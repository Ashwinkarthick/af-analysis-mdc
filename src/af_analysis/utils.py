"""
Utility functions and data classes for AlphaPulldown analysis
"""

import csv
import gzip
import json
import math
import shutil
import os
import pickle
import re
import shlex
import uuid

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except Exception:  # Plotly is optional at import time; the UI reports if it is unavailable.
    go = None




def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return default
            value = arr[0] if arr.size == 1 else np.nanmean(arr)
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"na", "n/a", "nan", "none", "null"}:
                return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _score_from_ranking(ranking_data: Dict[str, Any], key: str, model_name: str, default=None):
    values = ranking_data.get(key)
    if isinstance(values, dict) and model_name in values:
        return _to_float(values.get(model_name), default)
    return default


def _derive_iptm_ptm(iptm: Optional[float], ptm: Optional[float]) -> Optional[float]:
    """Return AF-Multimer-style combined score only when true ipTM and pTM exist."""
    if iptm is None or ptm is None:
        return None
    try:
        return 0.8 * float(iptm) + 0.2 * float(ptm)
    except Exception:
        return None


def _explicit_iptm_ptm_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    """Read only explicit ipTM+pTM fields; never use pLDDT/ranking_score as ipTM+pTM."""
    for key in ("iptm+ptm", "iptm_ptm", "iptm_ptm_score"):
        if key in payload:
            value = _to_float(payload.get(key), None)
            if value is not None:
                return value
    return None




def _load_pickle_scores(job_dir: Path, model_name: str) -> Dict[str, Optional[float]]:
    """Load true AF2 confidence scores from result_<model>.pkl(.gz) when present."""
    scores: Dict[str, Optional[float]] = {}
    for path in (job_dir / f"result_{model_name}.pkl", job_dir / f"result_{model_name}.pkl.gz"):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                result_data = pickle.load(handle)
            if isinstance(result_data, dict):
                for key in ("ptm", "iptm", "ranking_confidence", "iptm+ptm"):
                    if key in result_data:
                        scores[key] = _to_float(result_data.get(key), None)
            break
        except Exception:
            continue
    return scores

def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return None



# --- AF3 AlphaJudge compatibility helper inserted by patch_af_analysis_af3_alphajudge.py ---
def _link_or_copy_for_alphajudge(src: Optional[Path], dst: Path) -> bool:
    """Create a symlink/copy for AlphaJudge compatibility without touching originals."""
    if src is None or not src.exists() or src.stat().st_size == 0:
        return False
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return False
    except Exception:
        pass
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return True
    except Exception:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False


def prepare_af3_for_alphajudge(job_dir: Path) -> int:
    """Create AlphaJudge-compatible AF3 aliases for prefixed AF3 outputs.

    Adds standard aliases such as ranked_0_model.cif,
    ranked_0_summary_confidences.json, ranked_0_confidences.json, a minimal
    ranking_scores.csv, and seed-<seed>_sample-0/{model.cif,confidences.json,
    summary_confidences.json}. Originals are not renamed or overwritten.
    """
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        return 0

    summary_candidates = [
        p for p in job_dir.glob('*_summary_confidences.json')
        if not p.name.startswith('ranked_') and p.stat().st_size > 0
    ]
    plain_summary = job_dir / 'summary_confidences.json'
    if plain_summary.exists() and plain_summary.stat().st_size > 0:
        summary_candidates.append(plain_summary)
    if not summary_candidates:
        return 0

    summary = sorted(summary_candidates, key=lambda p: ('seed-' in str(p), len(p.name)))[0]
    prefix = job_dir.name if summary.name == 'summary_confidences.json' else summary.name[:-len('_summary_confidences.json')]

    conf: Optional[Path] = job_dir / f'{prefix}_confidences.json'
    if not conf.exists() or conf.stat().st_size == 0:
        confs = [
            p for p in job_dir.glob('*_confidences.json')
            if not p.name.endswith('_summary_confidences.json') and not p.name.startswith('ranked_') and p.stat().st_size > 0
        ]
        conf = sorted(confs, key=lambda p: len(p.name))[0] if confs else None

    model: Optional[Path] = job_dir / f'{prefix}_model.cif'
    if not model.exists() or model.stat().st_size == 0:
        models = [
            p for pattern in ('*.cif', '*.mmcif', '*.pdb') for p in job_dir.glob(pattern)
            if not p.name.startswith('ranked_') and p.stat().st_size > 0
        ]
        model = sorted(models, key=lambda p: len(p.name))[0] if models else None

    changed = 0
    for src, alias in (
        (summary, 'ranked_0_summary_confidences.json'),
        (conf, 'ranked_0_confidences.json'),
        (model, 'ranked_0_model.cif'),
    ):
        if _link_or_copy_for_alphajudge(src, job_dir / alias):
            changed += 1

    summary_data = _read_json(summary) or {}
    data = _read_json(job_dir / f'{prefix}_data.json') or {}
    seeds = data.get('modelSeeds') or data.get('model_seeds') or []
    try:
        seed = int(seeds[0]) if isinstance(seeds, list) and seeds else 1
    except Exception:
        seed = 1

    sample_dir = job_dir / f'seed-{seed}_sample-0'
    for src, alias in ((summary, 'summary_confidences.json'), (conf, 'confidences.json'), (model, 'model.cif')):
        if _link_or_copy_for_alphajudge(src, sample_dir / alias):
            changed += 1

    ranking_csv = job_dir / 'ranking_scores.csv'
    if not ranking_csv.exists() or ranking_csv.stat().st_size == 0:
        score = _to_float(summary_data.get('ranking_score'), None)
        if score is None:
            score = _to_float(summary_data.get('confidence_score'), None)
        if score is None:
            ptm = _to_float(summary_data.get('ptm'), None)
            iptm = _to_float(summary_data.get('iptm'), None)
            score = 0.8 * iptm + 0.2 * ptm if iptm is not None and ptm is not None else 0.0
        try:
            with ranking_csv.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['seed', 'sample', 'ranking_score'])
                writer.writerow([seed, 0, score])
            changed += 1
        except Exception:
            pass
    return changed
# --- end AF3 AlphaJudge compatibility helper ---


def patch_pae_json_for_alphajudge(job_dir: Path, model_name: str) -> bool:
    """Backfill score keys into compact AF2 PAE JSON files for AlphaJudge.

    Priority: true ptm/iptm from ranking_debug.json, then true ptm/iptm from
    result_<model>.pkl(.gz), then ranking_confidence / iptm+ptm from
    ranking_debug.json. The final fallback is marked as compatibility-only.
    """
    ranking_data = _read_json(job_dir / "ranking_debug.json") or {}
    pickle_scores = _load_pickle_scores(job_dir, model_name)

    iptm_ptm = _score_from_ranking(
        ranking_data,
        "iptm+ptm",
        model_name,
        pickle_scores.get("ranking_confidence", pickle_scores.get("iptm+ptm")),
    )
    ptm = _score_from_ranking(ranking_data, "ptm", model_name, pickle_scores.get("ptm"))
    iptm = _score_from_ranking(ranking_data, "iptm", model_name, pickle_scores.get("iptm"))

    score_source = "explicit_or_pickle"
    if ptm is None and iptm_ptm is not None:
        ptm = iptm_ptm
        score_source = "iptm+ptm_fallback_for_alphajudge"
    if iptm is None and iptm_ptm is not None:
        iptm = iptm_ptm
        score_source = "iptm+ptm_fallback_for_alphajudge"

    candidates = [job_dir / f"pae_{model_name}.json"]
    m = re.search(r"model_(\d+)", model_name)
    if m:
        n = m.group(1)
        candidates += [job_dir / f"pae_model_{n}_ptm_pred_0.json", job_dir / f"pae_model_{n}.json"]

    changed_any = False
    for path in dict.fromkeys(candidates):
        if not path.exists() or path.stat().st_size == 0:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        data = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(data, dict):
            continue
        if "predicted_aligned_error" not in data and "pae" not in data:
            continue
        changed = False
        for key, value in (("ptm", ptm), ("iptm", iptm), ("ranking_confidence", iptm_ptm), ("iptm+ptm", iptm_ptm)):
            if key not in data and value is not None:
                data[key] = float(value)
                changed = True
        if changed:
            data.setdefault("alphajudge_score_source", score_source)
            with path.open("w") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            changed_any = True

    return changed_any


def _looks_like_prediction_job(job_dir: Path) -> bool:
    """Fast prediction-job test without parsing large structures/PAE matrices."""
    try:
        if not job_dir.is_dir():
            return False
        direct_markers = (
            "ranking_debug.json",
            "ranking_scores.csv",
            "ranked_0.pdb",
            "ranked_0.pdb.gz",
            "ranked_0_model.cif",
            "ranked_0.cif",
        )
        if any((job_dir / marker).exists() for marker in direct_markers):
            return True
        if any(job_dir.glob("ranked_*_summary_confidences.json")):
            return True
        for child in job_dir.iterdir():
            if child.is_dir() and re.match(r"seed-[^/]+_sample-[^/]+", child.name):
                return True
    except Exception:
        return False
    return False


def discover_prediction_jobs(root_dir: Path, max_depth: int = 2) -> List[Path]:
    """Find prediction job directories at root/job and root/group/job depth.

    AlphaPulldown batches are commonly either one directory per job, or grouped
    under a second-level folder such as output/job_id/prediction_job. Keep this
    intentionally shallow to preserve the original fast AF-Analysis startup behavior.
    """
    root_dir = Path(root_dir)
    jobs: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            jobs.append(path)

    try:
        first_level = sorted(p for p in root_dir.iterdir() if p.is_dir())
    except Exception:
        return jobs

    for child in first_level:
        if _looks_like_prediction_job(child):
            add(child)
            continue
        if max_depth < 2:
            continue
        try:
            for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
                if _looks_like_prediction_job(grandchild):
                    add(grandchild)
        except Exception:
            continue
    return jobs


class AlphaPulldownAnalyzer:
    """Handles analysis of AlphaPulldown prediction directories"""

    def __init__(
        self,
        output_dir: str,
        compute_mean_pae: bool = False,
        enable_legacy_fill: bool = False,
        legacy_fill_jobs: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.compute_mean_pae = compute_mean_pae
        self.enable_legacy_fill = enable_legacy_fill
        self.legacy_fill_jobs = set(legacy_fill_jobs or [])
        self._job_cache: Dict[Path, Optional[Dict[str, Any]]] = {}

    def _job_label(self, job_dir: Path) -> str:
        try:
            return str(job_dir.relative_to(self.output_dir))
        except Exception:
            return job_dir.name

    def _job_selected_for_legacy_fill(self, job_dir: Path) -> bool:
        if not self.legacy_fill_jobs:
            return True
        label = self._job_label(job_dir)
        labels = {label, job_dir.name, str(job_dir), str(job_dir.resolve())}
        return bool(labels & self.legacy_fill_jobs)

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        return _to_float(value, default)

    @staticmethod
    def _select_best_interface_row(df: pd.DataFrame) -> Optional[pd.Series]:
        """Return the interface row with the highest ipTM (fallback to ipTM+PTM)."""
        if df is None or df.empty:
            return None

        working_df = df.copy()
        for col in ["iptm", "iptm_ptm"]:
            if col in working_df.columns:
                working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

        sort_cols = [col for col in ["iptm", "iptm_ptm"] if col in working_df.columns]
        if sort_cols:
            working_df = working_df.sort_values(
                sort_cols, ascending=False, na_position="last"
            )

        try:
            best_idx = working_df.index[0]
        except IndexError:
            return None
        return df.loc[best_idx]

    def _detect_job_type(self, job_dir: Path) -> Optional[str]:
        if (job_dir / "ranking_debug.json").exists():
            return "af2"
        if (job_dir / "ranking_scores.csv").exists():
            return "af3"

        # AlphaFold 3 jobs (without AlphaJudge) often only contain seed/sample folders
        # and ranked_* files but no ranking_scores.csv. Detect those patterns as AF3.
        has_seed_dirs = any(
            d.is_dir() and re.match(r"seed-[^/]+_sample-[^/]+", d.name)
            for d in job_dir.iterdir()
        )
        if has_seed_dirs:
            return "af3"

        if list(job_dir.glob("ranked_*_summary_confidences.json")):
            return "af3"
        if (job_dir / "ranked_0_model.cif").exists():
            return "af3"
        return None

    def _load_af2_models(self, job_dir: Path) -> List[Dict[str, Any]]:
        ranking_file = job_dir / "ranking_debug.json"
        if not ranking_file.exists():
            return []

        with ranking_file.open() as f:
            ranking_data = json.load(f)

        order = ranking_data.get("order", [])
        models: List[Dict[str, Any]] = []
        for rank_idx, model_name in enumerate(order):
            structure_path = job_dir / f"ranked_{rank_idx}.pdb"
            if not structure_path.exists():
                for alt in (job_dir / f"ranked_{rank_idx}.pdb.gz", job_dir / f"ranked_{rank_idx}.cif", job_dir / f"ranked_{rank_idx}.mmcif"):
                    if alt.exists():
                        structure_path = alt
                        break
            pickle_scores = _load_pickle_scores(job_dir, model_name)
            iptm = _score_from_ranking(ranking_data, "iptm", model_name, pickle_scores.get("iptm"))
            ptm = _score_from_ranking(ranking_data, "ptm", model_name, pickle_scores.get("ptm"))
            explicit_iptm_ptm = _score_from_ranking(
                ranking_data,
                "iptm+ptm",
                model_name,
                pickle_scores.get("iptm+ptm"),
            )
            iptm_ptm = explicit_iptm_ptm if explicit_iptm_ptm is not None else _derive_iptm_ptm(iptm, ptm)
            ranking_confidence = pickle_scores.get("ranking_confidence")
            suffix = structure_path.suffix.lower()
            structure_format = "mmcif" if suffix in {".cif", ".mmcif"} else "pdb"
            models.append(
                {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": iptm if iptm is not None else 0.0,
                    "iptm_ptm": iptm_ptm,
                    "ptm": ptm,
                    "confidence_score": iptm_ptm if iptm_ptm is not None else ranking_confidence,
                    "mean_plddt": ranking_confidence if iptm_ptm is None else None,
                    "structure_file": structure_path,
                    "structure_format": structure_format,
                    "pdb_file": structure_path,
                    "job_type": "af2",
                }
            )
        return models
    def _load_af3_models(self, job_dir: Path) -> List[Dict[str, Any]]:
        ranking_path = job_dir / "ranking_scores.csv"
        rows: List[Dict[str, Any]] = []

        if ranking_path.exists():
            with ranking_path.open(newline="") as f:
                rows = [row for row in csv.DictReader(f) if row]
        else:
            # Fallback 1: seed/sample directories
            for seed_dir in sorted(job_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                m = re.match(r"seed-(?P<seed>[^_]+)_sample-(?P<sample>.+)", seed_dir.name)
                if not m:
                    continue
                summary_path = seed_dir / "summary_confidences.json"
                row: Dict[str, Any] = {
                    "seed": m.group("seed"),
                    "sample": m.group("sample"),
                    "_model_dir": str(seed_dir),
                }
                if summary_path.exists():
                    try:
                        with summary_path.open() as f:
                            summary = json.load(f)
                        row["_cached_summary"] = summary
                        row["ranking_score"] = summary.get("ranking_score")
                    except Exception as exc:
                        st.warning(f"Could not read {summary_path}: {exc}")
                rows.append(row)

            # Fallback 2: ranked_* summary files only
            if not rows:
                ranked_summaries = sorted(job_dir.glob("ranked_*_summary_confidences.json"))
                for path in ranked_summaries:
                    rank_match = re.match(
                        r"ranked_(\d+)_summary_confidences\.json", path.name
                    )
                    rank_idx = int(rank_match.group(1)) if rank_match else 0
                    row = {
                        "seed": f"ranked_{rank_idx}",
                        "sample": str(rank_idx),
                        "ranking_score": None,
                        "_model_dir": str(job_dir),
                        "_summary_path": str(path),
                        "_ranked_idx": rank_idx,
                    }
                    try:
                        with path.open() as f:
                            summary = json.load(f)
                        row["_cached_summary"] = summary
                        row["ranking_score"] = summary.get("ranking_score")
                    except Exception as exc:
                        st.warning(f"Could not read {path}: {exc}")
                    rows.append(row)

        if not rows:
            return []

        def row_score(row: Dict[str, Any]) -> float:
            cached = row.get("_cached_summary")
            if cached:
                val = self._safe_float(cached.get("ranking_score"), float("-inf"))
                if val is not None:
                    return val
            val = self._safe_float(row.get("ranking_score"), float("-inf"))
            return val if val is not None else float("-inf")

        rows.sort(key=row_score, reverse=True)

        models: List[Dict[str, Any]] = []
        for rank_idx, row in enumerate(rows):
            seed = row.get("seed")
            sample = row.get("sample")
            if seed is None or sample is None:
                continue

            model_name = f"seed-{seed}_sample-{sample}"
            model_dir = Path(row.get("_model_dir")) if row.get("_model_dir") else job_dir / model_name
            summary_path = Path(row.get("_summary_path")) if row.get("_summary_path") else model_dir / "summary_confidences.json"
            summary: Dict[str, Any] = row.get("_cached_summary", {})

            if not summary and summary_path.exists():
                try:
                    with summary_path.open() as f:
                        summary = json.load(f)
                except Exception as exc:
                    st.warning(f"Could not read {summary_path}: {exc}")
                    summary = {}

            if not summary and "_ranked_idx" in row:
                fallback_summary = job_dir / f"ranked_{row['_ranked_idx']}_summary_confidences.json"
                if fallback_summary.exists():
                    try:
                        with fallback_summary.open() as f:
                            summary = json.load(f)
                    except Exception as exc:
                        st.warning(f"Could not read {fallback_summary}: {exc}")

            iptm = self._safe_float(summary.get("iptm"), None)
            ptm = self._safe_float(summary.get("ptm"), None)
            explicit_iptm_ptm = _explicit_iptm_ptm_from_payload(summary) if isinstance(summary, dict) else None
            iptm_ptm = explicit_iptm_ptm if explicit_iptm_ptm is not None else _derive_iptm_ptm(iptm, ptm)
            confidence_score = self._safe_float(
                summary.get("confidence_score", summary.get("ranking_score", row.get("ranking_score"))),
                None,
            )
            mean_plddt = self._safe_float(summary.get("plddt", summary.get("mean_plddt", summary.get("pLDDT"))), None)

            structure_path = Path(row.get("_structure_path")) if row.get("_structure_path") else model_dir / "model.cif"
            if not structure_path.exists():
                ranked_alt_idx = row.get("_ranked_idx", rank_idx)
                ranked_alt = job_dir / f"ranked_{ranked_alt_idx}_model.cif"
                if ranked_alt.exists():
                    structure_path = ranked_alt
                else:
                    pdb_alt = job_dir / f"ranked_{ranked_alt_idx}.pdb"
                    if pdb_alt.exists():
                        structure_path = pdb_alt
            suffix = structure_path.suffix.lower()
            structure_format = "mmcif" if suffix in {".cif", ".mmcif"} else "pdb"

            models.append(
                {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": iptm if iptm is not None else 0.0,
                    "iptm_ptm": iptm_ptm,
                    "ptm": ptm,
                    "structure_file": structure_path,
                    "structure_format": structure_format,
                    "pdb_file": structure_path,
                    "job_type": "af3",
                    "confidence_score": confidence_score,
                    "mean_plddt": mean_plddt,
                }
            )

        return models

    def _get_job_models(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        job_dir = job_dir.resolve()
        if job_dir in self._job_cache:
            return self._job_cache[job_dir]

        job_type = self._detect_job_type(job_dir)
        if job_type is None:
            self._job_cache[job_dir] = None
            return None

        if job_type == "af2":
            models = self._load_af2_models(job_dir)
        else:
            models = self._load_af3_models(job_dir)

        info = {"job_type": job_type, "models": models}
        self._job_cache[job_dir] = info
        return info

    def obtain_seq_lengths(self, result_dir: Path) -> List[int]:
        """Extract sequence lengths from PDB file by counting chains"""
        try:
            # Get the best ranked PDB file
            pdb_file = result_dir / "ranked_0.pdb"
            if not pdb_file.exists():
                return []

            # Parse PDB to get chain lengths
            chain_lengths = {}
            with open(pdb_file, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        res_num = int(line[22:26].strip())
                        if chain_id not in chain_lengths:
                            chain_lengths[chain_id] = 0
                        chain_lengths[chain_id] = max(chain_lengths[chain_id], res_num)

            # Return list of chain lengths
            return list(chain_lengths.values()) if chain_lengths else []

        except Exception as e:
            st.warning(f"Could not extract sequence lengths from {result_dir}: {e}")
            return []

    def obtain_pae_and_iptm(
        self, result_dir: Path, model_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Extract PAE matrix and ipTM score for a specific model."""
        try:
            pae_file = get_pae_file_for_model(result_dir, model_name)
            pae_mtx = load_pae_matrix_cached(str(pae_file)) if pae_file else None

            ranking_data: Dict[str, Any] = {}
            ranking_file = result_dir / "ranking_debug.json"
            if ranking_file.exists():
                with ranking_file.open() as f:
                    ranking_data = json.load(f)

            pickle_scores = _load_pickle_scores(result_dir, model_name)
            iptm_score = _score_from_ranking(ranking_data, "iptm", model_name, pickle_scores.get("iptm"))
            return pae_mtx, iptm_score

        except Exception as e:
            st.warning(f"Could not load PAE/ipTM data from {result_dir}: {e}")
            return None, None
    def get_all_models(self, result_dir: Path) -> List[Dict]:
        """Get information about all models in a prediction directory"""
        try:
            info = self._get_job_models(result_dir)
            if not info:
                return []
            return info.get("models", [])
        except Exception as e:
            st.warning(f"Could not load models from {result_dir}: {e}")
            return []

    def analyze_directory(self) -> pd.DataFrame:
        """Analyze all prediction jobs in the directory"""
        jobs = discover_prediction_jobs(self.output_dir, max_depth=2)
        if self.enable_legacy_fill and self.legacy_fill_jobs:
            jobs = [d for d in jobs if self._job_selected_for_legacy_fill(d)]

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, job_dir in enumerate(jobs):
            status_text.text(f"Processing {job_dir.name} ({idx + 1}/{len(jobs)})")

            try:
                job_info = self._get_job_models(job_dir)
                if not job_info or not job_info.get("models"):
                    continue

                models = job_info["models"]
                job_type = job_info["job_type"]

                best_model_info = models[0]
                best_model = best_model_info["model_name"]
                iptm_ptm_score = best_model_info.get("iptm_ptm")
                iptm_score = best_model_info.get("iptm")
                if iptm_score is None:
                    iptm_score = 0.0

                mean_pae = None
                should_legacy_fill_job = (
                    self.enable_legacy_fill
                    and self._job_selected_for_legacy_fill(job_dir)
                )

                seq_lengths: List[int] = []
                if job_type == "af2" and (self.compute_mean_pae or should_legacy_fill_job):
                    seq_lengths = self.obtain_seq_lengths(job_dir)

                pae_mtx = None
                if self.compute_mean_pae and job_type == "af2":
                    pae_mtx, _ = self.obtain_pae_and_iptm(job_dir, best_model)
                    if pae_mtx is not None:
                        if seq_lengths:
                            mean_pae = self.calculate_mean_inter_pae(
                                pae_mtx, seq_lengths
                            )

                interface_df = load_interfaces_csv(job_dir)
                interface_summary = (
                    self._select_best_interface_row(interface_df)
                    if interface_df is not None
                    else None
                )

                interface_metrics: Dict[str, Optional[float]] = {}
                if interface_summary is not None:
                    interface_value_map = {
                        "iptm": "iptm",
                        "iptm_ptm": "iptm_ptm",
                        "average_interface_pae": "mean_pae",
                        "ptm": "ptm",
                        "confidence_score": "confidence_score",
                        "DockQ": "dockq",
                        "dockq": "dockq",
                        "global_DockQ": "dockq",
                        "interface_DockQ": "dockq",
                        "interface_dockq": "dockq",
                        "DockQ_score": "dockq",
                        "dockq_score": "dockq",
                        "pDockQ/mpDockQ": "global_dockq",
                        "interface_pDockQ2": "best_interface_pdockq2",
                        "interface_ipSAE": "best_interface_ipsae",
                        "interface_LIS": "best_interface_lis",
                        "interface_score": "interface_score",
                        "interface_average_plddt": "interface_average_plddt",
                        "interface_num_intf_residues": "interface_residue_count",
                        "interface_contact_pairs": "interface_contact_pairs",
                        "interface_area": "interface_area",
                        "interface_solv_en": "interface_solv_energy",
                        "interface_polar": "interface_polar_fraction",
                        "interface_hydrophobic": "interface_hydrophobic_fraction",
                        "interface_charged": "interface_charged_fraction",
                    }

                    for src, dst in interface_value_map.items():
                        if src not in interface_summary:
                            continue
                        value = interface_summary.get(src)
                        interface_metrics[dst] = _to_float(value, None)

                    if not _is_missing_value(interface_metrics.get("iptm")):
                        iptm_score = interface_metrics.get("iptm", iptm_score)
                    if not _is_missing_value(interface_metrics.get("iptm_ptm")):
                        iptm_ptm_score = interface_metrics.get("iptm_ptm", iptm_ptm_score)
                    mean_pae = interface_metrics.get("mean_pae", mean_pae)

                used_legacy_fill = False
                if should_legacy_fill_job:
                    # One-shot in-house backfill only. Do not import external analysis packages here:
                    # its optional native dependency stack can be difficult to compile
                    # on HPC nodes. This path is selected only for rows with fillable
                    # missing values and is never part of the default fast load.
                    if pae_mtx is None and job_type == "af2":
                        pae_mtx, _ = self.obtain_pae_and_iptm(job_dir, best_model)
                    legacy_fallback_metrics = calculate_legacy_interface_fallback(
                        job_dir, best_model_info, pae_mtx, seq_lengths
                    )
                    interface_metrics, used_old_fill = fill_missing_interface_metrics(
                        interface_metrics, legacy_fallback_metrics
                    )
                    used_legacy_fill = used_legacy_fill or used_old_fill

                    if _is_missing_value(mean_pae) and not _is_missing_value(interface_metrics.get("mean_pae")):
                        mean_pae = interface_metrics.get("mean_pae")

                result_row = {
                    "job": self._job_label(job_dir),
                    "iptm_ptm": iptm_ptm_score,
                    "iptm": iptm_score,
                    "mean_pae": mean_pae,
                    "best_model": best_model,
                    "path": str(job_dir),
                    "n_models": len(models),
                    "job_type": job_type,
                    "interface_csv": str(interface_df["_interfaces_csv"].dropna().iloc[0])
                    if interface_df is not None and "_interfaces_csv" in interface_df.columns and not interface_df["_interfaces_csv"].dropna().empty
                    else None,
                    "interface_summary_model": interface_summary.get("model_used")
                    if interface_summary is not None
                    else None,
                    "interface_fill_source": (
                        "alphajudge_plus_interface_analysis_fill"
                        if interface_summary is not None and used_legacy_fill
                        else "alphajudge_interfaces_csv"
                        if interface_summary is not None
                        else "interface_analysis_fallback"
                        if used_legacy_fill
                        else None
                    ),
                }

                result_row.update(interface_metrics)
                results.append(result_row)

            except Exception as e:
                st.warning(f"Error processing {job_dir.name}: {e}")

            progress_bar.progress((idx + 1) / len(jobs))

        progress_bar.empty()
        status_text.empty()

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("iptm", ascending=False).reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    def calculate_mean_inter_pae(
        self, pae_mtx: np.ndarray, seq_lengths: List[int]
    ) -> float:
        """Calculate mean PAE for inter-chain regions"""
        try:
            pae_copy = pae_mtx.copy()
            old_length = 0
            for length in seq_lengths:
                new_length = old_length + length
                pae_copy[old_length:new_length, old_length:new_length] = np.nan
                old_length = new_length

            return float(np.nanmean(pae_copy))
        except Exception:
            return float("nan")



def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "n/a", "nan", "none", "null"}
    return False


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")


def _aa_class(residue_name: str) -> Tuple[bool, bool, bool]:
    polar = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
    charged = {"ARG", "LYS", "HIS", "ASP", "GLU"}
    return residue_name in polar, residue_name in hydrophobic, residue_name in charged


def _tokenize_cif_atom_line(line: str) -> List[str]:
    """Tokenize one mmCIF atom_site row.

    AlphaFold 3 writes ordinary whitespace-delimited atom_site rows. shlex also
    handles quoted residue/atom names if they appear. This intentionally avoids
    importing BioPython/MMCIFParser so AF-Analysis stays lightweight on the cluster.
    """
    try:
        return shlex.split(line, posix=True)
    except Exception:
        return line.split()


def _parse_mmcif_residues(structure_file: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Parse CA/CB atom coordinates and pLDDT from an AF3/mmCIF file.

    AF3 stores per-atom confidence in the _atom_site.B_iso_or_equiv column,
    analogous to AlphaFold PDB B-factors. The in-house fallback needs only
    chain id, residue number/name, atom name, coordinates, and pLDDT, so this
    parser reads just the atom_site loop rather than the full mmCIF grammar.
    """
    chains: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not structure_file or not structure_file.exists():
        return chains
    try:
        lines = structure_file.read_text(errors="ignore").splitlines()
    except Exception:
        return chains

    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: List[str] = []
        while i < n and lines[i].lstrip().startswith("_atom_site."):
            headers.append(lines[i].strip())
            i += 1
        if not headers or not any(h.startswith("_atom_site.Cartn_x") for h in headers):
            continue
        idx = {h: pos for pos, h in enumerate(headers)}

        def col(*names: str) -> Optional[int]:
            for name in names:
                full = name if name.startswith("_atom_site.") else f"_atom_site.{name}"
                if full in idx:
                    return idx[full]
            return None

        group_i = col("group_PDB")
        atom_i = col("label_atom_id", "auth_atom_id")
        comp_i = col("label_comp_id", "auth_comp_id")
        chain_i = col("auth_asym_id", "label_asym_id")
        seq_i = col("auth_seq_id", "label_seq_id")
        x_i = col("Cartn_x")
        y_i = col("Cartn_y")
        z_i = col("Cartn_z")
        b_i = col("B_iso_or_equiv")
        needed = [atom_i, comp_i, chain_i, seq_i, x_i, y_i, z_i]
        if any(v is None for v in needed):
            continue

        while i < n:
            raw = lines[i].strip()
            if not raw or raw == "#":
                i += 1
                break
            if raw == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                break
            parts = _tokenize_cif_atom_line(raw)
            if len(parts) < len(headers):
                i += 1
                continue
            try:
                if group_i is not None and parts[group_i].upper() not in {"ATOM", "HETATM"}:
                    i += 1
                    continue
                atom_name = parts[atom_i].strip().strip(chr(34) + chr(39))
                res_name = parts[comp_i].strip().strip(chr(34) + chr(39))
                chain_id = parts[chain_i].strip().strip(chr(34) + chr(39)) or "_"
                seq_token = parts[seq_i].strip().strip(chr(34) + chr(39))
                if seq_token in {".", "?"}:
                    i += 1
                    continue
                # auth_seq_id can contain insertion codes; keep the integer part.
                m = re.search(r"-?\d+", seq_token)
                if not m:
                    i += 1
                    continue
                res_num = int(m.group(0))
                x = float(parts[x_i])
                y = float(parts[y_i])
                z = float(parts[z_i])
                plddt = float(parts[b_i]) if b_i is not None and parts[b_i] not in {".", "?"} else float("nan")
            except Exception:
                i += 1
                continue
            chain = chains.setdefault(chain_id, {})
            residue = chain.setdefault(res_num, {"name": res_name, "plddt": plddt, "atoms": {}})
            residue["atoms"][atom_name] = np.array([x, y, z], dtype=float)
            if not np.isfinite(residue.get("plddt", np.nan)) and np.isfinite(plddt):
                residue["plddt"] = plddt
            elif np.isfinite(plddt):
                residue["plddt"] = plddt
            i += 1
    return chains


def _parse_pdb_residues(structure_file: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    chains: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not structure_file or not structure_file.exists():
        return chains
    if structure_file.name.lower().endswith((".cif", ".mmcif")):
        return _parse_mmcif_residues(structure_file)
    try:
        with _open_text(structure_file) as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                chain_id = line[21].strip() or "_"
                try:
                    res_num = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    plddt = float(line[60:66])
                except Exception:
                    continue
                chain = chains.setdefault(chain_id, {})
                residue = chain.setdefault(res_num, {"name": res_name, "plddt": plddt, "atoms": {}})
                residue["atoms"][atom_name] = np.array([x, y, z], dtype=float)
                residue["plddt"] = plddt
    except Exception:
        return {}
    return chains



def _first_existing_column(row: pd.Series, names: List[str], default=None):
    lower_map = {str(col).lower(): col for col in row.index}
    for name in names:
        col = lower_map.get(name.lower())
        if col is not None:
            value = row.get(col)
            if not _is_missing_value(value):
                return value
    return default


def calculate_external_analysis_fallback(job_dir: Path, job_type: str) -> Dict[str, Any]:
    """Deprecated compatibility stub.

    Earlier patched builds attempted to call an external general-purpose package here.
    That dependency path can pull in native code such as TMalign wrappers, which is
    fragile on shared HPC environments. AF-Analysis now keeps this as a no-op and uses
    the in-house legacy fallback directly.
    """
    return {}

def calculate_legacy_interface_fallback(job_dir: Path, best_model_info: Dict[str, Any], pae_mtx: Optional[np.ndarray], seq_lengths: List[int]) -> Dict[str, Any]:
    """Calculate interface-analysis-style fallback interface metrics for missing AlphaJudge cells.

    These are intentionally marked as legacy estimates. They are used only when
    AlphaJudge interfaces.csv is missing, empty, unreadable, or lacks a specific cell.
    """
    structure_file = Path(best_model_info.get("structure_file") or best_model_info.get("pdb_file") or "")
    chains = _parse_pdb_residues(structure_file)
    if len(chains) < 2:
        return {}

    chain_ids = sorted(chains.keys())
    chain_offsets: Dict[str, int] = {}
    offset = 0
    for idx, chain_id in enumerate(chain_ids):
        chain_offsets[chain_id] = offset
        if idx < len(seq_lengths):
            offset += int(seq_lengths[idx])
        else:
            offset += len(chains[chain_id])

    interface_residues = set()
    interface_plddts: List[float] = []
    interface_pae_values: List[float] = []
    contact_pairs = 0
    polar_count = hydrophobic_count = charged_count = 0
    cutoff = 8.0

    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            chain_a_id = chain_ids[i]
            chain_b_id = chain_ids[j]
            residues_a = chains[chain_a_id]
            residues_b = chains[chain_b_id]
            sorted_a = sorted(residues_a.items())
            sorted_b = sorted(residues_b.items())
            a_index = {resnum: pos for pos, (resnum, _) in enumerate(sorted_a)}
            b_index = {resnum: pos for pos, (resnum, _) in enumerate(sorted_b)}

            for res_a_num, res_a in sorted_a:
                atom_a = res_a["atoms"].get("CB", res_a["atoms"].get("CA"))
                if atom_a is None:
                    continue
                for res_b_num, res_b in sorted_b:
                    atom_b = res_b["atoms"].get("CB", res_b["atoms"].get("CA"))
                    if atom_b is None:
                        continue
                    if float(np.linalg.norm(atom_a - atom_b)) > cutoff:
                        continue
                    contact_pairs += 1
                    for chain_id, res_num, residue in ((chain_a_id, res_a_num, res_a), (chain_b_id, res_b_num, res_b)):
                        key = (chain_id, res_num)
                        if key not in interface_residues:
                            interface_residues.add(key)
                            interface_plddts.append(float(residue.get("plddt", np.nan)))
                            polar, hydro, charged = _aa_class(str(residue.get("name", "")))
                            polar_count += int(polar)
                            hydrophobic_count += int(hydro)
                            charged_count += int(charged)
                    if pae_mtx is not None and getattr(pae_mtx, "ndim", 0) == 2:
                        idx_a = chain_offsets.get(chain_a_id, 0) + a_index.get(res_a_num, 0)
                        idx_b = chain_offsets.get(chain_b_id, 0) + b_index.get(res_b_num, 0)
                        if idx_a < pae_mtx.shape[0] and idx_b < pae_mtx.shape[1]:
                            interface_pae_values.append(float(pae_mtx[idx_a, idx_b]))
                            interface_pae_values.append(float(pae_mtx[idx_b, idx_a]))

    if contact_pairs <= 0 or not interface_residues:
        return {}

    interface_count = len(interface_residues)
    avg_plddt = float(np.nanmean(interface_plddts)) if interface_plddts else None
    interface_score = (avg_plddt / 100.0 * math.log10(max(contact_pairs, 1))) if avg_plddt is not None else None
    mpdockq = (avg_plddt * math.log10(max(contact_pairs, 1))) if avg_plddt is not None else None

    fallback = {
        "global_dockq": mpdockq,
        "interface_score": interface_score,
        "interface_average_plddt": avg_plddt,
        "interface_residue_count": interface_count,
        "interface_contact_pairs": contact_pairs,
        "interface_area": float(contact_pairs * 20.0),
        "interface_solv_energy": float(-0.5 * contact_pairs),
        "interface_polar_fraction": polar_count / interface_count if interface_count else None,
        "interface_hydrophobic_fraction": hydrophobic_count / interface_count if interface_count else None,
        "interface_charged_fraction": charged_count / interface_count if interface_count else None,
        "legacy_fill_source": "in_house_structure_fallback",
    }
    if interface_pae_values:
        fallback["mean_pae"] = float(np.mean(interface_pae_values))
    return fallback


def fill_missing_interface_metrics(interface_metrics: Dict[str, Any], fallback_metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Fill only missing interface cells from fallback metrics."""
    if not fallback_metrics:
        return interface_metrics, False
    filled = dict(interface_metrics)
    changed = False
    for key, value in fallback_metrics.items():
        if key == "legacy_fill_source":
            continue
        if _is_missing_value(filled.get(key)) and not _is_missing_value(value):
            filled[key] = value
            changed = True
    if changed:
        filled["legacy_fill_source"] = fallback_metrics.get("legacy_fill_source", "interface_analysis_fallback")
    return filled, changed


@st.cache_data(show_spinner=False)
def load_pae_matrix_cached(pae_file: str) -> np.ndarray:
    """Load a PAE matrix once and reuse it across Streamlit reruns."""
    with open(pae_file, "r") as f:
        pae_data = json.load(f)

    if isinstance(pae_data, list):
        data = pae_data[0]
    else:
        data = pae_data

    matrix = data.get("predicted_aligned_error") or data.get("pae")
    if matrix is None:
        raise ValueError(f"No PAE matrix found in {pae_file}")
    return np.array(matrix, dtype=np.float32)


def _stride_downsample(matrix: np.ndarray, max_points: int = 900) -> Tuple[np.ndarray, int]:
    """Fast visual downsampling for large matrices shown in Streamlit."""
    n_rows, n_cols = matrix.shape[:2]
    stride = max(1, int(math.ceil(max(n_rows, n_cols) / max_points)))
    if stride == 1:
        return matrix, stride
    return matrix[::stride, ::stride], stride


def plot_pae_heatmap(pae_file: Path, figsize: Tuple[int, int] = (10, 10)) -> Optional[plt.Figure]:
    """Create a fast PAE heatmap plot for the web UI.

    Large PAE matrices are loaded through Streamlit's cache and visually
    downsampled by stride for display. The original JSON remains available via
    the download button, but reruns and model switching are much faster.
    """
    try:
        pae_mtx = load_pae_matrix_cached(str(pae_file))
        display_mtx, stride = _stride_downsample(pae_mtx, max_points=900)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            display_mtx,
            cmap="Greens_r",
            vmin=0,
            vmax=30,
            origin="lower",
            interpolation="nearest",
            rasterized=True,
        )

        total_residues = pae_mtx.shape[0]
        tick_count = 6
        residue_ticks = np.linspace(1, total_residues, min(tick_count, total_residues), dtype=int)
        plot_ticks = (residue_ticks - 1) / stride
        ax.set_xticks(plot_ticks)
        ax.set_xticklabels(residue_ticks)
        ax.set_yticks(plot_ticks)
        ax.set_yticklabels(residue_ticks)
        ax.tick_params(axis="both", which="major", length=5, width=1)

        ax.set_xlabel("Scored residue", fontsize=12)
        ax.set_ylabel("Aligned residue", fontsize=12)
        title = "Predicted Aligned Error (PAE)"
        if stride > 1:
            title += f" — displayed every {stride} residues"
        ax.set_title(title, fontsize=14, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Expected position error (Å)", rotation=270, labelpad=20)
        fig.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Could not create PAE plot: {e}")
        return None


def plot_pae_heatmap_interactive(pae_file: Path, max_points: int = 650):
    """Create a compact interactive PAE heatmap with residue-number hover.

    The full matrix is loaded from the JSON and visually downsampled for browser
    responsiveness. Hover text always reports residue numbers in the original
    coordinate system, not the downsampled pixel index.
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install AF-Analysis with the updated pyproject.toml dependencies.")

    pae_mtx = load_pae_matrix_cached(str(pae_file))
    display_mtx, stride = _stride_downsample(pae_mtx, max_points=max_points)
    total_residues = int(pae_mtx.shape[0])
    residue_axis = np.arange(1, total_residues + 1, stride, dtype=int)

    fig = go.Figure(
        data=go.Heatmap(
            z=display_mtx,
            x=residue_axis,
            y=residue_axis,
            zmin=0,
            zmax=30,
            colorscale="Greens_r",
            colorbar={"title": "PAE (Å)"},
            hovertemplate=(
                "Scored residue: %{x}<br>"
                "Aligned residue: %{y}<br>"
                "PAE: %{z:.2f} Å"
                "<extra></extra>"
            ),
        )
    )
    title = "Predicted Aligned Error (PAE)"
    if stride > 1:
        title += f" — displayed every {stride} residues"
    fig.update_layout(
        title=title,
        width=560,
        height=560,
        margin={"l": 55, "r": 25, "t": 55, "b": 50},
        xaxis_title="Scored residue",
        yaxis_title="Aligned residue",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_model_comparison(models: List[Dict]) -> Optional[plt.Figure]:
    """Create bar plot comparing all models"""
    if not models:
        return None

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ranks = [m["rank"] for m in models]
        iptm_scores = [np.nan if m.get("iptm") is None else m.get("iptm") for m in models]
        iptm_ptm_scores = [np.nan if m.get("iptm_ptm") is None else m.get("iptm_ptm") for m in models]

        # ipTM scores
        ax1.bar(ranks, iptm_scores, color="steelblue", alpha=0.8)
        ax1.set_xlabel("Model Rank", fontsize=11)
        ax1.set_ylabel("ipTM Score", fontsize=11)
        ax1.set_title("ipTM Scores", fontsize=12, fontweight="bold")
        ax1.set_xticks(ranks)
        ax1.set_ylim([0, 1])
        ax1.grid(axis="y", alpha=0.3)

        # ipTM+pTM scores
        ax2.bar(ranks, iptm_ptm_scores, color="coral", alpha=0.8)
        ax2.set_xlabel("Model Rank", fontsize=11)
        ax2.set_ylabel("ipTM+pTM Score", fontsize=11)
        ax2.set_title("ipTM+pTM Scores", fontsize=12, fontweight="bold")
        ax2.set_xticks(ranks)
        ax2.set_ylim([0, 1])
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Could not create comparison plot: {e}")
        return None


def create_3dmol_view(
    structure_file: Path,
    color_by: str = "plddt",
    structure_format: str = "pdb",
    width: int = 800,
    height: int = 600,
) -> str:
    """Create HTML for 3Dmol.js viewer"""
    try:
        with open(structure_file, "r") as f:
            structure_data = f.read()

        # Color schemes
        if color_by == "plddt":
            color_scheme = """
                viewer.setStyle({}, {
                    cartoon: {colorscheme: {
                        prop: 'b',
                        gradient: 'roygb',
                        min: 50,
                        max: 90
                    }}
                });
            """
        else:  # color by chain
            color_scheme = """
                viewer.setStyle({}, {cartoon: {colorscheme: 'chain'}});
            """

        data_literal = json.dumps(structure_data)
        model_format = (
            "mmcif"
            if structure_format.lower() in {"mmcif", "cif"}
            else structure_format.lower()
        )

        container_id = f"af-analysis-viewer-{uuid.uuid4().hex}"

        html = f"""
        <div id="{container_id}" style="width: {width}px; height: {height}px; position: relative;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            let viewer = $3Dmol.createViewer("{container_id}", {{
                backgroundColor: 'white'
            }});

            let pdbData = {data_literal};

            viewer.addModel(pdbData, "{model_format}");
            {color_scheme}
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(0.8, 1000);
        </script>
        """

        return html
    except Exception as e:
        st.error(f"Could not create 3D view: {e}")
        return ""


def get_pae_file_for_model(job_path: Path, model_name: str) -> Optional[Path]:
    """Find the PAE file for a specific model."""
    candidates = [job_path / f"pae_{model_name}.json"]
    match = re.search(r"model_(\d+)", model_name)
    if match:
        model_num = match.group(1)
        candidates.extend([
            job_path / f"pae_model_{model_num}_ptm_pred_0.json",
            job_path / f"pae_model_{model_num}.json",
        ])
    for candidate in dict.fromkeys(candidates):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_interfaces_csv_files(job_path: Path, max_depth: int = 5) -> List[Path]:
    """Find AlphaJudge interfaces.csv files directly or nested under a job."""
    if not job_path.exists():
        return []
    out: List[Path] = []
    for path in job_path.rglob("interfaces.csv"):
        try:
            rel_depth = len(path.relative_to(job_path).parts) - 1
        except ValueError:
            continue
        if rel_depth > max_depth:
            continue
        parts = set(path.relative_to(job_path).parts)
        if parts & {"_archive_duplicate_pairs", "__pycache__"}:
            continue
        if path.stat().st_size > 0:
            out.append(path)
    return sorted(out)


def load_interfaces_csv(job_path: Path) -> Optional[pd.DataFrame]:
    """Load direct or nested AlphaJudge interfaces.csv files for one job."""
    frames: List[pd.DataFrame] = []
    for csv_path in find_interfaces_csv_files(job_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["_interfaces_csv"] = str(csv_path)
            frames.append(df)
        except EmptyDataError:
            continue
        except Exception as e:
            st.warning(f"Could not load interfaces.csv from {csv_path}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_pae_plot_image(job_path: Path, model_name: str, rank: int) -> Optional[Path]:
    """Locate a precomputed PAE PNG image."""

    def _candidate_exists(path: Path) -> Optional[Path]:
        return path if path.exists() else None

    candidates = [
        job_path / f"pae_{model_name}.png",
        job_path / f"pae_plot_ranked_{rank}.png",
        job_path / f"pae_plot_ranked_{rank}.PNG",
    ]

    # Try a simplified model identifier if possible
    if "_" in model_name:
        model_num = model_name.split("_")[1]
        candidates.extend(
            [
                job_path / f"pae_model_{model_num}.png",
                job_path / f"pae_model_{model_num}_ptm_pred_0.png",
            ]
        )

    for candidate in candidates:
        existing = _candidate_exists(candidate)
        if existing:
            return existing

    return None


# ---------------------------------------------------------------------------
# af_analysis-lite backend
# ---------------------------------------------------------------------------
# This lightweight backend intentionally does NOT import af_analysis.  It keeps
# the useful part for AF-Analysis: broad result-folder discovery and dataframe-style
# metadata extraction for AF2/AF3/ColabFold-like outputs.  Heavy/compiled pieces
# such as pdb_cpp/TMalign, MDAnalysis, nglview, clustering, and notebook widgets
# are deliberately excluded so AF-Analysis remains installable on HPC systems.

_AF_LITE_SCORE_KEYS = (
    "ranking_confidence",
    "iptm+ptm",
    "iptm_ptm",
    "ranking_score",
    "confidence_score",
    "aggregate_score",
    "ptm",
    "iptm",
    "plddt",
    "mean_plddt",
    "pLDDT",
)

_AF_LITE_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif", ".pdb.gz"}


def _aflite_json(path: Path) -> Optional[Any]:
    return _read_json(path)


def _aflite_scalar_from_json(payload: Any, keys: Tuple[str, ...]) -> Optional[float]:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            val = _to_float(payload.get(key), None)
            if val is not None:
                return val
    return None


def _aflite_extract_scores(score_payload: Any) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {
        "iptm": None,
        "ptm": None,
        "iptm_ptm": None,
        "ranking_confidence": None,
        "confidence_score": None,
        "mean_plddt": None,
    }
    if isinstance(score_payload, list) and score_payload:
        score_payload = score_payload[0]
    if not isinstance(score_payload, dict):
        return scores

    scores["iptm"] = _to_float(score_payload.get("iptm"), None)
    scores["ptm"] = _to_float(score_payload.get("ptm"), None)
    scores["iptm_ptm"] = _explicit_iptm_ptm_from_payload(score_payload)
    if scores["iptm_ptm"] is None:
        scores["iptm_ptm"] = _derive_iptm_ptm(scores.get("iptm"), scores.get("ptm"))
    scores["ranking_confidence"] = _to_float(
        score_payload.get("ranking_confidence", score_payload.get("ranking_score", score_payload.get("aggregate_score"))),
        None,
    )
    scores["confidence_score"] = _to_float(
        score_payload.get("confidence_score", score_payload.get("ranking_score", score_payload.get("aggregate_score"))),
        None,
    )

    plddt = score_payload.get("plddt", score_payload.get("pLDDT", score_payload.get("mean_plddt")))
    scores["mean_plddt"] = _to_float(plddt, None)
    return scores


def _aflite_rank_from_name(name: str) -> int:
    patterns = [r"rank[_-]?(\d+)", r"ranked[_-]?(\d+)", r"model[_-]?(\d+)", r"sample[_-]?(\d+)"]
    for pattern in patterns:
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            try:
                value = int(m.group(1))
                # ColabFold ranks are usually 1-based; ranked_0 is 0-based.
                if "ranked" in pattern:
                    return value
                return max(0, value - 1) if "rank" in pattern and value > 0 else value
            except Exception:
                pass
    return 0


def _aflite_model_key_from_score_file(path: Path) -> str:
    stem = path.name
    stem = re.sub(r"\.json(\.gz)?$", "", stem)
    stem = re.sub(r"(_scores?|_summary_confidences|_confidence|_confidences)$", "", stem)
    return stem


def _aflite_structure_format(path: Path) -> str:
    name = path.name.lower()
    return "mmcif" if name.endswith((".cif", ".mmcif")) else "pdb"


def _aflite_find_matching_structure(job_dir: Path, score_path: Optional[Path], rank: int) -> Optional[Path]:
    candidates: List[Path] = []
    if score_path is not None:
        score_name = score_path.name
        prefixes = [
            re.sub(r"_scores.*$", "", score_name),
            re.sub(r"_summary_confidences.*$", "", score_name),
            re.sub(r"_confidence.*$", "", score_name),
            score_path.stem,
        ]
        for prefix in dict.fromkeys(prefixes):
            if not prefix:
                continue
            for ext in (".pdb", ".pdb.gz", ".cif", ".mmcif"):
                candidates += list(job_dir.glob(f"{prefix}*{ext}"))

    # Common names across AlphaFold/ColabFold/AF3/Boltz/Chai-like outputs.
    patterns = [
        f"ranked_{rank}.pdb",
        f"ranked_{rank}.pdb.gz",
        f"ranked_{rank}.cif",
        f"ranked_{rank}.mmcif",
        f"ranked_{rank}_model.cif",
        f"model_{rank}.pdb",
        f"model_{rank}.cif",
        f"model_{rank}.mmcif",
        f"prediction_{rank}.pdb",
        f"prediction_{rank}.cif",
        f"sample_{rank}.cif",
        f"sample_{rank}.pdb",
    ]
    for pattern in patterns:
        p = job_dir / pattern
        if p.exists():
            candidates.append(p)

    # Last resort: ranked structures first, then any structure file in this job.
    candidates += sorted(job_dir.glob("ranked_*.pdb"))
    candidates += sorted(job_dir.glob("ranked_*.pdb.gz"))
    candidates += sorted(job_dir.glob("ranked_*.cif"))
    candidates += sorted(job_dir.glob("ranked_*_model.cif"))
    candidates += sorted(p for p in job_dir.iterdir() if p.is_file() and p.name.lower().endswith((".pdb", ".cif", ".mmcif", ".pdb.gz")))
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _aflite_score_files(job_dir: Path) -> List[Path]:
    patterns = [
        "*_scores*.json",
        "*scores*.json",
        "*_summary_confidences.json",
        "summary_confidences.json",
        "*confidence*.json",
        "*_confidences.json",
        "ranking_scores.json",
        "ranking_debug.json",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(p for p in job_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0)
    # Exclude PAE-only JSONs from metadata model discovery.
    filtered = []
    for path in dict.fromkeys(files):
        lname = path.name.lower()
        if "pae" in lname or "predicted_aligned_error" in lname:
            continue
        filtered.append(path)
    return filtered


def _aflite_has_score_payload(path: Path) -> bool:
    payload = _aflite_json(path)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in _AF_LITE_SCORE_KEYS) or (
        "order" in payload and any(key in payload for key in ("iptm+ptm", "iptm", "ptm"))
    )


def _looks_like_prediction_job(job_dir: Path) -> bool:  # type: ignore[override]
    """Fast prediction-job test for AF2, AF3, ColabFold, and related outputs."""
    try:
        if not job_dir.is_dir():
            return False
        direct_markers = (
            "ranking_debug.json",
            "ranking_scores.csv",
            "ranked_0.pdb",
            "ranked_0.pdb.gz",
            "ranked_0_model.cif",
            "ranked_0.cif",
            "summary_confidences.json",
        )
        if any((job_dir / marker).exists() for marker in direct_markers):
            return True
        if any(job_dir.glob("ranked_*_summary_confidences.json")):
            return True
        if any(job_dir.glob("*_scores_rank_*.json")) or any(job_dir.glob("*scores*.json")):
            return True
        if any(job_dir.glob("*_unrelaxed_rank_*.pdb")) or any(job_dir.glob("*_relaxed_rank_*.pdb")):
            return True
        if any(job_dir.glob("*.cif")) and _aflite_score_files(job_dir):
            return True
        for child in job_dir.iterdir():
            if child.is_dir() and re.match(r"seed-[^/]+_sample-[^/]+", child.name):
                return True
    except Exception:
        return False
    return False


def _aflite_detect_job_type(self: AlphaPulldownAnalyzer, job_dir: Path) -> Optional[str]:
    # Preserve explicit high-confidence native detections first.
    if (job_dir / "ranking_debug.json").exists():
        return "af2"
    if (job_dir / "ranking_scores.csv").exists():
        return "af3"
    has_seed_dirs = any(
        d.is_dir() and re.match(r"seed-[^/]+_sample-[^/]+", d.name)
        for d in job_dir.iterdir()
    )
    if has_seed_dirs or list(job_dir.glob("ranked_*_summary_confidences.json")) or (job_dir / "ranked_0_model.cif").exists():
        return "af3"
    if any(job_dir.glob("*_scores_rank_*.json")) or any(job_dir.glob("*_unrelaxed_rank_*.pdb")) or any(job_dir.glob("*_relaxed_rank_*.pdb")):
        return "colabfold"
    # Generic parser for Boltz/Chai/MassiveFold-like folders with score JSON + structures.
    if _aflite_score_files(job_dir) and any(p.name.lower().endswith((".pdb", ".pdb.gz", ".cif", ".mmcif")) for p in job_dir.iterdir() if p.is_file()):
        return "generic"
    return None


def _aflite_load_colabfold_models(self: AlphaPulldownAnalyzer, job_dir: Path) -> List[Dict[str, Any]]:
    score_files = [p for p in _aflite_score_files(job_dir) if _aflite_has_score_payload(p)]
    if not score_files:
        # Structure-only ColabFold folder. Keep visible with minimal metadata.
        score_files = []
        structures = sorted(list(job_dir.glob("*_unrelaxed_rank_*.pdb")) + list(job_dir.glob("*_relaxed_rank_*.pdb")))
        return [
            {
                "rank": idx,
                "model_name": structure.stem,
                "iptm": 0.0,
                "iptm_ptm": None,
                "ptm": None,
                "structure_file": structure,
                "structure_format": _aflite_structure_format(structure),
                "pdb_file": structure,
                "job_type": "colabfold",
                "confidence_score": None,
            }
            for idx, structure in enumerate(structures)
        ]

    models: List[Dict[str, Any]] = []
    for path in score_files:
        payload = _aflite_json(path)
        scores = _aflite_extract_scores(payload)
        rank = _aflite_rank_from_name(path.name)
        structure = _aflite_find_matching_structure(job_dir, path, rank)
        if structure is None:
            continue
        ranking_conf = scores.get("ranking_confidence")
        iptm_ptm = scores.get("iptm_ptm")
        models.append(
            {
                "rank": rank,
                "model_name": _aflite_model_key_from_score_file(path),
                "iptm": scores.get("iptm") if scores.get("iptm") is not None else 0.0,
                "iptm_ptm": iptm_ptm,
                "ptm": scores.get("ptm"),
                "structure_file": structure,
                "structure_format": _aflite_structure_format(structure),
                "pdb_file": structure,
                "job_type": "colabfold",
                "confidence_score": scores.get("confidence_score") or ranking_conf,
                "mean_plddt": scores.get("mean_plddt"),
            }
        )
    models.sort(key=lambda m: (m.get("iptm_ptm") or m.get("confidence_score") or 0.0), reverse=True)
    for idx, model in enumerate(models):
        model["rank"] = idx
    return models


def _aflite_load_generic_models(self: AlphaPulldownAnalyzer, job_dir: Path) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    score_files = [p for p in _aflite_score_files(job_dir) if _aflite_has_score_payload(p)]
    structures = sorted(p for p in job_dir.iterdir() if p.is_file() and p.name.lower().endswith((".pdb", ".pdb.gz", ".cif", ".mmcif")))

    if score_files:
        for idx, path in enumerate(score_files):
            payload = _aflite_json(path)
            scores = _aflite_extract_scores(payload)
            rank = _aflite_rank_from_name(path.name)
            structure = _aflite_find_matching_structure(job_dir, path, rank)
            if structure is None and idx < len(structures):
                structure = structures[idx]
            if structure is None:
                continue
            ranking_conf = scores.get("ranking_confidence")
            iptm_ptm = scores.get("iptm_ptm")
            models.append(
                {
                    "rank": rank,
                    "model_name": _aflite_model_key_from_score_file(path),
                    "iptm": scores.get("iptm") if scores.get("iptm") is not None else 0.0,
                    "iptm_ptm": iptm_ptm,
                    "ptm": scores.get("ptm"),
                    "structure_file": structure,
                    "structure_format": _aflite_structure_format(structure),
                    "pdb_file": structure,
                    "job_type": "generic",
                    "confidence_score": scores.get("confidence_score") or ranking_conf,
                    "mean_plddt": scores.get("mean_plddt"),
                }
            )
    else:
        for idx, structure in enumerate(structures):
            models.append(
                {
                    "rank": idx,
                    "model_name": structure.stem,
                    "iptm": 0.0,
                    "iptm_ptm": None,
                    "ptm": None,
                    "structure_file": structure,
                    "structure_format": _aflite_structure_format(structure),
                    "pdb_file": structure,
                    "job_type": "generic",
                    "confidence_score": None,
                    "mean_plddt": None,
                }
            )

    models.sort(key=lambda m: (m.get("iptm_ptm") or m.get("confidence_score") or 0.0), reverse=True)
    for idx, model in enumerate(models):
        model["rank"] = idx
    return models


def _aflite_get_job_models(self: AlphaPulldownAnalyzer, job_dir: Path) -> Optional[Dict[str, Any]]:
    job_dir = job_dir.resolve()
    if job_dir in self._job_cache:
        return self._job_cache[job_dir]

    job_type = self._detect_job_type(job_dir)
    if job_type is None:
        self._job_cache[job_dir] = None
        return None

    if job_type == "af2":
        models = self._load_af2_models(job_dir)
    elif job_type == "af3":
        models = self._load_af3_models(job_dir)
    elif job_type == "colabfold":
        models = _aflite_load_colabfold_models(self, job_dir)
    else:
        models = _aflite_load_generic_models(self, job_dir)

    info = {"job_type": job_type, "models": models}
    self._job_cache[job_dir] = info
    return info


# Monkey-patch the class in-place so older AF-Analysis code keeps using the same public
# AlphaPulldownAnalyzer API while gaining af_analysis-style multi-format parsing.
AlphaPulldownAnalyzer._detect_job_type = _aflite_detect_job_type  # type: ignore[assignment]
AlphaPulldownAnalyzer._get_job_models = _aflite_get_job_models  # type: ignore[assignment]
