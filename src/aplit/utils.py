"""
Utility functions and data classes for AlphaPulldown analysis
"""

import csv
import gzip
import json
import math
import pickle
import re
import uuid

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import streamlit as st
import matplotlib.pyplot as plt




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

class AlphaPulldownAnalyzer:
    """Handles analysis of AlphaPulldown prediction directories"""

    def __init__(self, output_dir: str, compute_mean_pae: bool = False):
        self.output_dir = Path(output_dir)
        self.compute_mean_pae = compute_mean_pae
        self._job_cache: Dict[Path, Optional[Dict[str, Any]]] = {}

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
            iptm_ptm = _score_from_ranking(
                ranking_data,
                "iptm+ptm",
                model_name,
                pickle_scores.get("ranking_confidence", pickle_scores.get("iptm+ptm", 0.0)),
            )
            iptm = _score_from_ranking(ranking_data, "iptm", model_name, pickle_scores.get("iptm", 0.0))
            ptm = _score_from_ranking(ranking_data, "ptm", model_name, pickle_scores.get("ptm"))
            suffix = structure_path.suffix.lower()
            structure_format = "mmcif" if suffix in {".cif", ".mmcif"} else "pdb"
            models.append(
                {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": iptm if iptm is not None else 0.0,
                    "iptm_ptm": iptm_ptm if iptm_ptm is not None else 0.0,
                    "ptm": ptm,
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

            iptm = self._safe_float(summary.get("iptm"), 0.0)
            ptm = self._safe_float(summary.get("ptm"))
            iptm_ptm = self._safe_float(
                summary.get("ranking_score")
                or summary.get("iptm+ptm")
                or row.get("ranking_score"),
                0.0,
            )

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
                    "iptm_ptm": iptm_ptm if iptm_ptm is not None else 0.0,
                    "ptm": ptm,
                    "structure_file": structure_path,
                    "structure_format": structure_format,
                    "pdb_file": structure_path,
                    "job_type": "af3",
                    "confidence_score": self._safe_float(summary.get("confidence_score")),
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
        jobs = sorted(d for d in self.output_dir.iterdir() if d.is_dir())

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
                iptm_ptm_score = best_model_info.get("iptm_ptm", 0.0) or 0.0
                iptm_score = best_model_info.get("iptm", 0.0) or 0.0

                mean_pae = None
                if self.compute_mean_pae and job_type == "af2":
                    pae_mtx, _ = self.obtain_pae_and_iptm(job_dir, best_model)
                    if pae_mtx is not None:
                        seq_lengths = self.obtain_seq_lengths(job_dir)
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

                    iptm_score = interface_metrics.get("iptm", iptm_score)
                    iptm_ptm_score = interface_metrics.get("iptm_ptm", iptm_ptm_score)
                    mean_pae = interface_metrics.get("mean_pae", mean_pae)

                result_row = {
                    "job": job_dir.name,
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
def plot_model_comparison(models: List[Dict]) -> Optional[plt.Figure]:
    """Create bar plot comparing all models"""
    if not models:
        return None

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ranks = [m["rank"] for m in models]
        iptm_scores = [m["iptm"] for m in models]
        iptm_ptm_scores = [m["iptm_ptm"] for m in models]

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

        container_id = f"aplit-viewer-{uuid.uuid4().hex}"

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
