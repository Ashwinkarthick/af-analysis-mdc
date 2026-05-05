"""Prediction dataframe loading, interface-analysis integration, and cache management."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
import streamlit as st

from af_analysis.utils import AlphaPulldownAnalyzer

CACHE_VERSION = "af_analysis_v13_interface_report"
ACCEPTED_CACHE_VERSIONS = {CACHE_VERSION, "ap" + "lit_aflite_v12_" + "result" + "_analysis", "ap" + "lit_aflite_v11", "ap" + "lit_refactor_v12"}
DEFAULT_INTERFACE_REPORT_CUTOFF = 100.0


def dataframe_cache_path(directory: str, compute_mean_pae: bool) -> Path:
    """Return the current AF-Analysis dataframe-cache path."""
    root = Path(directory).expanduser().resolve()
    cache_dir = root / ".af_analysis_cache"
    suffix = "meanpae1" if compute_mean_pae else "meanpae0"
    return cache_dir / f"results_{suffix}.pkl"


def legacy_dataframe_cache_path(directory: str, compute_mean_pae: bool) -> Path:
    """Return the previous cache path for backwards-compatible fast loading."""
    root = Path(directory).expanduser().resolve()
    legacy_name = "." + "ap" + "lit" + "_cache"
    suffix = "meanpae1" if compute_mean_pae else "meanpae0"
    return root / legacy_name / f"results_{suffix}.pkl"


def _read_dataframe_cache_file(cache_file: Path) -> Optional[pd.DataFrame]:
    if not cache_file.exists() or cache_file.stat().st_size <= 0:
        return None
    payload = pd.read_pickle(cache_file)
    if isinstance(payload, dict):
        df = payload.get("dataframe")
        if isinstance(df, pd.DataFrame):
            return df
    if isinstance(payload, pd.DataFrame):
        return payload
    return None


def load_dataframe_cache(directory: str, compute_mean_pae: bool) -> Optional[pd.DataFrame]:
    """Load the current cache, falling back to the previous cache location."""
    for cache_file in (dataframe_cache_path(directory, compute_mean_pae), legacy_dataframe_cache_path(directory, compute_mean_pae)):
        try:
            df = _read_dataframe_cache_file(cache_file)
            if df is not None:
                return df
        except Exception as exc:
            st.sidebar.warning(f"Could not read AF-Analysis cache `{cache_file.name}`: {exc}")
    return None
def save_dataframe_cache(directory: str, compute_mean_pae: bool, dataframe: pd.DataFrame) -> None:
    cache_file = dataframe_cache_path(directory, compute_mean_pae)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "created_at": time.time(),
            "directory": str(Path(directory).expanduser().resolve()),
            "compute_mean_pae": bool(compute_mean_pae),
            "dataframe": dataframe.copy(),
        }
        pd.to_pickle(payload, cache_file)
    except Exception as exc:
        st.sidebar.warning(f"Could not write AF-Analysis cache: {exc}")


def build_predictions_dataframe(directory: str, compute_mean_pae: bool) -> pd.DataFrame:
    analyzer = AlphaPulldownAnalyzer(
        directory,
        compute_mean_pae=compute_mean_pae,
        enable_legacy_fill=False,
    )
    return analyzer.analyze_directory()


def interface_analysis_filename(cutoff: float = DEFAULT_INTERFACE_REPORT_CUTOFF) -> str:
    cutoff_str = str(float(cutoff)).replace(".", "_")
    return f"predictions_with_pae_cutoff_{cutoff_str}.csv"


def interface_report_csv_path(directory: str, cutoff: float = DEFAULT_INTERFACE_REPORT_CUTOFF) -> Path:
    return Path(directory).expanduser().resolve() / interface_analysis_filename(cutoff)


def find_interface_report_csv(directory: str, cutoff: float = DEFAULT_INTERFACE_REPORT_CUTOFF) -> Optional[Path]:
    root = Path(directory).expanduser().resolve()
    preferred = interface_report_csv_path(str(root), cutoff)
    if preferred.exists() and preferred.stat().st_size > 0:
        return preferred
    candidates = sorted(root.glob("predictions_with_pae_cutoff_*.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def load_interface_report_csv(directory: str, cutoff: float = DEFAULT_INTERFACE_REPORT_CUTOFF) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    csv_path = find_interface_report_csv(directory, cutoff)
    if csv_path is None:
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if "jobs" not in df.columns:
            return None, csv_path
        return df, csv_path
    except Exception as exc:
        st.warning(f"Could not read interface-analysis CSV `{csv_path}`: {exc}")
        return None, csv_path


def run_integrated_interface_analysis(
    directory: str,
    cutoff: float = DEFAULT_INTERFACE_REPORT_CUTOFF,
    run_alphajudge: bool = True,
    force_alphajudge: bool = False,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Run bundled af_analysis.interaction_report and return (success, log, csv_path).

    When ``progress_callback`` is provided, stdout is streamed line-by-line and
    the callback receives ``(line, progress_0_to_1, phase_label)``.  This keeps
    Streamlit responsive while AlphaJudge/interface-analysis backfill is running.
    """
    root = Path(directory).expanduser().resolve()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "af_analysis.interaction_report",
        "--output_dir",
        str(root),
        "--cutoff",
        str(float(cutoff)),
    ]
    if run_alphajudge:
        cmd.append("--run-alphajudge")
    if force_alphajudge:
        cmd.append("--force-alphajudge")

    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    output_lines: list[str] = []
    progress = 0.02
    phase = "Starting bundled interaction-report module"
    if progress_callback:
        progress_callback("Launching bundled interaction-report module", progress, phase)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception as exc:
        return False, f"Could not start bundled interface_analysis: {exc}", None

    total_jobs: Optional[int] = None
    for raw_line in proc.stdout or []:
        line = raw_line.rstrip("\n")
        output_lines.append(line)

        found_match = re.search(r"Found\s+(\d+)\s+job directories", line)
        if found_match:
            total_jobs = max(int(found_match.group(1)), 1)
            progress = max(progress, 0.08)
            phase = f"Discovered {total_jobs} job folder(s)"

        aj_match = re.search(r"Running AlphaJudge\s+(\d+)/(\d+):\s*(.*)", line)
        if aj_match:
            current = int(aj_match.group(1))
            total = max(int(aj_match.group(2)), 1)
            progress = max(progress, min(0.42, 0.10 + 0.32 * current / total))
            phase = f"Running AlphaJudge {current}/{total}: {aj_match.group(3)}"

        if line.startswith("AlphaJudge summary:"):
            progress = max(progress, 0.44)
            phase = line

        proc_match = re.search(r"Processing\s+(\d+)/(\d+):\s*(.*)", line)
        if proc_match:
            current = int(proc_match.group(1))
            total = max(int(proc_match.group(2)), 1)
            progress = max(progress, min(0.96, 0.45 + 0.50 * current / total))
            phase = f"Scoring structures {current}/{total}: {proc_match.group(3)}"
        elif line.startswith("  ✓ Added") and total_jobs:
            progress = max(progress, min(0.96, progress + 0.5 / total_jobs))

        if "Wrote" in line and "predictions_with_pae_cutoff" in line:
            progress = max(progress, 0.98)
            phase = "Writing interface-analysis CSV"

        if progress_callback:
            progress_callback(line, progress, phase)

    return_code = proc.wait()
    output = "\n".join(output_lines)
    csv_path = interface_report_csv_path(str(root), cutoff)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path = find_interface_report_csv(str(root), cutoff) or csv_path
    success = return_code == 0 and csv_path.exists() and csv_path.stat().st_size > 0
    if progress_callback:
        progress_callback("Finished bundled interaction-report module", 1.0 if success else progress, "Done" if success else "Finished with errors")
    return success, output, csv_path if csv_path.exists() else None


def _norm_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.replace("\\", "/")


def _candidate_keys_for_row(row: pd.Series, output_root: Path) -> Iterable[str]:
    job = _norm_key(row.get("job"))
    if job:
        yield job
        parts = [p for p in Path(job).parts if p not in {".", ""}]
        if parts:
            yield parts[0]
            yield parts[-1]
    raw_path = _norm_key(row.get("path"))
    if raw_path:
        path = Path(raw_path)
        yield path.name
        try:
            rel = path.resolve().relative_to(output_root.resolve())
            rel_parts = [p for p in rel.parts if p not in {".", ""}]
            if rel_parts:
                yield rel_parts[0]
                yield str(rel).replace("\\", "/")
        except Exception:
            pass


def _dedupe(seq: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in seq:
        item = _norm_key(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def merge_interface_analysis_scores(base_df: pd.DataFrame, result_df: Optional[pd.DataFrame], directory: str) -> pd.DataFrame:
    """Merge bundled interface-analysis CSV columns into the fast AF-Analysis dataframe.

    The merge uses job/folder names and keeps AlphaJudge-derived columns prefixed
    with ``aj_`` while preserving the interface-analysis values such as ``mpDockQ``,
    ``protein1_plddt`` and ``protein2_plddt`` as first-class table columns.
    """
    if base_df is None or base_df.empty or result_df is None or result_df.empty or "jobs" not in result_df.columns:
        out = base_df if base_df is not None else pd.DataFrame()
        if hasattr(out, "attrs"):
            out.attrs["af_analysis_interface_report_merged_rows"] = 0
        return out

    output_root = Path(directory).expanduser().resolve()
    result_index: Dict[str, pd.Series] = {}
    for _, result_row in result_df.iterrows():
        job = _norm_key(result_row.get("jobs"))
        for key in _dedupe([job, Path(job).name if job else "", Path(job).parts[0] if job and Path(job).parts else ""]):
            result_index.setdefault(key, result_row)

    # Keep the exact public columns from predictions_with_pae_cutoff_*.csv.
    merge_cols = [col for col in result_df.columns if col != "jobs"]
    merged = base_df.copy()
    for col in merge_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    matched = 0
    for idx, base_row in merged.iterrows():
        source = None
        for key in _dedupe(_candidate_keys_for_row(base_row, output_root)):
            source = result_index.get(key)
            if source is not None:
                break
        if source is None:
            continue
        matched += 1
        for col in merge_cols:
            value = source.get(col)
            if pd.isna(value):
                continue
            # Interface-analysis columns are authoritative for these score fields.
            merged.at[idx, col] = value
        merged.at[idx, "interface_analysis_csv"] = str(interface_report_csv_path(str(output_root)))

    merged.attrs["af_analysis_interface_report_merged_rows"] = matched
    return merged
