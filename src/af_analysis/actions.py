"""One-shot actions that modify or backfill prediction metadata."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, List

import pandas as pd

from af_analysis.utils import (
    AlphaPulldownAnalyzer,
    discover_prediction_jobs,
    patch_pae_json_for_alphajudge,
    prepare_af3_for_alphajudge,
)


def patch_in_house_metrics_for_directory(directory: str) -> int:
    root = Path(directory)
    if not root.exists():
        return 0

    patched = 0
    analyzer = AlphaPulldownAnalyzer(str(root))
    for job_dir in discover_prediction_jobs(root, max_depth=2):
        job_info = analyzer._get_job_models(job_dir)
        if not job_info or not job_info.get("models"):
            continue
        for model in job_info.get("models", []):
            if patch_pae_json_for_alphajudge(job_dir, str(model.get("model_name", ""))):
                patched += 1
    return patched


def _row_interfaces_csv_missing(row: pd.Series) -> bool:
    for col in ("interface_csv", "interfaces_csv"):
        value = row.get(col) if col in row.index else None
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"na", "n/a", "nan", "none", "null"}:
            path = Path(value)
            if path.exists() and path.stat().st_size > 0:
                return False
    path_value = row.get("path") if "path" in row.index else None
    if isinstance(path_value, str) and path_value:
        csv_path = Path(path_value) / "interfaces.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return False
    return True


def rows_needing_alphajudge(results_df: pd.DataFrame) -> List[str]:
    if results_df is None or results_df.empty or "path" not in results_df.columns:
        return []
    paths: List[str] = []
    for _, row in results_df.iterrows():
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        if _row_interfaces_csv_missing(row):
            paths.append(path)
    return list(dict.fromkeys(paths))


def run_alphajudge_for_paths(paths: List[str]) -> str:
    exe = shutil.which("alphajudge")
    if not exe:
        return "AlphaJudge is not available in PATH. No jobs were run."
    ran = skipped = failed = 0
    for raw_path in paths:
        job_dir = Path(raw_path)
        if not job_dir.is_dir():
            failed += 1
            continue
        interfaces = job_dir / "interfaces.csv"
        if interfaces.exists() and interfaces.stat().st_size > 0:
            skipped += 1
            continue
        analyzer = AlphaPulldownAnalyzer(str(job_dir.parent))
        info = analyzer._get_job_models(job_dir)
        if info and info.get("job_type") == "af2":
            for model in info.get("models", [])[:1]:
                patch_pae_json_for_alphajudge(job_dir, str(model.get("model_name", "")))
        elif info and info.get("job_type") == "af3":
            prepare_af3_for_alphajudge(job_dir)
        else:
            prepare_af3_for_alphajudge(job_dir)
        log_path = job_dir / "alphajudge.af_analysis.log"
        cmd = [
            exe,
            str(job_dir),
            "--models_to_analyse",
            "best",
            "--contact_thresh",
            "8.0",
            "--pae_filter",
            "100.0",
            "--ipsae_pae_cutoff",
            "10.0",
            "--cores",
            "1",
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("Command: " + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if proc.returncode == 0 and interfaces.exists() and interfaces.stat().st_size > 0:
            ran += 1
        else:
            failed += 1
    return (
        f"AlphaJudge: {ran} run, {skipped} already present, {failed} failed or unavailable. "
        "Logs are written as alphajudge.af_analysis.log inside each job folder."
    )


FILLABLE_BACKFILL_COLUMNS = [
    "mean_pae",
    "global_dockq",
    "interface_score",
    "interface_average_plddt",
    "interface_residue_count",
    "interface_contact_pairs",
    "interface_area",
    "interface_solv_energy",
    "interface_polar_fraction",
    "interface_hydrophobic_fraction",
    "interface_charged_fraction",
]


def rows_needing_backfill(results_df: pd.DataFrame) -> List[str]:
    if results_df is None or results_df.empty or "path" not in results_df.columns:
        return []

    fillable_cols = [
        "iptm",
        "iptm_ptm",
        "ptm",
        "mean_pae",
        "global_dockq",
        "best_interface_pdockq2",
        "best_interface_ipsae",
        "best_interface_lis",
        "interface_score",
        "interface_average_plddt",
        "interface_residue_count",
        "interface_contact_pairs",
        "interface_area",
        "interface_solv_energy",
        "interface_polar_fraction",
        "interface_hydrophobic_fraction",
        "interface_charged_fraction",
    ]
    existing = [col for col in fillable_cols if col in results_df.columns]

    if not existing:
        if "interface_csv" in results_df.columns:
            mask = results_df["interface_csv"].isna()
        else:
            mask = pd.Series(True, index=results_df.index)
    else:
        mask = results_df[existing].isna().any(axis=1)

    paths = results_df.loc[mask, "path"].dropna().astype(str).tolist()
    return list(dict.fromkeys(paths))


def merge_backfill_results(fast_df: pd.DataFrame, backfill_df: pd.DataFrame) -> pd.DataFrame:
    if fast_df is None or fast_df.empty or backfill_df is None or backfill_df.empty:
        out = fast_df
        if hasattr(out, "attrs"):
            out.attrs["af_analysis_backfill_changed_cells"] = 0
        return out
    if "path" not in fast_df.columns or "path" not in backfill_df.columns:
        out = fast_df
        out.attrs["af_analysis_backfill_changed_cells"] = 0
        return out

    merged = fast_df.copy()
    backfill_by_path = backfill_df.set_index("path", drop=False)
    fill_cols = [col for col in backfill_df.columns if col not in {"job", "path"}]
    changed_cells = 0

    for idx, row in merged.iterrows():
        path = row.get("path")
        if path not in backfill_by_path.index:
            continue
        source: Any = backfill_by_path.loc[path]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        row_changed = False
        for col in fill_cols:
            if col not in merged.columns:
                merged[col] = pd.NA
            current = merged.at[idx, col]
            value = source.get(col)
            if pd.isna(current) and not pd.isna(value):
                merged.at[idx, col] = value
                row_changed = True
                changed_cells += 1
        if row_changed:
            merged.at[idx, "interface_fill_source"] = source.get(
                "interface_fill_source", "one_shot_inhouse_interface_analysis_backfill"
            )
    merged.attrs["af_analysis_backfill_changed_cells"] = changed_cells
    return merged
