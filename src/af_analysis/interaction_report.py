#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated AF-Analysis interface-analysis module.

Create predictions_with_pae_cutoff_xxx.csv compatible with AlphaPulldown output format.

This script recreates the functionality of the official AlphaPulldown analysis pipeline
outside of the Singularity container, generating the standard results table with:
- iptm, iptm+ptm scores from AlphaFold
- mpDockQ scores 
- Interface analysis metrics
- Filtered by inter-chain PAE cutoff

Based on AlphaPulldown's get_good_inter_pae.py functionality.
"""

import argparse
import csv
import json
import math
import os
import pickle
import gzip
import glob
import re
import sys
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def to_scalar(value, default=None):
    """Convert numpy/list scalar-like values to a plain Python float when possible."""
    if value is None:
        return default
    try:
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            arr = np.asarray(value)
            if arr.size == 0:
                return default
            arr = arr.astype(float, copy=False).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return default
            return float(arr[0]) if arr.size == 1 else float(np.nanmean(arr))
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.size == 0:
                return default
            arr = arr.astype(float, copy=False).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return default
            return float(arr[0]) if arr.size == 1 else float(np.nanmean(arr))
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"nan", "none", "null", "na", "n/a"}:
                return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_round(value, digits: int, default=None):
    scalar = to_scalar(value, default=None)
    if scalar is None:
        return default
    return round(scalar, digits)


def safe_format(value, fmt: str, default: str = "N/A") -> str:
    scalar = to_scalar(value, default=None)
    if scalar is None:
        return default
    try:
        return format(scalar, fmt)
    except Exception:
        return default


def print_help():
    """Print basic help information about the script."""
    help_text = """
AF-Analysis Interaction Report Module
=====================================

USAGE:
    python -m af_analysis.interaction_report --output_dir <path> [--cutoff <value>] [--help] [--help-full]

REQUIRED ARGUMENTS:
    --output_dir PATH    Directory containing AlphaPulldown prediction results

CONNECTED WORKFLOW FRONT-END:
    bash /fast/AG_Daumke/ashwin/alphafold/lewin_10102023/alphapulldown/scripts/alphapulldown_workflow.sh report --output-dir <prediction_output_dir>

OPTIONAL ARGUMENTS:
    --cutoff FLOAT       Inter-chain PAE cutoff value (default: 10.0)
                        Lower values = higher confidence interfaces
    
    --run-alphajudge     Run AlphaJudge before merging interfaces.csv
    --force-alphajudge   Re-run AlphaJudge even when interfaces.csv exists
    --alphajudge-models best|all
    --alphajudge-contact-thresh FLOAT
    --alphajudge-pae-filter FLOAT
    --alphajudge-ipsae-pae-cutoff FLOAT
    --alphajudge-cores N
    --alphajudge-recursive
    --alphajudge-summary FILE
    --alphajudge-extra-flags "..."

    --help               Show this help message
    --help-full          Show comprehensive help with detailed explanations

KEY OUTPUT COLUMNS:
    - min_inter_chain_pae : Minimum PAE between chains (lower = better)
    - protein1/2_plddt   : Individual protein confidence (>70 = good)
    - iptm               : Interface confidence (>0.7 = high confidence)
    - iptm_ptm          : Combined score (>0.7 = high confidence)
    - mpDockQ           : Interface quality (>140 = excellent)

EXAMPLES:
    # Standard analysis (recommended for most cases)
    python -m af_analysis.interaction_report --output_dir output/predictions
    
    # High-confidence interactions only
    python -m af_analysis.interaction_report --output_dir output/predictions --cutoff 5.0
    
    # More permissive for exploratory analysis
    python -m af_analysis.interaction_report --output_dir output/predictions --cutoff 15.0

OUTPUT:
    Creates 'predictions_with_pae_cutoff_xxx.csv' with comprehensive interface analysis.

For detailed information about scores and interpretation, use --help-full
"""
    print(help_text)


def print_help_full():
    """Print comprehensive help information about the script."""
    help_text = """
AF-Analysis Interaction Report Module - Comprehensive Guide
==========================================================

OVERVIEW:
    This script creates comprehensive predictions_with_pae_cutoff_xxx.csv files
    compatible with AlphaPulldown output format. It processes AlphaFold multimer
    predictions and provides detailed interface analysis with confidence scores
    based on EBI AlphaFold training guidelines.

USAGE:
    python -m af_analysis.interaction_report --output_dir <path> [--cutoff <value>] [--help] [--help-full]

CONNECTED WORKFLOW FRONT-END:
    bash /fast/AG_Daumke/ashwin/alphafold/lewin_10102023/alphapulldown/scripts/alphapulldown_workflow.sh report --output-dir <prediction_output_dir>

ARGUMENTS:
    --output_dir PATH    Directory containing AlphaPulldown prediction results
                        Expected: job directories with ranking_debug.json,
                        ranked_0.pdb, and result_model_*.pkl(.gz) files
    
    --cutoff FLOAT       Inter-chain PAE cutoff (default: 10.0 Angstroms)
                        Filters models by minimum inter-chain PAE
                        Typical values: 5.0 (strict) to 20.0 (permissive)

    AlphaJudge backfill / merge:
    --run-alphajudge     Run AlphaJudge before merging interfaces.csv
    --force-alphajudge   Re-run AlphaJudge even when interfaces.csv exists
    --alphajudge-models best|all
                        Maps to AlphaJudge --models_to_analyse
    --alphajudge-contact-thresh FLOAT
                        Maps to AlphaJudge --contact_thresh; default 8.0
    --alphajudge-pae-filter FLOAT
                        Maps to AlphaJudge --pae_filter; default 100.0
    --alphajudge-ipsae-pae-cutoff FLOAT
                        Maps to AlphaJudge --ipsae_pae_cutoff; default 10.0
    --alphajudge-cores N
                        Maps to AlphaJudge --cores; 0 means all available cores
    --alphajudge-recursive
                        Maps to AlphaJudge -r/--recursive
    --alphajudge-summary FILE
                        Maps to AlphaJudge -o/--summary FILE
    --alphajudge-extra-flags "..."
                        Extra raw options appended to alphajudge

OUTPUT COLUMNS (in order of importance):

=== PRIMARY CONFIDENCE METRICS ===
    jobs                     : Prediction job name
    min_inter_chain_pae     : Minimum PAE between protein chains (Angstroms)
                             • < 5.0  = Very high confidence interface
                             • 5-10   = High confidence interface  
                             • 10-15  = Medium confidence interface
                             • > 15   = Low confidence interface
    
    protein1_plddt          : Per-residue confidence (first protein, 0-100)
    protein2_plddt          : Per-residue confidence (second protein, 0-100)
                             • > 90   = Very high confidence (dark blue)
                             • 70-90  = Confident (light blue) 
                             • 50-70  = Low confidence (yellow)
                             • < 50   = Very low confidence (orange/red)
    
    iptm                    : Interface predicted TM-score (0-1)
                             • > 0.8  = Very high confidence interface
                             • 0.6-0.8= High confidence interface
                             • 0.4-0.6= Medium confidence interface
                             • < 0.4  = Low confidence interface
    
    iptm_ptm               : Combined ranking score (0-1)
                             • > 0.8  = Very high overall confidence
                             • 0.6-0.8= High overall confidence
                             • 0.4-0.6= Medium overall confidence
                             • < 0.4  = Low overall confidence
    
    mpDockQ                : Interface quality score
                             • > 150  = Excellent interface
                             • 120-150= Good interface
                             • 100-120= Acceptable interface
                             • < 100  = Poor interface

=== INTERFACE COMPOSITION ===
    interface_residues     : Number of residues at the interface
    polar_fraction        : Fraction of polar residues (SER,THR,ASN,GLN,TYR,CYS)
    hydrophobic_fraction  : Fraction of hydrophobic residues (ALA,VAL,LEU,ILE,MET,PHE,TRP,PRO)
    charged_fraction      : Fraction of charged residues (ARG,LYS,HIS,ASP,GLU)

=== CONTACT ANALYSIS ===
    contact_pairs         : Inter-chain residue contacts (CB-CB <= 8A)
    sidechain_contacts    : Side chain contacts (distance <= 6A)
    hydrogen_bonds        : Hydrogen bonds (polar-polar <= 4A)
    salt_bridges         : Salt bridges (charged-charged <= 5A)

ALPHAFOLD CONFIDENCE INTERPRETATION (EBI Guidelines):

    VERY HIGH CONFIDENCE (Dark Blue, pLDDT > 90):
    - Individual domains are highly accurate
    - Side chain positions generally accurate
    - Suitable for drug design and detailed analysis

    CONFIDENT (Light Blue, pLDDT 70-90):
    - Generally accurate backbone
    - Some side chain uncertainty
    - Good for functional analysis

    LOW CONFIDENCE (Yellow, pLDDT 50-70):
    - Backbone generally correct
    - Side chains often incorrect
    - Use with caution for detailed analysis

    VERY LOW CONFIDENCE (Orange/Red, pLDDT < 50):
    - Often incorrect structure
    - Should not be used for detailed analysis
    - May indicate disordered regions

INTERFACE CONFIDENCE GUIDELINES:

    HIGH CONFIDENCE INTERFACES:
    - iptm > 0.7 AND min_inter_chain_pae < 10
    - Both proteins have pLDDT > 70
    - mpDockQ > 120
    
    MEDIUM CONFIDENCE INTERFACES:
    - iptm 0.4-0.7 OR min_inter_chain_pae 10-15
    - At least one protein pLDDT > 70
    - mpDockQ 100-120
    
    LOW CONFIDENCE INTERFACES:
    - iptm < 0.4 OR min_inter_chain_pae > 15
    - Low individual protein confidence
    - mpDockQ < 100

PRACTICAL USAGE EXAMPLES:

    # Publication-quality interactions (high confidence)
    python -m af_analysis.interaction_report --output_dir results --cutoff 5.0
    
    # Drug target validation (medium-high confidence)
    python -m af_analysis.interaction_report --output_dir results --cutoff 8.0
    
    # Pathway analysis (include medium confidence)
    python -m af_analysis.interaction_report --output_dir results --cutoff 12.0
    
    # Exploratory screening (permissive)
    python -m af_analysis.interaction_report --output_dir results --cutoff 20.0

TECHNICAL DETAILS:
    - Interface detection: 8A CB-CB cutoff (CA for Glycine)
    - mpDockQ = mean_interface_pLDDT * log10(contact_count)
    - PAE filtering: intra-chain regions masked to 50A
    - Amino acid classification by chemical properties

REFERENCES:
    - AlphaFold Training: https://www.ebi.ac.uk/training/online/courses/alphafold/
    - Jumper et al. Nature 2021 (AlphaFold2 methodology)
    - Evans et al. bioRxiv 2021 (AlphaFold multimer)
"""
    print(help_text)


def open_text_maybe_gzip(path: str):
    """Open plain-text or gzipped structure files in text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def find_ranked_pdb(result_subdir: str) -> Optional[str]:
    """Return the best available ranked structure file in one APD/AF output dir."""
    preferred = [
        "ranked_0.pdb",
        "ranked_model_0.pdb",
        "ranked_0.pdb.gz",
        "ranked_model_0.pdb.gz",
        "ranked_0.cif",
        "ranked_0.mmcif",
    ]
    for name in preferred:
        path = os.path.join(result_subdir, name)
        if os.path.exists(path):
            return path

    ranked_candidates: List[str] = []
    for dirpath, dirnames, filenames in os.walk(result_subdir):
        rel = os.path.relpath(dirpath, result_subdir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            dirnames[:] = []
            continue
        for filename in filenames:
            lower = filename.lower()
            if "ranked" in lower and lower.endswith((".pdb", ".pdb.gz", ".cif", ".mmcif")):
                ranked_candidates.append(os.path.join(dirpath, filename))
    return sorted(ranked_candidates)[0] if ranked_candidates else None


def _is_ignored_report_dir(name: str) -> bool:
    return (
        not name
        or name.startswith(".")
        or name.startswith("_archive")
        or name in {"__pycache__", "tmp", "logs", "input", "af3_json"}
    )


def _find_best_analysis_dir(job_root: str) -> Tuple[str, str]:
    """Return (analysis_dir, reason) for one top-level job folder.

    The analysis directory may be the top-level folder itself or a nested AF/APD
    output folder. The reason is used only for diagnostics when a job cannot be
    analyzed.
    """
    candidates: List[Tuple[int, int, str]] = []
    missing_ranked: List[str] = []
    max_depth = int(os.environ.get("APD_REPORT_MAXDEPTH", "5"))

    for dirpath, dirnames, filenames in os.walk(job_root):
        dirnames[:] = [d for d in dirnames if not _is_ignored_report_dir(d)]
        rel = os.path.relpath(dirpath, job_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        has_ranking = "ranking_debug.json" in filenames
        has_ranked = find_ranked_pdb(dirpath) is not None
        if has_ranking and has_ranked:
            candidates.append((depth, -int(os.path.getmtime(dirpath)), dirpath))
        elif has_ranking:
            missing_ranked.append(dirpath)

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][2], "ok"
    if missing_ranked:
        missing_ranked.sort(key=lambda x: (x.count(os.sep), x))
        return missing_ranked[0], "ranking_debug_without_ranked_structure"
    return job_root, "missing_ranking_debug"


def find_job_dirs(root: str) -> List[Tuple[str, str]]:
    """Find top-level AlphaPulldown/AF jobs and return (analysis_path, stable_job_name).

    The previous implementation returned only folders that already had both
    ranking_debug.json and a ranked structure, so incomplete/misnamed folders
    disappeared from the denominator. This version keeps the top-level job count
    stable and lets the main loop report why a job was skipped.
    """
    job_dirs: List[Tuple[str, str]] = []
    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        return job_dirs

    top_level = []
    for item in sorted(os.listdir(root_abs)):
        if _is_ignored_report_dir(item):
            continue
        path = os.path.join(root_abs, item)
        if os.path.isdir(path):
            top_level.append((item, path))

    # If output_dir itself is a single job folder, support that too.
    if not top_level and os.path.exists(os.path.join(root_abs, "ranking_debug.json")):
        return [(root_abs, os.path.basename(root_abs))]

    analyzable = 0
    missing = []
    for job_name, job_root in top_level:
        analysis_dir, reason = _find_best_analysis_dir(job_root)
        if reason == "ok":
            analyzable += 1
        else:
            missing.append((job_name, reason, analysis_dir))
        job_dirs.append((analysis_dir, job_name))

    if missing:
        skipped_path = os.path.join(root_abs, "report_omitted_or_incomplete_jobs.tsv")
        try:
            with open(skipped_path, "w", encoding="utf-8") as handle:
                handle.write("job\treason\tchecked_path\n")
                for job_name, reason, checked_path in missing:
                    handle.write(f"{job_name}\t{reason}\t{checked_path}\n")
            print(
                f"Report discovery: {len(top_level)} top-level job folders, "
                f"{analyzable} analyzable, {len(missing)} missing ranking/ranked evidence."
            )
            print(f"Omitted/incomplete job list: {skipped_path}")
        except Exception as exc:
            print(f"Warning: could not write omitted job report: {exc}")

    return sorted(job_dirs, key=lambda item: item[1])


def _aggregate_alphajudge_column(series: pd.Series, column_name: str) -> Optional[float]:
    """Aggregate one AlphaJudge numeric column to a single job-level value."""
    if series.empty:
        return None

    lower = column_name.lower()
    if "pae" in lower:
        return float(series.min())
    if any(token in lower for token in ["dockq", "ipsae", "lis", "score", "iptm", "plddt", "contact", "area", "hb", "sb", "sc"]):
        return float(series.max())
    return float(series.mean())


def load_alphajudge_features(output_dir: str) -> pd.DataFrame:
    """Load and aggregate per-job AlphaJudge interfaces.csv files under output_dir.

    AlphaJudge can be written directly under output/<job>/interfaces.csv or one
    level deeper when an AF runner nests the result folder. We aggregate by the
    top-level job name so the merge matches the report's stable `jobs` column.
    """
    records_by_job: Dict[str, List[pd.DataFrame]] = {}
    root_abs = os.path.abspath(output_dir)
    max_depth = int(os.environ.get("APD_REPORT_MAXDEPTH", "5"))

    for dirpath, dirnames, filenames in os.walk(root_abs):
        dirnames[:] = [d for d in dirnames if not _is_ignored_report_dir(d)]
        rel = os.path.relpath(dirpath, root_abs)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "interfaces.csv" not in filenames:
            continue
        interfaces_csv = os.path.join(dirpath, "interfaces.csv")
        if not os.path.exists(interfaces_csv) or os.path.getsize(interfaces_csv) == 0:
            continue
        parts = [] if rel == "." else rel.split(os.sep)
        job_name = parts[0] if parts else os.path.basename(root_abs)
        try:
            aj_df = pd.read_csv(interfaces_csv)
        except Exception as e:
            print(f"  Warning: Could not read AlphaJudge file {interfaces_csv}: {e}")
            continue
        if aj_df.empty:
            continue
        records_by_job.setdefault(job_name, []).append(aj_df)

    records: List[Dict] = []
    for job_name, frames in sorted(records_by_job.items()):
        aj_df = pd.concat(frames, ignore_index=True)
        row: Dict[str, Optional[float]] = {
            "jobs": job_name,
            "alphajudge_rows": int(len(aj_df)),
        }
        numeric_cols = [
            col for col in aj_df.columns
            if pd.api.types.is_numeric_dtype(aj_df[col])
        ]
        for col in numeric_cols:
            values = aj_df[col].dropna()
            if values.empty:
                continue
            agg_val = _aggregate_alphajudge_column(values, col)
            if agg_val is not None:
                row[f"aj_{col}"] = agg_val
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["jobs"])
    return pd.DataFrame(records)



def _link_or_copy_for_alphajudge(src: Optional[str], dst: str) -> bool:
    if not src or not os.path.exists(src) or os.path.getsize(src) == 0:
        return False
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if os.path.exists(dst) or os.path.islink(dst):
            os.unlink(dst)
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False


def prepare_af3_for_alphajudge_layout(job_dir: str) -> int:
    """Add AlphaJudge-compatible aliases for prefixed AF3 outputs."""
    if not os.path.isdir(job_dir):
        return 0
    summaries = [p for p in glob.glob(os.path.join(job_dir, '*_summary_confidences.json')) if not os.path.basename(p).startswith('ranked_') and os.path.getsize(p) > 0]
    plain_summary = os.path.join(job_dir, 'summary_confidences.json')
    if os.path.exists(plain_summary) and os.path.getsize(plain_summary) > 0:
        summaries.append(plain_summary)
    if not summaries:
        return 0
    summary = sorted(summaries, key=lambda p: ('seed-' in p, len(os.path.basename(p))))[0]
    base = os.path.basename(summary)
    prefix = os.path.basename(job_dir) if base == 'summary_confidences.json' else base[:-len('_summary_confidences.json')]
    conf = os.path.join(job_dir, f'{prefix}_confidences.json')
    if not os.path.exists(conf):
        confs = [p for p in glob.glob(os.path.join(job_dir, '*_confidences.json')) if not p.endswith('_summary_confidences.json') and not os.path.basename(p).startswith('ranked_')]
        conf = sorted(confs, key=lambda p: len(os.path.basename(p)))[0] if confs else None
    model = os.path.join(job_dir, f'{prefix}_model.cif')
    if not os.path.exists(model):
        models = []
        for pat in ('*.cif', '*.mmcif', '*.pdb'):
            models.extend(glob.glob(os.path.join(job_dir, pat)))
        models = [p for p in models if not os.path.basename(p).startswith('ranked_')]
        model = sorted(models, key=lambda p: len(os.path.basename(p)))[0] if models else None
    changed = 0
    for src, name in ((summary, 'ranked_0_summary_confidences.json'), (conf, 'ranked_0_confidences.json'), (model, 'ranked_0_model.cif')):
        changed += bool(_link_or_copy_for_alphajudge(src, os.path.join(job_dir, name)))
    summary_data = load_json(summary) or {}
    data = load_json(os.path.join(job_dir, f'{prefix}_data.json')) or {}
    seeds = data.get('modelSeeds') or data.get('model_seeds') or []
    try:
        seed = int(seeds[0]) if isinstance(seeds, list) and seeds else 1
    except Exception:
        seed = 1
    sample_dir = os.path.join(job_dir, f'seed-{seed}_sample-0')
    for src, name in ((summary, 'summary_confidences.json'), (conf, 'confidences.json'), (model, 'model.cif')):
        changed += bool(_link_or_copy_for_alphajudge(src, os.path.join(sample_dir, name)))
    ranking_csv = os.path.join(job_dir, 'ranking_scores.csv')
    if not os.path.exists(ranking_csv) or os.path.getsize(ranking_csv) == 0:
        score = _safe_float(summary_data.get('ranking_score'), None) or _safe_float(summary_data.get('confidence_score'), None)
        if score is None:
            iptm = _safe_float(summary_data.get('iptm'), None)
            ptm = _safe_float(summary_data.get('ptm'), None)
            score = 0.8 * iptm + 0.2 * ptm if iptm is not None and ptm is not None else 0.0
        try:
            with open(ranking_csv, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['seed', 'sample', 'ranking_score'])
                writer.writerow([seed, 0, score])
            changed += 1
        except Exception:
            pass
    return changed


def run_alphajudge_for_job_dirs(
    job_dirs: List[Tuple[str, str]],
    models_to_analyse: str = "best",
    contact_thresh: float = 8.0,
    pae_filter: float = 100.0,
    ipsae_pae_cutoff: float = 10.0,
    cores: int = 1,
    recursive: bool = False,
    summary_csv: Optional[str] = None,
    extra_flags: Optional[str] = None,
    force: bool = False,
) -> None:
    """Run AlphaJudge on analyzable job dirs before merging interfaces.csv."""
    alphajudge_bin = shutil.which("alphajudge")
    if not alphajudge_bin:
        print("WARNING: alphajudge not found in PATH; cannot run AlphaJudge backfill.")
        return

    total = 0
    ran = 0
    skipped = 0
    failed = 0
    for job_entry in job_dirs:
        job_dir = job_entry[0] if isinstance(job_entry, (tuple, list)) else str(job_entry)
        job_name = job_entry[1] if isinstance(job_entry, (tuple, list)) and len(job_entry) > 1 else os.path.basename(job_dir)
        if not os.path.isdir(job_dir):
            continue
        if find_ranked_pdb(job_dir) is None:
            continue
        total += 1
        interfaces_csv = os.path.join(job_dir, "interfaces.csv")
        if os.path.exists(interfaces_csv) and os.path.getsize(interfaces_csv) > 0 and not force:
            skipped += 1
            continue
        prepare_alphajudge_inputs(job_dir)
        af3_changed = prepare_af3_for_alphajudge_layout(job_dir)
        if af3_changed:
            print(f"  Prepared AF3 AlphaJudge-compatible aliases for {job_name}: {af3_changed} file(s).")
        log_path = os.path.join(job_dir, "alphajudge.log")
        cmd = [
            alphajudge_bin,
            job_dir,
            "--models_to_analyse", str(models_to_analyse),
            "--contact_thresh", str(contact_thresh),
            "--pae_filter", str(pae_filter),
            "--ipsae_pae_cutoff", str(ipsae_pae_cutoff),
            "--cores", str(cores),
        ]
        if recursive:
            cmd.append("-r")
        if summary_csv:
            cmd.extend(["-o", str(summary_csv)])
        if extra_flags:
            import shlex
            cmd.extend(shlex.split(extra_flags))
        print(f"Running AlphaJudge {ran + 1}/{total}: {job_name}")
        try:
            with open(log_path, "w", encoding="utf-8") as log_handle:
                proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        except Exception as exc:
            failed += 1
            print(f"  Warning: AlphaJudge could not start for {job_name}: {exc}")
            continue
        if proc.returncode != 0:
            failed += 1
            print(f"  Warning: AlphaJudge failed for {job_name} with exit code {proc.returncode}; see {log_path}")
            continue
        if not (os.path.exists(interfaces_csv) and os.path.getsize(interfaces_csv) > 0):
            failed += 1
            print(f"  Warning: AlphaJudge completed for {job_name} but interfaces.csv was not created; see {log_path}")
            continue
        ran += 1

    print(f"AlphaJudge summary: {ran} run, {skipped} already present, {failed} failed/unavailable, {total} analyzable jobs checked.")

def merge_alphajudge_features(base_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Merge aggregated AlphaJudge features into the main results table."""
    aj_df = load_alphajudge_features(output_dir)
    if aj_df.empty or len(aj_df.columns) <= 1:
        print("No AlphaJudge features found to merge.")
        return base_df

    merged = base_df.merge(aj_df, on="jobs", how="left")
    matched = int(merged["alphajudge_rows"].notna().sum()) if "alphajudge_rows" in merged.columns else 0
    print(f"Merged AlphaJudge features for {matched}/{len(merged)} jobs.")
    return merged

def _read_json_file(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_json_file(path: str, payload: Any) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return True
    except Exception:
        return False


def _model_score_from_ranking(ranking_data: Dict[str, Any], key: str, model_name: str, default=None):
    values = ranking_data.get(key)
    if isinstance(values, dict) and model_name in values:
        return to_scalar(values.get(model_name), default)
    return default


def _load_pickle_scores(result_subdir: str, model_name: str) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    for suffix, opener in ((".pkl", open), (".pkl.gz", gzip.open)):
        pkl_path = os.path.join(result_subdir, f"result_{model_name}{suffix}")
        if not os.path.exists(pkl_path):
            continue
        try:
            with opener(pkl_path, "rb") as handle:
                result_data = pickle.load(handle)
            if isinstance(result_data, dict):
                for key in ("ptm", "iptm", "ranking_confidence", "iptm+ptm"):
                    if key in result_data:
                        scores[key] = to_scalar(result_data.get(key), None)
            break
        except Exception:
            continue
    return scores


def normalize_pae_json_for_alphajudge(result_subdir: str, model_name: str) -> bool:
    """Add missing score keys to AF2 PAE JSONs for AlphaJudge compatibility.

    Compact AlphaPulldown PAE JSONs may contain only predicted_aligned_error and
    max_predicted_aligned_error. Some AlphaJudge versions expect ptm/iptm in the
    same JSON and otherwise fail with KeyError: 'ptm'.
    """
    candidate_paths = [os.path.join(result_subdir, f"pae_{model_name}.json")]
    match = re.search(r"model_(\d+)", model_name)
    if match:
        n = match.group(1)
        candidate_paths.extend([
            os.path.join(result_subdir, f"pae_model_{n}_ptm_pred_0.json"),
            os.path.join(result_subdir, f"pae_model_{n}.json"),
        ])

    ranking_data = _read_json_file(os.path.join(result_subdir, "ranking_debug.json")) or {}
    pickle_scores = _load_pickle_scores(result_subdir, model_name)
    ptm = _model_score_from_ranking(ranking_data, "ptm", model_name, pickle_scores.get("ptm", 0.0))
    iptm = _model_score_from_ranking(ranking_data, "iptm", model_name, pickle_scores.get("iptm", 0.0))
    iptm_ptm = _model_score_from_ranking(
        ranking_data,
        "iptm+ptm",
        model_name,
        pickle_scores.get("ranking_confidence", pickle_scores.get("iptm+ptm")),
    )

    changed_any = False
    for pae_path in dict.fromkeys(candidate_paths):
        if not os.path.exists(pae_path) or os.path.getsize(pae_path) == 0:
            continue
        payload = _read_json_file(pae_path)
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
        if changed and _write_json_file(pae_path, payload):
            changed_any = True
    return changed_any


def prepare_alphajudge_inputs(job_dir: str) -> None:
    """Patch PAE-only JSON files so AlphaJudge can run without KeyError: ptm."""
    ranking_data = _read_json_file(os.path.join(job_dir, "ranking_debug.json")) or {}
    order = ranking_data.get("order") or []
    if not order:
        order = sorted({name[4:-5] for name in os.listdir(job_dir) if name.startswith("pae_") and name.endswith(".json")})
    changed = 0
    for model_name in order:
        if normalize_pae_json_for_alphajudge(job_dir, str(model_name)):
            changed += 1
    if changed:
        print(f"Prepared {changed} PAE JSON file(s) for AlphaJudge compatibility in {job_dir}")


def normalize_seq_lengths(seq_lengths: List[int], matrix_size: int, context: str = "") -> Optional[List[int]]:
    """Normalize sequence lengths to match PAE matrix size."""
    valid_lengths = []
    for length in seq_lengths:
        try:
            length_int = int(length)
        except (TypeError, ValueError):
            continue
        if length_int > 0:
            valid_lengths.append(length_int)

    if len(valid_lengths) < 2:
        if context:
            print(f"  Warning: {context} has fewer than 2 valid chain lengths: {seq_lengths}")
        return None

    total = sum(valid_lengths)
    if total == matrix_size:
        return valid_lengths

    if total < matrix_size:
        remainder = matrix_size - total
        valid_lengths.append(remainder)
        if context:
            print(
                f"  Warning: {context} length sum ({total}) < PAE size ({matrix_size}); "
                f"appended inferred chain length {remainder}."
            )
        return valid_lengths

    overflow = total - matrix_size
    if context:
        print(
            f"  Warning: {context} length sum ({total}) > PAE size ({matrix_size}); "
            "trimming terminal chain lengths to fit."
        )

    while overflow > 0 and valid_lengths:
        last_len = valid_lengths[-1]
        if last_len > overflow:
            valid_lengths[-1] = last_len - overflow
            overflow = 0
        else:
            overflow -= last_len
            valid_lengths.pop()

    if overflow > 0 or len(valid_lengths) < 2 or sum(valid_lengths) != matrix_size:
        if context:
            print(f"  Warning: Could not normalize sequence lengths for {context}; skipping job.")
        return None

    return valid_lengths


def examine_inter_pae(pae_mtx: np.ndarray, seq_lengths: List[int], cutoff: float) -> bool:
    """Check if there are inter-chain PAE values below cutoff."""
    if pae_mtx is None or len(seq_lengths) < 2:
        return False

    if not isinstance(pae_mtx, np.ndarray) or pae_mtx.ndim != 2:
        return False
    if pae_mtx.shape[0] != pae_mtx.shape[1]:
        return False
    if sum(seq_lengths) != pae_mtx.shape[0]:
        return False
    
    mtx = pae_mtx.copy()
    old_length = 0
    
    # Mask intra-chain regions
    for length in seq_lengths:
        new_length = old_length + length
        mtx[old_length:new_length, old_length:new_length] = 50
        old_length = new_length
    
    # Check if any inter-chain PAE is below cutoff
    return bool(np.any(mtx < cutoff))


def obtain_seq_lengths(result_subdir: str) -> List[int]:
    """Get sequence lengths from the best ranked PDB/mmCIF file."""
    pdb_file = find_ranked_pdb(result_subdir)
    if not pdb_file or not os.path.exists(pdb_file):
        return []
    
    try:
        # Simple PDB parsing for chain lengths
        chain_residues = {}
        
        with open_text_maybe_gzip(pdb_file) as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain_id = line[21]
                    res_num = int(line[22:26].strip())
                    
                    if chain_id not in chain_residues:
                        chain_residues[chain_id] = set()
                    chain_residues[chain_id].add(res_num)
        
        # Calculate lengths
        seq_lengths = []
        for chain_id in sorted(chain_residues.keys()):
            seq_lengths.append(len(chain_residues[chain_id]))

        if not seq_lengths:
            print(f"Warning: No chain lengths parsed from {pdb_file}")
        return seq_lengths
        
    except Exception as e:
        print(f"Warning: Could not determine sequence lengths for {pdb_file}: {e}")
        return []


def obtain_pae_and_iptm(result_subdir: str, best_model: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Extract PAE matrix and iptm score from result files."""
    pae_mtx = None
    iptm_score = None
    
    # Try to get iptm from ranking_debug.json first
    try:
        ranking_path = os.path.join(result_subdir, "ranking_debug.json")
        with open(ranking_path, 'r') as f:
            ranking_data = json.load(f)
        
        if "iptm" in ranking_data and best_model in ranking_data["iptm"]:
            iptm_score = to_scalar(ranking_data["iptm"][best_model])
    except FileNotFoundError:
        print(f"  Warning: ranking_debug.json not found in {result_subdir}")
    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse ranking_debug.json in {result_subdir}: {e}")
    except KeyError as e:
        print(f"  Warning: Missing key while reading ranking_debug.json in {result_subdir}: {e}")
    
    # Try to get PAE from separate JSON file first
    pae_json_path = os.path.join(result_subdir, f"pae_{best_model}.json")
    if os.path.exists(pae_json_path):
        try:
            with open(pae_json_path, 'r') as f:
                pae_data = json.load(f)
            pae_mtx = np.array(pae_data[0]['predicted_aligned_error'])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Warning: Could not parse {pae_json_path}: {type(e).__name__}: {e}")
    
    # If PAE not found in JSON, try pickle files
    if pae_mtx is None or iptm_score is None:
        # Try regular pickle first
        pkl_path = os.path.join(result_subdir, f"result_{best_model}.pkl")
        pkl_gz_path = os.path.join(result_subdir, f"result_{best_model}.pkl.gz")
        
        result_data = None
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    result_data = pickle.load(f)
            except Exception as e:
                print(f"  Warning: Could not read {pkl_path}: {type(e).__name__}: {e}")
        elif os.path.exists(pkl_gz_path):
            try:
                with gzip.open(pkl_gz_path, 'rb') as f:
                    result_data = pickle.load(f)
            except Exception as e:
                print(f"  Warning: Could not read {pkl_gz_path}: {type(e).__name__}: {e}")
        
        if result_data:
            if pae_mtx is None and 'predicted_aligned_error' in result_data:
                pae_mtx = np.array(result_data['predicted_aligned_error'])
            if iptm_score is None and 'iptm' in result_data:
                iptm_score = to_scalar(result_data['iptm'])
    
    return pae_mtx, iptm_score


def calculate_amino_acid_properties(residue_name: str) -> Dict[str, bool]:
    """Get amino acid properties for interface composition analysis."""
    # Amino acid classification
    polar = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
    hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
    charged = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU'}
    
    return {
        'polar': residue_name in polar,
        'hydrophobic': residue_name in hydrophobic, 
        'charged': residue_name in charged
    }


def calculate_enhanced_interface_metrics(pdb_path: str, pae_mtx: np.ndarray, seq_lengths: List[int]) -> Dict:
    """Calculate comprehensive interface metrics matching AlphaPulldown format."""
    metrics = {
        'Num_intf_residues': 0,
        'Polar': 0.0,
        'Hydrophobic': 0.0,
        'Charged': 0.0,
        'contact_pairs': 0,
        'sc': 0,  # side chain contacts
        'hb': 0,  # hydrogen bonds (approximated)
        'sb': 0,  # salt bridges (approximated)
        'int_solv_en': 0.0,  # interface solvation energy (approximated)
        'int_area': 0.0,  # interface area (approximated)
        'pi_score': 0.0,  # interface quality score
        'average_interface_pae': None,
        'average_interface_plddt': None,
        'min_inter_chain_pae': None,
        'protein1_plddt': None,
        'protein2_plddt': None
    }
    
    if not os.path.exists(pdb_path) or pae_mtx is None:
        return metrics

    if not isinstance(pae_mtx, np.ndarray) or pae_mtx.ndim != 2 or pae_mtx.shape[0] != pae_mtx.shape[1]:
        print(f"Warning: Invalid PAE matrix shape for {pdb_path}: {getattr(pae_mtx, 'shape', None)}")
        return metrics

    if sum(seq_lengths) != pae_mtx.shape[0]:
        print(
            f"Warning: Sequence lengths ({sum(seq_lengths)}) do not match PAE matrix size "
            f"({pae_mtx.shape[0]}) for {pdb_path}"
        )
        return metrics
    
    try:
        # Parse PDB to get detailed residue information
        chains_data = {}
        residue_mapping = {}
        
        with open_text_maybe_gzip(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain_id = line[21]
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip()) - 1  # Convert to 0-based for PAE indexing
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    plddt = float(line[60:66])
                    
                    if chain_id not in chains_data:
                        chains_data[chain_id] = {}
                    
                    if res_num not in chains_data[chain_id]:
                        chains_data[chain_id][res_num] = {
                            'name': res_name,
                            'plddt': plddt,
                            'atoms': {}
                        }
                    
                    chains_data[chain_id][res_num]['atoms'][atom_name] = [x, y, z]
        
        # Calculate individual protein pLDDT scores
        chain_ids = sorted(chains_data.keys())
        if len(chain_ids) >= 2:
            chain1_plddts = []
            chain2_plddts = []
            
            for res_num in chains_data[chain_ids[0]]:
                chain1_plddts.append(chains_data[chain_ids[0]][res_num]['plddt'])
            
            for res_num in chains_data[chain_ids[1]]:
                chain2_plddts.append(chains_data[chain_ids[1]][res_num]['plddt'])
            
            metrics['protein1_plddt'] = float(np.mean(chain1_plddts)) if chain1_plddts else None
            metrics['protein2_plddt'] = float(np.mean(chain2_plddts)) if chain2_plddts else None
        
        # Create sequential mapping for PAE matrix indexing
        total_residues = 0
        for chain_id in sorted(chains_data.keys()):
            for res_num in sorted(chains_data[chain_id].keys()):
                residue_mapping[(chain_id, res_num)] = total_residues
                total_residues += 1
        
        # Find interface residues and analyze contacts
        interface_residues = []
        contact_pairs = 0
        sc_contacts = 0
        hb_contacts = 0
        sb_contacts = 0
        
        cutoff = 8.0
        
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                chain_a_id = chain_ids[i]
                chain_b_id = chain_ids[j]
                
                for res_a in chains_data[chain_a_id]:
                    for res_b in chains_data[chain_b_id]:
                        # Get CB coordinates (CA for Glycine)
                        atoms_a = chains_data[chain_a_id][res_a]['atoms']
                        atoms_b = chains_data[chain_b_id][res_b]['atoms']
                        
                        cb_a = atoms_a.get('CB', atoms_a.get('CA'))
                        cb_b = atoms_b.get('CB', atoms_b.get('CA'))
                        
                        if cb_a and cb_b:
                            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(cb_a, cb_b)))
                            
                            if distance <= cutoff:
                                contact_pairs += 1
                                
                                # Add to interface residues
                                if (chain_a_id, res_a) not in interface_residues:
                                    interface_residues.append((chain_a_id, res_a))
                                if (chain_b_id, res_b) not in interface_residues:
                                    interface_residues.append((chain_b_id, res_b))
                                
                                # Analyze contact types
                                res_name_a = chains_data[chain_a_id][res_a]['name']
                                res_name_b = chains_data[chain_b_id][res_b]['name']
                                
                                # Side chain contact (approximation)
                                if distance <= 6.0:
                                    sc_contacts += 1
                                
                                # Hydrogen bond approximation (polar-polar, distance < 4Å)
                                props_a = calculate_amino_acid_properties(res_name_a)
                                props_b = calculate_amino_acid_properties(res_name_b)
                                
                                if distance <= 4.0 and (props_a['polar'] or props_b['polar']):
                                    hb_contacts += 1
                                
                                # Salt bridge approximation (charged-charged, distance < 5Å)
                                if distance <= 5.0 and props_a['charged'] and props_b['charged']:
                                    sb_contacts += 1
        
        metrics['contact_pairs'] = contact_pairs
        metrics['sc'] = sc_contacts
        metrics['hb'] = hb_contacts
        metrics['sb'] = sb_contacts
        
        # Analyze interface composition
        if interface_residues:
            total_interface = len(interface_residues)
            polar_count = 0
            hydrophobic_count = 0
            charged_count = 0
            
            interface_plddts = []
            interface_pae_values = []
            
            for chain_id, res_num in interface_residues:
                res_name = chains_data[chain_id][res_num]['name']
                plddt = chains_data[chain_id][res_num]['plddt']
                
                # Count amino acid types
                props = calculate_amino_acid_properties(res_name)
                if props['polar']:
                    polar_count += 1
                if props['hydrophobic']:
                    hydrophobic_count += 1
                if props['charged']:
                    charged_count += 1
                
                interface_plddts.append(plddt)
                
                # Get PAE values for inter-chain contacts
                if (chain_id, res_num) in residue_mapping:
                    res_idx = residue_mapping[(chain_id, res_num)]
                    
                    for other_chain, other_res in interface_residues:
                        if other_chain != chain_id and (other_chain, other_res) in residue_mapping:
                            other_idx = residue_mapping[(other_chain, other_res)]
                            if res_idx < pae_mtx.shape[0] and other_idx < pae_mtx.shape[1]:
                                interface_pae_values.append(pae_mtx[res_idx, other_idx])
            
            # Calculate composition fractions
            metrics['Num_intf_residues'] = total_interface
            metrics['Polar'] = polar_count / total_interface if total_interface > 0 else 0.0
            metrics['Hydrophobic'] = hydrophobic_count / total_interface if total_interface > 0 else 0.0
            metrics['Charged'] = charged_count / total_interface if total_interface > 0 else 0.0
            
            # Calculate averages and minimum PAE
            metrics['average_interface_plddt'] = float(np.mean(interface_plddts)) if interface_plddts else None
            metrics['average_interface_pae'] = float(np.mean(interface_pae_values)) if interface_pae_values else None
            metrics['min_inter_chain_pae'] = float(np.min(interface_pae_values)) if interface_pae_values else None
            
            # Approximate interface area (based on contact count)
            metrics['int_area'] = contact_pairs * 20.0  # Rough approximation
            
            # Approximate solvation energy (negative for favorable)
            metrics['int_solv_en'] = -1.0 * contact_pairs * 0.5
            
            # PI score (approximation based on pLDDT and contacts)
            if metrics['average_interface_plddt']:
                metrics['pi_score'] = metrics['average_interface_plddt'] / 100.0 * math.log10(max(contact_pairs, 1))
    
    except Exception as e:
        print(f"Warning: Could not calculate enhanced interface metrics for {pdb_path}: {e}")
    
    return metrics


def calculate_mpdockq(pdb_path: str) -> Optional[float]:
    """Calculate mpDockQ score for the structure."""
    if not os.path.exists(pdb_path):
        return None
    
    try:
        # Parse PDB and extract interface residues and pLDDT scores
        chains_data = {}
        
        with open_text_maybe_gzip(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain_id = line[21]
                    atom_name = line[12:16].strip()
                    res_num = int(line[22:26].strip())
                    
                    # Only take CB atoms (CA for Glycine)
                    if atom_name == 'CB' or (atom_name == 'CA' and line[17:20].strip() == 'GLY'):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        plddt = float(line[60:66])  # B-factor contains pLDDT
                        
                        if chain_id not in chains_data:
                            chains_data[chain_id] = {'coords': [], 'plddts': []}
                        
                        chains_data[chain_id]['coords'].append([x, y, z])
                        chains_data[chain_id]['plddts'].append(plddt)
        
        # Convert to numpy arrays
        for chain_id in chains_data:
            chains_data[chain_id]['coords'] = np.array(chains_data[chain_id]['coords'])
            chains_data[chain_id]['plddts'] = np.array(chains_data[chain_id]['plddts'])
        
        if len(chains_data) < 2:
            return None
        
        # Find interface residues (within 8Å across chains)
        interface_residues = []
        contact_count = 0
        cutoff = 8.0
        
        chain_ids = list(chains_data.keys())
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                chain_a = chains_data[chain_ids[i]]
                chain_b = chains_data[chain_ids[j]]
                
                if len(chain_a['coords']) == 0 or len(chain_b['coords']) == 0:
                    continue
                
                # Calculate distances between all pairs
                coords_a = chain_a['coords']
                coords_b = chain_b['coords']
                
                # Broadcasting to calculate all pairwise distances
                diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=2))
                
                # Find contacts within cutoff
                contacts = np.where(distances <= cutoff)
                
                if len(contacts[0]) > 0:
                    contact_count += len(contacts[0])
                    
                    # Add interface residues
                    for idx_a in np.unique(contacts[0]):
                        interface_residues.append((chain_ids[i], idx_a))
                    for idx_b in np.unique(contacts[1]):
                        interface_residues.append((chain_ids[j], idx_b))
        
        if contact_count == 0 or len(interface_residues) == 0:
            return None
        
        # Calculate mean pLDDT of interface residues
        interface_plddts = []
        for chain_id, res_idx in interface_residues:
            if chain_id in chains_data and res_idx < len(chains_data[chain_id]['plddts']):
                interface_plddts.append(chains_data[chain_id]['plddts'][res_idx])
        
        if not interface_plddts:
            return None
        
        mean_interface_plddt = np.mean(interface_plddts)
        
        # Calculate mpDockQ: mean_interface_plddt * log10(contacts)
        mpdockq = mean_interface_plddt * math.log10(max(contact_count, 1))
        
        return float(mpdockq)
        
    except Exception as e:
        print(f"Warning: Could not calculate mpDockQ for {pdb_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create csv compatible with AlphaPulldown", add_help=False)
    parser.add_argument("--output_dir", help="Directory containing AlphaPulldown predictions")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Inter-chain PAE cutoff (default: 10.0)")
    parser.add_argument("--run-alphajudge", action="store_true", default=os.environ.get("APD_REPORT_RUN_ALPHAJUDGE", "0").lower() in {"1", "true", "yes", "on"}, help="Run AlphaJudge before merging interfaces.csv")
    parser.add_argument("--force-alphajudge", action="store_true", help="Re-run AlphaJudge even when interfaces.csv already exists")
    parser.add_argument("--alphajudge-models", default=os.environ.get("ALPHAJUDGE_MODELS_TO_ANALYSE", "best"), help="AlphaJudge --models_to_analyse value")
    parser.add_argument("--alphajudge-contact-thresh", type=float, default=float(os.environ.get("ALPHAJUDGE_CONTACT_THRESH", "8.0")), help="AlphaJudge contact threshold")
    parser.add_argument("--alphajudge-pae-filter", type=float, default=float(os.environ.get("ALPHAJUDGE_PAE_FILTER", "100.0")), help="AlphaJudge --pae_filter value")
    parser.add_argument("--alphajudge-ipsae-pae-cutoff", type=float, default=float(os.environ.get("ALPHAJUDGE_IPSAE_PAE_CUTOFF", "10.0")), help="AlphaJudge --ipsae_pae_cutoff value")
    parser.add_argument("--alphajudge-cores", type=int, default=int(os.environ.get("ALPHAJUDGE_CORES", "1")), help="AlphaJudge --cores value; 0 means all available cores in AlphaJudge")
    parser.add_argument("--alphajudge-recursive", action="store_true", default=os.environ.get("ALPHAJUDGE_RECURSIVE", "0").lower() in {"1", "true", "yes", "on"}, help="Pass -r/--recursive to AlphaJudge")
    parser.add_argument("--alphajudge-summary", default=os.environ.get("ALPHAJUDGE_SUMMARY", ""), help="Pass -o/--summary FILE to AlphaJudge")
    parser.add_argument("--alphajudge-extra-flags", default=os.environ.get("ALPHAJUDGE_EXTRA_FLAGS", ""), help="Additional raw flags appended to alphajudge")
    parser.add_argument("--help", action="store_true", help="Show basic help message")
    parser.add_argument("--help-full", action="store_true", help="Show comprehensive help with detailed explanations")
    
    args = parser.parse_args()
    
    # Handle help requests
    if args.help:
        print_help()
        return
    
    if hasattr(args, 'help_full') and args.help_full:
        print_help_full()
        return
    
    # Check required arguments
    if not args.output_dir:
        print("Error: --output_dir is required")
        print("Use --help for usage information")
        sys.exit(1)

    if args.cutoff <= 0:
        print("Error: --cutoff must be greater than 0")
        sys.exit(1)

    print(f"Analyzing predictions in: {args.output_dir}")
    print(f"Using PAE cutoff: {args.cutoff}")

    job_dirs = find_job_dirs(args.output_dir)
    print(f"Found {len(job_dirs)} job directories")

    if args.run_alphajudge:
        run_alphajudge_for_job_dirs(
            job_dirs,
            models_to_analyse=args.alphajudge_models,
            contact_thresh=args.alphajudge_contact_thresh,
            pae_filter=args.alphajudge_pae_filter,
            ipsae_pae_cutoff=args.alphajudge_ipsae_pae_cutoff,
            cores=args.alphajudge_cores,
            recursive=args.alphajudge_recursive,
            summary_csv=args.alphajudge_summary or None,
            extra_flags=args.alphajudge_extra_flags or None,
            force=args.force_alphajudge,
        )

    results = []
    
    for count, job_entry in enumerate(job_dirs, 1):
        # Backward-compatible with older find_job_dirs implementations that
        # returned only a path, and fixed for the newer (path, stable_name) form.
        if isinstance(job_entry, (tuple, list)):
            job_dir = job_entry[0]
            job_name = job_entry[1] if len(job_entry) > 1 else os.path.basename(job_dir)
        else:
            job_dir = job_entry
            job_name = os.path.basename(job_dir)
        print(f"Processing {count}/{len(job_dirs)}: {job_name}")
        
        try:
            # Load ranking debug to get best model and scores
            ranking_debug_path = os.path.join(job_dir, "ranking_debug.json")
            if not os.path.exists(ranking_debug_path):
                print(f"  Warning: No ranking_debug.json found for {job_name}; skipped from scored CSV")
                continue

            ranked_pdb_path = find_ranked_pdb(job_dir)
            if ranked_pdb_path is None:
                print(f"  Warning: No ranked PDB/CIF found for {job_name}; skipped from scored CSV")
                continue
            
            with open(ranking_debug_path, 'r') as f:
                ranking_data = json.load(f)
            
            if "order" not in ranking_data or not ranking_data["order"]:
                print(f"  Warning: No model order found in {job_name}")
                continue
                
            best_model = ranking_data["order"][0]
            
            # Get iptm+ptm score
            iptm_ptm_score = None
            if "iptm+ptm" in ranking_data and best_model in ranking_data["iptm+ptm"]:
                iptm_ptm_score = to_scalar(ranking_data["iptm+ptm"][best_model])
            
            if iptm_ptm_score is None:
                print(f"  Warning: No iptm+ptm score found for {job_name}")
                continue
            
            # Get PAE matrix and iptm score
            pae_mtx, iptm_score = obtain_pae_and_iptm(job_dir, best_model)
            
            if pae_mtx is None:
                print(f"  Warning: Could not load PAE matrix for {job_name}")
                continue
            
            # Get sequence lengths
            seq_lengths = obtain_seq_lengths(job_dir)
            normalized_seq_lengths = normalize_seq_lengths(
                seq_lengths,
                pae_mtx.shape[0],
                context=job_name
            )
            if normalized_seq_lengths is None:
                continue
            
            # Check inter-chain PAE
            passes_pae_cutoff = examine_inter_pae(pae_mtx, normalized_seq_lengths, args.cutoff)
            
            if not passes_pae_cutoff:
                print(f"  Skipping {job_name}: inter-chain PAE above cutoff")
                continue
            
            # Calculate mpDockQ and enhanced interface metrics
            mpdockq_score = calculate_mpdockq(ranked_pdb_path)
            enhanced_metrics = calculate_enhanced_interface_metrics(ranked_pdb_path, pae_mtx, normalized_seq_lengths)
            
            # Create result entry with intuitive column ordering
            result = {
                # Primary quality metrics (most important)
                "jobs": job_name,
                "min_inter_chain_pae": safe_round(enhanced_metrics['min_inter_chain_pae'], 2),
                "protein1_plddt": safe_round(enhanced_metrics['protein1_plddt'], 2),
                "protein2_plddt": safe_round(enhanced_metrics['protein2_plddt'], 2),
                "iptm": safe_round(iptm_score, 4),
                "iptm_ptm": safe_round(iptm_ptm_score, 4),
                "mpDockQ": safe_round(mpdockq_score, 2),
                "pi_score": safe_round(enhanced_metrics['pi_score'], 2, 0.0),
                
                # Interface composition and size
                "interface_residues": enhanced_metrics['Num_intf_residues'],
                "polar_fraction": safe_round(enhanced_metrics['Polar'], 3, 0.0),
                "hydrophobic_fraction": safe_round(enhanced_metrics['Hydrophobic'], 3, 0.0),
                "charged_fraction": safe_round(enhanced_metrics['Charged'], 3, 0.0),
                
                # Contact analysis (expanded names)
                "contact_pairs": enhanced_metrics['contact_pairs'],
                "sidechain_contacts": enhanced_metrics['sc'],
                "hydrogen_bonds": enhanced_metrics['hb'],
                "salt_bridges": enhanced_metrics['sb'],
                
                # Interface quality metrics
                "interface_solvation_energy": safe_round(enhanced_metrics['int_solv_en'], 2, 0.0),
                "interface_area": safe_round(enhanced_metrics['int_area'], 2, 0.0)
            }
            
            results.append(result)
            # Format output message
            iptm_str = safe_format(iptm_score, ".3f")
            mpdockq_str = safe_format(mpdockq_score, ".2f")
            contacts_str = enhanced_metrics['contact_pairs']
            pae_str = safe_format(enhanced_metrics['min_inter_chain_pae'], ".1f")
            print(f"  ✓ Added {job_name} (iptm: {iptm_str}, iptm+ptm: {safe_format(iptm_ptm_score, '.3f')}, mpDockQ: {mpdockq_str}, min_PAE: {pae_str})")
            
        except Exception as e:
            print(f"  Error processing {job_name}: {type(e).__name__}: {e}")
            continue
    
    # Create output DataFrame
    if results:
        df = pd.DataFrame(results)

        # Merge AlphaJudge-derived features (if interfaces.csv files are present)
        df = merge_alphajudge_features(df, args.output_dir)
        
        # Sort by iptm score descending
        if 'iptm' in df.columns:
            df = df.sort_values(by='iptm', ascending=False)
        
        # Save results
        cutoff_str = str(args.cutoff).replace(".", "_")
        output_filename = f"predictions_with_pae_cutoff_{cutoff_str}.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Wrote {len(df)} results to {output_path}")
        
        # Show sample results
        print("\nSample results:")
        print(df.head().to_string(index=False))
        
    else:
        print(f"\nNo results found. All models had inter-chain PAE above cutoff ({args.cutoff})")
        
        # Create empty file with intuitive column ordering
        empty_df = pd.DataFrame(columns=[
            "jobs", "min_inter_chain_pae", "protein1_plddt", "protein2_plddt", "iptm", "iptm_ptm", "mpDockQ", "pi_score",
            "interface_residues", "polar_fraction", "hydrophobic_fraction", "charged_fraction",
            "contact_pairs", "sidechain_contacts", "hydrogen_bonds", "salt_bridges",
            "interface_solvation_energy", "interface_area"
        ])
        cutoff_str = str(args.cutoff).replace(".", "_")
        output_filename = f"predictions_with_pae_cutoff_{cutoff_str}.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        empty_df.to_csv(output_path, index=False)
        print(f"Created empty results file: {output_path}")


if __name__ == "__main__":
    main()
