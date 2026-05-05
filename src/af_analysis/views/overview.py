"""Overview page: filters, table, downloads, and optional plots."""

from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from af_analysis.plots.overview import render_overview_plots
from af_analysis.state import navigate_to_viewer
from af_analysis.utils import AlphaPulldownAnalyzer

BASE_COLUMNS_EXCLUDED = {
    "job",
    "path",
    "best_model",
    "n_models",
    "job_type",
    "interface_csv",
    "interface_summary_model",
    "interface_fill_source",
    "interface_analysis_csv",
}

RESULT_ANALYSIS_COLUMNS = [
    ("min_inter_chain_pae", "min PAE", "Minimum inter-chain PAE from the bundled interface-analysis module"),
    ("protein1_plddt", "Protein 1 pLDDT", "Average pLDDT for the first protein/chain from the interface-analysis module"),
    ("protein2_plddt", "Protein 2 pLDDT", "Average pLDDT for the second protein/chain from the interface-analysis module"),
    ("mpDockQ", "mpDockQ", "Interface-analysis mpDockQ = mean interface pLDDT × log10(inter-chain contact count)"),
    ("pi_score", "PI score", "Interface-analysis interface quality score"),
    ("interface_residues", "Intf Res", "Interface residues from the interface-analysis module"),
    ("contact_pairs", "Contacts", "Inter-chain contact pairs from the interface-analysis module"),
    ("interface_area", "Interface area", "Interface area from the interface-analysis module when predictions_with_pae_cutoff_100_0.csv is merged"),
]

ALPHAJUDGE_COLUMNS = [
    ("dockq", "DockQ", "Explicit true DockQ score; values are not normalized by AF-Analysis"),
    ("global_dockq", "AJ pDockQ/mpDockQ", "AlphaJudge pDockQ/mpDockQ-style score; not used as true DockQ in overview plots"),
    ("best_interface_pdockq2", "AJ pDQ2", "Best AlphaJudge interface pDockQ2"),
    ("best_interface_ipsae", "AJ ipSAE", "Best AlphaJudge interface ipSAE"),
    ("best_interface_lis", "AJ LIS", "Best AlphaJudge interface LIS"),
    ("interface_score", "AJ Score", "AlphaJudge/interface score"),
    ("interface_average_plddt", "AJ interface pLDDT", "AlphaJudge interface average pLDDT"),
    ("interface_residue_count", "AJ Res", "AlphaJudge interface residues"),
    ("interface_contact_pairs", "AJ Pairs", "AlphaJudge interface contact pairs"),
    ("interface_solv_energy", "AJ SolvE", "AlphaJudge interface solvation energy"),
    ("interface_polar_fraction", "AJ Polar", "AlphaJudge interface polar fraction"),
    ("interface_hydrophobic_fraction", "AJ Hydro", "AlphaJudge interface hydrophobic fraction"),
    ("interface_charged_fraction", "AJ Charged", "AlphaJudge interface charged fraction"),
    ("alphajudge_rows", "AJ rows", "Number of AlphaJudge interface rows merged by the interface-analysis module"),
    ("aj_pDockQ/mpDockQ", "AJ pDockQ/mpDockQ", "AlphaJudge pDockQ/mpDockQ column merged by the interface-analysis module"),
    ("aj_interface_pDockQ2", "AJ pDQ2", "AlphaJudge interface_pDockQ2 merged by the interface-analysis module"),
    ("aj_interface_ipSAE", "AJ ipSAE", "AlphaJudge interface_ipSAE merged by the interface-analysis module"),
    ("aj_interface_LIS", "AJ LIS", "AlphaJudge interface_LIS merged by the interface-analysis module"),
    ("aj_interface_average_plddt", "AJ interface pLDDT", "AlphaJudge interface_average_plddt merged by the interface-analysis module"),
    ("aj_interface_num_intf_residues", "AJ Res", "AlphaJudge interface_num_intf_residues merged by the interface-analysis module"),
    ("aj_interface_contact_pairs", "AJ Pairs", "AlphaJudge interface_contact_pairs merged by the interface-analysis module"),
    ("aj_interface_area", "AJ Area", "AlphaJudge interface_area merged by the interface-analysis module"),
    ("aj_interface_score", "AJ Score", "AlphaJudge interface_score merged by the interface-analysis module"),
    ("aj_average_interface_pae", "AJ Avg PAE", "AlphaJudge average interface PAE merged by the interface-analysis module"),
]


def _interface_analysis_keys() -> set[str]:
    return {"iptm", "iptm_ptm", "mean_pae", "n_models"} | {key for key, _, _ in RESULT_ANALYSIS_COLUMNS} | {
        "polar_fraction",
        "hydrophobic_fraction",
        "charged_fraction",
        "sidechain_contacts",
        "hydrogen_bonds",
        "salt_bridges",
        "interface_solvation_energy",
    }


def _alphajudge_key(key: str) -> bool:
    key_lower = str(key).lower()
    return (
        key_lower.startswith("aj_")
        or key in {key for key, _, _ in ALPHAJUDGE_COLUMNS}
        or key_lower.startswith("best_interface_")
        or key_lower.startswith("interface_") and key not in _interface_analysis_keys()
        or key_lower in {"global_dockq", "alphajudge_rows"}
    )


def get_numeric_cols_for_view(results_df: pd.DataFrame, table_view: str) -> List[str]:
    cols = []
    result_keys = _interface_analysis_keys()
    for col in results_df.columns:
        if col in BASE_COLUMNS_EXCLUDED:
            continue
        if not (pd.api.types.is_numeric_dtype(results_df[col]) and results_df[col].notna().any()):
            continue
        if table_view == "AlphaJudge":
            if _alphajudge_key(col):
                cols.append(col)
        else:
            if col in result_keys or (col in {"ptm", "confidence_score"} and not _alphajudge_key(col)):
                cols.append(col)
    return cols


def format_value(key: str, value) -> str:
    if pd.isna(value):
        return "N/A"
    if key in {"iptm", "iptm_ptm", "dockq", "global_dockq", "best_interface_pdockq2", "best_interface_ipsae", "best_interface_lis", "aj_pDockQ/mpDockQ", "aj_interface_pDockQ2", "aj_interface_ipSAE", "aj_interface_LIS", "aj_iptm", "aj_iptm_ptm", "aj_ptm", "aj_confidence_score"}:
        return f"{float(value):.3f}"
    if key in {"mean_pae", "min_inter_chain_pae", "interface_score", "interface_average_plddt", "interface_solv_energy", "interface_solvation_energy", "protein1_plddt", "protein2_plddt", "mpDockQ", "pi_score", "aj_interface_average_plddt", "aj_interface_score", "aj_average_interface_pae"}:
        return f"{float(value):.2f}"
    if key in {"interface_residue_count", "interface_contact_pairs", "interface_residues", "contact_pairs", "sidechain_contacts", "hydrogen_bonds", "salt_bridges", "n_models", "alphajudge_rows", "aj_interface_num_intf_residues", "aj_interface_contact_pairs", "aj_interface_hb", "aj_interface_sb"}:
        return f"{int(float(value))}"
    if key in {"interface_area", "aj_interface_area"}:
        return f"{float(value):.0f}"
    if key in {"interface_polar_fraction", "interface_hydrophobic_fraction", "interface_charged_fraction", "polar_fraction", "hydrophobic_fraction", "charged_fraction", "aj_interface_polar", "aj_interface_hydrophobic", "aj_interface_charged", "aj_interface_sc"}:
        return f"{float(value):.2f}"
    return str(value)


def _with_unique_display_labels(specs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Attach duplicate-safe display labels for Streamlit/Arrow tables."""
    label_counts: Dict[str, int] = {}
    for spec in specs:
        label = spec.get("label", spec.get("key", "column"))
        label_counts[label] = label_counts.get(label, 0) + 1

    seen: Dict[str, int] = {}
    for spec in specs:
        label = spec.get("label", spec.get("key", "column"))
        if label_counts.get(label, 0) <= 1:
            spec["display_label"] = label
            continue
        seen[label] = seen.get(label, 0) + 1
        key = spec.get("key", "")
        source = "interface_analysis" if key.startswith("aj_") else "AF-Analysis"
        spec["display_label"] = f"{label} ({source})"
    return specs


def _append_if_present(specs: List[Dict[str, str]], used: set[str], df: pd.DataFrame, key: str, label: str, help_text: str) -> None:
    if key in used:
        return
    if key in df.columns and df[key].notna().any():
        specs.append({"key": key, "label": label, "help": help_text})
        used.add(key)


def make_column_specs(filtered_df: pd.DataFrame, table_view: str, show_detailed: bool = False) -> List[Dict[str, str]]:
    specs = [{"key": "job", "label": "Job", "help": "Prediction/job folder"}]
    used = {"job"}

    if table_view == "AlphaJudge":
        essential_keys = {
            "dockq",
            "global_dockq",
            "best_interface_pdockq2",
            "best_interface_ipsae",
            "best_interface_lis",
            "interface_score",
            "alphajudge_rows",
            "aj_pDockQ/mpDockQ",
            "aj_interface_pDockQ2",
            "aj_interface_ipSAE",
            "aj_interface_LIS",
            "aj_interface_area",
            "aj_interface_score",
            "aj_average_interface_pae",
        }
        for key, label, help_text in ALPHAJUDGE_COLUMNS:
            if show_detailed or key in essential_keys:
                _append_if_present(specs, used, filtered_df, key, label, help_text)
        if show_detailed:
            for key in get_numeric_cols_for_view(filtered_df, "AlphaJudge"):
                _append_if_present(specs, used, filtered_df, key, key, "Additional AlphaJudge numeric column")
    else:
        _append_if_present(specs, used, filtered_df, "iptm", "ipTM", "Interface predicted TM score from the interface-analysis module / ranking_debug")
        _append_if_present(specs, used, filtered_df, "iptm_ptm", "ipTM+pTM", "ipTM+pTM / AF-Multimer ranking score from the interface-analysis module / ranking_debug")
        _append_if_present(specs, used, filtered_df, "mean_pae", "Mean PAE", "Mean inter-chain/interface PAE from AF-Analysis cache when available")
        for key, label, help_text in RESULT_ANALYSIS_COLUMNS:
            _append_if_present(specs, used, filtered_df, key, label, help_text)
        if show_detailed:
            for key, label, help_text in (
                ("polar_fraction", "Polar", "Interface polar residue fraction from the interface-analysis module"),
                ("hydrophobic_fraction", "Hydrophobic", "Interface hydrophobic residue fraction from the interface-analysis module"),
                ("charged_fraction", "Charged", "Interface charged residue fraction from the interface-analysis module"),
                ("sidechain_contacts", "Sidechain contacts", "Side-chain contacts from the interface-analysis module"),
                ("hydrogen_bonds", "H-bonds", "Hydrogen bonds from the interface-analysis module"),
                ("salt_bridges", "Salt bridges", "Salt bridges from the interface-analysis module"),
                ("interface_solvation_energy", "SolvE", "Interface solvation energy from the interface-analysis module"),
                ("ptm", "pTM", "pTM if present in the interface-analysis output / ranking metadata"),
                ("confidence_score", "Confidence score", "Confidence/ranking score if present"),
            ):
                _append_if_present(specs, used, filtered_df, key, label, help_text)

    _append_if_present(specs, used, filtered_df, "n_models", "Models", "Number of models found")
    return _with_unique_display_labels(specs)


def _render_metric_guide(table_view: str) -> None:
    with st.expander("Metric guide: value ranges and how to use them", expanded=False):
        st.markdown(
            """
Use these ranges for **triage**, not as proof of binding. The most reliable candidates usually have agreement between: good chain confidence, low inter-chain PAE, high interface confidence, and a plausible interface.

EMBL-EBI's AlphaFold guide emphasizes that **pLDDT is local chain confidence**, **PAE is relative placement confidence**, and AlphaFold-Multimer adds **pTM/ipTM** for complex confidence. AF-Analysis keeps true DockQ, mpDockQ/pDockQ-style scores, interface-analysis values, and AlphaJudge values separate so the scales are not mixed.

References: [EMBL-EBI multimer confidence](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/confidence-scores-in-alphafold-multimer/), [EMBL-EBI PAE guide](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/pae-a-measure-of-global-confidence-in-alphafold-predictions/), [AlphaJudge](https://github.com/KosinskiLab/AlphaJudge).
"""
        )

        guide_tabs = st.tabs(["Quick ranges", "Result analysis", "AlphaJudge", "Plots"])

        with guide_tabs[0]:
            st.markdown(
                """
### Quick interpretation ranges

| Metric | Strong / good | Borderline / inspect | Weak / warning | Main caution |
|---|---:|---:|---:|---|
| **Protein pLDDT** | >90 very high; 70-90 confident | 50-70 low | <50 very low | Local chain confidence only; it does not prove docking. |
| **min inter-chain PAE** | <5 A excellent; 5-10 A good | 10-15 A caution | >15 A likely unreliable docking | Low PAE should cover the interface, not only one residue pair. |
| **average/interface PAE** | <5-10 A | 10-15 A | >15 A | Usually more robust than a single minimum. |
| **ipTM** | >0.8 confident | 0.6-0.8 grey zone | <0.6 likely failed | In fast screens, lower values can be used only for initial triage. |
| **pTM** | >0.5 suggests plausible global fold | around 0.5 | <0.5 likely wrong global fold | Less interface-specific than ipTM. |
| **Model confidence (ipTM+pTM)** | >0.8 high | 0.6-0.8 moderate | <0.6 weak | Good for sorting; still check PAE/interface metrics. |
| **AF-Analysis mpDockQ** | >150 excellent | 120-150 good; 100-120 acceptable | <100 poor | This bundled scale is `mean interface pLDDT x log10(contacts)`, not true DockQ. |
| **True DockQ** | >0.80 high | 0.49-0.80 medium; 0.23-0.49 acceptable | <0.23 incorrect | Only available with a reference. AF-Analysis never invents DockQ. |
| **PI score** | higher is better | compare within the run | near zero is weak | Relative AlphaPulldown/interface_analysis score, not a binding affinity. |
| **Interface area / contacts** | plausible and non-trivial | depends on complex type | tiny or sparse can be weak | Size alone is not evidence; large uncertain interfaces can be artefacts. |
"""
            )

        with guide_tabs[1]:
            st.markdown(
                """
### Interface-analysis / AlphaFold columns

| Column | Direction | Useful range or rule | What it means |
|---|---|---|---|
| **Protein 1 pLDDT / Protein 2 pLDDT** | higher better | >90 very high, 70-90 confident, 50-70 low, <50 very low | Average confidence of each chain. Good chains can still dock incorrectly. |
| **min PAE / min inter-chain PAE** | lower better | <5 excellent, 5-10 good, 10-15 caution, >15 poor | Best inter-chain relative-placement confidence. Use with average/interface PAE. |
| **Mean PAE / average interface PAE** | lower better | <10 usually useful, >15 suspicious | Relative placement confidence across more residue pairs. |
| **ipTM** | higher better | >0.8 confident; 0.6-0.8 grey; <0.6 weak | AlphaFold-Multimer interface/chain-placement confidence. |
| **pTM** | higher better | >0.5 suggests plausible global topology | Overall complex fold confidence, less interface-specific. |
| **Model confidence (ipTM+pTM)** | higher better | >0.8 high, 0.6-0.8 moderate, <0.6 weak | Ranking confidence; AF-Analysis labels this as model confidence. |
| **mpDockQ** | higher better | >150 excellent, 120-150 good, 100-120 acceptable, <100 poor | AF-Analysis interface-analysis heuristic from interface pLDDT and contact count. |
| **PI score** | higher better | compare within run; near 0 is weak | Interface-analysis interface-quality score used for Top 20 coloring. |
| **Interface residues** | context | no universal cutoff | Number of residues participating in the interface. |
| **Contact pairs** | context; often higher better | no universal cutoff | Inter-chain residue contacts; small interfaces may be valid if biologically expected. |
| **Side-chain contacts** | context; higher if plausible | no universal cutoff | Helps distinguish packed interfaces from backbone-only proximity. |
| **Hydrogen bonds** | context | no universal cutoff | Supports specificity in polar interfaces; absence is not fatal for hydrophobic interfaces. |
| **Salt bridges** | context | no universal cutoff | Strong specificity clue when geometry is plausible. |
| **Interface area** | context | compare related candidates | Bubble size in Top 20 plot. Very small areas may be weak; very large areas with high PAE are suspicious. |
| **Polar / hydrophobic / charged fraction** | context | 0-1 fraction | Chemistry sanity check; no single ideal composition. |
| **Solvation energy / SolvE** | more favorable is better | relative only | Approximate energy descriptor; keep secondary to PAE/ipTM/interface scores. |
"""
            )

        with guide_tabs[2]:
            st.markdown(
                """
### AlphaJudge columns

AlphaJudge values are shown in a separate table because they are generated and aggregated differently from bundled interface-analysis values.

| Column | Direction | Practical guide | What it means |
|---|---|---|---|
| **AJ Score / AJ confidence score** | higher better | compare within run; top-ranked values are stronger | AlphaJudge composite/interface score if present. |
| **AJ ipTM / AJ pTM** | higher better | use same spirit as ipTM/pTM; keep separate from base columns | AlphaFold confidence values parsed or propagated through AlphaJudge output. |
| **AJ pDockQ / AJ mpDockQ** | higher better | usually 0-1 or tool-specific; compare within run | DockQ-like interface-quality estimate, not true DockQ. |
| **AJ pDockQ2** | higher better | higher values are better; useful for ranking | PAE-aware predicted DockQ-style interface score. |
| **AJ average interface PAE** | lower better | <10 good, 10-15 caution, >15 weak | Mean PAE over AlphaJudge interface residues. |
| **AJ interface pLDDT** | higher better | >70 supportive, >90 very strong local confidence | Local confidence of interface residues; supporting evidence only for multimers. |
| **AJ ipSAE** | higher better | compare within run; high values support interface confidence | Interface-specific AlphaJudge score summarizing PAE-aware support. |
| **AJ LIS** | higher better | compare within run | Local interaction score; helps identify well-supported local contacts. |
| **AJ interface residues / contacts / area** | context | compare related candidates | Interface size and extent. Not enough by itself. |
| **AJ polar / hydrophobic / charged** | context | 0-1 fraction | Interface chemistry composition. |
| **AJ HB / SB / SC** | context; more can support specificity | no universal cutoff | Hydrogen bonds, salt bridges, and side-chain contacts. |
| **AJ SolvE / solvation** | more favorable is better | relative only | Approximate energetic/solvation descriptor. |
| **AJ rows** | debug/context | >1 means multiple interface rows were merged | Check raw interfaces.csv for multi-interface complexes. |
"""
            )

        with guide_tabs[3]:
            st.markdown(
                """
### Plot rules in AF-Analysis

| Plot | Ranking / axes | Color / size | How to use it |
|---|---|---|---|
| **Model confidence plot** | y = **Model confidence (ipTM+pTM)**; x = true DockQ only if explicit, otherwise mpDockQ | color = min inter-chain PAE, yellow low/good to purple high/poor | Best region is high confidence, low PAE, and high x-axis quality. |
| **Top 20 Interactions** | selected by composite triage score: 40% model confidence, 35% PAE confidence, 25% true DockQ or relative mpDockQ | size = interface area; color = PI score, yellow low to purple high | Use for shortlist generation, then inspect structure and PAE. |
| **AlphaJudge overview** | AlphaJudge-only metrics such as ipSAE, LIS, pDockQ2, interface score | AlphaJudge scales only | Use as an independent check; agreement with interface analysis is strongest. |

### Accept a candidate only after these checks

1. pLDDT is good enough for the chains or regions involved.
2. Inter-chain/interface PAE is low, ideally below 10 A and not just at one residue pair.
3. ipTM/model confidence is not weak.
4. mpDockQ, PI score, or AlphaJudge pDockQ2/ipSAE/LIS supports the interface.
5. Interface area and contacts are plausible for the expected biology.
6. Structure viewer does not show obvious clashes, chain crossings, or contacts dominated by low-confidence/disordered regions.
"""
            )


def make_display_df(filtered_df: pd.DataFrame, specs: List[Dict[str, str]]) -> pd.DataFrame:
    display_df = filtered_df[[spec["key"] for spec in specs]].copy()
    for spec in specs:
        key = spec["key"]
        if key != "job":
            display_df[key] = display_df[key].map(lambda value, k=key: format_value(k, value))
    return display_df.rename(columns={spec["key"]: spec.get("display_label", spec["label"]) for spec in specs})


@st.cache_data(show_spinner=False)
def build_rank0_zip(entries: Tuple[Tuple[str, str], ...]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, file_path in entries:
            path = Path(file_path)
            if path.exists():
                zf.write(path, arcname=arcname)
    return buffer.getvalue()


def collect_rank0_entries(filtered_df: pd.DataFrame) -> Tuple[Tuple[str, str], ...]:
    entries = []
    analyzer_cache: Dict[Path, AlphaPulldownAnalyzer] = {}
    for _, row in filtered_df.iterrows():
        job_path = Path(row["path"])
        if job_path.parent not in analyzer_cache:
            analyzer_cache[job_path.parent] = AlphaPulldownAnalyzer(str(job_path.parent))
        models = analyzer_cache[job_path.parent].get_all_models(job_path)
        if not models:
            continue
        rank0_model = next((m for m in models if m.get("rank") == 0), models[0])
        structure_file = rank0_model.get("structure_file")
        if not structure_file:
            continue
        structure_path = Path(structure_file)
        if not structure_path.exists():
            continue
        safe_job = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["job"]))
        entries.append((f"{safe_job}{structure_path.suffix}", str(structure_path)))
    return tuple(entries)


def filter_and_sort_results(results_df: pd.DataFrame, min_iptm: float, max_pae: float, table_view: str) -> pd.DataFrame:
    numeric_cols_for_view = get_numeric_cols_for_view(results_df, table_view)

    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1.2])
    with col1:
        search_term = st.text_input("Search predictions", "", placeholder="Filter by job name...")
    with col2:
        min_iptm = st.slider("Minimum ipTM", 0.0, 1.0, min_iptm, 0.05, key="iptm_filter")

    slider_max_pae = 30.0
    if "mean_pae" in results_df.columns and results_df["mean_pae"].notna().any():
        max_found_pae = float(results_df["mean_pae"].max(skipna=True))
        slider_max_pae = max(slider_max_pae, math.ceil(max_found_pae / 5.0) * 5.0)
    with col3:
        default_pae = slider_max_pae if slider_max_pae > max_pae else max_pae
        max_pae = st.slider("Maximum mean PAE", 0.0, slider_max_pae, default_pae, 0.5, key="pae_filter")

    sort_options = ["ipTM", "ipTM+pTM", "Job name"] + [col for col in numeric_cols_for_view if col not in {"iptm", "iptm_ptm"}]
    with col4:
        sort_by = st.selectbox("Sort by", sort_options)

    filtered_df = results_df.copy()
    filtered_df = filtered_df[filtered_df["iptm"] >= min_iptm]
    if "mean_pae" in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df["mean_pae"].isna()) | (filtered_df["mean_pae"] <= max_pae)]
    if search_term:
        filtered_df = filtered_df[
            filtered_df["job"].astype(str).str.contains(search_term, case=False, regex=False, na=False)
        ]

    alphajudge_ranges: Dict[str, Tuple[float, float]] = {}
    if numeric_cols_for_view:
        with st.expander(f"Additional {table_view} filters", expanded=False):
            filter_cols = st.columns(2)
            for idx, col_name in enumerate([c for c in numeric_cols_for_view if c not in {"iptm", "iptm_ptm", "n_models"}]):
                values = results_df[col_name].dropna()
                if values.empty:
                    continue
                min_val = float(values.min())
                max_val = float(values.max())
                with filter_cols[idx % 2]:
                    if min_val == max_val:
                        st.caption(f"{col_name}: {min_val:.3f}")
                    else:
                        step = max((max_val - min_val) / 100, 0.0001)
                        alphajudge_ranges[col_name] = st.slider(
                            col_name,
                            min_val,
                            max_val,
                            (min_val, max_val),
                            step=step,
                            key=f"alphajudge_filter_{col_name}",
                        )

    for col_name, value_range in alphajudge_ranges.items():
        filtered_df = filtered_df[
            filtered_df[col_name].isna()
            | ((filtered_df[col_name] >= value_range[0]) & (filtered_df[col_name] <= value_range[1]))
        ]

    if sort_by == "ipTM":
        filtered_df = filtered_df.sort_values("iptm", ascending=False)
    elif sort_by == "ipTM+pTM":
        filtered_df = filtered_df.sort_values("iptm_ptm", ascending=False)
    elif sort_by == "Job name":
        filtered_df = filtered_df.sort_values("job")
    else:
        filtered_df = filtered_df.sort_values(sort_by, ascending=False, na_position="last")

    return filtered_df.reset_index(drop=True)


def _render_optional_overview_plots(filtered_df: pd.DataFrame) -> None:
    st.divider()
    plot_col, hide_col = st.columns([1, 1])
    plot_key = f"overview_plots_visible_{st.session_state.get('active_directory', 'default')}"
    with plot_col:
        if st.button("Generate overview plots", use_container_width=True, help="Draw plots on demand. Interface-analysis Top 20 is ranked by a composite of ipTM+pTM/model confidence, min inter-chain PAE, and true DockQ when present otherwise relative mpDockQ; AlphaJudge plots use AlphaJudge metrics separately."):
            st.session_state[plot_key] = True
    with hide_col:
        if st.session_state.get(plot_key, False) and st.button("Hide overview plots", use_container_width=True):
            st.session_state[plot_key] = False
            st.rerun()
    if st.session_state.get(plot_key, False):
        selected_job = render_overview_plots(filtered_df)
        if selected_job:
            navigate_to_viewer(selected_job)


def render_overview_page(results_df: pd.DataFrame, min_iptm: float, max_pae: float) -> None:
    if results_df.empty:
        st.error("No valid multimer predictions found in the specified directory.")
        return

    table_view = st.radio(
        "Table view",
        ["Result analysis", "AlphaJudge"],
        horizontal=True,
        key="overview_table_view",
        help="Keep interface-analysis metrics separate from AlphaJudge metrics.",
    )
    show_detailed_columns = st.checkbox(
        "Show detailed columns",
        value=False,
        key="overview_show_detailed_columns",
        help="Default view keeps the table compact. Enable this for residue-class fractions, H-bonds, salt bridges, and all extra AlphaJudge columns.",
    )
    _render_metric_guide(table_view)

    filtered_df = filter_and_sort_results(results_df, min_iptm, max_pae, table_view)
    if filtered_df.empty:
        st.warning("No predictions match the current filters.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total predictions", len(filtered_df))
    col2.metric("Best ipTM", f"{filtered_df['iptm'].max():.3f}")
    col3.metric("Average ipTM", f"{filtered_df['iptm'].mean():.3f}")
    if "mean_pae" in filtered_df.columns and filtered_df["mean_pae"].notna().any():
        col4.metric("Mean PAE avg", f"{filtered_df['mean_pae'].mean():.1f} Å")
    else:
        col4.metric("Mean PAE avg", "N/A")

    _render_optional_overview_plots(filtered_df)

    st.divider()
    st.subheader(f"Predictions ({len(filtered_df)} results)")

    specs = make_column_specs(filtered_df, table_view, show_detailed_columns)
    display_df = make_display_df(filtered_df, specs)
    column_config = {
        spec.get("display_label", spec["label"]): st.column_config.TextColumn(
            spec.get("display_label", spec["label"]),
            help=spec.get("help"),
        )
        for spec in specs
    }

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config=column_config,
        selection_mode="single-row",
        on_select="rerun",
    )

    selected_rows = []
    if hasattr(event, "selection"):
        selected_rows = list(getattr(event.selection, "rows", []))
    elif isinstance(event, dict):
        selected_rows = list(event.get("selection", {}).get("rows", []))

    if selected_rows:
        selected_job = filtered_df.iloc[selected_rows[0]]["job"]
        if st.button(f"Open {selected_job}", type="primary"):
            navigate_to_viewer(selected_job)

    st.divider()
    _, col2, col3 = st.columns([2, 1, 1])
    with col2:
        rank0_entries = collect_rank0_entries(filtered_df)
        if rank0_entries:
            st.download_button(
                "📦 Download rank 0 models",
                build_rank0_zip(rank0_entries),
                file_name="rank0_models.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.info("No rank 0 models found for the current filters.")
    with col3:
        st.download_button(
            "📥 Download CSV",
            filtered_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
