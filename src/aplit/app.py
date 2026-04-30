#!/usr/bin/env python3
"""
APLit - AlphaPulldown / AlphaFold result browser.

This version focuses on fast reruns, theme-safe UI elements, and lazy loading of
large files such as PAE JSON matrices and structure ZIP exports.
"""

import io
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from aplit.utils import (
        AlphaPulldownAnalyzer,
        create_3dmol_view,
        get_pae_file_for_model,
        get_pae_plot_image,
        load_interfaces_csv,
        plot_model_comparison,
        plot_pae_heatmap,
    )
except ModuleNotFoundError:  # allows `streamlit run src/aplit/app.py`
    from utils import (
        AlphaPulldownAnalyzer,
        create_3dmol_view,
        get_pae_file_for_model,
        get_pae_plot_image,
        load_interfaces_csv,
        plot_model_comparison,
        plot_pae_heatmap,
    )


st.set_page_config(
    page_title="APLit Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --aplit-accent: #1f77b4;
    --aplit-muted: rgba(49, 51, 63, 0.68);
}
.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    color: var(--aplit-accent);
    margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
}
.subtitle {
    color: var(--aplit-muted);
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


BASE_COLUMNS_EXCLUDED = {
    "job",
    "iptm",
    "iptm_ptm",
    "mean_pae",
    "best_model",
    "path",
    "n_models",
    "job_type",
    "ptm",
    "confidence_score",
    "interface_csv",
    "interface_summary_model",
}

ALPHAJUDGE_COLUMNS = [
    ("global_dockq", "DockQ", "Global DockQ score"),
    ("best_interface_pdockq2", "pDQ2", "Best interface pDockQ2"),
    ("best_interface_ipsae", "ipSAE", "Best interface ipSAE"),
    ("best_interface_lis", "LIS", "Best interface LIS"),
    ("interface_score", "Score", "Interface score"),
    ("interface_average_plddt", "pLDDT", "Interface average pLDDT"),
    ("interface_residue_count", "Res", "Interface residues"),
    ("interface_contact_pairs", "Pairs", "Interface contact pairs"),
    ("interface_area", "Area", "Interface area (Å²)"),
    ("interface_solv_energy", "SolvE", "Interface solvation energy"),
    ("interface_polar_fraction", "Polar", "Interface polar fraction"),
    ("interface_hydrophobic_fraction", "Hydro", "Interface hydrophobic fraction"),
    ("interface_charged_fraction", "Charged", "Interface charged fraction"),
]


def initialize_session_state() -> None:
    st.session_state.setdefault("current_page", "overview")
    st.session_state.setdefault("selected_job", None)
    st.session_state.setdefault("results_df", None)


def get_default_directory() -> str:
    return os.environ.get("APLIT_DEFAULT_DIRECTORY", "")


def navigate_to_viewer(job_name: str) -> None:
    st.session_state.current_page = "viewer"
    st.session_state.selected_job = job_name
    st.rerun()


def render_header() -> None:
    st.markdown('<div class="main-title">APLit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AlphaPulldown / AlphaFold structure and confidence viewer</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(default_directory: str):
    st.sidebar.title("Configuration")
    directory = st.sidebar.text_input(
        "Predictions directory",
        value=default_directory,
        help="Parent directory containing AlphaPulldown/AlphaFold prediction jobs.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Navigation")
    if st.sidebar.button("← Overview", use_container_width=True):
        st.session_state.current_page = "overview"
        st.rerun()
    if st.sidebar.button(
        "Structure viewer →",
        use_container_width=True,
        disabled=st.session_state.selected_job is None,
    ):
        st.session_state.current_page = "viewer"
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Performance")
    compute_mean_pae = st.sidebar.checkbox(
        "Compute AF2 mean inter-chain PAE",
        value=False,
        help=(
            "This reads large PAE JSON files for every AF2 job. Leave it off for "
            "faster browsing. AlphaJudge average_interface_pae is still used when present."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Auto-refresh")
    auto_refresh = st.sidebar.checkbox("Enable", value=False)
    refresh_interval = st.sidebar.slider(
        "Interval (seconds)", 10, 300, 60, 10, disabled=not auto_refresh
    )
    if st.sidebar.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return directory, auto_refresh, refresh_interval, compute_mean_pae


def get_alphajudge_numeric_cols(results_df: pd.DataFrame) -> List[str]:
    cols = []
    for col in results_df.columns:
        if col in BASE_COLUMNS_EXCLUDED:
            continue
        if pd.api.types.is_numeric_dtype(results_df[col]) and results_df[col].notna().any():
            cols.append(col)
    return cols


def format_value(key: str, value) -> str:
    if pd.isna(value):
        return "N/A"
    if key in {"iptm", "iptm_ptm", "global_dockq", "best_interface_pdockq2", "best_interface_ipsae", "best_interface_lis"}:
        return f"{float(value):.3f}"
    if key in {"mean_pae", "interface_score", "interface_average_plddt", "interface_solv_energy"}:
        return f"{float(value):.1f}"
    if key in {"interface_residue_count", "interface_contact_pairs", "n_models"}:
        return f"{int(value)}"
    if key == "interface_area":
        return f"{float(value):.0f}"
    if key in {"interface_polar_fraction", "interface_hydrophobic_fraction", "interface_charged_fraction"}:
        return f"{float(value):.2f}"
    return str(value)


def make_column_specs(filtered_df: pd.DataFrame) -> List[Dict[str, str]]:
    specs = [
        {"key": "job", "label": "Job", "help": "Prediction/job folder"},
        {"key": "iptm", "label": "ipTM", "help": "Interface predicted TM score"},
        {"key": "iptm_ptm", "label": "ipTM+pTM", "help": "Combined ranking/confidence score"},
    ]
    if "mean_pae" in filtered_df.columns:
        specs.append({"key": "mean_pae", "label": "Mean PAE", "help": "Mean inter-chain/interface PAE"})
    for key, label, help_text in ALPHAJUDGE_COLUMNS:
        if key in filtered_df.columns and filtered_df[key].notna().any():
            specs.append({"key": key, "label": label, "help": help_text})
    specs.append({"key": "n_models", "label": "Models", "help": "Number of models found"})
    return specs


def make_display_df(filtered_df: pd.DataFrame, specs: List[Dict[str, str]]) -> pd.DataFrame:
    display_df = filtered_df[[spec["key"] for spec in specs]].copy()
    for spec in specs:
        key = spec["key"]
        if key != "job":
            display_df[key] = display_df[key].map(lambda value, k=key: format_value(k, value))
    return display_df.rename(columns={spec["key"]: spec["label"] for spec in specs})


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


def filter_and_sort_results(results_df: pd.DataFrame, min_iptm: float, max_pae: float):
    alphajudge_numeric_cols = get_alphajudge_numeric_cols(results_df)

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

    sort_options = ["ipTM", "ipTM+pTM", "Job name"] + alphajudge_numeric_cols
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
    if alphajudge_numeric_cols:
        with st.expander("AlphaJudge filters", expanded=False):
            filter_cols = st.columns(2)
            for idx, col_name in enumerate(alphajudge_numeric_cols):
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


def render_overview_page(results_df: pd.DataFrame, min_iptm: float, max_pae: float) -> None:
    if results_df.empty:
        st.error("No valid multimer predictions found in the specified directory.")
        return

    filtered_df = filter_and_sort_results(results_df, min_iptm, max_pae)
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

    st.divider()
    st.subheader(f"Predictions ({len(filtered_df)} results)")

    specs = make_column_specs(filtered_df)
    display_df = make_display_df(filtered_df, specs)
    column_config = {
        spec["label"]: st.column_config.TextColumn(spec["label"], help=spec.get("help"))
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


def render_viewer_page(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        st.error("No predictions available.")
        return

    job_list = results_df["job"].tolist()
    default_idx = 0
    if st.session_state.selected_job in job_list:
        default_idx = job_list.index(st.session_state.selected_job)

    select_col, next_col = st.columns([4, 1])
    with select_col:
        selected_job = st.selectbox("Select prediction", job_list, index=default_idx)
    with next_col:
        st.write("")
        st.write("")
        if st.button("Next prediction →", use_container_width=True):
            next_idx = (job_list.index(selected_job) + 1) % len(job_list)
            st.session_state.selected_job = job_list[next_idx]
            st.rerun()
    st.session_state.selected_job = selected_job

    job_row = results_df[results_df["job"] == selected_job].iloc[0]
    job_path = Path(job_row["path"])
    analyzer = AlphaPulldownAnalyzer(str(job_path.parent))
    models = analyzer.get_all_models(job_path)
    if not models:
        st.error(f"Could not load models for {selected_job}")
        return

    model_options = [f"Rank {m['rank']} ({m['model_name']})" for m in models]
    selected_model_idx = st.selectbox(
        "Select model",
        range(len(models)),
        format_func=lambda i: model_options[i],
    )
    selected_model = models[selected_model_idx]
    structure_file = selected_model.get("structure_file") or selected_model.get("pdb_file")
    structure_format = selected_model.get("structure_format", "pdb")

    pae_file = get_pae_file_for_model(job_path, selected_model["model_name"])
    pae_image = get_pae_plot_image(job_path, selected_model["model_name"], selected_model["rank"])
    interfaces_df = load_interfaces_csv(job_path)
    model_interfaces = (
        interfaces_df[interfaces_df["model_used"] == selected_model["model_name"]]
        if interfaces_df is not None and "model_used" in interfaces_df.columns
        else None
    )

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rank", f"#{selected_model['rank']}")
    col2.metric("ipTM", f"{selected_model['iptm']:.3f}")
    col3.metric("ipTM+pTM", f"{selected_model['iptm_ptm']:.3f}")
    col4.metric("Total models", len(models))

    if model_interfaces is not None and not model_interfaces.empty:
        st.success("AlphaJudge interface scores available for this model.")
        show_interface_metrics(model_interfaces)

    with st.expander("📊 Compare all models", expanded=False):
        fig = plot_model_comparison(models)
        if fig:
            st.pyplot(fig, clear_figure=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Structure (pLDDT)", "Structure (chains)", "PAE plot"])

    with tab1:
        render_structure_tab(structure_file, structure_format, "plddt")
    with tab2:
        render_structure_tab(structure_file, structure_format, "chain")
    with tab3:
        if pae_image:
            st.image(str(pae_image), use_container_width=True)
        elif pae_file:
            fig = plot_pae_heatmap(pae_file, figsize=(8, 8))
            if fig:
                st.pyplot(fig, clear_figure=True)
        else:
            st.warning("PAE file not found for this model.")

    render_downloads(selected_job, selected_model, structure_file, structure_format, pae_file)

    if interfaces_df is not None and model_interfaces is not None and not model_interfaces.empty:
        st.divider()
        with st.expander("Interface details (AlphaJudge)", expanded=False):
            download_df = model_interfaces.reset_index(drop=True)
            st.dataframe(download_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download interfaces CSV (model)",
                download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_job}_{selected_model['model_name']}_interfaces.csv",
                mime="text/csv",
            )


def show_interface_metrics(model_interfaces: pd.DataFrame) -> None:
    iface_numeric = model_interfaces.copy()
    for col in ["interface_pDockQ2", "interface_score", "average_interface_pae", "pDockQ/mpDockQ"]:
        if col in iface_numeric.columns:
            iface_numeric[col] = pd.to_numeric(iface_numeric[col], errors="coerce")
    sort_columns = [
        col for col in ["interface_pDockQ2", "interface_score", "average_interface_pae"] if col in iface_numeric.columns
    ]
    if sort_columns:
        ascending = [col == "average_interface_pae" for col in sort_columns]
        best_iface = iface_numeric.sort_values(sort_columns, ascending=ascending, na_position="last").iloc[0]
    else:
        best_iface = iface_numeric.iloc[0]

    specs = [
        ("Global DockQ", "pDockQ/mpDockQ", ".3f"),
        ("Best interface pDockQ2", "interface_pDockQ2", ".3f"),
        ("Best interface ipSAE", "interface_ipSAE", ".3f"),
        ("Best interface LIS", "interface_LIS", ".3f"),
        ("Interface score", "interface_score", ".1f"),
        ("Interface avg pLDDT", "interface_average_plddt", ".1f"),
        ("Interface residues", "interface_num_intf_residues", "int"),
        ("Contact pairs", "interface_contact_pairs", "int"),
        ("Interface area (Å²)", "interface_area", ".0f"),
        ("Interface solvation energy", "interface_solv_en", ".1f"),
        ("Avg interface PAE (Å)", "average_interface_pae", ".2f"),
        ("Interface polar fraction", "interface_polar", ".2f"),
        ("Interface hydrophobic fraction", "interface_hydrophobic", ".2f"),
        ("Interface charged fraction", "interface_charged", ".2f"),
    ]
    metrics = []
    for label, key, fmt in specs:
        if key in best_iface and pd.notna(best_iface[key]):
            value = int(best_iface[key]) if fmt == "int" else format(float(best_iface[key]), fmt)
            metrics.append((label, value))
    for start in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, metrics[start : start + 3]):
            col.metric(label, value)


def render_structure_tab(structure_file, structure_format: str, color_by: str) -> None:
    if structure_file and Path(structure_file).exists():
        html = create_3dmol_view(
            Path(structure_file),
            color_by=color_by,
            structure_format=structure_format,
            height=600,
        )
        components.html(html, height=650)
    else:
        st.warning("Structure file not found.")


def render_downloads(selected_job: str, selected_model: Dict, structure_file, structure_format: str, pae_file) -> None:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if structure_file and Path(structure_file).exists():
            with open(structure_file, "rb") as fh:
                st.download_button(
                    "Download structure",
                    fh,
                    file_name=f"{selected_job}_rank_{selected_model['rank']}{Path(structure_file).suffix}",
                    mime="chemical/x-pdb" if structure_format == "pdb" else "chemical/x-mmcif",
                    use_container_width=True,
                )
    with col2:
        if pae_file and Path(pae_file).exists():
            with open(pae_file, "rb") as fh:
                st.download_button(
                    "Download PAE JSON",
                    fh,
                    file_name=f"{selected_job}_rank_{selected_model['rank']}_pae.json",
                    mime="application/json",
                    use_container_width=True,
                )


def render_missing_directory_help() -> None:
    st.warning("Please enter a valid predictions directory in the sidebar.")
    st.info(
        """
**Expected directory structure:**
```
/path/to/predictions/
├── protein1_and_protein2/
│   ├── ranking_debug.json
│   ├── ranked_0.pdb
│   ├── pae_model_1_ptm_pred_0.json
│   └── ...
```
"""
    )


def main() -> None:
    initialize_session_state()
    render_header()

    directory, auto_refresh, refresh_interval, compute_mean_pae = render_sidebar(get_default_directory())
    if not directory or not Path(directory).exists():
        render_missing_directory_help()
        return

    @st.cache_data(ttl=refresh_interval if auto_refresh else None, show_spinner=False)
    def load_predictions(directory_path: str, compute_mean_pae_flag: bool) -> pd.DataFrame:
        analyzer = AlphaPulldownAnalyzer(directory_path, compute_mean_pae=compute_mean_pae_flag)
        return analyzer.analyze_directory()

    with st.spinner("Analyzing predictions..."):
        results_df = load_predictions(directory, compute_mean_pae)
    st.session_state.results_df = results_df

    if st.session_state.current_page == "viewer":
        render_viewer_page(results_df)
    else:
        render_overview_page(results_df, 0.0, 30.0)

    if auto_refresh and st.session_state.current_page == "overview":
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
