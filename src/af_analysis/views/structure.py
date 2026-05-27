"""Structure-viewer page."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from af_analysis.plots.pae import plot_pae_heatmap_interactive
from af_analysis.utils import (
    AlphaPulldownAnalyzer,
    create_3dmol_view,
    get_pae_plot_image,
    load_interfaces_csv,
    plot_model_comparison,
    plot_pae_heatmap,
)


def get_pae_file_for_model(job_path: Path, model_name: str, rank: Optional[int] = None) -> Optional[Path]:
    """Find AF2 or AF3 PAE/confidence JSON for a specific model.

    AF2/ColabFold usually writes ``pae_<model>.json``. AF3 stores PAE inside
    ``confidences.json`` or ``ranked_*_confidences.json`` files, so include
    those files before falling back to static PAE images.
    """
    job_path = Path(job_path)
    model_dir = job_path / model_name
    candidates = []

    if model_dir.is_dir():
        candidates.extend(
            [
                model_dir / "confidences.json",
                model_dir / "summary_confidences.json",
            ]
        )

    if rank is not None:
        candidates.extend(
            [
                job_path / f"ranked_{rank}_confidences.json",
                job_path / f"ranked_{rank}_summary_confidences.json",
                job_path / f"pae_ranked_{rank}.json",
            ]
        )

    candidates.append(job_path / f"pae_{model_name}.json")
    match = re.search(r"model_(\d+)", model_name)
    if match:
        model_num = match.group(1)
        candidates.extend(
            [
                job_path / f"pae_model_{model_num}_ptm_pred_0.json",
                job_path / f"pae_model_{model_num}.json",
            ]
        )

    candidates.extend(
        [
            job_path / "confidences.json",
            job_path / "summary_confidences.json",
        ]
    )
    candidates.extend(sorted(job_path.glob("ranked_*_confidences.json")))
    candidates.extend(sorted(job_path.glob("*_confidences.json")))

    for candidate in dict.fromkeys(candidates):
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        except Exception:
            continue
    return None


def metric_text(value, fmt: str = ".3f") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return format(float(value), fmt)
    except Exception:
        return "N/A"


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
        selected_job = st.selectbox(
            "Select prediction",
            job_list,
            index=default_idx,
            key=f"viewer_job_select_{st.session_state.get('active_directory', 'default')}",
        )
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
        key=f"viewer_model_select_{st.session_state.get('active_directory', 'default')}_{selected_job}",
    )
    selected_model = models[selected_model_idx]
    structure_file = selected_model.get("structure_file") or selected_model.get("pdb_file")
    structure_format = selected_model.get("structure_format", "pdb")

    pae_file = get_pae_file_for_model(job_path, selected_model["model_name"], selected_model["rank"])
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
    col2.metric("ipTM", metric_text(selected_model.get("iptm"), ".3f"))
    col3.metric("ipTM+pTM", metric_text(selected_model.get("iptm_ptm"), ".3f"))
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
        if pae_file:
            try:
                fig = plot_pae_heatmap_interactive(pae_file, max_points=650)
                st.plotly_chart(
                    fig,
                    use_container_width=False,
                    key=f"pae_heatmap_{st.session_state.get('active_directory', 'default')}_{selected_job}_{selected_model['model_name']}",
                )
            except Exception as exc:
                st.warning(f"Interactive PAE plot unavailable ({exc}). Showing static fallback.")
                fallback_fig = plot_pae_heatmap(pae_file, figsize=(5.6, 5.6))
                if fallback_fig:
                    st.pyplot(fallback_fig, clear_figure=True)
        elif pae_image:
            st.image(str(pae_image), width=560)
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
