#!/usr/bin/env python3
"""AF-Analysis - AlphaPulldown / AlphaFold result browser.

The Streamlit entry point is intentionally thin.  Data discovery, state
management, plotting, and page rendering live in separate modules so UI state and
large-file loading are easier to debug.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Hide Streamlit's automatic multipage navigation.  Older unpacked builds could
# leave a stale src/af_analysis/pages/ directory behind; that directory causes the
# unwanted top sidebar entries named app / overview / structure.  The environment
# flag handles supported Streamlit versions and the best-effort cleanup handles
# dirty in-place upgrades.
os.environ.setdefault("STREAMLIT_CLIENT_SHOW_SIDEBAR_NAVIGATION", "false")
try:
    stale_pages_dir = Path(__file__).resolve().parent / "pages"
    if stale_pages_dir.is_dir():
        shutil.rmtree(stale_pages_dir)
except Exception:
    pass

# Prefer the local source tree when this file is launched directly with
# `streamlit run src/af_analysis/app.py`. This avoids mixing a newly unpacked app.py
# with an older installed `af_analysis` package on shared systems.
_SRC_ROOT = str(Path(__file__).resolve().parents[1])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import streamlit as st

st.set_page_config(
    page_title="AF-Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


from af_analysis.actions import (
    merge_backfill_results,
    rows_needing_alphajudge,
    rows_needing_backfill,
    run_alphajudge_for_paths,
)
from af_analysis.data.cache import (
    DEFAULT_INTERFACE_REPORT_CUTOFF,
    build_predictions_dataframe,
    dataframe_cache_path,
    load_dataframe_cache,
    load_interface_report_csv,
    merge_interface_analysis_scores,
    run_integrated_interface_analysis,
    save_dataframe_cache,
)
from af_analysis.views.overview import render_overview_page
from af_analysis.views.structure import render_viewer_page
from af_analysis.state import (
    get_default_directory,
    initialize_session_state,
    sync_directory_state,
)
from af_analysis.ui import apply_global_style, render_header, render_missing_directory_help, render_sidebar
from af_analysis.utils import AlphaPulldownAnalyzer


def main() -> None:
    initialize_session_state()
    apply_global_style()
    render_header()

    (
        directory,
        auto_refresh,
        refresh_interval,
        compute_mean_pae,
        run_alphajudge_clicked,
        legacy_backfill_clicked,
        refresh_clicked,
    ) = render_sidebar(get_default_directory())

    if not directory:
        render_missing_directory_help()
        return

    directory = sync_directory_state(directory)
    if not Path(directory).expanduser().exists():
        render_missing_directory_help()
        return

    results_df = None
    cache_file = dataframe_cache_path(directory, compute_mean_pae)
    scan_requested = bool(refresh_clicked or auto_refresh)

    if not scan_requested:
        results_df = load_dataframe_cache(directory, compute_mean_pae)

    if results_df is None:
        with st.spinner("Loading predictions into the AF-Analysis dataframe cache..."):
            results_df = build_predictions_dataframe(directory, compute_mean_pae)
        save_dataframe_cache(directory, compute_mean_pae, results_df)
        st.sidebar.caption(f"Dataframe cache written: {cache_file.name}")
    else:
        st.sidebar.caption(f"Loaded dataframe cache: {cache_file.name}. Use Scan / refresh dataframe now to rescan files.")

    # Merge the interface-analysis output into the fast dataframe. If the CSV is
    # missing, run the bundled module automatically, including best-effort
    # AlphaJudge generation/merge. This makes mpDockQ and per-protein pLDDT
    # first-class AF-Analysis table columns without requiring a separate script.
    interface_analysis_df, result_csv_path = load_interface_report_csv(directory, DEFAULT_INTERFACE_REPORT_CUTOFF)
    if interface_analysis_df is None:
        st.info("Interface-analysis CSV not found. Running bundled interaction-report module with AlphaJudge backfill...")
        progress_bar = st.progress(0.0, text="Starting interface-analysis backfill...")
        progress_status = st.empty()
        progress_log_box = st.empty()
        streamed_lines: list[str] = []

        def _update_result_progress(line: str, progress: float, phase: str) -> None:
            streamed_lines.append(line)
            safe_progress = min(max(float(progress), 0.0), 1.0)
            progress_bar.progress(safe_progress, text=phase)
            progress_status.caption(phase)
            # Show a small live tail without flooding the page.
            progress_log_box.code("\n".join(streamed_lines[-12:]))

        success, result_log, result_csv_path = run_integrated_interface_analysis(
            directory,
            cutoff=DEFAULT_INTERFACE_REPORT_CUTOFF,
            run_alphajudge=True,
            force_alphajudge=False,
            progress_callback=_update_result_progress,
        )
        if success:
            progress_bar.progress(1.0, text="Interface-analysis backfill complete")
        with st.expander("Interface-analysis backfill log", expanded=not success):
            st.code(result_log[-12000:] if result_log else "No output captured.")
        if success:
            # AlphaJudge may have created interfaces.csv files; rescan once so
            # structure-view side panels and fast dataframe columns reflect them.
            with st.spinner("Refreshing AF-Analysis dataframe after interface analysis/AlphaJudge..."):
                results_df = build_predictions_dataframe(directory, compute_mean_pae)
            interface_analysis_df, result_csv_path = load_interface_report_csv(directory, DEFAULT_INTERFACE_REPORT_CUTOFF)
        else:
            st.warning("The bundled interaction-report module did not produce a usable CSV. AF-Analysis will continue with the fast dataframe only.")

    if interface_analysis_df is not None:
        results_df = merge_interface_analysis_scores(results_df, interface_analysis_df, directory)
        save_dataframe_cache(directory, compute_mean_pae, results_df)
        merged_rows = int(results_df.attrs.get("af_analysis_interface_report_merged_rows", 0))
        if result_csv_path is not None:
            st.sidebar.caption(f"Merged interface-analysis CSV: {result_csv_path.name} ({merged_rows} rows matched).")

    if run_alphajudge_clicked:
        aj_targets = rows_needing_alphajudge(results_df)
        if not aj_targets:
            st.success("No rows with missing/empty interfaces.csv were found.")
        else:
            with st.spinner(f"Running AlphaJudge for {len(aj_targets)} job(s)..."):
                aj_summary = run_alphajudge_for_paths(aj_targets)
            st.info(aj_summary)
            with st.spinner("Refreshing dataframe after AlphaJudge..."):
                results_df = build_predictions_dataframe(directory, compute_mean_pae)
            save_dataframe_cache(directory, compute_mean_pae, results_df)

    if legacy_backfill_clicked:
        backfill_jobs = rows_needing_backfill(results_df)
        if not backfill_jobs:
            st.success("No missing interface values found to backfill.")
        else:
            with st.spinner(f"Backfilling missing values for {len(backfill_jobs)} job(s)..."):
                analyzer = AlphaPulldownAnalyzer(
                    directory,
                    compute_mean_pae=compute_mean_pae,
                    enable_legacy_fill=True,
                    legacy_fill_jobs=backfill_jobs,
                )
                backfill_df = analyzer.analyze_directory()
            results_df = merge_backfill_results(results_df, backfill_df)
            changed_cells = int(results_df.attrs.get("af_analysis_backfill_changed_cells", 0))
            if changed_cells > 0:
                save_dataframe_cache(directory, compute_mean_pae, results_df)
                st.success(
                    f"Backfilled {changed_cells} missing value(s) across up to "
                    f"{len(backfill_jobs)} job folder(s). The cached dataframe was updated."
                )
            else:
                st.warning(
                    "Backfill ran, but no fillable missing values were recovered. The cache was not changed."
                )

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
