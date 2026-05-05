"""Shared Streamlit UI layout helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from af_analysis.state import navigate_to_overview

CSS = """
<style>
:root {
    --af-analysis-accent: #1f77b4;
    --af-analysis-muted: rgba(49, 51, 63, 0.68);
}
.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    color: var(--af-analysis-accent);
    margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
}
.subtitle {
    color: var(--af-analysis-muted);
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1rem;
}
</style>
"""


def apply_global_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown('<div class="main-title">AF-Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AlphaPulldown / AF2 / AF3 / ColabFold structure and confidence viewer</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(default_directory: str):
    st.sidebar.title("Configuration")

    with st.sidebar.expander("About AF-Analysis", expanded=False):
        st.markdown(
            """
**AF-Analysis** is an MDC-maintained analysis viewer for AlphaFold / AlphaPulldown prediction folders.

Built from and extended around these open-source projects:

- [APLit](https://github.com/KosinskiLab/aplit)
- [AlphaJudge](https://github.com/KosinskiLab/AlphaJudge)
- [af_analysis](https://github.com/samuelmurail/af_analysis)

**Modified and maintained by:** Ashwin Karthick Natarajan  
**Queries:** ashwin.natarajan@mdc-berlin.de
"""
        )

    directory = st.sidebar.text_input(
        "Predictions directory",
        value=default_directory,
        help="Parent directory containing AlphaPulldown, AF2, AF3, ColabFold, or compatible prediction jobs.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Navigation")
    if st.sidebar.button("← Overview", use_container_width=True):
        navigate_to_overview()
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
    st.sidebar.subheader("AlphaJudge / fallback")
    st.sidebar.caption("Default loading uses the local dataframe cache when available. Backfill is a one-shot action.")
    run_alphajudge_clicked = st.sidebar.button(
        "Run AlphaJudge for missing interfaces.csv",
        use_container_width=True,
        help=(
            "Runs the alphajudge command only for rows whose interfaces.csv is missing, empty, "
            "or unreadable. This is separate from legacy backfill."
        ),
    )
    legacy_backfill_clicked = st.sidebar.button(
        "Backfill missing values once (in-house)",
        use_container_width=True,
        help=(
            "One-shot pass: uses only in-house interface-analysis-style parsing for rows with fillable N/A "
            "interface values. Refreshing returns to fast loading."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Auto-refresh")
    auto_refresh = st.sidebar.checkbox("Enable", value=False)
    refresh_interval = st.sidebar.slider(
        "Interval (seconds)", 10, 300, 60, 10, disabled=not auto_refresh
    )
    refresh_clicked = st.sidebar.button(
        "Scan / refresh dataframe now",
        use_container_width=True,
        help="Re-read prediction folders and update the local .af_analysis_cache dataframe.",
    )
    if refresh_clicked:
        st.cache_data.clear()

    return directory, auto_refresh, refresh_interval, compute_mean_pae, run_alphajudge_clicked, legacy_backfill_clicked, refresh_clicked


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
