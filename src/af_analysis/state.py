"""Session-state and directory-navigation helpers for AF-Analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st


def initialize_session_state() -> None:
    st.session_state.setdefault("current_page", "overview")
    st.session_state.setdefault("selected_job", None)
    st.session_state.setdefault("results_df", None)
    st.session_state.setdefault("active_directory", None)
    st.session_state.setdefault("overview_plots_visible", False)


def _directory_from_argv() -> str:
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--directory" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--directory="):
            return arg.split("=", 1)[1]
    return ""


def get_default_directory() -> str:
    return os.environ.get("AF_ANALYSIS_DEFAULT_DIRECTORY", "") or os.environ.get("AP" + "LIT_DEFAULT_DIRECTORY", "") or _directory_from_argv()


def normalize_directory(directory: str) -> Optional[str]:
    if not directory:
        return None
    try:
        return str(Path(directory).expanduser().resolve())
    except Exception:
        return str(Path(directory).expanduser())


def reset_transient_view_state() -> None:
    """Clear state tied to a previous page, selected job, or folder."""
    st.session_state.current_page = "overview"
    st.session_state.selected_job = None
    st.session_state.results_df = None
    st.session_state.overview_plots_visible = False
    for key in list(st.session_state.keys()):
        if key.startswith((
            "viewer_",
            "structure_",
            "pae_",
            "model_",
            "overview_",
            "alphajudge_filter_",
            "iptm_filter",
            "pae_filter",
        )):
            del st.session_state[key]


def sync_directory_state(directory: str) -> str:
    resolved_directory = normalize_directory(directory) or directory
    if st.session_state.get("active_directory") != resolved_directory:
        st.session_state.active_directory = resolved_directory
        reset_transient_view_state()
    return resolved_directory


def navigate_to_viewer(job_name: str) -> None:
    st.session_state.current_page = "viewer"
    st.session_state.selected_job = job_name
    st.rerun()


def navigate_to_overview() -> None:
    st.session_state.current_page = "overview"
    st.session_state.selected_job = None
    for key in list(st.session_state.keys()):
        if key.startswith(("viewer_", "structure_", "model_", "pae_")):
            del st.session_state[key]
    st.rerun()
