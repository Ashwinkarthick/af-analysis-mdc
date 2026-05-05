"""PAE plotting utilities."""

from __future__ import annotations

import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at runtime in Streamlit UI
    go = None

try:
    from af_analysis.utils import plot_pae_heatmap_interactive as _utils_plot_pae_heatmap_interactive
except Exception:  # pragma: no cover
    _utils_plot_pae_heatmap_interactive = None


def _load_pae_matrix(pae_file: Path) -> np.ndarray:
    path = Path(pae_file)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if isinstance(payload, dict):
            matrix = payload.get("predicted_aligned_error", payload.get("pae"))
            if matrix is None:
                matrix = payload.get("pae_matrix")
        else:
            matrix = payload
    else:
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rb") as handle:
            payload = pickle.load(handle)
        matrix = payload.get("predicted_aligned_error", payload.get("pae")) if isinstance(payload, dict) else payload
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"PAE matrix is not a non-empty 2D array: {path}")
    return arr


def _downsample_matrix(matrix: np.ndarray, max_points: int = 650) -> Tuple[np.ndarray, int]:
    n = int(max(matrix.shape))
    stride = max(1, int(math.ceil(n / max_points)))
    return matrix[::stride, ::stride], stride


def plot_pae_heatmap_interactive(pae_file: Path, max_points: int = 650):
    """Return a compact interactive PAE heatmap with residue-number hover."""
    if _utils_plot_pae_heatmap_interactive is not None:
        return _utils_plot_pae_heatmap_interactive(pae_file, max_points=max_points)
    if go is None:
        raise RuntimeError("Plotly is not installed. Reinstall AF-Analysis after updating pyproject.toml.")

    pae_mtx = _load_pae_matrix(Path(pae_file))
    display_mtx, stride = _downsample_matrix(pae_mtx, max_points=max_points)
    residues_x = np.arange(1, int(pae_mtx.shape[1]) + 1, stride, dtype=int)
    residues_y = np.arange(1, int(pae_mtx.shape[0]) + 1, stride, dtype=int)

    fig = go.Figure(
        data=go.Heatmap(
            z=display_mtx,
            x=residues_x,
            y=residues_y,
            zmin=0,
            zmax=30,
            colorscale="Greens_r",
            colorbar={"title": "PAE (Å)"},
            hovertemplate=(
                "Scored residue: %{x}<br>"
                "Aligned residue: %{y}<br>"
                "PAE: %{z:.2f} Å"
                "<extra></extra>"
            ),
        )
    )
    title = "Predicted Aligned Error (PAE)"
    if stride > 1:
        title += f" - displayed every {stride} residues"
    fig.update_layout(
        title=title,
        width=560,
        height=560,
        margin={"l": 55, "r": 25, "t": 55, "b": 50},
        xaxis_title="Scored residue",
        yaxis_title="Aligned residue",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
