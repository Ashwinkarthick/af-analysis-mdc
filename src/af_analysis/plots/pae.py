"""PAE plotting utilities."""

from __future__ import annotations

import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at runtime in Streamlit UI
    go = None


def _read_payload(path: Path) -> Any:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    opener = gzip.open if suffixes.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def _first_existing(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _extract_pae_payload(payload: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the raw PAE matrix plus its metadata payload.

    AF2/ColabFold PAE JSONs usually store ``predicted_aligned_error`` or
    ``pae`` directly. AF3 stores token metadata next to ``pae`` in
    ``*_confidences.json``. Keep that metadata so token-level AF3 PAE can be
    converted to residue-level PAE before plotting.
    """
    if isinstance(payload, list) and payload:
        payload = payload[0]

    metadata: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        matrix = _first_existing(
            payload,
            (
                "predicted_aligned_error",
                "pae",
                "pae_matrix",
                "predicted_aligned_error_matrix",
            ),
        )
    else:
        matrix = payload

    if matrix is None:
        raise ValueError("No PAE matrix found in payload")

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("PAE matrix is not a non-empty 2D array")
    return arr, metadata


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _af3_token_groups(metadata: Dict[str, Any], n_tokens: int) -> Optional[List[np.ndarray]]:
    """Build consecutive token groups that map AF3 atom/tokens to residues.

    AF3 confidence files may contain one PAE row/column per model token. For
    ligands and other non-standard components this can be finer than residue
    resolution, which makes residue-level PAE plots appear blocky and distorted.
    When chain and residue ids are available, adjacent tokens with the same
    (chain, residue) pair are collapsed into one residue bin.
    """
    chain_ids = _as_list(
        _first_existing(
            metadata,
            (
                "token_chain_ids",
                "token_asym_ids",
                "token_chain_id",
                "chain_ids",
                "asym_ids",
            ),
        )
    )
    residue_ids = _as_list(
        _first_existing(
            metadata,
            (
                "token_res_ids",
                "token_residue_ids",
                "token_residue_id",
                "token_residue_indices",
                "token_residue_index",
                "residue_ids",
                "res_ids",
            ),
        )
    )

    if not chain_ids or not residue_ids:
        return None
    if len(chain_ids) != n_tokens or len(residue_ids) != n_tokens:
        return None

    groups: List[List[int]] = []
    current_key: Optional[Tuple[Any, Any]] = None
    current: List[int] = []
    for idx, (chain, residue) in enumerate(zip(chain_ids, residue_ids)):
        # Missing residue ids cannot be safely merged. Treat each such token as
        # a distinct residue/bin while preserving its chain for boundaries.
        if residue in (None, "", ".", "?"):
            key = (chain, f"__token_{idx}")
        else:
            key = (chain, residue)
        if current and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        current.append(idx)
    if current:
        groups.append(current)

    # Do not transform ordinary one-token-per-residue AF2/AF3 files.
    if len(groups) == n_tokens:
        return None
    return [np.asarray(group, dtype=int) for group in groups]


def _collapse_token_pae_to_residue_pae(matrix: np.ndarray, groups: List[np.ndarray]) -> np.ndarray:
    """Collapse token-level PAE to residue-level PAE using mean token pairs."""
    n_groups = len(groups)
    collapsed = np.empty((n_groups, n_groups), dtype=float)
    for i, rows in enumerate(groups):
        block_rows = matrix[rows, :]
        for j, cols in enumerate(groups):
            collapsed[i, j] = float(np.nanmean(block_rows[:, cols]))
    return collapsed


def _load_pae_matrix(pae_file: Path) -> np.ndarray:
    path = Path(pae_file)
    payload = _read_payload(path)
    arr, metadata = _extract_pae_payload(payload)

    # AF3 confidence files can be token-level. Collapse adjacent tokens that map
    # to the same residue so the plot resolution matches AF2 residue PAE plots.
    if arr.shape[0] == arr.shape[1]:
        groups = _af3_token_groups(metadata, int(arr.shape[0]))
        if groups is not None:
            arr = _collapse_token_pae_to_residue_pae(arr, groups)
    return arr


def _downsample_matrix(matrix: np.ndarray, max_points: int = 650) -> Tuple[np.ndarray, int]:
    n = int(max(matrix.shape))
    stride = max(1, int(math.ceil(n / max_points)))
    return matrix[::stride, ::stride], stride


def plot_pae_heatmap_interactive(pae_file: Path, max_points: int = 650):
    """Return a compact interactive Plotly PAE heatmap with residue-number hover."""
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
        title += f" — displayed every {stride} residues"
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
