"""Interactive overview plots for AlphaPulldown triage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
except Exception:  # pragma: no cover - handled in UI
    px = None
    go = None
    sample_colorscale = None


TRUE_DOCKQ_NAMES = {
    "dockq",
    "true_dockq",
    "interface_dockq",
    "dockq_score",
    "global_dockq_score",
}

# Names frequently present in AlphaJudge/AF-Analysis caches but not true DockQ.
# These metrics are useful, but must not be drawn as DockQ.
NOT_TRUE_DOCKQ_NAMES = {
    "global_dockq",
    "normalized_dockq",
    "pdockq",
    "mpdockq",
    "pdockq_mpdockq",
    "aj_pdockq_mpdockq",
}

OKABE_ITO = {
    "High": "#0072B2",
    "Medium": "#009E73",
    "Acceptable": "#E69F00",
    "Incorrect": "#D55E00",
    "Very high": "#0072B2",
    "Low": "#CC79A7",
    "Unknown": "#999999",
}


# Shared color-blind-friendly viridis direction for interface-analysis plots:
# low values start yellow and high values end purple. This matches the requested
# PAE scale and applies the same visual language to PI score.
RESULT_YELLOW_TO_PURPLE_SCALE = [
    "#FDE725",  # yellow
    "#35B779",  # green
    "#31688E",  # blue
    "#440154",  # deep purple
]

RESULT_TOP20_SCALE = RESULT_YELLOW_TO_PURPLE_SCALE
RESULT_PAE_SCALE = RESULT_YELLOW_TO_PURPLE_SCALE

ALPHAJUDGE_SCALE = [
    "#3B0F70",  # purple
    "#8C2981",  # magenta
    "#DE4968",  # coral
    "#FE9F6D",  # orange
    "#FDE725",  # yellow
]




def _selected_job_from_plot_event(event, source_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Extract the selected job from a Streamlit Plotly selection event."""
    points = []
    try:
        selection = getattr(event, "selection", None)
        if selection is not None:
            points = list(getattr(selection, "points", []) or [])
    except Exception:
        points = []
    if not points and isinstance(event, dict):
        selection = event.get("selection", {}) or {}
        points = list(selection.get("points", []) or [])

    if not points:
        return None

    point = points[0]
    customdata = None
    if isinstance(point, dict):
        customdata = point.get("customdata")
    else:
        customdata = getattr(point, "customdata", None)

    if isinstance(customdata, (list, tuple)) and customdata:
        job = customdata[0]
        if pd.notna(job):
            return str(job)
    if isinstance(customdata, str) and customdata:
        return customdata

    point_index = None
    if isinstance(point, dict):
        point_index = point.get("point_index", point.get("pointIndex", point.get("pointNumber")))
    else:
        point_index = getattr(point, "point_index", None)
        if point_index is None:
            point_index = getattr(point, "pointIndex", None)
        if point_index is None:
            point_index = getattr(point, "pointNumber", None)

    if source_df is not None and point_index is not None and "job" in source_df.columns:
        try:
            idx = int(point_index)
            if 0 <= idx < len(source_df):
                job = source_df.iloc[idx]["job"]
                if pd.notna(job):
                    return str(job)
        except Exception:
            return None
    return None


def _plotly_chart_with_click(fig, key: str, source_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Render a Plotly chart and return a clicked job when supported by Streamlit."""
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return None
    return _selected_job_from_plot_event(event, source_df)


def _colors_for_continuous_values(values: pd.Series, cmin: float, cmax: float, scale: list[str]) -> list[str]:
    """Return per-point CSS colors so thin text-x markers can be colored reliably.

    Plotly's built-in ``x-thin`` marker can render as monochrome in some
    versions because open-marker stroke color is not always mapped through the
    continuous colorscale. Text glyphs avoid that renderer-specific behavior.
    """
    colors: list[str] = []
    denom = max(float(cmax) - float(cmin), 1e-12)
    for value in pd.to_numeric(values, errors="coerce"):
        if pd.isna(value):
            colors.append("#999999")
            continue
        t = (float(value) - float(cmin)) / denom
        t = min(1.0, max(0.0, t))
        if sample_colorscale is None:
            colors.append(scale[-1])
        else:
            colors.append(sample_colorscale(scale, [t])[0])
    return colors

def _numeric_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series
    return None


def _numeric_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[Tuple[pd.Series, str]]:
    for col in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series, col
    return None


def _normalized_name(column_name: str) -> str:
    normalized = str(column_name).strip().lower().replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("/", "_").replace("+", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _is_true_dockq_column(column_name: str) -> bool:
    normalized = _normalized_name(column_name)
    if normalized in NOT_TRUE_DOCKQ_NAMES:
        return False
    if "normalized" in normalized or "mpdockq" in normalized or "pdockq" in normalized:
        return False
    return normalized in TRUE_DOCKQ_NAMES


def _read_dockq_from_interfaces_csv(csv_path: Path) -> Optional[float]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    dockq_cols = [col for col in df.columns if _is_true_dockq_column(col)]
    if not dockq_cols:
        return None
    values = pd.to_numeric(df[dockq_cols[0]], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _dockq_series_from_cached_paths(df: pd.DataFrame) -> Optional[pd.Series]:
    values = []
    found = False
    for _, row in df.iterrows():
        candidates = []
        interface_csv = row.get("interface_csv") if "interface_csv" in row.index else None
        if isinstance(interface_csv, str) and interface_csv.strip():
            candidates.append(Path(interface_csv))
        path_value = row.get("path") if "path" in row.index else None
        if isinstance(path_value, str) and path_value.strip():
            candidates.append(Path(path_value) / "interfaces.csv")
        value = None
        for candidate in candidates:
            value = _read_dockq_from_interfaces_csv(candidate)
            if value is not None:
                break
        if value is not None:
            found = True
        values.append(value)
    return pd.Series(values, index=df.index, dtype="float64") if found else None


def _dockq_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return true DockQ values only; never use pDockQ/mpDockQ/global_dockq fallbacks."""
    explicit_cols = [col for col in df.columns if _is_true_dockq_column(col)]
    series = _numeric_series(df, explicit_cols)
    if series is not None:
        return series
    return _dockq_series_from_cached_paths(df)


def _confidence_series(df: pd.DataFrame) -> Optional[pd.Series]:
    return _numeric_series(df, ["iptm_ptm", "confidence_score", "iptm"])


def _result_x_metric_series(df: pd.DataFrame) -> Optional[Tuple[pd.Series, str, str, bool]]:
    """Return x-axis metric as (series, column_name, display_label, is_true_dockq)."""
    dockq = _dockq_series(df)
    if dockq is not None:
        return dockq, "dockq", "DockQ", True
    mpdockq = _numeric_series(df, ["mpDockQ"])
    if mpdockq is not None:
        return mpdockq, "mpDockQ", "mpDockQ", False
    return None


def _dockq_accuracy(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "Unknown"
    if pd.isna(v):
        return "Unknown"
    if v >= 0.80:
        return "High"
    if v >= 0.49:
        return "Medium"
    if v >= 0.23:
        return "Acceptable"
    return "Incorrect"


PAE_INCORRECT_THRESHOLD = 15.0


def _pae_color_series(values: pd.Series) -> pd.Series:
    """Continuous PAE color scale with >15 Å clipped to the incorrect end."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.clip(upper=PAE_INCORRECT_THRESHOLD)


def _as_0_to_1(values: pd.Series, default: float = 0.0) -> pd.Series:
    """Convert confidence-like values to 0..1 without rescaling valid values."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.clip(lower=0.0, upper=1.0).fillna(default)


def _relative_0_to_1(values: pd.Series, higher_is_better: bool = True, default: float = 0.0) -> pd.Series:
    """Normalize arbitrary-score columns within the current filtered run."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(default, index=values.index, dtype="float64")
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        normalized = pd.Series(1.0, index=values.index, dtype="float64")
    else:
        normalized = (numeric - min_value) / (max_value - min_value)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized.clip(lower=0.0, upper=1.0).fillna(default)


def _pae_rank_score(values: pd.Series) -> pd.Series:
    """Map min inter-chain PAE to 0..1, clipping >15 Å to the incorrect end."""
    numeric = pd.to_numeric(values, errors="coerce")
    return (1.0 - numeric.clip(lower=0.0, upper=PAE_INCORRECT_THRESHOLD) / PAE_INCORRECT_THRESHOLD).clip(0.0, 1.0).fillna(0.0)


def _interface_quality_rank_score(plot_df: pd.DataFrame, is_true_dockq: bool) -> tuple[pd.Series, str]:
    """Return a 0..1 interface-quality score for ranking Top 20 interactions.

    True DockQ, when explicitly available, is already on a 0..1 quality scale.
    Otherwise AF-Analysis uses mpDockQ as a relative within-run proxy;
    it is not relabeled as DockQ.
    """
    if is_true_dockq:
        return _as_0_to_1(plot_df["overview_x"]), "DockQ"
    if "mpDockQ" in plot_df.columns and pd.to_numeric(plot_df["mpDockQ"], errors="coerce").notna().any():
        return _relative_0_to_1(plot_df["mpDockQ"], higher_is_better=True), "relative mpDockQ"
    return pd.Series(0.0, index=plot_df.index, dtype="float64"), "interface quality unavailable"


def _add_interaction_selection_score(plot_df: pd.DataFrame, is_true_dockq: bool) -> tuple[pd.DataFrame, str]:
    """Add a transparent composite ranking score for Top 20 interaction selection.

    The score prioritizes AlphaFold-Multimer ranking confidence, inter-chain PAE,
    and DockQ-style interface quality. PI score and interface area remain visual
    encodings, not the primary Top 20 selection rule.
    """
    plot_df = plot_df.copy()
    confidence_score = _as_0_to_1(plot_df["model_confidence"])
    pae_source = plot_df["pae_value"] if "pae_value" in plot_df.columns else pd.Series(pd.NA, index=plot_df.index)
    pae_score = _pae_rank_score(pae_source)
    quality_score, quality_label = _interface_quality_rank_score(plot_df, is_true_dockq)
    plot_df["top20_confidence_component"] = confidence_score
    plot_df["top20_pae_component"] = pae_score
    plot_df["top20_quality_component"] = quality_score
    plot_df["top20_score"] = (0.40 * confidence_score) + (0.35 * pae_score) + (0.25 * quality_score)
    return plot_df, quality_label

def _confidence_band(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "Unknown"
    if pd.isna(v):
        return "Unknown"
    if v >= 0.80:
        return "Very high"
    if v >= 0.60:
        return "High"
    if v >= 0.40:
        return "Medium"
    return "Low"


def _format_hover_value(value, digits: int = 3, missing: str = "N/A") -> str:
    try:
        if pd.isna(value):
            return missing
        return f"{float(value):.{digits}f}"
    except Exception:
        return missing if value is None else str(value)


def _add_hover_column(df: pd.DataFrame, source: str, output: str, digits: int = 3) -> None:
    if source in df.columns:
        df[output] = pd.to_numeric(df[source], errors="coerce").map(lambda v: _format_hover_value(v, digits))
    else:
        df[output] = "N/A"


def _prepare_result_plot_df(filtered_df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, str, bool, Optional[list]]]:
    x_metric = _result_x_metric_series(filtered_df)
    confidence = _confidence_series(filtered_df)
    if x_metric is None or confidence is None:
        st.info(
            "Interface-analysis overview plots require model-confidence values plus either true DockQ or "
            "mpDockQ. AF-Analysis does not substitute pDockQ/global_dockq as DockQ."
        )
        return None

    x_values, _x_name, x_label, is_true_dockq = x_metric
    plot_df = filtered_df.copy()
    plot_df["overview_x"] = x_values
    plot_df["model_confidence"] = confidence
    pae_values = _numeric_series(plot_df, ["min_inter_chain_pae", "mean_pae", "min_pae"])
    if pae_values is not None:
        plot_df["pae_value"] = pae_values
        plot_df["pae_color"] = _pae_color_series(pae_values)
    else:
        plot_df["pae_value"] = pd.NA
        plot_df["pae_color"] = pd.NA
        st.caption("min_inter_chain_pae was not found, so PAE-coloring is unavailable for this plot.")
    plot_df = plot_df[plot_df["overview_x"].notna() & plot_df["model_confidence"].notna()].copy()
    if plot_df.empty:
        st.info("No complete rows are available for interface-analysis overview plots after filtering.")
        return None

    x_min = float(plot_df["overview_x"].min(skipna=True))
    x_max = float(plot_df["overview_x"].max(skipna=True))
    x_axis_range = [0, 1] if is_true_dockq and x_min >= 0 and x_max <= 1 else None
    if is_true_dockq and x_axis_range is None:
        st.warning("The selected DockQ values are outside 0..1. AF-Analysis is showing raw values and is not normalizing them.")
    if not is_true_dockq:
        st.caption("No true DockQ column was found. The interface-analysis overview uses mpDockQ on the x-axis; it is not normalized or relabeled as DockQ.")
    return plot_df, x_label, is_true_dockq, x_axis_range


def _add_result_hover_columns(plot_df: pd.DataFrame, x_label: str) -> pd.DataFrame:
    plot_df = plot_df.copy()
    plot_df["hover_x_label"] = x_label
    plot_df["hover_x"] = plot_df["overview_x"].map(lambda v: _format_hover_value(v, 3))
    plot_df["hover_conf"] = plot_df["model_confidence"].map(lambda v: _format_hover_value(v, 3))
    _add_hover_column(plot_df, "iptm", "hover_iptm", 3)
    _add_hover_column(plot_df, "min_inter_chain_pae", "hover_min_pae", 2)
    _add_hover_column(plot_df, "mpDockQ", "hover_mpdockq", 2)
    _add_hover_column(plot_df, "interface_area", "hover_area", 0)
    _add_hover_column(plot_df, "pi_score", "hover_pi", 2)
    if "top20_score" in plot_df.columns:
        plot_df["hover_top20_score"] = pd.to_numeric(plot_df["top20_score"], errors="coerce").map(lambda v: _format_hover_value(v, 3))
    else:
        plot_df["hover_top20_score"] = "N/A"
    return plot_df


def _render_interface_analysis_overview(filtered_df: pd.DataFrame) -> Optional[str]:
    st.markdown("#### Interface-analysis overview")
    prepared = _prepare_result_plot_df(filtered_df)
    if prepared is None:
        return
    plot_df, x_label, is_true_dockq, x_axis_range = prepared
    plot_df = _add_result_hover_columns(plot_df, x_label)

    left, right = st.columns(2)
    with left:
        title = "Model Confidence vs DockQ" if is_true_dockq else "Model Confidence vs mpDockQ"
        color_args = {}
        if "pae_color" in plot_df.columns and pd.to_numeric(plot_df["pae_color"], errors="coerce").notna().any():
            pae_color_values = pd.to_numeric(plot_df["pae_color"], errors="coerce")
            cmin = float(pae_color_values.min(skipna=True))
            cmax = float(pae_color_values.max(skipna=True))
            if cmax <= cmin:
                cmax = cmin + 1e-6
            color_args = {
                "color": "pae_color",
                "color_continuous_scale": RESULT_PAE_SCALE,
                "range_color": [cmin, cmax],
            }
        # Plotly's built-in x-thin marker can render monochrome with continuous
        # colors in some deployed versions. Use a colored text glyph for the
        # visible thin x, plus an invisible marker trace that carries the
        # continuous PAE colorbar.
        custom_values = plot_df[["job", "hover_x_label", "hover_x", "hover_conf", "hover_iptm", "hover_min_pae", "hover_mpdockq"]].to_numpy()
        if color_args and go is not None:
            visible_colors = _colors_for_continuous_values(plot_df["pae_color"], cmin, cmax, RESULT_PAE_SCALE)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["overview_x"],
                    y=plot_df["model_confidence"],
                    mode="markers",
                    marker={
                        "size": 16,
                        "opacity": 0.001,
                        "color": plot_df["pae_color"],
                        "colorscale": RESULT_PAE_SCALE,
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": True,
                        "colorbar": {"title": "min PAE (Å)<br>>15 = incorrect"},
                    },
                    customdata=custom_values,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df["overview_x"],
                    y=plot_df["model_confidence"],
                    mode="text",
                    text=["×"] * len(plot_df),
                    textfont={"size": 15, "color": visible_colors},
                    customdata=custom_values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}: %{customdata[2]}<br>"
                        "Model confidence (ipTM+pTM): %{customdata[3]}<br>"
                        "ipTM: %{customdata[4]}<br>"
                        "min inter-chain PAE: %{customdata[5]} Å<br>"
                        "mpDockQ: %{customdata[6]}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            fig.update_layout(
                title=title,
                height=430,
                xaxis_title=x_label,
                yaxis_title="Model confidence (ipTM+pTM)",
                margin={"l": 40, "r": 20, "t": 60, "b": 45},
            )
        else:
            fig = px.scatter(
                plot_df,
                x="overview_x",
                y="model_confidence",
                custom_data=["job", "hover_x_label", "hover_x", "hover_conf", "hover_iptm", "hover_min_pae", "hover_mpdockq"],
                title=title,
                labels={
                    "overview_x": x_label,
                    "model_confidence": "Model confidence (ipTM+pTM)",
                    "pae_color": "min inter-chain PAE",
                },
                height=430,
                **color_args,
            )
            fig.update_traces(
                marker_symbol="x",
                marker_size=7,
                marker_line_width=0.5,
                opacity=0.9,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}: %{customdata[2]}<br>"
                    "Model confidence (ipTM+pTM): %{customdata[3]}<br>"
                        "ipTM: %{customdata[4]}<br>"
                        "min inter-chain PAE: %{customdata[5]} Å<br>"
                        "mpDockQ: %{customdata[6]}"
                    "<extra></extra>"
                ),
            )
            fig.update_coloraxes(colorbar_title="min PAE (Å)<br>>15 = incorrect")
        if x_axis_range is not None:
            fig.update_xaxes(range=x_axis_range)
        fig.update_yaxes(range=[0, 1])
        selected_job = _plotly_chart_with_click(fig, "overview_result_confidence_vs_metric", plot_df)
        if selected_job:
            return selected_job

    with right:
        required = ["interface_area", "pi_score"]
        missing = [col for col in required if col not in plot_df.columns or pd.to_numeric(plot_df[col], errors="coerce").notna().sum() == 0]
        if missing:
            st.info("Top 20 interface-analysis plot needs these columns from predictions_with_pae_cutoff_100_0.csv: " + ", ".join(missing))
            return
        top_df = plot_df.copy()
        top_df["interface_area_plot"] = pd.to_numeric(top_df["interface_area"], errors="coerce")
        top_df["pi_score_plot"] = pd.to_numeric(top_df["pi_score"], errors="coerce")
        top_df = top_df[top_df["interface_area_plot"].notna() & top_df["pi_score_plot"].notna()].copy()
        top_df, quality_label = _add_interaction_selection_score(top_df, is_true_dockq)
        if "pae_value" in top_df.columns and pd.to_numeric(top_df["pae_value"], errors="coerce").notna().any():
            non_incorrect = top_df[pd.to_numeric(top_df["pae_value"], errors="coerce") <= PAE_INCORRECT_THRESHOLD].copy()
            if len(non_incorrect) >= 20:
                top_df = non_incorrect
        top_df = top_df.sort_values(["top20_score", "model_confidence", "top20_pae_component", "top20_quality_component"], ascending=False).head(20)
        if top_df.empty:
            st.info("No complete rows are available for the Top 20 interface-analysis plot.")
            return
        top_df = _add_result_hover_columns(top_df, x_label)
        st.caption(
            "Top 20 is ranked by a composite interaction score: "
            "40% model confidence (ipTM+pTM), 35% min inter-chain PAE confidence, "
            f"25% {quality_label}. PI score colors the circles; interface area controls size."
        )
        # Build this with graph_objects instead of Plotly Express so the hover
        # content is exactly controlled. Plotly Express can add default x/y
        # fields such as a separate ipTM+pTM line in some versions.
        area_values = pd.to_numeric(top_df["interface_area_plot"], errors="coerce")
        min_area = float(area_values.min(skipna=True))
        max_area = float(area_values.max(skipna=True))
        if max_area <= min_area:
            marker_sizes = [22.0] * len(top_df)
        else:
            marker_sizes = (14.0 + 28.0 * (area_values - min_area) / (max_area - min_area)).fillna(14.0).tolist()

        pi_values = pd.to_numeric(top_df["pi_score_plot"], errors="coerce")
        cmin = float(pi_values.min(skipna=True))
        cmax = float(pi_values.max(skipna=True))
        if cmax <= cmin:
            cmax = cmin + 1e-6

        custom_values = top_df[[
            "job",
            "hover_x_label",
            "hover_x",
            "hover_conf",
            "hover_area",
            "hover_pi",
            "hover_min_pae",
            "hover_mpdockq",
            "hover_top20_score",
        ]].to_numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=top_df["overview_x"],
                y=top_df["model_confidence"],
                mode="markers",
                marker={
                    "size": marker_sizes,
                    "color": top_df["pi_score_plot"],
                    "colorscale": RESULT_TOP20_SCALE,
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": True,
                    "colorbar": {"title": "PI score"},
                    "opacity": 0.86,
                    "line": {"width": 0.6, "color": "rgba(30,30,30,0.45)"},
                },
                customdata=custom_values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}: %{customdata[2]}<br>"
                    "Model confidence (ipTM+pTM): %{customdata[3]}<br>"
                    "Interface area: %{customdata[4]} Å²<br>"
                    "PI score: %{customdata[5]}<br>"
                    "min inter-chain PAE: %{customdata[6]} Å<br>"
                    "mpDockQ: %{customdata[7]}<br>"
                    "Top-20 score: %{customdata[8]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.update_layout(
            title=f"Top 20 Interactions<br><sup>(ranked by confidence + PAE + interface quality; x = {x_label}; size = Interface Area; color = PI score)</sup>",
            height=430,
            xaxis_title=x_label,
            yaxis_title="Model confidence (ipTM+pTM)",
            showlegend=False,
            margin={"l": 40, "r": 20, "t": 70, "b": 45},
        )
        if x_axis_range is not None:
            fig.update_xaxes(range=x_axis_range)
        fig.update_yaxes(range=[0, 1])
        selected_job = _plotly_chart_with_click(fig, "overview_result_top20_interactions", top_df)
        if selected_job:
            return selected_job


def _first_available_metric(df: pd.DataFrame, candidates: Iterable[Tuple[str, str]]) -> Optional[Tuple[pd.Series, str, str]]:
    for col, label in candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return series, col, label
    return None


def _add_alpha_hover_columns(plot_df: pd.DataFrame, x_label: str, y_label: str, color_label: str, size_label: str) -> pd.DataFrame:
    plot_df = plot_df.copy()
    plot_df["hover_x_label"] = x_label
    plot_df["hover_y_label"] = y_label
    plot_df["hover_color_label"] = color_label
    plot_df["hover_size_label"] = size_label
    plot_df["hover_x"] = plot_df["aj_overview_x"].map(lambda v: _format_hover_value(v, 3))
    plot_df["hover_y"] = plot_df["aj_overview_y"].map(lambda v: _format_hover_value(v, 3))
    if "aj_color" in plot_df.columns:
        plot_df["hover_color"] = pd.to_numeric(plot_df["aj_color"], errors="coerce").map(lambda v: _format_hover_value(v, 3))
    else:
        plot_df["hover_color"] = "N/A"
    if "aj_size" in plot_df.columns:
        plot_df["hover_size"] = pd.to_numeric(plot_df["aj_size"], errors="coerce").map(lambda v: _format_hover_value(v, 0))
    else:
        plot_df["hover_size"] = "N/A"
    _add_hover_column(plot_df, "aj_average_interface_pae", "hover_aj_pae", 2)
    if "hover_aj_pae" not in plot_df.columns or (plot_df["hover_aj_pae"] == "N/A").all():
        _add_hover_column(plot_df, "average_interface_pae", "hover_aj_pae", 2)
    return plot_df


def _render_alphajudge_overview(filtered_df: pd.DataFrame) -> Optional[str]:
    st.markdown("#### AlphaJudge overview")
    st.caption("This plot intentionally avoids interface pLDDT as the main axis; for multimer triage, ipSAE/LIS/pDockQ-style interface scores are more informative.")
    x_metric = _first_available_metric(
        filtered_df,
        [
            ("aj_interface_ipSAE", "AlphaJudge ipSAE"),
            ("best_interface_ipsae", "AlphaJudge ipSAE"),
            ("aj_interface_pDockQ2", "AlphaJudge pDockQ2"),
            ("best_interface_pdockq2", "AlphaJudge pDockQ2"),
            ("aj_interface_LIS", "AlphaJudge LIS"),
            ("best_interface_lis", "AlphaJudge LIS"),
            ("aj_pDockQ/mpDockQ", "AlphaJudge pDockQ/mpDockQ"),
            ("global_dockq", "AlphaJudge pDockQ/mpDockQ"),
        ],
    )
    y_metric = _first_available_metric(
        filtered_df,
        [
            ("aj_interface_LIS", "AlphaJudge LIS"),
            ("best_interface_lis", "AlphaJudge LIS"),
            ("aj_interface_pDockQ2", "AlphaJudge pDockQ2"),
            ("best_interface_pdockq2", "AlphaJudge pDockQ2"),
            ("aj_pDockQ/mpDockQ", "AlphaJudge pDockQ/mpDockQ"),
            ("global_dockq", "AlphaJudge pDockQ/mpDockQ"),
            ("aj_interface_score", "AlphaJudge score"),
            ("interface_score", "AlphaJudge score"),
        ],
    )
    if x_metric is None or y_metric is None:
        st.info("AlphaJudge overview needs AlphaJudge interface score columns such as aj_interface_ipSAE, aj_interface_LIS, or aj_interface_pDockQ2.")
        return

    x_values, x_col, x_label = x_metric
    y_values, y_col, y_label = y_metric

    # Avoid plotting the same metric against itself when possible.
    if x_col == y_col:
        alternate_y = _first_available_metric(
            filtered_df,
            [
                ("aj_interface_pDockQ2", "AlphaJudge pDockQ2"),
                ("best_interface_pdockq2", "AlphaJudge pDockQ2"),
                ("aj_pDockQ/mpDockQ", "AlphaJudge pDockQ/mpDockQ"),
                ("global_dockq", "AlphaJudge pDockQ/mpDockQ"),
                ("aj_interface_score", "AlphaJudge score"),
                ("interface_score", "AlphaJudge score"),
                ("aj_interface_LIS", "AlphaJudge LIS"),
                ("best_interface_lis", "AlphaJudge LIS"),
            ],
        )
        if alternate_y is not None and alternate_y[1] != x_col:
            y_values, y_col, y_label = alternate_y

    plot_df = filtered_df.copy()
    plot_df["aj_overview_x"] = x_values
    plot_df["aj_overview_y"] = y_values

    color_metric = _first_available_metric(
        plot_df,
        [
            ("aj_interface_score", "AlphaJudge score"),
            ("interface_score", "AlphaJudge score"),
            ("aj_interface_ipSAE", "AlphaJudge ipSAE"),
            ("best_interface_ipsae", "AlphaJudge ipSAE"),
            ("aj_interface_LIS", "AlphaJudge LIS"),
            ("best_interface_lis", "AlphaJudge LIS"),
        ],
    )
    size_metric = _first_available_metric(
        plot_df,
        [
            ("aj_interface_area", "AlphaJudge interface area"),
            ("aj_interface_contact_pairs", "AlphaJudge contact pairs"),
            ("interface_contact_pairs", "AlphaJudge contact pairs"),
            ("aj_interface_num_intf_residues", "AlphaJudge residues"),
            ("interface_residue_count", "AlphaJudge residues"),
        ],
    )
    color_col = None
    size_col = None
    if color_metric is not None:
        plot_df["aj_color"] = color_metric[0]
        color_col = "aj_color"
        color_label = color_metric[2]
    else:
        color_label = "AlphaJudge score"
    if size_metric is not None:
        plot_df["aj_size"] = size_metric[0]
        size_col = "aj_size"
        size_label = size_metric[2]
    else:
        plot_df["aj_size"] = 10
        size_col = "aj_size"
        size_label = "Point size"

    plot_df = plot_df[plot_df["aj_overview_x"].notna() & plot_df["aj_overview_y"].notna()].copy()
    if plot_df.empty:
        st.info("No complete rows are available for the AlphaJudge overview after filtering.")
        return
    sort_cols = ["aj_overview_x", "aj_overview_y"]
    if color_col:
        sort_cols.insert(0, "aj_color")
    plot_df = plot_df.sort_values(sort_cols, ascending=False).head(50)
    plot_df = _add_alpha_hover_columns(plot_df, x_label, y_label, color_label, size_label)

    fig = px.scatter(
        plot_df,
        x="aj_overview_x",
        y="aj_overview_y",
        size=size_col,
        color=color_col,
        color_continuous_scale=ALPHAJUDGE_SCALE,
        custom_data=["job", "hover_x_label", "hover_x", "hover_y_label", "hover_y", "hover_size_label", "hover_size", "hover_color_label", "hover_color", "hover_aj_pae"],
        title=f"AlphaJudge overview<br><sup>(x = {x_label}; y = {y_label}; size = {size_label}; color = {color_label})</sup>",
        labels={
            "aj_overview_x": x_label,
            "aj_overview_y": y_label,
            "aj_color": color_label,
            "aj_size": size_label,
        },
        height=450,
    )
    fig.update_traces(
        mode="markers",
        marker={"opacity": 0.86, "line": {"width": 0.6, "color": "rgba(30,30,30,0.45)"}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}: %{customdata[2]}<br>"
            "%{customdata[3]}: %{customdata[4]}<br>"
            "%{customdata[5]}: %{customdata[6]}<br>"
            "%{customdata[7]}: %{customdata[8]}<br>"
            "Avg interface PAE: %{customdata[9]}"
            "<extra></extra>"
        ),
    )
    fig.update_coloraxes(colorbar_title=color_label)
    selected_job = _plotly_chart_with_click(fig, "overview_alphajudge_summary", plot_df)
    if selected_job:
        return selected_job
    return None


def render_overview_plots(filtered_df: pd.DataFrame) -> Optional[str]:
    """Render requested overview plots with separate interface analysis and AlphaJudge sections."""
    if px is None:
        st.info("Install Plotly to show interactive overview plots.")
        return None

    st.caption("Click any point in an overview plot to open that prediction in the structure viewer.")
    plot_tab, aj_tab = st.tabs(["Result analysis plots", "AlphaJudge plots"])
    with plot_tab:
        selected_job = _render_interface_analysis_overview(filtered_df)
    if selected_job:
        return selected_job
    with aj_tab:
        selected_job = _render_alphajudge_overview(filtered_df)
    return selected_job
