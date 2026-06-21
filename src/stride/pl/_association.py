"""Association plotting surfaces for STRIDE `.pl`."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from stride.errors import ContractError

from ._utils import (
    BIO_PASTEL_PALETTE,
    _apply_clean_axes_style,
    _apply_figure_margins,
    _effect_size_cmap,
    _save_or_return_figure,
    _set_right_angle_xticklabels,
)


def augmented_relation_association_bubble_plot(
    stats: pd.DataFrame,
    *,
    relation_id: str | None = None,
    comparison_id: str | None = None,
    state_labels: Sequence[str] | None = None,
    state_order: Sequence[int] | None = None,
    effect_size_type: Literal["cliffs_delta", "eta_squared"] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> Figure | None:
    """Plot augmented-entry group association results as a bubble matrix.

    Scientific question:
        Which native fitted STRIDE relation entries show caller-provided group
        association after multiple-testing correction?

    Required input:
        `stats` is a `.da` augmented-entry association table. It must already
        contain `row_id`, `col_id`, `entry_type`, `effect_size`,
        `effect_size_type`, `effect_direction`, `q_value`, `relation_id`, and
        `comparison_id`.

    Display object:
        The plot uses the same augmented matrix geometry as
        `M_aug = [[A, d], [e^T, masked]]`: the upper-left block is `A`, the
        right column is `d`, the bottom row is `e`, and the bottom-right cell is
        masked.

    Visual encoding:
        Bubble size encodes binned `-log10(q_value)` with levels 0-4. Level 0
        (`<1`) is a smallest grey point. Levels 1-4 use the effect-size color
        scale. Cliff's delta uses signed diverging color semantics. Eta-squared
        uses unsigned sequential color semantics.

    Boundary:
        This is a rendering function. It does not compute tests, p-values,
        q-values, effect sizes, BH correction, tensor decomposition, or fitted
        STRIDE outputs. It does not mutate inputs or write scientific result
        payloads.
    """
    plot_data = _prepare_stats(
        stats,
        relation_id=relation_id,
        comparison_id=comparison_id,
        effect_size_type=effect_size_type,
    )
    K = _n_states_from_stats(plot_data)
    order = _resolve_state_order(K, state_order)
    labels = _resolve_state_labels(K, state_labels)
    ordered_labels = [labels[index] for index in order]
    plot_data = _reindex_augmented_coordinates(plot_data, order=order, n_states=K)

    resolved_effect_type = _single_value(plot_data["effect_size_type"], "effect_size_type")
    if figsize is None:
        figsize = (max(7.0, 0.62 * (K + 1) + 4.2), max(5.2, 0.58 * (K + 1) + 2.8))
    fig, ax = plt.subplots(figsize=figsize)

    scatter, norm, cmap = _draw_association_bubbles(
        ax,
        plot_data,
        n_states=K,
        effect_size_type=resolved_effect_type,
    )
    x_labels = [*ordered_labels, "source open d"]
    y_labels = [*ordered_labels, "target open e"]
    ax.set_xticks(np.arange(K + 1), labels=x_labels)
    ax.set_yticks(np.arange(K + 1), labels=y_labels)
    _set_right_angle_xticklabels(ax)
    ax.set_xlim(-0.5, K + 0.5)
    ax.set_ylim(K + 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Target community")
    ax.set_ylabel("Source community")
    ax.axhline(K - 0.5, color="#4A4A4A", linewidth=0.8)
    ax.axvline(K - 0.5, color="#4A4A4A", linewidth=0.8)
    ax.scatter([K], [K], s=36.0, marker="s", color="#E8E8E8", edgecolors="none", zorder=1)
    _add_association_legends(fig, ax, scatter=scatter, norm=norm, cmap=cmap, effect_size_type=resolved_effect_type)
    _apply_clean_axes_style(ax)
    _apply_figure_margins(fig, left=0.24, right=0.66, bottom=0.30, top=0.95)
    return _save_or_return_figure(fig, save)


_REQUIRED_STATS_COLUMNS = {
    "relation_id",
    "comparison_id",
    "row_id",
    "col_id",
    "entry_type",
    "effect_size",
    "effect_size_type",
    "q_value",
}


def _prepare_stats(
    stats: pd.DataFrame,
    *,
    relation_id: str | None,
    comparison_id: str | None,
    effect_size_type: Literal["cliffs_delta", "eta_squared"] | None,
) -> pd.DataFrame:
    """Validate and filter the supplied downstream association table."""
    if not isinstance(stats, pd.DataFrame):
        raise ContractError("stats must be a pandas DataFrame")
    missing = sorted(_REQUIRED_STATS_COLUMNS.difference(stats.columns))
    if missing:
        raise ContractError("stats is missing required columns: " + ", ".join(missing))
    data = stats.copy()
    if relation_id is not None:
        data = data[data["relation_id"].astype(str) == str(relation_id)]
    if comparison_id is not None:
        data = data[data["comparison_id"].astype(str) == str(comparison_id)]
    if effect_size_type is not None:
        data = data[data["effect_size_type"].astype(str) == str(effect_size_type)]
    if data.empty:
        raise ContractError("stats filter produced no rows")

    data["row_id"] = _as_int_column(data["row_id"], name="row_id")
    data["col_id"] = _as_int_column(data["col_id"], name="col_id")
    data["effect_size"] = _as_finite_float_column(data["effect_size"], name="effect_size")
    data["q_value"] = _as_finite_float_column(data["q_value"], name="q_value")
    if ((data["q_value"] < 0.0) | (data["q_value"] > 1.0)).any():
        raise ContractError("stats q_value must satisfy 0 <= q_value <= 1")
    effect_types = set(data["effect_size_type"].astype(str))
    if effect_types - {"cliffs_delta", "eta_squared"}:
        raise ContractError("effect_size_type must be cliffs_delta or eta_squared")
    if len(effect_types) != 1:
        raise ContractError("stats must contain one effect_size_type after filtering")
    return data.reset_index(drop=True)


def _draw_association_bubbles(
    ax,
    data: pd.DataFrame,
    *,
    n_states: int,
    effect_size_type: str,
) -> tuple[object, Normalize, object]:
    """Draw one augmented association bubble matrix."""
    levels = np.asarray([_q_value_size_level(float(value)) for value in data["q_value"]])
    sizes = 28.0 + levels * 58.0
    if effect_size_type == "cliffs_delta":
        cmap = _effect_size_cmap(effect_size_type)
        norm = Normalize(vmin=-1.0, vmax=1.0)
        color_values = data["effect_size"].to_numpy(dtype=float)
    else:
        cmap = _effect_size_cmap(effect_size_type)
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(data["effect_size"].max())))
        color_values = np.maximum(data["effect_size"].to_numpy(dtype=float), 0.0)
    colors = cmap(norm(color_values))
    colors[levels == 0] = _hex_to_rgba(BIO_PASTEL_PALETTE["grey"], alpha=0.60)
    scatter = ax.scatter(
        data["col_id"].to_numpy(dtype=float),
        data["row_id"].to_numpy(dtype=float),
        s=sizes,
        c=colors,
        edgecolors="#2F3437",
        linewidths=0.4,
        zorder=2,
    )
    ax.grid(which="major", color="#D9DDE2", linewidth=0.5)
    ax.set_xticks(np.arange(n_states + 1))
    ax.set_yticks(np.arange(n_states + 1))
    return scatter, norm, cmap


def _add_association_legends(
    fig: Figure,
    ax,
    *,
    scatter: object,
    norm: Normalize,
    cmap: object,
    effect_size_type: str,
) -> None:
    """Add explicit bubble-size and effect-size legends."""
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.045,
        pad=0.03,
    )
    colorbar.set_label(_effect_size_label(effect_size_type))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#2F3437",
            markersize=np.sqrt(28.0 + level * 58.0) / 1.8,
            label=label,
        )
        for level, label in (
            (0, "<1"),
            (1, "1-2"),
            (2, "2-3"),
            (3, "3-4"),
            (4, ">=4"),
        )
    ]
    ax.legend(
        handles=handles,
        title="-log10(q)",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.28, 0.98),
    )
    # Keep the scatter object live for test inspection and backends that lazily draw collections.
    _ = scatter


def _effect_size_label(effect_size_type: str) -> str:
    if effect_size_type == "cliffs_delta":
        return "Cliff's delta"
    return "Eta-squared"


def _n_states_from_stats(data: pd.DataFrame) -> int:
    """Infer K from augmented-entry coordinates."""
    max_index = int(max(data["row_id"].max(), data["col_id"].max()))
    if max_index < 1:
        raise ContractError("stats must contain augmented coordinates for at least one state")
    if ((data["row_id"] == max_index) & (data["col_id"] == max_index)).any():
        raise ContractError("bottom-right augmented-matrix cell is masked")
    expected = {"A", "d", "e"}
    entry_types = set(data["entry_type"].astype(str))
    if not entry_types.issubset(expected):
        raise ContractError("stats entry_type must be one of A, d, or e")
    for _, row in data.iterrows():
        entry_type = str(row["entry_type"])
        row_id = int(row["row_id"])
        col_id = int(row["col_id"])
        if row_id < 0 or col_id < 0 or row_id > max_index or col_id > max_index:
            raise ContractError("stats row_id/col_id must be valid augmented coordinates")
        if entry_type == "A" and (row_id == max_index or col_id == max_index):
            raise ContractError("A entries must be in the upper-left augmented block")
        if entry_type == "d" and not (row_id < max_index and col_id == max_index):
            raise ContractError("d entries must be in the source-open column")
        if entry_type == "e" and not (row_id == max_index and col_id < max_index):
            raise ContractError("e entries must be in the target-open row")
    return max_index


def _reindex_augmented_coordinates(
    data: pd.DataFrame,
    *,
    order: tuple[int, ...],
    n_states: int,
) -> pd.DataFrame:
    """Apply state ordering to augmented coordinates without changing data values."""
    index_map = {old: new for new, old in enumerate(order)}
    reordered = data.copy()
    row_values = []
    col_values = []
    for _, row in reordered.iterrows():
        row_id = int(row["row_id"])
        col_id = int(row["col_id"])
        row_values.append(n_states if row_id == n_states else index_map[row_id])
        col_values.append(n_states if col_id == n_states else index_map[col_id])
    reordered["row_id"] = row_values
    reordered["col_id"] = col_values
    return reordered


def _resolve_state_order(
    n_states: int,
    state_order: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve display state order as a full permutation."""
    if state_order is None:
        return tuple(range(n_states))
    order = tuple(int(value) for value in state_order)
    if sorted(order) != list(range(n_states)):
        raise ContractError("state_order must be a complete permutation of 0..K-1")
    return order


def _resolve_state_labels(
    n_states: int,
    state_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve display labels for the shared state basis."""
    if state_labels is None:
        return tuple(f"C{index}" for index in range(n_states))
    labels = tuple(str(label) for label in state_labels)
    if len(labels) != n_states:
        raise ContractError("state_labels length must match n_states")
    return labels


def _q_value_size_level(q_value: float) -> int:
    """Map supplied q-values to documented bubble size levels."""
    if q_value <= 0.0:
        return 4
    score = -np.log10(q_value)
    if score < 1.0:
        return 0
    if score < 2.0:
        return 1
    if score < 3.0:
        return 2
    if score < 4.0:
        return 3
    return 4


def _single_value(values: pd.Series, name: str) -> str:
    """Return one unique string value from a column."""
    unique = tuple(dict.fromkeys(values.astype(str)))
    if len(unique) != 1:
        raise ContractError(f"stats must contain one {name} after filtering")
    return unique[0]


def _as_int_column(values: pd.Series, *, name: str) -> pd.Series:
    """Convert a stats coordinate column to integers."""
    try:
        return values.astype(int)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"stats {name} must contain integer coordinates") from exc


def _as_finite_float_column(values: pd.Series, *, name: str) -> pd.Series:
    """Convert a stats value column to finite floats."""
    try:
        result = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"stats {name} must be numeric") from exc
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ContractError(f"stats {name} must contain only finite values")
    return result


def _hex_to_rgba(hex_color: str, *, alpha: float) -> tuple[float, float, float, float]:
    """Convert a hex color to an RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
        alpha,
    )
