"""Private plotting helpers for the STRIDE `.pl` namespace."""
from __future__ import annotations

from collections.abc import Iterable
from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from stride.errors import ContractError

BIO_PASTEL_PALETTE = {
    "blue": "#1D52A1",
    "purple": "#716DB2",
    "cyan": "#65C8CC",
    "green": "#72C15A",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "yellow": "#F0E442",
    "navy": "#0072B2",
    "vermillion": "#D55E00",
    "magenta": "#CC79A7",
    "light_blue": "#D8E6F1",
    "light_grey": "#F2F4F6",
    "grey": "#6F7378",
}

DEFAULT_CATEGORICAL_COLORS = tuple(BIO_PASTEL_PALETTE.values())
SOURCE_TARGET_PALETTE = {
    "source": BIO_PASTEL_PALETTE["navy"],
    "target": BIO_PASTEL_PALETTE["magenta"],
}
DEFAULT_GROUP_PALETTE = (
    BIO_PASTEL_PALETTE["orange"],
    BIO_PASTEL_PALETTE["sky"],
    BIO_PASTEL_PALETTE["teal"],
    BIO_PASTEL_PALETTE["purple"],
    BIO_PASTEL_PALETTE["vermillion"],
)


def _bio_continuous_cmap() -> LinearSegmentedColormap:
    """Return the descriptive `.pl` continuous community-fraction cmap."""
    return LinearSegmentedColormap.from_list(
        "stride_bio_fraction",
        [
            "#F7FAFC",
            BIO_PASTEL_PALETTE["light_blue"],
            BIO_PASTEL_PALETTE["blue"],
            BIO_PASTEL_PALETTE["teal"],
        ],
    )


def _effect_size_cmap(effect_size_type: str):
    """Return the color scale for a supplied `.da` effect-size type."""
    if effect_size_type == "cliffs_delta":
        return plt.get_cmap("RdBu_r")
    return _bio_continuous_cmap()


def _ordered_unique(values: Iterable[object]) -> tuple[str, ...]:
    """Return unique non-missing values in first-observed string order."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        label = str(value)
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return tuple(unique)


def _categorical_color_map(values: Iterable[object]) -> dict[str, str]:
    """Map categorical labels to the default descriptive palette."""
    labels = tuple(sorted(_ordered_unique(values)))
    return {
        label: DEFAULT_CATEGORICAL_COLORS[index % len(DEFAULT_CATEGORICAL_COLORS)]
        for index, label in enumerate(labels)
    }


def _group_color_map(values: Iterable[object]) -> dict[str, str]:
    """Map patient group labels to a deterministic group palette."""
    labels = tuple(sorted(_ordered_unique(values)))
    return {
        label: DEFAULT_GROUP_PALETTE[index % len(DEFAULT_GROUP_PALETTE)]
        for index, label in enumerate(labels)
    }


def _set_right_angle_xticklabels(ax) -> None:
    """Use one vertical tick-label convention across STRIDE plots."""
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")
        label.set_va("center")
        label.set_rotation_mode("anchor")


def _auto_grid(n_panels: int, *, max_cols: int = 3) -> tuple[int, int]:
    """Return a compact deterministic grid for small multi-panel figures."""
    if n_panels < 1:
        raise ContractError("number of panels must be positive")
    n_cols = min(max_cols, max(1, ceil(sqrt(n_panels))))
    n_rows = ceil(n_panels / n_cols)
    return n_rows, n_cols


def _star_from_value(value: float) -> str:
    """Format a p/q value as fixed-threshold significance stars."""
    if not np.isfinite(value):
        raise ContractError("stats p_value/q_value must be finite")
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    if value <= 0.05:
        return "*"
    return "ns"


def _format_pq_label(
    value: float | None,
    kind: str,
    *,
    show_ns: bool,
) -> str | None:
    """Format a supplied p/q value for display without computing statistics."""
    if value is None:
        return None
    value_float = float(value)
    star = _star_from_value(value_float)
    if star == "ns" and not show_ns:
        return None
    return f"{kind}={value_float:.3g} {star}"


def _save_or_return_figure(fig: Figure, save: str | Path | None) -> Figure | None:
    """Return a figure or export it as a PDF and close it."""
    if save is None:
        return fig

    path = Path(save)
    if path.suffix.lower() != ".pdf":
        raise ContractError("save must be a PDF path ending in '.pdf'")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return None


def _apply_clean_axes_style(ax) -> None:
    """Apply the low-decoration scientific plotting style used by `.pl`."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#6F7378")
    ax.tick_params(width=0.6, color="#6F7378")


def _apply_figure_margins(
    fig: Figure,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    """Apply explicit figure margins for long labels and external legends."""
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )


def _default_figsize(
    kind: str,
    n_rows: int,
    n_cols: int,
    n_panels: int = 1,
) -> tuple[float, float]:
    """Estimate a deterministic figure size for descriptive plots."""
    if kind == "community_annotation":
        return (
            max(10.5, 0.58 * n_cols + 7.0),
            max(4.8, 0.42 * n_rows + 2.2),
        )
    if kind == "fov_composition":
        return (
            max(8.5, 0.58 * n_cols + 5.0),
            max(4.8, min(14.0, 0.16 * n_rows + 2.4)),
        )
    if kind == "fraction_comparison":
        n_rows, n_panel_cols = _auto_grid(n_panels, max_cols=3)
        return (
            max(6.5, min(16.0, 4.6 * n_panel_cols)),
            max(4.4, n_rows * (0.42 * n_cols + 2.8)),
        )
    return (7.0, 4.0)


__all__ = [
    "BIO_PASTEL_PALETTE",
    "DEFAULT_CATEGORICAL_COLORS",
    "DEFAULT_GROUP_PALETTE",
    "SOURCE_TARGET_PALETTE",
]
