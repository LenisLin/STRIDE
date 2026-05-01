"""SVG figure writers for the Task A descriptive atlas."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DescriptiveAtlasContractError

CONTINUOUS_START = "#f7fbff"
CONTINUOUS_END = "#08306b"
BAR_FILL = "#2171b5"
OVERLAY_BACKGROUND = "#d0d5dd"
COMMUNITY_PALETTE: tuple[str, ...] = (
    "#1b4965",
    "#ef476f",
    "#06d6a0",
    "#ffd166",
    "#118ab2",
    "#8d99ae",
    "#bc6c25",
    "#6a4c93",
    "#2a9d8f",
    "#e76f51",
)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _blend_hex(start: str, end: str, weight: float) -> str:
    clipped = min(max(float(weight), 0.0), 1.0)
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    blended = tuple(
        int(round(start_channel + (end_channel - start_channel) * clipped))
        for start_channel, end_channel in zip(start_rgb, end_rgb, strict=True)
    )
    return _rgb_to_hex(blended)


def _community_color(community_id: int) -> str:
    return COMMUNITY_PALETTE[int(community_id) % len(COMMUNITY_PALETTE)]


def _svg_text(
    *,
    x: float,
    y: float,
    text: str,
    font_size: int = 12,
    anchor: str = "start",
    fill: str = "#111827",
    extra: str = "",
) -> str:
    escaped = html.escape(str(text))
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{font_size}" '
        f'font-family="monospace" text-anchor="{anchor}" fill="{fill}" {extra}>{escaped}</text>'
    )


def _write_svg(path: Path, *, width: int, height: int, elements: list[str]) -> None:
    payload = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(payload), encoding="utf-8")


def write_heatmap_svg(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    value_label: str,
    annotate: bool,
) -> None:
    row_labels = [f"C{int(label)}" for label in matrix.index.tolist()]
    column_labels = [str(label) for label in matrix.columns.tolist()]
    n_rows, n_cols = matrix.shape
    cell_width = 22 if n_cols <= 6 else 18
    cell_height = 22 if n_rows <= 12 else 18
    left_margin = max(140, 12 * max((len(label) for label in row_labels), default=1))
    top_margin = 130 if n_cols > 6 else 90
    right_margin = 110
    bottom_margin = 50
    plot_width = max(1, n_cols) * cell_width
    plot_height = max(1, n_rows) * cell_height
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    values = matrix.to_numpy(dtype=float)
    value_max = float(np.nanmax(values)) if values.size else 0.0
    if value_max <= 0.0:
        value_max = 1.0
    elements = _heatmap_base_elements(title, value_label, left_margin, plot_width, top_margin, plot_height, value_max)
    for row_idx, row_label in enumerate(row_labels):
        y = top_margin + row_idx * cell_height
        elements.append(_svg_text(x=left_margin - 8, y=y + cell_height * 0.7, text=row_label, font_size=11, anchor="end"))
        for col_idx, _column_label in enumerate(column_labels):
            x = left_margin + col_idx * cell_width
            value = float(values[row_idx, col_idx]) if values.size else 0.0
            fill = _blend_hex(CONTINUOUS_START, CONTINUOUS_END, value / value_max)
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" '
                f'fill="{fill}" stroke="#e5e7eb" stroke-width="0.5"/>'
            )
            if annotate:
                label = f"{value:.2f}" if value_label.endswith("fraction") else f"{value:.0f}"
                color = "#111827" if value < value_max * 0.55 else "#ffffff"
                elements.append(_svg_text(x=x + cell_width / 2.0, y=y + cell_height * 0.68, text=label, font_size=9, anchor="middle", fill=color))
    for col_idx, column_label in enumerate(column_labels):
        x = left_margin + col_idx * cell_width + cell_width / 2.0
        y = top_margin - 10
        elements.append(_svg_text(x=x, y=y, text=column_label, font_size=10, extra=f'transform="rotate(-55 {x:.2f} {y:.2f})"'))
    _write_svg(path, width=width, height=height, elements=elements)


def _heatmap_base_elements(
    title: str,
    value_label: str,
    left_margin: int,
    plot_width: int,
    top_margin: int,
    plot_height: int,
    value_max: float,
) -> list[str]:
    elements = [
        _svg_text(x=24, y=30, text=title, font_size=18),
        _svg_text(x=24, y=54, text=value_label, font_size=12, fill="#374151"),
    ]
    legend_x = left_margin + plot_width + 24
    legend_y = top_margin
    legend_height = min(140, plot_height)
    for step in range(20):
        weight = step / 19 if step else 0.0
        color = _blend_hex(CONTINUOUS_START, CONTINUOUS_END, weight)
        y = legend_y + legend_height - ((step + 1) * legend_height / 20.0)
        elements.append(
            f'<rect x="{legend_x:.2f}" y="{y:.2f}" width="18" height="{legend_height / 20.0 + 0.5:.2f}" '
            f'fill="{color}" stroke="none"/>'
        )
    elements.append(
        f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="18" height="{legend_height:.2f}" '
        'fill="none" stroke="#6b7280" stroke-width="0.8"/>'
    )
    elements.append(_svg_text(x=legend_x + 28, y=legend_y + 10, text="max", font_size=11))
    elements.append(_svg_text(x=legend_x + 28, y=legend_y + legend_height + 2, text="0", font_size=11))
    elements.append(_svg_text(x=legend_x + 28, y=legend_y + 24, text=f"{value_max:.2f}", font_size=11, fill="#374151"))
    return elements


def write_horizontal_bar_svg(
    summary: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> None:
    ordered = summary.sort_values(
        by=["patient_prevalence", "total_cells", "community_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    row_height = 24
    top_margin = 70
    bottom_margin = 35
    left_margin = 110
    right_margin = 90
    width = 760
    plot_width = width - left_margin - right_margin
    height = top_margin + max(1, len(ordered)) * row_height + bottom_margin
    elements = [
        _svg_text(x=24, y=30, text=title, font_size=18),
        _svg_text(x=24, y=52, text="Sorted by patient prevalence, then total cells", font_size=12, fill="#374151"),
    ]
    elements.append(
        f'<line x1="{left_margin}" y1="{top_margin - 8}" x2="{left_margin}" y2="{height - bottom_margin}" '
        'stroke="#9ca3af" stroke-width="1"/>'
    )
    _append_bar_axis(elements, left_margin, top_margin, height, bottom_margin, plot_width)
    for row_idx, row in ordered.iterrows():
        y = top_margin + row_idx * row_height
        bar_width = float(row["patient_prevalence"]) * plot_width
        elements.append(_svg_text(x=left_margin - 10, y=y + row_height * 0.68, text=f"C{int(row['community_id'])}", font_size=11, anchor="end"))
        elements.append(
            f'<rect x="{left_margin:.2f}" y="{y + 4:.2f}" width="{bar_width:.2f}" height="{row_height - 8:.2f}" '
            f'fill="{BAR_FILL}" opacity="0.92"/>'
        )
        label = f"{float(row['patient_prevalence']):.2f} ({int(row['n_patients_present'])}/{int(row['n_patients_total'])})"
        elements.append(_svg_text(x=left_margin + bar_width + 8, y=y + row_height * 0.68, text=label, font_size=10, fill="#374151"))
    _write_svg(path, width=width, height=height, elements=elements)


def _append_bar_axis(elements: list[str], left_margin: int, top_margin: int, height: int, bottom_margin: int, plot_width: int) -> None:
    for tick in range(5):
        fraction = tick / 4 if tick else 0.0
        x = left_margin + fraction * plot_width
        elements.append(
            f'<line x1="{x:.2f}" y1="{top_margin - 6}" x2="{x:.2f}" y2="{height - bottom_margin}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(_svg_text(x=x, y=height - 12, text=f"{fraction:.2f}", font_size=10, anchor="middle", fill="#374151"))


def write_overlay_svg(
    roi_frame: pd.DataFrame,
    path: Path,
    *,
    community_id: int,
    patient_id: str,
    domain_label: str,
    fov_id: str,
    community_fraction: float,
    community_cells: int,
    roi_total_cells: int,
) -> None:
    if roi_frame.empty:
        raise DescriptiveAtlasContractError("Representative overlay generation received an empty ROI frame")
    x_min = float(roi_frame["x"].min())
    x_max = float(roi_frame["x"].max())
    y_min = float(roi_frame["y"].min())
    y_max = float(roi_frame["y"].max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    width, height, margin, plot_top = 560, 620, 30, 90
    plot_width = width - 2 * margin
    plot_height = height - plot_top - margin
    plot_side = min(plot_width, plot_height)
    x_offset = margin + (plot_width - plot_side) / 2.0
    y_offset = plot_top + (plot_height - plot_side) / 2.0

    community_mask = roi_frame["community_id"].astype(int) == int(community_id)
    ordered = pd.concat([roi_frame.loc[~community_mask], roi_frame.loc[community_mask]], ignore_index=True)
    elements = _overlay_base_elements(community_id, patient_id, domain_label, fov_id, community_fraction, community_cells, roi_total_cells, x_offset, y_offset, plot_side)
    highlight_color = _community_color(community_id)
    radius = 1.6 if roi_total_cells <= 1200 else 1.2
    for _, row in ordered.iterrows():
        scaled_x = x_offset + (float(row["x"]) - x_min) / x_span * plot_side
        scaled_y = y_offset + (float(row["y"]) - y_min) / y_span * plot_side
        is_highlight = int(row["community_id"]) == int(community_id)
        fill = highlight_color if is_highlight else OVERLAY_BACKGROUND
        opacity = "0.92" if is_highlight else "0.35"
        elements.append(
            f'<circle cx="{scaled_x:.2f}" cy="{scaled_y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="none"/>'
        )
    _write_svg(path, width=width, height=height, elements=elements)


def _overlay_base_elements(
    community_id: int,
    patient_id: str,
    domain_label: str,
    fov_id: str,
    community_fraction: float,
    community_cells: int,
    roi_total_cells: int,
    x_offset: float,
    y_offset: float,
    plot_side: float,
) -> list[str]:
    return [
        _svg_text(x=24, y=30, text=f"Task A Descriptive Atlas: Community {community_id} representative overlay", font_size=18),
        _svg_text(x=24, y=52, text=f"patient={patient_id} domain={domain_label} roi={fov_id} community_fraction={community_fraction:.3f}", font_size=12, fill="#374151"),
        _svg_text(x=24, y=70, text=f"community_cells={community_cells} roi_total_cells={roi_total_cells}", font_size=12, fill="#374151"),
        f'<rect x="{x_offset:.2f}" y="{y_offset:.2f}" width="{plot_side:.2f}" height="{plot_side:.2f}" '
        'fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>',
    ]


__all__ = [
    "write_heatmap_svg",
    "write_horizontal_bar_svg",
    "write_overlay_svg",
]
