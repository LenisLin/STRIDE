"""Task A descriptive atlas workflow.

This module consumes a frozen Step 1 prepare manifest and writes descriptive
tables, figures, and a provenance manifest for the Task A atlas layer. It is
strictly descriptive-only and does not run Block 0 or any downstream
inferential workflow.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError

from ..config import load_task_a_config_bundle
from ..contracts import (
    CONTRACT_PASSED_STATE,
    SCAFFOLD_ACTIVE_STATE,
    validate_task_a_artifact_state,
)
from .stride_adapter import load_task_a_dataset_handle

ATLAS_MANIFEST_FILENAME = "task_a_descriptive_atlas_manifest.json"
ATLAS_OUTPUT_INDEX_FILENAME = "task_a_descriptive_atlas_output_index.csv"
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"
OVERLAY_DIRNAME = "representative_spatial_overlays"
DEFAULT_MAX_OVERLAY_COMMUNITIES = 8

REQUIRED_PREPARE_MANIFEST_FIELDS: tuple[str, ...] = (
    "artifact_state",
    "block0_gate_status",
    "config_path",
    "core_fit_dry_run",
    "mapping_manifest",
    "mass_mode",
    "run_scope",
    "scientific_interpretation_allowed",
    "stage0_h5ad",
    "task_name",
)
REQUIRED_MAPPING_FIELDS: tuple[str, ...] = (
    "field_mapping",
    "patient_ids",
    "real_data_crosswalk",
)
EXPECTED_ATLAS_ARTIFACT_STATES: tuple[str, ...] = (
    SCAFFOLD_ACTIVE_STATE,
    CONTRACT_PASSED_STATE,
)
ATLAS_ROLE = "descriptive_only"

_CONTINUOUS_START = "#f7fbff"
_CONTINUOUS_END = "#08306b"
_BAR_FILL = "#2171b5"
_OVERLAY_BACKGROUND = "#d0d5dd"
_COMMUNITY_PALETTE: tuple[str, ...] = (
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


def _require_mapping(payload: dict[str, Any], *, required_fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ContractError(f"{label} is missing required fields: {missing}")


def _load_json_payload(path: str | Path, *, required_fields: tuple[str, ...], label: str) -> tuple[Path, dict[str, Any]]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {resolved_path}")
    _require_mapping(payload, required_fields=required_fields, label=label)
    return resolved_path, payload


def _validate_prepare_manifest(prepare_manifest: dict[str, Any]) -> None:
    validate_task_a_artifact_state(prepare_manifest["artifact_state"])
    if prepare_manifest["artifact_state"] not in EXPECTED_ATLAS_ARTIFACT_STATES:
        raise ContractError(
            "Task A descriptive atlas expects a Step 1 artifact_state of "
            f"{EXPECTED_ATLAS_ARTIFACT_STATES}; got {prepare_manifest['artifact_state']!r}"
        )
    if bool(prepare_manifest["scientific_interpretation_allowed"]):
        raise ContractError(
            "Task A descriptive atlas requires scientific_interpretation_allowed=false "
            "on the input prepare manifest"
        )
    if str(prepare_manifest["mass_mode"]) != "uniform":
        raise ContractError(
            "Task A descriptive atlas requires prepare-manifest mass_mode='uniform'"
        )


def _validate_mapping_alignment(
    *,
    prepare_manifest: dict[str, Any],
    mapping_payload: dict[str, Any],
    handle: Any,
) -> dict[str, str]:
    field_mapping = mapping_payload["field_mapping"]
    required_field_mapping_keys = (
        "patient_id_key",
        "fov_key",
        "domain_key",
        "cell_subtype_key",
        "state_id_key",
    )
    _require_mapping(
        field_mapping,
        required_fields=required_field_mapping_keys,
        label="Task A mapping field_mapping",
    )

    expected_pairs = {
        "patient_id_key": "patient_id",
        "fov_key": handle.fov_key,
        "domain_key": handle.domain_key,
        "cell_subtype_key": handle.cell_subtype_key,
        "state_id_key": handle.state_id_key,
    }
    for key, expected in expected_pairs.items():
        observed = str(field_mapping[key])
        if observed != str(expected):
            raise ContractError(
                "Task A descriptive atlas detected a mismatch between the Step 1 mapping "
                f"manifest and the loaded Stage 0 artifact for {key}: "
                f"{observed!r} != {expected!r}"
            )

    if "patient_subset" in prepare_manifest:
        expected_subset = tuple(str(patient_id) for patient_id in prepare_manifest["patient_subset"])
        observed_patient_ids = tuple(str(patient_id) for patient_id in mapping_payload["patient_ids"])
        if tuple(observed_patient_ids) != expected_subset:
            raise ContractError(
                "Task A descriptive atlas requires the prepare manifest patient_subset to "
                "match the Step 1 mapping manifest patient_ids exactly"
            )

    return {key: str(field_mapping[key]) for key in required_field_mapping_keys}


def _build_cell_frame(
    *,
    handle: Any,
    field_mapping: dict[str, str],
    patient_subset: tuple[str, ...] | None,
) -> pd.DataFrame:
    adata = handle.adata
    if "spatial" not in adata.obsm:
        raise ContractError("Task A descriptive atlas requires adata.obsm['spatial'] for overlays")
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ContractError("Task A descriptive atlas requires spatial coordinates with shape [n_cells, >=2]")

    obs_columns = [
        field_mapping["patient_id_key"],
        field_mapping["domain_key"],
        field_mapping["fov_key"],
        field_mapping["cell_subtype_key"],
        field_mapping["state_id_key"],
    ]
    missing_columns = [column for column in obs_columns if column not in adata.obs.columns]
    if missing_columns:
        raise ContractError(
            f"Task A descriptive atlas is missing required Stage 0 obs columns: {missing_columns}"
        )

    patient_ids = adata.obs[field_mapping["patient_id_key"]].astype(str)
    mask = np.ones(adata.n_obs, dtype=bool)
    if patient_subset is not None:
        mask = patient_ids.isin(set(patient_subset)).to_numpy(dtype=bool)
        if not bool(mask.any()):
            raise ContractError(
                "Task A descriptive atlas patient_subset did not match any cells in the Stage 0 h5ad"
            )

    obs = adata.obs.loc[mask, obs_columns].copy()
    frame = pd.DataFrame(
        {
            "patient_id": obs[field_mapping["patient_id_key"]].astype(str).to_numpy(),
            "domain_label": obs[field_mapping["domain_key"]].astype(str).to_numpy(),
            "fov_id": obs[field_mapping["fov_key"]].astype(str).to_numpy(),
            "cell_subtype_label": obs[field_mapping["cell_subtype_key"]].astype(str).to_numpy(),
            "community_id": obs[field_mapping["state_id_key"]].astype(int).to_numpy(),
            "x": spatial[mask, 0],
            "y": spatial[mask, 1],
        }
    )
    if frame.empty:
        raise ContractError("Task A descriptive atlas cannot run on an empty selected cohort")
    return frame


def _resolve_domain_order(frame: pd.DataFrame, config_bundle: Any) -> list[str]:
    configured = [str(domain) for domain in config_bundle.ordered_proxy.domains]
    observed = sorted(frame["domain_label"].astype(str).unique().tolist())
    extras = [domain for domain in observed if domain not in configured]
    ordered = [domain for domain in configured if domain in observed]
    ordered.extend(extras)
    return ordered


def _resolve_community_order(frame: pd.DataFrame, mapping_payload: dict[str, Any]) -> list[int]:
    observed = {int(community_id) for community_id in frame["community_id"].astype(int).unique().tolist()}
    configured = [
        int(community_id)
        for community_id in mapping_payload["field_mapping"].get("state_ids", [])
        if int(community_id) in observed
    ]
    if configured:
        return configured
    return sorted(observed)


def _prepare_matrix_frame(
    frame: pd.DataFrame,
    *,
    row_name: str,
    column_name: str,
    row_order: list[Any],
    column_order: list[Any],
) -> pd.DataFrame:
    matrix = pd.crosstab(frame[row_name], frame[column_name], dropna=False)
    row_index = pd.Index(row_order, name=row_name)
    column_index = pd.Index(column_order, name=column_name)
    matrix = matrix.reindex(index=row_index, columns=column_index, fill_value=0)
    matrix.index.name = row_name
    matrix.columns.name = column_name
    return matrix


def _build_domain_distribution_table(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    domain_order: list[str],
) -> pd.DataFrame:
    grouped = (
        frame.groupby(["community_id", "domain_label"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    index = pd.MultiIndex.from_product(
        [community_order, domain_order],
        names=["community_id", "domain_label"],
    )
    table = grouped.set_index(["community_id", "domain_label"]).reindex(index, fill_value=0).reset_index()
    community_totals = table.groupby("community_id", observed=False)["n_cells"].transform("sum")
    domain_totals = table.groupby("domain_label", observed=False)["n_cells"].transform("sum")
    table["community_total_cells"] = community_totals.astype(int)
    table["domain_total_cells"] = domain_totals.astype(int)
    table["fraction_within_community"] = np.where(
        community_totals > 0,
        table["n_cells"] / community_totals,
        0.0,
    )
    table["fraction_within_domain"] = np.where(
        domain_totals > 0,
        table["n_cells"] / domain_totals,
        0.0,
    )
    return table


def _build_domain_roi_prevalence_table(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    domain_order: list[str],
) -> pd.DataFrame:
    roi_frame = frame[["patient_id", "domain_label", "fov_id"]].drop_duplicates()
    total_rois = (
        roi_frame.groupby("domain_label", observed=False)
        .size()
        .rename("total_rois")
        .to_dict()
    )
    positive_rois = (
        frame.groupby(["community_id", "domain_label", "fov_id"], observed=False)
        .size()
        .reset_index(name="community_cells")
        .groupby(["community_id", "domain_label"], observed=False)["fov_id"]
        .nunique()
        .rename("positive_rois")
        .reset_index()
    )
    index = pd.MultiIndex.from_product(
        [community_order, domain_order],
        names=["community_id", "domain_label"],
    )
    table = (
        positive_rois.set_index(["community_id", "domain_label"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    table["total_rois"] = table["domain_label"].map(total_rois).fillna(0).astype(int)
    table["roi_prevalence"] = np.where(
        table["total_rois"] > 0,
        table["positive_rois"] / table["total_rois"],
        0.0,
    )
    return table


def _build_patient_occurrence_tables(
    frame: pd.DataFrame,
    *,
    community_order: list[int],
    patient_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_presence = (
        frame.groupby(["community_id", "patient_id"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    matrix_index = pd.Index(community_order, name="community_id")
    matrix_columns = pd.Index(patient_order, name="patient_id")
    matrix = pd.crosstab(patient_presence["community_id"], patient_presence["patient_id"], dropna=False)
    matrix = matrix.reindex(index=matrix_index, columns=matrix_columns, fill_value=0)
    matrix = (matrix > 0).astype(int)

    roi_presence = (
        frame.groupby(["community_id", "patient_id", "fov_id"], observed=False)
        .size()
        .rename("community_cells")
        .reset_index()
    )
    total_rois = int(frame[["patient_id", "fov_id"]].drop_duplicates().shape[0])
    patient_summary = (
        patient_presence.groupby("community_id", observed=False)
        .agg(
            n_patients_present=("patient_id", "nunique"),
            total_cells=("n_cells", "sum"),
            median_cells_per_positive_patient=("n_cells", "median"),
            max_cells_in_single_patient=("n_cells", "max"),
        )
        .reindex(matrix_index, fill_value=0)
        .reset_index()
    )
    positive_rois = (
        roi_presence.groupby("community_id", observed=False)["fov_id"]
        .nunique()
        .reindex(matrix_index, fill_value=0)
        .to_numpy(dtype=int)
    )
    patient_summary["n_patients_total"] = int(len(patient_order))
    patient_summary["patient_prevalence"] = np.where(
        patient_summary["n_patients_total"] > 0,
        patient_summary["n_patients_present"] / patient_summary["n_patients_total"],
        0.0,
    )
    patient_summary["n_positive_rois"] = positive_rois
    patient_summary["n_total_rois"] = total_rois
    patient_summary["roi_prevalence"] = np.where(
        patient_summary["n_total_rois"] > 0,
        patient_summary["n_positive_rois"] / patient_summary["n_total_rois"],
        0.0,
    )
    return patient_summary, matrix


def _select_representative_overlays(
    frame: pd.DataFrame,
    *,
    max_overlay_communities: int,
) -> pd.DataFrame:
    community_totals = (
        frame.groupby("community_id", observed=False)
        .size()
        .rename("community_total_cells")
        .sort_values(ascending=False)
    )
    selected_communities = community_totals.head(max_overlay_communities).index.tolist()
    roi_counts = (
        frame.groupby(["community_id", "patient_id", "domain_label", "fov_id"], observed=False)
        .size()
        .rename("community_cells")
        .reset_index()
    )
    roi_totals = (
        frame.groupby(["patient_id", "domain_label", "fov_id"], observed=False)
        .size()
        .rename("roi_total_cells")
        .reset_index()
    )
    selection = roi_counts.merge(
        roi_totals,
        on=["patient_id", "domain_label", "fov_id"],
        how="left",
        validate="many_to_one",
    )
    selection["community_fraction_in_roi"] = np.where(
        selection["roi_total_cells"] > 0,
        selection["community_cells"] / selection["roi_total_cells"],
        0.0,
    )
    selection["community_total_cells"] = selection["community_id"].map(community_totals.to_dict()).astype(int)
    records: list[pd.Series] = []
    for community_id in selected_communities:
        community_selection = selection.loc[selection["community_id"] == community_id].copy()
        community_selection = community_selection.sort_values(
            by=[
                "community_fraction_in_roi",
                "community_cells",
                "patient_id",
                "domain_label",
                "fov_id",
            ],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        if not community_selection.empty:
            records.append(community_selection.iloc[0])

    if not records:
        return pd.DataFrame(
            columns=[
                "community_id",
                "community_total_cells",
                "patient_id",
                "domain_label",
                "fov_id",
                "community_cells",
                "roi_total_cells",
                "community_fraction_in_roi",
            ]
        )

    return pd.DataFrame(records).reset_index(drop=True)


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
    return _COMMUNITY_PALETTE[int(community_id) % len(_COMMUNITY_PALETTE)]


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


def _write_heatmap_svg(
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

    elements: list[str] = [
        _svg_text(x=24, y=30, text=title, font_size=18),
        _svg_text(x=24, y=54, text=value_label, font_size=12, fill="#374151"),
    ]

    legend_x = left_margin + plot_width + 24
    legend_y = top_margin
    legend_height = min(140, plot_height)
    for step in range(20):
        weight = step / 19 if step else 0.0
        color = _blend_hex(_CONTINUOUS_START, _CONTINUOUS_END, weight)
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
    elements.append(
        _svg_text(
            x=legend_x + 28,
            y=legend_y + legend_height + 2,
            text="0",
            font_size=11,
        )
    )
    elements.append(
        _svg_text(
            x=legend_x + 28,
            y=legend_y + 24,
            text=f"{value_max:.2f}",
            font_size=11,
            fill="#374151",
        )
    )

    for row_idx, row_label in enumerate(row_labels):
        y = top_margin + row_idx * cell_height
        elements.append(
            _svg_text(
                x=left_margin - 8,
                y=y + cell_height * 0.7,
                text=row_label,
                font_size=11,
                anchor="end",
            )
        )
        for col_idx, _column_label in enumerate(column_labels):
            x = left_margin + col_idx * cell_width
            value = float(values[row_idx, col_idx]) if values.size else 0.0
            fill = _blend_hex(_CONTINUOUS_START, _CONTINUOUS_END, value / value_max)
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" '
                f'fill="{fill}" stroke="#e5e7eb" stroke-width="0.5"/>'
            )
            if annotate:
                label = f"{value:.2f}" if value_label.endswith("fraction") else f"{value:.0f}"
                elements.append(
                    _svg_text(
                        x=x + cell_width / 2.0,
                        y=y + cell_height * 0.68,
                        text=label,
                        font_size=9,
                        anchor="middle",
                        fill="#111827" if value < value_max * 0.55 else "#ffffff",
                    )
                )

    for col_idx, column_label in enumerate(column_labels):
        x = left_margin + col_idx * cell_width + cell_width / 2.0
        y = top_margin - 10
        elements.append(
            _svg_text(
                x=x,
                y=y,
                text=column_label,
                font_size=10,
                anchor="start",
                extra=f'transform="rotate(-55 {x:.2f} {y:.2f})"',
            )
        )

    _write_svg(path, width=width, height=height, elements=elements)


def _write_horizontal_bar_svg(
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
    elements: list[str] = [
        _svg_text(x=24, y=30, text=title, font_size=18),
        _svg_text(
            x=24,
            y=52,
            text="Sorted by patient prevalence, then total cells",
            font_size=12,
            fill="#374151",
        ),
    ]
    elements.append(
        f'<line x1="{left_margin}" y1="{top_margin - 8}" x2="{left_margin}" y2="{height - bottom_margin}" '
        'stroke="#9ca3af" stroke-width="1"/>'
    )
    for tick in range(5):
        fraction = tick / 4 if tick else 0.0
        x = left_margin + fraction * plot_width
        elements.append(
            f'<line x1="{x:.2f}" y1="{top_margin - 6}" x2="{x:.2f}" y2="{height - bottom_margin}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            _svg_text(
                x=x,
                y=height - 12,
                text=f"{fraction:.2f}",
                font_size=10,
                anchor="middle",
                fill="#374151",
            )
        )
    for row_idx, row in ordered.iterrows():
        y = top_margin + row_idx * row_height
        label = f"C{int(row['community_id'])}"
        bar_width = float(row["patient_prevalence"]) * plot_width
        elements.append(
            _svg_text(
                x=left_margin - 10,
                y=y + row_height * 0.68,
                text=label,
                font_size=11,
                anchor="end",
            )
        )
        elements.append(
            f'<rect x="{left_margin:.2f}" y="{y + 4:.2f}" width="{bar_width:.2f}" height="{row_height - 8:.2f}" '
            f'fill="{_BAR_FILL}" opacity="0.92"/>'
        )
        elements.append(
            _svg_text(
                x=left_margin + bar_width + 8,
                y=y + row_height * 0.68,
                text=(
                    f"{float(row['patient_prevalence']):.2f} "
                    f"({int(row['n_patients_present'])}/{int(row['n_patients_total'])})"
                ),
                font_size=10,
                fill="#374151",
            )
        )
    _write_svg(path, width=width, height=height, elements=elements)


def _write_overlay_svg(
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
        raise ContractError("Representative overlay generation received an empty ROI frame")
    x_min = float(roi_frame["x"].min())
    x_max = float(roi_frame["x"].max())
    y_min = float(roi_frame["y"].min())
    y_max = float(roi_frame["y"].max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    width = 560
    height = 620
    margin = 30
    plot_top = 90
    plot_width = width - 2 * margin
    plot_height = height - plot_top - margin
    plot_side = min(plot_width, plot_height)
    x_offset = margin + (plot_width - plot_side) / 2.0
    y_offset = plot_top + (plot_height - plot_side) / 2.0

    community_mask = roi_frame["community_id"].astype(int) == int(community_id)
    ordered = pd.concat(
        [
            roi_frame.loc[~community_mask],
            roi_frame.loc[community_mask],
        ],
        ignore_index=True,
    )
    highlight_color = _community_color(community_id)
    elements: list[str] = [
        _svg_text(
            x=24,
            y=30,
            text=f"Task A Descriptive Atlas: Community {community_id} representative overlay",
            font_size=18,
        ),
        _svg_text(
            x=24,
            y=52,
            text=(
                f"patient={patient_id} domain={domain_label} roi={fov_id} "
                f"community_fraction={community_fraction:.3f}"
            ),
            font_size=12,
            fill="#374151",
        ),
        _svg_text(
            x=24,
            y=70,
            text=f"community_cells={community_cells} roi_total_cells={roi_total_cells}",
            font_size=12,
            fill="#374151",
        ),
        f'<rect x="{x_offset:.2f}" y="{y_offset:.2f}" width="{plot_side:.2f}" height="{plot_side:.2f}" '
        'fill="#ffffff" stroke="#9ca3af" stroke-width="1"/>',
    ]

    radius = 1.6 if roi_total_cells <= 1200 else 1.2
    for _, row in ordered.iterrows():
        scaled_x = x_offset + (float(row["x"]) - x_min) / x_span * plot_side
        scaled_y = y_offset + (float(row["y"]) - y_min) / y_span * plot_side
        is_highlight = int(row["community_id"]) == int(community_id)
        fill = highlight_color if is_highlight else _OVERLAY_BACKGROUND
        opacity = "0.92" if is_highlight else "0.35"
        elements.append(
            f'<circle cx="{scaled_x:.2f}" cy="{scaled_y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="none"/>'
        )

    _write_svg(path, width=width, height=height, elements=elements)


def _relative_to_output(path: Path, output_root: Path) -> str:
    return path.resolve().relative_to(output_root.resolve()).as_posix()


def write_task_a_descriptive_atlas(
    *,
    prepare_manifest_path: str | Path,
    output_dir: str | Path,
    max_overlay_communities: int = DEFAULT_MAX_OVERLAY_COMMUNITIES,
) -> dict[str, Any]:
    if max_overlay_communities <= 0:
        raise ContractError("Task A descriptive atlas requires --max-overlay-communities >= 1")

    prepare_path, prepare_manifest = _load_json_payload(
        prepare_manifest_path,
        required_fields=REQUIRED_PREPARE_MANIFEST_FIELDS,
        label="Task A prepare manifest",
    )
    _validate_prepare_manifest(prepare_manifest)

    mapping_path, mapping_payload = _load_json_payload(
        prepare_manifest["mapping_manifest"],
        required_fields=REQUIRED_MAPPING_FIELDS,
        label="Task A mapping manifest",
    )
    config_bundle = load_task_a_config_bundle(prepare_manifest["config_path"])
    handle = load_task_a_dataset_handle(prepare_manifest["stage0_h5ad"])
    field_mapping = _validate_mapping_alignment(
        prepare_manifest=prepare_manifest,
        mapping_payload=mapping_payload,
        handle=handle,
    )

    patient_subset: tuple[str, ...] | None = None
    if "patient_subset" in prepare_manifest:
        patient_subset = tuple(str(patient_id) for patient_id in prepare_manifest["patient_subset"])
    frame = _build_cell_frame(
        handle=handle,
        field_mapping=field_mapping,
        patient_subset=patient_subset,
    )

    domain_order = _resolve_domain_order(frame, config_bundle)
    community_order = _resolve_community_order(frame, mapping_payload)
    patient_order = [
        str(patient_id)
        for patient_id in mapping_payload["patient_ids"]
        if str(patient_id) in set(frame["patient_id"].astype(str))
    ]
    cell_subtype_order = sorted(frame["cell_subtype_label"].astype(str).unique().tolist())
    observed_community_ids = sorted(frame["community_id"].astype(int).unique().tolist())
    if not patient_order:
        patient_order = sorted(frame["patient_id"].astype(str).unique().tolist())

    output_root = Path(output_dir).expanduser().resolve()
    tables_dir = output_root / TABLES_DIRNAME
    figures_dir = output_root / FIGURES_DIRNAME
    overlays_dir = figures_dir / OVERLAY_DIRNAME
    overlays_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    community_cell_subtype_counts = _prepare_matrix_frame(
        frame,
        row_name="community_id",
        column_name="cell_subtype_label",
        row_order=community_order,
        column_order=cell_subtype_order,
    )
    community_cell_subtype_row_fractions = (
        community_cell_subtype_counts.div(
            community_cell_subtype_counts.sum(axis=1).replace(0, np.nan),
            axis=0,
        ).fillna(0.0)
    )
    community_domain_distribution = _build_domain_distribution_table(
        frame,
        community_order=community_order,
        domain_order=domain_order,
    )
    community_domain_roi_prevalence = _build_domain_roi_prevalence_table(
        frame,
        community_order=community_order,
        domain_order=domain_order,
    )
    community_patient_occurrence_summary, community_patient_occurrence_matrix = _build_patient_occurrence_tables(
        frame,
        community_order=community_order,
        patient_order=patient_order,
    )
    representative_overlay_selection = _select_representative_overlays(
        frame,
        max_overlay_communities=max_overlay_communities,
    )

    counts_path = tables_dir / "community_cell_subtype_counts.csv"
    fractions_path = tables_dir / "community_cell_subtype_row_fractions.csv"
    domain_distribution_path = tables_dir / "community_domain_distribution.csv"
    domain_roi_prevalence_path = tables_dir / "community_domain_roi_prevalence.csv"
    patient_summary_path = tables_dir / "community_patient_occurrence_summary.csv"
    patient_matrix_path = tables_dir / "community_patient_occurrence_matrix.csv"
    overlay_selection_path = tables_dir / "representative_overlay_selection.csv"

    community_cell_subtype_counts.to_csv(counts_path)
    community_cell_subtype_row_fractions.to_csv(fractions_path)
    community_domain_distribution.to_csv(domain_distribution_path, index=False)
    community_domain_roi_prevalence.to_csv(domain_roi_prevalence_path, index=False)
    community_patient_occurrence_summary.to_csv(patient_summary_path, index=False)
    community_patient_occurrence_matrix.to_csv(patient_matrix_path)

    heatmap_path = figures_dir / "community_by_cell_subtype_heatmap.svg"
    domain_heatmap_path = figures_dir / "community_domain_abundance_heatmap.svg"
    domain_prevalence_heatmap_path = figures_dir / "community_domain_roi_prevalence_heatmap.svg"
    patient_prevalence_path = figures_dir / "patient_level_community_prevalence.svg"

    _write_heatmap_svg(
        community_cell_subtype_row_fractions,
        heatmap_path,
        title="Task A Descriptive Atlas: community x cell subtype",
        value_label="row fraction within each community",
        annotate=False,
    )
    _write_heatmap_svg(
        community_domain_distribution.pivot(
            index="community_id",
            columns="domain_label",
            values="fraction_within_community",
        ).reindex(index=community_order, columns=domain_order, fill_value=0.0),
        domain_heatmap_path,
        title="Task A Descriptive Atlas: community domain distribution",
        value_label="fraction within community",
        annotate=True,
    )
    _write_heatmap_svg(
        community_domain_roi_prevalence.pivot(
            index="community_id",
            columns="domain_label",
            values="roi_prevalence",
        ).reindex(index=community_order, columns=domain_order, fill_value=0.0),
        domain_prevalence_heatmap_path,
        title="Task A Descriptive Atlas: ROI prevalence by domain",
        value_label="ROI prevalence fraction",
        annotate=True,
    )
    _write_horizontal_bar_svg(
        community_patient_occurrence_summary,
        patient_prevalence_path,
        title="Task A Descriptive Atlas: patient-level community prevalence",
    )

    overlay_rows: list[dict[str, Any]] = []
    for _, selection in representative_overlay_selection.iterrows():
        community_id = int(selection["community_id"])
        patient_id = str(selection["patient_id"])
        domain_label = str(selection["domain_label"])
        fov_id = str(selection["fov_id"])
        overlay_path = overlays_dir / f"community_{community_id:02d}_overlay.svg"
        roi_frame = frame.loc[
            (frame["patient_id"] == patient_id)
            & (frame["domain_label"] == domain_label)
            & (frame["fov_id"] == fov_id)
        ].copy()
        _write_overlay_svg(
            roi_frame,
            overlay_path,
            community_id=community_id,
            patient_id=patient_id,
            domain_label=domain_label,
            fov_id=fov_id,
            community_fraction=float(selection["community_fraction_in_roi"]),
            community_cells=int(selection["community_cells"]),
            roi_total_cells=int(selection["roi_total_cells"]),
        )
        overlay_record = selection.to_dict()
        overlay_record["overlay_path"] = _relative_to_output(overlay_path, output_root)
        overlay_rows.append(overlay_record)
    representative_overlay_selection = pd.DataFrame(overlay_rows)
    representative_overlay_selection.to_csv(overlay_selection_path, index=False)

    output_rows = [
        {
            "relative_path": ATLAS_MANIFEST_FILENAME,
            "artifact_kind": "manifest",
            "category": "provenance",
            "format": "json",
            "description": "Task A descriptive atlas provenance manifest",
        },
        {
            "relative_path": ATLAS_OUTPUT_INDEX_FILENAME,
            "artifact_kind": "index",
            "category": "provenance",
            "format": "csv",
            "description": "Machine-readable index of atlas outputs",
        },
        {
            "relative_path": _relative_to_output(counts_path, output_root),
            "artifact_kind": "table",
            "category": "community_cell_subtype",
            "format": "csv",
            "description": "Community by cell-subtype counts",
        },
        {
            "relative_path": _relative_to_output(fractions_path, output_root),
            "artifact_kind": "table",
            "category": "community_cell_subtype",
            "format": "csv",
            "description": "Community by cell-subtype row fractions",
        },
        {
            "relative_path": _relative_to_output(domain_distribution_path, output_root),
            "artifact_kind": "table",
            "category": "community_domain_distribution",
            "format": "csv",
            "description": "Community abundance summaries across TC/IM/PT",
        },
        {
            "relative_path": _relative_to_output(domain_roi_prevalence_path, output_root),
            "artifact_kind": "table",
            "category": "community_domain_distribution",
            "format": "csv",
            "description": "Community ROI prevalence across TC/IM/PT",
        },
        {
            "relative_path": _relative_to_output(patient_summary_path, output_root),
            "artifact_kind": "table",
            "category": "patient_occurrence",
            "format": "csv",
            "description": "Patient-level community occurrence summary",
        },
        {
            "relative_path": _relative_to_output(patient_matrix_path, output_root),
            "artifact_kind": "table",
            "category": "patient_occurrence",
            "format": "csv",
            "description": "Binary community x patient occurrence matrix",
        },
        {
            "relative_path": _relative_to_output(overlay_selection_path, output_root),
            "artifact_kind": "table",
            "category": "representative_spatial_overlays",
            "format": "csv",
            "description": "Deterministic representative ROI selections for overlay figures",
        },
        {
            "relative_path": _relative_to_output(heatmap_path, output_root),
            "artifact_kind": "figure",
            "category": "community_cell_subtype",
            "format": "svg",
            "description": "Community x cell-subtype heatmap",
        },
        {
            "relative_path": _relative_to_output(domain_heatmap_path, output_root),
            "artifact_kind": "figure",
            "category": "community_domain_distribution",
            "format": "svg",
            "description": "Community abundance heatmap across tissue domains",
        },
        {
            "relative_path": _relative_to_output(domain_prevalence_heatmap_path, output_root),
            "artifact_kind": "figure",
            "category": "community_domain_distribution",
            "format": "svg",
            "description": "Community ROI prevalence heatmap across tissue domains",
        },
        {
            "relative_path": _relative_to_output(patient_prevalence_path, output_root),
            "artifact_kind": "figure",
            "category": "patient_occurrence",
            "format": "svg",
            "description": "Patient-level community prevalence summary",
        },
    ]
    for _, row in representative_overlay_selection.iterrows():
        output_rows.append(
            {
                "relative_path": str(row["overlay_path"]),
                "artifact_kind": "figure",
                "category": "representative_spatial_overlays",
                "format": "svg",
                "description": f"Representative spatial overlay for community {int(row['community_id'])}",
            }
        )
    output_index = pd.DataFrame(output_rows)
    output_index_path = output_root / ATLAS_OUTPUT_INDEX_FILENAME
    output_index.to_csv(output_index_path, index=False)

    manifest: dict[str, Any] = {
        "task_name": str(prepare_manifest["task_name"]),
        "workflow_name": "write_task_a_descriptive_atlas",
        "atlas_role": ATLAS_ROLE,
        "claim_scope": ATLAS_ROLE,
        "scientific_interpretation_allowed": False,
        "artifact_state": str(prepare_manifest["artifact_state"]),
        "block0_gate_status": str(prepare_manifest["block0_gate_status"]),
        "implementation_tier": str(prepare_manifest.get("implementation_tier", "descriptive_context")),
        "evidence_lineage": str(prepare_manifest.get("evidence_lineage", "canonical_rerun")),
        "prepare_manifest_path": str(prepare_path),
        "mapping_manifest_path": str(mapping_path),
        "config_path": str(Path(prepare_manifest["config_path"]).expanduser().resolve()),
        "stage0_h5ad": str(Path(prepare_manifest["stage0_h5ad"]).expanduser().resolve()),
        "run_scope": str(prepare_manifest["run_scope"]),
        "mass_mode": str(prepare_manifest["mass_mode"]),
        "community_id_key": field_mapping["state_id_key"],
        "cell_subtype_key": field_mapping["cell_subtype_key"],
        "domain_key": field_mapping["domain_key"],
        "fov_key": field_mapping["fov_key"],
        "patient_id_key": field_mapping["patient_id_key"],
        "spatial_key": "spatial",
        "configured_state_ids": [
            int(community_id)
            for community_id in mapping_payload["field_mapping"].get("state_ids", [])
        ],
        "observed_community_ids": observed_community_ids,
        "domain_labels": domain_order,
        "patient_ids": patient_order,
        "max_overlay_communities": int(max_overlay_communities),
        "overlay_selection_rule": (
            "top communities by total cell count; within each community choose the ROI with the "
            "highest within-ROI community fraction, then highest community cell count, then "
            "patient_id/domain/roi_id ascending"
        ),
        "n_cells": int(frame.shape[0]),
        "n_patients": int(len(patient_order)),
        "n_rois": int(frame[["patient_id", "fov_id"]].drop_duplicates().shape[0]),
        "n_observed_communities": int(len(observed_community_ids)),
        "n_cell_subtypes": int(len(cell_subtype_order)),
        "output_index": str(output_index_path),
    }
    if patient_subset is not None:
        manifest["patient_subset"] = list(patient_subset)
    if "demo_subset_name" in prepare_manifest:
        manifest["demo_subset_name"] = str(prepare_manifest["demo_subset_name"])
    if "demo_subset_rationale" in prepare_manifest:
        manifest["demo_subset_rationale"] = str(prepare_manifest["demo_subset_rationale"])

    manifest_path = output_root / ATLAS_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Task A descriptive atlas from a frozen Step 1 prepare manifest.",
    )
    parser.add_argument(
        "--prepare-manifest",
        required=True,
        help="Path to a task_a_prepare_manifest.json produced by Task A Step 1 prepare",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for atlas artifacts")
    parser.add_argument(
        "--max-overlay-communities",
        type=int,
        default=DEFAULT_MAX_OVERLAY_COMMUNITIES,
        help="Maximum number of top communities to render as representative overlays",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        manifest = write_task_a_descriptive_atlas(
            prepare_manifest_path=args.prepare_manifest,
            output_dir=args.output_dir,
            max_overlay_communities=args.max_overlay_communities,
        )
        output_index = manifest.get("output_index", args.output_dir)
        print(f"Wrote descriptive atlas manifest with {len(manifest)} keys and index {output_index}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = [
    "ATLAS_MANIFEST_FILENAME",
    "ATLAS_OUTPUT_INDEX_FILENAME",
    "DEFAULT_MAX_OVERLAY_COMMUNITIES",
    "write_task_a_descriptive_atlas",
]
