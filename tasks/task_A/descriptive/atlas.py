"""Task A descriptive atlas orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_task_a_config_bundle
from .contracts import (
    ATLAS_MANIFEST_FILENAME,
    ATLAS_OUTPUT_INDEX_FILENAME,
    ATLAS_ROLE,
    DEFAULT_MAX_OVERLAY_COMMUNITIES,
    FIGURES_DIRNAME,
    OVERLAY_DIRNAME,
    TABLES_DIRNAME,
    DescriptiveAtlasContractError,
    copy_stage0_field_mapping,
)
from .figures import write_heatmap_svg, write_horizontal_bar_svg, write_overlay_svg
from .tables import (
    build_cell_frame,
    build_domain_distribution_table,
    build_domain_roi_prevalence_table,
    build_patient_occurrence_tables,
    prepare_matrix_frame,
    resolve_community_order,
    resolve_domain_order,
    select_representative_overlays,
)

try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AtlasInputs:
    config_path: Path
    stage0_h5ad: Path
    config_bundle: Any
    adata: Any
    field_mapping: dict[str, str]
    patient_ids: tuple[str, ...] | None
    configured_community_ids: list[int]


@dataclass(frozen=True)
class AtlasTables:
    frame: pd.DataFrame
    community_order: list[int]
    domain_order: list[str]
    patient_order: list[str]
    cell_subtype_order: list[str]
    observed_community_ids: list[int]
    community_cell_subtype_counts: pd.DataFrame
    community_cell_subtype_row_fractions: pd.DataFrame
    community_domain_distribution: pd.DataFrame
    community_domain_roi_prevalence: pd.DataFrame
    community_patient_occurrence_summary: pd.DataFrame
    community_patient_occurrence_matrix: pd.DataFrame
    representative_overlay_selection: pd.DataFrame


@dataclass(frozen=True)
class AtlasPaths:
    output_root: Path
    tables_dir: Path
    figures_dir: Path
    overlays_dir: Path
    counts: Path
    fractions: Path
    domain_distribution: Path
    domain_roi_prevalence: Path
    patient_summary: Path
    patient_matrix: Path
    overlay_selection: Path
    heatmap: Path
    domain_heatmap: Path
    domain_prevalence_heatmap: Path
    patient_prevalence: Path
    output_index: Path
    manifest: Path


def _relative_to_output(path: Path, output_root: Path) -> str:
    return path.resolve().relative_to(output_root.resolve()).as_posix()


def _load_stage0_h5ad(stage0_h5ad: str | Path) -> tuple[Path, Any]:
    if ad is None:
        raise ModuleNotFoundError("anndata is required to load Task A Stage0 h5ad artifacts")
    resolved_path = Path(stage0_h5ad).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Task A Stage0 h5ad was not found: {resolved_path}")
    return resolved_path, ad.read_h5ad(resolved_path)


def _load_inputs(
    *,
    config_path: str | Path,
    stage0_h5ad: str | Path,
    patient_ids: tuple[str, ...] | None,
) -> AtlasInputs:
    config_bundle = load_task_a_config_bundle(config_path)
    stage0_path, adata = _load_stage0_h5ad(stage0_h5ad)
    configured_community_ids = list(range(int(config_bundle.data.k_full)))
    return AtlasInputs(
        config_path=config_bundle.config_path,
        stage0_h5ad=stage0_path,
        config_bundle=config_bundle,
        adata=adata,
        field_mapping=copy_stage0_field_mapping(),
        patient_ids=patient_ids,
        configured_community_ids=configured_community_ids,
    )


def _build_tables(inputs: AtlasInputs, *, max_overlay_communities: int) -> AtlasTables:
    frame = build_cell_frame(
        adata=inputs.adata,
        field_mapping=inputs.field_mapping,
        patient_ids=inputs.patient_ids,
    )
    domain_order = resolve_domain_order(frame, inputs.config_bundle)
    community_order = resolve_community_order(frame, inputs.configured_community_ids)
    if inputs.patient_ids is not None:
        patient_order = [
            str(patient_id)
            for patient_id in inputs.patient_ids
            if str(patient_id) in set(frame["patient_id"].astype(str))
        ]
    else:
        patient_order = sorted(frame["patient_id"].astype(str).unique().tolist())
    cell_subtype_order = sorted(frame["cell_subtype_label"].astype(str).unique().tolist())
    observed_community_ids = sorted(frame["community_id"].astype(int).unique().tolist())
    counts = prepare_matrix_frame(
        frame,
        row_name="community_id",
        column_name="cell_subtype_label",
        row_order=community_order,
        column_order=cell_subtype_order,
    )
    row_fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    patient_summary, patient_matrix = build_patient_occurrence_tables(
        frame,
        community_order=community_order,
        patient_order=patient_order,
    )
    return AtlasTables(
        frame=frame,
        community_order=community_order,
        domain_order=domain_order,
        patient_order=patient_order,
        cell_subtype_order=cell_subtype_order,
        observed_community_ids=observed_community_ids,
        community_cell_subtype_counts=counts,
        community_cell_subtype_row_fractions=row_fractions,
        community_domain_distribution=build_domain_distribution_table(
            frame,
            community_order=community_order,
            domain_order=domain_order,
        ),
        community_domain_roi_prevalence=build_domain_roi_prevalence_table(
            frame,
            community_order=community_order,
            domain_order=domain_order,
        ),
        community_patient_occurrence_summary=patient_summary,
        community_patient_occurrence_matrix=patient_matrix,
        representative_overlay_selection=select_representative_overlays(
            frame,
            max_overlay_communities=max_overlay_communities,
        ),
    )


def _resolve_paths(output_dir: str | Path) -> AtlasPaths:
    output_root = Path(output_dir).expanduser().resolve()
    tables_dir = output_root / TABLES_DIRNAME
    figures_dir = output_root / FIGURES_DIRNAME
    overlays_dir = figures_dir / OVERLAY_DIRNAME
    overlays_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return AtlasPaths(
        output_root=output_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        overlays_dir=overlays_dir,
        counts=tables_dir / "community_cell_subtype_counts.csv",
        fractions=tables_dir / "community_cell_subtype_row_fractions.csv",
        domain_distribution=tables_dir / "community_domain_distribution.csv",
        domain_roi_prevalence=tables_dir / "community_domain_roi_prevalence.csv",
        patient_summary=tables_dir / "community_patient_occurrence_summary.csv",
        patient_matrix=tables_dir / "community_patient_occurrence_matrix.csv",
        overlay_selection=tables_dir / "representative_overlay_selection.csv",
        heatmap=figures_dir / "community_by_cell_subtype_heatmap.svg",
        domain_heatmap=figures_dir / "community_domain_abundance_heatmap.svg",
        domain_prevalence_heatmap=figures_dir / "community_domain_roi_prevalence_heatmap.svg",
        patient_prevalence=figures_dir / "patient_level_community_prevalence.svg",
        output_index=output_root / ATLAS_OUTPUT_INDEX_FILENAME,
        manifest=output_root / ATLAS_MANIFEST_FILENAME,
    )


def _write_tables(tables: AtlasTables, paths: AtlasPaths) -> None:
    tables.community_cell_subtype_counts.to_csv(paths.counts)
    tables.community_cell_subtype_row_fractions.to_csv(paths.fractions)
    tables.community_domain_distribution.to_csv(paths.domain_distribution, index=False)
    tables.community_domain_roi_prevalence.to_csv(paths.domain_roi_prevalence, index=False)
    tables.community_patient_occurrence_summary.to_csv(paths.patient_summary, index=False)
    tables.community_patient_occurrence_matrix.to_csv(paths.patient_matrix)


def _write_summary_figures(tables: AtlasTables, paths: AtlasPaths) -> None:
    write_heatmap_svg(
        tables.community_cell_subtype_row_fractions,
        paths.heatmap,
        title="Task A Descriptive Atlas: community x cell subtype",
        value_label="row fraction within each community",
        annotate=False,
    )
    write_heatmap_svg(
        tables.community_domain_distribution.pivot(
            index="community_id",
            columns="domain_label",
            values="fraction_within_community",
        ).reindex(index=tables.community_order, columns=tables.domain_order, fill_value=0.0),
        paths.domain_heatmap,
        title="Task A Descriptive Atlas: community domain distribution",
        value_label="fraction within community",
        annotate=True,
    )
    write_heatmap_svg(
        tables.community_domain_roi_prevalence.pivot(
            index="community_id",
            columns="domain_label",
            values="roi_prevalence",
        ).reindex(index=tables.community_order, columns=tables.domain_order, fill_value=0.0),
        paths.domain_prevalence_heatmap,
        title="Task A Descriptive Atlas: ROI prevalence by domain",
        value_label="ROI prevalence fraction",
        annotate=True,
    )
    write_horizontal_bar_svg(
        tables.community_patient_occurrence_summary,
        paths.patient_prevalence,
        title="Task A Descriptive Atlas: patient-level community prevalence",
    )


def _write_overlay_figures(tables: AtlasTables, paths: AtlasPaths) -> pd.DataFrame:
    overlay_rows: list[dict[str, Any]] = []
    for _, selection in tables.representative_overlay_selection.iterrows():
        community_id = int(selection["community_id"])
        patient_id = str(selection["patient_id"])
        domain_label = str(selection["domain_label"])
        fov_id = str(selection["fov_id"])
        overlay_path = paths.overlays_dir / f"community_{community_id:02d}_overlay.svg"
        roi_frame = tables.frame.loc[
            (tables.frame["patient_id"] == patient_id)
            & (tables.frame["domain_label"] == domain_label)
            & (tables.frame["fov_id"] == fov_id)
        ].copy()
        write_overlay_svg(
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
        overlay_record["overlay_path"] = _relative_to_output(overlay_path, paths.output_root)
        overlay_rows.append(overlay_record)
    return pd.DataFrame(overlay_rows)


def _artifact_row(
    path: Path | str,
    paths: AtlasPaths,
    *,
    artifact_kind: str,
    category: str,
    fmt: str,
    description: str,
) -> dict[str, str]:
    relative_path = str(path) if isinstance(path, str) else _relative_to_output(path, paths.output_root)
    return {
        "relative_path": relative_path,
        "artifact_kind": artifact_kind,
        "category": category,
        "format": fmt,
        "description": description,
    }


def _build_output_index(paths: AtlasPaths, overlay_selection: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _artifact_row(
            ATLAS_MANIFEST_FILENAME,
            paths,
            artifact_kind="manifest",
            category="atlas_metadata",
            fmt="json",
            description="Task A descriptive atlas manifest",
        ),
        _artifact_row(
            ATLAS_OUTPUT_INDEX_FILENAME,
            paths,
            artifact_kind="index",
            category="atlas_metadata",
            fmt="csv",
            description="Machine-readable index of atlas outputs",
        ),
        _artifact_row(paths.counts, paths, artifact_kind="table", category="community_cell_subtype", fmt="csv", description="Community by cell-subtype counts"),
        _artifact_row(paths.fractions, paths, artifact_kind="table", category="community_cell_subtype", fmt="csv", description="Community by cell-subtype row fractions"),
        _artifact_row(paths.domain_distribution, paths, artifact_kind="table", category="community_domain_distribution", fmt="csv", description="Community abundance summaries across TC/IM/PT"),
        _artifact_row(paths.domain_roi_prevalence, paths, artifact_kind="table", category="community_domain_distribution", fmt="csv", description="Community ROI prevalence across TC/IM/PT"),
        _artifact_row(paths.patient_summary, paths, artifact_kind="table", category="patient_occurrence", fmt="csv", description="Patient-level community occurrence summary"),
        _artifact_row(paths.patient_matrix, paths, artifact_kind="table", category="patient_occurrence", fmt="csv", description="Binary community x patient occurrence matrix"),
        _artifact_row(paths.overlay_selection, paths, artifact_kind="table", category="representative_spatial_overlays", fmt="csv", description="Deterministic representative ROI selections for overlay figures"),
        _artifact_row(paths.heatmap, paths, artifact_kind="figure", category="community_cell_subtype", fmt="svg", description="Community x cell-subtype heatmap"),
        _artifact_row(paths.domain_heatmap, paths, artifact_kind="figure", category="community_domain_distribution", fmt="svg", description="Community abundance heatmap across tissue domains"),
        _artifact_row(paths.domain_prevalence_heatmap, paths, artifact_kind="figure", category="community_domain_distribution", fmt="svg", description="Community ROI prevalence heatmap across tissue domains"),
        _artifact_row(paths.patient_prevalence, paths, artifact_kind="figure", category="patient_occurrence", fmt="svg", description="Patient-level community prevalence summary"),
    ]
    for _, row in overlay_selection.iterrows():
        rows.append(
            _artifact_row(
                str(row["overlay_path"]),
                paths,
                artifact_kind="figure",
                category="representative_spatial_overlays",
                fmt="svg",
                description=f"Representative spatial overlay for community {int(row['community_id'])}",
            )
        )
    return pd.DataFrame(rows)


def _build_manifest(
    *,
    inputs: AtlasInputs,
    tables: AtlasTables,
    paths: AtlasPaths,
    max_overlay_communities: int,
) -> dict[str, Any]:
    field_mapping = inputs.field_mapping
    return {
        "task_name": str(inputs.config_bundle.raw_config.get("task_name", "Task A")),
        "workflow_name": "write_task_a_descriptive_atlas",
        "atlas_role": ATLAS_ROLE,
        "claim_scope": ATLAS_ROLE,
        "scientific_interpretation_allowed": False,
        "config_path": str(inputs.config_path),
        "stage0_h5ad": str(inputs.stage0_h5ad),
        "community_id_key": field_mapping["state_id_key"],
        "cell_subtype_key": field_mapping["cell_subtype_key"],
        "domain_key": field_mapping["domain_key"],
        "fov_key": field_mapping["fov_key"],
        "patient_id_key": field_mapping["patient_id_key"],
        "spatial_key": "spatial",
        "configured_community_ids": [int(community_id) for community_id in inputs.configured_community_ids],
        "observed_community_ids": tables.observed_community_ids,
        "domain_labels": tables.domain_order,
        "patient_ids": tables.patient_order,
        "max_overlay_communities": int(max_overlay_communities),
        "overlay_selection_rule": (
            "top communities by total cell count; within each community choose the ROI with the "
            "highest within-ROI community fraction, then highest community cell count, then "
            "patient_id/domain/roi_id ascending"
        ),
        "n_cells": int(tables.frame.shape[0]),
        "n_patients": int(len(tables.patient_order)),
        "n_rois": int(tables.frame[["patient_id", "fov_id"]].drop_duplicates().shape[0]),
        "n_observed_communities": int(len(tables.observed_community_ids)),
        "n_cell_subtypes": int(len(tables.cell_subtype_order)),
        "output_index": str(paths.output_index),
    }


def write_task_a_descriptive_atlas(
    *,
    config_path: str | Path,
    stage0_h5ad: str | Path,
    output_dir: str | Path,
    patient_ids: tuple[str, ...] | list[str] | None = None,
    max_overlay_communities: int = DEFAULT_MAX_OVERLAY_COMMUNITIES,
) -> dict[str, Any]:
    if max_overlay_communities <= 0:
        raise DescriptiveAtlasContractError("Task A descriptive atlas requires --max-overlay-communities >= 1")

    resolved_patient_ids = None
    if patient_ids is not None:
        resolved_patient_ids = tuple(str(patient_id) for patient_id in patient_ids)
        if not resolved_patient_ids:
            raise DescriptiveAtlasContractError("Task A descriptive atlas patient_ids must not be empty")

    inputs = _load_inputs(
        config_path=config_path,
        stage0_h5ad=stage0_h5ad,
        patient_ids=resolved_patient_ids,
    )
    tables = _build_tables(inputs, max_overlay_communities=max_overlay_communities)
    paths = _resolve_paths(output_dir)
    _write_tables(tables, paths)
    _write_summary_figures(tables, paths)
    overlay_selection = _write_overlay_figures(tables, paths)
    overlay_selection.to_csv(paths.overlay_selection, index=False)
    output_index = _build_output_index(paths, overlay_selection)
    output_index.to_csv(paths.output_index, index=False)
    manifest = _build_manifest(
        inputs=inputs,
        tables=tables,
        paths=paths,
        max_overlay_communities=max_overlay_communities,
    )
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


__all__ = ["write_task_a_descriptive_atlas"]
