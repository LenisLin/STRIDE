"""Objective Block 1 community-correspondence packet for Task A.

The packet is intentionally non-interpretive. It gathers machine-readable
tables that connect community ids across the Stage 0 atlas context and the
Block 1 community summary outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError

from ..contracts import EVIDENCE_READY_STATE, TaskAStage0StrideMappingSummary
from ..workflows.stride_adapter import load_task_a_dataset_handle


COMMUNITY_CORRESPONDENCE_DIRNAME = "community_correspondence"
COMMUNITY_CORRESPONDENCE_MANIFEST_FILENAME = "block1_community_correspondence_manifest.json"
COMMUNITY_CORRESPONDENCE_INDEX_FILENAME = "block1_community_correspondence_index.csv"


def _relative_to_output(path: Path, output_root: Path) -> str:
    return str(path.resolve().relative_to(output_root.resolve()))


def _build_cell_frame(
    *,
    data_path: str | Path,
    stage0_mapping: TaskAStage0StrideMappingSummary,
) -> pd.DataFrame:
    handle = load_task_a_dataset_handle(data_path)
    field_mapping = stage0_mapping.field_mapping
    obs_columns = [
        field_mapping.patient_id_key,
        field_mapping.cell_subtype_key,
        field_mapping.state_id_key,
    ]
    missing_columns = [column for column in obs_columns if column not in handle.adata.obs.columns]
    if missing_columns:
        raise ContractError(
            "Task A Block 1 community correspondence is missing required Stage 0 obs columns: "
            f"{missing_columns}"
        )
    obs = handle.adata.obs.loc[:, obs_columns].copy()
    frame = pd.DataFrame(
        {
            "patient_id": obs[field_mapping.patient_id_key].astype(str).to_numpy(),
            "cell_subtype_label": obs[field_mapping.cell_subtype_key].astype(str).to_numpy(),
            "community_id": obs[field_mapping.state_id_key].astype(int).to_numpy(),
        }
    )
    if frame.empty:
        raise ContractError("Task A Block 1 community correspondence cannot run on an empty cohort")
    return frame


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


def _configured_community_order(
    stage0_mapping: TaskAStage0StrideMappingSummary,
    observed_community_ids: set[int],
) -> list[int]:
    configured = [
        int(community_id)
        for community_id in stage0_mapping.field_mapping.state_ids
        if int(community_id) in observed_community_ids
    ]
    extras = sorted(observed_community_ids - set(configured))
    return configured + extras


def _build_source_major_targets_table(source_summary_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in source_summary_df.to_dict(orient="records"):
        for rank in (1, 2, 3):
            target_id = row.get(f"top_target_{rank}_id")
            target_value = row.get(f"top_target_{rank}_value")
            if pd.isna(target_id) or float(target_value) <= 0.0:
                continue
            records.append(
                {
                    "patient_id": str(row["patient_id"]),
                    "pair_family": str(row["pair_family"]),
                    "claim_role": str(row["claim_role"]),
                    "source_domain": str(row["source_domain"]),
                    "target_domain": str(row["target_domain"]),
                    "source_community_id": int(row["source_community_id"]),
                    "source_burden": float(row["source_burden"]),
                    "source_weight": float(row["source_weight"]),
                    "target_rank": int(rank),
                    "target_community_id": int(target_id),
                    "target_operator_value": float(target_value),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_family",
                "claim_role",
                "source_domain",
                "target_domain",
                "source_community_id",
                "source_burden",
                "source_weight",
                "target_rank",
                "target_community_id",
                "target_operator_value",
            ]
        )
    return frame.sort_values(
        ["patient_id", "pair_family", "source_community_id", "target_rank"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_source_burden_components_table(source_summary_df: pd.DataFrame) -> pd.DataFrame:
    if source_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_family",
                "claim_role",
                "source_domain",
                "target_domain",
                "source_community_id",
                "source_burden",
                "source_weight",
                "matched_burden",
                "self_retention_burden",
                "off_diagonal_burden",
                "depletion_burden",
            ]
        )
    frame = source_summary_df.loc[
        :,
        [
            "patient_id",
            "pair_family",
            "claim_role",
            "source_domain",
            "target_domain",
            "source_community_id",
            "source_burden",
            "source_weight",
            "self_retention_burden",
            "off_diagonal_burden",
            "depletion_burden",
        ],
    ].copy()
    frame["matched_burden"] = (
        frame["self_retention_burden"].astype(float)
        + frame["off_diagonal_burden"].astype(float)
    )
    frame = frame.loc[
        :,
        [
            "patient_id",
            "pair_family",
            "claim_role",
            "source_domain",
            "target_domain",
            "source_community_id",
            "source_burden",
            "source_weight",
            "matched_burden",
            "self_retention_burden",
            "off_diagonal_burden",
            "depletion_burden",
        ],
    ]
    return frame.sort_values(
        ["patient_id", "pair_family", "source_community_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_target_burden_components_table(target_summary_df: pd.DataFrame) -> pd.DataFrame:
    if target_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "pair_family",
                "claim_role",
                "source_domain",
                "target_domain",
                "target_community_id",
                "target_burden",
                "target_weight",
                "incoming_matched_operator",
                "incoming_matched_burden",
                "emergence_tendency",
                "emergence_burden",
            ]
        )
    return (
        target_summary_df.loc[
            :,
            [
                "patient_id",
                "pair_family",
                "claim_role",
                "source_domain",
                "target_domain",
                "target_community_id",
                "target_burden",
                "target_weight",
                "incoming_matched_operator",
                "incoming_matched_burden",
                "emergence_tendency",
                "emergence_burden",
            ],
        ]
        .sort_values(["patient_id", "pair_family", "target_community_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _pipe_join(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values if str(value).strip()})
    return "|".join(unique_values)


def _build_community_id_crosswalk(
    *,
    frame: pd.DataFrame,
    stage0_mapping: TaskAStage0StrideMappingSummary,
    source_summary_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    stage0_cell_counts = frame.groupby("community_id", observed=False).size().to_dict()
    stage0_patient_counts = (
        frame.groupby("community_id", observed=False)["patient_id"].nunique().to_dict()
    )
    source_row_counts = (
        source_summary_df.groupby("source_community_id", observed=False).size().to_dict()
        if not source_summary_df.empty
        else {}
    )
    source_patient_counts = (
        source_summary_df.groupby("source_community_id", observed=False)["patient_id"].nunique().to_dict()
        if not source_summary_df.empty
        else {}
    )
    target_row_counts = (
        target_summary_df.groupby("target_community_id", observed=False).size().to_dict()
        if not target_summary_df.empty
        else {}
    )
    target_patient_counts = (
        target_summary_df.groupby("target_community_id", observed=False)["patient_id"].nunique().to_dict()
        if not target_summary_df.empty
        else {}
    )
    source_pair_families = (
        source_summary_df.groupby("source_community_id", observed=False)["pair_family"].agg(_pipe_join).to_dict()
        if not source_summary_df.empty
        else {}
    )
    target_pair_families = (
        target_summary_df.groupby("target_community_id", observed=False)["pair_family"].agg(_pipe_join).to_dict()
        if not target_summary_df.empty
        else {}
    )

    configured_state_ids = [int(community_id) for community_id in stage0_mapping.field_mapping.state_ids]
    crosswalk_ids = list(configured_state_ids)
    observed_ids = {
        int(community_id) for community_id in frame["community_id"].astype(int).unique().tolist()
    }
    observed_ids.update(int(key) for key in source_row_counts)
    observed_ids.update(int(key) for key in target_row_counts)
    extras = sorted(observed_ids - set(crosswalk_ids))
    crosswalk_ids.extend(extras)
    configured_index = {community_id: idx for idx, community_id in enumerate(configured_state_ids)}

    records: list[dict[str, Any]] = []
    for community_id in crosswalk_ids:
        records.append(
            {
                "community_id": int(community_id),
                "configured_state_index": configured_index.get(int(community_id), np.nan),
                "observed_in_stage0": bool(stage0_cell_counts.get(int(community_id), 0) > 0),
                "n_stage0_cells": int(stage0_cell_counts.get(int(community_id), 0)),
                "n_stage0_patients": int(stage0_patient_counts.get(int(community_id), 0)),
                "observed_in_source_summary": bool(source_row_counts.get(int(community_id), 0) > 0),
                "n_source_summary_rows": int(source_row_counts.get(int(community_id), 0)),
                "n_source_summary_patients": int(source_patient_counts.get(int(community_id), 0)),
                "source_summary_pair_families": source_pair_families.get(int(community_id), ""),
                "observed_in_target_summary": bool(target_row_counts.get(int(community_id), 0) > 0),
                "n_target_summary_rows": int(target_row_counts.get(int(community_id), 0)),
                "n_target_summary_patients": int(target_patient_counts.get(int(community_id), 0)),
                "target_summary_pair_families": target_pair_families.get(int(community_id), ""),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["configured_state_index", "community_id"], kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )


def write_block1_community_correspondence_packet(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    stage0_mapping: TaskAStage0StrideMappingSummary,
    source_summary_path: str | Path,
    target_summary_path: str | Path,
    confirmatory_family_comparison_path: str | Path,
    exploratory_source_comparison_path: str | Path,
    exploratory_target_comparison_path: str | Path,
    source_summary_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    packet_root = output_root / COMMUNITY_CORRESPONDENCE_DIRNAME
    tables_dir = packet_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_cell_frame(data_path=data_path, stage0_mapping=stage0_mapping)
    observed_community_ids = {
        int(community_id) for community_id in frame["community_id"].astype(int).unique().tolist()
    }
    community_order = _configured_community_order(stage0_mapping, observed_community_ids)
    cell_subtype_order = sorted(frame["cell_subtype_label"].astype(str).unique().tolist())

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
    source_major_targets = _build_source_major_targets_table(source_summary_df)
    source_burden_components = _build_source_burden_components_table(source_summary_df)
    target_burden_components = _build_target_burden_components_table(target_summary_df)
    community_id_crosswalk = _build_community_id_crosswalk(
        frame=frame,
        stage0_mapping=stage0_mapping,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
    )

    counts_path = tables_dir / "community_cell_subtype_counts.csv"
    fractions_path = tables_dir / "community_cell_subtype_row_fractions.csv"
    source_targets_path = tables_dir / "source_community_major_targets.csv"
    source_burdens_path = tables_dir / "source_community_burden_components.csv"
    target_burdens_path = tables_dir / "target_community_burden_components.csv"
    crosswalk_path = tables_dir / "community_id_crosswalk.csv"

    community_cell_subtype_counts.to_csv(counts_path)
    community_cell_subtype_row_fractions.to_csv(fractions_path)
    source_major_targets.to_csv(source_targets_path, index=False)
    source_burden_components.to_csv(source_burdens_path, index=False)
    target_burden_components.to_csv(target_burdens_path, index=False)
    community_id_crosswalk.to_csv(crosswalk_path, index=False)

    source_summary_resolved = Path(source_summary_path).expanduser().resolve()
    target_summary_resolved = Path(target_summary_path).expanduser().resolve()
    family_comparison_resolved = Path(confirmatory_family_comparison_path).expanduser().resolve()
    source_comparison_resolved = Path(exploratory_source_comparison_path).expanduser().resolve()
    target_comparison_resolved = Path(exploratory_target_comparison_path).expanduser().resolve()

    index_rows = [
        {
            "relative_path": _relative_to_output(
                packet_root / COMMUNITY_CORRESPONDENCE_MANIFEST_FILENAME,
                output_root,
            ),
            "artifact_kind": "manifest",
            "category": "provenance",
            "format": "json",
            "description": "Block 1 objective community-correspondence manifest",
        },
        {
            "relative_path": _relative_to_output(
                packet_root / COMMUNITY_CORRESPONDENCE_INDEX_FILENAME,
                output_root,
            ),
            "artifact_kind": "index",
            "category": "provenance",
            "format": "csv",
            "description": "Machine-readable index of community-correspondence outputs",
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
            "relative_path": _relative_to_output(source_targets_path, output_root),
            "artifact_kind": "table",
            "category": "source_major_targets",
            "format": "csv",
            "description": "Source-community major target communities by family and patient",
        },
        {
            "relative_path": _relative_to_output(source_burdens_path, output_root),
            "artifact_kind": "table",
            "category": "source_burden_components",
            "format": "csv",
            "description": "Source-community matched, self-retention, remodeling, and depletion burdens",
        },
        {
            "relative_path": _relative_to_output(target_burdens_path, output_root),
            "artifact_kind": "table",
            "category": "target_burden_components",
            "format": "csv",
            "description": "Target-community incoming matched and emergence burden components",
        },
        {
            "relative_path": _relative_to_output(crosswalk_path, output_root),
            "artifact_kind": "table",
            "category": "community_crosswalk",
            "format": "csv",
            "description": "Community-id crosswalk connecting Stage 0 and Block 1 outputs",
        },
        {
            "relative_path": _relative_to_output(source_summary_resolved, output_root),
            "artifact_kind": "reference",
            "category": "summary_reference",
            "format": "csv",
            "description": "Primary Block 1 source-community summary surface",
        },
        {
            "relative_path": _relative_to_output(target_summary_resolved, output_root),
            "artifact_kind": "reference",
            "category": "summary_reference",
            "format": "csv",
            "description": "Primary Block 1 target-community summary surface",
        },
        {
            "relative_path": _relative_to_output(family_comparison_resolved, output_root),
            "artifact_kind": "reference",
            "category": "comparison_reference",
            "format": "csv",
            "description": "Confirmatory Block 1 family paired comparison surface",
        },
        {
            "relative_path": _relative_to_output(source_comparison_resolved, output_root),
            "artifact_kind": "reference",
            "category": "comparison_reference",
            "format": "csv",
            "description": "Exploratory Block 1 source-community paired comparison surface",
        },
        {
            "relative_path": _relative_to_output(target_comparison_resolved, output_root),
            "artifact_kind": "reference",
            "category": "comparison_reference",
            "format": "csv",
            "description": "Exploratory Block 1 target-community paired comparison surface",
        },
    ]
    index_path = packet_root / COMMUNITY_CORRESPONDENCE_INDEX_FILENAME
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    manifest = {
        "workflow_name": "write_task_a_block1_bundle",
        "packet_role": "objective_community_correspondence",
        "scientific_interpretation_allowed": False,
        "artifact_state": EVIDENCE_READY_STATE,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "stage0_h5ad": str(Path(data_path).expanduser().resolve()),
        "block1_output_dir": str(output_root),
        "community_id_key": stage0_mapping.field_mapping.state_id_key,
        "cell_subtype_key": stage0_mapping.field_mapping.cell_subtype_key,
        "patient_id_key": stage0_mapping.field_mapping.patient_id_key,
        "configured_state_ids": [int(community_id) for community_id in stage0_mapping.field_mapping.state_ids],
        "observed_community_ids": community_order,
        "patient_ids": [str(patient_id) for patient_id in stage0_mapping.patient_ids],
        "source_community_summary_path": str(source_summary_resolved),
        "target_community_summary_path": str(target_summary_resolved),
        "confirmatory_family_comparison_path": str(family_comparison_resolved),
        "exploratory_source_community_comparison_path": str(source_comparison_resolved),
        "exploratory_target_community_comparison_path": str(target_comparison_resolved),
        "output_index": str(index_path),
    }
    manifest_path = packet_root / COMMUNITY_CORRESPONDENCE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path, index_path


__all__ = [
    "COMMUNITY_CORRESPONDENCE_DIRNAME",
    "COMMUNITY_CORRESPONDENCE_INDEX_FILENAME",
    "COMMUNITY_CORRESPONDENCE_MANIFEST_FILENAME",
    "write_block1_community_correspondence_packet",
]
