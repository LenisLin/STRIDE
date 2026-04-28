"""Raw bundle layout and writer helpers for internal Block 3 Phase 3 execution.

Role:
    Define the raw artifact layout and write the raw bundle subset for one
    internal Block 3 subexperiment.

Authority anchors:
    - docs/task_A_spec.md §4.5.5, §4.5.6, §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §5.5, §5.6

Local boundary:
    - This module owns raw bundle schema layout and raw artifact writing only.
    - It does not reorganize data into review-facing tables.
    - It does not reopen public workflow entrypoints or the packet bridge.

Primary contents:
    - Raw artifact layout dataclasses.
    - Schema builders for `3A` versus method-bearing sections.
    - Raw bundle manifest/index writers for one subexperiment at a time.

Why this module exists:
    Block 3 raw outputs need a stable routing layer that is shared by the
    execution logic and downstream review builders. Keeping raw bundle concerns
    here makes the `3A` non-method-bearing schema split explicit without mixing
    it into orchestration or review-surface code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stride.errors import ContractError

from .contracts import Block3SubexperimentRawRows
from .execution import BLOCK3_PACKET_BRIDGE_POLICY, Block3ExecutionPlan


@dataclass(frozen=True)
class Block3ArtifactLayout:
    """Layout record for one raw Block 3 artifact."""

    role: str
    relative_path: str
    format: str
    schema_columns: tuple[str, ...]
    subexperiment_id: str | None = None


@dataclass(frozen=True)
class Block3BundleLayout:
    """Raw bundle layout describing the artifacts emitted for one execution pass."""

    artifacts: tuple[Block3ArtifactLayout, ...]
    artifact_state: str
    scientific_interpretation_allowed: bool
    packet_bridge_enabled: bool
    packet_bridge_policy: str
    manifest_name: str
    raw_index_name: str


@dataclass(frozen=True)
class Block3WrittenRawBundle:
    """Filesystem locations written for one raw Block 3 subexperiment bundle."""

    subexperiment_id: str
    manifest_path: Path
    raw_index_path: Path
    artifact_paths: dict[str, Path]


_BASE_ROUTING_COLUMNS: tuple[str, ...] = (
    "rerun_id",
    "subexperiment_id",
    "condition_id",
    "evaluation_family",
    "metric_name",
    "metric_role",
    "metric_status",
    "reported_value",
)

_METHOD_ROUTING_COLUMNS: tuple[str, ...] = (
    "method_name",
    "method_class",
)


def build_raw_table_schema(subexperiment_id: str, artifact_role: str) -> tuple[str, ...]:
    """Build the raw-table schema for one subexperiment artifact.

    Purpose:
        Return the frozen flat-column schema for one raw Block 3 artifact role.

    Inputs:
        subexperiment_id: Executable Block 3 subexperiment whose raw artifact is
            being described.
        artifact_role: Raw artifact role such as `3a_object_scores`,
            `3b1_patient_metrics`, or `3b2_patient_metrics`.

    Returns:
        The ordered tuple of flat column names expected in the corresponding raw
        CSV artifact.

    Raises:
        ContractError: The requested `(subexperiment_id, artifact_role)` pair is
            not part of the supported raw bundle surface.

    Core flow:
        1. Route `3A` object-score artifacts to the non-method-bearing schema.
        2. Route `3A` rerun-stability artifacts to the non-method-bearing
           schema plus stability-level metadata.
        3. Route patient-metric artifacts to the method-bearing schema with
           patient id.
        4. Route condition-summary artifacts to the method-bearing schema with
           summary-statistic fields.
        5. Fail fast for unsupported artifact roles.

    Notes:
        `3A` is the only non-method-bearing subexperiment. `3B`, `3C-1`, and
        `3C-2` retain method columns on patient and condition summary rows.
    """
    if artifact_role == "generator_rerun_registry":
        return (
            "rerun_id",
            "split_seed",
            "n_train_patients",
            "n_test_patients",
            "hidden_relation_condition_id",
        )
    if artifact_role == "generator_split_registry":
        return (
            "rerun_id",
            "split_seed",
            "patient_id",
            "split_role",
        )
    if artifact_role == "patient_truth_store":
        return (
            "rerun_id",
            "subexperiment_id",
            "condition_id",
            "patient_id",
            "x_json",
            "y_json",
            "A_json",
            "d_json",
            "e_json",
            "open_mass",
            "open_mass_scale",
        )
    if artifact_role == "method_native_output_store":
        return (
            "rerun_id",
            "subexperiment_id",
            "condition_id",
            "patient_id",
            "method_name",
            "method_class",
            "fit_status",
            "A_json",
            "d_json",
            "e_json",
            "mu_minus_json",
            "mu_plus_json",
            "P_json",
            "metadata_json",
            "open_mass_scale",
        )
    if subexperiment_id == "3A" and artifact_role == "3a_object_scores":
        return _BASE_ROUTING_COLUMNS[:4] + ("validation_object_id",) + _BASE_ROUTING_COLUMNS[4:]
    if subexperiment_id == "3A" and artifact_role == "3a_rerun_stability":
        return (
            _BASE_ROUTING_COLUMNS[:4]
            + ("validation_object_id",)
            + _BASE_ROUTING_COLUMNS[4:]
            + ("stability_summary_level",)
        )
    if artifact_role.endswith("patient_metrics"):
        return _BASE_ROUTING_COLUMNS[:4] + _METHOD_ROUTING_COLUMNS + _BASE_ROUTING_COLUMNS[4:] + (
            "open_mass_scale",
            "patient_id",
        )
    if artifact_role.endswith("condition_summary"):
        return _BASE_ROUTING_COLUMNS[:4] + _METHOD_ROUTING_COLUMNS + _BASE_ROUTING_COLUMNS[4:] + (
            "open_mass_scale",
            "summary_level",
            "mean_value",
            "ci_lower",
            "ci_upper",
            "paired_difference_vs_stride_reference",
        )
    raise ContractError(
        f"Unsupported Block 3 raw artifact_role {artifact_role!r} for subexperiment {subexperiment_id!r}"
    )


def _filter_bundle_artifacts(
    artifacts: tuple[Block3ArtifactLayout, ...],
    *,
    subexperiment_ids: tuple[str, ...] | None,
) -> tuple[Block3ArtifactLayout, ...]:
    """Keep only the raw artifacts relevant to the requested subexperiments.

    Purpose:
        Reduce the full raw bundle inventory to the subset needed for the
        current single-subexperiment write.

    Inputs / Returns:
        artifacts: Full tuple of raw artifact layouts.
        subexperiment_ids: Optional subexperiment filter; `None` keeps the full
            inventory.
        Returns the filtered tuple of `Block3ArtifactLayout` objects.

    Core flow:
        1. Return the full artifact tuple when no filter is provided.
        2. Build the allowed subexperiment-id set.
        3. Keep shared manifest/index artifacts plus layouts assigned to the
           requested subexperiment ids.
    """
    if subexperiment_ids is None:
        return artifacts
    allowed = set(subexperiment_ids)
    return tuple(
        artifact
        for artifact in artifacts
        if (
            artifact.role != "method_native_output_store"
            or allowed != {"3A"}
        )
        and (artifact.subexperiment_id is None or artifact.subexperiment_id in allowed)
    )


def build_block3_bundle_layout(
    plan: Block3ExecutionPlan,
    *,
    manifest_name: str | None = None,
    raw_index_name: str | None = None,
    subexperiment_ids: tuple[str, ...] | None = None,
) -> Block3BundleLayout:
    """Build the raw artifact layout for the internal Block 3 bundle surface.

    Purpose:
        Assemble the raw bundle layout used by one Block 3 internal execution
        pass.

    Inputs:
        plan: Execution-plan metadata supplying artifact-state and boundary
            flags.
        manifest_name: Optional override for the raw manifest filename.
        raw_index_name: Optional override for the raw index filename.
        subexperiment_ids: Optional filter restricting the layout to one or more
            executable subexperiments.

    Returns:
        A `Block3BundleLayout` describing the raw artifacts that should exist
        for the requested execution scope.

    Raises:
        ContractError: A review-facing artifact leaks into the raw bundle
            layout.

    Core flow:
        1. Pick manifest and raw-index filenames.
        2. Construct the full raw artifact inventory for `3A`, `3B`, `3C-1`,
           and `3C-2`.
        3. Filter that inventory to the requested subexperiment subset.
        4. Attach artifact-state and boundary flags from the execution plan.
        5. Fail fast if review-surface artifacts accidentally appear here.

    Notes:
        This builder defines raw-layout inventory only. It does not write files.
    """
    manifest_filename = manifest_name or "block3_method_validation_manifest.json"
    raw_index_filename = raw_index_name or "block3_raw_index.csv"
    all_artifacts = (
        Block3ArtifactLayout(
            role="bundle_manifest",
            relative_path=manifest_filename,
            format="json",
            schema_columns=(),
        ),
        Block3ArtifactLayout(
            role="raw_index",
            relative_path=raw_index_filename,
            format="csv",
            schema_columns=("artifact_role", "relative_path", "format", "subexperiment_id"),
        ),
        Block3ArtifactLayout(
            role="generator_rerun_registry",
            relative_path="raw/generator_rerun_registry.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3A", "generator_rerun_registry"),
        ),
        Block3ArtifactLayout(
            role="generator_split_registry",
            relative_path="raw/generator_split_registry.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3A", "generator_split_registry"),
        ),
        Block3ArtifactLayout(
            role="patient_truth_store",
            relative_path="raw/patient_truth_store.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3A", "patient_truth_store"),
        ),
        Block3ArtifactLayout(
            role="method_native_output_store",
            relative_path="raw/method_native_output_store.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3B", "method_native_output_store"),
        ),
        Block3ArtifactLayout(
            role="3a_object_scores",
            relative_path="raw/3a_object_scores.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3A", "3a_object_scores"),
            subexperiment_id="3A",
        ),
        Block3ArtifactLayout(
            role="3a_rerun_stability",
            relative_path="raw/3a_rerun_stability.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3A", "3a_rerun_stability"),
            subexperiment_id="3A",
        ),
        Block3ArtifactLayout(
            role="3b1_patient_metrics",
            relative_path="raw/3b1_patient_metrics.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3B-1", "3b1_patient_metrics"),
            subexperiment_id="3B-1",
        ),
        Block3ArtifactLayout(
            role="3b1_condition_summary",
            relative_path="raw/3b1_condition_summary.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3B-1", "3b1_condition_summary"),
            subexperiment_id="3B-1",
        ),
        Block3ArtifactLayout(
            role="3b2_patient_metrics",
            relative_path="raw/3b2_patient_metrics.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3B-2", "3b2_patient_metrics"),
            subexperiment_id="3B-2",
        ),
        Block3ArtifactLayout(
            role="3b2_condition_summary",
            relative_path="raw/3b2_condition_summary.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3B-2", "3b2_condition_summary"),
            subexperiment_id="3B-2",
        ),
        Block3ArtifactLayout(
            role="3c1_patient_metrics",
            relative_path="raw/3c1_patient_metrics.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3C-1", "3c1_patient_metrics"),
            subexperiment_id="3C-1",
        ),
        Block3ArtifactLayout(
            role="3c1_condition_summary",
            relative_path="raw/3c1_condition_summary.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3C-1", "3c1_condition_summary"),
            subexperiment_id="3C-1",
        ),
        Block3ArtifactLayout(
            role="3c2_patient_metrics",
            relative_path="raw/3c2_patient_metrics.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3C-2", "3c2_patient_metrics"),
            subexperiment_id="3C-2",
        ),
        Block3ArtifactLayout(
            role="3c2_condition_summary",
            relative_path="raw/3c2_condition_summary.csv",
            format="csv",
            schema_columns=build_raw_table_schema("3C-2", "3c2_condition_summary"),
            subexperiment_id="3C-2",
        ),
    )
    artifacts = _filter_bundle_artifacts(all_artifacts, subexperiment_ids=subexperiment_ids)
    layout = Block3BundleLayout(
        artifacts=artifacts,
        artifact_state=plan.artifact_state,
        scientific_interpretation_allowed=plan.scientific_interpretation_allowed,
        packet_bridge_enabled=plan.packet_bridge_enabled,
        packet_bridge_policy=BLOCK3_PACKET_BRIDGE_POLICY,
        manifest_name=manifest_filename,
        raw_index_name=raw_index_filename,
    )
    if any(artifact.role.endswith("review_surface") for artifact in layout.artifacts):
        raise ContractError("Block 3 bundle layout must not absorb review-facing artifacts")
    return layout


def build_bundle_manifest_payload(
    layout: Block3BundleLayout,
    *,
    workflow_name: str = "block3_internal_phase3_execution",
    subexperiment_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON payload written as the raw bundle manifest.

    Purpose:
        Convert a raw bundle layout into the manifest payload persisted beside
        the raw artifact index.

    Inputs:
        layout: Frozen raw bundle layout for the current write scope.
        workflow_name: Workflow label recorded in the manifest payload.
        subexperiment_id: Optional executable subexperiment recorded when the
            bundle only covers one subexperiment.

    Returns:
        A JSON-serializable manifest payload describing the raw bundle surface.

    Core flow:
        1. Copy artifact-state and packet-boundary metadata from the layout.
        2. Serialize each artifact layout into a flat manifest entry.
        3. Optionally record the single-subexperiment label for narrowed writes.

    Notes:
        This helper shapes metadata only. It does not touch the filesystem.
    """
    payload: dict[str, Any] = {
        "workflow_name": workflow_name,
        "artifact_state": layout.artifact_state,
        "scientific_interpretation_allowed": layout.scientific_interpretation_allowed,
        "packet_bridge_enabled": layout.packet_bridge_enabled,
        "packet_bridge_policy": layout.packet_bridge_policy,
        "raw_index_name": layout.raw_index_name,
        "artifacts": [
            {
                "artifact_role": artifact.role,
                "relative_path": artifact.relative_path,
                "format": artifact.format,
                "subexperiment_id": artifact.subexperiment_id,
            }
            for artifact in layout.artifacts
        ],
    }
    if subexperiment_id is not None:
        payload["subexperiment_id"] = subexperiment_id
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk for a raw bundle sidecar.

    Purpose:
        Persist a manifest-like JSON payload while ensuring parent directories
        exist.

    Inputs / Returns:
        path: Destination JSON path.
        payload: JSON-serializable mapping to write.
        Returns `None` after the file is written.

    Core flow:
        1. Create parent directories when needed.
        2. Serialize the payload with stable indentation and key ordering.
        3. Write the encoded text to the requested path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a pandas frame to disk for a raw bundle artifact.

    Purpose:
        Persist one raw artifact or index CSV while ensuring parent directories
        exist.

    Inputs / Returns:
        path: Destination CSV path.
        frame: Data frame already aligned to the target schema.
        Returns `None` after the file is written.

    Core flow:
        1. Create parent directories when needed.
        2. Serialize the frame without an implicit index column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _coerce_frame(
    records: list[dict[str, object]],
    *,
    schema_columns: tuple[str, ...],
    artifact_role: str,
) -> pd.DataFrame:
    """Coerce raw records into a schema-aligned data frame.

    Purpose:
        Convert flat raw-record dictionaries into a `DataFrame` whose columns
        exactly match the artifact schema.

    Inputs / Returns:
        records: Flat record dictionaries already shaped for one raw artifact.
        schema_columns: Ordered column tuple required by the artifact contract.
        artifact_role: Artifact label used only for fail-fast error reporting.
        Returns a schema-aligned pandas `DataFrame`.

    Raises:
        ContractError: The frame columns do not match the declared schema.

    Core flow:
        1. Build a frame using the declared schema as the column order.
        2. Compare the resulting frame columns to the schema tuple.
        3. Fail fast when any column is missing, reordered, or unexpected.
    """
    frame = pd.DataFrame.from_records(records, columns=schema_columns)
    if tuple(frame.columns) != schema_columns:
        raise ContractError(
            f"Block 3 raw artifact {artifact_role!r} columns do not match schema {schema_columns!r}"
        )
    return frame


def _raw_role_records(
    *,
    subexperiment_id: str,
    raw_rows: Block3SubexperimentRawRows,
) -> dict[str, list[dict[str, object]]]:
    """Map typed raw-row containers to raw artifact-role records.

    Purpose:
        Split one subexperiment's typed raw rows into the exact raw artifact
        roles that should be written to disk.

    Inputs / Returns:
        subexperiment_id: Executable Block 3 subexperiment being written.
        raw_rows: Typed raw-row container produced by the execution layer.
        Returns a mapping from raw artifact role to flat record list.

    Raises:
        ContractError: `3A` carries method-bearing rows, a method-bearing
            section carries `3A` rows, required row families are missing, or the
            subexperiment id is unsupported.

    Core flow:
        1. Route `3A` to object-score and rerun-stability records only.
        2. Reject method-bearing rows on `3A`.
        3. Reject `3A` validation-object rows on method-bearing sections.
        4. Route `3B`, `3C-1`, and `3C-2` to patient-metric and
           condition-summary artifact roles.
        5. Fail fast for unsupported subexperiment ids.

    Notes:
        `3A` and the method-bearing sections are mutually exclusive row
        families at this layer.
    """
    if subexperiment_id == "3A":
        if raw_rows.patient_metrics or raw_rows.condition_summaries:
            raise ContractError("3A raw artifacts must remain non-method-bearing")
        if not raw_rows.object_scores or not raw_rows.rerun_stability:
            raise ContractError("3A raw artifacts require object scores and rerun stability rows")
        role_records = {
            "3a_object_scores": [row.to_record() for row in raw_rows.object_scores],
            "3a_rerun_stability": [row.to_record() for row in raw_rows.rerun_stability],
        }
        role_records.update({role: list(records) for role, records in raw_rows.shared_tables.items()})
        return role_records
    if raw_rows.object_scores or raw_rows.rerun_stability:
        raise ContractError(f"{subexperiment_id} raw artifacts must not carry 3A validation-object rows")
    if not raw_rows.patient_metrics or not raw_rows.condition_summaries:
        raise ContractError(f"{subexperiment_id} raw artifacts require patient rows and condition summaries")
    suffix = {
        "3B-1": "3b1",
        "3B-2": "3b2",
        "3C-1": "3c1",
        "3C-2": "3c2",
    }.get(subexperiment_id)
    if suffix is None:
        raise ContractError(f"Unsupported Block 3 raw subexperiment {subexperiment_id!r}")
    role_records = {
        f"{suffix}_patient_metrics": [row.to_record() for row in raw_rows.patient_metrics],
        f"{suffix}_condition_summary": [row.to_record() for row in raw_rows.condition_summaries],
    }
    role_records.update({role: list(records) for role, records in raw_rows.shared_tables.items()})
    return role_records


def write_block3_subexperiment_raw_bundle(
    *,
    output_dir: str | Path,
    plan: Block3ExecutionPlan,
    subexperiment_id: str,
    raw_rows: Block3SubexperimentRawRows,
) -> Block3WrittenRawBundle:
    """Write the raw bundle subset for one Block 3 subexperiment.

    Purpose:
        Materialize the raw CSV artifacts, raw index, and manifest for a single
        internally executed Block 3 subexperiment.

    Inputs:
        output_dir: Root directory where the raw bundle subset should be
            written.
        plan: Execution plan carrying artifact-state and boundary metadata.
        subexperiment_id: Executable Block 3 subexperiment whose raw rows should
            be emitted.
        raw_rows: Typed raw rows already validated by the execution layer.

    Returns:
        A `Block3WrittenRawBundle` describing the raw manifest, raw index, and
        per-artifact output paths written for the requested subexperiment.

    Raises:
        ContractError: The row families do not match the subexperiment contract
            or the flat records do not match the declared schema.

    Core flow:
        1. Resolve the output directory and narrow the raw layout to the
           requested subexperiment.
        2. Convert typed rows into flat artifact-role records.
        3. Coerce each record set into a schema-aligned frame and write the
           per-artifact CSV files.
        4. Write the raw index describing the emitted artifact subset.
        5. Build and write the raw bundle manifest.
        6. Return the filesystem paths collected during the write.

    Notes:
        This writer emits only the requested subexperiment subset. It does not
        write review-facing artifacts.
    """
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    layout = build_block3_bundle_layout(plan, subexperiment_ids=(subexperiment_id,))
    role_records = _raw_role_records(subexperiment_id=subexperiment_id, raw_rows=raw_rows)

    artifact_paths: dict[str, Path] = {}
    for artifact in layout.artifacts:
        if artifact.role not in role_records:
            continue
        artifact_path = resolved_output_dir / artifact.relative_path
        frame = _coerce_frame(
            role_records[artifact.role],
            schema_columns=artifact.schema_columns,
            artifact_role=artifact.role,
        )
        _write_csv(artifact_path, frame)
        artifact_paths[artifact.role] = artifact_path

    raw_index_path = resolved_output_dir / layout.raw_index_name
    raw_index_frame = pd.DataFrame.from_records(
        [
            {
                "artifact_role": artifact.role,
                "relative_path": artifact.relative_path,
                "format": artifact.format,
                "subexperiment_id": artifact.subexperiment_id,
            }
            for artifact in layout.artifacts
        ],
        columns=("artifact_role", "relative_path", "format", "subexperiment_id"),
    )
    _write_csv(raw_index_path, raw_index_frame)

    manifest_path = resolved_output_dir / layout.manifest_name
    manifest_payload = build_bundle_manifest_payload(
        layout,
        workflow_name="block3_internal_phase3_execution",
        subexperiment_id=subexperiment_id,
    )
    _write_json(manifest_path, manifest_payload)
    return Block3WrittenRawBundle(
        subexperiment_id=subexperiment_id,
        manifest_path=manifest_path,
        raw_index_path=raw_index_path,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "Block3ArtifactLayout",
    "Block3BundleLayout",
    "Block3WrittenRawBundle",
    "build_block3_bundle_layout",
    "build_bundle_manifest_payload",
    "build_raw_table_schema",
    "write_block3_subexperiment_raw_bundle",
]
