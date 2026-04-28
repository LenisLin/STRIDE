"""Review-surface layout and writer helpers for internal Block 3 Phase 3 execution.

Role:
    Define the internal review-surface layout and write one review subset for a
    single Block 3 subexperiment.

Authority anchors:
    - docs/task_A_spec.md §4.5.6, §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §5.5, §5.6

Local boundary:
    - This module shapes internal review carriers that remain subordinate to the
      doc authority chain.
    - It does not define public review workflow behavior.
    - It does not reopen the packet bridge or turn review outputs into
      scientific authority.

Primary contents:
    - Review artifact layout dataclasses.
    - Schema builders for `3A` versus method-bearing review tables.
    - Review manifest/index writers for one subexperiment subset.

Why this module exists:
    The internal review surface needs flat, readable tables without mixing its
    layout logic into the raw bundle writer or the execution dispatcher. Keeping
    review-specific schema decisions here makes it explicit that `3A` remains
    non-method-bearing while `3B`/`3C-1`/`3C-2` carry summary and method
    columns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stride.errors import ContractError

from .bundle import Block3BundleLayout
from .contracts import Block3SubexperimentReviewRows
from .execution import BLOCK3_PACKET_BRIDGE_POLICY, Block3ExecutionPlan


@dataclass(frozen=True)
class Block3ReviewArtifactLayout:
    """Layout record for one Block 3 review artifact."""

    role: str
    relative_path: str
    format: str
    schema_columns: tuple[str, ...]
    subexperiment_id: str | None = None


@dataclass(frozen=True)
class Block3ReviewLayout:
    """Review layout describing the artifacts emitted for one write scope."""

    artifacts: tuple[Block3ReviewArtifactLayout, ...]
    artifact_state: str
    scientific_interpretation_allowed: bool
    packet_bridge_enabled: bool
    packet_bridge_policy: str
    manifest_name: str
    review_index_name: str


@dataclass(frozen=True)
class Block3WrittenReviewSurface:
    """Filesystem locations written for one review-surface subset."""

    subexperiment_id: str
    manifest_path: Path
    review_index_path: Path
    artifact_paths: dict[str, Path]


_BASE_REVIEW_COLUMNS: tuple[str, ...] = (
    "rerun_id",
    "subexperiment_id",
    "condition_id",
    "evaluation_family",
    "metric_name",
    "metric_role",
    "metric_status",
    "reported_value",
)

_METHOD_REVIEW_COLUMNS: tuple[str, ...] = (
    "method_name",
    "method_class",
)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "summary_level",
    "mean_value",
    "ci_lower",
    "ci_upper",
    "paired_difference_vs_stride_reference",
)


def build_review_table_schema(subexperiment_id: str) -> tuple[str, ...]:
    """Build the review-table schema for one Block 3 subexperiment.

    Purpose:
        Return the frozen flat-column schema for the review surface associated
        with one executable Block 3 subexperiment.

    Inputs:
        subexperiment_id: Executable Block 3 subexperiment whose review table is
            being described.

    Returns:
        The ordered tuple of flat column names expected in the corresponding
        review CSV artifact.

    Core flow:
        1. Route `3A` to the non-method-bearing review schema with validation
           object and stability-level columns.
        2. Route `3B`, `3C-1`, and `3C-2` to the method-bearing review schema
           with summary-statistic columns.

    Notes:
        `3A` review rows never carry `method_name` or `method_class`. The other
        executable sections always carry method and summary columns.
    """
    if subexperiment_id == "3A":
        return (
            _BASE_REVIEW_COLUMNS[:4]
            + ("validation_object_id",)
            + _BASE_REVIEW_COLUMNS[4:]
            + ("stability_summary_level", "review_surface_role")
        )
    return (
        _BASE_REVIEW_COLUMNS[:4]
        + _METHOD_REVIEW_COLUMNS
        + _BASE_REVIEW_COLUMNS[4:]
        + ("open_mass_scale",)
        + _SUMMARY_COLUMNS
        + ("section_title", "condition_title", "review_surface_role")
    )


def _filter_review_artifacts(
    artifacts: tuple[Block3ReviewArtifactLayout, ...],
    *,
    subexperiment_ids: tuple[str, ...] | None,
    include_extraction_route_index: bool,
) -> tuple[Block3ReviewArtifactLayout, ...]:
    """Keep only the review artifacts relevant to the requested write scope.

    Purpose:
        Reduce the full review-layout inventory to the subset needed for the
        current write.

    Inputs / Returns:
        artifacts: Full tuple of review artifact layouts.
        subexperiment_ids: Optional subexperiment filter; `None` keeps the full
            inventory.
        include_extraction_route_index: Whether the extraction-route index
            should remain in the returned inventory.
        Returns the filtered tuple of `Block3ReviewArtifactLayout` objects.

    Core flow:
        1. Decide whether the extraction-route index should be kept.
        2. If subexperiment filters are provided, keep only layouts assigned to
           the requested subexperiments.
        3. Return the filtered review artifact inventory.
    """
    allowed = None if subexperiment_ids is None else set(subexperiment_ids)
    filtered: list[Block3ReviewArtifactLayout] = []
    for artifact in artifacts:
        if artifact.role == "extraction_route_index" and not include_extraction_route_index:
            continue
        if allowed is not None and artifact.subexperiment_id is not None and artifact.subexperiment_id not in allowed:
            continue
        filtered.append(artifact)
    return tuple(filtered)


def build_block3_review_layout(
    plan: Block3ExecutionPlan,
    bundle_layout: Block3BundleLayout,
    *,
    manifest_name: str | None = None,
    review_index_name: str | None = None,
    subexperiment_ids: tuple[str, ...] | None = None,
    include_extraction_route_index: bool = True,
) -> Block3ReviewLayout:
    """Build the review artifact layout for the internal Block 3 surface.

    Purpose:
        Assemble the review-layout inventory used by one Block 3 internal review
        write.

    Inputs:
        plan: Execution-plan metadata supplying artifact-state and boundary
            flags.
        bundle_layout: Raw bundle layout already selected for the same write
            scope; included to keep raw and review layers paired.
        manifest_name: Optional override for the review manifest filename.
        review_index_name: Optional override for the review index filename.
        subexperiment_ids: Optional filter restricting the layout to one or more
            executable subexperiments.
        include_extraction_route_index: Whether to include the extraction-route
            index artifact in the returned layout.

    Returns:
        A `Block3ReviewLayout` describing the review artifacts that should exist
        for the requested scope.

    Core flow:
        1. Accept the paired raw bundle layout for the same write scope.
        2. Pick manifest and review-index filenames.
        3. Construct the full review artifact inventory, including the optional
           extraction-route index.
        4. Filter that inventory to the requested subexperiment subset.
        5. Attach artifact-state and boundary flags from the execution plan.

    Notes:
        The extraction-route index remains part of the review-layout inventory
        because it describes review/extraction routing metadata even when a
        specific write elects not to emit it.
    """
    _ = bundle_layout
    manifest_filename = manifest_name or "block3_review_manifest.json"
    review_index_filename = review_index_name or "block3_review_index.csv"
    all_artifacts = (
        Block3ReviewArtifactLayout(
            role="review_manifest",
            relative_path=manifest_filename,
            format="json",
            schema_columns=(),
        ),
        Block3ReviewArtifactLayout(
            role="review_index",
            relative_path=review_index_filename,
            format="csv",
            schema_columns=("artifact_role", "relative_path", "format", "subexperiment_id"),
        ),
        Block3ReviewArtifactLayout(
            role="extraction_route_index",
            relative_path="review/block3_extraction_route_index.csv",
            format="csv",
            schema_columns=(
                "subexperiment_id",
                "condition_id",
                "evaluation_family",
                "method_name",
                "method_class",
                "metric_name",
                "metric_role",
                "metric_status",
            ),
        ),
        Block3ReviewArtifactLayout(
            role="3a_review_surface",
            relative_path="review/3a_review_surface.csv",
            format="csv",
            schema_columns=build_review_table_schema("3A"),
            subexperiment_id="3A",
        ),
        Block3ReviewArtifactLayout(
            role="3b1_review_surface",
            relative_path="review/3b1_review_surface.csv",
            format="csv",
            schema_columns=build_review_table_schema("3B-1"),
            subexperiment_id="3B-1",
        ),
        Block3ReviewArtifactLayout(
            role="3b2_review_surface",
            relative_path="review/3b2_review_surface.csv",
            format="csv",
            schema_columns=build_review_table_schema("3B-2"),
            subexperiment_id="3B-2",
        ),
        Block3ReviewArtifactLayout(
            role="3c1_review_surface",
            relative_path="review/3c1_review_surface.csv",
            format="csv",
            schema_columns=build_review_table_schema("3C-1"),
            subexperiment_id="3C-1",
        ),
        Block3ReviewArtifactLayout(
            role="3c2_review_surface",
            relative_path="review/3c2_review_surface.csv",
            format="csv",
            schema_columns=build_review_table_schema("3C-2"),
            subexperiment_id="3C-2",
        ),
    )
    artifacts = _filter_review_artifacts(
        all_artifacts,
        subexperiment_ids=subexperiment_ids,
        include_extraction_route_index=include_extraction_route_index,
    )
    return Block3ReviewLayout(
        artifacts=artifacts,
        artifact_state=plan.artifact_state,
        scientific_interpretation_allowed=plan.scientific_interpretation_allowed,
        packet_bridge_enabled=plan.packet_bridge_enabled,
        packet_bridge_policy=BLOCK3_PACKET_BRIDGE_POLICY,
        manifest_name=manifest_filename,
        review_index_name=review_index_filename,
    )


def build_review_manifest_payload(
    layout: Block3ReviewLayout,
    *,
    workflow_name: str = "block3_internal_phase3_review",
    subexperiment_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON payload written as the review manifest.

    Purpose:
        Convert a review layout into the manifest payload persisted beside the
        review index.

    Inputs:
        layout: Frozen review layout for the current write scope.
        workflow_name: Workflow label recorded in the manifest payload.
        subexperiment_id: Optional executable subexperiment recorded when the
            review write covers only one subexperiment.

    Returns:
        A JSON-serializable manifest payload describing the review-surface
        inventory.

    Core flow:
        1. Copy artifact-state and packet-boundary metadata from the layout.
        2. Serialize each review artifact layout into a flat manifest entry.
        3. Optionally record the single-subexperiment label for narrowed writes.
    """
    payload: dict[str, Any] = {
        "workflow_name": workflow_name,
        "artifact_state": layout.artifact_state,
        "scientific_interpretation_allowed": layout.scientific_interpretation_allowed,
        "packet_bridge_enabled": layout.packet_bridge_enabled,
        "packet_bridge_policy": layout.packet_bridge_policy,
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
    """Write a JSON payload to disk for a review-side sidecar.

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
    """Write a pandas frame to disk for a review artifact.

    Purpose:
        Persist one review artifact or review index CSV while ensuring parent
        directories exist.

    Inputs / Returns:
        path: Destination CSV path.
        frame: Data frame already aligned to the target review schema.
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
    """Coerce review records into a schema-aligned data frame.

    Purpose:
        Convert flat review-record dictionaries into a `DataFrame` whose columns
        exactly match the declared review artifact schema.

    Inputs / Returns:
        records: Flat record dictionaries already shaped for one review
            artifact.
        schema_columns: Ordered column tuple required by the review artifact.
        artifact_role: Artifact label used only for fail-fast error reporting.
        Returns a schema-aligned pandas `DataFrame`.

    Raises:
        ContractError: The frame columns do not match the declared review
            schema.

    Core flow:
        1. Build a frame using the declared schema as the column order.
        2. Compare the resulting columns to the schema tuple.
        3. Fail fast when any column is missing, reordered, or unexpected.
    """
    frame = pd.DataFrame.from_records(records, columns=schema_columns)
    if tuple(frame.columns) != schema_columns:
        raise ContractError(
            f"Block 3 review artifact {artifact_role!r} columns do not match schema {schema_columns!r}"
        )
    return frame


def _review_role_records(
    *,
    subexperiment_id: str,
    review_rows: Block3SubexperimentReviewRows,
) -> dict[str, list[dict[str, object]]]:
    """Map typed review-row containers to review artifact-role records.

    Purpose:
        Split one subexperiment's typed review rows into the exact review
        artifact role that should be written to disk.

    Inputs / Returns:
        subexperiment_id: Executable Block 3 subexperiment being written.
        review_rows: Typed review-row container produced by the execution layer.
        Returns a mapping from review artifact role to flat record list.

    Raises:
        ContractError: `3A` carries section rows, a method-bearing section
            carries generator rows, required row families are missing, or the
            subexperiment id is unsupported.

    Core flow:
        1. Route `3A` to the generator-review surface only.
        2. Reject section-review rows on `3A`.
        3. Reject generator-review rows on method-bearing sections.
        4. Route `3B`, `3C-1`, and `3C-2` to their section-review artifacts.
        5. Fail fast for unsupported subexperiment ids.

    Notes:
        `3A` review rows and method-bearing section review rows are mutually
        exclusive at this layer.
    """
    if subexperiment_id == "3A":
        if review_rows.section_rows:
            raise ContractError("3A review surface must remain non-method-bearing")
        if not review_rows.generator_rows:
            raise ContractError("3A review surface requires generator review rows")
        return {"3a_review_surface": [row.to_record() for row in review_rows.generator_rows]}
    if review_rows.generator_rows:
        raise ContractError(f"{subexperiment_id} review surface must not carry 3A review rows")
    if not review_rows.section_rows:
        raise ContractError(f"{subexperiment_id} review surface requires section review rows")
    suffix = {
        "3B-1": "3b1",
        "3B-2": "3b2",
        "3C-1": "3c1",
        "3C-2": "3c2",
    }.get(subexperiment_id)
    if suffix is None:
        raise ContractError(f"Unsupported Block 3 review subexperiment {subexperiment_id!r}")
    return {f"{suffix}_review_surface": [row.to_record() for row in review_rows.section_rows]}


def write_block3_subexperiment_review_surface(
    *,
    output_dir: str | Path,
    plan: Block3ExecutionPlan,
    bundle_layout: Block3BundleLayout,
    subexperiment_id: str,
    review_rows: Block3SubexperimentReviewRows,
) -> Block3WrittenReviewSurface:
    """Write the review-surface subset for one Block 3 subexperiment.

    Purpose:
        Materialize the review CSV artifact, review index, and review manifest
        for a single internally executed Block 3 subexperiment.

    Inputs:
        output_dir: Root directory where the review subset should be written.
        plan: Execution plan carrying artifact-state and boundary metadata.
        bundle_layout: Raw bundle layout paired with the same subexperiment
            write.
        subexperiment_id: Executable Block 3 subexperiment whose review rows
            should be emitted.
        review_rows: Typed review rows already validated by the execution layer.

    Returns:
        A `Block3WrittenReviewSurface` describing the review manifest, review
        index, and per-artifact output paths written for the requested
        subexperiment.

    Raises:
        ContractError: The row families do not match the subexperiment contract
            or the flat records do not match the declared schema.

    Core flow:
        1. Resolve the output directory and narrow the review layout to the
           requested subexperiment.
        2. Convert typed review rows into flat artifact-role records.
        3. Coerce each record set into a schema-aligned frame and write the
           per-artifact CSV files.
        4. Write the review index describing the emitted artifact subset.
        5. Build and write the review manifest.
        6. Return the filesystem paths collected during the write.

    Notes:
        This writer emits only the current subexperiment review subset. It does
        not expose a public review workflow.
    """
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    layout = build_block3_review_layout(
        plan,
        bundle_layout,
        subexperiment_ids=(subexperiment_id,),
        include_extraction_route_index=False,
    )
    role_records = _review_role_records(subexperiment_id=subexperiment_id, review_rows=review_rows)

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

    review_index_path = resolved_output_dir / layout.review_index_name
    review_index_frame = pd.DataFrame.from_records(
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
    _write_csv(review_index_path, review_index_frame)

    manifest_path = resolved_output_dir / layout.manifest_name
    manifest_payload = build_review_manifest_payload(
        layout,
        workflow_name="block3_internal_phase3_review",
        subexperiment_id=subexperiment_id,
    )
    _write_json(manifest_path, manifest_payload)
    return Block3WrittenReviewSurface(
        subexperiment_id=subexperiment_id,
        manifest_path=manifest_path,
        review_index_path=review_index_path,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "Block3ReviewArtifactLayout",
    "Block3ReviewLayout",
    "Block3WrittenReviewSurface",
    "build_block3_review_layout",
    "build_review_manifest_payload",
    "build_review_table_schema",
    "write_block3_subexperiment_review_surface",
]
