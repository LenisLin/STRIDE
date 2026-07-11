"""Relation-specific evidence assembly for STRIDE `.tl`.

This module converts one declared relation into estimator-ready evidence
blocks. It performs row selection, patient grouping, and deterministic subbag
construction. It does not validate `.pp` handoff slots and does not define the
shared state basis.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
)
from stride.errors import ContractError

BLOCK_CONSTRUCTION_POLICY = "partitioned_fov_subbag_v1"


@dataclass(frozen=True)
class EvidenceBlock:
    """One source/target FOV evidence block for a single patient.

    patient_id: patient identifier for the fitted relation.
    source_bag: source FOV composition matrix `[n_source_fov, K]`.
    target_bag: target FOV composition matrix `[n_target_fov, K]`.
    block_id: stable block identifier for provenance and loss records.
    metadata: block construction and support metadata.
    """

    patient_id: str
    source_bag: Any
    target_bag: Any
    block_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationInput:
    """Estimator input for one declared relation.

    relation_id: stable relation result key from `.io` config.
    source_timepoint: configured source side.
    target_timepoint: configured target side.
    source_domain: source-side `domain_label` selector.
    target_domain: target-side `domain_label` selector.
    patient_ids: eligible patient ids aligned to optimizer axis `P`.
    support_counts: support counts by patient and side.
    skipped_patient_ids: patients missing source or target support.
    blocks: relation evidence blocks.
    metadata: compact relation-level provenance.
    """

    relation_id: str
    source_timepoint: str
    target_timepoint: str
    source_domain: str
    target_domain: str
    patient_ids: tuple[str, ...]
    support_counts: Mapping[str, Mapping[str, int]]
    skipped_patient_ids: tuple[str, ...]
    blocks: tuple[EvidenceBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def resolve_relation(
    *,
    relation_id: str,
    source_timepoint: str,
    target_timepoint: str,
    source_domain: str,
    target_domain: str,
    metadata: pd.DataFrame,
    community_composition: Any,
) -> RelationInput:
    """Resolve one declared relation into estimator input.

    Purpose:
        Select source/target FOV rows, group support by patient, skip
        patient-level support gaps with warnings, and build paired subbags.

    Key variables:
        source_mask: rows matching source timepoint and source domain.
        target_mask: rows matching target timepoint and target domain.
        source_rows_by_patient: selected source rows grouped by patient.
        target_rows_by_patient: selected target rows grouped by patient.
        eligible_patient_ids: patients with source and target support.
        skipped_patient_ids: patients lacking one side of relation support.
    """
    # source_mask: relation-specific source evidence selector.
    # eligible_patient_ids: optimizer patient axis for this relation.
    source_mask = (metadata[OBS_TIMEPOINT_KEY] == source_timepoint) & (
        metadata[OBS_DOMAIN_KEY] == source_domain
    )
    target_mask = (metadata[OBS_TIMEPOINT_KEY] == target_timepoint) & (
        metadata[OBS_DOMAIN_KEY] == target_domain
    )
    source_rows = tuple(int(i) for i in np.flatnonzero(source_mask.to_numpy()))
    target_rows = tuple(int(i) for i in np.flatnonzero(target_mask.to_numpy()))
    patient_ids = metadata[OBS_PATIENT_KEY].astype(str).to_numpy(copy=False)

    source_rows_by_patient: dict[str, list[int]] = {}
    target_rows_by_patient: dict[str, list[int]] = {}
    for row in source_rows:
        patient_id = patient_ids[row]
        source_rows_by_patient.setdefault(patient_id, []).append(row)
    for row in target_rows:
        patient_id = patient_ids[row]
        target_rows_by_patient.setdefault(patient_id, []).append(row)

    patient_order: list[str] = []
    seen_patients: set[str] = set()
    selected_mask = source_mask | target_mask
    for row in np.flatnonzero(selected_mask.to_numpy()):
        patient_id = patient_ids[int(row)]
        if patient_id not in seen_patients:
            seen_patients.add(patient_id)
            patient_order.append(patient_id)

    support_counts: dict[str, dict[str, int]] = {}
    eligible_patient_ids: list[str] = []
    skipped_patient_ids: list[str] = []
    for patient_id in patient_order:
        n_source = len(source_rows_by_patient.get(patient_id, ()))
        n_target = len(target_rows_by_patient.get(patient_id, ()))
        support_counts[patient_id] = {"source": n_source, "target": n_target}
        if n_source > 0 and n_target > 0:
            eligible_patient_ids.append(patient_id)
        else:
            skipped_patient_ids.append(patient_id)

    if skipped_patient_ids:
        warnings.warn(
            (
                f"relation {relation_id!r} skipped patients missing source or target "
                f"support: {skipped_patient_ids}"
            ),
            UserWarning,
            stacklevel=2,
        )

    relation_metadata = {
        "block_construction_policy": BLOCK_CONSTRUCTION_POLICY,
        "n_source_rows": len(source_rows),
        "n_target_rows": len(target_rows),
        "n_eligible_patients": len(eligible_patient_ids),
        "n_skipped_patients": len(skipped_patient_ids),
    }
    if not eligible_patient_ids:
        warnings.warn(
            f"relation {relation_id!r} has no eligible patients",
            UserWarning,
            stacklevel=2,
        )
        return RelationInput(
            relation_id=relation_id,
            source_timepoint=source_timepoint,
            target_timepoint=target_timepoint,
            source_domain=source_domain,
            target_domain=target_domain,
            patient_ids=(),
            support_counts=support_counts,
            skipped_patient_ids=tuple(skipped_patient_ids),
            blocks=(),
            metadata=relation_metadata,
        )

    blocks: list[EvidenceBlock] = []
    for patient_id in eligible_patient_ids:
        blocks.extend(
            _build_evidence_blocks(
                patient_id=patient_id,
                source_rows=source_rows_by_patient[patient_id],
                target_rows=target_rows_by_patient[patient_id],
                metadata=metadata,
                community_composition=community_composition,
            )
        )

    return RelationInput(
        relation_id=relation_id,
        source_timepoint=source_timepoint,
        target_timepoint=target_timepoint,
        source_domain=source_domain,
        target_domain=target_domain,
        patient_ids=tuple(eligible_patient_ids),
        support_counts=support_counts,
        skipped_patient_ids=tuple(skipped_patient_ids),
        blocks=tuple(blocks),
        metadata=relation_metadata,
    )


def _order_observations(
    rows: Sequence[int],
    metadata: pd.DataFrame,
) -> tuple[int, ...]:
    """Return deterministic FOV row order within one patient-side bag.

    Purpose:
        Stabilize subbag construction and provenance across repeated runs.

    Key variables:
        rows: FOV row indices into `community_composition`.
        fov_id: FOV identifier used as primary stable sort key.
        domain_label: relation-side domain label retained for provenance.
        row_index: original matrix row used as final tie breaker.
    """
    # rows: selected FOV row indices before stable ordering.
    row_tuple = tuple(int(row) for row in rows)
    if not row_tuple:
        return ()
    fov_ids = metadata[OBS_FOV_KEY].astype(str).to_numpy(copy=False)
    domain_labels = metadata[OBS_DOMAIN_KEY].astype(str).to_numpy(copy=False)
    return tuple(
        sorted(
            row_tuple,
            key=lambda row: (
                fov_ids[row],
                domain_labels[row],
                int(row),
            ),
        )
    )


def _partition_subbags(
    row_indices: Sequence[int],
    n_parts: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition ordered FOV rows into deterministic non-empty subbags.

    Purpose:
        Implement the `partitioned_fov_subbag_v1` block policy.

    Key variables:
        base_size: minimum chunk size.
        remainder: number of chunks receiving one extra row.
        start: inclusive chunk start offset.
        stop: exclusive chunk stop offset.
    """
    # n_parts: number of paired source/target subbags.
    if isinstance(n_parts, bool) or not isinstance(n_parts, int) or n_parts <= 0:
        raise ContractError("subbag partition count must be a positive integer")
    row_tuple = tuple(int(row) for row in row_indices)
    if n_parts > len(row_tuple):
        raise ContractError("subbag partition count must not exceed row count")

    base_size, remainder = divmod(len(row_tuple), n_parts)
    chunks: list[tuple[int, ...]] = []
    start = 0
    for part_index in range(n_parts):
        size = base_size + (1 if part_index < remainder else 0)
        stop = start + size
        chunk = row_tuple[start:stop]
        if not chunk:
            raise ContractError("subbag partitions must be non-empty")
        chunks.append(chunk)
        start = stop
    return tuple(chunks)


def _build_evidence_blocks(
    *,
    patient_id: str,
    source_rows: Sequence[int],
    target_rows: Sequence[int],
    metadata: pd.DataFrame,
    community_composition: Any,
) -> tuple[EvidenceBlock, ...]:
    """Build paired source/target evidence blocks for one patient.

    Purpose:
        Convert selected row indices into source and target FOV bags with
        deterministic block identifiers and provenance metadata.

    Key variables:
        ordered_source_rows: stable source FOV row order.
        ordered_target_rows: stable target FOV row order.
        n_subbags: paired subbag count for this patient.
        source_chunk: row indices for one source subbag.
        target_chunk: row indices for one target subbag.
        block_id: provenance-stable block identifier.
    """
    # n_subbags: min source/target partition count for this patient.
    patient_id = str(patient_id).strip()
    if not patient_id:
        raise ContractError("patient_id must be non-empty")
    source_row_tuple = tuple(int(row) for row in source_rows)
    target_row_tuple = tuple(int(row) for row in target_rows)
    if not source_row_tuple or not target_row_tuple:
        raise ContractError(
            "eligible patient evidence requires non-empty source and target rows"
        )

    ordered_source_rows = _order_observations(source_row_tuple, metadata)
    ordered_target_rows = _order_observations(target_row_tuple, metadata)
    n_subbags = min(len(ordered_source_rows), len(ordered_target_rows))
    source_chunks = _partition_subbags(ordered_source_rows, n_subbags)
    target_chunks = _partition_subbags(ordered_target_rows, n_subbags)
    fov_ids = metadata[OBS_FOV_KEY].astype(str).to_numpy(copy=False)
    domain_labels = metadata[OBS_DOMAIN_KEY].astype(str).to_numpy(copy=False)

    blocks: list[EvidenceBlock] = []
    for subbag_index, (source_chunk, target_chunk) in enumerate(
        zip(source_chunks, target_chunks, strict=True)
    ):
        block_id = f"{patient_id}:subbag_{subbag_index}"
        blocks.append(
            EvidenceBlock(
                patient_id=patient_id,
                source_bag=_slice_composition_rows(community_composition, source_chunk),
                target_bag=_slice_composition_rows(community_composition, target_chunk),
                block_id=block_id,
                metadata={
                    "block_construction_policy": BLOCK_CONSTRUCTION_POLICY,
                    "subbag_index": subbag_index,
                    "n_blocks_for_patient": n_subbags,
                    "source_fov_ids": tuple(fov_ids[row] for row in source_chunk),
                    "target_fov_ids": tuple(fov_ids[row] for row in target_chunk),
                    "source_domain_labels": tuple(domain_labels[row] for row in source_chunk),
                    "target_domain_labels": tuple(domain_labels[row] for row in target_chunk),
                    "source_row_indices": source_chunk,
                    "target_row_indices": target_chunk,
                },
            )
        )
    return tuple(blocks)


def _slice_composition_rows(
    community_composition: Any,
    rows: Sequence[int],
) -> Any:
    """Return the FOV-by-state composition bag for selected rows.

    Purpose:
        Keep matrix slicing local to evidence construction.

    Key variables:
        community_composition: canonical `[n_fov, K]` matrix from `.pp`.
        rows: selected FOV row indices.
        bag: selected `[n_rows, K]` evidence matrix.
    """
    # bag: FOV composition support points used by the observation objective.
    row_tuple = tuple(int(row) for row in rows)
    if not row_tuple:
        raise ContractError("evidence bag rows must be non-empty")
    matrix = np.asarray(community_composition)
    if matrix.ndim != 2:
        raise ContractError("community_composition must be a 2D matrix")
    bag = matrix[list(row_tuple), :]
    if bag.ndim != 2 or bag.shape[0] != len(row_tuple):
        raise ContractError("sliced evidence bag must be a 2D row subset")
    return bag
