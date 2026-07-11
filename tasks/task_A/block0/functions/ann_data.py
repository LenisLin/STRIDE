"""AnnData helpers for the live Task A Block 0 fitting path."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError
from stride.pp import build_fov_observations, validate_ready

from ...workflows.fit_adapter import (
    derive_task_a_patient_side_compositions,
    read_task_a_fov_observations,
)
from .schemas import (
    SOURCE_DOMAIN,
    TARGET_DOMAIN,
    Block0DomainLabelPermutationAssignment,
)

_FOV_METADATA_COLUMNS = (
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBS_FOV_KEY,
    OBS_DOMAIN_KEY,
)
_ASSIGNMENT_LABELS = {SOURCE_DOMAIN, TARGET_DOMAIN}


def block0_fov_observation_signature(adata: AnnData) -> dict[str, object]:
    """Return a deterministic resume signature for the Block 0 AnnData source."""
    metadata, composition = _fov_metadata_and_composition(adata)
    digest = hashlib.sha256()
    observations = _sorted_metadata_with_composition(metadata, composition)
    patient_ids = tuple(sorted(metadata[OBS_PATIENT_KEY].astype(str).unique().tolist()))
    for row, vector in observations:
        descriptor = {
            OBS_PATIENT_KEY: str(row[OBS_PATIENT_KEY]),
            OBS_TIMEPOINT_KEY: str(row[OBS_TIMEPOINT_KEY]),
            OBS_FOV_KEY: str(row[OBS_FOV_KEY]),
            OBS_DOMAIN_KEY: str(row[OBS_DOMAIN_KEY]),
            "composition_shape": [int(value) for value in vector.shape],
        }
        digest.update(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(np.ascontiguousarray(vector, dtype=float).tobytes(order="C"))
    return {
        "label": "block0_tc_im_ann_data",
        "patient_ids": list(patient_ids),
        "n_observations": int(metadata.shape[0]),
        "digest": digest.hexdigest(),
    }


def build_null_tc_im_adata(
    real_adata: AnnData,
    assignments: Sequence[Block0DomainLabelPermutationAssignment],
    *,
    permutation_index: int,
) -> AnnData:
    """Copy a real pair AnnData and rebuild FOV observations after TC/IM relabeling."""
    if not assignments:
        raise ContractError("Block 0 null AnnData construction requires assignments")
    if any(int(assignment.permutation_index) != int(permutation_index) for assignment in assignments):
        raise ContractError("Block 0 null AnnData assignment permutation_index values must align")

    real_metadata, _composition = _fov_metadata_and_composition(real_adata)
    expected_keys = _assignment_keys_from_metadata(real_metadata)
    assignment_by_key = _assignment_by_key(assignments)
    if set(assignment_by_key) != expected_keys:
        missing = tuple(sorted(expected_keys - set(assignment_by_key)))
        extra = tuple(sorted(set(assignment_by_key) - expected_keys))
        raise ContractError(
            "Block 0 null AnnData assignments do not match real TC/IM FOV metadata; "
            f"missing={missing}, extra={extra}"
        )

    null_adata = real_adata.copy()
    _require_obs_columns(null_adata)
    for column in (OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY):
        null_adata.obs[column] = null_adata.obs[column].astype(str)
    original_patient_ids = null_adata.obs[OBS_PATIENT_KEY].astype(str).to_numpy(copy=True)
    original_fov_ids = null_adata.obs[OBS_FOV_KEY].astype(str).to_numpy(copy=True)
    original_domain_labels = null_adata.obs[OBS_DOMAIN_KEY].astype(str).to_numpy(copy=True)

    for key, assignment in assignment_by_key.items():
        patient_id, fov_id, original_domain_label = key
        mask = (
            (original_patient_ids == patient_id)
            & (original_fov_ids == fov_id)
            & (original_domain_labels == original_domain_label)
        )
        if not bool(mask.any()):
            raise ContractError(f"Block 0 null AnnData has no cells for assignment key {key!r}")
        null_adata.obs.loc[mask, OBS_DOMAIN_KEY] = str(assignment.permuted_domain_label)
        null_adata.obs.loc[mask, OBS_TIMEPOINT_KEY] = str(assignment.permuted_domain_label)

    stride_uns = _stride_uns_mapping(null_adata)
    stride_uns.pop(STRIDE_FOV_OBSERVATIONS_KEY, None)
    build_fov_observations(null_adata)
    validate_ready(null_adata)
    return null_adata


def derive_patient_side_burdens(adata: AnnData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return historical burden-named, composition-scale patient side vectors."""
    metadata, _composition = read_task_a_fov_observations(adata)
    patient_ids = tuple(sorted(metadata[OBS_PATIENT_KEY].astype(str).unique().tolist()))
    if not patient_ids:
        raise ContractError("Block 0 burden derivation requires at least one patient")
    return derive_task_a_patient_side_compositions(
        adata,
        source=SOURCE_DOMAIN,
        target=TARGET_DOMAIN,
        required_patient_ids=patient_ids,
    )


def _fov_metadata_and_composition(adata: AnnData) -> tuple[pd.DataFrame, np.ndarray]:
    return read_task_a_fov_observations(adata)


def _stride_uns_mapping(adata: AnnData) -> dict[str, object]:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, dict):
        raise ContractError("adata.uns['stride'] must be a mutable mapping")
    return stride_uns


def _sorted_metadata_with_composition(
    metadata: pd.DataFrame,
    composition: np.ndarray,
) -> tuple[tuple[pd.Series, np.ndarray], ...]:
    rows = []
    for row_index, row in metadata.iterrows():
        rows.append((row, composition[int(row_index)]))
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                str(item[0][OBS_PATIENT_KEY]),
                str(item[0][OBS_TIMEPOINT_KEY]),
                str(item[0][OBS_FOV_KEY]),
                str(item[0][OBS_DOMAIN_KEY]),
            ),
        )
    )


def _assignment_keys_from_metadata(metadata: pd.DataFrame) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in metadata.itertuples(index=False):
        patient_id = str(getattr(row, OBS_PATIENT_KEY))
        fov_id = str(getattr(row, OBS_FOV_KEY))
        domain_label = str(getattr(row, OBS_DOMAIN_KEY))
        if domain_label not in _ASSIGNMENT_LABELS:
            continue
        key = (patient_id, fov_id, domain_label)
        if key in keys:
            raise ContractError(
                "Block 0 real AnnData FOV metadata must be unique per "
                "(patient_id, fov_id, original_domain_label)"
            )
        keys.add(key)
    if not keys:
        raise ContractError("Block 0 real AnnData requires TC/IM FOV metadata")
    return keys


def _assignment_by_key(
    assignments: Sequence[Block0DomainLabelPermutationAssignment],
) -> dict[tuple[str, str, str], Block0DomainLabelPermutationAssignment]:
    assignment_by_key: dict[tuple[str, str, str], Block0DomainLabelPermutationAssignment] = {}
    for assignment in assignments:
        key = (
            str(assignment.patient_id),
            str(assignment.fov_id),
            str(assignment.original_domain_label),
        )
        if key in assignment_by_key:
            raise ContractError(
                "Block 0 null AnnData assignments must be unique per "
                "(patient_id, fov_id, original_domain_label)"
            )
        assignment_by_key[key] = assignment
    return assignment_by_key


def _require_obs_columns(adata: AnnData) -> None:
    missing = [
        column
        for column in (OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY)
        if column not in adata.obs
    ]
    if missing:
        raise ContractError("Block 0 AnnData is missing obs columns: " + ", ".join(missing))


__all__ = [
    "block0_fov_observation_signature",
    "build_null_tc_im_adata",
    "derive_patient_side_burdens",
]
