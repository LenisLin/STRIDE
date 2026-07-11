"""Within-patient label-permutation helpers for Task A Block 0 calibration.

Block 0 asks whether real `TC-IM` STRIDE relation structure departs from a
within-patient count-preserving FOV domain-label permutation null. The null
keeps patient identity, FOV composition, FOV count structure, and each
patient's `n_TC`/`n_IM` counts fixed while permuting `TC`/`IM` labels within
that same patient. Cross-patient label borrowing and relaxed fallback strata
are not allowed as null-model inputs. See
`tasks/task_A/README.md`, `tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import pandas as pd

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
)
from stride.errors import ContractError

if TYPE_CHECKING:
    from tasks.task_A.workflows.stride_adapter import TaskAFovRecord as ObservationRecord
else:
    ObservationRecord = Any

from .schemas import (
    SOURCE_DOMAIN,
    TARGET_DOMAIN,
    Block0DomainLabelPermutationAssignment,
    Block0PatientDomainCounts,
)

_ALLOWED_DOMAIN_LABELS = {SOURCE_DOMAIN, TARGET_DOMAIN}


def derive_block0_seed(
    *,
    master_seed: int,
    patient_id: str,
    permutation_index: int,
    namespace: str = "block0_calibration",
) -> int:
    """Derive a deterministic 32-bit seed from master seed, patient, and permutation."""
    payload = f"{namespace}|{int(master_seed)}|{patient_id}|{int(permutation_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _domain_label(observation: ObservationRecord) -> str:
    label = observation.domain_label if observation.domain_label is not None else observation.timepoint
    label = str(label)
    if label not in _ALLOWED_DOMAIN_LABELS:
        raise ContractError(
            f"Block 0 within-patient permutation requires TC/IM labels; observed {label!r}"
        )
    return label


def _group_observations_by_patient(
    real_observations: Iterable[ObservationRecord],
) -> dict[str, list[tuple[ObservationRecord, str]]]:
    grouped: dict[str, list[tuple[ObservationRecord, str]]] = defaultdict(list)
    seen_assignment_keys: set[tuple[str, str, str]] = set()
    for observation in real_observations:
        patient_id = str(observation.patient_id)
        fov_id = str(observation.fov_id)
        if patient_id.strip() == "":
            raise ContractError("Block 0 observations require non-empty patient_id")
        if fov_id.strip() == "":
            raise ContractError("Block 0 observations require non-empty fov_id")
        label = _domain_label(observation)
        assignment_key = (patient_id, fov_id, label)
        if assignment_key in seen_assignment_keys:
            raise ContractError(
                "Block 0 within-patient label permutation requires unique "
                "(patient_id, fov_id, original_domain_label) observations"
            )
        seen_assignment_keys.add(assignment_key)
        grouped[patient_id].append((observation, label))
    if not grouped:
        raise ContractError("Block 0 within-patient permutation requires observations")
    return grouped


def build_patient_domain_counts(
    real_observations: Iterable[ObservationRecord],
) -> tuple[Block0PatientDomainCounts, ...]:
    """Return per-patient `TC`/`IM` FOV counts preserved by the null model."""
    grouped = _group_observations_by_patient(real_observations)
    counts: list[Block0PatientDomainCounts] = []
    for patient_id in sorted(grouped):
        patient_observations = grouped[patient_id]
        labels = [label for _observation, label in patient_observations]
        n_TC = labels.count(SOURCE_DOMAIN)
        n_IM = labels.count(TARGET_DOMAIN)
        if n_TC == 0 or n_IM == 0:
            raise ContractError(
                f"Block 0 patient {patient_id!r} requires both TC and IM observations"
            )
        counts.append(
            Block0PatientDomainCounts(
                patient_id=patient_id,
                n_TC=n_TC,
                n_IM=n_IM,
                fov_ids=tuple(
                    sorted(str(observation.fov_id) for observation, _label in patient_observations)
                ),
            )
        )
    return tuple(counts)


def build_domain_label_permutation_assignments(
    real_observations: Iterable[ObservationRecord],
    *,
    permutation_index: int,
    master_seed: int,
) -> tuple[Block0DomainLabelPermutationAssignment, ...]:
    """Permute `TC`/`IM` FOV labels within each patient.

    The empirical null preserves patient identity, FOV ids, FOV composition,
    mass fields, and each patient's exact `n_TC`/`n_IM` counts. Identity
    permutations are allowed because this is a standard within-patient
    permutation null, not a no-identity reassignment constraint.
    """
    grouped = _group_observations_by_patient(real_observations)
    assignments: list[Block0DomainLabelPermutationAssignment] = []
    for patient_id in sorted(grouped):
        patient_observations = tuple(
            sorted(
                grouped[patient_id],
                key=lambda item: (str(item[0].fov_id), item[1]),
            )
        )
        original_labels = [label for _observation, label in patient_observations]
        if SOURCE_DOMAIN not in original_labels or TARGET_DOMAIN not in original_labels:
            raise ContractError(
                f"Block 0 patient {patient_id!r} requires both TC and IM observations"
            )
        patient_seed = derive_block0_seed(
            master_seed=master_seed,
            patient_id=patient_id,
            permutation_index=permutation_index,
            namespace="block0_within_patient_domain_labels",
        )
        permuted_labels = list(original_labels)
        random.Random(patient_seed).shuffle(permuted_labels)
        for (observation, original_label), permuted_label in zip(
            patient_observations,
            permuted_labels,
            strict=True,
        ):
            assignments.append(
                Block0DomainLabelPermutationAssignment(
                    permutation_index=int(permutation_index),
                    patient_id=patient_id,
                    fov_id=str(observation.fov_id),
                    original_domain_label=original_label,
                    permuted_domain_label=permuted_label,
                    seed=patient_seed,
                )
            )
    return tuple(assignments)


def build_domain_label_permutation_assignments_from_fov_metadata(
    metadata: object,
    *,
    permutation_index: int,
    master_seed: int,
) -> tuple[Block0DomainLabelPermutationAssignment, ...]:
    """Permute `TC`/`IM` FOV metadata labels within each patient."""
    grouped = _group_fov_metadata_by_patient(metadata)
    assignments: list[Block0DomainLabelPermutationAssignment] = []
    for patient_id in sorted(grouped):
        patient_rows = tuple(
            sorted(
                grouped[patient_id],
                key=lambda item: (item[1], item[2]),
            )
        )
        original_labels = [label for _patient_id, _fov_id, label in patient_rows]
        if SOURCE_DOMAIN not in original_labels or TARGET_DOMAIN not in original_labels:
            raise ContractError(
                f"Block 0 patient {patient_id!r} requires both TC and IM FOV metadata"
            )
        patient_seed = derive_block0_seed(
            master_seed=master_seed,
            patient_id=patient_id,
            permutation_index=permutation_index,
            namespace="block0_within_patient_domain_labels",
        )
        permuted_labels = list(original_labels)
        random.Random(patient_seed).shuffle(permuted_labels)
        for (_patient_id, fov_id, original_label), permuted_label in zip(
            patient_rows,
            permuted_labels,
            strict=True,
        ):
            assignments.append(
                Block0DomainLabelPermutationAssignment(
                    permutation_index=int(permutation_index),
                    patient_id=patient_id,
                    fov_id=fov_id,
                    original_domain_label=original_label,
                    permuted_domain_label=permuted_label,
                    seed=patient_seed,
                )
            )
    return tuple(assignments)


def _group_fov_metadata_by_patient(
    metadata: object,
) -> dict[str, list[tuple[str, str, str]]]:
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("Block 0 FOV metadata permutation requires a pandas DataFrame")
    missing = [
        column
        for column in (OBS_PATIENT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY)
        if column not in metadata
    ]
    if missing:
        raise ContractError("Block 0 FOV metadata is missing columns: " + ", ".join(missing))
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen_assignment_keys: set[tuple[str, str, str]] = set()
    for row in metadata.loc[:, [OBS_PATIENT_KEY, OBS_FOV_KEY, OBS_DOMAIN_KEY]].itertuples(index=False):
        patient_id = str(getattr(row, OBS_PATIENT_KEY)).strip()
        fov_id = str(getattr(row, OBS_FOV_KEY)).strip()
        label = str(getattr(row, OBS_DOMAIN_KEY)).strip()
        if label not in _ALLOWED_DOMAIN_LABELS:
            continue
        if patient_id == "":
            raise ContractError("Block 0 FOV metadata requires non-empty patient_id")
        if fov_id == "":
            raise ContractError("Block 0 FOV metadata requires non-empty fov_id")
        assignment_key = (patient_id, fov_id, label)
        if assignment_key in seen_assignment_keys:
            raise ContractError(
                "Block 0 within-patient label permutation requires unique "
                "(patient_id, fov_id, original_domain_label) FOV metadata rows"
            )
        seen_assignment_keys.add(assignment_key)
        grouped[patient_id].append(assignment_key)
    if not grouped:
        raise ContractError("Block 0 within-patient permutation requires TC/IM FOV metadata")
    return grouped


__all__ = [
    "build_domain_label_permutation_assignments",
    "build_domain_label_permutation_assignments_from_fov_metadata",
    "build_patient_domain_counts",
    "derive_block0_seed",
]
