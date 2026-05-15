"""Stage 0 observation surface for Task A Block 0 calibration.

Block 0 asks whether real `TC-IM` STRIDE relation structure departs from a
within-patient count-preserving FOV domain-label permutation null. Its hard
inputs are Task A config, Stage 0 h5ad, output dir, permutation count, master
seed, and optional selectors. Outputs remain calibration statistics, not
biological interpretation or downstream execution decisions. See
`tasks/task_A/README.md`,
`tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from stride import DatasetHandle
from stride.errors import ContractError
from stride.observation.contracts import FovObservation

from ...config import TaskAConfigBundle
from ...workflows.stride_adapter import (
    build_task_a_family_observations,
    resolve_task_a_state_basis,
)
from .permutation import build_patient_domain_counts
from .schemas import (
    DEMO_SUBSET_SCOPE,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    FULL_COHORT_SCOPE,
    PATIENT_SUBSET_SCOPE,
    REAL_FAMILY,
    SOURCE_DOMAIN,
    TARGET_DOMAIN,
    Block0DomainLabelPermutationAssignment,
    Block0RunConfig,
)


@dataclass(frozen=True)
class Block0ObservationBundle:
    """Observation bundle for one real or empirical-null Block 0 fit family."""

    label: str
    observations: tuple[FovObservation, ...]
    patient_ids: tuple[str, ...]
    permutation_index: int | None = None
    assignments: tuple[Block0DomainLabelPermutationAssignment, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in {FIT_LABEL_REAL, FIT_LABEL_NULL}:
            raise ContractError(f"Unsupported Block 0 observation label: {self.label!r}")
        if len(self.patient_ids) == 0:
            raise ContractError("Block 0 observation bundles require at least one patient")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ContractError("Block 0 observation bundle patient_ids must be unique")
        observed_patients = {str(observation.patient_id) for observation in self.observations}
        missing_observations = tuple(
            patient_id for patient_id in self.patient_ids if patient_id not in observed_patients
        )
        if missing_observations:
            raise ContractError(
                f"Block 0 observation bundle has patients without observations: {missing_observations}"
            )
        if self.label == FIT_LABEL_REAL:
            if self.permutation_index is not None or self.assignments:
                raise ContractError("Real Block 0 observations must not carry null assignment metadata")
        if self.label == FIT_LABEL_NULL:
            if (
                self.permutation_index is None
                or isinstance(self.permutation_index, bool)
                or not isinstance(self.permutation_index, int)
                or self.permutation_index < 0
            ):
                raise ContractError("Null Block 0 observations require a non-negative permutation_index")
            if not self.assignments:
                raise ContractError("Null Block 0 observations require domain-label assignments")


def resolve_block0_run_config(
    *,
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    n_permutations: int,
    master_seed: int,
    patient_ids: tuple[str, ...] | None = None,
    demo_subset_name: str | None = None,
) -> Block0RunConfig:
    """Resolve paths and diagnostic/full-cohort scope without loading real data."""
    if patient_ids is not None and demo_subset_name is not None:
        raise ContractError("Block 0 accepts either patient_ids or demo_subset_name, not both")
    if demo_subset_name is not None:
        run_scope = DEMO_SUBSET_SCOPE
    elif patient_ids is not None:
        run_scope = PATIENT_SUBSET_SCOPE
    else:
        run_scope = FULL_COHORT_SCOPE
    return Block0RunConfig(
        config_path=Path(config_path).expanduser().resolve(),
        data_path=Path(data_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
        run_scope=run_scope,
        n_permutations=int(n_permutations),
        master_seed=int(master_seed),
        patient_ids=None if patient_ids is None else tuple(dict.fromkeys(map(str, patient_ids))),
        demo_subset_name=None if demo_subset_name is None else str(demo_subset_name),
    )


def build_real_tc_im_observations(
    handle: DatasetHandle,
    config_bundle: TaskAConfigBundle,
    *,
    patient_ids: Sequence[str] | None = None,
) -> Block0ObservationBundle:
    """Build real `TC-IM` observations from Stage 0 plus Task A config only."""
    handle.validate(
        require_cell_type=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    selected_patient_ids = _resolve_requested_patients(handle, patient_ids)
    state_basis = resolve_task_a_state_basis(handle)
    family_spec = _resolve_tc_im_family(config_bundle)
    observations = build_task_a_family_observations(
        handle,
        family_spec,
        state_basis=state_basis,
        mass_mode=str(config_bundle.data.mass_mode),
        require_complete_patients=True,
    )
    if selected_patient_ids is not None:
        selected = set(selected_patient_ids)
        observations = tuple(
            observation
            for observation in observations
            if str(observation.patient_id) in selected
        )

    observed_patient_ids = tuple(sorted({str(observation.patient_id) for observation in observations}))
    if selected_patient_ids is not None:
        missing_complete = tuple(
            patient_id
            for patient_id in selected_patient_ids
            if patient_id not in set(observed_patient_ids)
        )
        if missing_complete:
            raise ContractError(
                "Block 0 requested patients require both TC and IM observations; "
                f"missing complete TC-IM support for {missing_complete}"
            )
    if not observations:
        raise ContractError("Block 0 real observations require both TC and IM observations")
    build_patient_domain_counts(observations)
    return Block0ObservationBundle(
        label=FIT_LABEL_REAL,
        observations=tuple(observations),
        patient_ids=observed_patient_ids,
    )


def _resolve_tc_im_family(config_bundle: TaskAConfigBundle):
    for family_spec in config_bundle.ordered_proxy.pair_families:
        if (
            family_spec.name == REAL_FAMILY
            and family_spec.source_domain == SOURCE_DOMAIN
            and family_spec.target_domain == TARGET_DOMAIN
        ):
            return family_spec
    raise ContractError("Task A config must define the Block 0 TC-IM ordered pair family")


def _resolve_requested_patients(
    handle: DatasetHandle,
    patient_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if patient_ids is None:
        return None
    requested = tuple(dict.fromkeys(str(patient_id) for patient_id in patient_ids))
    if not requested:
        raise ContractError("Block 0 patient selector must not be empty")
    available = set(handle.adata.obs["patient_id"].astype(str).unique().tolist())
    missing = tuple(patient_id for patient_id in requested if patient_id not in available)
    if missing:
        raise ContractError(f"Block 0 requested patients do not exist in Stage 0: {missing}")
    return requested


def build_null_tc_im_observations(
    real_observations: Block0ObservationBundle,
    assignments: Sequence[Block0DomainLabelPermutationAssignment],
    *,
    permutation_index: int,
) -> Block0ObservationBundle:
    """Build within-patient domain-label permuted observations for one null draw.

    The null keeps `patient_id`, `fov_id`, `community_composition`, `mass`, and
    `mass_mode` unchanged. It rewrites only `timepoint` and `domain_label`
    from the same-patient assignment table. Assignment metadata records the
    original label, permuted label, and permutation index for diagnostics only;
    it is not a new evidence input.
    """
    if real_observations.label != FIT_LABEL_REAL:
        raise ContractError("Null Block 0 observations must derive from a real observation bundle")
    if not assignments:
        raise ContractError("Null Block 0 observations require at least one assignment")
    if any(assignment.permutation_index != int(permutation_index) for assignment in assignments):
        raise ContractError("Null Block 0 assignment permutation_index values must align")

    assignment_by_key: dict[tuple[str, str, str], Block0DomainLabelPermutationAssignment] = {}
    for assignment in assignments:
        key = (
            str(assignment.patient_id),
            str(assignment.fov_id),
            str(assignment.original_domain_label),
        )
        if key in assignment_by_key:
            raise ContractError("Null Block 0 assignments must be unique per patient/FOV/original label")
        assignment_by_key[key] = assignment

    null_observations: list[FovObservation] = []
    observed_keys: set[tuple[str, str, str]] = set()
    allowed_labels = {SOURCE_DOMAIN, TARGET_DOMAIN}
    for observation in real_observations.observations:
        original_label = (
            observation.domain_label
            if observation.domain_label is not None
            else observation.timepoint
        )
        original_label = str(original_label)
        if original_label not in allowed_labels:
            raise ContractError("Block 0 null observations require TC/IM real labels")
        key = (str(observation.patient_id), str(observation.fov_id), original_label)
        if key in observed_keys:
            raise ContractError("Block 0 real observations must be unique per patient/FOV/original label")
        observed_keys.add(key)
        assignment = assignment_by_key.get(key)
        if assignment is None:
            raise ContractError(f"Missing Block 0 null assignment for observation key {key!r}")
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "block0_original_domain_label": assignment.original_domain_label,
                "block0_permuted_domain_label": assignment.permuted_domain_label,
                "block0_permutation_index": int(permutation_index),
            }
        )
        null_observations.append(
            replace(
                observation,
                timepoint=assignment.permuted_domain_label,
                domain_label=assignment.permuted_domain_label,
                metadata=metadata,
            )
        )

    extra_assignments = tuple(sorted(set(assignment_by_key) - observed_keys))
    if extra_assignments:
        raise ContractError(f"Block 0 null assignments do not match real observations: {extra_assignments}")

    return Block0ObservationBundle(
        label=FIT_LABEL_NULL,
        observations=tuple(null_observations),
        patient_ids=real_observations.patient_ids,
        permutation_index=int(permutation_index),
        assignments=tuple(assignments),
    )


__all__ = [
    "Block0ObservationBundle",
    "build_null_tc_im_observations",
    "build_real_tc_im_observations",
    "resolve_block0_run_config",
]
