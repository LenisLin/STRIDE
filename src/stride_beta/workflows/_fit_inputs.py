"""Internal patient-input grouping helpers for the STRIDE fit workflow."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..basis.contracts import StateBasis, validate_state_basis
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..observation.contracts import FovObservation, validate_fov_observation


def _require_nonempty_identifier(value: object, *, field_name: str) -> str:
    """Normalize one identifier field and require non-empty string content."""

    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string")
    return normalized


def _observation_key(observation: FovObservation) -> tuple[str, str, str, str]:
    """Build a stable observation identity key for grouping and validation."""

    return (
        str(observation.patient_id),
        str(observation.timepoint),
        str(observation.fov_id),
        str(observation.domain_label),
    )


def _normalize_nested_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Convert nested count mappings into plain serializable dictionaries."""

    return {
        str(group_label): {
            str(domain_label): int(count)
            for domain_label, count in domain_counts.items()
        }
        for group_label, domain_counts in counts.items()
    }


def _resolve_ordered_group_labels(
    observed_labels: tuple[str, ...],
    *,
    declared_order: tuple[str, ...],
) -> tuple[str, ...]:
    if not declared_order:
        return observed_labels

    extras = [label for label in observed_labels if label not in declared_order]
    if extras:
        raise ContractError(
            "Observed group labels are missing from TaskConfig.timepoint_order: "
            f"{tuple(extras)}"
        )

    return tuple(label for label in declared_order if label in observed_labels)


@dataclass(frozen=True)
class _FitObservationGroup:
    """One ordered-side/timepoint group of observations for a patient fit."""

    group_label: str
    observations: tuple[FovObservation, ...]
    observations_by_domain: Mapping[str, tuple[FovObservation, ...]]

    def __post_init__(self) -> None:
        _validate_fit_observation_group(self)


@dataclass(frozen=True)
class _PatientFitInput:
    """Canonical per-patient input bundle for STRIDE fitting."""

    patient_id: str
    ordered_group_labels: tuple[str, ...]
    groups: tuple[_FitObservationGroup, ...]
    n_observations_by_group: Mapping[str, int]
    n_observations_by_domain: Mapping[str, int]
    n_observations_by_group_and_domain: Mapping[str, Mapping[str, int]]
    state_basis: StateBasis | None = None
    geometry: StateGeometry | None = None
    mass_mode: str = "uniform"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_patient_fit_input(self)

    @property
    def observations(self) -> tuple[FovObservation, ...]:
        """Return the flattened ordered observation sequence across groups."""
        return tuple(
            observation
            for group in self.groups
            for observation in group.observations
        )

    @property
    def groups_by_label(self) -> dict[str, _FitObservationGroup]:
        """Return the grouped observations keyed by their declared label."""
        return {group.group_label: group for group in self.groups}


def _validate_fit_observation_group(group: _FitObservationGroup) -> None:
    """Validate one ordered-side/timepoint observation group."""
    group_label = _require_nonempty_identifier(
        group.group_label,
        field_name="group_label",
    )
    if len(group.observations) == 0:
        raise ContractError("_FitObservationGroup.observations must be non-empty")
    if len(group.observations_by_domain) == 0:
        raise ContractError("_FitObservationGroup.observations_by_domain must be non-empty")

    n_states: int | None = None
    observed_counter: Counter[tuple[str, str, str, str]] = Counter()
    observed_domain_counter: Counter[str] = Counter()
    for observation in group.observations:
        validate_fov_observation(observation)
        if observation.timepoint != group_label:
            raise ContractError(
                "All _FitObservationGroup.observations must share the declared group_label"
            )
        if observation.domain_label is None:
            raise ContractError("_FitObservationGroup observations must declare domain_label")

        observed_counter[_observation_key(observation)] += 1
        observed_domain_counter[str(observation.domain_label)] += 1
        current_n_states = int(
            np.asarray(observation.community_composition, dtype=float).shape[0]
        )
        if n_states is None:
            n_states = current_n_states
        elif current_n_states != n_states:
            raise ContractError(
                "_FitObservationGroup observations must share one K-state axis size"
            )

    grouped_counter: Counter[tuple[str, str, str, str]] = Counter()
    grouped_domain_counter: Counter[str] = Counter()
    for domain_label, domain_observations in group.observations_by_domain.items():
        normalized_domain = _require_nonempty_identifier(
            domain_label,
            field_name="domain_label",
        )
        if len(domain_observations) == 0:
            raise ContractError(
                "_FitObservationGroup.observations_by_domain entries must be non-empty"
            )
        for observation in domain_observations:
            validate_fov_observation(observation)
            if observation.timepoint != group_label:
                raise ContractError(
                    "Domain-grouped observations must share the _FitObservationGroup.group_label"
                )
            if observation.domain_label != normalized_domain:
                raise ContractError(
                    "_FitObservationGroup.observations_by_domain keys must match "
                    "observation.domain_label"
                )
            grouped_counter[_observation_key(observation)] += 1
            grouped_domain_counter[normalized_domain] += 1

    if grouped_counter != observed_counter:
        raise ContractError(
            "_FitObservationGroup.observations_by_domain must partition the declared observations"
        )
    if grouped_domain_counter != observed_domain_counter:
        raise ContractError(
            "_FitObservationGroup.observations_by_domain counts must match observed domain counts"
        )


def _validate_patient_fit_input(patient_input: _PatientFitInput) -> None:
    """Validate one canonical per-patient fit input bundle."""
    patient_id = _require_nonempty_identifier(
        patient_input.patient_id,
        field_name="patient_id",
    )
    if patient_input.mass_mode != "uniform":
        raise ContractError("_PatientFitInput.mass_mode must be 'uniform' in the current pass")
    if len(patient_input.groups) == 0:
        raise ContractError("_PatientFitInput.groups must be non-empty")

    expected_group_labels = tuple(group.group_label for group in patient_input.groups)
    if patient_input.ordered_group_labels != expected_group_labels:
        raise ContractError(
            "_PatientFitInput.ordered_group_labels must align with _PatientFitInput.groups"
        )
    if len(set(patient_input.ordered_group_labels)) != len(patient_input.ordered_group_labels):
        raise ContractError("_PatientFitInput.ordered_group_labels must not contain duplicates")

    counts_by_group: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    counts_by_group_and_domain: dict[str, dict[str, int]] = {}
    n_states: int | None = None
    for group in patient_input.groups:
        _validate_fit_observation_group(group)
        counts_by_group[group.group_label] = len(group.observations)
        group_domain_counts: dict[str, int] = {}

        for observation in group.observations:
            if observation.patient_id != patient_id:
                raise ContractError("_PatientFitInput observations must belong to one patient_id")
            if observation.mass_mode != patient_input.mass_mode:
                raise ContractError(
                    "_PatientFitInput observations must share _PatientFitInput.mass_mode"
                )

            current_n_states = int(
                np.asarray(observation.community_composition, dtype=float).shape[0]
            )
            if n_states is None:
                n_states = current_n_states
            elif current_n_states != n_states:
                raise ContractError("_PatientFitInput observations must share one K-state axis size")

            domain_label = str(observation.domain_label)
            counts_by_domain[domain_label] = counts_by_domain.get(domain_label, 0) + 1
            group_domain_counts[domain_label] = group_domain_counts.get(domain_label, 0) + 1

        counts_by_group_and_domain[group.group_label] = group_domain_counts

    if dict(patient_input.n_observations_by_group) != counts_by_group:
        raise ContractError(
            "_PatientFitInput.n_observations_by_group does not match grouped observations"
        )
    if dict(patient_input.n_observations_by_domain) != counts_by_domain:
        raise ContractError(
            "_PatientFitInput.n_observations_by_domain does not match grouped observations"
        )
    if (
        _normalize_nested_counts(patient_input.n_observations_by_group_and_domain)
        != counts_by_group_and_domain
    ):
        raise ContractError(
            "_PatientFitInput.n_observations_by_group_and_domain does not match grouped observations"
        )

    if n_states is None:
        raise ContractError("_PatientFitInput must contain at least one observation")

    if patient_input.state_basis is not None:
        validate_state_basis(patient_input.state_basis)
        if patient_input.state_basis.n_states != n_states:
            raise ContractError(
                "_PatientFitInput.state_basis must align to the observation shared K-state axis"
            )

    if patient_input.geometry is not None:
        geometry_shape = np.asarray(patient_input.geometry.cost_matrix, dtype=float).shape
        if len(geometry_shape) != 2 or geometry_shape[0] != geometry_shape[1]:
            raise ContractError("_PatientFitInput.geometry.cost_matrix must be square")
        if geometry_shape[0] != n_states:
            raise ContractError(
                "_PatientFitInput.geometry must align to the observation shared K-state axis"
            )
        if len(patient_input.geometry.state_ids) != n_states:
            raise ContractError(
                "_PatientFitInput.geometry.state_ids must align to the shared K-state axis"
            )

    if (
        patient_input.state_basis is not None
        and patient_input.geometry is not None
        and patient_input.state_basis.resolved_state_ids != tuple(patient_input.geometry.state_ids)
    ):
        raise ContractError(
            "_PatientFitInput.state_basis and geometry must share the same declared state_ids"
        )


def _build_patient_fit_inputs(
    observations: tuple[FovObservation, ...] | list[FovObservation],
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    timepoint_order: tuple[str, ...] = (),
) -> tuple[_PatientFitInput, ...]:
    """Group observations into canonical per-patient fit input bundles."""
    observation_sequence = tuple(observations)
    if len(observation_sequence) == 0:
        raise ContractError("observations must contain at least one FovObservation")

    patient_order: list[str] = []
    observed_group_order: dict[str, list[str]] = {}
    patient_groups: dict[str, dict[str, list[FovObservation]]] = {}

    for observation in observation_sequence:
        validate_fov_observation(observation)
        patient_id = _require_nonempty_identifier(observation.patient_id, field_name="patient_id")
        group_label = _require_nonempty_identifier(observation.timepoint, field_name="timepoint")
        _require_nonempty_identifier(observation.fov_id, field_name="fov_id")
        if observation.domain_label is None:
            raise ContractError("All fit observations must declare domain_label")

        if patient_id not in patient_groups:
            patient_order.append(patient_id)
            patient_groups[patient_id] = {}
            observed_group_order[patient_id] = []
        if group_label not in patient_groups[patient_id]:
            patient_groups[patient_id][group_label] = []
            observed_group_order[patient_id].append(group_label)
        patient_groups[patient_id][group_label].append(observation)

    patient_fit_inputs: list[_PatientFitInput] = []
    for patient_id in patient_order:
        observed_labels = tuple(observed_group_order[patient_id])
        ordered_labels = _resolve_ordered_group_labels(
            observed_labels,
            declared_order=tuple(timepoint_order),
        )

        groups: list[_FitObservationGroup] = []
        counts_by_group: dict[str, int] = {}
        counts_by_domain: dict[str, int] = {}
        counts_by_group_and_domain: dict[str, dict[str, int]] = {}
        for group_label in ordered_labels:
            group_observations = tuple(patient_groups[patient_id][group_label])
            observations_by_domain_lists: dict[str, list[FovObservation]] = {}
            for observation in group_observations:
                domain_label = str(observation.domain_label)
                observations_by_domain_lists.setdefault(domain_label, []).append(observation)
                counts_by_domain[domain_label] = counts_by_domain.get(domain_label, 0) + 1

            observations_by_domain = {
                domain_label: tuple(domain_observations)
                for domain_label, domain_observations in observations_by_domain_lists.items()
            }
            groups.append(
                _FitObservationGroup(
                    group_label=group_label,
                    observations=group_observations,
                    observations_by_domain=observations_by_domain,
                )
            )
            counts_by_group[group_label] = len(group_observations)
            counts_by_group_and_domain[group_label] = {
                domain_label: len(domain_observations)
                for domain_label, domain_observations in observations_by_domain.items()
            }

        patient_fit_inputs.append(
            _PatientFitInput(
                patient_id=patient_id,
                ordered_group_labels=ordered_labels,
                groups=tuple(groups),
                n_observations_by_group=counts_by_group,
                n_observations_by_domain=counts_by_domain,
                n_observations_by_group_and_domain=counts_by_group_and_domain,
                state_basis=state_basis,
                geometry=geometry,
                mass_mode="uniform",
                metadata={
                    "grouping_axis": "timepoint",
                    "declared_timepoint_order": tuple(timepoint_order),
                },
            )
        )

    return tuple(patient_fit_inputs)
