"""Validation helpers composed from canonical STRIDE modules."""
from __future__ import annotations

from stride.basis import StateBasis, validate_state_basis
from stride.data.longitudinal import validate_longitudinal_adata
from stride.errors import ContractError
from stride.latent.operators import PatientRelation, validate_patient_relation
from stride.latent.recurrence import RecurrenceResult, validate_recurrence_inputs
from stride.observation.contracts import FovObservation, validate_fov_observation
from stride.observation.validation import validate_observation_match_inputs
from stride.outputs.diagnostics import reject_pathological_relation


def validate_patient_relation_arrays(
    A: object,
    d: object,
    e: object,
    *,
    mu_minus: object | None = None,
    mu_plus: object | None = None,
    state_ids: tuple[int, ...] | None = None,
) -> None:
    """Validate a patient-level `(A_p, d_p, e_p)` payload."""
    validate_patient_relation(
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        state_ids=state_ids,
    )


def validate_patient_relation_object(relation: PatientRelation) -> None:
    """Validate one typed patient-level relation object."""
    validate_patient_relation(
        A=relation.A,
        d=relation.d,
        e=relation.e,
        mu_minus=relation.mu_minus,
        mu_plus=relation.mu_plus,
        state_ids=relation.state_ids,
    )


def validate_recurrence_family_arrays(template_A: object, template_d: object, template_e: object) -> None:
    """Validate one recurrence-family template on the shared state axis."""
    validate_patient_relation(A=template_A, d=template_d, e=template_e)


def validate_state_basis_object(state_basis: StateBasis) -> None:
    """Validate one typed shared-state basis object."""
    validate_state_basis(state_basis)


def validate_fov_observation_object(observation: FovObservation) -> None:
    """Validate one typed observation-layer FOV object."""
    validate_fov_observation(observation)


def validate_recurrence_result(result: RecurrenceResult) -> None:
    """Validate a typed recurrence result and its family templates."""
    for family in result.families:
        validate_patient_relation(A=family.template_A, d=family.template_d, e=family.template_e)


def detect_relation_pathologies(relation: PatientRelation) -> tuple[bool, tuple[object, ...]]:
    """Detect whether a patient relation should be rejected on pathology grounds."""
    return reject_pathological_relation(relation)


__all__ = [
    "ContractError",
    "detect_relation_pathologies",
    "validate_fov_observation_object",
    "validate_longitudinal_adata",
    "validate_observation_match_inputs",
    "validate_patient_relation_arrays",
    "validate_patient_relation_object",
    "validate_recurrence_family_arrays",
    "validate_recurrence_inputs",
    "validate_recurrence_result",
    "validate_state_basis_object",
]
