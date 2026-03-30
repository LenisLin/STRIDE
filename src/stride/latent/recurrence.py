"""Cohort-level recurrence contracts built from STRIDE patient relations."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..errors import ContractError
from .operators import PatientRelation, validate_patient_relation


@dataclass(frozen=True)
class RecurrenceParameters:
    """Shared low-dimensional recurrence parameters across patients."""

    basis_dim: int
    loadings: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRecurrenceEmbedding:
    """Low-dimensional embedding for one patient relation."""

    patient_id: str
    coordinates: np.ndarray
    fit_status: str = "ok"


@dataclass(frozen=True)
class RecurrenceFamily:
    """One recurrence-family summary on the shared state axis."""

    family_id: str
    template_A: np.ndarray
    template_d: np.ndarray
    template_e: np.ndarray
    support_n_patients: int
    within_family_dispersion: float | None = None
    fit_status: str = "ok"
    member_patient_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecurrenceResult:
    """Container for cohort-level recurrence outputs.

    The result holds the cohort members that were analyzed plus any learned
    recurrence families, low-dimensional embeddings, and fit metadata.
    """

    patient_ids: tuple[str, ...]
    families: tuple[RecurrenceFamily, ...]
    fit_status: str
    parameters: RecurrenceParameters | None = None
    embeddings: tuple[PatientRecurrenceEmbedding, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecurrenceConfig:
    """Configuration for the current cohort-level recurrence interface."""

    recurrence_unit: str = "patient"
    min_support_n_patients: int = 1
    basis_dim: int = 2
    mode: str = "deferred"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def validate_recurrence_inputs(relations: Sequence[PatientRelation]) -> None:
    """Validate model-layer patient relations before recurrence estimation."""
    if len(relations) == 0:
        raise ContractError("relations must contain at least one patient relation")
    for relation in relations:
        validate_patient_relation(
            A=relation.A,
            d=relation.d,
            e=relation.e,
            mu_minus=relation.mu_minus,
            mu_plus=relation.mu_plus,
            state_ids=relation.state_ids,
        )


def build_recurrence_result(
    patient_ids: Sequence[str],
    families: Sequence[RecurrenceFamily] = (),
    *,
    fit_status: str = "ok",
    parameters: RecurrenceParameters | None = None,
    embeddings: Sequence[PatientRecurrenceEmbedding] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RecurrenceResult:
    """Build a recurrence result from already-assembled family-level objects."""
    for family in families:
        validate_patient_relation(A=family.template_A, d=family.template_d, e=family.template_e)
    return RecurrenceResult(
        patient_ids=tuple(str(patient_id) for patient_id in patient_ids),
        families=tuple(families),
        fit_status=str(fit_status),
        parameters=parameters,
        embeddings=tuple(embeddings),
        metadata=dict(metadata or {}),
    )


def summarize_recurrence_support(result: RecurrenceResult) -> dict[str, int]:
    """Summarize support size by recurrence family."""
    return {family.family_id: int(family.support_n_patients) for family in result.families}


def estimate_recurrence(
    relations: Sequence[PatientRelation],
    config: RecurrenceConfig | None = None,
) -> RecurrenceResult:
    """Return the current deferred result for the canonical recurrence estimator.

    Recurrence remains a model-layer concept in STRIDE, but this entrypoint is
    intentionally narrow today: it validates the cohort input, records the
    requested embedding dimensionality, and marks the estimation status as
    deferred.
    """
    validate_recurrence_inputs(relations)
    resolved_config = config or RecurrenceConfig()
    patient_ids = tuple(relation.patient_id for relation in relations)
    zero_embeddings = tuple(
        PatientRecurrenceEmbedding(
            patient_id=relation.patient_id,
            coordinates=np.full(resolved_config.basis_dim, np.nan, dtype=float),
            fit_status="deferred",
        )
        for relation in relations
    )
    return RecurrenceResult(
        patient_ids=patient_ids,
        families=(),
        fit_status="deferred",
        parameters=RecurrenceParameters(
            basis_dim=resolved_config.basis_dim,
            loadings=None,
            metadata={"mode": resolved_config.mode},
        ),
        embeddings=zero_embeddings,
        metadata={
            "mode": resolved_config.mode,
            "message": "Canonical cohort-level recurrence estimation remains deferred.",
            **dict(resolved_config.metadata),
        },
    )


__all__ = [
    "PatientRecurrenceEmbedding",
    "RecurrenceConfig",
    "RecurrenceFamily",
    "RecurrenceParameters",
    "RecurrenceResult",
    "build_recurrence_result",
    "estimate_recurrence",
    "summarize_recurrence_support",
    "validate_recurrence_inputs",
]
