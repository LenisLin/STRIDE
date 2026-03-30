"""Latent relation and recurrence surfaces for STRIDE."""
from __future__ import annotations

from .emergence import EmergenceComponent
from .operators import (
    ContinuityOperator,
    DepletionComponent,
    PatientRelation,
    PatientRelationAudit,
    initialize_patient_relation,
    validate_patient_relation,
)
from .recurrence import (
    PatientRecurrenceEmbedding,
    RecurrenceConfig,
    RecurrenceFamily,
    RecurrenceParameters,
    RecurrenceResult,
    build_recurrence_result,
    estimate_recurrence,
    summarize_recurrence_support,
    validate_recurrence_inputs,
)

__all__ = [
    "ContinuityOperator",
    "DepletionComponent",
    "EmergenceComponent",
    "PatientRecurrenceEmbedding",
    "PatientRelation",
    "PatientRelationAudit",
    "RecurrenceConfig",
    "RecurrenceFamily",
    "RecurrenceParameters",
    "RecurrenceResult",
    "build_recurrence_result",
    "estimate_recurrence",
    "initialize_patient_relation",
    "summarize_recurrence_support",
    "validate_patient_relation",
    "validate_recurrence_inputs",
]
