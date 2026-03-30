"""Transition wrapper for canonical STRIDE recurrence interfaces."""
from __future__ import annotations

from stride.latent.recurrence import (
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
