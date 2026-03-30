"""Public summary dispatchers for stable STRIDE output objects."""
from __future__ import annotations

from ..latent.operators import PatientRelation
from ..latent.recurrence import RecurrenceResult
from ..outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from ..outputs.summaries import (
    summarize_patient_bridge_result,
    summarize_patient_relation,
    summarize_recurrence_support,
    summarize_stride_fit_result,
)


def summarize_fit(payload: object) -> object:
    """Return a compact summary for output objects with a stable summary surface."""
    if isinstance(payload, PatientRelation):
        return summarize_patient_relation(payload)
    if isinstance(payload, PatientBridgeResult):
        return summarize_patient_bridge_result(payload)
    if isinstance(payload, STRIDEFitResult):
        return summarize_stride_fit_result(payload)
    if isinstance(payload, RecurrenceResult):
        return summarize_recurrence_support(payload)
    return payload


__all__ = ["summarize_fit"]
