"""Public summary dispatchers for stable STRIDE output objects."""
from __future__ import annotations

from ..latent.operators import CohortRelation, PatientRelation
from ..outputs.fit_result import PatientRelationResult, STRIDEFitResult
from ..outputs.summaries import (
    summarize_cohort_relation,
    summarize_patient_relation_result,
    summarize_patient_relation,
    summarize_stride_fit_result,
)


def summarize_fit(payload: object) -> object:
    """Return a compact summary for output objects with a stable summary surface."""
    if isinstance(payload, PatientRelation):
        return summarize_patient_relation(payload)
    if isinstance(payload, PatientRelationResult):
        return summarize_patient_relation_result(payload)
    if isinstance(payload, STRIDEFitResult):
        return summarize_stride_fit_result(payload)
    if isinstance(payload, CohortRelation):
        return summarize_cohort_relation(payload)
    return payload


__all__ = ["summarize_fit"]
