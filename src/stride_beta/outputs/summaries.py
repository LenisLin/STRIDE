"""Reporting-layer summaries for STRIDE relation and fit artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..latent.operators import CohortRelation, PatientRelation
from .fit_result import PatientRelationResult, STRIDEFitResult


@dataclass(frozen=True)
class PatientRelationSummary:
    """Compact reporting-layer summary of one patient-level relation.

    This summary is intentionally lossy: it exposes stable, easy-to-compare
    totals without replacing the underlying model-layer ``PatientRelation``.
    """

    patient_id: str
    diagonal_continuity_mass: float
    offdiagonal_continuity_mass: float
    total_continuity_mass: float
    total_depletion_mass: float
    total_emergence_mass: float
    continuity_backbone_fraction: float
    open_channel_fraction: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def summarize_continuity_backbone(relation: PatientRelation) -> dict[str, float]:
    """Summarize diagonal, off-diagonal, and total continuity mass in ``A``."""
    A = np.asarray(relation.A, dtype=float)
    diagonal = float(np.trace(A))
    total = float(np.sum(A, dtype=float))
    return {
        "diagonal_continuity_mass": diagonal,
        "offdiagonal_continuity_mass": total - diagonal,
        "total_continuity_mass": total,
    }


def summarize_open_channels(relation: PatientRelation) -> dict[str, float]:
    """Summarize total depletion and emergence mass."""
    d_mass = float(np.sum(np.asarray(relation.d, dtype=float), dtype=float))
    e_mass = float(np.sum(np.asarray(relation.e, dtype=float), dtype=float))
    return {
        "total_depletion_mass": d_mass,
        "total_emergence_mass": e_mass,
    }


def summarize_patient_relation(relation: PatientRelation) -> PatientRelationSummary:
    """Build a compact summary for one model-layer patient relation."""
    backbone = summarize_continuity_backbone(relation)
    open_channels = summarize_open_channels(relation)
    total_mass = (
        backbone["total_continuity_mass"]
        + open_channels["total_depletion_mass"]
        + open_channels["total_emergence_mass"]
    )
    continuity_fraction = backbone["total_continuity_mass"] / total_mass if total_mass > 0.0 else float("nan")
    open_fraction = (
        (open_channels["total_depletion_mass"] + open_channels["total_emergence_mass"]) / total_mass
        if total_mass > 0.0
        else float("nan")
    )
    return PatientRelationSummary(
        patient_id=relation.patient_id,
        diagonal_continuity_mass=backbone["diagonal_continuity_mass"],
        offdiagonal_continuity_mass=backbone["offdiagonal_continuity_mass"],
        total_continuity_mass=backbone["total_continuity_mass"],
        total_depletion_mass=open_channels["total_depletion_mass"],
        total_emergence_mass=open_channels["total_emergence_mass"],
        continuity_backbone_fraction=float(continuity_fraction),
        open_channel_fraction=float(open_fraction),
        metadata=dict(relation.metadata),
    )


def summarize_cohort_relation(relation: CohortRelation) -> dict[str, object]:
    """Summarize one cohort common-structure relation."""
    return {
        "cohort_id": relation.cohort_id,
        "fit_status": relation.fit_status,
        "support_n_patients": len(relation.support_patient_ids),
        "support_patient_ids": tuple(relation.support_patient_ids),
        "dispersion": relation.dispersion,
        "metadata": dict(relation.metadata),
    }


def summarize_patient_relation_result(result: PatientRelationResult) -> object:
    """Summarize one patient relation result without inventing missing arrays."""
    relation = result.relation
    if relation is not None:
        return summarize_patient_relation(relation)
    return {
        "patient_id": result.patient_id,
        "fit_status": result.fit_status,
        "diagnostics": dict(result.diagnostics),
    }


def summarize_stride_fit_result(result: STRIDEFitResult) -> dict[str, object]:
    """Summarize the cohort-wide fit result with status-focused counts only."""
    patient_status_counts: dict[str, int] = {}
    for patient_result in result.patient_results:
        patient_status_counts[patient_result.fit_status] = (
            patient_status_counts.get(patient_result.fit_status, 0) + 1
        )

    return {
        "fit_status": result.fit_status,
        "implementation_tier": result.implementation_tier,
        "n_patients": len(result.patient_results),
        "patient_status_counts": patient_status_counts,
        "cohort_fit_status": result.cohort_relation.fit_status,
        "cohort_support_n_patients": len(result.cohort_relation.support_patient_ids),
        "objective_total": (result.objective.total if result.objective is not None else None),
    }


def summarize_outputs(payload: object) -> object:
    """Summarize canonical STRIDE outputs when a stable summary exists."""
    if isinstance(payload, PatientRelation):
        return summarize_patient_relation(payload)
    if isinstance(payload, PatientRelationResult):
        return summarize_patient_relation_result(payload)
    if isinstance(payload, STRIDEFitResult):
        return summarize_stride_fit_result(payload)
    if isinstance(payload, CohortRelation):
        return summarize_cohort_relation(payload)
    return payload


__all__ = [
    "PatientRelationSummary",
    "summarize_cohort_relation",
    "summarize_continuity_backbone",
    "summarize_open_channels",
    "summarize_outputs",
    "summarize_patient_relation_result",
    "summarize_patient_relation",
    "summarize_stride_fit_result",
]
