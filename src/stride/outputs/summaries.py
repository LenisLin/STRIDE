"""Reporting-layer summaries for patient relations, recurrence, and fit artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..latent.operators import PatientRelation
from ..latent.recurrence import RecurrenceResult, summarize_recurrence_support as _summarize_recurrence_support
from .fit_result import PatientBridgeResult, STRIDEFitResult


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


def summarize_recurrence_support(result: RecurrenceResult) -> dict[str, int]:
    """Thin recurrence-support summary export."""
    return _summarize_recurrence_support(result)


def summarize_patient_bridge_result(result: PatientBridgeResult) -> object:
    """Summarize one patient bridge result without inventing missing arrays."""
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
        "recurrence_fit_status": result.recurrence.fit_status,
        "n_recurrence_used_patients": len(result.recurrence.used_patient_ids),
        "objective_total": (result.objective.total if result.objective is not None else None),
    }


def summarize_outputs(payload: object) -> object:
    """Summarize canonical STRIDE outputs when a stable summary exists."""
    if isinstance(payload, PatientRelation):
        return summarize_patient_relation(payload)
    if isinstance(payload, PatientBridgeResult):
        return summarize_patient_bridge_result(payload)
    if isinstance(payload, STRIDEFitResult):
        return summarize_stride_fit_result(payload)
    if isinstance(payload, RecurrenceResult):
        return summarize_recurrence_support(payload)
    return payload


__all__ = [
    "PatientRelationSummary",
    "summarize_continuity_backbone",
    "summarize_open_channels",
    "summarize_outputs",
    "summarize_patient_bridge_result",
    "summarize_patient_relation",
    "summarize_recurrence_support",
    "summarize_stride_fit_result",
]
