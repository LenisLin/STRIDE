"""Diagnostic helpers for detecting pathological patient relations."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..latent.operators import PatientRelation
from ..optimize.stopping import detect_pathologies


@dataclass(frozen=True)
class PathologyDiagnostic:
    """Named diagnostic emitted by the output validation/support layer."""

    name: str
    triggered: bool
    detail: str
    reject: bool


def reject_pathological_relation(relation: PatientRelation) -> tuple[bool, tuple[PathologyDiagnostic, ...]]:
    """Return whether a patient relation should be rejected on pathology grounds."""
    checks = detect_pathologies(relation.A, relation.d, relation.e)
    diagnostics = tuple(
        PathologyDiagnostic(
            name=check.name,
            triggered=check.triggered,
            detail=check.detail,
            reject=check.triggered,
        )
        for check in checks
    )
    return any(diagnostic.reject for diagnostic in diagnostics), diagnostics


def audit_failure_modes(relations: Sequence[PatientRelation]) -> tuple[tuple[str, tuple[PathologyDiagnostic, ...]], ...]:
    """Collect pathology diagnostics across multiple patient relations."""
    return tuple((relation.patient_id, reject_pathological_relation(relation)[1]) for relation in relations)


__all__ = ["PathologyDiagnostic", "audit_failure_modes", "reject_pathological_relation"]
