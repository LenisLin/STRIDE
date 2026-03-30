"""Synthetic and semi-synthetic sanity-check interfaces for canonical STRIDE objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..patient_relation.inference import initialize_patient_relation


@dataclass(frozen=True)
class SyntheticPatientSpec:
    """Synthetic patient-level relation specification."""

    patient_id: str
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticCohortSpec:
    """Synthetic cohort-level relation specification."""

    patients: tuple[SyntheticPatientSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def generate_synthetic_relations(spec: SyntheticCohortSpec) -> tuple[object, ...]:
    """Materialize synthetic patient relations from an explicit cohort spec."""
    return tuple(
        initialize_patient_relation(
            patient_id=patient.patient_id,
            A=patient.A,
            d=patient.d,
            e=patient.e,
            metadata=patient.metadata,
        )
        for patient in spec.patients
    )


def score_relation_recovery(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute a simple cosine-style recovery score between two relation payloads."""
    ref = np.asarray(reference, dtype=float).reshape(-1)
    est = np.asarray(estimate, dtype=float).reshape(-1)
    ref_norm = float(np.linalg.norm(ref))
    est_norm = float(np.linalg.norm(est))
    if ref_norm == 0.0 or est_norm == 0.0:
        return float("nan")
    return float(np.dot(ref, est) / (ref_norm * est_norm))


__all__ = [
    "SyntheticCohortSpec",
    "SyntheticPatientSpec",
    "generate_synthetic_relations",
    "score_relation_recovery",
]

