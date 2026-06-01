"""Canonical result containers for STRIDE relation and cohort fit surfaces."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..latent.operators import (
    CohortRelation,
    PatientRelation,
    PatientRelationAudit,
    initialize_patient_relation,
)
from ._fit_result_validate import validate_patient_relation_result, validate_stride_fit_result
from .provenance import STRIDEFitProvenance
from .uncertainty import STRIDEBootstrapUncertaintyResult

if TYPE_CHECKING:
    from ..workflows._fit_inputs import _PatientFitInput


@dataclass(frozen=True)
class PatientRelationResult:
    """Per-patient relation output contract centered on ``A``, ``d``, and ``e``."""

    patient_id: str
    fit_status: str
    A: object | None = None
    d: object | None = None
    e: object | None = None
    mu_minus: object | None = None
    mu_plus: object | None = None
    state_ids: tuple[int, ...] | None = None
    audit: PatientRelationAudit | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    implementation_tier: str = "canonical_full"
    objective: Any | None = None

    def __post_init__(self) -> None:
        validate_patient_relation_result(self)

    @property
    def is_ok(self) -> bool:
        """Return whether the relation result carries a validated patient relation."""
        return self.fit_status == "ok"

    @property
    def is_deferred(self) -> bool:
        """Return whether the relation estimator remains intentionally deferred."""
        return self.fit_status == "deferred"

    @property
    def is_failed(self) -> bool:
        """Return whether relation fitting failed without emitting model arrays."""
        return self.fit_status == "failed"

    @property
    def is_canonical_full(self) -> bool:
        """Return whether the result comes from the canonical full-method path."""
        return self.implementation_tier == "canonical_full"

    @property
    def relation(self) -> PatientRelation | None:
        """Return a validated model-layer patient relation when arrays are available."""
        if self.A is None or self.d is None or self.e is None:
            return None
        return initialize_patient_relation(
            patient_id=self.patient_id,
            A=self.A,
            d=self.d,
            e=self.e,
            mu_minus=self.mu_minus,
            mu_plus=self.mu_plus,
            state_ids=self.state_ids,
            audit=self.audit,
            metadata=dict(self.auxiliary),
        )


@dataclass(frozen=True)
class STRIDEFitResult:
    """Canonical cohort-wide fit bundle for the STRIDE fit path."""

    patient_inputs: tuple[_PatientFitInput, ...]
    patient_results: tuple[PatientRelationResult, ...]
    cohort_relation: CohortRelation
    fit_status: str
    implementation_tier: str = "canonical_full"
    objective: Any | None = None
    provenance: STRIDEFitProvenance | Mapping[str, Any] | None = None
    summaries: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: STRIDEBootstrapUncertaintyResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_stride_fit_result(self)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """Return the ordered patient identifiers for the fit bundle."""
        return tuple(patient_result.patient_id for patient_result in self.patient_results)


__all__ = [
    "PatientRelationResult",
    "STRIDEFitResult",
    "validate_patient_relation_result",
    "validate_stride_fit_result",
]
