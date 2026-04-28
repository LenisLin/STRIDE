"""Canonical result containers for STRIDE bridge and cohort fit surfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..errors import ContractError
from ..objectives import LossBreakdown
from ..latent.operators import (
    PatientRelation,
    PatientRelationAudit,
    initialize_patient_relation,
    validate_patient_relation,
)
from ..latent.recurrence import RecurrenceResult
from .uncertainty import STRIDEBootstrapUncertaintyResult

if TYPE_CHECKING:
    from ..workflows.fit_stride import PatientBridgeInput


_ALLOWED_FIT_STATUSES: tuple[str, ...] = ("ok", "deferred", "failed")
_ALLOWED_IMPLEMENTATION_TIERS: tuple[str, ...] = (
    "canonical_full",
    "approximate_proxy",
    "assembled_relation",
)
_KNOWN_BRIDGE_AUXILIARY_ARRAY_FIELDS: tuple[str, ...] = (
    "matched_transition_burden",
    "raw_matched_transition_burden",
    "source_unmatched_burden",
    "target_unmatched_burden",
)


def _require_nonempty_message(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string when provided")
    return normalized


def _validate_known_bridge_auxiliary_arrays(result: PatientBridgeResult) -> None:
    if not result.auxiliary:
        return

    if result.fit_status != "ok":
        for field_name in _KNOWN_BRIDGE_AUXILIARY_ARRAY_FIELDS:
            if field_name in result.auxiliary:
                raise ContractError(
                    "Non-ok PatientBridgeResult objects must not carry realized auxiliary bridge arrays"
                )
        return

    if result.A is None or result.d is None or result.e is None:
        raise ContractError("Known bridge auxiliary validation requires realized A, d, and e")

    n_states = int(np.asarray(result.A, dtype=float).shape[0])
    mu_minus = (
        np.asarray(result.mu_minus, dtype=float)
        if result.mu_minus is not None
        else None
    )

    def _validate_nonnegative_auxiliary(
        field_name: str,
        *,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if field_name not in result.auxiliary:
            return None
        array = np.asarray(result.auxiliary[field_name], dtype=float)
        if array.shape != expected_shape:
            raise ContractError(
                f"PatientBridgeResult.auxiliary[{field_name!r}] must have shape {expected_shape}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ContractError(
                f"PatientBridgeResult.auxiliary[{field_name!r}] must be finite"
            )
        if (array < 0.0).any():
            raise ContractError(
                f"PatientBridgeResult.auxiliary[{field_name!r}] must be non-negative"
            )
        return array

    matched_transition_burden = _validate_nonnegative_auxiliary(
        "matched_transition_burden",
        expected_shape=(n_states, n_states),
    )
    raw_matched_transition_burden = _validate_nonnegative_auxiliary(
        "raw_matched_transition_burden",
        expected_shape=(n_states, n_states),
    )
    source_unmatched_burden = _validate_nonnegative_auxiliary(
        "source_unmatched_burden",
        expected_shape=(n_states,),
    )
    target_unmatched_burden = _validate_nonnegative_auxiliary(
        "target_unmatched_burden",
        expected_shape=(n_states,),
    )

    if mu_minus is None and any(
        array is not None
        for array in (
            matched_transition_burden,
            raw_matched_transition_burden,
            source_unmatched_burden,
            target_unmatched_burden,
        )
    ):
        raise ContractError(
            "Known burden-scale auxiliary arrays require mu_minus on ok PatientBridgeResult objects"
        )

    if matched_transition_burden is not None and raw_matched_transition_burden is not None:
        if np.any(matched_transition_burden > raw_matched_transition_burden + 1e-8):
            raise ContractError(
                "PatientBridgeResult.auxiliary['matched_transition_burden'] must not exceed "
                "PatientBridgeResult.auxiliary['raw_matched_transition_burden'] elementwise"
            )

    if matched_transition_burden is not None and mu_minus is not None:
        if np.any(np.sum(matched_transition_burden, axis=1, dtype=float) > mu_minus + 1e-8):
            raise ContractError(
                "PatientBridgeResult.auxiliary['matched_transition_burden'] row sums must not exceed mu_minus"
            )

    if matched_transition_burden is not None and mu_minus is not None:
        expected_transition_burden = np.asarray(result.A, dtype=float) * mu_minus[:, None]
        if not np.allclose(
            matched_transition_burden,
            expected_transition_burden,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ContractError(
                "PatientBridgeResult.auxiliary['matched_transition_burden'] must agree with "
                "A on the realized operator scale implied by mu_minus"
            )

    if source_unmatched_burden is not None and mu_minus is not None:
        if np.any(source_unmatched_burden > mu_minus + 1e-8):
            raise ContractError(
                "PatientBridgeResult.auxiliary['source_unmatched_burden'] must not exceed mu_minus"
            )

    if target_unmatched_burden is not None:
        expected_target_unmatched = np.asarray(result.e, dtype=float) * float(
            np.sum(mu_minus, dtype=float)
        )
        if not np.allclose(
            target_unmatched_burden,
            expected_target_unmatched,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ContractError(
                "PatientBridgeResult.auxiliary['target_unmatched_burden'] must agree with e "
                "under the realized emergence scaling"
            )


def _validate_patient_bridge_status_metadata(result: PatientBridgeResult) -> None:
    if result.audit is not None and str(result.audit.patient_id) != str(result.patient_id):
        raise ContractError("PatientBridgeResult.audit.patient_id must align with patient_id")

    if result.fit_status == "deferred":
        if "defer_reason" in result.diagnostics:
            _require_nonempty_message(
                result.diagnostics["defer_reason"],
                field_name="PatientBridgeResult.diagnostics['defer_reason']",
            )
        if result.audit is not None and "defer_reason" in result.audit.metadata:
            _require_nonempty_message(
                result.audit.metadata["defer_reason"],
                field_name="PatientBridgeResult.audit.metadata['defer_reason']",
            )

    if result.fit_status == "failed" and "defer_reason" in result.diagnostics:
        raise ContractError("Failed PatientBridgeResult objects must not report a defer_reason")


@dataclass(frozen=True)
class PatientBridgeResult:
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
    implementation_tier: str = "approximate_proxy"
    objective: LossBreakdown | None = None

    def __post_init__(self) -> None:
        validate_patient_bridge_result(self)

    @property
    def is_ok(self) -> bool:
        """Return whether the bridge result carries a validated patient relation."""
        return self.fit_status == "ok"

    @property
    def is_deferred(self) -> bool:
        """Return whether the bridge estimator remains intentionally deferred."""
        return self.fit_status == "deferred"

    @property
    def is_failed(self) -> bool:
        """Return whether bridge fitting failed without emitting model arrays."""
        return self.fit_status == "failed"

    @property
    def is_canonical_full(self) -> bool:
        """Return whether the result comes from the canonical full-method path."""
        return self.implementation_tier == "canonical_full"

    @property
    def is_proxy_path(self) -> bool:
        """Return whether the result comes from the explicit approximate proxy path."""
        return self.implementation_tier == "approximate_proxy"

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


def validate_patient_bridge_result(result: PatientBridgeResult) -> None:
    """Validate one canonical patient bridge output payload."""
    patient_id = str(result.patient_id).strip()
    if patient_id == "":
        raise ContractError("PatientBridgeResult.patient_id must be a non-empty string")
    if result.fit_status not in _ALLOWED_FIT_STATUSES:
        raise ContractError(
            "PatientBridgeResult.fit_status must be one of "
            f"{_ALLOWED_FIT_STATUSES}, got {result.fit_status!r}"
        )
    if result.implementation_tier not in _ALLOWED_IMPLEMENTATION_TIERS:
        raise ContractError(
            "PatientBridgeResult.implementation_tier must be one of "
            f"{_ALLOWED_IMPLEMENTATION_TIERS}, got {result.implementation_tier!r}"
        )

    has_core_array = any(array is not None for array in (result.A, result.d, result.e))
    has_all_core_arrays = all(array is not None for array in (result.A, result.d, result.e))
    if has_core_array and not has_all_core_arrays:
        raise ContractError("PatientBridgeResult must provide A, d, and e together")

    has_optional_arrays = any(array is not None for array in (result.mu_minus, result.mu_plus))
    if result.fit_status == "ok":
        if not has_all_core_arrays:
            raise ContractError("fit_status='ok' requires A, d, and e")
        validate_patient_relation(
            A=result.A,
            d=result.d,
            e=result.e,
            mu_minus=result.mu_minus,
            mu_plus=result.mu_plus,
            state_ids=result.state_ids,
        )
    else:
        if has_all_core_arrays or has_optional_arrays:
            raise ContractError("Non-ok PatientBridgeResult objects must not carry bridge arrays")
    if result.objective is not None and result.fit_status != "ok":
        raise ContractError("Only fit_status='ok' PatientBridgeResult objects may carry an objective")
    _validate_known_bridge_auxiliary_arrays(result)
    _validate_patient_bridge_status_metadata(result)


@dataclass(frozen=True)
class STRIDEFitResult:
    """Canonical cohort-wide fit bundle for the deferred STRIDE fit path."""

    patient_inputs: tuple["PatientBridgeInput", ...]
    patient_results: tuple[PatientBridgeResult, ...]
    recurrence: RecurrenceResult
    fit_status: str
    implementation_tier: str = "canonical_full"
    objective: LossBreakdown | None = None
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


def validate_stride_fit_result(result: STRIDEFitResult) -> None:
    """Validate patient/cohort alignment on the canonical STRIDE fit surface."""
    if result.fit_status not in _ALLOWED_FIT_STATUSES:
        raise ContractError(
            "STRIDEFitResult.fit_status must be one of "
            f"{_ALLOWED_FIT_STATUSES}, got {result.fit_status!r}"
        )
    if result.implementation_tier not in _ALLOWED_IMPLEMENTATION_TIERS:
        raise ContractError(
            "STRIDEFitResult.implementation_tier must be one of "
            f"{_ALLOWED_IMPLEMENTATION_TIERS}, got {result.implementation_tier!r}"
        )

    input_patient_ids = tuple(patient_input.patient_id for patient_input in result.patient_inputs)
    output_patient_ids = tuple(patient_result.patient_id for patient_result in result.patient_results)

    if len(set(input_patient_ids)) != len(input_patient_ids):
        raise ContractError("STRIDEFitResult.patient_inputs must have unique patient_id values")
    if len(set(output_patient_ids)) != len(output_patient_ids):
        raise ContractError("STRIDEFitResult.patient_results must have unique patient_id values")
    if input_patient_ids != output_patient_ids:
        raise ContractError(
            "STRIDEFitResult.patient_inputs and patient_results must align by ordered patient_id"
        )
    if tuple(result.recurrence.patient_ids) != output_patient_ids:
        raise ContractError(
            "STRIDEFitResult.recurrence.patient_ids must align with patient_results"
        )
    if result.recurrence.used_patient_ids:
        invalid_used_patient_ids = tuple(
            patient_id
            for patient_id in result.recurrence.used_patient_ids
            if patient_id not in output_patient_ids
        )
        if invalid_used_patient_ids:
            raise ContractError(
                "STRIDEFitResult.recurrence.used_patient_ids must be a subset of patient_results"
            )

    for patient_result in result.patient_results:
        validate_patient_bridge_result(patient_result)

    patient_status_counts: dict[str, int] = {}
    for patient_result in result.patient_results:
        patient_status_counts[patient_result.fit_status] = (
            patient_status_counts.get(patient_result.fit_status, 0) + 1
        )

    if result.fit_status == "ok":
        if any(patient_result.fit_status != "ok" for patient_result in result.patient_results):
            raise ContractError("fit_status='ok' requires all patient_results to be ok")
        if result.recurrence.fit_status != "ok":
            raise ContractError("fit_status='ok' requires recurrence.fit_status='ok'")
    if result.fit_status == "failed" and any(
        patient_result.fit_status == "ok" for patient_result in result.patient_results
    ):
        raise ContractError("fit_status='failed' must not contain ok patient_results")
    if result.objective is not None and result.fit_status == "failed":
        raise ContractError("fit_status='failed' STRIDEFitResult objects must not carry an objective")

    if "patient_status_counts" in result.summaries:
        normalized_summary_counts = {
            str(status): int(count)
            for status, count in dict(result.summaries["patient_status_counts"]).items()
        }
        if normalized_summary_counts != patient_status_counts:
            raise ContractError(
                "STRIDEFitResult.summaries['patient_status_counts'] must match patient_results"
            )
    if "patient_status_counts" in result.diagnostics:
        normalized_diagnostic_counts = {
            str(status): int(count)
            for status, count in dict(result.diagnostics["patient_status_counts"]).items()
        }
        if normalized_diagnostic_counts != patient_status_counts:
            raise ContractError(
                "STRIDEFitResult.diagnostics['patient_status_counts'] must match patient_results"
            )

    if result.uncertainty is not None:
        if tuple(result.uncertainty.patient_ids) != output_patient_ids:
            raise ContractError(
                "STRIDEFitResult.uncertainty.patient_ids must align with patient_results"
            )
        for patient_result, patient_uncertainty in zip(
            result.patient_results,
            result.uncertainty.patient_results,
            strict=True,
        ):
            if patient_uncertainty.realized_fit_status != patient_result.fit_status:
                raise ContractError(
                    "STRIDEFitResult.uncertainty.patient_results must preserve each realized patient fit_status"
                )
            if patient_result.fit_status != "ok" and patient_uncertainty.eligible:
                raise ContractError(
                    "STRIDEFitResult.uncertainty cannot mark non-ok patient_results as eligible"
                )


# Compatibility aliases retained for transition-era imports.
PatientRelationFitResult = PatientBridgeResult
FitResult = PatientRelationFitResult


__all__ = [
    "FitResult",
    "PatientBridgeResult",
    "PatientRelationFitResult",
    "STRIDEFitResult",
    "validate_patient_bridge_result",
    "validate_stride_fit_result",
]
