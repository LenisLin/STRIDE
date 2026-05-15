"""Private validators for STRIDE fit result containers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..errors import ContractError
from ..latent.operators import validate_cohort_relation, validate_patient_relation
from .provenance import STRIDEFitProvenance, validate_stride_fit_provenance

_ALLOWED_FIT_STATUSES: tuple[str, ...] = ("ok", "deferred", "failed")
_ALLOWED_IMPLEMENTATION_TIERS: tuple[str, ...] = ("canonical_full",)
_KNOWN_RELATION_AUXILIARY_ARRAY_FIELDS: tuple[str, ...] = (
    "matched_transition_burden",
    "raw_matched_transition_burden",
    "source_unmatched_burden",
    "target_unmatched_burden",
)
_NON_OK_STATUS_LABELS = frozenset({"deferred", "failed"})


def _require_nonempty_message(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string when provided")
    return normalized


def _validate_known_relation_auxiliary_arrays(result: PatientRelationResult) -> None:
    if not result.auxiliary:
        return

    if result.fit_status != "ok":
        for field_name in _KNOWN_RELATION_AUXILIARY_ARRAY_FIELDS:
            if field_name in result.auxiliary:
                raise ContractError(
                    "Non-ok PatientRelationResult objects must not carry realized auxiliary relation arrays"
                )
        return

    if result.A is None or result.d is None or result.e is None:
        raise ContractError("Known relation auxiliary validation requires realized A, d, and e")

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
                f"PatientRelationResult.auxiliary[{field_name!r}] must have shape {expected_shape}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ContractError(
                f"PatientRelationResult.auxiliary[{field_name!r}] must be finite"
            )
        if (array < 0.0).any():
            raise ContractError(
                f"PatientRelationResult.auxiliary[{field_name!r}] must be non-negative"
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
            "Known burden-scale auxiliary arrays require mu_minus on ok PatientRelationResult objects"
        )

    if (
        matched_transition_burden is not None
        and raw_matched_transition_burden is not None
        and np.any(matched_transition_burden > raw_matched_transition_burden + 1e-8)
    ):
        raise ContractError(
            "PatientRelationResult.auxiliary['matched_transition_burden'] must not exceed "
            "PatientRelationResult.auxiliary['raw_matched_transition_burden'] elementwise"
        )

    if (
        matched_transition_burden is not None
        and mu_minus is not None
        and np.any(np.sum(matched_transition_burden, axis=1, dtype=float) > mu_minus + 1e-8)
    ):
        raise ContractError(
            "PatientRelationResult.auxiliary['matched_transition_burden'] row sums must not exceed mu_minus"
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
                "PatientRelationResult.auxiliary['matched_transition_burden'] must agree with "
                "A on the realized operator scale implied by mu_minus"
            )

    if (
        source_unmatched_burden is not None
        and mu_minus is not None
        and np.any(source_unmatched_burden > mu_minus + 1e-8)
    ):
        raise ContractError(
            "PatientRelationResult.auxiliary['source_unmatched_burden'] must not exceed mu_minus"
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
                "PatientRelationResult.auxiliary['target_unmatched_burden'] must agree with e "
                "under the realized emergence scaling"
            )


def _validate_patient_relation_status_metadata(result: PatientRelationResult) -> None:
    if result.audit is not None and str(result.audit.patient_id) != str(result.patient_id):
        raise ContractError("PatientRelationResult.audit.patient_id must align with patient_id")

    if result.audit is not None:
        relation_status = str(result.audit.relation_status)
        observation_status = (
            None
            if result.audit.observation_fit_status is None
            else str(result.audit.observation_fit_status)
        )
        if result.fit_status == "ok":
            if relation_status in _NON_OK_STATUS_LABELS:
                raise ContractError(
                    "fit_status='ok' PatientRelationResult must not carry "
                    "audit.relation_status deferred/failed drift"
                )
            if observation_status in _NON_OK_STATUS_LABELS:
                raise ContractError(
                    "fit_status='ok' PatientRelationResult must not carry "
                    "audit.observation_fit_status deferred/failed drift"
                )
        else:
            if relation_status == "ok":
                raise ContractError(
                    "Non-ok PatientRelationResult must not carry audit.relation_status='ok'"
                )
            if observation_status == "ok":
                raise ContractError(
                    "Non-ok PatientRelationResult must not carry "
                    "audit.observation_fit_status='ok'"
                )

    if result.fit_status == "deferred":
        if "defer_reason" in result.diagnostics:
            _require_nonempty_message(
                result.diagnostics["defer_reason"],
                field_name="PatientRelationResult.diagnostics['defer_reason']",
            )
        if result.audit is not None and "defer_reason" in result.audit.metadata:
            _require_nonempty_message(
                result.audit.metadata["defer_reason"],
                field_name="PatientRelationResult.audit.metadata['defer_reason']",
            )

    if result.fit_status == "failed" and "defer_reason" in result.diagnostics:
        raise ContractError("Failed PatientRelationResult objects must not report a defer_reason")

def validate_patient_relation_result(result: PatientRelationResult) -> None:
    """Validate one canonical patient relation output payload."""
    patient_id = str(result.patient_id).strip()
    if patient_id == "":
        raise ContractError("PatientRelationResult.patient_id must be a non-empty string")
    if result.fit_status not in _ALLOWED_FIT_STATUSES:
        raise ContractError(
            "PatientRelationResult.fit_status must be one of "
            f"{_ALLOWED_FIT_STATUSES}, got {result.fit_status!r}"
        )
    if result.implementation_tier not in _ALLOWED_IMPLEMENTATION_TIERS:
        raise ContractError(
            "PatientRelationResult.implementation_tier must be one of "
            f"{_ALLOWED_IMPLEMENTATION_TIERS}, got {result.implementation_tier!r}"
        )

    has_core_array = any(array is not None for array in (result.A, result.d, result.e))
    has_all_core_arrays = all(array is not None for array in (result.A, result.d, result.e))
    if has_core_array and not has_all_core_arrays:
        raise ContractError("PatientRelationResult must provide A, d, and e together")

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
            raise ContractError("Non-ok PatientRelationResult objects must not carry relation arrays")
    if result.objective is not None and result.fit_status != "ok":
        raise ContractError("Only fit_status='ok' PatientRelationResult objects may carry an objective")
    _validate_known_relation_auxiliary_arrays(result)
    _validate_patient_relation_status_metadata(result)


def _stride_fit_provenance_payload(
    provenance: STRIDEFitProvenance | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(provenance, STRIDEFitProvenance):
        return provenance.to_dict()
    return provenance


def _require_positive_int_metadata(mapping: Mapping[str, Any], field_name: str) -> int:
    if field_name not in mapping:
        raise ContractError(f"STRIDEFitResult.metadata[{field_name!r}] is required when provenance is present")
    value = mapping[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"STRIDEFitResult.metadata[{field_name!r}] must be a positive integer")
    if int(value) <= 0:
        raise ContractError(f"STRIDEFitResult.metadata[{field_name!r}] must be positive")
    return int(value)


def _validate_stride_fit_result_provenance_metadata_count(result: STRIDEFitResult) -> None:
    if result.provenance is None:
        return
    if "n_evidence_blocks" not in result.metadata:
        return

    payload = _stride_fit_provenance_payload(result.provenance)
    comparison_plan = dict(payload["observation_comparison_plan"])
    provenance_count = int(comparison_plan["n_evidence_blocks"])
    metadata_count = _require_positive_int_metadata(result.metadata, "n_evidence_blocks")
    if provenance_count != metadata_count:
        raise ContractError(
            "STRIDEFitResult.provenance.observation_comparison_plan.n_evidence_blocks "
            "must match STRIDEFitResult.metadata['n_evidence_blocks']"
        )


def _validate_stride_fit_result_provenance_coherence(result: STRIDEFitResult) -> None:
    if result.provenance is None:
        return

    payload = _stride_fit_provenance_payload(result.provenance)
    initialization = dict(payload["objective_scale_initialization"])
    provenance_k = int(initialization["K"])
    patient_relation_dimensions = {
        int(np.asarray(patient_result.A, dtype=float).shape[0])
        for patient_result in result.patient_results
        if patient_result.fit_status == "ok"
    }
    if patient_relation_dimensions != {provenance_k}:
        raise ContractError(
            "STRIDEFitResult.provenance.objective_scale_initialization.K must match patient relation state dimension"
        )

    recurrence_payload = dict(payload["recurrence"])
    provenance_support = int(recurrence_payload["support_n_patients"])
    used_patient_count = len(result.cohort_relation.support_patient_ids)
    if provenance_support != used_patient_count:
        raise ContractError(
            "STRIDEFitResult.provenance.recurrence.support_n_patients must match "
            "cohort_relation.support_patient_ids"
        )
    cohort_dispersion = result.cohort_relation.dispersion
    if cohort_dispersion is None:
        raise ContractError(
            "STRIDEFitResult.cohort_relation.dispersion "
            "is required when compact provenance is present"
        )
    provenance_dispersion = float(recurrence_payload["dispersion"])
    if not np.isclose(
        provenance_dispersion,
        float(cohort_dispersion),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ContractError(
            "STRIDEFitResult.provenance.recurrence.dispersion must match "
            "cohort_relation.dispersion"
        )

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
    validate_cohort_relation(result.cohort_relation)
    if result.cohort_relation.fit_status == "ok" and tuple(result.cohort_relation.support_patient_ids) != output_patient_ids:
        raise ContractError(
            "ok STRIDEFitResult.cohort_relation.support_patient_ids must align with patient_results"
        )

    for patient_result in result.patient_results:
        validate_patient_relation_result(patient_result)

    patient_status_counts: dict[str, int] = {}
    for patient_result in result.patient_results:
        patient_status_counts[patient_result.fit_status] = (
            patient_status_counts.get(patient_result.fit_status, 0) + 1
        )

    if result.fit_status == "ok":
        if any(patient_result.fit_status != "ok" for patient_result in result.patient_results):
            raise ContractError("fit_status='ok' requires all patient_results to be ok")
        if result.cohort_relation.fit_status != "ok":
            raise ContractError("fit_status='ok' requires cohort_relation.fit_status='ok'")
    if result.fit_status == "failed" and any(
        patient_result.fit_status == "ok" for patient_result in result.patient_results
    ):
        raise ContractError("fit_status='failed' must not contain ok patient_results")
    if result.objective is not None and result.fit_status == "failed":
        raise ContractError("fit_status='failed' STRIDEFitResult objects must not carry an objective")
    if result.provenance is not None:
        validate_stride_fit_provenance(result.provenance)
        _validate_stride_fit_result_provenance_metadata_count(result)
        if result.fit_status != "ok":
            raise ContractError(
                "Only fit_status='ok' STRIDEFitResult objects may carry "
                "compact successful-fit provenance"
            )
    if (
        result.fit_status == "ok"
        and result.implementation_tier == "canonical_full"
        and result.provenance is None
    ):
        raise ContractError(
            "STRIDEFitResult fit_status='ok' and implementation_tier='canonical_full' "
            "requires compact successful-fit provenance"
        )
    if result.fit_status == "ok" and result.implementation_tier == "canonical_full":
        _validate_stride_fit_result_provenance_coherence(result)

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

__all__ = [
    "validate_patient_relation_result",
    "validate_stride_fit_result",
]
