"""Private payload helpers for STRIDE fit provenance."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ContractError

STRIDE_FIT_PROVENANCE_SCHEMA_VERSION = "stride_fit_provenance.v1"

_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "provenance_schema_version",
    "objective_contract_version",
    "random_seed",
    "objective_constants",
    "objective_scale_initialization",
    "optimizer_start_initialization",
    "loss",
    "e_bounds",
    "post_reconstruction_form",
    "observation_comparison_plan",
    "observation_discrepancy",
    "state_geometry",
    "optimizer",
    "recurrence",
    "detailed_optimizer_trace",
)
_OPTIONAL_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "objective_sensitivity",
    "optimizer_trace_ref",
)
_OPTIONAL_ABLATION_FIELDS: tuple[str, ...] = (
    "ablation_mode",
    "ablation_term_handling",
    "ablation_denominator_policy",
)
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    (*_REQUIRED_TOP_LEVEL_FIELDS, *_OPTIONAL_DIAGNOSTIC_FIELDS, *_OPTIONAL_ABLATION_FIELDS)
)
_LOSS_COMPONENTS: tuple[str, ...] = (
    "obs",
    "open",
    "geometry",
    "subbag_consistency",
    "recurrence",
)
_FORBIDDEN_PROVENANCE_FIELDS = frozenset(
    {
        "fit_status",
        "status",
        "status_counts",
        "fit_status_counts",
        "patient_status",
        "patient_statuses",
        "patient_fit_status",
        "patient_status_counts",
        "recurrence_status",
        "recurrence_fit_status",
        "evidence_block_status",
        "evidence_block_statuses",
        "evidence_block_status_counts",
        "per_patient_status",
        "per_evidence_block_status",
        "failure_reason",
        "optimizer_failure_reason",
        "defer_reason",
        "deferred_reason",
        "per_patient_records",
        "patient_records",
        "patient_results",
        "per_evidence_block_records",
        "evidence_block_records",
    }
)

def _payload_mapping(provenance: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(provenance, "to_dict") and provenance.__class__.__name__ == "STRIDEFitProvenance":
        return provenance.to_dict()
    if isinstance(provenance, Mapping):
        return provenance
    raise ContractError("STRIDE fit provenance payload must be a mapping")


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _copy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _copy_value(value) for key, value in mapping.items()}

__all__ = [
    "STRIDE_FIT_PROVENANCE_SCHEMA_VERSION",
    "_ALLOWED_TOP_LEVEL_FIELDS",
    "_FORBIDDEN_PROVENANCE_FIELDS",
    "_LOSS_COMPONENTS",
    "_OPTIONAL_ABLATION_FIELDS",
    "_OPTIONAL_DIAGNOSTIC_FIELDS",
    "_REQUIRED_TOP_LEVEL_FIELDS",
    "_copy_mapping",
    "_copy_value",
    "_payload_mapping",
]
