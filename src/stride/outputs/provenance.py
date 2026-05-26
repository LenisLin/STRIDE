"""Compact successful-fit provenance contracts for STRIDE outputs."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import ContractError
from ._provenance_payload import (
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    _OPTIONAL_ABLATION_FIELDS,
    _OPTIONAL_DIAGNOSTIC_FIELDS,
    _copy_mapping,
    _copy_value,
)
from ._provenance_validate import (
    _coerce_e_bounds,
    _require_bool,
    _require_int_or_none,
    _require_mapping,
    validate_stride_fit_provenance,
)


@dataclass(frozen=True)
class STRIDEFitProvenance:
    """Validated compact provenance for one successful full STRIDE fit."""

    provenance_schema_version: str
    objective_contract_version: str
    random_seed: int | None
    objective_constants: Mapping[str, Any]
    objective_scale_initialization: Mapping[str, Any]
    optimizer_start_initialization: Mapping[str, Any]
    loss: Mapping[str, Any]
    e_bounds: tuple[float, float]
    post_reconstruction_form: str
    observation_comparison_plan: Mapping[str, Any]
    observation_discrepancy: Mapping[str, Any]
    state_geometry: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    recurrence: Mapping[str, Any]
    detailed_optimizer_trace: bool = False
    objective_sensitivity: Any | None = None
    optimizer_trace_ref: Any | None = None
    ablation_mode: str | None = None
    ablation_term_handling: str | None = None
    ablation_denominator_policy: str | None = None

    def __post_init__(self) -> None:
        validate_stride_fit_provenance(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact mapping with absent optional diagnostics omitted."""
        payload: dict[str, Any] = {
            "provenance_schema_version": self.provenance_schema_version,
            "objective_contract_version": self.objective_contract_version,
            "random_seed": self.random_seed,
            "objective_constants": _copy_value(self.objective_constants),
            "objective_scale_initialization": _copy_value(self.objective_scale_initialization),
            "optimizer_start_initialization": _copy_value(self.optimizer_start_initialization),
            "loss": _copy_value(self.loss),
            "e_bounds": list(self.e_bounds),
            "post_reconstruction_form": self.post_reconstruction_form,
            "observation_comparison_plan": _copy_value(self.observation_comparison_plan),
            "observation_discrepancy": _copy_value(self.observation_discrepancy),
            "state_geometry": _copy_value(self.state_geometry),
            "optimizer": _copy_value(self.optimizer),
            "recurrence": _copy_value(self.recurrence),
            "detailed_optimizer_trace": self.detailed_optimizer_trace,
        }
        for field_name in (*_OPTIONAL_DIAGNOSTIC_FIELDS, *_OPTIONAL_ABLATION_FIELDS):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = _copy_value(value)
        return payload


def build_stride_fit_provenance(
    payload: Mapping[str, Any] | STRIDEFitProvenance,
    **overrides: Any,
) -> STRIDEFitProvenance:
    """Construct a validated STRIDE compact provenance object from a mapping."""
    if isinstance(payload, STRIDEFitProvenance):
        raw_payload = payload.to_dict()
    elif isinstance(payload, Mapping):
        raw_payload = dict(payload)
    else:
        raise ContractError("STRIDE fit provenance payload must be a mapping")
    raw_payload.update(overrides)
    validate_stride_fit_provenance(raw_payload)

    optional_values = {
        field_name: _copy_value(raw_payload[field_name])
        for field_name in (*_OPTIONAL_DIAGNOSTIC_FIELDS, *_OPTIONAL_ABLATION_FIELDS)
        if field_name in raw_payload
    }
    return STRIDEFitProvenance(
        provenance_schema_version=str(raw_payload["provenance_schema_version"]),
        objective_contract_version=str(raw_payload["objective_contract_version"]),
        random_seed=_require_int_or_none(raw_payload["random_seed"], "random_seed"),
        objective_constants=_copy_mapping(
            _require_mapping(raw_payload["objective_constants"], "objective_constants")
        ),
        objective_scale_initialization=_copy_mapping(
            _require_mapping(raw_payload["objective_scale_initialization"], "objective_scale_initialization")
        ),
        optimizer_start_initialization=_copy_mapping(
            _require_mapping(raw_payload["optimizer_start_initialization"], "optimizer_start_initialization")
        ),
        loss=_copy_mapping(_require_mapping(raw_payload["loss"], "loss")),
        e_bounds=_coerce_e_bounds(raw_payload["e_bounds"]),
        post_reconstruction_form=str(raw_payload["post_reconstruction_form"]),
        observation_comparison_plan=_copy_mapping(
            _require_mapping(
                raw_payload["observation_comparison_plan"],
                "observation_comparison_plan",
            )
        ),
        observation_discrepancy=_copy_mapping(
            _require_mapping(
                raw_payload["observation_discrepancy"],
                "observation_discrepancy",
            )
        ),
        state_geometry=_copy_mapping(
            _require_mapping(raw_payload["state_geometry"], "state_geometry")
        ),
        optimizer=_copy_mapping(_require_mapping(raw_payload["optimizer"], "optimizer")),
        recurrence=_copy_mapping(_require_mapping(raw_payload["recurrence"], "recurrence")),
        detailed_optimizer_trace=_require_bool(
            raw_payload["detailed_optimizer_trace"],
            "detailed_optimizer_trace",
        ),
        **optional_values,
    )


__all__ = [
    "STRIDE_FIT_PROVENANCE_SCHEMA_VERSION",
    "STRIDEFitProvenance",
    "build_stride_fit_provenance",
    "validate_stride_fit_provenance",
]
