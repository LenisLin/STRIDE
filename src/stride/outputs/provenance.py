"""Compact successful-fit provenance contracts for STRIDE outputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

from ..errors import ContractError


STRIDE_FIT_PROVENANCE_SCHEMA_VERSION = "stride_fit_provenance.v1"

_REQUIRED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "provenance_schema_version",
    "alpha",
    "random_seed",
    "initialization",
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
    "alpha_sensitivity",
    "legacy_observation_diagnostics",
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
_LOSS_COMPONENT_FIELDS: tuple[str, ...] = (
    "raw",
    "scale",
    "normalized",
    "floor_used",
)
_LOSS_COMPONENTS: tuple[str, ...] = (
    "obs",
    "open",
    "geometry",
    "consistency",
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


@dataclass(frozen=True)
class STRIDEFitProvenance:
    """Validated compact provenance for one successful full STRIDE fit."""

    provenance_schema_version: str
    alpha: float
    random_seed: int | None
    initialization: Mapping[str, Any]
    loss: Mapping[str, Any]
    e_bounds: tuple[float, float]
    post_reconstruction_form: str
    observation_comparison_plan: Mapping[str, Any]
    observation_discrepancy: Mapping[str, Any]
    state_geometry: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    recurrence: Mapping[str, Any]
    detailed_optimizer_trace: bool = False
    alpha_sensitivity: Any | None = None
    legacy_observation_diagnostics: Any | None = None
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
            "alpha": self.alpha,
            "random_seed": self.random_seed,
            "initialization": _copy_value(self.initialization),
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
        alpha=_require_finite_float(raw_payload["alpha"], "alpha"),
        random_seed=_require_int_or_none(raw_payload["random_seed"], "random_seed"),
        initialization=_copy_mapping(
            _require_mapping(raw_payload["initialization"], "initialization")
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


def validate_stride_fit_provenance(
    provenance: Mapping[str, Any] | STRIDEFitProvenance,
) -> None:
    """Validate the compact successful-fit provenance schema."""
    payload = _payload_mapping(provenance)
    _reject_forbidden_fields(payload)

    for field_name in _REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in payload:
            raise ContractError(
                f"STRIDE fit provenance missing required provenance field {field_name!r}"
            )
    for field_name in payload:
        if str(field_name) not in _ALLOWED_TOP_LEVEL_FIELDS:
            raise ContractError(f"STRIDE fit provenance unexpected provenance field {field_name!r}")

    _require_exact_string(
        payload["provenance_schema_version"],
        "provenance_schema_version",
        STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    )
    alpha = _require_finite_float(payload["alpha"], "alpha")
    _require_int_or_none(payload["random_seed"], "random_seed")
    _validate_initialization(payload["initialization"])
    _validate_ablation_extensions(payload)
    _validate_loss(payload["loss"], alpha=alpha, ablation_mode=_effective_ablation_mode(payload))
    _validate_e_bounds(payload["e_bounds"])
    _require_exact_string(
        payload["post_reconstruction_form"],
        "post_reconstruction_form",
        "normalize(q_minus @ A + e)",
    )
    _validate_observation_comparison_plan(payload["observation_comparison_plan"])
    _validate_observation_discrepancy(payload["observation_discrepancy"])
    _validate_state_geometry(payload["state_geometry"])
    _validate_optimizer(payload["optimizer"])
    _validate_recurrence(payload["recurrence"])
    _require_bool(payload["detailed_optimizer_trace"], "detailed_optimizer_trace")


def _payload_mapping(provenance: Mapping[str, Any] | STRIDEFitProvenance) -> Mapping[str, Any]:
    if isinstance(provenance, STRIDEFitProvenance):
        payload: dict[str, Any] = {
            "provenance_schema_version": provenance.provenance_schema_version,
            "alpha": provenance.alpha,
            "random_seed": provenance.random_seed,
            "initialization": provenance.initialization,
            "loss": provenance.loss,
            "e_bounds": provenance.e_bounds,
            "post_reconstruction_form": provenance.post_reconstruction_form,
            "observation_comparison_plan": provenance.observation_comparison_plan,
            "observation_discrepancy": provenance.observation_discrepancy,
            "state_geometry": provenance.state_geometry,
            "optimizer": provenance.optimizer,
            "recurrence": provenance.recurrence,
            "detailed_optimizer_trace": provenance.detailed_optimizer_trace,
        }
        for field_name in (*_OPTIONAL_DIAGNOSTIC_FIELDS, *_OPTIONAL_ABLATION_FIELDS):
            value = getattr(provenance, field_name)
            if value is not None:
                payload[field_name] = value
        return payload
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


def _reject_forbidden_fields(value: Any, *, path: str = "provenance") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PROVENANCE_FIELDS:
                raise ContractError(
                    "STRIDE fit provenance must not carry forbidden provenance field "
                    f"{key_text!r}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be a mapping")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be a bool")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be an int")
    return int(value)


def _require_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name)


def _require_finite_float(
    value: Any,
    field_name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be a finite float")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must be a finite float"
        ) from exc
    if not isfinite(numeric):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be finite")
    if nonnegative and numeric < 0.0:
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be non-negative")
    if positive and numeric <= 0.0:
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be positive")
    return numeric


def _require_exact_string(value: Any, field_name: str, expected: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must equal {expected!r}"
        )


def _require_numeric_value(value: Any, field_name: str, expected: float) -> None:
    numeric = _require_finite_float(value, field_name)
    if not isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must equal {expected!r}"
        )


def _require_numeric_formula_value(
    value: Any,
    field_name: str,
    expected: float,
    *,
    abs_tol: float = 1e-8,
) -> None:
    numeric = _require_finite_float(value, field_name, nonnegative=True)
    if not isclose(numeric, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must match formula value {expected!r}"
        )


def _require_int_value(value: Any, field_name: str, expected: int) -> None:
    numeric = _require_int(value, field_name)
    if numeric != expected:
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must equal {expected!r}"
        )


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractError(f"STRIDE fit provenance field {field_name!r} must be a sequence")
    return value


def _coerce_e_bounds(value: Any) -> tuple[float, float]:
    bounds = _require_sequence(value, "e_bounds")
    if len(bounds) != 2:
        raise ContractError("STRIDE fit provenance field 'e_bounds' must have length 2")
    return (
        _require_finite_float(bounds[0], "e_bounds[0]"),
        _require_finite_float(bounds[1], "e_bounds[1]"),
    )


def _validate_e_bounds(value: Any) -> None:
    bounds = _coerce_e_bounds(value)
    if not (
        isclose(bounds[0], 0.0, rel_tol=0.0, abs_tol=1e-12)
        and isclose(bounds[1], 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ContractError("STRIDE fit provenance field 'e_bounds' must equal [0.0, 1.0]")


def _validate_exact_numeric_sequence(
    value: Any,
    field_name: str,
    expected: tuple[float, ...],
) -> None:
    sequence = _require_sequence(value, field_name)
    if len(sequence) != len(expected):
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} must have length {len(expected)}"
        )
    for index, expected_value in enumerate(expected):
        _require_numeric_value(sequence[index], f"{field_name}[{index}]", expected_value)


def _validate_no_extra_keys(
    mapping: Mapping[str, Any],
    field_name: str,
    expected_keys: tuple[str, ...],
) -> None:
    expected = set(expected_keys)
    actual = {str(key) for key in mapping}
    missing = tuple(key for key in expected_keys if key not in actual)
    if missing:
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} missing required keys {missing!r}"
        )
    extra = tuple(sorted(actual - expected))
    if extra:
        raise ContractError(
            f"STRIDE fit provenance field {field_name!r} has unexpected keys {extra!r}"
        )


def _validate_initialization(value: Any) -> None:
    initialization = _require_mapping(value, "initialization")
    _validate_no_extra_keys(
        initialization,
        "initialization",
        ("policy", "delta_init", "K", "dtype"),
    )
    _require_exact_string(initialization["policy"], "initialization.policy", "identity_plus_small_open")
    delta_init = _require_finite_float(
        initialization["delta_init"],
        "initialization.delta_init",
        positive=True,
    )
    K = _require_int(initialization["K"], "initialization.K")
    if K <= 0:
        raise ContractError("STRIDE fit provenance field 'initialization.K' must be positive")
    expected_delta = min(0.05, 1.0 / float(K + 1))
    if not isclose(delta_init, expected_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "STRIDE fit provenance field 'initialization.delta_init' must equal "
            "min(0.05, 1 / (K + 1))"
        )
    _require_exact_string(initialization["dtype"], "initialization.dtype", "float64")


def _validate_loss_component(value: Any, field_name: str, *, component_name: str) -> None:
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(component, field_name, _LOSS_COMPONENT_FIELDS)
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    scale = _require_finite_float(component["scale"], f"{field_name}.scale", positive=True)
    if component_name == "open":
        _require_numeric_value(component["scale"], f"{field_name}.scale", 1.0)
    normalized = _require_finite_float(
        component["normalized"],
        f"{field_name}.normalized",
        nonnegative=True,
    )
    expected_normalized = raw / scale
    if not isclose(normalized, expected_normalized, rel_tol=0.0, abs_tol=1e-8):
        raise ContractError(
            f"STRIDE fit provenance field {field_name + '.normalized'!r} must equal raw / scale"
        )
    floor_used = _require_bool(component["floor_used"], f"{field_name}.floor_used")
    if component_name == "open" and floor_used:
        raise ContractError(
            f"STRIDE fit provenance field {field_name + '.floor_used'!r} must be False"
        )


def _validate_loss(value: Any, *, alpha: float, ablation_mode: str | None) -> None:
    loss = _require_mapping(value, "loss")
    _validate_no_extra_keys(
        loss,
        "loss",
        (
            "total",
            "local",
            "regularization",
            "epsilon_norm",
            "local_denominator",
            "regularization_denominator",
            "components",
        ),
    )
    _require_finite_float(loss["total"], "loss.total", nonnegative=True)
    _require_finite_float(loss["local"], "loss.local", nonnegative=True)
    _require_finite_float(loss["regularization"], "loss.regularization", nonnegative=True)
    _require_numeric_value(loss["epsilon_norm"], "loss.epsilon_norm", 0.01)
    _require_int_value(loss["local_denominator"], "loss.local_denominator", 3)
    _require_int_value(loss["regularization_denominator"], "loss.regularization_denominator", 2)
    components = _require_mapping(loss["components"], "loss.components")
    _validate_no_extra_keys(components, "loss.components", _LOSS_COMPONENTS)
    for component_name in _LOSS_COMPONENTS:
        _validate_loss_component(
            components[component_name],
            f"loss.components.{component_name}",
            component_name=component_name,
        )
    normalized = {
        component_name: _require_finite_float(
            _require_mapping(
                components[component_name],
                f"loss.components.{component_name}",
            )["normalized"],
            f"loss.components.{component_name}.normalized",
            nonnegative=True,
        )
        for component_name in _LOSS_COMPONENTS
    }
    if ablation_mode == "geometry":
        normalized["geometry"] = 0.0
    elif ablation_mode == "recurrence":
        normalized["recurrence"] = 0.0
    elif ablation_mode == "consistency":
        normalized["consistency"] = 0.0
    expected_local = (
        normalized["obs"] + normalized["open"] + normalized["geometry"]
    ) / 3.0
    expected_regularization = (normalized["consistency"] + normalized["recurrence"]) / 2.0
    expected_total = (1.0 - alpha) * expected_local + alpha * expected_regularization
    _require_numeric_formula_value(loss["local"], "loss.local", expected_local)
    _require_numeric_formula_value(
        loss["regularization"],
        "loss.regularization",
        expected_regularization,
    )
    _require_numeric_formula_value(loss["total"], "loss.total", expected_total)


def _validate_observation_comparison_plan(value: Any) -> None:
    plan = _require_mapping(value, "observation_comparison_plan")
    _validate_no_extra_keys(
        plan,
        "observation_comparison_plan",
        ("resolved_by", "n_evidence_blocks", "domain_policy"),
    )
    _require_exact_string(plan["resolved_by"], "observation_comparison_plan.resolved_by", "task_layer")
    n_blocks = _require_int(
        plan["n_evidence_blocks"],
        "observation_comparison_plan.n_evidence_blocks",
    )
    if n_blocks <= 0:
        raise ContractError(
            "STRIDE fit provenance field 'observation_comparison_plan.n_evidence_blocks' "
            "must be positive"
        )
    _require_exact_string(
        plan["domain_policy"],
        "observation_comparison_plan.domain_policy",
        "observation_layer_only",
    )


def _validate_observation_discrepancy(value: Any) -> None:
    discrepancy = _require_mapping(value, "observation_discrepancy")
    _validate_no_extra_keys(
        discrepancy,
        "observation_discrepancy",
        (
            "operator_version",
            "backend",
            "dtype",
            "inner_epsilon_schedule",
            "outer_epsilon_schedule",
            "max_iter",
            "tol",
            "warning_tol",
        ),
    )
    _require_exact_string(
        discrepancy["operator_version"],
        "observation_discrepancy.operator_version",
        "D_obs^BalancedSinkhornDivergence-v1",
    )
    _require_exact_string(discrepancy["backend"], "observation_discrepancy.backend", "torch")
    _require_exact_string(discrepancy["dtype"], "observation_discrepancy.dtype", "float64")
    _validate_exact_numeric_sequence(
        discrepancy["inner_epsilon_schedule"],
        "observation_discrepancy.inner_epsilon_schedule",
        (0.5, 0.2, 0.1),
    )
    _validate_exact_numeric_sequence(
        discrepancy["outer_epsilon_schedule"],
        "observation_discrepancy.outer_epsilon_schedule",
        (0.5, 0.2, 0.1),
    )
    _require_int_value(discrepancy["max_iter"], "observation_discrepancy.max_iter", 1000)
    _require_numeric_value(discrepancy["tol"], "observation_discrepancy.tol", 1e-6)
    _require_numeric_value(
        discrepancy["warning_tol"],
        "observation_discrepancy.warning_tol",
        1e-4,
    )


def _validate_state_geometry(value: Any) -> None:
    geometry = _require_mapping(value, "state_geometry")
    _validate_no_extra_keys(geometry, "state_geometry", ("normalization", "s_C"))
    _require_exact_string(
        geometry["normalization"],
        "state_geometry.normalization",
        "C_norm = C_raw / s_C",
    )
    _require_finite_float(geometry["s_C"], "state_geometry.s_C", positive=True)


def _validate_optimizer(value: Any) -> None:
    optimizer = _require_mapping(value, "optimizer")
    _validate_no_extra_keys(
        optimizer,
        "optimizer",
        ("framework", "algorithm", "weight_decay", "scheduler_policy"),
    )
    _require_exact_string(optimizer["framework"], "optimizer.framework", "torch")
    _require_exact_string(optimizer["algorithm"], "optimizer.algorithm", "AdamW")
    _require_numeric_value(optimizer["weight_decay"], "optimizer.weight_decay", 0.0)
    scheduler_policy = optimizer["scheduler_policy"]
    if scheduler_policy not in {"none", "ReduceLROnPlateau_on_total_objective"}:
        raise ContractError(
            "STRIDE fit provenance field 'optimizer.scheduler_policy' must be one of "
            "('none', 'ReduceLROnPlateau_on_total_objective')"
        )


def _validate_recurrence(value: Any) -> None:
    recurrence = _require_mapping(value, "recurrence")
    _validate_no_extra_keys(recurrence, "recurrence", ("support_n_patients", "dispersion"))
    support = _require_int(recurrence["support_n_patients"], "recurrence.support_n_patients")
    if support < 0:
        raise ContractError(
            "STRIDE fit provenance field 'recurrence.support_n_patients' must be non-negative"
        )
    _require_finite_float(recurrence["dispersion"], "recurrence.dispersion", nonnegative=True)


def _validate_ablation_extensions(payload: Mapping[str, Any]) -> None:
    present_fields = tuple(
        field_name for field_name in _OPTIONAL_ABLATION_FIELDS if field_name in payload
    )
    if not present_fields:
        return
    if payload.get("ablation_mode") == "none":
        if present_fields != ("ablation_mode",):
            raise ContractError(
                "STRIDE fit provenance field 'ablation_mode' may equal 'none' only "
                "without ablation_term_handling or ablation_denominator_policy"
            )
        return
    if len(present_fields) != len(_OPTIONAL_ABLATION_FIELDS):
        raise ContractError(
            "STRIDE fit provenance ablation provenance extension fields must be provided together"
        )
    if payload["ablation_mode"] not in {
        "recurrence",
        "geometry",
        "consistency",
    }:
        raise ContractError(
            "STRIDE fit provenance field 'ablation_mode' must be one of "
            "('recurrence', 'geometry', 'consistency')"
        )
    if payload["ablation_term_handling"] not in {
        "remove",
        "zero_weight",
    }:
        raise ContractError(
            "STRIDE fit provenance field 'ablation_term_handling' must be one of "
            "('remove', 'zero_weight')"
        )
    _require_exact_string(
        payload["ablation_denominator_policy"],
        "ablation_denominator_policy",
        "fixed_denominator_no_reweighting",
    )


def _effective_ablation_mode(payload: Mapping[str, Any]) -> str | None:
    ablation_mode = payload.get("ablation_mode")
    if ablation_mode in {"geometry", "recurrence", "consistency"}:
        return str(ablation_mode)
    return None


__all__ = [
    "STRIDE_FIT_PROVENANCE_SCHEMA_VERSION",
    "STRIDEFitProvenance",
    "build_stride_fit_provenance",
    "validate_stride_fit_provenance",
]
