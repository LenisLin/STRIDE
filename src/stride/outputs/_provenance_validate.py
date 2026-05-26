"""Private validators for STRIDE fit provenance payloads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any

from ..errors import ContractError
from ._provenance_payload import (
    _ALLOWED_TOP_LEVEL_FIELDS,
    _FORBIDDEN_PROVENANCE_FIELDS,
    _LOSS_COMPONENTS,
    _OPTIONAL_ABLATION_FIELDS,
    _REQUIRED_TOP_LEVEL_FIELDS,
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    _payload_mapping,
)

if TYPE_CHECKING:  # pragma: no cover
    from .provenance import STRIDEFitProvenance

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
    _require_exact_string(
        payload["objective_contract_version"],
        "objective_contract_version",
        "stride_full_estimator_three_block_v1",
    )
    _require_int_or_none(payload["random_seed"], "random_seed")
    _validate_objective_constants(payload["objective_constants"])
    _validate_initialization(payload["objective_scale_initialization"])
    _validate_optimizer_start_initialization(payload["optimizer_start_initialization"])
    _validate_ablation_extensions(payload)
    _validate_loss(
        payload["loss"],
        objective_constants=payload["objective_constants"],
        ablation_mode=_effective_ablation_mode(payload),
    )
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

def _reject_forbidden_fields(value: Any, *, path: str = "provenance") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            allowed_subbag_status = (
                path == "provenance.loss.components.subbag_consistency"
                and key_text == "status"
            )
            if key_text in _FORBIDDEN_PROVENANCE_FIELDS and not allowed_subbag_status:
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


def _validate_objective_constants(value: Any) -> None:
    constants = _require_mapping(value, "objective_constants")
    _validate_no_extra_keys(
        constants,
        "objective_constants",
        ("rho_subbag", "geometry_effective_weight", "s_cohort", "epsilon_norm"),
    )
    _require_numeric_value(constants["rho_subbag"], "objective_constants.rho_subbag", 1.0)
    _require_numeric_value(
        constants["geometry_effective_weight"],
        "objective_constants.geometry_effective_weight",
        0.01,
    )
    _require_numeric_value(constants["s_cohort"], "objective_constants.s_cohort", 0.01)
    _require_numeric_value(constants["epsilon_norm"], "objective_constants.epsilon_norm", 0.01)


def _validate_optimizer_start_initialization(value: Any) -> None:
    initialization = _require_mapping(value, "optimizer_start_initialization")
    _validate_no_extra_keys(
        initialization,
        "optimizer_start_initialization",
        ("policy", "delta_init", "offdiag_init_mass", "numerical_min_mass", "K", "dtype"),
    )
    _require_exact_string(
        initialization["policy"],
        "optimizer_start_initialization.policy",
        "offdiag_seeded_identity_plus_small_open",
    )
    delta_init = _require_finite_float(
        initialization["delta_init"],
        "optimizer_start_initialization.delta_init",
        positive=True,
    )
    K = _require_int(initialization["K"], "optimizer_start_initialization.K")
    if K <= 0:
        raise ContractError("STRIDE fit provenance field 'optimizer_start_initialization.K' must be positive")
    expected_delta = min(0.05, 1.0 / float(K + 1))
    if not isclose(delta_init, expected_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "STRIDE fit provenance field 'optimizer_start_initialization.delta_init' must equal "
            "min(0.05, 1 / (K + 1))"
        )
    _require_numeric_value(
        initialization["offdiag_init_mass"],
        "optimizer_start_initialization.offdiag_init_mass",
        0.01,
    )
    _require_numeric_value(
        initialization["numerical_min_mass"],
        "optimizer_start_initialization.numerical_min_mass",
        1e-12,
    )
    if 1.0 - delta_init - float(K - 1) * 0.01 <= 0.0:
        raise ContractError(
            "STRIDE fit provenance optimizer start initialization is invalid for K"
        )
    _require_exact_string(initialization["dtype"], "optimizer_start_initialization.dtype", "float64")


def _validate_obs_component(value: Any) -> float:
    field_name = "loss.components.obs"
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(component, field_name, ("raw", "scale", "normalized", "floor_used"))
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    scale = _require_finite_float(component["scale"], f"{field_name}.scale", positive=True)
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
    _require_bool(component["floor_used"], f"{field_name}.floor_used")
    return normalized


def _validate_open_component(value: Any) -> float:
    field_name = "loss.components.open"
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(component, field_name, ("raw", "normalized"))
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    normalized = _require_finite_float(
        component["normalized"],
        f"{field_name}.normalized",
        nonnegative=True,
    )
    if not isclose(normalized, raw, rel_tol=0.0, abs_tol=1e-8):
        raise ContractError(
            f"STRIDE fit provenance field {field_name + '.normalized'!r} must equal raw"
        )
    return normalized


def _validate_geometry_component(value: Any, *, objective_constants: Any, ablation_mode: str | None) -> float:
    field_name = "loss.components.geometry"
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(
        component,
        field_name,
        ("raw", "scale", "normalized", "effective", "floor_used"),
    )
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    scale = _require_finite_float(component["scale"], f"{field_name}.scale", positive=True)
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
    _require_bool(component["floor_used"], f"{field_name}.floor_used")
    effective = _require_finite_float(
        component["effective"],
        f"{field_name}.effective",
        nonnegative=True,
    )
    constants = _require_mapping(objective_constants, "objective_constants")
    expected_effective = float(constants["geometry_effective_weight"]) * normalized
    if ablation_mode == "geometry":
        expected_effective = 0.0
    _require_numeric_formula_value(
        effective,
        "loss.components.geometry.effective",
        expected_effective,
    )
    return effective


def _validate_subbag_consistency_component(
    value: Any,
    *,
    objective_constants: Any,
    ablation_mode: str | None,
) -> float:
    field_name = "loss.components.subbag_consistency"
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(component, field_name, ("raw", "effective", "status"))
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    effective = _require_finite_float(
        component["effective"],
        f"{field_name}.effective",
        nonnegative=True,
    )
    status = component["status"]
    if not isinstance(status, str) or status.strip() == "":
        raise ContractError(f"STRIDE fit provenance field {field_name + '.status'!r} must be a non-empty string")
    constants = _require_mapping(objective_constants, "objective_constants")
    expected_effective = float(constants["rho_subbag"]) * raw
    if ablation_mode == "consistency":
        expected_effective = 0.0
    _require_numeric_formula_value(
        effective,
        "loss.components.subbag_consistency.effective",
        expected_effective,
    )
    return effective


def _validate_recurrence_component(
    value: Any,
    *,
    objective_constants: Any,
    ablation_mode: str | None,
) -> float:
    field_name = "loss.components.recurrence"
    component = _require_mapping(value, field_name)
    _validate_no_extra_keys(component, field_name, ("raw", "cohort_scaled"))
    raw = _require_finite_float(component["raw"], f"{field_name}.raw", nonnegative=True)
    cohort_scaled = _require_finite_float(
        component["cohort_scaled"],
        f"{field_name}.cohort_scaled",
        nonnegative=True,
    )
    constants = _require_mapping(objective_constants, "objective_constants")
    expected_scaled = raw / float(constants["s_cohort"])
    _require_numeric_formula_value(
        cohort_scaled,
        "loss.components.recurrence.cohort_scaled",
        expected_scaled,
    )
    if ablation_mode == "recurrence":
        return 0.0
    return cohort_scaled


def _validate_loss(
    value: Any,
    *,
    objective_constants: Any,
    ablation_mode: str | None,
) -> None:
    loss = _require_mapping(value, "loss")
    _validate_no_extra_keys(
        loss,
        "loss",
        (
            "total",
            "fit",
            "prior",
            "cohort",
            "components",
        ),
    )
    _require_finite_float(loss["total"], "loss.total", nonnegative=True)
    _require_finite_float(loss["fit"], "loss.fit", nonnegative=True)
    _require_finite_float(loss["prior"], "loss.prior", nonnegative=True)
    _require_finite_float(loss["cohort"], "loss.cohort", nonnegative=True)
    components = _require_mapping(loss["components"], "loss.components")
    _validate_no_extra_keys(components, "loss.components", _LOSS_COMPONENTS)
    obs_normalized = _validate_obs_component(components["obs"])
    open_normalized = _validate_open_component(components["open"])
    geometry_effective = _validate_geometry_component(
        components["geometry"],
        objective_constants=objective_constants,
        ablation_mode=ablation_mode,
    )
    subbag_effective = _validate_subbag_consistency_component(
        components["subbag_consistency"],
        objective_constants=objective_constants,
        ablation_mode=ablation_mode,
    )
    recurrence_effective = _validate_recurrence_component(
        components["recurrence"],
        objective_constants=objective_constants,
        ablation_mode=ablation_mode,
    )
    expected_fit = obs_normalized + subbag_effective
    expected_prior = (open_normalized + geometry_effective) / 2.0
    expected_cohort = recurrence_effective
    expected_total = (expected_fit + expected_prior + expected_cohort) / 3.0
    _require_numeric_formula_value(loss["fit"], "loss.fit", expected_fit)
    _require_numeric_formula_value(loss["prior"], "loss.prior", expected_prior)
    _require_numeric_formula_value(loss["cohort"], "loss.cohort", expected_cohort)
    _require_numeric_formula_value(loss["total"], "loss.total", expected_total)


def _validate_observation_comparison_plan(value: Any) -> None:
    plan = _require_mapping(value, "observation_comparison_plan")
    _validate_no_extra_keys(
        plan,
        "observation_comparison_plan",
        (
            "resolved_by",
            "n_evidence_blocks",
            "domain_policy",
            "block_construction_policy",
            "n_blocks_by_patient",
        ),
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
    policy = plan["block_construction_policy"]
    if not isinstance(policy, str) or policy.strip() == "":
        raise ContractError(
            "STRIDE fit provenance field "
            "'observation_comparison_plan.block_construction_policy' "
            "must be a non-empty string"
        )
    blocks_by_patient = _require_mapping(
        plan["n_blocks_by_patient"],
        "observation_comparison_plan.n_blocks_by_patient",
    )
    if len(blocks_by_patient) == 0:
        raise ContractError(
            "STRIDE fit provenance field "
            "'observation_comparison_plan.n_blocks_by_patient' must be non-empty"
        )
    counted_blocks = 0
    for patient_id, count_value in blocks_by_patient.items():
        patient_key = str(patient_id).strip()
        if patient_key == "":
            raise ContractError(
                "STRIDE fit provenance field "
                "'observation_comparison_plan.n_blocks_by_patient' "
                "must use non-empty patient ids"
            )
        count = _require_int(
            count_value,
            f"observation_comparison_plan.n_blocks_by_patient[{patient_key!r}]",
        )
        if count <= 0:
            raise ContractError(
                "STRIDE fit provenance field "
                f"'observation_comparison_plan.n_blocks_by_patient[{patient_key!r}]' "
                "must be positive"
            )
        counted_blocks += count
    if counted_blocks != n_blocks:
        raise ContractError(
            "STRIDE fit provenance field "
            "'observation_comparison_plan.n_blocks_by_patient' "
            "must sum to observation_comparison_plan.n_evidence_blocks"
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
    _require_int_value(discrepancy["max_iter"], "observation_discrepancy.max_iter", 100)
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
        (
            "framework",
            "algorithm",
            "weight_decay",
            "protocol_name",
            "exit_flag",
            "warmup",
            "main",
            "cosine",
            "early_stop_thresholds",
        ),
    )
    _require_exact_string(optimizer["framework"], "optimizer.framework", "torch")
    _require_exact_string(optimizer["algorithm"], "optimizer.algorithm", "AdamW")
    _require_numeric_value(optimizer["weight_decay"], "optimizer.weight_decay", 0.0)
    _require_exact_string(
        optimizer["protocol_name"],
        "optimizer.protocol_name",
        "two_phase_warmup20_main100plus_v1",
    )
    if optimizer["exit_flag"] not in {"plateau_patience", "max_steps_exhausted_finite"}:
        raise ContractError(
            "STRIDE fit provenance field 'optimizer.exit_flag' must be one of "
            "('plateau_patience', 'max_steps_exhausted_finite')"
        )
    warmup = _require_mapping(optimizer["warmup"], "optimizer.warmup")
    _validate_no_extra_keys(
        warmup,
        "optimizer.warmup",
        ("steps", "lr", "scheduler_policy", "early_stop"),
    )
    _require_int_value(warmup["steps"], "optimizer.warmup.steps", 20)
    _require_numeric_value(warmup["lr"], "optimizer.warmup.lr", 0.02)
    _require_exact_string(
        warmup["scheduler_policy"],
        "optimizer.warmup.scheduler_policy",
        "none",
    )
    _require_exact_string(
        warmup["early_stop"],
        "optimizer.warmup.early_stop",
        "not_allowed",
    )
    main = _require_mapping(optimizer["main"], "optimizer.main")
    _validate_no_extra_keys(
        main,
        "optimizer.main",
        ("min_steps", "max_steps", "lr", "scheduler_policy", "early_stop"),
    )
    _require_int_value(main["min_steps"], "optimizer.main.min_steps", 100)
    _require_int_value(main["max_steps"], "optimizer.main.max_steps", 200)
    _require_numeric_value(main["lr"], "optimizer.main.lr", 0.05)
    _require_exact_string(
        main["scheduler_policy"],
        "optimizer.main.scheduler_policy",
        "CosineAnnealingLR",
    )
    _require_exact_string(
        main["early_stop"],
        "optimizer.main.early_stop",
        "main_after_min_steps",
    )
    cosine = _require_mapping(optimizer["cosine"], "optimizer.cosine")
    _validate_no_extra_keys(
        cosine,
        "optimizer.cosine",
        ("T_max", "eta_min"),
    )
    _require_int_value(cosine["T_max"], "optimizer.cosine.T_max", 200)
    _require_numeric_value(cosine["eta_min"], "optimizer.cosine.eta_min", 0.0)
    thresholds = _require_mapping(
        optimizer["early_stop_thresholds"],
        "optimizer.early_stop_thresholds",
    )
    _validate_no_extra_keys(
        thresholds,
        "optimizer.early_stop_thresholds",
        ("min_relative_improvement", "convergence_tol", "patience"),
    )
    _require_numeric_value(
        thresholds["min_relative_improvement"],
        "optimizer.early_stop_thresholds.min_relative_improvement",
        0.0,
    )
    _require_numeric_value(
        thresholds["convergence_tol"],
        "optimizer.early_stop_thresholds.convergence_tol",
        1e-6,
    )
    _require_int_value(
        thresholds["patience"],
        "optimizer.early_stop_thresholds.patience",
        5,
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
    "validate_stride_fit_provenance",
    "_coerce_e_bounds",
    "_require_bool",
    "_require_int_or_none",
    "_require_mapping",
]
