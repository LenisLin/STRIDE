"""Training runtime configuration for the canonical STRIDE optimizer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ..errors import ContractError
from ..losses.assembly import EPSILON_NORM
from ..observation.balanced_sinkhorn import BalancedSinkhornDivergenceConfig
from .schedules import (
    CosineConfig,
    OptimizationSchedule,
    REFERENCE_OPTIMIZER_PROTOCOL,
    build_reference_schedule,
)


SchedulerPolicy = Literal["none", "CosineAnnealingLR"]


@dataclass(frozen=True)
class TrainConfig:
    """Reference optimizer configuration plus runtime toggles for one STRIDE fit."""

    objective_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    convergence_tol: float = 1e-6
    patience: int = 5
    min_relative_improvement: float = 0.0
    epsilon_norm: float = EPSILON_NORM
    detailed_trace: bool = False
    device: object | None = None
    seed: int | None = None
    ablation_mode: str = "none"
    observation_config: BalancedSinkhornDivergenceConfig | None = None
    schedule: OptimizationSchedule = field(default_factory=build_reference_schedule)


def _require_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field_name} must be a positive integer")
    return int(value)


def _require_finite_float(
    value: object,
    *,
    field_name: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field_name} must be a finite float")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} must be a finite float") from exc
    if not math.isfinite(resolved):
        raise ContractError(f"{field_name} must be finite")
    if nonnegative and resolved < 0.0:
        raise ContractError(f"{field_name} must be nonnegative")
    if positive and resolved <= 0.0:
        raise ContractError(f"{field_name} must be positive")
    return resolved


def validate_train_config(config: TrainConfig | None) -> TrainConfig:
    """Return a validated training configuration."""
    resolved = config or TrainConfig()
    if not isinstance(resolved, TrainConfig):
        raise ContractError("config must be a TrainConfig object")
    if not isinstance(resolved.objective_weights, tuple) or len(resolved.objective_weights) != 3:
        raise ContractError("objective_weights must be a tuple ordered as (fit, prior, cohort)")
    objective_weights = tuple(
        _require_finite_float(weight, field_name="objective_weights", nonnegative=True)
        for weight in resolved.objective_weights
    )
    if sum(objective_weights) <= 0.0:
        raise ContractError("objective_weights must contain at least one positive entry")
    if objective_weights != (1.0, 1.0, 1.0):
        raise ContractError(
            "objective_weights is fixed to (1.0, 1.0, 1.0) for the "
            "stride_full_estimator_three_block_v1 reference objective"
        )
    _require_positive_int(resolved.patience, field_name="patience")
    _require_finite_float(resolved.convergence_tol, field_name="convergence_tol", nonnegative=True)
    _require_finite_float(
        resolved.min_relative_improvement,
        field_name="min_relative_improvement",
        nonnegative=True,
    )
    epsilon_norm = _require_finite_float(resolved.epsilon_norm, field_name="epsilon_norm", positive=True)
    if not math.isclose(epsilon_norm, EPSILON_NORM, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "epsilon_norm is fixed to 0.01 for the "
            "stride_full_estimator_three_block_v1 reference objective"
        )
    if not isinstance(resolved.detailed_trace, bool):
        raise ContractError("detailed_trace must be a bool")
    if resolved.seed is not None and (isinstance(resolved.seed, bool) or not isinstance(resolved.seed, int)):
        raise ContractError("seed must be an int or None")
    _validate_reference_schedule(resolved.schedule)
    return resolved


def _validate_reference_schedule(schedule: OptimizationSchedule) -> None:
    if not isinstance(schedule, OptimizationSchedule):
        raise ContractError("schedule must be an OptimizationSchedule object")
    if schedule.protocol_name != REFERENCE_OPTIMIZER_PROTOCOL:
        raise ContractError("schedule.protocol_name must equal the fixed STRIDE reference protocol")
    if len(schedule.stages) != 2:
        raise ContractError("schedule.stages must contain exactly warmup and main")
    warmup = schedule.warmup_stage
    main = schedule.main_stage
    _validate_stage(
        warmup,
        expected_name="warmup",
        expected_lr=0.02,
        expected_min_steps=20,
        expected_max_steps=20,
        expected_scheduler_policy="none",
        expected_allow_early_stop=False,
    )
    _validate_stage(
        main,
        expected_name="main",
        expected_lr=0.05,
        expected_min_steps=100,
        expected_max_steps=200,
        expected_scheduler_policy="CosineAnnealingLR",
        expected_allow_early_stop=True,
    )
    if schedule.early_stop_eligibility_policy != "main_after_min_steps":
        raise ContractError(
            "schedule.early_stop_eligibility_policy must equal 'main_after_min_steps'"
        )
    cosine = schedule.cosine
    if not isinstance(cosine, CosineConfig):
        raise ContractError("schedule.cosine must be a CosineConfig object")
    if _require_positive_int(cosine.T_max, field_name="schedule.cosine.T_max") != 200:
        raise ContractError("schedule.cosine.T_max must equal 200")
    _require_finite_float(
        cosine.eta_min,
        field_name="schedule.cosine.eta_min",
        nonnegative=True,
    )
    if not math.isclose(float(cosine.eta_min), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("schedule.cosine.eta_min must equal 0.0")


def _require_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field_name} must be a non-negative integer")
    return int(value)


def _validate_stage(
    stage: object,
    *,
    expected_name: str,
    expected_lr: float,
    expected_min_steps: int,
    expected_max_steps: int,
    expected_scheduler_policy: SchedulerPolicy,
    expected_allow_early_stop: bool,
) -> None:
    if not hasattr(stage, "name"):
        raise ContractError("schedule stages must be OptimizationStage objects")
    if str(getattr(stage, "name")) != expected_name:
        raise ContractError(f"schedule stage {expected_name!r} must keep its canonical name")
    lr = _require_finite_float(getattr(stage, "lr"), field_name=f"{expected_name}.lr", positive=True)
    if not math.isclose(lr, expected_lr, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(f"schedule stage {expected_name!r} lr must equal {expected_lr}")
    min_steps = _require_positive_int(getattr(stage, "min_steps"), field_name=f"{expected_name}.min_steps")
    max_steps = _require_positive_int(getattr(stage, "max_steps"), field_name=f"{expected_name}.max_steps")
    if min_steps != expected_min_steps or max_steps != expected_max_steps:
        raise ContractError(
            f"schedule stage {expected_name!r} must keep canonical min/max steps "
            f"({expected_min_steps}, {expected_max_steps})"
        )
    scheduler_policy = getattr(stage, "scheduler_policy")
    if scheduler_policy != expected_scheduler_policy:
        raise ContractError(
            f"schedule stage {expected_name!r} scheduler_policy must equal "
            f"{expected_scheduler_policy!r}"
        )
    if bool(getattr(stage, "allow_early_stop")) is not expected_allow_early_stop:
        raise ContractError(
            f"schedule stage {expected_name!r} allow_early_stop must equal "
            f"{expected_allow_early_stop!r}"
        )


__all__ = ["SchedulerPolicy", "TrainConfig", "validate_train_config"]
