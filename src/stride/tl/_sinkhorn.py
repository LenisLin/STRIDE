"""Canonical balanced Sinkhorn observation operator for STRIDE `.tl`.

This module owns the formal `D_obs^BalancedSinkhornDivergence-v1` surface.
The operator is part of `L_obs`; it is not a biological `d/e` axis and not an
independently weighted loss term.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from stride.errors import ContractError

BALANCED_SINKHORN_OPERATOR_VERSION = "D_obs^BalancedSinkhornDivergence-v1"
CANONICAL_EPSILON_SCHEDULE = (0.5, 0.2, 0.1)
CANONICAL_MAX_ITER = 100
CANONICAL_TOL = 1e-6
CANONICAL_WARNING_TOL = 1e-4
SMALL_NEGATIVE_TOL = 1e-10
INPUT_SIMPLEX_TOL = 1e-8


@dataclass(frozen=True)
class SinkhornConfig:
    """Fixed v1 numerical settings for the canonical observation operator.

    inner_epsilon_schedule: inner composition-distance epsilon schedule.
    outer_epsilon_schedule: outer FOV-bag divergence epsilon schedule.
    max_iter: maximum iterations per epsilon stage.
    tol: convergence tolerance.
    warning_tol: convergence warning threshold.
    backend: canonical backend name.
    dtype: canonical tensor dtype name.
    debiased: whether the balanced divergence is debiased.
    """

    inner_epsilon_schedule: tuple[float, ...] = CANONICAL_EPSILON_SCHEDULE
    outer_epsilon_schedule: tuple[float, ...] = CANONICAL_EPSILON_SCHEDULE
    max_iter: int = CANONICAL_MAX_ITER
    tol: float = CANONICAL_TOL
    warning_tol: float = CANONICAL_WARNING_TOL
    backend: str = "torch"
    dtype: str = "float64"
    debiased: bool = True

    def __post_init__(self) -> None:
        inner = _canonical_epsilon_schedule(
            self.inner_epsilon_schedule,
            field_name="inner_epsilon_schedule",
        )
        outer = _canonical_epsilon_schedule(
            self.outer_epsilon_schedule,
            field_name="outer_epsilon_schedule",
        )

        object.__setattr__(self, "inner_epsilon_schedule", inner)
        object.__setattr__(self, "outer_epsilon_schedule", outer)
        object.__setattr__(self, "max_iter", _canonical_max_iter(self.max_iter))
        object.__setattr__(
            self,
            "tol",
            _canonical_float(self.tol, field_name="tol", expected=CANONICAL_TOL),
        )
        object.__setattr__(
            self,
            "warning_tol",
            _canonical_float(
                self.warning_tol,
                field_name="warning_tol",
                expected=CANONICAL_WARNING_TOL,
            ),
        )
        object.__setattr__(self, "backend", _canonical_backend(self.backend))
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))
        object.__setattr__(self, "debiased", _canonical_debiased(self.debiased))

    def metadata(self) -> dict[str, Any]:
        """Return compact provenance-compatible configuration metadata."""
        return {
            "operator_version": BALANCED_SINKHORN_OPERATOR_VERSION,
            "backend": self.backend,
            "dtype": self.dtype,
            "inner_epsilon_schedule": list(self.inner_epsilon_schedule),
            "outer_epsilon_schedule": list(self.outer_epsilon_schedule),
            "max_iter": self.max_iter,
            "tol": self.tol,
            "warning_tol": self.warning_tol,
            "debiased": self.debiased,
        }


@dataclass(frozen=True)
class SinkhornResult:
    """Structured result for the canonical observation operator.

    value: scalar tensor value used by objective assembly.
    metadata: compact operator settings and convergence metadata.
    warnings: structured convergence or numerical warning records.
    """

    value: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[Mapping[str, Any], ...] = ()


def compute_sinkhorn_divergence(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    fov_cost_scale: float,
    fov_cost_scale_floor_used: bool = False,
    observed_self_ground_cost: torch.Tensor | None = None,
    observed_self_clipped_negative: bool = False,
    validate_inputs: bool = True,
    collect_warnings: bool = False,
    config: SinkhornConfig | None = None,
) -> SinkhornResult:
    """Compute canonical balanced Sinkhorn observation discrepancy.

    Purpose:
        Declare the v1 `D_obs` comparison between predicted and observed
        target-side FOV bags.

    Key variables:
        resolved_config: fixed v1 operator settings.
        C_norm: state cost matrix normalized by `cost_scale` (`s_C`).
        G_norm: FOV ground cost normalized by fixed `fov_cost_scale`
            (`s_G_init`).
        outer_value: debiased balanced Sinkhorn divergence over FOV bags.

    Notes:
        This function validates and uses `s_G_init`; it does not compute or
        recompute that evidence-block scale during optimization.
        `observed_self_ground_cost` may provide a setup-time observed-observed
        FOV ground cost for optimizer hot paths.
        `validate_inputs=False` is reserved for tensors already checked during
        relation-fit materialization, avoiding repeated CPU synchronization in
        optimizer hot paths.
        `collect_warnings=True` enables convergence diagnostics for profiling
        runs; the default value path avoids diagnostic CPU synchronization.
    """
    resolved_config = _resolve_sinkhorn_config(config)
    runtime_checks = bool(validate_inputs or collect_warnings)

    predicted_tensor = _as_float64_matrix(predicted, name="predicted")
    observed_tensor = _as_float64_matrix(
        observed,
        name="observed",
        device=predicted_tensor.device,
    )
    if validate_inputs:
        _validate_distribution_matrix(predicted_tensor, name="predicted")
        _validate_distribution_matrix(observed_tensor, name="observed")
    if predicted_tensor.shape[1] != observed_tensor.shape[1]:
        raise ContractError("predicted and observed bags must share the same K-state axis")

    C_norm = _normalized_cost_matrix(
        cost_matrix,
        cost_scale,
        n_states=int(predicted_tensor.shape[1]),
        device=predicted_tensor.device,
        validate=validate_inputs,
    )
    s_G_init = _validate_fov_cost_scale(
        fov_cost_scale,
        floor_used=fov_cost_scale_floor_used,
    )

    G_cross = _pairwise_composition_ground_cost_value(
        predicted_tensor,
        observed_tensor,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.cross",
        collect_warnings=collect_warnings,
        runtime_checks=runtime_checks,
    )
    G_pred = _pairwise_composition_ground_cost_value(
        predicted_tensor,
        predicted_tensor,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.predicted_self",
        collect_warnings=collect_warnings,
        runtime_checks=runtime_checks,
    )
    # Observed self-cost is invariant across optimizer steps and may be precomputed.
    if observed_self_ground_cost is None:
        G_obs = _pairwise_composition_ground_cost_value(
            observed_tensor,
            observed_tensor,
            C_norm,
            config=resolved_config,
            label="inner_composition_distance.observed_self",
            collect_warnings=collect_warnings,
            runtime_checks=runtime_checks,
        )
        G_obs_value = G_obs.value
        G_obs_clipped_negative = G_obs.clipped_negative
        G_obs_warnings = G_obs.warnings
    else:
        G_obs_value = _as_float64_matrix(
            observed_self_ground_cost,
            name="observed_self_ground_cost",
            device=predicted_tensor.device,
        )
        G_obs_clipped_negative = bool(observed_self_clipped_negative)
        G_obs_warnings = ()

    predicted_mass = torch.full(
        (predicted_tensor.shape[0],),
        1.0 / float(predicted_tensor.shape[0]),
        dtype=torch.float64,
        device=predicted_tensor.device,
    )
    observed_mass = torch.full(
        (observed_tensor.shape[0],),
        1.0 / float(observed_tensor.shape[0]),
        dtype=torch.float64,
        device=predicted_tensor.device,
    )
    outer = _batched_sinkhorn_divergence_value(
        predicted_mass.unsqueeze(0),
        observed_mass.unsqueeze(0),
        (G_cross.value / s_G_init).unsqueeze(0),
        (G_pred.value / s_G_init).unsqueeze(0),
        (G_obs_value / s_G_init).unsqueeze(0),
        epsilon_schedule=resolved_config.outer_epsilon_schedule,
        config=resolved_config,
        label="outer_fov_bag_divergence",
        collect_warnings=collect_warnings,
        runtime_checks=runtime_checks,
    )
    value, outer_clipped, outer_warnings = _apply_small_negative_rule(
        outer.value[0],
        label="outer_fov_bag_divergence",
        runtime_checks=runtime_checks,
    )

    clipped_negative = (
        G_cross.clipped_negative
        or G_pred.clipped_negative
        or G_obs_clipped_negative
        or outer_clipped
    )
    warnings = (
        G_cross.warnings
        + G_pred.warnings
        + G_obs_warnings
        + outer.warnings
        + outer_warnings
    )

    metadata = {
        **resolved_config.metadata(),
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": float(cost_scale),
        },
        "fov_ground_cost": {
            "normalization": "G_norm = G / s_G_init",
            "s_G_init": float(s_G_init),
            "s_G_init_floor_used": bool(fov_cost_scale_floor_used),
        },
        "clipped_negative": bool(clipped_negative),
    }
    return SinkhornResult(value=value, metadata=metadata, warnings=warnings)


def compute_fov_ground_cost_matrix(
    left: torch.Tensor,
    right: torch.Tensor,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    validate_inputs: bool = True,
    collect_warnings: bool = False,
    config: SinkhornConfig | None = None,
) -> SinkhornResult:
    """Compute setup-time FOV-level inner ground costs for `_train.py`.

    Purpose:
        Provide the fixed initialization-time FOV ground-cost matrix used to
        derive per-block `s_G_init`. This helper computes only inner
        composition distances; it does not compute the outer FOV-bag
        divergence.

    Key variables:
        left: left FOV composition bag `[N, K]`.
        right: right FOV composition bag `[M, K]`.
        C_norm: state cost matrix normalized by `cost_scale`.
        fov_ground_cost: matrix `[N, M]` of inner composition distances.
    """
    resolved_config = _resolve_sinkhorn_config(config)
    left_tensor = _as_float64_matrix(left, name="left")
    right_tensor = _as_float64_matrix(right, name="right", device=left_tensor.device)
    if validate_inputs:
        _validate_distribution_matrix(left_tensor, name="left")
        _validate_distribution_matrix(right_tensor, name="right")
    if left_tensor.shape[1] != right_tensor.shape[1]:
        raise ContractError("left and right bags must share the same K-state axis")

    C_norm = _normalized_cost_matrix(
        cost_matrix,
        cost_scale,
        n_states=int(left_tensor.shape[1]),
        device=left_tensor.device,
        validate=validate_inputs,
    )
    ground = _pairwise_composition_ground_cost_value(
        left_tensor,
        right_tensor,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.fov_ground_cost",
        collect_warnings=collect_warnings,
    )
    metadata = {
        **resolved_config.metadata(),
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": float(cost_scale),
        },
        "fov_ground_cost": {
            "role": "setup_time_inner_ground_cost_matrix",
            "shape": tuple(int(item) for item in ground.value.shape),
        },
        "clipped_negative": bool(ground.clipped_negative),
    }
    # This setup helper feeds fixed scales/cache, not the optimizer autograd graph.
    return SinkhornResult(
        value=ground.value.detach(),
        metadata=metadata,
        warnings=ground.warnings,
    )


def compute_observed_self_ground_cost(
    observed: torch.Tensor,
    cost_matrix: torch.Tensor,
    cost_scale: float,
    *,
    validate_inputs: bool = True,
    collect_warnings: bool = False,
    config: SinkhornConfig | None = None,
) -> SinkhornResult:
    """Compute fixed observed-observed FOV ground cost for one evidence block.

    Purpose:
        Provide the optimizer setup path for the observed self-cost term that
        is invariant across repeated objective evaluations.

    Key variables:
        C_norm: state cost matrix normalized by `cost_scale`.
        observed_self_ground_cost: FOV-by-FOV observed self ground cost.
    """
    resolved_config = _resolve_sinkhorn_config(config)
    observed_tensor = _as_float64_matrix(observed, name="observed")
    if validate_inputs:
        _validate_distribution_matrix(observed_tensor, name="observed")
    C_norm = _normalized_cost_matrix(
        cost_matrix,
        cost_scale,
        n_states=int(observed_tensor.shape[1]),
        device=observed_tensor.device,
        validate=validate_inputs,
    )
    G_obs = _pairwise_composition_ground_cost_value(
        observed_tensor,
        observed_tensor,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.observed_self",
        collect_warnings=collect_warnings,
    )
    metadata = {
        **resolved_config.metadata(),
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": float(cost_scale),
        },
        "clipped_negative": bool(G_obs.clipped_negative),
    }
    return SinkhornResult(value=G_obs.value.detach(), metadata=metadata, warnings=G_obs.warnings)


def _resolve_sinkhorn_config(
    config: SinkhornConfig | None,
) -> SinkhornConfig:
    """Return the canonical v1 Sinkhorn config.

    Purpose:
        Centralize default config resolution for the operator.

    Key variables:
        resolved_config: provided config or default `SinkhornConfig`.
    """
    if config is None:
        return SinkhornConfig()
    if not isinstance(config, SinkhornConfig):
        raise ContractError("config must be None or a SinkhornConfig instance")
    return config


@dataclass(frozen=True)
class _BatchedSinkhornValue:
    value: torch.Tensor
    final_updates: torch.Tensor
    iterations: torch.Tensor
    max_iter_reached: torch.Tensor


@dataclass(frozen=True)
class _DivergenceValue:
    value: torch.Tensor
    warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class _GroundCostValue:
    value: torch.Tensor
    clipped_negative: bool
    warnings: tuple[Mapping[str, Any], ...] = ()


def _canonical_epsilon_schedule(value: Any, *, field_name: str) -> tuple[float, ...]:
    try:
        schedule = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"SinkhornConfig.{field_name} must equal the canonical v1 schedule"
        ) from exc
    if schedule != CANONICAL_EPSILON_SCHEDULE:
        raise ContractError(f"SinkhornConfig.{field_name} must equal the canonical v1 schedule")
    return schedule


def _canonical_max_iter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != CANONICAL_MAX_ITER:
        raise ContractError("SinkhornConfig.max_iter must equal the canonical v1 default")
    return int(value)


def _canonical_float(value: Any, *, field_name: str, expected: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"SinkhornConfig.{field_name} must equal the canonical v1 default"
        ) from exc
    if resolved != expected:
        raise ContractError(f"SinkhornConfig.{field_name} must equal the canonical v1 default")
    return resolved


def _canonical_backend(value: Any) -> str:
    if str(value).strip().lower() != "torch":
        raise ContractError("SinkhornConfig.backend must equal 'torch'")
    return "torch"


def _canonical_dtype(value: Any) -> str:
    if str(value).strip().lower() != "float64":
        raise ContractError("SinkhornConfig.dtype must equal 'float64'")
    return "float64"


def _canonical_debiased(value: Any) -> bool:
    if value is not True:
        raise ContractError("SinkhornConfig.debiased must be True")
    return True


def _as_float64_matrix(value: Any, *, name: str, device: Any | None = None) -> torch.Tensor:
    try:
        if torch.is_tensor(value):
            resolved_device = value.device if device is None else device
            tensor = value.to(device=resolved_device, dtype=torch.float64)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be coercible to a float64 tensor") from exc

    if tensor.ndim != 2:
        raise ContractError(f"{name} must be a 2D tensor")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ContractError(f"{name} must be non-empty")
    return tensor


def _ensure_bool(value: torch.Tensor | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value.detach().cpu().item())


def _validate_distribution_matrix(matrix: torch.Tensor, *, name: str) -> None:
    if not _ensure_bool(torch.isfinite(matrix).all()):
        raise ContractError(f"{name} must contain only finite values")
    if _ensure_bool((matrix < 0.0).any()):
        raise ContractError(f"{name} entries must be nonnegative")

    row_sums = torch.sum(matrix, dim=1)
    if not _ensure_bool(
        torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=INPUT_SIMPLEX_TOL,
            atol=INPUT_SIMPLEX_TOL,
        )
    ):
        raise ContractError(f"{name} rows must sum to 1.0")


def _normalized_cost_matrix(
    cost_matrix: Any,
    cost_scale: float,
    *,
    n_states: int,
    device: Any,
    validate: bool = True,
) -> torch.Tensor:
    C_raw = _as_float64_matrix(cost_matrix, name="cost_matrix", device=device)
    if C_raw.shape != (n_states, n_states):
        raise ContractError("cost_matrix must be [K, K] aligned with FOV bags")

    try:
        scale = float(cost_scale)
    except (TypeError, ValueError) as exc:
        raise ContractError("cost_scale must be finite and strictly positive") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("cost_scale must be finite and strictly positive")

    if validate:
        if not _ensure_bool(torch.isfinite(C_raw).all()):
            raise ContractError("cost_matrix must contain only finite values")
        if _ensure_bool((C_raw < 0.0).any()):
            raise ContractError("cost_matrix must be nonnegative")
        if not _ensure_bool(torch.allclose(C_raw, C_raw.T, rtol=0.0, atol=1e-12)):
            raise ContractError("cost_matrix must be symmetric")
        if not _ensure_bool(
            torch.allclose(
                torch.diagonal(C_raw),
                torch.zeros(C_raw.shape[0], dtype=torch.float64, device=device),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ContractError("cost_matrix diagonal must be zero")
        off_diagonal = ~torch.eye(C_raw.shape[0], dtype=torch.bool, device=device)
        if not _ensure_bool(((C_raw > 0.0) & off_diagonal).any()):
            raise ContractError("cost_matrix must contain a positive off-diagonal cost")

    C_norm = C_raw / scale
    if validate and not _ensure_bool(torch.isfinite(C_norm).all()):
        raise ContractError("C_norm = C_raw / s_C must contain only finite values")
    return C_norm


def _validate_fov_cost_scale(fov_cost_scale: float, *, floor_used: bool) -> float:
    if not isinstance(floor_used, bool):
        raise ContractError("fov_cost_scale_floor_used must be a bool")
    try:
        scale = float(fov_cost_scale)
    except (TypeError, ValueError) as exc:
        raise ContractError("fov_cost_scale must be finite and strictly positive") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("fov_cost_scale must be finite and strictly positive")
    if floor_used and scale != 1.0:
        raise ContractError("fov_cost_scale_floor_used=True requires fov_cost_scale=1.0")
    return scale


def _safe_log_distribution(distribution: torch.Tensor) -> torch.Tensor:
    logged = torch.full_like(distribution, -torch.inf)
    positive = distribution > 0.0
    return torch.where(
        positive,
        torch.log(torch.clamp(distribution, min=torch.finfo(distribution.dtype).tiny)),
        logged,
    )


def _batched_sinkhorn_transport_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cost_matrix: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: SinkhornConfig,
    runtime_checks: bool = True,
) -> _BatchedSinkhornValue:
    if left_mass.ndim != 2 or right_mass.ndim != 2 or cost_matrix.ndim != 3:
        raise ContractError("batched Sinkhorn masses/costs must be [B,N], [B,M], [B,N,M]")
    batch_size = int(left_mass.shape[0])
    if batch_size <= 0:
        raise ContractError("batched Sinkhorn requires at least one transport item")
    if right_mass.shape[0] != batch_size or cost_matrix.shape[0] != batch_size:
        raise ContractError("batched Sinkhorn batch dimensions must align")
    if cost_matrix.shape[1:] != (left_mass.shape[1], right_mass.shape[1]):
        raise ContractError("batched Sinkhorn cost matrix shape must match masses")

    left_value = left_mass.detach()
    right_value = right_mass.detach()
    cost_value = cost_matrix.detach()
    log_left = _safe_log_distribution(left_value)
    log_right = _safe_log_distribution(right_value)
    left_positive = left_value > 0.0
    right_positive = right_value > 0.0
    neg_inf_left = torch.full_like(left_value, -torch.inf)
    neg_inf_right = torch.full_like(right_value, -torch.inf)

    f = torch.zeros_like(left_value)
    g = torch.zeros_like(right_value)
    n_stages = len(epsilon_schedule)
    final_updates = torch.full(
        (batch_size, n_stages),
        torch.inf,
        dtype=left_value.dtype,
        device=left_value.device,
    )
    iterations = torch.zeros(
        (batch_size, n_stages),
        dtype=torch.int64,
        device=left_value.device,
    )
    max_iter_reached = torch.zeros(
        (batch_size, n_stages),
        dtype=torch.bool,
        device=left_value.device,
    )

    with torch.no_grad():
        for stage_index, epsilon in enumerate(epsilon_schedule):
            eps = float(epsilon)
            active = torch.ones(batch_size, dtype=torch.bool, device=left_value.device)
            first_converged_residual = torch.full(
                (batch_size,),
                torch.inf,
                dtype=left_value.dtype,
                device=left_value.device,
            )
            first_converged_iteration = torch.zeros(
                (batch_size,),
                dtype=torch.int64,
                device=left_value.device,
            )
            last_residual = torch.full(
                (batch_size,),
                torch.inf,
                dtype=left_value.dtype,
                device=left_value.device,
            )
            if runtime_checks:
                saw_nonfinite_update = torch.zeros(
                    (batch_size,),
                    dtype=torch.bool,
                    device=left_value.device,
                )
                saw_nonfinite_residual = torch.zeros(
                    (batch_size,),
                    dtype=torch.bool,
                    device=left_value.device,
                )

            for iteration in range(1, int(config.max_iter) + 1):
                masked_g = torch.where(right_positive, g, neg_inf_right)
                proposed_f = eps * (
                    log_left
                    - torch.logsumexp(
                        (masked_g.unsqueeze(1) - cost_value) / eps,
                        dim=2,
                    )
                )
                next_f = torch.where(left_positive, proposed_f, torch.zeros_like(proposed_f))
                if runtime_checks:
                    saw_nonfinite_update = saw_nonfinite_update | ~torch.isfinite(next_f).all(dim=1)
                f = torch.where(active.unsqueeze(1), next_f, f)

                masked_f = torch.where(left_positive, f, neg_inf_left)
                proposed_g = eps * (
                    log_right
                    - torch.logsumexp(
                        (masked_f.unsqueeze(2) - cost_value) / eps,
                        dim=1,
                    )
                )
                next_g = torch.where(right_positive, proposed_g, torch.zeros_like(proposed_g))
                if runtime_checks:
                    saw_nonfinite_update = saw_nonfinite_update | ~torch.isfinite(next_g).all(dim=1)
                g = torch.where(active.unsqueeze(1), next_g, g)

                masked_f_plan = torch.where(left_positive, f, neg_inf_left)
                masked_g_plan = torch.where(right_positive, g, neg_inf_right)
                log_plan = (
                    masked_f_plan.unsqueeze(2) + masked_g_plan.unsqueeze(1) - cost_value
                ) / eps
                plan = torch.exp(log_plan)
                row_residual = torch.amax(torch.abs(plan.sum(dim=2) - left_value), dim=1)
                column_residual = torch.amax(torch.abs(plan.sum(dim=1) - right_value), dim=1)
                residual = torch.maximum(row_residual, column_residual)
                if runtime_checks:
                    saw_nonfinite_residual = saw_nonfinite_residual | ~torch.isfinite(residual)
                last_residual = torch.where(active, residual, last_residual)

                if runtime_checks:
                    converged_now = active & torch.isfinite(residual) & (residual <= float(config.tol))
                else:
                    converged_now = active & (residual <= float(config.tol))
                first_converged_residual = torch.where(
                    converged_now,
                    residual,
                    first_converged_residual,
                )
                first_converged_iteration = torch.where(
                    converged_now,
                    torch.full_like(first_converged_iteration, iteration),
                    first_converged_iteration,
                )
                active = active & ~converged_now

            if runtime_checks:
                if _ensure_bool(saw_nonfinite_update.any()):
                    raise ContractError("batched Sinkhorn iteration produced a non-finite update")
                if _ensure_bool(saw_nonfinite_residual.any()):
                    raise ContractError("batched Sinkhorn iteration produced a non-finite residual")

            final_updates[:, stage_index] = torch.where(
                active,
                last_residual,
                first_converged_residual,
            )
            iterations[:, stage_index] = torch.where(
                active,
                torch.full_like(first_converged_iteration, int(config.max_iter)),
                first_converged_iteration,
            )
            max_iter_reached[:, stage_index] = active

    final_epsilon = float(epsilon_schedule[-1])
    masked_f = torch.where(left_positive, f, neg_inf_left)
    masked_g = torch.where(right_positive, g, neg_inf_right)
    log_plan = (masked_f.unsqueeze(2) + masked_g.unsqueeze(1) - cost_value) / final_epsilon
    plan = torch.exp(log_plan)
    entropy_terms = torch.where(
        plan > 0.0,
        plan * (torch.log(torch.clamp(plan, min=torch.finfo(plan.dtype).tiny)) - 1.0),
        torch.zeros_like(plan),
    )
    primal_value = (plan * cost_value).sum(dim=(1, 2)) + final_epsilon * entropy_terms.sum(
        dim=(1, 2)
    )
    value = primal_value
    if left_mass.requires_grad:
        value = value + (f.detach() * (left_mass - left_mass.detach())).sum(dim=1)
    if right_mass.requires_grad:
        value = value + (g.detach() * (right_mass - right_mass.detach())).sum(dim=1)
    if cost_matrix.requires_grad:
        value = value + (plan.detach() * (cost_matrix - cost_matrix.detach())).sum(dim=(1, 2))
    if runtime_checks and not _ensure_bool(torch.isfinite(value).all()):
        raise ContractError("batched Sinkhorn transport cost is non-finite")

    return _BatchedSinkhornValue(
        value=value,
        final_updates=final_updates,
        iterations=iterations,
        max_iter_reached=max_iter_reached,
    )


def _sinkhorn_divergence_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cross_cost: torch.Tensor,
    left_self_cost: torch.Tensor,
    right_self_cost: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: SinkhornConfig,
    label: str,
    collect_warnings: bool = False,
    runtime_checks: bool = True,
) -> _DivergenceValue:
    if left_mass.ndim != 1 or right_mass.ndim != 1:
        raise ContractError("Sinkhorn divergence masses must be 1D")
    if cross_cost.shape != (left_mass.shape[0], right_mass.shape[0]):
        raise ContractError("cross cost matrix shape must match Sinkhorn masses")
    if left_self_cost.shape != (left_mass.shape[0], left_mass.shape[0]):
        raise ContractError("left self cost matrix shape must match left Sinkhorn mass")
    if right_self_cost.shape != (right_mass.shape[0], right_mass.shape[0]):
        raise ContractError("right self cost matrix shape must match right Sinkhorn mass")

    cross_forward = _batched_sinkhorn_transport_value(
        left_mass.unsqueeze(0),
        right_mass.unsqueeze(0),
        cross_cost.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    cross_reverse = _batched_sinkhorn_transport_value(
        right_mass.unsqueeze(0),
        left_mass.unsqueeze(0),
        cross_cost.T.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    left_self = _batched_sinkhorn_transport_value(
        left_mass.unsqueeze(0),
        left_mass.unsqueeze(0),
        left_self_cost.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    right_self = _batched_sinkhorn_transport_value(
        right_mass.unsqueeze(0),
        right_mass.unsqueeze(0),
        right_self_cost.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    divergence = (
        0.5 * (cross_forward.value[0] + cross_reverse.value[0])
        - 0.5 * left_self.value[0]
        - 0.5 * right_self.value[0]
    )
    warnings: tuple[Mapping[str, Any], ...] = ()
    if collect_warnings:
        warnings = (
            _compact_convergence_warnings(
                cross_forward,
                label=f"{label}.cross_forward",
                epsilon_schedule=epsilon_schedule,
                config=config,
            )
            + _compact_convergence_warnings(
                cross_reverse,
                label=f"{label}.cross_reverse",
                epsilon_schedule=epsilon_schedule,
                config=config,
            )
            + _compact_convergence_warnings(
                left_self,
                label=f"{label}.left_self",
                epsilon_schedule=epsilon_schedule,
                config=config,
            )
            + _compact_convergence_warnings(
                right_self,
                label=f"{label}.right_self",
                epsilon_schedule=epsilon_schedule,
                config=config,
            )
        )
    return _DivergenceValue(
        value=divergence,
        warnings=warnings,
    )


def _batched_sinkhorn_divergence_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cross_cost: torch.Tensor,
    left_self_cost: torch.Tensor,
    right_self_cost: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: SinkhornConfig,
    label: str,
    collect_warnings: bool = False,
    runtime_checks: bool = True,
) -> _DivergenceValue:
    """Compute debiased Sinkhorn divergence for a batch of FOV-bag problems.

    This is an execution-shape optimization of the canonical balanced Sinkhorn
    operator. It preserves the epsilon schedule, max_iter, tol, dtype, debiasing
    formula, and warning semantics. The equal-size branch fuses the four transport
    calls only when their mass/cost shapes can be concatenated safely.
    """
    if left_mass.ndim != 2 or right_mass.ndim != 2:
        raise ContractError("batched Sinkhorn divergence masses must be [B,N], [B,M]")
    if cross_cost.ndim != 3 or left_self_cost.ndim != 3 or right_self_cost.ndim != 3:
        raise ContractError("batched Sinkhorn divergence costs must be 3D")
    batch_size = int(left_mass.shape[0])
    if batch_size <= 0:
        raise ContractError("batched Sinkhorn divergence requires at least one item")
    if right_mass.shape[0] != batch_size:
        raise ContractError("batched Sinkhorn divergence mass batch dimensions must align")
    if cross_cost.shape != (batch_size, left_mass.shape[1], right_mass.shape[1]):
        raise ContractError("cross cost batch shape must match divergence masses")
    if left_self_cost.shape != (batch_size, left_mass.shape[1], left_mass.shape[1]):
        raise ContractError("left self cost batch shape must match left masses")
    if right_self_cost.shape != (batch_size, right_mass.shape[1], right_mass.shape[1]):
        raise ContractError("right self cost batch shape must match right masses")

    if left_mass.shape[1] == right_mass.shape[1]:
        all_left = torch.cat((left_mass, right_mass, left_mass, right_mass), dim=0)
        all_right = torch.cat((right_mass, left_mass, left_mass, right_mass), dim=0)
        all_cost = torch.cat(
            (
                cross_cost,
                cross_cost.transpose(1, 2),
                left_self_cost,
                right_self_cost,
            ),
            dim=0,
        )
        transport = _batched_sinkhorn_transport_value(
            all_left,
            all_right,
            all_cost,
            epsilon_schedule=epsilon_schedule,
            config=config,
            runtime_checks=runtime_checks,
        )
        values = transport.value.reshape(4, batch_size)
        divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
        if collect_warnings:
            final_updates = transport.final_updates.reshape(
                4,
                batch_size,
                transport.final_updates.shape[1],
            )
            iterations = transport.iterations.reshape(
                4,
                batch_size,
                transport.iterations.shape[1],
            )
            max_iter_reached = transport.max_iter_reached.reshape(
                4,
                batch_size,
                transport.max_iter_reached.shape[1],
            )
            transport_parts = tuple(
                _BatchedSinkhornValue(
                    value=values[index],
                    final_updates=final_updates[index],
                    iterations=iterations[index],
                    max_iter_reached=max_iter_reached[index],
                )
                for index in range(4)
            )
            transports = (
                (transport_parts[0], f"{label}.cross_forward"),
                (transport_parts[1], f"{label}.cross_reverse"),
                (transport_parts[2], f"{label}.left_self"),
                (transport_parts[3], f"{label}.right_self"),
            )
        else:
            transports = ()
    else:
        cross_forward = _batched_sinkhorn_transport_value(
            left_mass,
            right_mass,
            cross_cost,
            epsilon_schedule=epsilon_schedule,
            config=config,
            runtime_checks=runtime_checks,
        )
        cross_reverse = _batched_sinkhorn_transport_value(
            right_mass,
            left_mass,
            cross_cost.transpose(1, 2),
            epsilon_schedule=epsilon_schedule,
            config=config,
            runtime_checks=runtime_checks,
        )
        left_self = _batched_sinkhorn_transport_value(
            left_mass,
            left_mass,
            left_self_cost,
            epsilon_schedule=epsilon_schedule,
            config=config,
            runtime_checks=runtime_checks,
        )
        right_self = _batched_sinkhorn_transport_value(
            right_mass,
            right_mass,
            right_self_cost,
            epsilon_schedule=epsilon_schedule,
            config=config,
            runtime_checks=runtime_checks,
        )
        divergence = (
            0.5 * (cross_forward.value + cross_reverse.value)
            - 0.5 * left_self.value
            - 0.5 * right_self.value
        )
        transports = (
            (cross_forward, f"{label}.cross_forward"),
            (cross_reverse, f"{label}.cross_reverse"),
            (left_self, f"{label}.left_self"),
            (right_self, f"{label}.right_self"),
        )
    warnings: tuple[Mapping[str, Any], ...] = ()
    if collect_warnings:
        warnings = tuple(
            warning
            for transport, transport_label in transports
            for warning in _compact_convergence_warnings(
                transport,
                label=transport_label,
                epsilon_schedule=epsilon_schedule,
                config=config,
            )
        )
    return _DivergenceValue(value=divergence, warnings=warnings)


def _pairwise_composition_ground_cost_value(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: SinkhornConfig,
    label: str,
    collect_warnings: bool = False,
    runtime_checks: bool = True,
) -> _GroundCostValue:
    if left.ndim != 2 or right.ndim != 2 or C_norm.ndim != 2:
        raise ContractError(f"{label} inner composition inputs must be [N,K], [M,K], [K,K]")
    if left.shape[1] != right.shape[1] or C_norm.shape != (left.shape[1], left.shape[1]):
        raise ContractError(f"{label} inner composition K-state axes must align")

    n_left = int(left.shape[0])
    n_right = int(right.shape[0])
    n_states = int(left.shape[1])
    if n_left <= 0 or n_right <= 0:
        raise ContractError(f"{label} requires at least one FOV pair")

    left_batch = left[:, None, :].expand(n_left, n_right, n_states).reshape(-1, n_states)
    right_batch = right[None, :, :].expand(n_left, n_right, n_states).reshape(-1, n_states)
    n_pairs = int(left_batch.shape[0])
    C_batch = C_norm.expand(n_pairs, -1, -1)
    C_t_batch = C_norm.T.expand(n_pairs, -1, -1)

    all_left = torch.cat((left_batch, right_batch, left_batch, right_batch), dim=0)
    all_right = torch.cat((right_batch, left_batch, left_batch, right_batch), dim=0)
    all_cost = torch.cat((C_batch, C_t_batch, C_batch, C_batch), dim=0)
    transport = _batched_sinkhorn_transport_value(
        all_left,
        all_right,
        all_cost,
        epsilon_schedule=config.inner_epsilon_schedule,
        config=config,
        runtime_checks=runtime_checks,
    )
    values = transport.value.reshape(4, n_pairs)
    divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
    clipped_values, clipped_negative, clip_warnings = _apply_small_negative_rule(
        divergence,
        label=label,
        runtime_checks=runtime_checks,
    )
    warnings = clip_warnings
    if collect_warnings:
        warnings = (
            _compact_convergence_warnings(
                transport,
                label=label,
                epsilon_schedule=config.inner_epsilon_schedule,
                config=config,
            )
            + clip_warnings
        )
    return _GroundCostValue(
        value=clipped_values.reshape(n_left, n_right),
        clipped_negative=clipped_negative,
        warnings=warnings,
    )


def _apply_small_negative_rule(
    values: torch.Tensor,
    *,
    label: str,
    runtime_checks: bool = True,
) -> tuple[torch.Tensor, bool, tuple[Mapping[str, Any], ...]]:
    if not runtime_checks:
        failed = values < -SMALL_NEGATIVE_TOL
        clipped = torch.where(values < 0.0, torch.zeros_like(values), values)
        return torch.where(failed, torch.full_like(values, torch.nan), clipped), False, ()
    if not _ensure_bool(torch.isfinite(values).all()):
        raise ContractError(f"{label} contains NaN/Inf")

    detached = values.detach()
    below_failure = detached < -SMALL_NEGATIVE_TOL
    if _ensure_bool(below_failure.any()):
        min_value = float(torch.min(detached).detach().cpu())
        raise ContractError(
            f"{label} is below the small-negative tolerance: min_value={min_value:g}"
        )

    clipped_mask = (detached < 0.0) & (detached >= -SMALL_NEGATIVE_TOL)
    if not _ensure_bool(clipped_mask.any()):
        return values, False, ()

    clipped = torch.where(values < 0.0, torch.zeros_like(values), values)
    min_value = float(torch.min(detached).detach().cpu())
    return (
        clipped,
        True,
        (
            {
                "type": "tiny_negative_clipped",
                "label": label,
                "threshold": SMALL_NEGATIVE_TOL,
                "min_value": min_value,
            },
        ),
    )


def _compact_convergence_warnings(
    transport: _BatchedSinkhornValue,
    *,
    label: str,
    epsilon_schedule: tuple[float, ...],
    config: SinkhornConfig,
) -> tuple[Mapping[str, Any], ...]:
    warnings: list[Mapping[str, Any]] = []
    for stage_index, epsilon in enumerate(epsilon_schedule):
        if stage_index >= transport.max_iter_reached.shape[1]:
            break
        max_iter_mask = transport.max_iter_reached[:, stage_index]
        if not _ensure_bool(max_iter_mask.any()):
            continue
        stage_updates = transport.final_updates[:, stage_index]
        max_update = torch.max(stage_updates[max_iter_mask])
        warnings.append(
            {
                "type": "sinkhorn_max_iter_reached",
                "label": label,
                "epsilon": float(epsilon),
                "max_final_update": float(max_update.detach().cpu()),
                "warning_tol": float(config.warning_tol),
                "exceeds_warning_tol": bool((max_update > config.warning_tol).detach().cpu()),
                "n_items": int(max_iter_mask.sum().detach().cpu()),
            }
        )
    return tuple(warnings)
