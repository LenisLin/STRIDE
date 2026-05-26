"""Torch-native balanced Sinkhorn divergence observation operator."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry

try:  # pragma: no cover - exercised through _require_torch when unavailable
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


BALANCED_SINKHORN_OPERATOR_VERSION = "D_obs^BalancedSinkhornDivergence-v1"
CANONICAL_EPSILON_SCHEDULE = (0.5, 0.2, 0.1)
CANONICAL_MAX_ITER = 100
CANONICAL_TOL = 1e-6
CANONICAL_WARNING_TOL = 1e-4
SMALL_NEGATIVE_TOL = 1e-10
INPUT_SIMPLEX_TOL = 1e-8


def _canonical_epsilon_schedule(value: Any, *, field_name: str) -> tuple[float, ...]:
    try:
        schedule = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"BalancedSinkhornDivergenceConfig.{field_name} must equal the canonical v1 schedule"
        ) from exc
    if schedule != CANONICAL_EPSILON_SCHEDULE:
        raise ContractError(
            f"BalancedSinkhornDivergenceConfig.{field_name} must equal the canonical v1 schedule"
        )
    return schedule


def _canonical_max_iter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != CANONICAL_MAX_ITER:
        raise ContractError(
            "BalancedSinkhornDivergenceConfig.max_iter must equal the canonical v1 default"
        )
    return int(value)


def _canonical_float(value: Any, *, field_name: str, expected: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"BalancedSinkhornDivergenceConfig.{field_name} must equal the canonical v1 default"
        ) from exc
    if resolved != expected:
        raise ContractError(
            f"BalancedSinkhornDivergenceConfig.{field_name} must equal the canonical v1 default"
        )
    return resolved


def _canonical_backend(value: Any) -> str:
    if str(value).strip().lower() != "torch":
        raise ContractError(
            "BalancedSinkhornDivergenceConfig.backend must equal the canonical v1 backend 'torch'"
        )
    return "torch"


def _canonical_dtype(value: Any) -> str:
    if str(value).strip().lower() != "float64":
        raise ContractError(
            "BalancedSinkhornDivergenceConfig.dtype must equal the canonical v1 dtype 'float64'"
        )
    return "float64"


def _validate_canonical_config_fields(config: BalancedSinkhornDivergenceConfig) -> None:
    _canonical_epsilon_schedule(
        config.inner_epsilon_schedule,
        field_name="inner_epsilon_schedule",
    )
    _canonical_epsilon_schedule(
        config.outer_epsilon_schedule,
        field_name="outer_epsilon_schedule",
    )
    _canonical_max_iter(config.max_iter)
    _canonical_float(config.tol, field_name="tol", expected=CANONICAL_TOL)
    _canonical_float(
        config.warning_tol,
        field_name="warning_tol",
        expected=CANONICAL_WARNING_TOL,
    )
    _canonical_backend(config.backend)
    _canonical_dtype(config.dtype)


def _resolve_config(
    config: BalancedSinkhornDivergenceConfig | None,
) -> BalancedSinkhornDivergenceConfig:
    if config is None:
        return BalancedSinkhornDivergenceConfig()
    if not isinstance(config, BalancedSinkhornDivergenceConfig):
        raise ContractError(
            "config must be None or a BalancedSinkhornDivergenceConfig instance"
        )
    _validate_canonical_config_fields(config)
    return config


@dataclass(frozen=True)
class BalancedSinkhornDivergenceConfig:
    """Fixed v1 numerical settings for the canonical observation operator."""

    inner_epsilon_schedule: tuple[float, ...] = CANONICAL_EPSILON_SCHEDULE
    outer_epsilon_schedule: tuple[float, ...] = CANONICAL_EPSILON_SCHEDULE
    max_iter: int = CANONICAL_MAX_ITER
    tol: float = CANONICAL_TOL
    warning_tol: float = CANONICAL_WARNING_TOL
    backend: str = "torch"
    dtype: str = "float64"

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
        }


@dataclass(frozen=True)
class BalancedSinkhornDivergenceResult:
    """Structured result for the canonical balanced Sinkhorn observation operator."""

    value: torch.Tensor
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    clipped_negative: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SinkhornCostComputation:
    value: torch.Tensor
    warnings: tuple[str, ...]
    final_updates: tuple[float, ...]
    iterations: tuple[int, ...]
    max_iter_reached: tuple[bool, ...]
    row_sums: torch.Tensor
    column_sums: torch.Tensor


@dataclass(frozen=True)
class _BatchedSinkhornCostComputation:
    value: torch.Tensor
    warnings: tuple[str, ...]
    final_updates: tuple[float, ...]
    iterations: tuple[int, ...]
    max_iter_reached: tuple[bool, ...]
    row_sums: torch.Tensor
    column_sums: torch.Tensor


@dataclass(frozen=True)
class _BatchedSinkhornValueComputation:
    value: torch.Tensor
    final_updates: torch.Tensor
    iterations: torch.Tensor
    max_iter_reached: torch.Tensor
    row_sums: torch.Tensor
    column_sums: torch.Tensor


@dataclass(frozen=True)
class _DivergenceComputation:
    value: torch.Tensor
    warnings: tuple[str, ...]
    final_updates: tuple[float, ...]
    iterations: tuple[int, ...]
    max_iter_reached: tuple[bool, ...]
    cross_row_sums: torch.Tensor
    cross_column_sums: torch.Tensor


@dataclass(frozen=True)
class _DivergenceValueComputation:
    value: torch.Tensor
    cross_row_sums: torch.Tensor
    cross_column_sums: torch.Tensor


@dataclass(frozen=True)
class _GroundCostComputation:
    value: torch.Tensor
    clipped_negative: bool
    warnings: tuple[str, ...]
    final_updates: tuple[float, ...]
    iterations: tuple[int, ...]
    max_iter_reached: tuple[bool, ...]


@dataclass(frozen=True)
class _GroundCostValueComputation:
    value: torch.Tensor
    clipped_negative: bool
    clipped_mask: torch.Tensor


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on optional runtime
        raise ContractError(
            "D_obs^BalancedSinkhornDivergence-v1 requires the canonical torch backend"
        )
    return torch


def _as_float64_matrix(value: Any, *, name: str, device: Any | None = None) -> torch.Tensor:
    torch_module = _require_torch()
    if torch_module.is_tensor(value):
        resolved_device = value.device if device is None else device
        return value.to(device=resolved_device, dtype=torch_module.float64)
    return torch_module.as_tensor(value, dtype=torch_module.float64, device=device)


def _ensure_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().item())


def _validate_distribution_matrix(matrix: torch.Tensor, *, name: str) -> None:
    torch_module = _require_torch()
    if matrix.ndim != 2:
        raise ContractError(f"{name} must be a 2D [N, K] distribution matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ContractError(f"{name} must be a non-empty [N, K] distribution matrix")
    if not _ensure_bool(torch_module.isfinite(matrix).all()):
        raise ContractError(f"{name} must contain only finite values")
    if _ensure_bool((matrix < 0.0).any()):
        raise ContractError(f"{name} entries must be nonnegative")

    row_sums = torch_module.sum(matrix, dim=1)
    if not torch_module.allclose(
        row_sums,
        torch_module.ones_like(row_sums),
        rtol=INPUT_SIMPLEX_TOL,
        atol=INPUT_SIMPLEX_TOL,
    ):
        raise ContractError(f"{name} rows must sum to 1.0")


def _normalized_geometry_cost(
    geometry: StateGeometry,
    *,
    n_states: int,
    device: Any,
) -> tuple[torch.Tensor, float]:
    torch_module = _require_torch()
    if not isinstance(geometry, StateGeometry):
        raise ContractError("geometry must be a StateGeometry object")

    cost_scale = float(geometry.cost_scale)
    if not math.isfinite(cost_scale) or cost_scale <= 0.0:
        raise ContractError("StateGeometry.cost_scale must be finite and strictly positive")

    C_raw = torch_module.as_tensor(geometry.cost_matrix, dtype=torch_module.float64, device=device)
    if C_raw.ndim != 2 or C_raw.shape[0] != C_raw.shape[1]:
        raise ContractError("StateGeometry.cost_matrix must be square")
    if int(C_raw.shape[0]) != int(n_states):
        raise ContractError("StateGeometry.cost_matrix must align to the input K-state axis")
    if not _ensure_bool(torch_module.isfinite(C_raw).all()):
        raise ContractError("StateGeometry.cost_matrix must contain only finite values")
    if _ensure_bool((C_raw < 0.0).any()):
        raise ContractError("StateGeometry.cost_matrix must be nonnegative")
    if not torch_module.allclose(C_raw, C_raw.T, rtol=0.0, atol=1e-12):
        raise ContractError("StateGeometry.cost_matrix must be symmetric")
    if not torch_module.allclose(
        torch_module.diagonal(C_raw),
        torch_module.zeros(C_raw.shape[0], dtype=torch_module.float64, device=device),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ContractError("StateGeometry.cost_matrix diagonal must be zero")

    off_diagonal = ~torch_module.eye(C_raw.shape[0], dtype=torch_module.bool, device=device)
    if not _ensure_bool(((C_raw > 0.0) & off_diagonal).any()):
        raise ContractError("StateGeometry.cost_matrix must contain a positive off-diagonal cost")

    C_norm = C_raw / cost_scale
    if not _ensure_bool(torch_module.isfinite(C_norm).all()):
        raise ContractError("C_norm = C_raw / s_C must contain only finite values")
    return C_norm, cost_scale


def _safe_log_distribution(distribution: torch.Tensor) -> torch.Tensor:
    torch_module = _require_torch()
    logged = torch_module.full_like(distribution, -torch_module.inf)
    positive = distribution > 0.0
    if _ensure_bool(positive.any()):
        logged = logged.clone()
        logged[positive] = torch_module.log(distribution[positive])
    return logged


def _max_update(
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    torch_module = _require_torch()
    if not _ensure_bool(positive_mask.any()):
        return torch_module.zeros((), dtype=new_value.dtype, device=new_value.device)
    diff = torch_module.abs(new_value[positive_mask] - old_value[positive_mask])
    if diff.numel() == 0:
        return torch_module.zeros((), dtype=new_value.dtype, device=new_value.device)
    return torch_module.max(diff)


def _marginal_residual(
    *,
    f: torch.Tensor,
    g: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    left_positive: torch.Tensor,
    right_positive: torch.Tensor,
) -> torch.Tensor:
    torch_module = _require_torch()
    neg_inf_left = torch_module.full_like(left_mass, -torch_module.inf)
    neg_inf_right = torch_module.full_like(right_mass, -torch_module.inf)
    masked_f = torch_module.where(left_positive, f, neg_inf_left)
    masked_g = torch_module.where(right_positive, g, neg_inf_right)
    log_plan = (masked_f.unsqueeze(1) + masked_g.unsqueeze(0) - cost_matrix) / float(epsilon)
    plan = torch_module.exp(log_plan)
    row_residual = torch_module.max(torch_module.abs(torch_module.sum(plan, dim=1) - left_mass))
    column_residual = torch_module.max(torch_module.abs(torch_module.sum(plan, dim=0) - right_mass))
    return torch_module.maximum(row_residual, column_residual)


def _transport_value_with_envelope_gradient(
    *,
    primal_value: torch.Tensor,
    plan: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cost_matrix: torch.Tensor,
) -> torch.Tensor:
    """Attach OT envelope gradients without differentiating through iterations."""
    value = primal_value
    if left_mass.requires_grad:
        value = value + torch.sum(f.detach() * (left_mass - left_mass.detach()))
    if right_mass.requires_grad:
        value = value + torch.sum(g.detach() * (right_mass - right_mass.detach()))
    if cost_matrix.requires_grad:
        value = value + torch.sum(plan.detach() * (cost_matrix - cost_matrix.detach()))
    return value


def _sinkhorn_transport_cost(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cost_matrix: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _SinkhornCostComputation:
    """Return balanced entropic OT under the final epsilon stage."""
    if left_mass.ndim != 1 or right_mass.ndim != 1:
        raise ContractError(f"{label} Sinkhorn masses must be 1D")
    if cost_matrix.shape != (left_mass.shape[0], right_mass.shape[0]):
        raise ContractError(f"{label} cost matrix shape must match Sinkhorn masses")
    transport = _batched_sinkhorn_transport_cost(
        left_mass.unsqueeze(0),
        right_mass.unsqueeze(0),
        cost_matrix.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
        labels=(label,),
    )
    return _SinkhornCostComputation(
        value=transport.value[0],
        warnings=transport.warnings,
        final_updates=transport.final_updates,
        iterations=transport.iterations,
        max_iter_reached=transport.max_iter_reached,
        row_sums=transport.row_sums[0],
        column_sums=transport.column_sums[0],
    )


def _batched_sinkhorn_transport_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cost_matrix: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
) -> _BatchedSinkhornValueComputation:
    """Vectorized balanced entropic OT values without Python diagnostics."""
    torch_module = _require_torch()
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
    neg_inf_left = torch_module.full_like(left_value, -torch_module.inf)
    neg_inf_right = torch_module.full_like(right_value, -torch_module.inf)

    f = torch_module.zeros_like(left_value)
    g = torch_module.zeros_like(right_value)
    n_stages = len(epsilon_schedule)
    final_updates_by_item = torch_module.full(
        (batch_size, n_stages),
        torch_module.inf,
        dtype=left_value.dtype,
        device=left_value.device,
    )
    iterations_by_item = torch_module.zeros(
        (batch_size, n_stages),
        dtype=torch_module.int64,
        device=left_value.device,
    )
    max_iter_by_item = torch_module.zeros(
        (batch_size, n_stages),
        dtype=torch_module.bool,
        device=left_value.device,
    )

    with torch_module.no_grad():
        for stage_index, epsilon in enumerate(epsilon_schedule):
            eps = float(epsilon)
            active = torch_module.ones(batch_size, dtype=torch_module.bool, device=left_value.device)
            first_converged_residual = torch_module.full(
                (batch_size,),
                torch_module.inf,
                dtype=left_value.dtype,
                device=left_value.device,
            )
            first_converged_iteration = torch_module.zeros(
                (batch_size,),
                dtype=torch_module.int64,
                device=left_value.device,
            )
            last_residual = torch_module.full(
                (batch_size,),
                torch_module.inf,
                dtype=left_value.dtype,
                device=left_value.device,
            )
            saw_nonfinite_update = torch_module.zeros(
                (batch_size,),
                dtype=torch_module.bool,
                device=left_value.device,
            )
            saw_nonfinite_residual = torch_module.zeros(
                (batch_size,),
                dtype=torch_module.bool,
                device=left_value.device,
            )
            for iteration in range(1, int(config.max_iter) + 1):
                masked_g = torch_module.where(right_positive, g, neg_inf_right)
                proposed_f = eps * (
                    log_left - torch_module.logsumexp(
                        (masked_g.unsqueeze(1) - cost_value) / eps,
                        dim=2,
                    )
                )
                next_f = torch_module.where(left_positive, proposed_f, torch_module.zeros_like(proposed_f))
                saw_nonfinite_update = saw_nonfinite_update | ~torch_module.isfinite(next_f).all(dim=1)
                active_left = active.unsqueeze(1)
                f = torch_module.where(active_left, next_f, f)

                masked_f = torch_module.where(left_positive, f, neg_inf_left)
                proposed_g = eps * (
                    log_right - torch_module.logsumexp(
                        (masked_f.unsqueeze(2) - cost_value) / eps,
                        dim=1,
                    )
                )
                next_g = torch_module.where(right_positive, proposed_g, torch_module.zeros_like(proposed_g))
                saw_nonfinite_update = saw_nonfinite_update | ~torch_module.isfinite(next_g).all(dim=1)
                active_right = active.unsqueeze(1)
                g = torch_module.where(active_right, next_g, g)

                masked_f_plan = torch_module.where(left_positive, f, neg_inf_left)
                masked_g_plan = torch_module.where(right_positive, g, neg_inf_right)
                log_plan = (masked_f_plan.unsqueeze(2) + masked_g_plan.unsqueeze(1) - cost_value) / eps
                plan = torch_module.exp(log_plan)
                row_residual = torch_module.amax(torch_module.abs(plan.sum(dim=2) - left_value), dim=1)
                column_residual = torch_module.amax(torch_module.abs(plan.sum(dim=1) - right_value), dim=1)
                residual = torch_module.maximum(row_residual, column_residual)
                saw_nonfinite_residual = saw_nonfinite_residual | ~torch_module.isfinite(residual)
                last_residual = torch_module.where(active, residual, last_residual)

                converged_now = active & torch_module.isfinite(residual) & (residual <= float(config.tol))
                first_converged_residual = torch_module.where(
                    converged_now,
                    residual,
                    first_converged_residual,
                )
                first_converged_iteration = torch_module.where(
                    converged_now,
                    torch_module.full_like(first_converged_iteration, iteration),
                    first_converged_iteration,
                )
                active = active & ~converged_now

            if _ensure_bool(saw_nonfinite_update.any()):
                raise ContractError("batched Sinkhorn iteration produced a non-finite update")
            if _ensure_bool(saw_nonfinite_residual.any()):
                raise ContractError("batched Sinkhorn iteration produced a non-finite residual")

            stage_final_updates = torch_module.where(active, last_residual, first_converged_residual)
            stage_iterations = torch_module.where(
                active,
                torch_module.full_like(first_converged_iteration, int(config.max_iter)),
                first_converged_iteration,
            )
            stage_max_iter = active
            final_updates_by_item[:, stage_index] = stage_final_updates
            iterations_by_item[:, stage_index] = stage_iterations
            max_iter_by_item[:, stage_index] = stage_max_iter

    final_epsilon = float(epsilon_schedule[-1])
    masked_f = torch_module.where(left_positive, f, neg_inf_left)
    masked_g = torch_module.where(right_positive, g, neg_inf_right)
    log_plan = (masked_f.unsqueeze(2) + masked_g.unsqueeze(1) - cost_value) / final_epsilon
    plan = torch_module.exp(log_plan)
    positive_plan = plan > 0.0
    entropy_terms = torch_module.zeros_like(plan)
    if _ensure_bool(positive_plan.any()):
        entropy_terms = torch_module.where(
            positive_plan,
            plan * (torch_module.log(torch_module.clamp(plan, min=torch_module.finfo(plan.dtype).tiny)) - 1.0),
            torch_module.zeros_like(plan),
        )
    primal_value = (plan * cost_value).sum(dim=(1, 2)) + final_epsilon * entropy_terms.sum(dim=(1, 2))
    value = primal_value
    if left_mass.requires_grad:
        value = value + (f.detach() * (left_mass - left_mass.detach())).sum(dim=1)
    if right_mass.requires_grad:
        value = value + (g.detach() * (right_mass - right_mass.detach())).sum(dim=1)
    if cost_matrix.requires_grad:
        value = value + (plan.detach() * (cost_matrix - cost_matrix.detach())).sum(dim=(1, 2))
    if not _ensure_bool(torch_module.isfinite(value).all()):
        raise ContractError("batched Sinkhorn transport cost is non-finite")

    return _BatchedSinkhornValueComputation(
        value=value,
        final_updates=final_updates_by_item,
        iterations=iterations_by_item,
        max_iter_reached=max_iter_by_item,
        row_sums=plan.sum(dim=2),
        column_sums=plan.sum(dim=1),
    )


def _batched_sinkhorn_transport_cost(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cost_matrix: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
    labels: tuple[str, ...],
) -> _BatchedSinkhornCostComputation:
    """Vectorized batch of independent balanced entropic OT computations."""
    torch_module = _require_torch()
    batch_size = int(left_mass.shape[0]) if left_mass.ndim > 0 else 0
    if len(labels) != batch_size:
        raise ContractError("batched Sinkhorn labels must align to the batch dimension")

    transport = _batched_sinkhorn_transport_value(
        left_mass,
        right_mass,
        cost_matrix,
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    warnings: list[str] = []
    final_updates_by_item = transport.final_updates
    max_iter_by_item = transport.max_iter_reached
    for stage_index, epsilon in enumerate(epsilon_schedule):
        eps = float(epsilon)
        stage_max_iter = max_iter_by_item[:, stage_index]
        stage_final_updates = final_updates_by_item[:, stage_index]
        warning_items = tuple(
            int(item)
            for item in torch_module.nonzero(stage_max_iter, as_tuple=False).reshape(-1).detach().cpu().tolist()
        )
        for item_index in warning_items:
            final_update = float(stage_final_updates[item_index].detach().cpu())
            warning_text = (
                f"{labels[item_index]} Sinkhorn epsilon={eps:g} reached max_iter "
                f"with update={final_update:g}"
            )
            if final_update > float(config.warning_tol):
                warning_text = (
                    f"{warning_text}; final update exceeds "
                    f"warning_tol {float(config.warning_tol):g}"
                )
            warnings.append(warning_text)

    final_updates_tensor = transport.final_updates.detach().cpu().reshape(-1)
    iterations_tensor = transport.iterations.detach().cpu().reshape(-1)
    max_iter_tensor = transport.max_iter_reached.detach().cpu().reshape(-1)
    return _BatchedSinkhornCostComputation(
        value=transport.value,
        warnings=tuple(warnings),
        final_updates=tuple(float(item) for item in final_updates_tensor.tolist()),
        iterations=tuple(int(item) for item in iterations_tensor.tolist()),
        max_iter_reached=tuple(bool(item) for item in max_iter_tensor.tolist()),
        row_sums=transport.row_sums,
        column_sums=transport.column_sums,
    )


def _merge_cost_diagnostics(
    computations: tuple[_SinkhornCostComputation, ...],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...], tuple[bool, ...]]:
    warnings: list[str] = []
    final_updates: list[float] = []
    iterations: list[int] = []
    max_iter_reached: list[bool] = []
    for computation in computations:
        warnings.extend(computation.warnings)
        final_updates.extend(computation.final_updates)
        iterations.extend(computation.iterations)
        max_iter_reached.extend(computation.max_iter_reached)
    return (
        tuple(warnings),
        tuple(final_updates),
        tuple(iterations),
        tuple(max_iter_reached),
    )


def _sinkhorn_divergence(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cross_cost: torch.Tensor,
    left_self_cost: torch.Tensor,
    right_self_cost: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _DivergenceComputation:
    cross_forward = _sinkhorn_transport_cost(
        left_mass,
        right_mass,
        cross_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
        label=f"{label}.cross_forward",
    )
    cross_reverse = _sinkhorn_transport_cost(
        right_mass,
        left_mass,
        cross_cost.T,
        epsilon_schedule=epsilon_schedule,
        config=config,
        label=f"{label}.cross_reverse",
    )
    left_self = _sinkhorn_transport_cost(
        left_mass,
        left_mass,
        left_self_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
        label=f"{label}.left_self",
    )
    right_self = _sinkhorn_transport_cost(
        right_mass,
        right_mass,
        right_self_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
        label=f"{label}.right_self",
    )
    warnings, final_updates, iterations, max_iter_reached = _merge_cost_diagnostics(
        (cross_forward, cross_reverse, left_self, right_self)
    )
    return _DivergenceComputation(
        value=0.5 * (cross_forward.value + cross_reverse.value)
        - 0.5 * left_self.value
        - 0.5 * right_self.value,
        warnings=warnings,
        final_updates=final_updates,
        iterations=iterations,
        max_iter_reached=max_iter_reached,
        cross_row_sums=cross_forward.row_sums,
        cross_column_sums=cross_forward.column_sums,
    )


def _sinkhorn_divergence_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cross_cost: torch.Tensor,
    left_self_cost: torch.Tensor,
    right_self_cost: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
) -> _DivergenceValueComputation:
    """Return Sinkhorn divergence value without Python convergence diagnostics."""
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
    )
    cross_reverse = _batched_sinkhorn_transport_value(
        right_mass.unsqueeze(0),
        left_mass.unsqueeze(0),
        cross_cost.T.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    left_self = _batched_sinkhorn_transport_value(
        left_mass.unsqueeze(0),
        left_mass.unsqueeze(0),
        left_self_cost.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    right_self = _batched_sinkhorn_transport_value(
        right_mass.unsqueeze(0),
        right_mass.unsqueeze(0),
        right_self_cost.unsqueeze(0),
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    return _DivergenceValueComputation(
        value=0.5 * (cross_forward.value[0] + cross_reverse.value[0])
        - 0.5 * left_self.value[0]
        - 0.5 * right_self.value[0],
        cross_row_sums=cross_forward.row_sums[0],
        cross_column_sums=cross_forward.column_sums[0],
    )


def _batched_sinkhorn_divergence_value(
    left_mass: torch.Tensor,
    right_mass: torch.Tensor,
    cross_cost: torch.Tensor,
    left_self_cost: torch.Tensor,
    right_self_cost: torch.Tensor,
    *,
    epsilon_schedule: tuple[float, ...],
    config: BalancedSinkhornDivergenceConfig,
) -> _BatchedSinkhornValueComputation:
    """Return batched Sinkhorn divergence values for same-shape cost tensors."""
    if left_mass.ndim != 2 or right_mass.ndim != 2:
        raise ContractError("batched Sinkhorn divergence masses must be [B,N] and [B,M]")
    batch_size = int(left_mass.shape[0])
    if batch_size <= 0 or int(right_mass.shape[0]) != batch_size:
        raise ContractError("batched Sinkhorn divergence batch dimensions must align")
    if cross_cost.shape != (batch_size, left_mass.shape[1], right_mass.shape[1]):
        raise ContractError("batched cross cost shape must match Sinkhorn masses")
    if left_self_cost.shape != (batch_size, left_mass.shape[1], left_mass.shape[1]):
        raise ContractError("batched left self cost shape must match left Sinkhorn mass")
    if right_self_cost.shape != (batch_size, right_mass.shape[1], right_mass.shape[1]):
        raise ContractError("batched right self cost shape must match right Sinkhorn mass")

    cross_forward = _batched_sinkhorn_transport_value(
        left_mass,
        right_mass,
        cross_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    cross_reverse = _batched_sinkhorn_transport_value(
        right_mass,
        left_mass,
        cross_cost.transpose(1, 2),
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    left_self = _batched_sinkhorn_transport_value(
        left_mass,
        left_mass,
        left_self_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    right_self = _batched_sinkhorn_transport_value(
        right_mass,
        right_mass,
        right_self_cost,
        epsilon_schedule=epsilon_schedule,
        config=config,
    )
    return _BatchedSinkhornValueComputation(
        value=0.5 * (cross_forward.value + cross_reverse.value)
        - 0.5 * left_self.value
        - 0.5 * right_self.value,
        final_updates=cross_forward.final_updates,
        iterations=cross_forward.iterations,
        max_iter_reached=cross_forward.max_iter_reached,
        row_sums=cross_forward.row_sums,
        column_sums=cross_forward.column_sums,
    )


def _apply_small_negative_rule(
    values: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, bool, tuple[str, ...]]:
    """Clamp tiny negative Sinkhorn-divergence values and fail on larger negatives."""
    torch_module = _require_torch()
    tensor = values if torch_module.is_tensor(values) else torch_module.as_tensor(values)
    if not _ensure_bool(torch_module.isfinite(tensor).all()):
        raise ContractError(f"{label} contains NaN/Inf")

    detached = tensor.detach()
    below_failure = detached < -SMALL_NEGATIVE_TOL
    if _ensure_bool(below_failure.any()):
        min_value = float(torch_module.min(detached).cpu())
        raise ContractError(
            f"{label} is below the small-negative tolerance: min_value={min_value:g}"
        )

    clipped_mask = (detached < 0.0) & (detached >= -SMALL_NEGATIVE_TOL)
    if not _ensure_bool(clipped_mask.any()):
        return tensor, False, ()

    clipped = torch_module.where(tensor < 0.0, torch_module.zeros_like(tensor), tensor)
    min_value = float(torch_module.min(detached).cpu())
    return (
        clipped,
        True,
        (f"{label} clipped tiny negative Sinkhorn divergence value min={min_value:g}",),
    )


def _pairwise_composition_ground_cost_loop(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _GroundCostComputation:
    values: list[torch.Tensor] = []
    warnings: list[str] = []
    final_updates: list[float] = []
    iterations: list[int] = []
    max_iter_reached: list[bool] = []
    clipped_negative = False

    for left_index in range(left.shape[0]):
        for right_index in range(right.shape[0]):
            computation = _sinkhorn_divergence(
                left[left_index],
                right[right_index],
                C_norm,
                C_norm,
                C_norm,
                epsilon_schedule=config.inner_epsilon_schedule,
                config=config,
                label=f"{label}[{left_index},{right_index}]",
            )
            clipped_value, clipped, clipped_warnings = _apply_small_negative_rule(
                computation.value,
                label=f"{label}[{left_index},{right_index}]",
            )
            values.append(clipped_value)
            clipped_negative = clipped_negative or clipped
            warnings.extend(computation.warnings)
            warnings.extend(clipped_warnings)
            final_updates.extend(computation.final_updates)
            iterations.extend(computation.iterations)
            max_iter_reached.extend(computation.max_iter_reached)

    return _GroundCostComputation(
        value=torch.stack(values).reshape(left.shape[0], right.shape[0]),
        clipped_negative=clipped_negative,
        warnings=tuple(warnings),
        final_updates=tuple(final_updates),
        iterations=tuple(iterations),
        max_iter_reached=tuple(max_iter_reached),
    )


def _pairwise_composition_ground_cost_batched(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _GroundCostComputation:
    torch_module = _require_torch()
    if left.ndim != 2 or right.ndim != 2 or C_norm.ndim != 2:
        raise ContractError(f"{label} inner composition inputs must be [N,K], [M,K], [K,K]")
    if left.shape[1] != right.shape[1] or C_norm.shape != (left.shape[1], left.shape[1]):
        raise ContractError(f"{label} inner composition K-state axes must align")

    pairs = tuple(
        (left_index, right_index)
        for left_index in range(int(left.shape[0]))
        for right_index in range(int(right.shape[0]))
    )
    if not pairs:
        raise ContractError(f"{label} requires at least one FOV pair")
    left_batch = torch_module.stack([left[left_index] for left_index, _ in pairs], dim=0)
    right_batch = torch_module.stack([right[right_index] for _, right_index in pairs], dim=0)
    n_pairs = len(pairs)
    C_batch = C_norm.expand(n_pairs, -1, -1)
    C_t_batch = C_norm.T.expand(n_pairs, -1, -1)

    all_left = torch_module.cat(
        (left_batch, right_batch, left_batch, right_batch),
        dim=0,
    )
    all_right = torch_module.cat(
        (right_batch, left_batch, left_batch, right_batch),
        dim=0,
    )
    all_cost = torch_module.cat((C_batch, C_t_batch, C_batch, C_batch), dim=0)
    transport_labels = tuple(
        transport_label
        for suffix in ("cross_forward", "cross_reverse", "left_self", "right_self")
        for left_index, right_index in pairs
        for transport_label in (f"{label}[{left_index},{right_index}].{suffix}",)
    )
    transport = _batched_sinkhorn_transport_cost(
        all_left,
        all_right,
        all_cost,
        epsilon_schedule=config.inner_epsilon_schedule,
        config=config,
        labels=transport_labels,
    )
    values = transport.value.reshape(4, n_pairs)
    divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
    clipped_value_items: list[torch.Tensor] = []
    clipped_warnings: list[str] = []
    clipped_negative = False
    for pair_index, (left_index, right_index) in enumerate(pairs):
        clipped_value, clipped, warnings = _apply_small_negative_rule(
            divergence[pair_index],
            label=f"{label}[{left_index},{right_index}]",
        )
        clipped_value_items.append(clipped_value)
        clipped_negative = clipped_negative or clipped
        clipped_warnings.extend(warnings)
    clipped_values = torch_module.stack(clipped_value_items)
    n_stages = len(config.inner_epsilon_schedule)
    final_updates: list[float] = []
    iterations: list[int] = []
    max_iter_reached: list[bool] = []
    for pair_index in range(n_pairs):
        for suffix_index in range(4):
            item_index = suffix_index * n_pairs + pair_index
            start = item_index * n_stages
            stop = start + n_stages
            final_updates.extend(transport.final_updates[start:stop])
            iterations.extend(transport.iterations[start:stop])
            max_iter_reached.extend(transport.max_iter_reached[start:stop])
    return _GroundCostComputation(
        value=clipped_values.reshape(left.shape[0], right.shape[0]),
        clipped_negative=bool(clipped_negative),
        warnings=tuple(transport.warnings) + tuple(clipped_warnings),
        final_updates=tuple(final_updates),
        iterations=tuple(iterations),
        max_iter_reached=tuple(max_iter_reached),
    )


def _pairwise_composition_ground_cost_value(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _GroundCostValueComputation:
    """Return FOV-pair composition ground costs without Python diagnostics."""
    torch_module = _require_torch()
    if left.ndim != 2 or right.ndim != 2 or C_norm.ndim != 2:
        raise ContractError(f"{label} inner composition inputs must be [N,K], [M,K], [K,K]")
    if left.shape[1] != right.shape[1] or C_norm.shape != (left.shape[1], left.shape[1]):
        raise ContractError(f"{label} inner composition K-state axes must align")

    pairs = tuple(
        (left_index, right_index)
        for left_index in range(int(left.shape[0]))
        for right_index in range(int(right.shape[0]))
    )
    if not pairs:
        raise ContractError(f"{label} requires at least one FOV pair")
    left_batch = torch_module.stack([left[left_index] for left_index, _ in pairs], dim=0)
    right_batch = torch_module.stack([right[right_index] for _, right_index in pairs], dim=0)
    n_pairs = len(pairs)
    C_batch = C_norm.expand(n_pairs, -1, -1)
    C_t_batch = C_norm.T.expand(n_pairs, -1, -1)

    all_left = torch_module.cat(
        (left_batch, right_batch, left_batch, right_batch),
        dim=0,
    )
    all_right = torch_module.cat(
        (right_batch, left_batch, left_batch, right_batch),
        dim=0,
    )
    all_cost = torch_module.cat((C_batch, C_t_batch, C_batch, C_batch), dim=0)
    transport = _batched_sinkhorn_transport_value(
        all_left,
        all_right,
        all_cost,
        epsilon_schedule=config.inner_epsilon_schedule,
        config=config,
    )
    values = transport.value.reshape(4, n_pairs)
    divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
    clipped_mask = (divergence.detach() < 0.0) & (divergence.detach() >= -SMALL_NEGATIVE_TOL)
    clipped_values, clipped_negative, _ = _apply_small_negative_rule(
        divergence,
        label=label,
    )
    return _GroundCostValueComputation(
        value=clipped_values.reshape(left.shape[0], right.shape[0]),
        clipped_negative=bool(clipped_negative),
        clipped_mask=clipped_mask.reshape(left.shape[0], right.shape[0]),
    )


def _batched_pairwise_composition_ground_cost_value(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _GroundCostValueComputation:
    """Return grouped FOV-pair composition ground costs for same-shape blocks."""
    torch_module = _require_torch()
    if left.ndim != 3 or right.ndim != 3 or C_norm.ndim != 2:
        raise ContractError(f"{label} grouped composition inputs must be [B,N,K], [B,M,K], [K,K]")
    batch_size, n_left, n_states = (int(left.shape[0]), int(left.shape[1]), int(left.shape[2]))
    if batch_size <= 0 or int(right.shape[0]) != batch_size:
        raise ContractError(f"{label} grouped composition batch dimensions must align")
    n_right = int(right.shape[1])
    if n_left <= 0 or n_right <= 0:
        raise ContractError(f"{label} requires non-empty grouped FOV axes")
    if int(right.shape[2]) != n_states or C_norm.shape != (n_states, n_states):
        raise ContractError(f"{label} grouped composition K-state axes must align")

    left_batch = (
        left.unsqueeze(2)
        .expand(batch_size, n_left, n_right, n_states)
        .reshape(batch_size * n_left * n_right, n_states)
    )
    right_batch = (
        right.unsqueeze(1)
        .expand(batch_size, n_left, n_right, n_states)
        .reshape(batch_size * n_left * n_right, n_states)
    )
    n_pairs = int(left_batch.shape[0])
    C_batch = C_norm.expand(n_pairs, -1, -1)
    C_t_batch = C_norm.T.expand(n_pairs, -1, -1)
    all_left = torch_module.cat(
        (left_batch, right_batch, left_batch, right_batch),
        dim=0,
    )
    all_right = torch_module.cat(
        (right_batch, left_batch, left_batch, right_batch),
        dim=0,
    )
    all_cost = torch_module.cat((C_batch, C_t_batch, C_batch, C_batch), dim=0)
    transport = _batched_sinkhorn_transport_value(
        all_left,
        all_right,
        all_cost,
        epsilon_schedule=config.inner_epsilon_schedule,
        config=config,
    )
    values = transport.value.reshape(4, n_pairs)
    divergence = 0.5 * (values[0] + values[1]) - 0.5 * values[2] - 0.5 * values[3]
    clipped_mask = (divergence.detach() < 0.0) & (divergence.detach() >= -SMALL_NEGATIVE_TOL)
    clipped_values, clipped_negative, _ = _apply_small_negative_rule(
        divergence,
        label=label,
    )
    return _GroundCostValueComputation(
        value=clipped_values.reshape(batch_size, n_left, n_right),
        clipped_negative=bool(clipped_negative),
        clipped_mask=clipped_mask.reshape(batch_size, n_left, n_right),
    )


def _pairwise_composition_ground_cost(
    left: torch.Tensor,
    right: torch.Tensor,
    C_norm: torch.Tensor,
    *,
    config: BalancedSinkhornDivergenceConfig,
    label: str,
) -> _GroundCostComputation:
    return _pairwise_composition_ground_cost_batched(
        left,
        right,
        C_norm,
        config=config,
        label=label,
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _observed_self_ground_cost_cache_key(
    observed: torch.Tensor,
    C_norm: torch.Tensor,
    config: BalancedSinkhornDivergenceConfig,
) -> tuple[object, ...]:
    return (
        "observed_self_ground_cost_v1",
        str(observed.device),
        tuple(int(item) for item in observed.shape),
        _tensor_content_digest(observed),
        tuple(int(item) for item in C_norm.shape),
        _tensor_content_digest(C_norm),
        tuple(float(item) for item in config.inner_epsilon_schedule),
        int(config.max_iter),
        float(config.tol),
        float(config.warning_tol),
        str(config.backend),
        str(config.dtype),
    )


def _validate_fov_cost_scale(
    fov_cost_scale: float | None,
    *,
    floor_used: bool,
) -> float:
    if not isinstance(floor_used, bool):
        raise ContractError("s_G_init_floor_used must be a bool")
    if fov_cost_scale is None:
        raise ContractError("fov_cost_scale must be provided explicitly as s_G_init")
    scale = float(fov_cost_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("s_G_init must be finite and strictly positive")
    if floor_used and scale != 1.0:
        raise ContractError("s_G_init_floor_used=True requires explicit fallback s_G_init=1.0")
    return scale


def _tensor_to_float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _convergence_metadata(
    *,
    inner: tuple[_GroundCostComputation, ...],
    outer: _DivergenceComputation,
) -> dict[str, Any]:
    inner_updates: list[float] = []
    inner_iterations: list[int] = []
    inner_max_iter_reached: list[bool] = []
    for computation in inner:
        inner_updates.extend(computation.final_updates)
        inner_iterations.extend(computation.iterations)
        inner_max_iter_reached.extend(computation.max_iter_reached)
    return {
        "inner_max_final_update": max(inner_updates) if inner_updates else 0.0,
        "outer_max_final_update": max(outer.final_updates) if outer.final_updates else 0.0,
        "inner_max_iterations": max(inner_iterations) if inner_iterations else 0,
        "outer_max_iterations": max(outer.iterations) if outer.iterations else 0,
        "inner_max_iter_reached_count": int(sum(inner_max_iter_reached)),
        "outer_max_iter_reached_count": int(sum(outer.max_iter_reached)),
    }


def compute_balanced_sinkhorn_observation_value(
    predicted_target_fov_bag: Any,
    observed_target_fov_bag: Any,
    geometry: StateGeometry,
    *,
    fov_cost_scale: float | None = None,
    fov_cost_scale_floor_used: bool = False,
    config: BalancedSinkhornDivergenceConfig | None = None,
    observed_self_ground_cost_cache: (
        MutableMapping[tuple[object, ...], _GroundCostValueComputation] | None
    ) = None,
    observed_self_ground_cost_cache_key: tuple[object, ...] | None = None,
) -> BalancedSinkhornDivergenceResult:
    """Compute canonical observation value with compact production metadata."""
    torch_module = _require_torch()
    resolved_config = _resolve_config(config)

    predicted = _as_float64_matrix(predicted_target_fov_bag, name="predicted_target_fov_bag")
    observed = _as_float64_matrix(
        observed_target_fov_bag,
        name="observed_target_fov_bag",
        device=predicted.device,
    )
    _validate_distribution_matrix(predicted, name="predicted_target_fov_bag")
    _validate_distribution_matrix(observed, name="observed_target_fov_bag")
    if predicted.shape[1] != observed.shape[1]:
        raise ContractError("predicted and observed bags must share the same K-state axis")

    C_norm, cost_scale = _normalized_geometry_cost(
        geometry,
        n_states=int(predicted.shape[1]),
        device=predicted.device,
    )
    s_G_init = _validate_fov_cost_scale(
        fov_cost_scale,
        floor_used=fov_cost_scale_floor_used,
    )

    G_cross = _pairwise_composition_ground_cost_value(
        predicted,
        observed,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.cross",
    )
    G_pred = _pairwise_composition_ground_cost_value(
        predicted,
        predicted,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.predicted_self",
    )
    G_obs = None
    if observed_self_ground_cost_cache is not None and observed_self_ground_cost_cache_key is not None:
        G_obs = observed_self_ground_cost_cache.get(observed_self_ground_cost_cache_key)
        if G_obs is None:
            G_obs = _pairwise_composition_ground_cost_value(
                observed,
                observed,
                C_norm,
                config=resolved_config,
                label="inner_composition_distance.observed_self",
            )
            observed_self_ground_cost_cache[observed_self_ground_cost_cache_key] = G_obs
    if G_obs is None:
        G_obs = _pairwise_composition_ground_cost_value(
            observed,
            observed,
            C_norm,
            config=resolved_config,
            label="inner_composition_distance.observed_self",
        )

    predicted_fov_mass = torch_module.full(
        (predicted.shape[0],),
        1.0 / float(predicted.shape[0]),
        dtype=torch_module.float64,
        device=predicted.device,
    )
    observed_fov_mass = torch_module.full(
        (observed.shape[0],),
        1.0 / float(observed.shape[0]),
        dtype=torch_module.float64,
        device=predicted.device,
    )
    outer = _sinkhorn_divergence_value(
        predicted_fov_mass,
        observed_fov_mass,
        G_cross.value / s_G_init,
        G_pred.value / s_G_init,
        G_obs.value / s_G_init,
        epsilon_schedule=resolved_config.outer_epsilon_schedule,
        config=resolved_config,
    )
    value, outer_clipped, _ = _apply_small_negative_rule(
        outer.value,
        label="outer_fov_bag_divergence",
    )
    clipped_negative = (
        G_cross.clipped_negative
        or G_pred.clipped_negative
        or G_obs.clipped_negative
        or outer_clipped
    )
    return BalancedSinkhornDivergenceResult(
        value=value,
        status="ok_with_warnings" if clipped_negative else "ok",
        metadata={
            **resolved_config.metadata(),
            "state_geometry": {
                "normalization": "C_norm = C_raw / s_C",
                "s_C": cost_scale,
            },
        },
        clipped_negative=clipped_negative,
        warnings=(),
    )


def compute_balanced_sinkhorn_observation_discrepancy(
    predicted_target_fov_bag: Any,
    observed_target_fov_bag: Any,
    geometry: StateGeometry,
    *,
    fov_cost_scale: float | None = None,
    fov_cost_scale_floor_used: bool = False,
    config: BalancedSinkhornDivergenceConfig | None = None,
    observed_self_ground_cost_cache: (
        MutableMapping[tuple[object, ...], _GroundCostComputation] | None
    ) = None,
) -> BalancedSinkhornDivergenceResult:
    """Compute canonical ``D_obs^BalancedSinkhornDivergence-v1``.

    Inputs are normalized FOV/community-composition rows. The state-level
    geometry supplies ``C_raw`` and ``s_C``; this function uses
    ``C_norm = C_raw / s_C`` and does not infer or recompute the full-objective
    evidence-block ``s_G_init`` scale.
    """
    torch_module = _require_torch()
    resolved_config = _resolve_config(config)

    predicted = _as_float64_matrix(predicted_target_fov_bag, name="predicted_target_fov_bag")
    observed = _as_float64_matrix(
        observed_target_fov_bag,
        name="observed_target_fov_bag",
        device=predicted.device,
    )
    _validate_distribution_matrix(predicted, name="predicted_target_fov_bag")
    _validate_distribution_matrix(observed, name="observed_target_fov_bag")
    if predicted.shape[1] != observed.shape[1]:
        raise ContractError("predicted and observed bags must share the same K-state axis")

    C_norm, cost_scale = _normalized_geometry_cost(
        geometry,
        n_states=int(predicted.shape[1]),
        device=predicted.device,
    )
    s_G_init = _validate_fov_cost_scale(
        fov_cost_scale,
        floor_used=fov_cost_scale_floor_used,
    )

    G_cross = _pairwise_composition_ground_cost(
        predicted,
        observed,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.cross",
    )
    G_pred = _pairwise_composition_ground_cost(
        predicted,
        predicted,
        C_norm,
        config=resolved_config,
        label="inner_composition_distance.predicted_self",
    )
    G_obs = None
    if (
        observed_self_ground_cost_cache is not None
        and not bool(getattr(observed, "requires_grad", False))
        and not bool(getattr(C_norm, "requires_grad", False))
    ):
        cache_key = _observed_self_ground_cost_cache_key(observed, C_norm, resolved_config)
        G_obs = observed_self_ground_cost_cache.get(cache_key)
        if G_obs is None:
            G_obs = _pairwise_composition_ground_cost(
                observed,
                observed,
                C_norm,
                config=resolved_config,
                label="inner_composition_distance.observed_self",
            )
            observed_self_ground_cost_cache[cache_key] = G_obs
    if G_obs is None:
        G_obs = _pairwise_composition_ground_cost(
            observed,
            observed,
            C_norm,
            config=resolved_config,
            label="inner_composition_distance.observed_self",
        )

    predicted_fov_mass = torch_module.full(
        (predicted.shape[0],),
        1.0 / float(predicted.shape[0]),
        dtype=torch_module.float64,
        device=predicted.device,
    )
    observed_fov_mass = torch_module.full(
        (observed.shape[0],),
        1.0 / float(observed.shape[0]),
        dtype=torch_module.float64,
        device=predicted.device,
    )

    outer = _sinkhorn_divergence(
        predicted_fov_mass,
        observed_fov_mass,
        G_cross.value / s_G_init,
        G_pred.value / s_G_init,
        G_obs.value / s_G_init,
        epsilon_schedule=resolved_config.outer_epsilon_schedule,
        config=resolved_config,
        label="outer_fov_bag_divergence",
    )
    value, outer_clipped, outer_clipped_warnings = _apply_small_negative_rule(
        outer.value,
        label="outer_fov_bag_divergence",
    )

    clipped_negative = (
        G_cross.clipped_negative
        or G_pred.clipped_negative
        or G_obs.clipped_negative
        or outer_clipped
    )
    warnings = tuple(
        [
            *G_cross.warnings,
            *G_pred.warnings,
            *G_obs.warnings,
            *outer.warnings,
            *outer_clipped_warnings,
        ]
    )
    status = "ok_with_warnings" if warnings else "ok"
    metadata = {
        **resolved_config.metadata(),
        "status": status,
        "n_predicted_fovs": int(predicted.shape[0]),
        "n_observed_fovs": int(observed.shape[0]),
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": cost_scale,
            "C_norm": np.asarray(C_norm.detach().cpu(), dtype=float).tolist(),
        },
        "fov_ground_cost": {
            "normalization": "G_norm = G / s_G_init",
            "s_G_init": s_G_init,
            "s_G_init_floor_used": fov_cost_scale_floor_used,
        },
        "outer_transport_marginals": {
            "left_mass": _tensor_to_float_list(predicted_fov_mass),
            "right_mass": _tensor_to_float_list(observed_fov_mass),
            "row_sums": _tensor_to_float_list(outer.cross_row_sums),
            "column_sums": _tensor_to_float_list(outer.cross_column_sums),
        },
        "warning_flags": {
            "clipped_negative": clipped_negative,
            "has_warnings": bool(warnings),
        },
        "convergence": _convergence_metadata(
            inner=(G_cross, G_pred, G_obs),
            outer=outer,
        ),
    }
    return BalancedSinkhornDivergenceResult(
        value=value,
        status=status,
        metadata=metadata,
        clipped_negative=clipped_negative,
        warnings=warnings,
    )


__all__ = [
    "BALANCED_SINKHORN_OPERATOR_VERSION",
    "BalancedSinkhornDivergenceConfig",
    "BalancedSinkhornDivergenceResult",
    "compute_balanced_sinkhorn_observation_discrepancy",
]
