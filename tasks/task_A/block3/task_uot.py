"""Task-local soft unbalanced OT solver for the Block 3 comparator arm."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

STATUS_OK = "ok"
STATUS_EMPTY_SOURCE = "empty_source"
STATUS_EMPTY_TARGET = "empty_target"
STATUS_NUMERICAL_FAILURE = "numerical_failure"
STATUS_NOT_CONVERGED = "not_converged"


@dataclass(frozen=True)
class UOTSolverConfig:
    """Numerical controls for the Task A Block 3 soft-UOT comparator."""

    eps_schedule: tuple[float, ...] = (1.0, 0.2)
    max_iter: int = 2000
    tol: float = 1e-7

    def __post_init__(self) -> None:
        eps = np.asarray(self.eps_schedule, dtype=float)
        if eps.ndim != 1 or eps.size == 0:
            raise ValueError("eps_schedule must be a non-empty sequence")
        if not np.isfinite(eps).all() or np.any(eps <= 0.0):
            raise ValueError("eps_schedule values must be finite and positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and positive")


@dataclass(frozen=True)
class UOTBatchResult:
    """Dense plans and per-row status from a Task A Block 3 UOT solve."""

    plans: np.ndarray
    status: np.ndarray
    iterations: np.ndarray


def _validate_batch_inputs(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    match_penalties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_rows = np.asarray(source, dtype=float)
    target_rows = np.asarray(target, dtype=float)
    cost = np.asarray(cost_matrix, dtype=float)
    penalties = np.asarray(match_penalties, dtype=float)

    if source_rows.ndim != 2 or target_rows.shape != source_rows.shape:
        raise ValueError("source and target must be 2D arrays with the same shape")
    if source_rows.shape[0] == 0 or source_rows.shape[1] == 0:
        raise ValueError("source and target must contain at least one row and state")
    if not np.isfinite(source_rows).all() or not np.isfinite(target_rows).all():
        raise ValueError("source and target must contain finite values")
    if np.any(source_rows < 0.0) or np.any(target_rows < 0.0):
        raise ValueError("source and target must be non-negative")
    if cost.shape != (source_rows.shape[1], source_rows.shape[1]):
        raise ValueError("cost_matrix must match the shared state axis")
    if not np.isfinite(cost).all():
        raise ValueError("cost_matrix must contain finite values")
    if np.any(cost < 0.0):
        raise ValueError("cost_matrix must be non-negative")
    if penalties.shape != (source_rows.shape[0],):
        raise ValueError("match_penalties must have one value per input row")
    if not np.isfinite(penalties).all() or np.any(penalties <= 0.0):
        raise ValueError("match_penalties must be finite and positive")
    return source_rows, target_rows, cost, penalties


def _max_scaling_update(
    previous_u: np.ndarray,
    next_u: np.ndarray,
    previous_v: np.ndarray,
    next_v: np.ndarray,
    source_positive: np.ndarray,
    target_positive: np.ndarray,
) -> float:
    u_delta = np.abs(next_u[source_positive] - previous_u[source_positive])
    v_delta = np.abs(next_v[target_positive] - previous_v[target_positive])
    return float(max(np.max(u_delta, initial=0.0), np.max(v_delta, initial=0.0)))


def _solve_one(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    match_penalty: float,
    config: UOTSolverConfig,
) -> tuple[np.ndarray | None, str, int]:
    if float(np.sum(source, dtype=float)) <= 0.0:
        return None, STATUS_EMPTY_SOURCE, 0
    if float(np.sum(target, dtype=float)) <= 0.0:
        return None, STATUS_EMPTY_TARGET, 0

    source_positive = source > 0.0
    target_positive = target > 0.0
    log_source = np.full(source.shape, -np.inf, dtype=float)
    log_target = np.full(target.shape, -np.inf, dtype=float)
    log_source[source_positive] = np.log(source[source_positive])
    log_target[target_positive] = np.log(target[target_positive])
    log_u = np.zeros(source.shape, dtype=float)
    log_v = np.zeros(target.shape, dtype=float)
    total_iterations = 0
    final_log_kernel: np.ndarray | None = None

    for epsilon in config.eps_schedule:
        final_log_kernel = -(cost / float(epsilon))
        exponent = float(match_penalty) / (float(match_penalty) + float(epsilon))
        converged = False
        for _ in range(config.max_iter):
            total_iterations += 1
            log_kernel_v = logsumexp(final_log_kernel + log_v[None, :], axis=1)
            next_u = exponent * (log_source - log_kernel_v)
            log_kernel_t_u = logsumexp(final_log_kernel.T + next_u[None, :], axis=1)
            next_v = exponent * (log_target - log_kernel_t_u)

            if not np.isfinite(next_u[source_positive]).all() or not np.isfinite(
                next_v[target_positive]
            ).all():
                return None, STATUS_NUMERICAL_FAILURE, total_iterations
            max_update = _max_scaling_update(
                log_u,
                next_u,
                log_v,
                next_v,
                source_positive,
                target_positive,
            )
            log_u = next_u
            log_v = next_v
            if max_update <= config.tol:
                converged = True
                break
        if not converged:
            return None, STATUS_NOT_CONVERGED, total_iterations

    if final_log_kernel is None:
        return None, STATUS_NUMERICAL_FAILURE, total_iterations
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        plan = np.exp(log_u[:, None] + final_log_kernel + log_v[None, :])
    if not np.isfinite(plan).all() or np.any(plan < 0.0):
        return None, STATUS_NUMERICAL_FAILURE, total_iterations
    if float(np.sum(plan, dtype=float)) <= 0.0:
        return None, STATUS_NUMERICAL_FAILURE, total_iterations
    return plan, STATUS_OK, total_iterations


def solve_uot_batch(
    *,
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    match_penalties: np.ndarray,
    config: UOTSolverConfig | None = None,
    backend: str = "torch",
    device: str = "cuda:0",
) -> UOTBatchResult:
    """Solve independent soft-UOT plans on one shared state-cost matrix."""

    source_rows, target_rows, cost, penalties = _validate_batch_inputs(
        source,
        target,
        cost_matrix,
        match_penalties,
    )
    resolved_config = config or UOTSolverConfig()
    if backend == "torch":
        return _solve_torch_batch(
            source_rows,
            target_rows,
            cost,
            penalties,
            config=resolved_config,
            device=device,
        )
    if backend != "numpy":
        raise ValueError("backend must be 'torch' or 'numpy'")
    plans = np.full(
        (source_rows.shape[0], source_rows.shape[1], source_rows.shape[1]),
        np.nan,
        dtype=float,
    )
    status = np.full(source_rows.shape[0], STATUS_OK, dtype=object)
    iterations = np.zeros(source_rows.shape[0], dtype=int)
    for row_index in range(source_rows.shape[0]):
        plan, row_status, row_iterations = _solve_one(
            source_rows[row_index],
            target_rows[row_index],
            cost,
            float(penalties[row_index]),
            resolved_config,
        )
        status[row_index] = row_status
        iterations[row_index] = row_iterations
        if plan is not None:
            plans[row_index] = plan
    return UOTBatchResult(plans=plans, status=status, iterations=iterations)


def _solve_torch_batch(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    penalties: np.ndarray,
    *,
    config: UOTSolverConfig,
    device: str,
) -> UOTBatchResult:
    """Solve one batch in float64 log-domain arithmetic on an explicit GPU."""
    import torch

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("formal torch UOT requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("formal torch UOT requires torch.cuda.is_available()")
    if resolved_device.index is not None and resolved_device.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested UOT CUDA device is unavailable: {resolved_device}")

    n_rows, n_states = source.shape
    plans = np.full((n_rows, n_states, n_states), np.nan, dtype=float)
    status = np.full(n_rows, STATUS_OK, dtype=object)
    iterations = np.zeros(n_rows, dtype=int)
    source_mass = np.sum(source, axis=1, dtype=float)
    target_mass = np.sum(target, axis=1, dtype=float)
    status[source_mass <= 0.0] = STATUS_EMPTY_SOURCE
    status[(source_mass > 0.0) & (target_mass <= 0.0)] = STATUS_EMPTY_TARGET
    valid_indices = np.flatnonzero(status == STATUS_OK)
    if valid_indices.size == 0:
        return UOTBatchResult(plans=plans, status=status, iterations=iterations)

    source_tensor = torch.as_tensor(
        source[valid_indices], dtype=torch.float64, device=resolved_device
    )
    target_tensor = torch.as_tensor(
        target[valid_indices], dtype=torch.float64, device=resolved_device
    )
    cost_tensor = torch.as_tensor(cost, dtype=torch.float64, device=resolved_device)
    penalty_tensor = torch.as_tensor(
        penalties[valid_indices], dtype=torch.float64, device=resolved_device
    )
    source_positive = source_tensor > 0.0
    target_positive = target_tensor > 0.0
    negative_inf = torch.tensor(-torch.inf, dtype=torch.float64, device=resolved_device)
    log_source = torch.where(source_positive, torch.log(source_tensor), negative_inf)
    log_target = torch.where(target_positive, torch.log(target_tensor), negative_inf)
    log_u = torch.zeros_like(source_tensor)
    log_v = torch.zeros_like(target_tensor)
    total_iterations = 0
    final_log_kernel = None

    for epsilon in config.eps_schedule:
        epsilon_value = float(epsilon)
        final_log_kernel = -(cost_tensor / epsilon_value)
        exponent = penalty_tensor / (penalty_tensor + epsilon_value)
        converged = False
        for _ in range(config.max_iter):
            total_iterations += 1
            next_u = exponent[:, None] * (
                log_source
                - torch.logsumexp(final_log_kernel[None, :, :] + log_v[:, None, :], dim=2)
            )
            next_v = exponent[:, None] * (
                log_target
                - torch.logsumexp(
                    final_log_kernel.T[None, :, :] + next_u[:, None, :], dim=2
                )
            )
            finite = torch.isfinite(next_u[source_positive]).all() and torch.isfinite(
                next_v[target_positive]
            ).all()
            if not bool(finite):
                status[valid_indices] = STATUS_NUMERICAL_FAILURE
                iterations[valid_indices] = total_iterations
                return UOTBatchResult(plans=plans, status=status, iterations=iterations)
            u_delta = torch.where(source_positive, torch.abs(next_u - log_u), 0.0)
            v_delta = torch.where(target_positive, torch.abs(next_v - log_v), 0.0)
            max_update = torch.maximum(u_delta.amax(dim=1), v_delta.amax(dim=1))
            log_u = next_u
            log_v = next_v
            if bool(torch.all(max_update <= config.tol)):
                converged = True
                break
        if not converged:
            status[valid_indices] = STATUS_NOT_CONVERGED
            iterations[valid_indices] = total_iterations
            return UOTBatchResult(plans=plans, status=status, iterations=iterations)

    if final_log_kernel is None:
        status[valid_indices] = STATUS_NUMERICAL_FAILURE
        return UOTBatchResult(plans=plans, status=status, iterations=iterations)
    plan_tensor = torch.exp(
        log_u[:, :, None] + final_log_kernel[None, :, :] + log_v[:, None, :]
    )
    if not bool(torch.isfinite(plan_tensor).all()) or bool(torch.any(plan_tensor < 0.0)):
        status[valid_indices] = STATUS_NUMERICAL_FAILURE
        iterations[valid_indices] = total_iterations
        return UOTBatchResult(plans=plans, status=status, iterations=iterations)
    solved = plan_tensor.detach().cpu().numpy()
    positive_mass = np.sum(solved, axis=(1, 2), dtype=float) > 0.0
    failed_indices = valid_indices[~positive_mass]
    status[failed_indices] = STATUS_NUMERICAL_FAILURE
    plans[valid_indices[positive_mass]] = solved[positive_mass]
    iterations[valid_indices] = total_iterations
    return UOTBatchResult(plans=plans, status=status, iterations=iterations)


__all__ = [
    "STATUS_EMPTY_SOURCE",
    "STATUS_EMPTY_TARGET",
    "STATUS_NOT_CONVERGED",
    "STATUS_NUMERICAL_FAILURE",
    "STATUS_OK",
    "UOTBatchResult",
    "UOTSolverConfig",
    "solve_uot_batch",
]
