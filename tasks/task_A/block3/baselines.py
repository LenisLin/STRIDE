"""Baseline producer interfaces for Task A Block 3b internal Phase 3.

Task and purpose:
    Define stable baseline method names and deterministic/native plan
    producers for the split `3B-1 A benchmark` and `3B-2 d/e benchmark` routes.

Relevant document anchors:
    - docs/task_A/spec.md §4.5.5-§4.5.6 and §5.1 Phase 3
    - docs/task_A/block3/scientific_contract.md §4.3, §5.5, §5.6

Expected inputs and outputs:
    Inputs are endpoint profiles and a cost matrix when needed. Outputs are
    native matched plans `P` plus optional diagnostics that can be normalized by
    the shared analysis layer into `A/d/e` and serialized through native records.

Internal Phase 3 boundary:
    This file is an internal implementation surface. It does not define public
    workflow entrypoints, review commands, result-packet bridges, or scientific
    benchmark conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import UOTCalibrationResult, calibrate_uot_lambda
from .task_uot import STATUS_OK, UOTSolverConfig, solve_uot_batch


@dataclass(frozen=True)
class RuntimeSettings:
    """Task-local runtime request metadata retained for UOT comparator calls."""

    uot_backend: str = "numpy"
    device: str | None = None


LIVE_3B1_BASELINES: tuple[str, ...] = (
    "balanced_ot_baseline",
    "uot_baseline",
    "partial_ot_baseline",
    "diagonal_transport_baseline",
)
LIVE_3B2_BASELINES: tuple[str, ...] = (
    "uot_baseline",
    "partial_ot_baseline",
    "diagonal_transport_baseline",
)
PARTIAL_OT_NUMERICAL_ATOL = 1e-5


@dataclass(frozen=True)
class PlanBaselineResult:
    """Native plan output and method diagnostics for one baseline solve.

    Inputs are produced by baseline helper functions. Outputs carry `P`, a
    method status, and JSON-compatible metadata. Non-`ok` status means callers
    should propagate non-estimable metric rows instead of deriving fallback
    `A/d/e` arrays.
    """

    P: np.ndarray | None
    status: str = "ok"
    metadata: dict[str, object] = field(default_factory=dict)


def _validate_vector(name: str, values: np.ndarray) -> np.ndarray:
    """Return one finite nonnegative endpoint vector or raise `ValueError`."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values")
    if (arr < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    return arr


def _validate_cost_matrix(cost_matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a finite cost matrix matching the requested plan shape."""

    cost = np.asarray(cost_matrix, dtype=float)
    if cost.shape != shape:
        raise ValueError("cost_matrix shape must match (len(x), len(y))")
    if not np.isfinite(cost).all():
        raise ValueError("cost_matrix must contain finite values")
    return cost


def _validate_cost_matrix_or_default(
    cost_matrix: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return caller-provided costs or deterministic diagonal-first defaults."""

    if cost_matrix is not None:
        return _validate_cost_matrix(cost_matrix, shape)
    cost = np.ones(shape, dtype=float)
    diagonal_size = min(shape)
    cost[np.arange(diagonal_size), np.arange(diagonal_size)] = 0.0
    return cost


def diagonal_transport_plan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the exact diagonal native matched plan for shared state labels.

    Inputs are nonnegative source and target endpoint vectors on a shared state
    axis. Output `P` has `P[i, i] = min(x[i], y[i])` and all off-diagonal entries
    set to zero; callers derive `A/d/e` in the shared analysis layer.
    """

    source = _validate_vector("x", x)
    target = _validate_vector("y", y)
    plan = np.zeros((source.shape[0], target.shape[0]), dtype=float)
    diagonal_size = min(source.shape[0], target.shape[0])
    diagonal = np.minimum(source[:diagonal_size], target[:diagonal_size])
    # P is the native matched transport plan emitted by plan-based comparators.
    plan[np.arange(diagonal_size), np.arange(diagonal_size)] = diagonal
    return plan


def balanced_ot_plan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the exact balanced OT plan from POT on the shared state axis."""

    import ot

    source = _validate_vector("x", x)
    target = _validate_vector("y", y)
    if not np.isclose(source.sum(), target.sum(), atol=1e-10, rtol=0.0):
        raise ValueError("balanced OT requires equal source and target mass")
    cost = _validate_cost_matrix_or_default(None, (source.size, target.size))
    plan = np.asarray(ot.emd(source, target, cost), dtype=float)
    if plan.shape != cost.shape or not np.isfinite(plan).all():
        raise ValueError("balanced OT solver returned an invalid plan")
    return plan


def partial_ot_plan(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cost_matrix: np.ndarray | None = None,
    matched_mass_budget: float | None = None,
) -> PlanBaselineResult:
    """Build an exact fixed-mass partial-OT native plan.

    Inputs are nonnegative endpoint vectors, optional transport costs, and an
    optional requested matched-mass budget. Output carries `P` and clipping
    diagnostics. The native plan solves the POT fixed-mass partial Wasserstein
    problem. If no cost matrix is supplied, deterministic diagonal-first costs
    are used to prefer shared labels before off-diagonal assignments.
    """

    import ot

    source = _validate_vector("x", x)
    target = _validate_vector("y", y)
    cost = _validate_cost_matrix_or_default(cost_matrix, (source.size, target.size))
    if matched_mass_budget is None:
        # matched_mass_budget is the requested hard matched-mass comparator budget.
        matched_mass_budget = float(np.sum(np.minimum(source, target), dtype=float))
    requested_budget = float(matched_mass_budget)
    if not np.isfinite(requested_budget) or requested_budget < 0.0:
        raise ValueError("matched_mass_budget must be finite and non-negative")
    # effective_budget clips requested mass to feasible source and target mass.
    effective_budget = min(requested_budget, float(np.sum(source)), float(np.sum(target)))
    if effective_budget <= 0.0:
        plan = np.zeros((source.size, target.size), dtype=float)
    else:
        # P is the native matched transport plan emitted by plan-based comparators.
        plan = np.asarray(
            ot.partial.partial_wasserstein(
                source,
                target,
                cost,
                m=effective_budget,
            ),
            dtype=float,
        )
    if plan.shape != cost.shape:
        raise ValueError("partial OT solver returned a plan with unexpected shape")
    if not np.isfinite(plan).all():
        raise ValueError("partial OT solver returned non-finite plan entries")
    if np.any(plan < -PARTIAL_OT_NUMERICAL_ATOL):
        raise ValueError("partial OT solver returned negative plan entries")
    plan = np.where((plan < 0.0) & (plan >= -PARTIAL_OT_NUMERICAL_ATOL), 0.0, plan)
    row_sums = np.sum(plan, axis=1, dtype=float)
    col_sums = np.sum(plan, axis=0, dtype=float)
    transported_mass = float(np.sum(plan, dtype=float))
    if np.any(row_sums - source > PARTIAL_OT_NUMERICAL_ATOL):
        raise ValueError("partial OT plan violates source marginal upper bounds")
    if np.any(col_sums - target > PARTIAL_OT_NUMERICAL_ATOL):
        raise ValueError("partial OT plan violates target marginal upper bounds")
    if abs(transported_mass - effective_budget) > PARTIAL_OT_NUMERICAL_ATOL:
        raise ValueError("partial OT plan violates fixed matched-mass budget")
    return PlanBaselineResult(
        P=plan,
        metadata={
            "solver": "ot.partial.partial_wasserstein",
            "requested_budget": requested_budget,
            "effective_budget": effective_budget,
            "clipped": bool(effective_budget < requested_budget),
            "transported_mass": transported_mass,
            "objective_value": float(np.sum(plan * cost, dtype=float)),
        },
    )


def solve_uot_plan(
    *,
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix: np.ndarray,
    match_penalty: float,
    runtime_settings: RuntimeSettings | None = None,
) -> PlanBaselineResult:
    """Solve one UOT baseline and return its native matched plan.

    Inputs are nonnegative endpoint vectors, a finite cost matrix, and one
    positive match penalty. Output carries `details["matching_plan"]` as native
    `P` plus solver metadata. Non-ok solver status is returned without fallback
    plan so execution can propagate non-estimable metrics.
    """

    source = _validate_vector("x", x)
    target = _validate_vector("y", y)
    cost = _validate_cost_matrix(cost_matrix, (source.size, target.size))
    lambda_value = float(match_penalty)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("match_penalty must be finite and positive")
    resolved_runtime = runtime_settings or RuntimeSettings()
    result = solve_uot_batch(
        source=source[None, :],
        target=target[None, :],
        cost_matrix=cost,
        match_penalties=np.array([lambda_value], dtype=float),
        config=UOTSolverConfig(),
        backend=resolved_runtime.uot_backend,
        device=resolved_runtime.device or "cuda:0",
    )
    status_value = str(result.status[0])
    plan = np.asarray(result.plans[0], dtype=float)
    metadata = {
        "lambda": lambda_value,
        "solver": "tasks.task_A.block3.task_uot.solve_uot_batch",
        "solver_backend": f"{resolved_runtime.uot_backend}_log_domain",
        "requested_backend": resolved_runtime.uot_backend,
        "requested_device": resolved_runtime.device,
        "solver_status": status_value,
        "iterations": int(result.iterations[0]),
        "matched_mass": float(np.sum(plan, dtype=float)) if status_value == STATUS_OK else None,
    }
    if status_value != STATUS_OK:
        return PlanBaselineResult(P=None, status=status_value, metadata=metadata)
    return PlanBaselineResult(P=plan, status=STATUS_OK, metadata=metadata)


def solve_uot_plans_batch(
    *,
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    match_penalty: float,
    runtime_settings: RuntimeSettings,
) -> tuple[PlanBaselineResult, ...]:
    """Solve all test-side UOT rows in one task-local batch."""
    source_rows = np.asarray(source, dtype=float)
    target_rows = np.asarray(target, dtype=float)
    if source_rows.ndim != 2 or target_rows.shape != source_rows.shape:
        raise ValueError("source and target batches must share one 2D shape")
    cost = _validate_cost_matrix(
        cost_matrix,
        (source_rows.shape[1], target_rows.shape[1]),
    )
    lambda_value = float(match_penalty)
    result = solve_uot_batch(
        source=source_rows,
        target=target_rows,
        cost_matrix=cost,
        match_penalties=np.full(source_rows.shape[0], lambda_value, dtype=float),
        config=UOTSolverConfig(),
        backend=runtime_settings.uot_backend,
        device=runtime_settings.device or "cuda:0",
    )
    outputs: list[PlanBaselineResult] = []
    for row_index, row_status in enumerate(result.status):
        status_value = str(row_status)
        plan = np.asarray(result.plans[row_index], dtype=float)
        metadata = {
            "lambda": lambda_value,
            "solver": "tasks.task_A.block3.task_uot.solve_uot_batch",
            "solver_backend": f"{runtime_settings.uot_backend}_log_domain",
            "requested_backend": runtime_settings.uot_backend,
            "requested_device": runtime_settings.device,
            "solver_status": status_value,
            "iterations": int(result.iterations[row_index]),
            "matched_mass": (
                float(np.sum(plan, dtype=float)) if status_value == STATUS_OK else None
            ),
        }
        outputs.append(
            PlanBaselineResult(
                P=plan if status_value == STATUS_OK else None,
                status=status_value,
                metadata=metadata,
            )
        )
    return tuple(outputs)


def estimate_uot_matched_mass(
    *,
    train_pairs: tuple[tuple[np.ndarray, np.ndarray], ...],
    cost_matrix: np.ndarray,
    match_penalty: float,
    runtime_settings: RuntimeSettings | None = None,
) -> float:
    """Estimate train-side matched mass for one UOT lambda candidate.

    Inputs are train endpoint pairs, the shared cost matrix, and one lambda.
    Output is the mean matched plan mass across solver-ok rows. If no row is
    solver-ok, `inf` is returned so calibration will not select that candidate.
    """

    if not train_pairs:
        raise ValueError("train_pairs must be non-empty")
    source_rows = np.vstack([_validate_vector("x", x) for x, _y in train_pairs])
    target_rows = np.vstack([_validate_vector("y", y) for _x, y in train_pairs])
    if source_rows.shape != target_rows.shape:
        raise ValueError("train pair x and y arrays must share one shape")
    cost = _validate_cost_matrix(cost_matrix, (source_rows.shape[1], target_rows.shape[1]))
    lambda_value = float(match_penalty)
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("match_penalty must be finite and positive")
    resolved_runtime = runtime_settings or RuntimeSettings()
    result = solve_uot_batch(
        source=source_rows,
        target=target_rows,
        cost_matrix=cost,
        match_penalties=np.full(source_rows.shape[0], lambda_value, dtype=float),
        config=UOTSolverConfig(),
        backend=resolved_runtime.uot_backend,
        device=resolved_runtime.device or "cuda:0",
    )
    ok_mask = np.asarray(result.status == STATUS_OK, dtype=bool)
    if not np.any(ok_mask):
        return float("inf")
    return float(np.mean(np.sum(result.plans[ok_mask], axis=(1, 2), dtype=float)))


__all__ = [
    "LIVE_3B1_BASELINES",
    "LIVE_3B2_BASELINES",
    "PlanBaselineResult",
    "UOTCalibrationResult",
    "balanced_ot_plan",
    "calibrate_uot_lambda",
    "diagonal_transport_plan",
    "partial_ot_plan",
    "estimate_uot_matched_mass",
    "solve_uot_plan",
    "solve_uot_plans_batch",
]
