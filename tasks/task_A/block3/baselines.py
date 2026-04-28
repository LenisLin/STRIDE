"""Baseline producer interfaces for Task A Block 3b internal Phase 3.

Task and purpose:
    Define stable baseline method names and deterministic/native plan
    producers for the split `3B-1 A benchmark` and `3B-2 d/e benchmark` routes.

Relevant document anchors:
    - docs/task_A_spec.md §4.5.5-§4.5.6 and §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §4.3, §5.5, §5.6

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
from stride.adapters.ot_sinkhorn import (
    ObservationMatchConfig,
    batched_uot_solve,
    build_observation_kernels,
)


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
    """Return a deterministic balanced-route fallback plan for unit tests."""

    return diagonal_transport_plan(x, y)


def partial_ot_plan(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cost_matrix: np.ndarray | None = None,
    matched_mass_budget: float | None = None,
) -> PlanBaselineResult:
    """Build a hard-budget partial-OT native plan using greedy cost order.

    Inputs are nonnegative endpoint vectors, optional transport costs, and an
    optional requested matched-mass budget. Output carries `P` and clipping
    diagnostics. If no cost matrix is supplied, deterministic diagonal-first
    costs are used to prefer shared labels before off-diagonal assignments.
    """

    source = _validate_vector("x", x)
    target = _validate_vector("y", y)
    if cost_matrix is None:
        cost = np.ones((source.size, target.size), dtype=float)
        diagonal_size = min(source.size, target.size)
        cost[np.arange(diagonal_size), np.arange(diagonal_size)] = 0.0
    else:
        cost = _validate_cost_matrix(cost_matrix, (source.size, target.size))
    if matched_mass_budget is None:
        # matched_mass_budget is the requested hard matched-mass comparator budget.
        matched_mass_budget = float(np.sum(np.minimum(source, target), dtype=float))
    requested_budget = float(matched_mass_budget)
    if not np.isfinite(requested_budget) or requested_budget < 0.0:
        raise ValueError("matched_mass_budget must be finite and non-negative")
    # effective_budget clips requested mass to feasible source and target mass.
    effective_budget = min(requested_budget, float(np.sum(source)), float(np.sum(target)))
    remaining_source = source.copy()
    remaining_target = target.copy()
    remaining_budget = effective_budget
    # P is the native matched transport plan emitted by plan-based comparators.
    plan = np.zeros((source.size, target.size), dtype=float)
    for row_index, col_index in sorted(
        np.ndindex(cost.shape),
        key=lambda index: (float(cost[index]), index[0], index[1]),
    ):
        if remaining_budget <= 0.0:
            break
        amount = min(remaining_source[row_index], remaining_target[col_index], remaining_budget)
        if amount <= 0.0:
            continue
        plan[row_index, col_index] = amount
        remaining_source[row_index] -= amount
        remaining_target[col_index] -= amount
        remaining_budget -= amount
    return PlanBaselineResult(
        P=plan,
        metadata={
            "requested_budget": requested_budget,
            "effective_budget": effective_budget,
            "clipped": bool(effective_budget < requested_budget),
        },
    )


def solve_uot_plan(
    *,
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix: np.ndarray,
    match_penalty: float,
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
    cfg = ObservationMatchConfig(eps_schedule=(1.0, 0.2), max_iter=2000, tol=1e-7)
    kernels = build_observation_kernels(cost, cfg.eps_schedule, cost_scale=1.0)
    metrics, details, status = batched_uot_solve(
        A=source[None, :],
        B=target[None, :],
        lambda_pl=np.array([lambda_value], dtype=float),
        kernels=kernels,
        cfg=cfg,
        tau_external=None,
        return_plan=True,
    )
    status_value = str(status[0])
    metadata = {
        "lambda": lambda_value,
        "solver_status": status_value,
        "matched_mass": float(metrics["T"][0]) if np.isfinite(metrics["T"][0]) else None,
    }
    if status_value != "ok":
        return PlanBaselineResult(P=None, status=status_value, metadata=metadata)
    if "matching_plan" not in details:
        raise ValueError("UOT solver did not return details['matching_plan']")
    # P is the native matched transport plan emitted by plan-based comparators.
    plan = np.asarray(details["matching_plan"][0], dtype=float)
    return PlanBaselineResult(P=plan, status="ok", metadata=metadata)


def estimate_uot_matched_mass(
    *,
    train_pairs: tuple[tuple[np.ndarray, np.ndarray], ...],
    cost_matrix: np.ndarray,
    match_penalty: float,
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
    cfg = ObservationMatchConfig(eps_schedule=(1.0, 0.2), max_iter=2000, tol=1e-7)
    kernels = build_observation_kernels(cost, cfg.eps_schedule, cost_scale=1.0)
    _metrics, details, status = batched_uot_solve(
        A=source_rows,
        B=target_rows,
        lambda_pl=np.full(source_rows.shape[0], lambda_value, dtype=float),
        kernels=kernels,
        cfg=cfg,
        tau_external=None,
        return_plan=True,
    )
    ok_mask = np.asarray(status == "ok", dtype=bool)
    if not np.any(ok_mask):
        return float("inf")
    plans = np.asarray(details["matching_plan"], dtype=float)
    return float(np.mean(np.sum(plans[ok_mask], axis=(1, 2), dtype=float)))


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
]
