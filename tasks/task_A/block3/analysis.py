"""Shared analysis layer for Task A Block 3b internal Phase 3.

Task and purpose:
    Provide the deterministic interface that derives `A/d/e` from native
    plan-based comparator output `P` for `3B-1` and `3B-2` scoring surfaces.

Relevant document anchors:
    - docs/task_A_spec.md §4.5.5-§4.5.6 and §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §4.3.2, §5.5, §5.6

Expected inputs and outputs:
    Inputs are source profile `x`, target profile `y`, and matched plan `P`.
    Outputs are `(A, d, e)` arrays derived by the frozen row/column sum rule.

Internal Phase 3 boundary:
    This helper is an internal scaffold component. It does not create public
    runner, review workflow, or packet bridge authority.
"""
from __future__ import annotations

import numpy as np


def derive_A_d_e_from_plan(
    *,
    x: np.ndarray,
    y: np.ndarray,
    P: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive `A/d/e` from a native matched plan.

    Inputs are 1D nonnegative source/target vectors and a nonnegative plan
    matrix with shape ``(len(x), len(y))``. Outputs are STRIDE-native `A`, `d`,
    and `e` arrays. Zero source rows emit no fallback relation and status
    propagation remains the caller's responsibility at the metric layer.
    """

    source = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    # P is the native matched transport plan emitted by plan-based comparators.
    plan = np.asarray(P, dtype=float)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if plan.ndim != 2 or plan.shape != (source.size, target.size):
        raise ValueError("P shape must match (len(x), len(y))")
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(plan).all():
        raise ValueError("x, y, and P must contain finite values")
    if (source < 0.0).any() or (target < 0.0).any() or (plan < 0.0).any():
        raise ValueError("x, y, and P must be non-negative")
    if not np.isfinite(float(tol)) or float(tol) < 0.0:
        raise ValueError("tol must be finite and non-negative")
    # r and c are plan marginals used by the shared analysis layer.
    r = np.sum(plan, axis=1, dtype=float)
    c = np.sum(plan, axis=0, dtype=float)
    A = np.zeros_like(plan, dtype=float)
    positive = source > tol
    A[positive] = plan[positive] / source[positive, None]
    d = np.zeros_like(source, dtype=float)
    d[positive] = np.maximum((source[positive] - r[positive]) / source[positive], 0.0)
    e = np.maximum(target - c, 0.0)
    return A, d, e
