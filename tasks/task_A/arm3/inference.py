"""
Module: tasks.task_A.arm3.inference

Phase 6 of the Arm-3 pipeline: density-mode tensor assembly and metric computation.

Responsibilities:
- Assemble density-mode [N, K] tensors aligned to pair_meta row order (both
  full-coverage and pseudo-ROI density vectors use this same function).
- Freeze semantic support masks from full-coverage reference COUNT tensors.
- Broadcast frozen tau and lambda to per-row arrays aligned to pair_meta.
- Append primary and secondary Arm-3 density metrics to UOT result DataFrames.
- Compute row-level floor-dominated flags.

Design constraints (all locked):
- Do NOT call assemble_tensors from tasks.task_A.common for density mode;
  that function is count-mode only.
- Do NOT use T / (T + B_pos + D_pos + eps) as a transportability endpoint.
  This quantity is explicitly prohibited for Arm-3.
- Q_src_dens = T / (S_src + eps) is the primary relative transportability endpoint.
- Q_tgt_dens = T / (S_tgt + eps) is the mandatory audit / secondary endpoint.
- T_abs is secondary / audit context only.
- Frozen support masks K_r^100 must not shrink with coverage reduction.
- eta_floor padding must NOT activate prototypes outside the frozen semantic support.
- Per-row tau is assigned by compartment_a; per-row lambda by pair_family.

CRITICAL: freeze_support_masks expects COUNT tensors, not density tensors.
The caller (arm3_uq_stress.py) must pass full-coverage COUNT tensors assembled
from roi_block_summary (sum of count_k* columns across all blocks per ROI).
Do NOT pass density tensors to freeze_support_masks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import DENSITY_EPS
from slotar.exceptions import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)
from slotar.uot import UOTSolveConfig, batched_uot_solve
from slotar.utils import compute_active_mask

# ---------------------------------------------------------------------------
# Valid compartment / family sets (locked)
# ---------------------------------------------------------------------------

_VALID_COMPARTMENTS: frozenset[str] = frozenset({"TC", "IM", "PT"})
_VALID_FAMILIES: frozenset[str] = frozenset({"TC-IM", "IM-PT", "TC-PT"})


def assemble_density_tensors(
    roi_density_vectors: dict[str, np.ndarray],
    pair_meta: pd.DataFrame,
    k_full: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble density-mode [N, K] tensors aligned to pair_meta row order.

    Applicable to both full-coverage density vectors (from
    build_full_coverage_density_vectors) and pseudo-ROI density vectors (from
    build_pseudo_roi_density), since both have the same (K,) shape per ROI.

    Do NOT use this function with count-mode vectors; count-mode tensors are
    assembled by tasks.task_A.common.assemble_tensors.

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Maps roi_id -> vector of shape (K,).
        Values may be full-coverage density, pseudo-ROI density, or
        full-coverage COUNT vectors (for support mask construction).
    pair_meta : pd.DataFrame
        Ordered pair table aligned to the target batch dimension.
        Required columns: roi_a, roi_b.
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    A : np.ndarray, shape (N, K)
        Source-side tensors. Row i corresponds to pair_meta.iloc[i].
    B : np.ndarray, shape (N, K)
        Target-side tensors. Row i corresponds to pair_meta.iloc[i].
    """
    if pair_meta.empty:
        return (
            np.zeros((0, k_full), dtype=float),
            np.zeros((0, k_full), dtype=float),
        )

    for col in ("roi_a", "roi_b"):
        if col not in pair_meta.columns:
            raise ValueError(
                f"assemble_density_tensors: pair_meta is missing required column {col!r}"
            )

    # Validate all referenced roi_ids are present before building tensors
    missing: list[str] = []
    for roi_id in list(pair_meta["roi_a"].astype(str)) + list(pair_meta["roi_b"].astype(str)):
        if roi_id not in roi_density_vectors:
            missing.append(roi_id)
    if missing:
        unique_missing = sorted(set(missing))
        raise ValueError(
            f"assemble_density_tensors: {len(unique_missing)} roi_id(s) referenced "
            f"in pair_meta are absent from roi_density_vectors: "
            f"{unique_missing[:5]}{'...' if len(unique_missing) > 5 else ''}"
        )

    pair_meta = pair_meta.reset_index(drop=True)
    A = np.stack(
        [roi_density_vectors[str(rid)] for rid in pair_meta["roi_a"]],
        axis=0,
    ).astype(float, copy=False)
    B = np.stack(
        [roi_density_vectors[str(rid)] for rid in pair_meta["roi_b"]],
        axis=0,
    ).astype(float, copy=False)

    if A.shape[1] != k_full:
        raise ValueError(
            f"assemble_density_tensors: vectors have shape[1]={A.shape[1]} "
            f"but k_full={k_full}"
        )

    return A, B


def freeze_support_masks(
    A_count: np.ndarray,
    B_count: np.ndarray,
    n_min_proto: float,
    k_full: int,
) -> np.ndarray:
    """
    Compute frozen semantic support masks from full-coverage reference COUNT tensors.

    K_r^100 = {k : A_count[r, k] + B_count[r, k] >= n_min_proto}

    CRITICAL: This function must receive full-coverage COUNT tensors, not density
    tensors. The caller must aggregate count_k* from roi_block_summary (summing
    across all blocks per ROI) and assemble these counts into tensors before
    calling this function.

    This mask is computed once from full-coverage COUNT tensors and must NOT be
    recomputed on or updated by pseudo-ROI tensors. It must not shrink with
    coverage reduction. It is the authoritative semantic support for every
    pseudo-ROI replicate derived from reference pair r.

    eta_floor padding applied inside the solver must NOT activate prototypes
    outside this frozen mask.

    Parameters
    ----------
    A_count : np.ndarray, shape (N, K)
        Full-coverage source-side COUNT tensors (total cells per prototype,
        summed across all blocks in the ROI envelope).
        These are NOT density tensors.
    B_count : np.ndarray, shape (N, K)
        Full-coverage target-side COUNT tensors.
        These are NOT density tensors.
    n_min_proto : float
        Minimum combined COUNT mass for a prototype to enter semantic support.
        Sourced from uot_cfg.n_min_proto.
        Evaluated as: (A_count[r, k] + B_count[r, k]) >= n_min_proto
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    support_masks : np.ndarray, shape (N, K), dtype bool
        True where prototype k is in the frozen semantic support for pair r.
    """
    A_count = np.asarray(A_count, dtype=float)
    B_count = np.asarray(B_count, dtype=float)

    if A_count.shape != B_count.shape:
        raise ValueError(
            f"freeze_support_masks: A_count shape {A_count.shape} != "
            f"B_count shape {B_count.shape}"
        )
    if A_count.ndim != 2:
        raise ValueError(
            f"freeze_support_masks: expected 2D tensors, got shape {A_count.shape}"
        )
    if A_count.shape[1] != k_full:
        raise ValueError(
            f"freeze_support_masks: tensor K dimension {A_count.shape[1]} != k_full={k_full}"
        )

    support_masks = (A_count + B_count) >= n_min_proto
    return support_masks.astype(bool)


def broadcast_frozen_tau(
    pair_meta: pd.DataFrame,
    frozen_taus: dict[str, float],
) -> np.ndarray:
    """
    Assign per-row tau from frozen_taus by compartment_a.

    tau is compartment-specific. For an ordered inference row A -> B, the tau
    value is sourced from frozen_taus[compartment_a]. Side A is the reference
    side for tau assignment.

    Parameters
    ----------
    pair_meta : pd.DataFrame
        Must contain 'compartment_a' column with values in {'TC', 'IM', 'PT'}.
    frozen_taus : dict[str, float]
        Compartment-keyed frozen tau values from calibrate_tau_by_compartment.
        Expected keys: 'TC', 'IM', 'PT'.

    Returns
    -------
    np.ndarray, shape (N,), dtype float
        Per-row tau_external values aligned to pair_meta row order.
    """
    if "compartment_a" not in pair_meta.columns:
        raise ValueError(
            "broadcast_frozen_tau: pair_meta is missing required column 'compartment_a'"
        )

    present = set(frozen_taus.keys())
    unexpected = present - _VALID_COMPARTMENTS
    if unexpected:
        raise ValueError(
            f"broadcast_frozen_tau: frozen_taus contains unexpected compartment keys: "
            f"{sorted(unexpected)}. Expected only {sorted(_VALID_COMPARTMENTS)}."
        )
    missing = _VALID_COMPARTMENTS - present
    if missing:
        raise ValueError(
            f"broadcast_frozen_tau: frozen_taus is missing compartment keys: "
            f"{sorted(missing)}"
        )

    actual_comps = set(pair_meta["compartment_a"].astype(str).unique())
    unmapped = actual_comps - _VALID_COMPARTMENTS
    if unmapped:
        raise ValueError(
            f"broadcast_frozen_tau: pair_meta contains compartment_a values not in "
            f"{sorted(_VALID_COMPARTMENTS)}: {sorted(unmapped)}"
        )

    tau_arr = np.array(
        [float(frozen_taus[str(c)]) for c in pair_meta["compartment_a"]],
        dtype=float,
    )
    return tau_arr


def broadcast_frozen_lambda(
    pair_meta: pd.DataFrame,
    frozen_lambdas: dict[str, float],
) -> np.ndarray:
    """
    Assign per-row lambda_pl from frozen_lambdas by pair_family.

    Frozen lambda_dens values are broadcast from the full-coverage calibration
    phase. Pseudo-ROI inference must NOT recalibrate lambda.

    Parameters
    ----------
    pair_meta : pd.DataFrame
        Must contain 'pair_family' column with values in
        {'TC-IM', 'IM-PT', 'TC-PT'}.
    frozen_lambdas : dict[str, float]
        Family-keyed frozen lambda_dens values from calibrate_lambda_dens.

    Returns
    -------
    np.ndarray, shape (N,), dtype float
        Per-row lambda_pl values aligned to pair_meta row order.
    """
    if "pair_family" not in pair_meta.columns:
        raise ValueError(
            "broadcast_frozen_lambda: pair_meta is missing required column 'pair_family'"
        )

    present = set(frozen_lambdas.keys())
    unexpected = present - _VALID_FAMILIES
    if unexpected:
        raise ValueError(
            f"broadcast_frozen_lambda: frozen_lambdas contains unexpected family keys: "
            f"{sorted(unexpected)}. Expected only {sorted(_VALID_FAMILIES)}."
        )
    missing = _VALID_FAMILIES - present
    if missing:
        raise ValueError(
            f"broadcast_frozen_lambda: frozen_lambdas is missing family keys: "
            f"{sorted(missing)}"
        )

    actual_families = set(pair_meta["pair_family"].astype(str).unique())
    unmapped = actual_families - _VALID_FAMILIES
    if unmapped:
        raise ValueError(
            f"broadcast_frozen_lambda: pair_meta contains pair_family values not in "
            f"{sorted(_VALID_FAMILIES)}: {sorted(unmapped)}"
        )

    lambda_arr = np.array(
        [float(frozen_lambdas[str(f)]) for f in pair_meta["pair_family"]],
        dtype=float,
    )
    return lambda_arr


def compute_arm3_density_metrics(
    df_result: pd.DataFrame,
    A_dens: np.ndarray,
    B_dens: np.ndarray,
) -> pd.DataFrame:
    """
    Append primary and secondary Arm-3 density metrics to a UOT result DataFrame.

    Input df_result must already contain T, B_pos, D_pos, and uot_status from
    run_uot_batch_safe. Metrics on non-ok rows are set to NaN.

    Columns appended:
        U_abs_dens   = B_pos + D_pos      (primary; ok rows only)
        S_src        = sum_k(a_k^dens)    (scale audit; all rows)
        S_tgt        = sum_k(b_k^dens)    (scale audit; all rows)
        Delta_scale  = S_tgt - S_src      (scale audit; all rows)
        scale_ratio  = S_tgt / (S_src + DENSITY_EPS)   (all rows)
        Q_src_dens   = T / (S_src + DENSITY_EPS)  [primary endpoint; ok rows only]
        Q_tgt_dens   = T / (S_tgt + DENSITY_EPS)  [mandatory audit; ok rows only]

    Explicitly prohibited columns (must never be added here):
        T / (T + B_pos + D_pos + eps)  — this quantity is NOT a valid Arm-3
        transportability endpoint and must not appear in any output.

    Parameters
    ----------
    df_result : pd.DataFrame
        UOT results from run_uot_batch_safe. Modified in place (copy returned).
        Must contain columns: T, B_pos, D_pos, uot_status.
    A_dens : np.ndarray, shape (N, K)
        Source-side density tensors aligned to df_result rows.
    B_dens : np.ndarray, shape (N, K)
        Target-side density tensors aligned to df_result rows.

    Returns
    -------
    pd.DataFrame
        Copy of df_result with appended Arm-3 density metric columns.
    """
    for col in ("T", "B_pos", "D_pos", "uot_status"):
        if col not in df_result.columns:
            raise ValueError(
                f"compute_arm3_density_metrics: df_result is missing required column {col!r}"
            )

    n_rows = len(df_result)
    if A_dens.shape[0] != n_rows or B_dens.shape[0] != n_rows:
        raise ValueError(
            f"compute_arm3_density_metrics: A_dens/B_dens row count "
            f"({A_dens.shape[0]}, {B_dens.shape[0]}) does not match "
            f"df_result row count ({n_rows})"
        )

    df = df_result.copy()
    ok_mask = df["uot_status"] == "ok"

    # Scale audit quantities — computed for all rows
    S_src = A_dens.sum(axis=1)
    S_tgt = B_dens.sum(axis=1)
    Delta_scale = S_tgt - S_src
    scale_ratio = S_tgt / (S_src + DENSITY_EPS)

    df["S_src"] = S_src
    df["S_tgt"] = S_tgt
    df["Delta_scale"] = Delta_scale
    df["scale_ratio"] = scale_ratio

    # Primary metrics — ok rows only; NaN elsewhere
    U_abs_dens = np.full(n_rows, np.nan, dtype=float)
    Q_src_dens = np.full(n_rows, np.nan, dtype=float)
    Q_tgt_dens = np.full(n_rows, np.nan, dtype=float)

    ok_idx = np.flatnonzero(ok_mask.to_numpy())
    if ok_idx.size > 0:
        T_ok = df["T"].to_numpy(dtype=float)[ok_idx]
        B_ok = df["B_pos"].to_numpy(dtype=float)[ok_idx]
        D_ok = df["D_pos"].to_numpy(dtype=float)[ok_idx]
        S_src_ok = S_src[ok_idx]
        S_tgt_ok = S_tgt[ok_idx]

        U_abs_dens[ok_idx] = B_ok + D_ok
        Q_src_dens[ok_idx] = T_ok / (S_src_ok + DENSITY_EPS)
        Q_tgt_dens[ok_idx] = T_ok / (S_tgt_ok + DENSITY_EPS)

    df["U_abs_dens"] = U_abs_dens
    df["Q_src_dens"] = Q_src_dens
    df["Q_tgt_dens"] = Q_tgt_dens

    return df


def extract_prototype_event_marginals(
    A_dens: np.ndarray,
    B_dens: np.ndarray,
    df_result: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-prototype marginal event masses via proportional allocation.

    Allocates the row-level scalar transport/creation/destruction masses
    (T, B_pos, D_pos) across prototype dimensions using the source and target
    density vectors as weights:

        T_k[r, k] = T[r]     * A[r, k] / (S_src[r] + eps)
        D_k[r, k] = D_pos[r] * A[r, k] / (S_src[r] + eps)
        B_k[r, k] = B_pos[r] * B[r, k] / (S_tgt[r] + eps)

    This is an approximate decomposition: it allocates event mass proportionally
    to density, which is the natural first-order approximation when the full
    per-prototype transport plan is not exposed by the solver.

    Non-ok rows (uot_status != "ok") receive NaN for all prototypes.

    Parameters
    ----------
    A_dens : np.ndarray, shape (N, K)
        Source-side density tensors aligned to df_result rows.
    B_dens : np.ndarray, shape (N, K)
        Target-side density tensors aligned to df_result rows.
    df_result : pd.DataFrame
        UOT results with columns: T, B_pos, D_pos, uot_status.
        Must be row-aligned to A_dens / B_dens.

    Returns
    -------
    T_k : np.ndarray, shape (N, K)
        Per-prototype transport mass.
    B_k : np.ndarray, shape (N, K)
        Per-prototype creation (birth) mass.
    D_k : np.ndarray, shape (N, K)
        Per-prototype destruction (death) mass.
    """
    for col in ("T", "B_pos", "D_pos", "uot_status"):
        if col not in df_result.columns:
            raise ValueError(
                f"extract_prototype_event_marginals: df_result missing required "
                f"column {col!r}"
            )

    n_rows, k = A_dens.shape
    if B_dens.shape != (n_rows, k):
        raise ValueError(
            f"extract_prototype_event_marginals: A_dens shape {A_dens.shape} "
            f"!= B_dens shape {B_dens.shape}"
        )
    if len(df_result) != n_rows:
        raise ValueError(
            f"extract_prototype_event_marginals: df_result has {len(df_result)} rows "
            f"but A_dens has {n_rows}"
        )

    T_k = np.full((n_rows, k), np.nan, dtype=float)
    B_k = np.full((n_rows, k), np.nan, dtype=float)
    D_k = np.full((n_rows, k), np.nan, dtype=float)

    ok_mask = df_result["uot_status"].to_numpy() == "ok"
    ok_idx = np.flatnonzero(ok_mask)

    if ok_idx.size > 0:
        T_vals = df_result["T"].to_numpy(dtype=float)[ok_idx]
        B_vals = df_result["B_pos"].to_numpy(dtype=float)[ok_idx]
        D_vals = df_result["D_pos"].to_numpy(dtype=float)[ok_idx]

        A_ok = A_dens[ok_idx]  # (n_ok, K)
        B_ok = B_dens[ok_idx]  # (n_ok, K)

        S_src = A_ok.sum(axis=1, keepdims=True) + DENSITY_EPS  # (n_ok, 1)
        S_tgt = B_ok.sum(axis=1, keepdims=True) + DENSITY_EPS  # (n_ok, 1)

        T_k[ok_idx] = T_vals[:, None] * (A_ok / S_src)
        D_k[ok_idx] = D_vals[:, None] * (A_ok / S_src)
        B_k[ok_idx] = B_vals[:, None] * (B_ok / S_tgt)

    return T_k, B_k, D_k


def compute_floor_dominated_flags(
    A_dens: np.ndarray,
    support_masks: np.ndarray,
    eta_floor: float,
) -> np.ndarray:
    """
    Compute row-level floor-dominated flags.

    Rule (task-fixed):
        K_support_r  = number of True entries in support_masks[r]
        floor_mass_r = eta_floor * K_support_r
        S_src_r      = sum(A_dens[r, k] for supported k, i.e. support_masks[r])
        flag row r   iff floor_mass_r / (S_src_r + DENSITY_EPS) > 0.10

    This is the authoritative floor-dominated rule for Arm-3 Phase 5/6.

    Parameters
    ----------
    A_dens : np.ndarray, shape (N, K)
        Source-side density tensors for the rows being evaluated.
    support_masks : np.ndarray, shape (N, K), dtype bool
        Frozen semantic support masks from freeze_support_masks.
        Must have the same N and K as A_dens.
    eta_floor : float
        Floor padding value from uot_cfg.eta_floor.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True for rows where floor mass dominates the supported source signal.
    """
    A_dens = np.asarray(A_dens, dtype=float)
    support_masks = np.asarray(support_masks, dtype=bool)

    if A_dens.shape != support_masks.shape:
        raise ValueError(
            f"compute_floor_dominated_flags: A_dens shape {A_dens.shape} != "
            f"support_masks shape {support_masks.shape}"
        )

    n_rows = A_dens.shape[0]

    # K_support_r: number of supported prototypes per row
    K_support = support_masks.sum(axis=1).astype(float)  # (N,)

    # floor_mass_r = eta_floor * K_support_r
    floor_mass = eta_floor * K_support  # (N,)

    # S_src_r = sum of A_dens over supported prototypes only
    # For rows with no supported prototypes, S_src = 0.0
    S_src_supported = np.where(support_masks, A_dens, 0.0).sum(axis=1)  # (N,)

    # floor_dominated iff floor_mass / (S_src + DENSITY_EPS) > 0.10
    ratio = floor_mass / (S_src_supported + DENSITY_EPS)
    floor_dominated = ratio > 0.10

    return floor_dominated.astype(bool)


# ---------------------------------------------------------------------------
# Patch 1 — Combined UOT solver + prototype event extraction
# ---------------------------------------------------------------------------

#: Mirror of common.STATUS_TO_BYPASS — local copy so inference.py has no
#: dependency on common.py internals.
_ARM3_STATUS_TO_BYPASS: dict[str, str | None] = {
    "ok": None,
    ERR_UOT_EMPTY_MASS_SOURCE: "S0_zero",
    ERR_UOT_EMPTY_MASS_TARGET: "S1_zero",
    ERR_UOT_EMPTY_SUPPORT: "empty_support_after_prune",
    ERR_UOT_NUMERICAL: "uot_numerical_failure",
}

#: Scalar metric names returned by batched_uot_solve for ok rows.
_ARM3_MICRO_METRICS: tuple[str, ...] = (
    "T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau"
)


def run_uot_batch_with_events(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    uot_cfg: UOTSolveConfig,
    pair_meta: pd.DataFrame,
    tau_external: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the batched UOT solver ONCE and return both a scalar metrics DataFrame
    and per-prototype event mass tensors.

    This helper directly calls slotar.uot.batched_uot_solve (bypassing
    run_uot_batch_safe in common.py) so that prototype-level event masses can
    be extracted from the same single solver call.  Do NOT call the solver
    twice for the same A/B tensors.

    Scalar DataFrame is column-compatible with common.run_uot_batch_safe.
    Prototype event masses are computed via extract_prototype_event_marginals
    using the locked proportional-allocation formulas:

        T_k[i, k] = T[i]     * A[i, k] / (S_src[i] + eps)   — pi1 approximation
        D_k[i, k] = D_pos[i] * A[i, k] / (S_src[i] + eps)   — max(0, A-pi1) approx
        B_k[i, k] = B_pos[i] * B[i, k] / (S_tgt[i] + eps)   — max(0, B-pi2) approx

    These are equivalent to the locked definitions:
        T_k = pi1_{i,k}
        D_k = max(0, A_{i,k} - pi1_{i,k})
        B_k = max(0, B_{i,k} - pi2_{i,k})
    under the proportional-allocation approximation of pi1 and pi2.

    Non-ok rows receive NaN for all prototype event masses.

    Parameters
    ----------
    A : np.ndarray, shape (N, K)
        Source-side density tensors.
    B : np.ndarray, shape (N, K)
        Target-side density tensors.
    lambda_pl : np.ndarray, shape (N,)
        Per-row regularisation parameters.
    kernels : sequence of log-kernel arrays
        Precomputed log-kernels from precompute_logKernels.
    uot_cfg : UOTSolveConfig
        Numerical configuration.
    pair_meta : pd.DataFrame, shape (N, ...)
        Pair metadata aligned to the batch dimension.
    tau_external : np.ndarray, shape (N,) or None
        Per-row tau thresholds for retention computation.

    Returns
    -------
    df_result : pd.DataFrame
        Scalar metrics + pair metadata.  Column-compatible with
        common.run_uot_batch_safe output.
    T_k : np.ndarray, shape (N, K)
        Per-prototype transport mass (NaN for non-ok rows).
    B_k : np.ndarray, shape (N, K)
        Per-prototype creation (birth) mass (NaN for non-ok rows).
    D_k : np.ndarray, shape (N, K)
        Per-prototype destruction (death) mass (NaN for non-ok rows).
    """
    n = pair_meta.shape[0]
    if A.shape[0] != n or B.shape[0] != n:
        raise ValueError(
            "run_uot_batch_with_events: pair_meta must align with batch dimension of A/B"
        )
    if lambda_pl.shape != (n,):
        raise ValueError(
            "run_uot_batch_with_events: lambda_pl must have shape [N] matching pair_meta"
        )
    if tau_external is not None and tau_external.shape != (n,):
        raise ValueError(
            "run_uot_batch_with_events: tau_external must have shape [N] matching pair_meta"
        )

    # ------------------------------------------------------------------
    # Single solver call — no re-solve below
    # ------------------------------------------------------------------
    solver_metrics, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=uot_cfg,
        tau_external=tau_external,
    )

    # ------------------------------------------------------------------
    # Build scalar DataFrame (equivalent to common.run_uot_batch_safe)
    # ------------------------------------------------------------------
    result = pair_meta.reset_index(drop=True).copy()
    for metric_name, values in solver_metrics.items():
        result[metric_name] = values

    result["lambda_pl"] = lambda_pl
    result["uot_status"] = pd.Series(status, dtype="object")

    # bypass_reason
    unknown_statuses = sorted(set(status) - set(_ARM3_STATUS_TO_BYPASS))
    if unknown_statuses:
        raise ValueError(
            f"run_uot_batch_with_events: unknown uot_status values: {unknown_statuses}"
        )
    result["bypass_reason"] = result["uot_status"].map(_ARM3_STATUS_TO_BYPASS)

    # mass_pruned_ratio
    ratios = np.zeros(n, dtype=float)
    for idx in range(n):
        _, ratios[idx] = compute_active_mask(A[idx], B[idx], uot_cfg.n_min_proto)
    result["mass_pruned_ratio"] = ratios
    result["n_min_proto_used"] = float(uot_cfg.n_min_proto)

    S0 = np.sum(A, axis=1, dtype=float)
    S1 = np.sum(B, axis=1, dtype=float)
    result["S0"] = S0
    result["S1"] = S1
    result["scale_ratio"] = np.divide(
        S1, S0,
        out=np.full_like(S0, np.nan, dtype=float),
        where=S0 > 0.0,
    )
    result["log_scale"] = np.where(
        result["scale_ratio"] > 0.0,
        np.log(result["scale_ratio"]),
        np.nan,
    )
    result["U"] = result["B_pos"] + result["D_pos"]

    not_ok = result["uot_status"] != "ok"
    for metric_name in _ARM3_MICRO_METRICS:
        result.loc[not_ok, metric_name] = np.nan
    result.loc[not_ok, "U"] = np.nan

    # ------------------------------------------------------------------
    # Prototype event extraction — uses already-computed scalar results;
    # no additional solver call.
    # Formulas (locked):
    #   T_k[i,k] = T[i] * A[i,k] / (S_src[i] + eps)
    #   D_k[i,k] = D_pos[i] * A[i,k] / (S_src[i] + eps)
    #   B_k[i,k] = B_pos[i] * B[i,k] / (S_tgt[i] + eps)
    # ------------------------------------------------------------------
    T_k, B_k, D_k = extract_prototype_event_marginals(A, B, result)

    return result, T_k, B_k, D_k
