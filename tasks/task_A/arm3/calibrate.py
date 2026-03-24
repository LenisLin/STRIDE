"""
Module: tasks.task_A.arm3.calibrate

Phase 4 of the Arm-3 pipeline: full-coverage calibration of lambda_dens and tau.

Responsibilities:
- Calibrate one lambda_dens per unordered pair family using the full-coverage
  original ROI family reference pool (same calibrate_joint_lambda scan pattern
  as Arm-2, applied to density-mode tensors).
- Calibrate one tau per compartment using within-compartment full-coverage ROI
  reference pools.

Design constraints (all locked):
- Calibration is performed once on full-coverage original ROI reference pools only.
- No calibration is performed on pseudo-ROIs.
- tau is compartment-specific and is assigned by compartment_a in all inference.
- No mixed family-level tau values (no pooled TC-IM threshold from TC-TC + IM-IM).
- lambda_dens calibration pools both ordered directions within each unordered
  family (same joint-calibration pooling strategy as Arm-2).
- Full-coverage reference is built from the exact same frozen block universe used
  by bootstrap, not from uns['roi_areas'].

Implementation note: all open facts from the skeleton were resolved prior to coding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from slotar.uot import STATUS_OK, UOTSolveConfig, batched_uot_solve, calibrate_joint_lambda
from slotar.utils import weighted_quantile


def calibrate_lambda_dens(
    roi_density_vectors: dict[str, np.ndarray],
    pair_meta: pd.DataFrame,
    k_full: int,
    lambda_grid: tuple[float, ...],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    target_alpha: float = 0.05,
) -> dict[str, float]:
    """
    Calibrate lambda_dens once per pair family on full-coverage density tensors.

    Follows the same calibrate_joint_lambda scan-and-select pattern as Arm-2,
    applied to density-mode tensors instead of count-mode tensors. For each
    unordered pair family, both ordered directions are pooled for joint
    calibration (same strategy as Arm-2).

    Calibration is performed on the full-coverage original ROI reference pool
    only. Pseudo-ROI lambda recalibration is prohibited.

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Full-coverage density vectors per roi_id (cells / mm^2), shape (K,) each.
        Must be built from the frozen block universe (not from uns['roi_areas']).
    pair_meta : pd.DataFrame
        Full ordered pair metadata containing all six directions used for pooled
        calibration. Required columns: roi_a, roi_b, pair_family.
    k_full : int
        Prototype axis dimension (25 for Task A).
    lambda_grid : tuple[float, ...]
        Lambda candidates to scan. From config key 'arm3.lambda_grid' (falls back
        to 'arm2.lambda_grid' in the runner if absent).
    uot_cfg : UOTSolveConfig
        Frozen solver configuration.
    kernels : Sequence[np.ndarray]
        Pre-computed log-kernels (from precompute_logKernels).
    target_alpha : float
        Target unmatched fraction for calibration selection.

    Returns
    -------
    dict[str, float]
        Family-keyed frozen lambda_dens values:
        {"TC-IM": float, "IM-PT": float, "TC-PT": float}
    """
    from .constants import ARM3_PAIR_FAMILIES

    if pair_meta.empty:
        raise ValueError("calibrate_lambda_dens: pair_meta is empty")
    for col in ("roi_a", "roi_b", "pair_family"):
        if col not in pair_meta.columns:
            raise ValueError(
                f"calibrate_lambda_dens: pair_meta is missing required column {col!r}"
            )

    result: dict[str, float] = {}

    for pair_family in ARM3_PAIR_FAMILIES:
        family_mask = pair_meta["pair_family"].astype(str) == pair_family
        family_rows = pair_meta.loc[family_mask]

        if family_rows.empty:
            raise ValueError(
                f"calibrate_lambda_dens: no rows found for pair_family={pair_family!r} "
                "in pair_meta — cannot calibrate family-level lambda_dens"
            )

        # Assemble density tensors for this family; both ordered directions are
        # already present in pair_meta_full (same pooling logic as Arm-2).
        try:
            A = np.stack(
                [roi_density_vectors[str(roi_id)] for roi_id in family_rows["roi_a"]],
                axis=0,
            ).astype(float, copy=False)
            B = np.stack(
                [roi_density_vectors[str(roi_id)] for roi_id in family_rows["roi_b"]],
                axis=0,
            ).astype(float, copy=False)
        except KeyError as exc:
            raise ValueError(
                f"calibrate_lambda_dens: roi_id {exc} referenced in pair_meta "
                "is absent from roi_density_vectors"
            ) from exc

        if A.shape[1] != k_full:
            raise ValueError(
                f"calibrate_lambda_dens: density vectors have {A.shape[1]} prototype "
                f"dimensions but k_full={k_full} — mismatch in family {pair_family!r}"
            )

        family_lambda = calibrate_joint_lambda(
            A=A,
            B=B,
            lambda_grid=lambda_grid,
            kernels=kernels,
            cfg=uot_cfg,
            target_alpha=target_alpha,
        )
        result[pair_family] = family_lambda

    return result


def calibrate_tau_by_compartment(
    roi_density_vectors: dict[str, np.ndarray],
    roi_compartment_map: dict[str, str],
    roi_patient_map: dict[str, str],
    k_full: int,
    scaled_cost_matrix: np.ndarray,
    frozen_lambdas: dict[str, float],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    tau_q: float = 0.5,
) -> dict[str, float]:
    """
    Calibrate compartment-specific tau values on within-compartment full-coverage
    reference pools using the Pi-weighted cost quantile rule.

    For each compartment c in {TC, IM, PT}:
    - Assemble all within-patient, same-compartment ordered ROI pairs from the
      full-coverage density reference pool. Pairs are within-patient only; no
      cross-patient pooling.
    - Run batched_uot_solve on these pairs with tau_external=None and
      return_plan=True to obtain the unconstrained transport plans.
    - Pool the scaled cost entries C_jk across all successful plans, weighted by
      the corresponding transport masses Pi_ijk, excluding diagonal self-transport
      entries (j == k) from the quantile support.
    - Compute tau_c as the weighted quantile of that pooled cost distribution.

    tau is compartment-specific and is assigned exclusively by compartment_a in
    all downstream Arm-3 inference. No family-level tau pooling.
    Calibration is on the full-coverage reference pool only; no tau recalibration
    on pseudo-ROIs.

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Full-coverage density vectors per roi_id (cells / mm^2), shape (K,) each.
    roi_compartment_map : dict[str, str]
        Maps roi_id -> compartment label (TC, IM, or PT).
    roi_patient_map : dict[str, str]
        Maps roi_id -> patient_id. Used to enforce within-patient pairing.
    k_full : int
        Prototype axis dimension.
    scaled_cost_matrix : np.ndarray, shape (K, K)
        Explicit scaled cost matrix C used for Pi-weighted pooling.
        This must be passed directly; it is not reconstructed from kernels.
    frozen_lambdas : dict[str, float]
        Family-level lambda_dens from calibrate_lambda_dens. Lambda for
        within-compartment pairs is the mean of all three family lambdas (since
        within-compartment pairs map to no cross-compartment family).
    uot_cfg : UOTSolveConfig
        Frozen solver configuration.
    kernels : Sequence[np.ndarray]
        Pre-computed log-kernels.
    tau_q : float
        Quantile level applied to the pooled non-diagonal Pi-weighted scaled-cost
        distribution. Default 0.5 (median). Override via config key 'arm3.tau_q'.

    Returns
    -------
    dict[str, float]
        Compartment-keyed frozen tau values:
        {"TC": float, "IM": float, "PT": float}

    Raises
    ------
    ValueError
        If any compartment has no within-patient pairs or no valid transport
        solutions.
    """
    from collections import defaultdict

    if not roi_compartment_map:
        raise ValueError("calibrate_tau_by_compartment: roi_compartment_map is empty")
    if not roi_patient_map:
        raise ValueError("calibrate_tau_by_compartment: roi_patient_map is empty")
    if not (0.0 < tau_q < 1.0):
        raise ValueError(
            f"calibrate_tau_by_compartment: tau_q must be in (0, 1), got {tau_q!r}"
        )
    if not frozen_lambdas:
        raise ValueError("calibrate_tau_by_compartment: frozen_lambdas is empty")
    scaled_cost = np.asarray(scaled_cost_matrix, dtype=float)
    if scaled_cost.shape != (k_full, k_full):
        raise ValueError(
            "calibrate_tau_by_compartment: scaled_cost_matrix must have shape "
            f"{(k_full, k_full)}, got {scaled_cost.shape}"
        )
    if not np.isfinite(scaled_cost).all():
        raise ValueError("calibrate_tau_by_compartment: scaled_cost_matrix contains NaN/Inf")

    # Lambda for within-compartment pairs: mean over all calibrated family lambdas.
    # Within-compartment pairs (TC-TC, IM-IM, PT-PT) don't map to any
    # cross-compartment family, so we use the mean of all three as a neutral proxy.
    mean_lambda = float(np.mean(list(frozen_lambdas.values())))

    # Group ROIs by (patient_id, compartment) for within-patient pairing
    patient_comp_rois: dict[tuple[str, str], list[str]] = defaultdict(list)
    for roi_id, compartment in roi_compartment_map.items():
        patient_id = roi_patient_map.get(roi_id, "__unknown__")
        patient_comp_rois[(patient_id, compartment)].append(roi_id)

    COMPARTMENTS = ("TC", "IM", "PT")
    result: dict[str, float] = {}

    for compartment in COMPARTMENTS:
        # Enumerate all within-patient, same-compartment ordered pairs (A ≠ B)
        A_rows: list[np.ndarray] = []
        B_rows: list[np.ndarray] = []

        for (patient_id, comp), rois in patient_comp_rois.items():
            if comp != compartment:
                continue
            for roi_a in rois:
                for roi_b in rois:
                    if roi_a == roi_b:
                        continue
                    A_rows.append(roi_density_vectors[roi_a])
                    B_rows.append(roi_density_vectors[roi_b])

        if not A_rows:
            raise ValueError(
                f"calibrate_tau_by_compartment: no within-patient same-compartment "
                f"pairs found for compartment={compartment!r}. Each patient must "
                "have at least 2 ROIs in this compartment to form a reference pair."
            )

        A = np.stack(A_rows, axis=0).astype(float, copy=False)
        B = np.stack(B_rows, axis=0).astype(float, copy=False)
        n_pairs = A.shape[0]
        lambda_pl = np.full(n_pairs, mean_lambda, dtype=float)

        _metrics, details, status = batched_uot_solve(
            A=A,
            B=B,
            lambda_pl=lambda_pl,
            kernels=kernels,
            cfg=uot_cfg,
            tau_external=None,
            return_plan=True,
        )

        ok_mask = status == STATUS_OK
        ok_plan = np.asarray(details["Pi"][ok_mask], dtype=float)
        tau_c = _weighted_plan_cost_quantile(
            plans=ok_plan,
            scaled_cost_matrix=scaled_cost,
            q=tau_q,
        )

        if not np.isfinite(tau_c):
            raise ValueError(
                f"calibrate_tau_by_compartment: no valid UOT solutions for "
                f"compartment={compartment!r} reference pool "
                f"({n_pairs} pairs attempted, 0 yielded a positive finite pooled plan). "
                "Cannot calibrate tau."
            )

        result[compartment] = float(tau_c)

    return result


def _weighted_plan_cost_quantile(
    plans: np.ndarray,
    scaled_cost_matrix: np.ndarray,
    q: float,
) -> float:
    """
    Compute a pooled Pi-weighted quantile over non-diagonal scaled cost entries.

    Only finite strictly-positive plan entries contribute weight, and diagonal
    self-transport is excluded from support because tau is intended to reflect
    nontrivial within-compartment transport cost rather than identity-preserving
    anchor mass. This avoids materialising zero-mass entries from every dense
    [K, K] plan when pooling.
    """
    plan_arr = np.asarray(plans, dtype=float)
    cost_arr = np.asarray(scaled_cost_matrix, dtype=float)

    if plan_arr.ndim != 3:
        raise ValueError(
            f"_weighted_plan_cost_quantile: plans must be 3D [N, K, K], got {plan_arr.shape}"
        )
    if cost_arr.ndim != 2 or cost_arr.shape[0] != cost_arr.shape[1]:
        raise ValueError(
            "_weighted_plan_cost_quantile: scaled_cost_matrix must be a square [K, K] array"
        )
    if plan_arr.shape[1:] != cost_arr.shape:
        raise ValueError(
            "_weighted_plan_cost_quantile: plan K dimensions must match scaled_cost_matrix"
        )

    flat_cost = cost_arr.reshape(-1)                                       # (K*K,)
    diagonal_mask = np.eye(cost_arr.shape[0], dtype=bool).reshape(-1)
    eligible_cost = np.isfinite(flat_cost) & (~diagonal_mask)              # (K*K,)

    # Vectorised over all N plans at once — avoids per-plan Python loop.
    flat_plans = plan_arr.reshape(plan_arr.shape[0], -1)                   # (N, K*K)
    keep_mask = (
        np.isfinite(flat_plans) & (flat_plans > 0.0) & eligible_cost[None, :]
    )                                                                       # (N, K*K)

    row_has_data = keep_mask.any(axis=1)                                   # (N,)
    if not row_has_data.any():
        return float("nan")

    valid_plans = flat_plans[row_has_data]                                 # (M, K*K)
    valid_keep = keep_mask[row_has_data]                                   # (M, K*K)
    # broadcast_to avoids materialising a full (M, K*K) cost copy.
    pooled_costs = np.broadcast_to(flat_cost, valid_plans.shape)[valid_keep]
    pooled_weights = valid_plans[valid_keep]
    return float(weighted_quantile(pooled_costs, pooled_weights, q))
