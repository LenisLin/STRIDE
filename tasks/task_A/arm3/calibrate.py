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

Open implementation facts that still require human confirmation before coding:
- Exact pairing structure for within-compartment tau reference pools.
- Config keys: 'arm3.tau_grid' and 'arm3.target_retention'.
- Whether the lambda calibration uses all six ordered directions or only
  confirmatory families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from slotar.uot import STATUS_OK, UOTSolveConfig, batched_uot_solve, calibrate_joint_lambda


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
    - Run batched_uot_solve on these pairs with tau_external=None to obtain the
      unconstrained transport plans.
    - The Pi-weighted mean cost per pair is M_i (returned directly by the solver).
    - Compute tau_c = quantile({M_i : status == "ok"}, tau_q).

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
    frozen_lambdas : dict[str, float]
        Family-level lambda_dens from calibrate_lambda_dens. Lambda for
        within-compartment pairs is the mean of all three family lambdas (since
        within-compartment pairs map to no cross-compartment family).
    uot_cfg : UOTSolveConfig
        Frozen solver configuration.
    kernels : Sequence[np.ndarray]
        Pre-computed log-kernels.
    tau_q : float
        Quantile level applied to the per-pair M distribution. Default 0.5
        (median). Override via config key 'arm3.tau_q'.

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

        metrics, status = batched_uot_solve(
            A=A,
            B=B,
            lambda_pl=lambda_pl,
            kernels=kernels,
            cfg=uot_cfg,
            tau_external=None,
        )

        # M_i is the Pi-weighted mean cost per pair: sum_jk C_jk * Pi_jk / T.
        # Taking the tau_q quantile of {M_i} gives tau_c.
        ok_mask = status == STATUS_OK
        M_ok = metrics["M"][ok_mask]
        M_finite = M_ok[np.isfinite(M_ok)]

        if M_finite.size == 0:
            raise ValueError(
                f"calibrate_tau_by_compartment: no valid UOT solutions for "
                f"compartment={compartment!r} reference pool "
                f"({n_pairs} pairs attempted, 0 succeeded). Cannot calibrate tau."
            )

        tau_c = float(np.quantile(M_finite, tau_q))
        result[compartment] = tau_c

    return result
