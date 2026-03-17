"""
Module: tasks.task_A.common
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from slotar.exceptions import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)
from slotar.uot import UOTSolveConfig, batched_uot_solve
from slotar.utils import compute_active_mask

MICRO_METRICS: tuple[str, ...] = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")
STATUS_TO_BYPASS = {
    "ok": None,
    ERR_UOT_EMPTY_MASS_SOURCE: "S0_zero",
    ERR_UOT_EMPTY_MASS_TARGET: "S1_zero",
    ERR_UOT_EMPTY_SUPPORT: "empty_support_after_prune",
    ERR_UOT_NUMERICAL: "uot_numerical_failure",
}


def _require_count_mass_mode(mass_mode: str) -> None:
    if mass_mode != "count":
        raise ValueError(f"Patch-2 only supports mass_mode='count', got {mass_mode!r}")


def _build_roi_vectors(adata: AnnData, k_full: int) -> dict[str, np.ndarray]:
    obs = adata.obs.loc[:, ["roi_id", "proto_id"]].copy()
    if obs["proto_id"].isna().any():
        raise ValueError("adata.obs['proto_id'] contains NA values")

    proto_ids = obs["proto_id"].astype(int).to_numpy()
    if (proto_ids < 0).any() or (proto_ids >= k_full).any():
        raise ValueError("adata.obs['proto_id'] contains values outside the shared prototype axis")

    roi_vectors: dict[str, np.ndarray] = {}
    obs["roi_id"] = obs["roi_id"].astype(str)
    for roi_id, group in obs.groupby("roi_id", sort=False, observed=False):
        roi_vectors[str(roi_id)] = np.bincount(
            group["proto_id"].astype(int).to_numpy(),
            minlength=k_full,
        ).astype(float)
    return roi_vectors


def assemble_tensors(
    adata: AnnData,
    roi_pairs: pd.DataFrame,
    k_full: int,
    mass_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble full-axis count-mode tensors of shape [N, K_full] for Task A.
    """
    _require_count_mass_mode(mass_mode)

    if roi_pairs.empty:
        return (
            np.zeros((0, k_full), dtype=float),
            np.zeros((0, k_full), dtype=float),
            np.zeros(0, dtype=float),
        )

    roi_vectors = _build_roi_vectors(adata, k_full)
    missing = sorted(set(roi_pairs["roi_a"]).union(set(roi_pairs["roi_b"])) - set(roi_vectors))
    if missing:
        raise ValueError(f"ROI pairs reference unknown roi_id values: {missing}")

    A = np.vstack([roi_vectors[str(roi_id)] for roi_id in roi_pairs["roi_a"]]).astype(float, copy=False)
    B = np.vstack([roi_vectors[str(roi_id)] for roi_id in roi_pairs["roi_b"]]).astype(float, copy=False)

    sum_a = np.sum(A, axis=1, dtype=float)
    sum_b = np.sum(B, axis=1, dtype=float)
    denom = np.maximum(sum_a, sum_b)
    mass_gap = np.divide(
        np.abs(sum_a - sum_b),
        denom,
        out=np.zeros_like(sum_a, dtype=float),
        where=denom > 0.0,
    )
    return A, B, mass_gap


def _compute_mass_pruned_ratio(A: np.ndarray, B: np.ndarray, n_min_proto: float) -> np.ndarray:
    ratios = np.zeros(A.shape[0], dtype=float)
    for idx in range(A.shape[0]):
        _, ratios[idx] = compute_active_mask(A[idx], B[idx], n_min_proto)
    return ratios


def _map_bypass_reason(status: pd.Series) -> pd.Series:
    unknown = sorted(set(status) - set(STATUS_TO_BYPASS))
    if unknown:
        raise ValueError(f"Unknown uot_status values: {unknown}")
    return status.map(STATUS_TO_BYPASS)


def run_uot_batch_safe(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    uot_cfg: UOTSolveConfig,
    pair_meta: pd.DataFrame,
    tau_external: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run the frozen batched UOT solver and attach Task-A-specific audit fields.
    """
    if pair_meta.shape[0] != A.shape[0] or pair_meta.shape[0] != B.shape[0]:
        raise ValueError("pair_meta must align with the batch dimension of A/B")
    if lambda_pl.shape != (pair_meta.shape[0],):
        raise ValueError("lambda_pl must have shape [N] matching pair_meta")
    if tau_external is not None and tau_external.shape != (pair_meta.shape[0],):
        raise ValueError("tau_external must have shape [N] matching pair_meta")

    solver_metrics, _details, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=uot_cfg,
        tau_external=tau_external,
    )

    result = pair_meta.reset_index(drop=True).copy()
    for metric_name, values in solver_metrics.items():
        result[metric_name] = values

    result["lambda_pl"] = lambda_pl
    result["uot_status"] = pd.Series(status, dtype="object")
    result["bypass_reason"] = _map_bypass_reason(result["uot_status"])
    result["mass_pruned_ratio"] = _compute_mass_pruned_ratio(A, B, uot_cfg.n_min_proto)
    result["n_min_proto_used"] = float(uot_cfg.n_min_proto)

    S0 = np.sum(A, axis=1, dtype=float)
    S1 = np.sum(B, axis=1, dtype=float)
    result["S0"] = S0
    result["S1"] = S1
    result["scale_ratio"] = np.divide(
        S1,
        S0,
        out=np.full_like(S0, np.nan, dtype=float),
        where=S0 > 0.0,
    )
    result["log_scale"] = np.where(result["scale_ratio"] > 0.0, np.log(result["scale_ratio"]), np.nan)
    result["U"] = result["B_pos"] + result["D_pos"]

    not_ok = result["uot_status"] != "ok"
    for metric_name in MICRO_METRICS:
        result.loc[not_ok, metric_name] = np.nan
    result.loc[not_ok, "U"] = np.nan

    return result


def run_balanced_ot_batch(
    A: np.ndarray,
    B: np.ndarray,
    cost_matrix: np.ndarray,
    n_min_proto: float,
) -> np.ndarray:
    """
    Run rowwise same-pair shape-only Balanced OT on the scaled cost domain.
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape for Balanced OT")

    scaled_cost = np.asarray(cost_matrix, dtype=float)
    if scaled_cost.shape != (A.shape[1], A.shape[1]):
        raise ValueError(
            "Balanced OT cost_matrix shape must match the shared prototype axis: "
            f"expected {(A.shape[1], A.shape[1])}, got {scaled_cost.shape}"
        )

    try:
        import ot
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise ImportError("POT is required for the Task-A Balanced OT comparator") from exc

    balanced_cost = np.full(A.shape[0], np.nan, dtype=float)
    for idx in range(A.shape[0]):
        active_mask, _ = compute_active_mask(A[idx], B[idx], n_min_proto)
        if not np.any(active_mask):
            continue

        a_masked = A[idx, active_mask]
        b_masked = B[idx, active_mask]
        sum_a = float(np.sum(a_masked, dtype=float))
        sum_b = float(np.sum(b_masked, dtype=float))
        if not np.isfinite(sum_a) or not np.isfinite(sum_b) or sum_a <= 0.0 or sum_b <= 0.0:
            continue

        a_shape = a_masked / sum_a
        b_shape = b_masked / sum_b
        balanced_cost[idx] = float(
            ot.emd2(a_shape, b_shape, scaled_cost[np.ix_(active_mask, active_mask)])
        )

    return balanced_cost
