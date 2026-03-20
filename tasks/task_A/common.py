"""
Module: tasks.task_A.common
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
try:
    from anndata import AnnData
except ModuleNotFoundError:  # pragma: no cover - optional dependency for non-AnnData paths
    class AnnData:  # type: ignore[override]
        """Lightweight fallback so tensor-only Task-A workflows can import common."""

        pass

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
ARM_LOCKED_MASS_MODES: dict[str, str] = {
    "A1_baseline": "density",
    "A1_broken_reference": "density",
    "A2_cross_compartment": "density",
    "A3_uq_stress": "density",
}

def resolve_task_a_mass_mode(
    config: Mapping[str, Any],
    arm_name: str,
) -> str:
    """
    Resolve the evaluator-facing mass_mode for one enabled Task-A arm.

    Canonical mixed-arm configuration should use data.mass_mode_by_arm. The
    legacy global data.mass_mode surface is still accepted for single-mode arm
    sets where every enabled arm shares the same locked mass semantics.
    """
    if arm_name not in ARM_LOCKED_MASS_MODES:
        raise ValueError(f"Unknown Task-A arm {arm_name!r} for mass-mode resolution")

    expected_mode = ARM_LOCKED_MASS_MODES[arm_name]
    data_cfg_raw = config.get("data", {})
    if data_cfg_raw is None:
        data_cfg_raw = {}
    if not isinstance(data_cfg_raw, Mapping):
        raise ValueError("Task-A config key 'data' must be a mapping")

    enabled_arms = config.get("enabled_arms", [])
    enabled_known_arms = [str(name) for name in enabled_arms if str(name) in ARM_LOCKED_MASS_MODES]
    enabled_modes = {ARM_LOCKED_MASS_MODES[name] for name in enabled_known_arms}

    mass_mode_by_arm = data_cfg_raw.get("mass_mode_by_arm")
    if mass_mode_by_arm is not None:
        if not isinstance(mass_mode_by_arm, Mapping):
            raise ValueError("Task-A data.mass_mode_by_arm must be a mapping when provided")
        if arm_name not in mass_mode_by_arm:
            raise ValueError(
                f"Task-A data.mass_mode_by_arm is missing enabled arm {arm_name!r}"
            )
        resolved_mode = str(mass_mode_by_arm[arm_name])
    else:
        if len(enabled_modes) > 1 and "mass_mode" in data_cfg_raw:
            raise ValueError(
                "Task-A enabled_arms mix count-only and density-only arms; "
                "set data.mass_mode_by_arm to declare per-arm mass semantics explicitly"
            )
        if "mass_mode" not in data_cfg_raw:
            raise ValueError(
                "Task-A config must declare either data.mass_mode or data.mass_mode_by_arm"
            )
        resolved_mode = str(data_cfg_raw["mass_mode"])

    if resolved_mode != expected_mode:
        raise ValueError(
            f"Task-A arm {arm_name!r} requires mass_mode={expected_mode!r}, "
            f"got {resolved_mode!r}"
        )
    return resolved_mode


def validate_task_a_mass_mode_surface(
    config: Mapping[str, Any],
    enabled_arms: Sequence[str],
) -> None:
    """
    Fail fast when the configured Task-A mass-mode surface is incoherent.
    """
    for arm_name in enabled_arms:
        resolve_task_a_mass_mode(config, arm_name)


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


def assemble_pair_tensors_from_roi_vectors(
    roi_vectors: dict[str, np.ndarray],
    roi_pairs: pd.DataFrame,
    k_full: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble full-axis [N, K_full] tensors from a prepared roi_id -> vector mapping.
    """
    if roi_pairs.empty:
        return (
            np.zeros((0, k_full), dtype=float),
            np.zeros((0, k_full), dtype=float),
            np.zeros(0, dtype=float),
        )

    missing = sorted(set(roi_pairs["roi_a"]).union(set(roi_pairs["roi_b"])) - set(roi_vectors))
    if missing:
        raise ValueError(f"ROI pairs reference unknown roi_id values: {missing}")

    A = np.vstack([roi_vectors[str(roi_id)] for roi_id in roi_pairs["roi_a"]]).astype(float, copy=False)
    B = np.vstack([roi_vectors[str(roi_id)] for roi_id in roi_pairs["roi_b"]]).astype(float, copy=False)
    if A.shape[1] != k_full or B.shape[1] != k_full:
        raise ValueError(
            "Assembled ROI vectors do not match the declared shared prototype axis: "
            f"expected K={k_full}, observed {(A.shape[1], B.shape[1])}"
        )

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


def assemble_tensors(
    adata: AnnData,
    roi_pairs: pd.DataFrame,
    k_full: int,
    mass_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble full-axis Task-A tensors of shape [N, K_full] in the requested mass mode.
    """
    if mass_mode == "count":
        roi_vectors = _build_roi_vectors(adata, k_full)
    elif mass_mode == "density":
        roi_vectors, _roi_count_vectors, _roi_total_areas = build_task_a_density_reference_from_adata(
            adata,
            k_full=k_full,
        )
    else:
        raise ValueError(f"Unsupported Task-A mass_mode {mass_mode!r}")
    return assemble_pair_tensors_from_roi_vectors(roi_vectors, roi_pairs, k_full)


def build_task_a_density_reference_from_arrays(
    spatial_xy: np.ndarray,
    roi_ids: np.ndarray,
    proto_ids: np.ndarray,
    *,
    k_full: int,
    block_size_units: float | None = None,
    coord_to_mm2: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """
    Build Arm-3-consistent original-ROI density vectors for Task A.
    """
    from .arm3.block_partition import (
        build_full_coverage_density_vectors,
        build_grid_partition,
        compute_roi_block_summary,
    )
    from .arm3.constants import COORD_TO_MM2, DEFAULT_BLOCK_SIZE_UNITS

    block_size = float(DEFAULT_BLOCK_SIZE_UNITS if block_size_units is None else block_size_units)
    area_scale = float(COORD_TO_MM2 if coord_to_mm2 is None else coord_to_mm2)
    block_area_mm2 = block_size ** 2 * area_scale

    block_frame, roi_block_universe = build_grid_partition(
        spatial_xy=np.asarray(spatial_xy, dtype=float),
        roi_ids=np.asarray(roi_ids, dtype=object),
        proto_ids=np.asarray(proto_ids, dtype=int),
        block_size_units=block_size,
        coord_to_mm2=area_scale,
    )
    roi_block_summary = compute_roi_block_summary(
        block_frame=block_frame,
        roi_block_universe=roi_block_universe,
        k_full=k_full,
        block_area_mm2=block_area_mm2,
    )
    roi_density_vectors, roi_total_areas = build_full_coverage_density_vectors(
        roi_block_summary=roi_block_summary,
        k_full=k_full,
    )
    count_columns = [f"count_k{k}" for k in range(k_full)]
    roi_count_vectors = {
        str(roi_id): df[count_columns].sum(axis=0).to_numpy(dtype=float)
        for roi_id, df in roi_block_summary.items()
    }
    return roi_density_vectors, roi_count_vectors, roi_total_areas


def build_task_a_density_reference_from_adata(
    adata: AnnData,
    *,
    k_full: int,
    block_size_units: float | None = None,
    coord_to_mm2: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """
    Build Arm-3-consistent original-ROI density vectors directly from Task-A AnnData.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("Validated AnnData is missing obsm['spatial'] required for density construction")
    obs = adata.obs.loc[:, ["roi_id", "proto_id"]].copy()
    if obs["proto_id"].isna().any():
        raise ValueError("adata.obs['proto_id'] contains NA values")

    proto_ids = obs["proto_id"].astype(int).to_numpy()
    if (proto_ids < 0).any() or (proto_ids >= k_full).any():
        raise ValueError("adata.obs['proto_id'] contains values outside the shared prototype axis")

    return build_task_a_density_reference_from_arrays(
        spatial_xy=np.asarray(adata.obsm["spatial"], dtype=float),
        roi_ids=obs["roi_id"].astype(str).to_numpy(dtype=object),
        proto_ids=proto_ids,
        k_full=k_full,
        block_size_units=block_size_units,
        coord_to_mm2=coord_to_mm2,
    )


def _compute_mass_pruned_ratio(A: np.ndarray, B: np.ndarray, n_min_proto: float) -> np.ndarray:
    ratios = np.zeros(A.shape[0], dtype=float)
    for idx in range(A.shape[0]):
        _, ratios[idx] = compute_active_mask(A[idx], B[idx], n_min_proto)
    return ratios


def _compute_mass_pruned_ratio_from_mask(
    A: np.ndarray,
    B: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    ratios = np.zeros(A.shape[0], dtype=float)
    for idx in range(A.shape[0]):
        combined = np.asarray(A[idx], dtype=float) + np.asarray(B[idx], dtype=float)
        total_mass = float(np.sum(combined, dtype=float))
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            ratios[idx] = 0.0
            continue
        mask = np.asarray(support_mask[idx], dtype=bool)
        pruned_mass = float(np.sum(combined[~mask], dtype=float))
        ratios[idx] = pruned_mass / total_mass
    return ratios


def compute_pruning_audit_fields(
    A: np.ndarray,
    B: np.ndarray,
    n_min_proto: float,
    *,
    external_support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute authoritative and diagnostic pruning ratios for Task-A audit output.
    """
    density_ratios = _compute_mass_pruned_ratio(A, B, n_min_proto)
    if external_support_mask is None:
        return density_ratios, None
    return (
        _compute_mass_pruned_ratio_from_mask(A, B, external_support_mask),
        density_ratios,
    )


def _map_bypass_reason(status: pd.Series) -> pd.Series:
    unknown = sorted(set(status) - set(STATUS_TO_BYPASS))
    if unknown:
        raise ValueError(f"Unknown uot_status values: {unknown}")
    return status.map(STATUS_TO_BYPASS)


def build_task_a_uot_result_frame(
    *,
    pair_meta: pd.DataFrame,
    solver_metrics: dict[str, np.ndarray],
    status: np.ndarray,
    lambda_pl: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    uot_cfg: UOTSolveConfig,
    external_support_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Attach the shared Task-A audit surface to one batched UOT solve result.
    """
    result = pair_meta.reset_index(drop=True).copy()
    for metric_name, values in solver_metrics.items():
        result[metric_name] = values

    result["lambda_pl"] = lambda_pl
    result["uot_status"] = pd.Series(status, dtype="object")
    result["bypass_reason"] = _map_bypass_reason(result["uot_status"])
    mass_pruned_ratio, density_active_pruned_ratio = compute_pruning_audit_fields(
        A,
        B,
        uot_cfg.n_min_proto,
        external_support_mask=external_support_mask,
    )
    result["mass_pruned_ratio"] = mass_pruned_ratio
    if density_active_pruned_ratio is not None:
        result["density_active_pruned_ratio"] = density_active_pruned_ratio
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


def run_uot_batch_safe(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    uot_cfg: UOTSolveConfig,
    pair_meta: pd.DataFrame,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
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
    if external_support_mask is not None and external_support_mask.shape != A.shape:
        raise ValueError("external_support_mask must have shape [N, K] matching A/B")

    solver_metrics, _details, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=uot_cfg,
        tau_external=tau_external,
        external_support_mask=external_support_mask,
    )
    return build_task_a_uot_result_frame(
        pair_meta=pair_meta,
        solver_metrics=solver_metrics,
        status=status,
        lambda_pl=lambda_pl,
        A=A,
        B=B,
        uot_cfg=uot_cfg,
        external_support_mask=external_support_mask,
    )


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
