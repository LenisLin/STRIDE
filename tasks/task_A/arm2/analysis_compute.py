"""
Module: tasks.task_A.arm2.analysis_compute

One-time rerun/compute layer for the post-hoc Arm-II focused rewrite.

This module is the only place where the focused rewrite should:
- rebuild UOT solver settings for the frozen startup slice,
- rerun UOT on the fixed Arm-II pair set,
- rerun same-pair Balanced OT on the fixed Arm-II pair set,
- expose pair-level and pair-by-prototype compute surfaces for all active
  prototypes.

These surfaces must be rich enough for downstream modules to build:
- all-prototype transport summaries,
- all-prototype unmatched summaries,
- all-prototype recurrence,
- later prototype extraction/views without touching solver internals again.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from slotar.exceptions import ERR_UOT_NUMERICAL
from slotar.uot import (
    STATUS_OK,
    UOTSolveConfig,
    batched_uot_solve,
    precompute_logKernels,
)
from slotar.utils import compute_active_mask

from .analysis_contract import (
    BalancedPlanBundle,
    ComputedArm2Surfaces,
    LoadedArm2Inputs,
    PairPrototypeTransportSurface,
    PairPrototypeUnmatchedSurface,
    UotPlanBundle,
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with explicit NaN padding on zero denominators."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize a dense `[N, K]` matrix row-wise with NaN on zero totals."""

    values = np.asarray(values, dtype=float)
    totals = np.sum(values, axis=1, dtype=float)
    return _safe_divide(values, totals[:, None])


def _active_proto_ids(inputs: LoadedArm2Inputs) -> np.ndarray:
    """Return the validated active prototype IDs on the shared tensor axis."""

    proto_ids = np.asarray(inputs.stage0.active_prototype_ids, dtype=int)
    if proto_ids.ndim != 1:
        raise ValueError("Active prototype IDs must be a 1D array")
    if proto_ids.size == 0:
        raise ValueError("Active prototype ID list is empty")
    if not np.array_equal(proto_ids, np.sort(np.unique(proto_ids))):
        raise ValueError("Active prototype IDs must be unique and sorted")
    if proto_ids.min() < 0 or proto_ids.max() >= int(inputs.pair_tensors.k_full):
        raise ValueError("Active prototype IDs fall outside the shared prototype axis")
    return proto_ids


def _aligned_pair_metric_frame(
    inputs: LoadedArm2Inputs,
    columns: list[str],
) -> pd.DataFrame:
    """Align metrics-parquet columns to the stable sorted pair order."""

    lookup = inputs.metrics_df.loc[:, ["pair_id", *columns]].copy()
    aligned = inputs.pair_tensors.pair_metadata.loc[:, ["pair_id"]].merge(
        lookup,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[columns].isna().any().any():
        missing = aligned.loc[aligned[columns].isna().any(axis=1), "pair_id"].astype(str).tolist()
        raise ValueError(f"Metrics parquet is missing required aligned values for pair_id={missing}")
    return aligned


def _build_transport_surface(
    *,
    source_transport_marginal: np.ndarray,
    target_transport_marginal: np.ndarray,
    proto_ids: np.ndarray,
) -> PairPrototypeTransportSurface:
    """Project full-axis marginals to active-prototype transport surfaces."""

    source_abs_full = np.asarray(source_transport_marginal, dtype=float)
    target_abs_full = np.asarray(target_transport_marginal, dtype=float)
    source_abs = source_abs_full[:, proto_ids]
    target_abs = target_abs_full[:, proto_ids]
    return PairPrototypeTransportSurface(
        proto_ids=proto_ids.copy(),
        source_abs=source_abs,
        source_share=_normalize_rows(source_abs),
        target_abs=target_abs,
        target_share=_normalize_rows(target_abs),
    )


def _validate_surface_bundle(
    inputs: LoadedArm2Inputs,
    pair_level_transport: pd.DataFrame,
    uot_transport_surface: PairPrototypeTransportSurface,
    uot_unmatched_surface: PairPrototypeUnmatchedSurface,
    balanced_transport_surface: PairPrototypeTransportSurface,
) -> None:
    """Hard-fail on pair-axis or prototype-axis inconsistencies."""

    pair_count = int(inputs.pair_tensors.pair_metadata.shape[0])
    proto_ids = _active_proto_ids(inputs)

    if pair_level_transport.shape[0] != pair_count:
        raise ValueError(
            "Pair-level transport frame is misaligned with the ordered Arm-II pair axis: "
            f"expected_rows={pair_count}, observed_rows={pair_level_transport.shape[0]}"
        )

    for label, surface in (
        ("uot_transport_surface", uot_transport_surface),
        ("balanced_transport_surface", balanced_transport_surface),
    ):
        if not np.array_equal(np.asarray(surface.proto_ids, dtype=int), proto_ids):
            raise ValueError(f"{label} proto_ids do not match the active prototype axis")
        expected_shape = (pair_count, proto_ids.size)
        observed_shapes = (
            np.asarray(surface.source_abs).shape,
            np.asarray(surface.source_share).shape,
            np.asarray(surface.target_abs).shape,
            np.asarray(surface.target_share).shape,
        )
        if any(shape != expected_shape for shape in observed_shapes):
            raise ValueError(
                f"{label} arrays do not share the expected [pair, active_proto] shape: "
                f"expected={expected_shape}, observed={observed_shapes}"
            )

    if not np.array_equal(np.asarray(uot_unmatched_surface.proto_ids, dtype=int), proto_ids):
        raise ValueError("uot_unmatched_surface proto_ids do not match the active prototype axis")
    unmatched_shape = (pair_count, proto_ids.size)
    observed_unmatched_shapes = (
        np.asarray(uot_unmatched_surface.destroy_abs).shape,
        np.asarray(uot_unmatched_surface.destroy_share).shape,
        np.asarray(uot_unmatched_surface.birth_abs).shape,
        np.asarray(uot_unmatched_surface.birth_share).shape,
    )
    if any(shape != unmatched_shape for shape in observed_unmatched_shapes):
        raise ValueError(
            "UOT unmatched arrays do not share the expected [pair, active_proto] shape: "
            f"expected={unmatched_shape}, observed={observed_unmatched_shapes}"
        )


# ---------------------------------------------------------------------------
# Frozen startup-slice solver configuration
# ---------------------------------------------------------------------------


def build_uot_solver_config(task_config: dict[str, Any]) -> UOTSolveConfig:
    """Build the frozen UOT solver configuration for the post-hoc Arm-II pass."""

    uot_params = task_config.get("uot_params")
    if not isinstance(uot_params, dict):
        raise ValueError("Task config is missing the required 'uot_params' mapping")

    required = ("eps_schedule", "max_iter", "tol", "eta_floor", "n_min_proto")
    missing = [key for key in required if key not in uot_params]
    if missing:
        raise ValueError(f"Task config uot_params is missing required keys: {missing}")

    return UOTSolveConfig(
        eps_schedule=tuple(float(eps) for eps in uot_params["eps_schedule"]),
        max_iter=int(uot_params["max_iter"]),
        tol=float(uot_params["tol"]),
        eta_floor=float(uot_params["eta_floor"]),
        n_min_proto=float(uot_params["n_min_proto"]),
        tau_q=float(uot_params.get("tau_q", 0.25)),
        tau_mode=str(uot_params.get("tau_mode", "external_fixed_by_task")),
    )


def prepare_uot_solver_assets(
    inputs: LoadedArm2Inputs,
) -> tuple[UOTSolveConfig, tuple[np.ndarray, ...], np.ndarray]:
    """Build the shared UOT solver assets once for one Arm-II analysis chain."""

    cfg = build_uot_solver_config(inputs.task_config)
    aligned = _aligned_pair_metric_frame(inputs, ["lambda_pl"])
    lambda_pl = aligned["lambda_pl"].to_numpy(dtype=float)
    if not np.isfinite(lambda_pl).all() or (lambda_pl <= 0.0).any():
        raise ValueError("Aligned Arm-II lambda_pl values must be finite and strictly positive")

    kernels = tuple(
        precompute_logKernels(
            inputs.stage0.cost_matrix,
            cfg.eps_schedule,
            s_C=inputs.stage0.cost_scale,
        )
    )
    return cfg, kernels, lambda_pl


# ---------------------------------------------------------------------------
# Solver reruns
# ---------------------------------------------------------------------------


def rerun_uot(inputs: LoadedArm2Inputs) -> UotPlanBundle:
    """Rerun UOT on the fixed ordered Arm-II pair set exactly once."""

    cfg, kernels, lambda_pl = prepare_uot_solver_assets(inputs)
    metrics, details, status = batched_uot_solve(
        A=inputs.pair_tensors.A_density,
        B=inputs.pair_tensors.B_density,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=None,
        return_plan=False,
    )

    transport_mass = np.asarray(metrics["T"], dtype=float)
    source_transport_marginal = np.asarray(details["source_transport_marginal"], dtype=float)
    target_transport_marginal = np.asarray(details["target_transport_marginal"], dtype=float)
    if source_transport_marginal.shape != inputs.pair_tensors.A_density.shape:
        raise ValueError("UOT source transport marginal shape does not match the fixed Arm-II pair tensor")
    if target_transport_marginal.shape != inputs.pair_tensors.B_density.shape:
        raise ValueError("UOT target transport marginal shape does not match the fixed Arm-II pair tensor")

    status_columns = pd.DataFrame(
        {
            "lambda_pl": lambda_pl,
            "uot_status": pd.Series(status, dtype="object"),
            "T": np.asarray(metrics["T"], dtype=float),
            "D_pos": np.asarray(metrics["D_pos"], dtype=float),
            "B_pos": np.asarray(metrics["B_pos"], dtype=float),
            "U": np.asarray(metrics["D_pos"], dtype=float) + np.asarray(metrics["B_pos"], dtype=float),
            "d_rel": np.asarray(metrics["d_rel"], dtype=float),
            "b_rel": np.asarray(metrics["b_rel"], dtype=float),
            "M": np.asarray(metrics["M"], dtype=float),
            "R": np.asarray(metrics["R"], dtype=float),
            "tau": np.asarray(metrics["tau"], dtype=float),
        }
    )
    return UotPlanBundle(
        source_transport_marginal=source_transport_marginal,
        target_transport_marginal=target_transport_marginal,
        transport_mass=transport_mass,
        status_columns=status_columns,
    )


def rerun_balanced_ot(inputs: LoadedArm2Inputs) -> BalancedPlanBundle:
    """Rerun same-pair Balanced OT on the fixed ordered Arm-II pair set once."""

    try:
        import ot
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise ImportError("POT is required for the Task-A Balanced OT comparator") from exc

    cfg = build_uot_solver_config(inputs.task_config)
    A = inputs.pair_tensors.A_density
    B = inputs.pair_tensors.B_density
    n_items, k_full = A.shape
    scaled_cost = np.asarray(inputs.stage0.cost_matrix, dtype=float) / float(inputs.stage0.cost_scale)

    source_transport_marginal = np.full((n_items, k_full), np.nan, dtype=float)
    target_transport_marginal = np.full((n_items, k_full), np.nan, dtype=float)
    balanced_status = np.full(n_items, "ok", dtype=object)
    balanced_cost = np.full(n_items, np.nan, dtype=float)
    source_total_mass = np.sum(A, axis=1, dtype=float)
    target_total_mass = np.sum(B, axis=1, dtype=float)

    for idx in range(n_items):
        try:
            active_mask, _ = compute_active_mask(A[idx], B[idx], cfg.n_min_proto)
            if not np.any(active_mask):
                balanced_status[idx] = "empty_support_after_prune"
                continue

            a_masked = A[idx, active_mask]
            b_masked = B[idx, active_mask]
            sum_a = float(np.sum(a_masked, dtype=float))
            sum_b = float(np.sum(b_masked, dtype=float))
            if not np.isfinite(sum_a) or not np.isfinite(sum_b) or sum_a <= 0.0 or sum_b <= 0.0:
                balanced_status[idx] = "empty_mass_after_prune"
                continue

            masked_cost = scaled_cost[np.ix_(active_mask, active_mask)]
            masked_plan = np.asarray(
                ot.emd(a_masked / sum_a, b_masked / sum_b, masked_cost),
                dtype=float,
            )
            total_mass = float(np.sum(masked_plan, dtype=float))
            if not np.isfinite(total_mass) or total_mass <= 0.0:
                balanced_status[idx] = "empty_balanced_plan"
                continue

            normalized_plan = masked_plan / total_mass
            full_source = np.zeros(k_full, dtype=float)
            full_target = np.zeros(k_full, dtype=float)
            full_source[active_mask] = np.sum(normalized_plan, axis=1, dtype=float) * source_total_mass[idx]
            full_target[active_mask] = np.sum(normalized_plan, axis=0, dtype=float) * target_total_mass[idx]
            source_transport_marginal[idx] = full_source
            target_transport_marginal[idx] = full_target
            balanced_cost[idx] = float(np.sum(normalized_plan * masked_cost, dtype=float))
        except Exception as exc:  # pragma: no cover - exact branch depends on OT backend
            balanced_status[idx] = f"balanced_plan_failure::{type(exc).__name__}"

    status_columns = pd.DataFrame(
        {
            "balanced_status": pd.Series(balanced_status, dtype="object"),
            "M_balanced": balanced_cost,
        }
    )
    return BalancedPlanBundle(
        source_transport_marginal=source_transport_marginal,
        target_transport_marginal=target_transport_marginal,
        source_total_mass=source_total_mass,
        target_total_mass=target_total_mass,
        status_columns=status_columns,
    )


# ---------------------------------------------------------------------------
# Pair-level and pair-by-prototype compute surfaces
# ---------------------------------------------------------------------------


def build_pair_level_transport_frame(
    inputs: LoadedArm2Inputs,
    uot_plan: UotPlanBundle,
    balanced_plan: BalancedPlanBundle,
) -> pd.DataFrame:
    """
    Build the pair-level transport/unmatched frame used by downstream summaries.

    This pair-level frame is the scalar audit layer. It is distinct from the
    all-prototype surfaces below.
    """

    pair_level = inputs.pair_tensors.pair_metadata.reset_index(drop=True).copy()
    pair_level["source_total_mass"] = np.sum(inputs.pair_tensors.A_density, axis=1, dtype=float)
    pair_level["target_total_mass"] = np.sum(inputs.pair_tensors.B_density, axis=1, dtype=float)
    pair_level["uot_status"] = uot_plan.status_columns["uot_status"].astype(str)
    pair_level["balanced_status"] = balanced_plan.status_columns["balanced_status"].astype(str)

    for column in ("T", "D_pos", "B_pos", "U", "M"):
        pair_level[column] = pd.to_numeric(uot_plan.status_columns[column], errors="coerce").astype(float)
    pair_level["M_balanced"] = pd.to_numeric(
        balanced_plan.status_columns["M_balanced"],
        errors="coerce",
    ).astype(float)

    pair_level["T_abs"] = pair_level["T"].astype(float)
    pair_level["U_abs"] = pair_level["U"].astype(float)
    total_burden = pair_level["T_abs"].to_numpy(dtype=float) + pair_level["U_abs"].to_numpy(dtype=float)
    pair_level["transport_fraction"] = _safe_divide(
        pair_level["T_abs"].to_numpy(dtype=float),
        total_burden,
    )
    pair_level["unmatched_fraction"] = _safe_divide(
        pair_level["U_abs"].to_numpy(dtype=float),
        total_burden,
    )
    pair_level["balanced_minus_uot"] = (
        pd.to_numeric(pair_level["M_balanced"], errors="coerce").astype(float)
        - pd.to_numeric(pair_level["M"], errors="coerce").astype(float)
    )
    pair_level["balanced_to_uot_ratio"] = _safe_divide(
        pd.to_numeric(pair_level["M_balanced"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(pair_level["M"], errors="coerce").to_numpy(dtype=float),
    )
    return pair_level


def build_uot_pair_prototype_transport_surface(
    inputs: LoadedArm2Inputs,
    uot_plan: UotPlanBundle,
) -> PairPrototypeTransportSurface:
    """Build the all-prototype pair-by-prototype UOT transport surface."""

    proto_ids = _active_proto_ids(inputs)
    return _build_transport_surface(
        source_transport_marginal=np.asarray(uot_plan.source_transport_marginal, dtype=float),
        target_transport_marginal=np.asarray(uot_plan.target_transport_marginal, dtype=float),
        proto_ids=proto_ids,
    )


def build_uot_pair_prototype_unmatched_surface(
    inputs: LoadedArm2Inputs,
    uot_transport_surface: PairPrototypeTransportSurface,
) -> PairPrototypeUnmatchedSurface:
    """Build the all-prototype pair-by-prototype UOT unmatched surface."""

    proto_ids = np.asarray(uot_transport_surface.proto_ids, dtype=int)
    destroy_abs = np.maximum(
        inputs.pair_tensors.A_density[:, proto_ids] - uot_transport_surface.source_abs,
        0.0,
    )
    birth_abs = np.maximum(
        inputs.pair_tensors.B_density[:, proto_ids] - uot_transport_surface.target_abs,
        0.0,
    )
    return PairPrototypeUnmatchedSurface(
        proto_ids=proto_ids.copy(),
        destroy_abs=destroy_abs,
        destroy_share=_normalize_rows(destroy_abs),
        birth_abs=birth_abs,
        birth_share=_normalize_rows(birth_abs),
    )


def build_balanced_pair_prototype_transport_surface(
    inputs: LoadedArm2Inputs,
    balanced_plan: BalancedPlanBundle,
) -> PairPrototypeTransportSurface:
    """Build the all-prototype pair-by-prototype Balanced-OT transport surface."""

    proto_ids = _active_proto_ids(inputs)
    return _build_transport_surface(
        source_transport_marginal=np.asarray(balanced_plan.source_transport_marginal, dtype=float),
        target_transport_marginal=np.asarray(balanced_plan.target_transport_marginal, dtype=float),
        proto_ids=proto_ids,
    )


# ---------------------------------------------------------------------------
# Top-level compute bundle
# ---------------------------------------------------------------------------


def compute_all_surfaces(inputs: LoadedArm2Inputs) -> ComputedArm2Surfaces:
    """
    Execute the one-time compute stage for the focused Arm-II rewrite.

    Downstream analysis modules should consume this bundle instead of rerunning
    solver internals or reconstructing pair-by-prototype surfaces themselves.
    """

    uot_plan = rerun_uot(inputs)
    balanced_plan = rerun_balanced_ot(inputs)
    pair_level_transport = build_pair_level_transport_frame(
        inputs=inputs,
        uot_plan=uot_plan,
        balanced_plan=balanced_plan,
    )
    uot_transport_surface = build_uot_pair_prototype_transport_surface(inputs, uot_plan)
    uot_unmatched_surface = build_uot_pair_prototype_unmatched_surface(
        inputs,
        uot_transport_surface,
    )
    balanced_transport_surface = build_balanced_pair_prototype_transport_surface(
        inputs,
        balanced_plan,
    )
    _validate_surface_bundle(
        inputs=inputs,
        pair_level_transport=pair_level_transport,
        uot_transport_surface=uot_transport_surface,
        uot_unmatched_surface=uot_unmatched_surface,
        balanced_transport_surface=balanced_transport_surface,
    )
    return ComputedArm2Surfaces(
        pair_level_transport=pair_level_transport,
        uot_plan=uot_plan,
        balanced_plan=balanced_plan,
        uot_transport_surface=uot_transport_surface,
        uot_unmatched_surface=uot_unmatched_surface,
        balanced_transport_surface=balanced_transport_surface,
    )
