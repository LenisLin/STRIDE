"""Transition wrapper for the canonical STRIDE OT/Sinkhorn backend."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stride.adapters import ot_sinkhorn as _stride_ot_sinkhorn
from stride.adapters.ot_sinkhorn import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
    OBSERVATION_METRIC_NAMES,
    ObservationMatchConfig,
    STATUS_OK,
    _active_mask_from_combined_mass,
    compute_active_state_support,
    weighted_quantile,
)

_EXTRACTION_TARGET_PLAN_ELEMENTS = _stride_ot_sinkhorn._EXTRACTION_TARGET_PLAN_ELEMENTS


def build_observation_kernels(
    C: np.ndarray,
    eps_schedule: Sequence[float],
    cost_scale: float = 1.0,
) -> list[np.ndarray]:
    """Compatibility wrapper over canonical kernel building."""
    return _stride_ot_sinkhorn.build_observation_kernels(
        C=C,
        eps_schedule=eps_schedule,
        cost_scale=cost_scale,
    )


def calibrate_match_penalty(
    A: np.ndarray,
    B: np.ndarray,
    candidate_grid: Sequence[float],
    kernels: Sequence[np.ndarray],
    cfg: ObservationMatchConfig,
    target_alpha: float = 0.05,
) -> float:
    """Compatibility wrapper over canonical penalty calibration."""
    return _stride_ot_sinkhorn.calibrate_match_penalty(
        A=A,
        B=B,
        candidate_grid=candidate_grid,
        kernels=kernels,
        cfg=cfg,
        target_alpha=target_alpha,
    )


def batched_uot_solve(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    cfg: ObservationMatchConfig,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
    return_plan: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Compatibility wrapper that preserves legacy extraction tuning."""
    _stride_ot_sinkhorn._EXTRACTION_TARGET_PLAN_ELEMENTS = _EXTRACTION_TARGET_PLAN_ELEMENTS
    return _stride_ot_sinkhorn.batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
        external_support_mask=external_support_mask,
        return_plan=return_plan,
    )


__all__ = [
    "ObservationMatchConfig",
    "batched_uot_solve",
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
]
