"""Compatibility solver-path wrappers for legacy `slotar.uot` imports."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..backends import ot_sinkhorn as _ot_sinkhorn
from ..observation import ObservationMatchConfig
from .status import STATUS_OK
from .tables import MICRO_METRICS

UOTSolveConfig = ObservationMatchConfig
_EXTRACTION_TARGET_PLAN_ELEMENTS = _ot_sinkhorn._EXTRACTION_TARGET_PLAN_ELEMENTS


def precompute_logKernels(
    C: np.ndarray,
    eps_schedule: Sequence[float],
    s_C: float = 1.0,
) -> list[np.ndarray]:
    """Compatibility wrapper over the canonical kernel builder."""
    return _ot_sinkhorn.build_observation_kernels(C=C, eps_schedule=eps_schedule, cost_scale=s_C)


def calibrate_joint_lambda(
    A: np.ndarray,
    B: np.ndarray,
    lambda_grid: Sequence[float],
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    target_alpha: float = 0.05,
) -> float:
    """Compatibility wrapper over canonical penalty calibration."""
    return _ot_sinkhorn.calibrate_match_penalty(
        A=A,
        B=B,
        candidate_grid=lambda_grid,
        kernels=kernels,
        cfg=cfg,
        target_alpha=target_alpha,
    )


def batched_uot_solve(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
    return_plan: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Compatibility wrapper that preserves legacy extraction tuning."""
    _ot_sinkhorn._EXTRACTION_TARGET_PLAN_ELEMENTS = _EXTRACTION_TARGET_PLAN_ELEMENTS
    return _ot_sinkhorn.batched_uot_solve(
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
    "MICRO_METRICS",
    "ObservationMatchConfig",
    "STATUS_OK",
    "UOTSolveConfig",
    "_EXTRACTION_TARGET_PLAN_ELEMENTS",
    "batched_uot_solve",
    "calibrate_joint_lambda",
    "precompute_logKernels",
]
