"""Migration shim for the legacy `slotar.uot` import path."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .compat import uot as _compat_uot
from .compat.status import STATUS_OK
from .compat.tables import MICRO_METRICS

ObservationMatchConfig = _compat_uot.ObservationMatchConfig
UOTSolveConfig = _compat_uot.UOTSolveConfig
_EXTRACTION_TARGET_PLAN_ELEMENTS = _compat_uot._EXTRACTION_TARGET_PLAN_ELEMENTS


def precompute_logKernels(
    C: np.ndarray,
    eps_schedule: Sequence[float],
    s_C: float = 1.0,
) -> list[np.ndarray]:
    """Compatibility wrapper over the canonical kernel builder."""
    return _compat_uot.precompute_logKernels(C=C, eps_schedule=eps_schedule, s_C=s_C)


def calibrate_joint_lambda(
    A: np.ndarray,
    B: np.ndarray,
    lambda_grid: Sequence[float],
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    target_alpha: float = 0.05,
) -> float:
    """Compatibility wrapper over canonical penalty calibration."""
    return _compat_uot.calibrate_joint_lambda(
        A=A,
        B=B,
        lambda_grid=lambda_grid,
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
    _compat_uot._EXTRACTION_TARGET_PLAN_ELEMENTS = _EXTRACTION_TARGET_PLAN_ELEMENTS
    return _compat_uot.batched_uot_solve(
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
