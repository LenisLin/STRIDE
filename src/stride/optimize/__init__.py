"""Optimization scheduling and stopping helpers for STRIDE fitting flows."""
from __future__ import annotations

from .full_estimator import (
    FullEstimatorOptimizerConfig,
    FullEstimatorOptimizerResult,
    optimize_full_estimator,
)
from .schedules import (
    OptimizationSchedule,
    OptimizationStage,
    advance_stage,
    build_default_schedule,
)
from .stopping import PathologyCheck, StoppingCriteria, detect_pathologies, should_stop

__all__ = [
    "FullEstimatorOptimizerConfig",
    "FullEstimatorOptimizerResult",
    "OptimizationSchedule",
    "OptimizationStage",
    "PathologyCheck",
    "StoppingCriteria",
    "advance_stage",
    "build_default_schedule",
    "detect_pathologies",
    "optimize_full_estimator",
    "should_stop",
]
