"""Optimization scheduling and stopping helpers for STRIDE fitting flows."""
from __future__ import annotations

from .schedules import (
    OptimizationSchedule,
    OptimizationStage,
    advance_stage,
    build_default_schedule,
)
from .stopping import PathologyCheck, StoppingCriteria, detect_pathologies, should_stop

__all__ = [
    "OptimizationSchedule",
    "OptimizationStage",
    "PathologyCheck",
    "StoppingCriteria",
    "advance_stage",
    "build_default_schedule",
    "detect_pathologies",
    "should_stop",
]
