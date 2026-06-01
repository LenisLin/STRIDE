"""Optimization runtime and stopping helpers for STRIDE fitting flows."""
from __future__ import annotations

from .config import SchedulerPolicy, TrainConfig, validate_train_config
from .model import RelationModel
from .result import OptimizerRunInfo, TrainResult, TrainStatus
from .schedules import (
    CosineConfig,
    OptimizationSchedule,
    OptimizationStage,
    REFERENCE_OPTIMIZER_PROTOCOL,
    advance_stage,
    build_default_schedule,
    build_reference_schedule,
)
from .stopping import PathologyCheck, StoppingCriteria, detect_pathologies, should_stop
from .trainer import run_training

__all__ = [
    "CosineConfig",
    "OptimizationSchedule",
    "OptimizationStage",
    "PathologyCheck",
    "REFERENCE_OPTIMIZER_PROTOCOL",
    "RelationModel",
    "SchedulerPolicy",
    "StoppingCriteria",
    "TrainConfig",
    "OptimizerRunInfo",
    "TrainResult",
    "TrainStatus",
    "advance_stage",
    "build_default_schedule",
    "build_reference_schedule",
    "detect_pathologies",
    "run_training",
    "should_stop",
    "validate_train_config",
]
