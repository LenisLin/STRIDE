"""
Package: tasks.task_A.arm3

Arm-3 submodule package for Task A: density-primary coverage-reduction and UQ
stress test on the frozen Stage-0 artifact.

Public surface is intentionally minimal. Import submodules directly for access
to their individual functions.
"""

from .constants import (
    ARM3_ANCHOR_DIRECTIONS,
    ARM3_NAME,
    COORD_TO_MM2,
    COVERAGE_LEVELS,
    DEFAULT_BLOCK_SIZE_UNITS,
    DEFAULT_N_REPS,
    DEFAULT_RNG_SEED,
)

__all__ = [
    "ARM3_ANCHOR_DIRECTIONS",
    "ARM3_NAME",
    "COORD_TO_MM2",
    "COVERAGE_LEVELS",
    "DEFAULT_BLOCK_SIZE_UNITS",
    "DEFAULT_N_REPS",
    "DEFAULT_RNG_SEED",
]
