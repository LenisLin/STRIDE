"""Loss scale and initialization records for STRIDE losses.

Task: expose objective scale initialization, normalization floors, and
optimizer-start constants. Reference: ``docs/stride_design_freeze.md`` keeps
objective scale initialization separate from the off-diagonal-seeded optimizer
start and records both in successful-fit provenance.
"""
from __future__ import annotations

from .assembly import (
    EPSILON_NORM,
    NUMERICAL_MIN_MASS,
    OFFDIAG_INIT_MASS,
    FovCostScale,
    ObjectiveContext,
    ScaleInit,
)

__all__ = [
    "EPSILON_NORM",
    "NUMERICAL_MIN_MASS",
    "OFFDIAG_INIT_MASS",
    "FovCostScale",
    "ObjectiveContext",
    "ScaleInit",
]
