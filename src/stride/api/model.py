"""Lightweight facade objects for assembled STRIDE model state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..basis import StateBasis
from ..geometry import StateGeometry
from ..optimize import OptimizationSchedule


@dataclass(frozen=True)
class STRIDEModel:
    """Lightweight container for core STRIDE model-layer objects.

    This class packages already-prepared basis, geometry, and scheduling state.
    It is not itself a fitting engine or a promise of the final high-level model
    interface.
    """

    state_basis: StateBasis
    geometry: StateGeometry | None = None
    schedule: OptimizationSchedule | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["STRIDEModel"]
