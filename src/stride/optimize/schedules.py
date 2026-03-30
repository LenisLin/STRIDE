"""Staged optimization schedules for canonical relation fitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OptimizationStage:
    """One optimization stage in the canonical schedule."""

    name: str
    max_steps: int
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationSchedule:
    """Ordered sequence of optimization stages."""

    stages: tuple[OptimizationStage, ...]
    active_stage_index: int = 0

    @property
    def active_stage(self) -> OptimizationStage:
        """Return the current active stage."""
        return self.stages[self.active_stage_index]


def build_default_schedule() -> OptimizationSchedule:
    """Build a conservative default staged schedule."""
    return OptimizationSchedule(
        stages=(
            OptimizationStage(
                name="warm_start",
                max_steps=25,
                description="Initialize stable observation-to-patient estimates.",
            ),
            OptimizationStage(
                name="structure_fit",
                max_steps=100,
                description="Fit continuity and open-channel structure.",
            ),
            OptimizationStage(
                name="refine",
                max_steps=50,
                description="Refine the final patient-level relation.",
            ),
        )
    )


def advance_stage(schedule: OptimizationSchedule) -> OptimizationSchedule:
    """Advance to the next schedule stage when available."""
    if schedule.active_stage_index >= len(schedule.stages) - 1:
        return schedule
    return OptimizationSchedule(
        stages=schedule.stages,
        active_stage_index=schedule.active_stage_index + 1,
    )


__all__ = ["OptimizationSchedule", "OptimizationStage", "advance_stage", "build_default_schedule"]
