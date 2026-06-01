"""Reference optimizer schedules for canonical STRIDE relation fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SchedulerPolicy = Literal["none", "CosineAnnealingLR"]

REFERENCE_OPTIMIZER_PROTOCOL = "two_phase_warmup20_main100plus_v1"


@dataclass(frozen=True)
class CosineConfig:
    """Fixed cosine-decay parameters for the reference optimizer protocol."""

    T_max: int = 200
    eta_min: float = 0.0


@dataclass(frozen=True)
class OptimizationStage:
    """One fixed stage in the reference STRIDE optimizer schedule."""

    name: str
    lr: float
    min_steps: int
    max_steps: int
    scheduler_policy: SchedulerPolicy = "none"
    allow_early_stop: bool = False
    description: str = ""

    @property
    def fixed_steps(self) -> bool:
        """Return whether this stage is constrained to one exact step count."""
        return int(self.min_steps) == int(self.max_steps)


@dataclass(frozen=True)
class OptimizationSchedule:
    """Ordered optimizer schedule plus fixed cosine-decay metadata."""

    protocol_name: str
    stages: tuple[OptimizationStage, ...]
    cosine: CosineConfig
    early_stop_eligibility_policy: str
    active_stage_index: int = 0

    @property
    def active_stage(self) -> OptimizationStage:
        """Return the current active stage."""
        return self.stages[self.active_stage_index]

    @property
    def warmup_stage(self) -> OptimizationStage:
        """Return the reference warm-up stage."""
        return self.stages[0]

    @property
    def main_stage(self) -> OptimizationStage:
        """Return the reference main stage."""
        return self.stages[1]


def build_reference_schedule() -> OptimizationSchedule:
    """Build the fixed reference optimizer protocol for canonical STRIDE fits."""
    return OptimizationSchedule(
        protocol_name=REFERENCE_OPTIMIZER_PROTOCOL,
        stages=(
            OptimizationStage(
                name="warmup",
                lr=0.02,
                min_steps=20,
                max_steps=20,
                scheduler_policy="none",
                allow_early_stop=False,
                description="Stabilize the constrained A/d/e parameterization before the main fit.",
            ),
            OptimizationStage(
                name="main",
                lr=0.05,
                min_steps=100,
                max_steps=200,
                scheduler_policy="CosineAnnealingLR",
                allow_early_stop=True,
                description="Run the reference full-objective fit with cosine learning-rate decay.",
            ),
        ),
        cosine=CosineConfig(),
        early_stop_eligibility_policy="main_after_min_steps",
    )


def build_default_schedule() -> OptimizationSchedule:
    """Compatibility alias for the canonical reference schedule."""
    return build_reference_schedule()


def advance_stage(schedule: OptimizationSchedule) -> OptimizationSchedule:
    """Advance to the next stage when available."""
    if schedule.active_stage_index >= len(schedule.stages) - 1:
        return schedule
    return OptimizationSchedule(
        protocol_name=schedule.protocol_name,
        stages=schedule.stages,
        cosine=schedule.cosine,
        early_stop_eligibility_policy=schedule.early_stop_eligibility_policy,
        active_stage_index=schedule.active_stage_index + 1,
    )


__all__ = [
    "CosineConfig",
    "OptimizationSchedule",
    "OptimizationStage",
    "REFERENCE_OPTIMIZER_PROTOCOL",
    "SchedulerPolicy",
    "advance_stage",
    "build_default_schedule",
    "build_reference_schedule",
]
