"""Private objective policies used by controlled estimator evaluations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectivePolicy:
    """Effective objective-term multipliers with fixed block denominators."""

    name: str
    consistency_weight: float = 1.0
    geometry_weight: float = 1.0
    recurrence_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("objective policy name must be non-empty")
        for field_name in (
            "consistency_weight",
            "geometry_weight",
            "recurrence_weight",
        ):
            value = float(getattr(self, field_name))
            if value not in (0.0, 1.0):
                raise ValueError(f"{field_name} must be exactly 0.0 or 1.0")


REFERENCE_OBJECTIVE_POLICY = ObjectivePolicy("reference")
NO_CONSISTENCY_OBJECTIVE_POLICY = ObjectivePolicy(
    "consistency_ablation",
    consistency_weight=0.0,
)
NO_GEOMETRY_OBJECTIVE_POLICY = ObjectivePolicy(
    "geometry_ablation",
    geometry_weight=0.0,
)
NO_RECURRENCE_OBJECTIVE_POLICY = ObjectivePolicy(
    "recurrence_ablation",
    recurrence_weight=0.0,
)
