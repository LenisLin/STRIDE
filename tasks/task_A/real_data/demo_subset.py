"""Frozen real-data demo subsets for Task A interface alignment verification.

These subsets are recorded by patient_id only; no derived data is committed.
The actual filtering happens at adapter time via ``--patient-id`` or
``--demo-subset`` arguments to the prepare workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskADemoSubset:
    name: str
    patient_ids: tuple[str, ...]
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "patient_ids": list(self.patient_ids),
            "rationale": self.rationale,
        }


# ---- alignment_v1 ----
# Covers every real ROI-per-domain pattern present in the full cohort:
#   B10  -> 3 TC / 3 IM / 3 PT
#   B12  -> 2 TC / 4 IM / 3 PT
#   B3   -> 3 TC / 3 IM / 4 PT
#   W18  -> 3 TC / 3 IM / 2 PT
# Total: 215,605 cells, 36 ROIs, 4 patients.
# Cheap enough to run in under a minute yet exercises every boundary condition
# in the family-sliced STRIDE adapter path.
ALIGNMENT_V1 = TaskADemoSubset(
    name="alignment_v1",
    patient_ids=("B10", "B12", "B3", "W18"),
    rationale=(
        "Covers every real ROI-per-domain pattern (3/3/3, 2/4/3, 3/3/4, "
        "3/3/2) while staying cheap (215,605 cells, 36 ROIs, 4 patients)"
    ),
)

DEMO_SUBSETS: dict[str, TaskADemoSubset] = {
    ALIGNMENT_V1.name: ALIGNMENT_V1,
}


def resolve_demo_subset(name: str) -> TaskADemoSubset:
    """Resolve a named demo subset or raise ``KeyError``."""
    if name not in DEMO_SUBSETS:
        available = ", ".join(sorted(DEMO_SUBSETS))
        raise KeyError(
            f"Unknown demo subset {name!r}; available: {available}"
        )
    return DEMO_SUBSETS[name]


__all__ = [
    "ALIGNMENT_V1",
    "DEMO_SUBSETS",
    "TaskADemoSubset",
    "resolve_demo_subset",
]
