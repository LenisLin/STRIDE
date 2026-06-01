"""Typed fit-status facts for postfit audit/result assembly."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import ContractError
from ..optimize.result import TrainStatus


FitRunStage = Literal["optimizer", "task_feasibility"]


@dataclass(frozen=True)
class FitRunStatus:
    """Fit-level status facts before public diagnostics are assembled."""

    stage: FitRunStage
    status: TrainStatus
    reason: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in {"optimizer", "task_feasibility"}:
            raise ContractError("FitRunStatus.stage must be 'optimizer' or 'task_feasibility'")
        if self.status not in {"ok", "deferred", "failed"}:
            raise ContractError("FitRunStatus.status must be 'ok', 'deferred', or 'failed'")
        reason = None if self.reason is None else str(self.reason).strip()
        if self.status != "ok" and not reason:
            raise ContractError("non-ok FitRunStatus requires a non-empty reason")
        if not isinstance(self.context, Mapping):
            raise ContractError("FitRunStatus.context must be a mapping")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "context", dict(self.context))


__all__ = ["FitRunStage", "FitRunStatus"]
