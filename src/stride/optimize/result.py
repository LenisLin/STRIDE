"""Structured optimizer results for STRIDE training."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..losses.assembly import LossLedger
from .config import TrainConfig


TrainStatus = Literal["ok", "failed", "deferred"]


@dataclass(frozen=True)
class OptimizerRunInfo:
    """Typed optimizer-runtime facts produced by the training loop."""

    reason: str | None = None
    optimizer_exit_flag: str | None = None
    n_steps: int | None = None
    warmup_steps_completed: int | None = None
    main_steps_completed: int | None = None
    step: int | None = None
    initial_total: float | None = None
    final_total: float | None = None
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    optimizer_protocol: str = ""
    scheduler_policy: str = "none"
    ablation_mode: str = "none"
    message: str | None = None


@dataclass(frozen=True)
class TrainResult:
    """Detached optimizer result surface without audit/provenance assembly."""

    status: TrainStatus
    A: Any | None = None
    d: Any | None = None
    e: Any | None = None
    loss_ledger: LossLedger | None = None
    train_config: TrainConfig | None = None
    run_info: OptimizerRunInfo = field(default_factory=OptimizerRunInfo)
    trace: Mapping[str, Any] | None = None


__all__ = ["OptimizerRunInfo", "TrainResult", "TrainStatus"]
