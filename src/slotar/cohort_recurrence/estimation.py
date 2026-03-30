"""Transition wrapper for canonical STRIDE recurrence estimation helpers."""
from __future__ import annotations

from stride.latent.recurrence import (
    RecurrenceConfig,
    build_recurrence_result,
    estimate_recurrence,
    summarize_recurrence_support,
)

__all__ = [
    "RecurrenceConfig",
    "build_recurrence_result",
    "estimate_recurrence",
    "summarize_recurrence_support",
]
