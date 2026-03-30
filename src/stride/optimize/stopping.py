"""Stopping criteria and runtime pathology checks for staged optimization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StoppingCriteria:
    """Simple stopping criteria for iterative optimization."""

    max_steps: int
    tolerance: float = 1e-6
    patience: int = 5


@dataclass(frozen=True)
class PathologyCheck:
    """Named pathology flag emitted during optimization."""

    name: str
    triggered: bool
    detail: str


def should_stop(loss_history: np.ndarray, *, step: int, criteria: StoppingCriteria) -> bool:
    """Return whether optimization should stop at the current step."""
    history = np.asarray(loss_history, dtype=float).reshape(-1)
    if step >= criteria.max_steps:
        return True
    if history.size <= criteria.patience:
        return False
    recent = history[-criteria.patience :]
    return bool(np.nanmax(recent) - np.nanmin(recent) <= criteria.tolerance)


def detect_pathologies(A: np.ndarray, d: np.ndarray, e: np.ndarray) -> tuple[PathologyCheck, ...]:
    """Detect simple numerical or structural pathologies in one patient relation."""
    matrix = np.asarray(A, dtype=float)
    depletion = np.asarray(d, dtype=float)
    emergence = np.asarray(e, dtype=float)
    total = float(
        np.sum(matrix, dtype=float)
        + np.sum(depletion, dtype=float)
        + np.sum(emergence, dtype=float)
    )
    diagonal = float(np.trace(matrix))
    diagonal_fraction = (
        diagonal / np.sum(matrix, dtype=float) if np.sum(matrix, dtype=float) > 0.0 else float("nan")
    )
    open_fraction = (
        float(np.sum(depletion, dtype=float) + np.sum(emergence, dtype=float)) / total
        if total > 0.0
        else float("nan")
    )
    return (
        PathologyCheck(
            name="diagonal_collapse",
            triggered=bool(np.isfinite(diagonal_fraction) and diagonal_fraction > 0.99 and np.sum(matrix) > 0.0),
            detail=f"diagonal_fraction={diagonal_fraction}",
        ),
        PathologyCheck(
            name="open_channel_overuse",
            triggered=bool(np.isfinite(open_fraction) and open_fraction > 0.95),
            detail=f"open_fraction={open_fraction}",
        ),
        PathologyCheck(
            name="nonfinite_payload",
            triggered=bool(
                (not np.isfinite(matrix).all())
                or (not np.isfinite(depletion).all())
                or (not np.isfinite(emergence).all())
            ),
            detail="finite_check",
        ),
    )


__all__ = ["PathologyCheck", "StoppingCriteria", "detect_pathologies", "should_stop"]
