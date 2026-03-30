"""Grouped loss definitions and total-objective assembly for STRIDE fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossWeights:
    """Relative weights for grouped loss components."""

    data_fit: float = 1.0
    structural_bias: float = 0.0
    open_channel_control: float = 0.0


@dataclass(frozen=True)
class LossBreakdown:
    """Grouped loss summary for one fitting step."""

    data_fit: float
    structural_bias: float
    open_channel_control: float
    total: float


def compute_data_fit_loss(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute an L2 data-fit loss between observed and reconstructed payloads."""
    obs = np.asarray(observed, dtype=float)
    rec = np.asarray(reconstructed, dtype=float)
    return float(np.sum(np.square(obs - rec), dtype=float))


def compute_structural_bias_loss(A: np.ndarray, *, target_diagonal_fraction: float | None = None) -> float:
    """Compute a simple structural-bias penalty on the continuity operator."""
    matrix = np.asarray(A, dtype=float)
    diagonal = float(np.trace(matrix))
    total = float(np.sum(matrix, dtype=float))
    if total <= 0.0:
        return 0.0
    diagonal_fraction = diagonal / total
    if target_diagonal_fraction is None:
        return 0.0
    return float((diagonal_fraction - float(target_diagonal_fraction)) ** 2)


def compute_open_channel_control_loss(
    d: np.ndarray,
    e: np.ndarray,
    *,
    target_total: float | None = None,
) -> float:
    """Compute a simple open-channel control penalty on depletion/emergence mass."""
    depletion = float(np.sum(np.asarray(d, dtype=float), dtype=float))
    emergence = float(np.sum(np.asarray(e, dtype=float), dtype=float))
    total = depletion + emergence
    if target_total is None:
        return 0.0
    return float((total - float(target_total)) ** 2)


def evaluate_loss_bundle(
    *,
    observed: np.ndarray,
    reconstructed: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    weights: LossWeights | None = None,
    target_diagonal_fraction: float | None = None,
    target_open_channel_total: float | None = None,
) -> LossBreakdown:
    """Evaluate grouped loss components and return the weighted total."""
    resolved_weights = weights or LossWeights()
    data_fit = compute_data_fit_loss(observed, reconstructed)
    structural = compute_structural_bias_loss(A, target_diagonal_fraction=target_diagonal_fraction)
    open_control = compute_open_channel_control_loss(d, e, target_total=target_open_channel_total)
    total = (
        resolved_weights.data_fit * data_fit
        + resolved_weights.structural_bias * structural
        + resolved_weights.open_channel_control * open_control
    )
    return LossBreakdown(
        data_fit=data_fit,
        structural_bias=structural,
        open_channel_control=open_control,
        total=float(total),
    )


def build_total_objective(*args: object, **kwargs: object) -> LossBreakdown:
    """Thin public alias for the grouped loss bundle."""
    return evaluate_loss_bundle(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "LossBreakdown",
    "LossWeights",
    "build_total_objective",
    "compute_data_fit_loss",
    "compute_open_channel_control_loss",
    "compute_structural_bias_loss",
    "evaluate_loss_bundle",
]
