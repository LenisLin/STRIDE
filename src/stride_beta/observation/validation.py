"""Observation-layer validation helpers for STRIDE matching surfaces."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import ContractError


def validate_observation_match_inputs(
    A: np.ndarray,
    B: np.ndarray,
    match_penalty: np.ndarray,
    kernels: Sequence[np.ndarray],
) -> None:
    """Validate tensors for observation-layer cloud matching."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    match_penalty = np.asarray(match_penalty, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ContractError("A and B must be 2D arrays of shape [N, K]")
    if A.shape != B.shape:
        raise ContractError(f"A and B shape mismatch: {A.shape} vs {B.shape}")

    n_items, n_states = A.shape
    if match_penalty.shape != (n_items,):
        raise ContractError(f"match_penalty must have shape {(n_items,)}, got {match_penalty.shape}")

    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ContractError("A/B contain NaN/Inf")
    if (A < 0).any() or (B < 0).any():
        raise ContractError("A/B must be non-negative")

    if not np.isfinite(match_penalty).all() or (match_penalty <= 0).any():
        raise ContractError("match_penalty must be finite and strictly positive")

    if len(kernels) == 0:
        raise ContractError("kernels must be a non-empty epsilon schedule")
    for index, log_kernel in enumerate(kernels):
        log_kernel = np.asarray(log_kernel, dtype=float)
        if log_kernel.shape != (n_states, n_states):
            raise ContractError(
                f"kernels[{index}] must have shape {(n_states, n_states)}, got {log_kernel.shape}"
            )
        if not np.isfinite(log_kernel).all():
            raise ContractError(f"kernels[{index}] contains NaN/Inf")


__all__ = ["validate_observation_match_inputs"]
