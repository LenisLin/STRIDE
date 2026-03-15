"""
Module: src.slotar.utils
Architecture: Library Level
Constraints:
- Pure utility functions with strict isolation between semantic logic and numerical guards.
"""
from __future__ import annotations

import numpy as np


def _active_mask_from_combined_mass(
    combined_mass: np.ndarray,
    n_min_proto: float,
) -> np.ndarray:
    """
    Shared semantic rule for active-support inclusion.

    Accepts either a single [K] combined mass vector or a batched [N, K] combined
    mass tensor and applies the same inclusive threshold semantics.
    """
    combined = np.asarray(combined_mass, dtype=float)
    return float(n_min_proto) <= combined


def compute_active_mask(
    mass_source: np.ndarray,
    mass_target: np.ndarray,
    n_min_proto: float,
    eta_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Computes the mathematical active support mask for optimal transport, decoupling
    semantic pruning from numerical stabilization.

    Args:
        mass_source: 1D array of shape [K].
        mass_target: 1D array of shape [K].
        n_min_proto: Semantic threshold for active support inclusion.
        eta_floor: Numerical guard constant (used only for zero-division prevention, NOT masking).

    Returns:
        active_mask: 1D boolean array of shape [K]. True if (source + target >= n_min_proto).
        mass_pruned_ratio: Float indicating the ratio of mass lost due to semantic pruning.

    Constraints for Codex:
        1. The mask MUST purely evaluate `mass_source + mass_target >= n_min_proto`.
        2. `mass_pruned_ratio` MUST be calculated on the original mass sum before any 
           `eta_floor` padding is applied.
        3. Do NOT add `eta_floor` to the returned mask or manipulate the mass tensors here.
    """
    del eta_floor  # Numerical guards are intentionally excluded from mask semantics.

    source = np.asarray(mass_source, dtype=float)
    target = np.asarray(mass_target, dtype=float)
    if source.shape != target.shape:
        raise ValueError("mass_source and mass_target must have the same shape")

    combined = source + target
    active_mask = _active_mask_from_combined_mass(combined, n_min_proto)

    total_mass = float(np.sum(combined, dtype=float))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return active_mask, 0.0

    pruned_mass = float(np.sum(combined[~active_mask], dtype=float))
    mass_pruned_ratio = pruned_mass / total_mass
    return active_mask, mass_pruned_ratio


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """
    Computes a weighted quantile, used for setting the retention threshold (tau).
    
    Args:
        values: 1D array of shape [N] containing the data points (e.g., costs).
        weights: 1D array of shape [N] containing the weights (e.g., Pi matrix elements).
        q: Quantile to compute (0.0 <= q <= 1.0).
        
    Returns:
        The interpolated weighted quantile value.
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    wts = np.asarray(weights, dtype=float).reshape(-1)

    if vals.shape != wts.shape:
        raise ValueError("values and weights must have the same shape")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")

    finite = np.isfinite(vals) & np.isfinite(wts)
    vals = vals[finite]
    wts = wts[finite]
    if vals.size == 0:
        return float(np.nan)
    if (wts < 0.0).any():
        raise ValueError("weights must be non-negative")

    total_weight = float(np.sum(wts, dtype=float))
    if total_weight <= 0.0:
        return float(np.nan)

    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    wts = wts[order]

    cumulative = np.cumsum(wts, dtype=float)
    centers = (cumulative - 0.5 * wts) / total_weight

    if vals.size == 1:
        return float(vals[0])
    return float(np.interp(q, centers, vals, left=vals[0], right=vals[-1]))
