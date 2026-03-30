"""Compatibility utility wrappers for migration-era helper imports."""
from __future__ import annotations

import numpy as np

from ..backends.ot_sinkhorn import (
    _active_mask_from_combined_mass,
    compute_active_state_support,
    weighted_quantile,
)


def compute_active_mask(
    mass_source: np.ndarray,
    mass_target: np.ndarray,
    n_min_proto: float,
    eta_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Compatibility alias for `compute_active_state_support(...)`."""
    return compute_active_state_support(
        mass_source=mass_source,
        mass_target=mass_target,
        n_min_proto=n_min_proto,
        eta_floor=eta_floor,
    )


__all__ = [
    "_active_mask_from_combined_mass",
    "compute_active_mask",
    "compute_active_state_support",
    "weighted_quantile",
]
