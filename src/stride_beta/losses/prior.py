"""Prior-block loss helpers for STRIDE.

Task: expose open-channel complexity and geometry/locality terms used by
``L_prior = mean(normalized_L_open, L_geometry_effective)``.
Reference: ``docs/stride_design_freeze.md`` defines
``L_open_raw = mean(d_p) + mean(e_p)`` and
``L_geometry_effective = geometry_effective_weight * normalized_L_geometry``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._constants import GEOMETRY_EFFECTIVE_WEIGHT
from ._parameters import _normalized_geometry_cost, _validate_parameters, _validate_raw_loss

if TYPE_CHECKING:  # pragma: no cover
    from ..geometry.state_geometry import StateGeometry
    from ._parameters import ADEState


def compute_open_raw(params: "ADEState") -> Any:
    """Return ``mean(d_p) + mean(e_p)`` over fitted patients/components."""
    _, d, e, _ = _validate_parameters(params)
    raw = d.mean() + e.mean()
    _validate_raw_loss(raw, name="L_open_raw")
    return raw


def compute_geometry_raw(params: "ADEState", geometry: "StateGeometry") -> Any:
    """Return cohort mean ``(1/K) * sum_i sum_j A_p[i,j] * C_norm[i,j]``."""
    A, _, _, _ = _validate_parameters(params)
    K = int(A.shape[1])
    C_norm = _normalized_geometry_cost(geometry, K=K, device=A.device)
    per_patient = (A * C_norm.unsqueeze(0)).sum(dim=(1, 2)) / float(K)
    raw = per_patient.mean()
    _validate_raw_loss(raw, name="L_geometry_raw")
    return raw

__all__ = ["GEOMETRY_EFFECTIVE_WEIGHT", "compute_geometry_raw", "compute_open_raw"]
