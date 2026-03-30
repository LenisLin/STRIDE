"""Compatibility wrappers for the older uncertainty-module path."""
from __future__ import annotations

from ..uncertainty import bootstrap_observation_unit, estimate_log_measurement_error


def compute_log_measurement_error(
    theta_replicates: object,
    delta_stabilizer: float = 1e-4,
    s2_lower_bound: float = 1e-6,
) -> tuple[float, bool]:
    """Compatibility alias for `estimate_log_measurement_error(...)`."""
    return estimate_log_measurement_error(
        theta_replicates=theta_replicates,  # type: ignore[arg-type]
        delta_stabilizer=delta_stabilizer,
        s2_lower_bound=s2_lower_bound,
    )


def bootstrap_single_roi(
    adata: object,
    roi_id: str,
    G: int,
    B_boot: int,
) -> dict[str, object]:
    """Compatibility alias for `bootstrap_observation_unit(...)`."""
    return bootstrap_observation_unit(
        adata=adata,  # type: ignore[arg-type]
        observation_id=roi_id,
        G=G,
        B_boot=B_boot,
        observation_key="roi_id",
    )


__all__ = [
    "bootstrap_observation_unit",
    "bootstrap_single_roi",
    "compute_log_measurement_error",
    "estimate_log_measurement_error",
]
