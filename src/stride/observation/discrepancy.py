"""Observation-layer discrepancy helpers backed by OT adapter implementations."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..adapters.ot_sinkhorn import (
    batched_uot_solve,
    build_observation_kernels,
    calibrate_match_penalty,
    compute_active_state_support,
)
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from .contracts import (
    DomainStratifiedMeasure,
    ObservationDiscrepancy,
    ObservationDiscrepancyConfig,
    ObservationDiscrepancyResult,
)


def _cartesian_pair_state_matrices(
    pre_measure: DomainStratifiedMeasure,
    post_measure: DomainStratifiedMeasure,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    pre_state = np.asarray(pre_measure.state_matrix, dtype=float)
    post_state = np.asarray(post_measure.state_matrix, dtype=float)
    if pre_state.ndim != 2 or post_state.ndim != 2:
        raise ContractError("pre_measure and post_measure must expose stacked [N, K] state matrices")
    if pre_state.shape[1] != post_state.shape[1]:
        raise ContractError("pre_measure and post_measure must share the same K-state axis")

    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    pair_metadata: list[dict[str, str]] = []
    for pre_idx, pre_observation in enumerate(pre_measure.observations):
        for post_idx, post_observation in enumerate(post_measure.observations):
            pre_rows.append(pre_state[pre_idx])
            post_rows.append(post_state[post_idx])
            pair_metadata.append(
                {
                    "pre_fov_id": str(pre_observation.fov_id),
                    "post_fov_id": str(post_observation.fov_id),
                    "pre_observation_id": str(pre_observation.observation_id),
                    "post_observation_id": str(post_observation.observation_id),
                }
            )

    return (
        np.vstack(pre_rows).astype(float, copy=False),
        np.vstack(post_rows).astype(float, copy=False),
        pair_metadata,
    )


def match_observation_clouds(
    state_mass_pre: np.ndarray,
    state_mass_post: np.ndarray,
    match_penalty: np.ndarray,
    kernels: Sequence[np.ndarray],
    cfg: ObservationDiscrepancyConfig,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
    return_plan: bool = False,
) -> ObservationDiscrepancyResult:
    """Match paired observation clouds on a shared ``K``-state axis.

    This is the observation-layer adapter boundary: callers provide stacked
    state-mass matrices, precomputed kernels, and solver configuration, and
    receive canonical metrics/details without exposing low-level Sinkhorn internals.
    """
    metrics, details, status = batched_uot_solve(
        A=state_mass_pre,
        B=state_mass_post,
        lambda_pl=match_penalty,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
        external_support_mask=external_support_mask,
        return_plan=return_plan,
    )
    return ObservationDiscrepancyResult(metrics=metrics, details=details, status=status)


def compute_observation_discrepancy(
    pre_measure: DomainStratifiedMeasure,
    post_measure: DomainStratifiedMeasure,
    *,
    match_penalty: np.ndarray,
    geometry: StateGeometry,
    cfg: ObservationDiscrepancyConfig,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
    return_plan: bool = False,
) -> ObservationDiscrepancy:
    """Compute one named discrepancy between two domain-stratified measures.

    The function is intentionally narrow: it assumes both measures already live
    on the same shared state basis and uses the supplied geometry layer only to
    derive observation-matching kernels.
    """
    pre_state_mass, post_state_mass, pair_metadata = _cartesian_pair_state_matrices(
        pre_measure,
        post_measure,
    )
    n_pairs = int(pre_state_mass.shape[0])
    result = match_observation_clouds(
        state_mass_pre=pre_state_mass,
        state_mass_post=post_state_mass,
        match_penalty=np.broadcast_to(np.asarray(match_penalty, dtype=float), (n_pairs,)),
        kernels=build_observation_kernels(
            geometry.cost_matrix,
            cfg.eps_schedule,
            cost_scale=geometry.cost_scale,
        ),
        cfg=cfg,
        tau_external=tau_external,
        external_support_mask=external_support_mask,
        return_plan=return_plan,
    )
    return ObservationDiscrepancy(
        pre_domain=pre_measure.domain_label,
        post_domain=post_measure.domain_label,
        result=result,
        metadata={
            "mass_mode": pre_measure.mass_mode,
            "n_pre_observations": len(pre_measure.observations),
            "n_post_observations": len(post_measure.observations),
            "n_observation_pairs": n_pairs,
            "pair_metadata": pair_metadata,
        },
    )


__all__ = [
    "build_observation_kernels",
    "calibrate_match_penalty",
    "compute_active_state_support",
    "compute_observation_discrepancy",
    "match_observation_clouds",
]
