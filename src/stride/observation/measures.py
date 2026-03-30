"""Observation-layer builders for uniform-mass ROI/FOV community composition."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..basis.state_projection import aggregate_to_state_basis
from ..data.longitudinal import (
    CANONICAL_DOMAIN_KEY,
    CANONICAL_FOV_KEY,
    CANONICAL_TIMEPOINT_KEY,
    LONGITUDINAL_GROUP_KEYS,
    resolve_domain_key,
    resolve_fov_key,
    resolve_state_id_key,
    resolve_timepoint_key,
)
from ..errors import ContractError
from .contracts import DomainStratifiedMeasure, FovObservation, validate_fov_observation

try:
    from anndata import AnnData
except ModuleNotFoundError:  # pragma: no cover - optional dependency for non-AnnData paths
    AnnData = object  # type: ignore[assignment,misc]


def compute_fov_burden(state_mass: np.ndarray) -> float:
    """Compute total burden from one observation-layer state-mass vector."""
    mass = np.asarray(state_mass, dtype=float).reshape(-1)
    if mass.size == 0 or not np.isfinite(mass).all() or (mass < 0.0).any():
        raise ContractError("state_mass must be a finite non-negative 1D array")
    return float(np.sum(mass, dtype=float))


def compute_fov_composition(state_mass: np.ndarray) -> np.ndarray:
    """Compute the composition vector for one observation-layer state-mass vector."""
    mass = np.asarray(state_mass, dtype=float).reshape(-1)
    burden = compute_fov_burden(mass)
    if burden <= 0.0:
        return np.zeros_like(mass, dtype=float)
    return (mass / burden).astype(float, copy=False)


def _build_composite_ids(obs: pd.DataFrame) -> np.ndarray:
    return (
        obs.loc[:, ["patient_id", CANONICAL_TIMEPOINT_KEY, CANONICAL_FOV_KEY]]
        .astype(str)
        .agg("::".join, axis=1)
        .to_numpy(dtype=object)
    )


def build_fov_observations(
    adata: AnnData,
    *,
    state_key: str | None = None,
    n_states: int | None = None,
) -> tuple[FovObservation, ...]:
    """Build canonical ROI/FOV observations from per-cell community-state assignments."""
    active_state_key = state_key or resolve_state_id_key(adata)
    if active_state_key not in adata.obs.columns:
        raise ContractError(f"adata.obs is missing state assignment column {active_state_key!r}")

    timepoint_key = resolve_timepoint_key(adata)
    fov_key = resolve_fov_key(adata)
    domain_key = resolve_domain_key(adata)
    obs = adata.obs.loc[:, ["patient_id", timepoint_key, fov_key, domain_key, active_state_key]].copy()
    obs.columns = [
        "patient_id",
        CANONICAL_TIMEPOINT_KEY,
        CANONICAL_FOV_KEY,
        CANONICAL_DOMAIN_KEY,
        active_state_key,
    ]

    meta = obs.loc[:, [*LONGITUDINAL_GROUP_KEYS, CANONICAL_DOMAIN_KEY]].drop_duplicates()
    if meta.duplicated(subset=LONGITUDINAL_GROUP_KEYS).any():
        raise ContractError(
            "Each patient/timepoint/fov group must map to exactly one domain_label before observation construction"
        )

    composite_ids = _build_composite_ids(obs)
    projection = aggregate_to_state_basis(
        state_ids=obs[active_state_key].to_numpy(dtype=int),
        observation_ids=composite_ids,
        n_states=n_states,
    )

    meta = meta.assign(_composite_id=_build_composite_ids(meta)).set_index("_composite_id")
    observations: list[FovObservation] = []
    for idx, composite_id in enumerate(projection["observation_ids"].tolist()):
        if composite_id not in meta.index:
            raise ContractError(f"Missing metadata for observation group {composite_id!r}")
        row = meta.loc[str(composite_id)]
        composition = np.asarray(projection["state_composition"][idx], dtype=float)
        observation = FovObservation(
            patient_id=str(row["patient_id"]),
            timepoint=str(row[CANONICAL_TIMEPOINT_KEY]),
            fov_id=str(row[CANONICAL_FOV_KEY]),
            domain_label=str(row[CANONICAL_DOMAIN_KEY]),
            community_composition=composition,
            mass=1.0,
            mass_mode="uniform",
        )
        validate_fov_observation(observation)
        observations.append(observation)
    return tuple(observations)


def build_domain_stratified_measure(
    observations: Sequence[FovObservation],
    *,
    domain_label: str | None = None,
) -> DomainStratifiedMeasure:
    """Build one domain-stratified empirical measure from ROI/FOV observations."""
    if len(observations) == 0:
        raise ContractError("observations must contain at least one FOV observation")

    filtered = [
        observation
        for observation in observations
        if domain_label is None or observation.domain_label == domain_label
    ]
    if len(filtered) == 0:
        raise ContractError("No observations matched the requested domain_label")

    for observation in filtered:
        validate_fov_observation(observation)

    state_matrix = np.vstack(
        [np.asarray(observation.community_composition, dtype=float) for observation in filtered]
    ).astype(float, copy=False)
    burdens = np.full(len(filtered), 1.0, dtype=float)
    compositions = state_matrix.copy()
    resolved_domain = domain_label if domain_label is not None else str(filtered[0].domain_label or "undeclared")
    return DomainStratifiedMeasure(
        domain_label=resolved_domain,
        observations=tuple(filtered),
        state_matrix=state_matrix,
        burdens=burdens,
        compositions=compositions,
        mass_mode="uniform",
    )


def stack_observation_measures(measures: Sequence[DomainStratifiedMeasure]) -> dict[str, np.ndarray]:
    """Stack multiple domain-stratified measures into one batch-ready payload."""
    if len(measures) == 0:
        raise ContractError("measures must contain at least one DomainStratifiedMeasure")
    n_states = measures[0].state_matrix.shape[1]
    if any(measure.state_matrix.shape[1] != n_states for measure in measures):
        raise ContractError("All measures must share the same K-state axis")
    if any(measure.mass_mode != measures[0].mass_mode for measure in measures):
        raise ContractError("All measures must share the same mass_mode")

    return {
        "state_matrix": np.vstack([measure.state_matrix for measure in measures]).astype(
            float,
            copy=False,
        ),
        "burdens": np.concatenate([measure.burdens for measure in measures]).astype(
            float,
            copy=False,
        ),
        "compositions": np.vstack([measure.compositions for measure in measures]).astype(
            float,
            copy=False,
        ),
        "domain_labels": np.asarray(
            [measure.domain_label for measure in measures for _ in range(measure.state_matrix.shape[0])],
            dtype=object,
        ),
    }


__all__ = [
    "build_domain_stratified_measure",
    "build_fov_observations",
    "compute_fov_burden",
    "compute_fov_composition",
    "stack_observation_measures",
]
