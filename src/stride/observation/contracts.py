"""Observation-layer contracts for ROI/FOV units, measures, and discrepancies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..errors import ContractError
from ..settings.runtime import RuntimeSettings


@dataclass(frozen=True)
class FovObservation:
    """Canonical ROI/FOV observation on the shared community-state axis.

    Each object represents one ROI/FOV with normalized community composition.
    Tissue or compartment labels remain observation metadata (`domain_label`);
    they do not define the shared state identity or geometry.
    """

    patient_id: str
    timepoint: str
    fov_id: str
    community_composition: np.ndarray
    mass: float = 1.0
    mass_mode: str = "uniform"
    domain_label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state_mass(self) -> np.ndarray:
        """Compatibility alias for downstream code that still reads `state_mass`."""
        return self.community_composition

    @property
    def observation_id(self) -> str:
        """Compatibility alias for the canonical observation identifier."""
        return self.fov_id

    @property
    def burden(self) -> float:
        """Compatibility alias for total observation mass."""
        return self.mass

    @property
    def weight(self) -> float:
        """Compatibility alias for observation weight under the uniform-mass contract."""
        return self.mass


@dataclass(frozen=True)
class DomainStratifiedMeasure:
    """Observation-layer empirical measure for one declared domain stratum."""

    domain_label: str
    observations: tuple[FovObservation, ...]
    state_matrix: np.ndarray
    burdens: np.ndarray
    compositions: np.ndarray
    mass_mode: str


@dataclass(frozen=True)
class ObservationDiscrepancyConfig:
    """Canonical observation-layer discrepancy configuration."""

    eps_schedule: tuple[float, ...]
    max_iter: int = 2000
    tol: float = 1e-6
    eta_floor: float = 1e-12
    n_min_proto: float = 0.0
    tau_q: float = 0.25
    tau_mode: str = "pi_weighted_q25"
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_settings, RuntimeSettings):
            raise ContractError(
                "ObservationDiscrepancyConfig.runtime_settings must be a RuntimeSettings object"
            )


@dataclass(frozen=True)
class ObservationDiscrepancyResult:
    """Canonical result container for batched observation-layer discrepancies."""

    metrics: dict[str, np.ndarray]
    details: dict[str, np.ndarray]
    status: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok_mask(self) -> np.ndarray:
        """Boolean mask for rows whose observation fit succeeded."""
        return np.asarray(self.status == "ok", dtype=bool)

    @property
    def matched_mass(self) -> np.ndarray:
        """Canonical access to the matched observation-layer mass."""
        return self.metrics["T"]

    @property
    def source_unmatched_mass(self) -> np.ndarray:
        """Canonical access to pre-side unmatched observation mass."""
        return self.metrics["D_pos"]

    @property
    def target_unmatched_mass(self) -> np.ndarray:
        """Canonical access to post-side unmatched observation mass."""
        return self.metrics["B_pos"]

    @property
    def retention_fraction(self) -> np.ndarray:
        """Canonical access to thresholded retention diagnostics."""
        return self.metrics["R"]

    @property
    def retention_cost_threshold(self) -> np.ndarray:
        """Canonical access to the threshold used for `retention_fraction`."""
        return self.metrics["tau"]

    @property
    def matching_plan(self) -> np.ndarray | None:
        """Return the dense observation-layer matching plan when requested."""
        if "matching_plan" in self.details:
            return self.details["matching_plan"]
        if "Pi" in self.details:
            return self.details["Pi"]
        if "plan" in self.details:
            return self.details["plan"]
        return None


@dataclass(frozen=True)
class ObservationDiscrepancy:
    """Named observation-layer discrepancy between pre/post domain-stratified measures."""

    pre_domain: str
    post_domain: str
    result: ObservationDiscrepancyResult
    metadata: Mapping[str, Any] = field(default_factory=dict)


def validate_fov_observation(observation: FovObservation) -> None:
    """Validate the structural contract of one ROI/FOV observation."""
    composition = np.asarray(observation.community_composition, dtype=float).reshape(-1)
    if composition.ndim != 1 or composition.size == 0:
        raise ContractError("FovObservation.community_composition must be a non-empty 1D array")
    if not np.isfinite(composition).all():
        raise ContractError("FovObservation.community_composition contains NaN/Inf")
    if (composition < 0).any():
        raise ContractError("FovObservation.community_composition must be non-negative")
    if not np.isclose(float(np.sum(composition, dtype=float)), 1.0):
        raise ContractError("FovObservation.community_composition must sum to 1.0")
    if observation.mass_mode != "uniform":
        raise ContractError("FovObservation.mass_mode must be 'uniform'")
    if not np.isfinite(float(observation.mass)) or not np.isclose(float(observation.mass), 1.0):
        raise ContractError("FovObservation.mass must be finite and equal to 1.0")
    if observation.domain_label is not None and str(observation.domain_label).strip() == "":
        raise ContractError("FovObservation.domain_label must be non-empty when provided")


__all__ = [
    "DomainStratifiedMeasure",
    "FovObservation",
    "ObservationDiscrepancy",
    "ObservationDiscrepancyConfig",
    "ObservationDiscrepancyResult",
    "validate_fov_observation",
]
