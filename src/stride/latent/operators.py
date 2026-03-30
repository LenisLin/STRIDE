"""Model-layer patient relation contracts on the shared STRIDE state basis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..errors import ContractError


@dataclass(frozen=True)
class ContinuityOperator:
    """Patient-level continuity operator on the shared state basis."""

    weights: np.ndarray
    state_ids: tuple[int, ...] | None = None

    @property
    def n_states(self) -> int:
        """Return the shared state-axis size."""
        return int(np.asarray(self.weights).shape[0])


@dataclass(frozen=True)
class DepletionComponent:
    """Source-side unmatched or depleted mass on the shared state basis."""

    weights: np.ndarray
    state_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EmergenceComponent:
    """Target-side unmatched or emergent mass on the shared state basis."""

    weights: np.ndarray
    state_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PatientRelationAudit:
    """Provenance and workflow audit payload for one patient relation.

    This record stores how a relation was assembled or fit. It is descriptive
    metadata for the model layer, not a scientific summary of the relation.
    """

    patient_id: str
    timepoint_order: tuple[str, ...] = ()
    mass_mode: str = "unknown"
    n_pre_observations: int | None = None
    n_post_observations: int | None = None
    observation_fit_status: str | None = None
    bridge_status: str = "assembled"
    uncertainty_mode: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRelation:
    """Canonical patient-level remodeling relation on the shared state basis.

    A ``PatientRelation`` packages the continuity operator ``A`` together with
    depletion ``d`` and emergence ``e`` vectors, plus optional source/target
    marginals and audit metadata describing how the relation was produced.
    """

    patient_id: str
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    mu_minus: np.ndarray | None = None
    mu_plus: np.ndarray | None = None
    state_ids: tuple[int, ...] | None = None
    audit: PatientRelationAudit | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        """Return the shared-axis size `K`."""
        return int(np.asarray(self.A).shape[0])

    @property
    def continuity(self) -> ContinuityOperator:
        """Return the continuity/remodeling operator as a typed view."""
        return ContinuityOperator(weights=self.A, state_ids=self.state_ids)

    @property
    def depletion(self) -> DepletionComponent:
        """Return the depletion component as a typed view."""
        return DepletionComponent(weights=self.d, state_ids=self.state_ids)

    @property
    def emergence(self) -> EmergenceComponent:
        """Return the emergence component as a typed view."""
        return EmergenceComponent(weights=self.e, state_ids=self.state_ids)


def validate_patient_relation(
    *,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    mu_minus: np.ndarray | None = None,
    mu_plus: np.ndarray | None = None,
    state_ids: tuple[int, ...] | None = None,
    row_tolerance: float = 1e-8,
) -> None:
    """Validate one patient-level relation payload on a declared shared state axis.

    The validator checks only structural invariants such as non-negativity,
    square/shared-axis alignment, and the row-substochastic continuity plus
    depletion constraint. It does not normalize or infer missing values.
    """
    A_arr = np.asarray(A, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    e_arr = np.asarray(e, dtype=float)

    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ContractError("A must be a square [K, K] array")
    if d_arr.ndim != 1:
        raise ContractError("d must be a 1D [K] array")
    if e_arr.ndim != 1:
        raise ContractError("e must be a 1D [K] array")
    if d_arr.shape != (A_arr.shape[0],):
        raise ContractError(f"d must have shape {(A_arr.shape[0],)}, got {d_arr.shape}")
    if e_arr.shape != (A_arr.shape[0],):
        raise ContractError(f"e must have shape {(A_arr.shape[0],)}, got {e_arr.shape}")

    for name, array in (("A", A_arr), ("d", d_arr), ("e", e_arr)):
        if not np.isfinite(array).all():
            raise ContractError(f"{name} contains NaN/Inf")
        if (array < 0).any():
            raise ContractError(f"{name} must be non-negative")

    row_mass = np.sum(A_arr, axis=1, dtype=float) + d_arr
    if not np.allclose(row_mass, 1.0, rtol=0.0, atol=float(row_tolerance)):
        raise ContractError("A and d must satisfy the canonical row-substochastic continuity/depletion equality")

    if mu_minus is not None:
        mu_minus_arr = np.asarray(mu_minus, dtype=float)
        if mu_minus_arr.ndim != 1:
            raise ContractError("mu_minus must be a 1D [K] array")
        if mu_minus_arr.shape != (A_arr.shape[0],):
            raise ContractError(f"mu_minus must have shape {(A_arr.shape[0],)}, got {mu_minus_arr.shape}")
        if not np.isfinite(mu_minus_arr).all() or (mu_minus_arr < 0).any():
            raise ContractError("mu_minus must be finite and non-negative")

    if mu_plus is not None:
        mu_plus_arr = np.asarray(mu_plus, dtype=float)
        if mu_plus_arr.ndim != 1:
            raise ContractError("mu_plus must be a 1D [K] array")
        if mu_plus_arr.shape != (A_arr.shape[0],):
            raise ContractError(f"mu_plus must have shape {(A_arr.shape[0],)}, got {mu_plus_arr.shape}")
        if not np.isfinite(mu_plus_arr).all() or (mu_plus_arr < 0).any():
            raise ContractError("mu_plus must be finite and non-negative")

    if state_ids is not None:
        if len(state_ids) != A_arr.shape[0]:
            raise ContractError("state_ids must align to the shared K-state axis")
        if len(set(state_ids)) != len(state_ids):
            raise ContractError("state_ids must be unique along the shared K-state axis")


def initialize_patient_relation(
    *,
    patient_id: str,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    mu_minus: np.ndarray | None = None,
    mu_plus: np.ndarray | None = None,
    state_ids: tuple[int, ...] | None = None,
    audit: PatientRelationAudit | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PatientRelation:
    """Build a validated patient relation from explicit shared-axis arrays.

    When ``mu_minus`` or ``mu_plus`` are omitted, they are derived from
    ``A``, ``d``, and ``e`` so the returned object always carries explicit
    marginals.
    """
    A_arr = np.asarray(A, dtype=float)
    d_arr = np.asarray(d, dtype=float).reshape(-1)
    e_arr = np.asarray(e, dtype=float).reshape(-1)
    resolved_mu_minus = (
        np.asarray(mu_minus, dtype=float).reshape(-1)
        if mu_minus is not None
        else np.sum(A_arr, axis=1, dtype=float) + d_arr
    )
    resolved_mu_plus = (
        np.asarray(mu_plus, dtype=float).reshape(-1)
        if mu_plus is not None
        else np.sum(A_arr, axis=0, dtype=float) + e_arr
    )
    validate_patient_relation(
        A=A_arr,
        d=d_arr,
        e=e_arr,
        mu_minus=resolved_mu_minus,
        mu_plus=resolved_mu_plus,
        state_ids=state_ids,
    )
    return PatientRelation(
        patient_id=str(patient_id),
        A=A_arr,
        d=d_arr,
        e=e_arr,
        mu_minus=resolved_mu_minus,
        mu_plus=resolved_mu_plus,
        state_ids=state_ids,
        audit=audit,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "ContinuityOperator",
    "DepletionComponent",
    "EmergenceComponent",
    "PatientRelation",
    "PatientRelationAudit",
    "initialize_patient_relation",
    "validate_patient_relation",
]
