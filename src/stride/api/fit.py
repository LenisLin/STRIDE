"""Public first-pass fit facades for explicit patient relations.

`fit_stride(...)` and `build_patient_relation(...)` define the conservative
stable tier. Compatibility helpers and explicitly deferred estimator stubs stay
available from this module but are not package-root exports.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..basis.contracts import StateBasis
from ..data.longitudinal import AnnData, load_state_basis_from_adata, validate_longitudinal_adata
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry, build_state_geometry
from ..latent.operators import PatientRelation, PatientRelationAudit, initialize_patient_relation
from ..observation import FovObservation, build_fov_observations
from ..outputs.fit_result import PatientBridgeResult, PatientRelationFitResult, STRIDEFitResult
from ..workflows.fit_stride import STRIDEFitConfig, run_stride_fit
from .basis import BasisSpec
from .dataset import DatasetHandle
from .model import STRIDEModel


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for packaging explicit patient-level bridge artifacts."""

    bridge_mode: str = "explicit_patient_relation"
    preserve_emergence: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRelationFitConfig:
    """Compatibility configuration for the single-patient fit wrapper."""

    mode: str = "deferred"
    preserve_emergence: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_patient_relation(
    *,
    patient_id: str,
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    mu_minus: np.ndarray | None = None,
    mu_plus: np.ndarray | None = None,
    state_ids: tuple[int, ...] | None = None,
    audit: PatientRelationAudit | None = None,
    config: BridgeConfig | None = None,
) -> PatientRelation:
    """Assemble one validated patient relation from explicit shared-axis arrays."""
    resolved_config = config or BridgeConfig()
    resolved_audit = audit
    if resolved_audit is None:
        resolved_audit = PatientRelationAudit(
            patient_id=str(patient_id),
            bridge_status=resolved_config.bridge_mode,
            metadata=dict(resolved_config.metadata),
        )

    return initialize_patient_relation(
        patient_id=str(patient_id),
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        state_ids=state_ids,
        audit=resolved_audit,
        metadata=dict(resolved_config.metadata),
    )


def _coerce_observation_sequence(data: object) -> tuple[FovObservation, ...] | None:
    if isinstance(data, (str, bytes, bytearray)):
        return None
    if not isinstance(data, Sequence):
        return None
    observations = tuple(data)
    if all(isinstance(observation, FovObservation) for observation in observations):
        return observations
    return None


def _resolve_geometry(
    state_basis: StateBasis,
    *,
    geometry: StateGeometry | None,
) -> StateGeometry:
    if geometry is not None:
        return geometry
    return build_state_geometry(
        cost_matrix=state_basis.cost_matrix,
        cost_scale=state_basis.cost_scale,
        state_ids=state_basis.resolved_state_ids,
    )


def _normalize_observation_inputs(
    observations: tuple[FovObservation, ...],
    *,
    state_basis: StateBasis | None,
    geometry: StateGeometry | None,
    model: STRIDEModel | None,
) -> tuple[tuple[FovObservation, ...], StateBasis, StateGeometry]:
    resolved_basis = state_basis or (model.state_basis if model is not None else None)
    if resolved_basis is None:
        raise ContractError(
            "Direct FovObservation input requires an explicit state_basis or model.state_basis"
        )
    resolved_geometry = _resolve_geometry(
        resolved_basis,
        geometry=geometry or (model.geometry if model is not None else None),
    )
    return observations, resolved_basis, resolved_geometry


def _normalize_dataset_inputs(
    adata: AnnData,
    *,
    state_basis: StateBasis | None,
    geometry: StateGeometry | None,
    basis_spec: BasisSpec | None,
    model: STRIDEModel | None,
) -> tuple[tuple[FovObservation, ...], StateBasis, StateGeometry]:
    validate_longitudinal_adata(adata)

    resolved_basis: StateBasis
    try:
        attached_basis = load_state_basis_from_adata(adata)
    except ContractError as exc:
        if basis_spec is None:
            raise ContractError(
                "DatasetHandle/AnnData input requires an attached state axis or basis_spec "
                "to materialize the deterministic upstream path"
            ) from exc
        resolved_basis = basis_spec.fit(adata)
    else:
        if state_basis is not None and state_basis.n_states != attached_basis.n_states:
            raise ContractError(
                "Explicit state_basis does not align with the dataset-attached shared state axis"
            )
        if model is not None and model.state_basis is not None and model.state_basis.n_states != attached_basis.n_states:
            raise ContractError(
                "model.state_basis does not align with the dataset-attached shared state axis"
            )
        resolved_basis = attached_basis

    resolved_geometry = _resolve_geometry(
        resolved_basis,
        geometry=geometry or (model.geometry if model is not None else None),
    )
    observations = build_fov_observations(
        adata,
        state_key=resolved_basis.state_key,
        n_states=resolved_basis.n_states,
    )
    return observations, resolved_basis, resolved_geometry


def fit_patient_relation(
    observations: object,
    *,
    state_basis: object,
    geometry: object | None = None,
    schedule: object | None = None,
    config: PatientRelationFitConfig | None = None,
) -> PatientRelationFitResult:
    """Return the canonical single-patient bridge result from the deferred fit flow."""
    resolved_config = config or PatientRelationFitConfig()
    del schedule

    result = fit_stride(
        observations,
        state_basis=state_basis,  # type: ignore[arg-type]
        geometry=geometry,  # type: ignore[arg-type]
        config=STRIDEFitConfig(
            metadata={
                "compat_mode": resolved_config.mode,
                "preserve_emergence": resolved_config.preserve_emergence,
                **dict(resolved_config.metadata),
            }
        ),
    )
    if len(result.patient_results) != 1:
        raise ContractError(
            "fit_patient_relation expects input that resolves to exactly one patient"
        )

    patient_result = result.patient_results[0]
    return PatientBridgeResult(
        patient_id=patient_result.patient_id,
        fit_status=patient_result.fit_status,
        A=patient_result.A,
        d=patient_result.d,
        e=patient_result.e,
        mu_minus=patient_result.mu_minus,
        mu_plus=patient_result.mu_plus,
        state_ids=patient_result.state_ids,
        audit=patient_result.audit,
        diagnostics={
            **dict(patient_result.diagnostics),
            "compat_mode": resolved_config.mode,
            "preserve_emergence": resolved_config.preserve_emergence,
        },
        auxiliary=dict(patient_result.auxiliary),
    )


def bridge_observation_matches(*args: object, **kwargs: object) -> PatientRelation:
    """Reserved bridge estimator entrypoint kept honest by an explicit defer."""
    raise NotImplementedError(
        "Direct bridge_observation_matches(...) remains deferred as a standalone API. "
        "Use fit_stride(...) for the current narrow canonical observation-to-patient "
        "bridge path, or build_patient_relation(...) when A/d/e are already known."
    )


def fit_stride(
    data: object,
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    basis_spec: BasisSpec | None = None,
    model: STRIDEModel | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Normalize canonical STRIDE inputs and run the current narrow STRIDE fit path."""
    resolved_config = config or STRIDEFitConfig()

    observations = _coerce_observation_sequence(data)
    if observations is not None:
        normalized_observations, resolved_basis, resolved_geometry = _normalize_observation_inputs(
            observations,
            state_basis=state_basis,
            geometry=geometry,
            model=model,
        )
        return run_stride_fit(
            normalized_observations,
            state_basis=resolved_basis,
            geometry=resolved_geometry,
            config=resolved_config,
        )

    if isinstance(data, DatasetHandle):
        return fit_stride(
            data.adata,
            state_basis=state_basis,
            geometry=geometry,
            basis_spec=basis_spec,
            model=model,
            config=resolved_config,
        )

    if isinstance(data, AnnData):
        normalized_observations, resolved_basis, resolved_geometry = _normalize_dataset_inputs(
            data,
            state_basis=state_basis,
            geometry=geometry,
            basis_spec=basis_spec,
            model=model,
        )
        return run_stride_fit(
            normalized_observations,
            state_basis=resolved_basis,
            geometry=resolved_geometry,
            config=resolved_config,
        )

    raise ContractError(
        "fit_stride expects DatasetHandle, AnnData, or a sequence of FovObservation objects"
    )


__all__ = [
    "BridgeConfig",
    "STRIDEFitConfig",
    "build_patient_relation",
    "fit_stride",
]
