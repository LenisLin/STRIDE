"""Public fit facades for the canonical STRIDE interface."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from ..basis.contracts import StateBasis
from ..data.longitudinal import AnnData, load_state_basis_from_adata, validate_longitudinal_adata
from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry, build_state_geometry
from ..latent.operators import PatientRelation, PatientRelationAudit, initialize_patient_relation
from ..observation import FovObservation, build_fov_observations
from ..outputs.fit_result import STRIDEFitResult
from ..workflows.fit_stride import STRIDEFitConfig, run_stride_fit
from .basis import BasisSpec
from .dataset import DatasetHandle
from .model import STRIDEModel


def build_patient_relation(
    *,
    patient_id: str,
    A: object,
    d: object,
    e: object,
    mu_minus: object | None = None,
    mu_plus: object | None = None,
    state_ids: tuple[int, ...] | None = None,
    audit: PatientRelationAudit | None = None,
    relation_status: str = "explicit_patient_relation",
    metadata: Mapping[str, Any] | None = None,
) -> PatientRelation:
    """Assemble one validated patient relation from explicit shared-axis arrays."""
    metadata_dict = dict(metadata or {})
    resolved_audit = audit
    if resolved_audit is None:
        resolved_audit = PatientRelationAudit(
            patient_id=str(patient_id),
            bridge_status=str(relation_status),
            metadata=metadata_dict,
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
        metadata=metadata_dict,
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


def fit_stride(
    data: object,
    *,
    state_basis: StateBasis | None = None,
    geometry: StateGeometry | None = None,
    basis_spec: BasisSpec | None = None,
    model: STRIDEModel | None = None,
    config: STRIDEFitConfig | None = None,
) -> STRIDEFitResult:
    """Normalize canonical STRIDE inputs and run the canonical full-method path."""
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
    "STRIDEFitConfig",
    "build_patient_relation",
    "fit_stride",
]
