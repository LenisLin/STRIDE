"""Task-local adapter between Task A Stage 0 artifacts and stable STRIDE APIs.

This module builds the frozen field crosswalk, family-sliced observations, and
dry-run summaries needed by Task A. It does not push task-specific semantics
into `src/stride/`.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stride import BasisSpec, DatasetHandle
from stride.api.fit import STRIDEFitConfig, fit_stride
from stride.basis import load_state_basis
from stride.errors import ContractError
from stride.observation import FovObservation
from stride.outputs.fit_result import STRIDEFitResult

from ..config import TaskAConfigBundle, TaskAOrderedPairFamilySpec
from ..contracts import (
    TaskACoreFitDryRunRecord,
    TaskAFamilyStrideMappingSummary,
    TaskARealDataCrosswalk,
    TaskAStage0FieldMapping,
    TaskAStage0StrideMappingSummary,
)

try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


SEMANTIC_ALIGNMENT_ERROR_PREFIX = "Task A semantic alignment failed:"


def load_task_a_dataset_handle(stage0_h5ad_or_adata: str | Path | Any) -> DatasetHandle:
    if isinstance(stage0_h5ad_or_adata, DatasetHandle):
        return stage0_h5ad_or_adata
    if hasattr(stage0_h5ad_or_adata, "obs") and hasattr(stage0_h5ad_or_adata, "obsm"):
        return DatasetHandle(adata=stage0_h5ad_or_adata)
    if ad is None:
        raise ModuleNotFoundError("anndata is required to load Task-A stage0 h5ad artifacts")
    path = Path(stage0_h5ad_or_adata).expanduser().resolve()
    return DatasetHandle(adata=ad.read_h5ad(path))


def resolve_task_a_state_basis(
    handle: DatasetHandle,
    *,
    basis_spec: BasisSpec | None = None,
) -> Any:
    try:
        return handle.load_state_basis()
    except ContractError:
        adata = handle.adata
        centroids_key = None
        for candidate in ("state_centroids", "prototype_centroids"):
            if candidate in adata.uns:
                centroids_key = candidate
                break
        cost_scale_key = None
        for candidate in ("cost_scale", "s_C", "global_cost_scale"):
            if candidate in adata.uns:
                cost_scale_key = candidate
                break
        if centroids_key is not None and "cost_matrix" in adata.uns and cost_scale_key is not None:
            cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
            return load_state_basis(
                centroids=np.asarray(adata.uns[centroids_key], dtype=float),
                cost_matrix=cost_matrix,
                cost_scale=float(adata.uns[cost_scale_key]),
                feature_key=handle.feature_key,
                state_key=handle.state_id_key,
                state_ids=tuple(range(cost_matrix.shape[0])),
            )
        if basis_spec is None:
            raise
        return basis_spec.fit(handle.adata)


def build_task_a_stage0_field_mapping(
    handle: DatasetHandle,
    *,
    state_basis: Any,
) -> TaskAStage0FieldMapping:
    return TaskAStage0FieldMapping(
        patient_id_key="patient_id",
        timepoint_key=handle.timepoint_key,
        fov_key=handle.fov_key,
        domain_key=handle.domain_key,
        cell_subtype_key=handle.cell_subtype_key,
        state_id_key=handle.state_id_key,
        state_ids=tuple(int(state_id) for state_id in state_basis.resolved_state_ids),
        n_states=int(state_basis.n_states),
    )


def _state_index_lookup(state_ids: Sequence[int]) -> dict[int, int]:
    return {int(state_id): idx for idx, state_id in enumerate(state_ids)}


def _require_task_a_uniform_mass_mode(config_bundle: TaskAConfigBundle) -> str:
    mass_mode = str(config_bundle.data.mass_mode)
    if mass_mode != "uniform":
        raise ContractError(
            "Task A Step 1 requires data.mass_mode='uniform'; "
            f"got {mass_mode!r}"
        )
    return mass_mode


def build_task_a_family_observations(
    handle: DatasetHandle,
    family_spec: TaskAOrderedPairFamilySpec,
    *,
    state_basis: Any,
    mass_mode: str = "uniform",
    require_complete_patients: bool = True,
) -> tuple[FovObservation, ...]:
    if mass_mode != "uniform":
        raise ContractError(
            "Task A observation construction requires mass_mode='uniform'; "
            f"got {mass_mode!r}"
        )
    adata = handle.adata
    state_ids = tuple(int(state_id) for state_id in state_basis.resolved_state_ids)
    state_lookup = _state_index_lookup(state_ids)
    obs = adata.obs.loc[
        :,
        [
            "patient_id",
            handle.fov_key,
            handle.domain_key,
            handle.state_id_key,
        ],
    ].copy()
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs[handle.fov_key] = obs[handle.fov_key].astype(str)
    obs[handle.domain_key] = obs[handle.domain_key].astype(str)
    obs[handle.state_id_key] = obs[handle.state_id_key].astype(int)

    allowed_domains = {family_spec.source_domain, family_spec.target_domain}
    obs = obs.loc[obs[handle.domain_key].isin(allowed_domains)].copy()
    if obs.empty:
        return ()

    if require_complete_patients:
        patient_domains = (
            obs.groupby("patient_id", sort=True, observed=False)[handle.domain_key]
            .agg(lambda values: {str(value) for value in values})
        )
        eligible_patients = {
            patient_id
            for patient_id, domains in patient_domains.items()
            if allowed_domains.issubset(domains)
        }
        obs = obs.loc[obs["patient_id"].isin(eligible_patients)].copy()
        if obs.empty:
            return ()

    observations: list[FovObservation] = []
    grouped = obs.groupby(["patient_id", handle.domain_key, handle.fov_key], sort=True, observed=False)
    for (patient_id, domain_label, fov_id), group in grouped:
        counts = np.zeros(len(state_ids), dtype=float)
        for raw_state_id in group[handle.state_id_key].astype(int):
            if int(raw_state_id) not in state_lookup:
                raise ValueError(
                    f"Task-A Stage 0 state_id {raw_state_id} was not found in the resolved STRIDE state basis"
                )
            counts[state_lookup[int(raw_state_id)]] += 1.0
        total = float(np.sum(counts, dtype=float))
        if total <= 0.0:
            continue
        observations.append(
            FovObservation(
                patient_id=str(patient_id),
                timepoint=str(domain_label),
                fov_id=str(fov_id),
                domain_label=str(domain_label),
                community_composition=counts / total,
                mass=1.0,
                mass_mode=mass_mode,
                metadata={
                    "pair_family": family_spec.name,
                    "claim_role": family_spec.claim_role,
                },
            )
        )

    return tuple(observations)


def summarize_task_a_family_mapping(
    handle: DatasetHandle,
    family_spec: TaskAOrderedPairFamilySpec,
    *,
    state_basis: Any,
    mass_mode: str = "uniform",
) -> TaskAFamilyStrideMappingSummary:
    adata = handle.adata
    obs = adata.obs.loc[:, ["patient_id", handle.domain_key]].copy()
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs[handle.domain_key] = obs[handle.domain_key].astype(str)
    allowed_domains = {family_spec.source_domain, family_spec.target_domain}
    family_obs = obs.loc[obs[handle.domain_key].isin(allowed_domains)].copy()
    patients = tuple(sorted(family_obs["patient_id"].unique().tolist()))
    patient_domains = (
        family_obs.groupby("patient_id", sort=True, observed=False)[handle.domain_key]
        .agg(lambda values: {str(value) for value in values})
    )
    eligible_patients = tuple(
        sorted(
            patient_id
            for patient_id, domains in patient_domains.items()
            if allowed_domains.issubset(domains)
        )
    )
    skipped_patients = tuple(sorted(set(patients) - set(eligible_patients)))
    observations = build_task_a_family_observations(
        handle,
        family_spec,
        state_basis=state_basis,
        mass_mode=mass_mode,
        require_complete_patients=True,
    )
    n_source = sum(1 for observation in observations if observation.timepoint == family_spec.source_domain)
    n_target = sum(1 for observation in observations if observation.timepoint == family_spec.target_domain)
    return TaskAFamilyStrideMappingSummary(
        pair_family=family_spec.name,
        source_domain=family_spec.source_domain,
        target_domain=family_spec.target_domain,
        claim_role=family_spec.claim_role,
        pair_types=family_spec.pair_types,
        ordered_group_labels=family_spec.ordered_group_labels,
        eligible_patients=eligible_patients,
        skipped_patients=skipped_patients,
        n_observations=len(observations),
        n_source_observations=n_source,
        n_target_observations=n_target,
    )


def _assert_timepoint_inert(handle: DatasetHandle) -> tuple[str, ...]:
    """Check that the raw timepoint column carries only inert metadata values.

    Task A derives its ordered groups from the compartment/domain field, not
    from raw ``timepoint``.  If the real data suddenly has multiple distinct
    raw timepoint values this assumption no longer holds and should fail
    loudly rather than silently mixing semantics.

    Returns the tuple of observed raw timepoint values.
    """
    tp_key = handle.timepoint_key
    observed = tuple(sorted(handle.adata.obs[tp_key].astype(str).unique().tolist()))
    if len(observed) > 1:
        raise ContractError(
            f"{SEMANTIC_ALIGNMENT_ERROR_PREFIX} Task A expects raw timepoint to be inert single-value metadata, "
            f"but observed {len(observed)} distinct values under "
            f"'{tp_key}': {observed}.  The ordered-group derivation uses "
            f"compartment/domain, not raw timepoint.  If the data model "
            f"changed, the adapter must be updated."
        )
    return observed


def build_task_a_real_data_crosswalk(
    handle: DatasetHandle,
    *,
    mass_mode: str = "uniform",
) -> TaskARealDataCrosswalk:
    """Build a crosswalk recording the exact real-data field alias resolution."""
    if mass_mode != "uniform":
        raise ContractError(
            "Task A real-data crosswalk requires mass_mode='uniform'; "
            f"got {mass_mode!r}"
        )
    observed_tp = _assert_timepoint_inert(handle)
    return TaskARealDataCrosswalk(
        patient_id_raw="patient_id",
        patient_id_canonical="patient_id",
        patient_id_mapping="direct",
        fov_raw=handle.fov_key,
        fov_canonical="fov_id",
        fov_mapping="direct" if handle.fov_key == "fov_id" else "alias",
        domain_raw=handle.domain_key,
        domain_canonical="domain_label",
        domain_mapping="direct" if handle.domain_key == "domain_label" else "alias",
        timepoint_raw=handle.timepoint_key,
        timepoint_raw_observed_values=observed_tp,
        timepoint_inert=True,
        cell_subtype_raw=handle.cell_subtype_key,
        cell_subtype_canonical="cell_subtype_label",
        cell_subtype_mapping="direct" if handle.cell_subtype_key == "cell_subtype_label" else "alias",
        state_id_raw=handle.state_id_key,
        state_id_canonical="state_id",
        state_id_mapping="direct" if handle.state_id_key == "state_id" else "alias",
        feature_raw=handle.feature_key,
        feature_canonical="local_state_features",
        feature_mapping="direct" if handle.feature_key == "local_state_features" else "alias",
        centroids_raw="prototype_centroids" if "prototype_centroids" in handle.adata.uns else "state_centroids",
        centroids_canonical="state_centroids",
        centroids_mapping="direct" if "state_centroids" in handle.adata.uns else "alias",
        cost_scale_raw=next(
            (k for k in ("cost_scale", "s_C", "global_cost_scale") if k in handle.adata.uns),
            "cost_scale",
        ),
        cost_scale_canonical="cost_scale",
        cost_scale_mapping="direct" if "cost_scale" in handle.adata.uns else "alias",
        mass_mode=mass_mode,
        mass_note="Observation-layer mass is uniform at adapter time and matches data.mass_mode",
    )


def describe_task_a_stage0_stride_mapping(
    stage0_h5ad_or_adata: str | Path | Any,
    *,
    config_bundle: TaskAConfigBundle,
    basis_spec: BasisSpec | None = None,
) -> TaskAStage0StrideMappingSummary:
    handle = load_task_a_dataset_handle(stage0_h5ad_or_adata)
    mass_mode = _require_task_a_uniform_mass_mode(config_bundle)
    handle.validate(
        require_cell_type=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    resolved_basis = resolve_task_a_state_basis(handle, basis_spec=basis_spec)
    field_mapping = build_task_a_stage0_field_mapping(handle, state_basis=resolved_basis)
    crosswalk = build_task_a_real_data_crosswalk(handle, mass_mode=mass_mode)
    family_summaries = tuple(
        summarize_task_a_family_mapping(
            handle,
            family_spec,
            state_basis=resolved_basis,
            mass_mode=mass_mode,
        )
        for family_spec in config_bundle.ordered_proxy.pair_families
    )
    patient_ids = tuple(sorted(handle.adata.obs["patient_id"].astype(str).unique().tolist()))
    return TaskAStage0StrideMappingSummary(
        field_mapping=field_mapping,
        patient_ids=patient_ids,
        family_summaries=family_summaries,
        real_data_crosswalk=crosswalk,
    )


def run_task_a_family_core_fit_dry_run(
    stage0_h5ad_or_adata: str | Path | Any,
    *,
    config_bundle: TaskAConfigBundle,
    pair_families: Iterable[TaskAOrderedPairFamilySpec] | None = None,
    fit_metadata: dict[str, Any] | None = None,
    basis_spec: BasisSpec | None = None,
) -> tuple[pd.DataFrame, dict[str, STRIDEFitResult]]:
    handle = load_task_a_dataset_handle(stage0_h5ad_or_adata)
    mass_mode = _require_task_a_uniform_mass_mode(config_bundle)
    handle.validate(
        require_cell_type=True,
        require_state_axis=True,
        require_cost_scale=True,
        require_cost_matrix=True,
    )
    resolved_basis = resolve_task_a_state_basis(handle, basis_spec=basis_spec)
    active_families = tuple(pair_families or config_bundle.ordered_proxy.pair_families)

    records: list[TaskACoreFitDryRunRecord] = []
    results: dict[str, STRIDEFitResult] = {}
    for family_spec in active_families:
        observations = build_task_a_family_observations(
            handle,
            family_spec,
            state_basis=resolved_basis,
            mass_mode=mass_mode,
            require_complete_patients=True,
        )
        if not observations:
            continue
        metadata = dict(fit_metadata or {})
        metadata.update(
            {
                "task_pair_family": family_spec.name,
                "task_claim_role": family_spec.claim_role,
                "task_source_domain": family_spec.source_domain,
                "task_target_domain": family_spec.target_domain,
            }
        )
        result = fit_stride(
            observations,
            state_basis=resolved_basis,
            config=STRIDEFitConfig(
                timepoint_order=family_spec.ordered_group_labels,
                metadata=metadata,
            ),
        )
        recurrence_used_patient_ids = tuple(
            str(patient_id)
            for patient_id in (
                result.recurrence.used_patient_ids
                if result.recurrence.used_patient_ids
                else result.recurrence.patient_ids
            )
        )
        results[family_spec.name] = result
        for patient_result in result.patient_results:
            uncertainty_status = None
            if result.uncertainty is not None:
                for uncertainty_result in result.uncertainty.patient_results:
                    if uncertainty_result.patient_id == patient_result.patient_id:
                        uncertainty_status = str(uncertainty_result.uncertainty_status)
                        break
            defer_reason = None
            if "defer_reason" in patient_result.diagnostics:
                defer_reason = str(patient_result.diagnostics["defer_reason"])
            records.append(
                TaskACoreFitDryRunRecord(
                    pair_family=family_spec.name,
                    claim_role=family_spec.claim_role,
                    patient_id=str(patient_result.patient_id),
                    implementation_tier=str(result.implementation_tier),
                    fit_surface="fit_stride",
                    fit_status=str(patient_result.fit_status),
                    bridge_realized=bool(patient_result.is_ok),
                    defer_reason=defer_reason,
                    uncertainty_status=uncertainty_status,
                    cohort_recurrence_fit_status=str(result.recurrence.fit_status),
                    n_recurrence_families=int(len(result.recurrence.families)),
                    n_recurrence_used_patients=int(len(recurrence_used_patient_ids)),
                    source_domain=family_spec.source_domain,
                    target_domain=family_spec.target_domain,
                )
            )

    frame = pd.DataFrame.from_records([record.to_json_dict() for record in records])
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "pair_family",
                "claim_role",
                "patient_id",
                "implementation_tier",
                "fit_surface",
                "fit_status",
                "bridge_realized",
                "defer_reason",
                "uncertainty_status",
                "cohort_recurrence_fit_status",
                "n_recurrence_families",
                "n_recurrence_used_patients",
                "source_domain",
                "target_domain",
            ]
        )
    return frame, results


__all__ = [
    "build_task_a_family_observations",
    "build_task_a_real_data_crosswalk",
    "build_task_a_stage0_field_mapping",
    "describe_task_a_stage0_stride_mapping",
    "load_task_a_dataset_handle",
    "resolve_task_a_state_basis",
    "run_task_a_family_core_fit_dry_run",
    "SEMANTIC_ALIGNMENT_ERROR_PREFIX",
    "summarize_task_a_family_mapping",
]
