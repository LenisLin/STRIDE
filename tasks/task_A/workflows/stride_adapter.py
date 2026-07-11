"""Task-local adapter between Task A Stage 0 artifacts and stable STRIDE APIs.

This module builds the frozen field crosswalk, family-sliced observations, and
dry-run summaries needed by Task A. It does not push task-specific semantics
into `src/stride/`.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_CELL_TYPE_KEY,
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_STATE_ID_KEY,
    OBS_TIMEPOINT_KEY,
    OBSM_LOCAL_STATE_FEATURES_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_RELATION_IDS_KEY,
    STRIDE_RELATIONS_KEY,
    STRIDE_UNS_KEY,
    UNS_COST_MATRIX_KEY,
    UNS_COST_SCALE_KEY,
    UNS_STATE_CENTROIDS_KEY,
)
from stride.errors import ContractError
from stride.io import read_h5ad
from stride.pp import build_fov_observations, validate_ready
from stride.tl import FitResult, fit

from ..config import TaskAConfigBundle, TaskAOrderedPairFamilySpec
from ..contracts import (
    TaskACoreFitDryRunRecord,
    TaskAFamilyStrideMappingSummary,
    TaskARealDataCrosswalk,
    TaskAStage0FieldMapping,
    TaskAStage0StrideMappingSummary,
)

SEMANTIC_ALIGNMENT_ERROR_PREFIX = "Task A semantic alignment failed:"


@dataclass(frozen=True)
class TaskAStage0Handle:
    """Task-local Stage 0 handle with the field names used by legacy callers."""

    adata: AnnData
    patient_id_key: str = OBS_PATIENT_KEY
    timepoint_key: str = OBS_TIMEPOINT_KEY
    fov_key: str = OBS_FOV_KEY
    domain_key: str = OBS_DOMAIN_KEY
    cell_subtype_key: str = OBS_CELL_TYPE_KEY
    state_id_key: str = OBS_STATE_ID_KEY
    feature_key: str = OBSM_LOCAL_STATE_FEATURES_KEY

    def validate(
        self,
        *,
        require_cell_type: bool = False,
        require_state_axis: bool = False,
        require_cost_scale: bool = False,
        require_cost_matrix: bool = False,
    ) -> None:
        missing_obs = []
        if self.patient_id_key not in self.adata.obs:
            missing_obs.append(self.patient_id_key)
        if self.fov_key not in self.adata.obs:
            missing_obs.append(self.fov_key)
        if self.domain_key not in self.adata.obs:
            missing_obs.append(self.domain_key)
        if self.timepoint_key not in self.adata.obs:
            missing_obs.append(self.timepoint_key)
        if require_cell_type and self.cell_subtype_key not in self.adata.obs:
            missing_obs.append(self.cell_subtype_key)
        if require_state_axis and self.state_id_key not in self.adata.obs:
            missing_obs.append(self.state_id_key)
        if missing_obs:
            raise ContractError("Task A Stage 0 AnnData is missing obs columns: " + ", ".join(missing_obs))
        if require_state_axis and self.feature_key not in self.adata.obsm:
            raise ContractError(f"Task A Stage 0 AnnData is missing obsm[{self.feature_key!r}]")
        if require_cost_scale and UNS_COST_SCALE_KEY not in self.adata.uns:
            raise ContractError(f"Task A Stage 0 AnnData is missing uns[{UNS_COST_SCALE_KEY!r}]")
        if require_cost_matrix and UNS_COST_MATRIX_KEY not in self.adata.uns:
            raise ContractError(f"Task A Stage 0 AnnData is missing uns[{UNS_COST_MATRIX_KEY!r}]")

    def load_state_basis(self) -> TaskAStateAxis:
        return resolve_task_a_state_basis(self)


@dataclass(frozen=True)
class TaskAStateAxis:
    """Task-local state-axis summary retained for compatibility callers."""

    resolved_state_ids: tuple[int, ...]
    n_states: int
    centroids: np.ndarray | None = None
    cost_matrix: np.ndarray | None = None
    cost_scale: float | None = None


@dataclass(frozen=True)
class TaskAFovRecord:
    """Task-local FOV observation record for non-live compatibility helpers."""

    patient_id: str
    timepoint: str
    fov_id: str
    domain_label: str
    community_composition: np.ndarray
    mass: float = 1.0
    mass_mode: str = "uniform"
    metadata: dict[str, Any] | None = None


def load_task_a_dataset_handle(
    stage0_h5ad_or_adata: str | Path | Any,
    *,
    backed: bool = False,
) -> TaskAStage0Handle:
    """Load a Task A Stage 0 surface as a task-local handle."""
    if isinstance(stage0_h5ad_or_adata, TaskAStage0Handle):
        return stage0_h5ad_or_adata
    if hasattr(stage0_h5ad_or_adata, "obs") and hasattr(stage0_h5ad_or_adata, "obsm"):
        adata = normalize_task_a_stage0_aliases(stage0_h5ad_or_adata, copy_adata=True)
        _normalize_task_a_obs_aliases(adata, derive_timepoint_from_domain=False)
        return TaskAStage0Handle(adata=adata)
    if ad is None:
        raise ModuleNotFoundError("anndata is required to load Task-A stage0 h5ad artifacts")
    path = Path(stage0_h5ad_or_adata).expanduser().resolve()
    adata = load_task_a_adata(path, backed=backed)
    adata = normalize_task_a_stage0_aliases(adata, copy_adata=True)
    _normalize_task_a_obs_aliases(adata, derive_timepoint_from_domain=False)
    return TaskAStage0Handle(adata=adata)


def load_task_a_adata(
    stage0_h5ad_or_adata: str | Path | Any,
    *,
    backed: bool = False,
) -> AnnData:
    """Load a Task A Stage 0 h5ad as AnnData without constructing legacy handles."""
    if isinstance(stage0_h5ad_or_adata, AnnData):
        return stage0_h5ad_or_adata
    path = Path(stage0_h5ad_or_adata).expanduser().resolve()
    read_kwargs = {"backed": "r"} if backed else {}
    try:
        return read_h5ad(path, **read_kwargs)
    except TypeError:
        if not backed:
            raise
        return ad.read_h5ad(path, backed="r")


def normalize_task_a_stage0_aliases(
    adata: AnnData,
    *,
    copy_adata: bool = True,
) -> AnnData:
    """Materialize canonical STRIDE keys from Task A Stage 0 aliases.

    Existing canonical keys are never overwritten.
    """
    if copy_adata and adata.isbacked:
        target = adata.to_memory().copy()
    elif copy_adata:
        target = adata.copy()
    else:
        target = adata
    if OBS_STATE_ID_KEY not in target.obs and "proto_id" in target.obs:
        target.obs[OBS_STATE_ID_KEY] = target.obs["proto_id"]
    if UNS_STATE_CENTROIDS_KEY not in target.uns and "prototype_centroids" in target.uns:
        target.uns[UNS_STATE_CENTROIDS_KEY] = copy.deepcopy(target.uns["prototype_centroids"])
    if (
        OBSM_LOCAL_STATE_FEATURES_KEY not in target.obsm
        and "community_features" in target.obsm
    ):
        target.obsm[OBSM_LOCAL_STATE_FEATURES_KEY] = np.asarray(
            target.obsm["community_features"]
        ).copy()
    if OBS_CELL_TYPE_KEY not in target.obs and "cell_type" in target.obs:
        target.obs[OBS_CELL_TYPE_KEY] = target.obs["cell_type"]
    if UNS_COST_SCALE_KEY not in target.uns:
        for alias in ("s_C", "global_cost_scale"):
            if alias in target.uns:
                target.uns[UNS_COST_SCALE_KEY] = copy.deepcopy(target.uns[alias])
                break
    return target


def prepare_task_a_pair_adata(
    stage0_h5ad_or_adata: str | Path | Any,
    family_spec: TaskAOrderedPairFamilySpec,
    *,
    patient_ids: Sequence[str] | None = None,
    backed: bool = False,
    copy_adata: bool = True,
) -> AnnData:
    """Return a pp-ready AnnData for one Task A ordered pair family."""
    adata = load_task_a_adata(stage0_h5ad_or_adata, backed=backed)
    adata = normalize_task_a_stage0_aliases(adata, copy_adata=copy_adata)
    _normalize_task_a_obs_aliases(adata, derive_timepoint_from_domain=True)
    if patient_ids is not None:
        requested = tuple(dict.fromkeys(str(patient_id) for patient_id in patient_ids))
        if not requested:
            raise ContractError("Task A patient_ids selector must not be empty")
        available = set(adata.obs[OBS_PATIENT_KEY].astype(str).unique().tolist())
        missing = tuple(patient_id for patient_id in requested if patient_id not in available)
        if missing:
            raise ContractError(f"Task A requested patients do not exist in Stage 0: {missing}")
        patient_values = adata.obs[OBS_PATIENT_KEY].astype(str).to_numpy(copy=False)
        ordered_indices = np.concatenate(
            [np.flatnonzero(patient_values == patient_id) for patient_id in requested]
        )
        adata = adata[ordered_indices].copy()

    _set_task_a_pair_config(adata, family_spec)
    adata.uns[STRIDE_UNS_KEY].pop(STRIDE_FOV_OBSERVATIONS_KEY, None)
    build_fov_observations(adata)
    validate_ready(adata)
    return adata


def validate_task_a_pair_ready(adata: AnnData) -> AnnData:
    """Validate a Task A pair AnnData and return it unchanged."""
    validate_ready(adata)
    return adata


def _normalize_task_a_obs_aliases(
    adata: AnnData,
    *,
    derive_timepoint_from_domain: bool,
) -> None:
    if OBS_PATIENT_KEY not in adata.obs:
        raise ContractError("Task A Stage 0 AnnData requires adata.obs['patient_id']")
    if OBS_FOV_KEY not in adata.obs:
        adata.obs[OBS_FOV_KEY] = _resolve_task_a_obs_alias(
            adata,
            canonical=OBS_FOV_KEY,
            aliases=("roi_id",),
        )
    if OBS_DOMAIN_KEY not in adata.obs:
        adata.obs[OBS_DOMAIN_KEY] = _resolve_task_a_obs_alias(
            adata,
            canonical=OBS_DOMAIN_KEY,
            aliases=("compartment",),
        )
    if derive_timepoint_from_domain:
        # Task A's ordered proxy uses compartment/domain as the pair-local ordered side.
        adata.obs[OBS_TIMEPOINT_KEY] = adata.obs[OBS_DOMAIN_KEY]
    elif OBS_TIMEPOINT_KEY not in adata.obs:
        adata.obs[OBS_TIMEPOINT_KEY] = "0"


def _resolve_task_a_obs_alias(
    adata: AnnData,
    *,
    canonical: str,
    aliases: Sequence[str],
) -> pd.Series:
    for alias in aliases:
        if alias in adata.obs:
            return adata.obs[alias]
    raise ContractError(
        f"Task A Stage 0 AnnData requires adata.obs[{canonical!r}] "
        f"or one alias from {tuple(aliases)!r}"
    )


def _set_task_a_pair_config(
    adata: AnnData,
    family_spec: TaskAOrderedPairFamilySpec,
) -> None:
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, dict):
        stride_uns = dict(stride_uns or {})
        adata.uns[STRIDE_UNS_KEY] = stride_uns
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, dict):
        config = dict(config or {})
        stride_uns[STRIDE_CONFIG_KEY] = config

    n_states = config.get("n_states")
    if n_states is None:
        n_states = _infer_task_a_n_states(adata)
    config.update(
        {
            "source": str(family_spec.source_domain),
            "target": str(family_spec.target_domain),
            "time_order": list(family_spec.ordered_group_labels),
            STRIDE_RELATIONS_KEY: np.asarray(
                [(family_spec.source_domain, family_spec.target_domain)],
                dtype=object,
            ),
            STRIDE_RELATION_IDS_KEY: [
                (
                    f"{family_spec.source_domain}_{family_spec.source_domain}"
                    f"_to_{family_spec.target_domain}_{family_spec.target_domain}"
                )
            ],
        }
    )
    config.setdefault("community_mode", "fraction")
    config.setdefault("n_states", int(n_states))
    config.setdefault("k_neighbors", _infer_task_a_k_neighbors(adata))


def _infer_task_a_n_states(adata: AnnData) -> int:
    if UNS_COST_MATRIX_KEY in adata.uns:
        cost_matrix = np.asarray(adata.uns[UNS_COST_MATRIX_KEY])
        if cost_matrix.ndim == 2:
            return int(cost_matrix.shape[0])
    if UNS_STATE_CENTROIDS_KEY in adata.uns:
        centroids = np.asarray(adata.uns[UNS_STATE_CENTROIDS_KEY])
        if centroids.ndim >= 1:
            return int(centroids.shape[0])
    if OBS_STATE_ID_KEY in adata.obs:
        state_ids = np.asarray(adata.obs[OBS_STATE_ID_KEY], dtype=int)
        if state_ids.size:
            return int(state_ids.max()) + 1
    raise ContractError("Task A Stage 0 AnnData cannot infer config['n_states']")


def _infer_task_a_k_neighbors(adata: AnnData) -> int | None:
    config = adata.uns.get(STRIDE_UNS_KEY, {}).get(STRIDE_CONFIG_KEY, {})
    if isinstance(config, dict) and "k_neighbors" in config:
        value = config["k_neighbors"]
        return None if value is None else int(value)
    return 20


def resolve_task_a_state_basis(
    handle: TaskAStage0Handle,
    *,
    basis_spec: object | None = None,
) -> TaskAStateAxis:
    adata = normalize_task_a_stage0_aliases(handle.adata, copy_adata=False)
    _normalize_task_a_obs_aliases(adata, derive_timepoint_from_domain=False)
    if UNS_COST_MATRIX_KEY not in adata.uns:
        raise ContractError(f"Task A Stage 0 AnnData is missing uns[{UNS_COST_MATRIX_KEY!r}]")
    if UNS_COST_SCALE_KEY not in adata.uns:
        raise ContractError(f"Task A Stage 0 AnnData is missing uns[{UNS_COST_SCALE_KEY!r}]")
    cost_matrix = np.asarray(adata.uns[UNS_COST_MATRIX_KEY], dtype=float)
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ContractError("Task A Stage 0 cost_matrix must be square")
    if UNS_STATE_CENTROIDS_KEY in adata.uns:
        centroids = np.asarray(adata.uns[UNS_STATE_CENTROIDS_KEY], dtype=float)
    else:
        centroids = None
    state_ids = tuple(range(int(cost_matrix.shape[0])))
    return TaskAStateAxis(
        resolved_state_ids=state_ids,
        n_states=len(state_ids),
        centroids=centroids,
        cost_matrix=cost_matrix,
        cost_scale=float(adata.uns[UNS_COST_SCALE_KEY]),
    )


def build_task_a_stage0_field_mapping(
    handle: TaskAStage0Handle,
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
    handle: TaskAStage0Handle,
    family_spec: TaskAOrderedPairFamilySpec,
    *,
    state_basis: Any,
    mass_mode: str = "uniform",
    require_complete_patients: bool = True,
) -> tuple[TaskAFovRecord, ...]:
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

    observations: list[TaskAFovRecord] = []
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
            TaskAFovRecord(
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
    handle: TaskAStage0Handle,
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


def _assert_timepoint_inert(handle: TaskAStage0Handle) -> tuple[str, ...]:
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
    handle: TaskAStage0Handle,
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
    basis_spec: object | None = None,
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
    basis_spec: object | None = None,
) -> tuple[pd.DataFrame, dict[str, FitResult]]:
    mass_mode = _require_task_a_uniform_mass_mode(config_bundle)
    active_families = tuple(pair_families or config_bundle.ordered_proxy.pair_families)

    records: list[TaskACoreFitDryRunRecord] = []
    results: dict[str, FitResult] = {}
    for family_spec in active_families:
        pair_adata = prepare_task_a_pair_adata(
            stage0_h5ad_or_adata,
            family_spec,
        )
        if mass_mode != "uniform":
            raise ContractError(
                "Task A Step 1 dry-run requires data.mass_mode='uniform'; "
                f"got {mass_mode!r}"
            )
        result = fit(pair_adata, device="cpu")
        results[family_spec.name] = result
        for relation_id in result.relation_ids:
            relation = result.relations[relation_id]
            cohort = relation.cohort
            recurrence_used_patient_ids = tuple(
                str(patient_id) for patient_id in (() if cohort is None else cohort.patient_ids)
            )
            n_recurrence_families = 1 if cohort is not None and cohort.fit_status == "ok" else 0
            for patient_id in relation.patient_ids:
                fit_status = "ok"
                bridge_realized = True
                defer_reason = None
                if str(patient_id) not in recurrence_used_patient_ids:
                    fit_status = "ok"
                    bridge_realized = True
                records.append(
                    TaskACoreFitDryRunRecord(
                        pair_family=family_spec.name,
                        claim_role=family_spec.claim_role,
                        patient_id=str(patient_id),
                        implementation_tier="canonical_stride_tl",
                        fit_surface="stride.tl.fit",
                        fit_status=fit_status,
                        bridge_realized=bridge_realized,
                        defer_reason=defer_reason,
                        uncertainty_status=None,
                        cohort_recurrence_fit_status=(
                            "not_available" if cohort is None else str(cohort.fit_status)
                        ),
                        n_recurrence_families=n_recurrence_families,
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
    "TaskAFovRecord",
    "TaskAStage0Handle",
    "TaskAStateAxis",
    "build_task_a_family_observations",
    "build_task_a_real_data_crosswalk",
    "build_task_a_stage0_field_mapping",
    "describe_task_a_stage0_stride_mapping",
    "load_task_a_adata",
    "load_task_a_dataset_handle",
    "normalize_task_a_stage0_aliases",
    "prepare_task_a_pair_adata",
    "resolve_task_a_state_basis",
    "run_task_a_family_core_fit_dry_run",
    "SEMANTIC_ALIGNMENT_ERROR_PREFIX",
    "summarize_task_a_family_mapping",
    "validate_task_a_pair_ready",
]
