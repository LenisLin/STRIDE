"""Task-local adapters around the public STRIDE fit and result surfaces.

This module centralizes only execution and representation checks shared by
Task A Blocks 0, 1, and 3. Pair-family meaning, permutation semantics,
benchmark truth, and artifact schemas remain owned by their task blocks.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.da import patient_relation_arrays
from stride.errors import ContractError
from stride.tl import FitResult, fit

from .stride_adapter import validate_task_a_pair_ready

TASK_A_FIT_SURFACE = "stride.tl.fit"
_FOV_METADATA_COLUMNS = (
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
    OBS_FOV_KEY,
    OBS_DOMAIN_KEY,
)


@dataclass(frozen=True)
class TaskARelationArrays:
    """One validated Task A view of a fitted STRIDE relation."""

    relation_id: str
    patient_ids: tuple[str, ...]
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    cohort: object | None
    support: dict[str, object]
    warnings: tuple[dict[str, object], ...]

    @property
    def n_states(self) -> int:
        return int(self.A.shape[1])


def fit_task_a_pair(
    adata: AnnData,
    *,
    device: object | None = None,
    estimator: Callable[..., FitResult] = fit,
) -> FitResult:
    """Validate one pair-ready AnnData and run the public STRIDE estimator."""
    validate_task_a_pair_ready(adata)
    return estimator(adata, device="cuda:0" if device is None else device)


def extract_task_a_relations(result: FitResult) -> tuple[TaskARelationArrays, ...]:
    """Return validated patient arrays without adding task-specific meaning."""
    if not isinstance(result, FitResult):
        raise ContractError("Task A fit result must be a stride.tl.FitResult")
    if not result.relations:
        raise ContractError("Task A fit returned no realized relations")
    if not result.relation_ids:
        raise ContractError("Task A fit returned no declared relation IDs")
    if len(set(result.relation_ids)) != len(result.relation_ids):
        raise ContractError("Task A FitResult relation_ids must be unique")
    if set(result.relations) != set(result.relation_ids):
        raise ContractError("Task A FitResult relation keys must match relation_ids")

    extracted = patient_relation_arrays(result)
    relations: list[TaskARelationArrays] = []
    for relation_id in result.relation_ids:
        relation_groups = extracted.get(str(relation_id))
        if relation_groups is None:
            raise ContractError(f"Task A FitResult lacks relation arrays for {relation_id!r}")
        group = relation_groups.get("all")
        if group is None:
            raise ContractError(f"Task A FitResult relation {relation_id!r} lacks an 'all' group")
        patient_ids = tuple(str(value) for value in group.get("patient_ids", ()))
        A = np.asarray(group.get("A"), dtype=float)
        d = np.asarray(group.get("d"), dtype=float)
        e = np.asarray(group.get("e"), dtype=float)
        relation_result = result.relations[str(relation_id)]
        if relation_result.relation_id != str(relation_id):
            raise ContractError(
                f"Task A FitResult relation key {relation_id!r} does not match its relation_id"
            )
        _validate_relation_axes(
            relation_id=str(relation_id),
            patient_ids=patient_ids,
            A=A,
            d=d,
            e=e,
            expected_n_states=int(result.n_states),
        )
        relations.append(
            TaskARelationArrays(
                relation_id=str(relation_id),
                patient_ids=patient_ids,
                A=A.copy(),
                d=d.copy(),
                e=e.copy(),
                cohort=relation_result.cohort,
                support=dict(relation_result.support),
                warnings=tuple(dict(item) for item in relation_result.warnings),
            )
        )
    return tuple(relations)


def require_task_a_fit_support(result: FitResult, *, context: str) -> FitResult:
    """Fail when an expected Task A fit lacks usable patient relation arrays."""
    try:
        extract_task_a_relations(result)
    except ContractError as exc:
        raise ContractError(f"{context}: {exc}") from exc
    return result


def summarize_task_a_fit(
    result: FitResult,
    *,
    pair_family: str,
    run_scope: str | None = None,
) -> dict[str, object]:
    """Return compact, task-neutral fit facts for Task A manifests."""
    relations = extract_task_a_relations(result)
    patient_ids = {
        patient_id
        for relation in relations
        for patient_id in relation.patient_ids
    }
    summary: dict[str, object] = {
        "pair_family": str(pair_family),
        "fit_surface": TASK_A_FIT_SURFACE,
        "relation_count": len(relations),
        "relation_ids": [relation.relation_id for relation in relations],
        "patient_count": len(patient_ids),
        "k_states": int(result.n_states),
        "warning_count": len(result.warnings),
        "warnings": [dict(item) for item in result.warnings],
        "relation_support": {
            relation.relation_id: dict(relation.support) for relation in relations
        },
        "relation_warnings": {
            relation.relation_id: [dict(item) for item in relation.warnings]
            for relation in relations
        },
    }
    if run_scope is not None:
        summary["run_scope"] = str(run_scope)
    return summary


def read_task_a_fov_observations(adata: AnnData) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the canonical pp-ready FOV composition payload without reinterpreting it."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, dict):
        raise ContractError("Task A AnnData is missing adata.uns['stride']")
    payload = stride_uns.get(STRIDE_FOV_OBSERVATIONS_KEY)
    if not isinstance(payload, dict):
        raise ContractError("Task A AnnData is missing STRIDE FOV observations")
    metadata = payload.get("metadata")
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("Task A FOV observation metadata must be a DataFrame")
    if tuple(metadata.columns) != _FOV_METADATA_COLUMNS:
        raise ContractError(
            "Task A FOV metadata columns must be patient_id, timepoint, fov_id, domain_label"
        )
    metadata = metadata.reset_index(drop=True).copy()
    for column in _FOV_METADATA_COLUMNS:
        metadata[column] = metadata[column].astype(str).str.strip()
        if (metadata[column] == "").any():
            raise ContractError(f"Task A FOV metadata column {column!r} contains empty values")
    composition = np.asarray(payload.get("community_composition"), dtype=float)
    if composition.ndim != 2 or composition.shape[0] != metadata.shape[0]:
        raise ContractError("Task A FOV composition must align with metadata rows")
    if not np.isfinite(composition).all() or (composition < 0.0).any():
        raise ContractError("Task A FOV composition must be finite and nonnegative")
    return metadata, composition


def derive_task_a_patient_side_compositions(
    adata: AnnData,
    *,
    source: str | None = None,
    target: str | None = None,
    required_patient_ids: tuple[str, ...] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return patient-level source/target FOV-mean compositions.

    Task A artifact fields may retain the historical `burden` name, but these
    values remain composition-scale means under the current equal-FOV contract.
    """
    metadata, composition = read_task_a_fov_observations(adata)
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    config = stride_uns.get(STRIDE_CONFIG_KEY) if isinstance(stride_uns, dict) else None
    if not isinstance(config, dict):
        raise ContractError("Task A AnnData is missing STRIDE config")
    source_value = str(config.get("source") if source is None else source)
    target_value = str(config.get("target") if target is None else target)
    patient_array = metadata[OBS_PATIENT_KEY].astype(str).to_numpy(copy=False)
    time_array = metadata[OBS_TIMEPOINT_KEY].astype(str).to_numpy(copy=False)
    domain_array = metadata[OBS_DOMAIN_KEY].astype(str).to_numpy(copy=False)
    available = tuple(dict.fromkeys(patient_array.tolist()))
    requested = available if required_patient_ids is None else tuple(required_patient_ids)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing: list[str] = []
    for patient_id in requested:
        patient_mask = patient_array == str(patient_id)
        source_mask = patient_mask & (time_array == source_value) & (domain_array == source_value)
        target_mask = patient_mask & (time_array == target_value) & (domain_array == target_value)
        if not bool(source_mask.any()) or not bool(target_mask.any()):
            missing.append(str(patient_id))
            continue
        result[str(patient_id)] = (
            np.mean(composition[source_mask], axis=0, dtype=float),
            np.mean(composition[target_mask], axis=0, dtype=float),
        )
    if missing:
        raise ContractError(
            "Task A source/target FOV compositions are missing for patients: "
            f"{tuple(missing)}"
        )
    return result


def _validate_relation_axes(
    *,
    relation_id: str,
    patient_ids: tuple[str, ...],
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    expected_n_states: int,
) -> None:
    if not patient_ids:
        raise ContractError(f"Task A relation {relation_id!r} has no fitted patients")
    if len(set(patient_ids)) != len(patient_ids):
        raise ContractError(f"Task A relation {relation_id!r} patient IDs must be unique")
    if A.ndim != 3 or A.shape[0] != len(patient_ids) or A.shape[1] != A.shape[2]:
        raise ContractError(f"Task A relation {relation_id!r} A must have shape [P, K, K]")
    n_states = int(A.shape[1])
    if n_states != expected_n_states:
        raise ContractError(
            f"Task A relation {relation_id!r} K axis does not match FitResult.n_states"
        )
    if d.shape != (len(patient_ids), n_states):
        raise ContractError(f"Task A relation {relation_id!r} d must have shape [P, K]")
    if e.shape != (len(patient_ids), n_states):
        raise ContractError(f"Task A relation {relation_id!r} e must have shape [P, K]")
    if not np.isfinite(A).all() or not np.isfinite(d).all() or not np.isfinite(e).all():
        raise ContractError(f"Task A relation {relation_id!r} arrays must be finite")


__all__ = [
    "TASK_A_FIT_SURFACE",
    "TaskARelationArrays",
    "extract_task_a_relations",
    "derive_task_a_patient_side_compositions",
    "fit_task_a_pair",
    "read_task_a_fov_observations",
    "require_task_a_fit_support",
    "summarize_task_a_fit",
]
