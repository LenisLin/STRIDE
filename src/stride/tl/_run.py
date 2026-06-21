"""Public orchestration for the formal STRIDE fitting surface.

This module owns `stride.tl.fit`. It consumes `.pp-ready` AnnData slots by
fixed key, loops over declared relations in stored order, and delegates
relation-specific work to resolver, training, and output assembly modules.

Caller code must run `stride.pp.validate_ready(adata)` before entering this
surface. This module does not repeat `.io` or `.pp` handoff validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from stride._schema import (
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_RELATION_IDS_KEY,
    STRIDE_RELATIONS_KEY,
    STRIDE_UNS_KEY,
    UNS_COST_MATRIX_KEY,
    UNS_COST_SCALE_KEY,
)
from stride.errors import ContractError

from ._output import FitResult, assemble_fit_result, assemble_relation_result
from ._resolve import RelationInput, resolve_relation
from ._train import train_relation


def fit(adata: AnnData, *, device: Any = "cuda:0") -> FitResult:
    """Fit all declared STRIDE relations from a `.pp-ready` AnnData object.

    Purpose:
        Public `.tl` entry. Read fixed `.pp-ready` slots, resolve each
        declared relation, fit each realized relation, and return a
        relation-indexed `FitResult`.

    Key variables:
        slots: fixed AnnData handoff payload consumed by `.tl`.
        config: STRIDE config mapping under `adata.uns["stride"]["config"]`.
        relation_records: declared relation loop records in stored order.
        relation_input: estimator input for one declared relation.
        training_result: fitted internal `A/d/e` training result for one relation.
        relation_results: realized public relation outputs.
        warnings: compact skipped-support or runtime warning records.
    """
    # slots: `.pp-ready` payload; correctness is delegated to `.pp.validate_ready`.
    # relation_results: realized relation outputs in declared relation order.
    runtime_device = _resolve_fit_device(device)
    slots = _read_fit_slots(adata)
    relation_records = _iter_declared_relations(slots["config"])

    relation_results = []
    fit_warnings = []

    # Empty RelationInput objects represent declared relations skipped by support policy.
    for record in relation_records:
        relation_input = resolve_relation(
            relation_id=record["relation_id"],
            source_timepoint=record["source_timepoint"],
            target_timepoint=record["target_timepoint"],
            source_domain=record["source_domain"],
            target_domain=record["target_domain"],
            metadata=slots["metadata"],
            community_composition=slots["community_composition"],
        )

        if not relation_input.patient_ids:
            fit_warnings.append(_relation_skip_warning(relation_input))
            continue

        training_result = train_relation(
            relation_input,
            slots["cost_matrix"],
            slots["cost_scale"],
            device=runtime_device,
        )
        relation_results.append(
            assemble_relation_result(relation_input, training_result)
        )

    if not relation_results:
        raise ContractError("no realized STRIDE relations remain after support resolution")

    return assemble_fit_result(
        relations=tuple(relation_results),
        warnings=tuple(fit_warnings),
        source=slots["source"],
        target=slots["target"],
        n_states=slots["n_states"],
    )


def _resolve_fit_device(device: Any) -> torch.device:
    """Resolve the public fit device, using CPU when requested CUDA is absent."""
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ContractError(f"invalid runtime device: {device!r}") from exc

    if resolved.type != "cuda":
        return resolved
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise ContractError(f"requested runtime device is unavailable: {resolved}")
    return resolved


def _read_fit_slots(adata: AnnData) -> dict[str, Any]:
    """Read fixed AnnData slots required by `.tl.fit`.

    Purpose:
        Centralize fixed-key access without rebuilding or validating `.pp`
        outputs.

    Key variables:
        stride_uns: top-level STRIDE payload.
        config: declared source, target, relations, and relation ids.
        fov_observations: FOV-level observation payload.
        community_composition: FOV-by-state fraction matrix.
        metadata: FOV row metadata aligned to `community_composition`.
        cost_matrix: shared-state cost matrix.
        cost_scale: shared-state cost scale.
    """
    # stride_uns: canonical payload under `adata.uns["stride"]`.
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object")

    try:
        stride_uns = adata.uns[STRIDE_UNS_KEY]
    except KeyError as exc:
        raise ContractError(f"missing .pp-ready AnnData slot: {exc}") from exc
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")

    try:
        config = stride_uns[STRIDE_CONFIG_KEY]
    except KeyError as exc:
        raise ContractError(f"missing .pp-ready AnnData slot: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    try:
        fov_observations = stride_uns[STRIDE_FOV_OBSERVATIONS_KEY]
    except KeyError as exc:
        raise ContractError(f"missing .pp-ready AnnData slot: {exc}") from exc
    if not isinstance(fov_observations, Mapping):
        raise ContractError(
            "adata.uns['stride']['fov_observations'] must be a mapping"
        )

    try:
        community_composition = fov_observations["community_composition"]
        metadata = fov_observations["metadata"]
        cost_matrix = adata.uns[UNS_COST_MATRIX_KEY]
        cost_scale = adata.uns[UNS_COST_SCALE_KEY]
        n_states = config["n_states"]
        source = config["source"]
        target = config["target"]
    except KeyError as exc:
        raise ContractError(f"missing .pp-ready AnnData slot: {exc}") from exc

    return {
        "config": config,
        "source": source,
        "target": target,
        "n_states": _positive_int(n_states, name="config['n_states']"),
        "community_composition": community_composition,
        "metadata": metadata,
        "cost_matrix": cost_matrix,
        "cost_scale": cost_scale,
    }


def _iter_declared_relations(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return declared relation loop records in stored result order.

    Purpose:
        Pair each relation row with its stable relation id and configured
        source/target timepoints.

    Key variables:
        source_timepoint: configured source side for all relation rows.
        target_timepoint: configured target side for all relation rows.
        relations: two-column source/target domain-pair declaration.
        relation_ids: stable result ids aligned to `relations`.
        relation_index: row index into `relations`.
    """
    # relation_index: aligns one relation row with one stable relation id.
    if not isinstance(config, Mapping):
        raise ContractError("config must be a mapping")

    try:
        source_timepoint = _nonempty_identifier(
            config["source"], name="config['source']"
        )
        target_timepoint = _nonempty_identifier(
            config["target"], name="config['target']"
        )
        relations = np.asarray(config[STRIDE_RELATIONS_KEY], dtype=object)
        raw_relation_ids = config[STRIDE_RELATION_IDS_KEY]
    except KeyError as exc:
        raise ContractError(f"missing STRIDE config key: {exc}") from exc

    if relations.ndim != 2 or relations.shape[1] != 2:
        raise ContractError("config['relations'] shape must be [n_relations, 2]")

    relation_ids = _normalize_relation_ids(raw_relation_ids)
    if len(relation_ids) != relations.shape[0]:
        raise ContractError(
            "config['relation_ids'] length must match config['relations'] row count"
        )
    if len(set(relation_ids)) != len(relation_ids):
        raise ContractError("config['relation_ids'] contains duplicate values")

    records = []
    # Stored relation_ids define public result traversal order.
    for relation_index, (source_domain, target_domain) in enumerate(
        relations.tolist()
    ):
        records.append(
            {
                "relation_index": relation_index,
                "relation_id": relation_ids[relation_index],
                "source_timepoint": source_timepoint,
                "target_timepoint": target_timepoint,
                "source_domain": _nonempty_identifier(
                    source_domain,
                    name=f"config['relations'][{relation_index}, 0]",
                ),
                "target_domain": _nonempty_identifier(
                    target_domain,
                    name=f"config['relations'][{relation_index}, 1]",
                ),
            }
        )
    return tuple(records)


def _relation_skip_warning(relation: RelationInput) -> Mapping[str, Any]:
    return {
        "code": "relation_skipped_no_eligible_patients",
        "relation_id": str(relation.relation_id),
        "source_timepoint": str(relation.source_timepoint),
        "target_timepoint": str(relation.target_timepoint),
        "source_domain": str(relation.source_domain),
        "target_domain": str(relation.target_domain),
        "skipped_patient_ids": tuple(str(item) for item in relation.skipped_patient_ids),
        "support_counts": {
            str(patient_id): {
                str(side): int(count)
                for side, count in counts.items()
            }
            for patient_id, counts in relation.support_counts.items()
        },
    }


def _normalize_relation_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ContractError("config['relation_ids'] must be a sequence, not a string")
    try:
        relation_ids = tuple(
            _nonempty_identifier(item, name="config['relation_ids'] value")
            for item in value
        )
    except TypeError as exc:
        raise ContractError("config['relation_ids'] must be a sequence") from exc
    return relation_ids


def _nonempty_identifier(value: Any, *, name: str) -> str:
    if value is None:
        raise ContractError(f"{name} must not be None")
    if isinstance(value, (bool, np.bool_, dict, list, set, tuple)):
        raise ContractError(f"{name} must be a scalar identifier")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if not isinstance(missing, (bool, np.bool_)):
        raise ContractError(f"{name} must be a scalar identifier")
    if missing:
        raise ContractError(f"{name} contains a missing value")

    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{name} must be a non-empty identifier")
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ContractError(f"{name} must be a positive integer")
    return int(value)
