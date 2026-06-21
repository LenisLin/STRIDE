"""Patient-level fitted array extraction surfaces for STRIDE `.da`."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from stride.errors import ContractError
from stride.tl import FitResult, RelationResult


def patient_relation_arrays(
    result: FitResult | RelationResult,
    *,
    group_labels: Mapping[str, str] | pd.Series | None = None,
    relation_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Mapping[str, object]]]:
    """Extract raw fitted patient-level A/d/e arrays for downstream analysis.

    Scientific question:
        For each declared relation and optional caller-defined patient group,
        which fitted patient-level STRIDE arrays are available as direct
        downstream analysis inputs?

    Input:
        `result` is a `.tl` `RelationResult` or `FitResult`. The function
        consumes only `relation_id`, `patient_ids`, `A`, `d`, `e`, and
        `FitResult.relation_ids` / `relations` traversal fields.

    Grouping:
        `group_labels`, when supplied, maps patient ids to caller-owned group
        labels. Group labels are organization keys only. `.da` does not assign
        clinical or biological meaning to them.

    Output:
        Nested mapping:
        `relation_id -> group_id -> {"patient_ids": tuple[str, ...],
        "A": ndarray, "d": ndarray, "e": ndarray}`.

    Boundary:
        This is a thin extraction surface. It does not revalidate `.tl`
        contracts, inspect support/loss/provenance/warnings/cohort fields,
        derive features, compute statistics, filter patients, mutate inputs, or
        write files.
    """
    relations = _resolve_relations(result, relation_ids)
    extracted: dict[str, dict[str, Mapping[str, object]]] = {}
    for relation in relations:
        patient_ids = tuple(str(value) for value in relation.patient_ids)
        A, d, e = _relation_arrays(relation, n_patients=len(patient_ids))
        labels = _resolve_group_labels(patient_ids, group_labels)
        relation_groups: dict[str, Mapping[str, object]] = {}
        for group_id in _ordered_unique(labels):
            mask = np.asarray([label == group_id for label in labels], dtype=bool)
            relation_groups[group_id] = {
                "relation_id": str(relation.relation_id),
                "group_id": group_id,
                "patient_ids": tuple(
                    patient_id for patient_id, keep in zip(patient_ids, mask, strict=True) if keep
                ),
                "A": A[mask].copy(),
                "d": d[mask].copy(),
                "e": e[mask].copy(),
            }
        extracted[str(relation.relation_id)] = relation_groups
    return extracted


def _resolve_relations(
    result: FitResult | RelationResult,
    relation_ids: Sequence[str] | None,
) -> tuple[RelationResult, ...]:
    """Resolve public result containers into relation results."""
    if isinstance(result, RelationResult):
        if relation_ids is not None:
            requested = tuple(str(value) for value in relation_ids)
            if requested != (str(result.relation_id),):
                raise ContractError("relation_ids does not match RelationResult")
        return (result,)

    if isinstance(result, FitResult):
        if relation_ids is None:
            requested = tuple(str(value) for value in result.relation_ids)
        else:
            requested = tuple(str(value) for value in relation_ids)
        missing = [relation_id for relation_id in requested if relation_id not in result.relations]
        if missing:
            raise ContractError("unknown relation_id values: " + ", ".join(missing))
        return tuple(result.relations[relation_id] for relation_id in requested)

    raise ContractError("result must be a RelationResult or FitResult")


def _relation_arrays(
    relation: RelationResult,
    *,
    n_patients: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lightweight-checked fitted arrays as float copies."""
    A = _as_finite_array(relation.A, name="A")
    d = _as_finite_array(relation.d, name="d")
    e = _as_finite_array(relation.e, name="e")
    if A.ndim != 3 or A.shape[0] != n_patients or A.shape[1] != A.shape[2]:
        raise ContractError("RelationResult.A must have shape [P, K, K]")
    K = int(A.shape[1])
    if d.shape != (n_patients, K):
        raise ContractError("RelationResult.d must have shape [P, K]")
    if e.shape != (n_patients, K):
        raise ContractError("RelationResult.e must have shape [P, K]")
    return A, d, e


def _as_finite_array(value: object, *, name: str) -> np.ndarray:
    """Convert a public result array to a finite float copy."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"RelationResult.{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise ContractError(f"RelationResult.{name} must contain only finite values")
    return array.copy()


def _resolve_group_labels(
    patient_ids: Sequence[str],
    group_labels: Mapping[str, str] | pd.Series | None,
) -> tuple[str, ...]:
    """Resolve caller-owned group labels aligned to patient ids."""
    if group_labels is None:
        return tuple("all" for _ in patient_ids)
    labels = []
    missing = []
    for patient_id in patient_ids:
        if patient_id not in group_labels:
            missing.append(patient_id)
        else:
            label = group_labels[patient_id]
            if pd.isna(label):
                missing.append(patient_id)
            else:
                labels.append(str(label))
    if missing:
        raise ContractError("group_labels are missing patient ids: " + ", ".join(missing))
    return tuple(labels)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    """Return values in first-observed order."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)
