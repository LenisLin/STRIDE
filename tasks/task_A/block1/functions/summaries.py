"""Task-local Block 1 summary extraction from native relation exports."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError

from ...config import TaskAOrderedPairFamilySpec
from .native_export import Block1RelationExport, PatientRelationRecord

FAMILY_SUMMARY_FILENAME = "block1_family_summary.csv"
SOURCE_COMMUNITY_SUMMARY_FILENAME = "block1_source_community_summary.csv"
TARGET_COMMUNITY_SUMMARY_FILENAME = "block1_target_community_summary.csv"

SUMMARY_CONTRACT_VERSION = "task_a_block1_summary_v1"
SOURCE_ELIGIBILITY_RULE = "mu_minus > 0"
TARGET_ELIGIBILITY_RULE = "mu_plus > 0"

PROOF_CARRYING_SUMMARY_NAMES: tuple[str, ...] = (
    "self_retention",
    "depletion",
)
SUPPORTIVE_SUMMARY_NAMES: tuple[str, ...] = (
    "off_diagonal_remodeling",
    "emergence",
)
FAMILY_SUMMARY_SCALES: tuple[str, ...] = (
    "burden_weighted",
    "community_mean",
)

FAMILY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family",
    "claim_role",
    "source_domain",
    "target_domain",
    "summary_name",
    "summary_role",
    "scale",
    "value",
    "eligible_entity_axis",
    "eligible_entity_count",
    "burden_total",
)
SOURCE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family",
    "claim_role",
    "source_domain",
    "target_domain",
    "source_community_id",
    "source_burden",
    "source_weight",
    "self_retention",
    "depletion",
    "off_diagonal_remodeling",
    "self_retention_burden",
    "depletion_burden",
    "off_diagonal_burden",
)
TARGET_SUMMARY_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family",
    "claim_role",
    "source_domain",
    "target_domain",
    "target_community_id",
    "target_burden",
    "target_weight",
    "matched_incoming_burden",
    "open_incoming_tendency",
    "open_incoming_burden",
)


@dataclass(frozen=True)
class _PatientSummaryInputs:
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    mu_minus: np.ndarray
    mu_plus: np.ndarray
    state_ids: tuple[int, ...]
    transition_burden: np.ndarray
    depletion_burden_by_source: np.ndarray
    target_open_burden_by_target: np.ndarray


def _as_float_array(value: object, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ContractError(f"Task A summary extraction expected {field_name} to be 1D")
    return array


def _as_float_matrix(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ContractError(f"Task A summary extraction expected {field_name} to be square")
    return matrix


def _require_vector_shape(
    array: np.ndarray,
    *,
    n_states: int,
    field_name: str,
) -> np.ndarray:
    if array.shape != (n_states,):
        raise ContractError(
            "Task A summary extraction expected "
            f"{field_name} to have shape {(n_states,)}, got {array.shape}"
        )
    return array


def _extract_patient_summary_inputs(
    patient_record: PatientRelationRecord,
    *,
    state_ids: Sequence[int],
) -> _PatientSummaryInputs:
    if not patient_record.is_ok:
        raise ContractError("Task A summary extraction requires realized patient relation arrays")
    A = _as_float_matrix(patient_record.A, field_name="A")
    n_states = A.shape[0]
    d = _require_vector_shape(
        _as_float_array(patient_record.d, field_name="d"),
        n_states=n_states,
        field_name="d",
    )
    e = _require_vector_shape(
        _as_float_array(patient_record.e, field_name="e"),
        n_states=n_states,
        field_name="e",
    )
    mu_minus = _require_vector_shape(
        _as_float_array(patient_record.mu_minus, field_name="mu_minus"),
        n_states=n_states,
        field_name="mu_minus",
    )
    mu_plus = _require_vector_shape(
        _as_float_array(patient_record.mu_plus, field_name="mu_plus"),
        n_states=n_states,
        field_name="mu_plus",
    )
    resolved_state_ids = tuple(int(state_id) for state_id in state_ids)
    if len(resolved_state_ids) != n_states:
        raise ContractError("Task A summary extraction requires native-export state_ids to align with A/d/e")
    transition_burden = A * mu_minus[:, None]
    depletion_burden_by_source = d * mu_minus
    target_open_burden_by_target = e * float(np.sum(mu_minus, dtype=float))
    return _PatientSummaryInputs(
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        state_ids=resolved_state_ids,
        transition_burden=transition_burden,
        depletion_burden_by_source=depletion_burden_by_source,
        target_open_burden_by_target=target_open_burden_by_target,
    )


def _normalized_weights(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(values, dtype=float)
    total = float(np.sum(values[mask], dtype=float))
    if total <= 0.0:
        return weights
    weights[mask] = values[mask] / total
    return weights


def _summary_role(summary_name: str) -> str:
    if summary_name in PROOF_CARRYING_SUMMARY_NAMES:
        return "proof_carrying"
    if summary_name == "off_diagonal_remodeling":
        return "diagnostic_supportive"
    if summary_name == "emergence":
        return "supportive"
    raise ContractError(f"Unknown Task A summary_name {summary_name!r}")


def _build_family_summary_records(
    *,
    patient_record: PatientRelationRecord,
    family_spec: TaskAOrderedPairFamilySpec,
    state_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not patient_record.is_ok:
        return []

    inputs = _extract_patient_summary_inputs(patient_record, state_ids=state_ids)
    source_mask = np.asarray(inputs.mu_minus > 0.0, dtype=bool)
    target_mask = np.asarray(inputs.mu_plus > 0.0, dtype=bool)
    source_weights = _normalized_weights(inputs.mu_minus, source_mask)
    target_weights = _normalized_weights(inputs.mu_plus, target_mask)

    diagonal_operator = np.diag(inputs.A).astype(float, copy=False)
    off_diagonal_operator = np.sum(inputs.A, axis=1, dtype=float) - diagonal_operator

    records: list[dict[str, Any]] = []
    source_summary_inputs = {
        "self_retention": (
            float(np.sum(source_weights * diagonal_operator, dtype=float)),
            float(np.mean(diagonal_operator[source_mask], dtype=float)) if bool(source_mask.any()) else np.nan,
        ),
        "depletion": (
            float(np.sum(source_weights * inputs.d, dtype=float)),
            float(np.mean(inputs.d[source_mask], dtype=float)) if bool(source_mask.any()) else np.nan,
        ),
        "off_diagonal_remodeling": (
            float(np.sum(source_weights * off_diagonal_operator, dtype=float)),
            float(np.mean(off_diagonal_operator[source_mask], dtype=float)) if bool(source_mask.any()) else np.nan,
        ),
    }
    target_summary_inputs = {
        "emergence": (
            float(np.sum(target_weights * inputs.e, dtype=float)),
            float(np.mean(inputs.e[target_mask], dtype=float)) if bool(target_mask.any()) else np.nan,
        )
    }

    for summary_name, (bw_value, mean_value) in source_summary_inputs.items():
        summary_role = _summary_role(summary_name)
        records.append(
            {
                "patient_id": str(patient_record.patient_id),
                "pair_family": family_spec.name,
                "claim_role": family_spec.claim_role,
                "source_domain": family_spec.source_domain,
                "target_domain": family_spec.target_domain,
                "summary_name": summary_name,
                "summary_role": summary_role,
                "scale": "burden_weighted",
                "value": bw_value,
                "eligible_entity_axis": "source",
                "eligible_entity_count": int(np.sum(source_mask, dtype=int)),
                "burden_total": float(np.sum(inputs.mu_minus[source_mask], dtype=float)),
            }
        )
        records.append(
            {
                "patient_id": str(patient_record.patient_id),
                "pair_family": family_spec.name,
                "claim_role": family_spec.claim_role,
                "source_domain": family_spec.source_domain,
                "target_domain": family_spec.target_domain,
                "summary_name": summary_name,
                "summary_role": summary_role,
                "scale": "community_mean",
                "value": mean_value,
                "eligible_entity_axis": "source",
                "eligible_entity_count": int(np.sum(source_mask, dtype=int)),
                "burden_total": float(np.sum(inputs.mu_minus[source_mask], dtype=float)),
            }
        )

    for summary_name, (bw_value, mean_value) in target_summary_inputs.items():
        summary_role = _summary_role(summary_name)
        records.append(
            {
                "patient_id": str(patient_record.patient_id),
                "pair_family": family_spec.name,
                "claim_role": family_spec.claim_role,
                "source_domain": family_spec.source_domain,
                "target_domain": family_spec.target_domain,
                "summary_name": summary_name,
                "summary_role": summary_role,
                "scale": "burden_weighted",
                "value": bw_value,
                "eligible_entity_axis": "target",
                "eligible_entity_count": int(np.sum(target_mask, dtype=int)),
                "burden_total": float(np.sum(inputs.mu_plus[target_mask], dtype=float)),
            }
        )
        records.append(
            {
                "patient_id": str(patient_record.patient_id),
                "pair_family": family_spec.name,
                "claim_role": family_spec.claim_role,
                "source_domain": family_spec.source_domain,
                "target_domain": family_spec.target_domain,
                "summary_name": summary_name,
                "summary_role": summary_role,
                "scale": "community_mean",
                "value": mean_value,
                "eligible_entity_axis": "target",
                "eligible_entity_count": int(np.sum(target_mask, dtype=int)),
                "burden_total": float(np.sum(inputs.mu_plus[target_mask], dtype=float)),
            }
        )

    return records


def _build_source_summary_records(
    *,
    patient_record: PatientRelationRecord,
    family_spec: TaskAOrderedPairFamilySpec,
    state_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not patient_record.is_ok:
        return []

    inputs = _extract_patient_summary_inputs(patient_record, state_ids=state_ids)
    source_mask = np.asarray(inputs.mu_minus > 0.0, dtype=bool)
    source_weights = _normalized_weights(inputs.mu_minus, source_mask)
    records: list[dict[str, Any]] = []

    for source_index, state_id in enumerate(inputs.state_ids):
        eligible = bool(source_mask[source_index])
        row = inputs.A[source_index]
        self_retention = np.nan
        depletion = np.nan
        off_diagonal = np.nan
        diagonal_burden = np.nan
        depletion_burden = np.nan
        off_diagonal_burden = np.nan
        if eligible:
            self_retention = float(row[source_index])
            depletion = float(inputs.d[source_index])
            off_diagonal = float(np.sum(row, dtype=float) - self_retention)
            diagonal_burden = float(inputs.transition_burden[source_index, source_index])
            depletion_burden = float(inputs.depletion_burden_by_source[source_index])
            row_burden = np.asarray(inputs.transition_burden[source_index], dtype=float)
            off_diagonal_burden = float(np.sum(row_burden, dtype=float) - diagonal_burden)
        record: dict[str, Any] = {
            "patient_id": str(patient_record.patient_id),
            "pair_family": family_spec.name,
            "claim_role": family_spec.claim_role,
            "source_domain": family_spec.source_domain,
            "target_domain": family_spec.target_domain,
            "source_community_id": int(state_id),
            "source_burden": float(inputs.mu_minus[source_index]),
            "source_weight": float(source_weights[source_index]),
            "self_retention": self_retention,
            "depletion": depletion,
            "off_diagonal_remodeling": off_diagonal,
            "self_retention_burden": diagonal_burden,
            "depletion_burden": depletion_burden,
            "off_diagonal_burden": off_diagonal_burden,
        }
        records.append(record)
    return records


def _build_target_summary_records(
    *,
    patient_record: PatientRelationRecord,
    family_spec: TaskAOrderedPairFamilySpec,
    state_ids: Sequence[int],
) -> list[dict[str, Any]]:
    if not patient_record.is_ok:
        return []

    inputs = _extract_patient_summary_inputs(patient_record, state_ids=state_ids)
    target_mask = np.asarray(inputs.mu_plus > 0.0, dtype=bool)
    target_weights = _normalized_weights(inputs.mu_plus, target_mask)
    matched_incoming_burden = np.sum(inputs.transition_burden, axis=0, dtype=float)

    records: list[dict[str, Any]] = []
    for target_index, state_id in enumerate(inputs.state_ids):
        eligible = bool(target_mask[target_index])
        records.append(
            {
                "patient_id": str(patient_record.patient_id),
                "pair_family": family_spec.name,
                "claim_role": family_spec.claim_role,
                "source_domain": family_spec.source_domain,
                "target_domain": family_spec.target_domain,
                "target_community_id": int(state_id),
                "target_burden": float(inputs.mu_plus[target_index]),
                "target_weight": float(target_weights[target_index]),
                "matched_incoming_burden": (
                    float(matched_incoming_burden[target_index]) if eligible else np.nan
                ),
                "open_incoming_tendency": float(inputs.e[target_index]) if eligible else np.nan,
                "open_incoming_burden": (
                    float(inputs.target_open_burden_by_target[target_index]) if eligible else np.nan
                ),
            }
        )
    return records


def _finalize_summary_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    return (
        frame.loc[:, list(columns)]
        .sort_values(list(sort_columns), kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )


def build_block1_summary_frames(
    *,
    native_exports: Mapping[str, Block1RelationExport],
    pair_families: Iterable[TaskAOrderedPairFamilySpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []

    active_pair_families = tuple(pair_families)
    for family_spec in active_pair_families:
        export = native_exports.get(family_spec.name)
        if export is None:
            continue
        for patient_record in export.patient_records:
            family_records.extend(
                _build_family_summary_records(
                    patient_record=patient_record,
                    family_spec=family_spec,
                    state_ids=export.state_ids,
                )
            )
            source_records.extend(
                _build_source_summary_records(
                    patient_record=patient_record,
                    family_spec=family_spec,
                    state_ids=export.state_ids,
                )
            )
            target_records.extend(
                _build_target_summary_records(
                    patient_record=patient_record,
                    family_spec=family_spec,
                    state_ids=export.state_ids,
                )
            )

    family_frame = _finalize_summary_frame(
        pd.DataFrame.from_records(family_records),
        columns=FAMILY_SUMMARY_COLUMNS,
        sort_columns=("patient_id", "pair_family", "summary_name", "scale"),
    )
    source_frame = _finalize_summary_frame(
        pd.DataFrame.from_records(source_records),
        columns=SOURCE_SUMMARY_COLUMNS,
        sort_columns=("patient_id", "pair_family", "source_community_id"),
    )
    target_frame = _finalize_summary_frame(
        pd.DataFrame.from_records(target_records),
        columns=TARGET_SUMMARY_COLUMNS,
        sort_columns=("patient_id", "pair_family", "target_community_id"),
    )
    return family_frame, source_frame, target_frame


__all__ = [
    "FAMILY_SUMMARY_FILENAME",
    "FAMILY_SUMMARY_SCALES",
    "PROOF_CARRYING_SUMMARY_NAMES",
    "SOURCE_COMMUNITY_SUMMARY_FILENAME",
    "SOURCE_ELIGIBILITY_RULE",
    "SUMMARY_CONTRACT_VERSION",
    "SUPPORTIVE_SUMMARY_NAMES",
    "TARGET_COMMUNITY_SUMMARY_FILENAME",
    "TARGET_ELIGIBILITY_RULE",
    "build_block1_summary_frames",
]
