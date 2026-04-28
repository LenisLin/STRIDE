"""Task-local Block 1 paired comparison writers for Task A.

These utilities build patient-paired confirmatory `TC-IM` versus `TC-PT`
comparison surfaces from the frozen Block 1 summary outputs. They do not
recompute STRIDE fits or change `src/stride/` semantics.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stride.errors import ContractError

from .summaries import FAMILY_SUMMARY_SCALES


CONFIRMATORY_FAMILY_COMPARISON_FILENAME = "block1_confirmatory_family_comparison.csv"
EXPLORATORY_SOURCE_COMMUNITY_COMPARISON_FILENAME = (
    "block1_exploratory_source_community_comparison.csv"
)
EXPLORATORY_TARGET_COMMUNITY_COMPARISON_FILENAME = (
    "block1_exploratory_target_community_comparison.csv"
)
PAIRED_COMPARISON_CONTRACT_VERSION = "task_a_block1_paired_comparison_v1"

LEFT_PAIR_FAMILY = "TC-IM"
RIGHT_PAIR_FAMILY = "TC-PT"
CONFIRMATORY_PAIR_FAMILIES: tuple[str, str] = (LEFT_PAIR_FAMILY, RIGHT_PAIR_FAMILY)
CONFIRMATORY_SCOPE_ROLE = "confirmatory"
EXPLORATORY_SUPPORTIVE_SCOPE_ROLE = "exploratory_supportive"
MISSING_FIT_STATUS = "missing"
DEFERRED_FIT_STATUSES: frozenset[str] = frozenset({"deferred", "failed"})

FAMILY_SUMMARY_ROLE_BY_NAME: Mapping[str, str] = {
    "self_retention": "proof_carrying",
    "depletion": "proof_carrying",
    "off_diagonal_remodeling": "diagnostic_supportive",
    "emergence": "supportive",
}
FAMILY_ELIGIBLE_AXIS_BY_SUMMARY_NAME: Mapping[str, str] = {
    "self_retention": "source",
    "depletion": "source",
    "off_diagonal_remodeling": "source",
    "emergence": "target",
}
SOURCE_COMMUNITY_SUMMARY_NAMES: tuple[str, ...] = (
    "self_retention",
    "depletion",
    "off_diagonal_remodeling",
)
TARGET_COMMUNITY_SUMMARY_NAMES: tuple[str, ...] = (
    "incoming_matched_operator",
    "emergence_tendency",
)
TARGET_COMMUNITY_SUMMARY_ROLE_BY_NAME: Mapping[str, str] = {
    "incoming_matched_operator": "supportive",
    "emergence_tendency": "supportive",
}

FAMILY_COMPARISON_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family_left",
    "pair_family_right",
    "summary_name",
    "summary_role",
    "scale",
    "eligible_entity_axis",
    "tc_im_value",
    "tc_pt_value",
    "delta_tc_im_minus_tc_pt",
    "contrast_direction",
    "comparison_status",
    "tc_im_fit_status",
    "tc_pt_fit_status",
    "tc_im_defer_reason",
    "tc_pt_defer_reason",
    "tc_im_eligible_entity_count",
    "tc_pt_eligible_entity_count",
    "tc_im_burden_total",
    "tc_pt_burden_total",
    "comparison_scope_role",
)
SOURCE_COMMUNITY_COMPARISON_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family_left",
    "pair_family_right",
    "source_community_id",
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "tc_im_value",
    "tc_pt_value",
    "delta_tc_im_minus_tc_pt",
    "contrast_direction",
    "comparison_status",
    "tc_im_fit_status",
    "tc_pt_fit_status",
    "tc_im_defer_reason",
    "tc_pt_defer_reason",
    "tc_im_source_burden",
    "tc_pt_source_burden",
    "tc_im_source_weight",
    "tc_pt_source_weight",
    "comparison_scope_role",
)
TARGET_COMMUNITY_COMPARISON_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family_left",
    "pair_family_right",
    "target_community_id",
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "tc_im_value",
    "tc_pt_value",
    "delta_tc_im_minus_tc_pt",
    "contrast_direction",
    "comparison_status",
    "tc_im_fit_status",
    "tc_pt_fit_status",
    "tc_im_defer_reason",
    "tc_pt_defer_reason",
    "tc_im_target_burden",
    "tc_pt_target_burden",
    "tc_im_target_weight",
    "tc_pt_target_weight",
    "comparison_scope_role",
)


def _normalize_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if pd.isna(value):
        return None
    return str(value)


def _comparison_direction(delta: float | None) -> str:
    if delta is None or not np.isfinite(delta):
        return "not_estimable"
    if np.isclose(delta, 0.0, atol=1.0e-12):
        return "tc_im_eq_tc_pt"
    if delta > 0.0:
        return "tc_im_gt_tc_pt"
    return "tc_im_lt_tc_pt"


def _unique_records(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    if frame.empty:
        return lookup
    for row in frame.to_dict(orient="records"):
        key = tuple(row[column] for column in key_columns)
        if key in lookup:
            raise ContractError(f"Task A {label} contains duplicate rows for key {key!r}")
        lookup[key] = row
    return lookup


def _fit_status_records(
    dry_run_df: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    if dry_run_df.empty:
        return {}
    filtered = dry_run_df.loc[
        dry_run_df["pair_family"].astype(str).isin(CONFIRMATORY_PAIR_FAMILIES)
    ].copy()
    return _unique_records(
        filtered,
        key_columns=("patient_id", "pair_family"),
        label="Block 1 dry-run surface",
    )


def _fit_status_details(
    lookup: Mapping[tuple[str, str], dict[str, Any]],
    *,
    patient_id: str,
    pair_family: str,
) -> tuple[str, str | None]:
    row = lookup.get((patient_id, pair_family))
    if row is None:
        return MISSING_FIT_STATUS, None
    return str(row["fit_status"]), _normalize_optional_string(row.get("defer_reason"))


def _family_axis_rows() -> tuple[tuple[str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str]] = []
    ordered_summary_names = (
        "self_retention",
        "depletion",
        "off_diagonal_remodeling",
        "emergence",
    )
    for summary_name in ordered_summary_names:
        for scale in FAMILY_SUMMARY_SCALES:
            rows.append(
                (
                    summary_name,
                    FAMILY_SUMMARY_ROLE_BY_NAME[summary_name],
                    scale,
                    FAMILY_ELIGIBLE_AXIS_BY_SUMMARY_NAME[summary_name],
                )
            )
    return tuple(rows)


def _validate_family_summary_rows(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    filtered = frame.loc[
        frame["pair_family"].astype(str).isin(CONFIRMATORY_PAIR_FAMILIES)
    ].copy()
    for row in filtered.to_dict(orient="records"):
        summary_name = str(row["summary_name"])
        scale = str(row["scale"])
        if summary_name not in FAMILY_SUMMARY_ROLE_BY_NAME:
            raise ContractError(f"Unexpected Block 1 family summary_name {summary_name!r}")
        if scale not in FAMILY_SUMMARY_SCALES:
            raise ContractError(f"Unexpected Block 1 family summary scale {scale!r}")


def _validate_row_against_status(
    *,
    row: Mapping[str, Any] | None,
    fit_status: str,
    label: str,
    allow_missing_ok_row: bool,
) -> None:
    if fit_status == "ok":
        if row is None and not allow_missing_ok_row:
            raise ContractError(f"Task A {label} is missing an expected summary row for an ok fit")
        return
    if row is not None:
        raise ContractError(
            f"Task A {label} emitted a summary row despite fit_status={fit_status!r}"
        )


def _finalize_frame(
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


def _build_family_comparison_frame(
    *,
    dry_run_lookup: Mapping[tuple[str, str], dict[str, Any]],
    family_summary_df: pd.DataFrame,
    patient_ids: Sequence[str],
) -> pd.DataFrame:
    _validate_family_summary_rows(family_summary_df)
    summary_lookup = _unique_records(
        family_summary_df.loc[
            family_summary_df["pair_family"].astype(str).isin(CONFIRMATORY_PAIR_FAMILIES)
        ].copy(),
        key_columns=("patient_id", "pair_family", "summary_name", "scale"),
        label="Block 1 family summary surface",
    )

    records: list[dict[str, Any]] = []
    for patient_id in patient_ids:
        left_status, left_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=LEFT_PAIR_FAMILY,
        )
        right_status, right_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=RIGHT_PAIR_FAMILY,
        )
        for summary_name, summary_role, scale, eligible_axis in _family_axis_rows():
            left_row = summary_lookup.get((patient_id, LEFT_PAIR_FAMILY, summary_name, scale))
            right_row = summary_lookup.get((patient_id, RIGHT_PAIR_FAMILY, summary_name, scale))
            _validate_row_against_status(
                row=left_row,
                fit_status=left_status,
                label=f"family comparison {LEFT_PAIR_FAMILY}/{patient_id}/{summary_name}/{scale}",
                allow_missing_ok_row=False,
            )
            _validate_row_against_status(
                row=right_row,
                fit_status=right_status,
                label=f"family comparison {RIGHT_PAIR_FAMILY}/{patient_id}/{summary_name}/{scale}",
                allow_missing_ok_row=False,
            )

            comparison_status = "missing"
            delta: float | None = None
            left_value = np.nan
            right_value = np.nan
            if left_status in DEFERRED_FIT_STATUSES or right_status in DEFERRED_FIT_STATUSES:
                comparison_status = "deferred"
            elif left_status == "ok" and right_status == "ok":
                if left_row is None or right_row is None:
                    raise ContractError(
                        "Task A family comparison expected both confirmatory summary rows for "
                        f"patient_id={patient_id!r}, summary_name={summary_name!r}, scale={scale!r}"
                    )
                comparison_status = "estimable"
                left_value = float(left_row["value"])
                right_value = float(right_row["value"])
                delta = left_value - right_value
            elif left_status == "ok":
                if left_row is None:
                    raise ContractError(
                        f"Task A family comparison is missing the {LEFT_PAIR_FAMILY} row for patient {patient_id!r}"
                    )
                left_value = float(left_row["value"])
            elif right_status == "ok":
                if right_row is None:
                    raise ContractError(
                        f"Task A family comparison is missing the {RIGHT_PAIR_FAMILY} row for patient {patient_id!r}"
                    )
                right_value = float(right_row["value"])

            records.append(
                {
                    "patient_id": patient_id,
                    "pair_family_left": LEFT_PAIR_FAMILY,
                    "pair_family_right": RIGHT_PAIR_FAMILY,
                    "summary_name": summary_name,
                    "summary_role": summary_role,
                    "scale": scale,
                    "eligible_entity_axis": eligible_axis,
                    "tc_im_value": left_value,
                    "tc_pt_value": right_value,
                    "delta_tc_im_minus_tc_pt": np.nan if delta is None else float(delta),
                    "contrast_direction": _comparison_direction(delta),
                    "comparison_status": comparison_status,
                    "tc_im_fit_status": left_status,
                    "tc_pt_fit_status": right_status,
                    "tc_im_defer_reason": left_reason,
                    "tc_pt_defer_reason": right_reason,
                    "tc_im_eligible_entity_count": np.nan if left_row is None else int(left_row["eligible_entity_count"]),
                    "tc_pt_eligible_entity_count": np.nan if right_row is None else int(right_row["eligible_entity_count"]),
                    "tc_im_burden_total": np.nan if left_row is None else float(left_row["burden_total"]),
                    "tc_pt_burden_total": np.nan if right_row is None else float(right_row["burden_total"]),
                    "comparison_scope_role": CONFIRMATORY_SCOPE_ROLE,
                }
            )

    return _finalize_frame(
        pd.DataFrame.from_records(records),
        columns=FAMILY_COMPARISON_COLUMNS,
        sort_columns=("patient_id", "summary_name", "scale"),
    )


def _source_row_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, Any]]:
    if frame.empty and "pair_family" not in frame.columns:
        return {}
    filtered = frame.loc[
        frame["pair_family"].astype(str).isin(CONFIRMATORY_PAIR_FAMILIES)
    ].copy()
    if not filtered.empty:
        filtered["source_community_id"] = filtered["source_community_id"].astype(int)
    return _unique_records(
        filtered,
        key_columns=("patient_id", "pair_family", "source_community_id"),
        label="Block 1 source-community summary surface",
    )


def _target_row_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, Any]]:
    if frame.empty and "pair_family" not in frame.columns:
        return {}
    filtered = frame.loc[
        frame["pair_family"].astype(str).isin(CONFIRMATORY_PAIR_FAMILIES)
    ].copy()
    if not filtered.empty:
        filtered["target_community_id"] = filtered["target_community_id"].astype(int)
    return _unique_records(
        filtered,
        key_columns=("patient_id", "pair_family", "target_community_id"),
        label="Block 1 target-community summary surface",
    )


def _build_source_community_comparison_frame(
    *,
    dry_run_lookup: Mapping[tuple[str, str], dict[str, Any]],
    source_summary_df: pd.DataFrame,
    patient_ids: Sequence[str],
) -> pd.DataFrame:
    summary_lookup = _source_row_lookup(source_summary_df)
    communities_by_patient: dict[str, set[int]] = {patient_id: set() for patient_id in patient_ids}
    for patient_id, _pair_family, source_community_id in summary_lookup:
        communities_by_patient.setdefault(str(patient_id), set()).add(int(source_community_id))

    records: list[dict[str, Any]] = []
    for patient_id in patient_ids:
        left_status, left_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=LEFT_PAIR_FAMILY,
        )
        right_status, right_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=RIGHT_PAIR_FAMILY,
        )
        for source_community_id in sorted(communities_by_patient.get(patient_id, set())):
            left_row = summary_lookup.get((patient_id, LEFT_PAIR_FAMILY, source_community_id))
            right_row = summary_lookup.get((patient_id, RIGHT_PAIR_FAMILY, source_community_id))
            _validate_row_against_status(
                row=left_row,
                fit_status=left_status,
                label=f"source-community comparison {LEFT_PAIR_FAMILY}/{patient_id}/{source_community_id}",
                allow_missing_ok_row=True,
            )
            _validate_row_against_status(
                row=right_row,
                fit_status=right_status,
                label=f"source-community comparison {RIGHT_PAIR_FAMILY}/{patient_id}/{source_community_id}",
                allow_missing_ok_row=True,
            )

            for summary_name in SOURCE_COMMUNITY_SUMMARY_NAMES:
                comparison_status = "missing"
                delta: float | None = None
                left_value = np.nan
                right_value = np.nan
                if left_status in DEFERRED_FIT_STATUSES or right_status in DEFERRED_FIT_STATUSES:
                    comparison_status = "deferred"
                elif left_row is not None and right_row is not None:
                    comparison_status = "estimable"
                    left_value = float(left_row[summary_name])
                    right_value = float(right_row[summary_name])
                    delta = left_value - right_value
                elif left_row is not None:
                    left_value = float(left_row[summary_name])
                elif right_row is not None:
                    right_value = float(right_row[summary_name])

                records.append(
                    {
                        "patient_id": patient_id,
                        "pair_family_left": LEFT_PAIR_FAMILY,
                        "pair_family_right": RIGHT_PAIR_FAMILY,
                        "source_community_id": source_community_id,
                        "summary_name": summary_name,
                        "summary_role": FAMILY_SUMMARY_ROLE_BY_NAME[summary_name],
                        "eligible_entity_axis": "source",
                        "tc_im_value": left_value,
                        "tc_pt_value": right_value,
                        "delta_tc_im_minus_tc_pt": np.nan if delta is None else float(delta),
                        "contrast_direction": _comparison_direction(delta),
                        "comparison_status": comparison_status,
                        "tc_im_fit_status": left_status,
                        "tc_pt_fit_status": right_status,
                        "tc_im_defer_reason": left_reason,
                        "tc_pt_defer_reason": right_reason,
                        "tc_im_source_burden": np.nan if left_row is None else float(left_row["source_burden"]),
                        "tc_pt_source_burden": np.nan if right_row is None else float(right_row["source_burden"]),
                        "tc_im_source_weight": np.nan if left_row is None else float(left_row["source_weight"]),
                        "tc_pt_source_weight": np.nan if right_row is None else float(right_row["source_weight"]),
                        "comparison_scope_role": EXPLORATORY_SUPPORTIVE_SCOPE_ROLE,
                    }
                )

    return _finalize_frame(
        pd.DataFrame.from_records(records),
        columns=SOURCE_COMMUNITY_COMPARISON_COLUMNS,
        sort_columns=("patient_id", "source_community_id", "summary_name"),
    )


def _build_target_community_comparison_frame(
    *,
    dry_run_lookup: Mapping[tuple[str, str], dict[str, Any]],
    target_summary_df: pd.DataFrame,
    patient_ids: Sequence[str],
) -> pd.DataFrame:
    summary_lookup = _target_row_lookup(target_summary_df)
    communities_by_patient: dict[str, set[int]] = {patient_id: set() for patient_id in patient_ids}
    for patient_id, _pair_family, target_community_id in summary_lookup:
        communities_by_patient.setdefault(str(patient_id), set()).add(int(target_community_id))

    records: list[dict[str, Any]] = []
    for patient_id in patient_ids:
        left_status, left_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=LEFT_PAIR_FAMILY,
        )
        right_status, right_reason = _fit_status_details(
            dry_run_lookup,
            patient_id=patient_id,
            pair_family=RIGHT_PAIR_FAMILY,
        )
        for target_community_id in sorted(communities_by_patient.get(patient_id, set())):
            left_row = summary_lookup.get((patient_id, LEFT_PAIR_FAMILY, target_community_id))
            right_row = summary_lookup.get((patient_id, RIGHT_PAIR_FAMILY, target_community_id))
            _validate_row_against_status(
                row=left_row,
                fit_status=left_status,
                label=f"target-community comparison {LEFT_PAIR_FAMILY}/{patient_id}/{target_community_id}",
                allow_missing_ok_row=True,
            )
            _validate_row_against_status(
                row=right_row,
                fit_status=right_status,
                label=f"target-community comparison {RIGHT_PAIR_FAMILY}/{patient_id}/{target_community_id}",
                allow_missing_ok_row=True,
            )

            for summary_name in TARGET_COMMUNITY_SUMMARY_NAMES:
                comparison_status = "missing"
                delta: float | None = None
                left_value = np.nan
                right_value = np.nan
                if left_status in DEFERRED_FIT_STATUSES or right_status in DEFERRED_FIT_STATUSES:
                    comparison_status = "deferred"
                elif left_row is not None and right_row is not None:
                    comparison_status = "estimable"
                    left_value = float(left_row[summary_name])
                    right_value = float(right_row[summary_name])
                    delta = left_value - right_value
                elif left_row is not None:
                    left_value = float(left_row[summary_name])
                elif right_row is not None:
                    right_value = float(right_row[summary_name])

                records.append(
                    {
                        "patient_id": patient_id,
                        "pair_family_left": LEFT_PAIR_FAMILY,
                        "pair_family_right": RIGHT_PAIR_FAMILY,
                        "target_community_id": target_community_id,
                        "summary_name": summary_name,
                        "summary_role": TARGET_COMMUNITY_SUMMARY_ROLE_BY_NAME[summary_name],
                        "eligible_entity_axis": "target",
                        "tc_im_value": left_value,
                        "tc_pt_value": right_value,
                        "delta_tc_im_minus_tc_pt": np.nan if delta is None else float(delta),
                        "contrast_direction": _comparison_direction(delta),
                        "comparison_status": comparison_status,
                        "tc_im_fit_status": left_status,
                        "tc_pt_fit_status": right_status,
                        "tc_im_defer_reason": left_reason,
                        "tc_pt_defer_reason": right_reason,
                        "tc_im_target_burden": np.nan if left_row is None else float(left_row["target_burden"]),
                        "tc_pt_target_burden": np.nan if right_row is None else float(right_row["target_burden"]),
                        "tc_im_target_weight": np.nan if left_row is None else float(left_row["target_weight"]),
                        "tc_pt_target_weight": np.nan if right_row is None else float(right_row["target_weight"]),
                        "comparison_scope_role": EXPLORATORY_SUPPORTIVE_SCOPE_ROLE,
                    }
                )

    return _finalize_frame(
        pd.DataFrame.from_records(records),
        columns=TARGET_COMMUNITY_COMPARISON_COLUMNS,
        sort_columns=("patient_id", "target_community_id", "summary_name"),
    )


def build_block1_comparison_frames(
    *,
    dry_run_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    target_summary_df: pd.DataFrame,
    patient_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dry_run_lookup = _fit_status_records(dry_run_df)

    if patient_ids is None:
        discovered_patient_ids = {
            str(patient_id)
            for patient_id, _pair_family in dry_run_lookup
        }
        if not family_summary_df.empty:
            discovered_patient_ids.update(family_summary_df["patient_id"].astype(str).tolist())
        if not source_summary_df.empty:
            discovered_patient_ids.update(source_summary_df["patient_id"].astype(str).tolist())
        if not target_summary_df.empty:
            discovered_patient_ids.update(target_summary_df["patient_id"].astype(str).tolist())
        resolved_patient_ids = tuple(sorted(discovered_patient_ids))
    else:
        resolved_patient_ids = tuple(dict.fromkeys(str(patient_id) for patient_id in patient_ids))

    family_frame = _build_family_comparison_frame(
        dry_run_lookup=dry_run_lookup,
        family_summary_df=family_summary_df,
        patient_ids=resolved_patient_ids,
    )
    source_frame = _build_source_community_comparison_frame(
        dry_run_lookup=dry_run_lookup,
        source_summary_df=source_summary_df,
        patient_ids=resolved_patient_ids,
    )
    target_frame = _build_target_community_comparison_frame(
        dry_run_lookup=dry_run_lookup,
        target_summary_df=target_summary_df,
        patient_ids=resolved_patient_ids,
    )
    return family_frame, source_frame, target_frame


__all__ = [
    "CONFIRMATORY_FAMILY_COMPARISON_FILENAME",
    "EXPLORATORY_SOURCE_COMMUNITY_COMPARISON_FILENAME",
    "EXPLORATORY_SUPPORTIVE_SCOPE_ROLE",
    "EXPLORATORY_TARGET_COMMUNITY_COMPARISON_FILENAME",
    "FAMILY_COMPARISON_COLUMNS",
    "PAIRED_COMPARISON_CONTRACT_VERSION",
    "SOURCE_COMMUNITY_COMPARISON_COLUMNS",
    "TARGET_COMMUNITY_COMPARISON_COLUMNS",
    "build_block1_comparison_frames",
]
