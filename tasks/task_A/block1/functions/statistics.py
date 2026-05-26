"""Block 1 paired-patient statistical supplement builders."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from stride.errors import ContractError
from stride.outputs.fit_export import NativeRelationExport, PatientRelationRecord

from .schemas import (
    BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION,
    FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS,
    RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS,
    SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
    STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA,
    STATISTICAL_SUPPLEMENT_Q_ALPHA,
    TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
)


LEFT_PAIR_FAMILY = "TC-IM"
RIGHT_PAIR_FAMILY = "TC-PT"
WILCOXON_METHOD = "wilcoxon_signed_rank_two_sided"
SIGN_TEST_METHOD = "two_sided_binomial_sign_test"
BH_POLICY = "benjamini_hochberg_by_declared_surface"
ZERO_ATOL = 1.0e-12


def _comparison_direction(delta: float) -> str:
    if not np.isfinite(delta):
        return "not_estimable"
    if np.isclose(delta, 0.0, atol=ZERO_ATOL):
        return "tc_im_eq_tc_pt"
    if delta > 0.0:
        return "tc_im_gt_tc_pt"
    return "tc_im_lt_tc_pt"


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    q_values = [np.nan] * len(p_values)
    finite_indices = [idx for idx, value in enumerate(p_values) if np.isfinite(float(value))]
    if not finite_indices:
        return q_values
    ordered = sorted(finite_indices, key=lambda idx: float(p_values[idx]))
    m = float(len(finite_indices))
    running_min = 1.0
    for rank, idx in reversed(list(enumerate(ordered, start=1))):
        adjusted = float(p_values[idx]) * m / float(rank)
        running_min = min(running_min, adjusted)
        q_values[idx] = min(running_min, 1.0)
    return q_values


def _paired_test_payload(
    *,
    tc_im_values: Sequence[float],
    tc_pt_values: Sequence[float],
    deltas: Sequence[float],
) -> dict[str, object]:
    tc_im = np.asarray(tc_im_values, dtype=float)
    tc_pt = np.asarray(tc_pt_values, dtype=float)
    delta = np.asarray(deltas, dtype=float)
    estimable_mask = np.isfinite(tc_im) & np.isfinite(tc_pt) & np.isfinite(delta)
    tc_im = tc_im[estimable_mask]
    tc_pt = tc_pt[estimable_mask]
    delta = delta[estimable_mask]
    nonzero_delta = delta[np.abs(delta) > ZERO_ATOL]
    positive_n = int(np.sum(delta > ZERO_ATOL))
    negative_n = int(np.sum(delta < -ZERO_ATOL))
    zero_n = int(delta.size - positive_n - negative_n)

    wilcoxon_p = np.nan
    sign_p = np.nan
    if nonzero_delta.size > 0:
        wilcoxon_p = float(wilcoxon(nonzero_delta, alternative="two-sided").pvalue)
        sign_p = float(
            binomtest(
                positive_n,
                n=positive_n + negative_n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    median_delta = float(np.median(delta)) if delta.size else np.nan
    abs_median_delta = abs(median_delta) if np.isfinite(median_delta) else np.nan
    effect_floor_pass = bool(
        np.isfinite(abs_median_delta)
        and abs_median_delta >= STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA
    )
    return {
        "pair_family_left": LEFT_PAIR_FAMILY,
        "pair_family_right": RIGHT_PAIR_FAMILY,
        "n_patients": int(delta.size),
        "n_estimable": int(delta.size),
        "n_nonzero_delta": int(nonzero_delta.size),
        "support_positive_n": positive_n,
        "support_negative_n": negative_n,
        "support_zero_n": zero_n,
        "support_positive_fraction": positive_n / float(delta.size) if delta.size else np.nan,
        "support_negative_fraction": negative_n / float(delta.size) if delta.size else np.nan,
        "tc_im_median": float(np.median(tc_im)) if tc_im.size else np.nan,
        "tc_pt_median": float(np.median(tc_pt)) if tc_pt.size else np.nan,
        "median_delta": median_delta,
        "mean_delta": float(np.mean(delta)) if delta.size else np.nan,
        "abs_median_delta": abs_median_delta,
        "contrast_direction": _comparison_direction(median_delta),
        "wilcoxon_p_value": wilcoxon_p,
        "sign_test_p_value": sign_p,
        "q_alpha": STATISTICAL_SUPPLEMENT_Q_ALPHA,
        "effect_floor_abs_median_delta": STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA,
        "effect_floor_pass": effect_floor_pass,
    }


def _apply_q_values(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["bh_q_value"] = _benjamini_hochberg(output["wilcoxon_p_value"].astype(float).tolist())
    output["q_pass"] = output["bh_q_value"].astype(float) <= STATISTICAL_SUPPLEMENT_Q_ALPHA
    output["review_candidate"] = output["q_pass"].astype(bool) & output["effect_floor_pass"].astype(bool)
    return output


def _finalize(frame: pd.DataFrame, *, columns: Sequence[str], sort_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    return (
        frame.loc[:, list(columns)]
        .sort_values(list(sort_columns), kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )


def _estimable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[frame["comparison_status"].astype(str) == "estimable"].copy()


def _summary_group_rows(
    frame: pd.DataFrame,
    *,
    statistical_surface: str,
    group_columns: Sequence[str],
    bh_scope: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    estimable = _estimable_rows(frame)
    for key, group in estimable.groupby(list(group_columns), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_payload = dict(zip(group_columns, key_tuple))
        payload = _paired_test_payload(
            tc_im_values=group["tc_im_value"].astype(float).tolist(),
            tc_pt_values=group["tc_pt_value"].astype(float).tolist(),
            deltas=group["delta_tc_im_minus_tc_pt"].astype(float).tolist(),
        )
        first_row = group.iloc[0].to_dict()
        records.append(
            {
                "statistical_supplement_contract_version": BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION,
                "statistical_surface": statistical_surface,
                **key_payload,
                "summary_role": str(first_row.get("summary_role", "")),
                "eligible_entity_axis": str(first_row.get("eligible_entity_axis", "")),
                **payload,
                "statistical_test": WILCOXON_METHOD,
                "sign_test": SIGN_TEST_METHOD,
                "bh_scope": bh_scope,
                "multiple_testing_policy": BH_POLICY,
                "comparison_scope_role": str(first_row.get("comparison_scope_role", "")),
            }
        )
    return records


def build_family_statistical_supplement(family_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired-patient statistics over the Block 1 family comparison surface."""
    records = _summary_group_rows(
        family_comparison_df,
        statistical_surface="family",
        group_columns=("summary_name", "scale"),
        bh_scope="family_summary",
    )
    return _finalize(
        _apply_q_values(pd.DataFrame.from_records(records)),
        columns=FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS,
        sort_columns=("summary_name", "scale"),
    )


def build_source_community_statistical_supplement(source_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired-patient statistics over source-community comparison rows."""
    records = _summary_group_rows(
        source_comparison_df,
        statistical_surface="source_community",
        group_columns=("source_community_id", "summary_name"),
        bh_scope="source_community_summary",
    )
    return _finalize(
        _apply_q_values(pd.DataFrame.from_records(records)),
        columns=SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
        sort_columns=("source_community_id", "summary_name"),
    )


def build_target_community_statistical_supplement(target_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired-patient statistics over target-community comparison rows."""
    records = _summary_group_rows(
        target_comparison_df,
        statistical_surface="target_community",
        group_columns=("target_community_id", "summary_name"),
        bh_scope="target_community_summary",
    )
    return _finalize(
        _apply_q_values(pd.DataFrame.from_records(records)),
        columns=TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
        sort_columns=("target_community_id", "summary_name"),
    )


def _ok_patient_records(export: NativeRelationExport, *, label: str) -> dict[str, PatientRelationRecord]:
    records = {str(record.patient_id): record for record in export.patient_records if record.is_ok}
    if len(records) != len(export.patient_records):
        raise ContractError(f"Block 1 statistical supplement requires all {label} patient records to be ok")
    return records


def _require_pair_exports(native_exports: Mapping[str, NativeRelationExport]) -> tuple[NativeRelationExport, NativeRelationExport]:
    left_export = native_exports.get(LEFT_PAIR_FAMILY)
    right_export = native_exports.get(RIGHT_PAIR_FAMILY)
    if left_export is None or right_export is None:
        raise ContractError("Block 1 statistical supplement requires TC-IM and TC-PT native exports")
    if tuple(left_export.state_ids) != tuple(right_export.state_ids):
        raise ContractError("Block 1 statistical supplement requires shared state ids")
    return left_export, right_export


def _cohort_context_lookup(cohort_relation_comparison_df: pd.DataFrame) -> dict[tuple[str, int | None, int | None], dict[str, float]]:
    lookup: dict[tuple[str, int | None, int | None], dict[str, float]] = {}
    if cohort_relation_comparison_df.empty:
        return lookup
    for row in cohort_relation_comparison_df.to_dict(orient="records"):
        component = str(row["component"])
        source_value = row["source_community_id"]
        target_value = row["target_community_id"]
        source_id = None if pd.isna(source_value) else int(source_value)
        target_id = None if pd.isna(target_value) else int(target_value)
        if component == "template_A":
            key = ("A", source_id, target_id)
        elif component == "template_d":
            key = ("d", source_id, None)
        elif component == "template_e":
            key = ("e", None, target_id)
        else:
            continue
        lookup[key] = {
            "cohort_tc_im_value": float(row["tc_im_value"]),
            "cohort_tc_pt_value": float(row["tc_pt_value"]),
            "cohort_delta_tc_im_minus_tc_pt": float(row["delta_tc_im_minus_tc_pt"]),
            "cohort_comparison_scope_role": str(row["comparison_scope_role"]),
        }
    return lookup


def _patient_values(
    records: Mapping[str, PatientRelationRecord],
    patient_ids: Sequence[str],
    *,
    component: str,
    source_index: int | None = None,
    target_index: int | None = None,
) -> list[float]:
    values: list[float] = []
    for patient_id in patient_ids:
        record = records[patient_id]
        if component == "A":
            if source_index is None or target_index is None or record.A is None:
                raise ContractError("A relation-element statistics require source and target indices")
            values.append(float(record.A[source_index, target_index]))
        elif component == "d":
            if source_index is None or record.d is None:
                raise ContractError("d relation-element statistics require a source index")
            values.append(float(record.d[source_index]))
        elif component == "e":
            if target_index is None or record.e is None:
                raise ContractError("e relation-element statistics require a target index")
            values.append(float(record.e[target_index]))
        else:
            raise ContractError(f"Unexpected relation component {component!r}")
    return values


def _relation_record(
    *,
    component: str,
    relation_axis: str,
    state_ids: Sequence[int],
    source_index: int | None,
    target_index: int | None,
    patient_ids: Sequence[str],
    left_records: Mapping[str, PatientRelationRecord],
    right_records: Mapping[str, PatientRelationRecord],
    cohort_context: Mapping[tuple[str, int | None, int | None], Mapping[str, object]],
) -> dict[str, object]:
    source_id = None if source_index is None else int(state_ids[source_index])
    target_id = None if target_index is None else int(state_ids[target_index])
    tc_im_values = _patient_values(
        left_records,
        patient_ids,
        component=component,
        source_index=source_index,
        target_index=target_index,
    )
    tc_pt_values = _patient_values(
        right_records,
        patient_ids,
        component=component,
        source_index=source_index,
        target_index=target_index,
    )
    deltas = [left - right for left, right in zip(tc_im_values, tc_pt_values)]
    payload = _paired_test_payload(
        tc_im_values=tc_im_values,
        tc_pt_values=tc_pt_values,
        deltas=deltas,
    )
    context = dict(
        cohort_context.get(
            (component, source_id, target_id),
            {
                "cohort_tc_im_value": np.nan,
                "cohort_tc_pt_value": np.nan,
                "cohort_delta_tc_im_minus_tc_pt": np.nan,
                "cohort_comparison_scope_role": "",
            },
        )
    )
    return {
        "statistical_supplement_contract_version": BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION,
        "statistical_surface": "relation_element",
        "component": component,
        "relation_axis": relation_axis,
        "source_community_id": np.nan if source_id is None else source_id,
        "target_community_id": np.nan if target_id is None else target_id,
        **payload,
        "statistical_test": WILCOXON_METHOD,
        "sign_test": SIGN_TEST_METHOD,
        "bh_scope": f"relation_element_{component}",
        "multiple_testing_policy": BH_POLICY,
        **context,
    }


def build_relation_element_statistical_supplement(
    *,
    native_exports: Mapping[str, NativeRelationExport],
    cohort_relation_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build patient-supported paired statistics for native A/d/e elements."""
    left_export, right_export = _require_pair_exports(native_exports)
    left_records = _ok_patient_records(left_export, label=LEFT_PAIR_FAMILY)
    right_records = _ok_patient_records(right_export, label=RIGHT_PAIR_FAMILY)
    patient_ids = tuple(sorted(set(left_records) & set(right_records)))
    if tuple(sorted(left_records)) != patient_ids or tuple(sorted(right_records)) != patient_ids:
        raise ContractError("Block 1 relation-element statistics require matched ok patient ids")
    state_ids = tuple(int(state_id) for state_id in left_export.state_ids)
    cohort_context = _cohort_context_lookup(cohort_relation_comparison_df)

    records: list[dict[str, object]] = []
    for source_index in range(len(state_ids)):
        for target_index in range(len(state_ids)):
            records.append(
                _relation_record(
                    component="A",
                    relation_axis="source_target",
                    state_ids=state_ids,
                    source_index=source_index,
                    target_index=target_index,
                    patient_ids=patient_ids,
                    left_records=left_records,
                    right_records=right_records,
                    cohort_context=cohort_context,
                )
            )
    for source_index in range(len(state_ids)):
        records.append(
            _relation_record(
                component="d",
                relation_axis="source_open",
                state_ids=state_ids,
                source_index=source_index,
                target_index=None,
                patient_ids=patient_ids,
                left_records=left_records,
                right_records=right_records,
                cohort_context=cohort_context,
            )
        )
    for target_index in range(len(state_ids)):
        records.append(
            _relation_record(
                component="e",
                relation_axis="target_open",
                state_ids=state_ids,
                source_index=None,
                target_index=target_index,
                patient_ids=patient_ids,
                left_records=left_records,
                right_records=right_records,
                cohort_context=cohort_context,
            )
        )

    frame = pd.DataFrame.from_records(records)
    adjusted_frames = [_apply_q_values(group.copy()) for _name, group in frame.groupby("component", sort=False)]
    output = pd.concat(adjusted_frames, ignore_index=True) if adjusted_frames else frame
    return _finalize(
        output,
        columns=RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS,
        sort_columns=("component", "source_community_id", "target_community_id"),
    )


__all__ = [
    "BH_POLICY",
    "SIGN_TEST_METHOD",
    "WILCOXON_METHOD",
    "build_family_statistical_supplement",
    "build_relation_element_statistical_supplement",
    "build_source_community_statistical_supplement",
    "build_target_community_statistical_supplement",
]
