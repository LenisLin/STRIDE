"""Task-local Block 2 robustness helpers for Task A.

These helpers perturb the full-cohort Stage 0 surface, rebuild the frozen
Block 1 summary/comparison objects, and summarize whether the current Block 1
findings survive those perturbations. They keep all logic task-local and do not
change ``src/stride/`` semantics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stride.errors import ContractError

from ..block1.comparisons import build_block1_comparison_frames
from ..block1.summaries import build_block1_summary_frames
from ..config import TaskABlock2Config, TaskAConfigBundle
from ..workflows.stride_adapter import (
    describe_task_a_stage0_stride_mapping,
    load_task_a_dataset_handle,
    run_task_a_family_core_fit_dry_run,
)

try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    ad = None  # type: ignore[assignment]


ROUTE_PATIENT_SUBSAMPLE = "patient_subsample"
ROUTE_LEAVE_SOME_OUT = "leave_some_out"
ROUTE_SEED_RERUN = "seed_rerun"
ROUTE_ROI_DROP = "roi_drop_one_per_domain"

PRIMARY_ROUTE_GROUP = "primary"
SECONDARY_ROUTE_GROUP = "secondary"

SUMMARY_SCOPE_FAMILY = "family"
SUMMARY_SCOPE_SOURCE = "source_community"
SUMMARY_SCOPE_TARGET = "target_community"

REPLICATE_STATUS_EXECUTED = "executed"
REPLICATE_STATUS_FAILED = "failed"
REPLICATE_STATUS_PENDING = "pending"

DETAIL_ROBUST = "robust"
DETAIL_PARTIAL = "partial"
DETAIL_FAILURE = "failure"
DETAIL_NOT_ASSESSED = "not_assessed"

REPLICATE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "route_name",
    "route_group",
    "replicate_index",
    "replicate_label",
    "selection_seed",
    "patient_subset_json",
    "dropped_roi_ids_json",
    "route_note",
    "route_status",
    "failure_reason",
    "n_patients_retained",
    "n_rois_retained",
    "n_cells_retained",
)

ROUTE_ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "block",
    "route_name",
    "route_group",
    "finding_id",
    "summary_scope",
    "finding_priority",
    "summary_name",
    "scale",
    "community_id",
    "full_data_direction",
    "full_data_estimable_n",
    "full_data_support_n",
    "full_data_support_fraction",
    "full_data_median_delta",
    "full_data_rank",
    "full_data_tc_im_mode_top_target_1_id",
    "full_data_tc_pt_mode_top_target_1_id",
    "n_replicates_planned",
    "n_replicates_executed",
    "estimable_replicate_fraction",
    "direction_recovery_rate",
    "median_replicate_support_fraction",
    "median_replicate_delta",
    "median_replicate_rank",
    "replicate_rank_iqr",
    "tc_im_mode_recovery_rate",
    "tc_pt_mode_recovery_rate",
    "robustness_call",
)


@dataclass(frozen=True)
class RoutePlan:
    route_name: str
    route_group: str
    n_replicates: int
    description: str


def _require_anndata() -> None:
    if ad is None:
        raise ModuleNotFoundError("anndata is required to run Task A Block 2 robustness workflows")


def _resolve_route_group(route_name: str, *, primary_routes: tuple[str, ...]) -> str:
    return PRIMARY_ROUTE_GROUP if route_name in set(primary_routes) else SECONDARY_ROUTE_GROUP


def build_route_plans(block2_config: TaskABlock2Config) -> tuple[RoutePlan, ...]:
    plans = (
        RoutePlan(
            route_name=ROUTE_PATIENT_SUBSAMPLE,
            route_group=_resolve_route_group(
                ROUTE_PATIENT_SUBSAMPLE,
                primary_routes=block2_config.primary_routes,
            ),
            n_replicates=int(block2_config.patient_subsample_replicates),
            description="Repeated patient subsampling without replacement.",
        ),
        RoutePlan(
            route_name=ROUTE_LEAVE_SOME_OUT,
            route_group=_resolve_route_group(
                ROUTE_LEAVE_SOME_OUT,
                primary_routes=block2_config.primary_routes,
            ),
            n_replicates=int(block2_config.leave_some_out_replicates),
            description="Balanced leave-some-patients-out perturbations.",
        ),
        RoutePlan(
            route_name=ROUTE_SEED_RERUN,
            route_group=_resolve_route_group(
                ROUTE_SEED_RERUN,
                primary_routes=block2_config.primary_routes,
            ),
            n_replicates=int(block2_config.seed_rerun_replicates),
            description="Deterministic full-data reruns to audit seed sensitivity.",
        ),
        RoutePlan(
            route_name=ROUTE_ROI_DROP,
            route_group=_resolve_route_group(
                ROUTE_ROI_DROP,
                primary_routes=block2_config.primary_routes,
            ),
            n_replicates=int(block2_config.roi_drop_replicates),
            description="Drop one ROI per patient-domain stratum when scientifically supported.",
        ),
    )
    return tuple(plan for plan in plans if plan.n_replicates > 0)


def _load_stage0_adata(data_path: str | Path) -> Any:
    _require_anndata()
    return ad.read_h5ad(Path(data_path).expanduser().resolve())


def _filter_adata(
    adata: Any,
    *,
    patient_ids: tuple[str, ...],
    dropped_roi_ids: tuple[str, ...],
) -> Any:
    patient_mask = adata.obs["patient_id"].astype(str).isin(set(patient_ids)).to_numpy(dtype=bool)
    filtered = adata[patient_mask].copy()
    if not dropped_roi_ids:
        return filtered
    roi_mask = ~filtered.obs["roi_id"].astype(str).isin(set(dropped_roi_ids)).to_numpy(dtype=bool)
    return filtered[roi_mask].copy()


def _patient_domain_signature_frame(
    adata: Any,
    *,
    ordered_domains: tuple[str, ...],
) -> pd.DataFrame:
    roi_frame = (
        adata.obs.loc[:, ["patient_id", "compartment", "roi_id"]]
        .astype({"patient_id": str, "compartment": str, "roi_id": str})
        .drop_duplicates()
    )
    counts = (
        roi_frame.groupby(["patient_id", "compartment"], observed=False)["roi_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=list(ordered_domains), fill_value=0)
        .reset_index()
    )
    counts["signature"] = counts.loc[:, list(ordered_domains)].apply(
        lambda row: "|".join(f"{domain}:{int(row[domain])}" for domain in ordered_domains),
        axis=1,
    )
    return counts


def _resolve_subsample_size(
    n_patients: int,
    *,
    fraction: float,
    min_patients: int,
) -> int:
    if n_patients <= 1:
        return n_patients
    target = int(math.floor(float(n_patients) * fraction))
    target = max(int(min_patients), target)
    target = min(target, n_patients - 1)
    return max(1, target)


def _resolve_leave_out_count(
    n_patients: int,
    *,
    fraction: float,
    min_patients: int,
) -> int:
    if n_patients <= 1:
        return 0
    omit_count = int(round(float(n_patients) * fraction))
    omit_count = max(int(min_patients), omit_count)
    omit_count = min(omit_count, n_patients - 1)
    return max(1, omit_count)


def _allocate_stratum_counts(
    stratum_sizes: dict[str, int],
    *,
    target_n: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    if target_n <= 0:
        return {stratum: 0 for stratum in stratum_sizes}
    total = int(sum(stratum_sizes.values()))
    if total <= 0:
        return {stratum: 0 for stratum in stratum_sizes}

    floor_counts: dict[str, int] = {}
    fractional_parts: list[tuple[float, float, str]] = []
    for stratum, size in stratum_sizes.items():
        raw = float(target_n) * float(size) / float(total)
        floor_count = min(size, int(math.floor(raw)))
        floor_counts[stratum] = floor_count
        fractional_parts.append((raw - float(floor_count), float(rng.random()), stratum))

    remaining = int(target_n - sum(floor_counts.values()))
    for _fractional, _tie_breaker, stratum in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break
        if floor_counts[stratum] >= stratum_sizes[stratum]:
            continue
        floor_counts[stratum] += 1
        remaining -= 1

    if remaining > 0:
        for stratum in sorted(stratum_sizes):
            while remaining > 0 and floor_counts[stratum] < stratum_sizes[stratum]:
                floor_counts[stratum] += 1
                remaining -= 1
            if remaining <= 0:
                break
    return floor_counts


def sample_patient_subsets(
    signature_frame: pd.DataFrame,
    *,
    route_name: str,
    route_group: str,
    n_replicates: int,
    block2_config: TaskABlock2Config,
    omission_counts: dict[str, int] | None = None,
    stratum_omission_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    patient_ids = tuple(signature_frame["patient_id"].astype(str).tolist())
    n_patients = len(patient_ids)
    master_rng = np.random.default_rng(int(block2_config.master_seed))
    records: list[dict[str, Any]] = []

    for replicate_index in range(n_replicates):
        selection_seed = int(master_rng.integers(np.iinfo(np.int64).max))
        rng = np.random.default_rng(selection_seed)
        dropped_roi_ids: tuple[str, ...] = ()
        route_note = ""

        if route_name == ROUTE_PATIENT_SUBSAMPLE:
            target_n = _resolve_subsample_size(
                n_patients,
                fraction=block2_config.patient_subsample_fraction,
                min_patients=block2_config.patient_subsample_min_patients,
            )
            strata_to_patients = {
                str(signature): tuple(group["patient_id"].astype(str).tolist())
                for signature, group in signature_frame.groupby("signature", observed=False)
            }
            stratum_counts = _allocate_stratum_counts(
                {stratum: len(group) for stratum, group in strata_to_patients.items()},
                target_n=target_n,
                rng=rng,
            )
            selected: list[str] = []
            for stratum in sorted(strata_to_patients):
                candidates = np.asarray(strata_to_patients[stratum], dtype=object)
                if stratum_counts[stratum] <= 0:
                    continue
                chosen = rng.choice(candidates, size=stratum_counts[stratum], replace=False)
                selected.extend(str(value) for value in chosen.tolist())
            retained_patients = tuple(sorted(selected))
        elif route_name == ROUTE_LEAVE_SOME_OUT:
            omit_count = _resolve_leave_out_count(
                n_patients,
                fraction=block2_config.leave_some_out_fraction,
                min_patients=block2_config.leave_some_out_min_patients,
            )
            active_omission_counts = omission_counts if omission_counts is not None else {}
            active_stratum_omission_counts = (
                stratum_omission_counts if stratum_omission_counts is not None else {}
            )
            weights: list[float] = []
            for row in signature_frame.to_dict(orient="records"):
                patient_id = str(row["patient_id"])
                signature = str(row["signature"])
                patient_weight = 1.0 / (1.0 + float(active_omission_counts.get(patient_id, 0)))
                stratum_weight = 1.0 / (1.0 + float(active_stratum_omission_counts.get(signature, 0)))
                weights.append(patient_weight * stratum_weight)
            weight_array = np.asarray(weights, dtype=float)
            weight_array = weight_array / float(np.sum(weight_array, dtype=float))
            omitted = tuple(
                str(value)
                for value in rng.choice(
                    np.asarray(patient_ids, dtype=object),
                    size=omit_count,
                    replace=False,
                    p=weight_array,
                ).tolist()
            )
            retained_patients = tuple(sorted(set(patient_ids) - set(omitted)))
            for patient_id in omitted:
                active_omission_counts[patient_id] = int(active_omission_counts.get(patient_id, 0)) + 1
                signature = str(
                    signature_frame.loc[
                        signature_frame["patient_id"].astype(str) == patient_id,
                        "signature",
                    ].iloc[0]
                )
                active_stratum_omission_counts[signature] = (
                    int(active_stratum_omission_counts.get(signature, 0)) + 1
                )
        else:
            retained_patients = tuple(sorted(patient_ids))
            if route_name == ROUTE_SEED_RERUN:
                route_note = (
                    "Current Block 1 fitting path is effectively deterministic on the frozen "
                    "Stage 0 basis; this route audits workflow determinism rather than a new seed surface."
                )

        records.append(
            {
                "route_name": route_name,
                "route_group": route_group,
                "replicate_index": int(replicate_index),
                "replicate_label": f"{route_name}_{replicate_index:04d}",
                "selection_seed": selection_seed,
                "patient_subset_json": json.dumps(list(retained_patients)),
                "dropped_roi_ids_json": json.dumps(list(dropped_roi_ids)),
                "route_note": route_note,
            }
        )
    return pd.DataFrame.from_records(records)


def sample_roi_drop_replicates(
    adata: Any,
    *,
    route_group: str,
    n_replicates: int,
    block2_config: TaskABlock2Config,
) -> pd.DataFrame:
    patient_ids = tuple(sorted(adata.obs["patient_id"].astype(str).unique().tolist()))
    roi_frame = (
        adata.obs.loc[:, ["patient_id", "compartment", "roi_id"]]
        .astype({"patient_id": str, "compartment": str, "roi_id": str})
        .drop_duplicates()
    )
    eligible_groups = {
        (str(patient_id), str(compartment)): tuple(sorted(group["roi_id"].astype(str).tolist()))
        for (patient_id, compartment), group in roi_frame.groupby(
            ["patient_id", "compartment"],
            observed=False,
        )
        if int(group["roi_id"].nunique()) >= int(block2_config.roi_drop_min_rois_per_domain)
    }
    master_rng = np.random.default_rng(int(block2_config.master_seed) + 1)
    records: list[dict[str, Any]] = []
    for replicate_index in range(n_replicates):
        selection_seed = int(master_rng.integers(np.iinfo(np.int64).max))
        rng = np.random.default_rng(selection_seed)
        dropped: list[str] = []
        for _group_key, roi_ids in sorted(eligible_groups.items()):
            n_drop = min(len(roi_ids) - 1, int(block2_config.roi_drop_n_drop_per_domain))
            if n_drop <= 0:
                continue
            chosen = rng.choice(np.asarray(roi_ids, dtype=object), size=n_drop, replace=False)
            dropped.extend(str(value) for value in chosen.tolist())
        records.append(
            {
                "route_name": ROUTE_ROI_DROP,
                "route_group": route_group,
                "replicate_index": int(replicate_index),
                "replicate_label": f"{ROUTE_ROI_DROP}_{replicate_index:04d}",
                "selection_seed": selection_seed,
                "patient_subset_json": json.dumps(list(patient_ids)),
                "dropped_roi_ids_json": json.dumps(sorted(dropped)),
                "route_note": (
                    "No eligible patient-domain strata had enough ROIs to drop one ROI."
                    if not dropped
                    else "Dropped one ROI from each eligible patient-domain stratum."
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_replicate_manifest(
    adata: Any,
    *,
    config_bundle: TaskAConfigBundle,
) -> pd.DataFrame:
    signature_frame = _patient_domain_signature_frame(
        adata,
        ordered_domains=config_bundle.ordered_proxy.domains,
    )
    omission_counts: dict[str, int] = {}
    stratum_omission_counts: dict[str, int] = {}

    frames: list[pd.DataFrame] = []
    for plan in build_route_plans(config_bundle.block2):
        if plan.route_name == ROUTE_ROI_DROP:
            frames.append(
                sample_roi_drop_replicates(
                    adata,
                    route_group=plan.route_group,
                    n_replicates=plan.n_replicates,
                    block2_config=config_bundle.block2,
                )
            )
            continue
        frames.append(
            sample_patient_subsets(
                signature_frame,
                route_name=plan.route_name,
                route_group=plan.route_group,
                n_replicates=plan.n_replicates,
                block2_config=config_bundle.block2,
                omission_counts=omission_counts,
                stratum_omission_counts=stratum_omission_counts,
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(REPLICATE_MANIFEST_COLUMNS))
    manifest = pd.concat(frames, ignore_index=True)
    manifest["route_status"] = REPLICATE_STATUS_PENDING
    manifest["failure_reason"] = ""
    manifest["n_patients_retained"] = 0
    manifest["n_rois_retained"] = 0
    manifest["n_cells_retained"] = 0
    return manifest


def _empty_route_robustness_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ROUTE_ROBUSTNESS_COLUMNS))


def _comparison_support_counts(
    frame: pd.DataFrame,
    *,
    direction_sign: int,
) -> tuple[int, int, float, float]:
    estimable = frame.loc[frame["comparison_status"].astype(str) == "estimable"].copy()
    if estimable.empty:
        return 0, 0, float("nan"), float("nan")
    delta = estimable["delta_tc_im_minus_tc_pt"].astype(float)
    support_count = int((np.sign(delta.to_numpy(dtype=float)) == float(direction_sign)).sum())
    support_fraction = float(support_count / len(estimable))
    return len(estimable), support_count, support_fraction, float(delta.median())


def _mode_int(series: pd.Series) -> int | None:
    if series.empty:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    value_counts = clean.astype(int).value_counts()
    max_count = int(value_counts.max())
    tied = sorted(int(value) for value, count in value_counts.items() if int(count) == max_count)
    return tied[0] if tied else None


def _build_family_anchor_rows(full_family_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (summary_name, scale), group in full_family_df.groupby(["summary_name", "scale"], observed=False):
        estimable_n, support_n, support_fraction, median_delta = _comparison_support_counts(
            group,
            direction_sign=1 if float(group["delta_tc_im_minus_tc_pt"].median()) > 0.0 else -1,
        )
        if estimable_n <= 0 or not math.isfinite(median_delta) or np.isclose(median_delta, 0.0, atol=1.0e-12):
            continue
        direction_sign = 1 if median_delta > 0.0 else -1
        rows.append(
            {
                "finding_id": f"family::{summary_name}::{scale}",
                "summary_scope": SUMMARY_SCOPE_FAMILY,
                "finding_priority": (
                    "primary" if str(summary_name) in {"self_retention", "depletion"} else "secondary"
                ),
                "summary_name": str(summary_name),
                "scale": str(scale),
                "community_id": np.nan,
                "direction_sign": int(direction_sign),
                "full_data_direction": "tc_im_gt_tc_pt" if direction_sign > 0 else "tc_im_lt_tc_pt",
                "full_data_estimable_n": int(estimable_n),
                "full_data_support_n": int(support_n),
                "full_data_support_fraction": float(support_fraction),
                "full_data_median_delta": float(median_delta),
                "full_data_rank": np.nan,
                "full_data_tc_im_mode_top_target_1_id": np.nan,
                "full_data_tc_pt_mode_top_target_1_id": np.nan,
            }
        )
    return rows


def _build_community_anchor_rows(
    full_comparison_df: pd.DataFrame,
    *,
    scope: str,
    id_column: str,
    summary_names: tuple[str, ...],
    primary_communities: tuple[int, ...],
    secondary_communities: tuple[int, ...],
    full_source_summary_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configured_communities = tuple(dict.fromkeys(tuple(primary_communities) + tuple(secondary_communities)))
    for community_id in configured_communities:
        for summary_name in summary_names:
            group = full_comparison_df.loc[
                (full_comparison_df[id_column].astype(float) == float(community_id))
                & (full_comparison_df["summary_name"].astype(str) == str(summary_name))
            ].copy()
            if group.empty:
                continue
            raw_median = float(group.loc[group["comparison_status"].astype(str) == "estimable", "delta_tc_im_minus_tc_pt"].median())
            if not math.isfinite(raw_median) or np.isclose(raw_median, 0.0, atol=1.0e-12):
                continue
            direction_sign = 1 if raw_median > 0.0 else -1
            estimable_n, support_n, support_fraction, median_delta = _comparison_support_counts(
                group,
                direction_sign=direction_sign,
            )
            if estimable_n <= 0:
                continue
            finding_priority = "secondary"
            if int(community_id) in set(primary_communities) and not (
                scope == SUMMARY_SCOPE_TARGET and str(summary_name) == "emergence_tendency"
            ):
                finding_priority = "primary"
            row = {
                "finding_id": f"{scope}::{community_id}::{summary_name}",
                "summary_scope": scope,
                "finding_priority": finding_priority,
                "summary_name": str(summary_name),
                "scale": "community",
                "community_id": int(community_id),
                "direction_sign": int(direction_sign),
                "full_data_direction": "tc_im_gt_tc_pt" if direction_sign > 0 else "tc_im_lt_tc_pt",
                "full_data_estimable_n": int(estimable_n),
                "full_data_support_n": int(support_n),
                "full_data_support_fraction": float(support_fraction),
                "full_data_median_delta": float(median_delta),
                "full_data_rank": np.nan,
                "full_data_tc_im_mode_top_target_1_id": np.nan,
                "full_data_tc_pt_mode_top_target_1_id": np.nan,
            }
            if scope == SUMMARY_SCOPE_SOURCE and full_source_summary_df is not None:
                for pair_family, column_name in (
                    ("TC-IM", "full_data_tc_im_mode_top_target_1_id"),
                    ("TC-PT", "full_data_tc_pt_mode_top_target_1_id"),
                ):
                    full_mode = _mode_int(
                        full_source_summary_df.loc[
                            (full_source_summary_df["pair_family"].astype(str) == pair_family)
                            & (full_source_summary_df["source_community_id"].astype(float) == float(community_id)),
                            "top_target_1_id",
                        ]
                    )
                    row[column_name] = np.nan if full_mode is None else int(full_mode)
            rows.append(row)
    return rows


def build_anchor_findings(
    *,
    full_family_df: pd.DataFrame,
    full_source_comparison_df: pd.DataFrame,
    full_target_comparison_df: pd.DataFrame,
    full_source_summary_df: pd.DataFrame,
    block2_config: TaskABlock2Config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_build_family_anchor_rows(full_family_df))
    rows.extend(
        _build_community_anchor_rows(
            full_source_comparison_df,
            scope=SUMMARY_SCOPE_SOURCE,
            id_column="source_community_id",
            summary_names=("self_retention", "depletion"),
            primary_communities=block2_config.primary_source_communities,
            secondary_communities=block2_config.secondary_source_communities,
            full_source_summary_df=full_source_summary_df,
        )
    )
    rows.extend(
        _build_community_anchor_rows(
            full_target_comparison_df,
            scope=SUMMARY_SCOPE_TARGET,
            id_column="target_community_id",
            summary_names=("incoming_matched_operator", "emergence_tendency"),
            primary_communities=block2_config.primary_target_communities,
            secondary_communities=block2_config.secondary_target_communities,
            full_source_summary_df=None,
        )
    )
    anchor_df = pd.DataFrame.from_records(rows)
    if anchor_df.empty:
        return pd.DataFrame(
            columns=[
                "finding_id",
                "summary_scope",
                "finding_priority",
                "summary_name",
                "scale",
                "community_id",
                "direction_sign",
                "full_data_direction",
                "full_data_estimable_n",
                "full_data_support_n",
                "full_data_support_fraction",
                "full_data_median_delta",
                "full_data_rank",
                "full_data_tc_im_mode_top_target_1_id",
                "full_data_tc_pt_mode_top_target_1_id",
            ]
        )

    for scope in (SUMMARY_SCOPE_SOURCE, SUMMARY_SCOPE_TARGET):
        for summary_name, group in anchor_df.loc[
            anchor_df["summary_scope"].astype(str) == scope
        ].groupby("summary_name", observed=False):
            supportive_scores = (
                group["direction_sign"].astype(float) * group["full_data_median_delta"].astype(float)
            )
            ranks = supportive_scores.rank(method="min", ascending=False)
            anchor_df.loc[group.index, "full_data_rank"] = ranks.astype(float).to_numpy()
    return anchor_df.sort_values(
        ["summary_scope", "finding_priority", "summary_name", "scale", "community_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _community_rank_map(
    comparison_df: pd.DataFrame,
    *,
    scope: str,
    summary_name: str,
    anchor_subset: pd.DataFrame,
) -> dict[int, float]:
    if anchor_subset.empty:
        return {}
    id_column = "source_community_id" if scope == SUMMARY_SCOPE_SOURCE else "target_community_id"
    relevant_ids = set(int(value) for value in anchor_subset["community_id"].dropna().astype(int).tolist())
    estimable = comparison_df.loc[
        (comparison_df["summary_name"].astype(str) == str(summary_name))
        & (comparison_df["comparison_status"].astype(str) == "estimable")
        & (comparison_df[id_column].astype(float).isin({float(value) for value in relevant_ids}))
    ].copy()
    if estimable.empty:
        return {}
    medians = estimable.groupby(id_column, observed=False)["delta_tc_im_minus_tc_pt"].median().astype(float)
    score_map = {
        int(community_id): float(anchor_subset.loc[
            anchor_subset["community_id"].astype(float) == float(community_id),
            "direction_sign",
        ].iloc[0]) * float(median_delta)
        for community_id, median_delta in medians.items()
    }
    score_series = pd.Series(score_map, dtype=float)
    if score_series.empty:
        return {}
    ranks = score_series.rank(method="min", ascending=False)
    return {int(community_id): float(rank) for community_id, rank in ranks.items()}


def _source_mode_maps(source_summary_df: pd.DataFrame, *, community_ids: tuple[int, ...]) -> dict[str, dict[int, int | None]]:
    result: dict[str, dict[int, int | None]] = {"TC-IM": {}, "TC-PT": {}}
    for pair_family in result:
        family_df = source_summary_df.loc[source_summary_df["pair_family"].astype(str) == pair_family].copy()
        for community_id in community_ids:
            result[pair_family][community_id] = _mode_int(
                family_df.loc[
                    family_df["source_community_id"].astype(float) == float(community_id),
                    "top_target_1_id",
                ]
            )
    return result


def build_replicate_assessment_rows(
    *,
    anchor_df: pd.DataFrame,
    family_comparison_df: pd.DataFrame,
    source_comparison_df: pd.DataFrame,
    target_comparison_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    route_name: str,
    route_group: str,
    replicate_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_anchor = anchor_df.loc[anchor_df["summary_scope"].astype(str) == SUMMARY_SCOPE_SOURCE].copy()
    target_anchor = anchor_df.loc[anchor_df["summary_scope"].astype(str) == SUMMARY_SCOPE_TARGET].copy()
    source_rank_maps = {
        summary_name: _community_rank_map(
            source_comparison_df,
            scope=SUMMARY_SCOPE_SOURCE,
            summary_name=str(summary_name),
            anchor_subset=source_anchor.loc[source_anchor["summary_name"].astype(str) == str(summary_name)],
        )
        for summary_name in sorted(source_anchor["summary_name"].astype(str).unique().tolist())
    }
    target_rank_maps = {
        summary_name: _community_rank_map(
            target_comparison_df,
            scope=SUMMARY_SCOPE_TARGET,
            summary_name=str(summary_name),
            anchor_subset=target_anchor.loc[target_anchor["summary_name"].astype(str) == str(summary_name)],
        )
        for summary_name in sorted(target_anchor["summary_name"].astype(str).unique().tolist())
    }
    source_modes = _source_mode_maps(
        source_summary_df,
        community_ids=tuple(source_anchor["community_id"].dropna().astype(int).tolist()),
    )

    for finding in anchor_df.to_dict(orient="records"):
        scope = str(finding["summary_scope"])
        summary_name = str(finding["summary_name"])
        direction_sign = int(finding["direction_sign"])
        community_id = None if pd.isna(finding["community_id"]) else int(finding["community_id"])
        scale = str(finding["scale"])

        if scope == SUMMARY_SCOPE_FAMILY:
            frame = family_comparison_df.loc[
                (family_comparison_df["summary_name"].astype(str) == summary_name)
                & (family_comparison_df["scale"].astype(str) == scale)
            ].copy()
            replicate_rank = np.nan
            tc_im_mode_matches_full = np.nan
            tc_pt_mode_matches_full = np.nan
        elif scope == SUMMARY_SCOPE_SOURCE and community_id is not None:
            frame = source_comparison_df.loc[
                (source_comparison_df["source_community_id"].astype(float) == float(community_id))
                & (source_comparison_df["summary_name"].astype(str) == summary_name)
            ].copy()
            replicate_rank = float(source_rank_maps.get(summary_name, {}).get(community_id, np.nan))
            tc_im_mode = source_modes["TC-IM"].get(community_id)
            tc_pt_mode = source_modes["TC-PT"].get(community_id)
            full_tc_im = None if pd.isna(finding["full_data_tc_im_mode_top_target_1_id"]) else int(
                finding["full_data_tc_im_mode_top_target_1_id"]
            )
            full_tc_pt = None if pd.isna(finding["full_data_tc_pt_mode_top_target_1_id"]) else int(
                finding["full_data_tc_pt_mode_top_target_1_id"]
            )
            tc_im_mode_matches_full = (
                np.nan if full_tc_im is None or tc_im_mode is None else bool(tc_im_mode == full_tc_im)
            )
            tc_pt_mode_matches_full = (
                np.nan if full_tc_pt is None or tc_pt_mode is None else bool(tc_pt_mode == full_tc_pt)
            )
        elif scope == SUMMARY_SCOPE_TARGET and community_id is not None:
            frame = target_comparison_df.loc[
                (target_comparison_df["target_community_id"].astype(float) == float(community_id))
                & (target_comparison_df["summary_name"].astype(str) == summary_name)
            ].copy()
            replicate_rank = float(target_rank_maps.get(summary_name, {}).get(community_id, np.nan))
            tc_im_mode_matches_full = np.nan
            tc_pt_mode_matches_full = np.nan
        else:
            continue

        estimable_n, support_n, support_fraction, median_delta = _comparison_support_counts(
            frame,
            direction_sign=direction_sign,
        )
        replicate_support = bool(
            estimable_n > 0
            and math.isfinite(median_delta)
            and not np.isclose(median_delta, 0.0, atol=1.0e-12)
            and int(np.sign(median_delta)) == direction_sign
            and math.isfinite(support_fraction)
            and float(support_fraction) >= 0.5
        )
        rows.append(
            {
                "finding_id": str(finding["finding_id"]),
                "route_name": route_name,
                "route_group": route_group,
                "replicate_index": int(replicate_index),
                "summary_scope": scope,
                "finding_priority": str(finding["finding_priority"]),
                "summary_name": summary_name,
                "scale": scale,
                "community_id": np.nan if community_id is None else int(community_id),
                "estimable_patient_count": int(estimable_n),
                "support_patient_count": int(support_n),
                "support_patient_fraction": float(support_fraction) if math.isfinite(support_fraction) else np.nan,
                "median_delta": float(median_delta) if math.isfinite(median_delta) else np.nan,
                "replicate_supports_direction": bool(replicate_support),
                "replicate_rank": float(replicate_rank) if math.isfinite(replicate_rank) else np.nan,
                "tc_im_mode_matches_full": tc_im_mode_matches_full,
                "tc_pt_mode_matches_full": tc_pt_mode_matches_full,
            }
        )
    return rows


def run_block1_reestimate_for_block2(
    source: Any,
    *,
    config_bundle: TaskAConfigBundle,
    route_name: str,
    replicate_index: int,
    selection_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage0_mapping = describe_task_a_stage0_stride_mapping(
        source,
        config_bundle=config_bundle,
    )
    dry_run_df, dry_run_results = run_task_a_family_core_fit_dry_run(
        source,
        config_bundle=config_bundle,
        pair_families=config_bundle.ordered_proxy.confirmatory_pair_families,
        fit_metadata={
            "task_block": "block2_bounded_audit",
            "task_block2_route": route_name,
            "task_block2_replicate_index": int(replicate_index),
            "task_block2_selection_seed": int(selection_seed),
        },
    )
    family_summary_df, source_summary_df, target_summary_df = build_block1_summary_frames(
        fit_results=dry_run_results,
        pair_families=config_bundle.ordered_proxy.confirmatory_pair_families,
    )
    family_comparison_df, source_comparison_df, target_comparison_df = build_block1_comparison_frames(
        dry_run_df=dry_run_df,
        family_summary_df=family_summary_df,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
        patient_ids=stage0_mapping.patient_ids,
    )
    return family_comparison_df, source_comparison_df, target_comparison_df, source_summary_df


def _route_call(
    *,
    scope: str,
    direction_recovery_rate: float,
    estimable_replicate_fraction: float,
    median_replicate_support_fraction: float,
    block2_config: TaskABlock2Config,
) -> str:
    if not math.isfinite(direction_recovery_rate):
        return DETAIL_NOT_ASSESSED
    support_threshold = (
        block2_config.family_patient_support_threshold
        if scope == SUMMARY_SCOPE_FAMILY
        else block2_config.community_patient_support_threshold
    )
    estimable_threshold = (
        block2_config.family_estimable_fraction_threshold
        if scope == SUMMARY_SCOPE_FAMILY
        else block2_config.community_estimable_fraction_threshold
    )
    if (
        direction_recovery_rate >= block2_config.robust_direction_consistency_threshold
        and estimable_replicate_fraction >= estimable_threshold
        and math.isfinite(median_replicate_support_fraction)
        and median_replicate_support_fraction >= support_threshold
    ):
        return DETAIL_ROBUST
    if (
        direction_recovery_rate >= block2_config.partial_direction_consistency_threshold
        and estimable_replicate_fraction >= 0.5
    ):
        return DETAIL_PARTIAL
    return DETAIL_FAILURE


def summarize_assessments_by_route(
    *,
    anchor_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
    replicate_manifest_df: pd.DataFrame,
    block2_config: TaskABlock2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_counts = (
        replicate_manifest_df.groupby(["route_name", "route_group"], observed=False)["replicate_index"]
        .size()
        .rename("n_replicates_planned")
        .reset_index()
    )
    executed_counts = (
        replicate_manifest_df.loc[replicate_manifest_df["route_status"].astype(str) == REPLICATE_STATUS_EXECUTED]
        .groupby(["route_name", "route_group"], observed=False)["replicate_index"]
        .size()
        .rename("n_replicates_executed")
        .reset_index()
    )
    route_counts = route_counts.merge(
        executed_counts,
        on=["route_name", "route_group"],
        how="left",
    )
    route_counts["n_replicates_executed"] = route_counts["n_replicates_executed"].fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    if assessment_df.empty or anchor_df.empty:
        empty = _empty_route_robustness_df()
        return empty, empty, empty

    for route_record in route_counts.to_dict(orient="records"):
        route_name = str(route_record["route_name"])
        route_group = str(route_record["route_group"])
        route_assessment = assessment_df.loc[
            assessment_df["route_name"].astype(str) == route_name
        ].copy()
        for finding in anchor_df.to_dict(orient="records"):
            finding_rows = route_assessment.loc[
                route_assessment["finding_id"].astype(str) == str(finding["finding_id"])
            ].copy()
            executed_n = int(route_record["n_replicates_executed"])
            if executed_n <= 0:
                direction_recovery_rate = float("nan")
                estimable_fraction = float("nan")
                median_support_fraction = float("nan")
                median_delta = float("nan")
                median_rank = float("nan")
                rank_iqr = float("nan")
                tc_im_mode_recovery_rate = float("nan")
                tc_pt_mode_recovery_rate = float("nan")
                detail_call = DETAIL_NOT_ASSESSED
            else:
                estimable_fraction = float(
                    (finding_rows["estimable_patient_count"].astype(int) > 0).sum() / executed_n
                )
                direction_recovery_rate = float(
                    finding_rows["replicate_supports_direction"].astype(bool).sum() / executed_n
                )
                estimable_rows = finding_rows.loc[finding_rows["estimable_patient_count"].astype(int) > 0].copy()
                median_support_fraction = float(
                    estimable_rows["support_patient_fraction"].astype(float).median()
                ) if not estimable_rows.empty else float("nan")
                median_delta = float(
                    estimable_rows["median_delta"].astype(float).median()
                ) if not estimable_rows.empty else float("nan")
                rank_rows = finding_rows.loc[finding_rows["replicate_rank"].notna()].copy()
                median_rank = float(rank_rows["replicate_rank"].astype(float).median()) if not rank_rows.empty else float("nan")
                rank_iqr = float(
                    rank_rows["replicate_rank"].astype(float).quantile(0.75)
                    - rank_rows["replicate_rank"].astype(float).quantile(0.25)
                ) if not rank_rows.empty else float("nan")
                tc_im_mode_rows = finding_rows.loc[finding_rows["tc_im_mode_matches_full"].notna()].copy()
                tc_im_mode_recovery_rate = float(
                    tc_im_mode_rows["tc_im_mode_matches_full"].astype(bool).sum() / len(tc_im_mode_rows)
                ) if not tc_im_mode_rows.empty else float("nan")
                tc_pt_mode_rows = finding_rows.loc[finding_rows["tc_pt_mode_matches_full"].notna()].copy()
                tc_pt_mode_recovery_rate = float(
                    tc_pt_mode_rows["tc_pt_mode_matches_full"].astype(bool).sum() / len(tc_pt_mode_rows)
                ) if not tc_pt_mode_rows.empty else float("nan")
                detail_call = _route_call(
                    scope=str(finding["summary_scope"]),
                    direction_recovery_rate=direction_recovery_rate,
                    estimable_replicate_fraction=estimable_fraction,
                    median_replicate_support_fraction=median_support_fraction,
                    block2_config=block2_config,
                )
            rows.append(
                {
                    "block": "block2_bounded_audit",
                    "route_name": route_name,
                    "route_group": route_group,
                    "finding_id": str(finding["finding_id"]),
                    "summary_scope": str(finding["summary_scope"]),
                    "finding_priority": str(finding["finding_priority"]),
                    "summary_name": str(finding["summary_name"]),
                    "scale": str(finding["scale"]),
                    "community_id": finding["community_id"],
                    "full_data_direction": str(finding["full_data_direction"]),
                    "full_data_estimable_n": int(finding["full_data_estimable_n"]),
                    "full_data_support_n": int(finding["full_data_support_n"]),
                    "full_data_support_fraction": float(finding["full_data_support_fraction"]),
                    "full_data_median_delta": float(finding["full_data_median_delta"]),
                    "full_data_rank": finding["full_data_rank"],
                    "full_data_tc_im_mode_top_target_1_id": finding["full_data_tc_im_mode_top_target_1_id"],
                    "full_data_tc_pt_mode_top_target_1_id": finding["full_data_tc_pt_mode_top_target_1_id"],
                    "n_replicates_planned": int(route_record["n_replicates_planned"]),
                    "n_replicates_executed": int(route_record["n_replicates_executed"]),
                    "estimable_replicate_fraction": estimable_fraction,
                    "direction_recovery_rate": direction_recovery_rate,
                    "median_replicate_support_fraction": median_support_fraction,
                    "median_replicate_delta": median_delta,
                    "median_replicate_rank": median_rank,
                    "replicate_rank_iqr": rank_iqr,
                    "tc_im_mode_recovery_rate": tc_im_mode_recovery_rate,
                    "tc_pt_mode_recovery_rate": tc_pt_mode_recovery_rate,
                    "robustness_call": detail_call,
                }
            )

    summarized = pd.DataFrame.from_records(rows)
    if summarized.empty:
        empty = _empty_route_robustness_df()
        return empty, empty, empty

    family_df = summarized.loc[summarized["summary_scope"].astype(str) == SUMMARY_SCOPE_FAMILY].reset_index(drop=True)
    source_df = summarized.loc[summarized["summary_scope"].astype(str) == SUMMARY_SCOPE_SOURCE].reset_index(drop=True)
    target_df = summarized.loc[summarized["summary_scope"].astype(str) == SUMMARY_SCOPE_TARGET].reset_index(drop=True)
    return family_df, source_df, target_df


def build_block2_summary(
    *,
    family_robustness_df: pd.DataFrame,
    source_robustness_df: pd.DataFrame,
    target_robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    detailed = pd.concat(
        [family_robustness_df, source_robustness_df, target_robustness_df],
        ignore_index=True,
    )
    if detailed.empty:
        return pd.DataFrame(
            columns=[
                "block",
                "summary_scope",
                "finding_priority",
                "summary_name",
                "scale",
                "community_id",
                "full_data_direction",
                "full_data_support_fraction",
                "full_data_median_delta",
                "primary_routes_executed",
                "primary_routes_robust",
                "primary_routes_partial_or_better",
                "worst_direction_recovery_rate",
                "worst_estimable_replicate_fraction",
                "worst_median_replicate_support_fraction",
                "overall_robustness_call",
                "primary_route_names",
            ]
        )

    rows: list[dict[str, Any]] = []
    for finding_id, group in detailed.groupby("finding_id", observed=False):
        primary_rows = group.loc[group["route_group"].astype(str) == PRIMARY_ROUTE_GROUP].copy()
        if primary_rows.empty:
            overall_call = DETAIL_NOT_ASSESSED
            primary_routes_executed = 0
            robust_primary = 0
            partial_or_better = 0
            worst_direction = float("nan")
            worst_estimable = float("nan")
            worst_patient_support = float("nan")
            primary_route_names = ""
        else:
            calls = primary_rows["robustness_call"].astype(str).tolist()
            if calls and all(call == DETAIL_ROBUST for call in calls):
                overall_call = DETAIL_ROBUST
            elif any(call == DETAIL_FAILURE for call in calls):
                overall_call = DETAIL_FAILURE
            else:
                overall_call = DETAIL_PARTIAL
            primary_routes_executed = int(len(primary_rows))
            robust_primary = int((primary_rows["robustness_call"].astype(str) == DETAIL_ROBUST).sum())
            partial_or_better = int(
                primary_rows["robustness_call"].astype(str).isin({DETAIL_ROBUST, DETAIL_PARTIAL}).sum()
            )
            worst_direction = float(primary_rows["direction_recovery_rate"].astype(float).min())
            worst_estimable = float(primary_rows["estimable_replicate_fraction"].astype(float).min())
            worst_patient_support = float(
                primary_rows["median_replicate_support_fraction"].astype(float).min()
            )
            primary_route_names = "|".join(sorted(primary_rows["route_name"].astype(str).tolist()))

        exemplar = group.iloc[0]
        rows.append(
            {
                "block": "block2_bounded_audit",
                "summary_scope": str(exemplar["summary_scope"]),
                "finding_priority": str(exemplar["finding_priority"]),
                "summary_name": str(exemplar["summary_name"]),
                "scale": str(exemplar["scale"]),
                "community_id": exemplar["community_id"],
                "full_data_direction": str(exemplar["full_data_direction"]),
                "full_data_support_fraction": float(exemplar["full_data_support_fraction"]),
                "full_data_median_delta": float(exemplar["full_data_median_delta"]),
                "primary_routes_executed": int(primary_routes_executed),
                "primary_routes_robust": int(robust_primary),
                "primary_routes_partial_or_better": int(partial_or_better),
                "worst_direction_recovery_rate": worst_direction,
                "worst_estimable_replicate_fraction": worst_estimable,
                "worst_median_replicate_support_fraction": worst_patient_support,
                "overall_robustness_call": overall_call,
                "primary_route_names": primary_route_names,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["summary_scope", "finding_priority", "summary_name", "scale", "community_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


__all__ = [
    "DETAIL_FAILURE",
    "DETAIL_NOT_ASSESSED",
    "DETAIL_PARTIAL",
    "DETAIL_ROBUST",
    "PRIMARY_ROUTE_GROUP",
    "REPLICATE_STATUS_EXECUTED",
    "ROUTE_LEAVE_SOME_OUT",
    "ROUTE_PATIENT_SUBSAMPLE",
    "ROUTE_ROI_DROP",
    "ROUTE_SEED_RERUN",
    "REPLICATE_STATUS_PENDING",
    "SECONDARY_ROUTE_GROUP",
    "SUMMARY_SCOPE_FAMILY",
    "SUMMARY_SCOPE_SOURCE",
    "SUMMARY_SCOPE_TARGET",
    "build_anchor_findings",
    "build_block2_summary",
    "build_replicate_manifest",
    "build_route_plans",
    "build_replicate_assessment_rows",
    "run_block1_reestimate_for_block2",
    "summarize_assessments_by_route",
]
