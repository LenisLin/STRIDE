"""Objective Block 2 review-surface writers for Task A packets."""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stride.errors import ContractError

from ..config import load_task_a_config_bundle
from .robustness import (
    DETAIL_FAILURE,
    DETAIL_NOT_ASSESSED,
    DETAIL_PARTIAL,
    DETAIL_ROBUST,
    ROUTE_LEAVE_SOME_OUT,
    ROUTE_PATIENT_SUBSAMPLE,
    ROUTE_ROI_DROP,
    ROUTE_SEED_RERUN,
    SUMMARY_SCOPE_FAMILY,
    SUMMARY_SCOPE_SOURCE,
    SUMMARY_SCOPE_TARGET,
    build_route_plans,
)


BLOCK2_OBJECTIVE_REVIEW_MANIFEST_FILENAME = "block2_objective_review_manifest.json"
BLOCK2_ARTIFACT_INDEX_FILENAME = "block2_artifact_index.csv"
BLOCK2_ROUTE_SUMMARY_FILENAME = "block2_route_summary.csv"
BLOCK2_PRIMARY_FINDING_REVIEW_FILENAME = "block2_primary_finding_review_table.csv"
BLOCK2_FAMILY_REVIEW_FILENAME = "block2_family_review_table.csv"
BLOCK2_SOURCE_PRIMARY_REVIEW_FILENAME = "block2_source_primary_review_table.csv"
BLOCK2_TARGET_PRIMARY_REVIEW_FILENAME = "block2_target_primary_review_table.csv"
BLOCK2_CALL_SEMANTICS_FILENAME = "block2_call_semantics.csv"
BLOCK2_REPRODUCIBILITY_METADATA_FILENAME = "block2_reproducibility_metadata.json"
BLOCK2_HUMAN_INDEX_FILENAME = "BLOCK2_RESULTS_INDEX.md"
REVIEW_DIRNAME = "review"


@dataclass(frozen=True)
class Block2ReviewArtifact:
    artifact_name: str
    packet_relative_path: str
    source_path: str
    artifact_kind: str
    format: str
    claim_scope: str
    review_role: str
    artifact_evidence_class: str
    proof_carrying_status: str
    analysis_level: str
    family_surface_role: str
    robustness_routes: str
    rows_represent: str
    columns_represent: str
    source_workflow: str
    source_manifest_or_bundle: str
    notes: str
    review_rank: int


@dataclass(frozen=True)
class Block2ReviewSurface:
    manifest_path: Path
    artifact_index_path: Path
    human_index_path: Path
    artifacts: tuple[Block2ReviewArtifact, ...]


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_json_dict(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {resolved}")
    return payload


def _require_fields(payload: dict[str, Any], *, required_fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ContractError(f"{label} is missing required fields: {missing}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _csv_signature(path: Path) -> tuple[int, int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0, ""
        row_count = sum(1 for _ in reader)
    return row_count, len(header), "|".join(str(column) for column in header)


def _json_signature(path: Path) -> tuple[int | str, int | str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        keys = list(payload.keys())
        return 1, len(keys), "|".join(str(key) for key in keys)
    if isinstance(payload, list):
        keys: list[str] = []
        if payload and isinstance(payload[0], dict):
            keys = list(payload[0].keys())
        return len(payload), len(keys), "|".join(str(key) for key in keys)
    return "", "", ""


def _observe_artifact(path: Path) -> tuple[int | str, int | str, str]:
    if path.suffix.lower() == ".csv":
        return _csv_signature(path)
    if path.suffix.lower() == ".json":
        return _json_signature(path)
    return "", "", ""


def _route_parameter_summary(route_name: str, *, block2_config: Any) -> str:
    if route_name == ROUTE_PATIENT_SUBSAMPLE:
        return (
            f"retain_fraction={block2_config.patient_subsample_fraction};"
            f" min_patients={block2_config.patient_subsample_min_patients}"
        )
    if route_name == ROUTE_LEAVE_SOME_OUT:
        return (
            f"omit_fraction={block2_config.leave_some_out_fraction};"
            f" min_patients={block2_config.leave_some_out_min_patients}"
        )
    if route_name == ROUTE_SEED_RERUN:
        return f"configured_replicates={block2_config.seed_rerun_replicates}"
    if route_name == ROUTE_ROI_DROP:
        return (
            f"min_rois_per_domain={block2_config.roi_drop_min_rois_per_domain};"
            f" n_drop_per_domain={block2_config.roi_drop_n_drop_per_domain}"
        )
    return ""


def _route_description_map(config_bundle: Any) -> dict[str, str]:
    return {plan.route_name: plan.description for plan in build_route_plans(config_bundle.block2)}


def _configured_replicate_map(config_bundle: Any) -> dict[str, int]:
    return {plan.route_name: int(plan.n_replicates) for plan in build_route_plans(config_bundle.block2)}


def _all_route_names(
    *,
    replicate_manifest_df: pd.DataFrame,
    config_bundle: Any,
) -> tuple[str, ...]:
    ordered_names = [plan.route_name for plan in build_route_plans(config_bundle.block2)]
    observed_names = replicate_manifest_df.get("route_name", pd.Series(dtype="object")).astype(str).tolist()
    return tuple(dict.fromkeys([*ordered_names, *observed_names]))


def _route_status_summary(*, planned: int, executed: int, failed: int, pending: int) -> str:
    if planned <= 0:
        return "not_planned"
    if executed == planned and failed == 0 and pending == 0:
        return "all_executed"
    if failed > 0 and executed > 0:
        return "partially_failed"
    if failed > 0 and executed == 0:
        return "all_failed"
    if pending > 0 and executed > 0:
        return "partially_pending"
    if pending == planned:
        return "all_pending"
    return "mixed"


def _analysis_level_for_scope(scope: str) -> str:
    if scope == SUMMARY_SCOPE_FAMILY:
        return "family"
    if scope == SUMMARY_SCOPE_SOURCE:
        return "source_community"
    if scope == SUMMARY_SCOPE_TARGET:
        return "target_community"
    return "cohort_decision"


def _route_names_string(frame: pd.DataFrame) -> str:
    if frame.empty or "route_name" not in frame.columns:
        return ""
    return "|".join(sorted(frame["route_name"].astype(str).dropna().unique().tolist()))


def _copy_if_needed(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)


def _build_route_summary(
    *,
    replicate_manifest_df: pd.DataFrame,
    config_bundle: Any,
    primary_routes: tuple[str, ...],
) -> pd.DataFrame:
    description_map = _route_description_map(config_bundle)
    configured_map = _configured_replicate_map(config_bundle)
    rows: list[dict[str, Any]] = []

    for route_name in _all_route_names(replicate_manifest_df=replicate_manifest_df, config_bundle=config_bundle):
        group = replicate_manifest_df.loc[replicate_manifest_df["route_name"].astype(str) == route_name].copy()
        if group.empty:
            route_group = "primary" if route_name in set(primary_routes) else "secondary"
        else:
            route_group = str(group["route_group"].astype(str).iloc[0])
        planned = int(len(group))
        executed = int((group["route_status"].astype(str) == "executed").sum()) if not group.empty else 0
        failed = int((group["route_status"].astype(str) == "failed").sum()) if not group.empty else 0
        pending = int((group["route_status"].astype(str) == "pending").sum()) if not group.empty else 0
        executed_group = group.loc[group["route_status"].astype(str) == "executed"].copy()
        route_notes = "|".join(
            sorted(
                {
                    str(note).strip()
                    for note in group.get("route_note", pd.Series(dtype="object")).dropna().tolist()
                    if str(note).strip()
                }
            )
        )
        rows.append(
            {
                "route_name": route_name,
                "route_group": route_group,
                "included_in_overall_primary_call": bool(route_name in set(primary_routes)),
                "route_description": description_map.get(route_name, ""),
                "route_parameter_summary": _route_parameter_summary(
                    route_name,
                    block2_config=config_bundle.block2,
                ),
                "configured_replicates": int(configured_map.get(route_name, 0)),
                "planned_replicates": planned,
                "executed_replicates": executed,
                "failed_replicates": failed,
                "pending_replicates": pending,
                "route_status_summary": _route_status_summary(
                    planned=planned,
                    executed=executed,
                    failed=failed,
                    pending=pending,
                ),
                "unique_patient_subset_count": int(
                    group.get("patient_subset_json", pd.Series(dtype="object")).astype(str).nunique()
                )
                if not group.empty
                else 0,
                "unique_dropped_roi_pattern_count": int(
                    group.get("dropped_roi_ids_json", pd.Series(dtype="object")).astype(str).nunique()
                )
                if not group.empty
                else 0,
                "median_patients_retained": float(executed_group["n_patients_retained"].astype(float).median())
                if not executed_group.empty
                else float("nan"),
                "min_patients_retained": float(executed_group["n_patients_retained"].astype(float).min())
                if not executed_group.empty
                else float("nan"),
                "max_patients_retained": float(executed_group["n_patients_retained"].astype(float).max())
                if not executed_group.empty
                else float("nan"),
                "median_rois_retained": float(executed_group["n_rois_retained"].astype(float).median())
                if not executed_group.empty
                else float("nan"),
                "median_cells_retained": float(executed_group["n_cells_retained"].astype(float).median())
                if not executed_group.empty
                else float("nan"),
                "route_notes": route_notes,
            }
        )
    route_order = {name: idx for idx, name in enumerate(_all_route_names(replicate_manifest_df=replicate_manifest_df, config_bundle=config_bundle))}
    route_df = pd.DataFrame.from_records(rows)
    if route_df.empty:
        return pd.DataFrame(
            columns=[
                "route_name",
                "route_group",
                "included_in_overall_primary_call",
                "route_description",
                "route_parameter_summary",
                "configured_replicates",
                "planned_replicates",
                "executed_replicates",
                "failed_replicates",
                "pending_replicates",
                "route_status_summary",
                "unique_patient_subset_count",
                "unique_dropped_roi_pattern_count",
                "median_patients_retained",
                "min_patients_retained",
                "max_patients_retained",
                "median_rois_retained",
                "median_cells_retained",
                "route_notes",
            ]
        )
    return (
        route_df.assign(_route_order=route_df["route_name"].map(route_order))
        .sort_values(["route_group", "_route_order"], kind="mergesort")
        .drop(columns="_route_order")
        .reset_index(drop=True)
    )


def _build_primary_finding_review_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    detail_path_map = {
        SUMMARY_SCOPE_FAMILY: "block2/bundle/block2_family_robustness.csv",
        SUMMARY_SCOPE_SOURCE: "block2/bundle/block2_source_community_robustness.csv",
        SUMMARY_SCOPE_TARGET: "block2/bundle/block2_target_community_robustness.csv",
    }
    review_df = summary_df.copy()
    review_df["analysis_level"] = "cohort_decision"
    review_df["detail_artifact_path"] = review_df["summary_scope"].astype(str).map(detail_path_map).fillna("")
    ordered_columns = [
        "summary_scope",
        "analysis_level",
        "finding_priority",
        "summary_name",
        "scale",
        "community_id",
        "full_data_direction",
        "full_data_support_fraction",
        "full_data_median_delta",
        "primary_route_names",
        "primary_routes_executed",
        "primary_routes_robust",
        "primary_routes_partial_or_better",
        "worst_direction_recovery_rate",
        "worst_estimable_replicate_fraction",
        "worst_median_replicate_support_fraction",
        "overall_robustness_call",
        "detail_artifact_path",
    ]
    return review_df.loc[:, ordered_columns].sort_values(
        ["summary_scope", "finding_priority", "summary_name", "scale", "community_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _build_family_review_table(
    family_df: pd.DataFrame,
    *,
    route_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    route_meta = route_summary_df.loc[:, ["route_name", "route_description", "included_in_overall_primary_call"]]
    review_df = family_df.merge(route_meta, on="route_name", how="left")
    ordered_columns = [
        "route_name",
        "route_group",
        "included_in_overall_primary_call",
        "route_description",
        "finding_priority",
        "summary_name",
        "scale",
        "full_data_direction",
        "full_data_support_fraction",
        "full_data_median_delta",
        "n_replicates_planned",
        "n_replicates_executed",
        "estimable_replicate_fraction",
        "direction_recovery_rate",
        "median_replicate_support_fraction",
        "median_replicate_delta",
        "robustness_call",
    ]
    return review_df.loc[:, ordered_columns].sort_values(
        ["route_group", "route_name", "finding_priority", "summary_name", "scale"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_source_primary_review_table(
    source_df: pd.DataFrame,
    *,
    route_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    route_meta = route_summary_df.loc[:, ["route_name", "route_description", "included_in_overall_primary_call"]]
    review_df = source_df.loc[source_df["finding_priority"].astype(str) == "primary"].copy().merge(
        route_meta,
        on="route_name",
        how="left",
    )
    ordered_columns = [
        "route_name",
        "route_group",
        "included_in_overall_primary_call",
        "route_description",
        "community_id",
        "summary_name",
        "full_data_direction",
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
    ]
    return review_df.loc[:, ordered_columns].sort_values(
        ["route_group", "route_name", "summary_name", "community_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_target_primary_review_table(
    target_df: pd.DataFrame,
    *,
    route_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    route_meta = route_summary_df.loc[:, ["route_name", "route_description", "included_in_overall_primary_call"]]
    review_df = target_df.loc[target_df["finding_priority"].astype(str) == "primary"].copy().merge(
        route_meta,
        on="route_name",
        how="left",
    )
    ordered_columns = [
        "route_name",
        "route_group",
        "included_in_overall_primary_call",
        "route_description",
        "community_id",
        "summary_name",
        "full_data_direction",
        "full_data_support_fraction",
        "full_data_median_delta",
        "full_data_rank",
        "n_replicates_planned",
        "n_replicates_executed",
        "estimable_replicate_fraction",
        "direction_recovery_rate",
        "median_replicate_support_fraction",
        "median_replicate_delta",
        "median_replicate_rank",
        "replicate_rank_iqr",
        "robustness_call",
    ]
    return review_df.loc[:, ordered_columns].sort_values(
        ["route_group", "route_name", "summary_name", "community_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_call_semantics_table(config_bundle: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scope_specs = (
        (
            "family",
            "block2_family_robustness.csv",
            config_bundle.block2.family_estimable_fraction_threshold,
            config_bundle.block2.family_patient_support_threshold,
        ),
        (
            "source_community",
            "block2_source_community_robustness.csv",
            config_bundle.block2.community_estimable_fraction_threshold,
            config_bundle.block2.community_patient_support_threshold,
        ),
        (
            "target_community",
            "block2_target_community_robustness.csv",
            config_bundle.block2.community_estimable_fraction_threshold,
            config_bundle.block2.community_patient_support_threshold,
        ),
    )
    for analysis_level, surface_name, estimable_threshold, support_threshold in scope_specs:
        rows.extend(
            [
                {
                    "surface_name": surface_name,
                    "analysis_level": analysis_level,
                    "call_field": "robustness_call",
                    "call_value": DETAIL_ROBUST,
                    "meaning": "Route-level robustness-call outcome satisfied the robust thresholds.",
                    "rule_summary": (
                        f"direction_recovery_rate >= {config_bundle.block2.robust_direction_consistency_threshold}"
                        f"; estimable_replicate_fraction >= {estimable_threshold}"
                        f"; median_replicate_support_fraction >= {support_threshold}"
                    ),
                    "execution_failure_equivalent": False,
                },
                {
                    "surface_name": surface_name,
                    "analysis_level": analysis_level,
                    "call_field": "robustness_call",
                    "call_value": DETAIL_PARTIAL,
                    "meaning": "Route-level robustness-call outcome met the partial threshold but not the robust threshold.",
                    "rule_summary": (
                        f"direction_recovery_rate >= {config_bundle.block2.partial_direction_consistency_threshold}"
                        "; estimable_replicate_fraction >= 0.5"
                    ),
                    "execution_failure_equivalent": False,
                },
                {
                    "surface_name": surface_name,
                    "analysis_level": analysis_level,
                    "call_field": "robustness_call",
                    "call_value": DETAIL_FAILURE,
                    "meaning": (
                        "Route-level robustness-call outcome did not meet the partial criteria. "
                        "This is not an execution failure."
                    ),
                    "rule_summary": "Any route-level robustness call that is not robust, partial, or not_assessed.",
                    "execution_failure_equivalent": False,
                },
            ]
        )
    rows.extend(
        [
            {
                "surface_name": "block2_bounded_audit_summary.csv",
                "analysis_level": "cohort_decision",
                "call_field": "overall_robustness_call",
                "call_value": DETAIL_ROBUST,
                "meaning": "All primary routes returned route-level robustness_call = robust.",
                "rule_summary": "overall_robustness_call = robust iff every primary route is robust.",
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_bounded_audit_summary.csv",
                "analysis_level": "cohort_decision",
                "call_field": "overall_robustness_call",
                "call_value": DETAIL_PARTIAL,
                "meaning": "No primary route failed, but at least one primary route was not robust.",
                "rule_summary": (
                    "overall_robustness_call = partial iff there is no primary-route failure "
                    "and at least one primary route is partial or otherwise not robust."
                ),
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_bounded_audit_summary.csv",
                "analysis_level": "cohort_decision",
                "call_field": "overall_robustness_call",
                "call_value": DETAIL_FAILURE,
                "meaning": "At least one primary route returned route-level robustness_call = failure.",
                "rule_summary": "overall_robustness_call = failure iff any primary route fails.",
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_replicate_manifest.csv",
                "analysis_level": "replicate",
                "call_field": "route_status",
                "call_value": "executed",
                "meaning": "Replicate completed and contributed to route-level assessment rows.",
                "rule_summary": "Execution-state label for one perturbation replicate.",
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_replicate_manifest.csv",
                "analysis_level": "replicate",
                "call_field": "route_status",
                "call_value": "failed",
                "meaning": "Replicate execution failed before a usable assessment was retained.",
                "rule_summary": "Execution-state label for one perturbation replicate.",
                "execution_failure_equivalent": True,
            },
            {
                "surface_name": "block2_replicate_manifest.csv",
                "analysis_level": "replicate",
                "call_field": "route_status",
                "call_value": "pending",
                "meaning": "Replicate was planned but had not yet been executed when the manifest was observed.",
                "rule_summary": "Execution-state label for one perturbation replicate.",
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_source_community_robustness.csv | block2_target_community_robustness.csv",
                "analysis_level": "source_community|target_community",
                "call_field": "failure_note",
                "call_value": DETAIL_FAILURE,
                "meaning": (
                    "In source/target robustness tables, failure is a robustness-call outcome for a finding, "
                    "not a crashed run or missing file."
                ),
                "rule_summary": "Clarifying note for row-level robustness_call = failure.",
                "execution_failure_equivalent": False,
            },
            {
                "surface_name": "block2_family_robustness.csv | block2_source_community_robustness.csv | block2_target_community_robustness.csv",
                "analysis_level": "family|source_community|target_community",
                "call_field": "not_assessed_note",
                "call_value": DETAIL_NOT_ASSESSED,
                "meaning": "No executed route-level assessment was available for the finding.",
                "rule_summary": "Possible only when a route had zero executed replicates for the finding surface.",
                "execution_failure_equivalent": False,
            },
        ]
    )
    return pd.DataFrame.from_records(rows)


def _build_reproducibility_payload(
    *,
    block2_manifest_path: Path,
    block2_payload: dict[str, Any],
    block1_payload: dict[str, Any],
    config_bundle: Any,
    route_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    replicate_manifest_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "workflow_name": "write_block2_review_surface",
        "scientific_interpretation_allowed": False,
        "block2_manifest_path": str(block2_manifest_path),
        "block2_status": str(block2_payload.get("status", "")),
        "artifact_state": str(block2_payload.get("artifact_state", "")),
        "scientific_role": str(block2_payload.get("scientific_role", "")),
        "claim_scope": str(block2_payload.get("claim_scope", "")),
        "config_path": str(block2_payload.get("config_path", "")),
        "config_fingerprint": str(block2_payload.get("config_fingerprint", "")),
        "block1_bundle_path": str(block2_payload.get("block1_bundle_path", "")),
        "block0_bundle_path": str(block2_payload.get("block0_bundle_path", "")),
        "stage0_h5ad": str(block1_payload.get("stage0_h5ad", "")),
        "pair_families": list(block1_payload.get("pair_families", [])),
        "confirmatory_pair_families": list(block1_payload.get("confirmatory_pair_families", [])),
        "family_summary_scales": list(block1_payload.get("family_summary_scales", [])),
        "source_eligibility_rule": str(block1_payload.get("source_eligibility_rule", "")),
        "target_eligibility_rule": str(block1_payload.get("target_eligibility_rule", "")),
        "primary_routes": list(block2_payload.get("primary_routes", [])),
        "primary_source_communities": list(block2_payload.get("primary_source_communities", [])),
        "primary_target_communities": list(block2_payload.get("primary_target_communities", [])),
        "call_thresholds": {
            "robust_direction_consistency_threshold": config_bundle.block2.robust_direction_consistency_threshold,
            "partial_direction_consistency_threshold": config_bundle.block2.partial_direction_consistency_threshold,
            "family_estimable_fraction_threshold": config_bundle.block2.family_estimable_fraction_threshold,
            "family_patient_support_threshold": config_bundle.block2.family_patient_support_threshold,
            "community_estimable_fraction_threshold": config_bundle.block2.community_estimable_fraction_threshold,
            "community_patient_support_threshold": config_bundle.block2.community_patient_support_threshold,
        },
        "route_plan_summary": route_summary_df.to_dict(orient="records"),
        "observed_rows": {
            "block2_bounded_audit_summary": int(len(summary_df)),
            "block2_family_robustness": int(len(family_df)),
            "block2_source_community_robustness": int(len(source_df)),
            "block2_target_community_robustness": int(len(target_df)),
            "block2_replicate_manifest": int(len(replicate_manifest_df)),
        },
        "executed_replicates": int((replicate_manifest_df["route_status"].astype(str) == "executed").sum())
        if not replicate_manifest_df.empty
        else 0,
        "failed_replicates": int((replicate_manifest_df["route_status"].astype(str) == "failed").sum())
        if not replicate_manifest_df.empty
        else 0,
    }


def _human_index_lines(
    *,
    block2_payload: dict[str, Any],
    raw_artifacts: list[Block2ReviewArtifact],
    derived_artifacts: list[Block2ReviewArtifact],
) -> list[str]:
    lines = [
        "# Block 2 Objective Review Surface",
        "",
        "This surface repackages the live Block 2 outputs without biological interpretation.",
        "",
        "## Current Run",
        f"- Status: `{block2_payload.get('status', 'unknown')}`",
        f"- Artifact state: `{block2_payload.get('artifact_state', 'unknown')}`",
        f"- Primary routes: `{ '|'.join(str(value) for value in block2_payload.get('primary_routes', [])) }`",
        f"- Review-only packaging: `true`",
        "",
        "## Inspect First",
        "- `block2/review/block2_primary_finding_review_table.csv`",
        "- `block2/review/block2_family_review_table.csv`",
        "- `block2/review/block2_source_primary_review_table.csv`",
        "- `block2/review/block2_target_primary_review_table.csv`",
        "- `block2/review/block2_route_summary.csv`",
        "- `block2/review/block2_call_semantics.csv`",
        "- `block2/review/block2_artifact_index.csv`",
        "- `block2/bundle/block2_bounded_audit_summary.csv`",
        "",
        "## Raw Block 2 Artifacts",
    ]
    for artifact in sorted(raw_artifacts, key=lambda item: (item.review_rank, item.artifact_name)):
        lines.append(f"- `{artifact.packet_relative_path}`: {artifact.notes}")
    lines.extend(["", "## Derived Review Artifacts"])
    for artifact in sorted(derived_artifacts, key=lambda item: (item.review_rank, item.artifact_name)):
        lines.append(f"- `{artifact.packet_relative_path}`: {artifact.notes}")
    lines.extend(
        [
            "",
            "## Semantics Notes",
            "- `overall_robustness_call` is the primary-route aggregation surface, not a rerun status.",
            "- Row-level `robustness_call = failure` in source/target tables means a robustness-call outcome, not an execution crash.",
            "- Replicate execution failures appear only through `route_status` / `failure_reason` in `block2_replicate_manifest.csv`.",
        ]
    )
    return lines


def write_block2_review_surface(
    *,
    block2_manifest_path: str | Path,
    output_dir: str | Path,
) -> Block2ReviewSurface:
    block2_manifest = _resolve_path(block2_manifest_path)
    block2_payload = _load_json_dict(block2_manifest, label="Task A Block 2 manifest")
    _require_fields(
        block2_payload,
        required_fields=(
            "block1_bundle_path",
            "summary_path",
            "contract_path",
            "replicate_manifest_path",
            "family_robustness_path",
            "source_community_robustness_path",
            "target_community_robustness_path",
            "config_path",
        ),
        label="Task A Block 2 manifest",
    )

    block1_bundle_path = _resolve_path(block2_payload["block1_bundle_path"])
    block1_payload = _load_json_dict(block1_bundle_path, label="Task A Block 1 bundle")
    config_bundle = load_task_a_config_bundle(block2_payload["config_path"])

    block2_dir = _resolve_path(output_dir)
    bundle_dir = block2_dir / "bundle"
    review_dir = block2_dir / REVIEW_DIRNAME
    bundle_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(_resolve_path(block2_payload["summary_path"]))
    family_df = pd.read_csv(_resolve_path(block2_payload["family_robustness_path"]))
    source_df = pd.read_csv(_resolve_path(block2_payload["source_community_robustness_path"]))
    target_df = pd.read_csv(_resolve_path(block2_payload["target_community_robustness_path"]))
    replicate_manifest_df = pd.read_csv(_resolve_path(block2_payload["replicate_manifest_path"]))
    family_routes = _route_names_string(family_df)
    source_routes = _route_names_string(source_df)
    target_routes = _route_names_string(target_df)
    all_routes = _route_names_string(replicate_manifest_df)

    raw_specs = [
        {
            "artifact_name": "block2_bounded_audit_manifest.json",
            "source_path": block2_manifest,
            "packet_relative_path": "block2/bundle/block2_bounded_audit_manifest.json",
            "artifact_kind": "manifest",
            "format": "json",
            "claim_scope": "provenance",
            "review_role": "provenance",
            "artifact_evidence_class": "engineering_audit",
            "proof_carrying_status": "none",
            "analysis_level": "cohort_decision",
            "family_surface_role": "not_applicable",
            "robustness_routes": "|".join(str(value) for value in block2_payload.get("primary_routes", [])),
            "rows_represent": "Single JSON object for the canonical live Block 2 robustness manifest.",
            "columns_represent": "Top-level keys record raw Block 2 provenance, linked Block 1 inputs, linked Block 2 outputs, and configured primary routes.",
            "notes": "Canonical live Block 2 manifest mirrored without modification.",
            "review_rank": 10,
        },
        {
            "artifact_name": "block2_bounded_audit_summary.csv",
            "source_path": _resolve_path(block2_payload["summary_path"]),
            "packet_relative_path": "block2/bundle/block2_bounded_audit_summary.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "robustness",
            "review_role": "proof_carrying",
            "artifact_evidence_class": "proof_carrying",
            "proof_carrying_status": "partial",
            "analysis_level": "cohort_decision",
            "family_surface_role": "comparison_surface",
            "robustness_routes": "|".join(str(value) for value in block2_payload.get("primary_routes", [])),
            "rows_represent": "Rows represent one frozen Block 1 finding aggregated across the primary Block 2 routes.",
            "columns_represent": "Columns record the top-level overall_robustness_call plus the worst primary-route support quantities used for review.",
            "notes": "Top-level primary-route aggregation surface from the live Block 2 run.",
            "review_rank": 20,
        },
        {
            "artifact_name": "block2_family_robustness.csv",
            "source_path": _resolve_path(block2_payload["family_robustness_path"]),
            "packet_relative_path": "block2/bundle/block2_family_robustness.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "robustness",
            "review_role": "proof_carrying",
            "artifact_evidence_class": "proof_carrying",
            "proof_carrying_status": "all",
            "analysis_level": "family",
            "family_surface_role": "comparison_surface",
            "robustness_routes": family_routes,
            "rows_represent": "Rows represent route-level family robustness summaries for the frozen family findings.",
            "columns_represent": "Columns record route identity, direction recovery, estimability, patient support, and the route-level robustness_call.",
            "notes": "Proof-carrying family-level route summaries from the live Block 2 run.",
            "review_rank": 21,
        },
        {
            "artifact_name": "block2_source_community_robustness.csv",
            "source_path": _resolve_path(block2_payload["source_community_robustness_path"]),
            "packet_relative_path": "block2/bundle/block2_source_community_robustness.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "robustness",
            "review_role": "proof_carrying",
            "artifact_evidence_class": "proof_carrying",
            "proof_carrying_status": "partial",
            "analysis_level": "source_community",
            "family_surface_role": "not_applicable",
            "robustness_routes": source_routes,
            "rows_represent": "Rows represent route-level source-community robustness summaries for scoped Block 1 source findings.",
            "columns_represent": "Columns record community ids, direction recovery, estimability, patient support, rank stability, top-target mode recovery, and robustness_call.",
            "notes": "Route-level source-community robustness table from the live Block 2 run.",
            "review_rank": 22,
        },
        {
            "artifact_name": "block2_target_community_robustness.csv",
            "source_path": _resolve_path(block2_payload["target_community_robustness_path"]),
            "packet_relative_path": "block2/bundle/block2_target_community_robustness.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "robustness",
            "review_role": "proof_carrying",
            "artifact_evidence_class": "proof_carrying",
            "proof_carrying_status": "partial",
            "analysis_level": "target_community",
            "family_surface_role": "not_applicable",
            "robustness_routes": target_routes,
            "rows_represent": "Rows represent route-level target-community robustness summaries for scoped Block 1 target findings.",
            "columns_represent": "Columns record community ids, direction recovery, estimability, patient support, rank stability, and robustness_call.",
            "notes": "Route-level target-community robustness table from the live Block 2 run.",
            "review_rank": 23,
        },
        {
            "artifact_name": "block2_replicate_manifest.csv",
            "source_path": _resolve_path(block2_payload["replicate_manifest_path"]),
            "packet_relative_path": "block2/bundle/block2_replicate_manifest.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "provenance",
            "review_role": "provenance",
            "artifact_evidence_class": "engineering_audit",
            "proof_carrying_status": "none",
            "analysis_level": "replicate",
            "family_surface_role": "not_applicable",
            "robustness_routes": all_routes,
            "rows_represent": "Rows represent one attempted Block 2 perturbation replicate.",
            "columns_represent": "Columns record route identity, retained cohort sizes, perturbation membership, and replicate execution status/failure metadata.",
            "notes": "Replicate-level execution manifest from the live Block 2 run.",
            "review_rank": 24,
        },
        {
            "artifact_name": "block2_contract_audit.csv",
            "source_path": _resolve_path(block2_payload["contract_path"]),
            "packet_relative_path": "block2/bundle/block2_contract_audit.csv",
            "artifact_kind": "table",
            "format": "csv",
            "claim_scope": "provenance",
            "review_role": "provenance",
            "artifact_evidence_class": "engineering_audit",
            "proof_carrying_status": "none",
            "analysis_level": "cohort_decision",
            "family_surface_role": "not_applicable",
            "robustness_routes": all_routes,
            "rows_represent": "Rows represent one provenance or contract check for the Block 2 run.",
            "columns_represent": "Columns record the check name, pass/fail state, and supporting detail path or value.",
            "notes": "Engineering-audit contract checks from the live Block 2 run.",
            "review_rank": 25,
        },
    ]

    raw_artifacts: list[Block2ReviewArtifact] = []
    for spec in raw_specs:
        source_path = _resolve_path(spec["source_path"])
        destination = block2_dir.parent / spec["packet_relative_path"]
        _copy_if_needed(source=source_path, destination=destination)
        raw_artifacts.append(
            Block2ReviewArtifact(
                artifact_name=str(spec["artifact_name"]),
                packet_relative_path=str(spec["packet_relative_path"]),
                source_path=str(source_path),
                artifact_kind=str(spec["artifact_kind"]),
                format=str(spec["format"]),
                claim_scope=str(spec["claim_scope"]),
                review_role=str(spec["review_role"]),
                artifact_evidence_class=str(spec["artifact_evidence_class"]),
                proof_carrying_status=str(spec["proof_carrying_status"]),
                analysis_level=str(spec["analysis_level"]),
                family_surface_role=str(spec["family_surface_role"]),
                robustness_routes=str(spec["robustness_routes"]),
                rows_represent=str(spec["rows_represent"]),
                columns_represent=str(spec["columns_represent"]),
                source_workflow="write_block2_bundle",
                source_manifest_or_bundle=str(block2_manifest),
                notes=str(spec["notes"]),
                review_rank=int(spec["review_rank"]),
            )
        )

    route_summary_df = _build_route_summary(
        replicate_manifest_df=replicate_manifest_df,
        config_bundle=config_bundle,
        primary_routes=tuple(str(value) for value in block2_payload.get("primary_routes", [])),
    )
    primary_finding_review_df = _build_primary_finding_review_table(summary_df)
    family_review_df = _build_family_review_table(family_df, route_summary_df=route_summary_df)
    source_primary_review_df = _build_source_primary_review_table(source_df, route_summary_df=route_summary_df)
    target_primary_review_df = _build_target_primary_review_table(target_df, route_summary_df=route_summary_df)
    call_semantics_df = _build_call_semantics_table(config_bundle)
    reproducibility_payload = _build_reproducibility_payload(
        block2_manifest_path=block2_manifest,
        block2_payload=block2_payload,
        block1_payload=block1_payload,
        config_bundle=config_bundle,
        route_summary_df=route_summary_df,
        summary_df=summary_df,
        family_df=family_df,
        source_df=source_df,
        target_df=target_df,
        replicate_manifest_df=replicate_manifest_df,
    )

    route_summary_path = review_dir / BLOCK2_ROUTE_SUMMARY_FILENAME
    primary_finding_review_path = review_dir / BLOCK2_PRIMARY_FINDING_REVIEW_FILENAME
    family_review_path = review_dir / BLOCK2_FAMILY_REVIEW_FILENAME
    source_primary_review_path = review_dir / BLOCK2_SOURCE_PRIMARY_REVIEW_FILENAME
    target_primary_review_path = review_dir / BLOCK2_TARGET_PRIMARY_REVIEW_FILENAME
    call_semantics_path = review_dir / BLOCK2_CALL_SEMANTICS_FILENAME
    reproducibility_path = review_dir / BLOCK2_REPRODUCIBILITY_METADATA_FILENAME

    _write_csv(route_summary_path, route_summary_df)
    _write_csv(primary_finding_review_path, primary_finding_review_df)
    _write_csv(family_review_path, family_review_df)
    _write_csv(source_primary_review_path, source_primary_review_df)
    _write_csv(target_primary_review_path, target_primary_review_df)
    _write_csv(call_semantics_path, call_semantics_df)
    _write_json(reproducibility_path, reproducibility_payload)

    derived_artifacts: list[Block2ReviewArtifact] = [
        Block2ReviewArtifact(
            artifact_name=BLOCK2_ROUTE_SUMMARY_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_ROUTE_SUMMARY_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="replicate",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Rows represent one configured Block 2 robustness route aggregated over attempted replicates.",
            columns_represent="Columns record route description, replicate counts, retained cohort sizes, route-note text, and whether the route contributes to the overall primary-call surface.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Review-only route summary table faithful to the live replicate manifest and Task A config.",
            review_rank=30,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_PRIMARY_FINDING_REVIEW_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_PRIMARY_FINDING_REVIEW_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="robustness",
            review_role="proof_carrying",
            artifact_evidence_class="proof_carrying",
            proof_carrying_status="partial",
            analysis_level="cohort_decision",
            family_surface_role="comparison_surface",
            robustness_routes="|".join(str(value) for value in block2_payload.get("primary_routes", [])),
            rows_represent="Rows represent one top-level Block 2 finding decision from block2_bounded_audit_summary.csv.",
            columns_represent="Columns restate the cohort-level overall robustness call, worst primary-route support quantities, and the raw detail artifact path for follow-up review.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Review-only reorganization of the top-level Block 2 summary surface.",
            review_rank=31,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_FAMILY_REVIEW_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_FAMILY_REVIEW_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="robustness",
            review_role="proof_carrying",
            artifact_evidence_class="proof_carrying",
            proof_carrying_status="all",
            analysis_level="family",
            family_surface_role="comparison_surface",
            robustness_routes=family_routes,
            rows_represent="Rows represent one route-level family robustness row reorganized for review.",
            columns_represent="Columns foreground route identity, whether the route contributes to the top-level call, the full-data direction, and the main route-level support metrics.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Review-only family table faithful to block2_family_robustness.csv.",
            review_rank=32,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_SOURCE_PRIMARY_REVIEW_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_SOURCE_PRIMARY_REVIEW_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="robustness",
            review_role="proof_carrying",
            artifact_evidence_class="proof_carrying",
            proof_carrying_status="all",
            analysis_level="source_community",
            family_surface_role="not_applicable",
            robustness_routes=source_routes,
            rows_represent="Rows represent the primary source-community route-level robustness findings only.",
            columns_represent="Columns foreground community ids, full-data ranks, top-target mode recovery, and route-level support metrics without changing the scientific contents.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Primary-only source-community review table filtered from the live source robustness surface.",
            review_rank=33,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_TARGET_PRIMARY_REVIEW_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_TARGET_PRIMARY_REVIEW_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="robustness",
            review_role="proof_carrying",
            artifact_evidence_class="proof_carrying",
            proof_carrying_status="all",
            analysis_level="target_community",
            family_surface_role="not_applicable",
            robustness_routes=target_routes,
            rows_represent="Rows represent the primary target-community route-level robustness findings only.",
            columns_represent="Columns foreground community ids, rank stability, and route-level support metrics without changing the scientific contents.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Primary-only target-community review table filtered from the live target robustness surface.",
            review_rank=34,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_CALL_SEMANTICS_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_CALL_SEMANTICS_FILENAME}",
            source_path="",
            artifact_kind="table",
            format="csv",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="cohort_decision",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Rows represent one explicit Block 2 call-state or execution-state definition.",
            columns_represent="Columns record which surface/field the call belongs to, the exact call value, the threshold rule summary, and whether it should be read as an execution failure.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Machine-readable semantics table for route-level calls, top-level calls, and replicate execution states.",
            review_rank=35,
        ),
        Block2ReviewArtifact(
            artifact_name=BLOCK2_REPRODUCIBILITY_METADATA_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_REPRODUCIBILITY_METADATA_FILENAME}",
            source_path="",
            artifact_kind="manifest",
            format="json",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="cohort_decision",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Single JSON object for the packet-local Block 2 reproducibility and route-metadata summary.",
            columns_represent="Top-level keys record the authoritative raw inputs, route plans, call thresholds, and observed row counts for the review surface.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Packet-local reproducibility and route metadata extracted from the live Block 2 manifest, Block 1 bundle, config, and raw tables.",
            review_rank=36,
        ),
    ]

    human_index_path = block2_dir / BLOCK2_HUMAN_INDEX_FILENAME
    human_index_path.write_text(
        "\n".join(
            _human_index_lines(
                block2_payload=block2_payload,
                raw_artifacts=raw_artifacts,
                derived_artifacts=derived_artifacts,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    derived_artifacts.append(
        Block2ReviewArtifact(
            artifact_name=BLOCK2_HUMAN_INDEX_FILENAME,
            packet_relative_path=f"block2/{BLOCK2_HUMAN_INDEX_FILENAME}",
            source_path="",
            artifact_kind="report",
            format="md",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="cohort_decision",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Markdown guide for the Block 2 packet subsection.",
            columns_represent="Markdown prose headings and bullet lists only.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Human-readable Block 2 review guide for the packet subsection.",
            review_rank=39,
        )
    )

    review_manifest_payload = {
        "workflow_name": "write_block2_review_surface",
        "scientific_interpretation_allowed": False,
        "block2_manifest_path": str(block2_manifest),
        "review_index_path": str(review_dir / BLOCK2_ARTIFACT_INDEX_FILENAME),
        "human_index_path": str(human_index_path),
        "primary_routes": list(block2_payload.get("primary_routes", [])),
        "raw_artifacts": [
            {
                "artifact_name": artifact.artifact_name,
                "packet_relative_path": artifact.packet_relative_path,
                "artifact_evidence_class": artifact.artifact_evidence_class,
                "analysis_level": artifact.analysis_level,
                "robustness_routes": artifact.robustness_routes,
            }
            for artifact in sorted(raw_artifacts, key=lambda item: (item.review_rank, item.artifact_name))
        ],
        "derived_review_artifacts": [
            {
                "artifact_name": artifact.artifact_name,
                "packet_relative_path": artifact.packet_relative_path,
                "artifact_evidence_class": artifact.artifact_evidence_class,
                "analysis_level": artifact.analysis_level,
                "robustness_routes": artifact.robustness_routes,
            }
            for artifact in sorted(derived_artifacts, key=lambda item: (item.review_rank, item.artifact_name))
        ],
    }
    review_manifest_path = review_dir / BLOCK2_OBJECTIVE_REVIEW_MANIFEST_FILENAME
    _write_json(review_manifest_path, review_manifest_payload)
    derived_artifacts.append(
        Block2ReviewArtifact(
            artifact_name=BLOCK2_OBJECTIVE_REVIEW_MANIFEST_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_OBJECTIVE_REVIEW_MANIFEST_FILENAME}",
            source_path="",
            artifact_kind="manifest",
            format="json",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="cohort_decision",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Single JSON object for the packet-local Block 2 objective review surface.",
            columns_represent="Top-level keys record the authoritative raw inputs plus the packet-local Block 2 review artifacts and indexes.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Manifest for the packet-local Block 2 objective review surface.",
            review_rank=38,
        )
    )

    artifact_index_rows: list[dict[str, Any]] = []
    for artifact in sorted([*raw_artifacts, *derived_artifacts], key=lambda item: (item.review_rank, item.artifact_name)):
        destination = block2_dir.parent / artifact.packet_relative_path
        n_rows, n_columns, observed_columns = _observe_artifact(destination) if destination.exists() else ("", "", "")
        artifact_index_rows.append(
            {
                "review_rank": artifact.review_rank,
                "artifact_name": artifact.artifact_name,
                "packet_relative_path": artifact.packet_relative_path,
                "source_path": artifact.source_path,
                "artifact_kind": artifact.artifact_kind,
                "format": artifact.format,
                "artifact_evidence_class": artifact.artifact_evidence_class,
                "proof_carrying_status": artifact.proof_carrying_status,
                "claim_scope": artifact.claim_scope,
                "review_role": artifact.review_role,
                "analysis_level": artifact.analysis_level,
                "family_surface_role": artifact.family_surface_role,
                "robustness_routes": artifact.robustness_routes,
                "n_rows": n_rows,
                "n_columns": n_columns,
                "observed_columns": observed_columns,
                "rows_represent": artifact.rows_represent,
                "columns_represent": artifact.columns_represent,
                "source_workflow": artifact.source_workflow,
                "source_manifest_or_bundle": artifact.source_manifest_or_bundle,
                "notes": artifact.notes,
            }
        )
    artifact_index_path = review_dir / BLOCK2_ARTIFACT_INDEX_FILENAME
    _write_csv(artifact_index_path, pd.DataFrame.from_records(artifact_index_rows))
    derived_artifacts.append(
        Block2ReviewArtifact(
            artifact_name=BLOCK2_ARTIFACT_INDEX_FILENAME,
            packet_relative_path=f"block2/{REVIEW_DIRNAME}/{BLOCK2_ARTIFACT_INDEX_FILENAME}",
            source_path="",
            artifact_kind="index",
            format="csv",
            claim_scope="provenance",
            review_role="provenance",
            artifact_evidence_class="engineering_audit",
            proof_carrying_status="none",
            analysis_level="cohort_decision",
            family_surface_role="not_applicable",
            robustness_routes=all_routes,
            rows_represent="Rows represent one raw or derived Block 2 artifact in the packet-local review surface.",
            columns_represent="Columns record artifact provenance, evidence class, robustness-route coverage, analysis level, and the row/column meaning of each artifact.",
            source_workflow="write_block2_review_surface",
            source_manifest_or_bundle=str(block2_manifest),
            notes="Machine-readable Block 2 artifact catalog for the packet-local review surface.",
            review_rank=37,
        )
    )

    return Block2ReviewSurface(
        manifest_path=review_manifest_path,
        artifact_index_path=artifact_index_path,
        human_index_path=human_index_path,
        artifacts=tuple(sorted(derived_artifacts, key=lambda item: (item.review_rank, item.artifact_name))),
    )


__all__ = [
    "BLOCK2_ARTIFACT_INDEX_FILENAME",
    "BLOCK2_CALL_SEMANTICS_FILENAME",
    "BLOCK2_FAMILY_REVIEW_FILENAME",
    "BLOCK2_HUMAN_INDEX_FILENAME",
    "BLOCK2_OBJECTIVE_REVIEW_MANIFEST_FILENAME",
    "BLOCK2_PRIMARY_FINDING_REVIEW_FILENAME",
    "BLOCK2_REPRODUCIBILITY_METADATA_FILENAME",
    "BLOCK2_ROUTE_SUMMARY_FILENAME",
    "BLOCK2_SOURCE_PRIMARY_REVIEW_FILENAME",
    "BLOCK2_TARGET_PRIMARY_REVIEW_FILENAME",
    "Block2ReviewArtifact",
    "Block2ReviewSurface",
    "write_block2_review_surface",
]
