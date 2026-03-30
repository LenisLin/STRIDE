from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tasks.task_A.runtime_contract import (
    TASK_A_METRICS_FILENAME,
    load_task_a_run_manifest,
    resolve_task_a_arm_artifact_root,
    resolve_task_a_arm_bioinformed_output_dir,
    resolve_task_a_arm_focused_output_dir,
    resolve_task_a_manifest_path,
)

DEFAULT_TASK_A_ROOT = Path("/mnt/NAS_21T/ProjectResult/SLOTAR/task_A")
DEFAULT_ARM2_FOCUSED_DIR = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "analysis" / "focused"
DEFAULT_ARM2_BIOINFORMED_DIR = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "analysis" / "bioinformed"
DEFAULT_ARM3_RESULT_ROOT = DEFAULT_TASK_A_ROOT / "arm3_phase0_8_closure" / "full_2026-03-19"
DEFAULT_OUTPUT_DIR = DEFAULT_TASK_A_ROOT / "arm2_arm3_neutral_extraction"
DEFAULT_STAGE0_PATH = Path("/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad")
ARM2_NAME = "A2_cross_compartment"
ARM3_NAME = "A3_uq_stress"

ARM2_FAMILY_PREFIX = {
    "TC-IM": "tc_im",
    "TC-PT": "tc_pt",
}
ARM2_METRICS_CANDIDATES = (
    DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "task_A_metrics.parquet",
    DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "task_A_arm2_metrics.parquet",
    DEFAULT_TASK_A_ROOT / "task_A_metrics.parquet",
)

ARM2_FOCUSED_FILES = {
    "prototype_meaning": "01_prototype_biological_meaning_table.csv",
    "baseline_pair_audit": "02_baseline_pair_audit.csv",
    "baseline_prototype_summary": "03_baseline_prototype_confirmatory_summary.csv",
    "baseline_patient_family": "04_baseline_patient_family_confirmatory_summary.csv",
    "patient_continuity_backbone_summary": "05_patient_continuity_backbone_summary.csv",
    "trusted_continuity_anchors": "06_trusted_continuity_anchors.csv",
    "closed_comparator_forced_closure": "07_closed_comparator_forced_closure.csv",
    "bounded_residual_contributors": "08_bounded_residual_contributors.csv",
    "anchor_residual_overlap_audit": "09_anchor_residual_overlap_audit.csv",
    "confirmatory_family_backbone_summary": "10_confirmatory_family_backbone_summary.csv",
    "trusted_anchor_patient_recurrence": "11_trusted_anchor_patient_recurrence.csv",
    "memo": "00_task_a_real_data_mirror_memo.md",
}
ARM2_BIO_FILES = {
    "closed_open_contrast": "23_closed_vs_open_prototype_contrast.csv",
    "directional_residual_assignment_audit": "24_directional_residual_assignment_audit.csv",
    "block2_biointegrated_audit_table": "25_block2_biointegrated_audit_table.csv",
}
ARM3_FILES = {
    "full_results": "arm3_phase6_full_coverage_results.parquet",
    "bootstrap_results": "arm3_phase6_bootstrap_results.parquet",
    "balanced_results": "arm3_phase6_balanced_ot_results.parquet",
    "prototype_events_full": "arm3_phase6_prototype_events_full.parquet",
    "prototype_events_bootstrap": "arm3_phase6_prototype_events_bootstrap.parquet",
    "phase7_degradation": "arm3_phase7_degradation_summary.parquet",
    "phase7_contrast": "arm3_phase7_contrast_summary.parquet",
    "prototype_contrast_prep": "arm3_phase8_prototype_contrast_prep.parquet",
    "prototype_stability": "arm3_phase8_prototype_stability.parquet",
    "manifest": "arm3_phase0_manifest.json",
    "runtime_timing": "arm3_runtime_timing.json",
    "memo": "arm3_phase8_memo.md",
}

SOURCE_DOCS = (
    ("documented_spec", "specs_docs", "task_a_spec", REPO_ROOT / "docs" / "task_A_spec.md", "live TaskA arm contract"),
    ("documented_spec", "specs_docs", "data_contracts", REPO_ROOT / "docs" / "data_contracts.md", "current Arm3 data contract"),
    ("documented_spec", "specs_docs", "task_a_readme", REPO_ROOT / "tasks" / "task_A" / "README.md", "task-local TaskA readme"),
)
SOURCE_CODE = (
    ("code_fact", "code", "analyze_arm2_results", REPO_ROOT / "tasks" / "task_A" / "analyze_arm2_results.py", "Arm2 parquet discovery entrypoint"),
    ("code_fact", "code", "arm2_analysis_contract", REPO_ROOT / "tasks" / "task_A" / "arm2" / "analysis_contract.py", "Arm2 focused output contract"),
    ("code_fact", "code", "arm2_analysis_response", REPO_ROOT / "tasks" / "task_A" / "arm2" / "analysis_response.py", "Arm2 output builders"),
    ("code_fact", "code", "arm2_analysis_bioinformed", REPO_ROOT / "tasks" / "task_A" / "arm2" / "analysis_bioinformed.py", "Arm2 OT vs UOT and B/D output builders"),
    ("code_fact", "code", "arm3_uq_stress", REPO_ROOT / "tasks" / "task_A" / "arm3_uq_stress.py", "Arm3 phase runner"),
    ("code_fact", "code", "arm3_output", REPO_ROOT / "tasks" / "task_A" / "arm3" / "output.py", "Arm3 phase 8 output builders"),
)
SOURCE_CONFIGS = (
    ("documented_spec", "configs", "task_a_config", REPO_ROOT / "tasks" / "task_A" / "config.yaml", "TaskA canonical density config"),
    ("documented_spec", "configs", "task_a_arm3_config", REPO_ROOT / "tasks" / "task_A" / "config_arm3.yaml", "TaskA Arm3 config"),
)


@dataclass(frozen=True)
class NeutralExtractionPaths:
    repo_root: Path
    task_a_root: Path
    arm2_focused_dir: Path
    arm2_bio_dir: Path
    arm3_result_root: Path
    output_dir: Path
    stage0_path: Path
    task_a_manifest_path: Path | None = None
    task_a_run_root: Path | None = None
    arm2_metrics_parquet: Path | None = None


@dataclass(frozen=True)
class ExtractionPackage:
    source_inventory: pd.DataFrame
    arm2_prototype_family_evidence: pd.DataFrame
    arm2_patient_family_comparator: pd.DataFrame
    arm2_overlap_audit: pd.DataFrame
    arm3_prototype_coverage_stability: pd.DataFrame
    arm3_prototype_event_surface: pd.DataFrame
    arm3_pair_level_coverage_comparator: pd.DataFrame
    arm3_phase7_patient_summary: pd.DataFrame
    arm2_arm3_linkage_inventory: pd.DataFrame
    manifest: dict[str, Any]


def _load_manifest_from_args(args: argparse.Namespace):
    if args.task_a_manifest is not None:
        return load_task_a_run_manifest(args.task_a_manifest), resolve_task_a_manifest_path(args.task_a_manifest)
    if args.task_a_run_root is not None:
        return load_task_a_run_manifest(args.task_a_run_root), resolve_task_a_manifest_path(args.task_a_run_root)
    return None, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract neutral Arm2/Arm3 evidence matrices and linkage inventories "
            "without scientific adjudication."
        )
    )
    parser.add_argument("--task-a-manifest", default=None)
    parser.add_argument("--task-a-run-root", default=None)
    parser.add_argument("--task-a-root", default=str(DEFAULT_TASK_A_ROOT))
    parser.add_argument("--arm2-focused-dir", default=None)
    parser.add_argument("--arm2-bio-dir", default=None)
    parser.add_argument("--arm3-result-root", default=None)
    parser.add_argument("--stage0-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> NeutralExtractionPaths:
    manifest, manifest_path = _load_manifest_from_args(args)
    task_a_root = Path(args.task_a_root).expanduser().resolve()

    if manifest is not None:
        arm2_focused_dir = (
            Path(args.arm2_focused_dir).expanduser().resolve()
            if args.arm2_focused_dir is not None
            else resolve_task_a_arm_focused_output_dir(manifest, ARM2_NAME)
        )
        arm2_bio_dir = (
            Path(args.arm2_bio_dir).expanduser().resolve()
            if args.arm2_bio_dir is not None
            else resolve_task_a_arm_bioinformed_output_dir(manifest, ARM2_NAME)
        )
        arm3_result_root = (
            Path(args.arm3_result_root).expanduser().resolve()
            if args.arm3_result_root is not None
            else (
                resolve_task_a_arm_artifact_root(manifest, ARM3_NAME)
                if ARM3_NAME in manifest.arm_artifact_roots
                else DEFAULT_ARM3_RESULT_ROOT.expanduser().resolve()
            )
        )
        stage0_path = (
            Path(args.stage0_path).expanduser().resolve()
            if args.stage0_path is not None
            else manifest.stage0_h5ad
        )
        arm2_metrics_parquet = manifest.metrics_parquet
        task_a_run_root = manifest.run_root
    else:
        arm2_focused_dir = Path(args.arm2_focused_dir or DEFAULT_ARM2_FOCUSED_DIR).expanduser().resolve()
        arm2_bio_dir = Path(args.arm2_bio_dir or DEFAULT_ARM2_BIOINFORMED_DIR).expanduser().resolve()
        arm3_result_root = Path(args.arm3_result_root or DEFAULT_ARM3_RESULT_ROOT).expanduser().resolve()
        stage0_path = Path(args.stage0_path or DEFAULT_STAGE0_PATH).expanduser().resolve()
        arm2_metrics_parquet = None
        task_a_run_root = None

    return NeutralExtractionPaths(
        repo_root=REPO_ROOT,
        task_a_root=task_a_root,
        arm2_focused_dir=arm2_focused_dir,
        arm2_bio_dir=arm2_bio_dir,
        arm3_result_root=arm3_result_root,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        stage0_path=stage0_path,
        task_a_manifest_path=manifest_path,
        task_a_run_root=task_a_run_root,
        arm2_metrics_parquet=arm2_metrics_parquet,
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table suffix for {path}")


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _coalesce_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.combine_first(right)


def _family_specific_value(df: pd.DataFrame, pair_family_col: str, mapping: dict[str, str], out_col: str) -> pd.DataFrame:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for family, source_col in mapping.items():
        if source_col not in df.columns:
            continue
        mask = df[pair_family_col].astype(str) == family
        result.loc[mask] = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
    df[out_col] = result
    return df


def longify_arm2_patient_family_wide_table(
    df_wide: pd.DataFrame,
    family_prefix_map: dict[str, str],
    namespace: str,
) -> pd.DataFrame:
    if "patient_id" not in df_wide.columns:
        raise ValueError("Expected patient_id column in Arm2 patient-family wide table")

    comparison_cols = [
        col for col in df_wide.columns if col != "patient_id" and col.startswith("patient_")
    ]
    frames: list[pd.DataFrame] = []
    for pair_family, prefix in family_prefix_map.items():
        family_cols = [col for col in df_wide.columns if col.startswith(f"{prefix}_")]
        subset = df_wide[["patient_id", *comparison_cols, *family_cols]].copy()
        rename_map: dict[str, str] = {}
        for col in family_cols:
            base_name = col[len(prefix) + 1 :]
            if namespace == "baseline":
                if base_name == "ordered_pair_count":
                    rename_map[col] = "ordered_pair_count"
                elif base_name == "prototype_count":
                    rename_map[col] = "prototype_count"
                else:
                    rename_map[col] = f"baseline_{base_name}"
            elif namespace == "transport":
                if base_name == "ordered_pair_count":
                    rename_map[col] = "ordered_pair_count_transport"
                else:
                    rename_map[col] = base_name
            else:
                rename_map[col] = base_name
        subset = subset.rename(columns=rename_map)
        subset.insert(1, "pair_family", pair_family)
        frames.append(subset)

    return (
        pd.concat(frames, axis=0, ignore_index=True, sort=False)
        .sort_values(["patient_id", "pair_family"], kind="stable")
        .reset_index(drop=True)
    )


def build_arm2_patient_family_comparator_from_frames(
    df_baseline: pd.DataFrame,
    df_transport: pd.DataFrame,
) -> pd.DataFrame:
    baseline_long = longify_arm2_patient_family_wide_table(df_baseline, ARM2_FAMILY_PREFIX, namespace="baseline")
    transport_long = longify_arm2_patient_family_wide_table(df_transport, ARM2_FAMILY_PREFIX, namespace="transport")
    merged = baseline_long.merge(
        transport_long,
        on=["patient_id", "pair_family"],
        how="outer",
        validate="one_to_one",
        suffixes=("_baseline_context", "_transport_context"),
    )

    if "ordered_pair_count" in merged.columns and "ordered_pair_count_transport" in merged.columns:
        merged.insert(
            2,
            "ordered_pair_count_coalesced",
            _coalesce_series(
                pd.to_numeric(merged["ordered_pair_count"], errors="coerce"),
                pd.to_numeric(merged["ordered_pair_count_transport"], errors="coerce"),
            ),
        )
    elif "ordered_pair_count" in merged.columns:
        merged.insert(2, "ordered_pair_count_coalesced", pd.to_numeric(merged["ordered_pair_count"], errors="coerce"))
    elif "ordered_pair_count_transport" in merged.columns:
        merged.insert(
            2,
            "ordered_pair_count_coalesced",
            pd.to_numeric(merged["ordered_pair_count_transport"], errors="coerce"),
        )
    else:
        merged.insert(2, "ordered_pair_count_coalesced", pd.Series(np.nan, index=merged.index, dtype=float))

    merged.insert(0, "artifact_present", True)
    merged.insert(1, "source_dir", "focused")
    merged.insert(2, "baseline_table_name", ARM2_FOCUSED_FILES["baseline_patient_family"])
    merged.insert(3, "transport_table_name", ARM2_FOCUSED_FILES["patient_continuity_backbone_summary"])
    merged.insert(4, "table_name", "04_baseline_patient_family_confirmatory_summary.csv|05_patient_continuity_backbone_summary.csv")

    return merged.sort_values(["patient_id", "pair_family"], kind="stable").reset_index(drop=True)


def build_arm2_prototype_family_evidence_from_frames(
    df_family_summary: pd.DataFrame,
    df_recurrence: pd.DataFrame,
    df_bd: pd.DataFrame,
    df_contrast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_family_cols = {"pair_family", "proto_id"}
    missing_family_cols = required_family_cols - set(df_family_summary.columns)
    if missing_family_cols:
        raise ValueError(f"Arm2 family summary is missing required columns: {sorted(missing_family_cols)}")

    out = df_family_summary.copy()

    recurrence_cols = [
        "proto_id",
        "trusted_anchor_score",
        "trusted_anchor_positive_patient_count",
        "trusted_anchor_positive_patient_count_prop",
        "forced_closure_score",
        "forced_closure_positive_patient_count_tc_im",
        "forced_closure_positive_patient_count_tc_im_prop",
        "forced_closure_positive_patient_count_tc_pt",
        "forced_closure_positive_patient_count_tc_pt_prop",
        "forced_closure_positive_patient_count_any_confirmatory",
        "forced_closure_positive_patient_count_any_confirmatory_prop",
        "bounded_residual_score",
        "bounded_residual_positive_patient_count_tc_im",
        "bounded_residual_positive_patient_count_tc_im_prop",
        "bounded_residual_positive_patient_count_tc_pt",
        "bounded_residual_positive_patient_count_tc_pt_prop",
        "bounded_residual_positive_patient_count_any_confirmatory",
        "bounded_residual_positive_patient_count_any_confirmatory_prop",
        "trusted_anchor_and_bounded_residual_any_patient_count",
        "trusted_anchor_and_bounded_residual_any_patient_count_prop",
    ]
    available_recurrence_cols = [col for col in recurrence_cols if col in df_recurrence.columns]
    if available_recurrence_cols:
        out = out.merge(
            df_recurrence[available_recurrence_cols].copy(),
            on="proto_id",
            how="left",
            validate="many_to_one",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "forced_closure_positive_patient_count_tc_im",
                "TC-PT": "forced_closure_positive_patient_count_tc_pt",
            },
            out_col="forced_closure_positive_patient_count_family",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "bounded_residual_positive_patient_count_tc_im",
                "TC-PT": "bounded_residual_positive_patient_count_tc_pt",
            },
            out_col="bounded_residual_positive_patient_count_family",
        )

    bd_primary = df_bd.copy()
    if "direction_role" in bd_primary.columns:
        bd_primary = bd_primary.loc[bd_primary["direction_role"].astype(str) == "primary_anchor"].copy()
    bd_keep = [
        "pair_family",
        "proto_id",
        "pair_type",
        "direction_role",
        "patient_count",
        "source_depletion_prone_share",
        "target_emergence_prone_share",
        "depletion_minus_emergence",
        "source_depletion_prone_abs",
        "target_emergence_prone_abs",
        "source_depletion_gt_emergence_patient_count",
        "source_depletion_gt_emergence_patient_prop",
        "emergence_gt_source_depletion_patient_count",
        "emergence_gt_source_depletion_patient_prop",
    ]
    bd_keep = [col for col in bd_keep if col in bd_primary.columns]
    if bd_keep:
        bd_primary = bd_primary[bd_keep].rename(
            columns={
                "pair_type": "bd_pair_type",
                "direction_role": "bd_direction_role",
                "patient_count": "bd_patient_count",
            }
        )
        out = out.merge(
            bd_primary,
            on=["pair_family", "proto_id"],
            how="left",
            validate="one_to_one",
        )

    if df_contrast is not None and not df_contrast.empty:
        contrast_keep = [
            "proto_id",
            "panel_name",
            "panel_rule",
            "is_borderline_tc_like",
            "closed_comparator_share_tc_im",
            "continuity_backbone_share_tc_im",
            "bounded_residual_share_tc_im",
            "forced_closure_excess_tc_im",
            "closed_comparator_share_tc_pt",
            "continuity_backbone_share_tc_pt",
            "bounded_residual_share_tc_pt",
            "forced_closure_excess_tc_pt",
        ]
        contrast_keep = [col for col in contrast_keep if col in df_contrast.columns]
        out = out.merge(
            df_contrast[contrast_keep].copy(),
            on="proto_id",
            how="left",
            validate="many_to_one",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "closed_comparator_share_tc_im",
                "TC-PT": "closed_comparator_share_tc_pt",
            },
            out_col="contrast_closed_comparator_share",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "continuity_backbone_share_tc_im",
                "TC-PT": "continuity_backbone_share_tc_pt",
            },
            out_col="contrast_continuity_backbone_share",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "bounded_residual_share_tc_im",
                "TC-PT": "bounded_residual_share_tc_pt",
            },
            out_col="contrast_bounded_residual_share",
        )
        out = _family_specific_value(
            out,
            pair_family_col="pair_family",
            mapping={
                "TC-IM": "forced_closure_excess_tc_im",
                "TC-PT": "forced_closure_excess_tc_pt",
            },
            out_col="contrast_forced_closure_excess",
        )

    out.insert(0, "artifact_present", True)
    out.insert(1, "source_dir", "focused|bioinformed")
    out.insert(2, "table_name", ARM2_FOCUSED_FILES["confirmatory_family_backbone_summary"])
    out.insert(3, "family_summary_table_name", ARM2_FOCUSED_FILES["confirmatory_family_backbone_summary"])
    out.insert(4, "recurrence_table_name", ARM2_FOCUSED_FILES["trusted_anchor_patient_recurrence"])
    out.insert(5, "bd_table_name", ARM2_BIO_FILES["directional_residual_assignment_audit"])
    out.insert(6, "contrast_table_name", ARM2_BIO_FILES["closed_open_contrast"] if df_contrast is not None else pd.NA)
    out.insert(7, "recurrence_artifact_present", "trusted_anchor_score" in out.columns)
    out.insert(8, "bd_artifact_present", out.get("bd_pair_type", pd.Series(pd.NA, index=out.index)).notna())
    out.insert(9, "contrast_artifact_present", out.get("panel_name", pd.Series(pd.NA, index=out.index)).notna())

    sort_cols = [col for col in ["pair_family", "baseline_priority_rank", "proto_id"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_arm2_overlap_audit_from_frame(df_overlap: pd.DataFrame) -> pd.DataFrame:
    out = df_overlap.copy()
    out.insert(0, "artifact_present", True)
    out.insert(1, "source_dir", "focused")
    out.insert(2, "table_name", ARM2_FOCUSED_FILES["anchor_residual_overlap_audit"])
    return out


def _normalize_arm3_uot_frame(df: pd.DataFrame, table_name: str, default_coverage: float | None) -> pd.DataFrame:
    out = df.copy()
    if "coverage" not in out.columns:
        out["coverage"] = float(default_coverage) if default_coverage is not None else np.nan
    if "replicate_id" not in out.columns:
        out["replicate_id"] = -1
    out.insert(0, "artifact_present", True)
    out.insert(1, "table_name", table_name)
    return out


def build_arm3_pair_level_coverage_comparator_from_frames(
    df_full: pd.DataFrame,
    df_bootstrap: pd.DataFrame,
    df_balanced: pd.DataFrame,
) -> pd.DataFrame:
    full_norm = _normalize_arm3_uot_frame(df_full, ARM3_FILES["full_results"], default_coverage=1.0)
    bootstrap_norm = _normalize_arm3_uot_frame(df_bootstrap, ARM3_FILES["bootstrap_results"], default_coverage=None)
    uot = pd.concat([full_norm, bootstrap_norm], axis=0, ignore_index=True, sort=False)

    key_cols = ["pair_id", "patient_id", "pair_type", "pair_family", "coverage", "replicate_id"]
    balanced_keep = [
        "pair_id",
        "patient_id",
        "pair_type",
        "pair_family",
        "coverage",
        "replicate_id",
        "compartment_a",
        "compartment_b",
        "balanced_ot_cost",
        "comparator_type",
    ]
    balanced = df_balanced[[col for col in balanced_keep if col in df_balanced.columns]].copy()
    merged = uot.merge(
        balanced,
        on=key_cols,
        how="left",
        validate="one_to_one",
        suffixes=("", "_balanced"),
    )

    if "compartment_a_balanced" in merged.columns:
        merged["compartment_a"] = _coalesce_series(merged.get("compartment_a"), merged["compartment_a_balanced"])
        merged = merged.drop(columns=["compartment_a_balanced"])
    if "compartment_b_balanced" in merged.columns:
        merged["compartment_b"] = _coalesce_series(merged.get("compartment_b"), merged["compartment_b_balanced"])
        merged = merged.drop(columns=["compartment_b_balanced"])

    merged.insert(2, "balanced_table_name", ARM3_FILES["balanced_results"])
    sort_cols = [col for col in ["pair_type", "coverage", "replicate_id", "patient_id", "pair_id"] if col in merged.columns]
    return merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_arm3_prototype_event_surface_from_frames(
    df_full: pd.DataFrame,
    df_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    full_norm = _normalize_arm3_uot_frame(df_full, ARM3_FILES["prototype_events_full"], default_coverage=1.0)
    bootstrap_norm = _normalize_arm3_uot_frame(df_bootstrap, ARM3_FILES["prototype_events_bootstrap"], default_coverage=None)
    out = pd.concat([full_norm, bootstrap_norm], axis=0, ignore_index=True, sort=False)
    sort_cols = [col for col in ["prototype_k", "coverage", "replicate_id", "pair_type", "patient_id", "pair_id"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_arm3_phase7_patient_summary_from_frames(
    df_degradation: pd.DataFrame,
    df_contrast: pd.DataFrame,
) -> pd.DataFrame:
    degradation = df_degradation.copy()
    degradation.insert(0, "artifact_present", True)
    degradation.insert(1, "table_name", ARM3_FILES["phase7_degradation"])
    degradation.insert(2, "summary_type", "degradation")

    contrast = df_contrast.copy()
    contrast.insert(0, "artifact_present", True)
    contrast.insert(1, "table_name", ARM3_FILES["phase7_contrast"])
    contrast.insert(2, "summary_type", "contrast")

    out = pd.concat([degradation, contrast], axis=0, ignore_index=True, sort=False)
    sort_cols = [col for col in ["summary_type", "coverage", "patient_id", "pair_type", "quantity", "contrast_name"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_arm3_prototype_coverage_stability_from_frame(df_stability: pd.DataFrame) -> pd.DataFrame:
    out = df_stability.copy()
    out.insert(0, "artifact_present", True)
    out.insert(1, "table_name", ARM3_FILES["prototype_stability"])
    sort_cols = [col for col in ["prototype_k", "coverage"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _prototype_label_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    label_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        proto_col = None
        if "prototype_k" in frame.columns:
            proto_col = "prototype_k"
        elif "proto_id" in frame.columns:
            proto_col = "proto_id"
        if proto_col is None or "prototype_label" not in frame.columns:
            continue
        label_frames.append(
            frame[[proto_col, "prototype_label"]]
            .rename(columns={proto_col: "arm3_prototype_k", "prototype_label": "arm3_label_field"})
            .copy()
        )
    if not label_frames:
        return pd.DataFrame(columns=["arm3_prototype_k", "arm3_label_field"])

    labels = pd.concat(label_frames, axis=0, ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    for prototype_k, grp in labels.groupby("arm3_prototype_k", sort=True):
        unique_labels = sorted({str(x) for x in grp["arm3_label_field"].dropna().astype(str).tolist()})
        records.append(
            {
                "arm3_prototype_k": int(prototype_k),
                "arm3_label_field": unique_labels[0] if unique_labels else pd.NA,
            }
        )
    return pd.DataFrame.from_records(records)


def build_arm2_arm3_linkage_inventory_from_frames(
    df_arm2_meaning: pd.DataFrame,
    df_arm3_labels: pd.DataFrame,
    shared_stage0_path: str,
    linkage_source_artifact: str,
) -> pd.DataFrame:
    arm2 = df_arm2_meaning[[col for col in ["proto_id", "dominant_cell_type", "prototype_label_top3"] if col in df_arm2_meaning.columns]].copy()
    arm2 = arm2.rename(
        columns={
            "proto_id": "arm2_proto_id",
            "prototype_label_top3": "arm2_label_field",
        }
    )
    arm3 = df_arm3_labels.copy()
    merged = arm2.merge(
        arm3,
        left_on="arm2_proto_id",
        right_on="arm3_prototype_k",
        how="outer",
        validate="one_to_one",
    )

    linkage_type = np.where(
        merged["arm2_proto_id"].notna() & merged["arm3_prototype_k"].notna(),
        "shared_prototype_index",
        np.where(merged["arm2_proto_id"].notna(), "arm2_only", "arm3_only"),
    )
    merged.insert(0, "artifact_present", True)
    merged.insert(1, "table_name", "01_prototype_biological_meaning_table.csv|arm3_phase8_prototype_stability.parquet|arm3_phase8_prototype_contrast_prep.parquet")
    merged.insert(2, "shared_stage0_path", shared_stage0_path)
    merged.insert(3, "linkage_type", linkage_type)
    merged.insert(4, "linkage_source_artifact", linkage_source_artifact)

    for col in ["arm2_proto_id", "arm3_prototype_k"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    merged = merged.sort_values(
        by=["arm2_proto_id", "arm3_prototype_k"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return merged


def _tabular_source_summary(path: Path) -> tuple[float | int | None, float | int | None, str | None]:
    if not path.exists():
        return (np.nan, np.nan, None)
    if path.suffix.lower() not in {".csv", ".parquet"}:
        return (np.nan, np.nan, None)
    df = read_table(path)
    return (int(len(df)), int(df.shape[1]), _to_json(list(df.columns)))


def build_source_inventory(paths: NeutralExtractionPaths) -> pd.DataFrame:
    entries: list[dict[str, Any]] = []

    for evidence_class, group_name, logical_name, path, note in (*SOURCE_DOCS, *SOURCE_CODE, *SOURCE_CONFIGS):
        line_count = np.nan
        if path.exists() and path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".json"}:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        row_count, column_count, column_names_json = _tabular_source_summary(path)
        entries.append(
            {
                "artifact_present": path.exists(),
                "evidence_class": evidence_class,
                "group_name": group_name,
                "logical_name": logical_name,
                "path": str(path),
                "file_format": path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
                "line_count": line_count,
                "row_count": row_count,
                "column_count": column_count,
                "column_names_json": column_names_json,
                "note": note,
            }
        )

    artifact_entries = [
        ("artifact_fact", "results_artifacts", "task_a_stage0_h5ad", paths.stage0_path, "shared Stage0 prototype axis"),
        ("artifact_fact", "results_artifacts", "arm2_focused_dir", paths.arm2_focused_dir, "Arm2 focused result directory"),
        ("artifact_fact", "results_artifacts", "arm2_bio_dir", paths.arm2_bio_dir, "Arm2 bioinformed result directory"),
        ("artifact_fact", "results_artifacts", "arm3_result_root", paths.arm3_result_root, "Arm3 result directory"),
        ("artifact_fact", "results_artifacts", "task_a_manifest", paths.task_a_manifest_path, "TaskA formal run manifest") if paths.task_a_manifest_path is not None else None,
        ("artifact_fact", "results_artifacts", "arm2_metrics_parquet", paths.arm2_metrics_parquet, "TaskA formal Arm2 metrics parquet") if paths.arm2_metrics_parquet is not None else None,
        ("artifact_fact", "logs_memos_reports", "arm2_focused_memo", paths.arm2_focused_dir / ARM2_FOCUSED_FILES["memo"], "Arm2 focused memo"),
        ("artifact_fact", "logs_memos_reports", "arm2_bio_memo_table", paths.arm2_bio_dir / ARM2_BIO_FILES["block2_biointegrated_audit_table"], "Arm2 Block-2 biointegrated audit table"),
        ("artifact_fact", "logs_memos_reports", "arm3_manifest", paths.arm3_result_root / ARM3_FILES["manifest"], "Arm3 phase0 manifest"),
        ("artifact_fact", "logs_memos_reports", "arm3_runtime_timing", paths.arm3_result_root / ARM3_FILES["runtime_timing"], "Arm3 runtime timing"),
        ("artifact_fact", "logs_memos_reports", "arm3_phase8_memo", paths.arm3_result_root / ARM3_FILES["memo"], "Arm3 phase8 memo"),
    ]
    for key, filename in ARM2_FOCUSED_FILES.items():
        if key == "memo":
            continue
        artifact_entries.append(("artifact_fact", "results_artifacts", f"arm2_{key}", paths.arm2_focused_dir / filename, f"Arm2 focused artifact {filename}"))
    for key, filename in ARM2_BIO_FILES.items():
        if key == "block2_biointegrated_audit_table":
            continue
        artifact_entries.append(("artifact_fact", "results_artifacts", f"arm2_{key}", paths.arm2_bio_dir / filename, f"Arm2 bio artifact {filename}"))
    for key, filename in ARM3_FILES.items():
        if key in {"manifest", "runtime_timing", "memo"}:
            continue
        artifact_entries.append(("artifact_fact", "results_artifacts", f"arm3_{key}", paths.arm3_result_root / filename, f"Arm3 artifact {filename}"))
    for idx, candidate in enumerate(_arm2_metrics_candidates(paths), start=1):
        role = "manifest_or_canonical_candidate" if idx == 1 else "duplicate_or_fallback_candidate"
        artifact_entries.append(("artifact_fact", "results_artifacts", f"arm2_metrics_candidate_{idx}", candidate, role))

    for entry in artifact_entries:
        if entry is None:
            continue
        evidence_class, group_name, logical_name, path, note = entry
        row_count, column_count, column_names_json = _tabular_source_summary(path)
        line_count = np.nan
        if path.exists() and path.suffix.lower() in {".md", ".json"}:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        entries.append(
            {
                "artifact_present": path.exists(),
                "evidence_class": evidence_class,
                "group_name": group_name,
                "logical_name": logical_name,
                "path": str(path),
                "file_format": path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
                "line_count": line_count,
                "row_count": row_count,
                "column_count": column_count,
                "column_names_json": column_names_json,
                "note": note,
            }
        )

    return pd.DataFrame.from_records(entries).sort_values(["evidence_class", "group_name", "logical_name"], kind="stable").reset_index(drop=True)


def _arm2_metrics_candidates(paths: NeutralExtractionPaths) -> tuple[Path, ...]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    def _append(candidate: Path | None) -> None:
        if candidate is None:
            return
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(resolved)

    _append(paths.arm2_metrics_parquet)
    for candidate in ARM2_METRICS_CANDIDATES:
        _append(candidate)
    return tuple(ordered)


def select_arm2_metrics_parquet(paths: NeutralExtractionPaths) -> Path | None:
    for candidate in _arm2_metrics_candidates(paths):
        if candidate.exists():
            return candidate.resolve()
    return None


def summarize_arm2_metrics_candidates(paths: NeutralExtractionPaths) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in _arm2_metrics_candidates(paths):
        record: dict[str, Any] = {
            "path": str(candidate),
            "artifact_present": candidate.exists(),
        }
        if candidate.exists():
            df = pd.read_parquet(candidate)
            record["row_count"] = int(len(df))
            record["column_count"] = int(df.shape[1])
            record["arm_values"] = sorted(df.get("arm", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
            record["mass_mode_values"] = sorted(df.get("mass_mode", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
            if "pair_family" in df.columns:
                record["pair_family_counts"] = {
                    str(key): int(val)
                    for key, val in df["pair_family"].astype(str).value_counts().sort_index().items()
                }
        summaries.append(record)
    return summaries


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(paths: NeutralExtractionPaths, package: ExtractionPackage) -> dict[str, Any]:
    arm3_manifest_path = paths.arm3_result_root / ARM3_FILES["manifest"]
    arm3_manifest = load_json(arm3_manifest_path)
    selected_arm2_metrics = select_arm2_metrics_parquet(paths)

    return {
        "output_dir": str(paths.output_dir),
        "repo_root": str(paths.repo_root),
        "task_a_root": str(paths.task_a_root),
        "task_a_manifest_path": str(paths.task_a_manifest_path) if paths.task_a_manifest_path is not None else None,
        "task_a_run_root": str(paths.task_a_run_root) if paths.task_a_run_root is not None else None,
        "arm2_focused_dir": str(paths.arm2_focused_dir),
        "arm2_bio_dir": str(paths.arm2_bio_dir),
        "arm3_result_root": str(paths.arm3_result_root),
        "selected_arm2_metrics_parquet": str(selected_arm2_metrics) if selected_arm2_metrics is not None else None,
        "arm2_metrics_candidates": summarize_arm2_metrics_candidates(paths),
        "shared_stage0_path": str(arm3_manifest.get("stage0_path", paths.stage0_path)),
        "output_row_counts": {
            "source_inventory": int(len(package.source_inventory)),
            "arm2_prototype_family_evidence": int(len(package.arm2_prototype_family_evidence)),
            "arm2_patient_family_comparator": int(len(package.arm2_patient_family_comparator)),
            "arm2_overlap_audit": int(len(package.arm2_overlap_audit)),
            "arm3_prototype_coverage_stability": int(len(package.arm3_prototype_coverage_stability)),
            "arm3_prototype_event_surface": int(len(package.arm3_prototype_event_surface)),
            "arm3_pair_level_coverage_comparator": int(len(package.arm3_pair_level_coverage_comparator)),
            "arm3_phase7_patient_summary": int(len(package.arm3_phase7_patient_summary)),
            "arm2_arm3_linkage_inventory": int(len(package.arm2_arm3_linkage_inventory)),
        },
        "output_files": {
            "manifest": "00_neutral_extraction_manifest.json",
            "source_inventory": "01_artifact_source_inventory.csv",
            "arm2_prototype_family_evidence": "02_arm2_prototype_family_evidence.csv",
            "arm2_patient_family_comparator": "03_arm2_patient_family_comparator.csv",
            "arm2_overlap_audit": "04_arm2_overlap_audit.csv",
            "arm3_prototype_coverage_stability": "05_arm3_prototype_coverage_stability.csv",
            "arm3_prototype_event_surface": "06_arm3_prototype_event_surface.parquet",
            "arm3_pair_level_coverage_comparator": "07_arm3_pair_level_coverage_comparator.parquet",
            "arm3_phase7_patient_summary": "08_arm3_phase7_patient_summary.csv",
            "arm2_arm3_linkage_inventory": "09_arm2_arm3_linkage_inventory.csv",
        },
    }


def extract_neutral_evidence(paths: NeutralExtractionPaths) -> ExtractionPackage:
    source_inventory = build_source_inventory(paths)

    arm2_prototype_meaning = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["prototype_meaning"])
    arm2_family_summary = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["confirmatory_family_backbone_summary"])
    arm2_recurrence = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["trusted_anchor_patient_recurrence"])
    arm2_bd = read_table(paths.arm2_bio_dir / ARM2_BIO_FILES["directional_residual_assignment_audit"])
    arm2_contrast = read_table(paths.arm2_bio_dir / ARM2_BIO_FILES["closed_open_contrast"])
    arm2_baseline_patient = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["baseline_patient_family"])
    arm2_transport_patient = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["patient_continuity_backbone_summary"])
    arm2_overlap = read_table(paths.arm2_focused_dir / ARM2_FOCUSED_FILES["anchor_residual_overlap_audit"])

    arm3_stability = read_table(paths.arm3_result_root / ARM3_FILES["prototype_stability"])
    arm3_proto_contrast_prep = read_table(paths.arm3_result_root / ARM3_FILES["prototype_contrast_prep"])
    arm3_events_full = read_table(paths.arm3_result_root / ARM3_FILES["prototype_events_full"])
    arm3_events_bootstrap = read_table(paths.arm3_result_root / ARM3_FILES["prototype_events_bootstrap"])
    arm3_full = read_table(paths.arm3_result_root / ARM3_FILES["full_results"])
    arm3_bootstrap = read_table(paths.arm3_result_root / ARM3_FILES["bootstrap_results"])
    arm3_balanced = read_table(paths.arm3_result_root / ARM3_FILES["balanced_results"])
    arm3_phase7_degradation = read_table(paths.arm3_result_root / ARM3_FILES["phase7_degradation"])
    arm3_phase7_contrast = read_table(paths.arm3_result_root / ARM3_FILES["phase7_contrast"])
    arm3_manifest = load_json(paths.arm3_result_root / ARM3_FILES["manifest"])

    arm2_prototype_family_evidence = build_arm2_prototype_family_evidence_from_frames(
        arm2_family_summary,
        arm2_recurrence,
        arm2_bd,
        df_contrast=arm2_contrast,
    )
    arm2_patient_family_comparator = build_arm2_patient_family_comparator_from_frames(
        arm2_baseline_patient,
        arm2_transport_patient,
    )
    arm2_overlap_audit = build_arm2_overlap_audit_from_frame(arm2_overlap)
    arm3_prototype_coverage_stability = build_arm3_prototype_coverage_stability_from_frame(arm3_stability)
    arm3_prototype_event_surface = build_arm3_prototype_event_surface_from_frames(
        arm3_events_full,
        arm3_events_bootstrap,
    )
    arm3_pair_level_coverage_comparator = build_arm3_pair_level_coverage_comparator_from_frames(
        arm3_full,
        arm3_bootstrap,
        arm3_balanced,
    )
    arm3_phase7_patient_summary = build_arm3_phase7_patient_summary_from_frames(
        arm3_phase7_degradation,
        arm3_phase7_contrast,
    )

    arm3_label_lookup = _prototype_label_lookup(arm3_stability, arm3_proto_contrast_prep, arm3_prototype_event_surface)
    linkage_source_artifact = str(paths.arm3_result_root / ARM3_FILES["manifest"])
    shared_stage0_path = str(arm3_manifest.get("stage0_path", paths.stage0_path))
    arm2_arm3_linkage_inventory = build_arm2_arm3_linkage_inventory_from_frames(
        arm2_prototype_meaning,
        arm3_label_lookup,
        shared_stage0_path=shared_stage0_path,
        linkage_source_artifact=linkage_source_artifact,
    )

    package = ExtractionPackage(
        source_inventory=source_inventory,
        arm2_prototype_family_evidence=arm2_prototype_family_evidence,
        arm2_patient_family_comparator=arm2_patient_family_comparator,
        arm2_overlap_audit=arm2_overlap_audit,
        arm3_prototype_coverage_stability=arm3_prototype_coverage_stability,
        arm3_prototype_event_surface=arm3_prototype_event_surface,
        arm3_pair_level_coverage_comparator=arm3_pair_level_coverage_comparator,
        arm3_phase7_patient_summary=arm3_phase7_patient_summary,
        arm2_arm3_linkage_inventory=arm2_arm3_linkage_inventory,
        manifest={},
    )
    manifest = build_manifest(paths, package)
    return ExtractionPackage(**{**package.__dict__, "manifest": manifest})


def write_extraction_package(package: ExtractionPackage, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "manifest": output_dir / "00_neutral_extraction_manifest.json",
        "source_inventory": output_dir / "01_artifact_source_inventory.csv",
        "arm2_prototype_family_evidence": output_dir / "02_arm2_prototype_family_evidence.csv",
        "arm2_patient_family_comparator": output_dir / "03_arm2_patient_family_comparator.csv",
        "arm2_overlap_audit": output_dir / "04_arm2_overlap_audit.csv",
        "arm3_prototype_coverage_stability": output_dir / "05_arm3_prototype_coverage_stability.csv",
        "arm3_prototype_event_surface": output_dir / "06_arm3_prototype_event_surface.parquet",
        "arm3_pair_level_coverage_comparator": output_dir / "07_arm3_pair_level_coverage_comparator.parquet",
        "arm3_phase7_patient_summary": output_dir / "08_arm3_phase7_patient_summary.csv",
        "arm2_arm3_linkage_inventory": output_dir / "09_arm2_arm3_linkage_inventory.csv",
    }

    output_paths["manifest"].write_text(_to_json(package.manifest), encoding="utf-8")
    package.source_inventory.to_csv(output_paths["source_inventory"], index=False)
    package.arm2_prototype_family_evidence.to_csv(output_paths["arm2_prototype_family_evidence"], index=False)
    package.arm2_patient_family_comparator.to_csv(output_paths["arm2_patient_family_comparator"], index=False)
    package.arm2_overlap_audit.to_csv(output_paths["arm2_overlap_audit"], index=False)
    package.arm3_prototype_coverage_stability.to_csv(output_paths["arm3_prototype_coverage_stability"], index=False)
    package.arm3_prototype_event_surface.to_parquet(output_paths["arm3_prototype_event_surface"], index=False)
    package.arm3_pair_level_coverage_comparator.to_parquet(output_paths["arm3_pair_level_coverage_comparator"], index=False)
    package.arm3_phase7_patient_summary.to_csv(output_paths["arm3_phase7_patient_summary"], index=False)
    package.arm2_arm3_linkage_inventory.to_csv(output_paths["arm2_arm3_linkage_inventory"], index=False)
    return output_paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    package = extract_neutral_evidence(paths)
    output_paths = write_extraction_package(package, paths.output_dir)
    summary = {
        "output_dir": str(paths.output_dir),
        "files_written": {key: str(value) for key, value in output_paths.items()},
        "row_counts": package.manifest["output_row_counts"],
    }
    print(_to_json(summary))


if __name__ == "__main__":
    main()
