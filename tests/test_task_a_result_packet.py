from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.errors import ContractError


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_prepare_bundle(base: Path, *, patient_id: str = "P01") -> Path:
    mapping_path = _write_json(
        base / "task_a_stride_mapping.json",
        {
            "field_mapping": {
                "patient_id_key": "patient_id",
                "fov_key": "roi_id",
                "domain_key": "compartment",
                "cell_subtype_key": "cell_type",
                "state_id_key": "proto_id",
            },
            "patient_ids": [patient_id],
            "family_summaries": [],
            "real_data_crosswalk": {"crosswalk": []},
        },
    )
    core_fit_path = _write_csv(
        base / "task_a_core_fit_dry_run.csv",
        [
            {
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "patient_id": patient_id,
                "fit_status": "ok",
                "bridge_realized": True,
                "defer_reason": "",
                "uncertainty_status": "",
                "source_domain": "TC",
                "target_domain": "IM",
            }
        ],
    )
    return _write_json(
        base / "task_a_prepare_manifest.json",
        {
            "task_name": "Task A test prepare",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "mapping_manifest": str(mapping_path),
            "core_fit_dry_run": str(core_fit_path),
            "pair_families": ["TC-IM", "TC-PT"],
            "confirmatory_pair_families": ["TC-IM", "TC-PT"],
            "run_scope": "full_cohort_alignment_check",
            "artifact_state": "contract_passed",
            "block0_gate_status": "not_passed",
            "scientific_interpretation_allowed": False,
            "mass_mode": "uniform",
            "fit_surface": "fit_stride",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
        },
    )


def _write_atlas_bundle(base: Path) -> Path:
    atlas_dir = base / "atlas"
    _write_csv(
        atlas_dir / "tables" / "community_cell_subtype_row_fractions.csv",
        [{"community_id": 0, "TC_EpCAM": 0.8, "CD8T": 0.2}],
    )
    _write_csv(
        atlas_dir / "tables" / "community_domain_distribution.csv",
        [
            {
                "community_id": 0,
                "domain_label": "TC",
                "n_cells": 10,
                "community_total_cells": 10,
                "domain_total_cells": 10,
                "fraction_within_community": 1.0,
                "fraction_within_domain": 1.0,
            }
        ],
    )
    _write_csv(
        atlas_dir / "tables" / "community_patient_occurrence_summary.csv",
        [
            {
                "community_id": 0,
                "n_patients_present": 1,
                "total_cells": 10,
                "median_cells_per_positive_patient": 10,
                "max_cells_in_single_patient": 10,
                "n_patients_total": 1,
                "patient_prevalence": 1.0,
                "n_positive_rois": 1,
                "n_total_rois": 1,
                "roi_prevalence": 1.0,
            }
        ],
    )
    _write_text(atlas_dir / "figures" / "community_by_cell_subtype_heatmap.svg", "<svg></svg>\n")
    output_index_rows = [
        {
            "relative_path": "task_a_descriptive_atlas_manifest.json",
            "artifact_kind": "manifest",
            "category": "atlas_metadata",
            "format": "json",
            "description": "Task A descriptive atlas manifest",
        },
        {
            "relative_path": "task_a_descriptive_atlas_output_index.csv",
            "artifact_kind": "index",
            "category": "atlas_metadata",
            "format": "csv",
            "description": "Machine-readable index of atlas outputs",
        },
        {
            "relative_path": "tables/community_cell_subtype_row_fractions.csv",
            "artifact_kind": "table",
            "category": "community_cell_subtype",
            "format": "csv",
            "description": "Community by cell-subtype row fractions",
        },
        {
            "relative_path": "tables/community_domain_distribution.csv",
            "artifact_kind": "table",
            "category": "community_domain_distribution",
            "format": "csv",
            "description": "Community abundance summaries across TC/IM/PT",
        },
        {
            "relative_path": "tables/community_patient_occurrence_summary.csv",
            "artifact_kind": "table",
            "category": "patient_occurrence",
            "format": "csv",
            "description": "Patient-level community occurrence summary",
        },
        {
            "relative_path": "figures/community_by_cell_subtype_heatmap.svg",
            "artifact_kind": "figure",
            "category": "community_cell_subtype",
            "format": "svg",
            "description": "Community x cell-subtype heatmap",
        },
    ]
    output_index_path = _write_csv(
        atlas_dir / "task_a_descriptive_atlas_output_index.csv",
        output_index_rows,
    )
    return _write_json(
        atlas_dir / "task_a_descriptive_atlas_manifest.json",
        {
            "workflow_name": "write_task_a_descriptive_atlas",
            "atlas_role": "descriptive_only",
            "claim_scope": "descriptive_only",
            "scientific_interpretation_allowed": False,
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "stage0.h5ad"),
            "community_id_key": "proto_id",
            "cell_subtype_key": "cell_type",
            "domain_key": "compartment",
            "fov_key": "roi_id",
            "patient_id_key": "patient_id",
            "spatial_key": "spatial",
            "configured_community_ids": [0],
            "observed_community_ids": [0],
            "output_index": str(output_index_path),
        },
    )


def _write_block0_run(base: Path) -> tuple[Path, Path, Path]:
    prepare_manifest_path = _write_prepare_bundle(base / "p0_prepare_full")
    p0_suitability_path = _write_json(
        base / "p0_suitability_full" / "task_a_pre_block0_data_suitability.json",
        {
            "task_name": "Task A test suitability",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "report_scope": "pre_block0_data_suitability",
            "run_scope": "full_cohort_alignment_check",
            "artifact_state": "contract_passed",
            "block0_gate_status": "not_passed",
            "scientific_interpretation_allowed": False,
            "mass_mode": "uniform",
            "fit_surface": "fit_stride",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "confirmatory_pair_families": ["TC-IM", "TC-PT"],
            "audit_pair_families": ["IM-PT"],
            "stage0_validation": {"artifact_state": "contract_passed"},
        },
    )
    block0_dir = base / "p3_block0_full"
    pair_metrics_path = _write_csv(
        block0_dir / "block0_pair_metrics.csv",
        [
            {
                "comparison_id": "legacy-001",
                "pair_family": "locality_anchor",
                "control_family": "cross_patient_pseudopair",
                "draw_number": 1,
                "draw_label": "draw_0001",
                "slot_index": 1,
                "run_scope": "full_cohort",
                "anchor_patient_id": "P01",
                "anchor_compartment": "TC",
                "anchor_roi_id": "ROI_01",
                "partner_patient_id": "P02",
                "partner_compartment": "IM",
                "partner_roi_id": "ROI_02",
                "same_patient": False,
                "same_compartment": False,
                "pair_type": "TC-IM",
                "partner_match_l1": 0.0,
                "match_penalty": 1.0,
                "retention_threshold": 0.5,
                "T": 1.0,
                "D_pos": 0.1,
                "B_pos": 0.2,
                "d_rel": 0.1,
                "b_rel": 0.2,
                "M": 0.3,
                "R": 0.7,
                "tau": 0.5,
                "observation_fit_status": "ok",
            }
        ],
    )
    _write_json(
        block0_dir / "task_a_pre_block0_data_suitability.json",
        {
            "task_name": "Task A block0 suitability",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "report_scope": "pre_block0_data_suitability",
            "run_scope": "full_cohort_alignment_check",
            "artifact_state": "contract_passed",
            "block0_gate_status": "not_passed",
            "scientific_interpretation_allowed": False,
            "mass_mode": "uniform",
            "fit_surface": "fit_stride",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
        },
    )
    _write_json(
        base / "execution_status.json",
        {
            "run_id": "test_run",
            "executed_phases": {"P3": {"status": "completed", "bundle_status": "failed"}},
            "blocked_phases": {"P4": "blocked"},
        },
    )
    _write_text(base / "engineering_execution_report.md", "# Task A execution report\n")
    block0_bundle_path = _write_json(
        block0_dir / "block0_bundle.json",
        {
            "block": "block0_locality_gate",
            "status": "failed",
            "artifact_state": "scaffold_active",
            "block0_passed": False,
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "output_dir": str(block0_dir),
            "bundle_path": str(block0_dir / "block0_bundle.json"),
            "pair_metrics_path": str(pair_metrics_path),
            "confirmatory_pair_families": ["locality_anchor"],
            "control_families": ["cross_patient_pseudopair"],
            "gate_checks": {"paired_control_checks": {}},
            "metrics_summary": {"paired_support": 1},
            "failure_reasons": ["legacy_test_failure"],
            "inputs": {"run_scope": "full_cohort"},
        },
    )
    return prepare_manifest_path, block0_bundle_path, p0_suitability_path


def _write_current_contract_block0_run(base: Path) -> tuple[Path, Path, Path]:
    prepare_manifest_path = _write_prepare_bundle(base / "p0_prepare_full")
    block0_dir = base / "p3_block0_full_current_contract"
    suitability_payload = {
        "task_name": "Task A current-contract suitability",
        "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
        "stage0_h5ad": str(base / "fixture.h5ad"),
        "report_scope": "pre_block0_data_suitability",
        "run_scope": "full_cohort",
        "artifact_state": "contract_passed",
        "block0_gate_status": "not_passed",
        "scientific_interpretation_allowed": False,
        "mass_mode": "uniform",
        "confirmatory_pair_families": ["TC-IM", "TC-PT"],
        "audit_pair_families": ["IM-PT"],
        "stage0_validation": {"artifact_state": "contract_passed"},
    }
    suitability_path = _write_json(
        block0_dir / "task_a_pre_block0_data_suitability.json",
        suitability_payload,
    )
    pair_metrics_path = _write_csv(
        block0_dir / "block0_pair_metrics.csv",
        [
            {
                "comparison_id": "block0_locality_gate::TC-IM::P01",
                "run_scope": "full_cohort",
                "pair_family": "TC-IM",
                "null_family": "TC-IM_randomized_target",
                "anchor_patient_id": "P01",
                "null_target_donor_patient_id": "P02",
                "source_domain": "TC",
                "target_domain": "IM",
                "n_source_observations": 2,
                "n_target_observations": 2,
                "count_stratum_key": "TC:2|IM:2",
                "selection_seed": 7,
                "null_assignment_status": "assigned",
                "null_assignment_reason": "",
                "real_fit_status": "ok",
                "null_fit_status": "ok",
                "real_defer_reason": "",
                "null_defer_reason": "",
                "real_total_continuity_mass": 19.0,
                "null_total_continuity_mass": 17.5,
                "delta_total_continuity_mass": 1.5,
                "real_total_depletion_mass": 6.0,
                "null_total_depletion_mass": 7.5,
                "delta_total_depletion_mass": -1.5,
                "real_total_emergence_mass": 0.2,
                "null_total_emergence_mass": 0.5,
                "delta_total_emergence_mass": -0.3,
            }
        ],
    )
    block0_bundle_path = _write_json(
        block0_dir / "block0_bundle.json",
        {
            "block": "block0_locality_gate",
            "status": "passed",
            "artifact_state": "contract_passed",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "run_scope": "full_cohort",
            "block0_passed": True,
            "config_fingerprint": "fixture-current-contract",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "output_dir": str(block0_dir),
            "bundle_path": str(block0_dir / "block0_bundle.json"),
            "pair_metrics_path": str(pair_metrics_path),
            "real_families": ["TC-IM"],
            "null_families": ["TC-IM_randomized_target"],
            "pre_block0_data_suitability": suitability_payload,
            "gate_checks": {
                "eligible_patients_positive": {"observed": 1, "passed": True},
                "fraction_real_total_continuity_mass_gt_null_above_half": {
                    "observed": 1.0,
                    "passed": True,
                    "threshold": 0.5,
                },
                "fraction_real_total_emergence_mass_lt_null_above_half": {
                    "observed": 1.0,
                    "passed": True,
                    "threshold": 0.5,
                },
                "full_cohort_scope_required_for_pass": {
                    "observed_run_scope": "full_cohort",
                    "passed": True,
                },
                "median_delta_total_continuity_mass_positive": {
                    "observed": 1.5,
                    "passed": True,
                    "threshold": 0.0,
                },
                "median_delta_total_emergence_mass_negative": {
                    "observed": -0.3,
                    "passed": True,
                    "threshold": 0.0,
                },
                "paired_support": {"observed": 1, "passed": True, "threshold": 1},
                "pre_block0_data_suitability_contract_passed": {
                    "observed_artifact_state": "contract_passed",
                    "passed": True,
                },
            },
            "metrics_summary": {
                "eligible_patients": 1,
                "required_support": 1,
                "gate_summary_quantities": [
                    "delta_total_continuity_mass",
                    "delta_total_emergence_mass",
                ],
                "real_family": {
                    "family_name": "TC-IM",
                    "fit_status_counts": {"ok": 1, "deferred": 0, "failed": 0},
                    "median_total_continuity_mass": 19.0,
                    "median_total_depletion_mass": 6.0,
                    "median_total_emergence_mass": 0.2,
                    "n_patients": 1,
                },
                "null_family": {
                    "family_name": "TC-IM_randomized_target",
                    "fit_status_counts": {"ok": 1, "deferred": 0, "failed": 0},
                    "median_total_continuity_mass": 17.5,
                    "median_total_depletion_mass": 7.5,
                    "median_total_emergence_mass": 0.5,
                    "n_patients": 1,
                },
                "paired_comparisons": {
                    "fraction_real_total_continuity_mass_gt_null": 1.0,
                    "fraction_real_total_depletion_mass_lt_null": 1.0,
                    "fraction_real_total_emergence_mass_lt_null": 1.0,
                    "median_delta_total_continuity_mass": 1.5,
                    "median_delta_total_depletion_mass": -1.5,
                    "median_delta_total_emergence_mass": -0.3,
                    "paired_support": 1,
                    "required_support": 1,
                },
            },
            "failure_reasons": [],
            "inputs": {
                "task_config": str(ROOT / "tasks" / "task_A" / "config.yaml"),
                "stage0_h5ad": str(base / "fixture.h5ad"),
                "run_scope": "full_cohort",
                "random_seed": 7,
                "real_family_definition": {
                    "pair_family": "TC-IM",
                    "source_domain": "TC",
                    "target_domain": "IM",
                    "construction": "task_a_stride_adapter_family_slice",
                    "fit_surface": "fit_stride",
                },
                "null_family_definition": {
                    "pair_family": "TC-IM_randomized_target",
                    "source_domain": "TC",
                    "target_domain": "IM",
                    "construction": "same_anchor_source_with_target_group_reassigned_from_different_patient",
                    "stratification_fields": [
                        "n_source_observations",
                        "n_target_observations",
                    ],
                    "donor_policy": "seeded_derangement_within_exact_count_strata",
                    "singleton_stratum_policy": "emit_null_fit_status_deferred_for_anchor_patient",
                },
                "gate_summary_quantities": {
                    "delta_total_continuity_mass": {
                        "definition": "sum(A_real) - sum(A_null)",
                        "decision_rule": "median > 0 and fraction_real_total_continuity_mass_gt_null > 0.5",
                    },
                    "delta_total_emergence_mass": {
                        "definition": "sum(e_real) - sum(e_null)",
                        "decision_rule": "median < 0 and fraction_real_total_emergence_mass_lt_null > 0.5",
                    },
                },
            },
        },
    )
    return prepare_manifest_path, block0_bundle_path, suitability_path


def _write_block1_bundle(base: Path) -> Path:
    block1_dir = base / "block1"
    community_dir = block1_dir / "community_correspondence"
    tables_dir = community_dir / "tables"
    mapping_path = _write_json(
        block1_dir / "block1_stage0_mapping.json",
        {
            "field_mapping": {
                "patient_id_key": "patient_id",
                "cell_subtype_key": "cell_type",
                "state_id_key": "proto_id",
            },
            "patient_ids": ["P01"],
            "family_summaries": [],
            "real_data_crosswalk": {"crosswalk": []},
        },
    )
    core_fit_path = _write_csv(
        block1_dir / "block1_core_fit_dry_run.csv",
        [
            {
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "patient_id": "P01",
                "implementation_tier": "canonical_full",
                "fit_surface": "fit_stride",
                "fit_status": "ok",
                "bridge_realized": True,
                "defer_reason": "",
                "uncertainty_status": "",
                "cohort_recurrence_fit_status": "ok",
                "n_recurrence_families": 1,
                "n_recurrence_used_patients": 1,
                "source_domain": "TC",
                "target_domain": "IM",
            }
        ],
    )
    recurrence_summary_path = _write_json(
        block1_dir / "block1_recurrence_summary.json",
        {
            "fit_surface": "fit_stride",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "cohort_recurrence_fit_status": "ok",
            "cohort_recurrence_fit_status_by_pair_family": {"TC-IM": "ok"},
            "cohort_recurrence_family_count": 1,
            "cohort_recurrence_family_count_by_pair_family": {"TC-IM": 1},
            "n_recurrence_used_patients": 1,
            "n_recurrence_used_patients_by_pair_family": {"TC-IM": 1},
            "pair_families": [
                {
                    "pair_family": "TC-IM",
                    "implementation_tier": "canonical_full",
                    "fit_surface": "fit_stride",
                    "cohort_recurrence_fit_status": "ok",
                    "n_recurrence_families": 1,
                    "n_recurrence_used_patients": 1,
                    "recurrence_unit": "patient",
                    "basis_dim": 2,
                    "patient_ids": ["P01"],
                    "used_patient_ids": ["P01"],
                    "family_ids": ["family_0"],
                    "metadata": {},
                }
            ],
        },
    )
    recurrence_families_path = _write_json(
        block1_dir / "block1_recurrence_families.json",
        [
            {
                "pair_family": "TC-IM",
                "family_id": "family_0",
                "fit_status": "ok",
                "support_n_patients": 1,
                "within_family_dispersion": 0.0,
                "member_patient_ids": ["P01"],
                "template_A": [[0.5, 0.5], [0.2, 0.8]],
                "template_d": [0.1, 0.1],
                "template_e": [0.0, 0.1],
            }
        ],
    )
    recurrence_embeddings_path = _write_csv(
        block1_dir / "block1_recurrence_embeddings.csv",
        [
            {
                "pair_family": "TC-IM",
                "patient_id": "P01",
                "fit_status": "ok",
                "used_for_recurrence": True,
                "coord_1": 0.0,
                "coord_2": 0.0,
            }
        ],
    )
    family_summary_path = _write_csv(
        block1_dir / "block1_family_summary.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "scale": "burden_weighted",
                "value": 0.5,
                "eligible_entity_axis": "source",
                "eligible_entity_count": 1,
                "burden_total": 1.0,
            }
        ],
    )
    source_summary_path = _write_csv(
        block1_dir / "block1_source_community_summary.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "source_community_id": 0,
                "source_burden": 1.0,
                "source_weight": 1.0,
                "self_retention": 0.5,
                "depletion": 0.2,
                "off_diagonal_remodeling": 0.3,
                "self_retention_burden": 0.5,
                "depletion_burden": 0.2,
                "off_diagonal_burden": 0.3,
                "top_target_1_id": 1,
                "top_target_1_value": 0.3,
            }
        ],
    )
    target_summary_path = _write_csv(
        block1_dir / "block1_target_community_summary.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "target_community_id": 1,
                "target_burden": 1.0,
                "target_weight": 1.0,
                "incoming_matched_operator": 0.5,
                "incoming_matched_burden": 0.5,
                "emergence_tendency": 0.1,
                "emergence_burden": 0.1,
            }
        ],
    )
    family_comparison_path = _write_csv(
        block1_dir / "block1_confirmatory_family_comparison.csv",
        [
            {
                "patient_id": "P01",
                "pair_family_left": "TC-IM",
                "pair_family_right": "TC-PT",
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "scale": "burden_weighted",
                "eligible_entity_axis": "source",
                "tc_im_value": 0.5,
                "tc_pt_value": 0.4,
                "delta_tc_im_minus_tc_pt": 0.1,
                "contrast_direction": "positive",
                "comparison_status": "estimable",
                "comparison_scope_role": "confirmatory",
            }
        ],
    )
    source_comparison_path = _write_csv(
        block1_dir / "block1_exploratory_source_community_comparison.csv",
        [
            {
                "patient_id": "P01",
                "pair_family_left": "TC-IM",
                "pair_family_right": "TC-PT",
                "source_community_id": 0,
                "summary_name": "self_retention",
                "summary_role": "exploratory_supportive",
                "tc_im_value": 0.5,
                "tc_pt_value": 0.4,
                "delta_tc_im_minus_tc_pt": 0.1,
                "comparison_status": "estimable",
                "comparison_scope_role": "exploratory_supportive",
            }
        ],
    )
    target_comparison_path = _write_csv(
        block1_dir / "block1_exploratory_target_community_comparison.csv",
        [
            {
                "patient_id": "P01",
                "pair_family_left": "TC-IM",
                "pair_family_right": "TC-PT",
                "target_community_id": 1,
                "summary_name": "emergence_tendency",
                "summary_role": "exploratory_supportive",
                "tc_im_value": 0.1,
                "tc_pt_value": 0.0,
                "delta_tc_im_minus_tc_pt": 0.1,
                "comparison_status": "estimable",
                "comparison_scope_role": "exploratory_supportive",
            }
        ],
    )
    correspondence_manifest_path = _write_json(
        community_dir / "block1_community_correspondence_manifest.json",
        {
            "workflow_name": "write_task_a_block1_bundle",
            "packet_role": "objective_community_correspondence",
            "scientific_interpretation_allowed": False,
            "artifact_state": "evidence_ready",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "community_id_key": "proto_id",
            "cell_subtype_key": "cell_type",
            "patient_id_key": "patient_id",
            "configured_state_ids": [0, 1],
            "observed_community_ids": [0, 1],
            "output_index": str(community_dir / "block1_community_correspondence_index.csv"),
        },
    )
    _write_csv(
        community_dir / "block1_community_correspondence_index.csv",
        [
            {
                "relative_path": "community_correspondence/block1_community_correspondence_manifest.json",
                "artifact_kind": "manifest",
                "category": "provenance",
                "format": "json",
                "description": "Block 1 objective community-correspondence manifest",
            },
            {
                "relative_path": "community_correspondence/tables/community_id_crosswalk.csv",
                "artifact_kind": "table",
                "category": "community_crosswalk",
                "format": "csv",
                "description": "Community-id crosswalk",
            },
        ],
    )
    _write_csv(
        tables_dir / "community_cell_subtype_counts.csv",
        [{"community_id": 0, "TC_EpCAM": 10}],
    )
    _write_csv(
        tables_dir / "community_cell_subtype_row_fractions.csv",
        [{"community_id": 0, "TC_EpCAM": 1.0}],
    )
    _write_csv(
        tables_dir / "source_community_major_targets.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "source_community_id": 0,
                "source_burden": 1.0,
                "source_weight": 1.0,
                "target_rank": 1,
                "target_community_id": 1,
                "target_operator_value": 0.3,
            }
        ],
    )
    _write_csv(
        tables_dir / "source_community_burden_components.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "source_community_id": 0,
                "source_burden": 1.0,
                "source_weight": 1.0,
                "matched_burden": 0.8,
                "self_retention_burden": 0.5,
                "off_diagonal_burden": 0.3,
                "depletion_burden": 0.2,
            }
        ],
    )
    _write_csv(
        tables_dir / "target_community_burden_components.csv",
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "target_community_id": 1,
                "target_burden": 1.0,
                "target_weight": 1.0,
                "incoming_matched_operator": 0.5,
                "incoming_matched_burden": 0.5,
                "emergence_tendency": 0.1,
                "emergence_burden": 0.1,
            }
        ],
    )
    _write_csv(
        tables_dir / "community_id_crosswalk.csv",
        [
            {
                "community_id": 0,
                "configured_state_index": 0,
                "observed_in_stage0": True,
                "n_stage0_cells": 10,
                "n_stage0_patients": 1,
                "observed_in_source_summary": True,
                "n_source_summary_rows": 1,
                "n_source_summary_patients": 1,
                "source_summary_pair_families": "TC-IM",
                "observed_in_target_summary": False,
                "n_target_summary_rows": 0,
                "n_target_summary_patients": 0,
                "target_summary_pair_families": "",
            }
        ],
    )
    _write_json(
        block1_dir / "block1_workflow_manifest.json",
        {
            "block": "block1_continuity_backbone",
            "scientific_role": "real_data_biological_discovery",
            "status": "active",
            "artifact_state": "evidence_ready",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "fit_surface": "fit_stride",
            "block0_bundle_path": str(base / "block0_bundle.json"),
            "block0_gate_status": "passed",
            "config_fingerprint": "test",
            "bundle_path": str(block1_dir / "block1_bundle.json"),
            "core_fit_dry_run_path": str(core_fit_path),
            "mapping_manifest_path": str(mapping_path),
            "recurrence_summary_path": str(recurrence_summary_path),
            "recurrence_families_path": str(recurrence_families_path),
            "recurrence_embeddings_path": str(recurrence_embeddings_path),
            "family_summary_path": str(family_summary_path),
            "source_community_summary_path": str(source_summary_path),
            "target_community_summary_path": str(target_summary_path),
            "confirmatory_family_comparison_path": str(family_comparison_path),
            "exploratory_source_community_comparison_path": str(source_comparison_path),
            "exploratory_target_community_comparison_path": str(target_comparison_path),
            "community_correspondence_manifest_path": str(correspondence_manifest_path),
            "community_correspondence_index_path": str(community_dir / "block1_community_correspondence_index.csv"),
            "summary_contract_version": "task_a_block1_summary_v1",
            "paired_comparison_contract_version": "task_a_block1_paired_comparison_v1",
            "proof_carrying_family_summaries": ["self_retention", "depletion"],
            "supportive_family_summaries": ["off_diagonal_remodeling", "emergence"],
            "family_summary_scales": ["burden_weighted", "community_mean"],
            "cohort_recurrence_fit_status": "ok",
            "cohort_recurrence_fit_status_by_pair_family": {"TC-IM": "ok"},
            "cohort_recurrence_family_count": 1,
            "cohort_recurrence_family_count_by_pair_family": {"TC-IM": 1},
            "n_recurrence_used_patients": 1,
            "n_recurrence_used_patients_by_pair_family": {"TC-IM": 1},
        },
    )
    return _write_json(
        block1_dir / "block1_bundle.json",
        {
            "block": "block1_continuity_backbone",
            "scientific_role": "real_data_biological_discovery",
            "status": "active",
            "artifact_state": "evidence_ready",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "fit_surface": "fit_stride",
            "block0_bundle_path": str(base / "block0_bundle.json"),
            "block0_gate_status": "passed",
            "config_fingerprint": "test",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "output_dir": str(block1_dir),
            "mapping_manifest_path": str(mapping_path),
            "core_fit_dry_run_path": str(core_fit_path),
            "recurrence_summary_path": str(recurrence_summary_path),
            "recurrence_families_path": str(recurrence_families_path),
            "recurrence_embeddings_path": str(recurrence_embeddings_path),
            "family_summary_path": str(family_summary_path),
            "source_community_summary_path": str(source_summary_path),
            "target_community_summary_path": str(target_summary_path),
            "confirmatory_family_comparison_path": str(family_comparison_path),
            "exploratory_source_community_comparison_path": str(source_comparison_path),
            "exploratory_target_community_comparison_path": str(target_comparison_path),
            "community_correspondence_manifest_path": str(correspondence_manifest_path),
            "community_correspondence_index_path": str(community_dir / "block1_community_correspondence_index.csv"),
            "bundle_path": str(block1_dir / "block1_bundle.json"),
            "pair_families": ["TC-IM", "TC-PT"],
            "confirmatory_pair_families": ["TC-IM", "TC-PT"],
            "summary_contract_version": "task_a_block1_summary_v1",
            "paired_comparison_contract_version": "task_a_block1_paired_comparison_v1",
            "proof_carrying_family_summaries": ["self_retention", "depletion"],
            "supportive_family_summaries": ["off_diagonal_remodeling", "emergence"],
            "family_summary_scales": ["burden_weighted", "community_mean"],
            "source_eligibility_rule": "mu_minus > 0",
            "target_eligibility_rule": "mu_plus > 0",
            "fit_result_counts": {"ok": 1},
            "cohort_recurrence_fit_status": "ok",
            "cohort_recurrence_fit_status_by_pair_family": {"TC-IM": "ok"},
            "cohort_recurrence_family_count": 1,
            "cohort_recurrence_family_count_by_pair_family": {"TC-IM": 1},
            "n_recurrence_used_patients": 1,
            "n_recurrence_used_patients_by_pair_family": {"TC-IM": 1},
            "stage0_mapping": {"field_mapping": {"patient_id_key": "patient_id"}},
        },
    )


def _write_block2_manifest(base: Path, *, block1_bundle_path: Path) -> Path:
    block2_dir = base / "block2"
    summary_path = _write_csv(
        block2_dir / "block2_bounded_audit_summary.csv",
        [
            {
                "block": "block2_bounded_audit",
                "summary_scope": "family",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "burden_weighted",
                "community_id": "",
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": 0.1,
                "primary_routes_executed": 2,
                "primary_routes_robust": 2,
                "primary_routes_partial_or_better": 2,
                "worst_direction_recovery_rate": 0.8,
                "worst_estimable_replicate_fraction": 1.0,
                "worst_median_replicate_support_fraction": 0.75,
                "overall_robustness_call": "robust",
                "primary_route_names": "leave_some_out|patient_subsample",
            }
        ],
    )
    contract_path = _write_csv(
        block2_dir / "block2_contract_audit.csv",
        [{"check": "config_enables_block2", "passed": True, "detail": "True"}],
    )
    replicate_manifest_path = _write_csv(
        block2_dir / "block2_replicate_manifest.csv",
        [
            {
                "route_name": "patient_subsample",
                "route_group": "primary",
                "replicate_index": 0,
                "replicate_label": "patient_subsample_0000",
                "selection_seed": 17,
                "patient_subset_json": "[\"P01\"]",
                "dropped_roi_ids_json": "[]",
                "route_note": "",
                "route_status": "executed",
                "failure_reason": "",
                "n_patients_retained": 1,
                "n_rois_retained": 2,
                "n_cells_retained": 8,
            }
        ],
    )
    family_robustness_path = _write_csv(
        block2_dir / "block2_family_robustness.csv",
        [
            {
                "block": "block2_bounded_audit",
                "route_name": "patient_subsample",
                "route_group": "primary",
                "finding_id": "family::self_retention::burden_weighted",
                "summary_scope": "family",
                "finding_priority": "primary",
                "summary_name": "self_retention",
                "scale": "burden_weighted",
                "community_id": "",
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_estimable_n": 1,
                "full_data_support_n": 1,
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": 0.1,
                "full_data_rank": "",
                "full_data_tc_im_mode_top_target_1_id": "",
                "full_data_tc_pt_mode_top_target_1_id": "",
                "n_replicates_planned": 1,
                "n_replicates_executed": 1,
                "estimable_replicate_fraction": 1.0,
                "direction_recovery_rate": 1.0,
                "median_replicate_support_fraction": 1.0,
                "median_replicate_delta": 0.1,
                "median_replicate_rank": "",
                "replicate_rank_iqr": "",
                "tc_im_mode_recovery_rate": "",
                "tc_pt_mode_recovery_rate": "",
                "robustness_call": "robust",
            }
        ],
    )
    source_robustness_path = _write_csv(
        block2_dir / "block2_source_community_robustness.csv",
        [
            {
                "block": "block2_bounded_audit",
                "route_name": "patient_subsample",
                "route_group": "primary",
                "finding_id": "source_community::0::self_retention",
                "summary_scope": "source_community",
                "finding_priority": "secondary",
                "summary_name": "self_retention",
                "scale": "community",
                "community_id": 0,
                "full_data_direction": "tc_im_gt_tc_pt",
                "full_data_estimable_n": 1,
                "full_data_support_n": 1,
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": 0.1,
                "full_data_rank": 1.0,
                "full_data_tc_im_mode_top_target_1_id": 1,
                "full_data_tc_pt_mode_top_target_1_id": 1,
                "n_replicates_planned": 1,
                "n_replicates_executed": 1,
                "estimable_replicate_fraction": 1.0,
                "direction_recovery_rate": 1.0,
                "median_replicate_support_fraction": 1.0,
                "median_replicate_delta": 0.1,
                "median_replicate_rank": 1.0,
                "replicate_rank_iqr": 0.0,
                "tc_im_mode_recovery_rate": 1.0,
                "tc_pt_mode_recovery_rate": 1.0,
                "robustness_call": "robust",
            }
        ],
    )
    target_robustness_path = _write_csv(
        block2_dir / "block2_target_community_robustness.csv",
        [
            {
                "block": "block2_bounded_audit",
                "route_name": "patient_subsample",
                "route_group": "primary",
                "finding_id": "target_community::1::incoming_matched_operator",
                "summary_scope": "target_community",
                "finding_priority": "secondary",
                "summary_name": "incoming_matched_operator",
                "scale": "community",
                "community_id": 1,
                "full_data_direction": "tc_im_lt_tc_pt",
                "full_data_estimable_n": 1,
                "full_data_support_n": 1,
                "full_data_support_fraction": 1.0,
                "full_data_median_delta": -0.1,
                "full_data_rank": 1.0,
                "full_data_tc_im_mode_top_target_1_id": "",
                "full_data_tc_pt_mode_top_target_1_id": "",
                "n_replicates_planned": 1,
                "n_replicates_executed": 1,
                "estimable_replicate_fraction": 1.0,
                "direction_recovery_rate": 1.0,
                "median_replicate_support_fraction": 1.0,
                "median_replicate_delta": -0.1,
                "median_replicate_rank": 1.0,
                "replicate_rank_iqr": 0.0,
                "tc_im_mode_recovery_rate": "",
                "tc_pt_mode_recovery_rate": "",
                "robustness_call": "robust",
            }
        ],
    )
    block1_payload = json.loads(block1_bundle_path.read_text(encoding="utf-8"))
    return _write_json(
        block2_dir / "block2_bounded_audit_manifest.json",
        {
            "block": "block2_bounded_audit",
            "scientific_role": "robustness_over_block1_summaries",
            "status": "active",
            "artifact_state": "evidence_ready",
            "scientific_interpretation_allowed": False,
            "claim_scope": "block1_summary_robustness",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "fit_surface": "fit_stride",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "config_fingerprint": "test",
            "output_dir": str(block2_dir),
            "bundle_path": str(block2_dir / "block2_bounded_audit_manifest.json"),
            "block1_bundle_path": str(block1_bundle_path),
            "block0_bundle_path": str(block1_payload["block0_bundle_path"]),
            "block1_stage0_mapping_path": str(block1_payload["mapping_manifest_path"]),
            "block1_core_fit_dry_run_path": str(block1_payload["core_fit_dry_run_path"]),
            "block1_recurrence_summary_path": str(block1_payload["recurrence_summary_path"]),
            "block1_recurrence_families_path": str(block1_payload["recurrence_families_path"]),
            "block1_recurrence_embeddings_path": str(block1_payload["recurrence_embeddings_path"]),
            "block1_family_summary_path": str(block1_payload["family_summary_path"]),
            "block1_source_community_summary_path": str(block1_payload["source_community_summary_path"]),
            "block1_target_community_summary_path": str(block1_payload["target_community_summary_path"]),
            "block1_confirmatory_family_comparison_path": str(
                block1_payload["confirmatory_family_comparison_path"]
            ),
            "block1_exploratory_source_community_comparison_path": str(
                block1_payload["exploratory_source_community_comparison_path"]
            ),
            "block1_exploratory_target_community_comparison_path": str(
                block1_payload["exploratory_target_community_comparison_path"]
            ),
            "summary_path": str(summary_path),
            "contract_path": str(contract_path),
            "replicate_manifest_path": str(replicate_manifest_path),
            "family_robustness_path": str(family_robustness_path),
            "source_community_robustness_path": str(source_robustness_path),
            "target_community_robustness_path": str(target_robustness_path),
            "summary_rows": 1,
            "replicate_rows": 1,
            "block1_cohort_recurrence_fit_status": "ok",
            "block1_cohort_recurrence_fit_status_by_pair_family": {"TC-IM": "ok"},
            "block1_cohort_recurrence_family_count": 1,
            "block1_cohort_recurrence_family_count_by_pair_family": {"TC-IM": 1},
            "block1_n_recurrence_used_patients": 1,
            "block1_n_recurrence_used_patients_by_pair_family": {"TC-IM": 1},
            "primary_routes": ["patient_subsample", "leave_some_out"],
        },
    )


def test_result_packet_workflow_is_exported() -> None:
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.workflows, "write_task_a_result_packet")
    assert hasattr(tasks.task_A.workflows, "validate_task_a_result_packet")


def test_block0_review_helper_imports() -> None:
    from tasks.task_A.block0.review import write_block0_review_surface

    assert callable(write_block0_review_surface)


def test_block2_review_helper_imports() -> None:
    from tasks.task_A.block2.review import write_block2_review_surface

    assert callable(write_block2_review_surface)


def test_result_packet_packages_available_atlas_block0_and_missing_block1(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        output_dir=tmp_path / "packet",
    )

    assert packet.manifest_path.exists()
    assert packet.index_path.exists()
    assert packet.human_index_path.exists()
    assert (packet.packet_root / "atlas" / "atlas_layer_manifest.json").exists()
    assert (packet.packet_root / "block0" / "block0_layer_manifest.json").exists()
    assert (packet.packet_root / "block1" / "block1_layer_manifest.json").exists()
    assert (packet.packet_root / "block2" / "block2_layer_manifest.json").exists()
    assert not (packet.packet_root / "block3").exists()
    assert (packet.packet_root / "atlas" / "bundle" / "task_a_descriptive_atlas_manifest.json").exists()
    assert (packet.packet_root / "block0" / "bundle" / "block0_bundle.json").exists()
    assert (packet.packet_root / "block0" / "provenance" / "p0_suitability" / "task_a_pre_block0_data_suitability.json").exists()
    assert (packet.packet_root / "block0" / "review" / "block0_objective_review_manifest.json").exists()
    assert (packet.packet_root / "block0" / "BLOCK0_RESULTS_INDEX.md").exists()

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    manifest_payload = json.loads(packet.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["included_layers"] == ["atlas", "block0", "block1", "block2"]
    assert manifest_payload["deferred_layers"] == ["block3"]
    assert manifest_payload["surface_lineage"]["block0"]["implementation_tier"] == "legacy_live_run"
    assert manifest_payload["surface_lineage"]["block0"]["evidence_lineage"] == "proxy_history"
    assert {"implementation_tier", "evidence_lineage"}.issubset(index_df.columns)
    block0_bundle_row = index_df.loc[index_df["packet_relative_path"] == "block0/bundle/block0_bundle.json"].iloc[0]
    assert block0_bundle_row["contract_alignment"] == "legacy_live_run"
    assert block0_bundle_row["implementation_tier"] == "legacy_live_run"
    assert block0_bundle_row["evidence_lineage"] == "proxy_history"
    suitability_row = index_df.loc[
        index_df["packet_relative_path"]
        == "block0/provenance/p0_suitability/task_a_pre_block0_data_suitability.json"
    ].iloc[0]
    assert suitability_row["implementation_tier"] == "canonical_full"
    assert suitability_row["evidence_lineage"] == "canonical_rerun"
    legacy_review_manifest = json.loads(
        (packet.packet_root / "block0" / "review" / "block0_objective_review_manifest.json").read_text(encoding="utf-8")
    )
    assert legacy_review_manifest["schema_variant"] == "legacy_live_run"
    missing_names = {row["artifact_name"] for row in legacy_review_manifest["missing_artifacts"]}
    assert "block0_patient_review_table.csv" in missing_names
    assert "block0_gate_summary.csv" in missing_names

    missing_block1_row = index_df.loc[index_df["expected_relative_path"] == "block1_bundle.json"].iloc[0]
    assert missing_block1_row["artifact_status"] == "missing_on_disk"
    assert missing_block1_row["packet_relative_path"] == ""
    missing_block2_row = index_df.loc[
        index_df["expected_relative_path"] == "block2_bounded_audit_manifest.json"
    ].iloc[0]
    assert missing_block2_row["artifact_status"] == "missing_on_disk"
    assert missing_block2_row["packet_relative_path"] == ""


def test_result_packet_generates_current_contract_block0_review_tables(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_current_contract_block0_run(
        tmp_path / "block0_run_current"
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        output_dir=tmp_path / "packet_current",
    )

    patient_review_path = packet.packet_root / "block0" / "review" / "block0_patient_review_table.csv"
    family_summary_path = packet.packet_root / "block0" / "review" / "block0_family_summary.csv"
    gate_summary_path = packet.packet_root / "block0" / "review" / "block0_gate_summary.csv"
    null_provenance_path = packet.packet_root / "block0" / "review" / "block0_null_provenance.csv"
    repro_path = packet.packet_root / "block0" / "review" / "block0_reproducibility_metadata.json"
    block0_human_index_path = packet.packet_root / "block0" / "BLOCK0_RESULTS_INDEX.md"

    assert patient_review_path.exists()
    assert family_summary_path.exists()
    assert gate_summary_path.exists()
    assert null_provenance_path.exists()
    assert repro_path.exists()
    assert block0_human_index_path.exists()

    patient_df = pd.read_csv(patient_review_path)
    assert patient_df["anchor_patient_id"].tolist() == ["P01"]
    assert patient_df["real_total_continuity_mass_gt_null"].tolist() == [True]
    assert patient_df["real_total_emergence_mass_lt_null"].tolist() == [True]

    gate_df = pd.read_csv(gate_summary_path)
    assert set(gate_df["quantity_name"]) == {
        "delta_total_continuity_mass",
        "delta_total_depletion_mass",
        "delta_total_emergence_mass",
    }
    continuity_row = gate_df.loc[gate_df["quantity_name"] == "delta_total_continuity_mass"].iloc[0]
    assert bool(continuity_row["participates_in_pass_decision"]) is True
    assert float(continuity_row["median_delta_value"]) == pytest.approx(1.5)

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    gate_row = index_df.loc[
        index_df["packet_relative_path"] == "block0/review/block0_gate_summary.csv"
    ].iloc[0]
    assert gate_row["review_role"] == "proof_carrying"
    assert gate_row["analysis_level"] == "cohort_level"
    assert gate_row["family_surface_role"] == "comparison_surface"
    assert gate_row["proof_carrying_status"] == "all"

    block0_review_index = pd.read_csv(packet.packet_root / "block0" / "block0_review_index.csv", keep_default_na=False)
    assert {"review_role", "analysis_level", "family_surface_role"}.issubset(block0_review_index.columns)

    review_manifest = json.loads(
        (packet.packet_root / "block0" / "review" / "block0_objective_review_manifest.json").read_text(encoding="utf-8")
    )
    assert review_manifest["schema_variant"] == "current_contract_available"
    assert review_manifest["missing_artifacts"] == []


def test_result_packet_can_mirror_optional_block1_bundle(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")
    block1_bundle_path = _write_block1_bundle(tmp_path / "block1_run")

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        block1_bundle_path=block1_bundle_path,
        output_dir=tmp_path / "packet",
    )

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    block1_bundle_row = index_df.loc[index_df["expected_relative_path"] == "block1_bundle.json"].iloc[0]
    assert block1_bundle_row["artifact_status"] == "available"
    assert block1_bundle_row["packet_relative_path"] == "block1/bundle/block1_bundle.json"
    assert block1_bundle_row["implementation_tier"] == "canonical_full"
    assert block1_bundle_row["evidence_lineage"] == "canonical_rerun"
    recurrence_row = index_df.loc[
        index_df["packet_relative_path"] == "block1/bundle/block1_recurrence_summary.json"
    ].iloc[0]
    assert recurrence_row["implementation_tier"] == "canonical_full"
    assert recurrence_row["evidence_lineage"] == "canonical_rerun"
    assert (packet.packet_root / "block1" / "bundle" / "block1_bundle.json").exists()
    assert (packet.packet_root / "block1" / "bundle" / "block1_recurrence_summary.json").exists()
    assert (packet.packet_root / "block1" / "bundle" / "block1_recurrence_families.json").exists()
    assert (packet.packet_root / "block1" / "bundle" / "block1_recurrence_embeddings.csv").exists()
    assert (
        packet.packet_root
        / "block1"
        / "bundle"
        / "community_correspondence"
        / "tables"
        / "community_id_crosswalk.csv"
    ).exists()


def test_result_packet_can_mirror_optional_block2_manifest(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")
    block1_bundle_path = _write_block1_bundle(tmp_path / "block1_run")
    block2_manifest_path = _write_block2_manifest(
        tmp_path / "block2_run",
        block1_bundle_path=block1_bundle_path,
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        block1_bundle_path=block1_bundle_path,
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "packet",
    )

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    manifest_payload = json.loads(packet.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["included_layers"] == ["atlas", "block0", "block1", "block2"]
    assert manifest_payload["deferred_layers"] == ["block3"]
    block2_manifest_row = index_df.loc[
        index_df["expected_relative_path"] == "block2_bounded_audit_manifest.json"
    ].iloc[0]
    assert block2_manifest_row["artifact_status"] == "available"
    assert block2_manifest_row["packet_relative_path"] == "block2/bundle/block2_bounded_audit_manifest.json"
    assert block2_manifest_row["implementation_tier"] == "canonical_full"
    assert block2_manifest_row["evidence_lineage"] == "canonical_rerun"
    summary_row = index_df.loc[
        index_df["packet_relative_path"] == "block2/bundle/block2_bounded_audit_summary.csv"
    ].iloc[0]
    assert summary_row["review_role"] == "proof_carrying"
    assert summary_row["claim_scope"] == "robustness"
    assert summary_row["analysis_level"] == "cohort_decision"
    assert (packet.packet_root / "block2" / "bundle" / "block2_family_robustness.csv").exists()
    assert (packet.packet_root / "block2" / "block2_review_index.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_route_summary.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_primary_finding_review_table.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_family_review_table.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_source_primary_review_table.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_target_primary_review_table.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_call_semantics.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_artifact_index.csv").exists()
    assert (packet.packet_root / "block2" / "review" / "block2_objective_review_manifest.json").exists()
    assert (packet.packet_root / "block2" / "BLOCK2_RESULTS_INDEX.md").exists()

    source_primary_row = index_df.loc[
        index_df["packet_relative_path"] == "block2/review/block2_source_primary_review_table.csv"
    ].iloc[0]
    assert source_primary_row["analysis_level"] == "source_community"
    assert source_primary_row["review_role"] == "proof_carrying"

    artifact_index_df = pd.read_csv(
        packet.packet_root / "block2" / "review" / "block2_artifact_index.csv",
        keep_default_na=False,
    )
    assert "artifact_evidence_class" in artifact_index_df.columns
    assert "robustness_routes" in artifact_index_df.columns
    raw_summary_index_row = artifact_index_df.loc[
        artifact_index_df["artifact_name"] == "block2_bounded_audit_summary.csv"
    ].iloc[0]
    assert raw_summary_index_row["analysis_level"] == "cohort_decision"
    assert raw_summary_index_row["artifact_evidence_class"] == "proof_carrying"

    semantics_df = pd.read_csv(
        packet.packet_root / "block2" / "review" / "block2_call_semantics.csv",
        keep_default_na=False,
    )
    failure_note_row = semantics_df.loc[
        (semantics_df["call_field"] == "failure_note") & (semantics_df["call_value"] == "failure")
    ].iloc[0]
    assert failure_note_row["execution_failure_equivalent"] in {False, "False"}


def test_result_packet_workflow_main_writes_manifest(tmp_path: Path) -> None:
    from tasks.task_A.workflows.package_results import main

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")
    output_dir = tmp_path / "packet"

    main(
        [
            "--atlas-manifest",
            str(atlas_manifest_path),
            "--prepare-manifest",
            str(prepare_manifest_path),
            "--block0-bundle",
            str(block0_bundle_path),
            "--block0-suitability-report",
            str(suitability_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "task_a_result_packet_manifest.json").exists()


def test_result_packet_workflow_rejects_block3_manifest_until_clean_bridge_exists(tmp_path: Path) -> None:
    from tasks.task_A.workflows.package_results import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")
    block3_manifest_path = _write_json(
        tmp_path / "block3" / "block3_method_validation_manifest.json",
        {"block": "block3_method_validation", "artifact_state": "evidence_ready"},
    )

    with pytest.raises(
        ContractError,
        match="Block 3 packet integration is deferred / non-authority / pending clean bridge spec",
    ):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_bundle_path=block0_bundle_path,
            block0_suitability_report_path=suitability_path,
            block3_manifest_path=block3_manifest_path,
            output_dir=tmp_path / "packet",
        )


def test_result_packet_validation_fails_when_mirrored_file_is_removed(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import validate_task_a_result_packet, write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_bundle_path, suitability_path = _write_block0_run(tmp_path / "block0_run")

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_bundle_path=block0_bundle_path,
        block0_suitability_report_path=suitability_path,
        output_dir=tmp_path / "packet",
    )

    mirrored_block0_bundle = packet.packet_root / "block0" / "bundle" / "block0_bundle.json"
    mirrored_block0_bundle.unlink()

    with pytest.raises(ContractError, match="Referenced packet artifact is missing"):
        validate_task_a_result_packet(packet.manifest_path)
