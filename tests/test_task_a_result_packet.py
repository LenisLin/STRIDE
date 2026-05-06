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


def _write_invalid_block0_payload(base: Path) -> tuple[Path, Path]:
    prepare_manifest_path = _write_prepare_bundle(base / "p0_prepare_full")
    invalid_payload_path = _write_json(
        base / "retired_block0_payload.json",
        {
            "block": "retired_block0_payload",
            "status": "retired",
        },
    )
    return prepare_manifest_path, invalid_payload_path


def _write_block0_calibration_run(base: Path) -> tuple[Path, Path]:
    prepare_manifest_path = _write_prepare_bundle(base / "p0_prepare_full")
    block0_dir = base / "block0_calibration"
    execution_manifest_path = _write_json(
        block0_dir / "block0_execution_manifest.json",
        {
            "task_name": "block0_execution_cache",
            "fit_cache_path": str(block0_dir / "block0_fit_cache.npz"),
            "fit_cache_index_path": str(block0_dir / "block0_fit_cache_index.csv"),
            "fit_cache_sha256": "fixture-cache-sha256",
            "fit_cache_index_sha256": "fixture-index-sha256",
        },
    )
    _write_text(block0_dir / "block0_fit_cache.npz", "fixture cache is not mirrored\n")
    _write_csv(
        block0_dir / "block0_fit_cache_index.csv",
        [
            {
                "record_id": 0,
                "fit_label": "real",
                "permutation_index": "",
                "patient_id": "P01",
                "fit_status": "ok",
            }
        ],
    )
    patient_calibration_path = _write_csv(
        block0_dir / "block0_patient_calibration.csv",
        [
            {
                "patient_id": "P01",
                "run_scope": "demo_subset",
                "real_family": "TC-IM",
                "null_family": "TC-IM_within_patient_domain_label_permutation_null",
                "n_permutations": 2,
                "real_fit_status": "ok",
                "null_fit_status": "ok",
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "eligible_entity_axis": "source",
                "scale": "burden_weighted",
                "reference_stat": "median",
                "expected_tail": "left",
                "real_value": 0.96,
                "null_reference": 0.962,
                "empirical_p_value": 0.3333333333,
                "primary_tail_fraction": 0.0,
                "opposite_tail_fraction": 1.0,
                "effect_delta": -0.002,
                "effect_ratio": 0.9979209979,
                "effect_ratio_status": "estimable",
                "readiness_status": "diagnostic",
            }
        ],
    )
    metric_summary_path = _write_csv(
        block0_dir / "block0_metric_summary.csv",
        [
            {
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "eligible_entity_axis": "source",
                "scale": "burden_weighted",
                "cohort_stat": "median",
                "expected_tail": "left",
                "real_value": 0.96,
                "null_reference": 0.962,
                "empirical_p_value": 0.3333333333,
                "primary_tail_fraction": 0.0,
                "opposite_tail_fraction": 1.0,
                "effect_delta": -0.002,
                "effect_ratio": 0.9979209979,
                "effect_ratio_status": "estimable",
                "n_patient_delta_positive": 0,
                "n_patient_delta_negative": 1,
                "n_patient_delta_zero": 0,
                "readiness_status": "diagnostic",
            }
        ],
    )
    calibration_manifest_path = _write_json(
        block0_dir / "block0_calibration_manifest.json",
        {
            "task_name": "block0_calibration",
            "config_path": str(ROOT / "tasks" / "task_A" / "config.yaml"),
            "stage0_h5ad": str(base / "fixture.h5ad"),
            "run_scope": "full_cohort",
            "n_permutations": 2,
            "master_seed": 7,
            "seed_derivation_policy": "sha256(namespace|master_seed|patient_id|permutation_index)",
            "real_family": "TC-IM",
            "null_family": "TC-IM_within_patient_domain_label_permutation_null",
            "permutation_policy": "within-patient TC/IM domain-label permutation",
            "summary_roles": {
                "self_retention": "proof_carrying",
                "depletion": "proof_carrying",
                "off_diagonal_remodeling": "diagnostic_supportive",
                "emergence": "supportive",
            },
            "fit_status": "ok",
            "readiness_status": "diagnostic",
            "analysis_spec_version": "block1_family_summary_calibration_v1",
            "source_execution_manifest_path": str(execution_manifest_path),
            "source_fit_cache_path": str(block0_dir / "block0_fit_cache.npz"),
            "source_fit_cache_index_path": str(block0_dir / "block0_fit_cache_index.csv"),
            "source_fit_cache_sha256": "fixture-cache-sha256",
            "source_fit_cache_index_sha256": "fixture-index-sha256",
            "patient_calibration_path": str(patient_calibration_path),
            "metric_summary_path": str(metric_summary_path),
        },
    )
    return prepare_manifest_path, calibration_manifest_path


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


def test_block2_review_helper_imports() -> None:
    from tasks.task_A.block2.review import write_block2_review_surface

    assert callable(write_block2_review_surface)


def test_result_packet_packages_available_atlas_block0_and_missing_block1(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
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
    assert (
        packet.packet_root
        / "block0"
        / "calibration"
        / "block0_calibration_manifest.json"
    ).exists()
    assert (
        packet.packet_root
        / "block0"
        / "calibration"
        / "block0_patient_calibration.csv"
    ).exists()
    assert (
        packet.packet_root
        / "block0"
        / "calibration"
        / "block0_metric_summary.csv"
    ).exists()
    assert not (packet.packet_root / "block0" / "bundle" / "block0_bundle.json").exists()
    assert not (packet.packet_root / "block0" / "bundle" / "block0_pair_metrics.csv").exists()
    assert not (packet.packet_root / "block0" / "review" / "block0_gate_summary.csv").exists()
    assert not (packet.packet_root / "block0" / "BLOCK0_RESULTS_INDEX.md").exists()

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    manifest_payload = json.loads(packet.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["included_layers"] == ["atlas", "block0", "block1", "block2"]
    assert manifest_payload["deferred_layers"] == ["block3"]
    assert manifest_payload["surface_lineage"]["block0"]["implementation_tier"] == "canonical_full"
    assert manifest_payload["surface_lineage"]["block0"]["evidence_lineage"] == "calibration_context"
    assert (
        manifest_payload["surface_lineage"]["block0"]["analysis_spec_version"]
        == "block1_family_summary_calibration_v1"
    )
    assert (
        manifest_payload["input_sources"]["block0_calibration_manifest_path"]
        == str(block0_calibration_manifest_path.resolve())
    )
    assert "block0_bundle_path" not in manifest_payload["input_sources"]
    assert {"implementation_tier", "evidence_lineage"}.issubset(index_df.columns)
    calibration_manifest_row = index_df.loc[
        index_df["packet_relative_path"]
        == "block0/calibration/block0_calibration_manifest.json"
    ].iloc[0]
    assert calibration_manifest_row["contract_alignment"] == "current_calibration_manifest"
    assert calibration_manifest_row["claim_scope"] == "calibration_context"
    assert calibration_manifest_row["proof_carrying_status"] == "none"
    assert calibration_manifest_row["implementation_tier"] == "canonical_full"
    assert calibration_manifest_row["evidence_lineage"] == "calibration_context"
    packet_relative_paths = "\n".join(index_df["packet_relative_path"].astype(str).tolist())
    for retired_token in (
        "block0_bundle.json",
        "block0_pair_metrics.csv",
        "TC-IM_randomized_target",
        "gate_checks",
        "block0_passed",
        "block0_fit_cache.npz",
    ):
        assert retired_token not in packet_relative_paths

    missing_block1_row = index_df.loc[index_df["expected_relative_path"] == "block1_bundle.json"].iloc[0]
    assert missing_block1_row["artifact_status"] == "missing_on_disk"
    assert missing_block1_row["packet_relative_path"] == ""
    missing_block2_row = index_df.loc[
        index_df["expected_relative_path"] == "block2_bounded_audit_manifest.json"
    ].iloc[0]
    assert missing_block2_row["artifact_status"] == "missing_on_disk"
    assert missing_block2_row["packet_relative_path"] == ""


def test_result_packet_rejects_legacy_proxy_block0_schema(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, invalid_block0_payload_path = _write_invalid_block0_payload(
        tmp_path / "legacy_block0_run"
    )

    with pytest.raises(ContractError, match="Block 0 calibration manifest is missing required fields"):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_calibration_manifest_path=invalid_block0_payload_path,
            output_dir=tmp_path / "packet",
        )


def test_result_packet_rejects_block0_calibration_manifest_with_wrong_analysis_spec(
    tmp_path: Path,
) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run_wrong_spec"
    )
    manifest_payload = json.loads(block0_calibration_manifest_path.read_text(encoding="utf-8"))
    manifest_payload["analysis_spec_version"] = "legacy_a_d_e_distance_v1"
    _write_json(block0_calibration_manifest_path, manifest_payload)

    with pytest.raises(ContractError, match="analysis_spec_version"):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_calibration_manifest_path=block0_calibration_manifest_path,
            output_dir=tmp_path / "packet_wrong_spec",
        )


def test_result_packet_rejects_block0_patient_calibration_with_legacy_columns(
    tmp_path: Path,
) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run_legacy_patient"
    )
    manifest_payload = json.loads(block0_calibration_manifest_path.read_text(encoding="utf-8"))
    _write_csv(
        Path(manifest_payload["patient_calibration_path"]),
        [
            {
                "patient_id": "P01",
                "metric_name": "A",
                "typical_distance_stat": "median",
                "A_real_vs_null_median_distance": 0.01,
            }
        ],
    )

    with pytest.raises(ContractError, match="block0_patient_calibration.csv columns do not match"):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_calibration_manifest_path=block0_calibration_manifest_path,
            output_dir=tmp_path / "packet_legacy_patient",
        )


def test_result_packet_rejects_block0_metric_summary_with_legacy_columns(
    tmp_path: Path,
) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run_legacy_metric"
    )
    manifest_payload = json.loads(block0_calibration_manifest_path.read_text(encoding="utf-8"))
    _write_csv(
        Path(manifest_payload["metric_summary_path"]),
        [
            {
                "metric_name": "A",
                "metric_role": "proof_carrying",
                "typical_distance_stat": "median",
                "p_value_tail": "right",
                "right_tail_null_fraction": 0.5,
            }
        ],
    )

    with pytest.raises(ContractError, match="block0_metric_summary.csv columns do not match"):
        write_task_a_result_packet(
            atlas_manifest_path=atlas_manifest_path,
            prepare_manifest_path=prepare_manifest_path,
            block0_calibration_manifest_path=block0_calibration_manifest_path,
            output_dir=tmp_path / "packet_legacy_metric",
        )


def test_result_packet_does_not_generate_block0_gate_review_tables(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run_current"
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
        output_dir=tmp_path / "packet_current",
    )

    assert not (packet.packet_root / "block0" / "review").exists()
    assert not (packet.packet_root / "block0" / "BLOCK0_RESULTS_INDEX.md").exists()

    index_df = pd.read_csv(packet.index_path, keep_default_na=False)
    block0_rows = index_df.loc[index_df["layer"] == "block0"].copy()
    assert set(block0_rows["proof_carrying_status"].astype(str)) == {"none"}
    assert "comparison_surface" not in set(block0_rows["family_surface_role"].astype(str))
    calibration_rows = block0_rows.loc[
        block0_rows["packet_relative_path"].astype(str).str.startswith("block0/calibration/")
    ]
    assert set(calibration_rows["claim_scope"].astype(str)) == {"calibration_context"}
    assert set(calibration_rows["review_role"].astype(str)) == {"calibration"}
    assert set(calibration_rows["family_surface_role"].astype(str)) == {"calibration_context"}
    block0_review_index = pd.read_csv(
        packet.packet_root / "block0" / "block0_review_index.csv",
        keep_default_na=False,
    )
    assert "gate_checks" not in block0_review_index.to_csv(index=False)


def test_result_packet_can_mirror_optional_block1_bundle(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )
    block1_bundle_path = _write_block1_bundle(tmp_path / "block1_run")

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
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
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )
    block1_bundle_path = _write_block1_bundle(tmp_path / "block1_run")
    block2_manifest_path = _write_block2_manifest(
        tmp_path / "block2_run",
        block1_bundle_path=block1_bundle_path,
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
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
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )
    output_dir = tmp_path / "packet"

    main(
        [
            "--atlas-manifest",
            str(atlas_manifest_path),
            "--prepare-manifest",
            str(prepare_manifest_path),
            "--block0-calibration-manifest",
            str(block0_calibration_manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "task_a_result_packet_manifest.json").exists()


def test_result_packet_workflow_rejects_block3_manifest_until_clean_bridge_exists(tmp_path: Path) -> None:
    from tasks.task_A.workflows.package_results import write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )
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
            block0_calibration_manifest_path=block0_calibration_manifest_path,
            block3_manifest_path=block3_manifest_path,
            output_dir=tmp_path / "packet",
        )


def test_result_packet_validation_fails_when_mirrored_file_is_removed(tmp_path: Path) -> None:
    from tasks.task_A.result_packet import validate_task_a_result_packet, write_task_a_result_packet

    atlas_manifest_path = _write_atlas_bundle(tmp_path / "atlas_source")
    prepare_manifest_path, block0_calibration_manifest_path = _write_block0_calibration_run(
        tmp_path / "block0_run"
    )

    packet = write_task_a_result_packet(
        atlas_manifest_path=atlas_manifest_path,
        prepare_manifest_path=prepare_manifest_path,
        block0_calibration_manifest_path=block0_calibration_manifest_path,
        output_dir=tmp_path / "packet",
    )

    mirrored_calibration_manifest = (
        packet.packet_root
        / "block0"
        / "calibration"
        / "block0_calibration_manifest.json"
    )
    mirrored_calibration_manifest.unlink()

    with pytest.raises(ContractError, match="Referenced packet artifact is missing"):
        validate_task_a_result_packet(packet.manifest_path)
