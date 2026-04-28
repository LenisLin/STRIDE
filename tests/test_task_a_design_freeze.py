from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None


def test_design_freeze_surface_registry_freezes_task_a_entrypoints() -> None:
    from tasks.task_A.contracts import TASK_A_SURFACE_SPEC_BY_NAME

    assert set(TASK_A_SURFACE_SPEC_BY_NAME) == {
        "build_stage0_artifacts",
        "check_task_a_pre_block0_data_suitability",
        "prepare_task_a_stage0_mapping",
        "write_task_a_descriptive_atlas",
        "run_block0_workflow",
        "write_task_a_block1_bundle",
        "write_block2_bundle",
        "write_semisynthetic_artifacts",
        "TaskADemoSubset registry",
        "stride_adapter helpers",
        "Task A contract helpers",
    }

    prepare_surface = TASK_A_SURFACE_SPEC_BY_NAME["prepare_task_a_stage0_mapping"]
    assert prepare_surface.execution_status == "executable"
    assert prepare_surface.emitted_artifact_states == ("scaffold_active", "contract_passed")
    assert "Satisfy the Block 0 scientific gate" in prepare_surface.does_not_do

    atlas_surface = TASK_A_SURFACE_SPEC_BY_NAME["write_task_a_descriptive_atlas"]
    assert atlas_surface.execution_status == "executable"
    assert atlas_surface.emitted_artifact_states == ("scaffold_active", "contract_passed")
    assert "Emit confirmatory or inferential claims" in atlas_surface.does_not_do

    block0_surface = TASK_A_SURFACE_SPEC_BY_NAME["run_block0_workflow"]
    assert block0_surface.execution_status == "executable"
    assert block0_surface.produces == ("block0_bundle.json", "block0_pair_metrics.csv")
    assert block0_surface.emitted_artifact_states == ("scaffold_active", "contract_passed")
    assert "Read Step 1 files as hard inputs" in block0_surface.does_not_do
    assert "Use observation-match-specific R/M gate metrics" in block0_surface.does_not_do

    block1_surface = TASK_A_SURFACE_SPEC_BY_NAME["write_task_a_block1_bundle"]
    assert block1_surface.execution_status == "executable"
    assert block1_surface.produces == (
        "block1_stage0_mapping.json",
        "block1_core_fit_dry_run.csv",
        "block1_family_summary.csv",
        "block1_source_community_summary.csv",
        "block1_target_community_summary.csv",
        "block1_confirmatory_family_comparison.csv",
        "block1_exploratory_source_community_comparison.csv",
        "block1_exploratory_target_community_comparison.csv",
        "community_correspondence/block1_community_correspondence_manifest.json",
        "community_correspondence/block1_community_correspondence_index.csv",
        "community_correspondence tables",
        "block1_bundle.json",
        "block1_workflow_manifest.json",
    )
    assert "Change the legacy Block 1 compatibility identifier" in block1_surface.does_not_do

    block2_surface = TASK_A_SURFACE_SPEC_BY_NAME["write_block2_bundle"]
    assert block2_surface.emitted_artifact_states == ("evidence_ready",)
    assert block2_surface.produces == (
        "block2_bounded_audit_summary.csv",
        "block2_contract_audit.csv",
        "block2_replicate_manifest.csv",
        "block2_family_robustness.csv",
        "block2_source_community_robustness.csv",
        "block2_target_community_robustness.csv",
        "block2_bounded_audit_manifest.json",
    )
    assert "Change the frozen Block 1 summary contract" in block2_surface.does_not_do
    assert "Compare baselines or run ablations" in block2_surface.does_not_do
    assert "Claim true disappearance or emergence" in block2_surface.does_not_do


def test_design_freeze_execution_graph_freezes_canonical_order_and_blockers() -> None:
    from tasks.task_A.contracts import TASK_A_EXECUTION_GRAPH, TASK_A_EXECUTION_NODE_BY_NAME

    assert [node.name for node in TASK_A_EXECUTION_GRAPH] == [
        "stage0_artifact_builder",
        "step1_prepare_full_cohort",
        "descriptive_atlas_context_layer",
        "pre_block0_data_suitability_report",
        "step1_prepare_subset_or_demo",
        "block0_locality_gate",
        "block1_continuity_backbone",
        "block2_bounded_audit",
        "semisynthetic_benchmark_export",
    ]

    atlas_node = TASK_A_EXECUTION_NODE_BY_NAME["descriptive_atlas_context_layer"]
    assert atlas_node.execution_status == "executable"
    assert atlas_node.hard_prerequisites == ("task_a_prepare_manifest.json",)
    assert atlas_node.canonical_predecessors == ("step1_prepare_full_cohort",)

    block0_node = TASK_A_EXECUTION_NODE_BY_NAME["block0_locality_gate"]
    assert block0_node.execution_status == "executable"
    assert block0_node.produced_artifacts == ("block0_bundle.json", "block0_pair_metrics.csv")
    assert block0_node.blocker_reason is None

    block1_node = TASK_A_EXECUTION_NODE_BY_NAME["block1_continuity_backbone"]
    assert block1_node.canonical_predecessors == (
        "step1_prepare_full_cohort",
        "descriptive_atlas_context_layer",
        "block0_locality_gate",
    )
    assert block1_node.hard_prerequisites == (
        "stage0-h5ad",
        "task-config",
        "passed block0 bundle",
    )
    assert block1_node.produced_artifacts == (
        "block1_stage0_mapping.json",
        "block1_core_fit_dry_run.csv",
        "block1_family_summary.csv",
        "block1_source_community_summary.csv",
        "block1_target_community_summary.csv",
        "block1_confirmatory_family_comparison.csv",
        "block1_exploratory_source_community_comparison.csv",
        "block1_exploratory_target_community_comparison.csv",
        "community_correspondence/block1_community_correspondence_manifest.json",
        "community_correspondence/block1_community_correspondence_index.csv",
        "community_correspondence tables",
        "block1_bundle.json",
        "block1_workflow_manifest.json",
    )

    block2_node = TASK_A_EXECUTION_NODE_BY_NAME["block2_bounded_audit"]
    assert block2_node.hard_prerequisites == ("block1 bundle",)
    assert block2_node.execution_status == "executable"
    assert block2_node.produced_artifacts == (
        "block2_bounded_audit_summary.csv",
        "block2_contract_audit.csv",
        "block2_replicate_manifest.csv",
        "block2_family_robustness.csv",
        "block2_source_community_robustness.csv",
        "block2_target_community_robustness.csv",
        "block2_bounded_audit_manifest.json",
    )

    assert "block3_method_validation" not in TASK_A_EXECUTION_NODE_BY_NAME


def test_design_freeze_artifact_registry_freezes_state_sources() -> None:
    from tasks.task_A.contracts import TASK_A_ARTIFACT_SPEC_BY_NAME

    stage0_validation = TASK_A_ARTIFACT_SPEC_BY_NAME["stage0_validation_report"]
    assert stage0_validation.artifact_state_location == "embedded field"
    assert stage0_validation.allowed_artifact_states == (
        "scaffold_active",
        "contract_passed",
    )

    step1_mapping = TASK_A_ARTIFACT_SPEC_BY_NAME["step1_stride_mapping"]
    assert step1_mapping.artifact_state_location == "task_a_prepare_manifest.json.artifact_state"
    assert step1_mapping.readiness_classification == "contract_passed"

    atlas_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["descriptive_atlas_manifest"]
    assert atlas_manifest.artifact_state_location == "embedded field"
    assert atlas_manifest.allowed_artifact_states == ("scaffold_active", "contract_passed")
    assert "atlas_role" in atlas_manifest.minimum_fields

    atlas_index = TASK_A_ARTIFACT_SPEC_BY_NAME["descriptive_atlas_output_index"]
    assert atlas_index.artifact_state_location == "task_a_descriptive_atlas_manifest.json.artifact_state"
    assert atlas_index.allowed_artifact_states == ("scaffold_active", "contract_passed")

    block2_summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_summary"]
    assert block2_summary.allowed_artifact_states == ("evidence_ready",)
    assert block2_summary.artifact_state_location == "block2_bounded_audit_manifest.json.artifact_state"
    assert "overall_robustness_call" in block2_summary.minimum_fields

    block0_pair_metrics = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_pair_metrics"]
    assert block0_pair_metrics.artifact_state_location == "block0_bundle.json.artifact_state"
    assert block0_pair_metrics.allowed_artifact_states == (
        "scaffold_active",
        "contract_passed",
    )
    assert "null_target_donor_patient_id" in block0_pair_metrics.minimum_fields
    assert "delta_total_continuity_mass" in block0_pair_metrics.minimum_fields

    block0_bundle = TASK_A_ARTIFACT_SPEC_BY_NAME["block0_bundle_contract"]
    assert "inputs" in block0_bundle.minimum_fields
    assert "real_families" in block0_bundle.minimum_fields
    assert "null_families" in block0_bundle.minimum_fields

    block1_family_summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_family_summary"]
    assert block1_family_summary.artifact_state_location == "block1_bundle.json.artifact_state"
    assert block1_family_summary.allowed_artifact_states == ("evidence_ready",)
    assert "summary_name" in block1_family_summary.minimum_fields
    assert "scale" in block1_family_summary.minimum_fields

    block1_source_summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_source_community_summary"]
    assert block1_source_summary.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "off_diagonal_remodeling" in block1_source_summary.minimum_fields
    assert "top_target_1_id" in block1_source_summary.minimum_fields

    block1_target_summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_target_community_summary"]
    assert block1_target_summary.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "incoming_matched_operator" in block1_target_summary.minimum_fields
    assert "emergence_burden" in block1_target_summary.minimum_fields

    block1_family_comparison = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_confirmatory_family_comparison"]
    assert block1_family_comparison.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "delta_tc_im_minus_tc_pt" in block1_family_comparison.minimum_fields
    assert "comparison_scope_role" in block1_family_comparison.minimum_fields

    block1_source_comparison = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_exploratory_source_community_comparison"]
    assert block1_source_comparison.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "source_community_id" in block1_source_comparison.minimum_fields
    assert "comparison_scope_role" in block1_source_comparison.minimum_fields

    block1_target_comparison = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_exploratory_target_community_comparison"]
    assert block1_target_comparison.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "target_community_id" in block1_target_comparison.minimum_fields
    assert "comparison_scope_role" in block1_target_comparison.minimum_fields

    block1_correspondence_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_community_correspondence_manifest"]
    assert block1_correspondence_manifest.artifact_state_location == "block1_bundle.json.artifact_state"
    assert "packet_role" in block1_correspondence_manifest.minimum_fields
    assert "output_index" in block1_correspondence_manifest.minimum_fields

    block1_correspondence_index = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_community_correspondence_index"]
    assert block1_correspondence_index.artifact_state_location == "block1_bundle.json.artifact_state"
    assert block1_correspondence_index.minimum_fields == (
        "relative_path",
        "artifact_kind",
        "category",
        "format",
        "description",
    )

    block1_bundle = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_bundle"]
    assert "scientific_role" in block1_bundle.minimum_fields
    assert "family_summary_path" in block1_bundle.minimum_fields
    assert "confirmatory_family_comparison_path" in block1_bundle.minimum_fields
    assert "community_correspondence_manifest_path" in block1_bundle.minimum_fields
    assert "paired_comparison_contract_version" in block1_bundle.minimum_fields
    assert "summary_contract_version" in block1_bundle.minimum_fields
    assert "proof_carrying_family_summaries" in block1_bundle.minimum_fields
    assert "source_eligibility_rule" in block1_bundle.minimum_fields
    assert "primary_evidence_lines" not in block1_bundle.minimum_fields

    block1_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block1_workflow_manifest"]
    assert "scientific_role" in block1_manifest.minimum_fields
    assert "family_summary_path" in block1_manifest.minimum_fields
    assert "confirmatory_family_comparison_path" in block1_manifest.minimum_fields
    assert "community_correspondence_index_path" in block1_manifest.minimum_fields
    assert "paired_comparison_contract_version" in block1_manifest.minimum_fields
    assert "summary_contract_version" in block1_manifest.minimum_fields

    semisynthetic_contract = TASK_A_ARTIFACT_SPEC_BY_NAME["semisynthetic_contract"]
    assert semisynthetic_contract.allowed_artifact_states == ("contract_passed",)
    assert semisynthetic_contract.minimum_fields[0] == "artifact_state"

    block2_summary = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_summary"]
    assert block2_summary.allowed_artifact_states == ("evidence_ready",)
    assert block2_summary.artifact_state_location == "block2_bounded_audit_manifest.json.artifact_state"
    assert "summary_scope" in block2_summary.minimum_fields
    assert "summary_name" in block2_summary.minimum_fields
    assert "primary_routes_robust" in block2_summary.minimum_fields
    assert "overall_robustness_call" in block2_summary.minimum_fields

    block2_replicate_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_replicate_manifest"]
    assert "route_name" in block2_replicate_manifest.minimum_fields
    assert "n_patients_retained" in block2_replicate_manifest.minimum_fields
    assert "failure_reason" in block2_replicate_manifest.minimum_fields

    block2_family_robustness = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_family_robustness"]
    assert "direction_recovery_rate" in block2_family_robustness.minimum_fields
    assert "robustness_call" in block2_family_robustness.minimum_fields

    block2_source_robustness = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_source_community_robustness"]
    assert "community_id" in block2_source_robustness.minimum_fields
    assert "median_replicate_rank" in block2_source_robustness.minimum_fields

    block2_target_robustness = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_target_community_robustness"]
    assert "community_id" in block2_target_robustness.minimum_fields
    assert "median_replicate_rank" in block2_target_robustness.minimum_fields

    block2_manifest = TASK_A_ARTIFACT_SPEC_BY_NAME["block2_manifest"]
    assert "scientific_role" in block2_manifest.minimum_fields
    assert "config_path" in block2_manifest.minimum_fields
    assert "block1_family_summary_path" in block2_manifest.minimum_fields
    assert "block1_source_community_summary_path" in block2_manifest.minimum_fields
    assert "block1_target_community_summary_path" in block2_manifest.minimum_fields
    assert "replicate_manifest_path" in block2_manifest.minimum_fields
    assert "family_robustness_path" in block2_manifest.minimum_fields
    assert "primary_routes" in block2_manifest.minimum_fields

    assert "block3_method_registry" not in TASK_A_ARTIFACT_SPEC_BY_NAME
    assert "block3_subexperiment_registry" not in TASK_A_ARTIFACT_SPEC_BY_NAME
    assert "block3_execution_plan" not in TASK_A_ARTIFACT_SPEC_BY_NAME
    assert "block3_summary" not in TASK_A_ARTIFACT_SPEC_BY_NAME
    assert "block3_manifest" not in TASK_A_ARTIFACT_SPEC_BY_NAME


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_stage0_validation_report_emits_frozen_artifact_state() -> None:
    import anndata as ad

    from tasks.task_A.stage0.build_artifacts import build_stage0_h5ad_validation_report
    from tests.helpers_task_a_fixture import build_task_a_fixture

    adata = build_task_a_fixture()
    n_features = int(adata.obsm["community_features"].shape[1])
    adata.uns["scaler_params"] = {
        "feature_names": [f"feature_{idx}" for idx in range(n_features)],
        "center": [0.0] * n_features,
        "scale": [1.0] * n_features,
    }
    passed_report = build_stage0_h5ad_validation_report(
        adata,
        require_all_proto_ids=False,
    )
    assert passed_report["artifact_state"] == "contract_passed"

    broken_obs = adata.obs.drop(columns=["proto_id"]).copy()
    broken = ad.AnnData(X=adata.X, obs=broken_obs, obsm=dict(adata.obsm), uns=dict(adata.uns))
    failed_report = build_stage0_h5ad_validation_report(
        broken,
        require_all_proto_ids=False,
    )
    assert failed_report["artifact_state"] == "scaffold_active"


def test_semisynthetic_contract_includes_artifact_state(tmp_path: Path) -> None:
    from tasks.task_A.benchmarks.semisynthetic import write_semisynthetic_artifacts

    manifest_path = write_semisynthetic_artifacts(
        output_root=tmp_path,
        manifest_filename="task_a_semisynthetic_manifest.csv",
        n_patients=4,
        seed=3,
    )

    contract_path = tmp_path / "task_a_semisynthetic_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert manifest_path.exists()
    assert payload["artifact_state"] == "contract_passed"
    assert payload["stronger_continuity_score"] > payload["weaker_continuity_score"]
