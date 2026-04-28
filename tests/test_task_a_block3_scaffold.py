from __future__ import annotations

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

from stride.errors import ContractError


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_block3_upstream_fixture(base: Path) -> Path:
    block1_bundle_path = _write_json(
        base / "block1" / "block1_bundle.json",
        {
            "block": "block1_continuity_backbone",
            "scientific_role": "real_data_biological_discovery",
            "artifact_state": "evidence_ready",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
        },
    )
    return _write_json(
        base / "block2" / "block2_bounded_audit_manifest.json",
        {
            "block": "block2_bounded_audit",
            "scientific_role": "robustness_of_frozen_block1_findings",
            "artifact_state": "evidence_ready",
            "implementation_tier": "canonical_full",
            "evidence_lineage": "canonical_rerun",
            "block1_bundle_path": str(block1_bundle_path),
        },
    )


def test_block3_package_exports_registry_first_contract_surface() -> None:
    import tasks.task_A.block3 as block3

    assert "Block3SubexperimentId" in set(block3.__all__)
    assert "Block3MethodName" in set(block3.__all__)
    assert "Block3MetricName" in set(block3.__all__)
    assert "MetricStatus" in set(block3.__all__)
    assert "get_live_block3_registry" in set(block3.__all__)
    assert "build_phase2_execution_plan" not in set(block3.__all__)
    assert "build_block3_bundle_layout" not in set(block3.__all__)
    assert "build_block3_review_layout" not in set(block3.__all__)


def test_block3_metric_value_requires_explicit_status_nullability() -> None:
    from tasks.task_A.block3.contracts import Block3MetricName, MetricStatus, make_metric_value

    reported = make_metric_value(
        metric_name=Block3MetricName.A_MAE_ACTIVE,
        value=0.125,
        status=MetricStatus.REPORTED,
    )
    assert reported.value == pytest.approx(0.125)
    assert reported.status is MetricStatus.REPORTED

    with pytest.raises(ContractError, match="reported metrics require a numeric value"):
        make_metric_value(
            metric_name=Block3MetricName.A_MAE_ACTIVE,
            value=None,
            status=MetricStatus.REPORTED,
        )

    with pytest.raises(ContractError, match="non-reported metrics must keep value=None"):
        make_metric_value(
            metric_name=Block3MetricName.TARGET_RECALL_AT_K,
            value=1.0,
            status=MetricStatus.NOT_APPLICABLE,
        )


def test_live_block3_registry_freezes_subexperiments_methods_and_metrics() -> None:
    from tasks.task_A.block3.contracts import (
        Block3MethodClass,
        Block3MetricName,
        Block3MethodName,
    )
    from tasks.task_A.block3.registry import get_live_block3_registry, get_subexperiment_spec

    registry = get_live_block3_registry()

    assert set(registry.subexperiments) == {"3A", "3B-1", "3B-2", "3C-1", "3C-2"}
    assert registry.section_groups["3B"].child_subexperiments == ("3B-1", "3B-2")
    assert registry.section_groups["3C"].child_subexperiments == ("3C-1", "3C-2")
    assert registry.subexperiments["3A"].methods == ()
    assert registry.subexperiments["3B-1"].methods == (
        Block3MethodName.STRIDE_REFERENCE,
        Block3MethodName.BALANCED_OT_BASELINE,
        Block3MethodName.UOT_BASELINE,
        Block3MethodName.PARTIAL_OT_BASELINE,
        Block3MethodName.DIAGONAL_TRANSPORT_BASELINE,
    )
    assert registry.subexperiments["3B-2"].methods == (
        Block3MethodName.STRIDE_REFERENCE,
        Block3MethodName.UOT_BASELINE,
        Block3MethodName.PARTIAL_OT_BASELINE,
        Block3MethodName.DIAGONAL_TRANSPORT_BASELINE,
    )
    assert registry.subexperiments["3C-1"].methods == (
        Block3MethodName.STRIDE_REFERENCE,
        Block3MethodName.OPEN_CHANNEL_ABLATION,
    )
    assert registry.subexperiments["3C-2"].methods == (
        Block3MethodName.STRIDE_REFERENCE,
        Block3MethodName.COHORT_ABLATION,
    )
    assert registry.subexperiments["3B-1"].metrics == (
        Block3MetricName.A_MAE_ACTIVE,
        Block3MetricName.A_MSE_ACTIVE,
        Block3MetricName.TARGET_RECALL_AT_K,
    )
    assert registry.subexperiments["3B-2"].metrics == (
        Block3MetricName.OPEN_SUPPORT_F1,
        Block3MetricName.D_MAE,
        Block3MetricName.E_MAE,
        Block3MetricName.D_MSE,
        Block3MetricName.E_MSE,
    )
    assert registry.subexperiments["3C-1"].metrics == (
        Block3MetricName.OPEN_SUPPORT_F1,
        Block3MetricName.D_MAE,
        Block3MetricName.E_MAE,
        Block3MetricName.D_MSE,
        Block3MetricName.E_MSE,
    )
    assert registry.subexperiments["3C-2"].metrics == (
        Block3MetricName.A_MAE_ACTIVE,
        Block3MetricName.A_MSE_ACTIVE,
        Block3MetricName.OPEN_SUPPORT_F1,
        Block3MetricName.D_MAE,
        Block3MetricName.E_MAE,
        Block3MetricName.D_MSE,
        Block3MetricName.E_MSE,
    )
    assert "NOT_APPLICABLE" not in Block3MethodClass.__members__
    assert not hasattr(registry, "stale_surface_aliases")

    with pytest.raises(ContractError, match="does not define an executable Block 3 subexperiment"):
        get_subexperiment_spec("3C")


def test_phase2_execution_plan_stays_scaffold_only_and_non_authority(tmp_path: Path) -> None:
    from tasks.task_A.block3.execution import build_phase2_execution_plan, resolve_upstream_inputs

    block2_manifest_path = _write_block3_upstream_fixture(tmp_path)
    inputs = resolve_upstream_inputs(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_output",
    )
    plan = build_phase2_execution_plan(inputs)

    assert plan.artifact_state == "scaffold_active"
    assert plan.scientific_interpretation_allowed is False
    assert plan.packet_bridge_enabled is False
    assert plan.packet_bridge_policy == "deferred_non_authority_pending_clean_bridge_spec"
    assert plan.workflow_entrypoints == ()
    assert plan.subexperiment_order == ("3A", "3B-1", "3B-2", "3C-1", "3C-2")
    assert plan.method_routes["3B-1"] == (
        "stride_reference",
        "balanced_ot_baseline",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    )
    assert plan.method_routes["3B-2"] == (
        "stride_reference",
        "uot_baseline",
        "partial_ot_baseline",
        "diagonal_transport_baseline",
    )


def test_bundle_and_review_layouts_preserve_section_routes_and_metric_status(tmp_path: Path) -> None:
    from tasks.task_A.block3.bundle import (
        build_block3_bundle_layout,
        build_bundle_manifest_payload,
        build_raw_table_schema,
    )
    from tasks.task_A.block3.execution import build_phase2_execution_plan, resolve_upstream_inputs
    from tasks.task_A.block3.review import (
        build_block3_review_layout,
        build_review_manifest_payload,
        build_review_table_schema,
    )

    block2_manifest_path = _write_block3_upstream_fixture(tmp_path)
    inputs = resolve_upstream_inputs(
        block2_manifest_path=block2_manifest_path,
        output_dir=tmp_path / "block3_output",
    )
    plan = build_phase2_execution_plan(inputs)
    bundle_layout = build_block3_bundle_layout(plan)
    review_layout = build_block3_review_layout(plan, bundle_layout)

    assert bundle_layout.packet_bridge_enabled is False
    assert bundle_layout.packet_bridge_policy == "deferred_non_authority_pending_clean_bridge_spec"
    assert review_layout.packet_bridge_enabled is False
    assert review_layout.packet_bridge_policy == "deferred_non_authority_pending_clean_bridge_spec"
    assert build_bundle_manifest_payload(bundle_layout)["workflow_name"] == "block3_internal_phase3_execution"
    assert build_review_manifest_payload(review_layout)["workflow_name"] == "block3_internal_phase3_review"

    assert {artifact.role for artifact in bundle_layout.artifacts} == {
        "bundle_manifest",
        "raw_index",
        "generator_rerun_registry",
        "generator_split_registry",
        "patient_truth_store",
        "method_native_output_store",
        "3a_object_scores",
        "3a_rerun_stability",
        "3b1_patient_metrics",
        "3b1_condition_summary",
        "3b2_patient_metrics",
        "3b2_condition_summary",
        "3c1_patient_metrics",
        "3c1_condition_summary",
        "3c2_patient_metrics",
        "3c2_condition_summary",
    }
    assert {artifact.role for artifact in review_layout.artifacts} == {
        "review_manifest",
        "review_index",
        "extraction_route_index",
        "3a_review_surface",
        "3b1_review_surface",
        "3b2_review_surface",
        "3c1_review_surface",
        "3c2_review_surface",
    }

    generator_metric_schema = build_raw_table_schema("3A", "3a_object_scores")
    assert "rerun_id" in generator_metric_schema
    assert "subexperiment_id" in generator_metric_schema
    assert "condition_id" in generator_metric_schema
    assert "evaluation_family" in generator_metric_schema
    assert "validation_object_id" in generator_metric_schema
    assert "method_name" not in generator_metric_schema
    assert "method_class" not in generator_metric_schema
    assert "metric_name" in generator_metric_schema
    assert "metric_role" in generator_metric_schema
    assert "metric_status" in generator_metric_schema

    split_schema = build_raw_table_schema("3A", "generator_split_registry")
    assert split_schema == (
        "rerun_id",
        "split_seed",
        "patient_id",
        "split_role",
    )

    truth_schema = build_raw_table_schema("3B-1", "patient_truth_store")
    assert "subexperiment_id" in truth_schema
    assert "condition_id" in truth_schema
    assert "x_json" in truth_schema
    assert "y_json" in truth_schema
    assert "A_json" in truth_schema
    assert "d_json" in truth_schema
    assert "e_json" in truth_schema
    assert "open_mass_scale" in truth_schema

    native_schema = build_raw_table_schema("3B-1", "method_native_output_store")
    assert "method_name" in native_schema
    assert "method_class" in native_schema
    assert "fit_status" in native_schema
    assert "A_json" in native_schema
    assert "d_json" in native_schema
    assert "e_json" in native_schema
    assert "P_json" in native_schema
    assert "metadata_json" in native_schema
    assert "open_mass_scale" in native_schema

    native_schema_3b2 = build_raw_table_schema("3B-2", "method_native_output_store")
    assert "method_name" in native_schema_3b2
    assert "method_class" in native_schema_3b2
    assert "fit_status" in native_schema_3b2
    assert "A_json" in native_schema_3b2
    assert "d_json" in native_schema_3b2
    assert "e_json" in native_schema_3b2
    assert "P_json" in native_schema_3b2
    assert "metadata_json" in native_schema_3b2
    assert "open_mass_scale" in native_schema_3b2

    metric_schema = build_raw_table_schema("3B-1", "3b1_patient_metrics")
    assert "rerun_id" in metric_schema
    assert "subexperiment_id" in metric_schema
    assert "condition_id" in metric_schema
    assert "open_mass_scale" in metric_schema
    assert "evaluation_family" in metric_schema
    assert "method_name" in metric_schema
    assert "method_class" in metric_schema
    assert "metric_name" in metric_schema
    assert "metric_role" in metric_schema
    assert "metric_status" in metric_schema

    generator_review_schema = build_review_table_schema("3A")
    assert "evaluation_family" in generator_review_schema
    assert "validation_object_id" in generator_review_schema
    assert "review_surface_role" in generator_review_schema
    assert "method_name" not in generator_review_schema
    assert "method_class" not in generator_review_schema
    assert "metric_status" in generator_review_schema
    assert "reported_value" in generator_review_schema

    review_schema = build_review_table_schema("3C-2")
    assert "evaluation_family" in review_schema
    assert "metric_status" in review_schema
    assert "reported_value" in review_schema
    assert "family_ARI" not in review_schema
    assert "template_MSE" not in review_schema
