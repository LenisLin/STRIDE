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
            "fit_surface": "stride.tl.fit",
            "implementation_tier": "canonical_stride_tl",
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


def test_result_packet_workflow_is_exported() -> None:
    import tasks.task_A.workflows

    assert hasattr(tasks.task_A.workflows, "write_task_a_result_packet")
    assert hasattr(tasks.task_A.workflows, "validate_task_a_result_packet")


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
    assert manifest_payload["included_layers"] == ["atlas", "block0", "block1"]
    assert manifest_payload["deferred_layers"] == ["block3"]
    assert manifest_payload["surface_lineage"]["block0"]["implementation_tier"] == "canonical_stride_tl"
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
    assert calibration_manifest_row["implementation_tier"] == "canonical_stride_tl"
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
