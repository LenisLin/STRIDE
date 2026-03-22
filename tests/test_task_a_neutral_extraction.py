from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.extract_arm2_arm3_neutral import (
    ExtractionPackage,
    NeutralExtractionPaths,
    build_arm2_arm3_linkage_inventory_from_frames,
    build_arm2_patient_family_comparator_from_frames,
    build_arm2_prototype_family_evidence_from_frames,
    build_arm3_pair_level_coverage_comparator_from_frames,
    extract_neutral_evidence,
    write_extraction_package,
)


def test_build_arm2_patient_family_comparator_longifies_wide_tables() -> None:
    df_baseline = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "tc_im_ordered_pair_count": [3],
            "tc_pt_ordered_pair_count": [5],
            "tc_im_prototype_count": [2],
            "tc_pt_prototype_count": [2],
            "tc_im_median_abs_delta_share": [0.10],
            "tc_pt_median_abs_delta_share": [0.30],
            "patient_median_tc_pt_minus_tc_im_abs_delta_share": [0.20],
        }
    )
    df_transport = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "tc_im_ordered_pair_count": [3],
            "tc_pt_ordered_pair_count": [5],
            "tc_im_valid_uot_pair_count": [3],
            "tc_pt_valid_uot_pair_count": [5],
            "tc_im_median_U_abs": [11.0],
            "tc_pt_median_U_abs": [22.0],
            "tc_im_median_M_balanced": [0.40],
            "tc_pt_median_M_balanced": [0.60],
            "tc_im_median_balanced_minus_uot": [0.01],
            "tc_pt_median_balanced_minus_uot": [0.02],
        }
    )

    result = build_arm2_patient_family_comparator_from_frames(df_baseline, df_transport)

    assert list(result["pair_family"]) == ["TC-IM", "TC-PT"]
    tc_im = result.loc[result["pair_family"] == "TC-IM"].iloc[0]
    tc_pt = result.loc[result["pair_family"] == "TC-PT"].iloc[0]
    assert tc_im["ordered_pair_count_coalesced"] == 3
    assert tc_im["baseline_median_abs_delta_share"] == 0.10
    assert tc_im["valid_uot_pair_count"] == 3
    assert tc_pt["median_M_balanced"] == 0.60
    assert tc_pt["median_balanced_minus_uot"] == 0.02


def test_build_arm2_prototype_family_evidence_merges_recurrence_bd_and_contrast() -> None:
    df_family_summary = pd.DataFrame(
        [
            {
                "baseline_priority_rank": 1,
                "pair_family": "TC-IM",
                "proto_id": 7,
                "dominant_cell_type": "TC_EpCAM",
                "prototype_label_top3": "p7 | TC_EpCAM",
                "paired_confirmatory_patient_count": 4,
                "baseline_median_abs_delta_share": 0.2,
                "uot_transport_share_median": 0.3,
                "balanced_transport_share_median": 0.4,
                "balanced_minus_uot_transport_share_median": 0.1,
                "uot_unmatched_share_median": 0.05,
            },
            {
                "baseline_priority_rank": 2,
                "pair_family": "TC-PT",
                "proto_id": 7,
                "dominant_cell_type": "TC_EpCAM",
                "prototype_label_top3": "p7 | TC_EpCAM",
                "paired_confirmatory_patient_count": 4,
                "baseline_median_abs_delta_share": 0.4,
                "uot_transport_share_median": 0.1,
                "balanced_transport_share_median": 0.2,
                "balanced_minus_uot_transport_share_median": 0.1,
                "uot_unmatched_share_median": 0.07,
            },
        ]
    )
    df_recurrence = pd.DataFrame(
        {
            "proto_id": [7],
            "shared_transport_anchor_score": [0.3],
            "shared_transport_positive_patient_count": [4],
            "balanced_ot_forced_transport_score": [0.1],
            "forced_transport_positive_patient_count_tc_im": [2],
            "forced_transport_positive_patient_count_tc_im_prop": [0.5],
            "forced_transport_positive_patient_count_tc_pt": [3],
            "forced_transport_positive_patient_count_tc_pt_prop": [0.75],
            "forced_transport_positive_patient_count_any_confirmatory": [4],
            "forced_transport_positive_patient_count_any_confirmatory_prop": [1.0],
            "uot_unmatched_contributor_score": [0.07],
            "unmatched_positive_patient_count_tc_im": [4],
            "unmatched_positive_patient_count_tc_im_prop": [1.0],
            "unmatched_positive_patient_count_tc_pt": [2],
            "unmatched_positive_patient_count_tc_pt_prop": [0.5],
            "unmatched_positive_patient_count_any_confirmatory": [4],
            "unmatched_positive_patient_count_any_confirmatory_prop": [1.0],
            "shared_transport_and_unmatched_any_patient_count": [4],
            "shared_transport_and_unmatched_any_patient_count_prop": [1.0],
        }
    )
    df_bd = pd.DataFrame(
        [
            {
                "pair_type": "TC->IM",
                "pair_family": "TC-IM",
                "direction_role": "primary_anchor",
                "proto_id": 7,
                "patient_count": 4,
                "destroy_share": 0.02,
                "birth_share": 0.03,
                "destroy_minus_birth_share": -0.01,
                "destroy_abs": 5.0,
                "birth_abs": 6.0,
                "destroy_gt_birth_patient_count": 1,
                "destroy_gt_birth_patient_prop": 0.25,
                "birth_gt_destroy_patient_count": 3,
                "birth_gt_destroy_patient_prop": 0.75,
            },
            {
                "pair_type": "TC->PT",
                "pair_family": "TC-PT",
                "direction_role": "primary_anchor",
                "proto_id": 7,
                "patient_count": 4,
                "destroy_share": 0.05,
                "birth_share": 0.01,
                "destroy_minus_birth_share": 0.04,
                "destroy_abs": 8.0,
                "birth_abs": 2.0,
                "destroy_gt_birth_patient_count": 4,
                "destroy_gt_birth_patient_prop": 1.0,
                "birth_gt_destroy_patient_count": 0,
                "birth_gt_destroy_patient_prop": 0.0,
            },
            {
                "pair_type": "PT->TC",
                "pair_family": "TC-PT",
                "direction_role": "audit_only",
                "proto_id": 7,
                "patient_count": 4,
                "destroy_share": 0.99,
                "birth_share": 0.99,
                "destroy_minus_birth_share": 0.0,
                "destroy_abs": 99.0,
                "birth_abs": 99.0,
                "destroy_gt_birth_patient_count": 0,
                "destroy_gt_birth_patient_prop": 0.0,
                "birth_gt_destroy_patient_count": 0,
                "birth_gt_destroy_patient_prop": 0.0,
            },
        ]
    )
    df_contrast = pd.DataFrame(
        {
            "proto_id": [7],
            "panel_name": ["tc_dominant"],
            "panel_rule": ["rule"],
            "is_borderline_tc_like": [False],
            "balanced_transport_share_tc_im": [0.4],
            "uot_transport_share_tc_im": [0.3],
            "uot_unmatched_share_tc_im": [0.05],
            "balanced_minus_uot_tc_im": [0.1],
            "balanced_transport_share_tc_pt": [0.2],
            "uot_transport_share_tc_pt": [0.1],
            "uot_unmatched_share_tc_pt": [0.07],
            "balanced_minus_uot_tc_pt": [0.1],
        }
    )

    result = build_arm2_prototype_family_evidence_from_frames(
        df_family_summary,
        df_recurrence,
        df_bd,
        df_contrast,
    )

    tc_im = result.loc[result["pair_family"] == "TC-IM"].iloc[0]
    tc_pt = result.loc[result["pair_family"] == "TC-PT"].iloc[0]
    assert tc_im["forced_transport_positive_patient_count_family"] == 2
    assert tc_pt["forced_transport_positive_patient_count_family"] == 3
    assert tc_im["bd_pair_type"] == "TC->IM"
    assert tc_pt["bd_pair_type"] == "TC->PT"
    assert tc_pt["destroy_abs"] == 8.0
    assert tc_im["contrast_balanced_transport_share"] == 0.4
    assert tc_pt["contrast_uot_unmatched_share"] == 0.07


def test_build_arm3_pair_level_coverage_comparator_merges_balanced_costs() -> None:
    df_full = pd.DataFrame(
        {
            "pair_id": ["pair_full"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "compartment_a": ["TC"],
            "compartment_b": ["IM"],
            "U_abs_dens": [10.0],
            "Q_src_dens": [0.8],
            "Q_tgt_dens": [0.7],
        }
    )
    df_bootstrap = pd.DataFrame(
        {
            "pair_id": ["pair_boot"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "compartment_a": ["TC"],
            "compartment_b": ["IM"],
            "coverage": [0.75],
            "replicate_id": [0],
            "U_abs_dens": [12.0],
            "Q_src_dens": [0.85],
            "Q_tgt_dens": [0.75],
            "floor_dominated": [False],
        }
    )
    df_balanced = pd.DataFrame(
        {
            "pair_id": ["pair_full", "pair_boot"],
            "patient_id": ["P1", "P1"],
            "pair_type": ["TC->IM", "TC->IM"],
            "pair_family": ["TC-IM", "TC-IM"],
            "compartment_a": ["TC", "TC"],
            "compartment_b": ["IM", "IM"],
            "replicate_id": [-1, 0],
            "coverage": [1.0, 0.75],
            "balanced_ot_cost": [0.33, 0.44],
            "comparator_type": ["shape_only_full", "shape_only_bootstrap"],
        }
    )

    result = build_arm3_pair_level_coverage_comparator_from_frames(df_full, df_bootstrap, df_balanced)

    assert len(result) == 2
    full_row = result.loc[result["pair_id"] == "pair_full"].iloc[0]
    boot_row = result.loc[result["pair_id"] == "pair_boot"].iloc[0]
    assert full_row["coverage"] == 1.0
    assert full_row["replicate_id"] == -1
    assert full_row["balanced_ot_cost"] == 0.33
    assert boot_row["balanced_ot_cost"] == 0.44
    assert boot_row["comparator_type"] == "shape_only_bootstrap"


def test_build_arm2_arm3_linkage_inventory_uses_shared_index() -> None:
    df_arm2 = pd.DataFrame(
        {
            "proto_id": [0, 1],
            "dominant_cell_type": ["TC", "Mono"],
            "prototype_label_top3": ["p0", "p1"],
        }
    )
    df_arm3_labels = pd.DataFrame(
        {
            "arm3_prototype_k": [0, 2],
            "arm3_label_field": ["proto_0", "proto_2"],
        }
    )

    result = build_arm2_arm3_linkage_inventory_from_frames(
        df_arm2,
        df_arm3_labels,
        shared_stage0_path="/tmp/stage0.h5ad",
        linkage_source_artifact="/tmp/manifest.json",
    )

    assert list(result["linkage_type"]) == ["shared_prototype_index", "arm2_only", "arm3_only"]
    assert str(result.loc[0, "arm2_proto_id"]) == "0"
    assert str(result.loc[2, "arm3_prototype_k"]) == "2"


def test_extract_neutral_evidence_writes_expected_output_package(tmp_path: Path) -> None:
    arm2_focused = tmp_path / "arm2_focused"
    arm2_bio = tmp_path / "arm2_bio"
    arm3_root = tmp_path / "arm3"
    output_dir = tmp_path / "out"
    for path in [arm2_focused, arm2_bio, arm3_root]:
        path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "proto_id": [0],
            "dominant_cell_type": ["TC_CAIX"],
            "prototype_label_top3": ["p0 | TC_CAIX"],
        }
    ).to_csv(arm2_focused / "01_prototype_biological_meaning_table.csv", index=False)
    pd.DataFrame(
        {
            "patient_id": ["P1"],
            "tc_im_ordered_pair_count": [1],
            "tc_pt_ordered_pair_count": [1],
            "tc_im_prototype_count": [1],
            "tc_pt_prototype_count": [1],
            "tc_im_median_abs_delta_share": [0.1],
            "tc_pt_median_abs_delta_share": [0.2],
            "patient_median_tc_pt_minus_tc_im_abs_delta_share": [0.1],
        }
    ).to_csv(arm2_focused / "04_baseline_patient_family_confirmatory_summary.csv", index=False)
    pd.DataFrame(
        {
            "patient_id": ["P1"],
            "tc_im_ordered_pair_count": [1],
            "tc_pt_ordered_pair_count": [1],
            "tc_im_valid_uot_pair_count": [1],
            "tc_pt_valid_uot_pair_count": [1],
            "tc_im_median_U_abs": [11.0],
            "tc_pt_median_U_abs": [22.0],
            "tc_im_median_transport_fraction": [0.9],
            "tc_pt_median_transport_fraction": [0.8],
            "tc_im_median_unmatched_fraction": [0.1],
            "tc_pt_median_unmatched_fraction": [0.2],
            "tc_im_median_M": [0.3],
            "tc_pt_median_M": [0.4],
            "tc_im_median_D_pos": [1.0],
            "tc_pt_median_D_pos": [2.0],
            "tc_im_median_B_pos": [1.5],
            "tc_pt_median_B_pos": [2.5],
            "tc_im_median_T_abs": [5.0],
            "tc_pt_median_T_abs": [6.0],
            "tc_im_median_M_balanced": [0.31],
            "tc_pt_median_M_balanced": [0.42],
            "tc_im_median_balanced_minus_uot": [0.01],
            "tc_pt_median_balanced_minus_uot": [0.02],
        }
    ).to_csv(arm2_focused / "05_global_transport_summary.csv", index=False)
    pd.DataFrame(
        {
            "baseline_priority_rank": [1, 1],
            "pair_family": ["TC-IM", "TC-PT"],
            "proto_id": [0, 0],
            "dominant_cell_type": ["TC_CAIX", "TC_CAIX"],
            "prototype_label_top3": ["p0 | TC_CAIX", "p0 | TC_CAIX"],
            "paired_confirmatory_patient_count": [1, 1],
            "baseline_median_abs_delta_share": [0.1, 0.2],
            "uot_transport_share_median": [0.3, 0.2],
            "balanced_transport_share_median": [0.4, 0.25],
            "balanced_minus_uot_transport_share_median": [0.1, 0.05],
            "uot_unmatched_share_median": [0.05, 0.07],
        }
    ).to_csv(arm2_focused / "10_prototype_family_specific_summary.csv", index=False)
    pd.DataFrame(
        {
            "proto_id": [0],
            "shared_transport_anchor_score": [0.3],
            "shared_transport_positive_patient_count": [1],
            "balanced_ot_forced_transport_score": [0.1],
            "forced_transport_positive_patient_count_tc_im": [1],
            "forced_transport_positive_patient_count_tc_im_prop": [1.0],
            "forced_transport_positive_patient_count_tc_pt": [1],
            "forced_transport_positive_patient_count_tc_pt_prop": [1.0],
            "forced_transport_positive_patient_count_any_confirmatory": [1],
            "forced_transport_positive_patient_count_any_confirmatory_prop": [1.0],
            "uot_unmatched_contributor_score": [0.07],
            "unmatched_positive_patient_count_tc_im": [1],
            "unmatched_positive_patient_count_tc_im_prop": [1.0],
            "unmatched_positive_patient_count_tc_pt": [1],
            "unmatched_positive_patient_count_tc_pt_prop": [1.0],
            "unmatched_positive_patient_count_any_confirmatory": [1],
            "unmatched_positive_patient_count_any_confirmatory_prop": [1.0],
            "shared_transport_and_unmatched_any_patient_count": [1],
            "shared_transport_and_unmatched_any_patient_count_prop": [1.0],
        }
    ).to_csv(arm2_focused / "11_prototype_patient_recurrence_summary.csv", index=False)
    pd.DataFrame(
        {
            "row_type": ["summary"],
            "overlap_label": ["anchor_vs_forced_top10"],
            "top_n": [10],
            "intersection_size": [2],
        }
    ).to_csv(arm2_focused / "09_prototype_overlap_conflict_audit.csv", index=False)
    (arm2_focused / "00_arm2_focused_results_memo.md").write_text("memo\n", encoding="utf-8")
    pd.DataFrame(
        {
            "proto_id": [0],
            "panel_name": ["tc_dominant"],
            "panel_rule": ["rule"],
            "is_borderline_tc_like": [False],
            "balanced_transport_share_tc_im": [0.4],
            "uot_transport_share_tc_im": [0.3],
            "uot_unmatched_share_tc_im": [0.05],
            "balanced_minus_uot_tc_im": [0.1],
            "balanced_transport_share_tc_pt": [0.25],
            "uot_transport_share_tc_pt": [0.2],
            "uot_unmatched_share_tc_pt": [0.07],
            "balanced_minus_uot_tc_pt": [0.05],
        }
    ).to_csv(arm2_bio / "23_ot_vs_uot_prototype_contrast.csv", index=False)
    pd.DataFrame(
        {
            "pair_type": ["TC->IM", "TC->PT"],
            "pair_family": ["TC-IM", "TC-PT"],
            "direction_role": ["primary_anchor", "primary_anchor"],
            "proto_id": [0, 0],
            "patient_count": [1, 1],
            "destroy_share": [0.02, 0.05],
            "birth_share": [0.03, 0.01],
            "destroy_minus_birth_share": [-0.01, 0.04],
            "destroy_abs": [5.0, 8.0],
            "birth_abs": [6.0, 2.0],
            "destroy_gt_birth_patient_count": [0, 1],
            "destroy_gt_birth_patient_prop": [0.0, 1.0],
            "birth_gt_destroy_patient_count": [1, 0],
            "birth_gt_destroy_patient_prop": [1.0, 0.0],
        }
    ).to_csv(arm2_bio / "24_bd_unmatched_directionality.csv", index=False)
    pd.DataFrame({"contrast": ["x"], "source_file": ["24_bd_unmatched_directionality.csv"]}).to_csv(
        arm2_bio / "25_arm2_biointegrated_memo_table.csv", index=False
    )

    pd.DataFrame(
        {
            "prototype_k": [0],
            "prototype_label": ["proto_0"],
            "coverage": [0.75],
            "sign_consistency_rate": [1.0],
            "n_evaluable_patients": [1],
            "n_zero_reference_patients": [0],
            "correlation_to_full_cov": [1.0],
            "n_correlation_patients": [1],
        }
    ).to_parquet(arm3_root / "arm3_phase8_prototype_stability.parquet", index=False)
    pd.DataFrame(
        {
            "patient_id": ["P1", "P1"],
            "coverage": [1.0, 0.75],
            "replicate_id": [np.nan, 0],
            "prototype_k": [0, 0],
            "prototype_label": ["proto_0", "proto_0"],
            "U_k_TC_IM": [5.0, 4.0],
            "U_k_TC_PT": [2.0, 1.5],
            "Delta_U_k": [-3.0, -2.5],
        }
    ).to_parquet(arm3_root / "arm3_phase8_prototype_contrast_prep.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["pair_full"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "coverage": [1.0],
            "replicate_id": [-1],
            "prototype_k": [0],
            "prototype_label": ["proto_0"],
            "T_mass": [4.0],
            "B_mass": [1.0],
            "D_mass": [0.0],
        }
    ).to_parquet(arm3_root / "arm3_phase6_prototype_events_full.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["pair_boot"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "coverage": [0.75],
            "replicate_id": [0],
            "prototype_k": [0],
            "prototype_label": ["proto_0"],
            "T_mass": [3.0],
            "B_mass": [1.2],
            "D_mass": [0.1],
        }
    ).to_parquet(arm3_root / "arm3_phase6_prototype_events_bootstrap.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["pair_full"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "compartment_a": ["TC"],
            "compartment_b": ["IM"],
            "U_abs_dens": [10.0],
            "Q_src_dens": [0.8],
            "Q_tgt_dens": [0.7],
        }
    ).to_parquet(arm3_root / "arm3_phase6_full_coverage_results.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["pair_boot"],
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "pair_family": ["TC-IM"],
            "compartment_a": ["TC"],
            "compartment_b": ["IM"],
            "coverage": [0.75],
            "replicate_id": [0],
            "U_abs_dens": [12.0],
            "Q_src_dens": [0.85],
            "Q_tgt_dens": [0.75],
            "floor_dominated": [False],
        }
    ).to_parquet(arm3_root / "arm3_phase6_bootstrap_results.parquet", index=False)
    pd.DataFrame(
        {
            "pair_id": ["pair_full", "pair_boot"],
            "patient_id": ["P1", "P1"],
            "pair_type": ["TC->IM", "TC->IM"],
            "pair_family": ["TC-IM", "TC-IM"],
            "compartment_a": ["TC", "TC"],
            "compartment_b": ["IM", "IM"],
            "replicate_id": [-1, 0],
            "coverage": [1.0, 0.75],
            "balanced_ot_cost": [0.33, 0.44],
            "comparator_type": ["shape_only_full", "shape_only_bootstrap"],
        }
    ).to_parquet(arm3_root / "arm3_phase6_balanced_ot_results.parquet", index=False)
    pd.DataFrame(
        {
            "patient_id": ["P1"],
            "pair_type": ["TC->IM"],
            "coverage": [0.75],
            "quantity": ["U_abs_dens"],
            "median_abs_degradation": [1.0],
            "sign_consistency_rate": [1.0],
            "floor_dominated_rate": [0.0],
            "mean_replicate_value": [10.0],
            "std_replicate_value": [0.5],
        }
    ).to_parquet(arm3_root / "arm3_phase7_degradation_summary.parquet", index=False)
    pd.DataFrame(
        {
            "patient_id": ["P1"],
            "coverage": [0.75],
            "contrast_name": ["Delta_U_abs"],
            "reference_contrast": [2.0],
            "median_replicate_contrast": [1.8],
            "abs_degradation": [0.2],
            "sign_consistency_rate": [1.0],
            "n_evaluable": [1],
            "n_zero_reference_sign": [0],
        }
    ).to_parquet(arm3_root / "arm3_phase7_contrast_summary.parquet", index=False)
    (arm3_root / "arm3_phase8_memo.md").write_text("memo\n", encoding="utf-8")
    (arm3_root / "arm3_runtime_timing.json").write_text(json.dumps({"coverage_levels": [0.75], "n_reps": 1}), encoding="utf-8")
    (arm3_root / "arm3_phase0_manifest.json").write_text(json.dumps({"stage0_path": str(tmp_path / "stage0.h5ad")}), encoding="utf-8")

    paths = NeutralExtractionPaths(
        repo_root=ROOT,
        task_a_root=tmp_path,
        arm2_focused_dir=arm2_focused,
        arm2_bio_dir=arm2_bio,
        arm3_result_root=arm3_root,
        output_dir=output_dir,
        stage0_path=tmp_path / "stage0.h5ad",
    )
    package = extract_neutral_evidence(paths)
    written = write_extraction_package(package, output_dir)

    assert isinstance(package, ExtractionPackage)
    assert written["arm2_prototype_family_evidence"].exists()
    assert written["arm3_pair_level_coverage_comparator"].exists()
    linkage = pd.read_csv(written["arm2_arm3_linkage_inventory"])
    assert "linkage_type" in linkage.columns
    assert set(package.arm2_prototype_family_evidence["pair_family"]) == {"TC-IM", "TC-PT"}
    assert set(package.arm3_prototype_coverage_stability["coverage"]) == {0.75}
