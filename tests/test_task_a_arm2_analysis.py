from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OT_AVAILABLE = importlib.util.find_spec("ot") is not None
if not OT_AVAILABLE:
    sys.modules["ot"] = types.SimpleNamespace(emd=lambda *args, **kwargs: None)
if importlib.util.find_spec("anndata") is None:
    anndata_stub = types.ModuleType("anndata")

    class _AnnData:  # pragma: no cover - import stub only
        pass

    anndata_stub.AnnData = _AnnData
    sys.modules["anndata"] = anndata_stub

from tasks.task_A.analyze_arm2_results import (
    CONFIRMATORY_SUMMARY_METRICS_WITH_NORMALIZED_T,
    PROTOTYPE_CONTRAST_QUANTITIES,
    add_transport_scale_metrics,
    aggregate_prototype_patient_family_tables,
    aggregate_prototype_transport_patient_family_table,
    build_patient_roi_audit_from_roi_table,
    compute_balanced_row_prototype_transport_quantities,
    compute_transport_row_prototype_quantities,
    compute_uot_row_prototype_quantities,
    confirmatory_mean_median_comparison_table,
    confirmatory_patient_contrast_table,
    confirmatory_patient_family_summary_table,
    prototype_confirmatory_contrast_table,
    prototype_confirmatory_support_summary_table,
    prototype_uot_vs_balanced_confirmatory_contrast_table,
    prototype_uot_vs_balanced_patient_family_delta_table,
    validate_prototype_transport_delta_table,
    validate_prototype_transport_table,
    validate_transport_scale_metrics,
)


def _row(
    patient_id_a: str,
    pair_family: str,
    *,
    T: float,
    U: float,
    D_pos: float,
    B_pos: float,
    M: float,
    balanced_minus_uot: float,
) -> dict[str, object]:
    return {
        "patient_id_a": patient_id_a,
        "pair_family": pair_family,
        "T": T,
        "U": U,
        "D_pos": D_pos,
        "B_pos": B_pos,
        "M": M,
        "balanced_minus_uot": balanced_minus_uot,
    }


def test_build_patient_roi_audit_from_roi_table_captures_nominal_and_deviations() -> None:
    roi_table = pd.DataFrame(
        [
            *[
                {"patient_id": "P1", "compartment": "TC", "roi_id": f"P1_TC_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P1", "compartment": "IM", "roi_id": f"P1_IM_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P1", "compartment": "PT", "roi_id": f"P1_PT_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P2", "compartment": "TC", "roi_id": f"P2_TC_{idx}"}
                for idx in range(2)
            ],
            *[
                {"patient_id": "P2", "compartment": "IM", "roi_id": f"P2_IM_{idx}"}
                for idx in range(4)
            ],
            *[
                {"patient_id": "P2", "compartment": "PT", "roi_id": f"P2_PT_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P3", "compartment": "TC", "roi_id": f"P3_TC_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P3", "compartment": "IM", "roi_id": f"P3_IM_{idx}"}
                for idx in range(3)
            ],
            *[
                {"patient_id": "P3", "compartment": "PT", "roi_id": f"P3_PT_{idx}"}
                for idx in range(4)
            ],
            {"patient_id": "P2", "compartment": "TC", "roi_id": "P2_TC_0"},
        ]
    )

    audit, deviation_counts = build_patient_roi_audit_from_roi_table(roi_table)

    by_patient = audit.set_index("patient_id")
    assert bool(by_patient.loc["P1", "is_nominal_3_3_3"])
    assert by_patient.loc["P1", "ordered_rows_TC_IM"] == 18
    assert by_patient.loc["P2", "deviation_pattern"] == "TC=2,IM=4,PT=3"
    assert by_patient.loc["P2", "ordered_rows_TC_IM"] == 16
    assert by_patient.loc["P2", "ordered_rows_TC_PT"] == 12
    assert by_patient.loc["P2", "ordered_rows_IM_PT"] == 24
    assert by_patient.loc["P3", "deviation_pattern"] == "TC=3,IM=3,PT=4"

    counts = deviation_counts.set_index("deviation_pattern")["patient_count"].to_dict()
    assert counts == {
        "TC=2,IM=4,PT=3": 1,
        "TC=3,IM=3,PT=4": 1,
        "nominal_3_3_3": 1,
    }


def test_confirmatory_patient_family_summaries_and_contrasts_are_analysis_only() -> None:
    pair_level = pd.DataFrame(
        [
            _row("P1", "TC-IM", T=10.0, U=1.0, D_pos=0.5, B_pos=0.5, M=0.8, balanced_minus_uot=0.3),
            _row("P1", "TC-IM", T=12.0, U=2.0, D_pos=1.0, B_pos=1.0, M=0.9, balanced_minus_uot=0.2),
            _row("P1", "TC-PT", T=6.0, U=4.0, D_pos=2.0, B_pos=2.0, M=1.1, balanced_minus_uot=0.4),
            _row("P1", "TC-PT", T=8.0, U=5.0, D_pos=2.5, B_pos=2.5, M=1.2, balanced_minus_uot=0.5),
            _row("P1", "IM-PT", T=7.0, U=3.0, D_pos=1.5, B_pos=1.5, M=1.0, balanced_minus_uot=0.1),
            _row("P2", "TC-IM", T=9.0, U=1.0, D_pos=0.5, B_pos=0.5, M=0.7, balanced_minus_uot=-2.0),
            _row("P2", "TC-IM", T=9.0, U=1.0, D_pos=0.5, B_pos=0.5, M=0.7, balanced_minus_uot=0.1),
            _row("P2", "TC-IM", T=9.0, U=9.0, D_pos=4.5, B_pos=4.5, M=0.7, balanced_minus_uot=0.1),
            _row("P2", "TC-PT", T=5.0, U=5.0, D_pos=2.5, B_pos=2.5, M=1.3, balanced_minus_uot=0.2),
            _row("P2", "TC-PT", T=5.0, U=5.0, D_pos=2.5, B_pos=2.5, M=1.3, balanced_minus_uot=0.2),
            _row("P2", "TC-PT", T=5.0, U=5.0, D_pos=2.5, B_pos=2.5, M=1.3, balanced_minus_uot=0.2),
            _row("P2", "IM-PT", T=8.0, U=2.0, D_pos=1.0, B_pos=1.0, M=1.0, balanced_minus_uot=0.0),
        ]
    )

    median_summary = confirmatory_patient_family_summary_table(pair_level, summary_method="median")
    mean_summary = confirmatory_patient_family_summary_table(pair_level, summary_method="mean")
    contrasts = confirmatory_patient_contrast_table(median_summary, mean_summary)
    comparison = confirmatory_mean_median_comparison_table(median_summary, mean_summary)

    assert set(median_summary["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(mean_summary["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}

    median_lookup = median_summary.set_index(["patient_id_a", "pair_family"])
    mean_lookup = mean_summary.set_index(["patient_id_a", "pair_family"])
    assert median_lookup.loc[("P1", "TC-IM"), "T"] == 11.0
    assert mean_lookup.loc[("P1", "TC-PT"), "U"] == 4.5
    assert median_lookup.loc[("P2", "TC-IM"), "U"] == 1.0
    assert mean_lookup.loc[("P2", "TC-IM"), "U"] == 11.0 / 3.0

    contrast_lookup = contrasts.set_index("patient_id_a")
    assert contrast_lookup.loc["P1", "median_contrast_tc_pt_minus_tc_im_T"] == -4.0
    assert contrast_lookup.loc["P1", "median_contrast_tc_pt_minus_tc_im_U"] == 3.0
    assert np.isclose(
        contrast_lookup.loc["P2", "mean_contrast_tc_pt_minus_tc_im_U"],
        5.0 - (11.0 / 3.0),
    )

    flip_row = comparison.loc[
        (comparison["patient_id_a"] == "P2")
        & (comparison["pair_family"].astype(str) == "TC-IM")
        & (comparison["metric"] == "balanced_minus_uot")
    ].iloc[0]
    assert bool(flip_row["sign_flip"])
    assert "descriptive_material_divergence_flag" in comparison.columns
    assert bool(flip_row["descriptive_material_divergence_flag"])


def test_compute_uot_row_prototype_quantities_preserves_component_masses() -> None:
    A = np.array(
        [
            [1.0, 1.0, 1.5],
            [0.8, 0.5, 0.7],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, 1.0, 1.5],
            [0.4, 1.0, 0.6],
        ],
        dtype=float,
    )
    edge_shares = np.array(
        [
            [
                [0.25, 0.0, 0.0],
                [0.10, 0.15, 0.0],
                [0.0, 0.0, 0.50],
            ],
            [
                [0.0, 0.4, 0.1],
                [0.0, 0.2, 0.0],
                [0.1, 0.0, 0.2],
            ],
        ],
        dtype=float,
    )
    transport_mass = np.array([2.0, 1.0], dtype=float)

    quantities = compute_uot_row_prototype_quantities(
        A=A,
        B=B,
        edge_shares=edge_shares,
        transport_mass=transport_mass,
    )

    np.testing.assert_allclose(
        np.sum(quantities["transport_source_abs"], axis=1),
        transport_mass,
    )
    np.testing.assert_allclose(
        np.sum(quantities["transport_target_abs"], axis=1),
        transport_mass,
    )
    np.testing.assert_allclose(
        np.sum(quantities["destroy_abs"], axis=1),
        np.sum(A, axis=1) - transport_mass,
    )
    np.testing.assert_allclose(
        np.sum(quantities["birth_abs"], axis=1),
        np.sum(B, axis=1) - transport_mass,
    )

    for key in (
        "transport_source_share",
        "transport_target_share",
        "destroy_share",
        "birth_share",
    ):
        np.testing.assert_allclose(np.sum(quantities[key], axis=1), np.array([1.0, 1.0], dtype=float))


def test_prototype_patient_family_aggregation_and_confirmatory_contrasts() -> None:
    pair_level = pd.DataFrame(
        {
            "patient_id_a": ["P1", "P1", "P1", "P1", "P1"],
            "pair_family": ["TC-IM", "TC-IM", "TC-PT", "TC-PT", "IM-PT"],
        }
    )
    annotation_summary = pd.DataFrame(
        {
            "proto_id": [0, 1],
            "dominant_cell_type": ["T_cell", "Fibroblast"],
            "dominant_cell_type_fraction": [0.7, 0.6],
            "top_cell_types": ["T_cell:0.7000", "Fibroblast:0.6000"],
            "prototype_total_cells": [10.0, 12.0],
        }
    )
    prototype_quantities = {
        name: np.zeros((5, 2), dtype=float)
        for name in PROTOTYPE_CONTRAST_QUANTITIES
    }
    prototype_quantities["transport_source_abs"] = np.array(
        [[1.0, 2.0], [0.5, 0.5], [0.5, 1.0], [0.5, 0.5], [9.0, 9.0]],
        dtype=float,
    )
    prototype_quantities["transport_target_abs"] = prototype_quantities["transport_source_abs"].copy()
    prototype_quantities["destroy_abs"] = np.array(
        [[0.2, 0.1], [0.0, 0.2], [0.6, 0.2], [0.2, 0.0], [1.0, 1.0]],
        dtype=float,
    )
    prototype_quantities["birth_abs"] = np.array(
        [[0.1, 0.2], [0.1, 0.0], [0.4, 0.3], [0.1, 0.2], [1.0, 1.0]],
        dtype=float,
    )

    transport, unmatched = aggregate_prototype_patient_family_tables(
        pair_level,
        active_proto_ids=np.array([0, 1], dtype=int),
        annotation_summary=annotation_summary,
        prototype_quantities=prototype_quantities,
    )
    contrasts = prototype_confirmatory_contrast_table(transport, unmatched)
    support = prototype_confirmatory_support_summary_table(contrasts)

    assert transport.shape[0] == 6
    assert unmatched.shape[0] == 6

    transport_lookup = transport.set_index(["patient_id_a", "pair_family", "proto_id"])
    assert transport_lookup.loc[("P1", "TC-IM", 0), "transport_source_abs"] == 1.5
    assert transport_lookup.loc[("P1", "TC-IM", 1), "transport_source_share"] == 2.5 / 4.0
    assert transport_lookup.loc[("P1", "TC-PT", 0), "transport_source_abs"] == 1.0

    contrast_lookup = contrasts.set_index(["patient_id_a", "proto_id"])
    assert contrast_lookup.loc[("P1", 0), "contrast_tc_pt_minus_tc_im_transport_source_abs"] == -0.5
    assert np.isclose(
        contrast_lookup.loc[("P1", 0), "contrast_tc_pt_minus_tc_im_transport_source_share"],
        0.4 - 0.375,
    )
    assert np.isclose(
        contrast_lookup.loc[("P1", 0), "contrast_tc_pt_minus_tc_im_destroy_abs"],
        0.6,
    )
    assert np.isclose(
        contrast_lookup.loc[("P1", 0), "contrast_tc_pt_minus_tc_im_destroy_share"],
        0.8 - 0.4,
    )

    support_lookup = support.set_index(["proto_id", "quantity"])
    assert support_lookup.loc[(0, "transport_source_abs"), "patient_count"] == 1
    assert support_lookup.loc[(0, "destroy_abs"), "expected_direction"] == "positive"


def test_transport_scale_metrics_preserve_absolute_quantities_and_normalize_fractions() -> None:
    pair_level = pd.DataFrame(
        [
            _row("P1", "TC-IM", T=6.0, U=2.0, D_pos=1.0, B_pos=1.0, M=0.8, balanced_minus_uot=0.1),
            _row("P1", "TC-PT", T=3.0, U=5.0, D_pos=2.5, B_pos=2.5, M=1.1, balanced_minus_uot=0.2),
            _row("P1", "IM-PT", T=4.0, U=4.0, D_pos=2.0, B_pos=2.0, M=0.9, balanced_minus_uot=0.0),
        ]
    )
    pair_level["M_balanced"] = pair_level["M"] + pair_level["balanced_minus_uot"]
    A = np.array([[8.0, 2.0], [4.0, 4.0], [5.0, 3.0]], dtype=float)
    B = np.array([[5.0, 5.0], [3.0, 6.0], [4.0, 4.0]], dtype=float)

    scaled = add_transport_scale_metrics(pair_level, A=A, B=B)
    validate_transport_scale_metrics(scaled)

    np.testing.assert_allclose(scaled["T_abs"].to_numpy(dtype=float), scaled["T"].to_numpy(dtype=float))
    np.testing.assert_allclose(scaled["U_abs"].to_numpy(dtype=float), scaled["U"].to_numpy(dtype=float))
    np.testing.assert_allclose(
        scaled["transport_fraction"].to_numpy(dtype=float) + scaled["unmatched_fraction"].to_numpy(dtype=float),
        np.ones(scaled.shape[0], dtype=float),
    )
    assert np.all((scaled["transport_fraction"].to_numpy(dtype=float) >= 0.0) & (scaled["transport_fraction"].to_numpy(dtype=float) <= 1.0))
    assert np.all((scaled["unmatched_fraction"].to_numpy(dtype=float) >= 0.0) & (scaled["unmatched_fraction"].to_numpy(dtype=float) <= 1.0))

    median_summary = confirmatory_patient_family_summary_table(
        scaled,
        summary_method="median",
        metrics=CONFIRMATORY_SUMMARY_METRICS_WITH_NORMALIZED_T,
    )
    mean_summary = confirmatory_patient_family_summary_table(
        scaled,
        summary_method="mean",
        metrics=CONFIRMATORY_SUMMARY_METRICS_WITH_NORMALIZED_T,
    )
    contrasts = confirmatory_patient_contrast_table(
        median_summary,
        mean_summary,
        metrics=CONFIRMATORY_SUMMARY_METRICS_WITH_NORMALIZED_T,
    )

    assert set(median_summary["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert "median_contrast_tc_pt_minus_tc_im_transport_fraction" in contrasts.columns
    assert "median_contrast_tc_pt_minus_tc_im_transport_over_source_total" in contrasts.columns


def test_balanced_transport_comparison_tables_are_internally_consistent() -> None:
    pair_level = pd.DataFrame(
        {
            "patient_id_a": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "pair_family": ["TC-IM", "TC-IM", "TC-PT", "TC-PT", "IM-PT", "IM-PT"],
        }
    )
    annotation_summary = pd.DataFrame(
        {
            "proto_id": [0, 1],
            "dominant_cell_type": ["Tumor", "Myeloid"],
            "dominant_cell_type_fraction": [0.8, 0.7],
            "top_cell_types": ["Tumor:0.8000", "Myeloid:0.7000"],
            "prototype_total_cells": [20.0, 15.0],
        }
    )

    uot_edge_shares = np.array(
        [
            [[0.6, 0.0], [0.0, 0.4]],
            [[0.5, 0.1], [0.0, 0.4]],
            [[0.3, 0.2], [0.1, 0.4]],
            [[0.2, 0.2], [0.1, 0.5]],
            [[0.4, 0.1], [0.1, 0.4]],
            [[0.3, 0.2], [0.2, 0.3]],
        ],
        dtype=float,
    )
    balanced_edge_shares = np.array(
        [
            [[0.7, 0.0], [0.0, 0.3]],
            [[0.6, 0.1], [0.0, 0.3]],
            [[0.5, 0.2], [0.1, 0.2]],
            [[0.4, 0.2], [0.1, 0.3]],
            [[0.5, 0.1], [0.1, 0.3]],
            [[0.4, 0.2], [0.2, 0.2]],
        ],
        dtype=float,
    )
    uot_source_mass = np.array([5.0, 6.0, 4.0, 5.0, 3.0, 3.5], dtype=float)
    uot_target_mass = uot_source_mass.copy()
    balanced_source_total = np.array([8.0, 8.0, 7.0, 7.0, 6.0, 6.0], dtype=float)
    balanced_target_total = np.array([7.0, 7.0, 6.0, 6.0, 5.0, 5.0], dtype=float)

    uot_quantities = compute_transport_row_prototype_quantities(
        edge_shares=uot_edge_shares,
        source_transport_mass=uot_source_mass,
        target_transport_mass=uot_target_mass,
    )
    balanced_quantities = compute_balanced_row_prototype_transport_quantities(
        edge_shares=balanced_edge_shares,
        source_total_mass=balanced_source_total,
        target_total_mass=balanced_target_total,
    )

    uot_transport = aggregate_prototype_transport_patient_family_table(
        pair_level,
        active_proto_ids=np.array([0, 1], dtype=int),
        annotation_summary=annotation_summary,
        prototype_quantities=uot_quantities,
        transport_method="uot",
    )
    balanced_transport = aggregate_prototype_transport_patient_family_table(
        pair_level,
        active_proto_ids=np.array([0, 1], dtype=int),
        annotation_summary=annotation_summary,
        prototype_quantities=balanced_quantities,
        transport_method="balanced",
    )
    validate_prototype_transport_table(uot_transport)
    validate_prototype_transport_table(balanced_transport)

    patient_family_delta = prototype_uot_vs_balanced_patient_family_delta_table(
        uot_transport,
        balanced_transport,
    )
    validate_prototype_transport_delta_table(patient_family_delta)
    confirmatory_contrasts = prototype_uot_vs_balanced_confirmatory_contrast_table(patient_family_delta)

    assert set(confirmatory_contrasts["patient_id_a"].astype(str)) == {"P1"}
    assert set(patient_family_delta["pair_family"].astype(str)) == {"TC-IM", "TC-PT", "IM-PT"}

    lookup = patient_family_delta.set_index(["patient_id_a", "pair_family", "proto_id"])
    proto0_tc_im = lookup.loc[("P1", "TC-IM", 0)]
    expected_delta = proto0_tc_im["transport_source_abs_balanced"] - proto0_tc_im["transport_source_abs_uot"]
    assert np.isclose(proto0_tc_im["delta_transport_source_abs"], expected_delta)

    contrast_lookup = confirmatory_contrasts.set_index(["patient_id_a", "proto_id"])
    expected_confirmatory = (
        lookup.loc[("P1", "TC-PT", 0), "delta_transport_source_share"]
        - lookup.loc[("P1", "TC-IM", 0), "delta_transport_source_share"]
    )
    assert np.isclose(
        contrast_lookup.loc[("P1", 0), "contrast_tc_pt_minus_tc_im_delta_transport_source_share"],
        expected_confirmatory,
    )
