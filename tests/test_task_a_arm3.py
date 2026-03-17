from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import re
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

if importlib.util.find_spec("anndata") is None:
    anndata_stub = types.ModuleType("anndata")
    anndata_stub.__spec__ = importlib.util.spec_from_loader("anndata", loader=None)

    class _AnnData:  # pragma: no cover - import stub only
        pass

    anndata_stub.AnnData = _AnnData
    sys.modules["anndata"] = anndata_stub

from slotar.uot import UOTSolveConfig
from tasks.task_A.arm3 import calibrate as arm3_calibrate
from tasks.task_A.arm3 import inference as arm3_inference
from tasks.task_A.arm3 import output as arm3_output
from tasks.task_A import arm3_uq_stress


def test_run_uot_batch_with_events_uses_exact_solver_details_and_skips_proportional_path(
    monkeypatch,
) -> None:
    A = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    pair_meta = pd.DataFrame({"pair_id": ["pair_0", "pair_1"]})
    lambda_pl = np.array([1.0, 1.5], dtype=float)
    tau_external = np.array([0.3, 0.7], dtype=float)
    uot_cfg = UOTSolveConfig(eps_schedule=[1.0])

    expected_metrics = {
        "T": np.array([2.0, 1.0], dtype=float),
        "D_pos": np.array([1.5, 0.8], dtype=float),
        "B_pos": np.array([1.0, 1.2], dtype=float),
        "d_rel": np.array([0.5, 0.4], dtype=float),
        "b_rel": np.array([0.25, 0.3], dtype=float),
        "M": np.array([0.4, 0.6], dtype=float),
        "R": np.array([0.7, 0.8], dtype=float),
        "tau": tau_external.copy(),
    }
    expected_T_k = np.array(
        [
            [0.5, 1.5],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    expected_D_k = np.array(
        [
            [1.5, 0.0],
            [0.8, 0.0],
        ],
        dtype=float,
    )
    expected_B_k = np.array(
        [
            [0.25, 0.75],
            [0.2, 1.0],
        ],
        dtype=float,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Proportional allocation fallback must not run in the Arm-3 main path")

    def fake_batched_uot_solve(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels,
        cfg: UOTSolveConfig,
        tau_external: np.ndarray | None = None,
        external_support_mask: np.ndarray | None = None,
        return_plan: bool = False,
    ):
        del kernels, cfg, external_support_mask
        assert tau_external is not None
        assert return_plan is False
        np.testing.assert_allclose(A, np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float))
        np.testing.assert_allclose(B, np.array([[1.0, 3.0], [2.0, 2.0]], dtype=float))
        np.testing.assert_allclose(lambda_pl, np.array([1.0, 1.5], dtype=float))
        details = {
            "source_transport_marginal": expected_T_k.copy(),
            "target_transport_marginal": np.array([[0.6, 1.4], [0.0, 1.0]], dtype=float),
            "T_k": expected_T_k.copy(),
            "D_k": expected_D_k.copy(),
            "B_k": expected_B_k.copy(),
        }
        return expected_metrics, details, np.array(["ok", "ok"], dtype=object)

    monkeypatch.setattr(arm3_inference, "extract_prototype_event_marginals", fail_if_called)
    monkeypatch.setattr(arm3_inference, "batched_uot_solve", fake_batched_uot_solve)

    df_result, T_k, B_k, D_k = arm3_inference.run_uot_batch_with_events(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=[np.zeros((2, 2), dtype=float)],
        uot_cfg=uot_cfg,
        pair_meta=pair_meta,
        tau_external=tau_external,
    )

    np.testing.assert_allclose(T_k, expected_T_k)
    np.testing.assert_allclose(B_k, expected_B_k)
    np.testing.assert_allclose(D_k, expected_D_k)
    np.testing.assert_allclose(df_result["T"].to_numpy(dtype=float), expected_metrics["T"])
    np.testing.assert_allclose(df_result["D_pos"].to_numpy(dtype=float), expected_metrics["D_pos"])
    np.testing.assert_allclose(df_result["B_pos"].to_numpy(dtype=float), expected_metrics["B_pos"])
    np.testing.assert_allclose(T_k.sum(axis=1), df_result["T"].to_numpy(dtype=float))
    np.testing.assert_allclose(B_k.sum(axis=1), df_result["B_pos"].to_numpy(dtype=float))
    np.testing.assert_allclose(D_k.sum(axis=1), df_result["D_pos"].to_numpy(dtype=float))

    proportional_T = expected_metrics["T"][:, None] * (
        A / (A.sum(axis=1, keepdims=True) + arm3_inference.DENSITY_EPS)
    )
    assert not np.allclose(T_k, proportional_T)

    forbidden_tensor_cols = {"T_k", "B_k", "D_k", "source_transport_marginal", "target_transport_marginal", "Pi"}
    assert forbidden_tensor_cols.isdisjoint(df_result.columns)


def test_run_uot_batch_with_events_passes_frozen_support_mask_to_solver(monkeypatch) -> None:
    A = np.array([[0.2, 0.8, 0.0]], dtype=float)
    B = np.array([[0.0, 0.5, 0.5]], dtype=float)
    A_count = np.array([[1.0, 4.0, 2.0]], dtype=float)
    B_count = np.array([[0.0, 1.0, 2.0]], dtype=float)
    support_mask = arm3_inference.freeze_support_masks(
        A_count=A_count,
        B_count=B_count,
        n_min_proto=2.0,
        k_full=3,
    )
    captured: dict[str, np.ndarray] = {}

    def fake_batched_uot_solve(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels,
        cfg: UOTSolveConfig,
        tau_external: np.ndarray | None = None,
        external_support_mask: np.ndarray | None = None,
        return_plan: bool = False,
    ):
        del A, B, lambda_pl, kernels, cfg, tau_external, return_plan
        assert external_support_mask is not None
        captured["external_support_mask"] = np.asarray(external_support_mask, dtype=bool).copy()
        metrics = {
            "T": np.array([1.0], dtype=float),
            "D_pos": np.array([0.0], dtype=float),
            "B_pos": np.array([0.5], dtype=float),
            "d_rel": np.array([0.0], dtype=float),
            "b_rel": np.array([0.1], dtype=float),
            "M": np.array([0.25], dtype=float),
            "R": np.array([np.nan], dtype=float),
            "tau": np.array([np.nan], dtype=float),
        }
        details = {
            "T_k": np.array([[0.0, 0.5, 0.5]], dtype=float),
            "D_k": np.array([[0.0, 0.0, 0.0]], dtype=float),
            "B_k": np.array([[0.0, 0.0, 0.5]], dtype=float),
        }
        return metrics, details, np.array(["ok"], dtype=object)

    monkeypatch.setattr(arm3_inference, "batched_uot_solve", fake_batched_uot_solve)

    df_result, T_k, B_k, D_k = arm3_inference.run_uot_batch_with_events(
        A=A,
        B=B,
        lambda_pl=np.array([1.0], dtype=float),
        kernels=[np.zeros((3, 3), dtype=float)],
        uot_cfg=UOTSolveConfig(eps_schedule=[1.0], n_min_proto=2.0),
        pair_meta=pd.DataFrame({"pair_id": ["pair_0"]}),
        external_support_mask=support_mask,
    )

    np.testing.assert_array_equal(captured["external_support_mask"], support_mask)
    assert captured["external_support_mask"].dtype == bool
    np.testing.assert_allclose(T_k, np.array([[0.0, 0.5, 0.5]], dtype=float))
    np.testing.assert_allclose(B_k, np.array([[0.0, 0.0, 0.5]], dtype=float))
    np.testing.assert_allclose(D_k, np.array([[0.0, 0.0, 0.0]], dtype=float))
    assert "T_k" not in df_result.columns


def test_weighted_plan_cost_quantile_excludes_diagonal_mass_from_tau_support() -> None:
    plans = np.array(
        [
            [[9.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=float,
    )
    scaled_cost_matrix = np.array(
        [
            [0.0, 4.0],
            [7.0, 0.0],
        ],
        dtype=float,
    )

    tau = arm3_calibrate._weighted_plan_cost_quantile(
        plans=plans,
        scaled_cost_matrix=scaled_cost_matrix,
        q=0.5,
    )

    assert tau == 4.0


def test_calibrate_tau_by_compartment_uses_explicit_scaled_cost_matrix_and_non_diagonal_pooled_pi_weights(
    monkeypatch,
) -> None:
    roi_density_vectors = {
        "tc_1": np.array([1.0, 0.0], dtype=float),
        "tc_2": np.array([0.0, 1.0], dtype=float),
        "im_1": np.array([1.0, 0.0], dtype=float),
        "im_2": np.array([0.0, 1.0], dtype=float),
        "pt_1": np.array([1.0, 0.0], dtype=float),
        "pt_2": np.array([0.0, 1.0], dtype=float),
    }
    roi_compartment_map = {
        "tc_1": "TC",
        "tc_2": "TC",
        "im_1": "IM",
        "im_2": "IM",
        "pt_1": "PT",
        "pt_2": "PT",
    }
    roi_patient_map = {roi_id: "P01" for roi_id in roi_density_vectors}
    scaled_cost_matrix = np.array(
        [
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=float,
    )

    def fake_batched_uot_solve(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels,
        cfg: UOTSolveConfig,
        tau_external: np.ndarray | None = None,
        external_support_mask: np.ndarray | None = None,
        return_plan: bool = False,
    ):
        del A, B, lambda_pl, kernels, cfg, tau_external, external_support_mask
        assert return_plan is True
        metrics = {
            "T": np.array([100.0, 1.0], dtype=float),
            "D_pos": np.array([0.0, 0.0], dtype=float),
            "B_pos": np.array([0.0, 0.0], dtype=float),
            "d_rel": np.array([0.0, 0.0], dtype=float),
            "b_rel": np.array([0.0, 0.0], dtype=float),
            "M": np.array([0.0, 15.0], dtype=float),
            "R": np.array([np.nan, np.nan], dtype=float),
            "tau": np.array([np.nan, np.nan], dtype=float),
        }
        details = {
            "Pi": np.array(
                [
                    [[100.0, 1.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ],
                dtype=float,
            ),
        }
        return metrics, details, np.array(["ok", "ok"], dtype=object)

    monkeypatch.setattr(arm3_calibrate, "batched_uot_solve", fake_batched_uot_solve)

    tau_by_compartment = arm3_calibrate.calibrate_tau_by_compartment(
        roi_density_vectors=roi_density_vectors,
        roi_compartment_map=roi_compartment_map,
        roi_patient_map=roi_patient_map,
        k_full=2,
        scaled_cost_matrix=scaled_cost_matrix,
        frozen_lambdas={"TC-IM": 1.0, "IM-PT": 1.0, "TC-PT": 1.0},
        uot_cfg=UOTSolveConfig(eps_schedule=[1.0]),
        kernels=[np.array([[-123.0, -456.0], [-789.0, -1011.0]], dtype=float)],
        tau_q=0.5,
    )

    assert tau_by_compartment == {"TC": 10.0, "IM": 10.0, "PT": 10.0}
    assert tau_by_compartment["TC"] != 0.0
    assert tau_by_compartment["TC"] != 7.5


def _phase8_proto_contrast_records_for_all_prototypes() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    patients = ["P01", "P02"]
    coverages = [0.75, 0.50, 0.25]

    for proto_k in range(25):
        label = f"proto_{proto_k}"
        for patient_idx, patient_id in enumerate(patients):
            full_ref = float(proto_k + patient_idx + 1)
            records.append(
                {
                    "patient_id": patient_id,
                    "coverage": 1.0,
                    "replicate_id": np.nan,
                    "prototype_k": proto_k,
                    "prototype_label": label,
                    "U_k_TC_IM": 0.0,
                    "U_k_TC_PT": full_ref,
                    "Delta_U_k": full_ref,
                }
            )
            for coverage in coverages:
                for rep_id in range(2):
                    records.append(
                        {
                            "patient_id": patient_id,
                            "coverage": coverage,
                            "replicate_id": rep_id,
                            "prototype_k": proto_k,
                            "prototype_label": label,
                            "U_k_TC_IM": 0.0,
                            "U_k_TC_PT": full_ref + coverage + rep_id,
                            "Delta_U_k": full_ref + coverage + rep_id,
                        }
                    )

    return pd.DataFrame.from_records(records)


def test_build_prototype_stability_table_covers_all_prototypes_and_excludes_recurrence() -> None:
    df_proto_contrast = _phase8_proto_contrast_records_for_all_prototypes()

    df_result = arm3_output.build_prototype_stability_table(df_proto_contrast)

    assert list(df_result.columns) == [
        "prototype_k",
        "prototype_label",
        "coverage",
        "sign_consistency_rate",
        "n_evaluable_patients",
        "n_zero_reference_patients",
        "correlation_to_full_cov",
        "n_correlation_patients",
    ]
    assert "recurrence_proportion" not in df_result.columns
    assert df_result["prototype_k"].nunique() == 25
    assert sorted(df_result["coverage"].unique().tolist()) == [0.25, 0.5, 0.75]
    assert len(df_result) == 25 * 3


def test_build_prototype_stability_table_uses_patient_medians_and_zero_reference_exclusion() -> None:
    df_proto_contrast = pd.DataFrame.from_records(
        [
            {
                "patient_id": "P1",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 1.0,
                "Delta_U_k": 1.0,
            },
            {
                "patient_id": "P2",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": -2.0,
                "Delta_U_k": -2.0,
            },
            {
                "patient_id": "P3",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.0,
                "Delta_U_k": 0.0,
            },
            {
                "patient_id": "P4",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 1.0,
                "Delta_U_k": 1.0,
            },
            {
                "patient_id": "P1",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 1.5,
                "Delta_U_k": 1.5,
            },
            {
                "patient_id": "P1",
                "coverage": 0.5,
                "replicate_id": 1,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.5,
                "Delta_U_k": 0.5,
            },
            {
                "patient_id": "P2",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": -1.0,
                "Delta_U_k": -1.0,
            },
            {
                "patient_id": "P2",
                "coverage": 0.5,
                "replicate_id": 1,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.0,
                "Delta_U_k": 0.0,
            },
            {
                "patient_id": "P3",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 9.0,
                "Delta_U_k": 9.0,
            },
            {
                "patient_id": "P3",
                "coverage": 0.5,
                "replicate_id": 1,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 9.0,
                "Delta_U_k": 9.0,
            },
            {
                "patient_id": "P4",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.0,
                "Delta_U_k": 0.0,
            },
            {
                "patient_id": "P4",
                "coverage": 0.5,
                "replicate_id": 1,
                "prototype_k": 7,
                "prototype_label": "proto_7",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.0,
                "Delta_U_k": 0.0,
            },
        ]
    )

    df_result = arm3_output.build_prototype_stability_table(df_proto_contrast)
    row = df_result.iloc[0]

    assert row["prototype_k"] == 7
    assert row["coverage"] == 0.5
    assert row["n_evaluable_patients"] == 3
    assert row["n_zero_reference_patients"] == 1
    assert row["n_correlation_patients"] == 3
    assert row["sign_consistency_rate"] == 2.0 / 3.0

    expected_corr = np.corrcoef(
        np.array([1.0, -2.0, 1.0], dtype=float),
        np.array([1.0, -0.5, 0.0], dtype=float),
    )[0, 1]
    assert row["correlation_to_full_cov"] == expected_corr


def test_build_prototype_stability_table_returns_nan_correlation_for_insufficient_or_constant_inputs() -> None:
    df_proto_contrast = pd.DataFrame.from_records(
        [
            {
                "patient_id": "P1",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 2,
                "prototype_label": "proto_2",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 1.0,
                "Delta_U_k": 1.0,
            },
            {
                "patient_id": "P2",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 2,
                "prototype_label": "proto_2",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 0.0,
                "Delta_U_k": 0.0,
            },
            {
                "patient_id": "P1",
                "coverage": 0.25,
                "replicate_id": 0,
                "prototype_k": 2,
                "prototype_label": "proto_2",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 2.0,
                "Delta_U_k": 2.0,
            },
            {
                "patient_id": "P2",
                "coverage": 0.25,
                "replicate_id": 0,
                "prototype_k": 2,
                "prototype_label": "proto_2",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 3.0,
                "Delta_U_k": 3.0,
            },
            {
                "patient_id": "P1",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 3,
                "prototype_label": "proto_3",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 1.0,
                "Delta_U_k": 1.0,
            },
            {
                "patient_id": "P2",
                "coverage": 1.0,
                "replicate_id": np.nan,
                "prototype_k": 3,
                "prototype_label": "proto_3",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 2.0,
                "Delta_U_k": 2.0,
            },
            {
                "patient_id": "P1",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 3,
                "prototype_label": "proto_3",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 5.0,
                "Delta_U_k": 5.0,
            },
            {
                "patient_id": "P2",
                "coverage": 0.5,
                "replicate_id": 0,
                "prototype_k": 3,
                "prototype_label": "proto_3",
                "U_k_TC_IM": 0.0,
                "U_k_TC_PT": 5.0,
                "Delta_U_k": 5.0,
            },
        ]
    )

    df_result = arm3_output.build_prototype_stability_table(df_proto_contrast)

    row_insufficient = df_result.loc[
        (df_result["prototype_k"] == 2) & (df_result["coverage"] == 0.25)
    ].iloc[0]
    row_constant = df_result.loc[
        (df_result["prototype_k"] == 3) & (df_result["coverage"] == 0.5)
    ].iloc[0]

    assert np.isnan(row_insufficient["correlation_to_full_cov"])
    assert row_insufficient["n_correlation_patients"] == 1
    assert np.isnan(row_constant["correlation_to_full_cov"])
    assert row_constant["n_correlation_patients"] == 2


def test_build_arm3_memo_stays_descriptive() -> None:
    df_degradation = pd.DataFrame(
        {
            "patient_id": ["P01"],
            "pair_type": ["TC->IM"],
            "coverage": [0.5],
            "quantity": ["U_abs_dens"],
            "median_abs_degradation": [1.25],
            "sign_consistency_rate": [0.75],
            "floor_dominated_rate": [0.0],
            "mean_replicate_value": [2.0],
            "std_replicate_value": [0.3],
        }
    )
    df_contrast = pd.DataFrame(
        {
            "patient_id": ["P01"],
            "coverage": [0.5],
            "contrast_name": ["Delta_U_abs"],
            "reference_contrast": [1.0],
            "median_replicate_contrast": [1.5],
            "abs_degradation": [0.5],
            "sign_consistency_rate": [0.75],
            "n_evaluable": [4],
            "n_zero_reference_sign": [1],
        }
    )
    df_prototype_stability = pd.DataFrame(
        {
            "prototype_k": [0],
            "prototype_label": ["proto_0"],
            "coverage": [0.5],
            "sign_consistency_rate": [0.75],
            "n_evaluable_patients": [4],
            "n_zero_reference_patients": [1],
            "correlation_to_full_cov": [0.33],
            "n_correlation_patients": [4],
        }
    )

    memo_text = arm3_output.build_arm3_memo(
        result_root=ROOT / "tmp_phase8_result",
        df_degradation=df_degradation,
        df_contrast=df_contrast,
        df_prototype_stability=df_prototype_stability,
        calibration_record={
            "run_utc": "2026-03-17T00:00:00+00:00",
            "lambda_dens": {"TC-IM": 10.0},
            "tau_by_compartment": {"TC": 0.8},
            "tau_q": 0.5,
            "tau_calibration_rule": "do_not_use_this_text",
        },
    )

    assert "Result root" in memo_text
    assert "Thresholded interpretation is deferred to downstream analysis." in memo_text
    assert "tau_calibration_rule" not in memo_text
    prohibited = re.compile(r"\b(stable|unstable|retained|weakened|passed|failed)\b", re.IGNORECASE)
    assert prohibited.search(memo_text) is None


def test_finalize_arm3_phase8_loads_artifacts_and_writes_outputs(tmp_path) -> None:
    df_proto_contrast = _phase8_proto_contrast_records_for_all_prototypes()
    df_degradation = pd.DataFrame(
        {
            "patient_id": ["P01"],
            "pair_type": ["TC->IM"],
            "coverage": [0.5],
            "quantity": ["U_abs_dens"],
            "median_abs_degradation": [1.25],
            "sign_consistency_rate": [0.75],
            "floor_dominated_rate": [0.0],
            "mean_replicate_value": [2.0],
            "std_replicate_value": [0.3],
        }
    )
    df_contrast = pd.DataFrame(
        {
            "patient_id": ["P01"],
            "coverage": [0.5],
            "contrast_name": ["Delta_U_abs"],
            "reference_contrast": [1.0],
            "median_replicate_contrast": [1.5],
            "abs_degradation": [0.5],
            "sign_consistency_rate": [0.75],
            "n_evaluable": [4],
            "n_zero_reference_sign": [1],
        }
    )

    df_degradation.to_parquet(
        tmp_path / "arm3_phase7_degradation_summary.parquet",
        index=False,
    )
    df_contrast.to_parquet(
        tmp_path / "arm3_phase7_contrast_summary.parquet",
        index=False,
    )
    df_proto_contrast.to_parquet(
        tmp_path / "arm3_phase8_prototype_contrast_prep.parquet",
        index=False,
    )
    with open(tmp_path / "arm3_phase4_calibration_record.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_utc": "2026-03-17T00:00:00+00:00",
                "lambda_dens": {"TC-IM": 10.0, "TC-PT": 10.0, "IM-PT": 10.0},
                "tau_by_compartment": {"TC": 0.8, "IM": 0.6, "PT": 0.5},
                "tau_q": 0.5,
                "lambda_grid_used": [0.1, 1.0, 10.0],
                "target_alpha": 0.05,
                "tau_calibration_rule": "ignore_this_text",
            },
            fh,
            indent=2,
        )

    df_phase8, memo_text = arm3_uq_stress.finalize_arm3_phase8(result_root=tmp_path)

    assert df_phase8["prototype_k"].nunique() == 25
    assert len(df_phase8) == 25 * 3
    assert (tmp_path / "arm3_phase8_prototype_stability.parquet").exists()
    assert (tmp_path / "arm3_phase8_prototype_stability.csv").exists()
    assert (tmp_path / "arm3_phase8_memo.md").exists()
    assert "tau_calibration_rule" not in memo_text
    assert "recurrence_proportion" not in df_phase8.columns


def test_arm3_metadata_writers_reflect_current_phase_scope(tmp_path) -> None:
    block_frame = pd.DataFrame({"block_id": ["roi_1::0::0"]})
    roi_block_universe = {"roi_1": ["roi_1::0::0"]}
    roi_block_summary = {"roi_1": pd.DataFrame({"block_id": ["roi_1::0::0"]})}
    roi_density_vectors = {"roi_1": np.zeros(25, dtype=float)}
    roi_total_areas = {"roi_1": 1.0}
    pair_meta_full = pd.DataFrame({"pair_id": ["pair_full_1"]})
    pair_meta_anchor = pd.DataFrame({"pair_id": ["pair_anchor_1"]})

    arm3_uq_stress._write_phase0_3_outputs(
        result_root=tmp_path,
        stage0_path=tmp_path / "task_A_stage0_k25.h5ad",
        block_size_units=100.0,
        block_area_mm2=0.01,
        spatial_xy=np.array([[0.0, 1.0]], dtype=float),
        block_frame=block_frame,
        roi_block_universe=roi_block_universe,
        roi_block_summary=roi_block_summary,
        roi_density_vectors=roi_density_vectors,
        roi_total_areas=roi_total_areas,
        pair_meta_full=pair_meta_full,
        pair_meta_anchor=pair_meta_anchor,
    )
    arm3_uq_stress._write_phase4_outputs(
        result_root=tmp_path,
        lambda_dens={"TC-IM": 10.0, "TC-PT": 10.0, "IM-PT": 10.0},
        lambda_grid=(0.1, 1.0, 10.0),
        target_alpha=0.05,
        tau_by_compartment={"TC": 0.8, "IM": 0.6, "PT": 0.5},
        tau_q=0.5,
    )

    with open(tmp_path / "arm3_phase0_manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(tmp_path / "arm3_phase4_calibration_record.json", "r", encoding="utf-8") as fh:
        calibration_record = json.load(fh)

    assert manifest["phase_implemented"] == "0-8"
    assert manifest["phase_4_plus"] == "IMPLEMENTED"
    assert calibration_record["phase"] == 4
    assert calibration_record["phase_5_plus"] == "IMPLEMENTED"
