from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def test_generate_cross_compartment_pairs_is_deterministic_and_exhaustive() -> None:
    from tasks.task_A.arm2_spatial_gradient import generate_cross_compartment_pairs
    from tests.helpers_task_a_fixture import build_task_a_fixture, expected_arm2_pair_records

    adata = build_task_a_fixture()
    pair_meta = generate_cross_compartment_pairs(adata)
    repeated = generate_cross_compartment_pairs(adata)
    expected = pd.DataFrame.from_records(expected_arm2_pair_records())

    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), repeated.reset_index(drop=True))
    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), expected)
    assert pair_meta.shape[0] == 48
    assert pair_meta["patient_group_id"].is_unique
    assert pair_meta["same_patient"].all()
    assert (~pair_meta["same_compartment"]).all()
    assert (pair_meta["patient_id_a"] == pair_meta["patient_id_b"]).all()
    assert (pair_meta["compartment_a"] != pair_meta["compartment_b"]).all()


def test_run_arm2_calibrates_per_family_and_keeps_tau_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slotar.uot import UOTSolveConfig, precompute_logKernels
    from tasks.task_A.arm2_spatial_gradient import ARM_NAME, PAIR_FAMILIES, run_arm2
    from tests.helpers_task_a_fixture import K_FULL, build_task_a_fixture

    adata = build_task_a_fixture()
    cfg = {
        "data": {"k_full": K_FULL, "mass_mode": "count"},
        "arm2": {"target_alpha": 0.05, "lambda_grid": [0.05, 0.1, 1.0]},
    }
    uot_cfg = UOTSolveConfig(eps_schedule=[1.0], max_iter=100, tol=1e-8, n_min_proto=0.0)
    kernels = precompute_logKernels(np.asarray(adata.uns["cost_matrix"], dtype=float), uot_cfg.eps_schedule)

    calibrated_values = iter([0.5, 1.5, 2.5])
    captured: dict[str, object] = {}

    def fake_calibrate_joint_lambda(
        A: np.ndarray,
        B: np.ndarray,
        lambda_grid: tuple[float, ...],
        kernels: list[np.ndarray],
        cfg: UOTSolveConfig,
        target_alpha: float = 0.05,
    ) -> float:
        del kernels, cfg
        captured.setdefault("calibration_calls", []).append(
            {
                "n_rows": A.shape[0],
                "lambda_grid": tuple(lambda_grid),
                "target_alpha": target_alpha,
            }
        )
        return next(calibrated_values)

    def fake_run_uot_batch_safe(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels: list[np.ndarray],
        uot_cfg: UOTSolveConfig,
        pair_meta: pd.DataFrame,
        tau_external: np.ndarray | None = None,
    ) -> pd.DataFrame:
        del A, B, kernels, uot_cfg
        captured["lambda_pl"] = lambda_pl.copy()
        captured["pair_meta"] = pair_meta.copy()
        captured["tau_external"] = tau_external

        result = pair_meta.reset_index(drop=True).copy()
        n_rows = result.shape[0]
        result["lambda_pl"] = lambda_pl
        result["uot_status"] = np.full(n_rows, "ok", dtype=object)
        result["bypass_reason"] = pd.Series([None] * n_rows, dtype="object")
        result["mass_pruned_ratio"] = np.zeros(n_rows, dtype=float)
        result["n_min_proto_used"] = 0.0
        result["S0"] = np.full(n_rows, 4.0, dtype=float)
        result["S1"] = np.full(n_rows, 4.0, dtype=float)
        result["scale_ratio"] = np.full(n_rows, 1.0, dtype=float)
        result["log_scale"] = np.zeros(n_rows, dtype=float)
        result["T"] = np.full(n_rows, 2.0, dtype=float)
        result["D_pos"] = np.full(n_rows, 0.3, dtype=float)
        result["B_pos"] = np.full(n_rows, 0.2, dtype=float)
        result["d_rel"] = np.full(n_rows, 0.1, dtype=float)
        result["b_rel"] = np.full(n_rows, 0.1, dtype=float)
        result["M"] = np.full(n_rows, 0.4, dtype=float)
        result["R"] = np.full(n_rows, np.nan, dtype=float)
        result["tau"] = np.full(n_rows, np.nan, dtype=float)
        result["U"] = result["D_pos"] + result["B_pos"]
        return result

    def fake_run_balanced_ot_batch(
        A: np.ndarray,
        B: np.ndarray,
        cost_matrix: np.ndarray,
        n_min_proto: float,
    ) -> np.ndarray:
        del A, B, cost_matrix, n_min_proto
        return np.linspace(1.0, 2.0, 48, dtype=float)

    monkeypatch.setattr("tasks.task_A.arm2_spatial_gradient.calibrate_joint_lambda", fake_calibrate_joint_lambda)
    monkeypatch.setattr("tasks.task_A.arm2_spatial_gradient.run_uot_batch_safe", fake_run_uot_batch_safe)
    monkeypatch.setattr("tasks.task_A.arm2_spatial_gradient.run_balanced_ot_batch", fake_run_balanced_ot_batch)

    df_result = run_arm2(adata, cfg, uot_cfg, kernels)

    calibration_calls = captured["calibration_calls"]
    assert len(calibration_calls) == len(PAIR_FAMILIES)
    assert [call["n_rows"] for call in calibration_calls] == [16, 16, 16]
    assert all(call["lambda_grid"] == (0.05, 0.1, 1.0) for call in calibration_calls)
    assert all(call["target_alpha"] == 0.05 for call in calibration_calls)
    assert captured["tau_external"] is None

    pair_meta = captured["pair_meta"]
    lambda_pl = captured["lambda_pl"]
    expected_lambda = {"TC-IM": 0.5, "IM-PT": 1.5, "TC-PT": 2.5}
    for pair_family, expected_value in expected_lambda.items():
        family_mask = pair_meta["pair_family"].astype(str) == pair_family
        np.testing.assert_allclose(lambda_pl[family_mask.to_numpy()], expected_value)

    assert (df_result["arm"] == ARM_NAME).all()
    assert (df_result["lambda_mode"] == "pair_specific_joint").all()
    assert (df_result["tau_mode"] == "unavailable").all()
    assert df_result["tau"].isna().all()
    assert df_result["R"].isna().all()
    assert np.isfinite(df_result["M_balanced"].to_numpy(dtype=float)).all()


def test_run_balanced_ot_batch_returns_shape_only_costs() -> None:
    from tasks.task_A.common import run_balanced_ot_batch

    A = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    result = run_balanced_ot_batch(A, B, cost_matrix=cost, n_min_proto=0.0)

    np.testing.assert_allclose(result[:2], np.array([1.0, 0.0], dtype=float))
    assert np.isnan(result[2])
