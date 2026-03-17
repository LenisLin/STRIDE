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


def test_generate_within_compartment_pairs_is_seeded_and_draw_aware() -> None:
    from tasks.task_A.arm1_noise_baseline import generate_within_compartment_pairs
    from tests.helpers_task_a_fixture import build_task_a_fixture, expected_arm1_pair_records

    adata = build_task_a_fixture()
    n_draws = 2
    random_seed = 7

    pair_meta = generate_within_compartment_pairs(
        adata,
        n_draws=n_draws,
        random_seed=random_seed,
    )
    repeated = generate_within_compartment_pairs(
        adata,
        n_draws=n_draws,
        random_seed=random_seed,
    )
    expected = pd.DataFrame.from_records(
        expected_arm1_pair_records(
            n_draws=n_draws,
            random_seed=random_seed,
        )
    )

    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), repeated.reset_index(drop=True))
    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), expected)
    assert pair_meta.shape[0] == n_draws * 6
    assert pair_meta["patient_group_id"].is_unique
    assert pair_meta["same_patient"].all()
    assert pair_meta["same_compartment"].all()
    assert (pair_meta["roi_a"] != pair_meta["roi_b"]).all()


def test_generate_broken_reference_pairs_is_seeded_and_breaks_locality() -> None:
    from tasks.task_A.arm1_broken_reference import generate_broken_reference_pairs
    from tests.helpers_task_a_fixture import (
        build_task_a_fixture,
        expected_broken_reference_pair_records,
    )

    adata = build_task_a_fixture()
    n_draws = 2
    random_seed = 7

    pair_meta = generate_broken_reference_pairs(
        adata,
        n_draws=n_draws,
        random_seed=random_seed,
    )
    repeated = generate_broken_reference_pairs(
        adata,
        n_draws=n_draws,
        random_seed=random_seed,
    )
    expected = pd.DataFrame.from_records(
        expected_broken_reference_pair_records(
            n_draws=n_draws,
            random_seed=random_seed,
        )
    )

    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), repeated.reset_index(drop=True))
    pd.testing.assert_frame_equal(pair_meta.reset_index(drop=True), expected)
    assert pair_meta.shape[0] == n_draws * 6
    assert pair_meta["patient_group_id"].is_unique
    assert (~pair_meta["same_patient"]).all()
    assert (~pair_meta["same_compartment"]).all()
    assert (pair_meta["patient_id_a"] != pair_meta["patient_id_b"]).all()
    assert (pair_meta["compartment_a"] != pair_meta["compartment_b"]).all()
    assert (pair_meta["roi_a"] != pair_meta["roi_b"]).all()


def test_assemble_tensors_count_mode_matches_fixture_vectors() -> None:
    from tasks.task_A.common import assemble_tensors
    from tests.helpers_task_a_fixture import K_FULL, build_task_a_fixture, expected_roi_vectors

    adata = build_task_a_fixture()
    pair_meta = pd.DataFrame(
        [
            {"roi_a": "P01_TC_01", "roi_b": "P01_TC_02"},
            {"roi_a": "P02_IM_01", "roi_b": "P02_IM_02"},
        ]
    )

    A, B, mass_gap = assemble_tensors(adata, pair_meta, k_full=K_FULL, mass_mode="count")
    vectors = expected_roi_vectors()

    np.testing.assert_allclose(A[0], vectors["P01_TC_01"])
    np.testing.assert_allclose(B[0], vectors["P01_TC_02"])
    np.testing.assert_allclose(A[1], vectors["P02_IM_01"])
    np.testing.assert_allclose(B[1], vectors["P02_IM_02"])
    np.testing.assert_allclose(mass_gap, np.zeros(2, dtype=float))

    with pytest.raises(ValueError, match="mass_mode='count'"):
        assemble_tensors(adata, pair_meta, k_full=K_FULL, mass_mode="density")


def test_run_arm1_supplies_fixed_lambda_and_tau_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    from slotar.uot import UOTSolveConfig, precompute_logKernels
    from tasks.task_A.arm1_noise_baseline import ARM_NAME, FIXED_MODE, run_arm1
    from tests.helpers_task_a_fixture import K_FULL, build_task_a_fixture

    adata = build_task_a_fixture()
    cfg = {
        "data": {"k_full": K_FULL, "mass_mode": "count"},
        "arm1": {
            "n_draws": 2,
            "random_seed": 7,
            "fixed_lambda_by_compartment": {"TC": 0.5, "IM": 1.5, "PT": 2.5},
            "fixed_tau_by_compartment": {"TC": 0.2, "IM": 0.4, "PT": 0.6},
        },
    }
    uot_cfg = UOTSolveConfig(eps_schedule=[1.0], max_iter=100, tol=1e-8, n_min_proto=0.0)
    kernels = precompute_logKernels(np.asarray(adata.uns["cost_matrix"], dtype=float), uot_cfg.eps_schedule)
    captured: dict[str, np.ndarray] = {}

    def fake_batched_uot_solve(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels: list[np.ndarray],
        cfg: UOTSolveConfig,
        tau_external: np.ndarray | None = None,
        external_support_mask: np.ndarray | None = None,
        return_plan: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        del kernels, cfg
        assert external_support_mask is None
        assert return_plan is False
        captured["A"] = A.copy()
        captured["B"] = B.copy()
        captured["lambda_pl"] = lambda_pl.copy()
        assert tau_external is not None
        captured["tau_external"] = tau_external.copy()
        n_rows = A.shape[0]
        metrics = {
            "T": np.full(n_rows, 1.0, dtype=float),
            "D_pos": np.zeros(n_rows, dtype=float),
            "B_pos": np.zeros(n_rows, dtype=float),
            "d_rel": np.zeros(n_rows, dtype=float),
            "b_rel": np.zeros(n_rows, dtype=float),
            "M": np.full(n_rows, 0.25, dtype=float),
            "R": np.full(n_rows, 0.75, dtype=float),
            "tau": tau_external.copy(),
        }
        details = {
            "source_transport_marginal": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "target_transport_marginal": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "T_k": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "D_k": np.zeros((n_rows, A.shape[1]), dtype=float),
            "B_k": np.zeros((n_rows, A.shape[1]), dtype=float),
        }
        return metrics, details, np.full(n_rows, "ok", dtype=object)

    monkeypatch.setattr("tasks.task_A.common.batched_uot_solve", fake_batched_uot_solve)

    df_result = run_arm1(adata, cfg, uot_cfg, kernels)

    expected_lambda = np.tile(np.array([1.5, 2.5, 0.5, 1.5, 2.5, 0.5], dtype=float), 2)
    expected_tau = np.tile(np.array([0.4, 0.6, 0.2, 0.4, 0.6, 0.2], dtype=float), 2)
    assert captured["A"].shape == (12, K_FULL)
    assert captured["B"].shape == (12, K_FULL)
    assert captured["lambda_pl"].shape == (12,)
    assert captured["tau_external"].shape == (12,)
    np.testing.assert_allclose(captured["lambda_pl"], expected_lambda)
    np.testing.assert_allclose(captured["tau_external"], expected_tau)

    assert (df_result["arm"] == ARM_NAME).all()
    assert (df_result["lambda_mode"] == FIXED_MODE).all()
    assert (df_result["tau_mode"] == FIXED_MODE).all()
    assert (df_result["patient_id"] == df_result["patient_id_a"]).all()
    assert (df_result["compartment"] == df_result["compartment_a"]).all()
    assert df_result["same_patient"].all()
    assert df_result["same_compartment"].all()
    assert df_result["patient_group_id"].is_unique
    assert df_result["pair_id"].str.contains("draw_0001|draw_0002", regex=True).all()
    assert (df_result["roi_a"] != df_result["roi_b"]).all()
    np.testing.assert_allclose(df_result["lambda_pl"].to_numpy(dtype=float), expected_lambda)
    np.testing.assert_allclose(df_result["tau"].to_numpy(dtype=float), expected_tau)


def test_run_broken_reference_anchors_lambda_and_tau_to_compartment_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slotar.uot import UOTSolveConfig, precompute_logKernels
    from tasks.task_A.arm1_broken_reference import ARM_NAME, run_arm1
    from tasks.task_A.arm1_noise_baseline import FIXED_MODE
    from tests.helpers_task_a_fixture import K_FULL, build_task_a_fixture

    adata = build_task_a_fixture()
    cfg = {
        "data": {"k_full": K_FULL, "mass_mode": "count"},
        "arm1": {
            "n_draws": 2,
            "random_seed": 7,
            "fixed_lambda_by_compartment": {"TC": 0.5, "IM": 1.5, "PT": 2.5},
            "fixed_tau_by_compartment": {"TC": 0.2, "IM": 0.4, "PT": 0.6},
        },
    }
    uot_cfg = UOTSolveConfig(eps_schedule=[1.0], max_iter=100, tol=1e-8, n_min_proto=0.0)
    kernels = precompute_logKernels(np.asarray(adata.uns["cost_matrix"], dtype=float), uot_cfg.eps_schedule)
    captured: dict[str, np.ndarray] = {}

    def fake_batched_uot_solve(
        *,
        A: np.ndarray,
        B: np.ndarray,
        lambda_pl: np.ndarray,
        kernels: list[np.ndarray],
        cfg: UOTSolveConfig,
        tau_external: np.ndarray | None = None,
        external_support_mask: np.ndarray | None = None,
        return_plan: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        del kernels, cfg
        assert external_support_mask is None
        assert return_plan is False
        captured["A"] = A.copy()
        captured["B"] = B.copy()
        captured["lambda_pl"] = lambda_pl.copy()
        assert tau_external is not None
        captured["tau_external"] = tau_external.copy()
        n_rows = A.shape[0]
        metrics = {
            "T": np.full(n_rows, 1.0, dtype=float),
            "D_pos": np.zeros(n_rows, dtype=float),
            "B_pos": np.zeros(n_rows, dtype=float),
            "d_rel": np.zeros(n_rows, dtype=float),
            "b_rel": np.zeros(n_rows, dtype=float),
            "M": np.full(n_rows, 0.25, dtype=float),
            "R": np.full(n_rows, 0.75, dtype=float),
            "tau": tau_external.copy(),
        }
        details = {
            "source_transport_marginal": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "target_transport_marginal": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "T_k": np.full((n_rows, A.shape[1]), 0.5, dtype=float),
            "D_k": np.zeros((n_rows, A.shape[1]), dtype=float),
            "B_k": np.zeros((n_rows, A.shape[1]), dtype=float),
        }
        return metrics, details, np.full(n_rows, "ok", dtype=object)

    monkeypatch.setattr("tasks.task_A.common.batched_uot_solve", fake_batched_uot_solve)

    df_result = run_arm1(adata, cfg, uot_cfg, kernels)

    expected_lambda = np.tile(np.array([1.5, 2.5, 0.5, 1.5, 2.5, 0.5], dtype=float), 2)
    expected_tau = np.tile(np.array([0.4, 0.6, 0.2, 0.4, 0.6, 0.2], dtype=float), 2)
    assert captured["A"].shape == (12, K_FULL)
    assert captured["B"].shape == (12, K_FULL)
    np.testing.assert_allclose(captured["lambda_pl"], expected_lambda)
    np.testing.assert_allclose(captured["tau_external"], expected_tau)

    assert (df_result["arm"] == ARM_NAME).all()
    assert (df_result["lambda_mode"] == FIXED_MODE).all()
    assert (df_result["tau_mode"] == FIXED_MODE).all()
    assert (~df_result["same_patient"]).all()
    assert (~df_result["same_compartment"]).all()
    assert (df_result["patient_id"] == df_result["patient_id_a"]).all()
    assert (df_result["compartment"] == df_result["compartment_a"]).all()
    assert (df_result["patient_id_a"] != df_result["patient_id_b"]).all()
    assert (df_result["compartment_a"] != df_result["compartment_b"]).all()
    np.testing.assert_allclose(df_result["lambda_pl"].to_numpy(dtype=float), expected_lambda)
    np.testing.assert_allclose(df_result["tau"].to_numpy(dtype=float), expected_tau)
