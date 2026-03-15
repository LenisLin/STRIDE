from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slotar.contracts import DataContractError
from slotar.exceptions import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)
from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels


def test_batched_uot_solve_raises_on_shape_mismatch() -> None:
    A = np.ones((2, 3), dtype=float)
    B = np.ones((2, 2), dtype=float)
    lam_pl = np.ones(2, dtype=float)
    kernels = [np.zeros((3, 3), dtype=float)]
    cfg = UOTSolveConfig(eps_schedule=[1.0])

    with pytest.raises(DataContractError, match="shape mismatch"):
        batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)


def test_batched_uot_solve_raises_on_negative_mass() -> None:
    A = np.array([[1.0, -0.1]], dtype=float)
    B = np.array([[1.0, 1.0]], dtype=float)
    lam_pl = np.array([1.0], dtype=float)
    kernels = [np.zeros((2, 2), dtype=float)]
    cfg = UOTSolveConfig(eps_schedule=[1.0])

    with pytest.raises(DataContractError, match="non-negative"):
        batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)


def test_batched_uot_solve_raises_on_nan_input() -> None:
    A = np.array([[1.0, np.nan]], dtype=float)
    B = np.array([[1.0, 1.0]], dtype=float)
    lam_pl = np.array([1.0], dtype=float)
    kernels = [np.zeros((2, 2), dtype=float)]
    cfg = UOTSolveConfig(eps_schedule=[1.0])

    with pytest.raises(DataContractError, match="NaN/Inf"):
        batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)


def test_batched_uot_solve_rejects_zero_lambda() -> None:
    A = np.ones((1, 2), dtype=float)
    B = np.ones((1, 2), dtype=float)
    lam_pl = np.array([0.0], dtype=float)
    kernels = [np.zeros((2, 2), dtype=float)]
    cfg = UOTSolveConfig(eps_schedule=[1.0])

    with pytest.raises(DataContractError, match="strictly positive"):
        batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)


def test_batched_uot_solve_rejects_kernel_schedule_length_mismatch() -> None:
    A = np.ones((1, 2), dtype=float)
    B = np.ones((1, 2), dtype=float)
    lam_pl = np.array([1.0], dtype=float)
    cfg = UOTSolveConfig(eps_schedule=[1.0, 0.5])
    kernels = [np.zeros((2, 2), dtype=float)]

    with pytest.raises(DataContractError, match="kernels length must match"):
        batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)


def test_batched_uot_solve_batch_isolation_for_item_degeneracies() -> None:
    N, K = 5, 2
    A = np.array(
        [
            [2.0, 1.0],      # ok
            [0.0, 0.0],      # empty source
            [1.0, 1.0],      # empty target
            [1e-8, 1e-8],    # empty support after pruning
            [1e308, 1e308],  # numerical failure (overflow in row reductions)
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.0, 2.0],      # ok
            [1.0, 1.0],      # empty source
            [0.0, 0.0],      # empty target
            [1e-8, 1e-8],    # empty support after pruning
            [1e308, 1e308],  # numerical failure
        ],
        dtype=float,
    )
    lam_pl = np.ones(N, dtype=float)
    cfg = UOTSolveConfig(eps_schedule=[1.0], n_min_proto=1e-5)
    kernels = precompute_logKernels(np.zeros((K, K), dtype=float), cfg.eps_schedule)

    metrics, status = batched_uot_solve(A=A, B=B, lambda_pl=lam_pl, kernels=kernels, cfg=cfg)
    single_metrics, single_status = batched_uot_solve(
        A=A[[0]],
        B=B[[0]],
        lambda_pl=lam_pl[[0]],
        kernels=kernels,
        cfg=cfg,
    )

    assert status.shape == (N,)
    assert status[0] == "ok"
    assert status[1] == ERR_UOT_EMPTY_MASS_SOURCE
    assert status[2] == ERR_UOT_EMPTY_MASS_TARGET
    assert status[3] == ERR_UOT_EMPTY_SUPPORT
    assert status[4] == ERR_UOT_NUMERICAL

    expected_core_metrics = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M")
    for name in expected_core_metrics:
        assert metrics[name].shape == (N,)
        assert np.isfinite(metrics[name][0])
        assert np.isclose(metrics[name][0], single_metrics[name][0])
        assert np.isnan(metrics[name][1])
        assert np.isnan(metrics[name][2])
        assert np.isnan(metrics[name][3])
        assert np.isnan(metrics[name][4])

    assert single_status[0] == "ok"
    assert np.isnan(metrics["R"][0])
    assert np.isnan(metrics["tau"][0])
    assert np.isnan(metrics["R"][1])
    assert np.isnan(metrics["tau"][1])
    assert np.isnan(metrics["R"][2])
    assert np.isnan(metrics["tau"][2])
    assert np.isnan(metrics["R"][3])
    assert np.isnan(metrics["tau"][3])
    assert np.isnan(metrics["R"][4])
    assert np.isnan(metrics["tau"][4])
