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
from slotar.exceptions import ERR_UOT_EMPTY_SUPPORT
from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels
from slotar.utils import compute_active_mask


def _run_singletons(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: list[np.ndarray],
    cfg: UOTSolveConfig,
    tau_external: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    metric_names = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")
    metrics = {name: np.full(A.shape[0], np.nan, dtype=float) for name in metric_names}
    status = np.empty(A.shape[0], dtype=object)

    for idx in range(A.shape[0]):
        tau_arg = None if tau_external is None else tau_external[idx : idx + 1]
        row_metrics, row_status = batched_uot_solve(
            A=A[idx : idx + 1],
            B=B[idx : idx + 1],
            lambda_pl=lambda_pl[idx : idx + 1],
            kernels=kernels,
            cfg=cfg,
            tau_external=tau_arg,
        )
        status[idx] = row_status[0]
        for name in metric_names:
            metrics[name][idx] = row_metrics[name][0]

    return metrics, status


def test_batched_uot_solve_lambda_affects_outputs() -> None:
    A = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    B = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    cost = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=float)

    cfg = UOTSolveConfig(eps_schedule=[1.0, 0.2], max_iter=5000)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    batched_metrics, batched_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=np.array([0.05, 10.0], dtype=float),
        kernels=kernels,
        cfg=cfg,
    )
    singleton_metrics, singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=np.array([0.05, 10.0], dtype=float),
        kernels=kernels,
        cfg=cfg,
    )

    np.testing.assert_array_equal(batched_status, singleton_status)
    np.testing.assert_allclose(batched_metrics["T"], singleton_metrics["T"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(batched_metrics["B_pos"], singleton_metrics["B_pos"], rtol=1e-6, atol=1e-8)
    assert batched_status[0] == "ok"
    assert batched_status[1] == "ok"
    assert not np.isclose(batched_metrics["T"][0], batched_metrics["T"][1])
    assert not np.isclose(batched_metrics["B_pos"][0], batched_metrics["B_pos"][1])


def test_batched_uot_solve_omitted_tau_external_returns_nan_tau_and_r() -> None:
    A = np.array([[0.7, 0.3]], dtype=float)
    B = np.array([[0.3, 0.7]], dtype=float)
    cost = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)

    cfg = UOTSolveConfig(eps_schedule=[1.0, 0.2], max_iter=5000)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    metrics, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=np.array([1.0], dtype=float),
        kernels=kernels,
        cfg=cfg,
    )

    assert status[0] == "ok"
    assert np.isnan(metrics["tau"][0])
    assert np.isnan(metrics["R"][0])
    assert np.isfinite(metrics["T"][0])
    assert np.isfinite(metrics["M"][0])


def test_batched_uot_solve_uses_external_tau_for_retention() -> None:
    A = np.array([[1.0, 0.0]], dtype=float)
    B = np.array([[0.0, 1.0]], dtype=float)
    cost = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=float)
    cfg = UOTSolveConfig(eps_schedule=[1.0, 0.2], max_iter=5000)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)
    lambda_pl = np.array([10.0], dtype=float)

    low_tau_metrics, low_tau_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=np.array([0.0], dtype=float),
    )
    high_tau_metrics, high_tau_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=np.array([10.0], dtype=float),
    )

    assert low_tau_status[0] == "ok"
    assert high_tau_status[0] == "ok"
    assert low_tau_metrics["tau"][0] == 0.0
    assert high_tau_metrics["tau"][0] == 10.0
    assert np.isclose(low_tau_metrics["T"][0], high_tau_metrics["T"][0])
    assert np.isclose(low_tau_metrics["M"][0], high_tau_metrics["M"][0])
    assert low_tau_metrics["R"][0] <= high_tau_metrics["R"][0]
    assert not np.isclose(low_tau_metrics["R"][0], high_tau_metrics["R"][0])


def test_batched_uot_solve_rejects_bad_tau_external_shape() -> None:
    A = np.array([[1.0, 0.0]], dtype=float)
    B = np.array([[0.0, 1.0]], dtype=float)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    cfg = UOTSolveConfig(eps_schedule=[1.0], max_iter=5000)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    with pytest.raises(DataContractError, match="tau_external must have shape"):
        batched_uot_solve(
            A=A,
            B=B,
            lambda_pl=np.array([1.0], dtype=float),
            kernels=kernels,
            cfg=cfg,
            tau_external=np.array([0.1, 0.2], dtype=float),
        )


def test_batched_uot_solve_eps_schedule_affects_outputs() -> None:
    A = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    B = np.array([[0.3, 0.7], [0.8, 0.2]], dtype=float)
    cost = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
    lambda_pl = np.array([1.0, 2.0], dtype=float)

    coarse_cfg = UOTSolveConfig(eps_schedule=[5.0], max_iter=5000)
    fine_cfg = UOTSolveConfig(eps_schedule=[0.05], max_iter=5000)
    coarse_kernels = precompute_logKernels(cost, coarse_cfg.eps_schedule)
    fine_kernels = precompute_logKernels(cost, fine_cfg.eps_schedule)

    coarse_metrics, coarse_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=coarse_kernels,
        cfg=coarse_cfg,
    )
    fine_metrics, fine_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=fine_kernels,
        cfg=fine_cfg,
    )
    coarse_singleton_metrics, coarse_singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=coarse_kernels,
        cfg=coarse_cfg,
    )
    fine_singleton_metrics, fine_singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=fine_kernels,
        cfg=fine_cfg,
    )

    np.testing.assert_array_equal(coarse_status, coarse_singleton_status)
    np.testing.assert_array_equal(fine_status, fine_singleton_status)
    np.testing.assert_allclose(coarse_metrics["M"], coarse_singleton_metrics["M"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fine_metrics["M"], fine_singleton_metrics["M"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(coarse_metrics["T"], coarse_singleton_metrics["T"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fine_metrics["T"], fine_singleton_metrics["T"], rtol=1e-6, atol=1e-8)
    assert np.all(coarse_status == "ok")
    assert np.all(fine_status == "ok")
    assert not np.allclose(coarse_metrics["M"], fine_metrics["M"])
    assert not np.allclose(coarse_metrics["T"], fine_metrics["T"])


def test_compute_active_mask_threshold_is_inclusive_and_matches_solver_screening() -> None:
    A = np.array(
        [
            [0.10, 0.15, 0.00],
            [0.124, 0.124, 0.00],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.15, 0.10, 0.00],
            [0.124, 0.124, 0.00],
        ],
        dtype=float,
    )
    cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    cfg = UOTSolveConfig(eps_schedule=[1.0], n_min_proto=0.25, max_iter=5000)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    active_mask, _ = compute_active_mask(A[0], B[0], cfg.n_min_proto)
    assert active_mask.tolist() == [True, True, False]

    metrics, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=np.ones(2, dtype=float),
        kernels=kernels,
        cfg=cfg,
    )

    assert status[0] == "ok"
    assert status[1] == ERR_UOT_EMPTY_SUPPORT
    assert np.isfinite(metrics["T"][0])
    assert np.isnan(metrics["T"][1])


def test_batched_uot_solve_active_mask_pruning_is_exercised() -> None:
    A = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.12, 0.12, 0.00],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.10, 0.80, 0.10],
            [0.12, 0.12, 0.00],
        ],
        dtype=float,
    )
    cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    loose_cfg = UOTSolveConfig(eps_schedule=[1.0], n_min_proto=0.0, max_iter=5000)
    pruned_cfg = UOTSolveConfig(eps_schedule=[1.0], n_min_proto=0.25, max_iter=5000)
    loose_kernels = precompute_logKernels(cost, loose_cfg.eps_schedule)
    pruned_kernels = precompute_logKernels(cost, pruned_cfg.eps_schedule)
    lambda_pl = np.ones(2, dtype=float)

    loose_metrics, loose_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=loose_kernels,
        cfg=loose_cfg,
    )
    pruned_metrics, pruned_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=pruned_kernels,
        cfg=pruned_cfg,
    )
    pruned_singleton_metrics, pruned_singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=pruned_kernels,
        cfg=pruned_cfg,
    )

    active_mask, mass_pruned_ratio = compute_active_mask(A[0], B[0], pruned_cfg.n_min_proto)
    assert active_mask.tolist() == [True, True, False]
    assert np.isclose(mass_pruned_ratio, 0.1)

    assert loose_status.shape == (2,)
    assert pruned_status.shape == (2,)
    assert loose_status[0] == "ok"
    assert pruned_status[0] == "ok"
    assert pruned_status[1] == ERR_UOT_EMPTY_SUPPORT
    np.testing.assert_array_equal(pruned_status, pruned_singleton_status)
    np.testing.assert_allclose(
        pruned_metrics["T"][:1],
        pruned_singleton_metrics["T"][:1],
        rtol=1e-6,
        atol=1e-8,
    )
    assert not np.isclose(loose_metrics["T"][0], pruned_metrics["T"][0])
