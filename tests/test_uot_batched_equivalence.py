from __future__ import annotations

import numpy as np

from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels

CORE_METRICS = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M")
ALL_METRICS = (*CORE_METRICS, "R", "tau")


def _run_singletons(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: list[np.ndarray],
    cfg: UOTSolveConfig,
    tau_external: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    metrics = {name: np.full(A.shape[0], np.nan, dtype=float) for name in ALL_METRICS}
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
        for name in ALL_METRICS:
            metrics[name][idx] = row_metrics[name][0]

    return metrics, status


def _assert_metric_match(
    batched_metrics: dict[str, np.ndarray],
    singleton_metrics: dict[str, np.ndarray],
    *,
    names: tuple[str, ...],
) -> None:
    for name in names:
        np.testing.assert_allclose(
            batched_metrics[name],
            singleton_metrics[name],
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )


def test_batched_uot_matches_singletons_with_external_tau() -> None:
    A = np.array(
        [
            [0.70, 0.20, 0.10, 0.00],
            [0.40, 0.00, 0.60, 0.00],
            [0.00, 0.80, 0.10, 0.10],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.30, 0.50, 0.20, 0.00],
            [0.20, 0.30, 0.50, 0.00],
            [0.10, 0.00, 0.70, 0.20],
        ],
        dtype=float,
    )
    lambda_pl = np.array([0.3, 1.0, 4.0], dtype=float)
    tau_external = np.array([0.4, 0.9, 1.4], dtype=float)
    cost = np.array(
        [
            [0.0, 0.6, 1.4, 2.0],
            [0.6, 0.0, 1.1, 1.8],
            [1.4, 1.1, 0.0, 0.7],
            [2.0, 1.8, 0.7, 0.0],
        ],
        dtype=float,
    )

    cfg = UOTSolveConfig(eps_schedule=[1.5, 0.4, 0.1], max_iter=4000, tol=1e-8)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    batched_metrics, batched_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
    )
    singleton_metrics, singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
    )

    np.testing.assert_array_equal(batched_status, singleton_status)
    _assert_metric_match(batched_metrics, singleton_metrics, names=ALL_METRICS)


def test_batched_uot_matches_singletons_without_tau_external() -> None:
    A = np.array(
        [
            [0.55, 0.45, 0.00],
            [0.10, 0.30, 0.60],
            [0.80, 0.10, 0.10],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.35, 0.65, 0.00],
            [0.20, 0.40, 0.40],
            [0.10, 0.70, 0.20],
        ],
        dtype=float,
    )
    lambda_pl = np.array([0.5, 1.5, 3.0], dtype=float)
    cost = np.array(
        [
            [0.0, 1.0, 1.8],
            [1.0, 0.0, 0.9],
            [1.8, 0.9, 0.0],
        ],
        dtype=float,
    )

    cfg = UOTSolveConfig(eps_schedule=[1.0, 0.2], max_iter=4000, tol=1e-8)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    batched_metrics, batched_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
    )
    singleton_metrics, singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
    )

    np.testing.assert_array_equal(batched_status, singleton_status)
    _assert_metric_match(batched_metrics, singleton_metrics, names=CORE_METRICS)
    assert np.isnan(batched_metrics["tau"]).all()
    assert np.isnan(batched_metrics["R"]).all()
    assert np.isnan(singleton_metrics["tau"]).all()
    assert np.isnan(singleton_metrics["R"]).all()


def test_batched_uot_matches_singletons_with_distinct_active_supports() -> None:
    A = np.array(
        [
            [0.50, 0.30, 0.20, 0.00],
            [0.00, 0.15, 0.55, 0.30],
            [0.45, 0.00, 0.10, 0.45],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.20, 0.60, 0.20, 0.00],
            [0.00, 0.35, 0.25, 0.40],
            [0.55, 0.00, 0.05, 0.40],
        ],
        dtype=float,
    )
    lambda_pl = np.array([0.8, 1.1, 2.5], dtype=float)
    tau_external = np.array([0.5, 0.7, 1.2], dtype=float)
    cost = np.array(
        [
            [0.0, 0.9, 1.5, 2.2],
            [0.9, 0.0, 0.7, 1.6],
            [1.5, 0.7, 0.0, 0.8],
            [2.2, 1.6, 0.8, 0.0],
        ],
        dtype=float,
    )

    cfg = UOTSolveConfig(eps_schedule=[1.2, 0.3], n_min_proto=0.25, max_iter=4000, tol=1e-8)
    kernels = precompute_logKernels(cost, cfg.eps_schedule)

    batched_metrics, batched_status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
    )
    singleton_metrics, singleton_status = _run_singletons(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=tau_external,
    )

    np.testing.assert_array_equal(batched_status, singleton_status)
    _assert_metric_match(batched_metrics, singleton_metrics, names=ALL_METRICS)
