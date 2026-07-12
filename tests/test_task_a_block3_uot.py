from __future__ import annotations

import numpy as np
import pytest

from tasks.task_A.block3.baselines import (
    RuntimeSettings,
    estimate_uot_matched_mass,
    solve_uot_plan,
)
from tasks.task_A.block3.task_uot import STATUS_EMPTY_SOURCE, STATUS_OK, solve_uot_batch


def test_task_local_uot_returns_finite_native_plan() -> None:
    source = np.array([0.7, 0.3], dtype=float)
    target = np.array([0.2, 0.8], dtype=float)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    result = solve_uot_plan(
        x=source,
        y=target,
        cost_matrix=cost,
        match_penalty=1.0,
        runtime_settings=RuntimeSettings(uot_backend="numpy", device=None),
    )

    assert result.status == STATUS_OK
    assert result.P is not None
    assert result.P.shape == (2, 2)
    assert np.isfinite(result.P).all()
    assert np.all(result.P >= 0.0)
    np.testing.assert_allclose(
        result.P,
        np.array(
            [
                [0.3483298268424, 0.1478942981534],
                [0.00004924671488483, 0.4605564311035],
            ]
        ),
        rtol=1e-7,
        atol=1e-10,
    )
    assert result.metadata["solver"] == "tasks.task_A.block3.task_uot.solve_uot_batch"
    assert result.metadata["solver_backend"] == "numpy_log_domain"
    assert result.metadata["requested_device"] is None


def test_task_local_uot_calibration_mass_is_finite() -> None:
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    train_pairs = (
        (np.array([0.7, 0.3]), np.array([0.2, 0.8])),
        (np.array([0.4, 0.6]), np.array([0.5, 0.5])),
    )

    matched_mass = estimate_uot_matched_mass(
        train_pairs=train_pairs,
        cost_matrix=cost,
        match_penalty=1.0,
    )

    assert np.isfinite(matched_mass)
    assert matched_mass > 0.0


def test_task_local_uot_reports_empty_source_without_fallback() -> None:
    result = solve_uot_batch(
        source=np.array([[0.0, 0.0]], dtype=float),
        target=np.array([[0.5, 0.5]], dtype=float),
        cost_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        match_penalties=np.array([1.0], dtype=float),
        backend="numpy",
    )

    assert result.status.tolist() == [STATUS_EMPTY_SOURCE]
    assert np.isnan(result.plans).all()


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA is required for the formal UOT backend",
)
def test_torch_uot_matches_numpy_reference() -> None:
    source = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    target = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=float)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    penalties = np.array([1.0, 2.0], dtype=float)

    numpy_result = solve_uot_batch(
        source=source,
        target=target,
        cost_matrix=cost,
        match_penalties=penalties,
        backend="numpy",
    )
    torch_result = solve_uot_batch(
        source=source,
        target=target,
        cost_matrix=cost,
        match_penalties=penalties,
        backend="torch",
        device="cuda:0",
    )

    assert numpy_result.status.tolist() == [STATUS_OK, STATUS_OK]
    assert torch_result.status.tolist() == [STATUS_OK, STATUS_OK]
    np.testing.assert_allclose(torch_result.plans, numpy_result.plans, atol=1e-7, rtol=0.0)
