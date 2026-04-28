from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.task_A.block3.baselines import (
    calibrate_uot_lambda,
    diagonal_transport_plan,
    partial_ot_plan,
    solve_uot_plan,
)
from tasks.task_A.block3.execution import (
    Block3MethodOutput,
    Block3PatientTruth,
    _build_open_metric_rows,
    _method_native_records,
    _run_uot_baseline,
    _run_diagonal_transport_baseline,
    _run_partial_ot_baseline,
)
from tasks.task_A.block3.contracts import MetricStatus


def _truth(patient_id: str, x: np.ndarray, y: np.ndarray) -> Block3PatientTruth:
    return Block3PatientTruth(
        rerun_id="rerun_01",
        patient_id=patient_id,
        x=x,
        y=y,
        A=np.zeros((x.size, y.size), dtype=float),
        d=np.zeros(x.size, dtype=float),
        e=np.zeros(y.size, dtype=float),
        open_mass=0.0,
    )


def test_diagonal_transport_plan_emits_exact_native_plan() -> None:
    plan = diagonal_transport_plan(np.array([2.0, 1.0, 0.5]), np.array([1.0, 3.0, 0.25]))

    np.testing.assert_allclose(
        plan,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.25],
            ]
        ),
    )


def test_partial_ot_plan_uses_cost_order_and_clipping_metadata() -> None:
    result = partial_ot_plan(
        x=np.array([2.0, 1.0]),
        y=np.array([1.0, 2.0]),
        cost_matrix=np.array([[3.0, 1.0], [2.0, 4.0]]),
        matched_mass_budget=5.0,
    )

    np.testing.assert_allclose(result.P, np.array([[0.0, 2.0], [1.0, 0.0]]))
    assert result.metadata["requested_budget"] == 5.0
    assert result.metadata["effective_budget"] == 3.0
    assert result.metadata["clipped"] is True


def test_uot_calibration_tie_breaks_to_smaller_lambda_and_reports_boundary() -> None:
    train_pairs = (
        (np.array([1.0, 0.0]), np.array([0.5, 0.5])),
        (np.array([0.2, 0.8]), np.array([0.2, 0.1])),
    )
    achieved = {0.05: 0.4, 0.1: 0.4, 0.5: 0.9}

    result = calibrate_uot_lambda(
        train_pairs=train_pairs,
        lambda_grid=(0.05, 0.1, 0.5),
        achieved_mass_fn=lambda lam, _pairs: achieved[lam],
    )

    assert result.selected_lambda == 0.05
    assert result.target_overlap == 0.4
    assert result.boundary_hit is True


def test_solve_uot_plan_emits_matching_plan_and_metadata() -> None:
    result = solve_uot_plan(
        x=np.array([0.6, 0.4]),
        y=np.array([0.5, 0.5]),
        cost_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        match_penalty=1.0,
    )

    assert result.status == "ok"
    assert result.P.shape == (2, 2)
    assert np.isfinite(result.P).all()
    assert np.sum(result.P) > 0.0
    assert result.metadata["lambda"] == 1.0


def test_uot_baseline_stores_calibration_diagnostic_metadata() -> None:
    truths = [_truth("P01", np.array([0.6, 0.4]), np.array([0.5, 0.5]))]
    outputs = _run_uot_baseline(
        truths=truths,
        cost_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        match_penalty=1.0,
        calibration_metadata={
            "selected_lambda": 1.0,
            "target_overlap": 0.9,
            "boundary_hit": False,
            "achieved_by_lambda": {0.5: 0.8, 1.0: 0.9},
            "absolute_error_by_lambda": {0.5: 0.1, 1.0: 0.0},
        },
    )

    records = _method_native_records(
        subexperiment_id="3B-2",
        condition_id="open_mass_scale_grid",
        rerun_id="rerun_01",
        method_name="uot_baseline",
        outputs=outputs,
        open_mass_scale=0.0,
    )
    metadata = json.loads(str(records[0]["metadata_json"]))

    assert records[0]["open_mass_scale"] == 0.0
    assert metadata["lambda"] == 1.0
    assert metadata["solver_status"] == "ok"
    assert metadata["matched_mass"] > 0.0
    assert metadata["selected_lambda"] == 1.0
    assert metadata["target_overlap"] == 0.9
    assert metadata["boundary_hit"] is False
    assert metadata["achieved_by_lambda"]["0.5"] == 0.8
    assert metadata["absolute_error_by_lambda"]["1.0"] == 0.0


def test_plan_baselines_store_P_json_and_metadata_json() -> None:
    truths = [_truth("P01", np.array([2.0, 1.0]), np.array([1.0, 2.0]))]
    outputs = _run_partial_ot_baseline(
        truths=truths,
        cost_matrix=np.array([[3.0, 1.0], [2.0, 4.0]]),
        matched_mass_budget=5.0,
    )

    records = _method_native_records(
        subexperiment_id="3B-2",
        condition_id="open_mass_scale_grid",
        rerun_id="rerun_01",
        method_name="partial_ot_baseline",
        outputs=outputs,
        open_mass_scale=0.0,
    )

    assert records[0]["open_mass_scale"] == 0.0
    assert records[0]["P_json"].startswith("[")
    metadata = json.loads(str(records[0]["metadata_json"]))
    assert metadata["effective_budget"] == 3.0
    assert metadata["clipped"] is True


def test_diagonal_baseline_derives_A_d_e_from_plan() -> None:
    truths = [_truth("P01", np.array([2.0, 1.0]), np.array([1.0, 2.0]))]
    output = _run_diagonal_transport_baseline(truths=truths)["P01"]

    np.testing.assert_allclose(output.P, np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(output.A, np.array([[0.5, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(output.d, np.array([0.5, 0.0]))
    np.testing.assert_allclose(output.e, np.array([0.0, 1.0]))


def test_open_mass_zero_marks_only_support_f1_not_applicable() -> None:
    truth = _truth("P01", np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    output = Block3MethodOutput(
        patient_id="P01",
        fit_status="ok",
        A=np.eye(2),
        d=np.zeros(2),
        e=np.zeros(2),
        mu_minus=np.array([1.0, 1.0]),
        mu_plus=np.array([1.0, 1.0]),
        P=np.eye(2),
    )

    rows = _build_open_metric_rows(
        rerun_id="rerun_01",
        subexperiment_id="3B-2",
        condition_id="open_mass_scale_grid",
        evaluation_family="open_recovery",
        method_name="diagonal_transport_baseline",
        truth=truth,
        output=output,
    )
    statuses = {row.metric_value.metric_name.value: row.metric_value.status for row in rows}

    assert statuses["open_support_F1"] is MetricStatus.NOT_APPLICABLE
    assert statuses["d_MAE"] is MetricStatus.REPORTED
    assert statuses["e_MAE"] is MetricStatus.REPORTED
    assert statuses["d_MSE"] is MetricStatus.REPORTED
    assert statuses["e_MSE"] is MetricStatus.REPORTED
