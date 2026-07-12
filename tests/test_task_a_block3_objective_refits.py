from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tasks.task_A.block3.execution as execution
from stride.tl._objective import (
    NO_CONSISTENCY_OBJECTIVE_POLICY,
    NO_GEOMETRY_OBJECTIVE_POLICY,
    NO_RECURRENCE_OBJECTIVE_POLICY,
)


@pytest.mark.parametrize(
    ("ablation_mode", "expected_policy"),
    [
        ("consistency", NO_CONSISTENCY_OBJECTIVE_POLICY),
        ("geometry", NO_GEOMETRY_OBJECTIVE_POLICY),
        ("recurrence", NO_RECURRENCE_OBJECTIVE_POLICY),
    ],
)
def test_block3_ablation_runs_private_independent_refit(
    monkeypatch: pytest.MonkeyPatch,
    ablation_mode: str,
    expected_policy: object,
) -> None:
    calls: list[object] = []
    fit_result = object()
    relation = SimpleNamespace(
        patient_ids=("p1",),
        A=np.full((1, 2, 2), 0.25),
        d=np.full((1, 2), 0.5),
        e=np.full((1, 2), 0.1),
    )
    truth = SimpleNamespace(
        patient_id="p1",
        x=np.array([0.6, 0.4]),
        y=np.array([0.5, 0.5]),
    )

    monkeypatch.setattr(execution, "_make_adata_for_truths", lambda *a, **k: object())
    monkeypatch.setattr(execution, "extract_task_a_relations", lambda result: (relation,))
    monkeypatch.setattr(
        execution,
        "_compact_stride_fit_metadata",
        lambda **kwargs: {"fit_surface": "stride.tl.fit"},
    )

    def fake_private_fit(adata, *, device, objective_policy):
        calls.append(objective_policy)
        return fit_result

    monkeypatch.setattr("stride.tl._run._fit_with_objective_policy", fake_private_fit)

    def fake_fit_task_a_pair(adata, *, device, estimator):
        return estimator(adata, device=device)

    monkeypatch.setattr(execution, "fit_task_a_pair", fake_fit_task_a_pair)

    outputs = execution._run_stride_method(
        cohort_inputs=SimpleNamespace(),
        truths=[truth],
        runtime=execution.Block3RuntimeControls(device="cuda:0"),
        ablation_mode=ablation_mode,
    )

    assert calls == [expected_policy]
    assert outputs["p1"].metadata["zeroed_objective_term"] == ablation_mode
    assert outputs["p1"].metadata["fixed_denominator_policy"] is True


def test_block3_reference_and_ablation_do_not_share_fit_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"reference": 0, "ablation": 0}
    truth = SimpleNamespace(
        patient_id="p1",
        x=np.array([0.6, 0.4]),
        y=np.array([0.5, 0.5]),
    )

    monkeypatch.setattr(execution, "_make_adata_for_truths", lambda *a, **k: object())
    monkeypatch.setattr(
        execution,
        "_compact_stride_fit_metadata",
        lambda **kwargs: {"fit_surface": "stride.tl.fit"},
    )

    def relation(value: float) -> SimpleNamespace:
        return SimpleNamespace(
            patient_ids=("p1",),
            A=np.full((1, 2, 2), value),
            d=np.full((1, 2), value),
            e=np.full((1, 2), value),
        )

    reference_result = object()
    ablation_result = object()
    monkeypatch.setattr(
        execution,
        "extract_task_a_relations",
        lambda result: (relation(0.2 if result is reference_result else 0.3),),
    )

    def fake_public_fit(adata, *, device):
        call_count["reference"] += 1
        return reference_result

    def fake_private_fit(adata, *, device, objective_policy):
        call_count["ablation"] += 1
        return ablation_result

    monkeypatch.setattr(execution, "stride_tl_fit", fake_public_fit)
    monkeypatch.setattr("stride.tl._run._fit_with_objective_policy", fake_private_fit)
    monkeypatch.setattr(
        execution,
        "fit_task_a_pair",
        lambda adata, *, device, estimator: estimator(adata, device=device),
    )

    reference = execution._run_stride_method(
        cohort_inputs=SimpleNamespace(),
        truths=[truth],
        runtime=execution.Block3RuntimeControls(device="cuda:0"),
    )
    ablation = execution._run_stride_method(
        cohort_inputs=SimpleNamespace(),
        truths=[truth],
        runtime=execution.Block3RuntimeControls(device="cuda:0"),
        ablation_mode="geometry",
    )

    assert call_count == {"reference": 1, "ablation": 1}
    assert not np.array_equal(reference["p1"].A, ablation["p1"].A)


def test_exact_comparators_run_through_spawn_pool() -> None:
    truths = [
        SimpleNamespace(
            patient_id="p1",
            x=np.array([0.6, 0.4]),
            y=np.array([0.5, 0.5]),
        ),
        SimpleNamespace(
            patient_id="p2",
            x=np.array([0.3, 0.7]),
            y=np.array([0.4, 0.6]),
        ),
    ]
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])

    balanced = execution._run_balanced_ot_baseline(
        cohort_inputs=SimpleNamespace(cost_matrix=cost),
        truths=truths,
    )
    partial = execution._run_partial_ot_baseline(
        truths=truths,
        cost_matrix=cost,
        matched_mass_budget=0.8,
    )

    assert tuple(balanced) == ("p1", "p2")
    assert tuple(partial) == ("p1", "p2")
    assert all(output.fit_status == "ok" for output in balanced.values())
    assert all(output.fit_status == "ok" for output in partial.values())
    assert all(output.P is not None for output in balanced.values())
    assert all(output.P is not None for output in partial.values())
