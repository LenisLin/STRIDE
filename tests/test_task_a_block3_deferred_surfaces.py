from __future__ import annotations

import json

import pytest

from tasks.task_A.block3.execution import (
    Block3DeferredExecutionResult,
    execute_internal_block3_experiment,
)


@pytest.mark.parametrize(
    ("experiment_name", "requested_ablation"),
    (
        ("subbag_consistency_ablation", "subbag_consistency"),
        ("geometry_ablation", "geometry"),
        ("recurrence_ablation", "recurrence"),
    ),
)
def test_block3_ablation_exits_with_structured_deferred_status(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
    requested_ablation: str,
) -> None:
    monkeypatch.setattr(
        "tasks.task_A.block3.execution._build_block3_cohort_inputs_from_stage0",
        lambda **kwargs: pytest.fail("deferred ablation must not build cohort inputs"),
    )

    result = execute_internal_block3_experiment(
        experiment_name=experiment_name,
        task_config_path=tmp_path / "missing.yaml",
        stage0_h5ad=tmp_path / "missing.h5ad",
        output_dir=tmp_path,
    )

    assert isinstance(result, Block3DeferredExecutionResult)
    assert result.requested_ablation == requested_ablation
    assert result.fit_surface == "stride.tl.fit"
    assert result.reason_code == "public_estimator_ablation_hook_unavailable"
    payload = json.loads(result.status_path.read_text())
    assert payload["status"] == "deferred"
    assert payload["supported"] is False
    assert payload["requested_ablation"] == requested_ablation
    assert payload["fit_surface"] == "stride.tl.fit"
    assert payload["frozen_contract_reference"].startswith(
        "docs/task_A/block3/scientific_contract.md"
    )
