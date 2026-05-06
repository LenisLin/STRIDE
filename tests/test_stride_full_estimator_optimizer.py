from __future__ import annotations

import math

import numpy as np
import pytest

from stride.errors import ContractError
from stride.geometry import build_state_geometry
from stride.objectives import FullEstimatorEvidenceBlock
from stride.optimize import FullEstimatorOptimizerConfig, optimize_full_estimator
from stride.outputs.provenance import validate_stride_fit_provenance


def _geometry():
    return build_state_geometry(
        cost_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        cost_scale=1.0,
        state_ids=(0, 1),
    )


def _blocks() -> tuple[FullEstimatorEvidenceBlock, ...]:
    return (
        FullEstimatorEvidenceBlock(
            patient_id="p1",
            source_bag=np.asarray([[0.80, 0.20], [0.70, 0.30]], dtype=float),
            target_bag=np.asarray([[0.25, 0.75], [0.30, 0.70]], dtype=float),
            block_id="p1:TC->TC",
        ),
        FullEstimatorEvidenceBlock(
            patient_id="p1",
            source_bag=np.asarray([[0.60, 0.40]], dtype=float),
            target_bag=np.asarray([[0.40, 0.60]], dtype=float),
            block_id="p1:IM->IM",
        ),
    )


def test_optimizer_runs_adamw_weight_decay_zero_and_builds_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stride.optimize.full_estimator as optimizer_module

    captured: dict[str, float] = {}
    original_adamw = optimizer_module.torch.optim.AdamW

    class RecordingAdamW(original_adamw):  # type: ignore[misc]
        def __init__(self, params, **kwargs):  # type: ignore[no-untyped-def]
            captured["weight_decay"] = float(kwargs["weight_decay"])
            super().__init__(params, **kwargs)

    monkeypatch.setattr(optimizer_module.torch.optim, "AdamW", RecordingAdamW)

    result = optimize_full_estimator(
        patient_ids=("p1",),
        K=2,
        evidence_blocks=_blocks(),
        geometry=_geometry(),
        config=FullEstimatorOptimizerConfig(
            max_steps=2,
            learning_rate=0.02,
            gradient_norm_tol=0.1,
        ),
    )

    assert result.status == "ok"
    assert result.diagnostics["completion_reason"] in {
        "gradient_norm_met",
        "absolute_objective_delta_met",
        "relative_objective_delta_met",
    }
    assert captured["weight_decay"] == pytest.approx(0.0)
    assert result.provenance is not None
    validate_stride_fit_provenance(result.provenance)
    payload = result.provenance.to_dict()
    assert payload["optimizer"]["algorithm"] == "AdamW"
    assert payload["optimizer"]["scheduler_policy"] == "none"
    assert payload["detailed_optimizer_trace"] is False
    assert "ablation_mode" not in payload
    assert math.isfinite(float(result.final_ledger.total))


def test_optimizer_can_return_detailed_trace_and_ablation_provenance() -> None:
    result = optimize_full_estimator(
        patient_ids=("p1",),
        K=2,
        evidence_blocks=_blocks(),
        geometry=_geometry(),
        config=FullEstimatorOptimizerConfig(
            max_steps=2,
            min_steps=1,
            learning_rate=0.01,
            detailed_optimizer_trace=True,
            ablation_mode="geometry",
            min_relative_improvement=0.0,
            gradient_norm_tol=0.1,
        ),
    )

    assert result.status == "ok"
    assert result.trace is not None
    assert result.provenance is not None
    payload = result.provenance.to_dict()
    assert payload["detailed_optimizer_trace"] is True
    assert payload["optimizer_trace_ref"] == "optimizer_trace:result.trace"
    assert payload["ablation_mode"] == "geometry"
    assert payload["ablation_term_handling"] == "zero_weight"
    assert payload["ablation_denominator_policy"] == "fixed_denominator_no_reweighting"


def test_optimizer_defers_without_provenance_on_max_step_exhaustion() -> None:
    result = optimize_full_estimator(
        patient_ids=("p1",),
        K=2,
        evidence_blocks=_blocks(),
        geometry=_geometry(),
        config=FullEstimatorOptimizerConfig(
            max_steps=1,
            min_steps=1,
            learning_rate=0.01,
            min_relative_improvement=0.0,
            convergence_tol=0.0,
            gradient_norm_tol=0.0,
            patience=10,
        ),
    )

    assert result.status == "deferred"
    assert result.provenance is None
    assert result.parameters is None
    assert result.final_ledger is None
    assert result.diagnostics["defer_reason"] == "max_steps_exhausted_without_completion"


def test_optimizer_does_not_complete_from_positive_improvement_alone() -> None:
    result = optimize_full_estimator(
        patient_ids=("p1",),
        K=2,
        evidence_blocks=_blocks(),
        geometry=_geometry(),
        config=FullEstimatorOptimizerConfig(
            max_steps=2,
            min_steps=1,
            learning_rate=0.01,
            min_relative_improvement=0.0,
            convergence_tol=0.0,
            gradient_norm_tol=0.0,
            patience=10,
        ),
    )

    assert result.status == "deferred"
    assert result.provenance is None
    assert result.diagnostics["defer_reason"] in {
        "max_steps_exhausted_without_completion",
        "insufficient_objective_improvement",
    }


def test_optimizer_reuses_fixed_objective_cache_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stride.objectives.full_estimator as objective_module

    call_count = 0
    original = objective_module.compute_init_fov_cost_scale

    def counting_compute_init_fov_cost_scale(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        objective_module,
        "compute_init_fov_cost_scale",
        counting_compute_init_fov_cost_scale,
    )

    result = optimize_full_estimator(
        patient_ids=("p1",),
        K=2,
        evidence_blocks=_blocks(),
        geometry=_geometry(),
        config=FullEstimatorOptimizerConfig(
            max_steps=3,
            min_steps=3,
            learning_rate=0.01,
            min_relative_improvement=0.0,
            convergence_tol=0.0,
            gradient_norm_tol=0.0,
            patience=10,
        ),
    )

    assert result.status in {"ok", "deferred"}
    assert result.diagnostics["n_steps"] >= 3
    assert call_count == len(_blocks())


def test_optimizer_rejects_invalid_scheduler_policy() -> None:
    with pytest.raises(ContractError, match="scheduler_policy"):
        optimize_full_estimator(
            patient_ids=("p1",),
            K=2,
            evidence_blocks=_blocks(),
            geometry=_geometry(),
            config=FullEstimatorOptimizerConfig(scheduler_policy="cosine"),  # type: ignore[arg-type]
        )


def test_optimizer_reports_torch_unavailable_as_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stride.optimize.full_estimator as optimizer_module

    monkeypatch.setattr(optimizer_module, "torch", None)

    with pytest.raises(ContractError, match="requires torch"):
        optimize_full_estimator(
            patient_ids=("p1",),
            K=2,
            evidence_blocks=_blocks(),
            geometry=_geometry(),
        )
