from __future__ import annotations

import dataclasses

import pytest
import torch

import stride.tl._losses as losses_module
import stride.tl._train as train_module
from stride.errors import ContractError
from stride.tl._losses import LossContext, LossLedger
from stride.tl._parameters import initialize_parameters
from stride.tl._resolve import EvidenceBlock, RelationInput
from stride.tl._sinkhorn import SinkhornResult
from stride.tl._train import (
    RelationModel,
    TrainingResult,
    _compute_block_fov_cost_scale,
    _materialize_relation_inputs_once,
    _plateau_condition_met,
    _populate_batched_block_context,
    _resolve_runtime_device,
    _scale_initial_parameters,
    train_relation,
)


def _relation_fixture() -> tuple[RelationInput, torch.Tensor, float]:
    source = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]], dtype=torch.float64)
    target = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2]], dtype=torch.float64)
    relation = RelationInput(
        relation_id="r0",
        source_timepoint="pre",
        target_timepoint="post",
        source_domain="TC",
        target_domain="IM",
        patient_ids=("p1",),
        support_counts={"p1": {"source": 2, "target": 2}},
        skipped_patient_ids=(),
        blocks=(
            EvidenceBlock(
                patient_id="p1",
                source_bag=source,
                target_bag=target,
                block_id="p1:subbag_0",
                metadata={"subbag_index": 0},
            ),
        ),
        metadata={"block_construction_policy": "partitioned_fov_subbag_v1"},
    )
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]],
        dtype=torch.float64,
    )
    return relation, cost, 1.5


def _multi_shape_relation_fixture() -> tuple[RelationInput, torch.Tensor, float]:
    blocks = (
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64),
            target_bag=torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64),
            block_id="p1_1x1",
        ),
        EvidenceBlock(
            patient_id="p1",
            source_bag=torch.tensor([[0.2, 0.6, 0.2]], dtype=torch.float64),
            target_bag=torch.tensor(
                [[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]],
                dtype=torch.float64,
            ),
            block_id="p1_1x2",
        ),
        EvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64),
            target_bag=torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float64),
            block_id="p2_1x1",
        ),
        EvidenceBlock(
            patient_id="p2",
            source_bag=torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64),
            target_bag=torch.tensor(
                [[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]],
                dtype=torch.float64,
            ),
            block_id="p2_1x2",
        ),
    )
    relation = RelationInput(
        relation_id="r0",
        source_timepoint="pre",
        target_timepoint="post",
        source_domain="TC",
        target_domain="IM",
        patient_ids=("p1", "p2"),
        support_counts={
            "p1": {"source": 2, "target": 3},
            "p2": {"source": 2, "target": 3},
        },
        skipped_patient_ids=(),
        blocks=blocks,
        metadata={"block_construction_policy": "partitioned_fov_subbag_v1"},
    )
    cost = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]],
        dtype=torch.float64,
    )
    return relation, cost, 1.5


def test_training_scaffold_imports_expected_public_symbols() -> None:
    assert dataclasses.is_dataclass(TrainingResult)
    assert callable(train_relation)


def test_training_result_has_no_status_field() -> None:
    field_names = {field.name for field in dataclasses.fields(TrainingResult)}

    assert field_names == {"parameters", "loss_ledger", "run_info", "trace"}
    assert "status" not in field_names


def test_training_run_info_keeps_random_seed_field_without_public_control() -> None:
    run_info = train_module.TrainingRunInfo()

    assert run_info.random_seed is None


def test_training_run_info_records_optimizer_runtime_facts() -> None:
    run_info = train_module.TrainingRunInfo(
        reason="max_steps",
        optimizer_exit_flag="max_steps_exhausted_finite",
        n_steps=3,
        warmup_steps_completed=1,
        main_steps_completed=2,
        initial_total=4.0,
        final_total=3.5,
        absolute_improvement=0.5,
        relative_improvement=0.125,
    )

    assert run_info.reason == "max_steps"
    assert run_info.n_steps == 3
    assert run_info.warmup_steps_completed == 1
    assert run_info.main_steps_completed == 2
    assert run_info.initial_total == 4.0
    assert run_info.final_total == 3.5
    assert run_info.absolute_improvement == 0.5
    assert run_info.relative_improvement == 0.125


def test_resolve_runtime_device_accepts_cpu() -> None:
    assert _resolve_runtime_device("cpu") == torch.device("cpu")


def test_resolve_runtime_device_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ContractError, match="unavailable"):
        _resolve_runtime_device("cuda:0")


def test_relation_model_registers_initial_logits_as_parameters_on_device() -> None:
    initial = initialize_parameters(["p1", "p2"], 3, device=torch.device("cpu"))

    model = RelationModel(initial)

    assert model.row_logits.device == initial.row_logits.device
    assert model.e_logits.device == initial.e_logits.device
    assert isinstance(model.row_logits, torch.nn.Parameter)
    assert isinstance(model.e_logits, torch.nn.Parameter)
    assert dict(model.named_parameters()).keys() == {"row_logits", "e_logits"}


def test_relation_model_state_returns_constrained_parameters_aligned_to_patients() -> None:
    initial = initialize_parameters(["p2", "p1"], 2)
    model = RelationModel(initial)

    parameters = model.state()

    assert parameters.patient_ids == ("p2", "p1")
    assert parameters.A.shape == (2, 2, 2)
    assert parameters.d.shape == (2, 2)
    assert parameters.e.shape == (2, 2)
    torch.testing.assert_close(
        parameters.A.sum(dim=2) + parameters.d,
        torch.ones((2, 2), dtype=torch.float64),
    )
    assert bool((parameters.A >= 0.0).all())
    assert bool((parameters.d >= 0.0).all())
    assert bool(((parameters.e >= 0.0) & (parameters.e <= 1.0)).all())


def test_scale_initial_parameters_uses_identity_plus_small_open_without_offdiag_seed() -> None:
    parameters = _scale_initial_parameters(("p1", "p2"), 3, device=torch.device("cpu"))

    expected_A = torch.diag(torch.full((3,), 0.95, dtype=torch.float64))
    torch.testing.assert_close(parameters.A[0], expected_A)
    torch.testing.assert_close(parameters.A[1], expected_A)
    torch.testing.assert_close(parameters.d, torch.full((2, 3), 0.05, dtype=torch.float64))
    torch.testing.assert_close(
        parameters.e,
        torch.full((2, 3), 0.05 / 3.0, dtype=torch.float64),
    )


def test_scale_initial_parameters_handles_single_state() -> None:
    parameters = _scale_initial_parameters(("p1",), 1, device=torch.device("cpu"))

    torch.testing.assert_close(parameters.A, torch.tensor([[[0.95]]], dtype=torch.float64))
    torch.testing.assert_close(parameters.d, torch.tensor([[0.05]], dtype=torch.float64))
    torch.testing.assert_close(parameters.e, torch.tensor([[0.05]], dtype=torch.float64))


def test_materialize_relation_inputs_once_moves_cost_and_blocks_to_target_dtype() -> None:
    relation, cost, scale = _relation_fixture()

    materialized, blocks, materialized_cost, materialized_scale = (
        _materialize_relation_inputs_once(
            relation,
            cost.float(),
            scale,
            device=torch.device("cpu"),
        )
    )

    assert materialized is not relation
    assert materialized.blocks == blocks
    assert materialized_scale == scale
    assert materialized_cost.dtype == torch.float64
    assert blocks[0].source_bag.dtype == torch.float64
    assert blocks[0].target_bag.dtype == torch.float64
    assert blocks[0].source_bag.device == materialized_cost.device
    assert blocks[0].target_bag.device == materialized_cost.device


def test_materialize_relation_inputs_once_rejects_bad_cost_scale() -> None:
    relation, cost, _ = _relation_fixture()

    with pytest.raises(ContractError):
        _materialize_relation_inputs_once(
            relation,
            cost,
            0.0,
            device=torch.device("cpu"),
        )


def test_compute_block_fov_cost_scale_uses_median_positive_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation, cost, scale = _relation_fixture()
    parameters = _scale_initial_parameters(relation.patient_ids, 3, device=torch.device("cpu"))

    def fake_ground(*args: object, **kwargs: object) -> SinkhornResult:
        return SinkhornResult(
            value=torch.tensor([[0.0, 2.0, 10.0]], dtype=torch.float64),
            metadata={},
        )

    def fake_observed_self(*args: object, **kwargs: object) -> SinkhornResult:
        return SinkhornResult(
            value=torch.zeros((2, 2), dtype=torch.float64),
            metadata={"clipped_negative": False},
        )

    monkeypatch.setattr(train_module, "compute_fov_ground_cost_matrix", fake_ground)
    monkeypatch.setattr(train_module, "compute_observed_self_ground_cost", fake_observed_self)

    s_G_init, floor_used, observed_self, clipped_negative = _compute_block_fov_cost_scale(
        relation.blocks[0],
        parameters=parameters,
        cost_matrix=cost,
        cost_scale=scale,
        sinkhorn_config=train_module.SinkhornConfig(),
    )

    assert s_G_init == 6.0
    assert floor_used is False
    torch.testing.assert_close(observed_self, torch.zeros((2, 2), dtype=torch.float64))
    assert clipped_negative is False


def test_compute_block_fov_cost_scale_uses_unit_floor_when_no_positive_ground_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation, cost, scale = _relation_fixture()
    parameters = _scale_initial_parameters(relation.patient_ids, 3, device=torch.device("cpu"))

    def fake_ground(*args: object, **kwargs: object) -> SinkhornResult:
        return SinkhornResult(value=torch.zeros((1, 3), dtype=torch.float64), metadata={})

    def fake_observed_self(*args: object, **kwargs: object) -> SinkhornResult:
        return SinkhornResult(
            value=torch.zeros((2, 2), dtype=torch.float64),
            metadata={"clipped_negative": True},
        )

    monkeypatch.setattr(train_module, "compute_fov_ground_cost_matrix", fake_ground)
    monkeypatch.setattr(train_module, "compute_observed_self_ground_cost", fake_observed_self)

    s_G_init, floor_used, _, clipped_negative = _compute_block_fov_cost_scale(
        relation.blocks[0],
        parameters=parameters,
        cost_matrix=cost,
        cost_scale=scale,
        sinkhorn_config=train_module.SinkhornConfig(),
    )

    assert s_G_init == 1.0
    assert floor_used is True
    assert clipped_negative is True


def test_batched_block_context_matches_single_block_context() -> None:
    relation, cost, scale = _multi_shape_relation_fixture()
    parameters = _scale_initial_parameters(
        relation.patient_ids,
        3,
        device=torch.device("cpu"),
    )
    config = train_module.SinkhornConfig()
    batched_scales: dict[str, float] = {}
    batched_floor: dict[str, bool] = {}
    batched_observed_self: dict[str, torch.Tensor] = {}
    batched_clipped: dict[str, bool] = {}

    _populate_batched_block_context(
        blocks=relation.blocks,
        parameters=parameters,
        cost_matrix=cost,
        cost_scale=scale,
        sinkhorn_config=config,
        fov_cost_scales=batched_scales,
        fov_cost_scale_floor_used=batched_floor,
        observed_self_ground_costs=batched_observed_self,
        observed_self_clipped_negative=batched_clipped,
    )

    for block in relation.blocks:
        (
            single_scale,
            single_floor,
            single_observed_self,
            single_clipped,
        ) = _compute_block_fov_cost_scale(
            block,
            parameters=parameters,
            cost_matrix=cost,
            cost_scale=scale,
            sinkhorn_config=config,
        )
        assert batched_scales[block.block_id] == pytest.approx(single_scale, abs=1e-12)
        assert bool(batched_floor[block.block_id]) is bool(single_floor)
        torch.testing.assert_close(
            batched_observed_self[block.block_id],
            single_observed_self,
            atol=1e-8,
            rtol=0.0,
        )
        assert bool(batched_clipped[block.block_id]) is bool(single_clipped)


def test_batched_block_context_preserves_observed_self_clipped_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation, cost, scale = _relation_fixture()
    parameters = _scale_initial_parameters(
        relation.patient_ids,
        3,
        device=torch.device("cpu"),
    )
    original_result = train_module._batched_pairwise_composition_ground_cost_result

    def fake_batched_ground_result(
        left: torch.Tensor,
        right: torch.Tensor,
        C_norm: torch.Tensor,
        *,
        config: train_module.SinkhornConfig,
        label: str,
        runtime_checks: bool,
    ) -> losses_module._BatchedGroundCostValue:
        assert runtime_checks is True
        if "observed_self" not in label:
            return original_result(
                left,
                right,
                C_norm,
                config=config,
                label=label,
                runtime_checks=runtime_checks,
            )
        value = torch.zeros(
            (left.shape[0], left.shape[1], right.shape[1]),
            dtype=torch.float64,
            device=left.device,
        )
        return losses_module._BatchedGroundCostValue(
            value=value,
            clipped_negative=tuple(True for _ in range(left.shape[0])),
        )

    monkeypatch.setattr(
        train_module,
        "_batched_pairwise_composition_ground_cost_result",
        fake_batched_ground_result,
    )
    batched_scales: dict[str, float] = {}
    batched_floor: dict[str, bool] = {}
    batched_observed_self: dict[str, torch.Tensor] = {}
    batched_clipped: dict[str, bool] = {}

    _populate_batched_block_context(
        blocks=relation.blocks,
        parameters=parameters,
        cost_matrix=cost,
        cost_scale=scale,
        sinkhorn_config=train_module.SinkhornConfig(),
        fov_cost_scales=batched_scales,
        fov_cost_scale_floor_used=batched_floor,
        observed_self_ground_costs=batched_observed_self,
        observed_self_clipped_negative=batched_clipped,
    )

    assert batched_clipped == {"p1:subbag_0": True}


def test_train_relation_runs_status_free_smoke_path(monkeypatch: pytest.MonkeyPatch) -> None:
    relation, cost, scale = _relation_fixture()
    context = LossContext(
        obs_scale=torch.tensor(1.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"p1:subbag_0": 1.0},
    )

    def fake_build_context_once(*args: object, **kwargs: object) -> LossContext:
        return context

    def fake_compute_total_loss(*args: object, **kwargs: object) -> LossLedger:
        parameters = args[0]
        total = parameters.e.sum() * 0.01
        return LossLedger(
            total=total,
            fit=total,
            prior=total,
            cohort=total,
            components={
                "obs_raw": total,
                "geometry_raw": total,
            },
        )

    monkeypatch.setattr(train_module, "_build_loss_context_once", fake_build_context_once)
    monkeypatch.setattr(train_module, "compute_total_loss", fake_compute_total_loss)
    monkeypatch.setattr(train_module, "WARMUP_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MIN_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MAX_STEPS", 2)
    monkeypatch.setattr(train_module, "PLATEAU_PATIENCE", 1)

    result = train_relation(
        relation,
        cost,
        scale,
        device="cpu",
    )

    assert result.parameters is not None
    assert result.loss_ledger is not None
    assert result.run_info.optimizer_exit_flag in {
        "max_steps_exhausted_finite",
        "plateau_patience",
    }
    assert result.run_info.reason in {"max_steps", "plateau_patience"}
    assert result.run_info.n_steps == 1 + result.run_info.main_steps_completed
    assert result.run_info.warmup_steps_completed == 1
    assert result.run_info.initial_total is not None
    assert result.run_info.final_total is not None
    assert result.run_info.absolute_improvement is not None
    assert result.run_info.relative_improvement is not None
    assert result.run_info.random_seed is None
    assert result.trace is None


def test_train_relation_passes_policy_through_warmup_main_and_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stride.tl._objective import NO_GEOMETRY_OBJECTIVE_POLICY

    relation, cost, scale = _relation_fixture()
    context = LossContext(
        obs_scale=torch.tensor(1.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"p1:subbag_0": 1.0},
    )
    policies: list[object] = []

    monkeypatch.setattr(train_module, "_build_loss_context_once", lambda **kwargs: context)

    def fake_compute_total_loss(*args: object, **kwargs: object) -> LossLedger:
        policies.append(kwargs["objective_policy"])
        parameters = args[0]
        total = parameters.e.sum() * 0.01
        return LossLedger(
            total=total,
            fit=total,
            prior=total,
            cohort=total,
            components={"obs_raw": total, "geometry_raw": total},
        )

    monkeypatch.setattr(train_module, "compute_total_loss", fake_compute_total_loss)
    monkeypatch.setattr(train_module, "WARMUP_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MIN_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MAX_STEPS", 1)
    monkeypatch.setattr(train_module, "PLATEAU_PATIENCE", 1)

    train_relation(
        relation,
        cost,
        scale,
        device="cpu",
        objective_policy=NO_GEOMETRY_OBJECTIVE_POLICY,
    )

    assert policies == [NO_GEOMETRY_OBJECTIVE_POLICY] * 3


def test_plateau_condition_uses_absolute_magnitude_and_disabled_relative_gate() -> None:
    assert not _plateau_condition_met(
        absolute_improvement=-10.0,
        relative_improvement=-10.0,
    )
    assert _plateau_condition_met(
        absolute_improvement=5e-7,
        relative_improvement=5e-7,
    )
    assert not _plateau_condition_met(
        absolute_improvement=1.0,
        relative_improvement=0.0,
    )


def test_train_relation_plateau_counts_only_completed_main_optimizer_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation, cost, scale = _relation_fixture()
    step_counter = _patch_optimizer_step_counter(monkeypatch)
    _patch_training_totals(monkeypatch, [1.0, 1.0 + 5e-7])
    monkeypatch.setattr(train_module, "WARMUP_STEPS", 0)
    monkeypatch.setattr(train_module, "MAIN_MIN_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MAX_STEPS", 4)
    monkeypatch.setattr(train_module, "PLATEAU_PATIENCE", 1)

    result = train_relation(
        relation,
        cost,
        scale,
        device="cpu",
    )

    assert result.run_info.optimizer_exit_flag == "plateau_patience"
    assert result.run_info.main_steps_completed == 2
    assert step_counter["optimizer_steps"] == result.run_info.main_steps_completed


def test_train_relation_worsening_total_exhausts_main_steps_without_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation, cost, scale = _relation_fixture()
    step_counter = _patch_optimizer_step_counter(monkeypatch)
    _patch_training_totals(monkeypatch, [1.0, 11.0, 21.0])
    monkeypatch.setattr(train_module, "WARMUP_STEPS", 0)
    monkeypatch.setattr(train_module, "MAIN_MIN_STEPS", 1)
    monkeypatch.setattr(train_module, "MAIN_MAX_STEPS", 3)
    monkeypatch.setattr(train_module, "PLATEAU_PATIENCE", 1)

    result = train_relation(
        relation,
        cost,
        scale,
        device="cpu",
    )

    assert result.run_info.optimizer_exit_flag == "max_steps_exhausted_finite"
    assert result.run_info.main_steps_completed == 3
    assert step_counter["optimizer_steps"] == 3


def _patch_training_totals(
    monkeypatch: pytest.MonkeyPatch,
    totals: list[float],
) -> dict[str, int]:
    context = LossContext(
        obs_scale=torch.tensor(1.0, dtype=torch.float64),
        geometry_scale=torch.tensor(1.0, dtype=torch.float64),
        fov_cost_scales={"p1:subbag_0": 1.0},
    )
    call_counter = {"loss_calls": 0}

    def fake_build_context_once(*args: object, **kwargs: object) -> LossContext:
        return context

    def fake_compute_total_loss(*args: object, **kwargs: object) -> LossLedger:
        parameters = args[0]
        index = min(call_counter["loss_calls"], len(totals) - 1)
        call_counter["loss_calls"] += 1
        constant = torch.tensor(
            float(totals[index]),
            dtype=torch.float64,
            device=parameters.e.device,
        )
        total = parameters.e.sum() * 0.0 + constant
        return LossLedger(
            total=total,
            fit=total,
            prior=total,
            cohort=total,
            components={
                "obs_raw": total,
                "geometry_raw": total,
            },
        )

    monkeypatch.setattr(train_module, "_build_loss_context_once", fake_build_context_once)
    monkeypatch.setattr(train_module, "compute_total_loss", fake_compute_total_loss)
    return call_counter


def _patch_optimizer_step_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    step_counter = {"optimizer_steps": 0}

    class CountingAdamW(torch.optim.AdamW):
        def step(self, closure: object = None) -> object:
            step_counter["optimizer_steps"] += 1
            return super().step(closure=closure)

    def fake_create_adamw(
        parameters: object,
        *,
        lr: float,
    ) -> torch.optim.AdamW:
        return CountingAdamW(parameters, lr=float(lr), weight_decay=0.0)

    monkeypatch.setattr(train_module, "create_adamw", fake_create_adamw)
    return step_counter
