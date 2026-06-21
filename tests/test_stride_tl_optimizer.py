from __future__ import annotations

import dataclasses

import torch

from stride.tl import _optimizer


def test_optimizer_constants_match_frozen_protocol() -> None:
    assert _optimizer.REFERENCE_OPTIMIZER_PROTOCOL == "two_phase_warmup20_main100plus_v1"
    assert _optimizer.WARMUP_STEPS == 20
    assert _optimizer.WARMUP_LR == 0.02
    assert _optimizer.MAIN_LR == 0.05
    assert _optimizer.MAIN_MIN_STEPS == 100
    assert _optimizer.MAIN_MAX_STEPS == 200
    assert _optimizer.COSINE_T_MAX == 200
    assert _optimizer.COSINE_ETA_MIN == 0.0
    assert _optimizer.CONVERGENCE_TOL == 1e-6
    assert _optimizer.PLATEAU_PATIENCE == 5
    assert _optimizer.MIN_RELATIVE_IMPROVEMENT == 0.0
    assert _optimizer.WEIGHT_DECAY == 0.0


def test_optimizer_handoff_returns_fixed_protocol_facts() -> None:
    handoff = _optimizer.optimizer_handoff()

    assert dataclasses.is_dataclass(handoff)
    assert handoff == _optimizer.OptimizerHandoff(
        protocol_name="two_phase_warmup20_main100plus_v1",
        warmup_steps=20,
        warmup_lr=0.02,
        main_lr=0.05,
        main_min_steps=100,
        main_max_steps=200,
        scheduler_policy="CosineAnnealingLR",
        cosine_T_max=200,
        cosine_eta_min=0.0,
        convergence_tol=1e-6,
        patience=5,
        min_relative_improvement=0.0,
        weight_decay=0.0,
    )


def test_optimizer_handoff_has_no_status_result_loss_or_evidence_fields() -> None:
    field_names = {
        field.name for field in dataclasses.fields(_optimizer.OptimizerHandoff)
    }

    assert field_names.isdisjoint(
        {
            "status",
            "fit_status",
            "result",
            "loss",
            "loss_ledger",
            "evidence",
            "evidence_blocks",
        }
    )


def test_create_adamw_uses_fixed_weight_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))

    optimizer = _optimizer.create_adamw([parameter], lr=0.123)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.123
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_create_main_scheduler_uses_fixed_cosine_protocol() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = _optimizer.create_adamw([parameter], lr=_optimizer.MAIN_LR)

    scheduler = _optimizer.create_main_scheduler(optimizer)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 200
    assert scheduler.eta_min == 0.0
