"""Optimizer primitives for STRIDE `.tl`.

This module owns only the fixed AdamW optimizer protocol, scheduler construction,
and step-level torch optimizer helpers. It does not own relation fitting,
training lifecycle, evidence materialization, failure policy, or result objects.
"""
from __future__ import annotations

from collections.abc import Iterable as _Iterable
from dataclasses import dataclass

import torch as _torch

__all__ = [
    "OptimizerHandoff",
    "REFERENCE_OPTIMIZER_PROTOCOL",
    "WARMUP_STEPS",
    "WARMUP_LR",
    "MAIN_LR",
    "MAIN_MIN_STEPS",
    "MAIN_MAX_STEPS",
    "COSINE_T_MAX",
    "COSINE_ETA_MIN",
    "CONVERGENCE_TOL",
    "PLATEAU_PATIENCE",
    "MIN_RELATIVE_IMPROVEMENT",
    "WEIGHT_DECAY",
    "create_adamw",
    "create_main_scheduler",
    "optimizer_handoff",
]

REFERENCE_OPTIMIZER_PROTOCOL = "two_phase_warmup20_main100plus_v1"

WARMUP_STEPS = 20
WARMUP_LR = 0.02
MAIN_LR = 0.05
MAIN_MIN_STEPS = 100
MAIN_MAX_STEPS = 200
COSINE_T_MAX = 200
COSINE_ETA_MIN = 0.0
CONVERGENCE_TOL = 1e-6
PLATEAU_PATIENCE = 5
MIN_RELATIVE_IMPROVEMENT = 0.0
WEIGHT_DECAY = 0.0


@dataclass(frozen=True)
class OptimizerHandoff:
    """Fixed optimizer protocol facts shared with training/output layers.

    This object is a read-only protocol handoff. It carries no relation data,
    loss values, fit status, evidence metadata, or runtime failure information.
    """

    protocol_name: str
    warmup_steps: int
    warmup_lr: float
    main_lr: float
    main_min_steps: int
    main_max_steps: int
    scheduler_policy: str
    cosine_T_max: int
    cosine_eta_min: float
    convergence_tol: float
    patience: int
    min_relative_improvement: float
    weight_decay: float


def create_adamw(
    parameters: _Iterable[_torch.nn.Parameter],
    *,
    lr: float,
) -> _torch.optim.AdamW:
    """Construct AdamW with the fixed STRIDE weight-decay policy.

    The caller owns training-loop timing and selected stage learning rate.
    """
    return _torch.optim.AdamW(parameters, lr=float(lr), weight_decay=WEIGHT_DECAY)


def create_main_scheduler(
    optimizer: _torch.optim.Optimizer,
) -> _torch.optim.lr_scheduler.CosineAnnealingLR:
    """Construct the canonical main-stage cosine scheduler.

    The training runtime decides when the main stage starts and when
    scheduler.step() is called.
    """
    return _torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=COSINE_T_MAX,
        eta_min=COSINE_ETA_MIN,
    )


def _set_optimizer_lr(optimizer: _torch.optim.Optimizer, lr: float) -> None:
    """Reset optimizer group learning rates for stage transitions."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def optimizer_handoff() -> OptimizerHandoff:
    """Return fixed optimizer protocol facts without runtime status.

    This is used by training/output handoff code to avoid duplicating protocol
    metadata outside `_optimizer.py`.
    """
    return OptimizerHandoff(
        protocol_name=REFERENCE_OPTIMIZER_PROTOCOL,
        warmup_steps=WARMUP_STEPS,
        warmup_lr=WARMUP_LR,
        main_lr=MAIN_LR,
        main_min_steps=MAIN_MIN_STEPS,
        main_max_steps=MAIN_MAX_STEPS,
        scheduler_policy="CosineAnnealingLR",
        cosine_T_max=COSINE_T_MAX,
        cosine_eta_min=COSINE_ETA_MIN,
        convergence_tol=CONVERGENCE_TOL,
        patience=PLATEAU_PATIENCE,
        min_relative_improvement=MIN_RELATIVE_IMPROVEMENT,
        weight_decay=WEIGHT_DECAY,
    )
