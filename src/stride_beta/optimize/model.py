"""Torch model for constrained STRIDE patient relations."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import ContractError
from ..losses.assembly import ADEState, LogitState, parameters_from_unconstrained, unconstrained_from_initialization

try:  # pragma: no cover - exercised through optimizer runtime
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def require_torch() -> Any:
    if torch is None:  # pragma: no cover
        raise ContractError("canonical STRIDE optimizer requires torch")
    return torch


class RelationModel(require_torch().nn.Module if torch is not None else object):
    """Trainable logits whose forward pass returns constrained ``A, d, e``."""

    def __init__(self, *, patient_ids: Sequence[str], K: int, device: Any | None = None) -> None:
        torch_module = require_torch()
        super().__init__()
        initial = unconstrained_from_initialization(patient_ids, K, device=device)
        self.patient_ids = tuple(initial.patient_ids)
        self.row_logits = torch_module.nn.Parameter(initial.row_logits.clone().detach())
        self.e_logits = torch_module.nn.Parameter(initial.e_logits.clone().detach())

    def logit_state(self) -> LogitState:
        return LogitState(
            patient_ids=self.patient_ids,
            row_logits=self.row_logits,
            e_logits=self.e_logits,
        )

    def state(self) -> ADEState:
        return parameters_from_unconstrained(self.logit_state())

    def forward(self) -> tuple[Any, Any, Any]:
        state = self.state()
        return state.A, state.d, state.e


__all__ = ["RelationModel", "require_torch"]
