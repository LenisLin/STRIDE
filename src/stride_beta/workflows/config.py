"""Workflow-level task configuration for STRIDE fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..errors import ContractError


def _normalize_identifier(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if normalized == "":
        raise ContractError(f"{field_name} must be a non-empty string")
    return normalized


@dataclass(frozen=True)
class TaskConfig:
    """Task semantic inputs resolved before optimizer/runtime code."""

    source: str
    target: str
    K: int
    patient_ids: tuple[str, ...] | None = None
    timepoint_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source = _normalize_identifier(self.source, field_name="source")
        target = _normalize_identifier(self.target, field_name="target")
        if source == target:
            raise ContractError("source and target must be distinct")
        if isinstance(self.K, bool) or not isinstance(self.K, int) or int(self.K) <= 0:
            raise ContractError("K must be a positive integer")
        patient_ids = None
        if self.patient_ids is not None:
            patient_ids = tuple(
                _normalize_identifier(patient_id, field_name="patient_ids item")
                for patient_id in self.patient_ids
            )
            if len(set(patient_ids)) != len(patient_ids):
                raise ContractError("patient_ids must not contain duplicates")
        timepoint_order = tuple(
            _normalize_identifier(label, field_name="timepoint_order label")
            for label in self.timepoint_order
        )
        if len(set(timepoint_order)) != len(timepoint_order):
            raise ContractError("timepoint_order must not contain duplicates")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "patient_ids", patient_ids)
        object.__setattr__(self, "timepoint_order", timepoint_order)


def coerce_patient_ids(patient_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if patient_ids is None:
        return None
    return tuple(str(patient_id) for patient_id in patient_ids)


__all__ = ["TaskConfig", "coerce_patient_ids"]
