"""Task-resolved evidence contracts for STRIDE losses.

Task: expose source/target FOV-bag evidence blocks after workflow resolution.
Reference: ``docs/constraints.md`` C17 states that task layers own source and
target declaration, while the reusable loss core receives resolved evidence and
does not treat domain as a loss, state, relation, or recurrence axis.
"""
from __future__ import annotations

from .assembly import EvidenceBlock

EvidenceSet = tuple[EvidenceBlock, ...]

__all__ = ["EvidenceBlock", "EvidenceSet"]
