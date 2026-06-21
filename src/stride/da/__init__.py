"""Downstream analysis namespace for fitted STRIDE outputs."""
from __future__ import annotations

from ._arrays import patient_relation_arrays
from ._association import augmented_entry_group_association
from ._programs import (
    relation_program_decomposition,
    relation_program_group_association,
    relation_program_rank_diagnostics,
)

__all__ = (
    "patient_relation_arrays",
    "augmented_entry_group_association",
    "relation_program_rank_diagnostics",
    "relation_program_decomposition",
    "relation_program_group_association",
)
