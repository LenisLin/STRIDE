"""STRIDE AnnData assembly and h5ad I/O helpers."""
from __future__ import annotations

from ._build import build_adata
from ._h5ad import read_h5ad, write_h5ad
from ._r_handover import (
    write_cohort_table,
    write_descriptive_tables,
    write_fraction_table,
    write_program_score_table,
    write_r_handover,
)

__all__ = [
    "build_adata",
    "read_h5ad",
    "write_cohort_table",
    "write_descriptive_tables",
    "write_fraction_table",
    "write_h5ad",
    "write_program_score_table",
    "write_r_handover",
]
