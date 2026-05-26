"""Loss ledger records for STRIDE.

Task: expose schema records for canonical raw components, normalized
components, optimizer-effective components, block fractions, geometry effective
weight, recurrence scale, and off-diagonal optimizer start mass. Reference:
``docs/api_specs.md`` compact provenance schema and ``docs/decisions.md`` D015.
"""
from __future__ import annotations

from .assembly import (
    GEOMETRY_EFFECTIVE_WEIGHT,
    OFFDIAG_INIT_MASS,
    RHO_SUBBAG,
    S_COHORT,
    CohortLossLedger,
    ConsistencyPatientLedger,
    LossComponent,
    LossLedger,
    LossTotals,
    ObservationBlockLedger,
)

__all__ = [
    "GEOMETRY_EFFECTIVE_WEIGHT",
    "OFFDIAG_INIT_MASS",
    "RHO_SUBBAG",
    "S_COHORT",
    "CohortLossLedger",
    "ConsistencyPatientLedger",
    "LossComponent",
    "LossLedger",
    "LossTotals",
    "ObservationBlockLedger",
]
