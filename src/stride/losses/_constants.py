"""Private constants for the STRIDE v1 loss contract."""
from __future__ import annotations

EPSILON_NORM = 1e-2
RHO_SUBBAG = 1.0
GEOMETRY_EFFECTIVE_WEIGHT = 1e-2
S_COHORT = 1e-2
OFFDIAG_INIT_MASS = 1e-2
ABLATION_MODES = frozenset({"none", "geometry", "recurrence", "consistency"})
ABLATION_TERM_HANDLING = "zero_weight"
NUMERICAL_MIN_MASS = 1e-12
OBJECTIVE_CONTRACT_VERSION = "stride_full_estimator_three_block_v1"
S_G_INIT_RTOL = 1e-7
S_G_INIT_ATOL = 1e-10

__all__ = [
    "ABLATION_MODES",
    "ABLATION_TERM_HANDLING",
    "EPSILON_NORM",
    "GEOMETRY_EFFECTIVE_WEIGHT",
    "NUMERICAL_MIN_MASS",
    "OBJECTIVE_CONTRACT_VERSION",
    "OFFDIAG_INIT_MASS",
    "RHO_SUBBAG",
    "S_COHORT",
    "S_G_INIT_ATOL",
    "S_G_INIT_RTOL",
]
