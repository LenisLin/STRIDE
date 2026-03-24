from __future__ import annotations

# SLOTAR library — architecture layers:
#   uot          : solver core (stateless, thread-safe)
#   io.bridge    : export contract (no config parsing)
#   contracts    : input validation
#   utils        : shared math primitives
#
# Concurrency contract: batched_uot_solve, calibrate_joint_lambda, and
# precompute_logKernels are safe for concurrent invocation via
# ThreadPoolExecutor. Each call allocates all working arrays internally.
# Callers must not share mutable state (e.g. timing dicts) across threads.

from .contracts import (
    DataContractError,
    validate_adata_inputs,
    validate_events_table,
    validate_metrics_table,
    validate_uot_inputs,
)
from .io.bridge import save_for_r
from .uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels

__all__ = [
    "__version__",
    "DataContractError",
    "UOTSolveConfig",
    "batched_uot_solve",
    "precompute_logKernels",
    "save_for_r",
    "validate_adata_inputs",
    "validate_events_table",
    "validate_metrics_table",
    "validate_uot_inputs",
]

__version__ = "0.1.0"
