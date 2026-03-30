"""Compatibility re-exports for older contract-centric imports."""
from __future__ import annotations

from ..errors import ContractError, DataContractError
from ..io.longitudinal import (
    CANONICAL_COST_SCALE_KEY,
    COST_SCALE_ALIASES,
    OPTIONAL_OBS_COLS,
    OPTIONAL_OBSM_KEYS,
    OPTIONAL_UNS_KEYS,
    REQUIRED_OBS_COLS,
    REQUIRED_OBSM_KEYS,
    REQUIRED_UNS_KEYS,
    validate_longitudinal_adata,
)
from ..validation import validate_observation_match_inputs
from .status import CANONICAL_BYPASS_REASONS, CANONICAL_UOT_STATUSES
from .tables import MICRO_METRICS, validate_events_table, validate_metrics_table


def validate_adata_inputs(
    adata: object,
    *,
    require_cell_type: bool = False,
    require_representation: bool = False,
    require_prototypes: bool = False,
    require_cost_scale: bool = False,
    require_cost_matrix: bool = False,
) -> None:
    """Compatibility wrapper over `validate_longitudinal_adata(...)`."""
    validate_longitudinal_adata(
        adata,  # type: ignore[arg-type]
        require_cell_type=require_cell_type,
        require_representation=require_representation,
        require_state_axis=require_prototypes,
        require_cost_scale=require_cost_scale,
        require_cost_matrix=require_cost_matrix,
    )


def validate_uot_inputs(
    A: object,
    B: object,
    lambda_pl: object,
    kernels: object,
) -> None:
    """Compatibility wrapper over canonical observation validation."""
    validate_observation_match_inputs(A=A, B=B, match_penalty=lambda_pl, kernels=kernels)  # type: ignore[arg-type]


__all__ = [
    "CANONICAL_BYPASS_REASONS",
    "CANONICAL_COST_SCALE_KEY",
    "CANONICAL_UOT_STATUSES",
    "COST_SCALE_ALIASES",
    "ContractError",
    "DataContractError",
    "MICRO_METRICS",
    "OPTIONAL_OBS_COLS",
    "OPTIONAL_OBSM_KEYS",
    "OPTIONAL_UNS_KEYS",
    "REQUIRED_OBS_COLS",
    "REQUIRED_OBSM_KEYS",
    "REQUIRED_UNS_KEYS",
    "validate_adata_inputs",
    "validate_events_table",
    "validate_metrics_table",
    "validate_uot_inputs",
]
