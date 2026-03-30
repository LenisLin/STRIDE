"""Compatibility status vocabularies preserved during the migration window."""

STATUS_OK: str = "ok"
ERR_UOT_EMPTY_MASS_SOURCE: str = "ERR_UOT_EMPTY_MASS_SOURCE"
ERR_UOT_EMPTY_MASS_TARGET: str = "ERR_UOT_EMPTY_MASS_TARGET"
ERR_UOT_EMPTY_SUPPORT: str = "ERR_UOT_EMPTY_SUPPORT"
ERR_UOT_NUMERICAL: str = "ERR_UOT_NUMERICAL"

CANONICAL_UOT_STATUSES: tuple[str, ...] = (
    STATUS_OK,
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)

CANONICAL_BYPASS_REASONS: tuple[str, ...] = (
    "S0_zero",
    "S1_zero",
    "empty_support_after_prune",
    "uot_numerical_failure",
)

__all__ = [
    "CANONICAL_BYPASS_REASONS",
    "CANONICAL_UOT_STATUSES",
    "ERR_UOT_EMPTY_MASS_SOURCE",
    "ERR_UOT_EMPTY_MASS_TARGET",
    "ERR_UOT_EMPTY_SUPPORT",
    "ERR_UOT_NUMERICAL",
    "STATUS_OK",
]
