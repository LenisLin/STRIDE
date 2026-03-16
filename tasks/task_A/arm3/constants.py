"""
Module: tasks.task_A.arm3.constants

Locked numerical constants for Task A Arm-3: density-primary coverage-reduction
and UQ stress test.

All constants in this module are either locked by design specification or are
clearly labelled as placeholders pending human confirmation. No runtime logic.
"""

# ---------------------------------------------------------------------------
# Arm identity
# ---------------------------------------------------------------------------

ARM3_NAME: str = "A3_uq_stress"

# ---------------------------------------------------------------------------
# Shared prototype axis (locked; matches frozen Stage-0 k=25 artifact)
# ---------------------------------------------------------------------------

K_FULL: int = 25  # number of prototypes on the shared prototype axis

# ---------------------------------------------------------------------------
# Coverage grid (locked)
# Arm-3 bootstrap reduced-coverage levels: 75%, 50%, 25%.
# The 100% full-coverage reference baseline is built separately and must NOT
# be added to this bootstrap loop constant.
# ---------------------------------------------------------------------------

COVERAGE_LEVELS: tuple[float, ...] = (0.75, 0.50, 0.25)

# ---------------------------------------------------------------------------
# Anchor directions (locked)
# Primary Arm-3 relative transportability analysis uses only these two ordered
# confirmatory anchor directions. Reverse and exploratory directions are
# audit-only and must not enter primary degradation summaries.
# ---------------------------------------------------------------------------

ARM3_ANCHOR_DIRECTIONS: tuple[str, str] = ("TC->IM", "TC->PT")

# ---------------------------------------------------------------------------
# Spatial grid (locked)
# Main run uses a 100 x 100 block size in coordinate units.
# 200 x 200 is explicitly out of scope for Arm-3 v1.
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE_UNITS: float = 100.0  # coordinate units per block side

# ---------------------------------------------------------------------------
# Coordinate-to-area conversion (locked)
# COORD_TO_MM2 = 1e-6 implies 1 coordinate unit = 1 µm.
# Block geometric area = DEFAULT_BLOCK_SIZE_UNITS^2 * COORD_TO_MM2 mm^2.
# Do NOT substitute uns['roi_areas'] or cell_area_sum for density area.
# ---------------------------------------------------------------------------

COORD_TO_MM2: float = 1e-6  # (coord_unit)^2 -> mm^2; 1 coord unit = 1 µm (locked)

# ---------------------------------------------------------------------------
# Bootstrap parameters
# DEFAULT_N_REPS: placeholder — confirm exact value before implementation.
# DEFAULT_RNG_SEED: placeholder — confirm before implementation.
# ---------------------------------------------------------------------------

DEFAULT_N_REPS: int = 100       # PLACEHOLDER: awaiting human confirmation
DEFAULT_RNG_SEED: int = 42      # PLACEHOLDER: awaiting human confirmation

# ---------------------------------------------------------------------------
# Pair families (reused from Arm-2 definition)
# ---------------------------------------------------------------------------

ARM3_PAIR_FAMILIES: tuple[str, ...] = ("TC-IM", "IM-PT", "TC-PT")

# ---------------------------------------------------------------------------
# Numerical epsilon used in Q_src_dens / Q_tgt_dens denominators.
# Keep consistent with solver-level floor choices.
# ---------------------------------------------------------------------------

DENSITY_EPS: float = 1e-12  # small stabilisation term only; not a semantic floor
