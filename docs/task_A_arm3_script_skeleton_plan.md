# Task A Arm-3 Script Skeleton Plan

## 1. Executive Summary

This document defines the script-level skeleton plan for Task A Arm-3: the density-primary coverage-reduction and UQ stress test on the frozen Stage-0 artifact. It is designed for human review before any code is written.

Arm-3 ingests the frozen Stage-0 `.h5ad`, partitions each ROI into a uniform spatial grid across its full envelope (including zero-cell blocks), builds the 100% full-coverage reference baseline from all blocks in each ROI envelope, calibrates `lambda_dens` (per family) and `tau` (per compartment) on that full-coverage reference, then runs bootstrap pseudo-ROI inference at 75 / 50 / 25% reduced-coverage levels. Outputs are parquet/csv/markdown consistent with current Task A patterns, written to the configured Arm-3 result root.

**Out of scope at this stage:**
- Implementing any script body
- Arm-IV (synthetic drift)
- Python→R bridge
- Any changes to `src/slotar/` except as explicitly delegated
- `bootstrap_single_roi` in `src/slotar/uq.py` (stub, not needed for Arm-3 v1)
- Separate full-reference technical calibration resampling layer (not in Arm-3 v1; 100% remains the full-coverage reference baseline)

---

## 2. Confirmed Local Reuse Points

| Existing path | Function | Why reusable in Arm-3 |
|---|---|---|
| `tasks/task_A/arm2/analysis_io.py` | `load_stage0_analysis_bundle(path, expected_k)` | Already handles HDF5 categorical layout directly; extracts cost_matrix, s_C, proto_ids, roi vectors, patient/compartment/cell-type tables without requiring anndata |
| `tasks/task_A/arm2_spatial_gradient.py` | `ORDERED_PAIR_SPECS`, `PAIR_FAMILIES` | Defines the full ordered pair universe used in Arm-2; Arm-3 will filter this to TC->IM and TC->PT primary anchors |
| `tasks/task_A/arm2_spatial_gradient.py` | `generate_cross_compartment_pairs(adata)` | Produces the canonical within-patient cross-compartment pair table; Arm-3 filters its output rather than defining its own pair generator |
| `src/slotar/uot.py` | `calibrate_joint_lambda(A, B, lambda_grid, kernels, cfg, target_alpha)` | Scan-and-select lambda calibration; reused for `lambda_dens` calibration on full-coverage density tensors |
| `src/slotar/uot.py` | `precompute_logKernels(C, eps_schedule, s_C)` | Kernel precomputation; same call pattern as Arm-2 |
| `src/slotar/uot.py` | `batched_uot_solve(A, B, lambda_pl, kernels, cfg, tau_external)` | Core solver; accepts per-row tau_external, needed for compartment-specific tau broadcast |
| `tasks/task_A/common.py` | `run_uot_batch_safe(A, B, lambda_pl, kernels, uot_cfg, pair_meta, tau_external)` | Wraps batched_uot_solve and attaches Task-A audit fields; use directly for Arm-3 inference once density tensors are assembled |
| `tasks/task_A/common.py` | `run_balanced_ot_batch(A, B, cost_matrix, n_min_proto)` | Same-pair shape-only forced-match comparator; Arm-3 runs this on pseudo-ROI shape vectors (derived from density) for comparator context |
| `tasks/task_A/arm2/analysis_compute.py` | `build_uot_solver_config(task_config)` | UOTSolveConfig construction from task config; reuse pattern directly |
| `tasks/task_A/pipeline.py` | `DataFrame.to_parquet(...)` pattern | Only currently implemented Task-A write path for metrics output; Arm-3 follows the same direct parquet write style |

**Not reused:**
- `tasks/task_A/common.py::assemble_tensors` — count mode only; Arm-3 needs density-mode assembly from block partitions (different logic)
- `src/slotar/uq.py::bootstrap_single_roi` — still a stub; Arm-3 implements its own task-layer block bootstrap

---

## 3. Proposed Script/Module Layout

```
tasks/task_A/
├── arm3_uq_stress.py           ← TOP-LEVEL RUNNER (needs full rewrite from stub)
└── arm3/
    ├── __init__.py
    ├── constants.py            ← Locked Arm-3 constants (grid params, coverage grid, n_reps)
    ├── block_partition.py      ← Phase 1: spatial grid construction + per-block count extraction
    ├── pseudo_roi.py           ← Phase 5: bootstrap pseudo-ROI reconstruction from blocks
    ├── calibrate.py            ← Phase 4: lambda_dens + tau calibration on full-coverage reference
    ├── inference.py            ← Phase 6: density tensor assembly + UOT + Balanced OT
    ├── retention.py            ← Phase 7: continuous retention summaries (no pass/fail)
    └── output.py               ← Phase 8: parquet/csv/markdown writers
```

### Responsibilities

**`arm3_uq_stress.py`** — Top-level runner
Orchestrates the 8-phase sequence. Accepts config dict, frozen artifact path, and result root path. Returns the full Arm-3 results DataFrame. Registered in `pipeline.py` under arm name `A3_density_uq`. Contains no scientific logic; delegates all computation to `arm3/` submodules.

**`arm3/constants.py`** — Constants and locked parameters
All Arm-3 task-fixed constants: `COVERAGE_LEVELS = (0.75, 0.50, 0.25)`, `ARM3_ANCHOR_DIRECTIONS = ("TC->IM", "TC->PT")`, `DEFAULT_BLOCK_SIZE_UNITS = 100`, `COORD_TO_MM2 = 1e-6`, `ARM3_NAME = "A3_density_uq"`. The 100% full-coverage reference baseline is built separately outside the bootstrap loop. No logic.

**`arm3/block_partition.py`** — Phase 1: Grid partition
Reads `obsm['spatial']` coordinates alongside `obs['proto_id']` and `obs['roi_id']` from the frozen Stage-0 artifact. For each ROI, defines a deterministic axis-aligned grid envelope and enumerates all grid cells in that envelope — including cells that contain zero cells. Assigns each physical cell to its grid block. Computes per-block geometric area in mm². The ROI effective area is the sum of all block geometric areas in that ROI envelope.

**`arm3/pseudo_roi.py`** — Phase 5: Bootstrap resampling
Given the frozen block-level summary for an ROI (all blocks in the ROI envelope), samples blocks with replacement until a target coverage level is reached. Reconstructs pseudo-ROI density vector and total pseudo-area from sampled blocks. Zero-cell blocks may be sampled and contribute area without adding prototype counts. Operates independently per ROI side.

**`arm3/calibrate.py`** — Phase 4: Calibration
Calibrates `lambda_dens` per pair family and `tau` per compartment (TC, IM, PT) on the full-coverage density vectors. Calls `calibrate_joint_lambda` for lambda. Provides `calibrate_tau_by_compartment` for tau. Returns frozen calibration records that are broadcast to all downstream inference.

**`arm3/inference.py`** — Phase 6: Inference
Assembles density tensors `[N, K]` from block summaries or pseudo-ROI reconstructions. Freezes semantic support masks from full-coverage pairs. Runs `run_uot_batch_safe` with density inputs, frozen lambda_pl, and per-row tau (from compartment_a). Computes primary Arm-3 metrics `U_abs_dens`, `Q_src_dens`, `Q_tgt_dens`, and scale audit fields. Runs `run_balanced_ot_batch` for shape comparator.

**`arm3/retention.py`** — Phase 7: Retention summaries
Computes absolute degradation `d(p, c, m)`, sign consistency `pi_c(m)`, and floor-dominated rate `phi_c` against the frozen full-coverage reference. Emits continuous statistics only. No pass/fail thresholding.

**`arm3/output.py`** — Writers
Writes all Arm-3 output files to the configured result root (provided as a path argument; not hard-coded in the module). Uses `DataFrame.to_parquet()` and `DataFrame.to_csv()` patterns consistent with current Task A. Writes one markdown memo.

---

## 4. ROI Grid Envelope and Block Universe Contract

This section defines the block-universe geometry contract for Arm-3 v1. It must be implemented before any coverage bootstrap logic.

**Grid type:** Axis-aligned uniform grid. Block side length = `block_size_units` coordinate units. Block geometric area = `block_size_units^2 * coord_to_mm2` mm² (same for every block).

**ROI grid envelope:** For each ROI, determine the axis-aligned bounding box of all cells assigned to that ROI from `obsm['spatial']`. Expand this bounding box to align with the fixed grid by snapping each edge outward to the nearest grid line. The result is the ROI grid envelope.

**ROI block universe:** The complete set of grid cells whose footprint falls within the ROI grid envelope. This set is determined once per ROI from the frozen Stage-0 spatial coordinates and is frozen before any bootstrap sampling.

**Zero-cell blocks:** Grid blocks within the ROI envelope that contain no physical cells are included in the ROI block universe. They contribute their geometric area to the ROI effective area and may be sampled during bootstrap. When sampled, they add area but contribute zero prototype counts to the pseudo-ROI reconstruction.

**ROI effective area:** `A_roi = sum of block_area_mm2 for all blocks in the ROI block universe`. This replaces `uns['roi_areas']` (trivial, all 1.0) and `cell_area_sum`. It is the only Arm-3 area definition.

**Grid origin:** Arm-3 v1 uses a per-ROI grid origin anchored to the floor of the ROI bounding box (snapped outward). The grid is not shared globally across ROIs. This is the simplest deterministic choice. If a global shared grid is required by a later extension, it must be introduced as a separate config option.

---

## 5. Phase-by-Phase Execution Plan

### Phase 0 — Constants and Manifest

**Purpose:** Lock all Arm-3 numerical constants before pipeline execution.

**Required inputs:** None (constants only)

**Produced outputs:** `arm3/constants.py` import surface

**Existing code to reuse:** None

**New code needed:**
- `arm3/constants.py`: all locked constants as module-level variables

---

### Phase 1 — Frozen Grid Partition

**Purpose:** Partition the frozen Stage-0 artifact into a uniform spatial grid. For each ROI, build the complete ROI block universe including zero-cell blocks. Assign each physical cell to its grid block. Compute per-block geometric area.

**Required inputs:**
- `obsm['spatial']` — shape `(1873179, 2)` per manifest; coordinates passed explicitly
- `obs/roi_id` — per-cell ROI identity (decoded strings)
- `obs/proto_id` — per-cell prototype assignments
- `block_size_units` — locked constant (default 100)
- `coord_to_mm2` — locked constant (1e-6)

**Produced outputs:**
- `block_frame: pd.DataFrame` — columns: `cell_idx`, `roi_id`, `block_id`, `proto_id`, `block_area_mm2`; one row per physical cell
- `roi_block_summary: dict[str, pd.DataFrame]` — keyed by `roi_id`; each DataFrame has one row per block in the ROI block universe (including zero-cell blocks), with columns: `block_id`, `block_area_mm2`, `count_k0..count_k{K-1}`

**Existing code to reuse:**
- `load_stage0_analysis_bundle` for loading the h5ad; Phase 1 reads `obsm['spatial']` in addition to what the existing loader already reads (the loader does not currently read spatial coordinates; this is a new read)

**New code needed:**
- `build_grid_partition(...)` in `arm3/block_partition.py`
- `compute_roi_block_summary(...)` in `arm3/block_partition.py`

---

### Phase 2 — Full-Coverage Reference from All Blocks

**Purpose:** Build full-coverage density vectors for every ROI by aggregating all blocks in the ROI block universe.

**Required inputs:** `roi_block_summary` (from Phase 1) — must include zero-cell blocks

**Produced outputs:**
- `roi_density_vectors: dict[str, np.ndarray]` — keyed by `roi_id`, each vector shape `(K,)` in cells/mm²; computed as `sum_b(n_{b,k}) / sum_b(Area_b)` over all blocks in the ROI envelope
- `roi_total_areas: dict[str, float]` — keyed by `roi_id`; total geometric area in mm² = `sum_b(Area_b)` over all blocks in the ROI envelope

**Existing code to reuse:** None (count mode uses `_build_roi_vectors`; density from block partition is new)

**New code needed:**
- `build_full_coverage_density_vectors(roi_block_summary, k_full)` in `arm3/block_partition.py`

---

### Phase 3 — Pair Universe and Anchor Subset

**Purpose:** Generate the full ordered cross-compartment pair table and filter to the primary Arm-3 anchor directions (TC->IM, TC->PT).

**Required inputs:**
- `adata` (or a lightweight struct with `obs[roi_id, compartment, patient_id]`)
- `ARM3_ANCHOR_DIRECTIONS = ("TC->IM", "TC->PT")` from constants

**Produced outputs:**
- `pair_meta_full: pd.DataFrame` — all six ordered directions (retained for audit)
- `pair_meta_anchor: pd.DataFrame` — filtered to TC->IM and TC->PT only (primary analysis)

**Existing code to reuse:**
- `generate_cross_compartment_pairs(adata)` — produces all six directions
- `ORDERED_PAIR_SPECS` — filtered by direction label

**New code needed:**
- `filter_to_anchor_directions(pair_meta, anchor_directions)` in `arm3_uq_stress.py` or `arm3/inference.py` — simple DataFrame filter, one line

---

### Phase 4 — Full-Coverage Calibration

**Purpose:** Calibrate `lambda_dens` (per unordered pair family) and `tau` (per compartment) on the full-coverage density reference. Freeze calibration outputs before any pseudo-ROI inference.

**Required inputs:**
- `roi_density_vectors` (from Phase 2)
- `pair_meta_full` (from Phase 3)
- `lambda_grid` — from task config (`arm3.lambda_grid`)
- `tau_grid` — from task config (`arm3.tau_grid`; new config key, see Open Facts)
- `uot_cfg` — from `build_uot_solver_config`
- `kernels` — from `precompute_logKernels`

**Produced outputs:**
- `frozen_lambdas: dict[str, float]` — `{"TC-IM": ..., "IM-PT": ..., "TC-PT": ...}`
- `frozen_taus: dict[str, float]` — `{"TC": tau_TC, "IM": tau_IM, "PT": tau_PT}`
- Calibration audit record (dict): records frozen calibration values + grid params; written to `arm3_calibration_record.json` at run start

**Existing code to reuse:**
- `calibrate_joint_lambda` from `src/slotar/uot.py` — for `lambda_dens` per family
- `run_uot_batch_safe` internally within tau calibration sweep

**New code needed:**
- `calibrate_lambda_dens(...)` in `arm3/calibrate.py` — thin wrapper around `calibrate_joint_lambda` using density tensors
- `calibrate_tau_by_compartment(...)` in `arm3/calibrate.py` — sweeps `tau_grid` per compartment reference pool, selects tau closest to target retention rate

---

### Phase 5 — Pseudo-ROI Bootstrap

**Purpose:** For each coverage level and each bootstrap replicate, independently resample blocks from each ROI's frozen block universe and reconstruct pseudo-ROI density vectors. Support masks are frozen before this phase.

**Required inputs:**
- `roi_block_summary` (from Phase 1) — the frozen block universe including zero-cell blocks
- `pair_meta_anchor` (from Phase 3) — only anchor-direction pairs enter the primary bootstrap
- `frozen_support_masks` — computed from full-coverage pairs before this phase (see Phase 6 pre-step)
- `coverage_levels = (0.75, 0.50, 0.25)` from constants
- `N_REPS` — number of replicates per coverage level (to be confirmed; stub uses 100)
- `rng_seed` — from config for reproducibility

**Note on ordering:** Frozen semantic support masks must be derived from full-coverage reference pairs (Phase 2 + Phase 6 setup) before any pseudo-ROI resampling begins. This sub-step is implemented as the first action inside Phase 6 before the bootstrap loop starts.

**Produced outputs:**
- Per replicate (in-memory): `(A_dens_pseudo, B_dens_pseudo, pseudo_meta_row)`
- Pseudo-ROI audit table (materialized after all replicates): written to `arm3_pseudo_roi_audit.parquet`

**Existing code to reuse:** None — entirely new block-level resampling logic

**New code needed:**
- `sample_blocks_to_coverage(block_ids, target_coverage, rng)` in `arm3/pseudo_roi.py`
- `build_pseudo_roi_density(roi_block_df, sampled_block_ids, k_full)` in `arm3/pseudo_roi.py`
- `run_bootstrap_pass(roi_block_summary, pair_meta, coverage, n_reps, k_full, rng_seed)` in `arm3/pseudo_roi.py`

---

### Phase 6 — Arm-3 Inference

**Purpose:** Run density-mode UOT (with frozen lambda_dens and compartment-specific tau) and shape-only Balanced OT on both the full-coverage reference pairs and each set of pseudo-ROI replicates. Compute primary and secondary Arm-3 metrics.

**Required inputs:**
- Full-coverage density tensors `A_dens_full, B_dens_full` (from Phase 2 via `assemble_density_tensors`)
- Pseudo-ROI density tensors per replicate (from Phase 5)
- `frozen_lambdas` (from Phase 4) — broadcast per pair family
- `frozen_taus` (from Phase 4) — assigned by `compartment_a`
- `frozen_support_masks` — computed once from full-coverage reference before bootstrap
- `kernels`, `uot_cfg`, `cost_matrix`, `s_C`

**Produced outputs:**
- `df_full_cov_results: pd.DataFrame` — full-coverage inference metrics (one row per pair)
- `df_bootstrap_results: pd.DataFrame` — all replicate inference metrics; columns include `coverage`, `replicate_id`, all primary/secondary metrics, audit fields

**Primary metrics computed per row:**
- `U_abs_dens = B_pos + D_pos` (from solver; ok rows only)
- `Q_src_dens = T / (S_src + eps)` where `S_src = sum_k(a_k^dens)` (primary; anchor-direction summaries only)
- `Q_tgt_dens = T / (S_tgt + eps)` where `S_tgt = sum_k(b_k^dens)` (mandatory audit)
- `S_src`, `S_tgt`, `Delta_scale = S_tgt - S_src`, `scale_ratio = S_tgt / (S_src + eps)` (scale audit)

**Existing code to reuse:**
- `run_uot_batch_safe` — core UOT dispatch; accepts `tau_external` per row
- `run_balanced_ot_batch` — shape-only comparator (pass shape vectors derived from density)
- `batched_uot_solve` — called with `tau_external` per-row

**New code needed:**
- `assemble_density_tensors(roi_density_vectors, pair_meta, k_full)` in `arm3/inference.py`
- `freeze_support_masks(A_full, B_full, n_min_proto, k_full)` in `arm3/inference.py`
- `apply_frozen_support(A_dens, B_dens, support_mask)` in `arm3/inference.py`
- `broadcast_frozen_tau(pair_meta, frozen_taus)` in `arm3/inference.py`
- `broadcast_frozen_lambda(pair_meta, frozen_lambdas)` in `arm3/inference.py`
- `compute_arm3_density_metrics(df_result, A_dens, B_dens)` in `arm3/inference.py`

---

### Phase 7 — Continuous Retention Summary

**Purpose:** For each monitored quantity and coverage level, compute absolute degradation, sign consistency, and floor-dominated rate against the frozen full-coverage reference. No pass/fail decisions. No tolerance-calibration layer.

**Required inputs:**
- `df_full_cov_results` (from Phase 6, full-coverage reference)
- `df_bootstrap_results` (from Phase 6, all replicates, anchor directions only for primary summary)
- Monitored quantity list: `["U_abs_dens", "Q_src_dens"]` plus optional secondary `["Q_tgt_dens", "T"]`
- Floor-dominated flag rule (task-fixed before execution; see Open Facts)

**Produced outputs:**
- `df_degradation: pd.DataFrame` — columns: `coverage`, `quantity`, `patient_id`, `median_abs_degradation`, `sign_consistency_rate`, `floor_dominated_rate`
  - Only anchor directions (TC->IM, TC->PT) are included in this summary

**Existing code to reuse:** None

**New code needed:**
- `compute_degradation_summary(df_full_cov, df_reduced, monitored_quantities)` in `arm3/retention.py`
- `compute_floor_dominated_flags(A_dens, B_dens, eta_floor, support_masks)` in `arm3/retention.py`
- `compute_sign_consistency(m_full, m_reduced_per_replicate)` in `arm3/retention.py`

---

### Phase 8 — Prototype Stability and Memo

**Purpose:** Report prototype-level stability across coverage levels and write all Arm-3 outputs to the configured result root.

**Required inputs:**
- `df_full_cov_results` (from Phase 6)
- `df_bootstrap_results` (from Phase 6)
- Frozen prototype audit set (active prototypes from full-coverage reference)
- Configured `result_root` path (from config or CLI)

**Produced outputs:**
- `arm3_prototype_stability.csv`
- `arm3_memo.md`
- All other final output files listed in Section 6

**Existing code to reuse:**
- Output writing patterns from `arm2/analysis_output.py` (markdown memo structure)

**New code needed:**
- `build_prototype_stability_table(df_full_cov, df_bootstrap, frozen_proto_audit_set)` in `arm3/output.py`
- `write_arm3_outputs(result_root, ...)` in `arm3/output.py`
- `build_arm3_memo(...)` in `arm3/output.py`

---

## 6. Proposed Function Skeletons

### `arm3/block_partition.py`

```python
def build_grid_partition(
    spatial_xy: np.ndarray,
    roi_ids: np.ndarray,
    proto_ids: np.ndarray,
    block_size_units: float,
    coord_to_mm2: float,
) -> pd.DataFrame:
    """
    Assign each physical cell to a grid block and enumerate the full ROI block universe.

    For each ROI, computes the axis-aligned bounding box from spatial_xy, snaps outward
    to the grid, enumerates all grid cells in the resulting envelope (the ROI block
    universe), then maps each physical cell to its grid block.

    Parameters
    ----------
    spatial_xy : np.ndarray, shape (N_cells, 2)
        Spatial coordinates from obsm['spatial'] in the frozen Stage-0 artifact.
        Passed explicitly; not read from adata inside this function.
    roi_ids : np.ndarray, shape (N_cells,)
        Per-cell roi_id values decoded from obs/roi_id/codes and obs/roi_id/categories.
    proto_ids : np.ndarray, shape (N_cells,)
        Per-cell prototype assignments from obs/proto_id.
    block_size_units : float
        Grid cell side length in coordinate units. Use DEFAULT_BLOCK_SIZE_UNITS.
    coord_to_mm2 : float
        Conversion factor: (coord_unit)^2 -> mm^2. Use COORD_TO_MM2 = 1e-6.

    Returns
    -------
    pd.DataFrame with columns:
        cell_idx: int         original cell index (0-based)
        roi_id: str           ROI identity
        block_id: str         encodes (roi_id, grid_col, grid_row)
        proto_id: int         prototype assignment for this cell
        block_area_mm2: float block_size_units^2 * coord_to_mm2 (constant)
    One row per physical cell. Zero-cell blocks are NOT in this per-cell frame
    but ARE present in the roi_block_summary output from compute_roi_block_summary.
    """


def compute_roi_block_summary(
    block_frame: pd.DataFrame,
    roi_block_universe: dict[str, list[str]],
    k_full: int,
    block_area_mm2: float,
) -> dict[str, pd.DataFrame]:
    """
    Build the complete per-block prototype count summary for every ROI.

    All blocks in the ROI block universe are represented, including blocks with
    zero physical cells. Zero-cell blocks have count_k* = 0 and nonzero block_area_mm2.

    Parameters
    ----------
    block_frame : pd.DataFrame
        Per-cell block assignments from build_grid_partition.
    roi_block_universe : dict[str, list[str]]
        Maps roi_id -> complete list of block_ids in that ROI's envelope.
        Produced as a side output of build_grid_partition.
    k_full : int
        Number of prototypes on the shared prototype axis (25 for Task A).
    block_area_mm2 : float
        Geometric area per block in mm^2 (constant for uniform grid).

    Returns
    -------
    dict keyed by roi_id (str).
    Each value is a DataFrame with one row per block in the ROI block universe:
        block_id: str
        block_area_mm2: float  (same value for all blocks)
        count_k0: float, count_k1: float, ..., count_k{K-1}: float
    Zero-cell blocks are present with all count_k* = 0.0.
    """


def build_full_coverage_density_vectors(
    roi_block_summary: dict[str, pd.DataFrame],
    k_full: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Build full-coverage density vectors from the complete ROI block universe.

    Area denominator uses ALL blocks in the ROI block universe, including zero-cell blocks.
    This ensures density area = true geometric footprint of the ROI envelope.

    Parameters
    ----------
    roi_block_summary : dict[str, pd.DataFrame]
        Output of compute_roi_block_summary. Must include zero-cell blocks.
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    roi_density_vectors : dict[str, np.ndarray]
        Keyed by roi_id. Shape (K,): sum_b(n_{b,k}) / sum_b(Area_b).
        Units: cells / mm^2.
    roi_total_areas : dict[str, float]
        Keyed by roi_id. sum_b(Area_b) over all blocks in ROI envelope (mm^2).
    """
```

### `arm3/pseudo_roi.py`

```python
def sample_blocks_to_coverage(
    available_block_ids: list[str],
    target_coverage: float,
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample block IDs with replacement to reach a target coverage count.

    n_sampled = max(1, floor(target_coverage * len(available_block_ids))).
    available_block_ids is the complete ROI block universe, including zero-cell blocks.

    Parameters
    ----------
    available_block_ids : list[str]
        Complete ROI block universe (frozen). Includes zero-cell blocks.
    target_coverage : float
        Fraction of total blocks to sample, e.g. 0.75 for 75%.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list[str]: sampled block IDs (may contain repeats).
    """


def build_pseudo_roi_density(
    roi_block_df: pd.DataFrame,
    sampled_block_ids: list[str],
    k_full: int,
) -> tuple[np.ndarray, float]:
    """
    Reconstruct pseudo-ROI density vector from sampled blocks (with repetition).

    Zero-cell blocks in sampled_block_ids contribute area but no prototype counts.

    Parameters
    ----------
    roi_block_df : pd.DataFrame
        One entry from roi_block_summary; includes zero-cell blocks.
    sampled_block_ids : list[str]
        Block IDs selected by sample_blocks_to_coverage (with repetition).
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    density_vector : np.ndarray, shape (K,)
        n_k^pseudo / A^pseudo for each prototype k.
        n_k^pseudo = sum over sampled blocks of count_k (with repetition).
        A^pseudo = sum over sampled blocks of block_area_mm2 (with repetition).
    total_pseudo_area : float
        A^pseudo in mm^2.
    """


def run_bootstrap_pass(
    roi_block_summary: dict[str, pd.DataFrame],
    pair_meta: pd.DataFrame,
    coverage: float,
    n_reps: int,
    k_full: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run one full bootstrap pass for a given coverage level.

    Side A and side B resample independently from their own frozen block universes.

    Parameters
    ----------
    roi_block_summary : dict[str, pd.DataFrame]
        Frozen block partition including zero-cell blocks.
    pair_meta : pd.DataFrame
        Ordered pair metadata: roi_a, roi_b, patient_id, pair_family, compartment_a, etc.
    coverage : float
        Target coverage fraction (0.0 to 1.0).
    n_reps : int
        Number of bootstrap replicates.
    k_full : int
        Prototype axis dimension.
    rng_seed : int
        Base seed; replicate i uses rng_seed + i for reproducibility.

    Returns
    -------
    A_reps : np.ndarray, shape (n_reps, N_pairs, K)
    B_reps : np.ndarray, shape (n_reps, N_pairs, K)
    pseudo_meta : pd.DataFrame
        One row per (replicate_id, pair_id):
        coverage, pseudo_area_a_mm2, pseudo_area_b_mm2,
        n_blocks_sampled_a, n_blocks_sampled_b.
    """
```

### `arm3/calibrate.py`

```python
def calibrate_lambda_dens(
    roi_density_vectors: dict[str, np.ndarray],
    pair_meta: pd.DataFrame,
    k_full: int,
    lambda_grid: tuple[float, ...],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    target_alpha: float = 0.05,
) -> dict[str, float]:
    """
    Calibrate lambda_dens once per pair family on full-coverage density tensors.

    Follows the same calibrate_joint_lambda scan-and-select pattern as Arm-2,
    applied to density-mode tensors. Calibration pools both ordered directions
    within each unordered family (same pooling strategy as Arm-2).

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Full-coverage density vectors per roi_id (cells/mm^2).
    pair_meta : pd.DataFrame
        Full ordered pair metadata (all six directions used for pooled calibration).
    k_full : int
        Prototype axis dimension.
    lambda_grid : tuple[float, ...]
        Candidates to scan. From config key 'arm3.lambda_grid'.
    uot_cfg : UOTSolveConfig
        Frozen solver config.
    kernels : Sequence[np.ndarray]
        Pre-computed log-kernels.
    target_alpha : float
        Target unmatched fraction.

    Returns
    -------
    dict[str, float]: {"TC-IM": lambda_dens_TCIM, "IM-PT": ..., "TC-PT": ...}
    """


def calibrate_tau_by_compartment(
    roi_density_vectors: dict[str, np.ndarray],
    roi_compartment_map: dict[str, str],
    k_full: int,
    tau_grid: tuple[float, ...],
    frozen_lambdas: dict[str, float],
    uot_cfg: UOTSolveConfig,
    kernels: Sequence[np.ndarray],
    target_retention: float,
) -> dict[str, float]:
    """
    Calibrate compartment-specific tau values on within-compartment full-coverage
    reference pools.

    For each compartment c in {TC, IM, PT}:
    - Assemble same-patient, within-compartment ROI pairs as the reference pool.
    - Scan tau_grid; for each candidate, run batched_uot_solve with tau_external.
    - Select tau_c closest to target_retention on the reference pool.

    tau is assigned by compartment_a in all downstream inference.
    No family-level tau pooling.

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Full-coverage density vectors.
    roi_compartment_map : dict[str, str]
        Maps roi_id -> compartment label. Built from obs/compartment in Stage-0.
    k_full : int
        Prototype axis dimension.
    tau_grid : tuple[float, ...]
        Candidates. From config key 'arm3.tau_grid'.
    frozen_lambdas : dict[str, float]
        Used to assign lambda to within-compartment reference pairs.
    uot_cfg : UOTSolveConfig
        Frozen solver config.
    kernels : Sequence[np.ndarray]
        Pre-computed log-kernels.
    target_retention : float
        Target retained-transport fraction. From config key 'arm3.target_retention'.

    Returns
    -------
    dict[str, float]: {"TC": tau_TC, "IM": tau_IM, "PT": tau_PT}
    """
```

### `arm3/inference.py`

```python
def assemble_density_tensors(
    roi_density_vectors: dict[str, np.ndarray],
    pair_meta: pd.DataFrame,
    k_full: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble density-mode [N, K] tensors aligned to pair_meta row order.

    Parameters
    ----------
    roi_density_vectors : dict[str, np.ndarray]
        Maps roi_id -> density vector (K,) in cells/mm^2.
        May be full-coverage or pseudo-ROI reconstruction.
    pair_meta : pd.DataFrame
        Ordered pair table; must contain roi_a, roi_b columns.
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    A : np.ndarray, shape (N, K)
    B : np.ndarray, shape (N, K)
    """


def freeze_support_masks(
    A_full: np.ndarray,
    B_full: np.ndarray,
    n_min_proto: float,
    k_full: int,
) -> np.ndarray:
    """
    Compute frozen semantic support masks from full-coverage reference tensors.

    K_r^100 = {k : A_full[r,k] + B_full[r,k] >= n_min_proto}.
    This mask does not shrink with coverage reduction.

    Parameters
    ----------
    A_full : np.ndarray, shape (N, K)
    B_full : np.ndarray, shape (N, K)
    n_min_proto : float
        Minimum combined mass to enter the semantic support.
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    support_masks : np.ndarray, shape (N, K), dtype bool
    """


def broadcast_frozen_tau(
    pair_meta: pd.DataFrame,
    frozen_taus: dict[str, float],
) -> np.ndarray:
    """
    Assign per-row tau from frozen_taus by compartment_a.

    Parameters
    ----------
    pair_meta : pd.DataFrame
        Must contain 'compartment_a' column with values in {'TC', 'IM', 'PT'}.
    frozen_taus : dict[str, float]
        Maps compartment -> tau scalar.

    Returns
    -------
    np.ndarray, shape (N,), float: per-row tau_external values.
    """


def broadcast_frozen_lambda(
    pair_meta: pd.DataFrame,
    frozen_lambdas: dict[str, float],
) -> np.ndarray:
    """
    Assign per-row lambda_pl from frozen_lambdas by pair_family.

    Parameters
    ----------
    pair_meta : pd.DataFrame
        Must contain 'pair_family' column.
    frozen_lambdas : dict[str, float]
        Maps pair_family -> lambda_dens scalar.

    Returns
    -------
    np.ndarray, shape (N,), float: per-row lambda_pl values.
    """


def compute_arm3_density_metrics(
    df_result: pd.DataFrame,
    A_dens: np.ndarray,
    B_dens: np.ndarray,
) -> pd.DataFrame:
    """
    Append primary and secondary Arm-3 density metrics to UOT result DataFrame.

    Input df_result must already contain T, B_pos, D_pos from run_uot_batch_safe.

    Appended columns:
    - U_abs_dens = B_pos + D_pos  (ok rows only)
    - S_src = sum_k(a_k^dens)
    - S_tgt = sum_k(b_k^dens)
    - Delta_scale = S_tgt - S_src
    - scale_ratio = S_tgt / (S_src + eps)
    - Q_src_dens = T / (S_src + eps)   [primary; anchor-direction summaries only]
    - Q_tgt_dens = T / (S_tgt + eps)   [mandatory audit]

    T from the solver is left as-is (secondary/audit context only).
    T / (T + B_pos + D_pos + eps) MUST NOT be added.
    """
```

### `arm3/retention.py`

```python
def compute_degradation_summary(
    df_full_cov: pd.DataFrame,
    df_reduced: pd.DataFrame,
    monitored_quantities: list[str],
    pair_id_col: str = "pair_id",
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Compute per-patient continuous degradation statistics at a given coverage level.

    For each patient p and monitored quantity m:
        d(p, c, m) = |median_replicate(m, p, c) - m_full_cov(p)|

    Both df_full_cov and df_reduced must already be filtered to anchor directions
    before calling this function (TC->IM and TC->PT only).

    Parameters
    ----------
    df_full_cov : pd.DataFrame
        Full-coverage reference results (one row per pair).
    df_reduced : pd.DataFrame
        Bootstrap results at one coverage level. Must contain replicate_id column.
    monitored_quantities : list[str]
        Minimum: ['U_abs_dens', 'Q_src_dens'].
    pair_id_col : str
    patient_id_col : str

    Returns
    -------
    pd.DataFrame columns:
        patient_id, quantity, coverage,
        median_abs_degradation, sign_consistency_rate, floor_dominated_rate,
        mean_replicate_value, std_replicate_value
    No pass/fail columns. No boolean retention flags.
    """


def compute_floor_dominated_flags(
    A_dens: np.ndarray,
    B_dens: np.ndarray,
    eta_floor: float,
    support_masks: np.ndarray,
) -> np.ndarray:
    """
    Flag pseudo-ROI replicates where eta_floor padding dominates meaningful mass.

    The exact criterion MUST be supplied as a task-fixed rule before this function
    is implemented. See Open Implementation Facts.

    Parameters
    ----------
    A_dens : np.ndarray, shape (N, K)
    B_dens : np.ndarray, shape (N, K)
    eta_floor : float
        Numerical floor from UOTSolveConfig.
    support_masks : np.ndarray, shape (N, K), bool
        Frozen semantic support mask.

    Returns
    -------
    np.ndarray, shape (N,), bool: True where replicate is floor-dominated.
    """
```

---

## 7. File I/O Contract

**Intermediate outputs (in-memory):**

| Object | Type | Description |
|---|---|---|
| `block_frame` | `pd.DataFrame` | Per-physical-cell block assignments |
| `roi_block_universe` | `dict[str, list[str]]` | Complete block ID list per ROI (including zero-cell blocks) |
| `roi_block_summary` | `dict[str, pd.DataFrame]` | Per-block count tables per ROI (including zero-cell blocks) |
| `roi_density_vectors` | `dict[str, np.ndarray]` | Full-coverage density vectors |
| `roi_total_areas` | `dict[str, float]` | Full-coverage total geometric areas |
| `pair_meta_full` | `pd.DataFrame` | All six ordered directions |
| `pair_meta_anchor` | `pd.DataFrame` | TC->IM and TC->PT only |
| `frozen_lambdas` | `dict[str, float]` | Family-level lambda_dens |
| `frozen_taus` | `dict[str, float]` | Compartment-level tau |
| `frozen_support_masks` | `np.ndarray (N, K)` | Semantic support masks per reference pair |
| `df_full_cov_results` | `pd.DataFrame` | Full-coverage UOT + BOT metrics |
| `A_reps, B_reps` | `np.ndarray (n_reps, N, K)` | Bootstrap pseudo-ROI density tensors |

**Final outputs (written to the configured Arm-3 result root):**

Result root is provided via config (`arm3.result_root`) or a CLI argument. It is not hard-coded in any module. The conventional external Task A location is `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_density_uq/` but scripts must not depend on this literal path.

| Filename | Format | Description |
|---|---|---|
| `arm3_calibration_record.json` | json | Frozen lambda_dens, tau values, grid params, n_reps, coverage_levels, rng_seed |
| `arm3_full_coverage_reference.parquet` | parquet | Full-coverage UOT + BOT results for all pairs |
| `arm3_pseudo_roi_audit.parquet` | parquet | Per-replicate pseudo-ROI audit: pair_id, coverage, replicate_id, pseudo_area_a/b, n_blocks_a/b |
| `arm3_bootstrap_results.parquet` | parquet | All bootstrap UOT + BOT results with coverage, replicate_id, all metrics |
| `arm3_density_family_direction_summary.csv` | csv | All-direction summaries of S_src, S_tgt, Delta_scale (including audit directions) |
| `arm3_baseline_scale_audit.csv` | csv | Per-pair, per-coverage scale audit: S_src, S_tgt, scale_ratio |
| `arm3_degradation_summary.csv` | csv | Anchor-direction continuous retention: patient × quantity × coverage degradation + sign consistency + floor rate |
| `arm3_prototype_stability.csv` | csv | Per-prototype recurrence proportion, sign consistency, correlation to full-coverage reference |
| `arm3_balanced_ot_comparator.csv` | csv | Balanced OT comparator summary by family/direction/coverage |
| `arm3_memo.md` | markdown | Arm-3 memo: retained findings, weakened findings, unresolved findings |

---

## 8. Data Contract Alignment Notes

**Density semantics:**
All `A` and `B` tensors use cells/mm² throughout. Full-coverage density = `sum_b(n_{b,k}) / sum_b(Area_b)` over all blocks in the ROI envelope. Pseudo-ROI density = `sum_sampled_b(n_{b,k}) / sum_sampled_b(Area_b)` over sampled blocks (with repetition). Count-mode tensors are audit-only; the UOT solver receives density tensors.

**Geometric area semantics:**
Block area = `block_size_units^2 * coord_to_mm2`. ROI effective area = sum of ALL block geometric areas in the ROI block universe, including zero-cell blocks. `cell_area_sum` is not used. `uns['roi_areas']` trivial values (all 1.0) are ignored.

**Frozen block bootstrap:**
The ROI block universe (including zero-cell blocks) is frozen from the Stage-0 artifact before any coverage reduction. Side A and B resample independently from their own frozen block universes. Support masks are computed from full-coverage pairs once before the bootstrap loop and do not change with coverage. `eta_floor` applies only within the frozen support; it does not reactivate zero-support prototypes.

**`compartment_a` tau assignment:**
Per-row tau is assigned exclusively by `compartment_a` via `broadcast_frozen_tau`. Tau values are `tau_TC`, `tau_IM`, `tau_PT` only. No mixed family-level pooling (e.g., no single tau for TC-IM built from both compartments). Side A is the reference side.

**Anchor-direction-only `Q_src` in primary summaries:**
`Q_src_dens` is computed for all rows. Primary retention summaries (`arm3_degradation_summary.csv`) are restricted to `pair_type IN ("TC->IM", "TC->PT")`. Reverse and audit directions appear only in `arm3_density_family_direction_summary.csv`. No unordered family median collapsing.

---

## 9. Open Implementation Facts Still Needing Human Confirmation

**1. Spatial coordinate unit interpretation (CRITICAL)**
`coord_to_mm2 = 1e-6` implies 1 coord unit = 1 µm (block area = 0.01 mm² at 100×100). This gives ~100 blocks per nominal 1 mm² ROI. This interpretation has NOT been verified against the actual `obsm['spatial']` coordinate range in the frozen `.h5ad`. A human must confirm the coordinate range (e.g., ~0–1000 per ROI) before block partition parameters are locked.

**2. Tau calibration reference pool structure**
The spec specifies "full-coverage original ROI compartment reference pool" but does not define the exact pairing structure. Does the TC reference pool consist of all within-patient, within-TC-compartment ROI pairs (all distinct ordered pairs of TC ROIs for the same patient)? Confirm before `calibrate_tau_by_compartment` is implemented.

**3. N_REPS (bootstrap replicate count)**
The stub hard-codes 100. The spec does not lock this number. Confirm: is 100 the approved count? Should it live in `constants.py` or in the task config YAML?

**4. Coverage sampling rule**
`n_sampled = max(1, floor(target_coverage * n_total_blocks))` is the proposed rule (equivalent to area-based coverage on a uniform grid). Confirm this floor-and-max rule is intended.

**5. Floor-dominated replicate flag rule**
Must be task-fixed before implementation. No current definition exists. Humans must supply a concrete criterion (e.g., ratio of eta_floor mass to total support mass exceeding a threshold) before `compute_floor_dominated_flags` can be written.

**6. Tau calibration target statistic and config keys**
`calibrate_tau_by_compartment` requires a `target_retention` value and a definition of how to measure retention from solver output (R is tau-dependent and circular in calibration; a proxy or indirect measure is needed). Also, the config keys `arm3.tau_grid` and `arm3.target_retention` do not yet exist in the task config YAML. Both must be specified before implementation.

---

## 10. Recommended Implementation Order

1. **`arm3/constants.py`** — First. No dependencies.

2. **`arm3/block_partition.py`** — Second. Critical path dependency for everything downstream. Validate coordinate unit interpretation (Open Fact 1) before coding block_size_units.

3. **`arm3/pseudo_roi.py`** — Third. Depends only on block_partition output. Can be unit-tested with a mock block summary including zero-cell blocks.

4. **`arm3/calibrate.py`** — Fourth. Requires resolved Open Facts 2 (tau reference pool), 6 (tau calibration target + config keys). Calls `calibrate_joint_lambda` which is already tested.

5. **`arm3/inference.py`** — Fifth. Depends on block_partition and calibrate. `assemble_density_tensors` and `freeze_support_masks` can be unit-tested before full pipeline runs.

6. **`arm3/retention.py`** — Sixth. Requires resolved Open Fact 5 (floor-dominated flag rule).

7. **`arm3/output.py`** — Seventh. Depends on computed results. Accept result_root as a path argument; do not hard-code paths.

8. **`arm3_uq_stress.py` (rewrite)** — Last. Orchestrates all above in phase order. Wire config keys including result_root. Register as `A3_density_uq` in pipeline.py.

9. **`arm3/__init__.py`** — Minimal; expose top-level run entry point only.

---

## 11. "Do Not Do" List for Implementation

- **Do NOT use `cell_area_sum` as density area.** Only geometric grid block areas.
- **Do NOT use `uns['roi_areas']` (all 1.0) as effective area.** Ignored in Arm-3.
- **Do NOT limit the ROI block universe to blocks that contain at least one cell.** Zero-cell blocks must be present in the block summary and included in ROI effective area.
- **Do NOT filter blocks by cell count.** No N_MIN_BLOCK filter anywhere in block construction. Semantic pruning is handled by support masks at the solver level.
- **Do NOT calibrate `lambda_dens` or `tau` on pseudo-ROIs.** Calibration is full-coverage only. Frozen values are broadcast to all downstream inference.
- **Do NOT compute `Q_src_dens` as `T / (T + B_pos + D_pos + eps)`.** This ratio is explicitly forbidden as an Arm-3 transportability endpoint.
- **Do NOT collapse TC->IM and TC->PT into unordered family medians for primary summaries.** Keep anchor directions separate.
- **Do NOT include reverse directions (IM->TC, PT->TC) in the primary degradation summary.** Reserve for audit outputs.
- **Do NOT add pass/fail boolean threshold logic anywhere in the pipeline.** Phase 7 outputs continuous statistics only.
- **Do NOT add a separate full-reference technical calibration resampling layer in Arm-3 v1.** The 100% full-coverage reference baseline is not a bootstrap level.
- **Do NOT hard-code the result output directory inside any module.** Result root is a config/CLI-provided argument.
- **Do NOT move Arm-3 orchestration or task-specific calibration into `src/slotar/`.** All Arm-3 logic stays in `tasks/task_A/arm3/`.
- **Do NOT call `assemble_tensors` from `common.py` for density mode.** That function raises on non-count input. Use `assemble_density_tensors` from `arm3/inference.py`.
- **Do NOT mix the 200×200 grid into the main implementation skeleton.** If needed, it is a later sensitivity rerun isolated by a config parameter.
- **Do NOT allow `eta_floor` to activate prototypes outside the frozen `K_r^100` support.** Floor padding is numerical stabilization only; it must not reactivate zero-support prototypes.
