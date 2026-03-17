# Task A Arm-3 Phase 0–3 Implementation Note

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

**Date:** 2026-03-15
**Branch:** chore/slotar-library-tasks
**Status:** Phase 0–3 implemented; Phase 4+ remains NotImplementedError.

---

## 1. Files Edited

| File | What changed |
|------|-------------|
| `tasks/task_A/arm3/constants.py` | Added `K_FULL = 25` (explicit prototype-axis constant); current local reduced-coverage constant now uses `(0.75, 0.50, 0.25)` while `100%` remains the separate reference baseline |
| `tasks/task_A/arm3/block_partition.py` | All 3 functions fully implemented (was all `NotImplementedError`) |
| `tasks/task_A/arm3_uq_stress.py` | Rewrote as Phase 0–3 runner; changed `run_arm3` signature; added `_load_arm3_stage0_inputs`, `_decode_h5_strings`, `_build_pair_universe`, `_write_phase0_3_outputs` |

No other files were edited. `calibrate.py`, `inference.py`, `retention.py`, `output.py`, `pseudo_roi.py`, `arm3/__init__.py`, `pipeline.py`, `src/slotar/`, and all spec/result docs are unchanged.

---

## 2. What Phase 0–3 Now Actually Implements

### Phase 0 — Constants and manifest
- Validates `config['data']['k_full']` against locked `K_FULL = 25`.
- Resolves `block_size_units` from `config['arm3']['block_size_units']` or falls back to `DEFAULT_BLOCK_SIZE_UNITS = 100.0`.
- Writes `arm3_phase0_manifest.json` with all run parameters.

### Phase 1 — Frozen grid partition
- Loads from the Stage-0 `.h5ad` via **direct h5py read** (no full AnnData load):
  - `obsm/spatial` → `spatial_xy` (tries `obsm/X_spatial` as fallback)
  - `obs/proto_id` → `proto_ids`
  - `obs/roi_id/{codes,categories}` → decoded per-cell `roi_ids`
  - `obs/patient_id/{codes,categories}`, `obs/compartment/{codes,categories}` → per-cell decoded arrays for roi_table construction
  - Validates `uns/cost_matrix` shape against `k_full`
- `build_grid_partition`: strict floor/ceil envelope snapping, column-major block ID encoding `"{roi_id}::{col}::{row}"`, clips at boundary.
- `compute_roi_block_summary`: initialises full-universe count array with zeros before scatter via `np.add.at`; zero-cell blocks guaranteed to survive.
- Writes `arm3_phase1_block_frame.parquet`, `arm3_phase1_roi_block_universe.json`, `arm3_phase1_roi_block_summary.parquet` (stacked long format with explicit `roi_id` column).

### Phase 2 — Full-coverage density reference
- `build_full_coverage_density_vectors`: sums counts across all blocks (including zero-cell), divides by total geometric block area.
- Validates total area > 0 for each ROI.
- Writes `arm3_phase2_roi_density_reference.parquet` with columns `roi_id`, `total_area_mm2`, `density_k0` … `density_k24`.

### Phase 3 — Pair universe and anchor subset
- Reuses `ORDERED_PAIR_SPECS` from `arm2_spatial_gradient.py` directly.
- Pair generation follows the same Cartesian product logic as `generate_cross_compartment_pairs` but operates on the HDF5-derived `roi_table` (no AnnData required).
- `pair_id` format: `"{ARM3_NAME}::{pair_type}::{patient_id}::{roi_a}::{roi_b}"`.
- Anchor subset filtered by `pair_type in ARM3_ANCHOR_DIRECTIONS` = `("TC->IM", "TC->PT")`.
- Validates that all ROIs referenced in pair_meta exist in `roi_density_vectors`.
- Writes `arm3_phase3_pair_meta_full.parquet` and `arm3_phase3_pair_meta_anchor.parquet`.

### Intentional stop
After writing Phase 0–3 outputs the runner raises:
```
NotImplementedError("Arm-3 Phase 4+ (calibration, bootstrap, inference, retention, outputs) not implemented in this tranche.")
```

---

## 3. Output Files Written by the Phase 0–3 Runner

| File | Content |
|------|---------|
| `arm3_phase0_manifest.json` | Run parameters: ARM3_NAME, K_FULL, block_size_units, COORD_TO_MM2, block_area_mm2, anchor_directions, spatial range, pair counts, run timestamp, phase flags |
| `arm3_phase1_block_frame.parquet` | Per-cell: cell_idx, roi_id, block_id, proto_id, block_area_mm2 |
| `arm3_phase1_roi_block_universe.json` | Per-ROI complete block ID list (includes zero-cell blocks) |
| `arm3_phase1_roi_block_summary.parquet` | Stacked long format: roi_id, block_id, block_area_mm2, count_k0..count_k24 |
| `arm3_phase2_roi_density_reference.parquet` | Per-ROI: roi_id, total_area_mm2, density_k0..density_k24 |
| `arm3_phase3_pair_meta_full.parquet` | All six ordered directions: pair_id, patient_id, compartment_a/b, pair_type, pair_family, roi_a, roi_b |
| `arm3_phase3_pair_meta_anchor.parquet` | Anchor subset (TC->IM, TC->PT only) |

---

## 4. Signature Changes from Prior Skeleton

### `run_arm3` signature simplified
- Skeleton: `run_arm3(stage0_path, config, uot_cfg, kernels, result_root)`
- Implementation: `run_arm3(stage0_path, config, result_root)`
- **Reason:** `uot_cfg` and `kernels` are only needed from Phase 4 onwards. Requiring them as Phase 0–3 inputs would force callers to instantiate solver infrastructure before any geometric validation is possible. They will be re-added to the signature when Phase 4 is implemented.

### `build_grid_partition` — no change from prior skeleton
The skeleton already documented the `tuple[pd.DataFrame, dict[str, list[str]]]` return type. No further change.

### `_load_arm3_stage0_inputs` — new private helper
Encapsulates the h5py read for Phase 0–3. Not part of the original skeleton (which only had `_load_stage0_bundle` delegating to Arm-2's loader). Arm-2's loader does not expose `obsm/spatial`. A separate minimal HDF5 read is therefore required; Arm-2's loader is not modified.

---

## 5. Remaining Blockers Before Phase 4

| # | Blocker | Blocks |
|---|---------|--------|
| 1 | **Spatial coordinate unit verification** — Need to confirm `obsm['spatial']` values are in µm before `block_size_units=100` is geophysically meaningful | Phase 1 interpretation |
| 2 | **Within-compartment tau reference pool structure** | `calibrate_tau_by_compartment` |
| 3 | **Config keys `arm3.tau_grid` and `arm3.target_retention`** — Not yet present in any config YAML | `calibrate_tau_by_compartment` |
| 4 | **N_REPS confirmation** — `DEFAULT_N_REPS=100` is a placeholder | `run_bootstrap_pass` |
| 5 | **Coverage sampling rule** — `max(1, floor(coverage × n_blocks))` must be confirmed | `sample_blocks_to_coverage` |
| 6 | **Floor-dominated criterion** — No definition; blocks `compute_floor_dominated_flags` | `retention.py` |
| 7 | **Zero-sign tie-breaking rule** — Must be task-fixed for `compute_degradation_summary` | `retention.py` |
| 8 | **Bootstrap side A/B seeding strategy** — Per-replicate offset rule for independence | `run_bootstrap_pass` |

---

## 6. Confirmations

- **Phase 4+ remains unimplemented.** `calibrate.py`, `inference.py`, `retention.py`, `output.py`, and `pseudo_roi.py` all raise `NotImplementedError` in every function body. The Phase 0–3 runner raises `NotImplementedError` at the Phase 4 boundary explicitly.
- **No scientific spec files were changed.** `docs/task_A_spec.md` and all results files are untouched.
- **`src/slotar/` was untouched.** No files under `src/slotar/` were read or modified.
- **`pipeline.py` was untouched.** `SUPPORTED_ARM_MODULES` is unchanged; Arm-3 is not registered.
- **Current local coverage ladder:** `100%` remains the frozen full-coverage reference baseline; reduced bootstrap levels are `75% / 50% / 25%` only.
- **Zero-cell blocks are preserved.** `compute_roi_block_summary` uses `np.add.at` scatter from the full block universe initialised with zeros, guaranteeing zero-cell blocks survive aggregation. The runner validates this explicitly before writing outputs.
- **No absolute paths are hard-coded in modules.** `result_root` is always caller-provided.
