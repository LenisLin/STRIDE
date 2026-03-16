# Task A Arm-3 Skeleton Generation Note

**Date:** 2026-03-15
**Branch:** chore/slotar-library-tasks
**Status:** Skeleton generation complete; awaiting human review before implementation.

---

## 1. Files Created / Rewritten

| File | Action | Notes |
|------|--------|-------|
| `tasks/task_A/arm3/__init__.py` | Created | Minimal export surface; re-exports constants only |
| `tasks/task_A/arm3/constants.py` | Created | Locked constants + clearly labelled placeholders |
| `tasks/task_A/arm3/block_partition.py` | Created | 3 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3/pseudo_roi.py` | Created | 3 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3/calibrate.py` | Created | 2 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3/inference.py` | Created | 5 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3/retention.py` | Created | 2 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3/output.py` | Created | 3 public functions; all `NotImplementedError` bodies |
| `tasks/task_A/arm3_uq_stress.py` | Rewritten | Orchestration skeleton; 1 public + 4 private helpers; all `NotImplementedError` bodies |

No files were created beyond the 9 listed above. No additional modules were added silently.

---

## 2. Public Functions Added Per File

### `arm3/constants.py`
- No functions; constants only.
- `ARM3_NAME`, `COVERAGE_LEVELS`, `ARM3_ANCHOR_DIRECTIONS`, `DEFAULT_BLOCK_SIZE_UNITS`, `COORD_TO_MM2`, `DEFAULT_N_REPS` (placeholder), `DEFAULT_RNG_SEED` (placeholder), `ARM3_PAIR_FAMILIES`, `DENSITY_EPS`.

### `arm3/block_partition.py`
- `build_grid_partition(spatial_xy, roi_ids, proto_ids, block_size_units, coord_to_mm2) -> tuple[pd.DataFrame, dict[str, list[str]]]`
- `compute_roi_block_summary(block_frame, roi_block_universe, k_full, block_area_mm2) -> dict[str, pd.DataFrame]`
- `build_full_coverage_density_vectors(roi_block_summary, k_full) -> tuple[dict[str, np.ndarray], dict[str, float]]`

### `arm3/pseudo_roi.py`
- `sample_blocks_to_coverage(available_block_ids, target_coverage, rng) -> list[str]`
- `build_pseudo_roi_density(roi_block_df, sampled_block_ids, k_full) -> tuple[np.ndarray, float]`
- `run_bootstrap_pass(roi_block_summary, pair_meta, coverage, n_reps, k_full, rng_seed) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]`

### `arm3/calibrate.py`
- `calibrate_lambda_dens(roi_density_vectors, pair_meta, k_full, lambda_grid, uot_cfg, kernels, target_alpha) -> dict[str, float]`
- `calibrate_tau_by_compartment(roi_density_vectors, roi_compartment_map, k_full, tau_grid, frozen_lambdas, uot_cfg, kernels, target_retention) -> dict[str, float]`

### `arm3/inference.py`
- `assemble_density_tensors(roi_density_vectors, pair_meta, k_full) -> tuple[np.ndarray, np.ndarray]`
- `freeze_support_masks(A_full, B_full, n_min_proto, k_full) -> np.ndarray`
- `broadcast_frozen_tau(pair_meta, frozen_taus) -> np.ndarray`
- `broadcast_frozen_lambda(pair_meta, frozen_lambdas) -> np.ndarray`
- `compute_arm3_density_metrics(df_result, A_dens, B_dens) -> pd.DataFrame`

### `arm3/retention.py`
- `compute_degradation_summary(df_full_cov, df_reduced, monitored_quantities, pair_id_col, patient_id_col) -> pd.DataFrame`
- `compute_floor_dominated_flags(A_dens, B_dens, eta_floor, support_masks) -> np.ndarray`

### `arm3/output.py`
- `write_arm3_outputs(result_root, df_full_cov, df_pseudo_roi_audit, df_bootstrap, df_density_family_dir, df_scale_audit, df_degradation, df_prototype_stability, df_balanced_ot, calibration_record, memo_text) -> None`
- `build_arm3_memo(df_full_cov, df_degradation, df_prototype_stability, calibration_record) -> str`
- `build_prototype_stability_table(df_full_cov, df_bootstrap, frozen_prototype_audit_set) -> pd.DataFrame`

### `arm3/__init__.py`
- No functions; re-exports constants from `constants.py`.

### `arm3_uq_stress.py`
- `run_arm3(stage0_path, config, uot_cfg, kernels, result_root) -> None` (public runner)
- `_load_stage0_bundle(stage0_path, k_full)` (private; delegates to arm2 HDF5 loader)
- `_build_pair_universe(stage0_bundle) -> tuple[pd.DataFrame, pd.DataFrame]` (private; Phase 3)
- `_run_full_coverage_inference(...)` (private; Phase 6a)
- `_run_bootstrap_coverage_loop(...)` (private; Phase 5 + 6b)

---

## 3. Signature Adjustments from Plan

### `build_grid_partition` return type
- Plan: returns `pd.DataFrame` (block_frame only).
- Skeleton: returns `tuple[pd.DataFrame, dict[str, list[str]]]` — block_frame plus `roi_block_universe`.
- **Reason:** `roi_block_universe` (the complete list of block_ids per ROI, including zero-cell blocks) must be passed to `compute_roi_block_summary` to ensure zero-cell blocks are present. Returning it from the same function that computes it avoids a second enumeration pass and makes the contract explicit. This is a minor structural adjustment, not a semantic change.
- **Documented in:** block_partition.py docstring.

### `output.py` — `build_prototype_stability_table` included
- Plan listed this function as "if clearly needed by the plan".
- It was included because the plan's Phase 8 and output contract explicitly list `arm3_prototype_stability.csv` as a required output, and the memo requires prototype stability data.
- No deviation; explicit inclusion is consistent with the plan.

No other signature deviations from the approved plan.

---

## 4. Unresolved Open Facts Blocking Real Implementation

The following items must be resolved by humans before any Phase 4-8 implementation begins:

| # | Open Fact | Blocking which function |
|---|-----------|------------------------|
| 1 | **Spatial coordinate unit verification** — `COORD_TO_MM2 = 1e-6` implies 1 coord unit = 1 µm; the actual coordinate range in `obsm['spatial']` of the frozen `.h5ad` must be verified before `DEFAULT_BLOCK_SIZE_UNITS = 100.0` is meaningful | `build_grid_partition` |
| 2 | **Within-compartment pairing structure for tau calibration** — Exact pair-assembly rule (same-patient, all-pairs Cartesian, leave-one-out, etc.) not yet confirmed | `calibrate_tau_by_compartment` |
| 3 | **Config keys for tau calibration** — `arm3.tau_grid` and `arm3.target_retention` are expected but not yet present in any config file; must be added and confirmed | `calibrate_tau_by_compartment` |
| 4 | **N_REPS confirmation** — `DEFAULT_N_REPS = 100` is taken from the old stub; must be confirmed as the intended value or moved to config YAML | `run_bootstrap_pass`, constants.py |
| 5 | **Coverage sampling rule confirmation** — `n_sampled = max(1, floor(target_coverage * n_total_blocks))` is used in the skeleton docstring but must be explicitly confirmed before implementation | `sample_blocks_to_coverage` |
| 6 | **Floor-dominated criterion** — No definition exists; must be task-fixed and recorded in Arm-3 run metadata before `compute_floor_dominated_flags` is implemented | `compute_floor_dominated_flags`, `compute_degradation_summary` |
| 7 | **Zero-sign tie-breaking rule for sign consistency** — Must be task-fixed before `compute_degradation_summary` is implemented | `compute_degradation_summary` |
| 8 | **Side A / side B seeding strategy in bootstrap** — Per-replicate seed is `rng_seed + i`; side offset for A vs B independence requires a concrete rule | `run_bootstrap_pass` |

---

## 5. Confirmations

- **No scientific logic was implemented.** All non-trivial function bodies raise `NotImplementedError`. `constants.py` contains only constant definitions; no runtime logic.
- **No code in `src/slotar/` was changed.** The `src/slotar/` tree is untouched.
- **`pipeline.py` was not modified.** Arm-3 is not registered; `SUPPORTED_ARM_MODULES` is unchanged.
- **No Task A spec or result docs were changed.** `docs/task_A_spec.md` and all results files are untouched.
- **No hidden assumptions for unresolved Open Facts.** Where an open fact affects a function body, the body raises `NotImplementedError` with an explicit message naming the unresolved item.
- **No additional modules were created silently.** The one structural addition (`roi_block_universe` as a second return value from `build_grid_partition`) is documented in Section 3 above.
