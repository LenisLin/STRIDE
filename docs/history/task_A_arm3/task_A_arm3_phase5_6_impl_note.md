# Task A Arm-3 Phase 5/6 Implementation Note

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

**Date:** 2026-03-15
**Branch:** chore/slotar-library-tasks
**Status:** Phase 5 and Phase 6 fully implemented. Phase 7+ remains `NotImplementedError`.

---

## 1. Files Edited

| File | What changed |
|------|-------------|
| `tasks/task_A/arm3/pseudo_roi.py` | Full implementation of `sample_blocks_to_coverage`, `build_pseudo_roi_density`, `run_bootstrap_pass`. Added `SIDE_B_SEED_OFFSET = 1_000_000` as an explicit named constant. |
| `tasks/task_A/arm3/inference.py` | Full implementation of `assemble_density_tensors`, `freeze_support_masks`, `broadcast_frozen_tau`, `broadcast_frozen_lambda`, `compute_arm3_density_metrics`, `compute_floor_dominated_flags`. |
| `tasks/task_A/arm3_uq_stress.py` | Extended `run_arm3` to run Phase 5 and Phase 6. Added imports for Phase 5/6 modules. Added Phase 5/6 output filename constants. Added helper functions `_select_full_coverage_columns`, `_build_support_mask_audit`, `_build_metric_summary_anchor`, `_write_phase5_6_outputs`. Updated `NotImplementedError` boundary from Phase 5 to Phase 7. |
| `docs/history/task_A_arm3/task_A_arm3_phase5_6_impl_note.md` | This note. |

No other files were edited. `constants.py`, `block_partition.py`, `calibrate.py`, `output.py`, `retention.py`, `arm3/__init__.py`, `pipeline.py`, `src/slotar/`, and all spec/result docs are unchanged.

---

## 2. Phase 5: Pseudo-ROI Bootstrap

### What is implemented

**`sample_blocks_to_coverage(available_block_ids, target_coverage, rng)`**
- `n_sampled = max(1, floor(target_coverage * n_total_blocks))` — exact locked rule.
- Samples with replacement using `rng.integers(0, n_total, size=n_sampled)`.
- Returns a list of sampled block IDs (may contain repeats).
- Zero-cell blocks are included in `available_block_ids` and may be sampled.

**`build_pseudo_roi_density(roi_block_df, sampled_block_ids, k_full)`**
- Builds a `block_id -> row_index` lookup dict from the DataFrame.
- Accumulates `count_k*` and `block_area_mm2` for each sampled block (with repetition).
- Returns `total_counts / total_area` as the density vector and `total_area` as pseudo area.
- Zero-cell blocks contribute area but zero counts — no exclusion.

**`run_bootstrap_pass(roi_block_summary, pair_meta, coverage, n_reps, k_full, rng_seed)`**
- Pre-caches per-ROI numpy arrays for `block_areas` and `counts` to avoid repeated DataFrame indexing.
- For each replicate and each pair: samples independently using separate RNG streams.
- Returns `A_reps (n_reps, N_pairs, K)`, `B_reps (n_reps, N_pairs, K)`, and `pseudo_meta` audit DataFrame.

### Side A / side B seeding rule (EXACT)

```
base_seed        = rng_seed + replicate_idx
side A RNG       = np.random.default_rng(base_seed)
side B RNG       = np.random.default_rng(base_seed + SIDE_B_SEED_OFFSET)
SIDE_B_SEED_OFFSET = 1_000_000   # explicit named constant in pseudo_roi.py
```

Side A and side B are seeded with entirely separate generators. They never share state or sequence. The offset `1_000_000` is large enough to prevent overlap for any realistic replicate count.

---

## 3. Phase 6: Arm-3 Inference

### What is implemented

**`assemble_density_tensors(roi_density_vectors, pair_meta, k_full)`**
- Stacks density vectors (or count vectors) in `pair_meta` row order for `roi_a` and `roi_b`.
- Raises `ValueError` if any referenced `roi_id` is absent.
- Works for full-coverage density vectors, pseudo-ROI density vectors, and full-coverage COUNT vectors (the caller controls which dict is passed).

**`freeze_support_masks(A_count, B_count, n_min_proto, k_full)`**
- Computes `support_masks[r, k] = (A_count[r, k] + B_count[r, k]) >= n_min_proto`.
- Returns a `(N, K)` bool array.
- Computed once; never updated or recomputed on pseudo-ROI inputs.

**`broadcast_frozen_tau(pair_meta, frozen_taus)`**
- Maps `compartment_a` -> frozen tau value.
- Validates that only `TC`, `IM`, `PT` are present.
- Fails loudly if any unmapped compartment is found.

**`broadcast_frozen_lambda(pair_meta, frozen_lambdas)`**
- Maps `pair_family` -> frozen lambda value.
- Validates that only `TC-IM`, `IM-PT`, `TC-PT` are present.
- Fails loudly if any unmapped family is found.

**`compute_arm3_density_metrics(df_result, A_dens, B_dens)`**
- Appends: `S_src`, `S_tgt`, `Delta_scale`, `scale_ratio`, `U_abs_dens`, `Q_src_dens`, `Q_tgt_dens`.
- `S_src`/`S_tgt`/`Delta_scale`/`scale_ratio` computed for all rows.
- `U_abs_dens`, `Q_src_dens`, `Q_tgt_dens` set to `NaN` for non-ok rows.
- The forbidden ratio `T / (T + B_pos + D_pos + eps)` is never computed or stored.

**`compute_floor_dominated_flags(A_dens, support_masks, eta_floor)`**
- Implements the exact task-fixed floor-dominated rule.

---

## 4. Floor-Dominated Rule (EXACT)

```
K_support_r     = sum(support_masks[r])          # count of True entries
floor_mass_r    = eta_floor * K_support_r
S_src_r         = sum(A_dens[r, k] for k where support_masks[r, k] is True)
floor_dominated = floor_mass_r / (S_src_r + DENSITY_EPS) > 0.10
```

- Threshold: `0.10` (10%).
- `DENSITY_EPS = 1e-12` from `arm3.constants`.
- `support_masks` are the FROZEN masks from the full-coverage reference (not recomputed per replicate).
- Implemented in `tasks/task_A/arm3/inference.py:compute_floor_dominated_flags`.

No adjustment from the specified rule was necessary.

---

## 5. Support Masks Are COUNT-Based (CONFIRMATION)

`freeze_support_masks` receives `A_count` and `B_count`, which are **full-coverage COUNT tensors** assembled from `roi_block_summary` by summing `count_k*` columns across all blocks per ROI. These are **not** density tensors.

In `arm3_uq_stress.py`:
```python
roi_count_vectors = {
    roi_id: df[count_col_names].sum(axis=0).to_numpy(dtype=float)
    for roi_id, df in roi_block_summary.items()
}
A_count, B_count = assemble_density_tensors(roi_count_vectors, pair_meta_anchor, k_full)
support_masks = freeze_support_masks(A_count=A_count, B_count=B_count, n_min_proto=..., k_full=k_full)
```

The `n_min_proto` threshold is compared against COUNT mass, not density. Density tensors are never passed to `freeze_support_masks`.

---

## 6. Balanced OT Uses L1-Normalised Shape Tensors (CONFIRMATION)

Before calling `run_balanced_ot_batch`, density tensors are L1-normalised:

```python
A_shape = A_dens / (A_dens.sum(axis=1, keepdims=True) + DENSITY_EPS)
B_shape = B_dens / (B_dens.sum(axis=1, keepdims=True) + DENSITY_EPS)
# Self-check: assert A_shape.sum(axis=1) <= 1 + 1e-9
bot_costs = run_balanced_ot_batch(A_shape, B_shape, cost_matrix, n_min_proto=0.0)
```

`n_min_proto=0.0` is passed to `run_balanced_ot_batch` because the shape-only comparator should not apply an additional count-based filter on already-normalised inputs.

A runtime assertion verifies that shape tensors are L1-normalised before being passed to the Balanced OT solver.

---

## 7. Output Files Written

| File | Written | Content |
|------|---------|---------|
| `arm3_phase0_manifest.json` | Always | Run parameters |
| `arm3_phase1_block_frame.parquet` | Always | Per-cell block assignments |
| `arm3_phase1_roi_block_universe.json` | Always | Complete ROI block ID lists |
| `arm3_phase1_roi_block_summary.parquet` | Always | Stacked per-block prototype count summary |
| `arm3_phase2_roi_density_reference.parquet` | Always | Per-ROI density vectors |
| `arm3_phase3_pair_meta_full.parquet` | Always | All six ordered pair directions |
| `arm3_phase3_pair_meta_anchor.parquet` | Always | Anchor subset (TC->IM, TC->PT) |
| `arm3_phase4_lambda_dens.json` | Always | Family-keyed `lambda_dens` values |
| `arm3_phase4_tau_by_compartment.json` | Always | Compartment-keyed `tau` values |
| `arm3_phase4_calibration_record.json` | Always | Full Phase 4 metadata |
| `arm3_phase5_pseudo_roi_audit.parquet` | Always | Per-(replicate, pair) bootstrap audit |
| `arm3_phase6_full_coverage_results.parquet` | Always | Full-coverage reference UOT results |
| `arm3_phase6_bootstrap_results.parquet` | Always | Bootstrap UOT results (all coverages × replicates) |
| `arm3_phase6_balanced_ot_results.parquet` | Always | Shape-only Balanced OT comparator outputs |
| `arm3_phase6_support_mask_audit.parquet` | Always | Per-pair support mask summary |
| `arm3_phase6_metric_summary_anchor.parquet` | Always | Grouped anchor metrics (TC->IM, TC->PT) |

---

## 8. Remaining Blockers Before Phase 7

| # | Blocker | Blocks |
|---|---------|--------|
| 1 | **Floor-dominated criterion in Phase 7** — the row-level flag is now computed in Phase 6, but the Phase 7 retention/degradation summary aggregation logic (`compute_floor_dominated_flags` → `compute_degradation_summary`) is not yet defined | Phase 7 `retention.py` |
| 2 | **Zero-sign tie-breaking rule** — must be task-fixed for `compute_degradation_summary` | Phase 7 `retention.py` |
| 3 | **Spatial coordinate unit verification** — need to confirm `obsm['spatial']` values are in µm before `block_size_units=100` is geophysically meaningful | Phase 1 interpretation |
| 4 | **N_REPS confirmation** — `DEFAULT_N_REPS=100` is a placeholder | Run parameters |
| 5 | **Phase 8 prototype stability memo** — not yet defined | Phase 8 `output.py` |

The lambda, tau, support mask, and floor-dominated blockers from the previous tranche are resolved.

---

## 9. Confirmations

- **Phase 7+ remains unimplemented.** `run_arm3` raises `NotImplementedError("Arm-3 Phase 7+ (retention summaries, prototype stability memo) not implemented in this tranche...")` after writing Phase 5/6 outputs.
- **`src/slotar/` was untouched.** `batched_uot_solve`, `STATUS_OK`, `precompute_logKernels`, and `UOTSolveConfig` are called but not modified.
- **`pipeline.py` was untouched.** `SUPPORTED_ARM_MODULES` is unchanged; Arm-3 is not registered.
- **No scientific spec files were changed.**
- **Current local coverage ladder:** `100%` remains the frozen full-coverage reference baseline; reduced bootstrap levels are `75% / 50% / 25%` only.
- **Support masks are COUNT-based.** `freeze_support_masks` receives COUNT tensors assembled from `roi_block_summary`, not density tensors.
- **Balanced OT uses L1-normalised shape tensors.** Density tensors are normalised with `DENSITY_EPS` guard before passing to `run_balanced_ot_batch`.
- **Side A / side B independence is explicit.** `SIDE_B_SEED_OFFSET = 1_000_000` is a named constant. Side A and side B never share RNG state.
