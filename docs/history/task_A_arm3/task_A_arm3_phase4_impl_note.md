# Task A Arm-3 Phase 4 Implementation Note

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

**Date:** 2026-03-15
**Branch:** chore/slotar-library-tasks
**Status:** Phase 4 fully implemented. `lambda_dens` and `tau_by_compartment` both calibrated.

---

## 1. Files Edited

| File | What changed |
|------|-------------|
| `tasks/task_A/arm3/calibrate.py` | Implemented `calibrate_lambda_dens` fully. Replaced the `calibrate_tau_by_compartment` `NotImplementedError` stub with a full within-patient same-compartment Pi-weighted quantile implementation. Updated signature: removed `tau_grid`/`target_retention`; added `roi_patient_map` and `tau_q`. |
| `tasks/task_A/arm3_uq_stress.py` | Extended `run_arm3` to run Phase 4 (Phase 0–3 tranche). Fixed grid dimension bug in `_check_block_universe_integrity` (`max+1` → range-safe `max-min+1`). Updated tau section: removed `tau_grid`/`target_retention` config reading; reads `arm3.tau_q` (default 0.5); builds `roi_patient_map` from `roi_table`; calls updated `calibrate_tau_by_compartment`; updated `_write_phase4_outputs` signature. |
| `docs/history/task_A_arm3/task_A_arm3_phase4_impl_note.md` | This note. |

No other files were edited. `constants.py`, `block_partition.py`, `inference.py`, `retention.py`, `output.py`, `pseudo_roi.py`, `arm3/__init__.py`, `pipeline.py`, `src/slotar/`, and all spec/result docs are unchanged.

---

## 2. What Phase 4 Actually Implements

### A. Grid dimension bug fix in `_check_block_universe_integrity`

**Bug:** `n_cols = max(cols) + 1` assumed 0-based column indices, which would give a wrong count if the snapped grid does not start at column 0.

**Fix:**
```python
# Before (unsafe — assumes 0-based indexing)
n_cols = max(cols) + 1
n_rows = max(rows) + 1

# After (range-safe)
n_cols = max(cols) - min(cols) + 1
n_rows = max(rows) - min(rows) + 1
```

The coordinate-set distinctness check (`len(set(zip(cols, rows))) == len(block_ids)`) catches duplicates independently of the dimension formula.

### B. `lambda_dens` calibration — FULLY IMPLEMENTED

`calibrate_lambda_dens` in [tasks/task_A/arm3/calibrate.py](tasks/task_A/arm3/calibrate.py):
- For each unordered family in `("TC-IM", "IM-PT", "TC-PT")`, masks `pair_meta_full` by `pair_family` (both ordered directions pooled; same joint-calibration strategy as Arm-2).
- Assembles `A` and `B` density tensors by indexing `roi_density_vectors`.
- Calls `slotar.uot.calibrate_joint_lambda(A, B, lambda_grid, kernels, cfg, target_alpha)`.
- Returns `{"TC-IM": float, "IM-PT": float, "TC-PT": float}`.
- Config: `arm3.lambda_grid` → `arm2.lambda_grid` fallback. `arm3.target_alpha` → `arm2.target_alpha` → `0.05`.

### C. `tau_by_compartment` calibration — FULLY IMPLEMENTED

`calibrate_tau_by_compartment` in [tasks/task_A/arm3/calibrate.py](tasks/task_A/arm3/calibrate.py):

**Reference pool:** Full-coverage original ROI pairs only. For each patient with ≥2 ROIs in compartment `c`, all within-patient, same-compartment ordered ROI pairs `(roi_a, roi_b)` where `roi_a ≠ roi_b` are enumerated. Cross-patient pairs are excluded. No calibration is performed on pseudo-ROIs.

**`mean_lambda` proxy (local implementation rule):** Within-compartment pairs (TC-TC, IM-IM, PT-PT) do not map to any cross-compartment pair family. The current implementation uses the arithmetic mean of the three already-calibrated family lambdas (`TC-IM`, `IM-PT`, `TC-PT`) as a neutral lambda proxy to drive the unconstrained solve and obtain the cost statistic distribution used for tau calibration. This is a local implementation proxy specific to this calibration pass. It is not a universal theorem or permanent project-wide law. Later refinement remains possible without rollback of the current implementation.

**Calibration math (unconstrained solve):**
1. Run `batched_uot_solve(A, B, lambda_pl, kernels, cfg, tau_external=None)`. The solve is unconstrained — `tau_external=None` means no tau truncation is applied during the calibration pass.
2. Extract `M_i` (Pi-weighted mean transported cost per pair; returned by the solver as `metrics["M"]`).
3. `tau_c = np.quantile({M_i : status == "ok"}, tau_q)`.

**Config key:** `arm3.tau_q` (default `0.5`). No `tau_grid` or `target_retention` scanning.

**Returns:** `{"TC": float, "IM": float, "PT": float}`.

---

## 3. Output Files Written

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
| `arm3_phase4_calibration_record.json` | Always | Full Phase 4 metadata: lambda, tau, tau_q, rule description |

---

## 4. Remaining Blockers Before Phase 5

| # | Blocker | Blocks |
|---|---------|--------|
| 1 | **Spatial coordinate unit verification** — need to confirm `obsm['spatial']` values are in µm before `block_size_units=100` is geophysically meaningful | Phase 1 interpretation |
| 2 | **N_REPS confirmation** — `DEFAULT_N_REPS=100` is a placeholder | `run_bootstrap_pass` (Phase 5) |
| 3 | **Coverage sampling rule** — `max(1, floor(coverage × n_blocks))` must be confirmed | `sample_blocks_to_coverage` (Phase 5) |
| 4 | **Floor-dominated criterion** — no definition for `compute_floor_dominated_flags` | `retention.py` (Phase 7) |
| 5 | **Zero-sign tie-breaking rule** — must be task-fixed for `compute_degradation_summary` | `retention.py` (Phase 7) |
| 6 | **Bootstrap side A/B seeding strategy** — per-replicate offset rule for independence | `run_bootstrap_pass` (Phase 5) |

The tau and lambda calibration blockers from the previous tranche are resolved. Phase 4 is complete.

---

## 5. Confirmations

- **Phase 5+ remains unimplemented.** `run_arm3` raises `NotImplementedError("Arm-3 Phase 5+ (bootstrap, inference, retention, outputs) not implemented in this tranche...")` after writing Phase 4 outputs.
- **`src/slotar/` was untouched.** `calibrate_joint_lambda`, `batched_uot_solve`, `STATUS_OK`, and `precompute_logKernels` are called but not modified.
- **`pipeline.py` was untouched.** `SUPPORTED_ARM_MODULES` is unchanged; Arm-3 is not registered.
- **No scientific spec files were changed.**
- **Current local coverage ladder:** `100%` remains the frozen full-coverage reference baseline; reduced bootstrap levels are `75% / 50% / 25%` only.
- **No `constants.py` changes were needed.** Defaults fall back to existing config keys.
