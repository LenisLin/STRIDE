# Task A Arm-3 Phase 7/8 Interface Patch Note

**Date:** 2026-03-16
**Branch:** chore/slotar-library-tasks
**Status:** Phase 7/8 interface patch applied.  Full Phase 8 memo not implemented.

---

## 1. Files Edited

| File | What changed |
|------|-------------|
| `tasks/task_A/arm3/inference.py` | Added `run_uot_batch_with_events`; added module-level imports for `batched_uot_solve` and slotar exception constants |
| `tasks/task_A/arm3/retention.py` | Added `compute_contrast_degradation_summary` (Patch 2A); added `build_prototype_contrast_table` (Patch 2B); added sign-consistency audit print inside `compute_contrast_degradation_summary` |
| `tasks/task_A/arm3_uq_stress.py` | Phase 6B loop: replaced `run_uot_batch_safe` with `run_uot_batch_with_events`; added `all_proto_boot_rows` accumulator; added prototype event write block with two self-checks; added Phase 7 contrast compute + write; added Phase 8 prep table write; added `_build_prototype_events_df` helper; updated filename constants |

No other files were edited.

**Unchanged (confirmed):**
- `src/slotar/` — untouched
- `pipeline.py` — untouched
- `common.py` — untouched
- `docs/task_A_spec.md` and all results docs — untouched
- `calibrate.py`, `pseudo_roi.py`, `block_partition.py`, `constants.py` — untouched
- Phase 4 calibration logic — unchanged
- Coverage ladder (75%/50%/25%) — unchanged
- Geometric block logic and zero-cell block handling — unchanged
- Floor-dominated rule — unchanged
- Stage-0 loader `_load_arm3_stage0_inputs` — not patched for `proto_labels`;
  `proto_labels` was already extracted by the existing loader and is simply
  passed downstream to `_build_prototype_events_df`.

---

## 2. Patch 1 — Prototype Event Extraction

### How `batched_uot_solve` is invoked

`run_uot_batch_with_events` in `tasks/task_A/arm3/inference.py` directly calls
`slotar.uot.batched_uot_solve` — the same solver entry point used by
`calibrate_tau_by_compartment` in `calibrate.py`.

The solver is called **exactly once** per inference batch.  After that single
call the scalar metrics DataFrame is assembled from `solver_metrics` and
`status`.  Prototype-level event tensors are then computed from the already-
returned scalars via `extract_prototype_event_marginals` — no second solver
invocation.

`run_uot_batch_safe` from `common.py` is no longer called in Phase 6A or 6B;
`run_uot_batch_with_events` replaces it.

### Prototype event formulas (locked)

The solver's metrics dict returns scalar aggregates `T`, `D_pos`, `B_pos`
per row but does **not** expose per-prototype marginals `pi1`, `pi2`.
`extract_prototype_event_marginals` implements density-proportional allocation
as the locked first-order approximation:

```
T_k[i, k] = T[i]     * A[i, k] / (S_src[i] + eps)
D_k[i, k] = D_pos[i] * A[i, k] / (S_src[i] + eps)
B_k[i, k] = B_pos[i] * B[i, k] / (S_tgt[i] + eps)
```

This is equivalent to the locked positive-part marginal definitions:

```
T_k ≡ pi1_{i,k}           (proportional-allocation approximation)
D_k ≡ max(0, A_{i,k} - pi1_{i,k})
B_k ≡ max(0, B_{i,k} - pi2_{i,k})
```

Non-ok rows receive NaN for all K prototype entries.

### Output files written

| File | Content |
|------|---------|
| `arm3_phase6_prototype_events_full.parquet` | Full-coverage prototype events (replicate_id = −1, coverage = 1.0) |
| `arm3_phase6_prototype_events_bootstrap.parquet` | Bootstrap prototype events (all coverages × replicates) |

**Schema per row (both files):**
`pair_id`, `patient_id`, `pair_type`, `pair_family`, `coverage`,
`replicate_id`, `prototype_k`, `prototype_label`, `T_mass`, `B_mass`, `D_mass`.

---

## 3. Phase 7 Contrast Definitions (Patch 2A)

### Why contrast-based sign consistency

`U_abs_dens = B_pos + D_pos` and `Q_src_dens = T / (S_src + eps)` are
always-nonnegative quantities.  Applying sign consistency directly to them
is not informative — the sign of a nonnegative quantity is always ≥ 0.

The biologically meaningful sign-consistency objects for the TC→IM / TC→PT
ordered-anchor comparison are the **between-direction contrasts**:

```
Delta_U_abs = U_abs_dens(TC->PT) − U_abs_dens(TC->IM)
Delta_Q_src = Q_src_dens(TC->IM) − Q_src_dens(TC->PT)
```

Both are signed quantities: `Delta_U_abs > 0` means TC→PT carries more
unmatched burden than TC→IM; `Delta_Q_src > 0` means TC→IM is more
transportable than TC→PT.

### Alignment rule (Pandas safety)

Before subtraction, TC→IM and TC→PT values are aggregated via median per
`(patient_id, pair_type)` for full coverage, and per
`(patient_id, pair_type, coverage, replicate_id)` for bootstrap.  They are
then merged on the alignment key so both sides are in the **same row** before
the subtraction is performed.  Rows missing either anchor direction are dropped
and a warning is printed; they are never silently fabricated.

### Sign rule (locked)

For any signed contrast `x`:
- sign = `+1` if `x > 0`
- sign = `0` if `x == 0`
- sign = `−1` if `x < 0`

**Zero-sign tie-breaking rule (locked, same as `compute_degradation_summary`):**
- If `sign_100 == 0`: patient is **non-evaluable** (excluded from denominator).
- If `sign_100 ≠ 0` and `sign_c == 0`: counts as **failure** (numerator does
  not increment).
- `pi_c = n_sign_retained / n_evaluable`   (NaN if `n_evaluable == 0`)

### Sign-consistency audit

`compute_contrast_degradation_summary` prints a per-`(contrast_name, coverage)`
audit line:

```
[Phase 7 contrast] contrast=Delta_U_abs, coverage=75%:
    n_total=32, n_evaluable=30, n_zero_reference_sign=2 (non-evaluable),
    n_sign_retained=25, sign_consistency_rate=0.833
```

`n_zero_reference_sign` is also stored in the output DataFrame.

### Existing scalar outputs preserved

`compute_degradation_summary` on raw `U_abs_dens` and `Q_src_dens` is still
called and written as `arm3_phase7_degradation_summary.{parquet,csv}`.
The contrast-based summary is written **in addition** as
`arm3_phase7_contrast_summary.{parquet,csv}`.

### Output files written (Phase 7)

| File | Content |
|------|---------|
| `arm3_phase7_degradation_summary.parquet` / `.csv` | Existing raw-quantity degradation summary (unchanged) |
| `arm3_phase7_contrast_summary.parquet` / `.csv` | New contrast-based sign-consistency summary (Patch 2A) |

---

## 4. Prototype-Level `Delta_U_k` for Phase 8 (Patch 2B)

### Definition (locked)

For each pair row `i` and prototype `k`:

```
U_k = B_k + D_k
```

For each patient, coverage level / replicate, and prototype `k`:

```
Delta_U_k = U_k(TC->PT) − U_k(TC->IM)
```

This is the signed prototype-level contrast for Phase 8 sign consistency.

`build_prototype_contrast_table` in `retention.py` takes the prototype event
parquets, computes `U_k` per row, aggregates (mean) across multiple ROI pairs
within the same patient/pair_type/prototype slot, and then merges TC→IM and
TC→PT columns strictly per
`(patient_id, coverage, replicate_id, prototype_k)` before subtraction.

### Phase 8 data path prepared

The contrast table is written to `arm3_phase8_prototype_contrast_prep.parquet`
with columns: `patient_id`, `coverage`, `replicate_id`, `prototype_k`,
`prototype_label`, `U_k_TC_IM`, `U_k_TC_PT`, `Delta_U_k`.

Phase 8 can use this table to compute:
- Recurrence proportion: fraction of patients / replicates where `|Delta_U_k|`
  is above a task-fixed threshold
- Sign consistency: `sign(Delta_U_k)` vs full-coverage reference sign per
  prototype
- Correlation of `Delta_U_k` to full-coverage reference across patients

Full Phase 8 implementation (prototype stability memo) is **not implemented**
in this patch pass.

---

## 5. Self-Checks Added

| Check | Location | What is verified |
|-------|----------|-----------------|
| Prototype label integrity | `arm3_uq_stress.py` after Phase 6 | Every `prototype_k ∈ [0, K)` has a valid non-empty string label |
| Event mass nonnegativity | `arm3_uq_stress.py` after Phase 6 | `T_mass ≥ 0`, `B_mass ≥ 0`, `D_mass ≥ 0` on full-coverage ok rows; no NaN on ok rows |
| Contrast completeness | `compute_contrast_degradation_summary` | Both TC→IM and TC→PT present before any subtraction; drops missing patients with a warning |
| Sign-consistency audit | `compute_contrast_degradation_summary` | `n_zero_reference_sign` counted and printed per (contrast, coverage); stored in output DataFrame |

---

## 6. Scalar Output Preservation

All Phase 0–6 scalar outputs are preserved unchanged:
- `arm3_phase6_full_coverage_results.parquet`
- `arm3_phase6_bootstrap_results.parquet`
- `arm3_phase6_balanced_ot_results.parquet`
- `arm3_phase6_support_mask_audit.parquet`
- `arm3_phase6_metric_summary_anchor.parquet`
- `arm3_phase7_degradation_summary.parquet` / `.csv`

The prototype event parquets and contrast summaries are **additional** outputs,
not replacements.

---

## 7. Confirmations

| Item | Status |
|------|--------|
| Stage-0 loader not patched for `proto_labels` | ✓ Confirmed — `proto_labels` already extracted by existing `_load_arm3_stage0_inputs`; simply passed downstream |
| `common.py` untouched | ✓ Confirmed |
| Phase 4 calibration logic unchanged | ✓ Confirmed |
| `src/slotar/` untouched | ✓ Confirmed |
| `pipeline.py` untouched | ✓ Confirmed |
| No scientific spec files changed | ✓ Confirmed |
| No pass/fail thresholding added | ✓ Confirmed — outputs are continuous statistics only |
| Coverage ladder unchanged | ✓ Confirmed — 75% / 50% / 25% only |
| Floor-dominated rule unchanged | ✓ Confirmed |
| Geometric block logic unchanged | ✓ Confirmed |
| Solver called exactly once per batch | ✓ Confirmed — `run_uot_batch_with_events` calls `batched_uot_solve` once; `extract_prototype_event_marginals` uses the scalar results only |
