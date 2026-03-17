# Arm-3 Phase 7 Implementation Note

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

**Date:** 2026-03-16
**Tranche:** Phase 7 — Continuous retention summary

---

## 1. Files edited

| File | Change |
|------|--------|
| `tasks/task_A/arm3/retention.py` | Replaced `NotImplementedError` stub in `compute_degradation_summary` with full implementation. Module docstring updated to record the locked Zero-Sign Tie-Breaking Rule. `compute_floor_dominated_flags` (standalone version with `B_dens` / `support_masks` signature) remains `NotImplementedError` — it is superseded by `inference.compute_floor_dominated_flags` which is already task-fixed and used in Phase 5/6. |
| `tasks/task_A/arm3_uq_stress.py` | Added `_PHASE7_DEGRADATION_FILENAME` constant; added `compute_degradation_summary` import; replaced the Phase 0–6 intentional-stop `NotImplementedError` with the Phase 7 block; updated module docstring and `run_arm3` docstring to reflect Phase 0–7 tranche. |

---

## 2. Zero-Sign Tie-Breaking Rule (exact as implemented)

For each `(pair_type, coverage, quantity)` group, `sign_consistency_rate` (`pi_c(m)`) is computed as follows across all patients in that group:

```
sign_100 = np.sign(m_100)          # full-coverage reference sign for patient p
sign_c   = np.sign(median_rep)     # sign of median across reduced-coverage replicates

Exclusion rule:
  if sign_100 == 0:
      patient p is EXCLUDED from both numerator and denominator.

Failure rule:
  if sign_100 != 0 and sign_c == 0:
      patient p is counted in denominator but NOT in numerator (failure).

Match rule:
  if sign_100 != 0 and sign_c == sign_100:
      patient p is counted in both denominator and numerator (success).

pi_c(m) = n_match / n_denom
         (nan when n_denom == 0, i.e., all reference values are exactly zero)
```

`sign_consistency_rate` is a **population-level** statistic for the group `(pair_type, coverage, quantity)`. It is broadcast to all patient rows in that group.

---

## 3. No pass/fail thresholds introduced

Phase 7 produces only continuous statistics:

- `median_abs_degradation` — per-patient `|median_rep - m_100|`
- `sign_consistency_rate` — population-level fraction (broadcast to patient rows)
- `floor_dominated_rate` — per-patient mean of boolean `floor_dominated` flag across replicates
- `mean_replicate_value`, `std_replicate_value` — descriptive replicate stats

No boolean retention flags, no threshold comparisons, and no columns named `pass`, `fail`, `retained`, or any variant are present in the Phase 7 output. Threshold constants (`D_MAX_m`, `PI_MIN_m`, `PHI_MAX`) remain externally applied and are not referenced inside `retention.py` or `arm3_uq_stress.py`.

---

## 4. Phase 8 remains blocked

The runner raises at the Phase 8 boundary with:

```python
raise NotImplementedError(
    "Arm-3 Phase 8 (Prototype stability and Memo) not implemented in this tranche"
)
```

No Phase 8 logic, output filenames (`_PHASE8_STABILITY_FILENAME`, `_PHASE8_MEMO_FILENAME`), or prototype audit logic (`_PROTOTYPE_AUDIT_LABELS`) has been touched. Phase 8 remains fully unimplemented.
