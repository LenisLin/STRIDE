# Task A Arm-II Current Local Conclusion

## 1. Title and scope

This document records the current local Arm-II conclusion for Task A from the focused output package at `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused`.

The conclusion is stage-limited and startup-slice bounded. It applies only to the current Arm-II startup slice on the frozen Stage-0 artifact and current local metrics parquet. It is not a claim of full Arm-II closure.

## 2. Scientific question

The Arm-II question at the current stage is whether, on biologically ordered within-patient `TC-IM` versus `TC-PT` comparisons, UOT provides a biologically interpretable view of shared transport plus unmatched structure relative to same-pair Balanced OT.

The point is not a generic method contest. The point is whether the ordered tissue families support the interpretation that plausible shared structure remains transportable, weakly corresponding structure is not merely force-matched, and unmatched structure adds interpretable information that Balanced OT does not natively expose.

## 3. Link from Arm-I to Arm-II

Arm-I provided the required entry condition for Arm-II at the current local stage. Under the constrained-versus-broken locality comparison, the present locality-sensitive scaffold did not behave like a random or degenerate baseline.

This supports moving to biologically ordered real-tissue comparisons. It does not by itself settle the Arm-II question, and it does not imply closure beyond the current startup slice.

## 4. Data / scope boundary

- Current scope is the startup slice only.
- Confirmatory families are `TC-IM` and `TC-PT`.
- `IM-PT` remains exploratory and audit-only.
- The confirmatory analysis unit is patient.
- The current focused package covers 32 patients.
- The current ordered-pair audit counts are 558 `TC-IM`, 522 `TC-PT`, and 630 `IM-PT`.
- `tau` and `R` are unavailable in the current slice and are not interpreted.

## 5. Baseline evidence

Before transport is invoked, the baseline layer establishes that the ordered pair set is real, patient-indexed, and nontrivial on the confirmatory families.

- Output `02` provides the pair-level baseline audit on the exact ordered Arm-II pair set.
- Output `03` provides all-prototype confirmatory baseline summaries across 25 prototypes.
- Output `04` provides patient-level confirmatory baseline summaries across 32 patients.
- In output `04`, the across-patient median of patient-level median absolute share difference is 0.019 for `TC-IM` and 0.023 for `TC-PT`.
- In output `04`, 23 of 32 patients have positive `TC-PT - TC-IM` median absolute share difference.
- In output `03`, the highest baseline-ranked prototypes include `Mono_CD11c`, `TC_EpCAM`, `UNKNOWN`, `NK`, and `Macro_CD163`, each with larger confirmatory baseline anchor on `TC-PT` than on `TC-IM`.

This baseline layer does not decide the method comparison. It establishes that confirmatory tissue differences are already present before transport and should remain analytically separate from transport claims.

## 6. Transport evidence

The confirmatory transport evidence is patient-level and ordered by tissue family.

- In output `05`, the across-patient median `U_abs` is 864.9 for `TC-IM` and 1457.8 for `TC-PT`.
- In output `05`, the across-patient median transport fraction is 0.867 for `TC-IM` and 0.820 for `TC-PT`.
- In output `05`, the across-patient median unmatched fraction is 0.133 for `TC-IM` and 0.180 for `TC-PT`.
- In output `05`, 29 of 32 patients show higher `TC-PT` than `TC-IM` unmatched fraction.
- In output `05`, 29 of 32 patients show lower `TC-PT` than `TC-IM` transport fraction.
- In output `05`, the median patient-level Balanced-minus-UOT readout is small and mixed in sign at the family level: 0.005 for `TC-IM` and -0.005 for `TC-PT`.

The current transport readout therefore supports a family-level interpretation in which `TC-PT` carries greater unmatched burden and lower transport fraction than `TC-IM`, while Balanced OT remains a contextual forced-match comparator rather than a sole winner metric.

## 7. Prototype-level interpretation

The prototype layer shows that UOT provides a distinct perspective relative to Balanced OT rather than merely duplicating a forced-match view.

- Output `06` shows prototypes with positive transport on both confirmatory families, including `Mono_CD11c` and `TC_EpCAM`, indicating that plausible shared structure remains transportable.
- Output `06` also shows prototypes where transport and unmatched structure coexist rather than collapsing into a single forced-match account. Examples include `Mono_CD11c` (`TC-PT` transport share 0.055, unmatched share 0.067) and `Macro_CD163` (`TC-PT` transport share 0.037, unmatched share 0.057).
- Output `06` further shows prototypes with weak or asymmetric correspondence across the ordered families, such as `UNKNOWN` and `NK`, where `TC-IM` transport is very small while `TC-PT` retains positive transport plus positive unmatched mass.
- Output `07` shows that these prototype-level patterns recur at the patient level rather than appearing as single-patient artifacts.
- Output `08` confirms that Balanced OT has no unmatched semantics in this package by construction.

The current prototype-level interpretation is therefore that UOT retains plausible shared transport where it exists, but does not require all biologically weak or tissue-specific structure to be absorbed into matched transport. The unmatched component is part of the interpretation, not a residual to be ignored.

## 8. Current claim

- Arm-I established that the current locality-sensitive scaffold is not behaving randomly under the constrained-versus-broken locality comparison.
- On the current startup slice and confirmatory ordered families `TC-IM` versus `TC-PT`, UOT provides a distinct perspective relative to same-pair Balanced OT.
- The supported Arm-II interpretation is that plausible shared structure remains transportable, while weakly corresponding or tissue-specific structure is not merely force-matched.
- UOT unmatched structure adds biologically interpretable information that Balanced OT does not natively provide on the same pair set.
- This claim is bounded to the current startup slice, current frozen Stage-0 artifact, current patient set, and current focused outputs.

## 9. Non-claims

- This is not a claim that Arm-II is fully passed.
- This is not a generic claim that UOT beats Balanced OT.
- This is not a claim of full mechanism, causality, or biological closure.
- This is not a confirmatory claim based on `IM-PT`.
- This is not a claim about `tau`- or `R`-dependent semantics, because `tau` and `R` are unavailable in the current slice.
- This is not a robustness claim under reduced coverage, perturbation, or drift.

## 10. Relation to Arm-3

Arm-3 is the next robustness step. Its purpose is to test whether the current Arm-II pattern remains stable under coverage reduction.

No stronger Arm-3 statement is supported by the present Arm-II outputs alone.

## 11. Operative outputs

The current conclusion is supported by the following nine focused outputs.

- `00_arm2_focused_results_memo.md`
- `01_prototype_biological_meaning_table.csv`
- `02_baseline_pair_audit.csv`
- `03_baseline_prototype_confirmatory_summary.csv`
- `04_baseline_patient_family_confirmatory_summary.csv`
- `05_global_transport_summary.csv`
- `06_key_prototype_comparison.csv`
- `07_key_prototype_patient_recurrence.csv`
- `08_minimal_appendix_audit.csv`
