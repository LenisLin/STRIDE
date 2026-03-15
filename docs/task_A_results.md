# Task A Current Results Summary

## 1. Scope

This document records the current local Task-A results state.

Task A currently contains completed Arm-I current-stage evidence and current-stage Arm-II evidence, while Arm-III remains the next robustness step.

The present summary is stage-limited and startup-slice bounded. It is not a claim of full Task-A completion.

## 2. Current evidence boundary

- Current evidence is based on the frozen Stage-0 artifact and the current local Task-A metrics parquet.
- Arm-I is interpreted at its current documented stage.
- Arm-II is interpreted only on the current startup slice.
- Arm-II confirmatory families are `TC-IM` and `TC-PT`.
- `IM-PT` remains exploratory and audit-only.
- The confirmatory analysis unit is patient.
- The current focused Arm-II package covers 32 patients.
- `tau` and `R` are unavailable in the current Arm-II startup slice and are not interpreted.

## 3. Arm-I as the entry condition

Arm-I provides the entry condition for Arm-II at the current local stage.

Under the present metric set, the current locality-sensitive scaffold shows separation between locality-preserving constrained pairing and locality-breaking pairing. This supports proceeding to biologically ordered tissue comparisons on the frozen Stage-0 representation.

This Arm-I result is not a claim of full Task-A completion, final calibration, or downstream robustness.

## 4. Arm-II baseline layer

The Arm-II baseline layer is interpreted separately from transport.

- Output `02` establishes the ordered Arm-II pair audit on the exact startup-slice pair set.
- Output `03` summarizes confirmatory baseline differences across 25 prototypes.
- Output `04` summarizes confirmatory baseline differences across 32 patients.
- In output `04`, the across-patient median of patient-level median absolute share difference is 0.019 for `TC-IM` and 0.023 for `TC-PT`.
- In output `04`, 23 of 32 patients show positive `TC-PT - TC-IM` median absolute share difference.

The baseline layer therefore establishes that confirmatory tissue differences are already present before transport is invoked. It does not, by itself, decide the method comparison.

## 5. Arm-II transport layer

The Arm-II transport layer is interpreted on the same ordered pair set and remains analytically distinct from the baseline layer.

- In output `05`, the across-patient median `U_abs` is 864.9 for `TC-IM` and 1457.8 for `TC-PT`.
- In output `05`, the across-patient median transport fraction is 0.867 for `TC-IM` and 0.820 for `TC-PT`.
- In output `05`, the across-patient median unmatched fraction is 0.133 for `TC-IM` and 0.180 for `TC-PT`.
- In output `05`, 29 of 32 patients show higher unmatched fraction on `TC-PT` than on `TC-IM`.
- In output `05`, 29 of 32 patients show lower transport fraction on `TC-PT` than on `TC-IM`.
- Balanced OT remains a same-pair forced-match comparator. In output `05`, the patient-level Balanced-minus-UOT summary is small and mixed in sign at the family level rather than a one-direction global winner readout.

On the current startup slice, the transport layer therefore supports a confirmatory tissue-level ordering in which `TC-PT` carries greater unmatched burden and lower transport fraction than `TC-IM`.

## 6. Arm-II prototype-level interpretation

The prototype layer is the current Arm-II interpretation layer relative to same-pair Balanced OT.

- Output `06` shows prototypes with positive transport on both confirmatory families, including `Mono_CD11c` and `TC_EpCAM`, consistent with plausible shared structure remaining transportable.
- Output `06` also shows prototypes where transport and unmatched structure coexist rather than collapsing into a single forced-match account. Examples include `Mono_CD11c` and `Macro_CD163` on `TC-PT`.
- Output `06` further shows prototypes with weak or asymmetric correspondence across the ordered families, including `UNKNOWN` and `NK`, where `TC-IM` transport is small while `TC-PT` retains positive transport plus positive unmatched structure.
- Output `07` shows that these prototype-level patterns recur across patients rather than appearing as isolated single-patient events.
- Output `08` confirms that Balanced OT has no unmatched semantics in the current package.

The current prototype-level readout therefore supports the interpretation that UOT provides a distinct and biologically interpretable perspective relative to same-pair Balanced OT: plausible shared structure remains transportable, weakly corresponding structure is not merely force-matched, and unmatched structure contributes interpretable information.

## 7. Current claim

- Arm-I establishes the current entry condition by showing separation between locality-preserving constrained pairing and locality-breaking pairing under the present metric set.
- Arm-II, on the current startup slice and on confirmatory `TC-IM` versus `TC-PT`, provides a biologically ordered validation layer rather than a generic OT-versus-UOT score contest.
- The current supported Arm-II claim is that UOT provides a distinct and biologically interpretable perspective relative to same-pair Balanced OT.
- This claim is bounded to the current startup slice, current frozen Stage-0 artifact, current patient set, and current focused outputs.

## 8. Non-claims

- This is not a claim that Arm-II is fully passed.
- This is not a claim that Task A is complete.
- This is not a generic claim that UOT beats Balanced OT.
- This is not a claim of mechanism, causality, or full biological closure.
- This is not a confirmatory claim based on `IM-PT`.
- This is not a robustness claim under coverage reduction or drift.

## 9. Arm-III relation

Arm-III is the next robustness step. Its purpose is to test stability of the current Arm-II pattern under coverage reduction.

## 10. Supporting Arm-II outputs

The current Arm-II evidence record is supported by the focused output package at `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused`.

- `00_arm2_focused_results_memo.md`
- `01_prototype_biological_meaning_table.csv`
- `02_baseline_pair_audit.csv`
- `03_baseline_prototype_confirmatory_summary.csv`
- `04_baseline_patient_family_confirmatory_summary.csv`
- `05_global_transport_summary.csv`
- `06_key_prototype_comparison.csv`
- `07_key_prototype_patient_recurrence.csv`
- `08_minimal_appendix_audit.csv`
