# Task A Current Results

## 1. Evidence Basis and Boundary
- This document summarizes the current persisted Task A real-data outputs already on disk.
- Current persisted status: Arm I current-stage real-data output is present, Arm II startup-slice output is present, and the Arm III Phase 0-8 density-primary bundle is present.
- Current evidence is bounded to the frozen Stage-0 artifact and the following current persisted outputs:
  - Arm I: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm1_realdata_2026-03-19/task_A_metrics.parquet`
  - Arm II: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/task_A_metrics.parquet` plus `analysis/focused/` and `analysis/bioinformed/`
  - Arm III: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/task_A_metrics.parquet` plus phase 6-8 outputs
- Current result scope:
  - Arm I: current constrained-versus-broken comparison stage
  - Arm II: current startup slice on confirmatory `TC-IM` and `TC-PT`; `IM-PT` remains exploratory
  - Arm III: current density-primary reduced-coverage robustness layer on `TC->IM` and `TC->PT`
- Current evidence supports a bounded Arms I-III narrative; it does not close Task A or authorize Arm IV.

## 2. Overall Current Task-A Claim
- On the frozen Stage-0 single-timepoint IMC representation, the current persisted evidence supports a three-step Task A narrative: Arm I shows locality-sensitive separation between constrained and broken pairing, Arm II supports a biologically ordered UOT interpretation on confirmatory `TC-IM` versus `TC-PT`, and Arm III shows that the unmatched-burden side of that Arm-II interpretation remains readable under reduced coverage.
- The current supported contribution is bounded to this startup-slice and reduced-coverage chain and does not support generic OT superiority, confirmatory `IM-PT` claims, full Arm-II or Arm-III scientific closure, or Arm IV readiness.

## 3. Arm-I Current Result
- The current Arm-I persisted output contains `38400` rows: `19200` `A1_baseline` rows and `19200` `A1_broken_reference` rows, all `ok`, all `density`, with the intended locality invariants preserved.
- The constrained baseline separates from the broken-locality comparator in the expected directions by median:

| field | constrained baseline | broken-locality reference |
| --- | ---: | ---: |
| `M` | 0.150480 | 0.581892 |
| `R` | 0.987103 | 0.860789 |
| `T` | 4241.798305 | 3624.202555 |
| `U` | 4309.428986 | 5730.270572 |
| `D_pos` | 2102.985496 | 2781.667171 |
| `B_pos` | 2102.519533 | 2785.450689 |

- Current result interpretation: Arm I is strong enough as the locality-sensitive entry condition for Arm II.
- Current non-claim boundary: Arm I is not a prototype-biology result, not full Task A completion, and not direct evidence for Arm II / III / IV by itself.

## 4. Arm-II Current Result
### Baseline context
- The non-transport baseline layer is non-null on the same confirmatory pair set and is kept separate from the transport interpretation.
- In `04_baseline_patient_family_confirmatory_summary.csv`, the across-patient median of patient-level median absolute share difference is `0.019153` for `TC-IM` and `0.023092` for `TC-PT`, and `23/32` patients show positive `TC-PT - TC-IM` baseline contrast.
- This means confirmatory tissue differences are already visible before transport is invoked.

### Confirmatory tissue-level result
- The current confirmatory surface covers `558` `TC-IM` rows and `522` `TC-PT` rows across `32` patients for each family.
- In `05_global_transport_summary.csv`, the patient-level confirmatory medians are:

| field | `TC-IM` | `TC-PT` |
| --- | ---: | ---: |
| `U_abs` | 864.869392 | 1457.782025 |
| `transport_fraction` | 0.867354 | 0.820392 |
| `unmatched_fraction` | 0.132646 | 0.179608 |
| `M` | 0.652815 | 0.917032 |

- The confirmatory patient-level direction is consistent with the intended biological ordering: `29/32` patients show higher unmatched burden on `TC-PT` than on `TC-IM`, `29/32` show lower transport fraction on `TC-PT`, and `29/32` show higher unmatched fraction on `TC-PT`.
- Balanced OT remains same-pair comparator context rather than a one-direction winner signal: the patient-level `balanced_minus_uot` median is `+0.005208` on `TC-IM` and `-0.004532` on `TC-PT`.

### Prototype-level interpretation
- The current prototype layer supports an allocation/interpretability claim rather than a scalar-winner claim.
- Recurrent shared-transport anchors include `Mono_CD11c`, `TC_EpCAM`, and `TC_Ki67` prototypes, while recurrent unmatched contributors include `Mono_CD11c`, `Macro_CD163`, `CD8T`, and `SC_COLLAGEN`.
- Patient-level recurrence is strong for the leading signals: proto `14` (`Mono_CD11c`) is positive in shared transport `32/32` patients and in unmatched structure `32/32` patients, proto `3` (`TC_EpCAM`) is shared-transport-positive in `32/32` patients, and proto `2` (`Macro_CD163`) is unmatched-positive in `32/32` patients.
- The strongest current OT-vs-UOT wording is bounded. OT transport and UOT shared transport overlap heavily (`9/10` shared top-10 prototypes), but UOT shared transport and UOT unmatched also overlap heavily (`8/10` shared top-10 prototypes). The current evidence therefore supports UOT as adding an interpretable unmatched-allocation layer on top of largely shared transport anchors, not a clean transport-only versus unmatched-only prototype partition.
- Prototype annotation is mixed rather than singular: only `5/25` prototypes have `top1_fraction >= 0.50`, and `13/25` have `top12_fraction_sum < 0.50`. Biological labels should therefore be used as interpretive context rather than strict single-cell-type identities.

### Current Arm-II non-claim boundary
- The current Arm-II results do not support generic `UOT > Balanced OT`, confirmatory `IM-PT` claims, tau/retention interpretation, a cleanly disjoint transport-versus-unmatched prototype partition, or a statement that Arm II is fully passed.

## 5. Arm-II vs Arm-III Full-Coverage Shared Surface
- On the `540` natural-key-matched forward-direction rows that Arm II and Arm III share at full coverage, the directly comparable pair-level fields are numerically near-identical.
- `U`, `T`, `D_pos`, `B_pos`, and `M` are exactly identical on all `540` shared rows, `scale_ratio` differs only at machine precision (`max_abs_diff = 1.33e-15`), and Arm-II `M_balanced` versus Arm-III full-coverage `balanced_ot_cost` differs only at machine precision (`max_abs_diff = 6.66e-16`).
- This supports using Arm III as a reduced-coverage robustness continuation of the same full-coverage UOT surface on the shared anchor rows.
- The cross-arm boundary remains important: Arm II exposes no usable `tau` / `R` on the startup slice, whereas Arm III does. Current evidence therefore supports shared-surface numerical continuity plus bounded robustness, not full transport-side revalidation in the same form.

## 6. Arm-III Current Result
### Tissue-level robustness
- The confirmatory unmatched-burden ordering remains visible across the reduced-coverage ladder: `TC->PT` stays above `TC->IM` on `U_abs_dens` at full, `0.75`, `0.50`, and `0.25` coverage, while the `Q_src_dens` contrast remains directionally stable.

| coverage | `TC->IM` `U_abs_dens` | `TC->PT` `U_abs_dens` | `TC->IM` `Q_src_dens` | `TC->PT` `Q_src_dens` |
| --- | ---: | ---: | ---: | ---: |
| full | 880.974475 | 1355.978754 | 0.944980 | 0.972643 |
| 0.75 | 893.874107 | 1374.319192 | 0.946557 | 0.972021 |
| 0.50 | 902.072165 | 1374.165342 | 0.946437 | 0.970833 |
| 0.25 | 925.234265 | 1373.784934 | 0.946697 | 0.970404 |

- Phase-7 degradation summaries are directionally stable rather than collapsing: pair-type sign consistency is `1.000000` for both `U_abs_dens` and `Q_src_dens`, and the patient-level `Delta_U_abs` contrast keeps sign consistency at `0.968750` across `0.75`, `0.50`, and `0.25` coverage.

### Prototype trackability
- The main prototype contrast signals remain trackable at low coverage, especially proto `14` (`sign_consistency_rate=0.967742`, `correlation_to_full_cov=0.997495`), proto `2` (`0.903226`, `0.998280`), and proto `23` (`1.000000`, `0.991744`).
- Using `arm3_phase8_prototype_contrast_prep.parquet`, the largest full-to-`0.25` drops in prototype-wise median absolute patient-level `Delta_U_k` (with the `0.25` side taken after per-patient bootstrap-median collapse) occur on proto `2` (`4.517743`), proto `16` (`4.363830`), and proto `12` (`4.125294`), so the current supported wording is trackability under reduction rather than invariance.

### Current Arm-III non-claim boundary
- The current Arm-III results support the robustness side of the Arm-II interpretation, not a full scientific pass, not full transport-side revalidation of Arm II in the same form, and not Arm IV readiness.

## 7. Current Limitations and Readiness Boundary
- Task A is not complete.
- Arm II is not fully passed.
- Arm III is not fully passed.
- `IM-PT` remains exploratory only.
- The repository does not currently support a generic `UOT > Balanced OT` claim.
- The current Arm-II / Arm-III transport-side endpoint-definition gap remains open, so the current readiness judgement remains `INSUFFICIENT_EVIDENCE_FOR_ARM4`.

## 8. Evidence Map
- Arm I entry-condition result: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm1_realdata_2026-03-19/task_A_metrics.parquet`
- Arm II baseline context: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/02_baseline_pair_audit.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/03_baseline_prototype_confirmatory_summary.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/04_baseline_patient_family_confirmatory_summary.csv`
- Arm II confirmatory tissue-level result: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/task_A_metrics.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/05_global_transport_summary.csv`
- Arm II prototype and OT-vs-UOT boundary: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/01_prototype_biological_meaning_table.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/06_uot_shared_transport_anchors.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/07_balanced_ot_forced_transport_prototypes.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/08_uot_unmatched_contributors.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/09_prototype_overlap_conflict_audit.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/10_prototype_family_specific_summary.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/11_prototype_patient_recurrence_summary.csv`
- Arm II auxiliary legacy and audit context: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/12_auxiliary_legacy_prototype_comparison.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/13_auxiliary_legacy_prototype_patient_recurrence.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/14_minimal_appendix_audit.csv`
- Arm II bioinformed supporting context: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/bioinformed/20_tc_dominant_family_summary.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/bioinformed/21_mixed_interface_family_summary.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/bioinformed/23_ot_vs_uot_prototype_contrast.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/bioinformed/25_arm2_biointegrated_memo_table.csv`
- Arm II vs Arm III full-coverage shared-surface consistency: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/task_A_metrics.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_full_coverage_results.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_balanced_ot_results.parquet`
- Arm III tissue-level robustness: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/task_A_metrics.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_bootstrap_results.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase7_degradation_summary.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase7_contrast_summary.parquet`
- Arm III prototype trackability: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase8_prototype_stability.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase8_prototype_contrast_prep.parquet`
