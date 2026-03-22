# Task A Current Results

## 1. Evidence Basis and Boundary
- This document summarizes the current persisted Task A real-data outputs already on disk.
- Current persisted status: Arm I current-stage real-data output is present, Arm II startup-slice output is present, the Arm III Phase 0-8 density-primary bundle is present, and the neutral Arm2/Arm3 extraction bundle is present.
- Current evidence is bounded to the frozen Stage-0 artifact and the following current persisted outputs:
  - Arm I: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm1_realdata_2026-03-19/task_A_metrics.parquet`
  - Arm II: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/task_A_metrics.parquet` plus `analysis/focused/` and `analysis/bioinformed/`
  - Arm III: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/task_A_metrics.parquet` plus phase 6-8 outputs
  - Neutral Arm2/Arm3 alignment bundle: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/00_neutral_extraction_manifest.json` through `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/09_arm2_arm3_linkage_inventory.csv`
- Current result scope:
  - Arm I: current constrained-versus-broken comparison stage
  - Arm II: current startup slice on confirmatory `TC-IM` and `TC-PT`; `IM-PT` remains exploratory
  - Arm III: current density-primary reduced-coverage continuation layer on `TC->IM` and `TC->PT`
- Current evidence supports a bounded Arms I-III narrative aligned within current paper-oriented scope. It does not close Task A as a full scientific program and does not authorize Arm IV.

## 2. Overall Current Task-A Claim
- On the frozen Stage-0 single-timepoint IMC representation, the current persisted evidence supports a bounded three-step Task A narrative: Arm I shows locality-sensitive separation between constrained and broken pairing; Arm II shows that confirmatory `TC-PT` versus `TC-IM` family separation is already visible before transport and is then refined/decomposed by UOT rather than created by it; trusted TC-related `TC-IM` prototypes behave as family-valid anchors while myeloid / immune-like unmatched semantics remain bounded candidate interpretations; Arm III then shows that selected Arm-II patterns remain trackable under reduced coverage.
- The current supported contribution is therefore bounded to Arms I-III within current paper-oriented scope. It does not support generic OT superiority, confirmatory `IM-PT` claims, TC-related `TC-PT` prototype proof of UOT necessity, standalone overlap-mechanism proof, full Arm-II or Arm-III scientific closure, or Arm IV readiness.

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
### Family-level ordering and baseline pre-existence
- The non-transport baseline layer is non-null on the same confirmatory pair set and is kept separate from the transport interpretation.
- In `03_arm2_patient_family_comparator.csv`, the across-patient median of patient-level median absolute share difference is `0.019153` for `TC-IM` and `0.023092` for `TC-PT`, and `23/32` patients show positive `TC-PT - TC-IM` baseline contrast.
- The current confirmatory surface covers `558` `TC-IM` rows and `522` `TC-PT` rows across `32` patients for each family. In the same comparator surface, the patient-level confirmatory medians are:

| field | `TC-IM` | `TC-PT` |
| --- | ---: | ---: |
| `U_abs` | 864.869392 | 1457.782025 |
| `transport_fraction` | 0.867354 | 0.820392 |
| `unmatched_fraction` | 0.132646 | 0.179608 |
| `M` | 0.652815 | 0.917032 |

- The confirmatory patient-level direction is consistent with the intended biological ordering: `29/32` patients show higher unmatched burden on `TC-PT` than on `TC-IM`, `29/32` show lower transport fraction on `TC-PT`, and `29/32` show higher unmatched fraction on `TC-PT`.
- Current interpretation: baseline separation between `TC-IM` and `TC-PT` is already visible before transport; UOT refines/decomposes this existing family separation rather than creating it.
- Balanced OT remains same-pair comparator context rather than a one-direction winner signal: the patient-level `balanced_minus_uot` median is `+0.005208` on `TC-IM` and `-0.004532` on `TC-PT`.

### TC-related prototypes in `TC-IM` as family-valid anchors
- In `02_arm2_prototype_family_evidence.csv`, trusted TC-related `TC-IM` prototypes retain broadly concordant OT and UOT transport shares: proto `3` (`TC_EpCAM`) is `0.058514` versus `0.058426`, proto `16` (`TC_EpCAM`) is `0.043040` versus `0.043174`, and proto `17` (`TC_Ki67`) is `0.043095` versus `0.043987`.
- These TC-related exemplars are recurrent across patients and behave as transportable anchors on the confirmatory `TC-IM` family.
- Current interpretation: these prototypes support family-valid correspondence between methods on a biologically trusted family. They do not, by themselves, establish UOT necessity.

### TC-related prototypes in `TC-PT`: supplementary restriction
- The same neutral extraction does not support a stronger claim that OT is forcing TC-related `TC-PT` prototypes into transport while UOT stably reallocates them into `D`.
- For the main TC-related `TC-PT` exemplars, OT and UOT transport shares remain broadly similar and UOT unmatched shares are present but modest: proto `3` (`TC_EpCAM`) is `0.043007` versus `0.045990` with unmatched `0.007874`, proto `16` (`TC_EpCAM`) is `0.034407` versus `0.038852` with unmatched `0.006213`, and proto `17` (`TC_Ki67`) is `0.021791` versus `0.025926` with unmatched `0.003274`.
- Current interpretation: TC-related `TC-PT` prototypes may be cited only as supplementary correspondence-preserving context, not as the main UOT-necessity argument.

### Myeloid / immune-like candidate evidence
- Some myeloid / immune-like prototypes provide bounded candidate evidence that UOT preserves unmatched structure where OT-only transport is harder to interpret.
- In `02_arm2_prototype_family_evidence.csv`, proto `2` (`Macro_CD163`) remains the clearest current exemplar: on `TC-PT`, UOT unmatched share is `0.057353` while the paired OT-forced transport score remains smaller (`0.015831`); on `TC-IM`, the same prototype remains recurrent but less asymmetric.
- `Treg` can be used only as a cautious secondary example: on `TC-PT`, UOT unmatched share is `0.027950` with UOT transport `0.022200` and OT transport `0.025018`.
- Current interpretation: `Macro_CD163`, and more cautiously `Treg`, support candidate allocation semantics only. They are not decisive proof of UOT necessity.

### Overlap audit as heterogeneity support
- The overlap audit is useful as a structural support surface, not as a standalone mechanism proof.
- In `04_arm2_overlap_audit.csv`, top-10 overlap is partial rather than identical: anchor versus forced transport shares `2/10` prototypes, anchor versus unmatched shares `8/10`, and forced versus unmatched shares `4/10`.
- Concrete examples are supportive rather than dispositive: TC-related entries such as `TC_EpCAM` illustrate that anchor and unmatched surfaces can coexist, while `Macro_CD163` illustrates that forced and unmatched surfaces are not the same ranking.
- Prototype annotation remains mixed rather than singular: only `5/25` prototypes have `top1_fraction >= 0.50`, and `13/25` have `top12_fraction_sum < 0.50`, so biological labels remain interpretive context rather than strict single-cell-type identities.

### Current Arm-II non-claim boundary
- The current Arm-II results do not support generic `UOT > Balanced OT`, confirmatory `IM-PT` claims, tau/retention interpretation, TC-related `TC-PT` prototype proof of UOT necessity, overlap audit as standalone mechanism proof, a cleanly disjoint transport-versus-unmatched prototype partition, or a statement that Arm II is scientifically complete.

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

### Prototype trackability of selected Arm-II patterns
- The Arm3 linkage inventory ties the selected Arm2 prototypes of current interest to the same prototype indices in Arm3, so the reduced-coverage review can be written as continuity of selected Arm2 patterns rather than as a disconnected robustness bundle.
- In `09_arm2_arm3_linkage_inventory.csv`, Arm2 proto `3` (`TC_EpCAM`), proto `17` (`TC_Ki67`), and proto `2` (`Macro_CD163`) map directly to Arm3 prototypes `3`, `17`, and `2`.
- In `05_arm3_prototype_coverage_stability.csv`, these linked prototypes remain trackable even at `0.25` coverage: proto `3` has `sign_consistency_rate=0.906250` and `correlation_to_full_cov=0.991798`, proto `17` has `0.933333` and `0.992879`, and proto `2` has `0.903226` and `0.998280`.
- Using `arm3_phase8_prototype_contrast_prep.parquet`, the largest full-to-`0.25` drops in prototype-wise median absolute patient-level `Delta_U_k` still include proto `2` (`4.517743`), proto `16` (`4.363830`), and proto `12` (`4.125294`), so the current supported wording remains trackability under reduction rather than invariance.

### Current Arm-III non-claim boundary
- The current Arm-III results support a targeted reduced-coverage continuation of selected Arm-II patterns, not a full scientific validation layer, not full transport-side revalidation of Arm II in the same form, and not Arm IV readiness.

## 7. Current Limitations and Readiness Boundary
- Arms I-III are aligned within the current paper-oriented scope.
- Task A is not scientifically complete.
- Arm II current scope is bounded to confirmatory `TC-IM` versus `TC-PT` ordering, family-valid anchor interpretation, supplementary `TC-PT` restriction, and bounded candidate myeloid / immune-like allocation language.
- Arm III current scope is bounded to targeted reduced-coverage continuation of selected Arm2 patterns rather than generic validation of every Arm2 endpoint.
- `IM-PT` remains exploratory only.
- The repository does not currently support a generic `UOT > Balanced OT` claim.
- Arm IV remains the major pending step, and the current readiness judgement remains `INSUFFICIENT_EVIDENCE_FOR_ARM4`.

## 8. Evidence Map
- Arm I entry-condition result: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm1_realdata_2026-03-19/task_A_metrics.parquet`
- Canonical Arm2/Arm3 alignment bundle: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/00_neutral_extraction_manifest.json`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/01_artifact_source_inventory.csv`
- Arm II family-level ordering and baseline pre-existence: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/03_arm2_patient_family_comparator.csv`
- Arm II prototype family evidence and bounded candidate interpretation: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/02_arm2_prototype_family_evidence.csv`
- Arm II overlap heterogeneity support: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/04_arm2_overlap_audit.csv`
- Underlying Arm II focused and bioinformed provenance is catalogued in `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/01_artifact_source_inventory.csv`, including the original focused tables, the OT-versus-UOT contrast surface, and the B/D directionality surface.
- Arm II vs Arm III full-coverage shared-surface consistency: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/task_A_metrics.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_full_coverage_results.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_balanced_ot_results.parquet`
- Arm III tissue-level robustness: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/task_A_metrics.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase6_bootstrap_results.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase7_degradation_summary.parquet`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase7_contrast_summary.parquet`
- Arm III prototype trackability and cross-arm linkage: `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/05_arm3_prototype_coverage_stability.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction/09_arm2_arm3_linkage_inventory.csv`, `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm3_phase0_8_closure/full_2026-03-19/arm3_phase8_prototype_contrast_prep.parquet`
