# Task A Specification
**Title:** Single-timepoint IMC Selection-aware Stress Test

## 1. Role and Current Boundary
- Task A is the single-timepoint controlled stress test within the broader validation chain, intended to examine selection-awareness, uncertainty calibration, and drift sensitivity under a strict counterfactual constraint on the frozen Stage-0 IMC representation.
- The current live Task A scope is bounded to Arms I-III on the frozen Stage-0 artifact. Arm I provides the locality-sensitive entry condition, Arm II provides the biologically ordered real-tissue interpretation layer, and Arm III provides the density-primary reduced-coverage robustness layer for that Arm-II interpretation.
- Current Arm-II / Arm-III document alignment is grounded in the neutral extraction bundle at `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_arm3_neutral_extraction`, with the manifest and source inventory acting as the documentation crosswalk.
- Current live Task A documentation treats Arms I-III as aligned within current paper-oriented scope. This is a bounded documentation state rather than a claim of full scientific completion.
- The current local Task A outputs remain task-layer validation and analysis artifacts rather than bridge/export deliverables.
- Current live Task A documentation does not authorize Arm IV and does not treat Task A as scientifically complete.

## 2. Global Constraints
- Data source: single-timepoint IMC, frozen Stage-0 artifact, 32 retained patients, operative compartments `TC`, `IM`, and `PT`, `K=25`, `kNN=20`.
- Nominal `3 TC / 3 IM / 3 PT` per-patient design language may be used for intent, but the active frozen Stage-0 artifact contains retained non-nominal cases and is the operative truth for current interpretation.
- Arm II uses one shared `lambda_pl` per unordered pair family, jointly calibrated across both ordered directions in that family.
- In the current Arm-II startup slice, `tau_mode="unavailable"` and `tau` / `R` remain unavailable on otherwise-ok rows.
- Arm III is density-primary and reuses frozen representation, prototype, cost, support, and calibration objects rather than relearning them on pseudo-ROIs.

## 3. Cross-Arm Narrative Contract
- Arm I asks whether the current UOT-based Task A scaffold responds appropriately to locality-preserving structure rather than behaving similarly under a broken-locality negative control.
- Arm II asks whether, on biologically ordered within-patient `TC / IM / PT` comparisons, confirmatory `TC-PT` versus `TC-IM` family ordering is visible at the patient level and whether a non-transport baseline separation is already present before transport is invoked.
- Arm II then asks whether trusted TC-related `TC-IM` prototypes behave as family-valid anchors under both Balanced OT and UOT, while TC-related `TC-PT` examples remain supplementary only and myeloid / immune-like unmatched semantics remain candidate-only.
- Arm III asks whether selected Arm2 patterns of real interest remain trackable when coverage is reduced on the same frozen Stage-0 surface.
- The intended current Task A narrative is therefore sequential and bounded: locality-sensitive behavior first, baseline-backed family ordering and bounded prototype interpretation second, targeted robustness continuation third.

## 4. Arm I: Locality-Sensitive Null / Specificity Baseline
### Intended role
- Arm I is the official locality-preserving random-vs-random null / specificity baseline.
- The broken-locality comparison is a negative-control reference, not a redefinition of Arm I.

### Design contract
- For each eligible `(patient_id, compartment)` slot, side A and side B are sampled from the same patient and same compartment under the official Arm-I null.
- Current locked locality constraints are: compartments limited to `TC`, `IM`, and `PT`; `TLS` excluded upstream; side A and side B disjoint within a draw; sampling without replacement within a draw and with replacement across repeated draws; `m=1` at the current milestone.
- The broken-locality comparator keeps the same draw/slot structure and side-A context but forces side B to come from a different patient and a different compartment.
- Under the current shared Task-A operator surface, Arm I is evaluated on original-ROI full-coverage density.

### Current-stage review target
- Current-stage Arm-I review is aggregate rather than prototype-level.
- The constrained baseline is expected to show lower `M`, higher `R`, higher `T`, and lower `U`, `D_pos`, and `B_pos` than the broken-locality comparator.

### Current-stage pass boundary
- Arm I is strong enough for current-stage closure when constrained-versus-broken separation is visible in the expected directions on the shared Task-A metric surface.
- Arm-I closure is an entry-condition statement for Arm II, not a downstream biological claim.

### Non-claims
- Arm I is not a prototype-level biological interpretation layer.
- Arm I does not by itself complete Task A, finalize `lambda` / `tau` calibration, establish Arm II / III / IV conclusions, or authorize Arm IV.
- Arm-I `R` remains a temporary scaffolded retention summary because `tau` is task-fixed by `compartment_a` on the current Arm-I surface.

## 5. Arm II: Biologically Ordered Validation Design
### Intended role and confirmatory scope
- Arm II is a biologically ordered validation design, not a generic cross-compartment contrast and not a global `UOT > Balanced OT` contest.
- `IM` is treated as the interface / transition zone between `TC` and `PT`, and Arm II has two linked readout layers: patient-level family ordering and bounded prototype-level interpretation.
- Confirmatory pair families are `TC-IM` and `TC-PT`.
- `IM-PT` remains exploratory and audit-only and does not support the main confirmatory claim.
- The row unit is one ordered within-patient cross-compartment ROI pair, but the confirmatory analysis unit is the patient.

### Required benchmark ladder
- Layer 1: abundance/compositional baseline.
- Layer 2: non-transport overlap/similarity baseline.
- Layer 3: same-pair Balanced OT forced-match comparator.
- Layer 4: UOT transport-plus-unmatched interpretation.
- Arm-II interpretation is not complete if only one layer is visible; the benchmark ladder is part of the Arm-II contract.

### Interpretation logic
- The confirmatory tissue-ordering target is asymmetric: `TC-IM` is partially transportable, whereas `TC-PT` is the negative-control-like confirmatory family and should show weaker direct transport meaning and greater unmatched structure than `TC-IM`.
- Balanced OT is the required same-pair forced-match comparator on the exact Arm-II pairs. It provides transport context but no native unmatched semantics.
- UOT is retained because it can represent unmatched `B` / `D` structure that Balanced OT cannot natively express.
- The primary Arm-II methodological question is biological allocation and interpretability: what both methods transport, what Balanced OT forces into transport, and what UOT leaves unmatched in a biologically meaningful way.
- Any OT-versus-UOT statement must therefore remain comparator-specific and interpretation-focused rather than scalar-superiority-focused.
- For full scientific closure, prototype review would need to examine transport anchors, transport that becomes more prominent under Balanced OT, and unmatched structure that is visible only under UOT on the exact confirmatory pairs.

### Current paper-oriented bounded closure
- Arm II can be documented as aligned within current paper-oriented scope when patient-aggregated confirmatory family ordering is visible, the non-transport baseline already separates `TC-PT` from `TC-IM`, and UOT is described as refining/decomposing that existing family separation rather than creating it.
- TC-related `TC-IM` prototypes may be used as family-valid anchors when OT and UOT transport shares remain broadly similar on trusted TC-related exemplars.
- TC-related `TC-PT` prototypes must remain supplementary only and must not be used as the main UOT-necessity argument.
- Myeloid / immune-like examples such as `Macro_CD163`, and more cautiously `Treg`, may be used only as candidate unmatched-allocation interpretations.
- Overlap audit may be used only as supporting evidence that anchor / forced / unmatched surfaces are heterogeneous, not as standalone mechanism proof.

### Current-stage review targets
- Tissue-level review should emphasize confirmatory family ordering using patient-aggregated summaries and explicit baseline-versus-transport separation.
- Prototype-level review should separate family-valid anchors, supplementary `TC-PT` correspondence, and bounded candidate unmatched semantics rather than collapsing them into one prototype hierarchy.
- Primary Arm-II biological readouts are `U_abs`, `transport_fraction`, `unmatched_fraction`, `M`, prototype-level transport anchors, and prototype-level unmatched structure.
- `D_pos` and `B_pos` are supporting unmatched-decomposition quantities rather than independent proof lines.
- In the current Arm-II startup slice, `tau` and `R` are unavailable by design and must not be interpreted.

### Ideal/full scientific closure boundary
- A full scientific Arm-II pass would require visible confirmatory tissue-level ordering, biologically interpretable prototype transport anchors, biologically interpretable UOT unmatched structure, and benchmark-ladder evidence that UOT adds interpretation beyond the simpler baselines.
- Availability of tables alone is not sufficient to call Arm II passed or closed.

### Non-claims
- Arm II does not support a generic `UOT > Balanced OT` claim.
- Arm II does not support confirmatory `IM-PT` claims.
- Arm II does not support tau/retention interpretation on the current startup slice.
- Arm II does not support TC-related `TC-PT` prototypes as proof of UOT necessity.
- Arm II does not support overlap audit as a standalone mechanism proof.
- Arm II does not convert candidate myeloid / immune-like unmatched interpretations into decisive proof.
- Arm II does not require or imply a cleanly disjoint transport-only versus unmatched-only prototype partition.
- Arm II does not by itself authorize Arm IV.

## 6. Arm III: Density-Primary Coverage Robustness
### Intended role and ordered anchors
- Arm III is the density-primary coverage-reduction and uncertainty-quantification stress test for the current Arm-II interpretation.
- In the current live scope, Arm III is a targeted robustness continuation of selected Arm2 patterns rather than a disconnected bundle or a generic full validation layer.
- Arm III operates on the frozen Stage-0 representation artifact and compares reduced-coverage pseudo-ROI inference back to a frozen full-coverage reference.
- The primary ordered confirmatory anchors are `TC->IM` and `TC->PT`, and they must remain separate in the primary Arm-III review surface.

### Frozen objects and primary endpoints
- Arm III must freeze the representation artifact, prototypes, prototype assignments, cost matrix, support, pair-generation logic, and full-coverage calibration objects before pseudo-ROI inference.
- Arm III is density-primary; shape may be derived for comparator analyses and count is audit-only.
- The primary absolute unmatched-burden endpoint is `U_abs_dens = B_pos + D_pos`.
- The primary relative transportability endpoint is `Q_src_dens = T / (source total density + eps)`.
- `Q_tgt_dens` is a mandatory audit endpoint.
- `U_abs_dens` must be interpreted together with baseline scale context (`S_src`, `S_tgt`, `Delta_scale`, `scale_ratio`).
- Balanced OT remains comparator context only in Arm III and is not a primary confirmatory Arm-III endpoint.
- Prototype stability is part of the Arm-III review surface, but the current prototype layer is descriptive rather than a thresholded biological pass/fail layer.

### Current paper-oriented bounded closure
- Arm III can be documented as aligned within current paper-oriented scope when ordered-anchor tissue contrasts remain readable under coverage reduction, the full-coverage shared surface stays numerically continuous with Arm II on shared anchor rows, and selected linked Arm2 patterns remain trackable in descriptive prototype-stability outputs.
- Current Arm3 prototype review is therefore continuity/trackability oriented, not a thresholded biological pass/fail layer.
- Balanced OT remains comparator context only; Arm III does not need to re-establish every Arm-II endpoint in the same form to satisfy current paper-oriented scope.

### Ideal/full scientific closure boundary
- If a scientific Arm-III pass is to be asserted, the review must use degradation, sign-consistency, and floor-dominated-rate thresholds on the ordered-anchor summaries rather than ratio rules or CI heuristics alone.
- The current local runner emits continuous degradation, contrast, and prototype-stability summaries; it does not itself emit scientific pass/fail verdicts.

### Non-claims
- Arm III is a bounded robustness continuation of Arm II, not a standalone mechanism-claim layer and not a generic full robustness validation layer.
- Arm III does not by itself fully re-establish every Arm-II endpoint in the same form.
- Arm III does not by itself complete Task A or authorize Arm IV.

## 7. Metric Interpretation Notes
- `T` denotes transported / transport-explained mass.
- `U` denotes unmatched mass and, on `uot_status == "ok"` rows, follows the current local implementation `U = B_pos + D_pos`.
- `B` and `D` refer to the currently reported positive-part unmatched components `B_pos` and `D_pos`.
- `M` is the average transported cost per transported mass on the scaled cost domain used by the solver.
- `M_balanced` is the same-pair shape-only Balanced OT comparator on the current scaled cost domain.
- In Arm II, `M_balanced` versus `M` is comparator context rather than a fully like-for-like primary burden readout.
- In Arm I, `R` is temporarily usable only as a scaffolded retention summary because `tau` is task-fixed by `compartment_a`.
- In Arm II, `tau` and `R` are unavailable by design and remain `NA` on otherwise-ok rows.

## 8. Scope Boundary
- Current live Task A documentation covers Arms I-III only.
- Arms I-III are aligned within current paper-oriented scope, using bounded rather than full-scientific closure language for Arms II and III.
- Arm IV remains future scope, is the major pending step, and is not part of the current live Task A claim set.
