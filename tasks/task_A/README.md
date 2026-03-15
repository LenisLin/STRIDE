# Task A

This task directory records the completed local Arm-I stage on validated
SLOTAR-ready `.h5ad` input and now hosts the implemented Arm-II startup slice
plus its current analysis-only follow-up.
Arm-I remains documented and closed enough for the current local track; the
current Task-A focus is the Arm-II interpretation and integrated follow-up.

## Current local Arm-I milestone
- Stage 0 is complete enough for the current Task-A track and is not being reopened here.
- Active input route: validated Stage-0 `.h5ad` artifact at `K=25`.
- Active constrained null arm: `A1_baseline`.
- Active broken-locality comparator arm: `A1_broken_reference`.
- Active constrained Arm-I semantics: within-patient, same-compartment, `m=1`, A/B disjoint within a draw, without replacement within a draw, with replacement across repeated draws, task-layer-only provenance.
- Active broken-locality comparator semantics: preserve the same draw/slot structure and the same side-A constrained context, then force side B to come from a different patient and a different compartment.
- Active mass mode: `count` only.
- Active scaffold rule: current task-fixed `lambda_pl` and `tau_external` are assigned from `compartment_a` only.

## Arm-I interpretation boundary
- Official Arm-I null: the constrained locality-preserving random-vs-random baseline.
- Not the Arm-I null: the anchored broken-locality reference.
- Why the distinction matters: the broken-locality reference is a negative-control comparator for locality sensitivity, not a redefinition of Arm I.

## What the current Arm-I milestone has accomplished
- It establishes a runnable and auditable constrained Arm-I baseline.
- It establishes a directly comparable broken-locality negative-control reference under the same A-side context and current scaffold.
- Under the current local Task-A scaffold, the constrained-vs-broken comparison has shown separation on the main MRTU comparison metrics in the current local review.
- A minority of patient-compartment groups still deserve manual review, but they do not block closing the current Arm-I stage in documentation.
- It does not mean Arm I has scientifically passed, and it does not complete all of Task A.

## Current Arm-II startup slice
- Active Arm-II startup arm: `A2_cross_compartment`.
- Arm-II startup is within-patient only, `k=1` only, `count` mode only, and uses deterministic exhaustive ordered ROI pairing.
- Startup unordered pair families: `TC-IM`, `IM-PT`, `TC-PT`.
- Startup ordered directions retained for audit: `TC->IM`, `IM->TC`, `IM->PT`, `PT->IM`, `TC->PT`, `PT->TC`.
- Arm-II row unit: one ordered within-patient cross-compartment ROI pair per row.
- Arm-II calibration rule: one shared `lambda_pl` per unordered pair family, jointly calibrated using both ordered directions in that family, then broadcast to both directions.
- Arm-II same-pair comparator: `M_balanced` from task-layer shape-only Balanced OT on the exact same pairs.
- Arm-II tau rule: `tau_mode="unavailable"` and `tau` / `R` remain `NA` on ok rows.
- Current project interpretation for the local Arm-II track: Arm II is documented as a biologically ordered benchmark ladder on `TC / IM / PT`, with `IM` treated as the interface / transition zone between `TC` and `PT`.
- Confirmatory pair families: `TC-IM`, `TC-PT`.
- Exploratory / secondary / excluded from main confirmatory analysis: `IM-PT`.
- Direction-specific rows are retained for audit, but primary confirmatory interpretation is family-level and patient-aggregated.
- Current local Arm-II readout is not a single global UOT-vs-Balanced-OT score contest. It is a biologically ordered readout at two linked levels: tissue-family ordering and prototype-level ordering.
- Nominal local design wording may still refer to 3 `TC` / 3 `IM` / 3 `PT` per retained patient, but the active frozen Stage-0 artifact used for the current Arm-II slice is not strictly uniform 3/3/3; it contains retained non-nominal cases and is the operative analysis truth.
- Confirmatory interpretation is patient-level; multiple ROI-pair rows from the same patient are audit rows, not independent confirmatory observations.
- This remains a live local Task-A analysis-and-interpretation stage on frozen Stage-0 artifacts and temporary task-layer outputs; it is not final Arm-II closure and it is not full V1.6 / V2.0 platform validation.

## What the current local Arm-II slice already shows
- The locked Arm-II startup slice has been implemented, run, and analyzed on the frozen Stage-0 artifact.
- The ordered within-patient Arm-II pair set and the confirmatory/exploratory family split are therefore fixed for the current local slice rather than hypothetical.
- Same-pair Balanced OT comparison is present on the exact Arm-II ordered pairs, and UOT unmatched quantities are present on the same slice.
- The current slice already supports family-level summaries plus prototype-level transport and unmatched review on the operative frozen artifact.
- This is enough to keep Arm II active as a biological interpretation stage, but not enough to call Arm II passed.

## Arm-II tissue-level biological ordering
- `TC-IM` is the confirmatory partially transportable family.
- `TC-PT` is the confirmatory negative-control-like family and should show weaker direct transport meaning and more unmatched structure than `TC-IM`.
- `IM-PT` remains exploratory for the current phase and is not part of the main confirmatory family set.

## Arm-II prototype-level biological ordering
- Prototype-level review is part of the Arm-II scientific readout, not optional background context.
- For each prototype, Arm II should review how it is transported under UOT and how it is transported under same-pair Balanced OT.
- This review should be organized by method, by tissue family, and across patients.
- Where useful, it should also descend to source-to-target prototype edges so recurrent edge assignments can be checked for biological plausibility.
- Arm II should also examine which concrete prototypes contribute to UOT birth-like (`B`) and destruction-like (`D`) unmatched components.
- Where possible, prototype interpretation should be linked back to available biological annotation or cell-type composition basis.
- If higher-level prototype biology labels are curated later, they should be used as an interpretation layer rather than assumed to already be complete.

## Arm-II method comparison logic
- Arm-I null remains the earlier baseline/null/specificity reference.
- Under the current constrained baseline/null scaffold, Arm I established that UOT does not show pathological behavior severe enough to block moving to ordered real-tissue comparisons.
- Balanced OT is the required same-pair forced-match comparator.
- UOT is the current unbalanced transport model and can leave prototype mass unmatched.
- Arm-II interpretation should therefore ask how each prototype is transported under UOT, how each prototype is transported under Balanced OT, which prototypes are additionally forced into transport under Balanced OT, which prototypes are left unmatched under UOT, and which unmatched prototype patterns contribute to birth-like (`B`) or destruction-like (`D`) UOT components.
- Balanced OT does not natively provide `B` / `D` semantics.
- The comparison is therefore not only about total transport cost or total transported mass; it is also about what each method transports and what only UOT leaves unmatched and interprets through `B` / `D`.
- The current Arm-II objective is to determine whether, in biologically ordered tissue comparisons, UOT preserves biologically plausible shared transport while also providing biologically meaningful unmatched interpretation for tissue-specific or weakly corresponding prototype structure.
- Arm II is not primarily a global scalar superiority contest and should not be presented as generic OT-vs-UOT winner selection.
- Current local Arm-II interpretation is judged mainly by tissue-family ordering, prototype-level biological plausibility, and whether UOT adds interpretable unmatched structure beyond the forced-match Balanced OT baseline.
- `M_balanced` versus `M` remains useful same-pair stress context on the current scaled cost domain, but it is a secondary forced-match comparator with different mass semantics, not a fully like-for-like primary burden comparison.
- Current Arm-II follow-up focuses on whether family-level summaries, per-prototype transport comparison, and UOT-only unmatched interpretation remain consistent under the planned stratified analysis.

## Arm-II benchmark ladder
- Layer 1, simple abundance / compositional baseline: asks what prototype abundance or composition differences are already visible before transport is invoked; it does not answer transport allocation or unmatched semantics; it is needed so Arm-II does not attribute obvious composition changes to transport modeling alone.
- Layer 2, non-transport overlap / similarity baseline: asks how similar paired prototype distributions are without transport; it does not answer which prototypes should align under transport or which structure should remain unmatched; it is needed to show what is already visible before transport allocation.
- Layer 3, Balanced OT forced-match transport baseline: asks what transport structure appears when full matching is enforced on the exact same ordered pairs; it does not answer whether weakly corresponding structure should remain unmatched and it does not natively provide `B` / `D` semantics; it is needed as the same-pair forced-match comparator.
- Layer 4, UOT transport-plus-unmatched interpretation: asks which structure appears plausibly shared and transportable and which structure is better represented as unmatched birth-like or destruction-like mass; it does not by itself prove mechanistic truth or generic superiority; it is needed because Arm II asks whether unmatched semantics add interpretation beyond the simpler layers.
- If one of the simpler benchmark layers is not yet stable on the current pair set, it should be treated as a required next-step output or check rather than described as already closed.

## Intended Arm-II result presentation order
1. Prototype biological annotation / interpretation map: define what each prototype biologically represents, summarize its cell-type composition basis, and optionally add broader biological grouping later if curated.
2. Non-transport baselines: show abundance / compositional differences and non-transport similarity differences before transport modeling.
3. Global tissue-level transport summaries: present confirmatory family-level ordering next, while separating primary biological readouts from contextual quantities.
4. Prototype-level OT and UOT transport comparison: show recurrent transport anchors, shared transport patterns, and prototypes that gain extra transport under Balanced OT.
5. UOT-only unmatched interpretation: show which prototypes contribute to birth-like / destruction-like unmatched structure and why this interpretation is unavailable in Balanced OT.
6. Patient-level recurrence and audit context: close with recurrence across patients, confirmatory versus exploratory scope, and comparator limits / non-claims.

## Arm-II prototype-level interpretation target
- Shared / biologically plausible transportable prototypes: examples include patterns that plausibly recur between `TC` and `IM`. The target is that UOT should preserve reasonable shared transport structure here, and Balanced OT may be similar on some of these anchors.
- Tissue-specific / weakly corresponding prototypes: examples include patterns that appear more specific to `PT`-side, immune-side, or interface-like contexts. The target is that Balanced OT may continue to force-match part of this structure, whereas UOT may leave part unmatched and interpret it through `B` / `D`.
- This distinction is an interpretive target for Arm II, not a completed proof and not a mechanistic claim.

## Arm-II current-stage success logic
- Arm-II would count as successful at the current stage only if biological ordering remains visible at the confirmatory tissue-family level.
- Arm-II would count as successful at the current stage only if prototype-level transport anchors are biologically interpretable.
- Arm-II would count as successful at the current stage only if UOT unmatched structure is biologically interpretable at the prototype level.
- Arm-II would count as successful at the current stage only if the benchmark ladder shows that UOT adds interpretation beyond simpler baselines.
- Any OT-versus-UOT claim must remain limited to biological meaningfulness and comparator-specific context, not generic superiority.
- Arm II is not yet considered closed; final judgment still depends on the pending products and checks listed below.

## What still needs to be generated / checked before Arm-II can be judged passed
- A stable prototype biological annotation / prototype interpretation map for figure and report use.
- Explicit non-transport baseline outputs on the current Arm-II ordered pair set.
- Clean prototype-level OT and UOT transport summaries.
- Clear prototype-level UOT unmatched summaries.
- Patient-level recurrence summaries for key prototype patterns.
- A final integrated memo that states what is and is not supported under the current local implementation.

## Current local metric interpretation notes
- In the current local Task-A implementation, `T` is transported mass and is interpreted in Arm II as matched / transport-explained mass.
- When Arm-II analysis summaries expose `T_abs`, it should be read as absolute transported mass.
- Under the current soft-marginal UOT semantics, `T_abs` should not be treated as an unqualified conserved matched-mass quantity and is not necessarily the primary transportability readout.
- Normalized transportability summaries can be more appropriate for comparing transportability across tissue families or patients, but they do not replace the unmatched-structure readout.
- In the current local Task-A implementation, `D_pos` and `B_pos` are positive-part unmatched mass summaries.
- In current Arm-II interpretation shorthand, `D` and `B` refer to the currently reported `D_pos` and `B_pos`; they are not separate additional metrics.
- In the current local Task-A implementation, `U` is currently implemented as `B_pos + D_pos` on ok rows; it is not documented here as a normalized ratio.
- In current Arm-II interpretation, `U`, `B`, and `D` are biologically meaningful because they describe unmatched structure that Balanced OT does not natively provide.
- In the current local Task-A implementation, `M` is a cost-weighted transport summary on the current scaled cost domain.
- In the current local Task-A Arm-I implementation, `R` is a tau-dependent retained transport fraction using task-fixed `tau` assigned from `compartment_a`.
- In the current local Task-A Arm-II startup implementation, `M_balanced` is the same-pair shape-only comparator on the current scaled cost domain.
- Primary biological readouts in current Arm-II interpretation should emphasize `U_abs`, `transport_fraction`, `unmatched_fraction`, `M`, prototype-level transport anchors, and prototype-level unmatched structure.
- Supporting unmatched decomposition quantities should emphasize `D_pos` and `B_pos`; they refine unmatched structure and should not be casually over-counted as fully independent evidence lines.
- Contextual / diagnostic quantities should include `T_abs`, `M_balanced`, `balanced_minus_uot`, `balanced_to_uot_ratio`, `transport_over_source_total`, and `transport_over_target_total`.
- `T_abs` therefore remains visible as absolute transported mass, but it should not be treated as an unqualified conserved matched-mass quantity.
- `M_balanced` versus `M` remains secondary comparator context, not a fully like-for-like primary burden comparison.
- `transport_over_source_total` and `transport_over_target_total` should not be presented as primary literal fractions under the current implementation.
- In the current local Task-A Arm-II startup implementation, `tau` and `R` are unavailable by design and remain `NA` on ok rows; `R` therefore remains an intended semantic component rather than a current Arm-II startup output.

## Current local real-data workflow context
- Real cohort source: `/mnt/NAS_21T/ProjectData/SLOTAR/CRLM_Cohort.rds`
- Current Task-A planning keeps only `TC`, `IM`, and `PT`
- `TLS` is excluded from the current Task-A track
- Current local working subset retains 32 eligible patients after applying the present cohort filters
- Nominal local design remains 3 `TC`, 3 `IM`, and 3 `PT` ROIs per retained patient where design intent is being described.
- The active frozen Stage-0 artifact used for the current Arm-II slice is not uniformly nominal 3/3/3; it contains retained non-nominal cases and should be treated as the operative analysis truth.

## Temporary task-layer scaffolding
- Arm I currently supplies `lambda_pl` from `arm1.fixed_lambda_by_compartment`.
- Arm I currently supplies `tau_external` from `arm1.fixed_tau_by_compartment`.
- These values are temporary task-layer scaffolding, not calibration outputs.
- Arm-II startup introduces family-level shared `lambda_pl` calibration but still does not introduce a startup tau calibration rule.

## Deferred work
- final lambda/tau redesign/calibration beyond the Arm-II startup slice
- Arm III / Arm IV
- UQ and drift workflows
- bridge-based AVCP export compliance

Task A currently writes one temporary direct metrics parquet for validation only.
That write path is not the canonical AVCP Python-to-R handoff and must not be
treated as final export behavior.

## Local path policy
- Project code and project docs stay in `/home/lenislin/Experiment/projects/SLOTAR`.
- Raw and project data stay in `/mnt/NAS_21T/ProjectData/SLOTAR`.
- Temporary scripts, intermediate outputs, reports, analysis products, and generated result artifacts stay in `/mnt/NAS_21T/ProjectResult/SLOTAR`.
- Current Task-A result and analysis artifacts should be written under `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/`.
- Repo-local generated outputs are not part of the active local layout and should not be recreated under the repository tree.

## Transition note
- Arm I remains documented and closed at the current local stage.
- Arm II remains the current main Task-A focus, but the next step is the integrated analysis-only follow-up rather than a broad pipeline redesign.
- This README does not expand Arm III / IV beyond that transition boundary.
