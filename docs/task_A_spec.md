# Task A Specification (v1.2 Locked)
**Title:** Single-timepoint IMC Selection-aware Stress Test

> Status note: the current local Task-A path has completed Stage 0 sufficiently for Arm-I work and has completed the current Arm-I constrained-vs-broken comparison stage on the Stage-0 `.h5ad` route at `K=25`. Arm I remains closed at that current local stage. The current Arm-II startup slice has been implemented, run, and analyzed on the frozen Stage-0 artifact. Arm III / IV remain scientifically specified but inactive. The current Task-A write path remains a temporary task-layer metrics parquet for validation and analysis, not the canonical AVCP bridge export.

## 1. Global Constraints
- **Data Source**: Single-timepoint IMC, $1\text{ mm}^2$ nominal area, 32 retained patients. Nominal local design language remains 3 `TC`, 3 `IM`, and 3 `PT` ROIs per retained patient where design intent is being described, but the active frozen Stage-0 artifact used for the current Arm-II slice contains retained non-nominal cases and is the operative analysis truth. Current local `TC` corresponds to older `CT` wording from earlier materials.
- **Model Params**: $K=25$, $kNN=20$.
- **Calibration Rule**: Arm-II startup uses one shared `lambda_pl` per unordered pair family, jointly calibrated using both ordered directions in that family.
- **Tau Rule**: Arm-II startup does NOT introduce a new pair-specific tau rule. In the current local Task-A Arm-II startup implementation, `tau_mode="unavailable"` and `tau` / `R` remain `NA` on ok rows.

## 2. Experimental Arms
### Arm I: Constrained Random vs Random (Official Null / Specificity Baseline)
- Official Arm-I null: for each repeated draw and each eligible `(patient_id, compartment)` slot, sample side A and side B from the same patient and the same compartment.
- Current locked Arm-I locality constraints: compartments are limited to `TC`, `IM`, and `PT`; `TLS` is excluded upstream; A and B are disjoint within a draw; sampling is without replacement within a draw and with replacement across repeated draws; `m=1` for the current milestone.
- Current milestone meaning: establish the constrained Arm-I baseline distribution under the implemented locality-preserving null design.
- Current milestone boundary: this makes Arm I implemented, runnable, and auditable for constrained-vs-broken comparison; it does not by itself complete Task A and does not support claims about Arm II / III / IV, uncertainty calibration, drift diagnostics, or final retention calibration.

### Arm I Broken-Locality Comparator (Negative-Control Reference)
- This comparator is NOT the official Arm-I null.
- Current intended next comparison: keep the same Arm-I draw structure and slot structure, keep side A anchored to the same constrained local group used by the Arm-I null, and replace side B with a locality-breaking partner.
- Broken-locality rule for side B: sample from the retained Arm-I ROI pool after upstream filtering, subject to `roi_b != roi_a`, `patient_id_b != patient_id_a`, and `compartment_b != compartment_a`.
- Current scaffold rule: task-fixed `lambda` and `tau` remain assigned from `compartment_a` only. Side A is the reference side for the current comparator design.
- Interpretation goal: test whether locality-preserving partner selection yields meaningfully different current Task-A metric behavior than locality-breaking partner selection under the same A-side context.
- Under the current local Task-A Arm-I metric definitions, the constrained baseline is expected to show lower `M`, higher `R`, higher `T`, and lower `U`, `D_pos`, and `B_pos` than the broken-locality reference.
- Failure signal for the next review step: if constrained Arm-I output does not separate meaningfully from this broken-locality reference, that weakens evidence that the current Task-A Arm-I design is sensitive to locality-preserving transport structure.

### Arm I Current Local Closure Note
- Under the current local Task-A scaffold, constrained baseline versus anchored broken-locality reference has shown separation on the main MRTU comparison metrics in the current local review.
- This supports constrained-vs-broken locality-sensitive separation under the current local Task-A scaffold.
- This does NOT complete all of Task A, does NOT finalize `lambda` / `tau` calibration, and does NOT establish Arm II / III / IV results.
- A minority of patient-compartment groups still warrant manual review, but they do not block closing the current Arm-I stage in documentation.

### Arm II: Biologically Ordered Validation Design
- Arm II is documented here as a biologically ordered benchmark ladder, not as a generic cross-compartment contrast and not as a single global `UOT beats OT` contest.
- Current project interpretation for the local Arm-II track: `IM` is treated as the interface / transition zone between `TC` and `PT`, and the Arm-II objective is defined at two linked biological levels: tissue-level ordering and prototype-level ordering.
- Startup scope is locked to: within-patient only, `k=1` only, `count` mode only, deterministic exhaustive ordered pairing, and temporary task-layer output only.
- Startup unordered pair families: `TC-IM`, `IM-PT`, `TC-PT`.
- Startup ordered directions retained for audit: `TC->IM`, `IM->TC`, `IM->PT`, `PT->IM`, `TC->PT`, `PT->TC`.
- Startup row unit: one ordered within-patient cross-compartment ROI pair per row, with one ROI on side A and one ROI on side B.
- Startup pair-generation rule: for each patient, enumerate ROI IDs by compartment; for each ordered direction above, generate the full Cartesian product of source-compartment ROI IDs and target-compartment ROI IDs exactly once.
- Calibration rule: for each unordered pair family, jointly calibrate one shared `lambda_pl` using both ordered directions in that family; then broadcast that family-level shared `lambda_pl` to all rows in both directions of that family.
- Baseline requirement: MUST run same-pair Balanced OT (shape-only) on the exact same Arm-II A/B pairs and report `M_balanced`.
- Startup tau rule: in the current local Task-A Arm-II startup implementation, `tau_mode="unavailable"` and `tau` / `R` remain unavailable on otherwise-ok rows.

#### Arm-II confirmatory scope
- Confirmatory pair families: `TC-IM`, `TC-PT`.
- Exploratory / excluded from main confirmatory analysis: `IM-PT`.
- Direction-specific rows are retained for audit, but primary confirmatory interpretation is family-level and patient-aggregated.

#### Nominal design versus active frozen artifact
- Nominal local design wording may still refer to a 3 `TC` / 3 `IM` / 3 `PT` per-patient layout when describing intended cohort structure.
- The active frozen Stage-0 artifact used for the current Arm-II slice is not uniformly nominal 3/3/3; it contains retained non-nominal cases and is the operative analysis truth for current Arm-II interpretation.
- Current Arm-II summaries should therefore be indexed to the frozen artifact composition rather than to an idealized uniform layout.

#### Current local slice: what is already shown
- The current Arm-II startup slice has been implemented, run, and analyzed on the frozen Stage-0 artifact under the locked startup contract above.
- The ordered within-patient pair set, confirmatory / exploratory family split, and same-pair Balanced OT comparator are therefore fixed for the current local slice rather than hypothetical.
- The current local slice already provides auditable family-level summaries plus method-specific prototype transport and UOT unmatched quantities on the operative frozen artifact.
- This is enough to treat Arm II as an active biological interpretation stage on real ordered tissue comparisons, but it is not enough to declare Arm II passed.

#### Tissue-level biological ordering
- `TC <-> IM` is the confirmatory partially transportable family; some mass may be biologically relatable / transportable, and some mass should remain unmatched.
- `TC <-> PT` is the confirmatory negative-control-like family in the current study framing; direct transport is not considered strongly biologically meaningful, so this family should show weaker direct transport meaning, weaker transport-explained structure, and more unmatched mass than `TC <-> IM`.
- `IM <-> PT` remains biologically ambiguous in the current project phase and is therefore exploratory only, not a co-equal confirmatory family.

#### Prototype-level biological ordering
- Each prototype on the shared representation should be interpreted biologically where possible, using available biological annotation or cell-type composition basis rather than treating the prototype axis as purely abstract.
- Arm-II should examine how each prototype behaves across methods, across tissue families, and across patients, not only whether family-level scalar summaries separate.
- Prototype-level review should also reach edge level where useful, so that recurrent source-to-target prototype assignments can be checked for biological plausibility under each method.
- Prototype-level review is part of the Arm-II scientific objective and readout, not optional background context.
- If higher-level prototype biology categories are curated later, such as tumor-core-like, stromal-like, immune/myeloid-like, interface-like, or ambiguous, they should be used as an interpretation layer rather than assumed to be already complete.

#### Primary Arm-II question
- Arm-I established, under the current constrained baseline/null scaffold, that UOT does not show pathological behavior severe enough to block moving to ordered real-tissue comparisons.
- Arm II now asks whether, in biologically ordered `TC / IM / PT` comparisons, UOT preserves biologically plausible shared transport while also providing biologically meaningful unmatched interpretation for tissue-specific or weakly corresponding prototype structure.
- Arm II is therefore not primarily a global scalar superiority contest against same-pair Balanced OT.
- The current readout asks both what each method transports and what only UOT leaves unmatched: which tissue families and prototypes are transported under each method, which prototype patterns are additionally forced into transport under Balanced OT, and which prototype patterns remain unmatched only under UOT in a biologically interpretable way.
- The scientific comparison concerns interpretability and biological plausibility, not only scalar advantage on one cost summary.
- Arm-I remains the earlier baseline/null/specificity reference, but it is not the sole Arm-II decision rule.

#### Balanced OT transport comparison
- Balanced OT is the required same-pair forced-match comparator on the exact Arm-II A/B pairs.
- In Arm II, Balanced OT is used as a transport baseline that reveals what transport structure appears when full matching is enforced on the same pairs.
- Balanced OT can therefore help identify which prototypes recurrently act as transport anchors and which prototype patterns become more prominent only when matching is forced.
- Balanced OT does not natively provide unmatched semantics and does not natively provide `B` / `D` semantics.

#### UOT unmatched semantics
- SLOTAR UOT is the unbalanced transport model in this slice and can leave prototype mass unmatched.
- The Arm-II method comparison therefore is not only about total transport cost or total transported mass. It must compare which prototypes are transported under each method, which prototype patterns are additionally forced into transport under Balanced OT, which prototype patterns remain unmatched under UOT, and which unmatched prototype patterns are interpreted by UOT as birth-like (`B`) or destruction-like (`D`) components.
- `B` and `D` are not only abstract unmatched summaries in Arm II; they are prototype-level interpretable unmatched components that should be traced back to concrete prototypes where possible.
- Even when global scalar UOT-versus-Balanced separation is weak, interpretable prototype contributions to `B` and `D` remain a major scientific reason to retain UOT in Arm II.

#### Decomposition semantics
- `T` = matched / transport-explained mass.
- `U` = unmatched mass, consistent with the current local implementation `U = B_pos + D_pos` on ok rows.
- `B` = born / appearing unmatched component, tied to the currently reported `B_pos` and intended for prototype-level biological interpretation rather than only abstract aggregation; Arm II should examine which concrete prototypes contribute to `B`.
- `D` = destroyed / disappearing unmatched component, tied to the currently reported `D_pos` and intended for prototype-level biological interpretation rather than only abstract aggregation; Arm II should examine which concrete prototypes contribute to `D`.
- `R` = intended retained-within-matched semantic component, but it is not currently available in Arm-II startup outputs because `tau` is unavailable.

#### Required prototype-level comparison across methods, tissue families, and patients
- For each prototype, Arm II should review its transport role under UOT and its transport role under same-pair Balanced OT.
- This review should be organized by method, by tissue family, and across patients.
- Where useful, the same review should descend to source-to-target prototype edges rather than only prototype marginals.
- Where possible, the analysis should identify prototypes that recurrently act as transport anchors, prototypes that are additionally forced into transport under Balanced OT, prototypes that are left unmatched under UOT, and prototypes that contribute to birth-like / destruction-like UOT components.
- Prototype-level interpretation should be linked back to available biological annotation or cell-type composition basis.
- Global scalar summaries alone are therefore not treated as sufficient for the current local Arm-II readout.

#### Prototype-level interpretation target
- Shared / biologically plausible transportable prototypes: examples include prototype patterns that plausibly recur between `TC` and `IM`. The interpretive target is that UOT should preserve reasonable shared transport structure here, and Balanced OT may look similar on some of these anchors.
- Tissue-specific / weakly corresponding prototypes: examples include prototype patterns that appear more specific to `PT`-side, immune-side, or interface-like contexts. The interpretive target is that Balanced OT may continue to force-match part of this structure, whereas UOT may leave part unmatched and express it through `B` / `D`.
- This distinction is an Arm-II interpretation target to be tested and curated, not a completed proof and not a mechanistic claim.

#### Statistical unit
- Audit row unit: one ordered within-patient cross-compartment ROI pair.
- Main confirmatory analysis unit: patient.
- Multiple ROI-pair rows from the same patient are not independent confirmatory observations.

#### Benchmark ladder
- Layer 1, simple abundance / compositional baseline: answers what prototype abundance or composition differences are already visible before transport is invoked; does not answer transport allocation or unmatched semantics; needed so Arm-II does not attribute obvious compositional differences to transport modeling alone. If this layer is not yet stable on the current pair set, it remains a required next-step benchmark product rather than a completed closure artifact.
- Layer 2, non-transport overlap / similarity baseline: answers how similar paired prototype distributions are without transport; does not answer which prototypes should be aligned under transport or which structure should remain unmatched; needed to show what is already visible before any transport allocation. If this layer is not yet stable on the current pair set, it remains a required next-step benchmark product rather than a completed closure artifact.
- Layer 3, Balanced OT forced-match transport baseline: answers what transport structure appears when full matching is enforced on the exact same ordered pairs; does not answer whether weakly corresponding structure should remain unmatched, and does not provide native `B` / `D` semantics; needed as the same-pair forced-match transport comparator.
- Layer 4, UOT transport-plus-unmatched interpretation: answers which structure appears plausibly shared and transportable and which structure is better represented as unmatched birth-like or destruction-like mass; does not by itself prove mechanistic truth or generic superiority over OT; needed because Arm II asks whether unmatched semantics add biological interpretation beyond the simpler layers.

#### Intended result presentation order
1. Prototype biological annotation / interpretation map: define what each prototype biologically represents, summarize the cell-type composition basis, and optionally add broader biological grouping later if curated.
2. Non-transport baselines: present abundance / compositional differences and non-transport similarity differences first so the reader sees what is already visible before transport modeling.
3. Global tissue-level transport summaries: present confirmatory family-level ordering next, while explicitly separating primary biological readouts from contextual quantities.
4. Prototype-level OT and UOT transport comparison: show which prototypes are recurrent transport anchors, which prototypes gain extra transport under Balanced OT, and which transport patterns are shared between methods.
5. UOT-only unmatched interpretation: show which prototypes contribute to birth-like / destruction-like unmatched structure and why this is biologically meaningful and unavailable in Balanced OT.
6. Patient-level recurrence and audit context: close with recurrence across patients, confirmatory versus exploratory scope, and the comparator limits / non-claims that bound the interpretation.

#### Current-stage Arm-II success logic
- Current-stage Arm-II success would require biologically ordered structure to remain visible at the tissue-family level on the confirmatory families.
- Current-stage Arm-II success would require prototype-level transport anchors to be biologically interpretable rather than only numerically recurrent.
- Current-stage Arm-II success would require UOT unmatched structure to be biologically interpretable at the prototype level.
- Current-stage Arm-II success would require the benchmark ladder to show that UOT adds interpretation beyond simpler baselines, not merely a scalar contrast against Balanced OT.
- Any OT-versus-UOT claim must remain limited to biological meaningfulness and comparator-specific context, not generic superiority.
- Arm II should not be called passed until the next required analysis products below are generated or checked and one final integrated memo states both what is supported and what is not supported.

#### Next required analysis products
- A stable prototype biological annotation / prototype interpretation map for figure and report use.
- Explicit non-transport baseline outputs on the current Arm-II ordered pair set.
- Clean prototype-level OT and UOT transport summaries, including recurrent anchor-oriented summaries rather than only raw pair-level tables.
- Clear prototype-level UOT unmatched summaries showing `B` / `D` contributors without overstating them as independent proof lines.
- Patient-level recurrence summaries for key prototype patterns, with confirmatory families kept distinct from exploratory `IM-PT`.
- A final integrated memo that states what is supported, what is not supported, and which claims remain outside the current implementation boundary.

### Arm III: Coverage Reduction (Density-Primary Stress Test)

#### Positioning
- Arm III is the density-primary coverage-reduction and UQ stress test for the current Task-A Arm-II interpretation.
- Arm III runs on the frozen Stage-0 representation artifact and uses pseudo-ROIs reconstructed by frozen block resampling.
- Full-coverage reference inference is run first on the original ROI family pairs. Coverage-reduced pseudo-ROI inference is then compared back to that frozen full-coverage reference.
- Density is the primary Arm-III mass route. Shape may be derived from density when required by comparator analyses. Count is audit-only in Arm III.

#### Frozen Arm-III objects
- The following objects MUST be frozen before any pseudo-ROI inference: the Stage-0 representation artifact, frozen `kNN`, frozen community features, prototype dictionary (`K=25`), frozen prototype assignment, `cost_matrix`, `s_C`, the current ordered pair-generation logic, the confirmatory anchor directions used for Arm-III transportability analysis, the same-pair Balanced OT comparator definition, and all full-coverage calibration outputs.
- Arm-III pseudo-ROI inference MUST NOT relearn community features, prototypes, prototype assignments, `cost_matrix`, or calibration parameters.

#### Coverage grid and pseudo-ROI construction
- The full-coverage reference baseline remains 100% and is built once outside the bootstrap loop.
- Reduced bootstrap coverage levels are the locked Arm-III grid: 75%, 50%, and 25%.
- For each original ROI, Arm III uses the valid frozen block partition generated on the frozen Stage-0 artifact.
- For each Arm-III pseudo-ROI replicate, sample valid frozen blocks with replacement from the source ROI block set until the requested coverage target is reached. Side A and side B resample independently from their own original ROI block sets.
- Let `n_{b,k}` be the frozen count assigned to prototype `k` in sampled block `b`, and let `Area_b` be the valid effective area of that block. Reconstruct each pseudo-ROI by:

  $$
  n_k^{pseudo} = \sum_b n_{b,k}, \qquad
  A^{pseudo} = \sum_b Area_b, \qquad
  a_k^{dens} = \frac{n_k^{pseudo}}{A^{pseudo}}
  $$

- When a shape-only comparator or audit is required, derive pseudo-ROI shape from the density vector:

  $$
  \bar{a}_k^{dens} = \frac{a_k^{dens}}{\sum_i a_i^{dens} + \varepsilon}
  $$

- Arm III may report pseudo-count totals and total pseudo-area for audit, but pseudo-counts do not replace the density-primary route.

#### Semantic support and numerical floor
- For each ordered full-coverage reference pair `r`, freeze the semantic active set on the original full-coverage pair before any coverage reduction:

  $$
  \mathcal{K}_r^{100} = \left\{ k : N_{r,A,k}^{100} + N_{r,B,k}^{100} \ge n_{\min}^{proto} \right\}
  $$

- `\mathcal{K}_r^{100}` is the Arm-III semantic support for every pseudo-ROI replicate derived from reference pair `r`. It MUST NOT shrink with coverage reduction.
- `eta_floor` is numerical stabilization only. Within the frozen semantic support, Arm-III may apply:

  $$
  a_k^{dens} \leftarrow \max(a_k^{dens}, \eta_{floor}), \qquad
  b_k^{dens} \leftarrow \max(b_k^{dens}, \eta_{floor})
  $$

- `eta_floor` MUST NOT redefine semantic support, MUST NOT reactivate prototypes outside `\mathcal{K}_r^{100}`, and MUST NOT be used as an Arm-III biological retention rule.

#### Full-coverage calibration and broadcast rules
- All Arm-III calibration is performed once on the full-coverage original ROI reference pools and then frozen before pseudo-ROI inference.
- `lambda_dens` MUST be calibrated once on the full-coverage original ROI family reference pool generated by the frozen ordered pair-generation logic. Arm-III pseudo-ROI inference MUST reuse those frozen family-level `lambda_dens` values and MUST NOT recalibrate `lambda_dens` on pseudo-ROIs.
- `tau` MUST be available in Arm III. It MUST be calibrated once on the full-coverage original ROI compartment reference pool and stored as compartment-specific task-fixed values: `tau_TC`, `tau_IM`, and `tau_PT`.
- For an ordered Arm-III inference row `A -> B`, assign `tau` by `compartment_a`. The ordered row inherits `tau_TC`, `tau_IM`, or `tau_PT` from its source compartment. Arm III MUST NOT introduce mixed family-level `tau` values such as pooled `TC-IM` thresholds built from `TC-TC` and `IM-IM` reference sets.
- Side A remains the reference side for ordered Arm-III assignment. Frozen full-coverage calibration outputs are broadcast to all reduced-coverage pseudo-ROI inference without direction-specific retuning on the pseudo-ROIs.

#### Ordered anchor directions
- The primary Arm-III relative transportability analysis uses only the ordered confirmatory anchor directions `TC -> IM` and `TC -> PT`.
- Arm III MUST keep these anchor directions separate. It MUST NOT collapse them into unordered family medians and MUST NOT mix reverse directions into the same primary summary.
- Reverse directions and exploratory directions may remain in audit outputs, but they do not enter the primary Arm-III ordered-anchor transportability summary.

#### Primary Arm-III endpoints
- For Arm-III density inference on a pseudo-ROI pair `(a^{dens}, b^{dens})`, the primary absolute unmatched burden endpoint is:

  $$
  U_{abs}^{dens} = B_{pos} + D_{pos}
  $$

- The primary relative transportability endpoint is source-normalized transportability:

  $$
  Q_{src}^{dens} = \frac{T}{\sum_i a_i^{dens} + \varepsilon}
  $$

- Arm III MUST also report the mandatory audit / secondary target-normalized transportability:

  $$
  Q_{tgt}^{dens} = \frac{T}{\sum_j b_j^{dens} + \varepsilon}
  $$

- `T_abs` may remain visible as supportive audit context only. It is not a primary Arm-III endpoint.
- The invalid transportability quantity

  $$
  \frac{T}{T + B_{pos} + D_{pos} + \varepsilon}
  $$

  MUST NOT be used anywhere as a primary Arm-III transportability endpoint.

#### Baseline scale audit
- Before interpreting `U_{abs}^{dens}`, Arm III MUST report baseline scale context for every ordered pair and coverage level:

  $$
  S_{src} = \sum_i a_i^{dens}, \qquad
  S_{tgt} = \sum_j b_j^{dens}, \qquad
  \Delta_{scale} = S_{tgt} - S_{src}, \qquad
  scale\_ratio = \frac{S_{tgt}}{S_{src} + \varepsilon}
  $$

- Arm III MUST report family- and direction-level summaries of source scale, target scale, and source-target scale asymmetry.
- `U_{abs}^{dens}` MUST NOT be interpreted in isolation from this scale audit.

#### Retention logic and technical tolerance
- Arm-III retention is evaluated on a task-fixed monitored set of patient-level contrast quantities `m`. This monitored set MUST include the ordered-anchor contrast summaries derived from `U_{abs}^{dens}` and `Q_{src}^{dens}` and may include secondary audits such as `Q_{tgt}^{dens}`, `T_abs`, and prototype-level recurrence summaries.
- For patient `p`, coverage level `c`, and monitored quantity `m`, let `\tilde{m}_{p,c}` denote the Arm-III patient-level estimate at coverage `c`, and let `\tilde{m}_{p,100}` denote the frozen full-coverage reference estimate. Define absolute technical degradation by:

  $$
  d_{p,c}(m) = \left| \tilde{m}_{p,c} - \tilde{m}_{p,100} \right|
  $$

- Define sign consistency and floor-dominance summaries by:

  $$
  \pi_c(m) = \frac{1}{P}\sum_{p=1}^{P}\mathbf{1}\left[\operatorname{sign}\!\left(\tilde{m}_{p,c}\right) = \operatorname{sign}\!\left(\tilde{m}_{p,100}\right)\right]
  $$

  $$
  \phi_c = \frac{\#\{\text{Arm-III pseudo-ROI replicates flagged floor-dominated}\}}{\#\{\text{Arm-III pseudo-ROI replicates}\}}
  $$

- The zero-sign tie rule for `\pi_c(m)` and the audit rule used to flag a replicate as floor-dominated MUST be task-fixed before Arm-III execution and recorded in the Arm-III run metadata.
- Arm-III retention at coverage `c` requires all of the following on the monitored quantity set:
  - median absolute degradation bounded by the task-fixed tolerance `D_MAX_m`
  - sign consistency rate `\pi_c(m)` bounded below by the task-fixed minimum `PI_MIN_m`
  - floor-dominated replicate rate `\phi_c` bounded above by the task-fixed maximum `PHI_MAX`
- Ratio retention rules of the form `m_c / m_100` are not the Arm-III primary gate.
- Bootstrap CI lower bound greater than zero is descriptive only and MUST NOT be the sole Arm-III retention rule.
- Arm III does NOT add a separate frozen-block calibration resampling layer at full reference coverage in the main pipeline. The frozen 100% full-coverage reference baseline remains outside the bootstrap loop and is the sole reference target for reduced-coverage comparison.
- The rule used to supply or interpret `D_MAX_m` MUST be task-fixed before reduced-coverage interpretation and recorded in the Arm-III audit metadata without treating 100% as an additional bootstrap resampling level.

#### Uncertainty role
- Arm-III bootstrap confidence intervals are required descriptive uncertainty outputs.
- Bootstrap confidence intervals do not replace the degradation, sign-consistency, and floor-dominance retention rules.

#### Prototype stability and Balanced OT comparator
- Arm III MUST freeze a prototype audit set from the full-coverage reference before pseudo-ROI inference.
- Arm III MUST report prototype recurrence proportion, sign consistency, and correlation to the frozen 100% full-coverage reference for that locked prototype audit set.
- Balanced OT remains the same-pair, shape-only, forced-match comparator and MUST run on the exact same pseudo-ROI pairs and frozen geometry used by the Arm-III UOT layer where applicable.
- Balanced OT is comparator context only. It provides slope / forced-match reference, does not provide unmatched semantics, and is not a primary confirmatory Arm-III endpoint.

#### Required Arm-III outputs
- pseudo-ROI audit table
- density family / direction summary
- baseline scale audit summary
- degradation / sign-consistency summary
- prototype stability summary
- Balanced OT comparator summary
- one Arm-III memo stating retained findings, weakened findings, and unresolved findings

### Arm IV: Synthetic Drift
- **Action**: Inject known drift $\delta$ (offset/gain) into B.
- **Constraint**: The drift vector passed to the alignment module must be the injected ground truth $\delta$ mapped through the robust scaler, NOT estimated from sparse anchors.

## 3. Current local metric interpretation notes
- Current local Task-A output remains a temporary task-layer metrics parquet for validation and analysis only.
- In the current local Task-A implementation, `T` is transported mass.
- When Arm-II analysis summaries expose `T_abs`, it should be read as absolute transported mass.
- Under the current soft-marginal UOT semantics, `T_abs` should not be treated as an unqualified conserved matched-mass partition and is not necessarily the primary transportability readout.
- Normalized transportability quantities can be more appropriate than absolute transported mass for comparing relative transportability across tissue families or patients, but they supplement rather than replace `T_abs`.
- In the current local Task-A implementation, `D_pos` and `B_pos` are positive-part unmatched mass summaries relative to the transport marginals on the active support.
- In current Arm-II interpretation shorthand, `B` and `D` refer to the currently reported `B_pos` and `D_pos`; they are not additional separate metrics.
- In the current local Task-A implementation, `U` is the task-local unmatched-mass summary `B_pos + D_pos` on `uot_status == "ok"` rows. Historical normalized-ratio wording for `U` is not the current local Task-A implementation and must not be assumed here.
- In Arm-II interpretation, `U`, `B`, and `D` are biologically meaningful because they describe unmatched structure that Balanced OT does not natively express.
- In the current local Task-A implementation, `M` is the average transported cost per transported mass on the scaled cost domain used by the solver.
- In the current local Task-A Arm-I milestone, `R` is the tau-dependent retained transport fraction and must be interpreted only as a temporary scaffolded retention summary because `tau` is task-fixed by `compartment_a`.
- In the current local Task-A Arm-II startup implementation, `M_balanced` is the same-pair shape-only Balanced OT comparator on the current scaled cost domain.
- In current Arm-II interpretation, `M_balanced` versus `M` is useful comparator context but remains a secondary forced-match comparison with different mass semantics, not a fully like-for-like primary burden readout.
- In the current local Task-A Arm-II startup implementation, `tau` and `R` are unavailable by design and remain `NA` on ok rows.
- Primary biological readouts in current Arm-II interpretation should emphasize `U_abs`, `transport_fraction`, `unmatched_fraction`, `M`, prototype-level transport anchors, and prototype-level unmatched structure.
- Supporting unmatched decomposition quantities should emphasize `D_pos` and `B_pos`; they refine the unmatched pattern and should not be casually over-counted as fully independent evidence lines.
- Contextual / diagnostic quantities should include `T_abs`, `M_balanced`, `balanced_minus_uot`, `balanced_to_uot_ratio`, `transport_over_source_total`, and `transport_over_target_total`.
- `T_abs` remains visible but should not be treated as an unqualified conserved matched-mass partition under the current soft-marginal UOT semantics.
- `M_balanced` versus `M` remains secondary contextual comparison, not a fully like-for-like primary burden claim.
- `transport_over_source_total` and `transport_over_target_total` should not be presented as primary literal fractions under the current implementation.

## 4. Local path policy
- Project code and project docs stay in `/home/lenislin/Experiment/projects/SLOTAR`.
- Raw and project data stay in `/mnt/NAS_21T/ProjectData/SLOTAR`.
- Temporary scripts, intermediate outputs, reports, analysis products, and generated result artifacts stay in `/mnt/NAS_21T/ProjectResult/SLOTAR`.
- Current Task-A analysis artifacts should be written under `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/`.
- Repo-local generated outputs are not part of the active local layout and should not be recreated under the repository tree.

## 5. Transition note
- Arm I remains documented and closed at the current local stage.
- The active local Task-A step is to revise the Arm-II interpretation and run the integrated analysis-only follow-up described above.
- This document does not expand Arm III / IV beyond that transition boundary.
