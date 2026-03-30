# Task A Live Specification

This file is the sole live scientific Task A document.

## 1. Task A in STRIDE
STRIDE is a longitudinal spatial remodeling analysis framework centered on
patient-level open remodeling relations.

For patient `p`, the primary scientific object is `(T_p, e_p)` with
`T_p = [A_p | d_p]`.

- `A_p` is the patient-level retention/remodeling relation across a shared
  `K`-state axis.
- Canonical patient-level `A_p` is row-substochastic, with
  `sum_j A_{p,ij} + d_{p,i} = 1`.
- If exposition needs a normalized conditional kernel, it is the derived
  auxiliary object `R_{p,ij} = A_{p,ij} / (1 - d_{p,i})` when
  `1 - d_{p,i} > 0`; `A_p` remains the canonical STRIDE object.
- `d_p` is pre-side depletion tendency.
- `e_p` is post-side emergence.
- Patient-level pre/post burden vectors live on a pseudo-mass / burden scale;
  normalized compositions are derived views only.
- Cohort-level meaning is defined through recurrence across patient-level
  relations.
- FOV/ROI-level fitting remains part of the model as the observation layer.
- The Task A observation surface compares domain-stratified bags of FOV
  community-composition vectors with equal ROI/FOV mass in the current
  first pass.
- `TC`, `IM`, and `PT` are observation-layer tissue-domain strata, not part of
  canonical state identity.
- OT / Sinkhorn is an observation-layer cloud comparison tool, not the primary
  biological object.

Task A is a bounded validation task inside this remodeling-first framework. It
does not fully validate the complete `(T_p, e_p)` object in the strongest
longitudinal sense. Instead, it tests a constrained proxy question about
continuity under an ordered single-timepoint tissue setting.

## 2. Operative Data Setting
- Task A uses a single-timepoint IMC cohort rather than true longitudinal
  before/after sampling.
- The operative data surface is a frozen task-local shared-state
  representation with multi-ROI, within-patient observations on a shared state
  axis whose state identity is independent of tissue domain.
- The current Task A cohort retains 32 patients and uses tissue-domain
  observations labeled `TC`, `IM`, and `PT`.
- Nominal study language may refer to a `3 TC / 3 IM / 3 PT` per-patient
  design, but the frozen working representation retains non-nominal cases and
  is the operative truth for Task A interpretation.
- Because coverage is partial and the observation units are FOV/ROI level,
  Task A remains a proxy validation problem rather than a direct patient-level
  longitudinal event readout.

## 3. Biological Setup and Ordered Tissue-Domain Proxy
- Task A uses within-patient tissue ordering `TC`, `IM`, and `PT` as a
  biological proxy rather than as literal time ordering.
- `IM` is treated as the interface domain between `TC` and `PT`.
- The confirmatory Task A families are `TC-IM` and `TC-PT`.
- `IM-PT` may appear only as exploratory or audit context and does not carry
  the confirmatory Task A claim.
- Ordering is retained on the analysis surface, but the confirmatory
  interpretation is patient-level and cohort-level rather than row-level.
- This ordered tissue-domain proxy acts on observation-layer strata; it does
  not redefine the canonical shared `K`-state basis.
- This ordered tissue-domain proxy is useful because it can test continuity
  structure under a constrained biological ordering without claiming true
  longitudinal before/after semantics.

## 4. Central Task A Question
Task A asks the following bounded question:

> Under the ordered tissue-domain proxy `TC -> IM -> PT`, does the cohort
> support a stable, cross-patient continuity backbone, and is that backbone
> not reducible to marginal abundance summaries alone?

This is the strongest Task A question. It is narrower than validation of the
full longitudinal remodeling object and narrower than any direct claim about
true biological emergence or disappearance.

## 5. Evidence Structure
Task A is structured around three evidence blocks. They are not interchangeable.

### Block 0: Locality / Specificity Gate
- Block 0 is the entry gate for Task A interpretation.
- It asks whether the observation-layer comparison surface responds to
  locality-preserving structure rather than behaving similarly under a
  broken-locality reference.
- Passing Block 0 supports using the ordered proxy surface for Task A
  interpretation.
- Block 0 is not itself a biological mechanism claim and is
  not a direct validation of `(A_p, d_p, e_p)`.

### Block 1: Stable Continuity Backbone Evidence
- Block 1 is the only primary Task A block.
- It asks whether the confirmatory ordered proxy supports a stable
  patient-level continuity backbone in which `TC-IM` behaves as the more
  continuous family and `TC-PT` behaves as the weaker continuity comparator
  family.
- Its primary evidence lines are patient-level ordering stability and trusted
  anchor recurrence on the confirmatory surface.
- Additional continuity-supporting checks may contribute context inside
  Block 1, but they do not define another primary block.

### Block 2: Bounded Open-Channel Assignment Audit
- Block 2 is secondary.
- It audits model-based assignment outside the continuity backbone, including
  source-side depletion-prone and target-side emergence-prone structure.
- These interpretations remain bounded, candidate-level, and assignment-based.
- Block 2 does not upgrade audit quantities into direct proof of true
  disappearance or true emergence.

### 5.1 Operational Compatibility Labels
- Current task utilities still expose legacy runtime labels, but the live
  scientific interpretation follows the three evidence blocks above rather than
  the historical label scheme.
- `A1_baseline` and `A1_broken_reference` correspond to the Block 0
  locality/specificity gate.
- `A2_cross_compartment` carries the confirmatory real-data surface for Block 1
  together with the bounded Block 2 audit.
- `A3_uq_stress` is a robustness/sensitivity continuation surface only. It does
  not define a second primary biological claim and does not replace Block 1.
- These task-layer labels remain operational compatibility residue. They do not
  redefine the live scientific framing of Task A.

## 6. Incremental Value and Comparator Ontology
- Abundance summaries and static overlap can see marginal composition or burden
  on a pair surface, but they do not define a source-to-target continuity
  relation.
- Relation modeling defines which states or communities participate in the
  continuity backbone and which assignments remain outside it.
- Same-pair closed matching can force backbone-external mass into apparent
  continuity, even when a bounded residual interpretation is more appropriate.
- Same-pair Balanced OT is the required closed comparator on the same pair
  observation surface. It provides forced-match context, not the main Task A
  ontology.
- Comparator surfaces support interpretation. They do not replace the
  continuity-backbone question and do not define the primary biological claim
  of Task A.

The incremental-value statement for Task A is therefore not "more complex is
better." It is that abundance sees marginals only, while relation modeling can
separate continuity backbone from bounded residual assignment on the same
surface.

## 7. Semi-Synthetic Gain Design
The central Task A gain experiment is semi-synthetic rather than fully
synthetic.

### 7.1 Goal
- Preserve the real patients, tissue domains, shared state axis, and observed
  marginal burdens from the frozen Task A representation.
- Replace only the hidden relation truth with a known benchmark target.
- Use that benchmark target to test whether relation-aware modeling recovers
  continuity structure that abundance-only summaries cannot identify.

### 7.2 Inputs
For each patient, the semi-synthetic benchmark starts from the real frozen Task
A representation:

- `TC`, `IM`, and `PT` tissue domains,
- ROI/FOV observations,
- the shared `K`-state or community axis,
- the real source and target state/community pseudo-burdens for each pair
  family.

### 7.3 Hidden Relation Truth
For a given ordered pair, such as `TC-IM`, the benchmark defines a hidden
proxy-surface relation object `M` with the following role:

- `M` is a Task A validation object on the ordered proxy surface, not the
  canonical patient-level `A_p`,
- continuity-focused benchmark worlds may use `M` to preserve the real source
  and target pseudo-burden marginals while changing only relation structure,
- bounded-residual benchmark worlds may keep source-side depletion-prone and
  target-side emergence-prone pseudo-burden outside the matched continuity
  backbone rather than forcing everything into `M`,
- different worlds can share identical marginals while differing sharply in
  relation structure.

This is the key benchmark move: keep the real pseudo-burden marginals and only
synthesize the hidden relation truth that real data do not directly reveal.

### 7.4 Challenge A: Same Marginals, Different Relation
For a teaching example on `TC -> IM`, consider the same source and target
pseudo-burden marginals in two different hidden worlds:

```text
s = [40, 30, 20, 10]
t = [35, 25, 25, 15]

World A =
[[30,  5,  3,  2],
 [ 4, 18,  6,  2],
 [ 1,  2, 15,  2],
 [ 0,  0,  1,  9]]

World B =
[[10, 18,  8,  4],
 [20,  4,  4,  2],
 [ 5,  3,  8,  4],
 [ 0,  0,  5,  5]]
```

- World A and World B have identical source and target pseudo-burden marginals.
- World A has a stronger diagonal-like continuity backbone.
- World B reroutes the same mass into a much weaker backbone.
- Abundance-only summaries cannot distinguish these worlds because the
  marginals are identical.

For exposition, the benchmark may use a simple teaching metric such as:

```text
ContinuityScore = sum_k M[k, k] / sum_i,j M[i, j]
```

This score is not a STRIDE estimand or canonical metric. It is a transparent
demonstration that the hidden relation can change while abundance does not.

### 7.5 Cohort-Level Semi-Synthetic Benchmark
The formal benchmark should be defined across all 32 patients rather than as a
single toy case.

- Build each patient from the real frozen pseudo-burden marginals on the shared
  state axis.
- Plant stronger continuity-backbone truth in `TC-IM`.
- Plant weaker continuity-backbone truth in `TC-PT`.
- Allow backbone strength, trusted anchor identity, and residual proportion to
  vary across patients.

This cohort construction directly targets the two primary Block 1 evidence
lines:

- patient-level ordering stability,
- trusted anchor recurrence.

### 7.6 Challenge B: Forced Closure Versus Bounded Residual
The semi-synthetic benchmark must also include pairs where full closed matching
is a bad explanation.

- Some pairs should place only part of the mass into a continuity backbone.
- Remaining source-side mass should be represented as depletion-prone residual
  rather than forced continuity.
- Remaining target-side mass should be represented as emergence-prone residual
  rather than forced continuity.

This challenge is necessary because "same marginals, different relation" only
shows that abundance is insufficient. Task A must also show why a closed
relation comparator can hallucinate continuity when the benchmark truth
contains bounded residual structure.

## 8. Real-Data Execution Plan
The real-data task is not to prove the hidden truth directly. It is to mirror
the semi-synthetic logic on the operative frozen Task A cohort representation.

### 8.1 Block 0: Locality / Specificity Gate
- Compare real within-patient confirmatory pairs against broken-locality
  references rather than against random noise alone.
- These comparisons live on the same domain-stratified bag-of-FOV composition
  surface with equal ROI/FOV mass used throughout Task A.
- The broken-locality references should include at least:
  - within-patient shuffling of ROI/FOV adjacency or locality structure,
  - cross-patient composition-matched pseudo-pairs.
- A passing result means the ordered proxy surface is sensitive to
  locality-preserving structure and is therefore usable for Task A
  interpretation.
- A passing result does not establish mechanism and does not directly validate
  the full patient-level `(A_p, d_p, e_p)` object.

### 8.2 Block 1: Primary Continuity Backbone Evidence
Real-data Block 1 should focus on only two primary evidence lines.

- Patient-level ordering stability:
  the confirmatory families should support `TC-IM > TC-PT` for a majority of
  patients on the chosen continuity-backbone summaries.
- Trusted anchor recurrence:
  a subset of shared states or communities should repeatedly support the
  `TC-IM` continuity backbone across the cohort, and those anchors should not
  collapse to "simply the largest abundance groups."

Other continuity-supporting views may be reported as context, but they do not
create a second primary claim.

### 8.3 Block 2: Bounded Open-Channel Assignment Audit
Real-data Block 2 remains secondary and bounded.

- It may report candidate-level source-side depletion-prone structure and
  target-side emergence-prone structure.
- It may highlight interface-like or tumor-associated assignments when those
  patterns are biologically coherent.
- It must keep the language assignment-based, bounded, and proxy-scoped.
- It must not be written as direct proof of true biological disappearance or
  true biological emergence.

## 9. Main-Text Narrative and Supplementary Layout
The main text should center one conceptual split:

> Same abundance does not imply the same continuity relation.

### 9.1 Main-Text Figure Logic
The main figure should be compressed into three panels.

- Panel A: biological setup
  - explain `TC -> IM -> PT` as an ordered tissue-domain proxy,
  - emphasize that `IM` is the interface domain,
  - state that `TC-IM` and `TC-PT` are the confirmatory families.
- Panel B: semi-synthetic gain experiment
  - show "same marginals, different relation,"
  - show that abundance cannot distinguish the two worlds,
  - show that same-pair Balanced OT forces closure where bounded residual is
    needed,
  - show that STRIDE recovers continuity backbone plus bounded residual split.
- Panel C: real-data mirror
  - show patient-level `TC-IM > TC-PT` ordering stability,
  - show recurrent trusted anchors,
  - show one exemplar pair contrasting same-pair Balanced OT forced closure
    against STRIDE bounded residual handling.

### 9.2 Supplementary Material
Supplementary material should carry the broader but lower-priority context:

- community identity heatmaps,
- full abundance and static-overlap panels,
- robustness and sensitivity analyses,
- subsampling or discovery/test split analyses,
- retained non-nominal case sensitivity analyses,
- the full ablation suite,
- expanded Block 2 audit views,
- exploratory `IM-PT` results.

## 10. Pass Boundary
- Task A passes when Block 0 supports locality/specificity, Block 1 supports a
  stable patient-level continuity backbone with recurrent trusted anchors on
  the confirmatory families, and Block 2 remains bounded as secondary audit
  rather than promoted to primary proof.
- Task A pass is a bounded proxy-validation statement, not a full validation
  of the complete longitudinal STRIDE method.
- Task A does not require every comparator to be numerically worse on every
  scalar summary.
- Task A does not require open-channel audit structure to resolve into
  definitive biological event labels.

## 11. Non-Claim Boundary
- The strongest live Task A claim is no stronger than stable continuity
  backbone evidence under an ordered tissue-domain proxy.
- Task A does not justify language implying true temporal transition, direct
  biological disappearance, or direct biological emergence.
- Task A does not support confirmatory claims on `IM-PT`.
- Task A does not support generic claims that `UOT` is globally superior to
  `Balanced OT`.
- Comparator results cannot, by themselves, carry the primary Task A claim.
- Observation-layer transport summaries and assignment surfaces are not
  themselves the primary STRIDE scientific object.
- Task A does not turn `TC`, `IM`, or `PT` into canonical state identities.
- Task A does not by itself close the bridge from observation-layer fitting to
  full patient-level `(T_p, e_p)` validation.
