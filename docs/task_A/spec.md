# Task A Live Design Specification

This file is the top-level live Task A design document. It integrates the
former Task A rewiring plan; no standalone live rewiring document exists.

For the canonical full STRIDE method definition, use
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md).
This file must not be used to redefine STRIDE itself.

Task A remains a bounded validation and migration surface inside STRIDE. Its
historical proxy execution stack is preserved as history; current live Task A
contracts describe how Task A consumes the full STRIDE design without replacing
the method-level authority in `docs/stride_design_freeze.md`.

This revision freezes the scientific structure, claim boundaries, and execution
logic of Task A around the new `Descriptive atlas + Block 0/1/3` framing.
The Block 1 summary estimand layer and the main `TC-IM` versus `TC-PT`
comparison logic are now frozen. Remaining deferred statistical and artifact
closure items are listed explicitly in `Freeze Status and Remaining Deferred
Decisions` below.

Task A remains intentionally narrower than the ideal full STRIDE target
because it still depends on the narrow first-pass patient-relation
implementation and ordered tissue-domain family slicing, even though the
canonical Step 3 rerun now emits first-pass cohort consensus recurrence on the full
STRIDE path.

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
- `e_p` is target-side incoming open-entry tendency. It is not direct evidence
  of true biological emergence.
- Patient-level pre/post burden vectors live on a pseudo-mass / burden scale;
  normalized compositions are derived views only.
- Cohort-level meaning is defined through recurrence across patient-level
  relations.
- FOV/ROI-level fitting remains part of the model as the observation layer.
- `TC`, `IM`, and `PT` are observation-layer tissue-domain strata, not part of
  canonical state identity.
- OT / Sinkhorn is an observation-layer comparison tool, not the primary
  biological object.

Task A is a bounded validation task inside this remodeling-first framework. It
does not fully validate the complete `(T_p, e_p)` object in the strongest
longitudinal sense. Instead, it tests whether STRIDE can recover meaningful and
non-random relation structure on an ordered tissue-domain proxy defined on a
single-timepoint cohort.

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

## 3. Ordered Tissue-Domain Proxy and Central Question

- Task A uses within-patient tissue ordering `TC -> IM -> PT` as an ordered
  tissue-domain proxy rather than as literal time ordering.
- `IM` is treated as the interface domain between `TC` and `PT`.
- `TC-IM` and `TC-PT` are the confirmatory real-data families for Task A.
- `IM-PT` may appear only as exploratory or audit context and does not carry
  the confirmatory Task A claim.
- Ordering is retained on the analysis surface, but confirmatory interpretation
  remains patient-level and cohort-level rather than row-level.
- Community state construction must remain tissue-agnostic. Tissue domains act
  only on the observation surface and do not redefine the canonical shared
  `K`-state basis.

Task A asks the following bounded question:

> Under the ordered tissue-domain proxy `TC -> IM -> PT`, can STRIDE recover
> non-random relation structure, relation patterns that can support downstream
> biological interpretation, and evidence that
> these findings are not reducible to simpler baselines or accidental component
> choices?

This question is stronger than the earlier continuity-only framing, but it is
still narrower than full validation of the complete longitudinal STRIDE object
and does not justify direct temporal event claims.

## 4. Scientific Evidence Stack

The Task A scientific stack is ordered. The blocks are not interchangeable, and
downstream interpretation depends on upstream context. Block 1 reads Stage 0
plus config directly.

### 4.1 Descriptive Atlas

The descriptive atlas is the biological context layer that precedes formal
validation.

Core questions:

- Which cell subtypes dominate each community.
- How communities distribute across `TC`, `IM`, and `PT`.
- Whether each community is common or rare across the cohort.
- What representative ROI/FOV spatial morphologies look like for the main
  communities.

Main outputs:

- community-by-cell-subtype heatmaps
- community abundance / prevalence summaries across tissue domains
- representative spatial overlays
- patient-level community occurrence summaries

Role:

- The atlas provides histologic context for later `A`, `d`, and `e`-derived
  findings.
- The atlas is descriptive only and is not itself a hypothesis-testing layer.

### 4.2 Block 0: TC-IM Empirical Null Calibration

Block 0 is the empirical calibration layer for the `TC-IM` relation surface.

Biological logic:

- An ordered tissue proxy is not automatically an informative relation surface.
- Before interpreting downstream real-data patterns, Task A records whether
  the real `TC-IM` STRIDE relation structure differs from an empirical
  count-preserving permutation null that breaks the proxy relation.

Frozen scope:

- Real family: `TC-IM`
- Null family: patient-level empirical count-preserving `TC-IM` permutation
  null that permutes TC/IM labels within each patient while preserving exact
  per-patient `n_TC`/`n_IM` counts
- Hard inputs: Stage 0 h5ad plus Task A config. Prepare and descriptive-atlas
  artifacts may not be read as evidence or hard inputs.
- Formal full calibration uses `B=199` permutations. `B` remains configurable
  for diagnostics; smaller checks use smaller `B` or subsets rather than a
  separate smoke-mode contract.

Execution and analysis layers:

- execution layer: run real `TC-IM` and B permutation-null full STRIDE fits and
  persist a reusable per-fit cache over `A`, `d`, `e`, `mu_minus`, and
  `mu_plus`
- analysis layer: derive calibration tables from an existing cache without
  rerunning `fit_stride(...)`
- analysis uses a fixed Block 1-facing family-summary calibration surface over
  `self_retention`, `depletion`, `off_diagonal_remodeling`, and `emergence`;
  those outputs must remain traceable to the source execution cache

Role:

- Block 0 does not interpret biology.
- Block 0 execution emits cache/provenance artifacts only, not p-values or
  derived scientific metrics.
- Block 0 analysis reports cache-derived calibration statistics only.
  Interpretation and downstream scientific judgement are performed in later
  review stages.

### 4.3 Block 1: Real-Data Discovery

Block 1 is the descriptive real-data discovery layer for the approved Stage 0
shared community basis.

Biological logic:

- Block 1 asks what patient-level and cohort-level full STRIDE relation
  structure is observed for the prespecified real-data families `TC-IM` and
  `TC-PT`.
- `TC-IM` and `TC-PT` retain their near/far ordered-proxy design labels, but
  those labels do not authorize engineering filters or direct biological
  interpretation inside Block 1.
- Block 1 reads Stage 0 h5ad and Task A config directly. It does not read
  Block 0 outputs, descriptive-atlas outputs, result packets, or proxy-history
  artifacts.

Frozen family-level scope:

- confirmatory comparison: `TC-IM` versus `TC-PT`
- `TC-IM` is the nearer design-label family
- `TC-PT` is the farther design-label family
- `IM-PT` is not a confirmatory family in this block
- family-level estimands are exported on two frozen scales:
  `burden_weighted` and `community_mean`
- source-side summary names are fixed to:
  - `SR` for self-retention
  - `D` for source-open depletion tendency, with a burden counterpart
  - `R` for off-diagonal remodeling tendency, with a burden counterpart and
    `R_i = sum_{j != i} A_ij` at community level
- Target-side `E` remains a supportive family-level target-open accounting
  rollup. It is not evidence of true biological emergence.

Required community-level scope:

- source-community `SR_i`, `D_i`, and `R_i` with burden-scale counterparts
- target-community `target_burden`
- target-community `target_weight`, a normalized target composition helper
- target-community `matched_incoming_burden`
- target-community `open_incoming_tendency`
- target-community `open_incoming_burden`

The Block 1 core schema does not export top-k target rankings. It does not
export unweighted target incoming operator column sums. Off-diagonal
destinations remain reconstructable from native `A` and `state_ids`; an
R-friendly full source-target long CSV is deferred until after 3.6 diagnostic
validation.

Block 1 also emits a statistical supplement over the frozen comparison
surfaces. This supplement is a review surface over already exported
patient-level paired `TC-IM` versus `TC-PT` contrasts; it does not change the
core descriptive comparison contract and does not rerun STRIDE fits. The
family, source-community, and target-community supplement tables use
patient-paired Wilcoxon signed-rank tests, two-sided sign tests, BH-adjusted
`q` values within their declared surfaces, and an absolute median-delta effect
floor of `0.05`. The native relation-element supplement applies the same
paired-patient logic to `A`, `d`, and `e` elements, with separate BH correction
for `A`, `d`, and `e`. Cohort relation comparison remains a cohort-level
effect map only; it does not carry cohort-level p-values or significance
labels.

Main outputs:

- raw full STRIDE fit outputs and raw cohort recurrence/common-structure
  outputs from the execution phase
- patient-level family summaries on frozen `summary_name x scale x pair_family`
  axes
- patient-level source-community summaries on frozen
  `patient_id x pair_family x source_community` axes
- patient-level target-community summaries on frozen
  `patient_id x pair_family x target_community` axes
- descriptive `TC-IM` versus `TC-PT` contrast tables
- statistical supplement tables over the paired-patient family,
  source-community, target-community, and native relation-element surfaces

Role:

- Block 1 records observed real-data STRIDE relation structure. Its core
  comparison tables remain descriptive. Its statistical supplement may emit
  p-values, BH-adjusted `q` values, effect-floor flags, and review-candidate
  flags. It does not emit figures, descriptive community annotation tables,
  biological interpretation prose, or textual significance labels.
- Formal full execution fails fast if any expected patient/family fit is
  non-ok. Diagnostic subsets are readiness checks only and cannot become
  formal Block 1 evidence.

### 4.4 Block 3: Semisynthetic Truth-Recovery Benchmark on the Full STRIDE Reference

Block 3 is a standalone cost-based semi-synthetic benchmark in the Task A
Stage0/TC-IM context. It is not a new biology block, and it does not reopen the
Block 0/1 biological interpretation stack.

Under the Step 1 freeze, the currently existing live Block 3 packet remains
proxy execution history rather than final full-STRIDE validation closure.

The rebuilt Block 3 reference method targets the canonical full STRIDE method as
currently defined in `docs/stride_design_freeze.md`:

- patient-level relation objects `(A_p, d_p, e_p)` on the shared `K`-state
  basis
- explicit open relation through depletion and emergence rather than forced
  closure
- a patient-relation fitting path that keeps FOV/ROI structure in the model
- cohort-level common/recurrent structure defined on patient relations rather
  than on pooled observations

This is a formal reference-method target. The current repository implements the
bounded first-pass PyTorch/AdamW full-estimator support envelope, compact
successful-fit provenance, and recurrence/geometry/consistency refit switches
described in `docs/state.md`; inputs outside that envelope remain explicit
non-`ok` or compatibility-route surfaces rather than successful full-objective
fits.

Block 3 execution is driven by Stage 0 h5ad plus Task A config only. The
Block 3 runtime contract does not consume Block 0, Block 1,
descriptive-atlas, result-packet, or proxy-history artifacts as hard inputs.
Block 3 shares Stage0, TC-IM, the K-state surface, and identity-derived
geometry with Task A, but it does not validate Block 1 biological findings.
Its role is to test method behavior under bounded semi-synthetic
truth-recovery questions.

#### 4.4.1 Frozen scientific role and hypotheses

The user-approved scientific framing is binding for rebuilt Block 3.

Block 3 records two method-level benchmark questions:

- `H1`: Under the shared train-template multi-FOV semi-synthetic realization,
  how do `stride_reference` and transport-style comparator arms differ on
  relation and open-profile recovery? The benchmark route is
  `3B baseline comparison`.
- `H2`: Under the shared multi-FOV generated realization, how do consistency,
  geometry, and recurrence refit ablations differ from `stride_reference` on
  native patient-level recovery of `A_p`, `d_p`, and `e_p`? The benchmark route
  is `3C ablation study`.

Block 3 therefore evaluates method behavior under the following bounded
questions and no broader claim:

- relation and open-profile recovery under the shared train-template
  multi-FOV realization
- recovery changes after consistency, geometry, and recurrence refit ablations
- whether exported metrics warrant later interpretation against explicit
  benchmark questions

Block 3 does not:

- generate a new biology claim that supersedes Block 1
- reinterpret the descriptive atlas as proof
- redefine or overwrite canonical Block 0 or Block 1 outputs
- validate Block 1 biological findings
- create a standalone semi-synthetic discovery block

#### 4.4.2 Frozen Block 3 subexperiments

Rebuilt Block 3 is frozen as three Task A Block 3 sections:
`3A generator validation`, `3B baseline comparison`, and
`3C ablation study`. The executable subexperiment structure is:

- `3A = generator validation`
- `3B-1 = A benchmark`
- `3B-2 = d/e benchmark`
- `3C-1 = subbag consistency ablation`
- `3C-2 = geometry ablation`
- `3C-3 = recurrence ablation`

`3B` carries no-`d/e`, no-open-channel, closed, balanced, and transport-style
comparator semantics. `3C` contains only core STRIDE refit ablations under the
consistency / geometry / recurrence ordering above.
This section structure is exhaustive for the rebuilt Block 3 public surface.

Hard inputs: Stage 0 h5ad and Task A config only.

Block 3 must not read Block 0, Block 1, descriptive-atlas outputs,
result packets, or preserved proxy-history artifacts as hard inputs. Result
interpretation is deferred until exported metrics are reviewed against explicit
benchmark questions, but execution inputs are Stage 0 plus config only.

Block 3 derives community identity vectors `g_k` internally from the Stage 0
shared-state/cell-subtype surface. The descriptive atlas may explain community
meaning but is not a hard input.

`block3.benchmark_pair_family: "TC-IM"` is the single explicit Task A config
field for the Block 3 benchmark family. Current validation accepts only
`TC-IM`.

The rebuilt benchmark now uses one shared hidden-program design:

- a real-data-derived baseline composition `x_p` on the shared `K`-state axis
- a hidden patient program `(A_p, d_p, e_p)` used only as the latent generator
- a method-facing target composition `y_p` derived from `(x_p, A_p, d_p, e_p)`
- endpoint-only baselines consume deterministic endpoint projections of the
  generated multi-FOV observations on the same shared `K`-state axis
- `stride_reference` and STRIDE ablation arms consume generated source/target
  FOV observations
- Task A resolves source/target endpoint comparisons and valid domain strata,
  then passes the resolved comparison plan and source/target observation
  evidence blocks to the formal `fit_stride(...)` surface for
  `stride_reference`
- Task A adapters may perform input conversion and comparison-plan
  instantiation; they must not substitute a task-local observation solver or
  task-local STRIDE estimator. Domain resolution remains task-layer
  provenance and does not become a core loss/state/relation/recurrence axis.
- If `fit_stride(...)` emits compact successful-fit provenance for
  `stride_reference`, Task A receives and preserves that full-estimator
  provenance as provenance attached to the reference fit. Task A does not
  expand it into a per-patient or per-evidence-block status hierarchy.

No ranked benchmark method, including `stride_reference`, may read hidden
`(A_p, d_p, e_p)` or any additional non-public truth companion. Deterministic
endpoint projections and generated FOV observations are the only method-facing
inputs in Block 3. Any exact truth companion may exist only as an internal
scoring/debugging object and must not appear as a ranked benchmark arm.

The detailed alignment note for this transitional redesign lives in
[`docs/task_A/block3/scientific_contract.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/scientific_contract.md).

The rebuilt benchmark is additionally frozen around one held-out outer design:

- each rerun uses a patient-level `24 train / 8 test` split
- `train` patients contribute only weak empirical calibration from real
  `TC -> IM` change
- `test` patients contribute only their real `TC` carrier `x_p`
- generator reruns are repeated `10` times and remain the outer statistical
  unit

To keep the semi-synthetic benchmark scientifically interpretable rather than
turning it into a large hyperparameter surface, Block 3 uses one
train-template multi-FOV generator realization shared by `3A`, `3B`, and
`3C-*`:

- train TC-IM endpoint residuals are decomposed into geometry-gated
  off-diagonal remodeling and open residuals using the normalized state cost
  matrix with fixed generator gate `tau=2.0`
- each train patient contributes a row-simplex template
  `[A^(q) | d^(q)]` plus an emergence-shape template `s^(q)`
- source rows that are not identifiable in an individual train template are
  filled from train-row pooled templates, with diagonal no-remodeling fallback
  only when no train patient identifies that row
- held-out patient truth mixes the train-bank medoid template with one sampled
  individual train template using `lambda_individual=0.10`
- the held-out patient's open emergence vector is
  `e_p = (x_p * d_p).sum() * s_p`, where `s_p` is the medoid/individual
  emergence-shape mixture
- generated source FOVs are deterministic shrinkage mixtures of the held-out
  source endpoint and the real held-out source FOVs with `eta=0.3`
- generated target FOVs are patient-level truth projections from generated
  source-FOV anchors with no target noise
- benchmark comparison and ablation study reuse the same rerun-specific
  patient-level semi-synthetic realizations; generator settings are recorded
  for provenance and diagnostics, not promoted to public benchmark axes
- each shared community `k` receives a frozen community-identity vector `g_k`
  computed internally from the Stage 0 shared-state/cell-subtype surface;
  `g_k` retains the `UNKNOWN` subtype with no removal or reweighting in the
  main contract
- the shared-axis biological cost precursor is
  `C_raw[i,j] = sqrt(JS(g_i, g_j))`, with `C_raw[i,i] = 0`
- let `s_C` be the median of the positive off-diagonal entries of `C_raw`;
  the frozen normalized cost matrix is `C = C_raw / s_C`
- `C` is derived once from the frozen upstream community-identity surface and
  reused both as the shared-axis biological neighborhood for relation support
  and as the comparator-facing OT cost matrix; it is not a rerun-specific or
  train-estimated pairwise topology
- for `stride_reference`, `C_raw`, `s_C`, and `C` are the Task A
  adapter/benchmark cost source passed into the full-estimator shared-state
  cost contract; they do not define a task-local geometry prior
- the full estimator owns the `L_geometry` contract over raw canonical `A_p`;
  Task A mirrors that contract for `stride_reference` and
  `geometry_ablation`

The train-derived templates are generator objects, not biological truth claims
and not outputs of STRIDE or any comparator. Their role is to create a known
hidden `(A_p, d_p, e_p)` surface with controlled cohort structure while
preserving the method-facing input boundary. The formal Block 3 contract does
not expose additional generator-grid axes.

##### 3A. Generator validation

Manual sanity-check question:

- Do the exported generator-validation raw and review surfaces show finite
  metrics, no evident real/synthetic target-surface anomaly, and no evident
  uncontrolled rerun variability before `3B/3C` are run?

Generator-validation surface:

- `3A` exports generator-validation raw and review surfaces for manual
  inspection before running `3B/3C`; it is not a formal pass/fail gate
- `3A` exports the same rerun-specific patient-level semi-synthetic
  realizations that are later consumed by `3B` and `3C`
- additional generator strata are not part of the public `3A` contract
- if implementation exports descriptive slices by realized generator
  quantities, those slices are diagnostics only and are not section-defining
  benchmark axes
- `3A` has two frozen formal validation objects on the held-out `test`
  cohort of rerun `r`:
  - the `community-space` target fraction surface
    `S_real^(r) = mean_{p in test_r} q_{p,IM}` versus
    `S_syn^(r) = mean_{p in test_r} y_p`
  - the `g_k`-projected identity-aware target fraction surface
    `B_real^(r) = sum_k S_real^(r)[k] g_k` versus
    `B_syn^(r) = sum_k S_syn^(r)[k] g_k`

Formal metrics and stability summary:

- `Pearson correlation`, `MAE`, `MSE`, and `JS divergence` are computed on the
  held-out `community-space` object pair
  `(S_real^(r), S_syn^(r))`
- the same four metrics are computed on the held-out `g_k`-projected
  identity-aware object pair `(B_real^(r), B_syn^(r))`
- `rerun variability` means the between-rerun stability of those two
  cohort-level realism / plausibility objects and their associated metric
  summaries; it is not a detached third comparator axis

Interpretive role:

- `3A` is a manual generator sanity check and has no thresholded gate,
  pass/fail label, or standalone scientific conclusion
- `3A` reviews generator outputs in the same fraction-space semantic layer used
  by the rest of Task A, not method superiority, not patient reconstruction,
  and not a forecasting-style accuracy benchmark
- intuitive `real TC -> IM` versus `synthetic TC -> y` change language may
  still be used for explanation, but the formal contract is the held-out
  cohort `community-space` target fraction surface and the corresponding
  `g_k`-projected identity-aware surface
- `3A` does not claim to isolate or prove the marginal contribution of
  train-derived template structure; template identity, row-imputation mass,
  endpoint closure, and evidence-block counts remain diagnostics rather than
  public `3A` axes
- compatibility-era sanity exports may still be emitted when useful for packet
  continuity, but `3A` has no dedicated formal null/random baseline contract;
  such exports are optional sidecars only, not new sections, comparators, or
  formal requirements

##### 3B. Baseline comparison

Scientific role:

- `3B` is the external baseline-comparison umbrella section under shared
  rerun-specific semi-synthetic comparison conditions.
- Its live scientific contract is split into:
  `3B-1 A benchmark` and `3B-2 d/e benchmark`.

###### 3B-1. A benchmark

Scientific question:

- Under shared rerun-specific semi-synthetic comparison conditions, how do full
  STRIDE and the transport-family comparators differ on recovery of the shared
  relation surface `A_p` from their generated method-facing inputs?

Reference versus baselines:

- reference: `stride_reference`
- baseline: `balanced_ot_baseline`
- baseline: `uot_baseline`
- baseline: `partial_ot_baseline`
- baseline: `diagonal_transport_baseline`

Formal metrics:

- `F_L1_total`
- `g_L1_total`
- `e_L1_total`
- `offdiag_mass_abs_error`
- `depletion_mass_abs_error`
- `emergence_mass_abs_error`
- `offdiag_ratio`
- `depletion_capture`
- `emergence_capture`
- `endpoint_y_MAE`
- `A_MAE_active`
- `A_MSE_active`
- `target_recall_at_k`
- `open_support_F1`
- `d_MAE`
- `d_MSE`
- `e_MAE`
- `e_MSE`

Metric rules:

- `3B-1` uses one shared generated condition,
  `a_benchmark_shared_realization_set`
- the fixed shared-axis cost matrix `C`, derived from community-identity
  vectors `g_k`, defines the comparator-facing OT cost for
  `balanced_ot_baseline`, `uot_baseline`, and `partial_ot_baseline`
- the same rerun-specific held-out patients, generated endpoint projections,
  generated FOV observations, and hidden patient-level truth are reused across
  all `3B-1` methods
- off-diagonal relation structure is inherited from the train-template
  generator and is recorded through diagnostics and provenance
- transported-mass scoring is truth-anchored:
  `T_true[i,j] = x_true[i] * A_true[i,j]` and
  `T_hat[i,j] = x_true[i] * A_hat[i,j]`
- `A_MAE_active` and `A_MSE_active` score active-row conditional target-pattern
  recovery on truth-active source rows only; they are not total
  transported-mass error metrics
- `target_recall_at_k` scores truth-anchored off-diagonal transported-mass
  target-priority recovery rather than generic endpoint reconstruction
- `open_support_F1` is the support-level recovery metric on burden-scale
  depletion/emergence carriers; `d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE` are the
  corresponding quantitative open-profile fidelity metrics
- `target_recall_at_k` is `not_applicable` for any patient whose hidden truth
  contains no off-diagonal target set

Interpretive role:

- `3B-1` is the relation-surface benchmark for `A_p`
- the phase-1 engineering contract is to compute and export the frozen
  shared 18-metric vocabulary for each evaluated method and rerun
- judgments about competitiveness, acceptable loss, or headline ranking are
  deferred to post-processing and result interpretation
- this is the relation-recovery benchmark arm for `H1`

###### 3B-2. d/e benchmark

Scientific question:

- Under the same rerun-specific semi-synthetic comparison conditions, does
  full STRIDE recover `d_p` and `e_p` more faithfully than open-comparator
  baselines when all methods are evaluated through one shared analysis layer?

Reference versus baselines:

- reference: `stride_reference`
- baseline: `uot_baseline`
- baseline: `partial_ot_baseline`
- baseline: `diagonal_transport_baseline`

`balanced_ot_baseline` is intentionally excluded from `3B-2`. Its closed
marginal constraints force `P 1 = x` and `P^T 1 = y`; under the shared
`P -> A/d/e` analysis layer this implies `d_hat = 0` and `e_hat = 0` by
construction. It is therefore retained only in `3B-1` as a closed relation
comparator, not as an open-profile comparator. `diagonal_transport_baseline`
remains the simple transport negative-control baseline for `3B-2` because its
unmatched residuals produce auditable `d_hat/e_hat` surfaces.

Formal metrics:

- `F_L1_total`
- `g_L1_total`
- `e_L1_total`
- `offdiag_mass_abs_error`
- `depletion_mass_abs_error`
- `emergence_mass_abs_error`
- `offdiag_ratio`
- `depletion_capture`
- `emergence_capture`
- `endpoint_y_MAE`
- `A_MAE_active`
- `A_MSE_active`
- `target_recall_at_k`
- `open_support_F1`
- `d_MAE`
- `d_MSE`
- `e_MAE`
- `e_MSE`

Metric rules:

- `3B-2` reuses the same rerun-specific held-out patient identities, endpoint
  fractions, and hidden truth construction flow used by `3B-1`
- in `3B-2`, reused means shared patient identities, shared `x_p`, shared
  generated source/target FOV observations, shared endpoint projections,
  shared `C`, shared train-template provenance, and shared hidden
  patient-level `(A_p, d_p, e_p)` truth
- each method first emits its native matched/unmatched representation, and the
  shared `3B` analysis layer then derives the common `A/d/e` scoring surfaces
- `3B-2` is open-focused, but it retains the full relation, open-channel, mass,
  and endpoint metric vocabulary for direct comparison with `3B-1` and `3C-*`
- `open_support_F1` is the support-level recovery metric on the derived
  burden-scale depletion/emergence carriers
- `d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE` are the quantitative profile-recovery
  metrics on the same derived open surfaces
- `3B-2` reuses the full shared `3C` `open_support_F1` contract, including
  the burden-scale support definition and the channel-level / patient-level
  status semantics
- `profile TV` may still be exported as a diagnostic sidecar, but it is not a
  phase-1 headline `3B-2` metric

Interpretive role:

- `3B-2` is the open-surface benchmark for the shared analysis-layer `d/e`
  recovery problem
- its fixed evidential target is whether STRIDE provides more accurate open
  support and quantitative open-profile recovery than the open comparator set
  under matched rerun-specific realizations
- it is an analysis-layer comparison of recovered `d/e` surfaces and is not a
  biological-truth-equivalence claim about literal disappearance or emergence

##### 3C. Ablation study

Scientific question:

- Which frozen STRIDE modules remain necessary once generator validation and
  external baseline comparison are fixed?

Interpretive role:

- `3C` carries ablation claims only and does not act as an external baseline
  section
- all `3C` method arms are loss/regularization-level STRIDE reruns that
  remove or zero one core objective term and then refit `A_p`, `d_p`, and
  `e_p`
- the core `3C` ablation set is subbag consistency, geometry, and recurrence
- all `3C` ablations use the three-block reference objective with the ablated
  term set to zero and do not reweight retained objective terms
- Each `3C` arm is an independent refit of `A_p`, `d_p`, `e_p` under the
  corresponding `ablation_mode`. The arm must not mask reference output, must
  not perform post-hoc rescoring only, and must preserve fixed denominators
  without reweighting retained objective terms.
- each `3C` arm uses the same rerun-specific multi-FOV generated realization,
  resolved evidence blocks, deterministic initialization, and optimizer
  protocol as the reference fit; the only objective change is the corresponding
  consistency, geometry, or recurrence term removal or zero-weighting
- `3C` reuses the same generated realization for all three ablations
- three-block ablation objective semantics are:
  `geometry_ablation` uses
  `L_prior = mean(normalized_L_open, 0)`,
  `recurrence_ablation` uses
  `L_total = mean(L_fit, L_prior, 0)`, and
  `consistency_ablation` uses
  `L_fit = normalized_L_obs + 0`
- `3C` arms must not be implemented by masking `stride_reference` outputs or
  by post-hoc rescoring only
- no-`d/e`, open-channel-removal, closed, balanced, and transport-style
  comparisons belong to `3B` baseline/comparator semantics rather than to
  core STRIDE ablations
- the current repository proxy Block 3 ablation implementation is preserved as
  historical/proxy execution context only and is not the normative `3C`
  scientific contract

###### 3C-1. Subbag consistency ablation

Scientific question:

- What is lost when the subbag consistency term is removed from
  the STRIDE objective and the patient relation is refit?

Reference versus ablation:

- reference: `stride_reference`
- ablation: `consistency_ablation`

Ablation semantics:

- `3C-1` is a within-STRIDE module-necessity test, not an external baseline
  comparison and not a generator-gradient benchmark
- `3C-1` display name is `subbag consistency ablation`
- the method key remains `consistency_ablation`
- the core `ablation_mode` is `consistency`
- `consistency_ablation` removes or zeroes only the subbag consistency term in
  the objective
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- geometry/locality terms are retained
- cohort recurrence/common-structure terms are retained
- retained terms keep the same objective coefficients as the reference fit;
  the retained terms are not reweighted
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the ablation must not be implemented by masking `stride_reference` outputs
  or by post-hoc rescoring only
- the comparison reuses the same rerun-specific multi-FOV generated realization
  used across Block 3 and does not introduce a separate public generator axis

###### 3C-2. Geometry ablation

Scientific question:

- What is lost when the geometry/locality prior is removed from the STRIDE
  objective and the patient relation is refit?

Reference versus ablation:

- reference: `stride_reference`
- ablation: `geometry_ablation`

Ablation semantics:

- `geometry_ablation` removes or zeroes only the geometry/locality term in the
  objective
- `stride_reference` and `geometry_ablation` use the full-estimator
  `L_geometry` contract over raw canonical `A_p`
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- subbag consistency terms are retained
- cohort recurrence/common-structure terms are retained
- retained terms keep the same objective coefficients as the reference fit;
  the retained terms are not reweighted
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the ablation must not be implemented by masking `stride_reference` outputs
  or by post-hoc rescoring only
- the comparison reuses the same rerun-specific multi-FOV generated realization
  used across Block 3 and does not introduce a separate public generator axis

###### 3C-3. Recurrence ablation

Scientific question:

- What is lost when cohort recurrence/common-structure feedback is removed from
  the STRIDE objective and the patient relation is refit?

Reference versus ablation:

- reference: `stride_reference`
- ablation: `recurrence_ablation`

Ablation semantics:

- `recurrence_ablation` removes or zeroes only the cohort
  recurrence/common-structure term in the objective
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- geometry/locality terms are retained
- subbag consistency terms are retained
- retained terms keep the same objective coefficients as the reference fit;
  the retained terms are not reweighted
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the ablation must not be implemented by masking `stride_reference` outputs
  or by post-hoc rescoring only
- the comparison reuses the same rerun-specific multi-FOV generated realization
  used across Block 3 and does not introduce a separate public generator axis

Formal metrics:

- `F_L1_total`
- `g_L1_total`
- `e_L1_total`
- `offdiag_mass_abs_error`
- `depletion_mass_abs_error`
- `emergence_mass_abs_error`
- `offdiag_ratio`
- `depletion_capture`
- `emergence_capture`
- `endpoint_y_MAE`
- `A_MAE_active`
- `A_MSE_active`
- `target_recall_at_k`
- `open_support_F1`
- `d_MAE`
- `d_MSE`
- `e_MAE`
- `e_MSE`

Metric rules:

- `open_support_F1` is defined on burden-scale open carriers rather than on raw
  `d/e > threshold` masks
- for each patient and open channel (`depletion`, `emergence`), normalize the
  burden carrier to unit mass and define support as the smallest state set
  covering at least 95% of that normalized mass; ties break by increasing
  shared-axis state id
- compute channel-level support F1 on those support sets
- if one open channel has truth total mass `= 0`, that channel-level
  `open_support_F1` is `not_applicable`
- patient-level `open_support_F1` is the arithmetic mean across the reported
  channel-level values
- if both open channels are `not_applicable`, patient-level
  `open_support_F1` is `not_applicable`
- `open_support_F1` is the support-level recovery metric for the burden-scale
  depletion/emergence carriers; it should not be read on its own as full
  open-profile fidelity
- `d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE` remain reported throughout the
  `3C` evaluation surface as
  quantitative profile-recovery metrics for the depletion/emergence magnitude
  profiles
- `A_MAE_active` and `A_MSE_active` are the minimum patient-level relation
  recovery metrics in `3C` and reuse the truth-anchored active-row conditional
  target-pattern definition; here `A` always means the patient-level relation
  operator `A_p` on the shared `K`-state axis
- any shared cohort/common-structure generator quantity remains a train-derived
  internal generator detail recorded for reproducibility only, not a public
  benchmark axis
- here `shared` means that all held-out `test` patients in one rerun reuse the
  same train-derived template bank and medoid, while each patient's realized
  hidden truth still depends on that patient's own `x_p` and sampled
  individual template
- `stride_reference` and each `3C` ablation arm are compared on the same
  held-out patients and the same native truth outputs under matched
  rerun-specific realizations
- `open_support_F1` remains a native patient-level support-recovery metric and
  does not by itself imply full profile fidelity
- metric reporting must distinguish `reported`, `not_applicable`, and
  `not_estimable`

Interpretive role:

- `3C` is centered on loss of native patient-level `A_p`, `d_p`, and `e_p`
  recovery after consistency, geometry, or recurrence terms are removed and the
  estimator is refit
- `3C` does not create a separate public sensitivity axis over shared
  train-derived cohort-level generation and therefore does not by itself
  establish robustness across multiple cohort/common-structure generation
  settings
- only the native patient-level metrics listed above belong to the live `3C`
  metric contract
- patient-level helper quantities may still be exported as diagnostics, but
  they are not part of the formal Block 3 metric contract

#### 4.4.3 Frozen comparator registry

The rebuilt Block 3 method registry is frozen as follows.

`stride_reference`

- Uses the canonical full STRIDE method with patient relation, explicit open
  relation, patient-relation fitting, and cohort consensus recurrence/common
  structure.
- Calls the formal `fit_stride(...)` frozen reference configuration on
  Task A-resolved source/target endpoint comparison evidence blocks and the
  resolved comparison plan; Task A adapters only convert inputs and instantiate
  the comparison plan.
- Uses the canonical full-estimator observation discrepancy operator
  `D_obs^BalancedSinkhornDivergence-v1`; UOT is not the reference operator and
  appears only through the `3B` comparator/legacy diagnostic surface.
- Emits native fitted `A/d/e` and preserves compact successful-fit provenance
  from `fit_stride(...)` when that provenance is emitted.
- This registry entry is a formal method target. Successful `stride_reference`
  runs must preserve the full-estimator provenance/status emitted by
  `fit_stride(...)`; unsupported or compatibility-route outputs are not a
  replacement definition of `stride_reference`.
- This is the only reference method for Block 3.

`balanced_ot_baseline`

- Uses the exact balanced OT comparator on paired endpoint fractions over the
  shared `K`-community axis.
- It is the closed exact OT comparator on the fixed shared cost matrix `C`.
- For each patient, solves `P* = argmin_P <P, C>` subject to
  `P 1 = x_p`, `P^T 1 = y_p`, and `P >= 0`.
- The native method output is the matched plan `P*`, which is later converted
  by the shared `3B` analysis layer into `A/d/e`.

`uot_baseline`

- Uses a soft-unbalanced transport comparator on the same paired endpoint
  fractions and fixed shared cost matrix `C`.
- The rerun-specific `24`-patient `train` split calibrates one shared
  `lambda_match` value from endpoint-fraction statistics and then reuses it
  across the shared `3B-1` and `3B-2` realizations and all `8` held-out
  `test` patients in that rerun.
- Uses the frozen internal lambda grid
  `[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`. Boundary selection is reported as
  native metadata through `boundary_hit`; it is a diagnostic and does not
  trigger automatic grid expansion before the formal Block 3 run.
- The native method output is the matched plan `P`, which is later converted
  by the shared `3B` analysis layer into `A/d/e`.

`partial_ot_baseline`

- Uses an exact fixed-mass partial optimal transport comparator on the same
  paired endpoint fractions and fixed shared cost matrix `C`.
- The rerun-specific `24`-patient `train` split calibrates one shared
  `matched_mass_budget = mean(sum(min(x_train, y_train)))` from real train
  endpoint profiles.
- That budget is reused across the shared `3B-1` and `3B-2` realizations and
  all `8` held-out `test` patients in that rerun.
- For each test patient, the requested budget is clipped only to the feasible
  source/target mass bound.
- The native method output is the exact partial OT matched plan `P`, solving
  `min <P, C>` subject to `P 1 <= x`, `P^T 1 <= y`, `P >= 0`, and
  `sum(P) = effective_budget`.
- The shared `3B` analysis layer then derives `A/d/e` from `P`; no hidden
  `A/d/e` truth is exposed to the comparator.

`diagonal_transport_baseline`

- Uses strict diagonal matched transport plus residual open mass on the paired
  endpoint fractions.
- The native method output is the identity-only diagonal matched plan `P`,
  with `P[i,i] = min(x[i], y[i])`, which is later converted by the shared `3B`
  analysis layer into `A/d/e`.

No-`d/e`, no-open-channel, closed, balanced, and transport-style comparison
semantics belong to this `3B` comparator layer. They do not define core STRIDE
ablation arms.

Comparator implementation minimums:

```python
def run_diagonal_transport(x, y):
    P = zeros((k, k))
    for i in range(k):
        P[i, i] = min(x[i], y[i])
    return P
```

```python
def run_uot_baseline(x, y, C, lambda_match, solver_cfg):
    P = solve_uot_plan(
        source=x,
        target=y,
        cost=C,
        match_penalty=lambda_match,
        cfg=solver_cfg,
    )
    r = row_sum(P)
    c = col_sum(P)

    A = zeros_like(P)
    for i in range(k):
        if x[i] > 0:
            A[i, :] = P[i, :] / x[i]

    d = zeros(k)
    for i in range(k):
        if x[i] > 0:
            d[i] = max((x[i] - r[i]) / x[i], 0.0)

    e = maximum(y - c, 0.0)
    return A, d, e, P
```

```python
def run_partial_ot_baseline(x, y, C, matched_mass_budget):
    effective_budget = min(matched_mass_budget, sum(x), sum(y))
    P = ot.partial.partial_wasserstein(
        x,
        y,
        C,
        m=effective_budget,
    )
    return derive_A_d_e_from_plan(x=x, y=y, P=P)
```

Train-side shared calibration minimums:

- `uot_baseline`: estimate one rerun-shared `lambda_match` from the `24`
  `train` patients using endpoint-fraction statistics and reuse that same
  scalar across the shared `3B-1` and `3B-2` realizations and all `test`
  patients in that rerun
- `partial_ot_baseline`: estimate one rerun-shared `matched_mass_budget` from
  the `24` `train` patients using endpoint-fraction statistics and reuse that
  same scalar across the shared `3B-1` and `3B-2` realizations and all `test`
  patients in that rerun
- the shared calibration target is the train-side mean endpoint overlap proxy
  on the shared `K`-community axis

```python
def observed_overlap_fraction(x, y):
    return float(np.sum(np.minimum(x, y), dtype=float))
```

```python
def calibrate_uot_lambda(train_patients, C, lambda_grid, solver_cfg):
    target_overlap = mean(
        observed_overlap_fraction(p.x, p.y) for p in train_patients
    )

    best_lambda = None
    best_error = inf
    for lam in lambda_grid:
        predicted_matched_fraction = mean(
            float(np.sum(solve_uot_plan(p.x, p.y, C, lam, solver_cfg), dtype=float))
            for p in train_patients
        )
        error = abs(predicted_matched_fraction - target_overlap)
        if error < best_error:
            best_error = error
            best_lambda = lam

    return best_lambda
```

```python
def calibrate_partial_ot_budget(train_patients):
    return mean(
        observed_overlap_fraction(p.x, p.y) for p in train_patients
    )
```

`consistency_ablation`

- Removes or zeroes only the subbag consistency term and refits `A_p`, `d_p`,
  and `e_p`.
- Uses `L_fit = normalized_L_obs + 0` and does not reweight retained
  objective terms.
- Preserves observation discrepancy, explicit open-channel terms,
  geometry/locality, cohort recurrence/common structure, and
  audit/plausibility handling.
- This is the fixed `3C-1 subbag consistency ablation` arm.

`geometry_ablation`

- Removes or zeroes only the geometry/locality term and refits `A_p`, `d_p`,
  and `e_p`.
- Uses the full-estimator `L_geometry` contract over raw canonical `A_p` for
  the reference arm and removes or zeroes that term in the ablation arm.
- Uses `L_prior = mean(normalized_L_open, 0)` and does not reweight retained
  objective terms.
- Preserves observation discrepancy, explicit open-channel terms, subbag
  consistency, cohort recurrence/common structure, and
  audit/plausibility handling.
- This is the fixed `3C-2 geometry ablation` arm.

`recurrence_ablation`

- Removes or zeroes only the cohort consensus recurrence/common-structure term and
  refits `A_p`, `d_p`, and `e_p`.
- Uses `L_total = mean(L_fit, L_prior, 0)` and does not reweight retained
  terms.
- Preserves observation discrepancy, explicit open-channel terms,
  geometry/locality, subbag consistency, and audit/plausibility handling.
- This is the fixed `3C-3 recurrence ablation` arm.

#### 4.4.4 Embedded semi-synthetic companion design

Semi-synthetic benchmark conditions remain embedded inside the relevant Block 3
subexperiments. There is no standalone Block 3E.

All Block 3 companion conditions must expose the same hidden-truth interface:

- patient-level truth: `(A_p, d_p, e_p)`
- patient-level generated endpoint projections and generated source/target FOV
  observations
- section-specific native truth outputs needed for scoring on the live metric
  contract
- section-specific internal helper quantities may exist for scoring/debugging,
  but the public truth interface is determined by the live metric contract
- any exported `pair_family` / `evaluation_family` labels are reporting-only
  containers and are not part of the formal patient-level truth/scoring
  contract

Native representation and `3B` analysis layer:

- `stride_reference` provides native `A / d / e`
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` provide native matched plan `P`
- the shared `3B` analysis layer derives the common scoring surfaces
  `A / d / e` from each method's native representation before applying
  `3B-1` or `3B-2` metrics

```python
def derive_surfaces_from_native(native, x, y):
    if native.kind == "plan":
        P = native.P
        r = row_sum(P)
        c = col_sum(P)

        A = zeros_like(P)
        for i in range(k):
            if x[i] > 0:
                A[i, :] = P[i, :] / x[i]

        d = zeros(k)
        for i in range(k):
            if x[i] > 0:
                d[i] = max((x[i] - r[i]) / x[i], 0.0)

        e = maximum(y - c, 0.0)
        return A, d, e

    if native.kind == "stride_native":
        return native.A, native.d, native.e
```

The frozen companion-design flow is:

1. derive the held-out source endpoint `x_p` from real TC source FOVs
2. build train-derived templates from real train TC-IM endpoint residuals using
   the fixed geometry gate `tau=2.0`
3. fill unidentifiable source rows through train-row pooled templates
4. choose the train-bank medoid and sample one individual train template for
   each held-out patient
5. mix medoid and sampled individual templates with
   `lambda_individual=0.10` to define hidden `(A_p, d_p, s_p)`
6. set `e_p = (x_p * d_p).sum() * s_p` and
   `y_p = normalize(x_p A_p + e_p)`
7. generate source FOVs by deterministic shrinkage toward real held-out source
   FOVs with `eta=0.3`
8. generate target FOVs by projecting generated source-FOV anchors through the
   hidden patient-level truth with no target noise
9. expose generated endpoint projections to endpoint-only baselines and
   generated source/target FOV observations to STRIDE reference/ablation arms
10. evaluate recovered objects against hidden truth only after inference

The open and off-diagonal relation burdens are realized by the train-template
generator and recorded through diagnostics and provenance.

`3B-2` preserves the following quantities within each rerun:

- held-out patient identities
- generated source and target FOV observations
- hidden patient-level `(A_p, d_p, e_p)` truth
- `C`
- train-template provenance

The frozen companion condition families are:

- held-out cohort-level `3A` summaries on the `community-space` target
  fraction surface, the `g_k`-projected identity-aware target fraction
  surface, and rerun stability of those same objects
- `3B-1`: the shared generated realization scored with the `A` benchmark
  comparator set and full shared metric vocabulary
- `3B-2`: the shared generated realization scored with the open-focused
  comparator set and full shared metric vocabulary
- rerun-specific patient-level semi-synthetic realizations reused for all
  `3C` refit ablations
- rerun-specific patient-level semi-synthetic realizations with
  cohort/common-structure present in the generator reused for the recurrence
  ablation arm

Compatibility-era sanity exports may still be emitted for packet continuity,
but `3A` has no dedicated formal null/random baseline contract. Any such
export is optional sidecar context only, not a required section, comparator
arm, or benchmark axis.

Generator validation is continuous and descriptive inside `3A`:

- `community-space` cohort realism:
  `Pearson correlation`, `MAE`, `MSE`, and `JS divergence` on held-out real
  `q_IM` versus synthetic `y` target fraction surfaces
- `g_k`-projected identity-aware biological plausibility:
  the same four metrics on the corresponding held-out projected cohort
  surfaces
- `rerun variability`:
  between-rerun stability of the above realism / plausibility objects and
  their score summaries, rather than a standalone bare metric name

Generator reruns are the intended outer statistical unit for Block 3
comparison. Patients are nested within one generator realization, and
`patient × rerun` observations must not be described as IID.

The outer rerun design remains frozen to `10` reruns with `24 train / 8 test`
patients in each rerun.

#### 4.4.5 Frozen Block 3 metric contract

Block 3 must continue to reuse the current Task A real-data summary language
for raw supportive exports, but the canonical benchmark contract is now the
fixed generator-validation set plus one shared method-bearing metric vocabulary.

`3A`

- formal objects: held-out cohort `community-space` target fraction surfaces
  and their `g_k`-projected identity-aware target fraction surfaces
- formal metrics: `Pearson correlation`, `MAE`, `MSE`, and `JS divergence`
  applied to each object pair, plus `rerun variability` as the between-rerun
  stability summary for those same object-level validations
- reporting is descriptive across generator reruns and does not use paired
  method-difference tables
- `3A` has no dedicated null/random comparator requirement; any compatibility
  sanity export remains optional sidecar context only

Shared method-bearing metrics for `3B-1`, `3B-2`, and `3C-*`:

- `F_L1_total`
- `g_L1_total`
- `e_L1_total`
- `offdiag_mass_abs_error`
- `depletion_mass_abs_error`
- `emergence_mass_abs_error`
- `offdiag_ratio`
- `depletion_capture`
- `emergence_capture`
- `endpoint_y_MAE`
- `A_MAE_active`
- `A_MSE_active`
- `target_recall_at_k`
- `open_support_F1`
- `d_MAE`
- `d_MSE`
- `e_MAE`
- `e_MSE`

`3B-1`

- formal metrics: the shared method-bearing vocabulary above
- engineering-facing rerun summary rows are stratified by generated
  realization, method name, and metric name
- the minimum fixed `3B-1` summary cell is `(method_name, metric_name)` within
  a generator rerun and shared realization
- mean, 95% bootstrap CI, and paired difference versus `stride_reference`
  are post-processing summaries computed from those fixed-cell rerun summaries
- `A_MAE_active` and `A_MSE_active` remain active-row conditional
  target-pattern recovery metrics rather than total transported-mass error
  metrics
- `target_recall_at_k` remains a truth-anchored off-diagonal target-priority
  metric and is `reported` only when truth off-diagonal targets exist
- `target_recall_at_k` is `not_applicable` for patient-level truth rows with
  no off-diagonal target set

`3B-2`

- formal metrics: the shared method-bearing vocabulary above
- engineering-facing rerun summary rows are stratified by generated realization,
  method name, and metric name
- the minimum fixed `3B-2` summary cell is `(method_name, metric_name)` within a
  generator rerun and shared realization
- mean, 95% bootstrap CI, and paired difference versus `stride_reference`
  are post-processing summaries computed from those fixed-cell rerun summaries
- `open_support_F1` is the support-level recovery metric, whereas `d_MAE`,
  `d_MSE`, `e_MAE`, and `e_MSE` carry the quantitative profile-fidelity role
- `profile TV` stays outside the phase-1 `3B-2` headline set
- the live `3B-2` readout surface is open-focused but retains the complete
  shared metric vocabulary for horizontal comparison

`3C-1`

- ablation: `consistency_ablation`
- ablation_mode: `consistency`
- formal metrics: the shared method-bearing vocabulary above
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference` over the shared realization set
- the ablation removes or zeroes the subbag consistency term and refits `A_p`,
  `d_p`, and `e_p`
- retained objective terms are not reweighted

`3C-2`

- ablation: `geometry_ablation`
- ablation_mode: `geometry`
- formal metrics: the shared method-bearing vocabulary above
- evaluation surface: the same rerun-specific realizations are reused for
  reference and ablation scoring
- the ablation removes or zeroes the geometry/locality term and refits
  `A_p`, `d_p`, and `e_p`
- retained objective terms are not reweighted
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference`

`3C-3`

- ablation: `recurrence_ablation`
- ablation_mode: `recurrence`
- formal metrics: the shared method-bearing vocabulary above
- evaluation surface: the same rerun-specific realizations are reused for
  reference and ablation scoring
- the ablation removes or zeroes the cohort recurrence term and refits
  `A_p`, `d_p`, and `e_p`
- retained objective terms are not reweighted
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference`

For all `3C` arms, `A_MAE_active` and `A_MSE_active` refer to patient-level
relation-operator `A_p` recovery on the shared `K`-state axis.
`open_support_F1` is the support-level recovery metric, whereas `d_MAE`,
`d_MSE`, `e_MAE`, and `e_MSE` carry the quantitative profile-fidelity role;
`open_support_F1` alone is not full open-profile recovery.

Metric-level hierarchy is frozen:

- patient-level metrics remain patient-level only
- condition-level comparison first aggregates within a generator rerun and then
  compares rerun-level summaries
- near-versus-far preservation, preservation-ratio, or direction-consistency
  quantities are condition-level only
- the live Block 3 hierarchy is fully determined by the section and metric
  structure above
- for `3B-1`, the generated realization is shared and no separate generator
  axis is part of the fixed engineering result key
- for `3B-2`, the generated realization is shared and no separate public
  open-burden axis is part of the fixed engineering result key
- any later pooled, collapsed, or headline summary across multiple public
  evaluation-axis values is post-processing only and is not part of the
  phase-1 engineering contract
- if a compatibility surface still carries `pair_family` or
  `evaluation_family`, that label is a reporting-side container only and not a
  formal scoring axis
- generator rerun is the outer statistical unit; `patient × rerun` rows are
  not IID
- metric reporting must carry explicit status when a quantity is not
  applicable or not estimable; blanks must not silently mix with numeric zeros
- raw CSV and extraction CSV surfaces must carry an explicit metric-status
  field and keep non-reported numeric values null
- markdown surfaces must render `N/A (not_applicable)` or
  `N/A (not_estimable)` rather than silently dropping the row state
- JSON surfaces must encode non-reported numeric metrics as `null` plus an
  explicit status field

#### 4.4.6 Frozen Block 3 reporting logic

Block 3 no longer uses a predeclared review-policy label surface as its
scientific contract.

The canonical result-facing organization is now the three frozen sections
`3A generator validation`, `3B baseline comparison`, and
`3C ablation study`, with `3B-1 A benchmark`, `3B-2 d/e benchmark`,
`3C-1 subbag consistency ablation`, `3C-2 geometry ablation`, and
`3C-3 recurrence ablation` nested inside the umbrella sections. Supporting
sidecars may be exported, but they are not a separate scientific section.

Presentation discipline is frozen:

- `3A` carries generator-validation context only, remains continuous rather
  than a gate, and exports held-out cohort `community-space` realism,
  `g_k`-projected identity-aware plausibility, and rerun stability on those
  same objects for manual inspection
- `3B-1` answers how STRIDE compares to transport-family baselines on the
  shared `A` surface
- `3B-2` answers how STRIDE compares to open-comparator baselines on the
  shared analysis-layer `d/e` surface
- `3C-1`, `3C-2`, and `3C-3` answer what happens when consistency, geometry,
  or recurrence objective terms are removed and `A/d/e` is refit
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` belong only to `3B`
- no-`d/e`, no-open-channel, closed, balanced, and transport-style comparisons
  belong only to `3B`
- consistency, geometry, and recurrence refit ablations belong only to `3C`

The canonical reporting order for section-level comparison summaries is:

- mean
- 95% bootstrap CI
- paired difference versus `stride_reference`

Median and rank-based tests may still be exported as supportive diagnostics,
but they are not the primary contract.

Interpretation for the `3B` and `3C` benchmark questions should be written only
after exported raw metric tables and summaries are reviewed. Block 3 does not
produce an aggregate pass/fail label.

#### 4.4.7 Relation to the canonical Step 3 outputs

Block 3 shares the Stage0/TC-IM/K-state Task A context, but it does not consume
or validate the Block 0/1 biological result stack. Its live relation to the
canonical Step 3 outputs is limited to method-facing semi-synthetic benchmark
artifacts exported through the internal semantic CLI.

Block 3 is not allowed to:

- redefine the canonical Block 0/1 results
- elevate the preserved proxy-era Block 3 packet into canonical authority
- backfill new biology that was not already established upstream
- treat comparator success as a substitute for the primary Block 1 discovery
  layer

## 5. Execution Order and Interpretation Discipline

The canonical Task A scientific order is:

1. Descriptive atlas
2. Block 0: `TC-IM` empirical null calibration
3. Block 1: real-data discovery
4. Block 3: semisynthetic truth-recovery benchmark with
   `3A generator validation`, `3B baseline comparison`, and
   `3C ablation study`, plus supporting sidecars

For every execution block, the disciplined order is:

1. implement the engineering surface
2. run subset or demo validation
3. run the full-data analysis
4. export objective results
5. perform human interpretation

Codex is responsible for the first four steps. Human interpretation is outside
the implementation path.

Current task-local engineering contracts use `block1_real_data_discovery` as
the live Block 1 identifier. Active config, workflow routing, registry, and
Block 1-owned tests are aligned to this identifier.

### 5.1 Step 4 multi-phase execution plan

Step 4 is frozen as a five-phase execution sequence.

#### Phase 1: experiment flow and metric freeze

Objective:

- freeze the Block 3 scientific role, experiment flow, comparator structure,
  embedded semi-synthetic companions, proof-carrying summaries, and decision
  logic at the documentation layer

Expected artifacts:

- updated Task A spec text with the rebuilt Block 3 design freeze

In scope:

- documentation
- comparator and metric-contract freeze
- explicit phase definitions

Out of scope:

- code changes
- result rewriting
- experiment execution

Completion criteria:

- the Block 3 scientific design is decision-complete for code drafting

#### Phase 2: Block 3 code framework drafting

Objective:

- draft the Block 3 engineering scaffold so the reference/comparator registry,
  common summary interface, and review-surface contract exist coherently

Expected artifacts:

- Block 3 framework code
- draft manifest and review-surface scaffolding

In scope:

- framework structure
- artifact wiring
- non-final contract plumbing

Out of scope:

- full implementation detail
- final scientific results

Completion criteria:

- the code framework can express the frozen Block 3 design without inventing
  new scientific decisions

#### Phase 3: code-detail implementation and demo-data test

Objective:

- implement the comparator and review logic in detail and confirm that the
  engineering surface works on demo or reduced data

Expected artifacts:

- implemented comparator paths
- demo-data or reduced-data contract-valid outputs

In scope:

- implementation detail
- demo validation
- contract debugging

Out of scope:

- full-cohort scientific interpretation

Completion criteria:

- demo or reduced-data execution completes and emits contract-valid Block 3
  artifacts
- current repo status after Phase 3 implementation:
  - comparator, ablation, embedded companion, and review-generation logic are
    implemented for demo/reduced validation
  - full-cohort Block 3 execution is still pending and belongs only to Phase 4

#### Phase 4: full execution

Objective:

- run the full rebuilt Block 3 analysis on real data plus the embedded
  semi-synthetic companions

Expected artifacts:

- full Block 3 manifests
- raw family, source-community, target-community, and cohort-method outputs
- packet-local review surfaces

In scope:

- full execution
- objective export generation

Out of scope:

- narrative result interpretation beyond objective review surfaces

Completion criteria:

- full real-data and embedded semi-synthetic execution completes without
  unresolved contract failure

#### Phase 5: result summarization and reporting

Objective:

- summarize the rebuilt Block 3 objective outputs and integrate them into Task
  A reporting without redefining Block 0/1

Expected artifacts:

- updated objective review packet
- updated reporting memo surfaces

In scope:

- objective reporting
- canon/history boundary maintenance

Out of scope:

- reopening the scientific design freeze

Completion criteria:

- Block 3 exports the rapid semi-synthetic raw and review surfaces required
  for later metric review against explicit benchmark questions

## 6. Freeze Status and Remaining Deferred Decisions

The following items separate what is now frozen from what remains intentionally
deferred before the new scientific framing becomes a fully
decision-complete engineering and statistical contract.

### 6.1 Frozen Block 1 Summary Definitions

Frozen items:

- `continuity` is now operationalized as strict `self-retention` only; this
  pass does not introduce a neighborhood-based `N(i)` continuity definition.
- Family-level summaries export two frozen estimands:
  `burden_weighted` and `community_mean`.
- Source-side family summaries are fixed to `SR`, `D`, and `R`; `D` is a
  source-open depletion tendency with a burden counterpart, and `R` is an
  off-diagonal remodeling tendency with a burden counterpart, where
  `R = sum_{j != i} A_ij`.
- Target-side `E` remains a supportive family-level target-open accounting
  rollup. It is not evidence of true biological emergence.
- Community-level exports are fixed to source-community `SR_i`, `D_i`, and
  `R_i` with burden-scale counterparts, and target-community `target_burden`,
  `target_weight`, `matched_incoming_burden`, `open_incoming_tendency`, and
  `open_incoming_burden`. `target_weight` is a normalized target composition
  helper.
- The Block 1 core schema does not export top-k target rankings or unweighted
  target incoming operator column sums.
- Off-diagonal destinations remain reconstructable from native `A` and
  `state_ids`; an R-friendly full source-target long CSV is deferred until
  after 3.6 diagnostic validation.
- Source eligibility defaults to non-zero source burden only; this pass does
  not introduce a threshold `tau`.
- `burden_weighted` aggregation uses the same patient's `TC` source-burden
  normalization weights on the eligible source set.
- `community_mean` aggregation uses equal weights over the same eligible source
  set.
- `E` is exported on both `burden_weighted` and `community_mean` scales, but
  remains target-side supportive context.

Still deferred inside this frozen summary layer:

- multiplicity treatment for these summaries
- confirmatory versus exploratory labeling below the frozen axes

### 6.2 Block 0 Empirical Null Calibration Protocol

Frozen items:

- The real family is `TC-IM`.
- The null model is a patient-level empirical permutation null that preserves
  Stage 0 observation surface, patient identity, ROI count structure, and
  domain count structure.
- The permutation policy is within-patient TC/IM domain-label permutation with
  exact per-patient `n_TC`/`n_IM` counts. Identity permutations are allowed;
  cross-patient borrowing and relaxed fallback are disallowed.
- Formal full calibration uses `B=199` permutations. `B` is configurable for
  diagnostics.
- Randomness uses a `master_seed` with deterministic patient/permutation-level
  derived seeds, recorded in the execution manifest.
- Formal full execution fail-fast blocks cache output if the real fit or any
  null fit fails.
- Execution output is a reusable fit cache over `A`, `d`, `e`, `mu_minus`, and
  `mu_plus`; execution does not derive p-values or scientific metrics.
- Calibration analysis is a fixed cache-derived family-summary surface over
  `self_retention`, `depletion`, `off_diagonal_remodeling`, and `emergence`,
  derived from the cache without rerunning permutations.

Still deferred around this calibration:

- result interpretation and plotting from cache-derived calibration outputs

### 6.3 Confirmatory Scope and Multiplicity

Pending items:

- which family-level outputs are confirmatory versus secondary
- which community-level outputs are confirmatory versus exploratory
- how multiplicity will be controlled

Why this is pending:

- Block 1 can otherwise become too flexible and vulnerable to selective
  reporting.

### 6.4 Frozen Cross-Family Comparison Logic

Frozen items:

- the primary cohort comparison is patient-level, paired, and patient-equal-weight
- direct `TC-IM` versus `TC-PT` comparison is only made in patients with both
  real families realized
- paired comparison is defined on the frozen Block 1 summary axes rather than
  on ad hoc post hoc rankings
- `IM-PT` remains outside the confirmatory cross-family comparison contract

Still deferred:

- whether additional normalization is needed for later comparator-facing
  interfaces

### 6.5 Frozen Block 3 comparator and summary interface

Frozen items:

- generator validation, baseline comparison, and ablation sections share one
  generated multi-FOV realization on the shared `K`-state axis; endpoint-only
  baselines consume deterministic endpoint projections, while STRIDE reference
  and ablation arms consume generated source/target FOV observations
- Block 3 may still export supportive family-level, source-community, and
  target-community summaries aligned to the frozen Block 1 language,
  but those supportive summaries are downstream reporting surfaces rather than
  the method-facing benchmark input contract
- Block 3 implementation must export full objective result surfaces for all
  carried communities on the chosen Block 3 summary axes before any later human
  prioritization or narrative selection
- if Block 3 later uses a `primary` versus `secondary` community distinction,
  that distinction is review-layer prioritization only and is not a reason to
  suppress full output generation
- the reference method is `stride_reference`, meaning the canonical full
  patient-plus-cohort STRIDE path with explicit open relation and cohort
  consensus recurrence/common structure, inferred from generated FOV
  observations rather than from hidden truth through the formal
  `fit_stride(...)` estimator
- this names the formal reference target; current implementation lag in
  optimizer/objective/provenance/refit switches does not redefine the Block 3
  scientific contract
- Task A resolves source/target endpoint comparisons and valid domain strata;
  its adapters only convert inputs and instantiate the comparison plan and
  source/target evidence blocks for `stride_reference`
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`,
  `diagonal_transport_baseline`, `recurrence_ablation`,
  `geometry_ablation`, and `consistency_ablation` are the frozen public Block
  3 method names and roles
- the shared-axis geometry/cost freeze is now resolved: `g_k`, `C_raw`,
  `s_C`, and `C` are fixed once from the upstream community-identity surface
  with `UNKNOWN` retained, and that same `C` serves both relation-support
  derivation and the comparator-facing OT cost matrix
- for `stride_reference`, that fixed Task A cost source is routed into the
  full-estimator shared-state cost contract rather than into a task-local
  geometry prior
- no ranked Block 3 method may consume hidden `(A, d, e)`, the latent target,
  or a noiseless baseline carrier; any exact truth companion remains internal
  only
- `3B` is the only external baseline section; `3B-1` carries the `A`
  benchmark and `3B-2` carries the `d/e` benchmark
- `3B` also carries no-`d/e`, no-open-channel, closed, balanced, and
  transport-style comparator semantics
- `3C` is the only core STRIDE ablation section; recurrence, geometry, and
  consistency refit ablations stay there as loss/regularization-level STRIDE
  ablations
- relation and open-comparator semantics belong only to `3B`; `3C` does not
  introduce a separate public generator axis and instead reuses the same
  rerun-specific patient-level semi-synthetic realizations
- `3A` is generator validation only and remains continuous/descriptive rather
  than a separate gate; its public objects are held-out cohort
  `community-space` realism, `g_k`-projected identity-aware biological
  plausibility, and rerun stability on those same validations
- the public Block 3 contract is section-wise rather than a single flat metric
  list:
  `3A` applies `Pearson correlation`, `MAE`, `MSE`, and `JS divergence` to
  the held-out real versus synthetic `community-space` target fraction
  surfaces and to their `g_k`-projected identity-aware surfaces, then
  summarizes rerun stability of those same object-level validations;
  `3B-1`, `3B-2`, `3C-1`, `3C-2`, and `3C-3` use the shared method-bearing
  metric vocabulary under matched rerun-specific semi-synthetic realizations,
  with `3B-2` interpreted as open-focused and `3C-*` interpreted as refit
  ablation readouts
- any compatibility-era `pair_family` / `evaluation_family` labels are
  reporting-only containers rather than formal truth/scoring-layer objects
- Block 3 must still export the broader real-data and semi-synthetic summary
  surfaces for transparency, but those supportive exports are not the frozen
  benchmark contract
- metric reporting must carry explicit `reported`, `not_applicable`, and
  `not_estimable` status semantics with null numeric values when not reported
- section-level comparison summaries for `3B` and all `3C` refit-ablation
  arms must default to mean, 95% bootstrap CI, and paired difference versus
  `stride_reference`, with generator reruns treated as the outer statistical
  unit

Still deferred:

- exact Block 3 file-level artifact schemas and manifest names beyond the
  common review-surface model already established in the repo
- whether any later review packet will add a further teaching-only layer beyond
  the proof-carrying and supportive layers frozen here
- whether supportive community review sets expand beyond the current carried
  communities after execution

Why this is now frozen at the design level:

- `3B` and `3C` export readout surfaces for later interpretation of the same
  patient-level relation object scale used elsewhere in Task A. These surfaces
  do not by themselves validate Block 1 biological findings.

### 6.6 Frozen Block 3 decision logic and remaining deferred items

Frozen items:

- `H1` is evaluated from the executed `3B` primary metrics rather than from
  review-policy labels
- `H2` is evaluated from the executed `3C` consistency, geometry, and
  recurrence refit-ablation native patient-level recovery metrics under the
  shared method-bearing vocabulary and matched rerun-specific semi-synthetic
  realizations
- that `H2` readout remains tied to rerun-specific realizations whose
  generator uses one shared train-derived cohort-level generation context
  across held-out patients; it is not by itself a robustness-gradient claim
- `3A` remains descriptive generator-validation context and does not by itself
  satisfy `H1` or `H2`
- interpretation remains attached to executed raw metric tables rather than to
  a frozen pass/fail policy surface
- Block 3 remains a bounded semi-synthetic method benchmark and does not
  replace the Block 1 discovery layer in final Task A pass logic

Still deferred:

- exact Task A overall pass logic wording after the rebuilt Block 3 execution
  is complete
- final packet-level artifact closure text after rebuilt Block 3 outputs exist

Why these items remain deferred:

- the scientific design is frozen, but the final repo-level pass statement
  should still be written against executed Block 3 artifacts rather than only
  against anticipated ones

## 7. Pass Boundary

Task A passes only as a bounded proxy-validation statement, not as full
validation of the complete longitudinal STRIDE method.

At the scientific level, any final Task A pass statement requires all of the
following:

- the descriptive atlas establishes coherent tissue and community context
- Block 0 calibration reports whether the `TC-IM` relation structure departs
  from the empirical count-preserving permutation null
- Block 1 shows organized family-level and community-level relation patterns
  on real data that can support downstream biological interpretation
- Block 3 first exports rapid semi-synthetic raw and review surfaces;
  scientific interpretation is deferred until metrics are reviewed against
  explicit questions

Task A pass does not require:

- every comparator to be worse on every scalar summary
- every community-level pattern to be equally stable
- `d` or `e` structure to resolve into definitive biological event labels

## 8. Non-Claim Boundary

- The strongest Task A claim remains a bounded proxy-validation claim under an
  ordered tissue-domain proxy.
- Task A does not justify language implying true temporal transition.
- Task A does not directly prove true biological disappearance or emergence.
- Task A does not support confirmatory claims on `IM-PT`.
- Task A does not turn `TC`, `IM`, or `PT` into canonical state identities.
- The descriptive atlas does not itself carry a confirmatory claim.
- Comparator results cannot, by themselves, carry the primary biological claim.
- Block 3 does not support or negate Block 1 biological conclusions by
  itself; it remains a bounded method benchmark.
- Endpoint-transport summaries and assignment surfaces are not themselves the
  primary STRIDE scientific object.
- Task A does not by itself close the evidence gap from endpoint-fraction
  fitting to full patient-level `(T_p, e_p)` validation.

## 9. Block 3 Parameter Freeze Notes

The shared-axis geometry/cost freeze, the train-template multi-FOV generator,
the `3B-1/3B-2` split, the shared generated realization, and the
transport-comparator definitions are now all resolved in the live contract.
The remaining items from the earlier Block 3 open-question list are therefore
treated as frozen parameter notes rather than as reasons to reopen the adopted
scientific design.

- `3B-1` is frozen as the shared-`A` benchmark on the same train-template
  generated realization used by the rest of Block 3.
- `3B-2` is frozen as the open-focused shared analysis-layer `d/e` benchmark on
  the same generated realization, with the complete shared metric vocabulary
  retained for comparison.
- The generator builds train-derived templates from real train TC-IM endpoint
  residuals using geometry-gated residual coupling with `tau=2.0`.
- Held-out truth mixes the train-bank medoid and one sampled individual train
  template using `lambda_individual=0.10`.
- Generated source FOVs use deterministic shrinkage toward real held-out
  source FOVs with `eta=0.3`; generated target FOVs are patient-level truth
  projections with no target noise.
- Template identity, row-imputation mass, endpoint closure, evidence-block
  count, and geometry locality are generator diagnostics and provenance
  fields, not public benchmark axes.
- `balanced_ot_baseline` is frozen as the closed exact OT comparator on the
  fixed cost matrix `C`.
- `uot_baseline` is frozen as the soft-unbalanced transport comparator with
  rerun-shared train-side endpoint-fraction-statistics `lambda_match`
  calibration.
- `partial_ot_baseline` is frozen as the hard-budget transport comparator with
  exact fixed-mass partial OT solves and rerun-shared train-side
  endpoint-fraction-statistics `matched_mass_budget` calibration.
- `diagonal_transport_baseline` is frozen as strict diagonal matched transport
  plus residual open mass.
