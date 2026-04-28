# Task A Live Proxy Specification

This file is the sole live scientific Task A document for the current
proxy/approximate Task A stack.

For the canonical full STRIDE method definition, use
[`docs/stride_design_freeze.md`](/home/lenislin/Experiment/projects/STRIDE/docs/stride_design_freeze.md).
For the Task A migration target, use
[`docs/task_A_rewiring_plan.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_rewiring_plan.md).
This file must not be used to redefine STRIDE itself.

This revision freezes the scientific structure, claim boundaries, and execution
logic of Task A around the new `Descriptive atlas + Block 0/1/2/3` framing.
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
- `e_p` is post-side emergence tendency.
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
> non-random relation structure, biologically interpretable patterns, robust
> repeatable findings, and evidence that these findings are not reducible to
> simpler baselines or accidental component choices?

This question is stronger than the earlier continuity-only framing, but it is
still narrower than full validation of the complete longitudinal STRIDE object
and does not justify direct temporal event claims.

## 4. Scientific Evidence Stack

The Task A scientific stack is ordered. The blocks are not interchangeable, and
downstream interpretation depends on upstream passage.

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

### 4.2 Block 0: STRIDE-Native Null Gate

Block 0 is the entry gate for Task A discovery.

Biological logic:

- An ordered tissue proxy is not automatically an informative relation surface.
- Before interpreting real biological structure, Task A must show that the
  near-proxy family behaves differently from an appropriate null family that
  breaks the proxy relation.

Frozen scope:

- Real family: `TC-IM`
- Null family: a count-preserving randomized family that breaks the intended
  ordered proxy relation while keeping the Task A observation surface honest

Main summary types:

- continuity-like matched summary derived from `A`
- depletion burden derived from `d`
- emergence burden derived from `e`

Role:

- Block 0 does not interpret biology.
- Block 0 asks whether the real `TC-IM` family is meaningfully different from a
  broken-proxy null family on a small set of STRIDE-native derived summaries.
- If Block 0 does not pass, Task A does not advance to Block 1.

### 4.3 Block 1: Real-Data Biological Discovery

Block 1 is the main real-data biological discovery layer.

Biological logic:

- After Block 0 establishes that the near-proxy surface is not behaving like a
  broken null, Task A asks whether real cross-tissue families show organized,
  interpretable patterns that align with tissue context.

Frozen family-level scope:

- confirmatory comparison: `TC-IM` versus `TC-PT`
- `TC-IM` is the nearer real family
- `TC-PT` is the farther real family
- `IM-PT` is not a confirmatory family in this block
- family-level estimands are exported on two frozen scales:
  `burden_weighted` and `community_mean`
- source-side summary names are fixed to:
  - `SR` for self-retention
  - `D` for depletion burden
  - `R` for off-diagonal remodeling burden, with
    `R_i = sum_{j != i} A_ij` at community level
- target-side `E` remains exported as a supportive summary and carries more
  cautious interpretation than the source-side primary summaries

Required community-level scope:

- source-community `SR_i`
- source-community `D_i`
- source-community `R_i`
- source-community `TopTargets_i`
- target-community `I_j`
- target-community `E_j`

Main outputs:

- patient-level family summaries on frozen `summary_name x scale x pair_family`
  axes
- patient-level source-community summaries on frozen
  `patient_id x pair_family x source_community` axes
- patient-level target-community summaries on frozen
  `patient_id x pair_family x target_community` axes
- paired confirmatory comparisons between `TC-IM` and `TC-PT`

Role:

- Block 1 carries the primary biological discovery claim for Task A.
- Residual or open-channel interpretation is not a separate block here; it is
  incorporated into the same real-data discovery layer through `d` and `e`.

### 4.4 Block 2: Robustness of Biological Findings

Block 2 is the robustness validation layer for Block 1 findings.

Biological logic:

- Whether a pattern exists and whether it is stable are separate questions.
- Task A should not promote Block 1 discoveries unless they repeat under
  reasonable data perturbations.

Recommended robustness routes:

- repeated patient subsampling
- leave-some-patients-out analyses
- seed robustness
- ROI/FOV subsampling when needed
- all Block 2 comparisons must stay aligned to the frozen Block 1 summary axes:
  `summary_name x scale x pair_family`, plus matching source/target community
  dimensions when community-level summaries are consumed

Main outputs:

- robustness summaries over the frozen Block 1 family-level summaries
- robustness summaries over the frozen source-community and target-community
  Block 1 exports
- repeatability of directions, rankings, and top patterns

Role:

- Block 2 does not compare STRIDE against baselines.
- Block 2 does not assess method superiority.
- Block 2 asks whether Block 1 discoveries are stable enough to trust.

### 4.5 Block 3: Semisynthetic Truth-Recovery Benchmark on the Full STRIDE Reference

Block 3 is the Task A method-validation layer. It is not a new biology block,
and it does not reopen the Block 0-2 biological interpretation stack.

Under the Step 1 freeze, the currently existing live Block 3 packet remains
proxy execution history rather than final full-STRIDE validation closure.

The rebuilt Block 3 reference method is the canonical full STRIDE method as
currently defined in `docs/stride_design_freeze.md` and realized in the
canonical Step 3 rerun:

- patient-level relation objects `(A_p, d_p, e_p)` on the shared `K`-state
  basis
- explicit open relation through depletion and emergence rather than forced
  closure
- a patient-relation fitting path that keeps FOV/ROI structure in the model
- cohort-level common/recurrent structure defined on patient relations rather
  than on pooled observations

Block 3 sits strictly downstream of the canonical Block 0-2 evidence stack.
Its role is to test, through semi-synthetic truth-recovery benchmarks anchored
to that evidence chain, whether the already established Block 1 / Block 2
finding pattern is genuinely method-specific under the full STRIDE design
rather than recoverable from simpler alternatives or preserved after key
component removals.

#### 4.5.1 Frozen scientific role and hypotheses

The user-approved scientific framing is binding for rebuilt Block 3.

Block 3 tests two method-level hypotheses:

- `H1`: relation-aware modeling with explicit fitted open relation is
  biologically and methodologically valuable. The main Block 1 / Block 2
  finding pattern should weaken when STRIDE is compared against endpoint-only,
  no-`d/e`, no-open-channel, closed, balanced, or transport-style comparator
  arms. The frozen evidence route is `3B baseline comparison`.
- `H2`: core full-STRIDE objective terms should materially support native
  patient-level recovery under matched rerun-specific patient-level
  semi-synthetic realizations. When recurrence, geometry, or consistency terms
  are removed or zeroed and the estimator is refit, recovered patient-level
  `A_p`, `d_p`, and `e_p` should degrade relative to the full reference fit.
  The frozen evidence route is `3C ablation study`.

Block 3 therefore validates all of the following and no broader claim:

- why relation-aware modeling matters
- why explicit open relation matters under external comparator pressure
- why recurrence, geometry, and consistency objective terms matter after
  refitting the full patient relation
- why the established Block 0-2 evidence stack should be read as full-STRIDE
  method evidence rather than as a proxy-era artifact

Block 3 does not:

- generate a new biology claim that supersedes Block 1
- reinterpret the descriptive atlas as proof
- redefine or overwrite canonical Block 0, Block 1, or Block 2 outputs
- create a standalone semi-synthetic discovery block

#### 4.5.2 Frozen Block 3 subexperiments

Rebuilt Block 3 is frozen as three Task A Block 3 sections:
`3A generator validation`, `3B baseline comparison`, and
`3C ablation study`. `3B` carries no-`d/e`, no-open-channel, closed,
balanced, and transport-style comparator semantics. `3C` contains the core
STRIDE refit ablations: `3C-1 recurrence ablation`,
`3C-2 geometry ablation`, and `3C-3 consistency ablation`.
This section structure is exhaustive for the rebuilt Block 3 public surface.

The rebuilt benchmark now uses one shared hidden-program design:

- a real-data-derived baseline composition `x_p` on the shared `K`-state axis
- a hidden patient program `(A_p, d_p, e_p)` used only as the latent generator
- a method-facing target composition `y_p` derived from `(x_p, A_p, d_p, e_p)`
- method inputs restricted to paired endpoint fractions `x_p` and `y_p` on the
  same shared `K`-state axis
- Task A resolves source/target endpoint comparisons and valid domain strata,
  then passes the resolved comparison plan and source/target observation
  evidence blocks to the formal `fit_stride(...)` surface for
  `stride_reference`
- Task A adapters may perform input conversion and comparison-plan
  instantiation; they must not substitute a task-local observation solver or
  task-local STRIDE estimator. Domain resolution remains task-layer
  provenance and does not become a core loss/state/relation/recurrence axis.

No ranked benchmark method, including `stride_reference`, may read hidden
`(A_p, d_p, e_p)` or any additional non-public truth companion. `x_p` and
`y_p` are the method-facing endpoint observations in Block 3 v1. Any exact
truth companion
may exist only as an internal scoring/debugging object and must not appear as a
ranked benchmark arm.

The detailed alignment note for this transitional redesign lives in
[`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md).

The rebuilt benchmark is additionally frozen around one held-out outer design:

- each rerun uses a patient-level `24 train / 8 test` split
- `train` patients contribute only weak empirical calibration from real
  `TC -> IM` change
- `test` patients contribute only their real `TC` carrier `x_p`
- generator reruns are repeated `10` times and remain the outer statistical
  unit

To keep the semi-synthetic benchmark scientifically interpretable rather than
turning it into a large hyperparameter surface, Block 3 now exposes two
section-specific public controls plus one internal generator layer:

- `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` is the public
  matched-structure control used by `3B-1`
- `relation_strength` denotes the off-diagonal matched-mass fraction on the
  shared `K`-community axis
- `max_offdiag_targets_per_source` is deterministically derived from
  `relation_strength` as `(0.00 -> 0, 0.05 -> 1, 0.15 -> 1, 0.30 -> 2)`
- `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` is the public open-burden
  control used by `3B-2`
- patient-level open burden remains a rerun-specific generator realization.
  Within each rerun, `P(m)` is estimated on the `train` split from
  `m_proxy = 0.5 * || q_TC - q_IM ||_1`, used to sample held-out patient
  `m_p`, and then realized as patient-level semi-synthetic open burden
- benchmark comparison and ablation study reuse the same rerun-specific
  patient-level semi-synthetic realizations; `P(m)` is therefore a generator
  quantity rather than a section-facing comparison target or benchmark metric
- each shared community `k` is assigned a frozen community-identity vector
  `g_k` from the upstream atlas/community-correspondence
  `community x cell subtype` row-fraction surface; `g_k` retains the
  `UNKNOWN` subtype with no removal or reweighting in the main contract
- the shared-axis biological cost precursor is
  `C_raw[i,j] = sqrt(JS(g_i, g_j))`, with `C_raw[i,i] = 0`
- let `s_C` be the median of the positive off-diagonal entries of `C_raw`;
  the frozen normalized cost matrix is `C = C_raw / s_C`
- `C` is derived once from the frozen upstream community-identity surface and
  reused both as the shared-axis biological neighborhood for relation support
  and as the comparator-facing OT cost matrix; it is not a rerun-specific or
  train-estimated pairwise topology

All other generator quantities, including the empirical turnover distribution,
depletion/emergence propensities, and the shared train-derived cohort-level
generation quantities reused within the recurrence-ablation realization set,
are estimated from the `train` split and are not treated as free public
benchmark hyperparameters.
`train` therefore continues to estimate only `P(m)`, `pi_d`, `pi_e`,
`kappa_d`, and `kappa_e`. The
compatibility label `shared hidden cohort effect` used in the
recurrence-ablation arm therefore means only that all held-out `test` patients
in the same rerun are generated under the same train-derived cohort-level
generator quantities. It is recorded for reproducibility only, is not a public
benchmark axis, and does not define a separate scientific open question. These
train-derived quantities are
generator calibration / diagnostics-side quantities only, not formal Block 3
headline benchmark metrics.

Within each rerun, weak train calibration is frozen to state-level and
open-level priors only:

- define the train-side open-mass proxy as
  `m_proxy = 0.5 * || q_TC - q_IM ||_1`; the factor `0.5` is definitional,
  not a tuned hyperparameter
- `P(m)` is the empirical distribution of `m_proxy` over the `train`
  patients, and each held-out `test` patient samples `m_p` from that
  empirical distribution with replacement
- the resulting patient-level semi-synthetic open burden is reused across the
  benchmark and ablation evaluations in that rerun rather than exposed as a
  public fixed-`m` section axis
- `pi_d` and `pi_e` are the mean normalized positive-part depletion and
  emergence changes over informative `train` patients
- `kappa_d` and `kappa_e` are train-estimated cohort-level dispersion scalars
  for depletion/emergence shape heterogeneity around `pi_d` and `pi_e`; in v1
  first compute the robust median total-variation deviation `tv` of
  informative `train` patients from the corresponding channel centroid, then
  map that dispersion to the `Gamma -> normalize` concentration by
  `kappa = clip(1 / max(tv, 1e-8), 1, 200)`. They are not total
  change-magnitude parameters
- train calibration may not estimate state-pair relation templates,
  off-diagonal motifs, or source-target mapping rules

For each held-out `test` carrier `x_p`, the frozen v1 open scaffold is:

- depletion base measure
  `b_p^- = normalize(x_p ⊙ pi_d)`
- emergence base measure
  `b_p^+ = normalize((x_p + epsilon_fixed) ⊙ pi_e)`, where
  `epsilon_fixed = 0.01` is a fixed smoothing pseudocount / soft floor rather
  than a train-estimated biological random quantity or Gaussian-style noise
  term
- `d_p` is hard patient-anchored through `b_p^-`
- `e_p` is soft patient-anchored through `b_p^+`
- `Delta_p^-` and `Delta_p^+` are the direct generator objects and satisfy
  `sum(Delta_p^-) = sum(Delta_p^+) = m_p`
- `d_p` and `e_p` are derived from `Delta_p^-` and `Delta_p^+`; they are not
  independently sampled total-mass vectors
- the default v1 sampling family is `Gamma -> normalize`, with capped
  allocation on depletion so that `Delta_{p,i}^- <= x_{p,i}` by construction
- matched source mass is
  `u_p = x_p - Delta_p^-`
- the latent target is
  `y_p = colsum(M_p) + Delta_p^+`

Across `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]`, only the matched
component changes:

- `relation_strength = 0.00` keeps the matched component diagonal-only and
  sets `max_offdiag_targets_per_source = 0`
- `relation_strength = 0.05` sets `max_offdiag_targets_per_source = 1` with
  off-diagonal matched-mass fraction `0.05`
- `relation_strength = 0.15` sets `max_offdiag_targets_per_source = 1` with
  off-diagonal matched-mass fraction `0.15`
- `relation_strength = 0.30` sets `max_offdiag_targets_per_source = 2` with
  off-diagonal matched-mass fraction `0.30`
- if `max_offdiag_targets_per_source = 1`, all off-diagonal matched mass is
  allocated to the nearest neighbor under `C`
- if `max_offdiag_targets_per_source = 2`, off-diagonal matched mass is
  allocated across the `2` nearest neighbors by normalized `exp(-C[i,j])`
  weights with no extra temperature parameter
- off-diagonal support is nested across
  `relation_strength = 0.05 -> 0.15 -> 0.30`
- these predeclared relation-strength levels are benchmark regimes rather than
  empirical biological subclasses
- off-diagonal support must be induced only by predeclared shared-axis
  geometry or cost rules and never by train-derived state-pair frequencies

##### 3A. Generator validation

Scientific question:

- Does the frozen semi-synthetic generator produce held-out cohort-level
  target-fraction objects that are realistic, biologically plausible, and
  stable enough to support downstream `3B/3C` evaluation?

Generator-validation surface:

- `3A` validates the same rerun-specific patient-level semi-synthetic
  realizations that are later consumed by `3B` and `3C`
- fixed `m_p` strata are not part of the public `3A` contract
- if implementation exports descriptive slices by realized open burden or by
  `relation_strength`, those slices are diagnostics only and are not
  section-defining benchmark axes
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

- `3A` absorbs the earlier generator-validity layer and keeps it continuous
  and descriptive rather than turning it into a single pass/fail gate
- `3A` validates the generator in the same fraction-space semantic layer used
  by the rest of Task A, not method superiority, not patient reconstruction,
  not a forecasting-style accuracy benchmark, and not a comparison of `P(m)`
- intuitive `real TC -> IM` versus `synthetic TC -> y` change language may
  still be used for explanation, but the formal contract is the held-out
  cohort `community-space` target fraction surface and the corresponding
  `g_k`-projected identity-aware surface
- `3A` does not claim to isolate or prove the marginal contribution of
  train-derived cohort priors; `P(m)`, `pi_d`, `pi_e`, `kappa_d`, and
  `kappa_e` remain generator quantities rather than public `3A` axes
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
  relation surface `A_p` from the same paired endpoint-fraction inputs?

Reference versus baselines:

- reference: `stride_reference`
- baseline: `balanced_ot_baseline`
- baseline: `uot_baseline`
- baseline: `partial_ot_baseline`
- baseline: `diagonal_transport_baseline`

Formal metrics:

- `A_MAE_active`
- `A_MSE_active`
- `target_recall_at_k`

Metric rules:

- `3B-1` uses `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]`
- `3B-1` fixes `open_mass_scale = 1.0`, the canonical open regime
- the fixed shared-axis cost matrix `C`, derived from community-identity
  vectors `g_k`, is reused to define nearest-neighbor relation support and to
  define the comparator-facing OT cost for
  `balanced_ot_baseline`, `uot_baseline`, and `partial_ot_baseline`
- the same rerun-specific held-out patients, endpoint fractions `x_p` and
  `y_p`, and hidden patient-level truth are reused across all `3B-1` methods
- for the same patient within the same rerun, sampled `m_p`, derived `d_p`,
  derived `e_p`, and `open_mass_scale` remain fixed across `3B-1`; only
  `relation_strength` changes inside the `3B-1` grid
- `relation_strength` controls the off-diagonal matched-mass fraction on the
  shared `K`-community axis
- `max_offdiag_targets_per_source` is deterministically derived from
  `relation_strength` as `(0.00 -> 0, 0.05 -> 1, 0.15 -> 1, 0.30 -> 2)`
- `relation_strength = 0.00` keeps the matched mass diagonal-only
- `relation_strength = 0.05` and `relation_strength = 0.15` use the nearest
  `1` off-diagonal target under `C`
- `relation_strength = 0.30` uses the nearest `2` off-diagonal targets under
  `C`
- if `max_offdiag_targets_per_source = 1`, all off-diagonal matched mass goes
  to the nearest neighbor under `C`
- if `max_offdiag_targets_per_source = 2`, off-diagonal matched mass is split
  across the `2` nearest neighbors by normalized `exp(-C[i,j])` weights with
  no extra temperature parameter
- these relation-strength levels are the reviewer-facing benchmark challenge
  levels for `3B-1`
- transported-mass scoring is truth-anchored:
  `T_true[i,j] = x_true[i] * A_true[i,j]` and
  `T_hat[i,j] = x_true[i] * A_hat[i,j]`
- `A_MAE_active` and `A_MSE_active` score active-row conditional target-pattern
  recovery on truth-active source rows only; they are not total
  transported-mass error metrics
- `target_recall_at_k` scores truth-anchored off-diagonal transported-mass
  target-priority recovery rather than generic endpoint reconstruction
- at `relation_strength = 0.00`, `target_recall_at_k` is `not_applicable`
  because truth contains no off-diagonal target set (`k = 0`); any
  compatibility-only numeric display must be labeled as a structural ceiling
  rather than evidence of recovery quality

Interpretive role:

- `3B-1` is the relation-surface benchmark for `A_p`
- the phase-1 engineering contract is to compute and export the frozen
  `3B-1` metric set for each evaluated method, relation-strength level, and
  rerun
- judgments about competitiveness, acceptable loss, or headline ranking are
  deferred to post-processing and result interpretation
- this is the reviewer-facing relation-gradient benchmark arm for `H1`

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

Formal metrics:

- `open_support_F1`
- `d_MAE`
- `e_MAE`
- `d_MSE`
- `e_MSE`

Metric rules:

- `3B-2` reuses the same rerun-specific held-out patient identities, endpoint
  fractions, and hidden truth construction flow used by `3B-1`
- `3B-2` fixes `relation_strength = 0.15` and therefore
  `max_offdiag_targets_per_source = 1`
- `3B-2` uses `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` as its public
  open-burden gradient
- within each rerun, held-out patients are reused across the full
  `open_mass_scale_grid` at fixed `relation_strength = 0.15`
- in `3B-2`, reused means shared patient identities, shared `x_p`, shared
  matched-structure setting, shared support rule, shared `C`, and shared
  `pi_d / pi_e` ratio
- the truth-side open quantities `delta_minus_scaled`, `delta_plus_scaled`,
  and the derived `d_p`, `e_p`, `y_p` vary with `open_mass_scale`
- each method first emits its native matched/unmatched representation, and the
  shared `3B` analysis layer then derives the common `A/d/e` scoring surfaces
- `open_support_F1` is the support-level recovery metric on the derived
  burden-scale depletion/emergence carriers
- `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` are the quantitative profile-recovery
  metrics on the same derived open surfaces
- `3B-2` reuses the full shared `3C` `open_support_F1` contract, including
  the burden-scale support definition and the channel-level / patient-level
  status semantics
- because `open_mass_scale = 0.0` implies zero truth depletion and zero truth
  emergence by construction, `open_support_F1` is `not_applicable` under that
  condition
- `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` remain reported under
  `open_mass_scale = 0.0`
- `profile TV` may still be exported as a diagnostic sidecar, but it is not a
  phase-1 headline `3B-2` metric

Interpretive role:

- `3B-2` is the open-surface benchmark for the shared analysis-layer `d/e`
  recovery problem
- its fixed evidential target is whether STRIDE provides more accurate `d/e`
  recovery than the open comparator set under matched rerun-specific
  realizations
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
- the core `3C` ablation set is recurrence, geometry, and consistency
- no-`d/e`, open-channel-removal, closed, balanced, and transport-style
  comparisons belong to `3B` baseline/comparator semantics rather than to
  core STRIDE ablations
- the current repository proxy Block 3 ablation implementation is preserved as
  historical/proxy execution context only and is not the normative `3C`
  scientific contract

###### 3C-1. Recurrence ablation

Scientific question:

- What is lost when cohort consensus recurrence/common-structure feedback is removed
  from the STRIDE objective and the patient relation is refit?

Reference versus ablation:

- reference: `stride_reference`
- ablation: `recurrence_ablation`

Ablation semantics:

- `3C-1` is a within-STRIDE module-necessity test, not an external baseline
  comparison, not an open-gradient benchmark, and not a comparison of `P(m)`
- `recurrence_ablation` removes or zeroes only the cohort consensus
  recurrence/common-structure term in the objective
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- geometry/locality terms are retained
- patient consistency terms are retained
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations generated for Block 3 and does not introduce a separate public
  fixed-`m` axis

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
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- patient consistency terms are retained
- cohort consensus recurrence/common-structure terms are retained
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations generated for Block 3 and does not introduce a separate public
  fixed-`m` axis

###### 3C-3. Consistency ablation

Scientific question:

- What is lost when the patient-consistency term is removed from the STRIDE
  objective and the patient relation is refit?

Reference versus ablation:

- reference: `stride_reference`
- ablation: `consistency_ablation`

Ablation semantics:

- `consistency_ablation` removes or zeroes only the patient-consistency term in
  the objective
- observation discrepancy terms are retained
- explicit open-channel terms are retained
- geometry/locality terms are retained
- cohort consensus recurrence/common-structure terms are retained
- audit / plausibility handling is retained
- `A_p`, `d_p`, and `e_p` are refit under the ablated objective
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations generated for Block 3 and does not introduce a separate public
  fixed-`m` axis

Formal metrics:

- `A_MAE_active`
- `A_MSE_active`
- `open_support_F1`
- `d_MAE`
- `e_MAE`
- `d_MSE`
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
- `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` remain reported throughout the
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
  same train-derived cohort-level generator quantities
  (`P(m)`, `pi_d`, `pi_e`, `kappa_d`, `kappa_e`), while each patient's
  realized hidden truth still depends on that patient's own `x_p` plus
  patient-level sampling
- `stride_reference` and each `3C` ablation arm are compared on the same
  held-out patients and the same native truth outputs under matched
  rerun-specific realizations
- `open_support_F1` remains a native patient-level support-recovery metric and
  does not by itself imply full profile fidelity
- metric reporting must distinguish `reported`, `not_applicable`, and
  `not_estimable`

Interpretive role:

- `3C` is centered on loss of native patient-level `A_p`, `d_p`, and `e_p`
  recovery after recurrence, geometry, or consistency terms are removed and the
  estimator is refit
- `3C` does not create a separate public sensitivity axis over shared
  train-derived cohort-level generation and therefore does not by itself
  establish robustness across multiple cohort/common-structure generation
  settings
- only the native patient-level metrics listed above belong to the live `3C`
  metric contract
- patient-level helper quantities may still be exported as diagnostics, but
  they are not part of the formal Block 3 metric contract

#### 4.5.3 Frozen comparator registry

The rebuilt Block 3 method registry is frozen as follows.

`stride_reference`

- Uses the canonical full STRIDE method with patient relation, explicit open
  relation, patient-relation fitting, and cohort consensus recurrence/common
  structure.
- Calls the formal `fit_stride(...)` frozen reference configuration on
  Task A-resolved source/target endpoint comparison evidence blocks and the
  resolved comparison plan; Task A adapters only convert inputs and instantiate
  the comparison plan.
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
  across the full `3B-1 relation_strength_grid`, the full `3B-2`
  `open_mass_scale_grid`, and all `8` held-out `test` patients in that rerun.
- The native method output is the matched plan `P`, which is later converted
  by the shared `3B` analysis layer into `A/d/e`.

`partial_ot_baseline`

- Uses a hard-budget partial transport comparator on the same paired endpoint
  fractions and fixed shared cost matrix `C`.
- The rerun-specific `24`-patient `train` split calibrates one shared
  `matched_mass_budget` value from endpoint-fraction statistics and then
  reuses it across the full `3B-1 relation_strength_grid`, the full `3B-2`
  `open_mass_scale_grid`, and all `8` held-out `test` patients in that rerun.
- The native method output is the matched plan `P`, which is later converted
  by the shared `3B` analysis layer into `A/d/e`.

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
    P = solve_partial_ot_plan(
        source=x,
        target=y,
        cost=C,
        matched_mass_budget=matched_mass_budget,
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

Train-side shared calibration minimums:

- `uot_baseline`: estimate one rerun-shared `lambda_match` from the `24`
  `train` patients using endpoint-fraction statistics and reuse that same
  scalar across the full `3B-1 relation_strength_grid`, the full
  `3B-2 open_mass_scale_grid`, and all `test` patients in that rerun
- `partial_ot_baseline`: estimate one rerun-shared
  `matched_mass_budget` from the `24` `train` patients using
  endpoint-fraction statistics and reuse that same scalar across the full
  `3B-1 relation_strength_grid`, the full `3B-2 open_mass_scale_grid`, and
  all `test` patients in that rerun
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

`recurrence_ablation`

- Removes or zeroes only the cohort consensus recurrence/common-structure term and
  refits `A_p`, `d_p`, and `e_p`.
- Preserves observation discrepancy, explicit open-channel terms,
  geometry/locality, patient consistency, and audit/plausibility handling.
- This is the fixed `3C-1 recurrence ablation` arm.

`geometry_ablation`

- Removes or zeroes only the geometry/locality term and refits `A_p`, `d_p`,
  and `e_p`.
- Preserves observation discrepancy, explicit open-channel terms, patient
  consistency, cohort consensus recurrence/common structure, and
  audit/plausibility handling.
- This is the fixed `3C-2 geometry ablation` arm.

`consistency_ablation`

- Removes or zeroes only the patient-consistency term and refits `A_p`, `d_p`,
  and `e_p`.
- Preserves observation discrepancy, explicit open-channel terms,
  geometry/locality, cohort consensus recurrence/common structure, and
  audit/plausibility handling.
- This is the fixed `3C-3 consistency ablation` arm.

#### 4.5.4 Embedded semi-synthetic companion design

Semi-synthetic benchmark conditions remain embedded inside the relevant Block 3
subexperiments. There is no standalone Block 3E.

All Block 3 companion conditions must expose the same hidden-truth interface:

- patient-level truth: `(A_p, d_p, e_p)`
- patient-level paired endpoint fractions `x_p` and `y_p`
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

1. derive a real-data-grounded baseline composition `x_p`
2. estimate weak empirical priors from the `train` split only for generator
   calibration / diagnostics:
   `P(m)`, `pi_d`, `pi_e`, and `kappa_d`, `kappa_e`
3. sample `m_p` for each held-out patient from the empirical train-side
   `P(m)` distribution with replacement
4. treat the resulting patient-level semi-synthetic open burden as a
   rerun-specific generator realization that is reused across `3B` and `3C`
   rather than promoted to a public benchmark axis
5. construct the weak patient-anchored open scaffold
   `F_open(x_p) -> (b_p^-, b_p^+)`
6. sample `Delta_p^-` and `Delta_p^+` around that scaffold, derive
   `(d_p, e_p)`, and define matched source mass `u_p = x_p - Delta_p^-`
7. apply the frozen relation scenario to the matched mass only, producing
   `M_p`
8. construct a latent target composition
   `y_p = colsum(M_p) + Delta_p^+`
9. expose paired endpoint fractions `x_p` and `y_p` to ranked methods only
10. evaluate recovered objects against hidden truth only after inference

The frozen matched-structure control uses the parameter name
`relation_strength`:

- `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]`
- `3B-1` uses the full `relation_strength_grid`
- `3B-2` fixes `relation_strength = 0.15`
- `relation_strength` sets the off-diagonal matched-mass fraction
- `max_offdiag_targets_per_source` is derived from `relation_strength` as
  `(0.00 -> 0, 0.05 -> 1, 0.15 -> 1, 0.30 -> 2)`

The frozen open-burden control uses the existing parameter name
`open_mass_scale`:

- `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]`
- `3B-1` fixes `open_mass_scale = 1.0`
- `3B-2` uses the full `open_mass_scale_grid`
- `0.0` is the near-closed limit
- `1.0` is the canonical open regime
- the first frozen phase does not extend the public grid above `1.0`

```python
delta_minus_scaled = open_mass_scale * delta_minus
delta_plus_scaled = open_mass_scale * delta_plus
```

The dense `open_mass_scale` sweep in `3B-2` preserves the following
quantities:

- `x`
- `relation_strength = 0.15`
- support rule
- `C`
- `pi_d / pi_e` ratio

The frozen companion condition families are:

- held-out cohort-level `3A` summaries on the `community-space` target
  fraction surface, the `g_k`-projected identity-aware target fraction
  surface, and rerun stability of those same objects
- `3B-1`: `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` at fixed
  `open_mass_scale = 1.0`
- `3B-2`: `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` at fixed
  `relation_strength = 0.15`
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

#### 4.5.5 Frozen Block 3 metric contract

Block 3 must continue to reuse the current Task A real-data summary language
for raw supportive exports, but the canonical benchmark contract is now the
fixed section-wise metric set below.

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

`3B-1`

- formal metrics: `A_MAE_active`, `A_MSE_active`, and
  `target_recall_at_k`
- engineering-facing rerun summary rows must remain stratified by
  `relation_strength`
- the minimum fixed `3B-1` summary cell is
  `(relation_strength, method_name, metric_name)` under fixed
  `open_mass_scale = 1.0`
- mean, 95% bootstrap CI, and paired difference versus `stride_reference`
  are post-processing summaries computed from those fixed-cell rerun summaries
- `A_MAE_active` and `A_MSE_active` remain active-row conditional
  target-pattern recovery metrics rather than total transported-mass error
  metrics
- `target_recall_at_k` remains a truth-anchored off-diagonal target-priority
  metric and is `reported` only when truth off-diagonal targets exist
- at `relation_strength = 0.00`, `target_recall_at_k` is `not_applicable`
  because truth contains no off-diagonal target set (`k = 0`); any
  compatibility-only numeric display must be labeled as a structural ceiling
  rather than evidence of recovery quality

`3B-2`

- formal metrics: `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`,
  and `e_MSE`
- engineering-facing rerun summary rows must remain stratified by
  `open_mass_scale`
- the minimum fixed `3B-2` summary cell is
  `(open_mass_scale, method_name, metric_name)` under fixed
  `relation_strength = 0.15`
- mean, 95% bootstrap CI, and paired difference versus `stride_reference`
  are post-processing summaries computed from those fixed-cell rerun summaries
- `open_support_F1` is the support-level recovery metric, whereas `d_MAE`,
  `e_MAE`, `d_MSE`, and `e_MSE` carry the quantitative profile-fidelity role
- `profile TV` stays outside the phase-1 `3B-2` headline set
- the live `3B-2` readout surface is fully determined by
  `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`

`3C-1`

- ablation: recurrence
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`,
  `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference` over the shared realization set
- the ablation removes or zeroes the cohort consensus recurrence term and
  refits `A_p`, `d_p`, and `e_p`

`3C-2`

- ablation: geometry
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`, `d_MAE`,
  `e_MAE`, `d_MSE`, and `e_MSE`
- evaluation surface: the same rerun-specific realizations are reused for
  reference and ablation scoring
- the ablation removes or zeroes the geometry/locality term and refits
  `A_p`, `d_p`, and `e_p`
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference`

`3C-3`

- ablation: consistency
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`, `d_MAE`,
  `e_MAE`, `d_MSE`, and `e_MSE`
- evaluation surface: the same rerun-specific realizations are reused for
  reference and ablation scoring
- the ablation removes or zeroes the patient-consistency term and refits
  `A_p`, `d_p`, and `e_p`
- evaluation-surface summary table: mean, 95% bootstrap CI, and paired
  difference versus `stride_reference`

For all `3C` arms, `A_MAE_active` and `A_MSE_active` refer to patient-level
relation-operator `A_p` recovery on the shared `K`-state axis.
`open_support_F1` is the support-level recovery metric, whereas `d_MAE`,
`e_MAE`, `d_MSE`, and `e_MSE` carry the quantitative profile-fidelity role;
`open_support_F1` alone is not full open-profile recovery.

Metric-level hierarchy is frozen:

- patient-level metrics remain patient-level only
- condition-level comparison first aggregates within a generator rerun and then
  compares rerun-level summaries
- near-versus-far preservation, preservation-ratio, or direction-consistency
  quantities are condition-level only
- the live Block 3 hierarchy is fully determined by the section and metric
  structure above
- for `3B-1`, `relation_strength` is the public evaluation axis and is part of
  the fixed engineering result key; `open_mass_scale` is fixed to `1.0`
- for `3B-2`, `open_mass_scale` is the public evaluation axis and is part of
  the fixed engineering result key; `relation_strength` is fixed to `0.15`
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

#### 4.5.6 Frozen Block 3 reporting logic

Block 3 no longer uses a predeclared review-policy label surface as its
scientific contract.

The canonical result-facing organization is now the three frozen sections
`3A generator validation`, `3B baseline comparison`, and
`3C ablation study`, with `3B-1 A benchmark`, `3B-2 d/e benchmark`,
`3C-1 recurrence ablation`, `3C-2 geometry ablation`, and
`3C-3 consistency ablation` nested inside the umbrella sections. Supporting
sidecars may be exported, but they are not a separate scientific section.

Presentation discipline is frozen:

- `3A` carries generator-validation context only, remains continuous rather
  than a gate, and is defined by held-out cohort `community-space` realism,
  `g_k`-projected identity-aware plausibility, and rerun stability on those
  same objects
- `3B-1` answers how STRIDE compares to transport-family baselines on the
  shared `A` surface
- `3B-2` answers how STRIDE compares to open-comparator baselines on the
  shared analysis-layer `d/e` surface
- `3C-1`, `3C-2`, and `3C-3` answer what happens when recurrence, geometry,
  or consistency objective terms are removed and `A/d/e` is refit
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` belong only to `3B`
- no-`d/e`, no-open-channel, closed, balanced, and transport-style comparisons
  belong only to `3B`
- recurrence, geometry, and consistency refit ablations belong only to `3C`

The canonical reporting order for section-level comparison summaries is:

- mean
- 95% bootstrap CI
- paired difference versus `stride_reference`

Median and rank-based tests may still be exported as supportive diagnostics,
but they are not the primary contract.

Evidence for `H1` should therefore be argued from the executed `3B` raw metric
tables, while evidence for `H2` should be argued from the executed `3C`
recurrence, geometry, and consistency refit-ablation tables rather than from an
aggregate pass/fail label.

#### 4.5.7 Relation to the canonical Step 3 outputs

Block 3 is allowed to test only the now-established canonical Step 3 evidence
stack through Block 2:

- the descriptive atlas as biological context only
- the Block 0 near-proxy real-versus-null gate
- the Block 1 family-level proof-carrying contrast on `self_retention` and
  `depletion`
- the carried source-community and target-community patterns already scoped by
  Block 1 and sharpened by Block 2
- the canonical cohort consensus recurrence/common-structure outputs present in the
  Block 1 bundle
- the Block 2 robustness calls over those same carried findings

Block 3 is not allowed to:

- redefine the canonical Block 0-2 results
- elevate the preserved proxy-era Block 3 packet into canonical authority
- backfill new biology that was not already established upstream
- treat comparator success as a substitute for the primary Block 1 discovery
  layer

## 5. Execution Order and Interpretation Discipline

The canonical Task A scientific order is:

1. Descriptive atlas
2. Block 0: STRIDE-native real-versus-null gate
3. Block 1: real-data biological discovery
4. Block 2: robustness of findings
5. Block 3: semisynthetic truth-recovery benchmark with
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

Current task-local engineering contracts preserve the legacy block identifiers
`block1_continuity_backbone` and `block2_bounded_audit` as compatibility
labels, but the scientific role and output contract are now tied to the frozen
Block 1 summary surfaces and the Block 2 robustness-over-summaries consumer
surface.

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
  A reporting without redefining Block 0-2

Expected artifacts:

- updated objective review packet
- updated reporting memo surfaces

In scope:

- objective reporting
- canon/history boundary maintenance

Out of scope:

- reopening the scientific design freeze

Completion criteria:

- Block 3 is reportable as a downstream method-validation layer over the
  canonical evidence stack

## 6. Freeze Status and Remaining Deferred Decisions

The following items separate what is now frozen from what remains intentionally
deferred before the new scientific framing becomes a fully
decision-complete engineering and statistical contract.

### 6.1 Frozen Block 1 / Block 2 Summary Definitions

Frozen items:

- `continuity` is now operationalized as strict `self-retention` only; this
  pass does not introduce a neighborhood-based `N(i)` continuity definition.
- Family-level summaries export two frozen estimands:
  `burden_weighted` and `community_mean`.
- Source-side family summaries are fixed to `SR`, `D`, and `R`, where
  `R = sum_{j != i} A_ij`.
- Target-side `E` remains exported but is explicitly supportive rather than a
  primary proof-carrying summary.
- Community-level exports are fixed to `SR_i / D_i / R_i / TopTargets_i / I_j / E_j`.
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
- final resampling-based Block 2 stability estimands

### 6.2 Block 0 Null and Gate Protocol

Frozen items:

- The real family is `TC-IM`.
- The null family is `TC-IM_randomized_target`, constructed by holding each
  anchor patient's `TC` observations fixed and reassigning the `IM` group from
  a different patient in the same exact `(n_TC, n_IM)` count stratum.
- Null-family donor assignment is seeded and reproducible.
- Singleton strata emit deferred null fits rather than relaxing to a looser
  control.
- The live Block 0 exported summaries are derived directly from realized `A`,
  `d`, and `e`.
- Current pass/fail is driven by paired deltas on `sum(A)` and `sum(e)`; the
  `sum(d)` layer remains exported context rather than a primary decision
  statistic.

Still deferred around this gate:

- whether later robustness rounds should add repeated-null sensitivity analyses
  beyond the current seeded gate
- how future resampling depth should be reported once the later robustness pass
  is implemented

### 6.3 Confirmatory Scope and Multiplicity

Pending items:

- which family-level outputs are confirmatory versus secondary
- which community-level outputs are confirmatory versus exploratory
- how multiplicity will be controlled
- when stability can substitute for formal multiplicity control

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
- how these paired comparisons feed the final Block 2 resampling summaries

### 6.5 Frozen Block 3 comparator and summary interface

Frozen items:

- generator validation, baseline comparison, and ablation sections should all
  consume one common paired endpoint-fraction interface on the shared
  `K`-state axis, with `x_p` and `y_p` as the only method-facing inputs
- Block 3 may still export supportive family-level, source-community, and
  target-community summaries aligned to the frozen Block 1 / Block 2 language,
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
  consensus recurrence/common structure, inferred from paired endpoint
  fractions rather than from hidden truth through the formal `fit_stride(...)`
  estimator
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
- the reviewer-facing relation gradient belongs only to `3B`; `3C` does not
  introduce a separate public open-gradient axis and instead reuses the same
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
  `3B-1` uses `A_MAE_active`, `A_MSE_active`, and
  `target_recall_at_k`;
  `3B-2` uses `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`;
  `3C-1`, `3C-2`, and `3C-3` use `A_MAE_active`, `A_MSE_active`,
  `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` under matched
  rerun-specific semi-synthetic realizations after refitting `A/d/e`
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

- Block 3 should validate the same scientific objects as Block 1 / Block 2
  rather than introducing a disconnected benchmark language.

### 6.6 Frozen Block 3 decision logic and remaining deferred items

Frozen items:

- `H1` is argued from the executed `3B` primary metrics rather than from
  review-policy labels
- `H2` is argued from the executed `3C` recurrence, geometry, and consistency
  refit-ablation native patient-level recovery metrics `A_MAE_active`,
  `A_MSE_active`, `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`
  under matched rerun-specific semi-synthetic realizations
- that `H2` evidence remains tied to rerun-specific realizations whose
  generator uses one shared train-derived cohort-level generation context
  across held-out patients; it is not by itself a robustness-gradient claim
- `3A` remains descriptive generator-validation context and does not by itself
  satisfy `H1` or `H2`
- interpretation remains attached to executed raw metric tables rather than to
  a frozen pass/fail policy surface
- Block 3 remains a downstream method-validation layer and does not replace the
  Block 1 discovery layer in final Task A pass logic

Still deferred:

- exact Task A overall pass logic wording after the rebuilt Block 3 execution
  is complete
- final Block 2 resampling/stability closure language
- final packet-level artifact closure text after rebuilt Block 3 outputs exist

Why these items remain deferred:

- the scientific design is frozen, but the final repo-level pass statement
  should still be written against executed Block 3 artifacts rather than only
  against anticipated ones

## 7. Pass Boundary

Task A passes only as a bounded proxy-validation statement, not as full
validation of the complete longitudinal STRIDE method.

At the scientific level, Task A pass requires all of the following:

- the descriptive atlas establishes coherent tissue and community context
- Block 0 supports a non-random STRIDE-native signal on the near-proxy family
- Block 1 shows organized and biologically interpretable family-level and
  community-level patterns on real data
- Block 2 shows that the main Block 1 findings are stable under reasonable
  perturbations
- Block 3 shows that the main finding pattern is not reducible to simpler
  baselines, weakens under the relevant ablations, and remains informative on
  the embedded semi-synthetic truth-recovery benchmarks

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
- Block 3 supports method validation, but it does not replace Block 1 as the
  primary real-data biological discovery layer.
- Endpoint-transport summaries and assignment surfaces are not themselves the
  primary STRIDE scientific object.
- Task A does not by itself close the evidence gap from endpoint-fraction
  fitting to full patient-level `(T_p, e_p)` validation.

## 9. Block 3 Parameter Freeze Notes

The shared-axis geometry/cost freeze, the recurrence-ablation shared
train-derived cohort-level generation role (compatibility label:
`shared hidden cohort effect`), the `3B-1/3B-2` split, the dense
`open_mass_scale` sweep, and the transport-comparator definitions are now all
resolved in the live contract.
The remaining items from the earlier Block 3 open-question list are therefore
treated as frozen parameter notes rather than as reasons to reopen the adopted
scientific design.

- The compatibility label `shared hidden cohort effect` used in the
  recurrence-ablation arm
  describes one rerun-specific shared train-derived cohort-level generation
  context across held-out patients. It is recorded for reproducibility only,
  is not a public benchmark axis, and does not constitute a separate
  scientific design question.
- `P(m)`, `pi_d`, `pi_e`, `kappa_d`, and `kappa_e` are frozen as train-derived
  generator calibration / diagnostics-side quantities. They parameterize the
  hidden generator and its diagnostic checks, but they are not formal Block 3
  headline benchmark metrics.
- `3B-1` is frozen as the shared-`A` benchmark over
  `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` at fixed
  `open_mass_scale = 1.0`.
- `3B-2` is frozen as the shared analysis-layer `d/e` benchmark over
  `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` at fixed
  `relation_strength = 0.15`.
- `balanced_ot_baseline` is frozen as the closed exact OT comparator on the
  fixed cost matrix `C`.
- `uot_baseline` is frozen as the soft-unbalanced transport comparator with
  rerun-shared train-side endpoint-fraction-statistics `lambda_match`
  calibration.
- `partial_ot_baseline` is frozen as the hard-budget transport comparator with
  rerun-shared train-side endpoint-fraction-statistics
  `matched_mass_budget` calibration.
- `diagonal_transport_baseline` is frozen as strict diagonal matched transport
  plus residual open mass.
- `epsilon_fixed` is frozen as a fixed smoothing pseudocount / soft floor used
  only in the emergence base measure
  `b_p^+ = normalize((x_p + epsilon_fixed) ⊙ pi_e)`. It is not a
  train-estimated biological random quantity and not a Gaussian-style noise
  term. The recommended v1 fixed value is `0.01`.
- `kappa_d` and `kappa_e` are frozen as train-estimated cohort-level
  dispersion scalars for depletion/emergence shape heterogeneity around
  `pi_d` and `pi_e`, not total change-magnitude parameters. The recommended v1
  estimator first computes the robust median total-variation deviation `tv` of
  informative `train` patients from the corresponding channel centroid, then
  maps that dispersion to the `Gamma -> normalize` concentration by
  `kappa = clip(1 / max(tv, 1e-8), 1, 200)`.
