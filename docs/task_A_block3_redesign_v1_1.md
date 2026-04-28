# Task A Block 3 Redesign Alignment v1.1

This note records how the adopted Task A Block 3 redesign aligns with the
user-provided `Block 3 redesign proposal v1` and where explicit contract
repairs were added before implementation.

It is a design-alignment note, not an execution log. The normative scientific
freeze remains
[`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md).
This note is therefore an explanatory mirror of the live scientific contract,
not a competing implementation-first source.

## 1. Status

The adopted design is a transitional `v1.1` hardening of Block 3. It keeps the
main scientific direction of the original proposal:

- hidden patient program `(A, d, e)` remains the generator only
- method inputs move to paired endpoint fractions on the shared `K` axis
- the benchmark keeps a small number of headline metrics
- MSE and MAE are retained as magnitude-error quantities rather than being
  treated as the whole biological claim

The design is transitional because it freezes Block 3 directly in the same
community-fraction semantic space used elsewhere in Task A, rather than adding
an extra measurement-layer model.

Historical proxy Block 3 ablation behavior in the current repository or in
older review packets is compatibility context only. It is not the normative
`3C` scientific contract recorded in the live spec.

## 2. Decisions Preserved From The Original Proposal

- Block 3 is no longer framed as "recover STRIDE-shaped latent objects and
  score them directly." It is framed as "recover a hidden biological program
  from paired endpoint fractions and score recovery against truth."
- Real-data-derived baseline composition remains the starting substrate.
- The hidden generator still uses patient-level `(A, d, e)`.
- The frozen public section structure is now `3A generator validation`,
  `3B baseline comparison`, and `3C ablation study`.
- `3B` is split into `3B-1 A benchmark` and `3B-2 d/e benchmark`.
- `3C` is split into `3C-1 recurrence ablation`,
  `3C-2 geometry ablation`, and `3C-3 consistency ablation`.
- metric activity remains condition-specific rather than forcing every quantity
  into every comparison condition.

## 3. Added Contract Repairs In v1.1

These items were added to make the redesign decision-complete and to avoid
known failure modes in the earlier Block 3 contract.

### 3.1 Endpoint boundary

- All ranked methods, including `stride_reference`, consume paired endpoint
  fractions `x_p` and `y_p` only.
- For `stride_reference`, Task A resolves source/target endpoint comparisons
  and valid domain strata, then passes the resolved comparison plan and
  source/target observation evidence blocks to the formal `fit_stride(...)`
  estimator. Domain resolution remains task-layer provenance and does not
  become a core loss/state/relation/recurrence axis.
- No ranked method may read hidden `(A, d, e)` or any non-public truth object
  beyond the paired endpoint fractions.
- Any exact truth companion remains internal to scoring and debugging and is
  not a ranked benchmark method.

### 3.3 3B scoring anchors

- `3B` is now an umbrella with `3B-1 A benchmark` and `3B-2 d/e benchmark`.
- The original proposal wrote transported mass as
  `T_true = x_true * A_true` and `T_hat = x_hat * A_hat`.
- `v1.1` keeps truth-anchored transported mass during `3B-1` scoring:
  `T_true = x_true * A_true` and `T_hat = x_true * A_hat`.
- This keeps `3B-1` focused on relation recovery rather than mixing baseline
  reconstruction error into the main relation headline.
- `A_MAE_active` and `A_MSE_active` are therefore interpreted as active-row
  conditional target-pattern recovery metrics rather than total
  transported-mass error metrics.
- `target_recall_at_k` is interpreted as truth-anchored off-diagonal
  target-priority recovery and is `not_applicable` at
  `relation_strength = 0.00`, where truth contains no off-diagonal target set.
- `3B-2` uses a shared analysis layer in which STRIDE contributes native
  `A/d/e`, plan-based comparators contribute native matched plan `P`, and the
  common `A/d/e` surfaces are derived before applying the open-surface metrics.
- `3B-2` keeps `profile TV` outside the phase-1 headline set.

### 3.4 3C refit-ablation semantics

- `3C` is a within-STRIDE module-necessity test on rerun-specific
  patient-level semi-synthetic realizations.
- The live `3C` ablation set is recurrence, geometry, and consistency.
- Each `3C` arm removes or zeroes the corresponding objective term and then
  refits `A_p`, `d_p`, and `e_p`.
- Observation discrepancy, explicit open-channel terms, audit/plausibility
  handling, and the non-ablated objective terms are retained.
- `3C` arms use core estimator configurations and the canonical observation
  discrepancy backend rather than a Task A-local STRIDE estimator or
  observation backend replacement.
- No-`d/e`, no-open-channel, closed, balanced, and transport-style comparisons
  remain `3B` baseline/comparator semantics, not core STRIDE ablations.
- `3C` reuses the same rerun-specific patient-level semi-synthetic
  realizations used elsewhere in Block 3 and does not introduce a public
  fixed-`m` axis.

### 3.5 3C metric/status semantics

- The live `3C` headline metrics are `A_MAE_active`, `A_MSE_active`,
  `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE`.
- In `3C`, `A` explicitly means patient-level relation operator `A_p` recovery
  on the shared `K`-state axis.
- `open_support_F1` is defined on burden-scale depletion/emergence carriers,
  not on raw `d/e > threshold` masks.
- For each patient and channel, support is the smallest state set covering at
  least 95% of normalized open burden, with deterministic state-id
  tie-breaking.
- Compute channel-level support F1 on those support sets.
- If one open channel has truth total mass `= 0`, that channel-level
  `open_support_F1` is `not_applicable`.
- Patient-level `open_support_F1` is the arithmetic mean across the reported
  channel-level values.
- If both open channels are `not_applicable`, patient-level
  `open_support_F1` is `not_applicable`.
- `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` remain quantitative profile-recovery
  metrics.
- Any shared cohort/common-structure generator quantity is a train-derived
  internal generator detail recorded for reproducibility only, not a public
  benchmark axis or a reopened scientific design question.
- Here `shared` means that all held-out `test` patients in one rerun reuse the
  same train-derived cohort-level generator quantities
  (`P(m)`, `pi_d`, `pi_e`, `kappa_d`, `kappa_e`) while each patient's
  realized hidden truth still depends on that patient's own `x_p` plus
  patient-level sampling.

### 3.6 Explicit metric-status rule

- Metrics that are not applicable or not estimable must not silently collapse
  into blanks or zeros without explanation.
- At minimum, downstream reporting should distinguish:
  - `reported`
  - `not_applicable`
  - `not_estimable`

## 4. Adopted Benchmark Contract

### 4.1 Common generator flow

1. split patients into `24 train / 8 test`
2. derive a real-data-grounded baseline composition `x_p` from held-out `test`
   patients only
3. estimate weak empirical priors from real `TC -> IM` change in the `train`
   split only
4. define the train-side open-mass proxy as
   `m_proxy = 0.5 * || q_TC - q_IM ||_1`; the factor `0.5` is definitional,
   not a tuned hyperparameter
5. build `P(m)` as the empirical distribution of `m_proxy` over the `train`
   patients and sample each held-out `test` patient `m_p` from that empirical
   distribution with replacement
6. reuse the resulting patient-level semi-synthetic open-burden realizations
   across benchmark and ablation evaluations in that rerun rather than turning
   them into a public fixed-`m` section axis
7. estimate `pi_d` and `pi_e` as the mean normalized positive-part depletion
   and emergence changes over informative `train` patients, and estimate
   `kappa_d` and `kappa_e` as train-estimated cohort-level dispersion scalars
   for depletion/emergence shape heterogeneity around the corresponding
   channel centroid by first computing the robust median total-variation
   deviation `tv` and then mapping it to the `Gamma -> normalize`
   concentration with `kappa = clip(1 / max(tv, 1e-8), 1, 200)`
8. construct a weak patient-anchored open scaffold `F_open(x_p)` with
   depletion base `b_p^- = normalize(x_p ⊙ pi_d)` and emergence base
   `b_p^+ = normalize((x_p + epsilon_fixed) ⊙ pi_e)`, where
   `epsilon_fixed = 0.01` is a fixed smoothing pseudocount / soft floor
9. sample `Delta_p^-` and `Delta_p^+` from the frozen `Gamma -> normalize`
   family, using capped depletion allocation so that
   `sum(Delta_p^-) = sum(Delta_p^+) = m_p` and `Delta_{p,i}^- <= x_{p,i}` by
   construction
10. derive `d_p` and `e_p` from `Delta_p^-` and `Delta_p^+`, build matched
   source mass `u_p = x_p - Delta_p^-`, and apply the frozen relation scenario
   on `u_p` only to obtain `M_p`
11. build the latent target `y_p = colsum(M_p) + Delta_p^+`
12. expose paired endpoint fractions `x_p` and `y_p` to ranked methods only
13. score inferred outputs against hidden truth
14. repeat the generator rerun `10` times and treat rerun as the outer
   statistical unit

Within this common flow, `train` calibration is intentionally limited to weak
open-level and state-level priors:

- allowed train-derived quantities: `P(m)`, `pi_d`, `pi_e`, `kappa_d`, and
  `kappa_e`; these are generator calibration / diagnostics-side quantities, not
  Block 3 headline benchmark metrics
- in the recurrence-ablation realization set, these allowed train-derived
  quantities are the only cohort-level quantities shared across held-out
  patients within a rerun; the patient-specific hidden truth is still realized
  from patient-specific `x_p` plus patient-level sampling
- forbidden train-derived quantities: state-pair relation templates,
  off-diagonal source-target motifs, and direct source-target mapping rules
- each shared community `k` receives a frozen community-identity vector `g_k`
  from the upstream atlas/community-correspondence
  `community x cell subtype` row-fraction surface; `UNKNOWN` is retained in
  the main contract
- the shared-axis biological neighborhood is encoded by
  `C_raw[i,j] = sqrt(JS(g_i, g_j))`, with `C_raw[i,i] = 0`
- let `s_C` be the median of the positive off-diagonal entries of `C_raw`;
  the frozen normalized cost matrix is `C = C_raw / s_C`
- this fixed `C` is reused both as the benchmark biological neighborhood and
  as the comparator-facing OT cost matrix; it is not train-derived pairwise
  topology
- `train` therefore remains limited to `P(m)`, `pi_d`, `pi_e`, `kappa_d`, and
  `kappa_e`

### 4.2 3A generator validation

- role: continuous descriptive validation of the frozen generator and
  fraction-space generator semantics, not a standalone gate
- validation surface: the same rerun-specific patient-level semi-synthetic
  realizations later consumed by `3B` and `3C`
- fixed `m_p` strata are not part of the public `3A` contract
- frozen formal objects:
  - held-out cohort `community-space` target fraction surface
    `S_real^(r) = mean_{p in test_r} q_{p,IM}` versus
    `S_syn^(r) = mean_{p in test_r} y_p`
  - held-out `g_k`-projected identity-aware target fraction surface
    `B_real^(r) = sum_k S_real^(r)[k] g_k` versus
    `B_syn^(r) = sum_k S_syn^(r)[k] g_k`
- formal metrics: `Pearson correlation`, `MAE`, `MSE`, and `JS divergence`
  on each of those two object pairs
- `rerun variability`: between-rerun stability of the above realism /
  plausibility objects and their score summaries rather than a detached bare
  metric name
- interpretation: intuitive `real TC -> IM` versus `synthetic TC -> y`
  change language remains allowed, but the formal contract is cohort-level
  realism plus `g_k`-based identity-aware biological plausibility, not
  patient-by-patient reconstruction, not a comparator study, and not a public
  `P(m)` axis

### 4.3 3B baseline comparison

- `3B` is the external baseline-comparison umbrella and now splits into
  `3B-1 A benchmark` and `3B-2 d/e benchmark`
- `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` is the matched-structure
  control used by `3B-1`
- `relation_strength` denotes the off-diagonal matched-mass fraction on the
  shared `K`-community axis
- `max_offdiag_targets_per_source` is deterministically derived from
  `relation_strength` as `(0.00 -> 0, 0.05 -> 1, 0.15 -> 1, 0.30 -> 2)`
- `C` encodes biological neighborhood on the shared community axis, so
  "nearest" means lowest off-diagonal `C[i,j]`
- for the same patient within the same rerun, `3B-1` varies only
  `relation_strength` at fixed `open_mass_scale = 1.0`
- within `3B-1`, the held-out patient identities, `x_p`, sampled `m_p`, and
  the derived truth-side open quantities remain fixed across the
  `relation_strength` grid
- within `3B-2`, held-out patient identities are reused across the full
  `open_mass_scale_grid` at fixed `relation_strength = 0.15`
- in `3B-2`, reused means shared patient identities, shared `x_p`, shared
  matched-structure setting, shared support rule, shared `C`, and shared
  `pi_d / pi_e` ratio
- in `3B-2`, the truth-side open quantities `delta_minus_scaled`,
  `delta_plus_scaled`, and the derived `d_p`, `e_p`, `y_p` vary with
  `open_mass_scale`
- the open-burden design is now frozen as
  `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]`, with `0.0` as the
  near-closed limit and `1.0` as the canonical open regime in the first phase
- the dense sweep scales only the hidden open burden:

```python
delta_minus_scaled = open_mass_scale * delta_minus
delta_plus_scaled = open_mass_scale * delta_plus
```

- `3B-1` fixes `open_mass_scale = 1.0`
- `3B-2` fixes `relation_strength = 0.15`
- the dense `3B-2` sweep keeps `x`, `relation_strength = 0.15`, support rule,
  `C`, and the `pi_d / pi_e` ratio fixed
- rerun design remains frozen to `10` reruns with `24 train / 8 test`
- phase-1 engineering outputs remain stratified by the section-specific public
  control
- any later pooled or headline summary across multiple public-control values is
  deferred to post-processing and result interpretation

#### 4.3.1 3B-1 A benchmark

- methods: `stride_reference`, `balanced_ot_baseline`, `uot_baseline`,
  `partial_ot_baseline`, `diagonal_transport_baseline`
- `3B-1` compares recovery of the shared relation surface `A_p` across
  `stride_reference` and the transport-family comparators on the same paired
  endpoint inputs
- Phase 1 computes the frozen `3B-1` metric set only; competitiveness or
  acceptable-loss judgments are deferred to post-processing
- formal metrics: `A_MAE_active`, `A_MSE_active`, `target_recall_at_k`
- `3B-1` uses `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]`
- `3B-1` fixes `open_mass_scale = 1.0`
- `relation_strength = 0.00` keeps the matched mass diagonal-only and sets
  `max_offdiag_targets_per_source = 0`
- `relation_strength = 0.05` sets `max_offdiag_targets_per_source = 1` with
  off-diagonal matched-mass fraction `0.05`
- `relation_strength = 0.15` sets `max_offdiag_targets_per_source = 1` with
  off-diagonal matched-mass fraction `0.15`
- `relation_strength = 0.30` sets `max_offdiag_targets_per_source = 2` with
  off-diagonal matched-mass fraction `0.30`
- if `max_offdiag_targets_per_source = 1`, all off-diagonal matched mass goes
  to the nearest neighbor under `C`
- if `max_offdiag_targets_per_source = 2`, off-diagonal matched mass is split
  across the `2` nearest neighbors by normalized `exp(-C[i,j])` weights with
  no extra temperature parameter
- off-diagonal support is nested across
  `relation_strength = 0.05 -> 0.15 -> 0.30`
- these relation-strength levels are reviewer-facing benchmark challenge levels
  induced by the frozen support rule and shared cost matrix `C`; that
  generator-side structural bias remains an acknowledged design limitation
- truth-anchored transported-mass scoring remains the scoring anchor
- `target_recall_at_k` remains a truth-anchored off-diagonal target-priority
  metric and is `not_applicable` at `relation_strength = 0.00`, where truth
  contains no off-diagonal target set (`k = 0`)

#### 4.3.2 3B-2 d/e benchmark

- methods: `stride_reference`, `uot_baseline`, `partial_ot_baseline`,
  `diagonal_transport_baseline`
- `3B-2` asks whether STRIDE recovers `d/e` more accurately than the open
  comparator set under the same rerun-specific realizations
- formal metrics: `open_support_F1`, `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`
- `3B-2` fixes `relation_strength = 0.15` and therefore
  `max_offdiag_targets_per_source = 1`
- `3B-2` uses `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` as its public
  open-burden gradient
- within each rerun, held-out patients are reused across the full
  `open_mass_scale_grid` at fixed `relation_strength = 0.15`
- `3B-2` is defined on a shared analysis layer rather than on method-specific
  native output formats
- `3B-2` reuses the full shared `3C` `open_support_F1` contract, including
  the burden-scale support definition and the channel-level / patient-level
  status semantics
- because `open_mass_scale = 0.0` implies zero truth depletion and zero truth
  emergence by construction, `open_support_F1` is `not_applicable` under that
  condition
- `d_MAE`, `e_MAE`, `d_MSE`, and `e_MSE` remain reported under
  `open_mass_scale = 0.0`

Native representation and shared analysis layer:

- `stride_reference` calls the formal `fit_stride(...)` frozen reference
  configuration on Task A-declared source/target inputs and emits native
  fitted `A / d / e`
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` emit native matched plan `P`
- the shared `3B` analysis layer derives `A / d / e` before scoring

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

Comparator minimum pseudocode:

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

Train-side shared calibration minimum:

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

- `3B-2` keeps `profile TV` outside the phase-1 headline set

### 4.4 3C ablation study

#### 4.4.1 3C-1 recurrence ablation

- `3C-1` is a within-STRIDE module-necessity test, not an open-gradient
  benchmark and not an external baseline section
- methods: `stride_reference`, `recurrence_ablation`
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`,
  `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`
- `recurrence_ablation` removes or zeroes only cohort consensus
  recurrence/common-structure regularization and then refits `A_p`, `d_p`,
  and `e_p`
- observation discrepancy, explicit open-channel terms, geometry/locality,
  patient consistency, and audit / plausibility handling are retained
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations used elsewhere in Block 3

#### 4.4.2 3C-2 geometry ablation

- methods: `stride_reference`, `geometry_ablation`
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`,
  `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`
- `geometry_ablation` removes or zeroes only geometry/locality and then refits
  `A_p`, `d_p`, and `e_p`
- observation discrepancy, explicit open-channel terms, patient consistency,
  cohort consensus recurrence/common structure, and audit / plausibility
  handling are retained
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations used elsewhere in Block 3

#### 4.4.3 3C-3 consistency ablation

- methods: `stride_reference`, `consistency_ablation`
- formal metrics: `A_MAE_active`, `A_MSE_active`, `open_support_F1`,
  `d_MAE`, `e_MAE`, `d_MSE`, `e_MSE`
- `consistency_ablation` removes or zeroes only patient consistency and then
  refits `A_p`, `d_p`, and `e_p`
- observation discrepancy, explicit open-channel terms, geometry/locality,
  cohort consensus recurrence/common structure, and audit / plausibility
  handling are retained
- the comparison reuses the same rerun-specific patient-level semi-synthetic
  realizations used elsewhere in Block 3

For all `3C` arms, `A_MAE_active` and `A_MSE_active` reuse the same
truth-anchored active-row conditional target-pattern scoring definition as
`3B`, but here they serve as matched-realization patient-level
relation-recovery readouts rather than relation-gradient headline metrics.
No-`d/e`, no-open-channel, closed, balanced, or transport-style comparisons
belong to `3B`, not to `3C`.

## 5. Phase 1 Freeze Addendum

This addendum records the remaining Phase 1 freeze choices that are necessary
before Phase 2 script skeleton work can begin.

### 5.1 Source-of-truth policy for this pass

- This pass uses the on-disk live Task A documents as the normative authority
  layer.
- Normative scientific priority remains:
  1. [`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md)
  2. this note
  3. active derived docs as operational mirrors of the same live contract
- Audit/orientation filenames referenced in earlier discussion but not present
  on disk in the current workspace are not treated as normative sources for
  this pass.
- Active local Block 3 code/tests and legacy packet surfaces may expose
  compatibility lag or implementation reality, but they are evidence of lag
  only and do not enter the normative authority ladder.
- Current code facts may expose compatibility lag, but they do not reopen the
  scientific freeze if the spec already resolves the point.

### 5.2 Fraction-space v1 freeze

- The v1 public method-facing contract is paired endpoint fractions `x_p` and
  `y_p` on the shared `K`-state axis.
- `x_p` and `y_p` are strict simplex fractions; Block 3 v1 does not add a
  burden-like scale or a separate observation-count layer.
- The following remain hidden for truth-only evaluation:
  `(A_p, d_p, e_p)` plus any additional internal truth companions required for
  scoring/debugging.

### 5.3 Core v1 condition scope

- Core v1 / Phase 2 condition scope is:
  - `3A`: held-out cohort generator-validation summaries for
    `community-space` realism, `g_k`-projected identity-aware plausibility,
    and rerun stability
  - `3B-1`: `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` at fixed
    `open_mass_scale = 1.0`
  - `3B-2`: `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` at fixed
    `relation_strength = 0.15`
  - `3C-1`: rerun-specific patient-level semi-synthetic realizations reused for
    recurrence ablation
  - `3C-2`: rerun-specific patient-level semi-synthetic realizations reused for
    geometry ablation
  - `3C-3`: rerun-specific patient-level semi-synthetic realizations reused for
    consistency ablation
- `3B-1` is the matched-structure gradient benchmark.
- `3B-2` is the open-burden gradient benchmark.
- In `3C-1`, the method contrast is `stride_reference` versus
  `recurrence_ablation` on the same rerun-specific realization set.
- In `3C-2`, the method contrast is `stride_reference` versus
  `geometry_ablation` on the same rerun-specific realization set.
- In `3C-3`, the method contrast is `stride_reference` versus
  `consistency_ablation` on the same rerun-specific realization set.
- Optional sensitivity-only extensions are allowed as sidecars, but they are
  explicitly outside the core v1 / Phase 2 skeleton requirement set.

### 5.4 3C live metric/status contract

- The live `3C` contract reuses native-output metrics:
  `A_MAE_active`, `A_MSE_active`, `open_support_F1`, `d_MAE`, `e_MAE`,
  `d_MSE`, `e_MSE`.
- In `3C`, `A` means patient-level relation operator `A_p` recovery on the
  shared `K`-state axis.
- Standard `reported` / `not_applicable` / `not_estimable` status semantics may
  still be used downstream.
- Each `3C` arm must refit `A_p`, `d_p`, and `e_p` after removing or zeroing
  the corresponding recurrence, geometry, or consistency objective term.
- The live evidence path for `H2` is fully determined by the native
  patient-level metrics listed above.

### 5.5 Review, extraction, export, and rerun contract

- Existing review/extraction packet surfaces remain public reporting and
  compatibility routes only: primary method review, family review, cohort
  review, per-subexperiment extraction tables, the compact human summary, and
  the objective extraction memo.
- Those review/extraction surfaces, preserved historical packets, and any
  wrapper/builder result surfaces are result-facing implementation context
  only. They are not current live Block 3 scientific authority and not
  canonical Block 3 result authority for this pass.
- Baseline comparison and ablation remain separated by section and
  the compatibility/reporting label `evaluation_family`.
  `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` stay in `3B`, while
  `recurrence_ablation`, `geometry_ablation`, and `consistency_ablation` stay
  in `3C`. No-`d/e`, no-open-channel, closed, balanced, and transport-style
  comparisons remain `3B` comparator semantics.
- Task A may assemble source-target inputs, resolved evidence blocks, and
  comparison-plan metadata for `stride_reference`; the STRIDE reference fit
  uses the formal estimator and canonical observation backend.
- Layer boundaries remain explicit:
  patient-only metrics stay patient-level, condition summaries stay
  condition-level, and the live `3C` contract is expressed through native
  patient-level metrics only.
- Generator rerun remains the outer statistical unit. Raw, review, and
  extraction surfaces must preserve at least:
  `rerun_id`, `subexperiment_id`, `evaluation_family`,
  `metric_name`, `metric_role`, and metric-status fields across all Block 3
  sections. Method-bearing sections (`3B`, `3C-1`, `3C-2`, `3C-3`) must
  additionally preserve `method_name` and `method_class`.
- `3B-1` must additionally preserve `relation_strength`.
- `3B-2` must additionally preserve `open_mass_scale`.
- `3A` remains generator validation and must not be forced into method-bearing
  semantics. Within that routing layer, `evaluation_family` remains a
  reporting container rather than a formal truth/scoring axis.
- Implementation surfaces should use `rerun_id` together with the frozen
  subexperiment and condition identifiers as the public routing names for the
  rerun-specific semisynthetic context.
- Packet/index wording should mirror the same v1 metric and status contract
  immediately rather than lagging behind bundle output names or compatibility
  carriers, while remaining subordinate to the live doc authority chain above.

### 5.6 Phase 2 implementation boundary

- Expected Phase 2 touchpoints are the active Block 3 modules:
  `tasks/task_A/block3/contracts.py`,
  `tasks/task_A/block3/registry.py`,
  `tasks/task_A/block3/execution.py`,
  `tasks/task_A/block3/bundle.py`, and
  `tasks/task_A/block3/review.py`.
- Mirror updates may be needed in existing workflow, packet, contract, and test
  surfaces if they expose the public Block 3 contract, but Phase 2 should not
  introduce new benchmark families or new top-level workflow entrypoints.
- Phase 2 skeleton/header generation must cover the frozen condition and
  stratum names, metric
  names, metric-status semantics, review/extraction routing, and rerun-aware
  layer boundaries.
- Explicitly out of scope for Phase 2:
  full-data execution, smoke/full reruns, demo-test restart, Block 0-2 changes,
  reopening metric roles, and reopening the scientific redesign.

## 6. Scope Boundary

This note does not claim that the redesign is already implemented in code or
already reflected in historical Block 3 result packets. It records the adopted
scientific target for the next Block 3 rebuild.

## 7. Blocking Status

- No high-level Phase 2 scientific framing blocker remains in the on-disk
  canon after this addendum.
- The shared-axis geometry/cost freeze is now resolved in the on-disk canon and
  is no longer outstanding.
- The remaining gaps are documentation/code synchronization items around the
  already-frozen `epsilon_fixed`, `kappa_d/kappa_e`, and
  reproducibility-oriented documentation of the shared cohort-effect generator
  quantity for the recurrence-ablation realization set,
  plus compatibility-lag items in the current code surface around naming and
  routing cleanup.
- Those remaining items do not reopen the adopted redesign; they are tracked
  explicitly below as freeze notes or as Phase 2 implementation work.

## 8. Block 3 Parameter Freeze Notes

The shared-axis geometry/cost freeze, the `3B-1/3B-2` split, the dense
`open_mass_scale` sweep, and the transport-comparator family are now resolved
and are no longer outstanding. The remaining items from the earlier Block 3
open-question list are now treated as frozen parameter notes rather than as
reasons to reopen the adopted Block 3 framework.

- The compatibility label `shared hidden cohort effect` used in the
  recurrence-ablation arm
  describes one rerun-specific shared train-derived cohort-level generation
  context across held-out patients. It is recorded for reproducibility only,
  is not a public benchmark axis, and does not constitute a separate
  scientific design question.
- `P(m)`, `pi_d`, `pi_e`, `kappa_d`, and `kappa_e` are frozen as train-derived
  generator calibration / diagnostics-side quantities. They are not Block 3
  headline benchmark metrics.
- `3A` is frozen on two held-out cohort-level validation objects only: the
  `community-space` target fraction surface and the corresponding
  `g_k`-projected identity-aware surface. It has no dedicated formal
  null/random comparator contract, and any compatibility sanity export remains
  optional sidecar context only.
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
