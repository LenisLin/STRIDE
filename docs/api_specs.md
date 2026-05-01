# API Specifications

This file defines the current first-pass method-facing specification for
STRIDE. The contract is architecture-first: `src/stride/` is the reusable core
package surface, and broader estimator completeness remains a separate question.

## 1. Canonical Method Objects

### 1.1 Observation-layer object

The canonical current first-pass observation-layer quantity is the ordered
FOV/ROI community-composition vector:

- `v_{p,t,f} in R_+^K`
- `p`: patient
- `t`: ordered analysis side, typically a timepoint but allowed to be an
  equivalent declared ordered relation in a task-local setting
- `f`: FOV or ROI
- `K`: shared `K`-state basis

A valid `v_{p,t,f}` is:

- finite,
- nonnegative,
- normalized within the ROI/FOV in the current first pass so that
  `sum_k v_{p,t,f}[k] = 1` within declared numerical tolerance,
- defined on the same shared `K`-state basis as every other observation in the
  same analysis,
- accompanied by declared `mass_mode`, which is `uniform` in the current first
  pass,
- accompanied by observation mass, which is `1` for every ROI/FOV in the
  current first pass,
- accompanied by design-level `domain_label` metadata when
  domain-stratified comparison is used; the current AnnData route realizes this
  metadata through the concrete field `compartment`,
- linked explicitly to patient, ordered side, and observation-unit identifiers.

For the current first pass:

- `v_{p,t,f}[k]` is formed by counting the cells in ROI/FOV `f` assigned to
  shared community/state `k` and dividing by the total cell count in that
  ROI/FOV,
- `c(v_{p,t,f}) = v_{p,t,f}`,
- observation mass is carried separately as `mass = 1` rather than as ROI-level
  tissue amount or `||v||_1`.

When domain/compartment stratification is present, the canonical comparison
object is a domain-stratified bag-of-FOV empirical measure in
community-composition space:

- `nu_obs = sum_f w_f delta_{c(v_f)}`

where `w_f = 1` for each ROI/FOV in the current first pass.

The source/target comparison plan is caller- or task-declared and resolved by
the task layer before core estimation. Domain labels stratify observation
comparison during that resolution step and do not define state identity. The
core estimator receives resolved source/target observation evidence blocks
explicitly rather than inferring source-target biological semantics from
labels. After evidence blocks enter the core, domain is not a loss axis, state
axis, relation axis, or recurrence axis.

Important observation-layer boundary:

- canonical fitting at this layer is discrepancy or measure comparison over
  those empirical measures,
- canonical full-estimator `L_obs` uses the fixed
  `D_obs^BalancedSinkhornDivergence-v1` operator on predicted and observed
  target-side FOV bags,
- OT / Sinkhorn helpers belong to this layer only; UOT helpers are
  compatibility, diagnostic, or comparator surfaces rather than canonical
  full-estimator `L_obs`,
- the canonical docs do not treat multinomial, Dirichlet-multinomial, or
  logistic-normal generative models as the observation-layer default,
- raw histogram collapse is secondary to domain-stratified bag-of-FOV
  comparison.

The current first-pass state-construction route and the observation route are
intentionally distinct:

- shared community states are built tissue-agnostically before any
  tissue/domain stratification,
- the first-pass route is: per-cell subtype labels -> within-ROI kNN
  neighborhood subtype proportion vectors -> k-means shared community states,
- the neighborhood size `k` is user-configurable; the documented first-pass
  default is `20`,
- tissue/domain labels are attached only after shared community-state
  construction as observation-layer metadata,
- the cell-level neighborhood subtype composition used during state
  construction is not the same object as the ROI/FOV-level community
  composition used as `v_{p,t,f}`.

### 1.2 Patient-level object

The canonical patient-level method object is `(T_p, e_p)` with
`T_p = [A_p | d_p]`.

| Object | Shape | Meaning |
|---|---|---|
| `A_p` | `[K, K]` | patient-level continuity/remodeling operator |
| `d_p` | `[K]` | source-side outgoing open tendency |
| `e_p` | `[K]` | target-side incoming open-entry tendency |
| `T_p` | conceptual block object | `A_p` plus source-side open column `d_p`; storage may keep `A_p` and `d_p` separately |

Contract semantics:

- `A_p`, `d_p`, and `e_p` are fitted variables of the full STRIDE objective,
  not proxy post-processing results,
- each source row `[A_{p,i,*}, d_{p,i}]` lies on a simplex, with
  `sum_j A_{p,ij} + d_{p,i} = 1`,
- `e_p` is bounded as `0 <= e_{p,j} <= 1`,
- `d_p` and `e_p` share a bounded open-tendency scale while entering
  source-side outgoing accounting and target-side incoming reconstruction,
- composition-scale v1 post-side reconstruction uses
  `raw_post_p = q_p^- @ A_p + e_p` and
  `predicted_q_p^+ = normalize(raw_post_p)`,
- FOV-level observation fit uses
  `y_hat_f = normalize(v_source_f @ A_p + e_p)` to form the predicted
  target-side FOV bag for comparison with the observed target-side FOV bag,
- `diag(A_p)` is retention-like structure,
- `offdiag(A_p)` is remodeling-like structure,
- `A_p` is not the same thing as a normalized conditional kernel,
- if a conditional kernel is needed for exposition, it is the derived auxiliary
  object `R_{p,ij} = A_{p,ij} / (1 - d_{p,i})` when `1 - d_{p,i} > 0`,
- `d_p` is not a scalar discard term,
- `e_p` is not optional in the target design.

### 1.3 Burden-scale auxiliary objects

The patient-level scale contract also includes the following canonical
auxiliaries:

| Object | Shape | Meaning |
|---|---|---|
| `mu_p^-` | `[K]` | patient-level pre-side pseudo-mass / burden vector |
| `mu_p^+` | `[K]` | patient-level post-side pseudo-mass / burden vector |
| `q_p^-` | `[K]` | derived normalized pre-side composition |
| `q_p^+` | `[K]` | derived normalized post-side composition |

Any documented `m_p^(d)` or `m_p^(e)` summaries, whether total or state-wise,
live on the same pseudo-mass / burden scale as `mu_p^-` and `mu_p^+`.

Derived composition definitions:

- `q_p^- = mu_p^- / ||mu_p^-||_1`,
- `q_p^+ = mu_p^+ / ||mu_p^+||_1`.

Scale boundary:

- `mu_p^-` and `mu_p^+` are not canonical compositions,
- conservation is a soft burden-consistency anchor rather than literal physical
  conservation,
- composition-level structure may remain interpretable when burden-level
  comparability is weak,
- burden-level claims should be weakened or disabled when coverage or platform
  comparability is poor.

### 1.4 Cohort-level output layer

Cohort-level outputs are derived from patient-level objects. They are not a new
primary biological object, but full STRIDE v1 uses recurrence as a modeled
regularization layer that feeds back into patient-level estimation rather than
as post-hoc clustering alone. The v1 cohort recurrence interface must be able
to represent:

- which patients were included,
- which patient-level relations were compared,
- the single cohort consensus relation `R_bar = (A_bar, d_bar, e_bar)`,
- patient support count,
- dispersion around the consensus,
- recurrence fit status.

## 2. Required Inputs

### 2.1 Required identifiers and ordering

Any valid analysis must provide:

- patient identifiers,
- ordered timepoint identifiers or an equivalent ordered relation,
- FOV/ROI identifiers,
- a shared `K`-state basis or an official route to derive one,
- declared `mass_mode`, which is `uniform` in the current first pass,
- declared ordered source/target observation comparison plan, including valid
  domain strata when domain-stratified comparison is used,
- design-level `domain_label` metadata when domain-stratified observation
  comparison is used; the current AnnData route uses `compartment`,
- area or equivalent support metadata only for non-uniform future/custom
  density semantics.

Equivalent ordered relations must be explicit rather than implicit. They may be
task-local, but they must still declare the ordering used for analysis.

The state/domain boundary is part of the API contract:

- the shared `K`-state basis defines state identity,
- domain labels may stratify observation comparison, grouped discrepancies, and
  patient-relation input grouping,
- callers keep state construction and domain stratification as separate
  modeling layers,
- state geometry and the axes of `A_p`, `d_p`, and `e_p` are defined on the
  shared `K`-state basis.

### 2.2 Required quantitative inputs

A valid method entry surface must provide one of the following:

1. official observation data sufficient to derive the shared `K`-state basis
   and the observation-layer vectors `v_{p,t,f}`, or
2. direct FOV/ROI state vectors `v_{p,t,f}` on an already-declared shared
   `K`-state basis, together with explicit observation-layer domain metadata and
   declared observation mass semantics.

### 2.3 Optional priors and side inputs

Optional method inputs may include:

- `alpha`, the primary full-estimator hyperparameter controlling local fit
  versus regularization strength; the default is `0.5`,
- optional `alpha` sensitivity diagnostics,
- shared-state cost geometry for observation comparison and the
  geometry/locality objective component,
- geometry/locality prior configuration,
- grouping priors,
- drift vectors or drift-risk priors,
- uncertainty configuration,
- recurrence consensus configuration,
- patient-relation fitting hyperparameters,
- compact provenance configuration,
- additional diagnostics requested by the task or study design.

These optional priors supplement the required patient/time/FOV and shared-basis
contracts.

## 3. Minimal Computational Pipeline

The minimal method pipeline is:

1. validate identifiers, ordering, shared-basis consistency, and state/domain
   separation,
2. build or accept the shared `K`-state basis before any tissue/domain
   stratification,
3. in the official first-pass route, construct the shared basis tissue-agnostically
   from per-cell subtype labels -> within-ROI kNN neighborhood subtype
   proportion vectors -> k-means shared community states,
4. aggregate shared community/state assignments within each ROI/FOV into
   normalized community-composition vectors `v_{p,t,f}`,
5. attach observation-layer `domain_label` metadata and uniform observation
   mass (`mass = 1`, `mass_mode = "uniform"`),
6. accept the task-resolved source/target observation evidence blocks and the
   resolved comparison plan, including the provenance of valid domain strata,
   as explicit estimator input,
7. construct constrained patient-level fitted variables `A_p`, `d_p`, and
   `e_p` with a feasible parameterization in which each source row
   `[A_i,* , d_i]` lies on a simplex and `e` is bounded in `[0,1]`,
8. use deterministic identity-plus-small-open initialization
   `delta_init = min(0.05, 1 / (K + 1))`,
   `A_init = (1 - delta_init) * I_K`,
   `d_init = delta_init * 1_K`, and
   `e_init = (delta_init / K) * 1_K`; compute component baseline scales from
   that feasible numerical starting point on the same input. Except for open,
   use `scale_c = max(raw_L_c(theta_init), epsilon_norm)` and
   `normalized_L_c = raw_L_c(theta) / scale_c`, with
   `epsilon_norm = 1e-2`. `epsilon_norm` is a full-estimator
   loss-normalization floor recorded as dimensionless `float64` provenance,
   not the Task A Block 3 generator `epsilon_fixed`. A floor applied to a
   valid finite raw baseline scale is provenance-only and does not change
   `fit_status`; invalid raw losses or invalid inputs still fail explicitly,
9. optimize the full objective with default `alpha=0.5`:
   `L_total = (1 - alpha) * L_local + alpha * L_regularization`, with
   `L_local = mean(normalized_L_obs, normalized_L_open, normalized_L_geometry)`
   and
   `L_regularization = mean(normalized_L_consistency, normalized_L_recurrence)`.
   `alpha` remains the primary local-versus-regularization hyperparameter.
   `L_geometry` is a fixed normalized component of `L_local` rather than an
   independently weighted term. For each resolved evidence block, the
   observation fit uses source-side FOV vectors
   `y_hat_f = normalize(v_source_f @ A_p + e_p)` to induce the predicted
   target-side FOV bag. `D_obs` is the fixed operator inside `L_obs`, not a
   sixth loss or independently weighted component. The canonical pairwise
   observation loss is
   `L_obs_pair_raw = D_obs^BalancedSinkhornDivergence-v1(predicted target FOV bag, observed target FOV bag; C_norm)`.
   Here `C_norm = C_raw / s_C` is the state/community-level cost. The
   FOV-level ground cost is
   `G[f,g] = debiased balanced Sinkhorn composition distance(y_hat_f, y_true_g; C_norm)`;
   the outer divergence uses `G_norm = G / s_G_init`.
   `s_G_init` is fixed per evidence block from positive finite
   initialization-time FOV-level costs at deterministic
   identity-plus-small-open initialization, uses `1.0` with recorded floor
   usage if no positive finite costs exist, and is not recomputed dynamically
   during optimization. The same full estimator and canonical observation
   discrepancy operator are used across tasks. The geometry/locality raw form
   is
   `L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]`, using
   all `K` source rows and the raw canonical `A_p`. The objective-level raw
   value is the simple mean over valid fitted patients. The v1 open-channel
   complexity form is `L_open_raw = mean(d_p) + mean(e_p)` with fixed
   `scale_open = 1`, so `normalized_L_open = L_open_raw`. STRIDE v1 uses
   simple continuous differentiable component losses where feasible, but the
   assembled full objective is treated as a constrained non-convex numerical
   objective rather than as a globally convex program; the contract does not
   claim a global optimum,
10. estimate single cohort consensus recurrence with feedback into the fitted
   patient relations using `R_bar = (A_bar, d_bar, e_bar)` and
   `L_recurrence_raw = mean_p dist(R_p, R_bar)`,
11. refit or iterate patient and cohort components as specified by the frozen
   objective implementation,
12. derive patient and cohort summaries from those fitted objects,
13. emit biological outputs plus compact successful-fit provenance; audits,
    uncertainty summaries, and result-container status fields remain separate
    output surfaces.

Important boundary:

- task layers declare source/target comparisons, while the core applies the
  same full estimator and canonical observation discrepancy backend,
- the core receives resolved evidence blocks and does not treat domain as a
  core loss/state/relation/recurrence axis,
- OT / Sinkhorn and observation matching are backend or observation-layer
  comparison tools,
- they contribute to the observation data-fit term within the full objective,
- canonical `L_obs` has no observation-layer unbalanced unmatched residual;
  open behavior is expressed only through fitted biological `d/e`,
- task-local observation solver substitution is outside the `fit_stride(...)`
  full-estimator contract,
- canonical patient-level `A/d/e` outputs are emitted by the objective-driven
  patient-relation fit,
- `patient_consistency` is support for one patient-level relation across
  resolved observation evidence blocks; if fewer than two blocks are available
  for a patient, `L_consistency_raw(p) = 0` and
  `consistency_status = "insufficient_blocks"`,
- geometry/locality is a soft biological-complexity cost over raw `A_p` using
  shared-state geometry; it is not a hard support mask, and distant remodeling
  may remain when supported by the joint objective,
- `C_raw` and `C_norm` must be finite, nonnegative, symmetric `[K, K]`
  matrices on the shared state basis with diagonal entries equal to `0` within
  declared numerical tolerance; `s_C` is the median of positive finite
  off-diagonal entries of `C_raw`, and absence of such entries is a contract
  failure,
- full STRIDE v1 uses PyTorch as the canonical optimization framework, with
  AdamW as the outer optimizer and `weight_decay = 0.0`; optimizer mechanics
  are not biological regularization,
- implementations must expose optimizer availability and runtime status through
  explicit exception or result-container status surfaces when the canonical
  PyTorch/AdamW path is unavailable or cannot complete,
- any scheduler must be fixed, predeclared, and recorded in provenance; the
  canonical v1 recommendation is `ReduceLROnPlateau` on the total objective,
- the target pipeline preserves FOV-aware observation-layer comparison before
  patient-level summary derivation.

## 4. Primary Outputs

### 4.1 Required patient-level outputs

Any target-design method surface must be able to emit, for each patient:

- `A_p`,
- `d_p`,
- `e_p`,
- patient-level audit fields describing observation coverage, burden/composition
  semantics, and fit status.

### 4.2 Derived patient summaries

Derived patient summaries may include:

- diagonal retention summaries from `A_p`,
- off-diagonal remodeling summaries from `A_p`,
- source-side outgoing open-tendency summaries from `d_p`,
- target-side incoming open-entry summaries from `e_p`,
- burden-scale summaries derived from `mu_p^-`, `mu_p^+`, `m_p^(d)`, or
  `m_p^(e)`,
- derived composition summaries from `q_p^-` or `q_p^+`,
- uncertainty summaries,
- observation/objective diagnostics.

These derived summaries are secondary to the patient-level object itself.

### 4.3 Required cohort-level outputs

Any recurrence layer must be able to emit:

- patient membership/support,
- cohort consensus summary identifiers,
- cohort consensus `A/d/e` on the same shared `K`-state basis,
- patient support count,
- dispersion around the consensus,
- recurrence-layer fit status.

### 4.4 Required compact provenance

Default manuscript-level STRIDE results must include compact successful-fit
provenance alongside biological outputs. The provenance payload is a compact
parameter, loss, and protocol record for a successful full-estimator
`fit_stride(...)` fit. It is not a separate audit module and does not duplicate
result-container status fields.

Required compact successful-fit provenance schema:

```yaml
provenance_schema_version: "stride_fit_provenance.v1"
alpha: float
random_seed: int | null
initialization:
  policy: "identity_plus_small_open"
  delta_init: float
  K: int
  dtype: "float64"
loss:
  total: float
  local: float
  regularization: float
  epsilon_norm: 0.01
  local_denominator: 3
  regularization_denominator: 2
  components:
    obs: {raw: float, scale: float, normalized: float, floor_used: bool}
    open: {raw: float, scale: float, normalized: float, floor_used: bool}
    geometry: {raw: float, scale: float, normalized: float, floor_used: bool}
    consistency: {raw: float, scale: float, normalized: float, floor_used: bool}
    recurrence: {raw: float, scale: float, normalized: float, floor_used: bool}
e_bounds: [0.0, 1.0]
post_reconstruction_form: "normalize(q_minus @ A + e)"
observation_comparison_plan:
  resolved_by: "task_layer"
  n_evidence_blocks: int
  domain_policy: "observation_layer_only"
observation_discrepancy:
  operator_version: "D_obs^BalancedSinkhornDivergence-v1"
  backend: "torch"
  dtype: "float64"
  inner_epsilon_schedule: [0.5, 0.2, 0.1]
  outer_epsilon_schedule: [0.5, 0.2, 0.1]
  max_iter: 1000
  tol: 1e-6
  warning_tol: 1e-4
state_geometry:
  normalization: "C_norm = C_raw / s_C"
  s_C: float
optimizer:
  framework: "torch"
  algorithm: "AdamW"
  weight_decay: 0.0
  scheduler_policy: "none" | "ReduceLROnPlateau_on_total_objective"
recurrence:
  support_n_patients: int
  dispersion: float
detailed_optimizer_trace: bool
```

Ablation/refit experiment fits may add `ablation_mode`,
`ablation_term_handling`, and
`ablation_denominator_policy="fixed_denominator_no_reweighting"` as
experiment-only provenance. Ordinary successful reference-fit provenance is not
required to expose ablation as a user-level `fit_stride(...)` control.
Compatibility payloads may temporarily record `ablation_mode: "none"` only as a
migration label.

Only three optional provenance diagnostics are allowed:
`alpha_sensitivity`, `legacy_observation_diagnostics`, and
`optimizer_trace_ref`. These fields must not change objective semantics,
biological claims, or the canonical meaning of `L_obs`. Detailed optimizer
traces are off by default; if emitted, the compact payload keeps
`detailed_optimizer_trace: true` and points to the diagnostic through
`optimizer_trace_ref`.

The compact provenance schema does not require `fit_status`, `patient_status`,
`recurrence_status`, `evidence_block_status`, patient or evidence-block status
counts, `failure_reason`, `optimizer_failure_reason`, per-patient records, or
per-evidence-block records. Existing result containers may still carry their
own status fields.

## 5. Official Route Versus Custom Route

### 5.1 Official route

The official route is the task-insensitive route from spatial single-cell or
spot observations to:

- a shared `K`-state basis,
- valid observation-layer vectors `v_{p,t,f}`,
- the priors needed for observation-layer comparison,
- the audits needed for FOV-aware fitting and bootstrap uncertainty over
  fitted patient relation outputs.

In the current first-pass official route:

- shared community states are built tissue-agnostically from per-cell subtype
  labels -> within-ROI kNN neighborhood subtype proportion vectors -> k-means
  shared community states,
- the neighborhood size `k` is user-configurable; the documented first-pass
  default is `20`,
- ROI/FOV observation vectors are then formed by counting cells assigned to
  each shared community/state within the ROI/FOV and dividing by the ROI/FOV
  total cell count,
- observation mass is set separately to `1` with `mass_mode = "uniform"`,
- tissue/domain labels remain observation-layer metadata only.

The current implementation locations for the official route are mainly:

- `stride.basis` and `stride.api.basis` for shared-basis construction,
- `stride.data.longitudinal` and `stride.api.dataset` for longitudinal input
  validation and canonical-field normalization,
- `stride.observation` for observation-layer cloud comparison,
- `stride.api.fit`, `stride.workflows.fit_stride`, and
  `stride.outputs.fit_result` for the canonical full STRIDE fit surface,
- `stride.outputs.uncertainty` for bootstrap uncertainty over fitted patient
  relation outputs.


### 5.2 Custom route

A custom route is allowed only if it ends in the same contracts:

- same shared `K`-state semantics,
- valid nonnegative observation-layer vectors,
- declared `mass_mode`,
- explicit patient/ordering/FOV indexing,
- domain handling that does not redefine state identity,
- a compatible path to patient-level `(A_p, d_p, e_p)`.

## 6. Live API Surface

The live reusable API surface is organized under `stride.*`.

Stable first-pass public surfaces are:

- package root `stride`,
- `stride.api.dataset.DatasetHandle`,
- `stride.api.basis.BasisSpec`,
- `stride.api.fit.fit_stride(...)`,
- `stride.api.fit.build_patient_relation(...)`,
- result contracts in `stride.outputs.fit_result`,
- uncertainty contracts in `stride.outputs.uncertainty`,
- observation-layer helpers in `stride.observation` including
  `match_observation_clouds(...)`, `build_observation_kernels(...)`,
  `calibrate_match_penalty(...)`, and `compute_active_state_support(...)`.

Implementation namespaces that realize the current first-pass contract include:

- `stride.basis` for shared state-basis construction,
- `stride.data.longitudinal` for longitudinal input validation and
  canonical-field normalization,
- `stride.observation` for observation-layer cloud comparison,
- `stride.workflows.fit_stride` for the canonical fit workflow,
- `stride.latent.recurrence` for the conservative first-pass recurrence
  estimator.

Formal Full-Estimator Contract And Current Implementation Boundary:

- `fit_stride(...)` is the formal manuscript-level full STRIDE estimator
  surface.
- The full contract requires `fit_stride(...)` to fit objective-driven
  patient-level `A_p`, `d_p`, and `e_p` under source-row simplex accounting for
  `[A_p | d_p]` and bounded `e_p` constraints.
- The full contract requires `fit_stride(...)` to use the frozen objective
  grouping with default `alpha=0.5` as the primary
  local-versus-regularization hyperparameter, fixed normalized component
  combination, and baseline-scale normalization from the
  identity-plus-small-open initialization.
- The full contract requires `fit_stride(...)` to apply `L_geometry` as a
  fixed normalized component of `L_local` over raw `A_p` using valid
  shared-state `C_norm`; there is no geometry-specific objective weight.
- The full contract requires ablation configurations to refit `A_p`, `d_p`,
  and `e_p` under the ablated objective with fixed group denominators and
  without reweighting non-ablated components.
- The full contract requires `fit_stride(...)` to include single cohort
  consensus recurrence as a feedback term in estimation.
- The full contract requires `fit_stride(...)` to emit biological outputs plus
  compact provenance.
- The current code implementation supports this formal contract for the
  bounded first-pass input envelope described in `docs/state.md`: uniform-mass
  patient inputs, exactly two ordered groups per patient, valid shared-state
  geometry, PyTorch/AdamW full-objective fitting, compact successful-fit
  provenance, and recurrence/geometry/consistency refit switches.
- Inputs or runtimes outside that supported envelope must surface explicit
  non-`ok` optimizer/fit status or remain on a clearly identified compatibility
  route; they are not successful full-objective fits.
- Current implementation coverage is described by `docs/state.md`; this API
  specification records the formal contract and public interpretation boundary.
- `build_patient_relation(...)` assembles already constructed patient-level
  arrays into the canonical patient relation object.
- Observation matching contributes as an internal objective term, diagnostic,
  or backend comparison within the full-estimator contract.
- `estimate_recurrence(...)` currently implements a conservative
  consensus-template estimator with explicit deferred status when support is
  insufficient.

## 7. Backend Numerical Surfaces

Observation-layer OT/Sinkhorn helpers live behind `stride.observation` and
`stride.adapters`. These functions provide numerical comparison support for the
domain-stratified bag-of-FOV observation layer. Canonical full-estimator
`L_obs` uses the fixed, versioned, auditable
`D_obs^BalancedSinkhornDivergence-v1` operator with `C_norm = C_raw / s_C`.
The operator is torch-native, `float64`, log-domain, differentiable, balanced,
and debiased. Dense transport plans and solver diagnostics remain backend
payloads unless a task contract explicitly uses them, as in the Task A Block 3
baseline-comparison surface.

The shared-state cost matrix used by the observation operator and the
geometry/locality objective component must be validated before canonical
fitting. `C_raw` and `C_norm` are finite, nonnegative, symmetric `[K, K]`
state-geometry matrices on the shared state basis. Diagonal entries are `0`
within declared numerical tolerance. The default `s_C` is the median of
positive finite off-diagonal entries of `C_raw`. At least one such entry must
exist; otherwise canonical fitting fails explicitly. The canonical contract
does not fall back to `s_C = 1.0` for invalid state geometry and does not
silently disable geometry. `C` is a symmetric state-identity distance, while
`A_p` is a directed remodeling relation.

For each source FOV, `y_hat_f = normalize(v_source_f @ A_p + e_p)`. The
FOV-level ground cost is
`G[f,g] = debiased balanced Sinkhorn composition distance(y_hat_f, y_true_g; C_norm)`.
Tiny negative inner composition-distance values in `[-1e-10, 0)` are clamped
to `0` with provenance, and values below `-1e-10` are numerical failures. The
outer observation loss is the debiased balanced Sinkhorn divergence between the
predicted target FOV bag and observed target FOV bag under
`G_norm = G / s_G_init`. `s_G_init` is fixed per evidence block from positive
finite initialization-time FOV-level costs at deterministic
identity-plus-small-open
initialization; if no positive finite costs exist, `s_G_init = 1.0` and floor
usage is recorded. This scale is not recomputed dynamically during
optimization.

Fixed v1 numeric defaults are inner epsilon schedule `(0.5, 0.2, 0.1)`, outer
epsilon schedule `(0.5, 0.2, 0.1)`, `max_iter = 1000` per epsilon stage,
`tol = 1e-6`, `warning_tol = 1e-4`, backend `torch`, and dtype `float64`.
Empty source or target FOV bag, invalid simplex, negative entries, NaN/Inf,
invalid cost matrix, or unavailable `torch` for the canonical full estimator
fails explicitly. Padding is not allowed. Reaching `max_iter` with finite
values and final update `<= warning_tol` is usable with warning; final update
above `warning_tol` is numerical failure.

OT, Sinkhorn, and observation matching are backend or observation-comparison
surfaces. They are not canonical biology objects, do not define the public full
estimator, and do not emit canonical patient-level `A_p`, `d_p`, or `e_p`
except through the full STRIDE objective fit or explicit assembly of already
valid relation objects. Legacy UOT, `D_pos/B_pos`, and unbalanced observation
residuals are diagnostic or comparator surfaces only; they are not canonical
`L_obs`, not biological open-channel estimands, and not independent loss
components. Task A Block 3 preserves `uot_baseline` as an external comparator
name. The balanced Sinkhorn divergence operator solves the observation
discrepancy, while AdamW optimizes `A_p`, `d_p`, `e_p`, and any necessary
objective variables.
