# STRIDE Design Freeze

This document is the canonical Step 1 design freeze for the intended full
STRIDE method. It defines full STRIDE itself. It does not describe the current
live Task A proxy stack except where implementation status is stated
explicitly.

If a task doc, result memo, or archived proposal conflicts with this file, this
file wins.

## 1. Purpose

STRIDE is frozen here as a joint longitudinal spatial remodeling framework on a
shared community/state axis. The method is not defined as:

- per-patient estimation plus post hoc cohort narration,
- an observation-layer OT ontology by itself,
- a Task A proxy surface elevated into the global method definition.

The purpose of the full method is to estimate patient-level open remodeling
relations under a predefined objective and report cohort consensus
recurrence/dispersion support from partial, multi-FOV spatial observations.

## 2. Problem Definition

STRIDE addresses longitudinal or ordered-relational spatial data with:

- partial and uneven FOV/ROI coverage,
- heterogeneous within-patient spatial structure,
- no reliable one-to-one physical matching of observed regions,
- a need to reason at both patient and cohort scales.

The method therefore needs to represent both:

- patient-specific remodeling behavior on a shared state axis,
- cohort-level common structure that is estimated on that same axis rather than
  narrated afterward.

## 3. Core Modeled Objects

### 3.1 Shared axis

All full-STRIDE quantities are defined on one shared `K`-state/community axis.
Domain/compartment labels remain observation-layer strata and do not redefine
state identity.

### 3.2 Patient-level relation

For patient `p`, the primary method object is `(T_p, e_p)` with
`T_p = [A_p | d_p]`.

- `A_p in R_+^{K x K}` is the patient-level continuity/remodeling operator.
- `d_p in [0,1]^K` is the source-side outgoing open tendency.
- `e_p in [0,1]^K` is the target-side incoming open-entry tendency.
- Each source row `[A_{p,i,*}, d_{p,i}]` lies on a simplex; equivalently,
  `sum_j A_{p,ij} + d_{p,i} = 1`.
- `d_p` and `e_p` share a bounded open-tendency scale while entering different
  accounting views.
- Composition-scale v1 target-side reconstruction uses
  `raw_post_p = q_p^- @ A_p + e_p` and
  `predicted_q_p^+ = normalize(raw_post_p)`.
- Observation fit is evaluated over task-declared source/target observation
  comparisons.
- For each eligible source-side FOV/community-composition vector `v_source_f`,
  the patient relation induces
  `y_hat_f = normalize(v_source_f @ A_p + e_p)`.
- The induced predicted target-side FOV vectors form a domain-stratified
  bag-of-FOV empirical measure.
- This predicted measure is compared with the observed target-side
  domain-stratified bag-of-FOV empirical measure.
- `A_p`, not a normalized conditional kernel, is the canonical patient object.
- In the full estimator, `A_p`, `d_p`, and `e_p` are objective-driven fitted
  variables. Source-row simplex accounting and bounded `e_p` are hard
  contracts.

`A_p`, `d_p`, and `e_p` are model objects inferred under partial observation.
They are not direct proof of literal physical transport, true disappearance, or
true neogenesis.

For exposition, an optional derived augmented display object may be written as
`M_aug = [[A_p, d_p], [e_p^T, 0]]`. This display object summarizes source-side
outgoing and target-side incoming open-channel structure and does not replace
the fitted variables `A_p`, `d_p`, and `e_p`.

### 3.3 Burden-scale auxiliaries

Full STRIDE also carries patient burden-scale auxiliaries on the same shared
axis:

- `mu_p^-` and `mu_p^+` for pre/post pseudo-mass or burden,
- derived normalized compositions `q_p^-` and `q_p^+`,
- open-channel burden summaries such as `m_p^(d)` and `m_p^(e)`.

Burden and composition remain separate semantic scales.

### 3.4 Cohort-level common structure

Full STRIDE includes an explicit cohort-level common-structure layer defined on
patient relations, not on pooled FOVs as the primary unit.

For full-estimator v1, that cohort layer represents a single cohort consensus
relation on the same shared axis, together with support counts, patient
membership, dispersion around the consensus, optional low-dimensional cohort
embeddings or coordinates, and explicit fit/deferred status.

The cohort object is real modeled structure. It is not merely a descriptive
summary written after patient-level estimation.

## 4. What Cohort-Level Common Structure Means

In the frozen full-estimator v1 story, cohort-level common structure means:

- a cohort consensus relation over `(A_p, d_p, e_p)`,
- a shared recurrence space in which patient relations can be compared with the
  cohort-supported consensus,
- a modeled object that can be regularized, audited, and validated.

It is both:

- a real model output,
- and something induced by explicit loss/regularization across patients.

Full STRIDE therefore cannot be reduced to "estimate each patient separately,
then narrate the cohort later." The cohort layer is part of the intended
method.

## 5. Joint Modeling Statement

Full STRIDE is a joint framework over the shared state axis. Conceptually, it
contains all of the following components:

1. An observation-layer fit over domain-stratified bag-of-FOV empirical
   measures on the shared axis.
2. A patient relation fitting or assembly layer that estimates constrained
   fitted variables `(A_p, d_p, e_p)`.
3. An explicit open-relation treatment in which source-side outgoing and
   target-side incoming open-channel tendencies remain modeled objects rather
   than forced closure residuals.
4. A cohort-level common-structure layer that operates on patient relations and
   encourages recurrent/shared structure across patients.
5. Structural regularization and explicit audit/failure handling that keep the
   outputs scientifically honest.

The full-estimator objective contract is frozen at the objective-group and
normalization-policy level.

`L_total = (1 - alpha) * L_local + alpha * L_regularization`

with fixed group-internal means:

`L_local = mean(normalized_L_obs, normalized_L_open, normalized_L_geometry)`

`L_regularization = mean(normalized_L_consistency, normalized_L_recurrence)`

`L_local` combines:

- observation data fit,
- open-channel sparsity/complexity regularization,
- geometry/locality prior.

`L_regularization` combines:

- patient consistency,
- cohort recurrence.

The default `alpha` is `0.5`. `alpha` is the primary
local-versus-regularization hyperparameter. `alpha` sensitivity grids are
optional diagnostics and are recorded in provenance when run.

Except for the open-channel component, component losses are normalized by
baseline scales computed from the deterministic identity-plus-small-open
initialization on the same input:

`scale_c = max(raw_L_c(theta_init), epsilon_norm)`

`normalized_L_c = raw_L_c(theta) / scale_c`

where:

- `theta_init` is the deterministic identity-plus-small-open initialization
  with `delta_init = min(0.05, 1 / (K + 1))`,
  `A_init = (1 - delta_init) * I_K`, `d_init = delta_init * 1_K`, and
  `e_init = (delta_init / K) * 1_K`,
- `epsilon_norm = 1e-2`,
- `epsilon_norm` is the full-estimator loss-normalization floor, not the Task
  A Block 3 generator `epsilon_fixed`; equal numeric values do not imply equal
  semantics,
- the canonical dtype is `float64`, and `epsilon_norm` is recorded in
  provenance as a dimensionless modeling floor.

The open-channel component is the explicit exception:

`scale_open = 1`

`normalized_L_open = L_open_raw = mean(d_p) + mean(e_p)`

If a raw baseline loss is valid and finite but the baseline scale is near zero,
the normalization floor is successful-fit provenance only. It does not change
`fit_status` and does not by itself create a warning or failure. Provenance
records the component raw value, effective scale, normalized value,
`epsilon_norm`, and floor flag. The floor does not rescue invalid raw losses or
invalid inputs: NaN or Inf values, illegal negative raw losses, empty
observation bags, invalid simplexes, invalid cost matrices, unavailable
`torch`, and related contract violations still fail fast. Under C3, low-level
contract violations use
canonical `ContractError` semantics rather than provenance null fields. Because
the initialization is primarily diagonal and `C_norm[i,i] = 0`, the geometry
baseline may be zero or near zero; that is normal provenance rather than a
geometry-specific failure or scale rule.

For consistency normalization, the `L_obs` normalization scale is determined
first. `L_consistency_raw` is then computed from normalized `L_obs` block
values. The consistency normalization scale is then computed from
`L_consistency_raw(theta_init)` and `epsilon_norm`.

Ablations use fixed group denominators and do not re-average active terms. The
objective coefficient of every non-ablated component must match the reference
fit. Operationally, an ablation may be implemented as term removal or
zero-weighting, but the fixed-denominator objective is:

- geometry ablation:
  `L_local = (normalized_L_obs + normalized_L_open + 0) / 3`,
- recurrence ablation:
  `L_regularization = (normalized_L_consistency + 0) / 2`,
- consistency ablation:
  `L_regularization = (0 + normalized_L_recurrence) / 2`.

Ablation/refit provenance records `ablation_mode` and whether that experiment
arm used a remove or zero-weight implementation route. Ordinary successful
reference-fit provenance does not expose ablation as a required user-level API
control.

For patient `p` and resolved observation evidence block `b`, `L_obs` compares
the model-implied predicted target-side FOV bag induced by `(A_p, e_p)` with
the observed target-side FOV bag. The task layer owns source/target/domain
resolution and comparison-plan instantiation; the core estimator receives
resolved source/target observation evidence blocks. After that resolution,
domain is not a loss axis, state axis, relation axis, or recurrence axis inside
the core estimator.

`y_hat_{p,b,f} = normalize(v_source_{p,b,f} @ A_p + e_p)`

`D_obs` is the fixed operator inside `L_obs`. It is not a sixth loss term and
has no independent weight. `L_obs_raw` is averaged over patients and resolved
evidence blocks using balanced averaging. The canonical v1 pairwise
observation loss is:

`L_obs_pair_raw = D_obs^BalancedSinkhornDivergence-v1(predicted target FOV bag, observed target FOV bag; C_norm)`

where:

- `C_norm = C_raw / s_C` is the state/community-level cost normalization,
- `D_obs^BalancedSinkhornDivergence-v1` is a fixed, versioned, auditable,
  torch-native, `float64`, log-domain, differentiable, balanced, debiased
  Sinkhorn divergence operator,
- the FOV-level ground cost is
  `G[f,g] = debiased balanced Sinkhorn composition distance(y_hat_f, y_true_g; C_norm)`,
- tiny negative inner composition-distance values in `[-1e-10, 0)` are clamped
  to `0` with provenance; values below `-1e-10` are numerical failures,
- `L_obs_pair_raw` is the debiased balanced Sinkhorn divergence between the
  predicted target FOV bag and observed target FOV bag under `G_norm`,
- `G_norm = G / s_G_init`,
- `s_G_init` is a fixed evidence-block scale computed once at deterministic
  identity-plus-small-open initialization from positive finite
  initialization-time FOV-level costs; if no positive finite costs exist,
  `s_G_init = 1.0` and floor usage is recorded,
- `s_G_init` is not recomputed dynamically during optimization,
- fixed v1 numeric defaults are inner epsilon schedule `(0.5, 0.2, 0.1)`,
  outer epsilon schedule `(0.5, 0.2, 0.1)`, `max_iter = 1000` per epsilon
  stage, `tol = 1e-6`, `warning_tol = 1e-4`, backend `torch`, and dtype
  `float64`,
- empty source or target FOV bags, invalid simplexes, negative entries,
  NaN/Inf values, invalid cost matrices, or unavailable `torch` for the
  canonical full estimator fail explicitly; padding is not allowed,
- reaching `max_iter` with finite values and final update `<= warning_tol` is
  usable with warning; final update above `warning_tol` is numerical failure,
- compact successful-fit provenance records the canonical observation
  discrepancy operator version, backend, dtype, epsilon schedules, `max_iter`,
  `tol`, `warning_tol`, and state-geometry normalization used by the fit.

The canonical observation layer does not include an unbalanced unmatched
residual. Open behavior is expressed only by fitted biological `d/e`.
Legacy UOT, `D_pos/B_pos`, and unbalanced observation residual diagnostics may
appear only as diagnostic or external-comparator surfaces, including the Task A
Block 3 `uot_baseline` comparator name; they are not canonical `L_obs`, not
biological `d/e`, and not independently weighted loss components.

Patient-level `A/d/e` are fitted model quantities under partial observation.
Manuscript-level biological process claims require cohort-level
recurrence/support/dispersion rather than a single-patient observation
diagnostic or open tendency alone.

STRIDE v1 uses simple continuous differentiable component losses where
feasible. The assembled full objective is treated as a constrained non-convex
numerical objective. The live contract does not claim global convexity or a
global optimum. Local optimum risk is handled through deterministic
initialization, a fixed optimizer protocol recorded in successful-fit
provenance, explicit result status where applicable, and optional stability
diagnostics.

The default numerical initialization is identity-plus-small-open:

`delta_init = min(0.05, 1 / (K + 1))`

`A_init = (1 - delta_init) * I_K`

`d_init = delta_init * 1_K`

`e_init = (delta_init / K) * 1_K`

This is the feasible numerical starting point and the starting point for
baseline normalization scale computation. It is not a biological estimate and
does not serve as final evidence for `A/d/e`. The initialization is primarily
continuity-oriented while allowing small source-side depletion and target-side
emergence. The constrained patient relation is optimized with a feasible
parameterization: each source row `[A_i,* , d_i]` lies on a simplex, and `e` is
bounded in `[0,1]`.

Full STRIDE v1 uses PyTorch as the canonical optimization framework. The outer
full-objective optimization uses AdamW with `weight_decay = 0.0`. AdamW is the
numerical optimizer and is not a biological regularizer. Biological
regularization comes only from explicit objective components: `L_open`,
`L_geometry`, `L_consistency`, and `L_recurrence`. A scheduler may be used only
if it is fixed, predeclared, and recorded in provenance; the canonical v1
recommendation is `ReduceLROnPlateau` on the total objective. The observation
term uses the torch-native differentiable canonical
`D_obs^BalancedSinkhornDivergence-v1` operator to solve the observation
discrepancy, while AdamW optimizes `A_p`, `d_p`, `e_p`, and any necessary
objective variables.

Open-channel regularization is a tendency-level L1 usage complexity cost on the
use of `d/e`. The v1 raw form is
`L_open_raw = mean(d_p) + mean(e_p)`. The open normalization scale is fixed at
`scale_open = 1`, so `normalized_L_open = L_open_raw`. The term does not
introduce state-specific targets, budget targets, or additional tunable
subweights. The fitted `d/e` values are determined by the joint objective over
ROI/FOV observation fit, geometry/locality, patient consistency, and cohort
recurrence.

Patient consistency is defined as support for one patient-level `A/d/e`
relation across that patient's multiple resolved observation evidence blocks.
For patient `p` and evidence block `b`:

`l_{p,b} = normalized L_obs for patient p, evidence block b evaluated under shared A_p and e_p`

`L_consistency_raw(p) = mean_b (l_{p,b} - mean_b l_{p,b})^2`

Evidence blocks come from the resolved source/target comparison plan. Domain
does not become a consistency axis. If `n_blocks < 2`,
`L_consistency_raw(p) = 0` and
`consistency_status = "insufficient_blocks"`. `L_consistency` penalizes
block-level support dispersion and does not replace `L_obs`; if all blocks fit
poorly, the overall `L_obs` remains high.

Geometry/locality is a soft biological-complexity cost over the canonical raw
`A_p` operator on shared-state geometry. It raises the explanation cost of
complex or distant remodeling, but it is not a hard prohibition: distant
remodeling can remain in the fitted relation when supported by the joint
objective. The term acts on raw `A_p` and not on a derived conditional kernel
such as `A_p / (1 - d_p)`. The full `A_p` matrix is the contract object for
this prior.

The shared-state geometry cost is:

- `C_norm = C_raw / s_C`,
- `C_raw` and `C_norm` must be finite, nonnegative, symmetric `[K, K]`
  state-geometry matrices on the shared `K`-state basis,
- diagonal entries must be `0` within declared numerical tolerance, so
  diagonal self-retention pays no geometry cost,
- `s_C` defaults to the median of positive finite off-diagonal entries of
  `C_raw`,
- at least one positive finite off-diagonal entry must exist; otherwise the
  fit fails the contract explicitly,
- no fallback `s_C = 1.0` and no automatic geometry disablement is allowed for
  an invalid state-geometry cost matrix,
- `C` represents symmetric state-identity distance, while `A_p` represents a
  directed remodeling relation, so `A_p[i,j]` need not equal `A_p[j,i]`.

For a valid fitted patient:

`L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]`

All `K` source rows are included in this row mean, including rows with no
source-observed activity. At the cohort/objective level,
`L_geometry_raw` is the simple mean of `L_geometry_raw(p)` over valid fitted
patients. `normalized_L_geometry` uses the same deterministic
identity-plus-small-open baseline-scale policy and epsilon-floor policy as the
other normalized components. No geometry-specific objective weight is added;
`L_geometry` is a fixed normalized component of `L_local`.

Geometry information enters the objective, compact provenance, and the
geometry ablation. The default scientific result surface remains the fitted
`A_p`, `d_p`, and `e_p` objects and derived summaries from those objects; no
separate biological result category is introduced for cost-stratified
transitions.

Cohort recurrence feeds back into estimation as a single cohort consensus
relation in v1. Let:

`R_p = (A_p, d_p, e_p)`

`R_bar = (A_bar, d_bar, e_bar)`

`dist(R_p, R_bar) = mean((A_p - A_bar)^2) + mean((d_p - d_bar)^2) + mean((e_p - e_bar)^2)`

`L_recurrence_raw = mean_p dist(R_p, R_bar)`

`R_bar` is the cohort-level consensus relation. Recurrence v1 encourages
patient-level `A/d/e` not to deviate without constraint from cohort-supported
consensus structure. It outputs cohort consensus `A/d/e`, patient support
count, dispersion around consensus, and recurrence fit status. It does not
claim automatic discovery of multiple biological remodeling families. Multiple
remodeling-family recurrence is a future extension or exploratory downstream
surface, not the current full-estimator v1 core objective.

## 6. Conceptual Loss and Regularization Roles

The full design requires the following conceptual loss/regularization families:

- observation discrepancy or comparison terms at the FOV/ROI layer,
- patient-consistency terms over resolved multi-block evidence for one patient
  relation,
- open-channel sparsity/complexity regularization over fitted `d_p` and `e_p`,
- geometry/locality prior over fitted `A_p`,
- cohort consensus recurrence/common-structure terms,
- result audit/status surfaces and compact successful-fit provenance.

This freeze defines objective-group roles and normalization policy. Later
implementation passes may refine exact raw formulas while preserving these
roles and the cohort/common-structure layer in the method definition.

## 7. Expected Full-STRIDE Outputs

Full STRIDE is expected to produce:

- patient-level `A_p`, `d_p`, `e_p`, and `T_p = [A_p | d_p]`,
- patient-level burden/composition auxiliaries,
- patient-level audits, fit-status fields, and uncertainty summaries,
- cohort-level recurrence/common-structure outputs such as consensus `A/d/e`,
  patient support count, dispersion around consensus, and embeddings,
- compact successful-fit provenance for `fit_stride(...)` with schema version
  `stride_fit_provenance.v1`,
- downstream task-ready summaries derived from the patient and cohort objects.

Task summaries may focus on selected views of these outputs, but they do not
replace the underlying patient and cohort objects.

The compact provenance schema is the successful full-estimator fit record. It
is not a separate audit module and does not duplicate result-container status
fields. Invalid parameters, invalid inputs, empty observation bags, invalid
simplexes, invalid state-geometry costs, unavailable canonical dependencies,
and other low-level contract violations fail fast under canonical
`ContractError` semantics rather than being represented as null provenance
fields.

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

Ablation/refit fits may add experiment-only provenance fields:

```yaml
ablation_mode: "recurrence" | "geometry" | "consistency"
ablation_term_handling: "remove" | "zero_weight"
ablation_denominator_policy: "fixed_denominator_no_reweighting"
```

Compatibility payloads may temporarily record `ablation_mode: "none"` for a
reference fit, but that is a migration/provenance label rather than an ordinary
user-facing `fit_stride(...)` control.

The only optional provenance diagnostics are `alpha_sensitivity`,
`legacy_observation_diagnostics`, and `optimizer_trace_ref`. These optional
fields do not change objective semantics, biological claims, or the canonical
meaning of `L_obs`. Detailed optimizer traces are off by default and may be
emitted only as optional diagnostics. Compact provenance does not require
`fit_status`, `patient_status`, `recurrence_status`,
`evidence_block_status`, status counts, `failure_reason`,
`optimizer_failure_reason`, per-patient records, or per-evidence-block records.
Existing result containers may continue to carry their own status fields.

## 8. Current Live Implementation Status

The live repo now contains the bounded first-pass implementation of the full
objective contract described above for supported `fit_stride(...)` inputs. This
status is an implementation claim about the current Python surface, not a
claim of global optimizer optimality or unrestricted input coverage.

The current implementation surface includes:

- a tissue-agnostic shared-state construction route,
- a domain-stratified bag-of-FOV observation layer with `mass_mode="uniform"`,
- a PyTorch/AdamW constrained full-estimator fit of patient-level `A_p`, `d_p`,
  and `e_p` for supported patient bundles,
- the canonical `D_obs^BalancedSinkhornDivergence-v1` observation discrepancy
  operator and fixed component-normalization policy,
- compact successful-fit provenance for successful full-estimator fits,
- recurrence, geometry, and consistency objective-ablation refit modes for
  internal validation surfaces,
- bootstrap uncertainty over fitted patient relation outputs,
- explicit non-`ok` status surfaces for unsupported inputs or numerical
  non-completion.

The current full-estimator path supports a bounded first-pass configuration:

- exactly two ordered groups per patient,
- uniform-mass patient inputs,
- valid shared-state geometry with finite `C_raw`, positive `s_C`, and derived
  `C_norm`,
- deterministic initialization and explicit optimizer status.

Inputs outside that support are not silently promoted to successful full
estimator fits. They either receive an explicit non-`ok` full-estimator status
or, for non-ablation compatibility routes, remain on the local-initializer
fallback surface and must not be used as evidence that the full objective path
was successfully fit.

Task A currently operationalizes this supported full-estimator path as a
bounded validation surface. Historical proxy/result-packet outputs remain
preserved as historical evidence and are not retroactively relabeled as final
full-STRIDE evidence.

## 9. Non-Claim Boundary

Full STRIDE as frozen here still does not imply:

- lineage tracing,
- exact physical transport truth,
- exact one-to-one FOV matching,
- unbiased whole-lesion reconstruction,
- direct proof that modeled open-channel tendencies are free of sampling
  effects.

## 10. Source-of-Truth Order

Use the following order for future work:

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, and `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A_rewiring_plan.md`
5. `docs/task_A_spec.md`, `docs/task_A_result.md`, and `tasks/task_A/README.md`
   for the current Task A task layer, including the frozen Block 3 v2
   truth-recovery benchmark contract, the canonical Block 0-2 results memo,
   and the canonical rerun runbook
6. `history/docs/` and `tasks/task_A/result_packets/` as historical/proxy
   reference only

## 11. Minimal Canonical Document Set After This Freeze

The minimal canonical document set after Step 1 is:

- `docs/stride_design_freeze.md` for full STRIDE itself,
- `docs/task_A_rewiring_plan.md` for how Task A must move onto full STRIDE,
- `docs/state.md` for current live implementation status,
- `docs/task_A_spec.md` for the current live Task A proxy specification,
- `docs/task_A_result.md` for the canonical Task A result layer through Block
  2, with explicit preserved proxy-history context,
- `tasks/task_A/README.md` for the current canonical Block 0-2 rerun
  operations.
