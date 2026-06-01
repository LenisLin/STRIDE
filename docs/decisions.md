# Decisions

This file records the stable method decisions for STRIDE. It is a method
decision register, not a task log and not a changelog.

## D001 - STRIDE Is Remodeling-First And Patient-Level

- Context: Earlier repo framing over-centered transport machinery and
  solver-layer outputs, which made it too easy to treat transport as the
  biological estimand.
- Decision: The live scientific object is the patient-level pair
  `(T_p, e_p)` with `T_p = [A_p | d_p]`.
  - `A_p in R_+^{K x K}` is the patient-level continuity/remodeling operator on
    the shared `K`-state basis.
  - `d_p in [0,1]^K` is source-side outgoing open tendency.
  - `e_p in [0,1]^K` is target-side incoming open-entry tendency.
- Consequences:
  - STRIDE is documented as remodeling-first rather than transport-first.
  - observation-layer quantities are secondary evidence objects, not the
    primary biological object,
  - uncertainty, recurrence, and validation attach to patient-level relations
    rather than only to solver-level scalars.

## D002 - `A_p` Is Row-Substochastic, Not A Pure Conditional Kernel

- Context: Previous wording left room to read `A_p` as if it were already a
  normalized conditional kernel.
- Decision:
  - `A_p` is the canonical open continuity/remodeling operator,
  - it is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`,
  - if exposition needs a normalized conditional kernel, define the derived
    auxiliary object `R_{p,ij} = A_{p,ij} / (1 - d_{p,i})` only when
    `1 - d_{p,i} > 0`,
  - `A_p`, not `R_p`, remains the canonical STRIDE object.
- Consequences:
  - the diagonal of `A_p` is retention-like structure,
  - the off-diagonal of `A_p` is remodeling-like structure,
  - docs must not describe `A_p` as if every row already summed to one
    independent of source-side outgoing open tendency.

## D003 - Burden And Composition Must Remain Separate Scales

- Context: Prior wording blurred patient burden, normalized composition, and
  open-channel summaries into one scale.
- Decision:
  - `mu_p^-` and `mu_p^+` are patient-level pseudo-mass / burden vectors on the
    shared `K`-state basis,
  - normalized compositions are derived only:
    `q_p^- = mu_p^- / ||mu_p^-||_1` and
    `q_p^+ = mu_p^+ / ||mu_p^+||_1`,
  - any `m_p^(d)` or `m_p^(e)` summaries live on the same pseudo-mass / burden
    scale as `mu_p^-` and `mu_p^+`,
  - conservation is a soft burden-consistency anchor rather than literal
    physical conservation.
- Consequences:
  - composition-level structure may remain interpretable even when burden-level
    comparability is weak,
  - burden-level claims must be weakened or disabled when coverage or platform
    comparability is poor,
  - docs must not call `mu_p^-` and `mu_p^+` canonical compositions.

## D004 - The Canonical Observation Layer Is Domain-Stratified Bag-of-FOV Comparison

- Context: The observation layer was previously left abstract enough to invite
  transport-ontology or generative-observation interpretations.
- Decision:
  - each observed FOV/ROI is represented in the current first pass as a
    normalized community-composition vector `v_f in R_+^K` on the shared
    `K`-state basis,
  - `v_f[k]` is formed by counting the cells in ROI/FOV `f` assigned to shared
    community/state `k` and dividing by the total cell count in that ROI/FOV,
  - the current first-pass observation mass contract is degenerate/constant,
    with `mass_f = 1` and `mass_mode = "uniform"` for every ROI/FOV within a
    study,
  - the canonical observation object is a domain-stratified bag-of-FOV
    empirical measure in community-composition space with equal ROI/FOV mass:
    `nu_obs = sum_f w_f delta_{c(v_f)}`,
  - in the current first pass, `c(v_f) = v_f` and `w_f = 1`,
  - canonical fitting at this layer is discrepancy or measure comparison on
    those empirical measures, not a multinomial, Dirichlet-multinomial, or
    logistic-normal generative model.
- Consequences:
  - OT / Sinkhorn is documented as an observation-layer comparison tool,
  - ROI/FOV-level community composition remains distinct from the cell-level
    neighborhood subtype composition used during shared community-state
    construction,
  - domain-stratified bag-of-FOV comparison is preferred over collapsing to one
    raw histogram,
  - dense plans remain implementation details unless later decisions promote
    them explicitly.

## D005 - Domain Is Not Part Of Canonical State Identity

- Context: Domain labels are scientifically important, but folding them into the
  state basis and then conditioning on domain again causes double counting.
- Decision:
  - the shared `K`-state basis is built before any tissue/domain
    stratification, with domain excluded from canonical state identity itself,
  - the current first-pass state-construction route is tissue-agnostic:
    per-cell subtype labels -> within-ROI kNN neighborhood subtype proportion
    vectors -> k-means clustering to define shared community states,
  - the neighborhood size `k` is user-configurable; the documented first-pass
    default is `20`,
  - tissue/domain labels are not part of this state definition,
  - design-level `domain_label` metadata, and concrete current-route fields
    such as `compartment`, act only as observation-layer covariates,
    stratification variables, grouped discrepancy organization,
    patient-relation input grouping, or analysis surfaces,
  - docs and analyses must not encode domain into the state basis and then
    condition on domain again,
  - domain labels do not define state geometry or the axes of `A_p`, `d_p`,
    and `e_p`.
- Consequences:
  - domain-aware tasks remain compatible with one global state basis,
  - task-local ordered tissue-domain surfaces, including Task A, do not redefine
    the canonical state space,
  - tissue labels such as `TC`, `IM`, and `PT` remain observation-layer
    metadata rather than state names.

## D006 - Bootstrap Uncertainty Over Fitted Patient Relation Outputs

- Context: The live first-pass implementation already exposes bootstrap
  uncertainty summaries over fitted patient relation outputs, while the older
  Hurdle + Measurement Error framing is no longer the current core-package
  behavior.
- Decision:
  - current uncertainty means bootstrap/sampling-variance uncertainty over
    fitted patient relation outputs (`A_p`, `d_p`, `e_p`),
  - uncertainty remains attached to fitted patient-level outputs rather than
    defining a separate canonical estimand,
  - deferred or failed relation fits stay explicit rather than being coerced
    into apparently realized uncertainty summaries.
- Constraints:
  - replicate statuses remain explicit (`ok`, `deferred`, `failed`),
  - bootstrap summaries describe fitted-output variability, not all forms of
    bias, model misspecification, or selection uncertainty,
  - later uncertainty refactors must supersede this boundary explicitly rather
    than implicitly reviving older framing.
- Consequences:
  - canonical docs should describe the current first-pass uncertainty surface as
    bootstrap uncertainty,
  - older Hurdle + Measurement Error language is not authoritative for the live
    core-package pass.

## D007 - Explicit Failure Semantics And No Silent Fallback

- Context: Solver failures, empty supports, and structural sparsity must not be
  silently coerced into apparently valid biological outputs.
- Decision:
  - programmer-level contract violations fail fast,
  - per-item observation-layer degeneracies remain explicit in status fields,
  - patient-level or cohort-level outputs may be omitted only with an explicit
    audit trail.
- Consequences:
  - explicit fit-status fields remain a compatibility surface for current
    implementation-era observation fits,
  - future patient-level objects must continue to expose missingness and failure
    modes explicitly rather than fabricate fallback values.

## D008 - Observation-To-Relation Inference Remains In The Main Model And Must Keep Target-Side Open Entry Explicit

- Context: Longitudinal spatial data are observed through partial FOVs/ROIs with
  uneven coverage, heterogeneous composition, and no reliable one-to-one
  physical matching across time.
- Decision:
  - FOV/ROI-level observation fitting remains part of the main model,
  - STRIDE keeps an explicit observation-to-relation inference layer rather than
    collapsing directly to a pooled patient vector,
  - observation matching may contribute loss terms, diagnostics, or backend
    comparisons, but it does not define a public bridge estimator and does not
    generate canonical `A_p`, `d_p`, or `e_p` by itself,
  - target-side incoming open-entry tendency is represented by `e_p` and kept
    explicit in the fitted patient relation.
- Consequences:
  - FOVs/ROIs remain observation-layer units, not disposable preprocessing
    details,
  - coverage heterogeneity and within-patient spatial variation remain visible
    to the model and to the audit surface,
  - target-side open-entry preservation remains a required validation target.

## D009 - Cohort Recurrence Acts On Patient-Level Relations

- Context: Cohort-level claims become misleading if they are made directly from
  pooled FOV evidence rather than from patient-level remodeling objects.
- Decision: Cohort recurrence is defined over patient-level relations
  `(A_p, d_p, e_p)`. The cohort layer asks whether similar patient-level
  relations recur across patients, subgroups, or outcome strata.
- Consequences:
  - pooled FOVs are not the primary recurrence unit,
  - recurrence outputs must report patient support explicitly,
  - validation must include recurrence recovery and recurrence ablation tests.

## D010 - Task A Is Bounded Proxy Validation And Baselines Remain Mandatory

- Context: The method should remain interpretable without turning Task A into a
  claim of full longitudinal proof.
- Decision:
  - Task A's existing single-timepoint ordered tissue-domain proxy remains
    historical/proxy validation context,
  - future full-STRIDE Task A validation must use the formal full estimator
    contract rather than relabeling proxy outputs as manuscript-level STRIDE,
  - Task A does not redefine the global STRIDE object,
  - simple abundance and other non-relational baselines remain mandatory when
    STRIDE is used to claim added value,
  - STRIDE must not be documented as lineage tracing, exact physical transport,
    exact one-to-one FOV matching, or guaranteed unbiased whole-lesion
    dynamics.
- Consequences:
  - baselines explain what changes are already visible without the remodeling
    relation,
  - Task A language stays proxy-scoped and bounded,
  - method validation and task reporting are judged by contract behavior and
    falsifiable comparisons rather than attractive biological stories alone.

## D011 - Retired Compatibility Surfaces Remain Documented But Non-Canonical

- Context: Current code, task outputs, and historical docs still expose
  transport-centered surfaces.
- Decision:
  - keep explicit migration notes for retired transport-era fields and solver
    APIs,
  - do not describe those surfaces as the target public method contract,
  - do not force a one-to-one mapping where the relation to
    `(A_p, d_p, e_p)` is only partial.
- Consequences:
  - `docs/api_specs.md`, `docs/data_contracts.md`, and `docs/state.md` must
    distinguish target design from compatibility surface,
  - later code and task rewrites can be evaluated against a stable contract
    rather than against moving historical artifacts.

## D012 - STRIDE Package Identity, Public API, And Deferred Namespaces

- Context: The live repository now uses `stride` as the package and distribution
  identity. The codebase also contains implementation namespaces for losses,
  optimization, workflow orchestration, audit, and outputs. These internal
  namespaces must not be mistaken for a broad stable public API.
- Decision:
  - STRIDE-level method contracts are canonical at the design level,
  - the Python distribution name is `stride`,
  - `src/stride/` is the task-insensitive core architecture and live
    first-pass implementation surface,
  - the root public API remains deliberately small and beta:
    `fit_stride`, `build_patient_relation`, `summarize_fit`, `BasisSpec`,
    `DatasetHandle`, `ContractError`, and `__version__`,
  - `fit_stride(data, *, source, target, K, ...)` keeps an explicit
    keyword-only task surface rather than becoming a config-first public API,
  - `build_patient_relation(...)` remains keyword-only,
  - `losses`, `optimize`, `audit`, `workflows`, and private `_*.py` modules are
    implementation surfaces with no public stability commitment,
  - `fit_stride(...)` is the current implemented full-estimator entrypoint,
    with beta runtime/input-support status,
  - proxy or bridge-estimator style public entrypoints are not part of the live
    public API contract,
  - the target user-facing package architecture is recorded in
    `docs/package_api_design.md`; current implementation remains the beta root
    API until those namespaces are implemented,
  - scientific contract versions and Python API stability are separate: the
    objective/provenance/operator identifiers may remain v1 while the Python
    API remains beta.
- Consequences:
  - docs must name `stride` as the live package identity,
  - `fit_stride(...)` is the only live public full-estimator method surface,
  - proxy or bridge-estimator names must not be described as active full-STRIDE
    estimator surfaces,
  - archived/history paths must not be presented as active installable surfaces,
  - backend or compatibility functions such as solver kernels, observation
    matching helpers, and feature-building utilities must not be advertised as
    the primary biological interface.

## D013 - Full STRIDE Uses Objective-Driven Fitted `A/d/e`

- Context: Proxy initializers and post-hoc objective reporting can emit arrays
  that resemble the canonical patient relation. That resemblance is not enough
  to define the full estimator.
- Decision:
  - `fit_stride(...)` is the current implemented full STRIDE estimator
    entrypoint,
  - patient-level `A_p`, `d_p`, and `e_p` are objective-driven fitted
    variables,
  - proxy initializer output plus post-hoc objective reporting is not full
    STRIDE,
  - each source row `[A_{p,i,*}, d_{p,i}]` is simplex constrained, with
    `sum_j A_{p,ij} + d_{p,i} = 1` as a hard contract,
  - `e_p` is bounded with `0 <= e_{p,j} <= 1`,
  - composition-scale post-side reconstruction uses
    `predicted_q_p^+ = normalize(q_p^- @ A_p + e_p)`.
- Consequences:
  - canonical `A/d/e` outputs must come from the full objective fit or from
    explicit assembly of already valid relation objects,
  - observation-matching post-processing cannot be used to define the
    manuscript-level full estimator,
  - compatibility paths that depend on proxy initialization, bridge matching,
    or recurrence shrinkage are not manuscript-level full STRIDE fits unless
    the emitted `A_p`, `d_p`, and `e_p` come from objective-driven fitting under
    the frozen full-estimator contract.

## D014 - Proxy And Public Bridge Estimator Surfaces Are Retired

- Context: Historical Task A and transitional implementations used proxy and
  bridge naming that can be mistaken for the full method surface.
- Decision:
  - public `fit_stride_proxy(...)` is removed from the live public API contract,
  - public `bridge_observation_matches(...)` and bridge-estimator surfaces are
    removed from the live public API contract,
  - proxy may appear only as historical, retired, or compatibility context,
  - observation matching may remain inside the full estimator as a loss,
    diagnostic, or backend comparison but does not emit canonical `A/d/e`.
- Consequences:
  - live docs must define public STRIDE through `fit_stride(...)`,
  - implementation code retaining retired names is implementation debt unless
    explicitly marked as compatibility-only and excluded from the live method
    contract.

## D015 - Full Estimator Three-Block Objective

- Context: The full estimator has a frozen three-block objective,
  normalization policy, effective-loss ledger, and separate scale/start
  initialization contracts.
- Decision:
  - `L_total = mean(L_fit, L_prior, L_cohort)`,
  - `L_fit = normalized_L_obs + rho_subbag * L_subbag_consistency`,
  - `rho_subbag = 1.0`,
  - `L_prior = mean(normalized_L_open, L_geometry_effective)`,
  - `L_open_raw = mean(d_p) + mean(e_p)`,
  - `normalized_L_open = L_open_raw`,
  - `L_geometry_effective = geometry_effective_weight * normalized_L_geometry`,
  - `geometry_effective_weight = 0.01`,
  - `L_cohort = L_recurrence_raw / s_cohort`,
  - `s_cohort = 1e-2`,
  - `epsilon_norm = 1e-2`,
  - objective scale initialization and optimizer start initialization are
    separate contract objects,
  - objective scale initialization is identity-plus-small-open,
  - optimizer start initialization is off-diagonal-seeded
    identity-plus-small-open,
  - STRIDE v1 uses simple continuous differentiable component losses where
    feasible, but the assembled full objective is treated as a constrained
    non-convex numerical objective rather than as a globally convex program,
  - the live contract does not claim a global optimum.
- Consequences:
  - objective reports identify fit, prior, and cohort blocks,
  - provenance records raw, scale, normalized, and effective component values
    where applicable,
  - reference-fit outputs record `rho_subbag`, `geometry_effective_weight`,
    `s_cohort`, `epsilon_norm`, scale initialization, and optimizer start
    initialization,
  - local optimum risk is managed through fixed initialization, fixed optimizer
    protocol recorded in successful-fit provenance, explicit result status
    where applicable, and optional stability diagnostics rather than through a
    global-optimum claim.

## D016 - Open-Channel Variables Are Required Fitted Components

- Context: The full estimator treats open-channel behavior as fitted model
  structure tied to observations and regularization.
- Decision:
  - `d_p` and `e_p` are required bounded open-tendency fitted components in the
    full objective,
  - `d_p` closes source-side outgoing accounting,
  - `e_p` contributes to target-side incoming reconstruction,
  - `L_open_raw = mean(d_p) + mean(e_p)`,
  - `normalized_L_open = L_open_raw`,
  - `L_open` enters the revised v1 prior block together with
    `L_geometry_effective`,
  - `L_open` is a fixed tendency-level L1 open-channel usage complexity cost,
  - `L_open` does not introduce state-specific targets, budget targets, or
    additional tunable subweights,
  - `d/e` values are determined by the joint objective,
  - manuscript language should describe `d/e` as open tendencies under model
    and observations.
- Consequences:
  - open-channel estimates must be reported as fitted quantities with their
    objective context,
  - provenance must record the open-channel complexity form used for the fit.

## D017 - Cohort Recurrence Uses Row-Simplex Relation Feedback In V1

- Context: The full-estimator v1 recurrence claim requires recurrence terms to
  participate in estimation without claiming automatic discovery of multiple
  biological remodeling families.
- Decision:
  - `T_p = [A_p | d_p]`,
  - `T_bar = mean_p T_p`,
  - `e_bar = mean_p e_p`,
  - `L_T = mean_p mean_i sum_j (T_p[i,j] - T_bar[i,j])^2`,
  - `L_e_rec = mean_p mean_j (e_p[j] - e_bar[j])^2`,
  - `L_recurrence_raw = L_T + L_e_rec`,
  - `L_cohort = L_recurrence_raw / s_cohort`,
  - `s_cohort = 1e-2`,
  - recurrence v1 reports cohort consensus `T/e`, patient support count,
    dispersion around consensus, and recurrence fit status.
- Consequences:
  - estimator implementations must connect recurrence feedback to the fitted
    patient-level `A_p`, `d_p`, and `e_p` relation,
  - multiple remodeling-family analysis may be added later as an exploratory
    downstream or extension surface, but it is not the current v1 core
    objective.

## D018 - Core Ablation Semantics Require Refitting

- Context: Core ablations test estimator behavior under defined objective
  configurations.
- Decision:
  - `ablation_mode` supports `none`, `recurrence`, `geometry`, and
    `consistency`,
  - each ablation mode fits `A_p`, `d_p`, and `e_p` under the corresponding
    objective configuration,
  - `none` is the full reference fit,
  - the `recurrence`, `geometry`, and `consistency` modes fit under objective
    configurations with the corresponding term removed or zero-weighted,
  - ablations must refit `A_p`, `d_p`, and `e_p` under the ablated objective;
    they must not be implemented by masking reference-fit outputs or by
    post-hoc rescoring only,
  - ablations retain the reference fit's initialization, optimizer protocol,
    resolved evidence blocks, and rerun-specific semi-synthetic realization;
    the only objective change is removal or zero-weighting of the corresponding
    recurrence, geometry, or consistency term,
  - ablations use the three-block reference objective with the ablated term set
    to zero:
    `geometry` uses `L_prior = mean(normalized_L_open, 0)`,
    `recurrence` uses `L_total = mean(L_fit, L_prior, 0)`, and
    `consistency` uses `L_fit = normalized_L_obs + 0`,
  - no-`d/e`, open-channel removal, closed, balanced, and transport-style
    comparisons remain baseline/comparator surfaces rather than 3C ablations.
- Consequences:
  - ablation results must identify the fitted objective configuration,
  - ablation/refit experiment provenance must record `ablation_mode`, the
    three-block ablation objective policy, and the remove versus zero-weight
    implementation route for each ablated fit,
  - ordinary successful reference-fit provenance is not required to expose
    ablation as a user-level `fit_stride(...)` control; compatibility payloads
    may temporarily record `ablation_mode: "none"` only as a migration label.

## D019 - Compact Provenance Is Required For Manuscript-Level Results

- Context: Manuscript-level STRIDE outputs need compact provenance for a
  successful full-estimator fit without creating a new audit module or
  requiring detailed optimizer traces by default.
- Decision:
  - default STRIDE package output must include the biological result plus
    compact successful-fit provenance,
  - compact provenance is a parameter, loss, and protocol record for a
    successful `fit_stride(...)` full-estimator fit,
  - compact provenance does not replace fail-fast input or configuration
    validation and does not duplicate status fields already owned by result
    containers,
  - invalid parameters, invalid inputs, empty observation bags, invalid
    simplexes, invalid cost matrices, unavailable canonical dependencies, and
    low-level contract violations fail fast under canonical `ContractError`
    semantics rather than being represented as null provenance fields,
  - the required compact successful-fit provenance schema is:

```yaml
provenance_schema_version: "stride_fit_provenance.v1"
objective_contract_version: "stride_full_estimator_three_block_v1"
random_seed: int | null
objective_constants:
  rho_subbag: 1.0
  geometry_effective_weight: 0.01
  s_cohort: 0.01
  epsilon_norm: 0.01
objective_scale_initialization:
  policy: "identity_plus_small_open"
  delta_init: float
  K: int
  dtype: "float64"
optimizer_start_initialization:
  policy: "offdiag_seeded_identity_plus_small_open"
  delta_init: float
  offdiag_init_mass: 0.01
  numerical_min_mass: 1.0e-12
  K: int
  dtype: "float64"
loss:
  total: float
  fit: float
  prior: float
  cohort: float
  components:
    obs: {raw: float, scale: float, normalized: float, floor_used: bool}
    open: {raw: float, normalized: float}
    geometry: {raw: float, scale: float, normalized: float, effective: float, floor_used: bool}
    subbag_consistency: {raw: float, effective: float, status: string}
    recurrence: {raw: float, cohort_scaled: float}
e_bounds: [0.0, 1.0]
post_reconstruction_form: "normalize(q_minus @ A + e)"
observation_comparison_plan:
  resolved_by: "task_layer"
  n_evidence_blocks: int
  domain_policy: "observation_layer_only"
  block_construction_policy: string
  n_blocks_by_patient: {patient_id: int}
observation_discrepancy:
  operator_version: "D_obs^BalancedSinkhornDivergence-v1"
  backend: "torch"
  dtype: "float64"
  inner_epsilon_schedule: [0.5, 0.2, 0.1]
  outer_epsilon_schedule: [0.5, 0.2, 0.1]
  max_iter: 100
  tol: 1e-6
  warning_tol: 1e-4
state_geometry:
  normalization: "C_norm = C_raw / s_C"
  s_C: float
optimizer:
  framework: "torch"
  algorithm: "AdamW"
  weight_decay: 0.0
  scheduler_policy: "none" | "CosineAnnealingLR"
recurrence:
  support_n_patients: int
  dispersion: float
detailed_optimizer_trace: bool
```

  - the public adapter uses `partitioned_fov_subbag_v1` for `FovObservation`
    inputs; each evidence block remains a source/target bag of FOV observations,
  - ablation/refit experiment fits may add `ablation_mode`,
    `ablation_term_handling`, and
    `ablation_objective_policy="three_block_reference_term_zeroing"` as
    experiment-only provenance fields; they are not required user-facing fields
    for an ordinary successful `fit_stride(...)` fit,
  - `D_obs^BalancedSinkhornDivergence-v1` is the canonical successful-fit
    observation-discrepancy operator version in provenance,
  - detailed optimizer traces are off by default and may be emitted only as
    optional diagnostics,
  - optional provenance diagnostics are limited to `objective_sensitivity`
    and `optimizer_trace_ref`,
  - optional provenance diagnostics must not change objective semantics,
    biological claims, or the canonical meaning of `L_obs`,
  - compact provenance does not require `fit_status`, `patient_status`,
    `recurrence_status`, `evidence_block_status`, status counts,
    `failure_reason`, `optimizer_failure_reason`, per-patient records, or
    per-evidence-block records.
- Consequences:
  - biological result tables alone are incomplete manuscript-level outputs,
  - compact provenance records the successful fit's parameter, loss, and
    protocol context without defining a separate status or failure hierarchy,
  - existing result containers may continue to carry their own status fields.

## D020 - Source/Target Declaration And Observation-Fit Evidence Boundary

- Context: Full STRIDE needs task-specific source/target observation
  comparisons without allowing tasks to redefine the core estimator or convert
  observation diagnostics into biological claims.
- Decision:
  - tasks declare and resolve ordered source/target observation groups and the
    valid domain strata or domain mapping used for observation comparison,
  - the task layer instantiates the comparison plan,
  - the reusable core receives resolved source/target observation evidence
    blocks; it does not read task YAML and does not infer biological
    source-target semantics,
  - once resolved evidence blocks enter the core, domain is not a loss axis,
    state axis, relation axis, or recurrence axis,
  - once resolved, every analysis uses the same task-insensitive
    `fit_stride(...)` full estimator and canonical v1 observation-discrepancy
    operator,
  - `D_obs` is the fixed operator inside `L_obs`, not a sixth loss term and
    not an independently weighted component,
  - `L_obs` compares the model-implied predicted target-side FOV bag with the
    observed target-side FOV bag,
  - the source-side FOV form is
    `y_hat_f = normalize(v_source_f @ A_p + e_p)`,
  - `L_obs_pair_raw = D_obs^BalancedSinkhornDivergence-v1(predicted target FOV bag, observed target FOV bag; C_norm)`,
  - `C_norm = C_raw / s_C` remains the state/community-level cost
    normalization,
  - canonical v1 `D_obs` is the fixed, versioned, auditable, torch-native,
    `float64`, log-domain, differentiable, balanced, debiased Sinkhorn
    divergence operator,
  - the canonical observation layer has no unbalanced unmatched residual; open
    behavior is expressed only by fitted biological `d/e`,
  - retired UOT, `D_pos/B_pos`, and unbalanced observation residual diagnostics
    remain diagnostic or Block 3 comparator concepts, including the external
    Task A `uot_baseline` comparator name, rather than canonical `L_obs`,
    biological `d/e`, or independently weighted loss components,
  - biological interpretation requires the evidence chain from observation
    discrepancy to patient-level fitted `A/d/e` to cohort consensus
    recurrence/dispersion support.
- Consequences:
  - task layers may choose the comparison plan, but they do not define a
    task-local STRIDE estimator or replace the observation solver for
    `stride_reference`,
  - compact successful-fit provenance records the task-resolved observation
    comparison plan and the canonical
    `D_obs^BalancedSinkhornDivergence-v1` operator settings required by the
    success schema,
  - retired observation diagnostics are not part of compact successful-fit
    provenance,
  - manuscript-level claims must distinguish retired observation diagnostics
    from fitted patient-level open-channel quantities and cohort-level
    recurrence evidence.

## D021 - Full Estimator V1 Uses PyTorch AdamW As Numerical Optimizer

- Context: The full objective needs a fixed optimizer protocol for auditability
  without treating optimizer mechanics as biological regularization.
- Decision:
  - full STRIDE v1 uses PyTorch as the canonical optimization framework,
  - outer full-objective optimization uses AdamW,
  - `weight_decay = 0.0`,
  - AdamW is a numerical optimizer, not a biological regularizer,
  - all biological regularization comes only from explicit objective blocks and
    components: `L_fit`, `L_prior`, `L_cohort`, `L_open`,
    `L_geometry_effective`, `L_subbag_consistency`, and `L_recurrence_raw`,
  - the canonical reference protocol is fixed as `20` warm-up AdamW steps at
    `lr = 0.02`, followed by a main AdamW stage at `lr = 0.05` with
    `CosineAnnealingLR`,
  - the cosine settings are fixed as `T_max = 200` and `eta_min = 0.0`,
  - main-stage early stopping becomes eligible only after `100` main steps,
    with main-stage hard cap `200` steps,
  - the observation term uses the torch-native differentiable canonical
    `D_obs^BalancedSinkhornDivergence-v1` operator,
  - the balanced Sinkhorn divergence operator solves the observation
    discrepancy; AdamW optimizes `A_p`, `d_p`, `e_p`, and any necessary
    objective variables.
- Consequences:
  - compact successful-fit provenance records the fixed two-phase optimizer
    protocol, cosine settings, and early-stop thresholds as numerical
    protocol fields,
  - finite optimizer exits that reach the main-stage hard cap remain
    successful fits and record an explicit optimizer exit flag rather than
    becoming optimizer-level deferred results,
  - optimizer protocol provenance must distinguish numerical optimization
    settings from objective regularization,
  - optimizer or dependency contract violations fail fast through the same
    exception/status surfaces as other contract failures rather than becoming
    required compact provenance fields,
  - failure handling must not imply global optimality.

## D022 - Canonical Observation Discrepancy Uses Debiased Balanced Sinkhorn Divergence

- Context: The full-estimator observation term needs a fixed discrepancy
  operator that compares predicted and observed target-side FOV bags without
  introducing an observation-layer unmatched residual or a separate loss
  weight.
- Decision:
  - the canonical v1 operator is
    `D_obs^BalancedSinkhornDivergence-v1`,
  - `D_obs` is a fixed operator inside `L_obs`; it is not a sixth loss term and
    has no independent weight,
  - `y_hat_f = normalize(v_source_f @ A_p + e_p)`,
  - `L_obs` compares the model-implied predicted target-side FOV bag with the
    observed target-side FOV bag,
  - the operator is torch-native, `float64`, log-domain, differentiable,
    balanced, and debiased,
  - observation-layer unbalanced unmatched residuals are not part of canonical
    `L_obs`; open behavior is expressed only by fitted `d/e`,
  - `C_norm = C_raw / s_C` remains the state/community-level cost,
  - the FOV-level ground cost is
    `G[f,g] = debiased balanced Sinkhorn composition distance(y_hat_f, y_true_g; C_norm)`,
  - tiny negative inner composition-distance values in `[-1e-10, 0)` are
    clamped to `0` with provenance; values below `-1e-10` are numerical
    failure,
  - the outer loss is
    `L_obs_pair_raw = debiased balanced Sinkhorn divergence(predicted target FOV bag, observed target FOV bag; G_norm)`,
  - `G_norm = G / s_G_init`,
  - `s_G_init` is a fixed evidence-block scale computed at deterministic
    identity-plus-small-open initialization from positive finite
    initialization-time FOV-level costs,
  - if no positive finite initialization-time FOV-level costs exist,
    `s_G_init = 1.0` and floor usage is recorded,
  - `s_G_init` is not recomputed dynamically during optimization,
  - fixed v1 numeric defaults are inner epsilon schedule `(0.5, 0.2, 0.1)`,
    outer epsilon schedule `(0.5, 0.2, 0.1)`, `max_iter = 100` per epsilon
    stage, `tol = 1e-6`, `warning_tol = 1e-4`, backend `torch`, and dtype
    `float64`,
  - empty source or target FOV bag, invalid simplex, negative entries, NaN/Inf,
    invalid cost matrix, or unavailable `torch` for the canonical full
    estimator fails explicitly,
  - no padding is allowed,
  - reaching `max_iter` with finite values is usable with warning; updates
    larger than `warning_tol` are recorded convergence warnings on
    diagnostic/profiling surfaces rather than every production optimizer
    forward,
  - UOT, `D_pos/B_pos`, and unbalanced observation residuals are retired,
    diagnostic, or Block 3 comparator concepts only,
  - Task A Block 3 preserves `uot_baseline` as an external comparator name and
    does not rename it to the canonical full-estimator operator.
- Consequences:
  - compact successful-fit provenance records
    `operator_version = "D_obs^BalancedSinkhornDivergence-v1"`, backend
    `torch`, dtype `float64`, inner and outer epsilon schedules, `max_iter`,
    `tol`, `warning_tol`, and `C_norm = C_raw / s_C` state-geometry
    normalization,
  - UOT remains a retired diagnostic or Block 3 comparator concept only and is
    not the canonical full-estimator observation operator,
  - tests and documents that describe canonical full-estimator `L_obs` must not
    treat retired UOT or retired residual diagnostics as canonical biological
    open-channel quantities.

## D023 - Geometry/Locality Prior Uses Raw A And Symmetric Shared-State Cost

- Context: The full estimator needs a fixed geometry/locality contract that
  raises the explanation cost of complex or distant remodeling without turning
  state geometry into a hard support mask or an additional reported biology
  class.
- Decision:
  - `L_geometry` is a soft biological-complexity cost on shared-state
    geometry,
  - distant remodeling may remain in the fitted relation when supported by the
    joint objective,
  - the term acts directly on canonical raw `A_p`,
  - it does not act on a derived conditional kernel such as
    `A_p / (1 - d_p)`,
  - the full `A_p` matrix is the objective contract surface for the term,
  - `C_norm = C_raw / s_C`,
  - `C_raw` and `C_norm` must be finite, nonnegative, symmetric `[K, K]`
    state-geometry matrices on the shared `K`-state basis,
  - diagonal entries must be `0` within declared numerical tolerance,
  - `s_C` defaults to the median of positive finite off-diagonal entries of
    `C_raw`,
  - at least one positive finite off-diagonal entry must exist; otherwise the
    fit fails the contract explicitly,
  - invalid state geometry must not fall back to `s_C = 1.0` and must not
    silently disable the geometry term,
  - `C` represents symmetric state-identity distance, while `A_p` is directed,
  - for a valid fitted patient:
    `L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]`,
  - all `K` source rows are included in the row mean,
  - diagonal self-retention pays no geometry cost because `C_norm[i,i] = 0`,
  - cohort/objective-level `L_geometry_raw` is the simple mean over valid
    fitted patients,
  - `scale_geometry = max(L_geometry_raw(theta_scale), epsilon_norm)`,
  - `normalized_L_geometry = L_geometry_raw(theta) / scale_geometry`,
  - `L_geometry_effective = geometry_effective_weight * normalized_L_geometry`,
  - `geometry_effective_weight = 0.01` in the v1 reference objective,
  - a zero or near-zero geometry baseline uses `epsilon_norm = 1e-2` and
    records `loss.components.geometry.floor_used = true` as provenance-only
    normal provenance when the raw geometry loss is valid,
  - the default scientific result surface remains fitted `A_p`, `d_p`, and
    `e_p` plus existing derived summaries,
  - geometry information is emitted through the objective, compact provenance,
    and geometry ablation rather than as a separate biological result class.
- Consequences:
  - official and custom representation routes must provide a valid
    shared-state geometry cost matrix before canonical fitting,
  - compact provenance records `state_geometry.s_C`,
    `loss.components.geometry.{raw, scale, normalized, effective, floor_used}`,
    `epsilon_norm`, `geometry_effective_weight`, and ablation mode,
  - geometry ablation removes or zeroes the geometry/locality term and refits
    `A_p`, `d_p`, and `e_p` under the ablated objective,
  - later implementation work must preserve the raw, normalized, and effective
    geometry ledger distinction.

## D024 - STRIDE User Package API Target

- Context: The live root API is intentionally small while the repository also
  needs a positive design target for the eventual user-facing package layer.
- Decision:
  - the package target adopts compact user namespaces: `io`, `pp`, `tl`, `pl`,
    and `ds`,
  - current root `fit_stride(...)` remains the implemented entry to the core
    estimator contract,
  - implementation namespaces such as `losses`, `optimize`, `audit`, and
    `workflows` remain internal development surfaces.
- Consequences:
  - user-facing namespace implementation follows
    `docs/package_api_design.md`,
  - API migration and package cleanup use the review workflow in
    `docs/package_api_design.md`,
  - stable public exposure is created through target namespace contracts,
    implementation review, and narrow tests.
- Current status:
  - `stride.io` v1 is implemented with `build_adata`, `read_h5ad`, and
    `write_h5ad`,
  - root estimator entrypoints remain the implemented fitting surface,
  - `pp`, `tl`, `pl`, and `ds` remain target namespaces pending reviewed
    implementation.
