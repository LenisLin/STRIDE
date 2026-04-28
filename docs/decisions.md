# Decisions

This file records the stable method decisions for STRIDE. It is a method
decision register, not a task log and not a changelog. References to
`slotar.*` below denote current implementation locations during migration,
not the canonical future architecture.

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

## D011 - Legacy Compatibility Surfaces Remain Documented But Non-Canonical

- Context: Current code, task outputs, and historical docs still expose
  transport-centered surfaces.
- Decision:
  - keep explicit migration notes for legacy transport-era fields and solver
    APIs,
  - do not describe those surfaces as the target public method contract,
  - do not force a one-to-one mapping where the relation to
    `(A_p, d_p, e_p)` is only partial.
- Consequences:
  - `docs/api_specs.md`, `docs/data_contracts.md`, and `docs/state.md` must
    distinguish target design from compatibility surface,
  - later code and task rewrites can be evaluated against a stable contract
    rather than against moving historical artifacts.

## D012 - STRIDE Target Architecture Versus Transitional And Retired Namespaces

- Context: The repository now contains a live `stride` core package together with
  still-live `slotar` implementation surfaces, backend implementations, and
  legacy compatibility namespaces. The docs also need one rule for retired
  proxy and bridge-estimator names that must not remain live method surfaces.
- Decision:
  - STRIDE-level method contracts are canonical at the design level,
  - `src/stride/` is the task-insensitive core architecture and live
    first-pass implementation surface,
  - `src/slotar/` is the current transitional implementation and compatibility
    namespace,
  - `slotar.backends.ot_sinkhorn` is a backend implementation namespace, not a
    primary public method namespace,
  - `slotar.representation`, `slotar.uot`, `slotar.contracts`, `slotar.uq`,
    `slotar.utils`, `slotar.io.bridge`, `slotar.drift`, and `slotar.legacy.*`
    are compatibility or legacy surfaces,
  - `history/docs/` is archival only, historical code is archived outside the
    repo working tree, and `src/history/` is not a live package surface,
  - `fit_stride(...)` is the formal full STRIDE estimator entrypoint for
    manuscript-level use,
  - `fit_stride_proxy(...)` is retired from the live public API contract; if a
    compatibility implementation exists, it is historical/proxy context only,
  - `bridge_observation_matches(...)` and bridge-estimator style public
    surfaces are retired from the live public API contract; observation
    matching remains an internal loss/diagnostic/backend comparison role,
  - a namespace may be canonical even when its final estimator is still
    deferred; namespace stability and estimator completeness must be documented
    separately.
- Consequences:
  - docs must name `stride` as the architecture direction and name
    `slotar`/backend/legacy surfaces explicitly as implementation or
    compatibility layers,
  - `fit_stride(...)` is the only live public full-estimator method surface,
  - proxy or bridge-estimator names must not be described as active full-STRIDE
    estimator surfaces,
  - current `slotar.*` entrypoints may still be referenced when the docs need
    to name implementation locations or migration behavior,
  - docs must not present archived/history paths as active installable
    surfaces,
  - backend or compatibility functions such as `batched_uot_solve(...)`,
    `precompute_logKernels(...)`, and `build_community_features(...)` must not
    be advertised as the primary biological interface.

## D013 - Full STRIDE Uses Objective-Driven Fitted `A/d/e`

- Context: Proxy initializers and post-hoc objective reporting can emit arrays
  that resemble the canonical patient relation. That resemblance is not enough
  to define the full estimator.
- Decision:
  - `fit_stride(...)` is the formal manuscript-level full STRIDE estimator
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
    manuscript-level full estimator.

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

## D015 - Full Estimator Objective Grouping And `alpha` Semantics

- Context: The full estimator now has a frozen objective grouping, default
  `alpha`, normalization policy, and primary hyperparameter semantics.
  Remaining raw formulas remain tracked as implementation targets where not
  explicitly frozen by later decisions.
- Decision:
  - the full STRIDE objective is grouped as:
    `L_total = (1 - alpha) * L_local + alpha * L_regularization`,
  - default `alpha = 0.5`,
  - `alpha` is the primary local-versus-regularization hyperparameter,
  - `alpha` sensitivity grids are optional diagnostics,
  - `L_local` combines normalized observation data fit, open-channel
    sparsity/complexity regularization, and geometry/locality prior by fixed
    rule,
  - `L_regularization` combines normalized patient consistency and cohort
    recurrence by fixed rule,
  - component scales are computed from the deterministic
    identity-plus-small-open initialization,
  - near-zero scales use an epsilon floor and record floor usage in provenance,
  - group-internal component weighting is fixed by the contract rather than
    task-tuned,
  - STRIDE v1 uses simple continuous differentiable component losses where
    feasible, but the assembled full objective is treated as a constrained
    non-convex numerical objective rather than as a globally convex program,
  - the live contract does not claim a global optimum,
  - the deterministic identity-plus-small-open initialization is fixed as:
    `delta_init = min(0.05, 1 / (K + 1))`,
    `A_init = (1 - delta_init) * I_K`,
    `d_init = delta_init * 1_K`, and
    `e_init = (delta_init / K) * 1_K`,
  - that initialization is a feasible numerical starting point and the
    baseline normalization-scale starting point, not biological evidence for
    final `A/d/e`,
  - for patient consistency,
    `l_{p,b}` is the normalized `L_obs` for patient `p` and resolved evidence
    block `b` evaluated under shared `A_p` and `e_p`,
    `L_consistency_raw(p) = mean_b (l_{p,b} - mean_b l_{p,b})^2`,
  - if `n_blocks < 2`, `L_consistency_raw(p) = 0` and
    `consistency_status = "insufficient_blocks"`,
  - `L_consistency` penalizes block-level support dispersion and does not
    replace the observation data-fit term.
- Consequences:
  - objective reports must identify the local and regularization groups,
    normalized component losses, `alpha`, and any sensitivity grid that was
    run,
  - later optimizer and raw-formula choices must preserve the frozen grouping,
    normalization policy, and fixed group-internal weighting unless a later
    decision explicitly supersedes this one,
  - local optimum risk is managed through deterministic initialization, fixed
    optimizer protocol, optimizer status/provenance, and optional stability
    diagnostics rather than through a global-optimum claim.

## D016 - Open-Channel Variables Are Required Fitted Components

- Context: The full estimator treats open-channel behavior as fitted model
  structure tied to observations and regularization.
- Decision:
  - `d_p` and `e_p` are required bounded open-tendency fitted components in the
    full objective,
  - `d_p` closes source-side outgoing accounting,
  - `e_p` contributes to target-side incoming reconstruction,
  - `L_open_raw = mean(d_p) + mean(e_p)`,
  - `scale_open = 1`; `normalized_L_open = L_open_raw`,
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

## D017 - Cohort Recurrence Uses Single Consensus Feedback In V1

- Context: The full-estimator v1 recurrence claim requires recurrence terms to
  participate in estimation without claiming automatic discovery of multiple
  biological remodeling families.
- Decision:
  - cohort recurrence feeds back into estimation through the
    regularization/objective layer,
  - v1 uses a single cohort consensus relation:
    `R_p = (A_p, d_p, e_p)` and `R_bar = (A_bar, d_bar, e_bar)`,
  - `dist(R_p, R_bar) = mean((A_p - A_bar)^2) + mean((d_p - d_bar)^2) + mean((e_p - e_bar)^2)`,
  - `L_recurrence_raw = mean_p dist(R_p, R_bar)`,
  - `R_bar` is the cohort-level consensus relation,
  - recurrence v1 encourages patient-level `A/d/e` not to deviate without
    constraint from cohort-supported consensus structure,
  - recurrence v1 reports cohort consensus `A/d/e`, patient support count,
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
    configurations with the corresponding term disabled or zero-weighted.
- Consequences:
  - ablation results must identify the fitted objective configuration,
  - provenance must record `ablation_mode` for the full reference fit and each
    ablated fit.

## D019 - Compact Provenance Is Required For Manuscript-Level Results

- Context: Manuscript-level STRIDE outputs need enough provenance to audit the
  fit without requiring every detailed optimizer trace by default.
- Decision:
  - default STRIDE package output must include the biological result plus
    compact provenance,
  - detailed optimization traces are optional,
  - compact provenance must cover at least default `alpha` and any `alpha`
    sensitivity grid, loss decomposition and normalization, normalization
    baseline scales, epsilon-floor flags, initialization policy,
    `e_bound = [0, 1]`,
    `post_reconstruction_form = normalize(q_minus @ A + e)`,
    `observation_comparison_plan`,
    `observation_discrepancy_backend`,
    observation discrepancy operator version,
    `C_norm = C_raw / s_C` cost normalization,
    Sinkhorn/UOT backend version and status handling,
    observation-layer diagnostics such as `D_pos/B_pos` when emitted,
    `open_channel_complexity_form = mean(d)+mean(e)`,
    `open_channel_normalization_scale = 1`, optional augmented display object
    status if emitted, geometry cost normalization, optimizer framework,
    optimizer protocol, optimizer status, scheduler policy/status when used,
    recurrence consensus support/dispersion/status, ablation mode, random seed,
    and optimizer convergence or failure reason.
- Consequences:
  - biological result tables alone are incomplete manuscript-level outputs,
  - provenance schema design is required before final full-estimator promotion.

## D020 - Source/Target Declaration And Observation-Fit Evidence Boundary

- Context: Full STRIDE needs task-specific source/target observation
  comparisons without allowing tasks to redefine the core estimator or convert
  observation residuals into biological claims.
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
    backend,
  - `L_obs` compares predicted and observed domain-stratified bag-of-FOV
    empirical measures,
  - the source-side FOV form is
    `predicted_v_target = normalize(v_source @ A_p + e_p)`,
  - `L_obs_pair_raw = D_obs^UOT-v1(predicted target-side bag-of-FOV, observed target-side bag-of-FOV; C_norm)`,
  - `C_norm = C_raw / s_C`,
  - `D_obs^UOT-v1` is the fixed, versioned, auditable canonical observation
    discrepancy operator for v1,
  - observation diagnostics such as `D_pos/B_pos` remain observation-layer
    diagnostics rather than biological `d/e` or independently weighted loss
    components,
  - biological interpretation requires the evidence chain from observation
    discrepancy to patient-level fitted `A/d/e` to cohort consensus
    recurrence/dispersion support.
- Consequences:
  - task layers may choose the comparison plan, but they do not define a
    task-local STRIDE estimator or replace the observation solver for
    `stride_reference`,
  - provenance must record the observation comparison plan, observation
    discrepancy backend, operator version, cost normalization, Sinkhorn/UOT
    status handling, and emitted observation diagnostics,
  - manuscript-level claims must distinguish observation residual diagnostics
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
  - all biological regularization comes only from explicit objective
    components: `L_open`, `L_geometry`, `L_consistency`, and `L_recurrence`,
  - a scheduler may be used only if fixed, predeclared, and recorded in
    provenance; the canonical v1 recommendation is `ReduceLROnPlateau` on the
    total objective,
  - the observation term uses a torch-native differentiable canonical
    Sinkhorn/UOT-v1 operator,
  - Sinkhorn/UOT solves the observation discrepancy; AdamW optimizes `A_p`,
    `d_p`, `e_p`, and any necessary objective variables.
- Consequences:
  - optimizer provenance must distinguish numerical optimization settings from
    objective regularization,
  - status/failure handling must report optimizer convergence or failure
    without implying global optimality.
