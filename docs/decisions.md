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
  - `d_p in R_+^K` is depletion tendency on the pre side.
  - `e_p in R_+^K` is post-side emergence.
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
    independent of depletion.

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
    stratification variables, grouped discrepancy organization, bridge input
    grouping, or analysis surfaces,
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

## D006 - Bootstrap Uncertainty Over Realized Bridge Outputs

- Context: The live first-pass implementation already exposes bootstrap
  uncertainty summaries over realized patient bridge outputs, while the older
  Hurdle + Measurement Error framing is no longer the current core-package
  behavior.
- Decision:
  - current uncertainty means bootstrap/sampling-variance uncertainty over
    realized bridge outputs (`A_p`, `d_p`, `e_p`),
  - uncertainty remains attached to realized patient-level outputs rather than
    defining a separate canonical estimand,
  - deferred or failed bridge fits stay explicit rather than being coerced into
    apparently realized uncertainty summaries.
- Constraints:
  - replicate statuses remain explicit (`ok`, `deferred`, `failed`),
  - bootstrap summaries describe realized bridge variability, not all forms of
    bias or selection uncertainty,
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

## D008 - The FOV Bridge Remains In The Main Model And Must Keep Emergence Explicit

- Context: Longitudinal spatial data are observed through partial FOVs/ROIs with
  uneven coverage, heterogeneous composition, and no reliable one-to-one
  physical matching across time.
- Decision:
  - FOV-level fitting remains part of the main model,
  - STRIDE keeps an explicit observation-to-patient bridge rather than
    collapsing directly to a pooled patient vector,
  - post-side emergence is represented by `e_p`, not forced into `A_p`, not
    absorbed into depletion, and not reduced to a generic residual bucket.
- Consequences:
  - FOVs/ROIs remain observation-layer units, not disposable preprocessing
    details,
  - coverage heterogeneity and within-patient spatial variation remain visible
    to the model and to the audit surface,
  - emergence-preserving behavior remains a required validation target.

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
  - Task A remains a bounded validation task under a single-timepoint ordered
    tissue-domain proxy,
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

## D012 - STRIDE Target Architecture Versus Transitional Compatibility Namespaces

- Context: The repository now contains a live `stride` core package together with
  still-live `slotar` implementation surfaces, backend implementations, and
  legacy compatibility namespaces. The docs need one stable rule for how those
  layers are described.
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
  - a namespace may be canonical even when its final estimator is still
    deferred; namespace stability and estimator completeness must be documented
    separately.
- Consequences:
  - docs must name `stride` as the architecture direction and name
    `slotar`/backend/legacy surfaces explicitly as implementation or
    compatibility layers,
  - `fit_stride(...)` is the current narrow implemented observation-to-patient
    bridge path,
  - `bridge_observation_matches(...)` and `estimate_recurrence(...)` may be
    described as canonical reserved estimator surfaces, but not as fully
    implemented end-to-end estimators,
  - current `slotar.*` entrypoints may still be referenced when the docs need
    to name implementation locations or migration behavior,
  - docs must not present archived/history paths as active installable
    surfaces,
  - backend or compatibility functions such as `batched_uot_solve(...)`,
    `precompute_logKernels(...)`, and `build_community_features(...)` must not
    be advertised as the primary biological interface.
