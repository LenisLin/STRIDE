# API Specifications

This file defines the current first-pass method-facing specification for
STRIDE. The contract is architecture-first: `src/stride/` is the canonical
reusable-core direction, while `slotar.*` references below denote transitional
implementation surfaces during migration. The narrow first-pass `stride` API
already realizes the stable surfaces described here; broader estimator
completeness remains a separate question.

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

Important observation-layer boundary:

- canonical fitting at this layer is discrepancy or measure comparison over
  those empirical measures,
- OT / Sinkhorn belongs to this layer only,
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
| `d_p` | `[K]` | depletion tendency on the pre side |
| `e_p` | `[K]` | post-side emergence |
| `T_p` | conceptual block object | `A_p` plus depletion column `d_p`; storage may keep `A_p` and `d_p` separately |

Contract semantics:

- `A_p` is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`,
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
primary biological object. At minimum, a cohort recurrence interface must be
able to represent:

- which patients were included,
- which patient-level relations were compared,
- which recurrence families or summaries were found,
- how much patient support and dispersion each family carries.

## 2. Required Inputs

### 2.1 Required identifiers and ordering

Any valid analysis must provide:

- patient identifiers,
- ordered timepoint identifiers or an equivalent ordered relation,
- FOV/ROI identifiers,
- a shared `K`-state basis or an official route to derive one,
- declared `mass_mode`, which is `uniform` in the current first pass,
- design-level `domain_label` metadata when domain-stratified observation
  comparison is used; the current AnnData route uses `compartment`,
- area or equivalent support metadata only for non-uniform future/custom
  density semantics.

Equivalent ordered relations must be explicit rather than implicit. They may be
task-local, but they must still declare the ordering used for analysis.

The state/domain boundary is part of the API contract:

- the shared `K`-state basis excludes domain identity,
- domain labels may stratify observation comparison, grouped discrepancies, and
  bridge input grouping,
- callers must not encode domain in the basis and then condition on domain
  again,
- domain labels do not define state geometry or the axes of `A_p`, `d_p`, and
  `e_p`.

### 2.2 Required quantitative inputs

A valid method entry surface must provide one of the following:

1. official observation data sufficient to derive the shared `K`-state basis
   and the observation-layer vectors `v_{p,t,f}`, or
2. direct FOV/ROI state vectors `v_{p,t,f}` on an already-declared shared
   `K`-state basis, together with explicit observation-layer domain metadata and
   declared observation mass semantics.

### 2.3 Optional priors and side inputs

Optional method inputs may include:

- observation-layer cost geometry,
- grouping priors,
- drift vectors or drift-risk priors,
- uncertainty configuration,
- recurrence hyperparameters,
- bridge hyperparameters.

These are optional priors. They do not replace the required patient/time/FOV
and shared-basis contracts.

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
6. fit observation-layer comparisons on ordered, domain-stratified empirical
   measures,
7. apply the FOV bridge to produce patient-level `(A_p, d_p, e_p)`,
8. derive patient summaries from those objects,
9. estimate cohort recurrence from patient-level objects,
10. emit audits, uncertainty summaries, and failure states explicitly.

Important boundary:

- OT / Sinkhorn belongs to step 6 only,
- it informs the bridge but does not replace steps 7 through 9,
- direct pooling to a patient-level vector before the observation-layer
  comparison step is not the target design.

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
- depletion summaries from `d_p`,
- emergence summaries from `e_p`,
- burden-scale summaries derived from `mu_p^-`, `mu_p^+`, `m_p^(d)`, or
  `m_p^(e)`,
- derived composition summaries from `q_p^-` or `q_p^+`,
- uncertainty summaries,
- bridge diagnostics.

These derived summaries are secondary to the patient-level object itself.

### 4.3 Required cohort-level outputs

Any recurrence layer must be able to emit:

- patient membership/support,
- family or summary identifiers,
- family-level summaries on the same shared `K`-state basis,
- dispersion or stability diagnostics,
- recurrence-layer fit status.

## 5. Official Route Versus Custom Route

### 5.1 Official route

The official route is the task-insensitive route from spatial single-cell or
spot observations to:

- a shared `K`-state basis,
- valid observation-layer vectors `v_{p,t,f}`,
- the priors needed for observation-layer comparison,
- the audits needed for FOV-aware fitting and bootstrap uncertainty over
  realized bridge outputs.

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
- `stride.api.fit` and `stride.outputs.fit_result` for the current narrow
  observation-to-patient bridge path,
- `stride.outputs.uncertainty` for bootstrap uncertainty over realized bridge
  outputs.

`slotar.*` remains available as a transitional wrapper/compatibility layer
around parts of that live first-pass route.

### 5.2 Custom route

A custom route is allowed only if it ends in the same contracts:

- same shared `K`-state semantics,
- valid nonnegative observation-layer vectors,
- declared `mass_mode`,
- explicit patient/ordering/FOV indexing,
- domain handling that does not redefine state identity,
- a compatible path to patient-level `(A_p, d_p, e_p)`.

## 6. Target Architecture Areas Versus Current Implementation Surfaces

The canonical architecture direction is `src/stride/`. Much of the current
working implementation still lives under `slotar.*`. Treat the table below as a
migration map, not as a promise that every `stride.*` area is already a stable
public API.

| Design area | Current implementation surface | Target architecture area | Status |
|---|---|---|
| top-level export story | `slotar` | `stride` | transition in progress |
| shared basis construction | `slotar.state_space` | `stride.basis`, `stride.api.basis` | live first-pass implementation present |
| observation-layer comparison | `slotar.observation` | `stride.observation` | live first-pass implementation present |
| patient-level relation assembly | `stride.api.fit`, `slotar.bridge`, `slotar.patient` | `stride.latent`, `stride.api.fit`, `stride.outputs` | live first-pass implementation present; broader standalone estimator still deferred |
| cohort recurrence | `slotar.recurrence` | `stride.latent.recurrence`, `stride.outputs` | deferred estimator namespace present |
| validation and longitudinal input checking | `slotar.validation`, `slotar.io.longitudinal` | `stride.data`, `stride.types` | live first-pass implementation present |
| compatibility/adapters | `slotar.compat.*` and shim modules | `stride.adapters` | transition only |

The conservative stable first-pass public tier is:

- package root `stride`,
- `stride.api.dataset.DatasetHandle`,
- `stride.api.basis.BasisSpec`,
- `stride.api.fit.fit_stride(...)`,
- `stride.api.fit.build_patient_relation(...)`,
- result contracts in `stride.outputs.fit_result`,
- uncertainty contracts in `stride.outputs.uncertainty`.

The current implementation entrypoints that realize parts of this contract
include:

- `BasisSpec` in `stride.api.basis`,
- `DatasetHandle` in `stride.api.dataset`,
- `fit_stride(...)` and `build_patient_relation(...)` in `stride.api.fit`,
- `PatientBridgeResult` and `STRIDEFitResult` in `stride.outputs.fit_result`,
- `PatientBootstrapConfig` and `STRIDEBootstrapUncertaintyResult` in
  `stride.outputs.uncertainty`,
- `build_local_state_features(...)` and `learn_shared_state_axis(...)` in
  `stride.basis`,
- `match_observation_clouds(...)`, `build_observation_kernels(...)`,
  `calibrate_match_penalty(...)`, and `compute_active_state_support(...)` in
  `stride.observation`,
- `validate_longitudinal_adata(...)` in `stride.data.longitudinal`,
- `slotar.*` wrappers only where migration compatibility still requires them.

Important current boundary:

- the implementation locations above do not redefine the canonical
  architecture story,
- `fit_stride(...)` is the current implemented narrow observation-to-patient
  bridge path,
- `build_patient_relation(...)` is a canonical assembly surface for already
  constructed patient-level arrays,
- `bridge_observation_matches(...)` is a reserved canonical bridge estimator,
  not yet the implemented patient-level estimation algorithm,
- `estimate_recurrence(...)` in `stride.latent.recurrence` is the canonical
  recurrence estimator namespace, but it currently returns an explicit deferred
  result rather than a finalized family estimator.

## 7. Backend, Compatibility, And Legacy Surfaces

The following surfaces remain importable, but they are not the canonical public
method interface.

### 7.1 Backend-only surface

- `slotar.backends.ot_sinkhorn` contains the numerical OT / Sinkhorn backend
  used by the canonical observation layer and by compatibility shims.

Backend-only functions may remain important implementation surfaces, including:

- `batched_uot_solve(...)`,
- `build_observation_kernels(...)`,
- `calibrate_match_penalty(...)`,
- `compute_active_state_support(...)`.

These are backend or compatibility solver surfaces, not primary biological
interfaces.

### 7.2 Compatibility wrappers and shim paths

| Surface | Current role | Status |
|---|---|---|
| `slotar.compat.*` | migration-only compatibility implementations | compatibility |
| `slotar.representation` | top-level shim forwarding to `slotar.compat.representation` | compatibility shim |
| `slotar.uot` | top-level shim forwarding to `slotar.compat.uot` | compatibility shim |
| `slotar.contracts` | top-level shim forwarding to `slotar.compat.contracts` | compatibility shim |
| `slotar.uq` | top-level shim forwarding to `slotar.compat.uq` | compatibility shim |
| `slotar.utils` | top-level shim forwarding to `slotar.compat.utils` | compatibility shim |
| `slotar.io.bridge` | top-level shim forwarding to `slotar.compat.io.bridge` | compatibility shim |
| `slotar.drift` | top-level shim forwarding to `slotar.compat.drift` | compatibility shim |
| `slotar.exceptions` | top-level shim forwarding to `slotar.compat.exceptions` | compatibility shim |

The current compatibility entrypoints that may still appear in code or older
docs include:

- `build_community_features(...)`,
- `learn_global_prototypes(...)`,
- `batched_uot_solve(...)`,
- `precompute_logKernels(...)`,
- `calibrate_joint_lambda(...)`,
- `bootstrap_single_roi(...)`,
- `compute_active_mask(...)`,
- `flag_drift(...)`.

The most important compatibility rule is:

- these compatibility entrypoints are not re-exported from the target
  architecture story and should be treated as transitional implementation
  surfaces,
- `batched_uot_solve(...)` emits observation-layer details and diagnostics,
  not the final patient-level `(A_p, d_p, e_p)` contract.

## 8. Observation-Layer Detail Boundary

The current implementation container for the observation-layer result surface is
`slotar.observation.ObservationMatchResult`.

Important output boundary:

- it is an observation-layer container, not a patient-level biological object,
- it may expose observation-layer compatibility diagnostics internally during
  migration,
- when dense observation plans are returned, `matching_plan` is the preferred
  canonical label on the observation result surface,
- dense-plan aliases may still appear in backend or compatibility payloads
  during migration.

These details may inform the bridge, but they do not replace the patient-level
contract.

## 9. Legacy Output Boundary

The current code still emits observation-layer compatibility diagnostics,
regularization or control fields, and explicit fit-status metadata.

These remain useful implementation diagnostics and migration surfaces, but they
are not the target public method contract.

## 10. Explicit Non-Claim Boundary

These APIs define modeling interfaces and method objects. They do not, by
themselves, imply:

- lineage tracing,
- exact physical transport truth,
- exact one-to-one FOV matching,
- unbiased whole-lesion reconstruction.
