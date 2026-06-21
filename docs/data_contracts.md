# Data Contracts

This file defines the live data and artifact contracts for STRIDE. The
contracts are package-neutral at the design level and use canonical STRIDE
object semantics.

## 1. Scope and Notation

Notation used throughout the contract:

- `p`: patient
- `t`: ordered analysis side, typically a timepoint but allowed to be an
  equivalent declared ordered relation in a task-local setting
- `f`: FOV or ROI
- `g`: declared domain/compartment stratum
- `k`: state index on the shared `K`-state basis
- `K`: number of states

The canonical hierarchy is:

1. observation units,
2. FOV/ROI observation vectors on the shared `K`-state basis,
3. patient-level remodeling objects,
4. cohort-level recurrence summaries.

## 2. Canonical Names Versus Accepted Aliases

Canonical STRIDE names are used in docs, new code, and new exports. The
current longitudinal validator accepts the aliases listed below and normalizes
them before canonical fitting.

### 2.1 Canonical names

The preferred canonical field and key names are:

| Contract item | Canonical name |
|---|---|
| timepoint identifier | `timepoint` |
| FOV/ROI identifier | `fov_id` |
| observation domain label | `domain_label` |
| observation feature key | `local_state_features` |
| shared-basis state identifier | `state_id` |
| shared-basis centroids | `state_centroids` |
| cost scaling object | `cost_scale` |

Current domain-label note:

- the design-level observation-layer metadata concept is `domain_label`,
- the official AnnData route currently realizes that metadata through the
  concrete field `compartment`,
- labels such as `TC`, `IM`, and `PT` are observation-layer strata.

### 2.2 Accepted aliases in the current longitudinal validator

The current implementation validator accepts the following alias sets:

| Contract item | Canonical name | Accepted alias set |
|---|---|---|
| timepoint identifier | `timepoint` | `timepoint`, `timepoint_id` |
| FOV/ROI identifier | `fov_id` | `fov_id`, `roi_id` |
| observation feature key | `local_state_features` | `local_state_features`, `community_features` |
| shared-basis state identifier | `state_id` | `state_id`, `proto_id` |
| shared-basis centroids | `state_centroids` | `state_centroids`, `prototype_centroids` |
| cost scaling object | `cost_scale` | `cost_scale`, `s_C`, `global_cost_scale` |

Normalization rule:

- validators normalize accepted aliases into canonical names before fitting,
- canonical builders write canonical names first.

## 3. Identifier And Official-Route Input Contract

Any valid analysis surface must be able to represent the following identifiers
explicitly:

| Field | Level | Requirement | Meaning |
|---|---|---|---|
| `patient_id` | patient | required | patient membership key |
| `timepoint` | timepoint | required canonically | human-readable ordered-side label under the canonical naming surface |
| `timepoint_order` | timepoint | required logically | ordered relation used for analysis direction |
| `fov_id` | FOV/ROI | required canonically | observation-layer unit key |
| `roi_id` | FOV/ROI | accepted alias | modality-specific synonym when the observation unit is called an ROI |
| `domain_label` | observation stratum | required at the design level for the domain-aware route | observation-layer tissue/domain metadata used for stratification, grouped discrepancy organization, and patient-relation input grouping |
| `compartment` | observation stratum | required for the current AnnData official route | concrete AnnData field realizing design-level `domain_label` metadata |
| `state_id` | shared basis | required | index on the shared `K`-state basis |
| `group_id` | optional grouping | optional | benchmark-agnostic grouping variable if used |

The method contract does not allow implicit time ordering or implicit
shared-basis mixing.

An equivalent ordered relation may replace literal chronological time in a
task-local analysis, but it must still be declared explicitly through the same
ordering contract. Task A's ordered tissue-domain proxy is one example of this
kind of task-local instantiation.

The state/domain split is also part of the input contract:

- domain labels stratify the observation layer,
- domain labels organize grouped discrepancies and patient-relation input
  grouping,
- the shared `K`-state basis defines state identity,
- callers keep state construction and domain stratification as separate
  modeling layers,
- state geometry and the patient-level axes of `A_p`, `d_p`, and `e_p` are
  defined on the shared `K`-state basis,
- after task-layer source/target/domain resolution, the reusable core receives
  resolved source/target observation evidence blocks and does not use domain as
  a loss axis, state axis, relation axis, or recurrence axis.

### 3.1 Source/target observation comparison plan

The task layer must provide resolved ordered comparison metadata before calling
the reusable estimator. The core receives this plan and the resolved
source/target observation evidence blocks explicitly, then applies the same
task-insensitive `stride.tl.fit(...)` estimator to the resolved inputs.

Each comparison record must be able to represent:

| Field | Requirement | Meaning |
|---|---|---|
| `comparison_id` | required | stable key for the declared observation comparison |
| `source_group` | required | ordered source-side observation group |
| `target_group` | required | ordered target-side observation group |
| `valid_domain_strata` | required when domain-stratified comparison is used | permitted source/target domain strata or domain mapping |
| `patient_fov_linkage` | required | patient, source-side FOV, and target-side FOV membership used to build the empirical measures |

The comparison plan declares which observation groups and domain strata are
eligible for comparison. It does not redefine the shared `K`-state basis, the
fitted patient-level object, or any core recurrence axis. The resolved plan is
still recorded in provenance.

### 3.2 Official longitudinal AnnData route

For the current AnnData implementation route, the longitudinal validator
requires or resolves:

- `adata.obs['patient_id']`,
- one accepted timepoint identifier from the `timepoint` alias set,
- one accepted FOV/ROI identifier from the `fov_id` alias set,
- `adata.obs['compartment']`,
- `adata.obsm['spatial']`,
- ROI area metadata only for future/custom density community semantics.

Additional route requirements are stage-conditional:

- `cell_type` is required for the official shared-basis feature-construction
  route,
- `local_state_features` or `community_features` are required once the route
  expects a fitted observation representation,
- `state_id` or `proto_id` are required once the route expects a realized
  shared-state assignment,
- `cost_matrix` and an accepted `cost_scale` alias are required once the route
  enters observation-layer matching or full-estimator geometry/locality
  fitting.

Current first-pass canonical route:

- shared community-state construction is tissue-agnostic and happens before any
  tissue/domain stratification,
- the first-pass route is: per-cell subtype labels -> within-ROI kNN
  neighborhood subtype proportion vectors -> k-means shared community states,
- the neighborhood size `k` is user-configurable; the documented first-pass
  default is `20`,
- ROI/FOV observation vectors are constructed only after shared state
  assignments exist, by counting cells assigned to each shared community/state
  within the ROI/FOV and dividing by the ROI/FOV total cell count,
- this ROI/FOV-level community composition is distinct from the cell-level
  neighborhood subtype composition used during shared community-state
  construction.

Implementation note:

- implementation surfaces write canonical keys first,
- task wrappers may materialize accepted aliases when a task-level input route
  declares them.

## 4. Observation-Layer Contract

### 4.1 Observation vector

The canonical current first-pass observation-layer quantitative object is:

- `v_{p,t,f} in R_+^K`

Definition:

- `v_{p,t,f}[k]` is the normalized ROI/FOV community composition assigned to
  shared community/state `k` for patient `p`, ordered side `t`, and FOV/ROI
  `f`,
- in the current first pass,
  `v_{p,t,f}[k] = (# cells in ROI/FOV f assigned to state k) / (# total cells in ROI/FOV f)`.

A valid observation vector must satisfy all of the following:

- same `K` and same state ordering across the analysis,
- finite entries,
- nonnegative entries,
- sum to `1` within declared numerical tolerance in the current first pass,
- explicit linkage to `patient_id`, `timepoint`, and `fov_id`.

### 4.2 State-construction composition versus observation composition

- the cell-level neighborhood subtype composition used during shared
  community-state construction is tissue-agnostic and exists before any
  tissue/domain stratification,
- the ROI/FOV observation vector `v_{p,t,f}` is formed later, after cells have
  been assigned to the shared community/state basis,
- docs and interfaces must not blur these two layers of composition.

### 4.3 Domain-stratified empirical measure

The canonical observation-layer comparison object is a domain-stratified
bag-of-FOV empirical measure in community-composition space with one
observation support point per eligible ROI/FOV:

- `nu_obs = sum_f w_f delta_{c(v_f)}`

where:

- `f` indexes the FOVs/ROIs inside a declared domain stratum,
- `w_f = 1` in the current first pass,
- `c(v_f) = v_f` in the current first pass,
- the support points are ROI/FOV community compositions, not the cell-level
  neighborhood subtype compositions used during state construction and not raw
  coordinates or dense transport-plan cells.

Contract boundary:

- domain-stratified bag-of-FOV comparison is the canonical observation layer,
- canonical v1 `L_obs` uses the fixed, versioned, auditable
  `D_obs^BalancedSinkhornDivergence-v1` operator comparing predicted and
  observed target-side FOV bags,
- `D_obs` is a fixed operator inside `L_obs`, not a sixth loss term and not an
  independently weighted component,
- source-side FOV vectors are reconstructed as
  `y_hat_f = normalize(v_source_f @ A_p + e_p)`,
- `L_obs_pair_raw = D_obs^BalancedSinkhornDivergence-v1(predicted target FOV bag, observed target FOV bag; C_norm)`,
- `C_norm = C_raw / s_C` remains the state/community-level cost,
- the FOV-level ground cost is
  `G[f,g] = debiased balanced Sinkhorn composition distance(y_hat_f, y_true_g; C_norm)`,
- the outer loss is the debiased balanced Sinkhorn divergence between the
  predicted target FOV bag and observed target FOV bag under
  `G_norm = G / s_G_init`,
- `s_G_init` is a fixed evidence-block scale computed at deterministic
  identity-plus-small-open initialization from positive finite
  initialization-time FOV-level costs; if no positive finite costs exist,
  `s_G_init = 1.0` and floor usage is recorded,
- `s_G_init` is not recomputed dynamically during optimization,
- compact successful-fit provenance records backend `torch`, dtype `float64`,
  log-domain balanced/debiased Sinkhorn settings, inner and outer epsilon
  schedules `(0.5, 0.2, 0.1)`, `max_iter = 100` per epsilon stage,
  `tol = 1e-6`, `warning_tol = 1e-4`, and the state-geometry normalization
  used by the fit,
- tiny negative inner composition-distance values in `[-1e-10, 0)` are clamped
  to `0` with provenance; values below `-1e-10` are numerical failures,
- empty source or target FOV bags, invalid simplexes, negative entries,
  NaN/Inf values, invalid cost matrices, or unavailable `torch` for the
  canonical full estimator fail explicitly; padding is not allowed,
- reaching `max_iter` with finite values is usable with warning; updates larger
  than `warning_tol` are recorded convergence warnings on diagnostic/profiling
  surfaces, not necessarily every production optimizer forward,
- canonical `L_obs` has no observation-layer unbalanced unmatched residual;
  open behavior is expressed only by fitted biological `d/e`,
- retired UOT, `D_pos/B_pos`, and unbalanced observation residual diagnostics
  may be emitted only as diagnostic or comparator fields; Task A Block 3
  preserves `uot_baseline` as an external comparator name,
- docs should not promote raw histogram collapse as the default canonical
  observation object,
- docs should not leave the observation layer as an abstract placeholder `P`.

### 4.4 Current first-pass FOV observation semantics

The current canonical first-pass declaration is:

- current first-pass observation vectors are normalized community compositions
  of state assignments within each ROI/FOV,
- each eligible ROI/FOV contributes one support point to the bag-of-FOV
  empirical measure,
- ROI/FOVs are not weighted by tissue amount in the current first pass,
- task-specific count, density, or proportion inputs must first be mapped into
  the current observation contract.


### 4.5 Optional observation-layer priors

Optional priors or side inputs may include:

- shared-state cost geometry for observation comparison and geometry/locality,
- shared-basis construction metadata,
- drift vectors,
- support masks,
- uncertainty mode and resampling settings.

These are optional priors. They do not replace the observation-vector contract.

### 4.6 Shared-state geometry cost matrix

When a route provides the state-geometry cost for canonical observation
comparison or full-estimator geometry/locality fitting, the contract is:

- `C_norm = C_raw / s_C`,
- `C_raw` and `C_norm` have shape `[K, K]` on the same shared state basis as
  `A_p`, `d_p`, `e_p`, and the observation vectors,
- entries must be finite and nonnegative,
- matrices must be symmetric within declared numerical tolerance,
- diagonal entries must be `0` within declared numerical tolerance,
- the default `s_C` is the median of positive finite off-diagonal entries of
  `C_raw`,
- at least one positive finite off-diagonal entry must exist,
- absence of a valid positive off-diagonal scale is a contract failure,
- the route must not fall back to `s_C = 1.0` or automatically disable
  geometry for an invalid state-geometry cost matrix,
- `C` encodes symmetric state-identity distance, while `A_p` encodes directed
  remodeling relation mass.

For the full estimator,
`L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]`.
All `K` source rows are included. Diagonal self-retention pays no geometry cost
because `C_norm[i,i] = 0`. The objective-level raw value is the simple mean
over valid fitted patients. Geometry normalization uses
`scale_geometry = max(raw_L_geometry(theta_init), 1e-2)`, where `theta_init`
is the deterministic identity-plus-small-open initialization. The `1e-2` floor
is the full-estimator `epsilon_norm` loss-normalization floor. When
`raw_L_geometry(theta_init)` is valid and finite but zero or near zero, floor
use is provenance-only and records
`loss.components.geometry.floor_used = true` rather than changing
`fit_status` or creating a warning/failure.

## 5. Patient-Level Output Contract

### 5.1 Primary output object

For each patient `p`, the canonical primary output is `(T_p, e_p)` where
`T_p = [A_p | d_p]`.

| Object | Shape | Contract semantics |
|---|---|---|
| `A_p` | `[K, K]` | patient-level continuity/remodeling operator on the shared `K`-state basis |
| `d_p` | `[K]` | source-side outgoing open tendency |
| `e_p` | `[K]` | target-side incoming open-entry tendency |
| `T_p` | conceptual block object | may be stored as `A_p` plus `d_p`; the contract is the same |

Required properties:

- `A_p`, `d_p`, and `e_p` are finite when the patient fit is valid,
- `A_p` and `d_p` satisfy source-side row accounting: each source row
  `[A_{p,i,*}, d_{p,i}]` lies on a simplex, equivalently
  `sum_j A_{p,ij} + d_{p,i} = 1`,
- valid `e_p` is bounded in `[0,1]`,
- `A_p` is not required to be diagonal,
- `d_p` and `e_p` are not optional leftovers,
- if a conditional kernel is documented, it must be a derived auxiliary object
  from `A_p` and `d_p`, not the canonical patient contract itself.

Fitted-output reconstruction semantics:

- `raw_post = q_minus @ A + e`,
- `predicted_q_plus = normalize(raw_post)`,
- for source-side FOV vectors,
  `y_hat_f = normalize(v_source_f @ A_p + e_p)`,
- predicted target-side FOV vectors induce the predicted target-side
  bag-of-FOV empirical measure used by `L_obs`,
- retired `D_pos/B_pos` and other residual diagnostics may be emitted only as
  diagnostic or comparator fields. They are not canonical `L_obs` outputs, not
  fitted biological `d/e`, and not independently weighted fitted-output
  components.

### 5.2 Burden-scale auxiliary objects

The patient-level scale contract also includes:

| Object | Shape | Contract semantics |
|---|---|---|
| `mu_p^-` | `[K]` | pre-side patient pseudo-mass / burden vector |
| `mu_p^+` | `[K]` | post-side patient pseudo-mass / burden vector |
| `q_p^-` | `[K]` | derived normalized pre-side composition |
| `q_p^+` | `[K]` | derived normalized post-side composition |

Additional scale rules:

- `q_p^- = mu_p^- / ||mu_p^-||_1`,
- `q_p^+ = mu_p^+ / ||mu_p^+||_1`,
- `mu_p^-` and `mu_p^+` are not canonical compositions,
- `L_open = mean(d)+mean(e)` is a tendency-level open-channel objective cost
  on bounded fitted `d_p` and `e_p`,
- any documented `m_p^(d)` or `m_p^(e)` summaries live on the same
  pseudo-mass / burden scale as `mu_p^-` and `mu_p^+`,
- burden-scale summaries such as `m_p^(d)` and `m_p^(e)` are derived
  auxiliaries and should be interpreted on the declared pseudo-mass / burden
  scale,
- conservation is a soft burden-consistency anchor rather than literal
  physical conservation,
- composition-level structure may remain interpretable when burden-level
  comparability is weak,
- burden-level claims must be weakened or disabled when coverage or platform
  comparability is poor.

### 5.3 Patient relation container and audit payload

The current canonical implementation container surface is
`stride.latent.operators.PatientRelation`, optionally accompanied by
`stride.latent.operators.PatientRelationAudit`.

The current fit-result containers that carry these patient objects through the
workflow/API surface are:

- `stride.outputs.fit_result.PatientBridgeResult`,
- `stride.outputs.fit_result.STRIDEFitResult`.

Implementation-tier rule:

- canonical full-method workflow results report
  `implementation_tier="canonical_full"`,
- assembled explicit array payloads may report
  `implementation_tier="assembled_relation"` where appropriate.

`implementation_tier="canonical_full"` is a current container/lineage label for
the canonical workflow tier. It is not by itself evidence that the frozen
PyTorch/AdamW full-objective estimator has been implemented. Bridge-named
containers such as `PatientBridgeResult` remain implementation containers and
compatibility lineage surfaces; they do not define a live public bridge-method
contract.

The minimum patient-audit surface should be able to report:

| Audit field | Meaning |
|---|---|
| `patient_id` | patient membership key |
| `timepoint_order` | ordered relation used for the patient |
| `n_pre_observations` | count of pre-side FOVs/ROIs used |
| `n_post_observations` | count of post-side FOVs/ROIs used |
| `observation_fit_status` | observation-layer status summary |
| `relation_fit_status` | patient-relation fit or assembly status |
| `uncertainty_mode` | uncertainty route used, if any |
| `metadata` | additional audit metadata |

In the current first pass, uncertainty means bootstrap/sampling-variance
uncertainty over fitted patient relation outputs rather than a hurdle or
measurement-error estimator.

These audits do not replace the primary patient-level object. They remain
attached provenance and status surfaces.

### 5.4 Derived patient summaries

Derived summaries may include:

- diagonal retention summaries from `diag(A_p)`,
- off-diagonal remodeling summaries from `offdiag(A_p)`,
- total or state-specific source-side outgoing open-tendency summaries from
  `d_p`,
- total or state-specific target-side incoming open-entry summaries from
  `e_p`,
- burden-scale summaries from `mu_p^-`, `mu_p^+`, `m_p^(d)`, or `m_p^(e)`,
- derived composition summaries from `q_p^-` or `q_p^+`.

These summaries do not replace the primary object. The primary relation
contract remains the full `A_p` matrix rather than a selected derived subset.

## 6. Cohort-Level Output Contract

Cohort-level recurrence acts on patient-level objects. It does not act on
pooled FOV vectors as the primary contract. Full-estimator v1 recurrence uses a
cohort-supported consensus over `T_p = [A_p | d_p]` and `e_p` and reports
dispersion around that consensus.

Any cohort-level recurrence output must report:

- which patient-level objects were used,
- the recurrence unit (`patient`),
- patient support count for the consensus summary,
- cohort consensus `T/e` defined on the same shared `K`-state basis,
- dispersion around the consensus,
- recurrence-layer fit status.

### 6.1 Current recurrence implementation containers

The current implementation recurrence container surface is:

- `RecurrenceFamily` for family-level summaries,
- `RecurrenceResult` for cohort-level recurrence output.

If the recurrence layer produces family summaries, the minimal record should be
able to represent the v1 consensus summary through the compatibility container:

| Field | Meaning |
|---|---|
| `family_id` | compatibility key; v1 uses a single consensus summary |
| `template_A` | cohort consensus relation template on `[K, K]` |
| `template_d` | cohort consensus source-side outgoing open-tendency template on `[K]` |
| `template_e` | cohort consensus target-side incoming open-entry template on `[K]` |
| `support_n_patients` | number of patients supporting the consensus |
| `within_family_dispersion` | dispersion around the consensus |
| `fit_status` | explicit recurrence-layer status |

The minimal cohort-level record should also be able to represent:

| Field | Meaning |
|---|---|
| `patient_ids` | patients included in the recurrence pass |
| `used_patient_ids` | patients with realized relations actually used by the current estimator |
| `recurrence_unit` | recurrence unit, currently `patient` |
| `families` | compatibility container; v1 semantics are empty/deferred or one consensus summary |
| `parameters` | recurrence-space parameters such as basis dimension and loadings |
| `embeddings` | optional patient-level recurrence coordinates with explicit fit status |
| `fit_status` | overall recurrence-layer status |
| `metadata` | cohort-level recurrence metadata |

Current first-pass recurrence rule:

- the canonical recurrence surface supports a conservative first-pass
  consensus-template estimator,
- the `families` field may remain for container compatibility, but live v1
  semantics are a single consensus family/summary rather than automatic
  discovery of multiple remodeling families,
- it may still return `fit_status="deferred"` with no consensus summary when
  patient support is insufficient rather than fabricating assignments beyond
  the supported evidence.

## 7. Audit And Failure Contract

The data contract requires explicit auditability:

- no silent fallback from invalid to apparently valid outputs,
- no silent removal of patients or FOVs from outputs without an audit trail,
- explicit distinction between missing biological support and engineering
  failure,
- explicit declaration of uncertainty mode and patient-relation fit or
  assembly mode,
- explicit deferred status where patient or cohort support remains
  insufficient,
- explicit distinction between canonical full-method and approximate/proxy
  workflow tiers, without treating tier labels alone as proof that a supported
  full-objective optimizer run completed successfully.
- explicit state-geometry cost validation, `s_C`, and the geometry component
  provenance floor flag for full-estimator fits that use the
  geometry/locality component.
- normalization floors do not rescue invalid raw losses or invalid inputs;
  NaN/Inf values, illegal negative raw losses, empty observation bags, invalid
  simplexes, invalid cost matrices, unavailable `torch`, and related contract
  violations still fail explicitly.
- Compact successful-fit provenance does not replace fail-fast validation.
  Invalid observation vectors, invalid simplexes, invalid source or target FOV
  bags, invalid shared-state cost matrices, invalid configuration, and
  unavailable canonical dependencies remain explicit errors under canonical
  `ContractError` semantics where they are low-level contract violations.
- Compact successful-fit provenance is not required to contain per-patient
  records, per-evidence-block records, per-evidence-block status, status
  counts, `failure_reason`, or `optimizer_failure_reason`. Existing result
  containers may continue to carry their own status fields.

Observation-layer implementations expose explicit fit-status fields as audit
payloads.

## 8. Explicit Claim Boundary

These data contracts support longitudinal remodeling analysis with explicit
claim boundaries for:

- lineage-tracing interpretation,
- physical transport interpretation,
- FOV matching interpretation,
- whole-lesion dynamics interpretation.
