# Data Contracts

This file defines the target data contracts for STRIDE. The contracts are
package-neutral at the design level. Where `slotar.*` is named below, it refers
to the current implementation surface during migration rather than the
canonical future architecture. The live first-pass canonical implementation now
also exists under `stride.*`. Current task outputs and transport-era fields may
lag these contracts; their migration status is documented explicitly below.

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

Canonical STRIDE names should be preferred in docs, new code, and new exports.
The current longitudinal validator still accepts several implementation-era
aliases during migration.

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
- labels such as `TC`, `IM`, and `PT` are observation-layer strata, not part of
  state identity.

### 2.2 Accepted aliases in the current longitudinal validator

The current implementation validator, `slotar.io.longitudinal`, accepts the
following alias sets:

| Contract item | Canonical name | Accepted alias set during migration |
|---|---|---|
| timepoint identifier | `timepoint` | `timepoint`, `timepoint_id` |
| FOV/ROI identifier | `fov_id` | `fov_id`, `roi_id` |
| observation feature key | `local_state_features` | `local_state_features`, `community_features` |
| shared-basis state identifier | `state_id` | `state_id`, `proto_id` |
| shared-basis centroids | `state_centroids` | `state_centroids`, `prototype_centroids` |
| cost scaling object | `cost_scale` | `cost_scale`, `s_C`, `global_cost_scale` |

Migration rule:

- accepted aliases are compatibility behavior, not equal-status canonical names,
- canonical builders should write canonical names first; alias fields are added
  only by compatibility wrappers when older task code still needs them.

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
| `domain_label` | observation stratum | required at the design level for the domain-aware route | observation-layer tissue/domain metadata used for stratification, grouped discrepancy organization, and bridge input grouping |
| `compartment` | observation stratum | required for the current AnnData official route | concrete AnnData field realizing design-level `domain_label` metadata |
| `state_id` | shared basis | required | index on the shared `K`-state basis |
| `mass_mode` | analysis | required | declared semantics for observation mass; current first-pass canonical value is `uniform` |
| `group_id` | optional grouping | optional | benchmark-agnostic grouping variable if used |

The method contract does not allow implicit time ordering or implicit
shared-basis mixing.

An equivalent ordered relation may replace literal chronological time in a
task-local analysis, but it must still be declared explicitly through the same
ordering contract. Task A's ordered tissue-domain proxy is one example of this
kind of task-local instantiation.

The state/domain split is also part of the input contract:

- domain labels stratify the observation layer,
- domain labels organize grouped discrepancies and bridge input grouping,
- the shared `K`-state basis excludes domain identity,
- callers must not encode domain into the basis and then condition on domain
  again,
- domain labels do not define state geometry or the patient-level axes of
  `A_p`, `d_p`, and `e_p`.

### 3.1 Official longitudinal AnnData route

For the current AnnData implementation route, the longitudinal validator
requires or resolves:

- `adata.obs['patient_id']`,
- one accepted timepoint identifier from the `timepoint` alias set,
- one accepted FOV/ROI identifier from the `fov_id` alias set,
- `adata.obs['compartment']`,
- `adata.obsm['spatial']`,
- ROI area metadata only for future/custom non-uniform mass semantics.

Additional route requirements are stage-conditional:

- `cell_type` is required for the official shared-basis feature-construction
  route,
- `local_state_features` or `community_features` are required once the route
  expects a fitted observation representation,
- `state_id` or `proto_id` are required once the route expects a realized
  shared-state assignment,
- `cost_matrix` and an accepted `cost_scale` alias are required once the route
  enters observation-layer matching.

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

Current migration note:

- current `slotar.state_space` and `slotar.io.longitudinal` implementation
  surfaces write canonical keys first,
- compatibility wrappers may still materialize `community_features`,
  `proto_id`, `prototype_centroids`, or `s_C` when older task code requires
  them.

## 4. Observation-Layer Contract

### 4.1 Observation vector

The canonical current first-pass observation-layer quantitative object is:

- `v_{p,t,f} in R_+^K`

Definition:

- `v_{p,t,f}[k]` is the normalized ROI/FOV community composition assigned to
  shared community/state `k` for patient `p`, ordered side `t`, and FOV/ROI
  `f`,
- in the current first pass,
  `v_{p,t,f}[k] = (# cells in ROI/FOV f assigned to state k) / (# total cells in ROI/FOV f)`,
- current first-pass observation mass is separate from `v_{p,t,f}` and is fixed
  to `mass = 1` with `mass_mode = "uniform"` for each ROI/FOV in a study.

A valid observation vector must satisfy all of the following:

- same `K` and same state ordering across the analysis,
- finite entries,
- nonnegative entries,
- sum to `1` within declared numerical tolerance in the current first pass,
- explicit `mass_mode`,
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
bag-of-FOV empirical measure in community-composition space with equal
ROI/FOV mass:

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
- OT / Sinkhorn may compare these empirical measures,
- docs should not promote raw histogram collapse as the default canonical
  observation object,
- docs should not leave the observation layer as an abstract placeholder `P`.

### 4.4 Current first-pass `mass_mode` semantics

The current canonical first-pass declaration is:

- `mass_mode = "uniform"`,
- `mass = 1` for every ROI/FOV within a study,
- no ROI/FOV weighting by tissue amount is applied in the current first pass,
- `mass` remains part of the contract as a future-extensible field.

Current transitional implementations may still expose legacy `count`,
`density`, or `proportion` semantics. Those remain implementation or
compatibility surfaces during migration and do not redefine the current
canonical first-pass contract.

### 4.5 Optional observation-layer priors

Optional priors or side inputs may include:

- observation-layer cost geometry,
- shared-basis construction metadata,
- drift vectors,
- support masks,
- uncertainty mode and resampling settings.

These are optional priors. They do not replace the observation-vector contract.

## 5. Patient-Level Output Contract

### 5.1 Primary output object

For each patient `p`, the canonical primary output is `(T_p, e_p)` where
`T_p = [A_p | d_p]`.

| Object | Shape | Contract semantics |
|---|---|---|
| `A_p` | `[K, K]` | patient-level continuity/remodeling operator on the shared `K`-state basis |
| `d_p` | `[K]` | depletion tendency on the pre side |
| `e_p` | `[K]` | post-side emergence |
| `T_p` | conceptual block object | may be stored as `A_p` plus `d_p`; the contract is the same |

Required properties:

- `A_p`, `d_p`, and `e_p` are finite when the patient fit is valid,
- all three are nonnegative,
- `A_p` is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`,
- `A_p` is not required to be diagonal,
- `d_p` and `e_p` are not optional leftovers,
- if a conditional kernel is documented, it must be a derived auxiliary object
  from `A_p` and `d_p`, not the canonical patient contract itself.

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
- any documented `m_p^(d)` or `m_p^(e)` summaries live on the same
  pseudo-mass / burden scale as `mu_p^-` and `mu_p^+`,
- conservation is a soft burden-consistency anchor rather than literal
  physical conservation,
- composition-level structure may remain interpretable when burden-level
  comparability is weak,
- burden-level claims must be weakened or disabled when coverage or platform
  comparability is poor.

### 5.3 Patient relation container and audit payload

The current implementation container surface is
`slotar.patient.PatientRelation`, optionally accompanied by
`slotar.patient.PatientRelationAudit`.

The minimum patient-audit surface should be able to report:

| Audit field | Meaning |
|---|---|
| `patient_id` | patient membership key |
| `timepoint_order` | ordered relation used for the patient |
| `mass_mode` | declared mass semantics |
| `n_pre_observations` | count of pre-side FOVs/ROIs used |
| `n_post_observations` | count of post-side FOVs/ROIs used |
| `observation_fit_status` | observation-layer status summary |
| `bridge_status` | patient-assembly or bridge-estimation status |
| `uncertainty_mode` | uncertainty route used, if any |
| `metadata` | additional audit metadata |

In the current first pass, uncertainty means bootstrap/sampling-variance
uncertainty over realized bridge outputs rather than a hurdle or
measurement-error estimator.

These audits do not replace the primary patient-level object. They remain
attached provenance and status surfaces.

### 5.4 Derived patient summaries

Derived summaries may include:

- diagonal retention summaries from `diag(A_p)`,
- off-diagonal remodeling summaries from `offdiag(A_p)`,
- total depletion or state-specific depletion from `d_p`,
- total emergence or state-specific emergence from `e_p`,
- burden-scale summaries from `mu_p^-`, `mu_p^+`, `m_p^(d)`, or `m_p^(e)`,
- derived composition summaries from `q_p^-` or `q_p^+`.

These summaries do not replace the primary object.

## 6. Cohort-Level Output Contract

Cohort-level recurrence acts on patient-level objects. It does not act on
pooled FOV vectors as the primary contract.

Any cohort-level recurrence output must report:

- which patient-level objects were used,
- the recurrence unit (`patient`),
- support size per recurrence family or summary,
- a family- or cohort-level summary defined on the same shared `K`-state basis,
- dispersion or stability diagnostics,
- recurrence-layer fit status.

### 6.1 Current recurrence implementation containers

The current implementation recurrence container surface is:

- `RecurrenceFamily` for family-level summaries,
- `RecurrenceResult` for cohort-level recurrence output.

If the recurrence layer produces family summaries, the minimal record should be
able to represent:

| Field | Meaning |
|---|---|
| `family_id` | recurrence family key |
| `template_A` | family-level relation template on `[K, K]` |
| `template_d` | family-level depletion template on `[K]` |
| `template_e` | family-level emergence template on `[K]` |
| `support_n_patients` | number of patients supporting the family |
| `within_family_dispersion` | recurrence compactness / stability summary |
| `fit_status` | explicit recurrence-layer status |

The minimal cohort-level record should also be able to represent:

| Field | Meaning |
|---|---|
| `patient_ids` | patients included in the recurrence pass |
| `families` | family summaries returned, possibly empty |
| `fit_status` | overall recurrence-layer status |
| `metadata` | cohort-level recurrence metadata |

Current deferred-status rule:

- a canonical recurrence surface may return `fit_status="deferred"` with no
  families rather than fabricating family assignments before the final estimator
  exists.

## 7. Audit And Failure Contract

The data contract requires explicit auditability:

- no silent fallback from invalid to apparently valid outputs,
- no silent removal of patients or FOVs from outputs without an audit trail,
- explicit distinction between missing biological support and engineering
  failure,
- explicit declaration of uncertainty mode and bridge mode,
- explicit deferred status where a canonical namespace exists but its final
  estimator is not yet implemented.

Current observation-layer implementations may continue to expose explicit
fit-status fields. Those remain compatibility diagnostics until a future cleanup
retires them.

## 8. Legacy Compatibility And Migration Notes

Current code and task outputs still expose observation-layer compatibility
surfaces from earlier transport-centered implementations. They are not the
target patient-level contract. Their relationship to the current canonical
contract is only partial:

| Compatibility surface class | Current meaning | Relation to canonical contract | Migration note |
|---|---|---|---|
| observation-layer transport and unmatched summaries | solver-level comparison diagnostics | not patient-level remodeling objects | may remain bridge diagnostics only |
| thresholding, labeling, or regularization controls | observation-layer configuration or prior state | not biological outputs | keep as internal controls during migration |
| observation-layer fit-status fields | compatibility failure or audit metadata | audit only | remains useful until code refactor |
| dense observation-layer plan payloads and aliases | observation-layer matching detail | not a patient-level or cohort-level contract object | do not export as canonical method objects |

Migration rule:

- do not claim a one-to-one mapping from compatibility surfaces to
  `(A_p, d_p, e_p)` unless a future implementation adds that mapping
  explicitly,
- treat `slotar.compat.*` and the temporary top-level shim modules
  (`slotar.uot`, `slotar.representation`, `slotar.contracts`, `slotar.uq`,
  `slotar.utils`, `slotar.io.bridge`, `slotar.drift`, and
  `slotar.exceptions`) as migration residue rather than canonical interfaces,
- treat `stride` as the target architecture direction even when the current
  working implementation still lives under `slotar.*`.

## 9. Explicit Non-Claim Boundary

These data contracts support longitudinal remodeling analysis. They do not
support automatic claims of:

- lineage tracing,
- exact physical transport truth,
- exact one-to-one FOV matching,
- unbiased whole-lesion dynamics.
