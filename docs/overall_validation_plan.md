# Overall Validation Plan

This file defines the method-level validation plan for STRIDE. It is
task-agnostic by design. Task datasets and task papers are later execution
surfaces, not the primary validation logic in this document.

## 1. Validation Goals

The validation program is intended to answer the following questions:

1. does the method preserve the canonical patient-level target
   `(T_p, e_p)` rather than collapsing back to transport-first claims,
2. does it preserve the row-substochastic `A_p` contract rather than silently
   replacing it with a pure conditional-kernel story,
3. does it keep burden-scale and composition-scale semantics separate,
4. does it preserve the concrete observation layer as domain-stratified
   bag-of-FOV empirical measure comparison,
5. does it avoid state/domain double counting,
6. does it recover cohort consensus recurrence/dispersion support at the cohort
   level once the full objective-driven patient-relation estimator and
   recurrence estimator are fully live,
7. does it keep emergence explicit when post-side novelty is present,
8. does it fail in the expected direction under ablations and shuffled controls.

## 2. Validation Staging And Claim Boundary

The validation plan has two distinct stages. The docs must keep them separate.

### 2.1 Current normative validation layer

The following validation semantics are normative now:

- contract validity for observation, patient, and recurrence objects,
- the current narrow two-group uniform patient-relation implementation path,
- explicit separation between observation-layer diagnostics and patient-level
  outputs,
- explicit separation between any emitted retired observation residual
  diagnostics and fitted biological `d/e`,
- explicit row-substochastic `A_p` semantics,
- explicit bounded `e_p` semantics,
- explicit burden/composition scale separation,
- explicit domain-stratified bag-of-FOV observation semantics,
- explicit source/target declaration as task input rather than estimator
  identity,
- explicit no-double-counting treatment of domain versus state identity,
- explicit bootstrap/sampling-variance uncertainty over fitted patient relation
  outputs,
- explicit auditability and no silent fallback,
- honesty about deferred estimator state where canonical namespaces exist but
  final estimators are not yet complete.

This means the current docs may require:

- valid patient-level arrays when a patient relation is emitted,
- valid recurrence containers or explicit deferred recurrence status,
- no promotion of retired transport-era fields directly into canonical
  patient-level claims without an explicit patient-relation contract.

### 2.2 Deferred execution-stage validation layer

The following validation targets remain method-level requirements, but they are
execution-stage targets that apply once the full objective-driven
patient-relation estimator and non-deferred recurrence estimator are fully
implemented:

- synthetic recovery of planted patient-level relation structure,
- recovery of the cohort consensus relation and dispersion support from
  patient-level relations,
- optional exploratory recovery of multiple remodeling-family structure from
  fitted v1 outputs as a future extension rather than a current v1 requirement,
- numeric performance thresholds such as ARI or cosine-similarity targets,
- ablation sensitivities measured on end-to-end synthetic pipelines.

These targets remain the intended evidence standard. They are not grounds for
pretending those estimators are already complete.

## 3. Contract-Level Invariants

Every validation scenario should first check contract-level invariants:

- valid patient-level outputs have shapes `[K, K]`, `[K]`, and `[K]`,
- valid outputs are finite and nonnegative,
- `[A_p | d_p]` satisfies the row-simplex contract
  `sum_j A_{p,ij} + d_{p,i} = 1`,
- `e_p` is bounded in `[0,1]`,
- any documented conditional kernel is derived from `A_p` and `d_p` rather than
  substituted for them,
- shared-basis consistency is preserved across observations, patients, and
  cohort summaries,
- `mu_p^-`, `mu_p^+`, `m_p^(d)`, and `m_p^(e)` remain burden-scale quantities,
  while `q_p^-` and `q_p^+` remain derived compositions,
- current uncertainty outputs, when present, summarize bootstrap/sampling
  variance over fitted patient relation outputs rather than an older hurdle or
  measurement-error model,
- observation-layer outputs remain distinguishable from patient-level solver
  outputs and remain documented as domain-stratified bag-of-FOV measures in
  composition space,
- source/target declaration changes task input but not the identity of
  `fit_stride(...)` as the full estimator,
- any emitted retired observation residual diagnostics remain distinguishable
  from fitted biological `d/e`,
- state-geometry cost matrices are finite, nonnegative, symmetric on the
  shared `K`-state basis, have diagonal `0`, and provide a positive finite
  off-diagonal median scale before canonical fitting,
- the geometry/locality objective acts on raw canonical `A_p` and remains a
  soft biological-complexity cost rather than a hard support mask,
- manuscript-level biological process language requires cohort-level
  recurrence/support/dispersion over fitted `A/d/e`,
- domain is not encoded into state identity and then reused as a conditioning
  variable,
- failures remain explicit rather than silently coerced,
- deferred canonical estimator namespaces return explicit deferred states rather
  than fabricated end-to-end results.

## 4. Deferred Synthetic Recovery Suite

The synthetic recovery suite is the intended execution-stage validation surface
once the canonical end-to-end estimators are fully implemented.

### 4.1 Recovery of cohort consensus recurrence

Design:

- simulate patients from a cohort-supported consensus relation with controlled
  patient-level deviations,
- give the consensus a distinct `A_bar`, `d_bar`, and `e_bar` pattern,
- vary patient-level noise while keeping the consensus and dispersion
  recoverable,
- retain multiple FOVs per patient so the patient-relation fit remains
  observation-conditioned.

Required behavior:

- patient-level outputs recover the planted relation structure,
- recurrence analysis recovers the planted cohort consensus relation,
- support counts reflect the patients contributing to the consensus and
  dispersion reflects the planted deviation scale.

### 4.2 Coverage heterogeneity and burden/composition sensitivity

Design:

- vary the number of FOVs per patient and the relative coverage across ordered
  sides,
- hold the planted patient-level object fixed while perturbing observation
  coverage,
- include settings where composition remains stable while burden comparability
  becomes weak.

Required behavior:

- the model remains observation-aware rather than collapsing to naive pooled
  matching,
- patient-relation fit audits reflect the coverage differences explicitly,
- patient-level recovery degrades gracefully rather than failing silently,
- burden-level claims weaken before composition-level structure is discarded.

### 4.3 Emergence-present scenarios

Design:

- simulate patients with genuine post-side novelty that cannot be explained only
  by diagonal retention or off-diagonal remodeling in `A_p`,
- vary the amount and state concentration of emergence.

Required behavior:

- the model allocates meaningful signal to `e_p`,
- it does not force the signal into `A_p`, depletion, or an opaque unmatched
  scalar,
- cohort recurrence/dispersion reflects the supported emergence profile rather
  than converting it into an observation-layer diagnostic.

### 4.4 Domain-stratification and no-double-counting sensitivity

Design:

- simulate settings where domain-stratified observation structure matters,
- compare a correct shared-basis-plus-domain-stratification route against a
  leaked route that encodes domain in the basis and then conditions on domain
  again.

Required behavior:

- the correct route preserves interpretable patient-level structure,
- the leaked route shows inflated or unstable signal and therefore fails the
  contract boundary,
- validation surfaces make the double-counting failure visible.

## 5. Ablation Studies And Controls

These are execution-stage method validations for the completed end-to-end
canonical estimators.

The core STRIDE ablation vocabulary is `none`, `recurrence`, `geometry`, and
`consistency`; Block 3C exposes the refit arms as `recurrence_ablation`,
`geometry_ablation`, and `consistency_ablation`. No-`d/e`,
no-open-channel, closed, balanced, and transport-style variants remain
comparator/control vocabulary for baseline evaluation rather than core
full-estimator ablation names.

### 5.1 No recurrence layer

Remove or disable the recurrence layer while keeping patient-level objects.

Expected result:

- patient-level objects may still exist,
- cohort consensus support should weaken materially,
- cohort summaries should become less stable and less discriminative.

### 5.2 No geometry/locality prior

Remove or disable the geometry/locality prior while keeping the patient-level
relation variables and other objective terms active.

Expected result:

- recovery should become less stable when true remodeling follows the shared
  state geometry,
- implausibly distant transitions should increase under the ablated fit,
- patient-level relation recovery should degrade relative to the full fit.

The full fit is not expected to erase all distant remodeling. A distant
relation can remain when the joint objective supports it; the ablation checks
whether removing the biological-complexity cost increases unsupported distant
relations or destabilizes recovery.

### 5.3 No patient-consistency term

Remove or disable the patient-consistency term while keeping observation fit,
explicit open-channel terms, geometry/locality, and recurrence active.

Expected result:

- patient-level recovery should become less stable across partial observations,
- uncertainty and audit surfaces should show increased weak-support behavior,
- recurrence consensus recovery should degrade when patient-level relation
  estimates become less coherent.

### 5.4 Transport-first, closed-match, or no-open-channel comparator

Compare the remodeling-first design to an older transport-first or closed-match
framing, including no-`d/e` or no-open-channel comparator variants, that treats
the observation-layer comparison object as primary and downplays the fitted
open relation.

Expected result:

- the older framing should underperform when true emergence or depletion is
  present,
- it should be more vulnerable to forced diagonalization or forced one-to-one
  matching behavior,
- it should provide weaker cohort consensus recovery when recurrence truly acts
  on patient-level relations.

### 5.5 Histogram-collapse or state-domain leakage control

Replace domain-stratified bag-of-FOV comparison with either one pooled
histogram or a state basis that already encodes domain identity.

Expected result:

- patient-level structure should become less interpretable or less stable,
- recovered effects should become more sensitive to coverage artifacts,
- the failure should be visible as a contract violation or a degraded recovery
  surface rather than as a silent success.

### 5.6 Shuffled controls

Use patient-timepoint, recurrence, or state-label shuffling as weak controls.

Expected result:

- recovery of planted patient-level relations should collapse,
- recurrence consensus support should become unstable or non-recoverable,
- stability metrics should degrade toward weak-control behavior.

## 6. Failure-Mode Checks

The failure modes below define the direction in which the final method should be
stress-tested. Some checks can already be stated contractually; quantitative
recovery thresholds remain execution-stage targets.

### 6.1 Diagonal-retention collapse

Construct simulations with true off-diagonal remodeling mass.

Failure mode to detect:

- the model collapses toward an almost fully diagonal `A_p` even when the truth
  is strongly off-diagonal.

Required check:

- recovered off-diagonal structure must remain materially nonzero when planted
  off-diagonal remodeling is present.

### 6.2 Emergence overuse

Construct simulations with little or no true emergence.

Failure mode to detect:

- the model over-allocates signal into `e_p` instead of retention/remodeling or
  depletion.

Required check:

- recovered emergence should remain small in no-emergence scenarios and should
  increase only when genuine post-side novelty is planted.

## 7. Explicit Success Criteria

Validation success is defined by pipeline behavior and output contracts, not by
"nice-looking biology".

| Stage | Category | Success criterion |
|---|---|---|
| current normative layer | Contract validity | Any emitted patient-level object has `A_p`, `d_p`, and `e_p` with correct shapes and nonnegative finite values, `[A_p | d_p]` row-simplex accounting, and bounded `e_p` in `[0,1]`; invalid payloads fail explicitly rather than silently coercing values. |
| current normative layer | `A_p` semantics | Any emitted patient relation obeys row-substochastic semantics, and docs do not substitute a pure conditional kernel for canonical `A_p`. |
| current normative layer | Burden/composition honesty | `mu_p^-`, `mu_p^+`, `m_p^(d)`, and `m_p^(e)` remain burden-scale quantities, while normalized compositions remain derived views; burden claims are not documented as robust when comparability is weak. |
| current normative layer | Observation-layer boundary | Observation-layer outputs remain distinguishable from patient-level outputs and remain documented as domain-stratified bag-of-FOV empirical measures rather than as the primary biological object; retired residual diagnostics, when emitted, remain separate from fitted biological `d/e`. |
| current normative layer | Estimator invariance | Source/target declaration changes task input and comparison eligibility, while `fit_stride(...)` remains the same task-insensitive full estimator with the canonical observation discrepancy backend. |
| current normative layer | Claim boundary | Manuscript-level biological process language is supported only after cohort-level recurrence/support/dispersion over fitted patient-level `A/d/e`, not from a single-patient observation diagnostic or open tendency alone. |
| current normative layer | State/domain separation | Canonical docs and interfaces treat domain as an observation-layer stratification variable and do not encode domain identity into the shared state basis. |
| current normative layer | State-geometry validity | Canonical fitting validates finite, nonnegative, symmetric shared-state `C_raw/C_norm`, diagonal `0`, and a positive finite off-diagonal `s_C` scale before using geometry in the objective. |
| current normative layer | Deferred-estimator honesty | Canonical patient-relation and recurrence namespaces may return reserved or deferred states, but they must not fabricate patient-level relations or recurrence families that the implementation has not actually estimated. |
| current normative layer | Audit semantics | Missing biological support, numerical failure, and deferred estimator status remain explicit rather than being collapsed into apparently valid outputs. |
| future execution layer | Consensus recurrence recovery | In the base synthetic consensus-recovery suite, recovered cohort consensus `A/d/e` and patient-level dispersion match the planted consensus/dispersion within thresholds set by the objective-formula freeze. |
| future execution layer | Recurrence ablation sensitivity | Removing recurrence degrades consensus recovery or dispersion stability relative to the full model on the same synthetic suite; exact thresholds remain deferred until objective formula freeze. |
| future execution layer | Core ablation sensitivity | Removing recurrence, geometry, or consistency degrades the matching native recovery metric relative to the full fit on the same synthetic suite; exact geometry and consistency thresholds remain deferred until objective formula freeze. |
| future execution layer | Shuffled controls | Under patient-timepoint or recurrence shuffling, consensus recovery and stability metrics degrade toward weak-control behavior rather than remaining near the full-model regime. |
| future execution layer | Diagonal-collapse resistance | In simulations where planted off-diagonal mass fraction is at least 0.30, recovered off-diagonal mass fraction remains at least 0.15 and does not collapse almost entirely onto the diagonal. |
| future execution layer | Emergence restraint | In no-emergence simulations, median recovered total emergence mass ratio stays <= 0.10; in emergence-present simulations, recovered emergence magnitude tracks planted emergence with median correlation >= 0.80. |

## 8. Deferred Execution Surfaces

Task datasets remain important, but they remain deferred execution surfaces:

- task-specific validations are not the method contract itself,
- task rewrites and task papers should later inherit this validation logic,
- no success criterion in this file depends on visually compelling biology or on
  one benchmark alone.
