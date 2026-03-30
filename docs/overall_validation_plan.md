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
6. does it recover recurring patient-level relation families at the cohort
   level once the canonical bridge and recurrence estimators are fully live,
7. does it keep emergence explicit when post-side novelty is present,
8. does it fail in the expected direction under ablations and shuffled controls.

## 2. Validation Staging And Claim Boundary

The validation plan has two distinct stages. The docs must keep them separate.

### 2.1 Current normative validation layer

The following validation semantics are normative now:

- contract validity for observation, patient, and recurrence objects,
- the current narrow two-group uniform observation-to-patient bridge path,
- explicit separation between observation-layer diagnostics and patient-level
  outputs,
- explicit row-substochastic `A_p` semantics,
- explicit burden/composition scale separation,
- explicit domain-stratified bag-of-FOV observation semantics,
- explicit no-double-counting treatment of domain versus state identity,
- explicit bootstrap/sampling-variance uncertainty over realized bridge
  outputs,
- explicit auditability and no silent fallback,
- honesty about deferred estimator state where canonical namespaces exist but
  final estimators are not yet complete.

This means the current docs may require:

- valid patient-level arrays when a patient relation is emitted,
- valid recurrence containers or explicit deferred recurrence status,
- no promotion of legacy transport-era fields directly into canonical
  patient-level claims without an explicit bridge.

### 2.2 Deferred execution-stage validation layer

The following validation targets remain method-level requirements, but they are
execution-stage targets that apply once the canonical bridge estimator and
non-deferred recurrence estimator are fully implemented:

- synthetic recovery of planted patient-level relation structure,
- recovery of recurrence families from patient-level relations,
- numeric performance thresholds such as ARI or cosine-similarity targets,
- ablation sensitivities measured on end-to-end synthetic pipelines.

These targets remain the intended evidence standard. They are not grounds for
pretending those estimators are already complete.

## 3. Contract-Level Invariants

Every validation scenario should first check contract-level invariants:

- valid patient-level outputs have shapes `[K, K]`, `[K]`, and `[K]`,
- valid outputs are finite and nonnegative,
- `A_p` satisfies `sum_j A_{p,ij} + d_{p,i} = 1`,
- any documented conditional kernel is derived from `A_p` and `d_p` rather than
  substituted for them,
- shared-basis consistency is preserved across observations, patients, and
  cohort summaries,
- `mu_p^-`, `mu_p^+`, `m_p^(d)`, and `m_p^(e)` remain burden-scale quantities,
  while `q_p^-` and `q_p^+` remain derived compositions,
- current uncertainty outputs, when present, summarize bootstrap/sampling
  variance over realized bridge outputs rather than an older hurdle or
  measurement-error model,
- observation-layer outputs remain distinguishable from patient-level solver
  outputs and remain documented as domain-stratified bag-of-FOV measures in
  composition space,
- domain is not encoded into state identity and then reused as a conditioning
  variable,
- failures remain explicit rather than silently coerced,
- deferred canonical estimator namespaces return explicit deferred states rather
  than fabricated end-to-end results.

## 4. Deferred Synthetic Recovery Suite

The synthetic recovery suite is the intended execution-stage validation surface
once the canonical end-to-end estimators are fully implemented.

### 4.1 Recovery of recurring relation families

Design:

- simulate patients from a small number of planted relation families,
- give each family a distinct `A_p`, `d_p`, and `e_p` pattern,
- vary within-family noise while keeping family identities recoverable,
- retain multiple FOVs per patient so the bridge remains active.

Required behavior:

- patient-level outputs recover the planted relation structure,
- recurrence analysis recovers the planted family structure,
- support counts reflect the true patient family allocations.

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
- bridge audits reflect the coverage differences explicitly,
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
- cohort recurrence can distinguish families with different emergence profiles.

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

### 5.1 No recurrence layer

Remove or disable the recurrence layer while keeping patient-level objects.

Expected result:

- patient-level objects may still exist,
- recovery of recurring relation families should weaken materially,
- cohort summaries should become less stable and less discriminative.

### 5.2 No emergence bridge

Remove or disable the explicit emergence bridge so post-side novelty cannot
enter `e_p` directly.

Expected result:

- emergence-present simulations should show worse recovery,
- novelty should be misallocated into `A_p`, depletion, or residual diagnostics,
- cohort family separation should degrade when families differ mainly by
  emergence.

### 5.3 Transport-first or closed-match baseline

Compare the remodeling-first design to an older transport-first or closed-match
framing that treats the observation-layer comparison object as primary and
downplays the open relation.

Expected result:

- the older framing should underperform when true emergence or depletion is
  present,
- it should be more vulnerable to forced diagonalization or forced one-to-one
  matching behavior,
- it should provide weaker cohort-family recovery when recurrence truly acts on
  patient-level relations.

### 5.4 Histogram-collapse or state-domain leakage control

Replace domain-stratified bag-of-FOV comparison with either one pooled
histogram or a state basis that already encodes domain identity.

Expected result:

- patient-level structure should become less interpretable or less stable,
- recovered effects should become more sensitive to coverage artifacts,
- the failure should be visible as a contract violation or a degraded recovery
  surface rather than as a silent success.

### 5.5 Shuffled controls

Use patient-timepoint, recurrence, or state-label shuffling as weak controls.

Expected result:

- recovery of planted patient-level relations should collapse,
- recurrence families should become unstable or non-recoverable,
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
| current normative layer | Contract validity | Any emitted patient-level object has `A_p`, `d_p`, and `e_p` with correct shapes and nonnegative finite values; invalid payloads fail explicitly rather than silently coercing values. |
| current normative layer | `A_p` semantics | Any emitted patient relation obeys row-substochastic semantics, and docs do not substitute a pure conditional kernel for canonical `A_p`. |
| current normative layer | Burden/composition honesty | `mu_p^-`, `mu_p^+`, `m_p^(d)`, and `m_p^(e)` remain burden-scale quantities, while normalized compositions remain derived views; burden claims are not documented as robust when comparability is weak. |
| current normative layer | Observation-layer boundary | Observation-layer outputs remain distinguishable from patient-level outputs and remain documented as domain-stratified bag-of-FOV empirical measures rather than as the primary biological object. |
| current normative layer | State/domain separation | Canonical docs and interfaces treat domain as an observation-layer stratification variable and do not encode domain identity into the shared state basis. |
| current normative layer | Deferred-estimator honesty | Canonical bridge and recurrence namespaces may return reserved or deferred states, but they must not fabricate patient-level relations or recurrence families that the implementation has not actually estimated. |
| current normative layer | Audit semantics | Missing biological support, numerical failure, and deferred estimator status remain explicit rather than being collapsed into apparently valid outputs. |
| future execution layer | Family recovery | In the base synthetic family-recovery suite, patient family assignments achieve adjusted Rand index >= 0.80 and recovered family templates achieve median cosine similarity >= 0.80 against planted templates. |
| future execution layer | Recurrence ablation sensitivity | Removing recurrence lowers family assignment recovery by at least 0.15 ARI relative to the full model on the same synthetic suite. |
| future execution layer | Emergence ablation sensitivity | Removing the explicit emergence bridge increases emergence-related recovery error by at least 20% on emergence-present simulations. |
| future execution layer | Shuffled controls | Under patient-timepoint or recurrence shuffling, family assignment recovery falls to ARI <= 0.20 and stability metrics do not remain near the full-model regime. |
| future execution layer | Diagonal-collapse resistance | In simulations where planted off-diagonal mass fraction is at least 0.30, recovered off-diagonal mass fraction remains at least 0.15 and does not collapse almost entirely onto the diagonal. |
| future execution layer | Emergence restraint | In no-emergence simulations, median recovered total emergence mass ratio stays <= 0.10; in emergence-present simulations, recovered emergence magnitude tracks planted emergence with median correlation >= 0.80. |

## 8. Deferred Execution Surfaces

Task datasets remain important, but they remain deferred execution surfaces:

- task-specific validations are not the method contract itself,
- task rewrites and task papers should later inherit this validation logic,
- no success criterion in this file depends on visually compelling biology or on
  one benchmark alone.
