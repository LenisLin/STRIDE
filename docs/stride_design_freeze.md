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
- For each eligible source-side FOV/community-composition vector `v_source`,
  the patient relation induces
  `predicted_v_target = normalize(v_source @ A_p + e_p)`.
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

`L_local` is the fixed normalized combination of:

- observation data fit,
- open-channel sparsity/complexity regularization,
- geometry/locality prior.

`L_regularization` is the fixed normalized combination of:

- patient consistency,
- cohort recurrence.

The default `alpha` is `0.5`. `alpha` is the primary
local-versus-regularization hyperparameter. `alpha` sensitivity grids are
optional diagnostics and are recorded in provenance when run.

Component losses are normalized by baseline scales computed from the
deterministic identity-plus-small-open initialization on the same input.
Near-zero baseline scales use an epsilon floor, and floor usage is recorded in
provenance.

For patient `p` and resolved observation evidence block `b`, `L_obs` compares
the predicted target-side bag-of-FOV empirical measure induced by `(A_p, e_p)`
with the observed target-side bag-of-FOV empirical measure. The task layer
owns source/target/domain resolution and comparison-plan instantiation; the
core estimator receives resolved source/target observation evidence blocks.
After that resolution, domain is not a loss axis, state axis, relation axis, or
recurrence axis inside the core estimator.

`predicted_v_{p,b,f}^+ = normalize(v_{p,b,f}^- @ A_p + e_p)`

`L_obs_raw` is averaged over patients and resolved evidence blocks using
balanced averaging. The canonical v1 pairwise observation loss is:

`L_obs_pair_raw = D_obs^UOT-v1(predicted target-side bag-of-FOV, observed target-side bag-of-FOV; C_norm)`

where:

- `C_norm = C_raw / s_C`,
- `D_obs^UOT-v1` is a fixed, versioned, auditable canonical observation
  discrepancy operator,
- cost normalization, backend version, Sinkhorn/UOT status handling, and
  `D_pos/B_pos` diagnostics are recorded in provenance,
- `D_pos/B_pos` are observation-layer diagnostics. They are not biological
  `d/e` and are not independently weighted loss components.

Patient-level `A/d/e` are fitted model quantities under partial observation.
Manuscript-level biological process claims require cohort-level
recurrence/support/dispersion rather than a single-patient unmatched residual
or open tendency alone.

STRIDE v1 uses simple continuous differentiable component losses where
feasible. The assembled full objective is treated as a constrained non-convex
numerical objective. The live contract does not claim global convexity or a
global optimum. Local optimum risk is handled through deterministic
initialization, a fixed optimizer protocol, optimizer status/provenance, and
optional stability diagnostics.

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
term uses a torch-native differentiable canonical Sinkhorn/UOT-v1 operator:
Sinkhorn/UOT solves the observation discrepancy, while AdamW optimizes
`A_p`, `d_p`, `e_p`, and any necessary objective variables.

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

Geometry/locality is a soft prior over the full `A` operator. The current v1
raw candidate is the `A`-weighted normalized state-geometry cost.

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
- audit/provenance/status terms.

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
- compact provenance covering at least `alpha`, loss decomposition and
  normalization, optimizer framework/protocol/status, recurrence fit status,
  ablation mode, random seed, and failure or convergence reason,
- downstream task-ready summaries derived from the patient and cohort objects.

Task summaries may focus on selected views of these outputs, but they do not
replace the underlying patient and cohort objects.

## 8. Current Live Approximation Status

The live repo currently implements a narrower first-pass relation path and
remains an implementation target for the full objective contract described
above.

The current implementation surface includes:

- a tissue-agnostic shared-state construction route,
- a domain-stratified bag-of-FOV observation layer with `mass_mode="uniform"`,
- a narrow patient-relation construction path for supported patients,
- bootstrap uncertainty over fitted patient relation outputs,
- deferred canonical cohort consensus recurrence estimation.

The current patient-relation path supports a bounded first-pass configuration:

- exactly two ordered groups per patient,
- uniform-mass patient inputs,
- explicit deferred status for unsupported configurations.

Task A currently operationalizes this first-pass path as a bounded validation
surface while the full objective estimator remains the implementation target.

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
