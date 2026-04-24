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

The purpose of the full method is to recover patient-level remodeling
structure, explicit open-channel behavior, and cohort-level common/recurrent
structure from partial, multi-FOV spatial observations.

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
- `d_p in R_+^K` is the pre-side depletion component.
- `e_p in R_+^K` is the post-side emergence component.
- `A_p` is row-substochastic, with `sum_j A_{p,ij} + d_{p,i} = 1`.
- `A_p`, not a normalized conditional kernel, is the canonical patient object.

`A_p`, `d_p`, and `e_p` are model objects inferred under partial observation.
They are not direct proof of literal physical transport, true disappearance, or
true neogenesis.

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

At minimum, that cohort layer must be able to represent:

- recurrence or family templates on the same shared axis,
- support counts and patient membership,
- within-family dispersion or stability,
- optional low-dimensional cohort embeddings or coordinates,
- explicit fit/deferred status.

The cohort object is real modeled structure. It is not merely a descriptive
summary written after patient-level estimation.

## 4. What Cohort-Level Common Structure Means

In the frozen full-method story, cohort-level common structure means:

- recurrent relation families or templates over `(A_p, d_p, e_p)`,
- a shared coordinate or recurrence space in which patients can be compared,
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
2. A patient bridge or patient-assembly layer that produces `(A_p, d_p, e_p)`.
3. An explicit open-relation treatment in which depletion and emergence remain
   modeled objects rather than forced closure residuals.
4. A cohort-level common-structure layer that operates on patient relations and
   encourages recurrent/shared structure across patients.
5. Structural regularization and explicit audit/failure handling that keep the
   outputs scientifically honest.

The exact final numerical joint objective, weighting scheme, and optimizer are
not yet frozen in full detail. That lack of final numeric closure does not
change the method definition above: full STRIDE is still the joint patient plus
cohort framework with explicit open channels and explicit regularization.

## 6. Conceptual Loss and Regularization Roles

The full design requires the following conceptual loss/regularization families:

- observation discrepancy or comparison terms at the FOV/ROI layer,
- patient-level consistency terms connecting observation evidence to
  `(A_p, d_p, e_p)`,
- open-channel control terms that preserve explicit depletion/emergence instead
  of forcing closed matching,
- cohort-level recurrence/shared-structure terms that induce reusable common
  structure across patients,
- regularization and audit terms that keep failures, weak support, and
  implausible behavior explicit.

This freeze is intentionally conceptual rather than fully numeric. A later
implementation pass may refine exact formulas, but it must preserve these roles
and must not drop the cohort/common-structure layer from the method definition.

## 7. Expected Full-STRIDE Outputs

Full STRIDE is expected to produce:

- patient-level `A_p`, `d_p`, `e_p`, and `T_p = [A_p | d_p]`,
- patient-level burden/composition auxiliaries,
- patient-level audits, fit-status fields, and uncertainty summaries,
- cohort-level recurrence/common-structure outputs such as family templates,
  support counts, dispersion summaries, and embeddings,
- downstream task-ready summaries derived from the patient and cohort objects.

Task summaries may focus on selected views of these outputs, but they do not
replace the underlying patient and cohort objects.

## 8. Current Live Approximation Status

The current live implementation is narrower than the full design frozen here.

The live repo currently has:

- a tissue-agnostic shared-state construction route,
- a domain-stratified bag-of-FOV observation layer with `mass_mode="uniform"`,
- a narrow realized patient bridge for supported patients,
- bootstrap uncertainty over realized bridge outputs,
- deferred canonical cohort recurrence estimation.

The live bridge remains limited to the current supported case:

- exactly two ordered groups per patient,
- uniform-mass patient inputs,
- explicit deferred status outside that narrow case.

Task A currently operationalizes this narrow implementation through an ordered
tissue-domain proxy and task-local summary layers. That current Task A stack is
an approximate/proxy execution history. It does not redefine full STRIDE.

## 9. Non-Claim Boundary

Full STRIDE as frozen here still does not imply:

- lineage tracing,
- exact physical transport truth,
- exact one-to-one FOV matching,
- unbiased whole-lesion reconstruction,
- direct proof that modeled emergence or depletion is free of sampling effects.

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
