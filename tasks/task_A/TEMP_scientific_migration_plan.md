# Task A Temporary Scientific Migration Plan

Status: temporary control note.

This file records the working plan for the Task A destructive refactor and
scientific migration. It is not a source of scientific authority. If this file
conflicts with higher-priority contracts, the higher-priority contract wins.

## Authority And Scope

Authority order for Task A remains:

1. `docs/task_A/spec.md`
2. `docs/task_A/block3/scientific_contract.md`
3. `tasks/task_A/README.md`
4. `tasks/task_A/block3_execution_runbook.md`
5. `tasks/task_A/contracts/*`

This note is a coordination surface for the main control conversation. It may
guide implementation windows, but it must not redefine the scientific contract
without an explicit contract migration.

Frozen or preserved surfaces:

- Do not edit `docs/task_A/spec.md` unless a true contract migration is
  explicitly approved.
- Do not edit `docs/task_A/block3/scientific_contract.md` unless a true Block 3
  contract migration is explicitly approved.
- Do not rewrite `tasks/task_A/result_packets/` without explicit approval.
- Do not run full Task A validation from this plan without explicit approval.

## Approved Refactor Principles

- Treat Task A as a bounded validation surface for full STRIDE, not as a proxy
  method definition.
- Preserve the full STRIDE estimator implementation and full STRIDE core tests.
- Preserve Task A descriptive code and descriptive tests unless a later stage
  identifies a direct conflict with the live contract.
- Rebuild Task A in explicit scientific stages:
  `descriptive`, `block0`, `block1`, `block3a`, `block3b-1`,
  `block3b-2`, `block3c-1`, `block3c-2`, and `block3c-3`.
- Keep Block 3B-1 and Block 3B-2 in separate implementation files.
- Keep Block 3C implementation files named by scientific meaning when
  practical, with CLI or registry mapping to numbered subexperiments.
- Avoid legacy compatibility shells, retired public `run_block3` or
  `review_block3` restoration, and historical-result promotion.
- Enforce implementation size boundaries:
  scripts should stay under 1200 lines, and functions should stay under
  200 lines.
- Prefer narrow targeted tests before any long validation run.
- Full real-data runs require explicit confirmation.

## Block 3C Semantic Migration Proposal

Current frozen contract:

- `3C-1`: `recurrence_ablation`
- `3C-2`: `geometry_ablation`
- `3C-3`: `consistency_ablation`

Proposed live order after migration:

- `3C-1`: `consistency_ablation`
- `3C-2`: `geometry_ablation`
- `3C-3`: `recurrence_ablation`

Rationale:

- `3C-1 consistency_ablation` tests whether patient/evidence consistency
  contributes to recovery before testing higher-level structure.
- `3C-2 geometry_ablation` tests the locality or geometry prior.
- `3C-3 recurrence_ablation` tests cohort-level recurrence or shared
  structure as the final within-STRIDE module-necessity ablation.

Migration consequence:

- This is not a cosmetic rename.
- It requires synchronized updates to frozen Task A docs, Block 3 registry,
  bundle schema, review schema, tests, output naming, and manuscript-facing
  labels.
- Until that migration is approved and completed, implementation windows must
  treat the old mapping as the active frozen contract.

## Phase Order

Each phase follows the same gate sequence:

1. Confirm scientific question, semantics, and implementation feasibility.
2. Delete or retire only phase-local obsolete code, tests, and results after
   the destructive boundary is explicitly approved.
3. Plan the new directory, script, function, and artifact layout.
4. Specify each script and function before implementation:
   scientific question, required inputs, outputs, key variables, and cited
   contract documents.
5. Implement the phase with no oversized scripts or functions and no redundant
   historical compatibility layer.
6. Run a small real-data validation.
7. Ask for explicit approval before a full real-data run.

## Phase Specifications

### Descriptive

Scientific question:

- What is the Task A cohort, ROI, transition, annotation, and data-quality
  surface before modeling?

Primary role:

- Characterize the real-data cohort and define scope.
- Provide descriptive summaries only.
- Do not tune benchmark thresholds or scientific acceptance criteria.

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/spec.md`
- `tasks/task_A/README.md`

Preservation policy:

- Existing descriptive code and tests are eligible for preservation.
- Remove only redundant or stale wrappers identified during the phase review.

### Block 0

Scientific question:

- Does the Task A real-data surface support a STRIDE-native locality or
  data-suitability gate before downstream validation?

Primary role:

- Run the locality/data-suitability gate.
- Write a pass/fail bundle that downstream phases can reference.

Reference documents:

- `docs/task_A/spec.md`
- `tasks/task_A/README.md`
- `tasks/task_A/AGENTS.md`

Implementation expectation:

- Keep a first-class Block 0 CLI.
- Keep output contracts explicit and machine-checkable.

### Block 1

Scientific question:

- What biological discovery signal does full STRIDE report on the real Task A
  data under the approved Task A scope?

Primary role:

- Run real-data discovery summaries.
- Separate descriptive discovery from benchmark validation.

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/spec.md`
- `tasks/task_A/README.md`

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/spec.md`
- `tasks/task_A/README.md`

### Block 3A

Scientific question:

- Does the full STRIDE reference estimator recover native Task A semi-synthetic
  quantities under the approved semi-synthetic design?

Primary role:

- Establish internal full-STRIDE recovery behavior before baseline contrasts.

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

### Block 3B-1

Scientific question:

- How does recovery vary over relation strength at fixed
  `open_mass_scale=1.0`?

Primary role:

- Primary relation-strength benchmark.
- Keep sidecar open-mass sensitivity isolated as diagnostic-only artifact.

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

Implementation expectation:

- Use a dedicated file separate from Block 3B-2.
- The sidecar grid `open_mass_scale=[0.1, 0.25, 0.5]` must not alter the
  primary/frozen benchmark.

### Block 3B-2

Scientific question:

- How does open-transition recovery vary over the formal open-mass grid?

Primary role:

- Formal open-mass benchmark for `d` and `e`.

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

Implementation expectation:

- Use a dedicated file separate from Block 3B-1.
- Formal grid must be exactly `[0.0, 0.1, ..., 1.0]`.
- Formal grid must not include `0.25`.

### Block 3C-1

Proposed scientific question after migration:

- What is lost when patient/evidence consistency is removed from the full
  STRIDE objective and the model is refit?

Proposed method:

- `consistency_ablation`

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

Migration note:

- This phase currently conflicts with the frozen numbering, where `3C-1` is
  `recurrence_ablation`. Do not implement the new numbering until the contract
  migration is approved.

### Block 3C-2

Scientific question:

- What is lost when geometry or locality regularization is removed from the
  full STRIDE objective and the model is refit?

Method:

- `geometry_ablation`

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

### Block 3C-3

Proposed scientific question after migration:

- What is lost when cohort recurrence or shared structure is removed from the
  full STRIDE objective and the model is refit?

Proposed method:

- `recurrence_ablation`

Reference documents:

- `docs/task_A/spec.md`
- `docs/task_A/block3/scientific_contract.md`
- `tasks/task_A/block3_execution_runbook.md`

Migration note:

- This phase currently conflicts with the frozen numbering, where `3C-3` is
  `consistency_ablation`. Do not implement the new numbering until the contract
  migration is approved.

## Cleanup Boundaries

Eligible for phase-local destructive rebuild after explicit confirmation:

- Stage-local Task A block implementations under `tasks/task_A/block*`.
- Stage-local Task A workflow wrappers.
- Task A non-descriptive tests replaced by the new phase-specific tests.
- Temporary outputs under approved Task A output directories.

Not eligible without separate explicit confirmation:

- `tasks/task_A/result_packets/`
- `/mnt/NAS_21T/ProjectResult/STRIDE/task_A`
- frozen Task A docs
- full STRIDE core implementation
- full STRIDE core tests

## Verification Policy

Minimum checks before claiming a phase is complete:

- Static read of the relevant contract documents.
- Targeted unit or integration tests for the phase.
- Small real-data validation where the phase requires real inputs.
- No Task A full pipeline run unless explicitly approved.

Full validation commands and output directories should be proposed but not
executed from this note.

## Process Log

Use this section to record control decisions before implementation windows act.

| Date | Decision | Scope | Follow-up |
| --- | --- | --- | --- |
| 2026-04-30 | Created temporary migration plan. | Task A control surface only. | Confirm destructive boundary and 3C contract migration before implementation. |
