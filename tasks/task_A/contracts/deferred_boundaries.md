# Task A Deferred Boundaries

## Core surfaces that remain deferred

- Standalone bridge-expert APIs remain deferred.
- Task A uses task-local Block 0 and null-family orchestration on top of the
  canonical `fit_stride(...)` surface.

## Task-local interpretation boundary

- Block 1 self-retention / depletion / remodeling summaries are task-local
  Task A evidence surfaces.
- Any Block 1 continuity wording in task-local outputs refers only to the
  frozen self-retention summary on the ordered tissue-domain surface. The
  canonical STRIDE recurrence/common-structure exports are carried as separate
  cohort-level context and must not be silently folded into those patient-level
  Block 1 summary names.
- Block 2 is a robustness-over-summaries surface with bounded open-channel
  interpretation.

## Execution boundary

- This pass allows pre-Block 0 data suitability checks, executable Block 0 STRIDE-native gate runs, and strict subset/demo sidecars.
- Prepare, Block 1, and Block 2 consume their declared prerequisites and keep
  Block 0 as the scientific gate for downstream Task A evidence.
- Block-local full-cohort Block 2 robustness runs are allowed once an
  evidence-ready Block 1 bundle exists, including resumable reruns against the
  same output directory.
- The Step 3 objective result packet covers atlas plus Block 0-2 surfaces.
- The on-disk `tasks/task_A/block3/` package may run internal Phase 3 execution
  from evidence-ready Block 1/2 inputs.
- STRIDE core redesign and new bridge algorithms require separate contracts.
