# Task A Contracts

Task A keeps its contracts at the task layer so `src/stride/` stays task-insensitive.

- `design_freeze.py` is the machine-readable Step 2 freeze source for surface
  responsibilities, execution order, and artifact contracts.
- `stride_mapping.py` defines the code-level Task A to stable STRIDE mapping summaries.
- `artifact_contracts.md` freezes the file-level artifact schemas, readiness
  states, and non-goals for each major Task A output.
- `execution_graph.md` freezes the canonical execution order, hard
  prerequisites, and which stages are executable.
- `stage0_to_stride_mapping.md` freezes the required Stage 0 fields, ordered proxy rules, and adapter-side family slicing.
- `deferred_boundaries.md` records which Task A surfaces remain task-local or deferred after the Block 0 locality gate.
- [`../block3_execution_runbook.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/block3_execution_runbook.md)
  is the live operational checklist for the rewritten Block 3 execution path.
- The active Block 3 contract freezes lightweight generator validity, the
  `3A/3B/3C` benchmark sections, separated baseline-versus-ablation
  presentation, and generator reruns as the outer statistical unit.
- Block 0 freezes one STRIDE-native real-vs-null comparison:
  `TC-IM` versus `TC-IM_randomized_target`, where the null keeps anchor-patient
  `TC` observations fixed and reassigns the `IM` group from a different
  patient in the same exact `(n_TC, n_IM)` stratum under a fixed random seed.

These contracts describe Task A inputs, ordered tissue-domain proxy handling, block boundaries, and deferred surfaces. They do not redefine the STRIDE core method. Pre-Block 0 data suitability checks are allowed, but they must not be mistaken for Block 0 passage or primary evidence.
