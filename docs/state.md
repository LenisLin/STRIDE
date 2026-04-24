# Current Status

This file records the current live STRIDE repository state.

## Current Facts

- STRIDE is the live project and scientific identity.
- `src/stride/` is the canonical task-insensitive core package and live
  first-pass implementation surface.
- `tasks/` owns task-specific workflows, benchmark helpers, and operational
  task documentation.
- Current uncertainty means bootstrap/sampling-variance uncertainty over
  realized bridge outputs.
- `stride.api.fit.fit_stride(...)` is the canonical reusable fit surface.

## Stable Design Reference

The active STRIDE story is stable at the design level:

- the primary scientific object is `(T_p, e_p)` with `T_p = [A_p | d_p]`,
- `A_p` is row-substochastic and has an optional derived conditional-kernel
  view only when needed for exposition,
- burden and composition remain separate scales,
- the canonical observation layer is domain-stratified bag-of-FOV comparison,
- domain remains an observation-layer stratum.

The canonical full-method doc is:

- `docs/stride_design_freeze.md`

Supporting core method docs are:

- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/overall_validation_plan.md`
- `docs/constraints.md`

## Live Implementation Surface

- `fit_stride(...)` is the canonical reusable STRIDE path in `src/stride/`.
- The canonical path returns patient-level `A_p`, `d_p`, `e_p`, burden
  auxiliaries, grouped objective summaries, and a cohort-level recurrence or
  common-structure layer.
- The current recurrence implementation is a conservative first-pass template
  estimator with explicit deferred status when patient support is insufficient.
- The current realized patient bridge supports uniform-mass patient inputs with
  exactly two ordered groups.
- Task A Block 0-2 workflows call the canonical `fit_stride(...)` path.
- The Task A internal Block 3 rebuild package is the current method-validation
  implementation carrier for semisynthetic generator validation, baseline
  comparison, and ablation studies.

## Deferred Work

- Broader standalone observation-to-patient bridge estimation beyond the
  current two-group uniform-mass path.
- Richer recurrence estimators beyond the current conservative first-pass
  template family estimator.
- Task-pipeline expansions beyond the documented Task A operational surfaces.
- Additional public Block 3 workflow and packet integration surfaces.

## Non-Blocking Open Points

- The canonical first-pass patient-plus-cohort fit path is live in
  `stride.api.fit` and `stride.workflows.fit_stride`; broader standalone
  bridge surfaces remain deferred.
- Namespace direction and estimator completeness remain separate questions:
  `stride` is the target architecture even when some estimators are still
  deferred.
- Some current implementation terms still use `prototype` language where the
  design uses the broader term "shared `K`-state basis".
- The longitudinal validator still accepts implementation-era aliases for input
  tolerance.

## Source-of-Truth Order

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, and `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A_rewiring_plan.md`
5. `docs/task_A_spec.md`, `docs/task_A_result.md`, and
   `tasks/task_A/README.md` for the current Task A task layer
6. Task-local operational docs under `tasks/`

`docs/dev_log.md` is repo-memory only and not a design reference.
