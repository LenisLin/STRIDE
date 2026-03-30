# Migration Status

This file records the current repository state during the STRIDE documentation
and package-structure migration.

## Current Facts

- STRIDE is the live project and scientific identity.
- The Python distribution name remains `slotar` during the current transition.
- `src/stride/` is the canonical task-insensitive core package and live
  first-pass implementation surface.
- `src/slotar/` remains the current implementation and compatibility namespace.
- `tasks/` owns task-specific workflows, benchmark helpers, and operational
  task documentation.
- `history/docs/` preserves archived material only; historical code is no
  longer kept in the repo working tree.
- `src/history/` is not a live package surface and must not be presented as the
  active location of archived code.
- Current uncertainty means bootstrap/sampling-variance uncertainty over
  realized bridge outputs.

## Stable Design Reference Now

The active STRIDE story is stable at the design level:

- the primary scientific object is `(T_p, e_p)` with `T_p = [A_p | d_p]`,
- `A_p` is row-substochastic rather than a pure conditional kernel,
- burden and composition remain separate scales,
- the canonical observation layer is domain-stratified bag-of-FOV comparison,
- domain remains an observation-layer stratum rather than state identity,
- Task A remains a bounded proxy-validation task rather than a redefinition of
  the global STRIDE object.

The core method docs remain:

- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/overall_validation_plan.md`
- `docs/constraints.md`

## Structure Boundary

The active structure should be read as follows:

- `src/stride/` is the target reusable-core architecture.
- `src/stride/` already hosts the live narrow first-pass fit path.
- `src/slotar/` is a transitional layer that still hosts compatibility wrappers
  and migration-facing entrypoints.
- `tasks/` is the home of task-specific logic, runtime entrypoints, and
  benchmark orchestration.
- `tasks/_shared/benchmarks/` and task-local benchmark directories replace the
  old top-level `benchmarks/` surface.
- `history/docs/` preserves historical documents, and archived code now lives
  outside the live installable source tree.

## Migration-Only Surfaces Still Live

Important migration-only layers are still present:

- backend-only numerical implementation under `slotar.backends.ot_sinkhorn`,
- compatibility shims under `slotar.compat` and temporary top-level shim paths
  such as `slotar.representation`, `slotar.uot`, `slotar.contracts`,
  `slotar.uq`, `slotar.utils`, `slotar.io.bridge`, and `slotar.drift`,
- task/runtime labels that still use historical arm or transport-era naming,
- task outputs that are still pair/table oriented rather than centered on final
  patient-level `(A_p, d_p, e_p)` exports.

These surfaces remain documentation-relevant as compatibility residue. They are
not the canonical architecture target.

## Explicitly Deferred

The following work remains deferred to later source-migration passes:

- broader standalone observation-to-patient bridge estimation behind
  `bridge_observation_matches(...)` beyond the current narrow two-group
  uniform-mass path,
- implementation of a non-deferred recurrence estimator behind
  `estimate_recurrence(...)`,
- migration of remaining working implementations from transitional `slotar`
  surfaces into mature `stride` surfaces,
- task-pipeline rewrites,
- test rewrites,
- retirement of legacy observation metrics and compatibility shims once the
  source migration is complete.

## Non-Blocking Open Points

- The narrow first-pass observation-to-patient bridge is live in
  `stride.api.fit` and `stride.workflows.fit_stride`; broader standalone bridge
  surfaces remain deferred.
- Namespace direction and estimator completeness remain separate questions:
  `stride` may be the target architecture even when some estimators are still
  deferred.
- Some current implementation terms still use `prototype` language where the
  design uses the broader term "shared `K`-state basis".
- The longitudinal validator still accepts implementation-era aliases; that
  alias handling is compatibility behavior, not the final naming endpoint.

## Source-of-Truth Order

For repository framing and structure, use:

1. `README.md`
2. `docs/index.md`
3. `docs/architecture.md`
4. `docs/package_layout.md`
5. `docs/method_overview.md`

For method design, then use:

1. `docs/decisions.md`
2. `docs/api_specs.md`
3. `docs/data_contracts.md`
4. `docs/overall_validation_plan.md`
5. `docs/constraints.md`

For Task A scope and operational boundaries, use:

1. `docs/task_A_spec.md`
2. `tasks/task_A/README.md`

`docs/dev_log.md` is repo-memory only and not a design reference.
