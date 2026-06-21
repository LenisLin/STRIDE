# Architecture

This repository is in a documentation-first migration from older SLOTAR/UOT-era
surfaces toward a STRIDE-oriented core architecture.

## Active Boundary

- `src/stride/` is the canonical task-insensitive core package and live
  first-pass implementation surface.
- `tasks/` owns task-specific workflows, runtime configuration, benchmark
  helpers, and operational task documentation.
- `history/` is archival only and is not part of the live installable surface.

## `src/stride/`

The on-disk `stride` package is the active architectural direction for the
reusable core and now contains the live first-pass canonical implementation
surface. Its current top-level areas are:

- `stride.io`
- `stride.pp`
- `stride.tl`
- `stride.da`
- `stride.pl`

These namespaces document the implemented first-pass reusable core
decomposition. Their presence should not be read as a promise that every
function is production-stable public API.

The user-facing API layer is defined in `docs/package_api_design.md`.
`stride.io`, `stride.pp`, `stride.tl`, `stride.da`, and `stride.pl` are beta
package surfaces whose current scope is recorded in `docs/api_specs.md` and
`docs/state.md`.

## `tasks/`

Task-specific materials do not define the reusable core package. They belong in
`tasks/` or in explicitly labeled task docs.

- Cohort-specific logic, study design, reporting, and benchmark orchestration
  live under `tasks/`.
- Shared benchmark helpers now live under `tasks/_shared/benchmarks/`.
- Task-local benchmark code now lives with the relevant task, for example
  `tasks/task_A/benchmarks/`.
- The old top-level `benchmarks/` surface is no longer part of the active
  repository structure.

## `history/`

Archived materials are separated from live docs and live code:

- `history/docs/` contains archived historical documents.
- Historical code is archived outside the repo working tree.
- `src/history/` is not a live package surface and should not be referenced as
  the active home of archived code.
