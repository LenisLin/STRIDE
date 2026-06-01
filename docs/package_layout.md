# Package Layout

This file records the live repository structure for STRIDE.

| Path | Status | Role |
|---|---|---|
| `docs/` | active | canonical project docs plus maintenance docs |
| `src/stride/` | active core package | canonical task-insensitive core package and live first-pass implementation surface |
| `tasks/` | active | task-specific workflows, operational docs, and benchmark code |
| `tasks/_shared/benchmarks/` | active | shared benchmark helpers |
| `tasks/task_A/` | active task surface | Task A Block 0/1 workflows and internal Block 3 rebuild package |

## Placement Rules

- Put canonical project docs in `docs/`.
- Put task-specific runtime and operational materials under `tasks/` when they
  are task-owned.
- Keep task-scoped background notes in `docs/` only when they are explicitly
  labeled as task-specific and are not presented as the core package spec.
- Keep generated or run-specific outputs under the relevant task output
  directory with explicit manifests.

## Current Repository Reading

- `src/stride/` is the reusable core package surface.
- `src/stride/io/` contains the implemented v1 package I/O surface for raw
  AnnData assembly and h5ad persistence.
- `tasks/task_A/` owns the current Task A runtime, contracts, and internal
  Block 3 rebuild package.
- `tasks/_shared/benchmarks/` contains shared benchmark helpers used by task
  code.
- `docs/` contains the canonical method contract, task specs, and maintenance
  docs used by agents and developers.
- `docs/package_api_design.md` records the user package API architecture,
  current `stride.io` v1 surface, and cleanup review workflow.
