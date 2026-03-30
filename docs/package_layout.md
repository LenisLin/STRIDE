# Package Layout

This file records the current repository structure and the intended placement of
active versus archival documentation.

| Path | Status | Role |
|---|---|---|
| `docs/` | active | canonical project docs plus maintenance docs |
| `src/stride/` | active core package | canonical task-insensitive core package and live first-pass implementation surface |
| `src/slotar/` | transitional | compatibility namespace and migration bridge |
| `tasks/` | active | task-specific workflows, operational docs, and benchmark code |
| `tasks/_shared/benchmarks/` | active | shared benchmark helpers |
| `history/docs/` | archive | historical documents and legacy task notes |
| `src/history/` | not live | do not present as an active package surface |

## Placement Rules

- Put canonical project docs in `docs/`.
- Put task-specific runtime and operational materials under `tasks/` when they
  are task-owned.
- Keep task-scoped background notes in `docs/` only when they are explicitly
  labeled as task-specific and are not presented as the core package spec.
- Put historical documentation under `history/docs/`; keep historical code in a
  repo-external archive.
- Do not describe archived material as part of the current installable source
  tree.

## Current Repository Truth

- The live project identity is STRIDE.
- The package/distribution name remains `slotar` during the current transition.
- `stride` is the target reusable-core architecture and the live first-pass
  core-package surface.
- `slotar` remains the compatibility bridge to that target.
- Historical code is no longer kept in-tree.
