# Errors

Command failures and integration errors.

---

## [ERR-20260516-001] Shell Regex Command Substitution

**Logged**: 2026-05-16T10:42:29+08:00
**Priority**: low
**Status**: pending
**Area**: tooling

### Summary
An `rg` command failed because an unescaped backtick in the shell command was interpreted as command substitution.

### Details
When searching Markdown text that includes inline-code backticks, passing the pattern inside a shell command can trigger command substitution if the backticks are not quoted safely.

### Suggested Action
Use single quotes around regex patterns containing Markdown backticks, or split searches into simpler literal patterns.

### Metadata
- Source: error
- Related Files: docs/task_A/spec.md
- Tags: shell, rg, quoting

## [ERR-20260516-002] Direct Module CLI Missing src Import Path

**Logged**: 2026-05-16T10:42:29+08:00
**Priority**: medium
**Status**: resolved
**Area**: tooling

### Summary
`python -m tasks.task_A.block3 ... --dry-run` failed after the CLI imported the Block 3 registry because `src/` was not on `sys.path`.

### Details
The registry imports `stride.errors`, which is available under the repository `src/` tree during tests but was not visible during direct module execution from the repo root.

### Suggested Action
For package-local module CLIs that import `src` modules before dry-run exits, bootstrap the repo `src/` path before importing internal registry modules.

### Metadata
- Source: error
- Related Files: tasks/task_A/block3/__main__.py
- Tags: pythonpath, cli, dry-run

---

## [ERR-20260526-001] Git Index Write Blocked By Sandbox

**Logged**: 2026-05-26T22:18:57+08:00
**Priority**: low
**Status**: resolved
**Area**: tooling

### Summary
Creating requested per-file Git commits failed in the default sandbox because `.git/index.lock` could not be created.

### Error
```text
fatal: Unable to create '/home/lenislin/Experiment/projects/STRIDE/.git/index.lock': Read-only file system
```

### Context
The failing operation was a scripted sequence of `git add` and `git commit` calls. Re-running the same operation with escalated Git permissions succeeded.

### Suggested Fix
When Git index writes fail with a read-only filesystem error in this workspace, retry the same Git operation with explicit approval rather than changing working tree files.

### Metadata
- Reproducible: unknown
- Related Files: .git/index
- Tags: git, sandbox, commit

---

## [ERR-20260605-001] Full Pytest Collection Imports Retired Geometry Path

**Logged**: 2026-06-05T11:15:06+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
`env PYTHONPATH=src pytest -q` failed during collection because a Task A test imports `stride.geometry.state_geometry`, which is not present in the current source tree.

### Details
The failure occurs before the new `stride.pp` observation tests run. The current source tree contains `src/stride/pp/_geometry.py`, but no `src/stride/geometry/` package.

### Suggested Action
Resolve the Task A compatibility boundary separately: either restore/route the expected `stride.geometry` module path or update the Task A test/import surface to the current canonical geometry location.

### Metadata
- Source: error
- Related Files: tests/test_task_a_block3_generator_validation_profiles.py, src/stride/pp/_geometry.py
- Tags: pytest, import, task-a, geometry
