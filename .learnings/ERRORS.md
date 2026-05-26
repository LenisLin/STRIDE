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
