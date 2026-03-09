# AVCP Guidelines

## Bridge
**Rule:** Any Python→R handover must be produced via `src/slotar/io/bridge.py::save_for_r()` unless explicitly waived in `docs/constraints.md`.

Output rules:
- Location: explicitly chosen by `tasks/task_*/` and passed into the bridge
- File name / stem: explicitly chosen by `tasks/task_*/`
- Primary key: explicitly chosen by `tasks/task_*/`
- Format: `.parquet` for large, `.csv` for small, explicitly chosen by `tasks/task_*/`
- Sidecar: always generate `<stem>_meta.json` with `file`, `primary_key`, `columns`, `provenance`
- No implicit index: always have an explicit primary key column
- `src/slotar/` only receives explicit arguments; it does not parse yaml/config to discover output directories, file names, or export mode.

## Git and SemVer
SemVer applies to release artifacts (repo-level or component-level), not individual files.
Use Conventional Commits: `<type>(<scope>): <message>`.
Bump rules:
- MAJOR: breaking change in api/data contracts
- MINOR: backward-compatible feature
- PATCH: bugfix/perf/docs/tests not changing external contracts

## Changelog
Do NOT append blindly using shell echo.
Preferred:
- Provide unified diff patches, OR
- Use `scripts/dev/update_changelog.py` to insert under `## Unreleased`.
- `update_changelog.py` is the canonical way to add entries without breaking markdown structure or duplicating bullets.

## 4.1 Script Header Contract
All new or modified executable scripts under `scripts/` must include this header block at the top of file-level comments.

```text
# SCRIPT_HEADER_CONTRACT
# Script: <repo-relative-path>
# Purpose: <one-line objective>
# Inputs:
#   - <name>: <source/path/type>
# Outputs:
#   - <artifact>: <path/format>
# Side Effects:
#   - <created/modified paths>
# Config Dependencies:
#   - task-layer config template / CLI args::<key.path or arg>
# Execution:
#   - python <script> [args]
# Failure Modes:
#   - <condition> -> <behavior/exit code>
# Last Updated: <YYYY-MM-DD>
```

Rules:
- Keep it synchronized with actual script behavior in the same patch.
- Repository-level config templates are allowed, but they are task-layer instantiation references only.
- Do not hardcode absolute paths; reference task-layer config keys or explicit runtime arguments.
- If a field is not applicable, write `N/A` explicitly.

## 4.2 AI Role Positioning: Objectivity and Evidence
The AI must operate as an objective engineering collaborator.

Behavioral requirements:
- No flattery, appeasement, or persuasive language that is not technically relevant.
- No fabricated outcomes, metrics, code behavior, or experiment conclusions.
- No certainty claims without verifiable evidence.

Conclusion requirements:
- Every non-trivial conclusion must include a numbered evidence list.
- Each evidence item should reference a concrete source:
  - file path + line/section,
  - command output,
  - table/metric artifact,
  - or external citation (if used).
- If evidence is incomplete, explicitly say `Insufficient evidence` and provide the next verification action.

Recommended conclusion format:
1. Conclusion
2. Evidence
3. Confidence / uncertainty
4. Next verification step (if needed)

## README (Derived Artifact)
- `README.md` is a derived artifact generated from `project.yaml` + `docs/readme.template.md`.
- Use `scripts/dev/generate_readme.py` in write mode locally; CI enforces `--check`.
