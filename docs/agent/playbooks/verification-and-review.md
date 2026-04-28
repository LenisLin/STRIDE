# Verification And Review Playbook

## Trigger

Use this playbook before claiming work is complete, especially when a patch
touches runbooks, manifests, protocol docs, or Task A operational surfaces.

## Read First

- [`../../../AGENTS.md`](../../../AGENTS.md)
- [`../../../tasks/task_A/AGENTS.md`](../../../tasks/task_A/AGENTS.md) when the
  change touches Task A

## Allowed Commands

- `pytest -q <target>`
- `python scripts/dev/generate_readme.py --check`
- `git status --short`
- `rg` and `sed` for final spot checks

## Required Verification

- Re-run the exact targeted tests that prove the change.
- Re-run `pytest -q tests/test_agent_protocol.py` for any agent-protocol patch.
- Re-run `python scripts/dev/generate_readme.py --check` when README routing or
  template content changed.
- Confirm the final diff only touches intended files.

## Required Updates

- Sync README/template output if the README entry surface changed.
- Sync task-local runbooks and protocol docs together when behavior-level
  routing changed.
- Report any remaining unverified areas explicitly.

## Stop And Ask The User

- If verification requires a long rerun or unsupported environment
- If targeted tests fail for unrelated pre-existing reasons
- If the final diff reveals unintended scientific-contract edits
