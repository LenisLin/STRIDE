# Doc And Contract Sync Playbook

## Trigger

Use this playbook for doc-only changes, README routing updates, protocol-file
changes, or any patch whose primary purpose is to keep operational docs aligned
with higher-order contracts.

## Read First

- [`../../../AGENTS.md`](../../../AGENTS.md)
- [`../../../docs/index.md`](../../../docs/index.md)
- the highest-priority contract that governs the changed topic

## Allowed Commands

- `rg`, `sed`, and `git status --short`
- `pytest -q tests/test_agent_protocol.py`
- `python scripts/dev/generate_readme.py --check`

## Required Verification

- Run `pytest -q tests/test_agent_protocol.py` when protocol docs, playbooks,
  templates, or agent entrypoints changed.
- Run `python scripts/dev/generate_readme.py --check` when `README.md` or its
  template source changed.

## Required Updates

- Keep links and routing statements consistent across `AGENTS.md`,
  `docs/agent/README.md`, `docs/index.md`, and `README.md`.
- Do not restate or weaken the source-of-truth hierarchy.
- If a task-local doc changes, update the nearest task-local agent protocol if
  the routing implications changed.

## Stop And Ask The User

- If the doc change requires reopening a scientific decision
- If you find conflicting contract wording that cannot be resolved by routing
- If the requested doc change would silently override higher-priority docs
