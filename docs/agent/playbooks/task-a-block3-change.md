# Task A Block 3 Change Playbook

## Trigger

Use this playbook when a change touches Block 3 authority wording, deferred
`run_block3` / `review_block3` wrappers, Task A Block 3 docs, or any Task A
result-packet surface that mentions Block 3.

## Read First

- [`../../../tasks/task_A/AGENTS.md`](../../../tasks/task_A/AGENTS.md)
- [`../../../docs/task_A_spec.md`](../../../docs/task_A_spec.md)
- [`../../../docs/task_A_block3_redesign_v1_1.md`](../../../docs/task_A_block3_redesign_v1_1.md)
- [`../../../tasks/task_A/README.md`](../../../tasks/task_A/README.md)
- [`../../../tasks/task_A/block3_execution_runbook.md`](../../../tasks/task_A/block3_execution_runbook.md)

## Allowed Commands

- `rg` and `sed` against `tasks/task_A/`, `docs/`, and `tests/`
- targeted `pytest -q tests/test_task_a_block3_deferred_surfaces.py`
- targeted `pytest -q tests/test_task_a_canonical_step3_review.py`
- targeted `pytest -q tests/test_task_a_result_packet.py`
- targeted `pytest -q tests/test_task_a_design_freeze.py`

## Required Verification

- Run the narrowest Task A Block 3 tests that prove the change.
- Run `pytest -q tests/test_agent_protocol.py` if protocol docs or templates
  changed as part of the same patch.
- If README template content changed, also run
  `python scripts/dev/generate_readme.py --check`.

## Required Updates

- Sync any affected Task A operational mirror in `tasks/task_A/README.md`.
- Sync `tasks/task_A/AGENTS.md` if the routing, review, or resume boundary
  changed.
- Sync result-packet or review-surface docs when file names, packet-local
  semantics, or authority wording changed.

## Stop And Ask The User

- If the change would alter the frozen Block 3 scientific contract
- If you need a fresh scientific rerun instead of a targeted code/doc change
- If preserved proxy-history packets would need rewriting
- If a change would introduce a new public CLI or reframe Task A as canonical
  STRIDE
