# Task A Block 3 Change Playbook

## Trigger

Use this playbook when a change touches Block 3 authority wording, the
internal `python -m tasks.task_A.block3` CLI, Task A Block 3 docs, or any Task A
result-packet surface that mentions Block 3.

## Read First

- [`../../../tasks/task_A/AGENTS.md`](../../../tasks/task_A/AGENTS.md)
- [`../../../docs/task_A/spec.md`](../../../docs/task_A/spec.md)
- [`../../../docs/task_A/block3/scientific_contract.md`](../../../docs/task_A/block3/scientific_contract.md)
- [`../../../docs/task_A/block3/refactor_contract_map.md`](../../../docs/task_A/block3/refactor_contract_map.md)
- [`../../../tasks/task_A/README.md`](../../../tasks/task_A/README.md)
- [`../../../tasks/task_A/block3_execution_runbook.md`](../../../tasks/task_A/block3_execution_runbook.md)

## Allowed Commands

- `rg` and `sed` against `tasks/task_A/`, `docs/`, and `tests/`
- targeted `pytest -q tests/test_task_a_block3_contract_migration.py`
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
- Live Block 3 routing is Stage0-only and cost-only. Do not reintroduce
  Block1/2 manifest execution, motif-probe relation support, or diagnostic
  matrix routes as live paths.
- `3C-1`, `3C-2`, and `3C-3` use the shared multi-FOV hidden
  `(A_p, d_p, e_p)` truth and full refits for both reference and ablation arms.

## Stop And Ask The User

- If the change would alter the frozen Block 3 scientific contract
- If you need a fresh scientific rerun instead of a targeted code/doc change
- If preserved proxy-history packets would need rewriting
- If a change would introduce a new public CLI or reframe Task A as canonical
  STRIDE
