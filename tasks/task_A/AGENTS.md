# Task A Agent Protocol

This file defines how agents should work inside `tasks/task_A/` without
changing the scientific authority chain.

## Authority Chain

For Task A, use this order:

1. `docs/task_A/spec.md`
2. `docs/task_A/block3/scientific_contract.md`
3. Stage docs under `docs/task_A/block3/`
4. `docs/task_A/block3/refactor_contract_map.md` for migration mapping only
5. `docs/task_A/result.md` and `tasks/task_A/README.md`
6. `tasks/task_A/block3_execution_runbook.md` and `tasks/task_A/contracts/*`

Packet-local Block 3 mirrors are downstream/non-authority only and must not be
promoted to scientific authority.

## Workflow Boundaries

- `prepare`: Step 1 Task A entrypoint; validates Stage 0 alignment and writes
  Task A mapping/dry-run manifests.
- `tasks.task_A.block0` (`execute` / `analyze`): executable Block 0 calibration surface for execution-cache and cache-derived
  analysis artifacts used as empirical `TC-IM` null calibration context.
  Preserved legacy Block 0 artifacts must not be treated as live contract
  authority.
- `block1_real_data_discovery`: live Block 1 identifier for the descriptive
  real-data discovery surface. Use `python -m tasks.task_A.block1 execute`
  and `python -m tasks.task_A.block1 analyze` for the live command surface.
- `python -m tasks.task_A.block3 <experiment_name>`: internal Block 3 CLI.
  The experiment name is semantic; numbered ids remain `subexperiment_id`
  metadata.
- `package_results`: packet-local packager for atlas/Block 0/1 outputs
  only; it has no Block 3 manifest bridge parameter.

## Resume And Rerun Rules

- Block 3 has no checkpoint/resume path. The public Block 3 runner remains
  absent; internal single-subexperiment execution is package-local.
- Block 3 live execution is Stage0-only and cost-only. Use Stage 0 h5ad plus
  Task A config through `python -m tasks.task_A.block3 <semantic_name>`.
- Run `generator_validation` first for manual sanity review, then run
  `a_benchmark`, `de_benchmark`, `subbag_consistency_ablation`,
  `geometry_ablation`, and `recurrence_ablation` if review permits.
- Do not route live Block 3 through Block 1 manifests, motif-probe support,
  or a diagnostic matrix.

## Review And Result-Packet Boundary

- Review packets remain packet-local.
- Block 0-owned proxy/history packet contents under `tasks/task_A/result_packets/`
  are retired and should not be used as live context.
- Do not relabel packet-local review calls as live scientific estimands.

## Required Sync When Touching Block 3

If a change touches Block 3 contracts, review surfaces, or result packets, sync
the nearest affected surfaces:

- `docs/task_A/spec.md` only when a true contract change is explicitly intended
- `docs/task_A/block3/scientific_contract.md` and stage docs when the live
  Block 3 contract changes
- `tasks/task_A/README.md` when routing, boundaries, or launch guidance change
- `tasks/task_A/block3_execution_runbook.md` when launch/verification steps
  change
- `tests/test_task_a_block3_deferred_surfaces.py`
- `tests/test_task_a_result_packet.py`
- `tests/test_task_a_design_freeze.py`
- `tests/test_agent_protocol.py` when protocol wording or routing changes

## Stop And Ask The User

- If a patch would change the Task A authority chain
- If a change requires a fresh scientific rerun rather than a narrow fix
- If preserved packet outputs would need rewriting
- If a change would blur the boundary between packet-local review and
  scientific authority
