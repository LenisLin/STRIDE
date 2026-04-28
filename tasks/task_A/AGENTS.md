# Task A Agent Protocol

This file defines how agents should work inside `tasks/task_A/` without
changing the scientific authority chain.

## Authority Chain

For Task A, use this order:

1. `docs/task_A_spec.md`
2. `docs/task_A_block3_redesign_v1_1.md`
3. `tasks/task_A/README.md`
4. `tasks/task_A/block3_execution_runbook.md` and `tasks/task_A/contracts/*`

Retired `run_block3` / `review_block3` surfaces and any packet-local Block 3
mirrors are downstream/non-authority only and must not be promoted to
scientific authority.

## Workflow Boundaries

- `prepare`: Step 1 Task A entrypoint; validates Stage 0 alignment and writes
  Task A mapping/dry-run manifests.
- `run_block0`: writes the STRIDE-native Task A locality gate bundle.
- `run_block1`: writes the real-data biological discovery bundle and summaries.
- `run_block2`: writes robustness outputs and may be resumed with `--resume`.
- `run_block3`: retired from the active public path; currently a fail-fast
  deprecation stub. Internal Block 3 Phase 3 execution may exist inside
  `tasks/task_A/block3/`, but it is not a public workflow surface.
- `review_block3`: retired from the active path; currently a fail-fast
  deprecation stub rather than a live workflow surface.
- `package_results`: packet-local packager for atlas/Block 0/1/2 outputs
  only; it must fail-fast on `--block3-manifest` until a clean non-authority
  Block 3 bridge spec is approved.

## Resume And Rerun Rules

- `run_block2 --resume` is the only documented long-run resume surface in the
  current Task A stack.
- Block 3 has no checkpoint/resume path. The public Block 3 runner remains
  absent; any internal single-subexperiment Phase 3 execution inside
  `tasks/task_A/block3/` does not define a resume surface.
- Prefer reusing one evidence-ready Block 2 manifest before considering any
  upstream rerun.

## Review And Result-Packet Boundary

- Review packets remain packet-local.
- Preserved proxy-history packets under `tasks/task_A/result_packets/` are
  historical context only.
- Do not relabel packet-local review calls as live scientific estimands.

## Required Sync When Touching Block 3

If a change touches Block 3 contracts, review surfaces, or result packets, sync
the nearest affected surfaces:

- `docs/task_A_spec.md` only when a true contract change is explicitly intended
- `docs/task_A_block3_redesign_v1_1.md` when the live Block 3 alignment wording
  changes
- `tasks/task_A/README.md` when routing, boundaries, or launch guidance change
- `tasks/task_A/block3_execution_runbook.md` when launch/verification steps
  change
- `tests/test_task_a_block3_deferred_surfaces.py`
- `tests/test_task_a_canonical_step3_review.py`
- `tests/test_task_a_result_packet.py`
- `tests/test_task_a_design_freeze.py`
- `tests/test_agent_protocol.py` when protocol wording or routing changes

## Stop And Ask The User

- If a patch would change the Task A authority chain
- If a change requires a fresh scientific rerun rather than a narrow fix
- If preserved packet outputs would need rewriting
- If a change would blur the boundary between packet-local review and
  scientific authority
