# Task A Contract Alignment Review Memo

Status: superseded historical review memo

This memo records an earlier Block 1 alignment review. It is not a live
contract surface after the Block 1 `block1_real_data_discovery` cleanup. The
retired Block 1 bundle and `block1_continuity_backbone` references below are
historical context only.

Purpose:
- Record whether the current Task A documentation, bundle schema, workflow
  contract, and test surface all agree before the next Block 1 coding round.

Reviewed sources:
- `docs/task_A/spec.md`
- `tasks/task_A/README.md`
- `tasks/task_A/contracts/design_freeze.py`
- `tasks/task_A/contracts/execution_graph.md`
- `tasks/task_A/contracts/artifact_contracts.md`
- `tests/test_task_a_descriptive_atlas.py`
- `tests/test_task_a_design_freeze.py`
- `tests/test_task_a_block1_summaries.py`
- `tests/test_task_a_runtime.py`

Alignment findings:
- The live spec now uses the frozen Block 1 machine identifiers
  `burden_weighted` and `community_mean`, while keeping the scientific wording
  `SR / D / R / E` and the target-side supportive role of `E`.
- The live spec now treats Block 0 as a `TC-IM` real/null execution-cache
  surface plus cache-derived calibration analysis. The null is a within-patient
  count-preserving TC/IM domain-label permutation; deterministic gates and
  paired `sum(A)` / `sum(e)` deltas are not active Block 0 contract surfaces.
- `README.md`, `design_freeze.py`, `execution_graph.md`, and
  `artifact_contracts.md` agree that Block 1 emits:
  `block1_family_summary.csv`,
  `block1_source_community_summary.csv`,
  `block1_target_community_summary.csv`,
  `block1_bundle.json`, and `block1_workflow_manifest.json`.
- Legacy compatibility identifiers remain unchanged:
  `block1_continuity_backbone` and the existing Python workflow entrypoints.
- Non-claim boundary wording remains aligned with the scientific framing:
  no direct temporal claims, no proof of true emergence/disappearance, and
  target-side `E` remains supportive.

Targeted verification:
- Command run:
  `pytest -q tests/test_task_a_descriptive_atlas.py tests/test_task_a_design_freeze.py tests/test_task_a_block1_summaries.py tests/test_task_a_runtime.py`
- Result:
  `14 passed`

Residual note:
- `tasks/task_A/README.md` lists
  `task_a_pre_block0_data_suitability.json` inside the Block 0 row because
  the Block 0 package writes that sidecar into the same output directory. The
  machine-readable freeze models it as the separate
  `write_task_a_pre_block0_data_suitability_report` surface. This is a
  documentation nuance, not a Block 1 contract blocker.

## Required conclusion block

- Reviewer:
  `Codex`
- Review date:
  `2026-04-02`
- Spec versus contract status:
  aligned after updating the live spec to use the frozen scale identifiers and
  current Block 0 gate wording
- Contract versus tests status:
  aligned; targeted atlas/design-freeze/Block 1/runtime surface passed
  with `14` green tests
- Compatibility identifier status:
  `block1_continuity_backbone` and existing Python
  entrypoints remain unchanged
- Non-claim boundary status:
  aligned; wording still avoids direct temporal claims and keeps
  depletion/emergence interpretation cautious
- Blocking mismatches:
  none for the Block 1 summary contract; only a non-blocking README nuance on
  the co-located pre-Block 0 suitability sidecar
- Decision:
  `ready_for_block1_coding`
