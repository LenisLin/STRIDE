# Task A Block 3 Phase 3 Audit Memo

Status: reviewed

Purpose:
- Record whether the current internal Task A Block 3 Phase 3 implementation is
  a real doc-constrained execution surface rather than a demo, smoke run, or
  summary-only shell.
- Record whether the code now carries enough textual structure for a strict
  logic-and-contract audit at script, function, and key-variable level.

Reviewed sources:
- `docs/task_A_spec.md`
- `docs/task_A_block3_redesign_v1_1.md`
- `tasks/task_A/README.md`
- `tasks/task_A/block3_execution_runbook.md`
- `tasks/task_A/contracts/artifact_contracts.md`
- `tasks/task_A/contracts/execution_graph.md`
- `tasks/task_A/contracts/deferred_boundaries.md`

Reviewed code/tests:
- `tasks/task_A/block3/__init__.py`
- `tasks/task_A/block3/contracts.py`
- `tasks/task_A/block3/registry.py`
- `tasks/task_A/block3/bundle.py`
- `tasks/task_A/block3/review.py`
- `tasks/task_A/block3/execution.py`
- `src/stride/workflows/fit_stride.py`
- `tests/test_stride_fit_workflow.py`
- `tests/test_task_a_block3_scaffold.py`
- `tests/test_task_a_block3_internal_execution.py`
- `tests/test_task_a_block3_deferred_surfaces.py`
- `tests/test_task_a_block3_phase3_audit.py`

Traceability findings:
- The internal Block 3 execution path still requires an evidence-ready,
  `canonical_full`, `canonical_rerun` Block 2 manifest and an evidence-ready
  Block 1 bundle before any Phase 3 work begins.
- The generator path now enforces the frozen outer design of `10` reruns with
  `24 train / 8 test` per rerun and fails fast when fewer than `32` eligible
  patients are available.
- `3B` remains the only section that expands `relation_null`,
  `relation_weak`, `relation_mid`, and `relation_strong`.
- `3C-1` and `3C-2` consume opaque rerun-specific realization sets and do not
  expose a public relation axis.
- Raw artifacts now retain proof-carrying stores for rerun splits, hidden
  truth, and native method outputs rather than only summary metrics.
- `open_channel_ablation` now changes the proxy/local patient bridge inference
  path through a benchmark-controlled observation match penalty, so `3C-1`
  diverges before canonicalization and recurrence estimation rather than only
  at output projection.
- `cohort_ablation` continues to preserve the proxy/local patient bridge path
  while disabling only cohort recurrence shrinkage on the canonical full path.

Anti-fabrication findings:
- Production Block 3 modules no longer use `_DEMO_*`, `method_penalty`,
  `patient_offset`, or `rerun_offset` style score fabrication.
- Internal manifest workflow names and module wording no longer identify the
  current Phase 3 implementation as a `demo` surface.
- Public Block 3 entrypoints remain deferred/fail-fast, so the internal Phase 3
  implementation is not being relabeled as a live scientific workflow.

Annotation coverage findings:
- All audit-critical Block 3 modules now carry explicit module docstrings with
  role/boundary language.
- `tasks/task_A/block3/execution.py` now documents the generator flow,
  section-specific row builders, and the end-to-end execution entrypoint.
- `src/stride/workflows/fit_stride.py` now documents benchmark-mode semantics,
  canonicalization behavior, and the fit-entrypoint differences between proxy
  and canonical full STRIDE.
- Audit-critical constants such as rerun count, train/test split, epsilon,
  relation scenarios, and benchmark modes now carry local semantic guidance.

Blocking mismatches:
- none found for the current internal Phase 3 audit-use boundary. Block3b
  still has documented implementation-lag items that must be closed before the
  rebuilt 3B surface is treated as contract-aligned execution.

Targeted verification:
- Command run:
  `PYTHONPATH=.:src pytest -q tests/test_stride_fit_workflow.py tests/test_task_a_block3_scaffold.py tests/test_task_a_block3_internal_execution.py tests/test_task_a_block3_deferred_surfaces.py tests/test_task_a_block3_phase3_audit.py tests/test_agent_protocol.py`
- Command run:
  `python -m py_compile src/stride/workflows/fit_stride.py tasks/task_A/block3/__init__.py tasks/task_A/block3/contracts.py tasks/task_A/block3/registry.py tasks/task_A/block3/bundle.py tasks/task_A/block3/review.py tasks/task_A/block3/execution.py tests/test_task_a_block3_phase3_audit.py`

Decision:
- `ready_for_internal_phase3_audit_use`

Reviewer:
- `Codex`
