# Task A Block 3 Phase 3 Audit Memo

Status: superseded pre-implementation historical audit surface

Purpose:
- Preserve the pre-implementation Phase 3 audit context without redefining the
  live Task A Block 3 contract.
- Record implementation-lag items that must be resolved before internal Phase
  3 execution can be treated as aligned with the live `3B/3C` contract.

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
- The live `3C` contract contains `recurrence_ablation`,
  `geometry_ablation`, and `consistency_ablation`.
- Each live `3C` arm removes or zeroes the corresponding objective term and
  refits patient-level `A_p`, `d_p`, and `e_p` on the same rerun-specific
  realization set as `stride_reference`.
- Live `3C` uses native patient-level recovery metrics for `A_p`, `d_p`, and
  `e_p`; comparator-style transport or channel variants belong to the `3B`
  baseline-comparison contract.
- Raw artifacts now retain proof-carrying stores for rerun splits, hidden
  truth, and native method outputs rather than only summary metrics.

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
- The internal Phase 3 audit surface predates the live `3C` split into
  recurrence, geometry, and consistency refit ablations. It therefore remains
  historical audit context until implementation, artifacts, and tests are
  synchronized to that live contract.
- Block3b still has documented implementation-lag items that must be closed
  before the rebuilt 3B surface is treated as contract-aligned execution.

Targeted verification:
- Command run:
  `PYTHONPATH=.:src pytest -q tests/test_stride_fit_workflow.py tests/test_task_a_block3_scaffold.py tests/test_task_a_block3_internal_execution.py tests/test_task_a_block3_deferred_surfaces.py tests/test_task_a_block3_phase3_audit.py tests/test_agent_protocol.py`
- Command run:
  `python -m py_compile src/stride/workflows/fit_stride.py tasks/task_A/block3/__init__.py tasks/task_A/block3/contracts.py tasks/task_A/block3/registry.py tasks/task_A/block3/bundle.py tasks/task_A/block3/review.py tasks/task_A/block3/execution.py tests/test_task_a_block3_phase3_audit.py`

Decision:
- Historical audit context only. The live target is the frozen Task A Block 3
  contract in `docs/task_A_spec.md` and
  `docs/task_A_block3_redesign_v1_1.md`; implementation alignment remains
  pending for the refit-ablation `3C` surface.

Reviewer:
- `Codex`
