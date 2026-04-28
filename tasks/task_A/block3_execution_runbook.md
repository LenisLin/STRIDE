# Task A Block 3 Internal Execution Note

The scientific authority for Block 3 remains:

- [`docs/task_A_spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_spec.md)
- [`docs/task_A_block3_redesign_v1_1.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A_block3_redesign_v1_1.md)

The current on-disk `tasks/task_A/block3/` package hosts internal Phase 3
execution with semisynthetic generator logic, method execution, scoring, and
raw/review artifact writing. Public Block 3 workflow and packet integration
require a follow-up specification.

Current engineering boundary:

- [`tasks/task_A/block3/execution.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/block3/execution.py)
  may host internal single-subexperiment Phase 3 execution. This is an
  internal package helper only, not a public workflow entrypoint.
- [`tasks/task_A/workflows/run_block3.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/workflows/run_block3.py)
  remains reserved for a follow-up public workflow specification.
- [`tasks/task_A/workflows/review_block3.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/workflows/review_block3.py)
  remains reserved for a follow-up public workflow specification.
- [`tasks/task_A/workflows/package_results.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/workflows/package_results.py)
  accepts atlas plus Block 0-2 packet inputs for the current result packet.
- [`tasks/task_A/result_packet.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/result_packet.py)
  records the current packet layer boundaries.

Block3b internal implementation decisions:

- The next Block3b implementation pass is scoped only to the internal Phase 3
  carrier under `tasks/task_A/block3/`. Public `run_block3`, `review_block3`,
  and packet bridge behavior require a follow-up specification. Internal
  outputs remain Phase 3 implementation carriers.
- The umbrella `3B` route must be split into explicit executable `3B-1 A
  benchmark` and `3B-2 d/e benchmark` routes. `3B` remains an umbrella or
  section-group label only. `3B-1` uses
  `relation_strength_grid = [0.00, 0.05, 0.15, 0.30]` at fixed
  `open_mass_scale = 1.0`. `3B-2` uses
  `open_mass_scale_grid = [0.0, 0.1, ..., 1.0]` at fixed
  `relation_strength = 0.15`.
- The primary `3B-1` contract remains fixed at `open_mass_scale = 1.0`.
  Internal `3B-1` A-recovery sensitivity may evaluate
  `open_mass_scale = [0.1, 0.25, 0.5]` as a diagnostic sidecar only. The
  `0.25` value is diagnostic-only and is not part of the frozen public `3B-2`
  dense open-mass grid.
- `stride_reference` must call the formal `fit_stride(...)` frozen reference
  configuration and emit native fitted `A/d/e`. The Task A adapter may only
  convert Block 3 inputs, resolve source/target endpoint comparison evidence
  blocks, and instantiate the comparison plan, including valid domain strata,
  for the formal estimator input contract. Domain resolution remains
  task-layer provenance and does not become a core loss/state/relation/
  recurrence axis. The adapter must not implement a task-local semi-STRIDE,
  proxy initializer, STRIDE-like substitute estimator, or observation backend
  replacement.
- `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` emit native matched plan `P`.
  The live Block3b method routes are `stride_reference`,
  `balanced_ot_baseline`, `uot_baseline`, `partial_ot_baseline`, and
  `diagonal_transport_baseline` for `3B-1`; `3B-2` uses `stride_reference`,
  `uot_baseline`, `partial_ot_baseline`, and `diagonal_transport_baseline`.
- Balanced, closed, transport-style, and no-open-channel or no-`d/e`
  comparisons belong in Block 3B as baselines/comparators. They are not core
  STRIDE internal ablations.
- Block 3B baselines may use comparator-specific solvers under their own
  baseline contracts. Block 3C STRIDE ablations use core estimator
  configurations, retain the canonical observation discrepancy backend, and
  refit `A/d/e` under the ablated objective.
- `uot_baseline` must reuse the existing STRIDE Sinkhorn/UOT adapter,
  `src/stride/adapters/ot_sinkhorn.py::batched_uot_solve(..., return_plan=True)`.
  `lambda_match` uses rerun-shared train calibration over the fixed internal
  `lambda_grid = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`. The calibration
  target is train-side mean endpoint overlap, `mean(sum(min(x, y)))`. If
  multiple candidates are equally close, select the smaller `lambda`. If the
  implementation frequently selects boundary values after execution, record
  this as a diagnostic for a later grid update.
- `partial_ot_baseline` uses rerun-shared
  `matched_mass_budget = mean(sum(min(x_train, y_train)))`. For an individual
  test patient, if the shared budget exceeds that patient's feasible upper
  bound, clip it to the patient-specific feasible upper bound and record the
  clipping in calibration/native metadata. `diagonal_transport_baseline` uses
  exactly `P[i,i] = min(x[i], y[i])`.
- Plan-based baselines derive `A/d/e` from `P` in one shared analysis layer:
  `r = row_sum(P)`, `c = col_sum(P)`, `A[i,:] = P[i,:] / x[i]` only when
  `x[i] > tol`, `d[i] = max((x[i] - r[i]) / x[i], 0.0)` only when
  `x[i] > tol`, and `e = maximum(y - c, 0.0)`. When `x[i] == 0`, the analysis
  layer must not create a fallback relation; it keeps that row at `A = 0` and
  `d = 0`. In `3B-2` with `open_mass_scale = 0.0`, `open_support_F1` is
  `not_applicable`; `d/e_MAE` and `d/e_MSE` are still reported.
- Split raw/review artifacts use `3b1_*` and `3b2_*` files. Do not continue
  using one `3b_*` file that mixes `3B-1` and `3B-2`. Shared truth/native
  stores may remain shared during implementation if `subexperiment_id` is
  explicit; otherwise split them into `3b1_*` and `3b2_*` stores.

Block3c internal implementation decisions:

- The core STRIDE internal ablation set is restricted to `recurrence`,
  `geometry`, and `consistency`.
- Each Block 3C ablation must remove or zero the corresponding objective term
  and rerun/refit the estimator so that `A_p`, `d_p`, and `e_p` are newly fitted
  under the ablated objective.
- Block 3C must not implement ablations by masking fitted outputs or rewriting
  scores after the reference fit.
- Open-channel removal, no-`d/e`, balanced, closed, or transport-style
  comparisons remain Block 3B comparators, not Block 3C ablations.

Operational rule:

- `tasks/task_A/block3/` is the internal Phase 3 execution surface.
- Public Block 3 runner, review workflow, and packet bridge behavior require an
  explicit follow-up specification.
- Block 3 implementation work remains anchored to the docs hierarchy.

Audit surfaces:

- For the current logic/contract audit conclusion, see
  [`tasks/task_A/contracts/block3_phase3_audit_memo.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/block3_phase3_audit_memo.md).
- For the requirement-to-code traceability table, see
  [`tasks/task_A/contracts/block3_phase3_traceability_matrix.md`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/contracts/block3_phase3_traceability_matrix.md).
