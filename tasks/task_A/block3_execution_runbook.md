# Task A Block 3 Internal Execution Note

The scientific authority for Block 3 remains:

- [`docs/task_A/spec.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/spec.md)
- [`docs/task_A/block3/scientific_contract.md`](/home/lenislin/Experiment/projects/STRIDE/docs/task_A/block3/scientific_contract.md)

The current on-disk `tasks/task_A/block3/` package hosts internal Phase 3
execution with semisynthetic generator logic, method execution, scoring, and
raw/review artifact writing. Public Block 3 workflow and packet integration are
not restored.

Internal CLI experiment names are semantic:
`generator_validation`, `a_benchmark`, `de_benchmark`,
`subbag_consistency_ablation`, `geometry_ablation`, and
`recurrence_ablation`. The registry maps those names to `3A`, `3B-1`, `3B-2`,
`3C-1`, `3C-2`, and `3C-3` subexperiment ids.

Current engineering boundary:

- [`tasks/task_A/block3/execution.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/block3/execution.py)
  may host internal single-subexperiment Phase 3 execution. This is an
  internal package helper only, not a public workflow entrypoint.
- `python -m tasks.task_A.block3 <experiment_name> --task-config ... --stage0-h5ad ... --output-dir ...`
  is the internal package-local CLI surface.
- `--max-reruns <n>` is an internal smoke-test limiter only. Formal Block 3
  runs omit it and use the contract default rerun count.
- `--n-test <n>` is an internal smoke-test limiter for held-out test patients
  only. Formal Block 3 runs omit it and use the contract default of `8` test
  patients.
- Any run using `--max-reruns` or `--n-test` is an engineering correctness
  smoke run. Its raw and review manifests must record
  `execution_scope=subset_engineering_test`, and the output must not be read as
  a formal full-data result.
- [`tasks/task_A/workflows/package_results.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/workflows/package_results.py)
  accepts atlas plus Block 0/1 packet inputs for the current result packet.
- [`tasks/task_A/result_packet.py`](/home/lenislin/Experiment/projects/STRIDE/tasks/task_A/result_packet.py)
  records the current packet layer boundaries.

Block3b internal implementation decisions:

- Block3b implementation is scoped only to the internal Phase 3 carrier under
  `tasks/task_A/block3/`. Public Block 3 workflow and packet bridge behavior
  remain absent. Internal outputs remain Phase 3 implementation carriers.
- The umbrella `3B` route must be split into explicit executable `3B-1 A
  benchmark` and `3B-2 d/e benchmark` routes. `3B` remains an umbrella or
  section-group label only. Both executable routes use the shared multi-FOV
  generator realization. The generator builds train-derived templates from
  real train TC-IM endpoints, applies geometry-gated residual coupling with
  `tau=2.0`, mixes the medoid and sampled individual template with
  `lambda_individual=0.10`, and generates source/target FOVs with `eta=0.3`.
- `stride_reference` must call the formal `stride.tl.fit(...)` frozen reference
  configuration and emit native fitted `A/d/e`. The Task A adapter may only
  convert Block 3 inputs, resolve source/target endpoint comparison evidence
  blocks, and instantiate the comparison plan, including valid domain strata,
  for the formal estimator input contract. Domain resolution remains
  task-layer provenance and does not become a core loss/state/relation/
  recurrence axis. The adapter must not implement a task-local semi-STRIDE,
  STRIDE-like substitute estimator, or observation backend replacement.
- This is a formal reference target on the current bounded first-pass
  full-estimator support envelope. Unsupported inputs, compatibility fallback
  routes, or non-`ok` optimizer statuses are not successful `stride_reference`
  full-objective fits.
- Internal Block 3 `stride_reference` should preserve compact successful-fit
  provenance if it is emitted by `stride.tl.fit(...)`. The runbook does not
  require status/failure audit expansion or per-evidence-block provenance
  records.
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
  configurations, retain the canonical
  `D_obs^BalancedSinkhornDivergence-v1` observation discrepancy backend, and
  refit `A/d/e` under the ablated objective.
- For `stride_reference`, Task A passes its fixed `C_raw`, `s_C`, and `C`
  source as an adapter/benchmark cost source into the full-estimator
  shared-state cost contract. The runbook does not define a task-local
  geometry prior.
- `uot_baseline` must reuse the existing STRIDE Sinkhorn/UOT adapter,
  `src/stride/adapters/ot_sinkhorn.py::batched_uot_solve(..., return_plan=True)`.
  `lambda_match` uses rerun-shared train calibration over the fixed internal
  `lambda_grid = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`. The calibration
  target is train-side mean endpoint overlap, `mean(sum(min(x, y)))`. If
  multiple candidates are equally close, select the smaller `lambda`.
  Boundary selection is recorded through native `boundary_hit` metadata as a
  diagnostic and does not trigger automatic grid expansion before the formal
  Block 3 run.
- The internal Block 3 CLI optional `--device` flag is a runtime control only.
  `--device cuda` requests CUDA for STRIDE reference/refit methods and the UOT
  torch runtime used by both UOT train calibration and test solves. Exact
  transport comparators remain on their CPU solver routes.
- `partial_ot_baseline` uses rerun-shared
  `matched_mass_budget = mean(sum(min(x_train, y_train)))`.
  `partial_ot_baseline` must use `ot.partial.partial_wasserstein` as the exact
  fixed-mass partial OT solver. The metadata must record the requested budget,
  effective clipped budget, transported mass, objective value, solver name,
  and train-side budget calibration source. For an individual test patient, if
  the shared budget exceeds that patient's feasible upper bound, clip it to
  the patient-specific feasible upper bound and record the clipping in
  calibration/native metadata. `diagonal_transport_baseline` uses exactly
  `P[i,i] = min(x[i], y[i])`.
- Endpoint-only baselines consume deterministic endpoint projections of the
  generated multi-FOV observations. STRIDE reference and ablation arms consume
  the generated source/target FOV observations.
- Plan-based baselines derive `A/d/e` from `P` in one shared analysis layer:
  `r = row_sum(P)`, `c = col_sum(P)`, `A[i,:] = P[i,:] / x[i]` only when
  `x[i] > tol`, `d[i] = max((x[i] - r[i]) / x[i], 0.0)` only when
  `x[i] > tol`, and `e = maximum(y - c, 0.0)`. When `x[i] == 0`, the analysis
  layer must not create a fallback relation; it keeps that row at `A = 0` and
  `d = 0`.
- Method-bearing Block 3 metrics include `F_L1_total`, `g_L1_total`,
  `e_L1_total`, mass absolute-error metrics, `offdiag_ratio`,
  `depletion_capture`, `emergence_capture`, `endpoint_y_MAE`,
  `A_MAE_active`, `A_MSE_active`, `target_recall_at_k`, `open_support_F1`,
  `d_MAE`, `d_MSE`, `e_MAE`, and `e_MSE`.
- `de_benchmark` is open-focused, but it keeps the complete shared metric
  vocabulary so mass, relation, open-channel, and endpoint behavior remain
  comparable across `3B-1`, `3B-2`, and `3C-*`.
- Split raw/review artifacts use semantic roots such as `a_benchmark_*` and
  `de_benchmark_*`. Shared truth/native stores may remain shared during
  implementation if `subexperiment_id` is explicit.
- Reference reuse for `stride_reference` across the same generated realization
  and device is deferred as a full-data runtime optimization. It is not a
  subset smoke acceptance gate.

Block3c internal implementation decisions:

- The `3C` design remains frozen, but the current public estimator has no
  executable consistency, geometry, or recurrence ablation hook. Each `3C`
  semantic command therefore writes a structured deferred/unsupported status
  before Stage 0 loading or fitting. It must not emit fabricated raw/review
  metrics, copy the reference fit, or perform post-hoc masking.
- The core STRIDE internal ablation set is restricted to `consistency`,
  `geometry`, and `recurrence`.
- Under the current frozen numbering, `3C-1` is `consistency_ablation`, `3C-2`
  is `geometry_ablation`, and `3C-3` is `recurrence_ablation`.
- Each Block 3C ablation must remove or zero the corresponding objective term
  and rerun/refit the estimator so that `A_p`, `d_p`, and `e_p` are newly fitted
  under the ablated objective.
- `geometry_ablation` removes or zeroes the full-estimator `L_geometry` term
  over raw canonical `A_p` and then refits under that ablated objective.
- Block 3C must not implement ablations by masking fitted outputs from the
  reference fit.
- Block 3C must not implement ablations by post-hoc rescoring only.
- Block 3C must not reweight retained objective terms or re-average active
  terms after ablation; it keeps the fixed full-estimator group denominators.
- Each ablation uses the same deterministic initialization, optimizer
  protocol, resolved evidence blocks, and rerun-specific semi-synthetic
  realization as the reference fit.
- Ablation/refit experiment provenance records `ablation_mode` and whether the
  arm used a remove or zero-weight implementation route. Ordinary
  `stride_reference` fit provenance is not required to expose ablation as a
  user-level control.
- Block 3C method names are `recurrence_ablation`, `geometry_ablation`, and
  `consistency_ablation`.
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
