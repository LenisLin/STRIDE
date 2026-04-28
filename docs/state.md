# Current Status

This file records the current live STRIDE repository state.

## Current Facts

- STRIDE is the live project and scientific identity.
- `src/stride/` is the canonical task-insensitive core package and live
  first-pass implementation surface.
- `tasks/` owns task-specific workflows, benchmark helpers, and operational
  task documentation.
- Current uncertainty means bootstrap/sampling-variance uncertainty over
  fitted patient relation outputs.
- `stride.api.fit.fit_stride(...)` is the formal full-estimator contract
  surface for manuscript-level STRIDE.
- Source/target declaration is a task-layer input; the estimator remains
  task-insensitive once explicit comparison inputs are passed.

## Stable Design Reference

The active STRIDE story is stable at the design level:

- the primary scientific object is `(T_p, e_p)` with `T_p = [A_p | d_p]`,
- `A_p`, `d_p`, and `e_p` are objective-driven fitted variables,
- each source row `[A_{p,i,*}, d_{p,i}]` lies on a simplex,
- `e_p` is bounded in `[0,1]`,
- post-side composition reconstruction uses `normalize(q_minus @ A + e)`,
- `A_p` has an optional derived conditional-kernel view only when needed for
  exposition,
- burden and composition remain separate scales,
- the canonical observation layer is domain-stratified bag-of-FOV comparison,
- domain remains an observation-layer stratum,
- `L_obs` is fixed at object/scalar-role level as task-resolved source/target
  evidence-block comparison with the canonical `D_obs^UOT-v1` observation
  discrepancy operator and `C_norm = C_raw / s_C`,
- the frozen full-estimator objective skeleton is
  `L_total = (1 - alpha) * L_local + alpha * L_regularization`,
- default `alpha = 0.5`,
- `alpha` sensitivity grids are optional diagnostics,
- component losses are normalized by baseline scales from deterministic
  identity-plus-small-open initialization with
  `delta_init = min(0.05, 1 / (K + 1))`, `A_init = (1 - delta_init) * I_K`,
  `d_init = delta_init * 1_K`, and
  `e_init = (delta_init / K) * 1_K`, with epsilon floors for near-zero scales,
- `L_open` is fixed-scale L1 open-channel usage complexity over fitted `d_p`
  and `e_p`: `mean(d)+mean(e)` with scale `1`,
- `L_consistency` measures block-level support dispersion across resolved
  evidence blocks and returns `consistency_status = "insufficient_blocks"`
  with zero raw loss when fewer than two blocks are available,
- cohort recurrence uses a single cohort consensus relation `R_bar` and feeds
  back into estimation through dispersion around that consensus,
- the assembled full objective is treated as a constrained non-convex numerical
  objective; the contract does not claim global convexity or a global optimum.

The canonical full-method doc is:

- `docs/stride_design_freeze.md`

Supporting core method docs are:

- `docs/decisions.md`
- `docs/api_specs.md`
- `docs/data_contracts.md`
- `docs/overall_validation_plan.md`
- `docs/constraints.md`

## Live Implementation Surface

- Frozen next implementation target: `fit_stride(...)` is the manuscript-level
  full STRIDE estimator surface for objective-driven fitting of `A_p`, `d_p`,
  and `e_p` with compact provenance.
- The current first-pass patient-relation implementation covers a narrower
  supported configuration: uniform-mass inputs with exactly two ordered groups
  and explicit deferred status for unsupported configurations.
- The current recurrence implementation is a conservative first-pass
  consensus-template estimator with explicit deferred status when patient
  support is insufficient.
- Next implementation targets include feasible parameterization for
  `[A_i,* , d_i]` simplex rows with bounded `e`, deterministic
  identity-plus-small-open initialization, composition-scale post
  reconstruction, a single global objective fit, compact provenance, and
  cohort consensus recurrence feedback.
- Task A Block 0-2 workflows may call `fit_stride(...)` as part of first-pass
  validation. Full-estimator completion is tracked by the implementation
  targets above.
- The Task A internal Block 3 rebuild package is the current method-validation
  implementation carrier for semisynthetic generator validation, baseline
  comparison, and ablation studies. Its `stride_reference` target is the
  formal `fit_stride(...)` frozen reference configuration, with task adapters
  limited to input conversion and comparison-plan instantiation.

## Deferred Work

- Implementation of canonical `L_obs` scalar/provenance and tests for
  observation diagnostics, including cost normalization, backend version, and
  Sinkhorn/UOT status handling recorded in provenance.
- Implementation of deterministic identity-plus-small-open initialization with
  the fixed `delta_init = min(0.05, 1 / (K + 1))` formula.
- Implementation of the canonical PyTorch/AdamW optimizer protocol,
  `weight_decay = 0.0`, optional fixed scheduler provenance, and convergence
  criteria.
- Alignment of the dependency/runtime surface with the PyTorch/AdamW optimizer
  contract, including explicit optimizer availability, status, and failure
  provenance when the canonical optimizer cannot be executed.
- Implementation of single cohort consensus recurrence outputs, dispersion,
  and fit status.
- Implementation of bounded `e_p` parameterization.
- Implementation of the composition-scale post reconstruction form.
- Implementation of open-channel provenance fields, including `e_bounds`,
  `post_reconstruction_form`, and `open_channel_normalization_scale`.
- Full objective implementation for objective-driven fitted `A_p`, `d_p`, and
  `e_p` under the frozen objective grouping and normalization policy.
- Compact provenance schema covering default `alpha`, any `alpha` sensitivity
  grid, loss decomposition, normalization scales, epsilon-floor flags,
  initialization policy, `e_bounds`, `post_reconstruction_form`,
  `observation_comparison_plan`, `observation_discrepancy_backend`, operator
  version, cost normalization, Sinkhorn/UOT status handling, observation
  diagnostics when emitted, `open_channel_complexity_form`,
  `open_channel_normalization_scale`, optimizer framework/protocol/status,
  scheduler policy/status when used, recurrence consensus
  support/dispersion/status, ablation mode, random seed, and convergence or
  failure reason.
- Task A wiring validation showing that `stride_reference` calls the formal
  `fit_stride(...)` reference configuration and that Task A adapters only
  convert inputs and instantiate declared comparison plans.
- Task-pipeline expansions beyond the documented Task A operational surfaces.
- Additional public Block 3 workflow and packet integration surfaces.

## Non-Blocking Open Points

- The formal full-estimator path is frozen in the contract, but implementation
  completion remains an open engineering item.
- The full estimator v1 optimizer target is PyTorch/AdamW; current
  implementation and runtime packaging still need an explicit alignment pass
  before optimizer availability can be treated as guaranteed.
- Namespace direction and estimator completeness remain separate questions:
  `stride` is the target architecture even when some estimators are still
  deferred.
- Some current implementation terms still use `prototype` language where the
  design uses the broader term "shared `K`-state basis".
- The longitudinal validator still accepts implementation-era aliases for input
  tolerance.

## Source-of-Truth Order

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, and `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A_rewiring_plan.md`
5. `docs/task_A_spec.md`, `docs/task_A_result.md`, and
   `tasks/task_A/README.md` for the current Task A task layer
6. Task-local operational docs under `tasks/`

`docs/dev_log.md` is repo-memory only and not a design reference.
