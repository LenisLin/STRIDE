# Current Status

This file records the current live STRIDE repository state.

## Current Facts

- STRIDE is the live project and scientific identity.
- `src/stride/` is the canonical task-insensitive core package and live
  first-pass full-estimator implementation surface.
- `tasks/` owns task-specific workflows, benchmark helpers, and operational
  task documentation.
- Current uncertainty means bootstrap/sampling-variance uncertainty over
  fitted patient relation outputs.
- `stride.api.fit.fit_stride(...)` is the formal full-estimator contract
  surface for manuscript-level STRIDE.
- The current `fit_stride(...)` code path contains a bounded PyTorch/AdamW
  full-estimator implementation for supported inputs, with explicit
  optimizer/status surfaces for unsupported or numerically incomplete fits.
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
  evidence-block comparison with the canonical
  `D_obs^BalancedSinkhornDivergence-v1` observation discrepancy operator,
  `C_norm = C_raw / s_C`, and fixed `G_norm = G / s_G_init`,
- `D_obs` is the fixed operator inside `L_obs`, not a sixth loss term or an
  independently weighted component,
- `y_hat_f = normalize(v_source_f @ A_p + e_p)` defines the predicted
  target-side FOV vector before FOV-bag comparison,
- canonical `L_obs` is torch-native, `float64`, log-domain, differentiable,
  balanced, and debiased; open behavior is expressed only by fitted `d/e`,
- the frozen full-estimator objective skeleton is
  `L_total = (1 - alpha) * L_local + alpha * L_regularization`,
- the frozen group-internal means are
  `L_local = mean(normalized_L_obs, normalized_L_open, normalized_L_geometry)`
  and
  `L_regularization = mean(normalized_L_consistency, normalized_L_recurrence)`,
- default `alpha = 0.5`,
- `alpha` sensitivity grids are optional diagnostics,
- `L_geometry` is a fixed normalized component over raw canonical `A_p` using
  `L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]`, all
  `K` source rows, simple mean over valid fitted patients, and no separate
  geometry weight,
- `C_raw/C_norm` must be finite, nonnegative, symmetric `[K, K]` state
  geometry with diagonal `0`; `s_C` is the median of positive finite
  off-diagonal `C_raw` entries and invalid scale fails the contract,
- except for open, component losses are normalized by
  `scale_c = max(raw_L_c(theta_init), epsilon_norm)` and
  `normalized_L_c = raw_L_c(theta) / scale_c`, where
  `theta_init` is deterministic identity-plus-small-open initialization with
  `delta_init = min(0.05, 1 / (K + 1))`, `A_init = (1 - delta_init) * I_K`,
  `d_init = delta_init * 1_K`, and
  `e_init = (delta_init / K) * 1_K`; `epsilon_norm = 1e-2` is a
  dimensionless `float64` full-estimator loss-normalization floor,
- `L_open` is fixed-scale L1 open-channel usage complexity over fitted `d_p`
  and `e_p`: `mean(d)+mean(e)` with scale `1`,
- full-estimator ablations keep fixed group denominators and do not re-average
  active terms; geometry ablation uses
  `(normalized_L_obs + normalized_L_open + 0) / 3`, recurrence ablation uses
  `(normalized_L_consistency + 0) / 2`, and consistency ablation uses
  `(0 + normalized_L_recurrence) / 2`,
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

- `fit_stride(...)` is the manuscript-level full STRIDE estimator surface for
  objective-driven fitting of `A_p`, `d_p`, and `e_p` with compact
  successful-fit provenance.
- The supported first-pass full-estimator path uses PyTorch/AdamW over a
  constrained parameterization of source-row `[A_i,* , d_i]` simplexes and
  bounded `e`, deterministic identity-plus-small-open initialization,
  composition-scale post reconstruction, fixed component normalization, and the
  canonical `D_obs^BalancedSinkhornDivergence-v1` observation discrepancy
  operator.
- The supported first-pass input configuration is uniform-mass patient inputs
  with exactly two ordered groups per patient and valid shared-state geometry.
  Unsupported full-estimator requests receive explicit non-`ok` status, and
  non-ablation compatibility routes that fall back to the local initializer are
  not evidence that the full objective path successfully fit.
- Successful full-estimator fits emit compact `stride_fit_provenance.v1`
  records with loss raw/scale/normalized/floor values, optimizer protocol,
  observation backend configuration, state-geometry normalization, recurrence
  support/dispersion, and initialization metadata.
- Internal full-estimator ablations are recurrence, geometry, and consistency
  refits. They use fixed group denominators and do not reweight retained
  objective terms.
- The current recurrence implementation provides single cohort consensus
  recurrence outputs and explicit deferred status when patient support is
  insufficient.
- Task A Block 0-2 workflows may call `fit_stride(...)` as part of first-pass
  validation. Historical outputs that predate this implementation remain
  proxy/history unless rerun through the current contract.
- The Task A internal Block 3 rebuild package is the current method-validation
  implementation carrier for semisynthetic generator validation, baseline
  comparison, and ablation studies. Its `stride_reference` target is the
  formal `fit_stride(...)` frozen reference configuration, with task adapters
  limited to input conversion and comparison-plan instantiation.

## Remaining Engineering Work

- Broaden the full-estimator supported-input envelope beyond the first-pass
  uniform-mass, exactly-two-ordered-group configuration.
- Calibrate optimizer stopping criteria and runtime packaging at production
  scale while preserving explicit non-`ok` status for numerical
  non-completion.
- Maintain optional detailed optimizer traces as diagnostics rather than
  required compact provenance fields.
- Retire or further isolate local-initializer compatibility routes after all
  live task workflows have migrated to supported full-estimator inputs.
- Rerun approved Task A validation surfaces and refresh result packets only
  through the documented workflow, because historical packets remain proxy
  history until regenerated.
- Continue narrow tests for the public import surface, full-estimator objective,
  optimizer/provenance, observation backend, and Task A Block 3 method wiring.

## Non-Blocking Open Points

- The supported full-estimator path is an implemented numerical optimizer path,
  not a claim of global optimum for the non-convex objective.
- The full estimator v1 optimizer requires PyTorch availability at runtime;
  missing optimizer dependencies surface as explicit failures rather than
  successful compact provenance.
- Namespace direction and estimator completeness remain separate questions:
  `stride` is the target architecture while input-support expansion and
  historical-workflow migration continue.
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
