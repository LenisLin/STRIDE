# Current Status

This file records the current live STRIDE repository state.

## Current Facts

- STRIDE is the live project and scientific identity.
- `src/stride/` is the canonical task-insensitive core package and live
  first-pass full-estimator implementation surface.
- The installable Python distribution name is `stride`, with current source-tree
  version fallback `0.1.0`.
- `tasks/` owns task-specific workflows, benchmark helpers, and operational
  task documentation.
- Current uncertainty means bootstrap/sampling-variance uncertainty over
  fitted patient relation outputs.
- `stride.tl.fit(...)` is the current formal full-estimator API surface, with
  `stride.fit(...)` as the package-root convenience export.
- The public fitting wrapper permits default CUDA-to-CPU fallback when CUDA is
  unavailable; invalid explicit device identifiers still fail.
- The current `stride.tl.fit(...)` code path contains a bounded PyTorch/AdamW
  full-estimator implementation for supported inputs, with explicit
  optimizer/status surfaces for unsupported or numerically incomplete fits.
- The live reference optimizer protocol is fixed as warm-up `20` steps at
  `lr = 0.02`, then main-phase `lr = 0.05` with `CosineAnnealingLR`, main
  minimum `100` steps, and main
  hard cap `200` steps.
- Public API beta status is separate from the v1 scientific objective and
  provenance contracts: the contract is reference-stable, while runtime support
  and the input envelope are not production-stable.
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
- the frozen full-estimator objective is
  `L_total = mean(L_fit, L_prior, L_cohort)`,
- `L_fit = normalized_L_obs + rho_subbag * L_subbag_consistency`,
- `L_prior = mean(normalized_L_open, L_geometry_effective)`,
- `L_cohort = L_recurrence_raw / s_cohort`,
- reference constants are `rho_subbag = 1.0`,
  `geometry_effective_weight = 0.01`, `s_cohort = 1e-2`, and
  `epsilon_norm = 1e-2`,
- objective scale initialization and optimizer start initialization are
  separate contract objects,
- optimizer start initialization uses `offdiag_init_mass = 1e-2` and
  `numerical_min_mass = 1e-12`,
- the reference optimizer protocol is fixed and provenance-visible rather than
  caller-configurable through public `stride.tl.fit(...)` knobs,
- finite capped optimizer exits now remain successful fits and are annotated
  with an explicit optimizer exit flag,
- geometry records raw, scale, normalized, and effective values,
- recurrence acts on `T_p = [A_p | d_p]` and `e_p`,
- `L_geometry_raw(p) = (1 / K) * sum_i sum_j A_p[i,j] * C_norm[i,j]` uses all
  `K` source rows and the raw canonical `A_p`,
- `C_raw/C_norm` must be finite, nonnegative, symmetric `[K, K]` state
  geometry with diagonal `0`; `s_C` is the median of positive finite
  off-diagonal `C_raw` entries and invalid scale fails the contract,
- observation and geometry scales are computed from objective scale
  initialization, while subbag consistency and cohort recurrence have no
  independent baseline-normalization scales,
- `L_open` is fixed-scale L1 open-channel usage complexity over fitted `d_p`
  and `e_p`: `mean(d)+mean(e)`,
- full-estimator ablations use the three-block reference objective with the
  ablated term set to zero,
- subbag consistency measures block-level support dispersion across resolved
  evidence blocks and records insufficient block support with zero contribution
  when fewer than two blocks are available,
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

- The live `src/stride` implementation uses the three-block v1 reference
  objective and compact successful-fit provenance schema.
- `stride.tl.fit(...)` remains the current beta estimator surface for
  objective-driven fitting of `A_p`, `d_p`, and `e_p`, with `stride.fit(...)`
  exported at the package root.
- The root public package surface is intentionally small:
  `fit`, `FitResult`, `RelationResult`, `CohortResult`, `ContractError`, and
  `__version__`.
- Private `_*.py` modules and numerical helper internals are implementation
  surfaces, not stable public API.
- The selected user package namespace design is `stride.io`, `stride.pp`,
  `stride.tl`, `stride.pl`, and `stride.da`, recorded in
  `docs/package_api_design.md`.
- `stride.io` v1 is implemented with `build_adata`, `read_h5ad`, and
  `write_h5ad` for raw AnnData assembly and h5ad persistence, plus explicit
  CSV R handover helpers for downstream plotting tables.
- `stride.io.build_adata(...)` records `community_mode = "fraction"` for the
  current `.io -> .pp -> .tl` supported path. Density community observations
  are not part of that path.
- `stride.pp`, `stride.tl`, `stride.pl`, and `stride.da` are implemented beta
  namespaces. Their public contracts remain first-pass and bounded by
  `docs/api_specs.md`.
- public `stride.tl.fit(...)` no longer exposes direct `lr`, `max_steps`, or
  `min_steps` controls; reference optimizer protocol changes must go through
  the frozen docs/contracts first.
- Task A Block 0/1 workflows may call `stride.tl.fit(...)` as part of first-pass
  validation. Historical outputs that predate this implementation remain
  proxy/history unless rerun through the current contract.
- The Task A internal Block 3 rebuild package is the current method-validation
  implementation carrier for semisynthetic generator validation, baseline
  comparison, and ablation studies. Its `stride_reference` target is the
  formal `stride.tl.fit(...)` frozen reference configuration, with task adapters
  limited to input conversion and comparison-plan instantiation.

## Remaining Engineering Work

- Run approved small validation only after implementation migration.
- Continue package API cleanup review before expanding public objects through
  `stride.pp`, `stride.tl`, `stride.pl`, or `stride.da`, and before extending
  `stride.io` beyond v1 raw AnnData and h5ad persistence.
- Broaden the full-estimator supported-input envelope beyond the first-pass FOV
  community-composition, exactly-two-ordered-group configuration.
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
- Namespace direction is selected at the package-design level; input-support
  expansion, public-contract hardening, and historical-workflow migration remain
  open engineering work.
- The public API is beta even when the objective/provenance/operator contract
  versions are v1.
- Some current implementation terms still use `prototype` language where the
  design uses the broader term "shared `K`-state basis".
- The longitudinal validator still accepts implementation-era aliases for input
  tolerance.

## Source-of-Truth Order

1. `docs/stride_design_freeze.md`
2. `docs/decisions.md`, `docs/api_specs.md`, `docs/data_contracts.md`,
   `docs/overall_validation_plan.md`, and `docs/constraints.md`
3. `docs/state.md`
4. `docs/task_A/spec.md`
5. `docs/task_A/block3/scientific_contract.md` and stage docs under
   `docs/task_A/block3/`
6. `docs/task_A/block3/refactor_contract_map.md` for migration mapping only
7. `docs/task_A/result.md` and `tasks/task_A/README.md` for the current Task A
   task layer
8. Historical/proxy references only

`docs/dev_log.md` is repo-memory only and not a design reference.
