# Decisions

## D001 — SLOTAR V1.5 Core Algorithm Architecture
- **Context**: Longitudinal spatial omics data (e.g., IMC) suffers from structural asymmetry (unmatched mass) due to sampling bias and true biological remodeling (e.g., pCR vs NR in gastric cancer). Traditional paired tests fail under these conditions.
- **Decision**: Adopt the SLOTAR V1.5 framework as the mathematical engine.
  - Use community prototypes (kNN + block-wise robust scaling + k-means) to establish a unified semantic space.
  - Define the transport problem on ROI-/sample-level nonnegative mass distributions over that shared community/prototype axis; individual cells or spots are inputs to representation construction, not direct UOT nodes.
  - Decouple spatial changes into Density-level, Shape-level, and Scale ratio.
  - Employ Unbalanced Optimal Transport (UOT) with generalized KL divergence and entropic regularization ($\varepsilon$-scaling) to attribute changes to Retention, Remodeling, Creation, and Destruction.
  - Implement adaptive grid ($G \times G$) composition-stratified Block Bootstrap for single-ROI uncertainty quantification.
  - Enforce a global static cost scaling factor ($s_C$) to ensure $\lambda$ comparability across calibrations and inferences.
- **Alternatives Rejected**: 
  - Standard Balanced OT (rejected due to inability to handle unmatched mass like tissue creation/destruction).
  - Pure $2 \times 2$ static block bootstrap for single ROI (rejected due to statistical collapse/zero-variance false confidence).
  - Dynamic median cost scaling per UOT instance (rejected due to dimensional distortion of $\lambda$).
- **Consequences**: This mathematically bounds the problem and guarantees solvability, but requires rigorous $\varepsilon$-scaling and log-domain Sinkhorn implementations to prevent numerical underflow in sparse high-dimensional data.
- **Review Trigger**: If UOT solver fails to converge on real IMC data, or if block bootstrap yields undefined confidence intervals.

### V1.5 Algorithm Pseudocode (Locked Blueprint)
*(Refer to the agreed Proposal V1.5 for the full specifications of Algorithm 1: End-to-end, Algorithm 2: Group-wise calibration, and Algorithm 3: SolveUOT + Decompose)*

## D002 — SLOTAR V1.6 Core Algorithm Architecture Upgrade
- Context: V1.5 metrics (L1 distance, absolute density, arbitrary tau) lacked strict physical units, tightly bounded unmatched mass extraction, and robust single-ROI topology preservation.
- Decision: Upgrade to V1.6 mathematical engine.
  1. **Area-weighted aggregation**: Use $\frac{\sum N}{\sum Area}$ for density to ensure strict cells/mm² physics semantics.
  2. **Positive-part metrics**: Use $(x-m)_+$ for Creation/Destruction to accurately isolate unmatched mass under UOT KL relaxation.
  3. **Group-wise calibration**: Implement baseline calibration for $\tau_g$ (Retention threshold) using pre ROI-ROI pairs within the same group.
  4. **Active set separation**: Decouple semantic pruning (tracked explicitly by `mass_pruned_ratio`) from numerical stability limits (`eta_floor`).
  5. **Frozen-feature Bootstrap**: Enforce single-ROI adaptive grid block bootstrap with frozen kNN/prototypes to prevent topological tearing.
  6. **Explicit mass-mode declaration**: Aggregation outputs must declare `mass_mode` explicitly; `count`, `density`, and `proportion` are compatible routes, but they must not be silently mixed.
- Alternatives: L1 metrics (rejected due to confounding marginal relaxation with creation), moving-block bootstrap (kept as secondary option, too complex for default).
- Consequences: Output data contracts must be strictly expanded to include mandatory audit fields (`mass_pruned_ratio`, `eps_schedule_id`, etc.) to guarantee traceability and reproducibility.
- Review Trigger: Failure of log-domain Sinkhorn convergence or unacceptable `mass_pruned_ratio` (>0.5%) triggering sensitivity degradation.

## D004 — Hard Boundary Isolation (Library vs. Tasks)
- **Context**: To maintain method rigor while preventing benchmark overfitting, placement rules must be explicit without introducing extra physical layers.
- **Decision**: Enforce a strict two-tier physical boundary only: `src/slotar/` and `tasks/`. Inside `src/slotar`, allow two logical categories: (1) method kernel modules (UOT, contracts, representation, UQ, utilities), and (2) benchmark-agnostic method-defining core inference modules that are part of SLOTAR itself. `tasks/` owns study-specific design/orchestration and benchmark-scoped modeling/evaluation.
- **Consequences**: Any hard-coded cohort names, benchmark-specific clinical semantics, task-specific formula construction, or reporting logic inside `src/slotar/` violates architecture and should fail review.

## D005 — Batched Unbalanced Optimal Transport (Engineering Throughput)
- **Context**: The existing Python `for`-loop over bootstrap replicates and $\lambda$ candidate grids introduces unacceptable overhead, leading to severe computational bottlenecks for large cohorts.
- **Decision**: Upgrade the mathematical engine to a tensor-based Batched Unbalanced Log-domain Sinkhorn solver. The pipeline must construct `[N, K]` tensors to execute UOT solves simultaneously across batch dimensions (patients, lambdas, or replicates).
- **Constraints**: This is strictly an engineering optimization. The mathematical estimands, baseline calibration rules, and fail-fast behaviors must remain strictly equivalent to V1.6.

## D008 — Representation Route and Canonical UOT Level
- **Context**: The repository needs a sharper distinction between the official reference representation route and allowed custom routes, without changing the stable UOT interface.
- **Decision**:
  1. The canonical UOT operating level is ROI-/sample-level nonnegative mass distributions over a shared state/prototype axis of length `K`; SLOTAR does not perform UOT directly on individual cells.
  2. The official representation route lives in `src/slotar` as a closed benchmark-agnostic library-supported path from spatial single-cell or spot data to SLOTAR-ready objects.
  3. In the official route, high-level input validation, local community representation construction, and shared state/prototype construction belong to `src/slotar`.
  4. In the official route, shared state/prototype construction and cost geometry construction are tightly coupled and must remain geometrically consistent; users following the official route do not manually invent the canonical `cost_matrix`.
  5. Official aggregation is performed at the ROI-/sample-level and must declare `mass_mode` explicitly. `count`, `density`, and `proportion` are compatible modes; density depends on ROI area metadata and preserves cells/mm² semantics where relevant.
  6. Task pipelines define baseline subsets, grouping context, and pairing metadata first; library calibration then uses that task-compatible context; task pipelines assemble final solver batches; library batched UOT runs last.
  7. Custom representation routes are allowed, but may only replace how raw data are converted into the shared state space, aggregated masses, or compatible cost geometry; they may not redefine the mathematical meaning of canonical SLOTAR inputs.
  8. Before entering UOT, any route must provide or imply a shared state axis, ROI-/sample-level masses, compatible cost geometry, required global scaling objects such as `s_C`, and a valid route to `lambda_pl`.
- **Consequences**: Representation choice remains flexible, but the official route is documented as a complete library-supported pipeline, the SLOTAR-ready contract entering UOT stays fixed, and pairing source/target items remains task-level orchestration.

## D009 — Baseline Comparator Framework and Incremental Value
- **Context**: Scientific claims about SLOTAR require an explicit statement of what simpler baselines already explain and what additional structure SLOTAR contributes.
- **Decision**:
  1. Any task claiming added value from SLOTAR MUST define explicit baseline comparators. At minimum, this includes a simple state-abundance comparator.
  2. Baseline abundance methods answer whether states increase, decrease, disappear, or persist on the declared physical mass scale of the task.
  3. SLOTAR adds shared transport geometry, transport cost/work, decomposition into retention / remodeling / creation / destruction, uncertainty semantics, and audit / failure semantics.
  4. The same-scale principle is mandatory: baseline abundance comparators and SLOTAR inputs must be computed on the same physical mass scale. If SLOTAR uses area-weighted density, the baseline comparator must use that same density scale.
  5. It is invalid to present simple abundance changes as if they were uniquely discoverable only by SLOTAR.
  6. Baseline comparison outputs remain task-scoped. They may be handed off through the compliant bridge path as optional task auxiliary tables, but they are not promoted to core canonical SLOTAR artifacts.
- **Consequences**: Tasks must make falsifiable incremental-value claims: baselines establish whether states change, while SLOTAR explains how those changes are organized in transport space and what transport-structured uncertainty/audit information accompanies them.

## D007 — Batched UOT Failure Semantics (Issue A Lock)
- **Context**: Historical docs and code mixed two failure layers: programmer-level contract violations and per-item data degeneracies inside a batch.
- **Decision**:
  1. Programmer-level contract violations (shape mismatch, negative mass, NaN/Inf, invalid lambda shape, invalid kernel shape) MUST fail fast by raising `DataContractError`.
  2. Per-item data degeneracies in `batched_uot_solve` MUST be batch-isolated and MUST NOT crash the whole batch.
  3. Canonical per-item `uot_status` vocabulary is locked to:
     - `ok`
     - `ERR_UOT_EMPTY_MASS_SOURCE`
     - `ERR_UOT_EMPTY_MASS_TARGET`
     - `ERR_UOT_EMPTY_SUPPORT`
     - `ERR_UOT_NUMERICAL`
  4. Batch output shape `[N]` is preserved; failed items are not removed in library code; if `status != "ok"`, all micro UOT metrics are `NaN`.
  5. Task layer (not solver) maps `uot_status` to `bypass_reason` and omits non-`ok` rows from downstream inference.
- **Consequences**: Library contracts become strictly testable and deterministic, while task pipelines retain responsibility for inference-level missingness handling.

## D006 — Hurdle + Measurement Error Joint Model (SLOTAR V2.0 Inference)
- **Context**: Traditional Inverse Variance Weighting (IVW) on aggregated UQ estimates suffers from severe instability ("weight explosion") when local variance $\sigma_i^2$ approaches zero, and requires heuristic truncations that alter the inference estimand.
- **Decision**: Adopt a Hurdle + Measurement Error Joint Model. The parameter estimate $\hat{\theta}_i$ and its bootstrap-derived uncertainty are ingested directly into the likelihood function as a measurement error component $e_i \sim \mathcal{N}(0, \phi^2 + s_i^2)$.
- **Placement Clarification**: If this model is treated as part of the SLOTAR method definition, it belongs to the benchmark-agnostic core inference layer in `src/slotar`. Task pipelines remain responsible for endpoint definitions, covariate construction, subgroup analyses, and reporting.
- **Constraints (Locked)**:
  1. **Log-scale Empirical Variance**: $s_i^2$ must be computed empirically on the log-transformed bootstrap replicates: $s_i^2 := \text{Var}(\log(\hat{\theta}_i^{(b)} + \delta))$, not via Delta method on infinitesimal point estimates.
  2. **Numerical Stabilizer Bounds**: A strict numerical lower bound (`s2_lower_bound`) must be enforced on $s_i^2$ to prevent underestimation of total uncertainty.
  3. **Identifiability Diagnostics**: The pipeline must evaluate the heterogeneity of the $\{s_i^2\}$ distribution before drawing inference claims.
  4. **Strict Zero Stratification**: Only biologically justified "True Zeros" (e.g., compartment collapse) may enter the hurdle model's zero component. Engineering failures or pruning-induced zeros must be strictly logged as `NaN` (Missing Data).
  5. **Bias-Variance Boundary**: This model absorbs sampling variance, not systemic sampling bias.

## D010 — Task A Arm-III Documentation Boundary and Density-Primary Lock
- **Context**: Task-A Arm-II startup was intentionally implemented as a count-only startup slice on the frozen Stage-0 artifact so the ordered-pair scaffold, same-pair Balanced OT comparator, and initial Arm-II interpretation could be audited before the final coverage-stress contract was locked.
- **Decision**: Task-A Arm-III is now locked as a density-primary coverage-reduction stress test aligned to SLOTAR V1.6 area-weighted physical semantics. The execution contract lives in `docs/task_A_spec.md`; the historical rationale for the Arm-II startup slice and the Arm-III density-primary lock lives here in `docs/decisions.md`.
- **Consequences**: The Task-A spec stays operational and non-historical, while the repo retains a concise record of why Arm-II startup remained count-only and why Arm-III now uses density as the primary mass semantics.
