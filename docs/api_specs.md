# API Specifications

## Module 1: Representation and SLOTAR-Ready Input Construction
*Note: This module defines the official benchmark-agnostic route from spatial single-cell or spot data to SLOTAR-ready objects. UOT is defined on ROI-/sample-level nonnegative mass distributions over a shared state/prototype axis, not on individual cells. The currently documented official example route uses `.obsm['spatial']` plus `.obs['cell_type']`, but the general contract is defined by the SLOTAR-ready objects below. Custom representation routes are allowed only if they terminate in the same validated SLOTAR-ready contract.*

### `build_community_features(adata: AnnData, k: int = 30) -> None`
- **Inputs**: `adata` with `.obsm['spatial']` and `.obs['cell_type']`.
- **Side Effects**: Computes $\tilde{u}_c$ and stores it in `adata.obsm['community_features']`. Fails fast if coordinates contain NaNs.
- **Constraints**: Uses exact kNN. Computes density $\delta_c = k / (\pi \cdot r_k^2)$.

### `learn_global_prototypes(adata: AnnData, n_bal: int, K: int) -> None`
- **Side Effects**: Computes balanced sample $\mathcal{U}_{bal}$, runs KMeans, assigns `proto_id` to all cells in `adata.obs`. Establishes the shared state/prototype axis used for downstream ROI-/sample-level aggregation and cost construction. Computes and stores $s_C$ in adata.uns['s_C'].

### Official Route Pipeline (Ordered)
1. **Validate official input route**: Apply high-level AnnData contract checks for the official route before representation construction.
2. **Construct local community representation**: Build benchmark-agnostic local features from raw spatial single-cell or spot data.
3. **Construct shared state space and official cost geometry as one coupled phase**: Learn the shared prototype/state axis of length `K` and construct the canonical `cost_matrix` on that same axis inside `src/slotar`; these two objects are kept geometrically consistent and are not documented as independently user-editable official steps.
4. **Aggregate to ROI-/sample-level mass tables**: Convert each ROI/sample into a nonnegative mass vector over the shared axis with explicit `mass_mode` semantics. Supported documented modes are `count`, `density`, and `proportion`; density mode depends on ROI area metadata and preserves declared cells/mm² semantics where relevant.
5. **Prepare a calibration-compatible path**: Keep the resulting shared-axis masses and scaling objects compatible with downstream lambda calibration in the library.
- **Boundary**:
  - The official route ends at SLOTAR-ready objects, not at paired `A/B` tensors.
  - Pairing source/target items into batched `A` and `B` tensors remains task-level orchestration.
  - A custom route may replace the raw-data-to-state-space mapping, aggregation, or compatible cost construction, but it must preserve the same shared-axis semantics and validation contract.

## Module 2: UOT Engine & Calibration (`slotar.uot`)

### `calibrate_lambdas(adata: AnnData, target_alpha: float = 0.05) -> Tuple[Dict, Dict]`
- **Inputs**: Annotated `adata` with prototype counts or equivalent ROI-/sample-level masses on the shared state axis, together with task-defined baseline-compatible grouping context.
- **Outputs**: Returns two dictionaries `{group: lambda}` for density and shape levels.
- **Constraints**: Uses baseline-compatible task context (for example, subset/group definitions) rather than running in isolation before task metadata is defined. Calibration precedes final batch assembly, but depends on task-provided context.

### Calibration and Pairing Order
1. **Task layer defines baseline subsets / grouping / pairing metadata**.
2. **Library calibration uses that baseline-compatible task context** to obtain lambdas or a valid path to `lambda_pl`.
3. **Task layer assembles final solver batches** as `A`, `B`, and `lambda_pl`.
4. **Library batched UOT consumes the final `[N, K]` tensors**.

### `precompute_logKernels(C: np.ndarray, eps_schedule: Sequence[float], s_C: float = 1.0) -> list[np.ndarray]`
- **Inputs**:
  - `C`: Cost matrix on the shared state/prototype axis.
  - `eps_schedule`: Epsilon-scaling schedule used for batched UOT.
  - `s_C`: Positive finite cost-scale divisor applied before kernel precomputation.
- **Preconditions (Strict)**:
  - `C` MUST be array-like, numeric, finite, 2D, and square.
  - `eps_schedule` MUST be a non-empty positive 1D schedule.
  - `s_C` MUST be finite and strictly positive.
  - Invalid programmer-level inputs MUST raise `DataContractError`.
- **Outputs**:
  - `kernels`: List of precomputed log-domain kernels, one per epsilon in the schedule.
- **Constraints**:
  - This is the kernel-preparation step used before `batched_uot_solve(...)`.
  - Caller responsibility for supplying `C` and `s_C` is stage-conditional: once a pipeline enters kernel precomputation / UOT execution, the shared-axis cost geometry must already be available.

### `batched_uot_solve(A: np.ndarray, B: np.ndarray, lambda_pl: np.ndarray, kernels: list, solver_config: dict) -> Tuple[dict, np.ndarray]`
- **Inputs**: 
  - `A`, `B`: Non-negative ROI-/sample-level mass tensors of shape `[N, K]` on the shared state/prototype axis (where `N` is the batch dimension: paired ROI/sample items, lambdas, or bootstrap replicates).
  - `lambda_pl`: Array of shape `[N]` containing the regularization parameter for each batch item.
  - `kernels`: Precomputed log-domain kernels for the $\varepsilon$-scaling schedule.
- **Preconditions (Strict)**: 
  - `validate_uot_inputs(...)` MUST run before solve logic.
  - Programmer-level contract violations MUST raise `DataContractError` (shape mismatch, negative mass, NaN/Inf, invalid `lambda_pl` shape, invalid kernel shape).
  - Per-item data degeneracies MUST NOT crash the batch and MUST be reported via per-item `status`.
- **Outputs**: 
  - `metrics_dict`: Dictionary of batched tensors for `T`, `B_pos`, `D_pos`, `d_rel`, `b_rel`, `M`, `R`, `tau`.
  - `status_array`: Array of shape `[N]` with values:
    - `"ok"`
    - `"ERR_UOT_EMPTY_MASS_SOURCE"`
    - `"ERR_UOT_EMPTY_MASS_TARGET"`
    - `"ERR_UOT_EMPTY_SUPPORT"`
    - `"ERR_UOT_NUMERICAL"`
- **Constraints**: 
  - Batch dimension `[N]` MUST be preserved; failed items are not dropped inside library code.
  - If `status_array[i] != "ok"`, all micro metrics for item `i` MUST be `NaN`.

## Module 3: Uncertainty Quantification (`slotar.uq`)

### `bootstrap_single_roi(adata: AnnData, roi_id: str, G: int, B_boot: int) -> dict`
- **Inputs**: Subset of `adata` corresponding to a single ROI.
- **Outputs**: Dictionary of bootstrap replicates and the log-scale empirical variance.
- **Constraints**: 
  - MUST enforce frozen representations (Cannot recompute kNN or prototypes).
  - MUST compute the empirical measurement error strictly as $s_i^2 := \text{Var}(\log(\hat{\theta}_i^{(b)} + \delta))$, ensuring numerical bounds are applied per V2.0 contracts.

## Module 3B: Core Inference Layer (Benchmark-Agnostic, `src/slotar`)

### `run_core_inference(first_order_outputs: Mapping[str, Any], model_config: Mapping[str, Any]) -> CoreInferenceResult`
- **Role**: Defines benchmark-agnostic method-level inference logic that is part of SLOTAR itself (when such inference is treated as core method definition).
- **Inputs**:
  - `first_order_outputs`: Canonical SLOTAR first-order outputs (metrics/events plus required audit fields).
  - `model_config`: Explicitly passed inference hyperparameters from the task layer (no config parsing in library code).
- **Outputs**:
  - `CoreInferenceResult`: Standardized Python-side result contract containing in-memory estimates, uncertainty summaries, diagnostics, and run metadata.
- **Constraints**:
  - MUST remain benchmark-agnostic (no hard-coded cohort names, no benchmark-specific clinical assumptions, no task-specific formula construction).
  - MUST NOT perform reporting/visualization behavior.
  - This API section defines object-level contracts only; it does not define new file export formats.

### ST Modality Adaptation (Visium)
- **Inputs**: ST embedding (e.g., PCA on HVGs).
- **Adaptation**: 
  - $\mathbf{m}_c$: PCA vector of the spot.
  - $\bar{\mathbf{m}}_c$: kNN mean PCA vector ($k=20$).
  - $\delta_c$: Treated as constant (`delta_mode="const"`).
  - $\mathbf{p}_c$: Zero vector (`p_mode="zero"`) by default.
- **Constraints**: Enforces physical mapping without unsupported biological proxy claims.

## Module 4: Domain-Agnostic Utilities (`slotar.utils`)

### `build_grouping(adata, group_key: Optional[str] = None) -> np.ndarray`
- **Inputs**: `group_key` (column name in `adata.obs`).
- **Logic**: If `group_key` is None or not provided, MUST fallback to assigning all observations to a single group: `g="all"`. Implicit ROI-state clustering is strictly forbidden. 
- **Outputs**: Group assignments array.

### `compute_active_mask(mass_source: np.ndarray, mass_target: np.ndarray, n_min_proto: float) -> Tuple[np.ndarray, float]`
- **Inputs**: Source and target mass vectors (agnostic to temporal pre/post semantics).
- **Logic**: Mathematical pure mask `active_mask = (mass_source + mass_target >= n_min_proto)`. 
- **Outputs**: `active_mask` (boolean array) and `mass_pruned_ratio` (float).

### `flag_drift(events: dict, z: np.ndarray, drift_vector: Optional[np.ndarray] = None, thr: float = 0.85) -> dict`
- **Inputs**: `drift_vector` MUST be computed by the upstream task pipeline and passed explicitly. The library performs NO automatic drift estimation.
- **Logic**: If `drift_vector` is None, skips cosine computation.
- **Outputs**: Returns events with `drift_aligned` flags. If `drift_vector` is None, sets `drift_aligned = null` and signals unavailable mode.
