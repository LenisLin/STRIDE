# Task A Arm-3 Local Manifest

> Historical note: preserved Arm-3 planning/implementation artifact. Live methodology: `docs/task_A_spec.md`. Live API: `docs/api_specs.md`. Live current-output contracts: `docs/data_contracts.md`.

## A. Exact local files and paths

### Operative frozen Stage-0 bundle used by current Task A

- Canonical current Stage-0 `.h5ad`:
  - `/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad`
  - Evidence: this is the default `DEFAULT_STAGE0_PATH` in `tasks/task_A/analyze_arm2_results.py`, and the current focused memo at `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/00_arm2_focused_results_memo.md` records the same path.
- Stage-0 sidecars in the same canonical bundle:
  - `/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_roi_clinical.parquet`
  - `/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_validation.json`

### Historical Stage-0 copies present locally but not referenced by current Arm-II code

- `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/stage0_arm1/task_A_stage0_k25.h5ad`
- `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/stage0_arm1_rebuild/task_A_stage0_k25.h5ad`
- Concrete distinction:
  - Canonical current file size: `586,505,471` bytes.
  - Historical copies: `318,583,666` bytes each.
- Inference from current code and current focused memo: the operative current Task A artifact is the `ProjectData` copy, not either `ProjectResult` snapshot.

### Current Arm-II metrics parquet

- `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/task_A_metrics.parquet`
- Concrete facts from the live parquet:
  - `1710` rows, `40` columns.
  - `arm` values: only `A2_cross_compartment`.
  - `pair_family` counts: `TC-IM=558`, `IM-PT=630`, `TC-PT=522`.
  - `tau_mode='unavailable'`.
  - `R` is all `NA`.
  - `tau` is all `NA`.
  - `lambda_pl` is currently `10.0` for every row in all three pair families.

### Current focused analysis outputs

- Focused output directory:
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused`
- Current materialized files:
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/00_arm2_focused_results_memo.md`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/01_prototype_biological_meaning_table.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/02_baseline_pair_audit.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/03_baseline_prototype_confirmatory_summary.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/04_baseline_patient_family_confirmatory_summary.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/05_global_transport_summary.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/06_key_prototype_comparison.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/07_key_prototype_patient_recurrence.csv`
  - `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/arm2_cross_compartment/analysis/focused/08_minimal_appendix_audit.csv`

## B. Frozen artifact schema

Artifact inspected: `/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad`

### High-level counts

- Cells / obs rows: `1,873,179`
- ROI categories: `288`
- Patient categories: `32`
- Compartment categories: `3`
- Active prototype IDs: `25`
- Community-feature dimension: `22`

### `obs` columns actually present

- Index key: `CellID`
- Stored columns:
  - `patient_id`
  - `timepoint`
  - `roi_id`
  - `compartment`
  - `cell_type`
  - `cell_area`
  - `proto_id`

### `obsm` keys actually present

- `spatial`
  - shape: `(1873179, 2)`
- `community_features`
  - shape: `(1873179, 22)`

### `uns` keys actually present

- `cost_matrix`
  - shape: `(25, 25)`
- `prototype_centroids`
  - shape: `(25, 22)`
- `roi_areas`
  - dict-like group with `288` ROI keys
- `s_C`
  - scalar
- `scaler_params`
  - nested keys: `center`, `feature_names`, `scale`

### `proto_id`

- Already exists.
- Location: `obs/proto_id`
- Stored dtype: `int64`
- Observed range: `0` to `24`

### `block_id`

- Does not exist in the frozen artifact.
- Absent from `obs`.
- Absent from `obsm`.
- Absent from `uns`.

### ROI areas

- ROI areas already exist.
- Location: `uns/roi_areas`
- Stored as a dict-like HDF5 group keyed by `roi_id`
- Entry count: `288`
- Observed values: all current entries are `1.0`
- Important distinction:
  - `obs/cell_area` exists as a per-cell field.
  - There is no per-block area field.
  - There is no nontrivial per-ROI effective area field beyond `uns/roi_areas`.

### HDF5 storage detail relevant to implementation

- `patient_id`, `roi_id`, `compartment`, and `cell_type` are stored as categorical groups with `codes` and `categories`.
- `proto_id`, `timepoint`, `cell_area`, and `CellID` are stored as datasets.
- Current Arm-II focused loader already uses this HDF5 layout directly via `h5py` rather than requiring `anndata`.

## C. Existing reusable Task A code paths

### Pair-generation utilities

- `tasks/task_A/arm1_noise_baseline.py`
  - `build_arm1_roi_table` at line `21`
  - `generate_anchored_arm1_slots` at line `29`
  - `generate_within_compartment_pairs` at line `124`
- `tasks/task_A/arm1_broken_reference.py`
  - `generate_broken_reference_pairs` at line `26`
- `tasks/task_A/arm2_spatial_gradient.py`
  - `ORDERED_PAIR_SPECS` at line `22`
  - `PAIR_FAMILIES` at line `30`
  - `generate_cross_compartment_pairs` at line `105`
- `tasks/task_A/arm2/analysis_io.py`
  - `build_pair_metadata` reconstructs ordered Arm-II pair metadata from the current metrics parquet
  - `reconstruct_pair_tensors` at line `469`

### Arm-II lambda calibration helpers

- `tasks/task_A/arm2_spatial_gradient.py`
  - `run_arm2` at line `33`
  - For each `pair_family`, it calls the shared calibration helper and then broadcasts one family-wide `lambda_pl` across all rows in that family.
- `src/slotar/uot.py`
  - `calibrate_joint_lambda` at line `78`
  - Current behavior: scans `lambda_grid`, runs `batched_uot_solve`, computes family median unmatched ratio, and chooses the candidate closest to `target_alpha`.
- Live calibrated state in the current parquet:
  - `TC-IM -> 10.0`
  - `IM-PT -> 10.0`
  - `TC-PT -> 10.0`

### Balanced OT comparator code

- `tasks/task_A/common.py`
  - `run_balanced_ot_batch` at line `166`
  - Current rowwise comparator: same-pair, shape-only, `ot.emd2`, scaled cost domain.
- `tasks/task_A/arm2_spatial_gradient.py`
  - `run_arm2` attaches rowwise `M_balanced` to the live metrics parquet.
- `tasks/task_A/arm2/analysis_compute.py`
  - `rerun_balanced_ot` at line `342`
  - This is the richer focused-analysis comparator: it reconstructs full balanced plans, edge shares, and `M_balanced` for the fixed Arm-II pair set.

### Batch tensor / solver wrappers already in Task A

- `tasks/task_A/common.py`
  - `assemble_tensors` at line `55`
  - `run_uot_batch_safe` at line `107`
- `src/slotar/uot.py`
  - `precompute_logKernels` at line `56`
  - `batched_uot_solve` at line `464`

### Frozen Stage-0 loading and reconstruction already in Task A

- `tasks/task_A/arm2/analysis_io.py`
  - `load_stage0_analysis_bundle` at line `322`
  - `load_inputs` at line `521`
- Concrete reuse value:
  - This loader already reads `obs/proto_id`, ROI categorical codes, patient categorical codes, compartment categorical codes, and cell-type categorical codes directly from the `.h5ad` with `h5py`.
  - It already reconstructs ROI-by-prototype count vectors from the frozen artifact.

### Existing parquet / output writers

- `tasks/task_A/pipeline.py`
  - `TEMPORARY_METRICS_FILENAME = "task_A_metrics.parquet"` at line `21`
  - direct write at line `128`
  - Current Task A metrics parquet writing is just `df_metrics.to_parquet(...)`.
- `tasks/task_A/build_stage0_artifacts.py`
  - writes `task_A_stage0_roi_clinical.parquet`
- `tasks/task_A/arm2/analysis_output.py`
  - `write_output_package` at line `443`
  - Writes focused outputs as one markdown memo plus CSV tables, not parquet.
- `src/slotar/io/bridge.py`
  - `save_for_r` exists, but it is still `NotImplementedError`, so it is not a usable current writer path.

## D. Gaps that Arm-3 still needs

- No frozen `block_id` exists in the canonical Stage-0 `.h5ad`.
- No frozen block partition exists anywhere in the current Task A artifact bundle.
- No block-area store exists.
- Current `uns/roi_areas` is present but trivial: all `288` values are `1.0`.
- Current Task A tensor assembly is ROI-level and count-only.
  - `tasks/task_A/common.py::assemble_tensors` reconstructs ROI vectors from `obs['proto_id']`.
  - It does not support pseudo-ROI reconstruction, density mass, sampled block areas, or frozen support masks per reference pair.
- Current full-coverage calibration state is only partially available for Arm-3:
  - `lambda_pl` is present in the current Arm-II parquet.
  - `tau` calibration is not present.
  - `R` is unavailable in the current Arm-II startup slice.
- Current pair-generation logic emits all six ordered Arm-II directions.
  - Arm-III spec wants primary ordered-anchor analysis on `TC->IM` and `TC->PT`.
  - That anchor-direction filter does not yet exist as a dedicated Arm-III utility.
- `tasks/task_A/arm3_uq_stress.py` is not implementation-ready:
  - helper functions are `pass`
  - it sketches only a rough coverage loop
  - it references `np.ndarray` without importing `numpy`
  - it does not yet load frozen full-coverage lambdas, build blocks, construct pseudo-ROIs, freeze support, or write outputs
- `src/slotar/uq.py::bootstrap_single_roi` is also still a stub.
  - There is no working reusable block-bootstrap library in `src/slotar`.
- There is no current Arm-3 output contract, no Arm-3-focused writer, and no current Arm-3 result directory in the live result tree.

## E. Recommended implementation entrypoints

- Top-level Arm-3 runner:
  - `tasks/task_A/arm3_uq_stress.py`
  - Recommendation: treat this as the current intended entrypoint, but it needs real implementation work before it can run.
- Frozen artifact loading:
  - start from `tasks/task_A/arm2/analysis_io.py::load_stage0_analysis_bundle`
  - Reason: it already handles the actual HDF5 categorical layout without requiring `anndata`.
- Frozen ordered pair logic:
  - reuse `tasks/task_A/arm2_spatial_gradient.py::ORDERED_PAIR_SPECS`
  - reuse `tasks/task_A/arm2_spatial_gradient.py::generate_cross_compartment_pairs`
  - then add an Arm-3 layer that filters to the primary ordered anchors `TC->IM` and `TC->PT` while leaving reverse/audit directions optional.
- Full-coverage solver / comparator reuse:
  - `tasks/task_A/common.py::run_uot_batch_safe`
  - `tasks/task_A/common.py::run_balanced_ot_batch`
  - `src/slotar/uot.py::calibrate_joint_lambda`
- If Arm-3 needs post-hoc pair/prototype comparison tables similar to current Arm-II:
  - reuse `tasks/task_A/arm2/analysis_compute.py::rerun_uot`
  - reuse `tasks/task_A/arm2/analysis_compute.py::rerun_balanced_ot`
  - reuse `tasks/task_A/arm2/analysis_compute.py::build_pair_level_transport_frame`
- Output path convention:
  - follow the active external result root under `/mnt/NAS_21T/ProjectResult/SLOTAR/task_A/`
  - do not write generated Arm-3 artifacts into the repository tree
  - if Arm-3 needs parquet output before bridge/export work is revived, the only currently implemented pattern is the direct `DataFrame.to_parquet(...)` style used in `tasks/task_A/pipeline.py`
