# Task B TONIC Data Ecosystem Audit

This file is preserved as reusable dataset and ecosystem background for a
possible future Task B revisit. It is not the live task-execution surface.

## 1. Current code/spec fact

- `Spec fact:` [docs/task_B_spec.md](task_B_spec.md) defines the sole formal Task B cohort as the curated 40-patient `pre_nivo__on_nivo` subset in processed TONIC data.
- `Spec fact:` Task B pairing is patient-level sample pairing keyed by `Patient_ID`, `Timepoint`, and sample-level `Tissue_ID` or equivalent processed sample metadata.
- `Spec fact:` Task B is not lesion-matched and not FOV-to-FOV matched. Multiple FOVs under one sample must be aggregated at the sample level and retained only for FOV-aware uncertainty/sensitivity logic.
- `Spec fact:` The task-scoped processed compartment axis is `cancer_core`, `cancer_border`, `stroma_core`, `stroma_border`, and `immune_agg`.
- `Spec fact:` Canonical Task B masses must be reconstructed from processed cell-level data, cluster/state annotations, compartment assignments, and sample metadata.
- `Spec fact:` Density is the default physical scale because area metadata are expected to be available in TONIC. Count/proportion fallback is allowed only if area is irrecoverable for a subset and both baseline and STRIDE use the same fallback.
- `Spec fact:` Summary feature tables containing ratios, log-ratios, differences, normalized values, or any negative-valued summaries must not be used directly as transport masses.
- `Spec fact:` Public/local TONIC metadata are not assumed to provide reliable organ-site, biopsy-site, or lesion-location matching fields.

## 2. TONIC filesystem inventory

- `Current file fact:` `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort` is a flat processed-data root organized into `analysis_files/` and `intermediate_files/`; it is not a raw patient-directory hierarchy.
- `Current file fact (direct):` The root contains 34 CSV files `(direct)` and 1 H5AD file `(direct)`. No `.parquet`, `.xlsx`, `.tsv`, `.json`, `.yaml`, `.yml`, `.pkl`, `.rds`, `.loom`, `.md`, or `.txt` files were found under the TONIC root.
- `Current file fact:` No local README, schema note, or sidecar text file was found under the TONIC root.
- `Current file fact:` `analysis_files/harmonized_metadata.csv` and `intermediate_files/metadata/harmonized_metadata.csv` are byte-identical duplicate copies.

| Evidence type | Path | Format | Apparent purpose | Grain |
| --- | --- | --- | --- | --- |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/metadata/TONIC_data_per_patient.csv` | CSV | Patient-level cohort-flag table (`primary__baseline`, `baseline__pre_nivo`, `baseline__on_nivo`, `pre_nivo__on_nivo`) | patient |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/metadata/TONIC_data_per_timepoint.csv` | CSV | Sample/timepoint manifest with `Tissue_ID`, `Patient_ID`, `Timepoint`, `MIBI_data_generated`, cohort flags | sample/timepoint |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/harmonized_metadata.csv` | CSV | Primary FOV/core metadata with `fov`, `Tissue_ID`, `Patient_ID`, `Timepoint`, `rna_seq_sample_id`, `MIBI_data_generated`, cohort flags | FOV/core |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/metadata/TONIC_data_per_core.csv` | CSV | FOV/core metadata plus TMA coordinates and pathology screen (`TMAID`, `X_pos`, `Y_pos`, `Pathologist_assessment_HE`) | FOV/core |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/metadata/imaged_fovs.csv` | CSV | List of FOVs that survive into the processed cell/area layer | FOV/core |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/cell_table_clusters.csv` | CSV | Minimal processed cell table with `fov`, `label`, and three state/cluster axes | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/cell_table_counts.csv` | CSV | Per-cell marker-intensity table with state columns | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/cell_table_morph.csv` | CSV | Per-cell morphology table with `area` and nuclear morphology fields | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/cell_table_func_all.csv` | CSV | Per-cell functional-marker and double-positive summary table | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/cell_table_func_single_positive.csv` | CSV | Per-cell functional-marker single-positive summary table | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/combined_cell_table_normalized_cell_labels_updated.csv` | CSV | Wide processed cell table including intensities, morphology, `fov`, `label`, and state labels | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/adata_processed.h5ad` | H5AD | Processed cell object with `obs` fields including `fov`, per-cell `area`, `compartment`, and state axes | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/mask_dir/cell_annotation_mask.csv` | CSV | Cell-to-compartment map keyed by `fov`,`label` with `mask_name` | cell |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/intermediate_files/mask_dir/fov_annotation_mask_area.csv` | CSV | FOV-level compartment area table with `fov`,`compartment`,`area` | FOV/compartment |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/feature_metadata.csv` | CSV | Metadata for derived summary features (`feature_type`, `compartment`, `cell_pop_level`) | summary |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/combined_feature_data.csv` | CSV | FOV-level derived summary features with raw and normalized values | summary |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/combined_feature_data_filtered.csv` | CSV | Filtered FOV-level derived summary features | summary |
| `current file fact` | `/mnt/NAS_21T/ProjectData/SLOTAR/TONIC_Cohort/analysis_files/timepoint_features.csv` | CSV | Tissue/timepoint-level derived summary feature table | summary |

## 3. Cohort-scale summary

| Scope | Patients | Samples / tissues | FOVs / ROIs | Cells | Timepoints observed | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Patient-flag table | 117 `(direct)` | not applicable | not applicable | not applicable | not applicable | `TONIC_data_per_patient.csv` contains 117 `Patient_ID` values. |
| Sample/timepoint metadata | 113 `(direct)` | 553 `(direct)` | not applicable | not applicable | `baseline`, `pre_nivo`, `on_nivo`, `primary`, `primary_other`, `metastasis_1`, `lymphnode_neg`, `local_recurrence`, `biopsy`, `lymphnode_pos`, `metastasis`, `on_nivo_1_cycle`, `progression`, `metastasis_2`, `metastasis_3`, `lymphnode` `(direct)` | From `TONIC_data_per_timepoint.csv`. |
| FOV/core metadata | 113 `(direct)` | 394 unique `Tissue_ID` `(direct)` | 1,144 `(direct)` | not applicable | `baseline`, `primary`, `pre_nivo`, `on_nivo`, `primary_other`, `local_recurrence`, `metastasis_1`, `biopsy`, `on_nivo_1_cycle`, `metastasis_2`, `metastasis` `(direct)` | From `harmonized_metadata.csv`. |
| `MIBI_data_generated=True` subset of FOV/core metadata | 112 `(direct)` | 356 unique `Tissue_ID` `(direct)` | 949 `(direct)` | not applicable | Same timepoint vocabulary as `harmonized_metadata.csv`, but only rows with generated MIBI data `(direct)` | This is broader than the processed cell layer. |
| Processed cell/area intersection (`harmonized_metadata` intersect `cell_table_clusters` intersect `fov_annotation_mask_area`) | 111 `(direct)` | 294 unique `Tissue_ID` `(direct)` | 678 `(direct)` | 1,984,723 `(direct)` | `baseline`, `primary`, `pre_nivo`, `on_nivo` `(direct)` | `cell_table_clusters.csv`, `imaged_fovs.csv`, and `fov_annotation_mask_area.csv` all cover the same 678 FOVs. |
| Formal Task B processed pair slice (`pre_nivo__on_nivo=True`, processed FOV support, `Timepoint in {pre_nivo,on_nivo}`) | 40 `(direct)` | 80 `(direct)` | 163 `(direct)` | not directly counted | `pre_nivo`, `on_nivo` `(direct)` | 81 pre-nivo FOVs and 82 on-nivo FOVs `(direct)`; all 40 patients have both timepoints `(direct)`. |

- `Current file fact (direct):` The processed cell-supported timepoints are only `baseline`, `primary`, `pre_nivo`, and `on_nivo`.
- `Current file fact (direct):` Broader metadata tables contain additional timepoint names that do not survive into the processed cell layer, including `primary_other`, `local_recurrence`, `metastasis_1`, `metastasis_2`, `metastasis`, `biopsy`, `on_nivo_1_cycle`, `progression`, `lymphnode_neg`, `lymphnode_pos`, `metastasis_3`, and `lymphnode`.
- `Current file fact (direct):` The patient-level flag table is not perfectly aligned with downstream tables: `Patient_ID` values `112`, `114`, `115`, and `117` appear in `TONIC_data_per_patient.csv` but not in the sample/timepoint or FOV/core tables.
- `Inference:` The operative downstream cohort universe for actual sample/FOV design is 113 patients, not 117, because only 113 patients appear in the sample/timepoint and FOV/core metadata.

## 4. Clinical information audit

| Variable class | Availability | Evidence | Notes |
| --- | --- | --- | --- |
| Patient identifier | present | `current file fact` | `Patient_ID` is present in patient, sample/timepoint, FOV/core, and cohort-flag tables. |
| Sample / tissue identifier | present | `current file fact` | `Tissue_ID` is present in sample/timepoint and FOV/core tables. |
| Timepoint / treatment-phase label | present | `current file fact` | `Timepoint` is present in sample/timepoint and FOV/core tables and includes treatment-proximal labels such as `pre_nivo`, `on_nivo`, and `on_nivo_1_cycle`. |
| Generated-MIBI availability flag | present | `current file fact` | `MIBI_data_generated` is present in sample/timepoint and FOV/core tables. |
| RNA-seq linkage field | present but partially missing | `current file fact` | `rna_seq_sample_id` exists in `harmonized_metadata.csv`; 506 processed FOV rows `(direct)` have a non-empty value and 172 `(direct)` do not. |
| Cohort flags | present | `current file fact` | `primary__baseline`, `baseline__pre_nivo`, `baseline__on_nivo`, `pre_nivo__on_nivo` are present in patient, sample/timepoint, and FOV/core tables. |
| Tissue / block context | present | `current file fact` | `Tissue_ID_with_block` is present in `TONIC_data_per_core.csv`. |
| Pathology screening context | present | `current file fact` | `Pathologist_assessment_HE` is present in `TONIC_data_per_core.csv`; common values include `Include PA`, `No Tissue`, `Too few cells`, and `No Tumor`. |
| TMA grid / core coordinates | present | `current file fact` | `Study`, `MIBI_DATA_ACQUIRED`, `TMAID`, `X_pos`, and `Y_pos` are present in `TONIC_data_per_core.csv`. |
| Response-related field | absent | `current file fact` | No local CSV header matched response, responder, RECIST, or benefit terminology. |
| Survival / follow-up field | absent | `current file fact` | No local CSV header matched survival, death, follow-up, PFS, or OS terminology. |
| Organ-site / lesion-site / biopsy-site field | absent as explicit field | `current file fact` | No local CSV header matched organ-site, lesion-site, or biopsy-site terminology. |

- `Current file fact (direct):` `TONIC_data_per_timepoint.csv` includes `progression` as a timepoint label, but no explicit response-outcome field accompanies it.
- `Inference:` The available clinical layer is sufficient for sample pairing and treatment-phase filtering, but not for response-stratified or organ-site-aware Task B design.
- `Unresolved issue:` Timepoint labels such as `metastasis_1`, `metastasis_2`, `lymphnode_neg`, and `progression` likely encode clinical context, but there is no explicit dictionary in the local root to formalize them beyond string labels.

## 5. Hierarchy and keying audit

- `Current file fact:` The recoverable hierarchy is relational rather than directory-based: `patient -> sample/tissue -> FOV/core -> cell`.

| Level | Key columns | Primary supporting file(s) | Evidence type | Notes |
| --- | --- | --- | --- | --- |
| patient | `Patient_ID` | `TONIC_data_per_patient.csv`, `TONIC_data_per_timepoint.csv`, `harmonized_metadata.csv` | `current file fact` | Patient-level cohort flags live in `TONIC_data_per_patient.csv`. |
| sample / tissue | `Tissue_ID`, `Patient_ID`, `Timepoint` | `TONIC_data_per_timepoint.csv`, `harmonized_metadata.csv`, `TONIC_data_per_core.csv` | `current file fact` | `Tissue_ID` is the concrete sample key used downstream. |
| FOV / ROI | `fov` | `harmonized_metadata.csv`, `TONIC_data_per_core.csv`, `imaged_fovs.csv` | `current file fact` | `fov` is unique per FOV/core row in the processed-support intersection. |
| cell | `fov`, `label` | `cell_table_clusters.csv`, `cell_table_counts.csv`, `cell_table_morph.csv`, `cell_annotation_mask.csv` | `current file fact` | Cell tables share `fov`,`label` as the join key. |
| cell compartment | `fov`, `label`, `mask_name` or H5AD `obs.compartment` | `cell_annotation_mask.csv`, `adata_processed.h5ad` | `current file fact` | `mask_name` carries the same five processed compartment categories used by the H5AD. |
| FOV compartment area | `fov`, `compartment` | `fov_annotation_mask_area.csv` | `current file fact` | Contains area denominators for the five task compartments plus `all` and `empty_slide`. |
| state / cluster axis | `cell_cluster`, `cell_cluster_broad`, `cell_meta_cluster` | `cell_table_clusters.csv`, `adata_processed.h5ad` | `current file fact` | Three usable state/cluster axes are present. |

- `Current file fact (direct):` The processed cell tables do not carry `Patient_ID` or `Tissue_ID`; those must be recovered by joining `fov` through `harmonized_metadata.csv` or `TONIC_data_per_core.csv`.
- `Current file fact (direct):` In the processed intersection, `fov` is unique per metadata row and `Tissue_ID` is not reused across different patient/timepoint combinations.
- `Current file fact (direct):` In the broader `MIBI_data_generated=True` metadata, 1 patient/timepoint combination `(direct)` is multi-tissue: `Patient_ID=81`, `Timepoint=local_recurrence`, with tissues `T16-05179` and `T20-61039`.
- `Current file fact (direct):` In the formal processed `pre_nivo/on_nivo` slice, no patient/timepoint combination has more than one `Tissue_ID`.
- `Inference:` For Task B sample assembly, `Tissue_ID` is the safe sample key and `fov` is the subordinate ROI key under that sample.

## 6. Task B formal cohort audit

- `Current file fact (direct):` The curated cohort flag `pre_nivo__on_nivo` exists explicitly in the patient-level, sample/timepoint, and FOV/core metadata tables.
- `Current file fact (direct):` `pre_nivo__on_nivo=True` identifies 40 patients in `TONIC_data_per_patient.csv`.
- `Current file fact (direct):` At the FOV/core metadata level, `pre_nivo__on_nivo=True` covers 449 FOV rows, 162 unique tissues, and 40 patients.
- `Current file fact (direct):` Within those 449 FOV rows, only 401 rows and 155 tissues have `MIBI_data_generated=True`.
- `Current file fact (direct):` The flagged FOV/core rows are not limited to `pre_nivo` and `on_nivo`; they also include `baseline`, `primary`, `local_recurrence`, `metastasis_1`, and `metastasis_2`.
- `Current file fact (direct):` After restricting to processed FOV support and `Timepoint in {pre_nivo,on_nivo}`, the formal Task B slice contains 40 patients, 80 tissues, and 163 processed FOVs.
- `Current file fact (direct):` All 40 formal-cohort patients retain both `pre_nivo` and `on_nivo` in the processed pair slice.
- `Current file fact (direct):` The 80 formal paired tissues are supported by 59 multi-FOV tissues and 21 single-FOV tissues; the per-tissue FOV distribution is 21 tissues with 1 FOV, 37 with 2 FOVs, 20 with 3 FOVs, and 2 with 4 FOVs.
- `Inference:` The formal curated subset is explicitly present as a patient-membership flag, but it still must be filtered down to the actual Task B pair slice by applying both a processed-content intersection and a `Timepoint in {pre_nivo,on_nivo}` filter.
- `Unresolved issue:` None for identifying the formal processed `pre_nivo/on_nivo` Task B pair slice. The needed metadata logic is explicit and auditable.

## 7. State / compartment / area audit

- `Current file fact (direct):` `cell_table_clusters.csv` contains 1,984,723 cells across 678 FOVs with three state axes: 22 `cell_cluster` categories, 8 `cell_cluster_broad` categories, and 33 `cell_meta_cluster` categories.
- `Current file fact (direct):` `adata_processed.h5ad` independently confirms per-cell `fov`, per-cell `area`, `compartment`, `cell_cluster`, `cell_cluster_broad`, and `cell_meta_cluster` in `obs`.
- `Current file fact (direct):` The H5AD `obs.compartment` categories are exactly `cancer_border`, `cancer_core`, `immune_agg`, `stroma_border`, and `stroma_core`.
- `Current file fact (direct):` `cell_annotation_mask.csv` provides a pure-CSV compartment route on the same `fov`,`label` key, with `mask_name` counts across the same five compartments.
- `Current file fact (direct):` `fov_annotation_mask_area.csv` has 4,746 rows `(direct)`, which is 678 rows `(direct)` for each of 7 compartment labels `(direct)`: `all`, `empty_slide`, `cancer_core`, `cancer_border`, `stroma_core`, `stroma_border`, and `immune_agg`.
- `Current file fact (direct):` The area table, `imaged_fovs.csv`, and the processed cell tables cover the same 678 FOVs.
- `Current file fact (direct):` Positive area is not universal for every compartment in every FOV. For example, `immune_agg` has positive area in 127 FOVs `(direct)` and zero area in 551 FOVs `(direct)`.
- `Inference:` Density-scale sample-level nonnegative mass tables can be constructed from local data.
- `Inference:` The minimal objects needed are:
  1. `cell_table_clusters.csv` or `adata_processed.h5ad` for cell states.
  2. `cell_annotation_mask.csv` or H5AD `obs.compartment` for cell compartments.
  3. `harmonized_metadata.csv` or `TONIC_data_per_core.csv` for `fov -> Tissue_ID -> Patient_ID/Timepoint`.
  4. `fov_annotation_mask_area.csv` for compartment-level area denominators.
- `Unresolved issue:` None for density support in the formal processed cohort. The main care point is handling zero-area or absent-compartment rows explicitly rather than assuming every compartment is populated in every FOV.

## 8. File-format and interoperability audit

- `Current file fact:` Actual formats in the TONIC root are CSV and H5AD only.
- `Current file fact (direct):` The root contains 34 CSV files `(direct)` and 1 H5AD file `(direct)`, with no parquet, Excel, JSON, YAML, pickle, RDS, or loom artifacts.
- `Current file fact:` CSV files are directly consumable from Python with `csv` or `pandas`.
- `Current file fact:` `adata_processed.h5ad` is directly consumable from Python with `anndata` or `scanpy`, and its schema can also be inspected with `h5py`.
- `Inference:` No mandatory format conversion is required to map the formal Task B cohort into a Python-side task-local adapter.
- `Current file fact:` There is no local R-only serialization artifact that would force an R-based preprocessing step for the audited data layer.

## 9. Task B compatibility mapping

| Task B contract field | TONIC field(s) | Source object(s) | Compatibility status | Notes |
| --- | --- | --- | --- | --- |
| patient key | `Patient_ID` | `TONIC_data_per_patient.csv`, `TONIC_data_per_timepoint.csv`, `harmonized_metadata.csv` | `already compatible` | Explicit patient key exists. |
| sample key | `Tissue_ID` | `TONIC_data_per_timepoint.csv`, `harmonized_metadata.csv`, `TONIC_data_per_core.csv` | `already compatible` | Safe sample key for Task B aggregation. |
| timepoint key | `Timepoint` | `TONIC_data_per_timepoint.csv`, `harmonized_metadata.csv` | `already compatible` | Formal slice is recoverable with `pre_nivo` and `on_nivo`. |
| FOV / ROI key | `fov` | `harmonized_metadata.csv`, `TONIC_data_per_core.csv`, cell tables, area table | `already compatible` | Explicit FOV key exists across metadata, cell, and area layers. |
| cell key | `fov`,`label` | cell tables, `cell_annotation_mask.csv` | `already compatible` | Supports cell-level joins. |
| compartment key | H5AD `compartment` or CSV `mask_name` | `adata_processed.h5ad`, `cell_annotation_mask.csv` | `already compatible` | Native five-compartment processed axis is present. |
| state / cluster key | `cell_cluster`, `cell_cluster_broad`, `cell_meta_cluster` | `cell_table_clusters.csv`, `adata_processed.h5ad` | `already compatible` | Multiple usable state axes are present. |
| area key | `fov`,`compartment`,`area` | `fov_annotation_mask_area.csv` | `already compatible` | Supports density denominators. |
| sample-level mass assembly | join `fov -> Tissue_ID -> Patient_ID/Timepoint`, then aggregate cell counts by state and compartment and divide by area | `harmonized_metadata.csv` + cell state/compartment tables + area table | `compatible with thin task-local adapter` | Requires only task-local joins and aggregation; no new data object is missing. |

- `Current file fact:` The formal cohort flag, processed states, processed compartments, FOV-level areas, and FOV-to-sample linkage all exist locally.
- `Current file fact:` Summary feature tables such as `combined_feature_data.csv`, `combined_feature_data_filtered.csv`, and `timepoint_features.csv` are derived summary surfaces with normalized values and feature types including density ratios and compartment-area ratios; they are not valid direct transport masses under the Task B spec.
- `Inference:` TONIC can be mapped into the Task B contract with a thin task-local adapter over existing tables. No new upstream dataset is required for the formal processed `pre_nivo/on_nivo` slice.
- `Blocked by missing data object:` None for the formal processed `pre_nivo/on_nivo` slice.
- `Blocked by unresolved metadata ambiguity:` Broad non-formal timepoint expansions remain ambiguous because the sample/timepoint table, FOV/core table, and processed cell layer do not cover the same timepoint vocabulary.

## 10. Smallest real blockers

1. `Soft blocker:` `MIBI_data_generated=True` does not imply processed cell/area availability. The metadata layer reports 949 generated FOVs `(direct)`, but only 678 FOVs `(direct)` survive into both the processed cell tables and the FOV area table.
2. `Cleanup-only issue:` `TONIC_data_per_patient.csv` has 117 patient IDs `(direct)`, while the downstream sample/timepoint and FOV/core tables have 113 `(direct)`. The 4 extra IDs `(direct)` (`112`, `114`, `115`, `117`) are all-false cohort-flag rows with no downstream sample/FOV records.
3. `Interpretation limitation:` Broader metadata tables contain additional timepoints (`progression`, `lymphnode_*`, `metastasis_3`, `primary_other`, `local_recurrence`, `biopsy`) that are not all represented in the processed cell layer. This limits any expansion beyond the formal processed Task B slice.
4. `Interpretation limitation:` No explicit response, survival/follow-up, organ-site, lesion-site, or biopsy-site fields were found in local headers. This prevents response-stratified or site-matched interpretation from the public/local tables alone.
5. `Cleanup-only issue:` Derived summary feature tables are present beside the raw cell and area tables. They are useful for inventory, but they must be kept separate from canonical Task B mass construction because the spec forbids using summary or normalized feature tables as direct transport masses.
6. `Hard blocker:` None for the formal processed `pre_nivo/on_nivo` Task B design slice.

## 11. Ready-for-design decision

`READY for Task B experiment design`

- `Current file fact:` The formal `pre_nivo__on_nivo` cohort is explicitly present as a patient-membership flag and is reconstructible into a processed pair slice with 40 patients `(direct)`, 80 tissues `(direct)`, and 163 processed FOVs `(direct)`.
- `Current file fact:` The processed layer contains cell states, cell compartments, and FOV compartment areas on a shared 678-FOV universe `(direct)`.
- `Inference:` That is sufficient to build sample-level, density-scale nonnegative mass tables for the formal Task B cohort without new upstream data objects.
- `Current file fact:` The required caveat is operational, not structural: Task B design must use the processed intersection, not the broader `MIBI_data_generated=True` metadata universe.
- `Current file fact:` The main missing information is response/site/follow-up metadata, but those are outside the present Task B contract and do not block the response-blind external spatial-validity design.
