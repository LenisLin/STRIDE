# Figure 2 Visualization Strategy

This document is an internal visualization strategy note for Main Figure 2 and
two supporting supplementary figures. It is not manuscript-ready prose.
Visible figure titles, panel labels, legends, and axis labels should use concise
human-readable names rather than internal field names.

Each standalone plot should be generated from formal Task A outputs as a PDF.
Final panel assembly, alignment, and lettering will be handled manually in
Adobe Illustrator. This document defines the data interface so that the same
intent can be implemented in R, Python, or another plotting language.

## Result Roots

Use these root aliases in plotting scripts:

| Alias | Absolute path |
| --- | --- |
| `DESC` | `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/descriptive` |
| `B0` | `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0` |
| `B1` | `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1` |
| `B3` | `/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block3` |

## Cross-Language Data Interface

Plotting scripts should read formal output tables without modifying them.
Table shapes below are data rows x columns, excluding the header row. JSON
artifacts are metadata inputs and do not have a row-column shape.

### Display Label Map

Use formal field names in scripts and human-readable names in figure output.

| Formal field or metric | Figure label |
| --- | --- |
| `self_retention` | `Source retention` |
| `depletion` | `Source-open mass` |
| `off_diagonal_remodeling` | `Off-diagonal remodeling` |
| `emergence` | `Target-open mass` |
| `matched_incoming_burden` | `Matched incoming mass` |
| `F_L1_total` | `Total relation error` |
| `A_MAE_active` | `Active relation MAE` |
| `A_MSE_active` | `Active relation MSE` |
| `offdiag_mass_abs_error` | `Off-diagonal mass error` |
| `g_L1_total` | `Source-open total error` |
| `depletion_mass_abs_error` | `Source-open mass error` |
| `d_MAE` | `Source-open profile MAE` |
| `d_MSE` | `Source-open profile MSE` |
| `e_L1_total` | `Target-open total error` |
| `emergence_mass_abs_error` | `Target-open mass error` |
| `e_MAE` | `Target-open profile MAE` |
| `e_MSE` | `Target-open profile MSE` |
| `community_space_target_fraction` | `Community-space target` |
| `identity_projected_target_fraction` | `Identity-projected target` |
| `stride_reference` | `STRIDE` |
| `balanced_ot_baseline` | `Balanced OT` |
| `uot_baseline` | `Unbalanced OT` |
| `partial_ot_baseline` | `Partial OT` |
| `diagonal_transport_baseline` | `Diagonal transport` |
| `consistency_ablation` | `No consistency` |
| `geometry_ablation` | `No geometry` |
| `recurrence_ablation` | `No recurrence` |

### Shared Field Semantics

| Field | Meaning for plotting |
| --- | --- |
| `community_id` | Integer community identifier used for descriptive atlas panels. |
| `source_community_id` | Source community identifier for source-side relation summaries. |
| `target_community_id` | Target community identifier for target-side relation summaries. |
| `patient_id` | Patient identifier; use for paired lines and patient-level points. |
| `pair_family` | Tissue-domain relation family such as `TC-IM` or `TC-PT`. |
| `pair_family_left`, `pair_family_right` | Paired tissue-domain comparison, expected as `TC-IM` versus `TC-PT`. |
| `summary_name` | Formal real-data metric name; map to display labels before plotting. |
| `metric_name` | Formal benchmark metric name; map to display labels before plotting. |
| `tc_im_value`, `tc_pt_value` | Patient-level or cohort-level raw values for `TC -> IM` and `TC -> PT`. |
| `tc_im_median`, `tc_pt_median` | Cohort median values from statistical supplements. |
| `median_delta` | Median paired contrast; use for sorting or optional support annotation. |
| `bh_q_value`, `q_pass` | Existing BH-adjusted support fields for Block 1 supplements. |
| `method_name` | Formal method identifier in Block 3 benchmark outputs. |
| `method_class` | Method class, usually `reference`, `baseline`, or `ablation`. |
| `reported_value` | Raw per-patient metric value in Block 3 patient metrics. |
| `mean_value`, `ci_lower`, `ci_upper` | Summary value and bootstrap interval for Block 3 point-range panels. |
| `rerun_id` | Generator rerun identifier; use as pairing unit in Block 3 statistics. |
| `validation_object_id` | Generator target representation identifier in Stage 3A outputs. |

### General Plotting Rules

- Use raw values for statistical tests. Apply linear display scaling only to
  y-axis values, confidence intervals, and facet labels.
- Use lower-is-better orientation for benchmark and ablation error panels.
- Do not use `-log10(error)`, inverted axes, radar scores, or min-max recovery
  scores.
- If a method is not available for a displayed metric, leave an empty position
  and mark it as a light gray `NA`; do not add `NA` as a legend category.
- Use colorblind-safe palettes. Use shape, facet, or direct labeling when color
  alone could be ambiguous.
- Use concise figure labels. Keep formal field names in script variables,
  intermediate data frames, or code comments only.

### Display Scaling Rules

| Formal metric | Display transform | Axis label example |
| --- | --- | --- |
| `A_MAE_active` | `value * 1e3` | `Active relation MAE (x10^3)` |
| `d_MAE` | `value * 1e3` | `Source-open profile MAE (x10^3)` |
| `e_MAE` | `value * 1e3` | `Target-open profile MAE (x10^3)` |
| `A_MSE_active` | `value * 1e6` | `Active relation MSE (x10^6)` |
| `d_MSE` | `value * 1e3` | `Source-open profile MSE (x10^3)` |
| `e_MSE` | `value * 1e3` | `Target-open profile MSE (x10^3)` |

## Main Figure 2

Figure title:

Validation of STRIDE relation estimates in tissue-domain comparisons and
semi-synthetic benchmarks

### Panel A: Data Overview and Validation Workflow

Content:

- Multi-patient, multi-ROI, multi-tissue-domain data overview.
- Semi-synthetic hidden-truth benchmark workflow.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/task_a_descriptive_atlas_manifest.json` | JSON metadata | Cohort and atlas counts. |
| `B3/generator_validation/raw/generator_rerun_registry.csv` | CSV, `10 x 7` | Rerun count and split sizes. |
| `B3/generator_validation/raw/generator_split_registry.csv` | CSV, `320 x 4` | Train/test split membership. |
| `B3/generator_validation/raw/patient_truth_store.csv` | CSV, `80 x 16` | Hidden truth object availability. |

Key fields:

- Atlas metadata: `n_patients`, `n_rois`, `n_cells`,
  `n_observed_communities`, `domain_labels`.
- Rerun registry: `rerun_id`, `split_seed`, `n_train_patients`,
  `n_test_patients`, `hidden_relation_condition_id`.
- Split registry: `rerun_id`, `patient_id`, `split_role`.
- Truth store: `A_json`, `d_json`, `e_json`, `open_mass`,
  `endpoint_closure_l1`.

Visualization:

- Two-part schematic.
- Top: cohort and data overview.
- Bottom: repeated train-test split, generated source/target FOVs, hidden
  `(A,d,e)`, candidate methods, and scoring.

Axes and aesthetics:

- No quantitative axes.
- Display count labels from formal metadata.
- Use visible text labels only for necessary workflow entities.

Statistical comparison:

- None.

### Panel B: Community Annotation Overview

Content:

- Community-level annotation by cell-subpopulation composition, tissue-domain
  distribution, and patient/ROI prevalence.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/tables/community_cell_subtype_row_fractions.csv` | CSV, `25 x 22` | Community x cell-subtype composition. |
| `DESC/tables/community_domain_distribution.csv` | CSV, `75 x 7` | TC/IM/PT distribution for each community. |
| `DESC/tables/community_patient_occurrence_summary.csv` | CSV, `25 x 10` | Patient and ROI prevalence summary. |
| `DESC/tables/community_domain_roi_prevalence.csv` | CSV, `75 x 5` | Optional domain-specific ROI prevalence. |

Key fields:

- `community_cell_subtype_row_fractions.csv`: `community_id` plus the
  cell-subtype columns `B`, `CD4T`, `CD8T`, `Macro_CD11b`, `Macro_CD163`,
  `Macro_CD169`, `Macro_HLADR`, `Mono_CD11c`, `Mono_Classic`,
  `Mono_Intermediate`, `NK`, `SC_COLLAGEN`, `SC_FAP`, `SC_Vimentin`,
  `SC_aSMA`, `TC_CAIX`, `TC_EpCAM`, `TC_Ki67`, `TC_VEGF`, `Treg`,
  `UNKNOWN`.
- `community_domain_distribution.csv`: `community_id`, `domain_label`,
  `fraction_within_community`.
- `community_patient_occurrence_summary.csv`: `community_id`,
  `patient_prevalence`, `roi_prevalence`, `n_patients_present`.

Data transformation:

- Pivot cell-subtype columns to long format with fields
  `community_id`, `cell_subtype`, `fraction`.
- Compute `tumor_fraction = TC_CAIX + TC_EpCAM + TC_Ki67 + TC_VEGF` for
  ordering or annotation.

Visualization:

- Integrated annotation heatmap.
- Main body: community x cell-subpopulation heatmap.
- Right side: TC/IM/PT stacked annotation bar.
- Far right: patient prevalence and optional ROI prevalence bar.

Axes and aesthetics:

- y-axis: communities.
- x-axis: cell subpopulations.
- Fill: fraction within community.
- Domain annotation fill: `TC`, `IM`, `PT`.

Statistical comparison:

- None; descriptive annotation only.

### Panel C: Overall Raw Relation Summaries

Content:

- Cohort-level overall relation summaries for empirical null, `TC -> IM`, and
  `TC -> PT`.
- Patient-level distributions are reserved for Supplementary Figure 1D.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B0/block0_metric_summary.csv` | CSV, `16 x 18` | Empirical null and observed `TC -> IM` summaries. |
| `B1/block1_family_statistical_supplement.csv` | CSV, `8 x 35` | Paired `TC -> IM` versus `TC -> PT` cohort summaries and statistics. |

Key fields:

- Block 0: `summary_name`, `summary_role`, `eligible_entity_axis`, `scale`,
  `cohort_stat`, `real_value`, `null_reference`, `empirical_p_value`.
- Block 1: `summary_name`, `scale`, `eligible_entity_axis`, `tc_im_median`,
  `tc_pt_median`, `median_delta`, `wilcoxon_p_value`, `bh_q_value`,
  `q_pass`.

Required filters:

- `scale == "burden_weighted"`.
- `cohort_stat == "median"` for Block 0 rows.
- `summary_name in c("self_retention", "depletion",
  "off_diagonal_remodeling", "emergence")`.

Data interface for plotting:

- Build a long table with fields `metric_label`, `group`, `value`, `p_value`,
  `q_value`.
- `group` levels: `Null`, `TC -> IM`, `TC -> PT`.
- Map `depletion` to `Source-open mass` and `emergence` to
  `Target-open mass` for visible labels.

Visualization:

- Cohort-level grouped point or bar plot.

Axes and aesthetics:

- x-axis: metric label.
- y-axis: raw summary value.
- color or fill: `Null`, `TC -> IM`, `TC -> PT`.
- Keep the main panel compact; expanded patient-level support belongs in
  Supplementary Figure 1D.

Statistical comparison:

- Null versus `TC -> IM`: use Block 0 `empirical_p_value`.
- `TC -> IM` versus `TC -> PT`: use Block 1 `wilcoxon_p_value` and
  `bh_q_value`.
- Do not generate a null versus `TC -> PT` comparison.

### Panel D: Tumor-Dominant Community Relation Behavior

Content:

- Tumor-dominant source communities and their `TC -> IM` versus `TC -> PT`
  source-side relation behavior.
- Use two visual layers: a complete cohort-level screen plus selected
  patient-level examples.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/tables/community_cell_subtype_row_fractions.csv` | CSV, `25 x 22` | Tumor-dominant community selection. |
| `B1/block1_source_community_comparison.csv` | CSV, `2400 x 21` | Patient-level source community paired values. |
| `B1/block1_source_community_statistical_supplement.csv` | CSV, `75 x 35` | Cohort medians and statistical support. |
| `DESC/tables/community_patient_occurrence_summary.csv` | CSV, `25 x 10` | Optional prevalence annotation. |

Key fields:

- Selection: `TC_CAIX`, `TC_EpCAM`, `TC_Ki67`, `TC_VEGF`,
  `community_id`.
- Patient-level values: `patient_id`, `source_community_id`,
  `summary_name`, `tc_im_value`, `tc_pt_value`, `comparison_status`.
- Cohort summaries: `source_community_id`, `summary_name`, `tc_im_median`,
  `tc_pt_median`, `median_delta`, `bh_q_value`, `q_pass`,
  `review_candidate`.

Required filters:

- Tumor-dominant community rule:
  `TC_CAIX + TC_EpCAM + TC_Ki67 + TC_VEGF > 0.5`.
- Current formal outputs select communities `0`, `1`, `3`, `6`, `10`, `11`,
  `12`, `16`, and `17` under this rule.
- Main heatmap metrics:
  `summary_name in c("self_retention", "depletion")`.
- Patient example facets:
  - `source_community_id == 6` and `summary_name == "self_retention"`;
  - `source_community_id == 10` and `summary_name == "self_retention"`;
  - `source_community_id == 12` and `summary_name == "depletion"`.
- Patient example rows should use `comparison_status == "estimable"`.

Data interface for plotting:

- Heatmap table fields: `source_community_id`, `metric_label`,
  `pair_family`, `value`, `q_value`, `q_pass`.
- Patient example table fields: `patient_id`, `source_community_id`,
  `metric_label`, `pair_family`, `value`.
- Visible labels: `Source retention` and `Source-open mass`.

Visualization:

- Hybrid panel.
- Left or upper layer: heatmap covering all tumor-dominant communities.
- Right or lower layer: three paired patient boxplots as readable examples.

Axes and aesthetics:

- Heatmap rows: tumor-dominant source communities.
- Heatmap columns: `Source retention TC -> IM`, `Source retention TC -> PT`,
  `Source-open mass TC -> IM`, `Source-open mass TC -> PT`.
- Heatmap fill: raw cohort median value.
- Example boxplots x-axis: `TC -> IM` and `TC -> PT`.
- Example boxplots y-axis: raw metric value.
- Connect paired patient points within each example facet.

Statistical comparison:

- Use existing `bh_q_value` and `q_pass` from
  `block1_source_community_statistical_supplement.csv`.
- If visible support labels are used, keep them compact and tied to the
  displayed community-metric pairs.

### Panel E: Semi-Synthetic Benchmark Against Transport Baselines

Content:

- Controlled hidden-truth recovery benchmark for relation and open-channel
  error metrics.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/a_benchmark/raw/a_benchmark/condition_summary.csv` | CSV, `90 x 16` | Relation benchmark summary bars and intervals. |
| `B3/a_benchmark/raw/a_benchmark/patient_metrics.csv` | CSV, `7200 x 12` | Relation benchmark rerun-level statistical input. |
| `B3/de_benchmark/raw/de_benchmark/condition_summary.csv` | CSV, `72 x 16` | Open-channel benchmark summary bars and intervals. |
| `B3/de_benchmark/raw/de_benchmark/patient_metrics.csv` | CSV, `5760 x 12` | Open-channel benchmark rerun-level statistical input. |

Key fields:

- Summary tables: `method_name`, `method_class`, `metric_name`,
  `metric_role`, `mean_value`, `ci_lower`, `ci_upper`,
  `paired_difference_vs_stride_reference`.
- Patient metrics: `rerun_id`, `patient_id`, `method_name`, `method_class`,
  `metric_name`, `reported_value`.

Required metrics:

- Relation metrics from `a_benchmark`:
  `F_L1_total`, `A_MAE_active`, `offdiag_mass_abs_error`.
- Open-channel metrics from `de_benchmark`:
  `g_L1_total`, `d_MAE`, `e_L1_total`, `e_MAE`.

Required methods:

- `stride_reference`.
- `balanced_ot_baseline`.
- `uot_baseline`.
- `partial_ot_baseline`.
- `diagonal_transport_baseline`.

NA handling:

- Show one shared method universe across relation and open-channel facets.
- `balanced_ot_baseline` is not present in the open-channel benchmark outputs;
  show an empty position with a light gray `NA` in those facets.
- `diagonal_transport_baseline` has formal open-channel metric rows in
  `de_benchmark`; display those values.

Data interface for plotting:

- Build a summary plotting table with fields `metric_label`, `method_label`,
  `mean_display`, `ci_lower_display`, `ci_upper_display`, `is_missing`.
- Build a statistical table by aggregating `reported_value` to
  `rerun_id + method_name + metric_name`; keep unscaled raw means for tests.

Visualization:

- Single faceted lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: method.
- y-axis: error value, with display scaling stated in facet labels.
- facets: metric.
- Optional points: rerun-level mean values.

Statistical comparison:

- Compare each displayed baseline to `stride_reference` with paired Wilcoxon
  signed-rank tests across rerun-level means.
- Apply BH adjustment across displayed baseline-by-metric comparisons.
- Run tests on unscaled rerun-level values.

### Panel F: Semi-Synthetic Objective-Component Ablation

Content:

- Objective-component ablation benchmark using the same lower-is-better error
  vocabulary as Panel E where possible.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv` | CSV, `36 x 16` | Consistency ablation summary bars and intervals. |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Consistency ablation rerun-level statistical input. |
| `B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv` | CSV, `36 x 16` | Geometry ablation summary bars and intervals. |
| `B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Geometry ablation rerun-level statistical input. |
| `B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv` | CSV, `36 x 16` | Recurrence ablation summary bars and intervals. |
| `B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Recurrence ablation rerun-level statistical input. |

Key fields:

- Summary tables: `evaluation_family`, `method_name`, `method_class`,
  `metric_name`, `mean_value`, `ci_lower`, `ci_upper`.
- Patient metrics: `rerun_id`, `patient_id`, `evaluation_family`,
  `method_name`, `method_class`, `metric_name`, `reported_value`.

Required metrics:

- `F_L1_total`, `A_MAE_active`, `offdiag_mass_abs_error`, `g_L1_total`,
  `d_MAE`, `e_L1_total`, `e_MAE`.

Required methods:

- `stride_reference`.
- `consistency_ablation`.
- `geometry_ablation`.
- `recurrence_ablation`.

Data interface for plotting:

- Build a combined summary table with fields `ablation_family`,
  `method_label`, `metric_label`, `mean_display`, `ci_lower_display`,
  `ci_upper_display`.
- Build a statistical table by aggregating raw `reported_value` to
  `rerun_id + ablation_family + method_name + metric_name`.
- A shared reference bar may be used for display if selected reference summary
  rows are identical, but paired tests should use the matched reference rows
  from each ablation file.

Visualization:

- Faceted lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: reference and ablation arms.
- y-axis: error value, with display scaling stated in facet labels.
- facets: metric.

Statistical comparison:

- Compare each ablation to its matched `stride_reference` with paired
  Wilcoxon signed-rank tests across rerun-level means.
- Apply BH adjustment across displayed ablation-by-metric comparisons.
- Run tests on unscaled rerun-level values.

## Supplementary Figure 1

Figure title:

Extended real-data community composition and relation estimates

### Panel A: Community Distribution Across Tissue Domains

Content:

- TC/IM/PT distribution profile for each community.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/tables/community_domain_distribution.csv` | CSV, `75 x 7` | Domain fractions by community. |

Key fields:

- `community_id`, `domain_label`, `n_cells`, `community_total_cells`,
  `fraction_within_community`.

Visualization:

- Stacked bar plot.

Axes and aesthetics:

- x-axis: communities.
- y-axis: fraction within community.
- fill: `TC`, `IM`, `PT`.

Statistical comparison:

- None.

### Panel B: Selected Community Fractions Across Tissue Domains

Content:

- Patient-level tissue-domain composition for communities used by Main
  Figure 2D and SF1-G.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `/mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad` | H5AD, cell-level Stage 0 object | Read-only source for patient-domain community fractions. |

Key fields:

- `/obs/patient_id/categories` and `/obs/patient_id/codes`: patient ids.
- `/obs/compartment/categories` and `/obs/compartment/codes`: tissue-domain
  labels `TC`, `IM`, and `PT`.
- `/obs/proto_id`: shared community id.

Data transformation:

- Select communities `0`, `1`, `2`, `3`, `6`, `10`, `11`, `12`, `14`,
  `16`, `17`, and `23`.
- For each `patient_id x domain_label x community_id`, compute
  `community_fraction = community_cells / domain_total_cells`.
- Treat patient as the repeated unit; do not use ROI-level fractions as
  independent statistical replicates for this panel.

Visualization:

- Faceted boxplot with patient-level jitter and pairwise significance
  brackets.

Axes and aesthetics:

- x-axis: tissue domains `TC`, `IM`, `PT`.
- y-axis: community fraction within patient-domain.
- facets: selected communities.
- y-axis scaling: free y-axis by community facet to preserve low-abundance
  distribution detail.
- color/fill: tissue domain using the global `domain_colors`.

Statistical comparison:

- For each displayed community, run paired Wilcoxon tests for `TC-IM`,
  `TC-PT`, and `IM-PT`.
- Apply BH adjustment across the 36 displayed pairwise comparisons.
- Display compact significance brackets (`*`, `**`, `***`, `ns`) rather than
  full q-values to avoid overloading the small facets.

### Panel C: Representative Spatial Examples

Content:

- ROI-level spatial overlays for selected communities.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/tables/representative_overlay_selection.csv` | CSV, `8 x 9` | Overlay selection metadata. |
| `DESC/figures/representative_spatial_overlays/*.svg` | SVG images | Existing representative overlays. |

Key fields:

- `community_id`, `patient_id`, `domain_label`, `fov_id`,
  `community_fraction_in_roi`, `community_total_cells`, `overlay_path`.

Visualization:

- Multi-panel spatial overlay montage.

Axes and aesthetics:

- No quantitative axes.
- Label each overlay with community id, patient/FOV, and tissue domain.

Statistical comparison:

- None.

### Panel D: Patient-Level Overall Relation Summaries

Content:

- Patient-level support for the cohort-level patterns shown in Main Figure 2C.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B0/block0_patient_calibration.csv` | CSV, `512 x 22` | Patient-level null calibration values. |
| `B1/block1_confirmatory_family_comparison.csv` | CSV, `256 x 21` | Patient-level paired `TC -> IM` and `TC -> PT` values. |
| `B0/block0_metric_summary.csv` | CSV, `16 x 18` | Null comparison labels. |
| `B1/block1_family_statistical_supplement.csv` | CSV, `8 x 35` | Paired statistical support labels. |

Key fields:

- Block 0 patient rows: `patient_id`, `summary_name`, `scale`,
  `reference_stat`, `real_value`, `null_reference`, `empirical_p_value`.
- Block 1 paired rows: `patient_id`, `summary_name`, `scale`,
  `tc_im_value`, `tc_pt_value`, `delta_tc_im_minus_tc_pt`,
  `comparison_status`.

Required filters:

- `scale == "burden_weighted"`.
- `summary_name in c("self_retention", "depletion",
  "off_diagonal_remodeling", "emergence")`.

Visualization:

- Patient-level paired or jitter plot matching Main Figure 2C, with explicit
  statistical brackets for the two valid comparisons.

Axes and aesthetics:

- x-axis: `Null`, `TC -> IM`, `TC -> PT`.
- y-axis: raw metric value.
- facets: metric label.
- connect patient-level `TC -> IM` and `TC -> PT` values.
- bracket 1: `Null` versus `TC -> IM`.
- bracket 2: `TC -> IM` versus `TC -> PT`.

Statistical comparison:

- `Null` versus `TC -> IM`: use `B0/block0_metric_summary.csv`
  `empirical_p_value` from `scale == "burden_weighted"` and
  `cohort_stat == "median"`.
- `TC -> IM` versus `TC -> PT`: use
  `B1/block1_family_statistical_supplement.csv` `bh_q_value` from
  `scale == "burden_weighted"`.
- Do not compute or display a `Null` versus `TC -> PT` test.

### Panel E: Raw Extended Relation Matrices

Content:

- Full raw relation object for `TC -> IM` and `TC -> PT`.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B1/block1_cohort_relation_comparison.csv` | CSV, `675 x 13` | Cohort-level raw relation elements. |
| `B1/block1_relation_element_statistical_supplement.csv` | CSV, `675 x 38` | Optional q-supported cell outlines. |

Key fields:

- Relation elements: `component`, `relation_axis`, `source_community_id`,
  `target_community_id`, `tc_im_value`, `tc_pt_value`.
- Statistics: `component`, `relation_axis`, `source_community_id`,
  `target_community_id`, `bh_q_value`, `q_pass`.

Data transformation:

- Infer `K` from observed community identifiers; do not hard-code matrix size.
- For each pair family, construct the complete `(K + 1) x (K + 1)` extended
  relation matrix.
- Keep `template_A` as the community-by-community block, including its
  diagonal entries in their original matrix positions.
- Display `template_d` as the source-open column and `template_e` as the
  target-open row.
- Keep the conceptual bottom-right source-open/target-open intersection blank.
- Apply `log10(value + epsilon)` to all non-blank entries for display only,
  with `epsilon = 1e-4` unless explicitly changed in the script.
- Use a fixed display range of `[-4, 0]` for both pair families so separately
  exported panels can be compared after manual assembly.
- Do not transform, rescale, or re-export the raw relation values.

Visualization:

- Two standalone complete extended-matrix heatmaps: `TC -> IM relation object`
  and `TC -> PT relation object`, exported as separate PDFs for manual
  side-by-side assembly.

Axes and aesthetics:

- rows: source communities plus the target-open `e` row.
- columns: target communities plus the source-open `d` column.
- Use one shared display-only `log10(value + epsilon)` color scale for `A`,
  `d`, and `e`.
- Do not give `d` or `e` separate colors or separate scales; distinguish them
  by their source-open column and target-open row positions.
- Use the same community order, legend range, and display transform for
  `TC -> IM` and `TC -> PT`, with labels shown on the original-value scale
  where possible.
- State in the legend that the color scale is a display transform only; raw
  relation values and statistical interfaces are unchanged.

Statistical comparison:

- Optional cell outlines from `q_pass`; no new statistical tests.

### Panel F: Community-Level Relation Summary Dumbbell Plots

Content:

- Readable marginal view of source-side and target-side relation summaries,
  with direct `TC -> IM` versus `TC -> PT` comparison within each metric.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B1/block1_source_community_statistical_supplement.csv` | CSV, `75 x 35` | Source-side statistical support. |
| `B1/block1_target_community_statistical_supplement.csv` | CSV, `75 x 35` | Target-side statistical support. |

Key fields:

- Statistical supplements: community id, `summary_name`, `tc_im_median`,
  `tc_pt_median`, `bh_q_value`, `q_pass`.
- Source-side metrics: `self_retention`, `off_diagonal_remodeling`,
  `depletion`.
- Target-side metric: `matched_incoming_burden`.

Visualization:

- Four standalone metric-wise dumbbell plots:
  `Source retention`, `Off-diagonal remodeling`, `Source-open mass`, and
  `Matched incoming mass`.

Axes and aesthetics:

- y-axis: communities in the same order across all four outputs.
- x-axis: raw cohort median for the displayed metric.
- points: `TC -> IM` and `TC -> PT` using the global pair-family colors.
- segment: paired contrast between the two transfer families for the same
  community.
- x-axis ranges are metric-specific; do not use segment length across separate
  PDFs to compare absolute magnitude between different metrics.
- keep non-estimable target communities as empty rows in the
  `Matched incoming mass` output to preserve community-axis alignment.
- `q_pass` is a weak line-opacity cue only.
- standalone output PDFs:
  `sf1_panelF_source_retention_dumbbell.pdf`,
  `sf1_panelF_off_diagonal_remodeling_dumbbell.pdf`,
  `sf1_panelF_source_open_mass_dumbbell.pdf`, and
  `sf1_panelF_matched_incoming_mass_dumbbell.pdf`.

Statistical comparison:

- Use existing source or target statistical supplement fields; no new tests.

### Panel G: PT-Rich Target Patient-Level Incoming Burden Examples

Content:

- Patient-level distributions for PT-rich target communities, serving as the
  target-side patient-level complement to Main Figure 2D's tumor-dominant
  source-side examples.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B1/block1_target_community_comparison.csv` | CSV, `2400 x 21` | Target-side paired patient values. |
| `B1/block1_target_community_statistical_supplement.csv` | CSV, `75 x 35` | Target-side support labels. |
| `DESC/tables/community_domain_distribution.csv` | CSV, `75 x 7` | PT-rich target annotation support. |

Target-side filters:

- Select target communities from `community_domain_distribution.csv` where
  `domain_label == "PT"` and `fraction_within_community > 0.5`.
- Under the current formal atlas this rule gives C2, C13, C14, C20, C21,
  C22, and C23; plotting code should infer this set from the table rather
  than hard-coding it.
- `summary_name == "matched_incoming_burden"`.
- `comparison_status == "estimable"`.
- Plot only complete paired patient values with non-missing `tc_im_value` and
  `tc_pt_value`.
- Do not filter communities by effect size, q-value, or visual prominence.

Data interface for plotting:

- Build one long table with fields `patient_id`, `target_community_id`,
  `community_facet`, `pair_family`, and `value`.
- `value` is raw patient-level `matched_incoming_burden`.

Visualization:

- Boxplot plus jittered patient points and paired patient lines.

Axes and aesthetics:

- x-axis: `TC -> IM` and `TC -> PT`.
- y-axis: raw `Matched incoming mass`.
- facets: PT-rich target communities.
- connect paired patient points.
- use `facet_wrap(~ community_facet, scales = "free_y", ncol = 4)`.
- keep non-q-supported PT-rich communities, such as C13, because community
  selection is based on tissue composition rather than statistical support.

Statistical comparison:

- Use existing target statistical supplement q-values for displayed facets.
- No new statistical tests.

## Supplementary Figure 2

Figure title:

Supplemental semi-synthetic generator credibility and selected recovery metrics

Supplementary Figure 2 uses selected supplemental validation metrics rather
than the full formal metric vocabulary. It should not display endpoint-closure,
support-set, ratio, or capture metrics unless the visualization scope changes.
Benchmark and ablation panels retain lower-is-better error values with linear
display scaling where needed.

### Panel A: Generator Rerun Consistency

Content:

- Rerun variability for generator target summaries across repeated train-test
  splits.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/generator_validation/raw/generator_validation/rerun_stability.csv` | CSV, `2 x 10` | Between-rerun variability summary. |

Key fields:

- `validation_object_id`, `metric_name`, `reported_value`,
  `stability_summary_level`.

Required filters:

- `validation_object_id in c("community_space_target_fraction",
  "identity_projected_target_fraction")`.
- `metric_name == "rerun variability"`.

Visualization:

- Compact point or bar plot.

Axes and aesthetics:

- x-axis: validation object label.
- y-axis: rerun variability.

Statistical comparison:

- None; diagnostic display only.

### Panel B: Synthetic-Real Target Agreement

Content:

- Rerun-level numerical agreement between synthetic targets and held-out real
  target profiles.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/generator_validation/raw/generator_validation/object_scores.csv` | CSV, `80 x 9` | Rerun-level generator validation scores. |

Key fields:

- `rerun_id`, `validation_object_id`, `metric_name`, `reported_value`.

Required filters:

- `validation_object_id in c("community_space_target_fraction",
  "identity_projected_target_fraction")`.
- `metric_name in c("Pearson correlation", "MAE", "JS divergence")`.
- Omit `MSE` from the visual layer.

Visualization:

- Rerun-level dot plot with median or point-range summary.

Axes and aesthetics:

- x-axis: metric.
- y-axis: reported value.
- color or facet: validation object label.

Statistical comparison:

- None; diagnostic display only.

### Panel C: Relation Benchmark Supplemental Error

Content:

- Supplemental active relation MSE in the relation benchmark.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/a_benchmark/raw/a_benchmark/condition_summary.csv` | CSV, `90 x 16` | Summary bars and intervals. |
| `B3/a_benchmark/raw/a_benchmark/patient_metrics.csv` | CSV, `7200 x 12` | Rerun-level statistical input. |

Key fields:

- Summary: `method_name`, `method_class`, `metric_name`, `mean_value`,
  `ci_lower`, `ci_upper`.
- Patient metrics: `rerun_id`, `patient_id`, `method_name`, `metric_name`,
  `reported_value`.

Required filters:

- `metric_name == "A_MSE_active"`.
- Methods: `stride_reference`, `balanced_ot_baseline`, `uot_baseline`,
  `partial_ot_baseline`, `diagonal_transport_baseline`.

Visualization:

- Lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: method.
- y-axis: `Active relation MSE (x10^6)`.

Statistical comparison:

- Aggregate patient metrics to rerun-level means.
- Compare each baseline to `stride_reference` with paired Wilcoxon signed-rank
  tests.
- Apply BH adjustment across displayed baseline comparisons.
- Run tests on unscaled values.

### Panel D: Open Benchmark Supplemental Errors

Content:

- Supplemental open-channel mass and profile errors in the open benchmark.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/de_benchmark/raw/de_benchmark/condition_summary.csv` | CSV, `72 x 16` | Summary bars and intervals. |
| `B3/de_benchmark/raw/de_benchmark/patient_metrics.csv` | CSV, `5760 x 12` | Rerun-level statistical input. |

Key fields:

- Summary: `method_name`, `method_class`, `metric_name`, `mean_value`,
  `ci_lower`, `ci_upper`.
- Patient metrics: `rerun_id`, `patient_id`, `method_name`, `metric_name`,
  `reported_value`.

Required filters:

- `metric_name in c("depletion_mass_abs_error",
  "emergence_mass_abs_error", "d_MSE", "e_MSE")`.
- Methods: `stride_reference`, `uot_baseline`, `partial_ot_baseline`,
  `diagonal_transport_baseline`.
- This supplemental open-benchmark panel displays only methods present in the `de_benchmark` formal output; unlike Main Figure 2E, it does not add a missing `balanced_ot_baseline` NA slot.

Visualization:

- Faceted lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: method.
- y-axis: error value with display scaling in facet labels.
- facets: `Source-open mass error`, `Target-open mass error`,
  `Source-open profile MSE`, `Target-open profile MSE`.

Statistical comparison:

- Aggregate patient metrics to rerun-level means.
- Compare each baseline to `stride_reference` with paired Wilcoxon signed-rank
  tests within each displayed metric.
- Apply BH adjustment across displayed comparisons.
- Run tests on unscaled values.

### Panel E: Ablation Supplemental Relation and Mass Errors

Content:

- Supplemental relation and open-mass errors for objective-component ablations.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv` | CSV, `36 x 16` | Consistency ablation summary. |
| `B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv` | CSV, `36 x 16` | Geometry ablation summary. |
| `B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv` | CSV, `36 x 16` | Recurrence ablation summary. |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Consistency ablation rerun-level statistical input. |
| `B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Geometry ablation rerun-level statistical input. |
| `B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Recurrence ablation rerun-level statistical input. |

Key fields:

- Summary: `evaluation_family`, `method_name`, `method_class`,
  `metric_name`, `mean_value`, `ci_lower`, `ci_upper`.
- Patient metrics: `rerun_id`, `patient_id`, `evaluation_family`,
  `method_name`, `metric_name`, `reported_value`.

Required filters:

- `metric_name in c("A_MSE_active", "depletion_mass_abs_error",
  "emergence_mass_abs_error")`.
- Methods: `stride_reference`, `consistency_ablation`,
  `geometry_ablation`, `recurrence_ablation`.

Visualization:

- Faceted lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: method or ablation arm.
- y-axis: error value with display scaling in facet labels.
- facets: `Active relation MSE`, `Source-open mass error`,
  `Target-open mass error`.

Statistical comparison:

- Aggregate patient metrics to rerun-level means.
- Compare each ablation to its matched `stride_reference` with paired Wilcoxon
  signed-rank tests.
- Apply BH adjustment across displayed ablation-by-metric comparisons.
- Run tests on unscaled values.

### Panel F: Ablation Supplemental Open-Profile MSE

Content:

- Supplemental open-profile MSEs for objective-component ablations.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv` | CSV, `36 x 16` | Consistency ablation summary. |
| `B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv` | CSV, `36 x 16` | Geometry ablation summary. |
| `B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv` | CSV, `36 x 16` | Recurrence ablation summary. |
| `B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Consistency ablation rerun-level statistical input. |
| `B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Geometry ablation rerun-level statistical input. |
| `B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Recurrence ablation rerun-level statistical input. |

Key fields:

- Summary: `evaluation_family`, `method_name`, `method_class`,
  `metric_name`, `mean_value`, `ci_lower`, `ci_upper`.
- Patient metrics: `rerun_id`, `patient_id`, `evaluation_family`,
  `method_name`, `metric_name`, `reported_value`.

Required filters:

- `metric_name in c("d_MSE", "e_MSE")`.
- Methods: `stride_reference`, `consistency_ablation`,
  `geometry_ablation`, `recurrence_ablation`.

Visualization:

- Faceted lower-is-better point-range or bar plot.

Axes and aesthetics:

- x-axis: method or ablation arm.
- y-axis: error value.
- facets: `Source-open profile MSE (x10^3)` and
  `Target-open profile MSE (x10^3)`.

Statistical comparison:

- Aggregate patient metrics to rerun-level means.
- Compare each ablation to its matched `stride_reference` with paired Wilcoxon
  signed-rank tests.
- Apply BH adjustment across displayed ablation-by-metric comparisons.
- Run tests on unscaled values.

## Metrics Not Planned For Visual Panels

The following formal metrics are retained in formal outputs but are not planned
for Main Figure 2 or Supplementary Figure 2 visual panels under the current
visualization scope:

- `offdiag_ratio`;
- `depletion_capture`;
- `emergence_capture`;
- `endpoint_y_MAE`;
- `target_recall_at_k`;
- `open_support_F1`;
- generator `MSE`.
