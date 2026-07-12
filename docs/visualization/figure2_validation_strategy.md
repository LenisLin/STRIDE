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

The `B3` root was recreated by the formal 2026-07-12 Block 3 run. Plotting code
must fail on missing or incomplete Block 3 inputs rather than substitute
historical rows. The rendering script requires `block3_run_manifest.json` with
`execution_scope=formal_full_data` and all sections complete before reading
Block 3 tables.

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
| `B3/raw/generator_rerun_registry.csv` | CSV, `10 x 7` | Rerun count and split sizes. |
| `B3/raw/generator_split_registry.csv` | CSV, `320 x 4` | Train/test split membership. |
| `B3/raw/patient_truth_store.csv` | CSV, `80 x 16` | Hidden truth object availability. |

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
| `DESC/tables/community_cell_subtype_row_fractions.csv` | CSV, `K x 22` | Community x cell-subtype composition. |
| `DESC/tables/community_domain_distribution.csv` | CSV, `3K x 7` | TC/IM/PT distribution for each community. |
| `DESC/tables/community_patient_occurrence_summary.csv` | CSV, `K x 10` | Patient and ROI prevalence summary. |
| `DESC/tables/community_domain_roi_prevalence.csv` | CSV, `3K x 5` | Optional domain-specific ROI prevalence. |

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
| `DESC/tables/community_cell_subtype_row_fractions.csv` | CSV, `K x 22` | Tumor-dominant community selection. |
| `B1/block1_source_community_comparison.csv` | CSV, patient/community/metric rows, K-dependent | Patient-level source community paired values. |
| `B1/block1_source_community_statistical_supplement.csv` | CSV, `3K x 35` | Cohort medians and statistical support. |
| `DESC/tables/community_patient_occurrence_summary.csv` | CSV, `K x 10` | Optional prevalence annotation. |

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
- Main heatmap metrics:
  `summary_name in c("self_retention", "depletion")`.
- Patient example facets should use the tumor-dominant communities inferred
  from the current descriptive atlas table.
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

### Panel E: Benchmark Primary Community-Resolved MAE

Content:

- Controlled hidden-truth benchmark primary recovery display.
- Primary recovery panels report community-resolved MAE for `A`, `d`, and `e`.
- Errors for `A` and `d` are evaluated on burden-weighted carriers, `F = xA`
  and `g = xd`; `e` is evaluated directly.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/a_benchmark/patient_metrics.csv` | CSV, `7200 x 12` | Relation benchmark rerun-level statistical input. |
| `B3/raw/de_benchmark/patient_metrics.csv` | CSV, `5760 x 12` | Open-channel benchmark rerun-level statistical input. |

Key fields:

- Patient metrics: `rerun_id`, `patient_id`, `method_name`, `method_class`,
  `metric_name`, `reported_value`.

Required metrics:

| Formal metric | Display label |
| --- | --- |
| `A_MAE_active` | `A relation` |
| `d_MAE` | `d source-open` |
| `e_MAE` | `e target-open` |

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

- Aggregate `reported_value` to `rerun_id + method_name + metric_name`.
- Use scaled values only for display; keep unscaled rerun-level means for
  statistical tests.

Visualization:

- Rerun-level horizontal boxplot with overlaid rerun points.
- Right-side ladder brackets compare each baseline to STRIDE.

Axes and aesthetics:

- x-axis: community-resolved MAE, displayed as `x10^3`.
- y-axis: method.
- facets: `A relation`, `d source-open`, `e target-open`.

Statistical comparison:

- Compare each displayed baseline to `stride_reference` with paired Wilcoxon
  signed-rank tests across rerun-level means.
- Apply BH adjustment across displayed baseline-by-metric comparisons.
- Run tests on unscaled rerun-level values.
- Statistical labels show paired Wilcoxon signed-rank comparisons against
  STRIDE across rerun-level means, with BH adjustment across displayed
  comparisons.
- Display bracket labels only.

### Panel F: Ablation Primary Community-Resolved MAE

Content:

- Objective-component ablation primary recovery display.
- Primary recovery panels report community-resolved MAE for `A`, `d`, and `e`.
- Errors for `A` and `d` are evaluated on burden-weighted carriers, `F = xA`
  and `g = xd`; `e` is evaluated directly.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/subbag_consistency_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Consistency ablation rerun-level statistical input. |
| `B3/raw/geometry_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Geometry ablation rerun-level statistical input. |
| `B3/raw/recurrence_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Recurrence ablation rerun-level statistical input. |

Key fields:

- Patient metrics: `rerun_id`, `patient_id`, `evaluation_family`,
  `method_name`, `method_class`, `metric_name`, `reported_value`.

Required metrics:

| Formal metric | Display label |
| --- | --- |
| `A_MAE_active` | `A relation` |
| `d_MAE` | `d source-open` |
| `e_MAE` | `e target-open` |

Required methods:

- `stride_reference`.
- `consistency_ablation`.
- `geometry_ablation`.
- `recurrence_ablation`.

Data interface for plotting:

- Aggregate raw `reported_value` to
  `rerun_id + evaluation_family + method_name + metric_name`.
- A shared STRIDE row may be used for display, but paired tests use the matched
  STRIDE reference from each ablation family.

Visualization:

- Rerun-level horizontal boxplot with overlaid rerun points.
- Use the same visual grammar as Main Figure 2E.
- Right-side ladder brackets compare each ablation arm to its matched STRIDE
  reference.

Axes and aesthetics:

- x-axis: community-resolved MAE, displayed as `x10^3`.
- y-axis: STRIDE and ablation arms.
- facets: `A relation`, `d source-open`, `e target-open`.

Statistical comparison:

- Compare each ablation to its matched `stride_reference` with paired
  Wilcoxon signed-rank tests across rerun-level means.
- Apply BH adjustment across displayed ablation-by-metric comparisons.
- Run tests on unscaled rerun-level values.
- Statistical labels show paired Wilcoxon signed-rank comparisons against
  STRIDE across rerun-level means, with BH adjustment across displayed
  comparisons.
- Display bracket labels only.

## Supplementary Figure 1

Figure title:

Extended real-data community composition and relation estimates

### Panel A: Community Distribution Across Tissue Domains

Content:

- TC/IM/PT distribution profile for each community.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `DESC/tables/community_domain_distribution.csv` | CSV, `3K x 7` | Domain fractions by community. |

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
| `/mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0_k10/task_A_stage0_k10.h5ad` | H5AD, cell-level Stage 0 object | Read-only source for patient-domain community fractions. |

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
- Apply BH adjustment across all displayed pairwise comparisons.
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
| `B1/block1_cohort_relation_comparison.csv` | CSV, `(K^2 + 2K) x 13` | Cohort-level raw relation elements. |
| `B1/block1_relation_element_statistical_supplement.csv` | CSV, `(K^2 + 2K) x 38` | Optional q-supported cell outlines. |

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
| `B1/block1_source_community_statistical_supplement.csv` | CSV, `3K x 35` | Source-side statistical support. |
| `B1/block1_target_community_statistical_supplement.csv` | CSV, `3K x 35` | Target-side statistical support. |

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
| `B1/block1_target_community_comparison.csv` | CSV, patient/community/metric rows, K-dependent | Target-side paired patient values. |
| `B1/block1_target_community_statistical_supplement.csv` | CSV, `3K x 35` | Target-side support labels. |
| `DESC/tables/community_domain_distribution.csv` | CSV, `3K x 7` | PT-rich target annotation support. |

Target-side filters:

- Select target communities from `community_domain_distribution.csv` where
  `domain_label == "PT"` and `fraction_within_community > 0.5`.
- Plotting code should infer this set from the current descriptive atlas table
  rather than hard-coding community ids.
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
- keep non-q-supported PT-rich communities because community selection is
  based on tissue composition rather than statistical support.

Statistical comparison:

- Use existing target statistical supplement q-values for displayed facets.
- No new statistical tests.

## Supplementary Figure 2

Figure title:

Supplemental held-out generator audit and selected recovery metrics

Supplementary Figure 2 uses selected supplemental validation metrics rather
than the full formal metric vocabulary. It should not display endpoint-closure,
support-set, formal method-bearing ratio, or capture metrics unless the
visualization scope changes. Panels A and B audit the fixed semi-synthetic
generator in the held-out TC-to-IM proxy setting. They use existing
composition metrics to compare observed TC, observed IM, and generated IM
compositions without introducing a biological claim about real TC-to-IM
remodeling. Panels C and D report secondary recovery checks. Main Figure 2
uses community-resolved MAE as the primary recovery display; Supplementary
Figure 2C-D use community-level MSE sensitivity and overall amount error for
the same recovery objects.

Notation for Panels A and B:

- `x_TC` denotes the observed TC composition.
- `q_IM` denotes the observed IM composition.
- `\hat{q}_{IM}` denotes the generated IM composition.

### Panel A: Community-Level TC-to-IM Difference Audit

Content:

- Community-level comparison between `q_IM - x_TC` and
  `\hat{q}_{IM} - x_TC` for each rerun.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/generator_validation/target_surface_profiles.csv` | CSV, formal full-data `160K x 11` | Held-out observed IM community profiles. |
| `B3/raw/patient_truth_store.csv` | CSV | Observed TC compositions and generated IM compositions. |

Key fields:

- `rerun_id`, `patient_id`, `split_role`, `surface_source`,
  `validation_object_id`, `state_id`, `feature_index`, `reported_value`.
- `x_json` stores `x_TC` and `y_json` stores `\hat{q}_{IM}`.

Required filters:

- `split_role == "test"`.
- `surface_source == "heldout_real"`.
- `validation_object_id == "community_space_target_fraction"`.
- Use one row per `rerun_id` and `patient_id` from `patient_truth_store`.

Visualization:

- One side-by-side heatmap with two facets after averaging over held-out test
  patients: `q_IM - x_TC` and `\hat{q}_{IM} - x_TC`.

Axes and aesthetics:

- x-axis: rerun.
- y-axis: community/state.
- fill: mean community-fraction difference.
- Use one shared symmetric diverging color scale for both facets.

Statistical comparison:

- None at the community level. The panel is a visual audit of direction and
  magnitude patterns, not a per-community significance test.

Output PDF:

- `sf2_panelA_tc_to_im_difference_heatmap.pdf`.

### Panel B: Rerun-Level Generator Audit

Content:

- Rerun-level direct comparisons among `x_TC`, `q_IM`, and
  `\hat{q}_{IM}` using Pearson Cor, MAE, and JS profile metrics.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/generator_validation/target_surface_profiles.csv` | CSV, formal full-data `160K x 11` | Held-out observed IM community profiles. |
| `B3/raw/patient_truth_store.csv` | CSV | Observed TC compositions and generated IM compositions. |

Key fields:

- Target profiles: `rerun_id`, `patient_id`, `split_role`, `surface_source`,
  `validation_object_id`, `state_id`, `feature_index`, `reported_value`.
- Patient truth store: `rerun_id`, `patient_id`, `x_json`, `y_json`.

Required filters:

- Target profiles: `split_role == "test"`,
  `surface_source == "heldout_real"`, and
  `validation_object_id == "community_space_target_fraction"`.
- Use one row per `rerun_id` and `patient_id` from `patient_truth_store`.

Visualization:

- One PDF with two side-by-side comparison blocks.
- Left block: comparison to `q_IM`, with facets for Pearson Cor, MAE, and JS.
- Right block: comparison to `x_TC`, with facets for MAE and JS.
- Each facet displays rerun-level boxplots, paired rerun points, and one
  paired-comparison bracket.

Axes and aesthetics:

- x-axis: displayed pair of compositions.
- y-axis: rerun-level value.
- facet labels: metric name.

Statistical comparison:

- Aggregate patient-level profile metrics to rerun-level means before testing.
- Left block comparisons:
  - Pearson Cor: `cor(\hat{q}_{IM}, q_IM)` versus `cor(x_TC, q_IM)`.
  - MAE: `MAE(\hat{q}_{IM}, q_IM)` versus `MAE(x_TC, q_IM)`.
  - JS: `JS(\hat{q}_{IM}, q_IM)` versus `JS(x_TC, q_IM)`.
- Right block comparisons:
  - MAE: `MAE(\hat{q}_{IM}, x_TC)` versus `MAE(q_IM, x_TC)`.
  - JS: `JS(\hat{q}_{IM}, x_TC)` versus `JS(q_IM, x_TC)`.
- Use paired Wilcoxon tests across rerun-level summaries with BH adjustment
  across the five displayed comparisons.
- Display only bracket labels (`*`, `**`, `***`, or `ns`) in the panel.
- The right-side comparisons are not equivalence tests. A non-significant
  result would indicate no detectable rerun-level difference under this test,
  not proof of matched or calibrated change magnitude.
- Do not display generator MSE in this panel.

Output PDF:

- `sf2_panelB_rerun_audit_dotplot.pdf`.

### Panel C: Benchmark Secondary Checks

Content:

- Secondary benchmark checks for the same recovery objects shown in Main
  Figure 2E.
- Secondary panels report community-level MSE sensitivity and overall amount
  error for the same recovery objects.
- Errors for `A` and `d` are evaluated on burden-weighted carriers, `F = xA`
  and `g = xd`; `e` is evaluated directly.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/a_benchmark/condition_summary.csv` | CSV, `90 x 16` | Relation benchmark summary intervals. |
| `B3/raw/a_benchmark/patient_metrics.csv` | CSV, `7200 x 12` | Relation benchmark rerun-level statistical input. |
| `B3/raw/de_benchmark/condition_summary.csv` | CSV, `72 x 16` | Open-channel benchmark summary intervals. |
| `B3/raw/de_benchmark/patient_metrics.csv` | CSV, `5760 x 12` | Open-channel benchmark rerun-level statistical input. |

Metric mapping:

| Block | Formal metric | Display label |
| --- | --- | --- |
| Community-level error | `A_MSE_active` | `A relation` |
| Community-level error | `d_MSE` | `d source-open` |
| Community-level error | `e_MSE` | `e target-open` |
| Overall amount error | `offdiag_mass_abs_error` | `A off-diagonal` |
| Overall amount error | `depletion_mass_abs_error` | `d source-open` |
| Overall amount error | `emergence_mass_abs_error` | `e target-open` |

Required methods:

- `stride_reference`.
- `balanced_ot_baseline`.
- `uot_baseline`.
- `partial_ot_baseline`.
- `diagonal_transport_baseline`.

NA handling:

- Show one shared method universe across benchmark secondary-check facets.
- `balanced_ot_baseline` has relation metrics from `a_benchmark`; open-channel
  metrics absent from `de_benchmark` are displayed as light-gray `NA` slots.

Visualization:

- One PDF with two vertically stacked point-range blocks.
- Top block: `Community-level error`.
- Bottom block: `Overall amount error`.
- Each block uses right-side ladder brackets comparing each baseline to STRIDE.

Statistical comparison:

- Aggregate patient metrics to rerun-level means before testing.
- Statistical labels show paired Wilcoxon signed-rank comparisons against
  STRIDE across rerun-level means, with BH adjustment across displayed
  comparisons.
- Run tests on unscaled values.
- Display bracket labels only.

Output PDF:

- `sf2_panelC_benchmark_secondary_checks.pdf`.

### Panel D: Ablation Secondary Checks

Content:

- Secondary objective-component ablation checks for the same recovery objects
  shown in Main Figure 2F.
- Secondary panels report community-level MSE sensitivity and overall amount
  error for the same recovery objects.
- Errors for `A` and `d` are evaluated on burden-weighted carriers, `F = xA`
  and `g = xd`; `e` is evaluated directly.

Input data:

| Path | Format and shape | Role |
| --- | --- | --- |
| `B3/raw/subbag_consistency_ablation/condition_summary.csv` | CSV, `36 x 16` | Consistency ablation summary intervals. |
| `B3/raw/geometry_ablation/condition_summary.csv` | CSV, `36 x 16` | Geometry ablation summary intervals. |
| `B3/raw/recurrence_ablation/condition_summary.csv` | CSV, `36 x 16` | Recurrence ablation summary intervals. |
| `B3/raw/subbag_consistency_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Consistency ablation rerun-level statistical input. |
| `B3/raw/geometry_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Geometry ablation rerun-level statistical input. |
| `B3/raw/recurrence_ablation/patient_metrics.csv` | CSV, `2880 x 12` | Recurrence ablation rerun-level statistical input. |

Metric mapping:

| Block | Formal metric | Display label |
| --- | --- | --- |
| Community-level error | `A_MSE_active` | `A relation` |
| Community-level error | `d_MSE` | `d source-open` |
| Community-level error | `e_MSE` | `e target-open` |
| Overall amount error | `offdiag_mass_abs_error` | `A off-diagonal` |
| Overall amount error | `depletion_mass_abs_error` | `d source-open` |
| Overall amount error | `emergence_mass_abs_error` | `e target-open` |

Required methods:

- `stride_reference`.
- `consistency_ablation`.
- `geometry_ablation`.
- `recurrence_ablation`.

Visualization:

- One PDF with two vertically stacked point-range blocks.
- Top block: `Community-level error`.
- Bottom block: `Overall amount error`.
- Each block uses right-side ladder brackets comparing each ablation arm to its
  matched STRIDE reference.

Statistical comparison:

- Aggregate patient metrics to rerun-level means before testing.
- Compare each ablation arm to the matched `stride_reference` rows from the
  same ablation family.
- Statistical labels show paired Wilcoxon signed-rank comparisons against
  STRIDE across rerun-level means, with BH adjustment across displayed
  comparisons.
- Run tests on unscaled values.
- Display bracket labels only.

Output PDF:

- `sf2_panelD_ablation_secondary_checks.pdf`.

## Metrics Not Planned For Visual Panels

Formal outputs retain additional Block 3 metrics beyond the figure-level
display vocabulary. Figure 2 visual panels use the MAE/MSE/overall-amount
vocabulary defined above. `F_L1_total`, `g_L1_total`, and `e_L1_total` remain
available in formal outputs and are not part of the revised Figure 2 or
Supplementary Figure 2 display vocabulary.

The direct rerun-level comparisons in Supplementary Figure 2B are held-out
generator-audit displays. They are separate from the formal Block 3
method-bearing ratio metrics listed below.

- `offdiag_ratio`;
- `depletion_capture`;
- `emergence_capture`;
- `endpoint_y_MAE`;
- `target_recall_at_k`;
- `open_support_F1`;
