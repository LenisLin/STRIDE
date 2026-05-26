# Task A Figure 2 Panel PDF Workflow Skeleton
#
# Purpose:
#   Generate standalone PDF panels for Main Figure 2, Supplementary Figure 1,
#   and Supplementary Figure 2 from formal Task A outputs.
#
# Boundary:
#   - Main Figure 2A is a workflow schematic and is not generated here.
#   - This script does not modify formal output tables.
#   - This script does not assemble final multi-panel figures; final assembly,
#     alignment, and lettering are handled manually in Adobe Illustrator.
#   - Main Figure 2B is the only composite heatmap expected to be exported as
#     one assembled PDF from this script. Other multi-part panels should export
#     separate component PDFs for manual Illustrator assembly.
#   - This script is intentionally workflow-style. Do not define reusable
#     project-level plotting/helper functions in this file.
#   - Allowed global objects include path constants, label maps, factor levels,
#     named color vectors, and theme objects.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(dittoSeq)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
  library(rhdf5)
})

###--GLOBAL PATHS, LABELS, AND COLORS--###

# Result roots.
DESC <- "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/descriptive"
B0 <- "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block0"
B1 <- "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block1"
B3 <- "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/block3"
FIG_DIR <- "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/figures"
STAGE0_H5AD <- "/mnt/NAS_21T/ProjectData/STRIDE/task_A_stage0/task_A_stage0_k25.h5ad"

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Formal field names stay in code; human-readable names are used in figures.
metric_labels <- c(
  self_retention = "Source retention",
  depletion = "Source-open mass",
  off_diagonal_remodeling = "Off-diagonal remodeling",
  emergence = "Target-open mass",
  matched_incoming_burden = "Matched incoming mass",
  F_L1_total = "Total relation error",
  A_MAE_active = "Active relation MAE",
  A_MSE_active = "Active relation MSE",
  offdiag_mass_abs_error = "Off-diagonal mass error",
  g_L1_total = "Source-open total error",
  depletion_mass_abs_error = "Source-open mass error",
  d_MAE = "Source-open profile MAE",
  d_MSE = "Source-open profile MSE",
  e_L1_total = "Target-open total error",
  emergence_mass_abs_error = "Target-open mass error",
  e_MAE = "Target-open profile MAE",
  e_MSE = "Target-open profile MSE",
  community_space_target_fraction = "Community-space target",
  identity_projected_target_fraction = "Identity-projected target"
)

method_labels <- c(
  stride_reference = "STRIDE",
  balanced_ot_baseline = "Balanced OT",
  uot_baseline = "Unbalanced OT",
  partial_ot_baseline = "Partial OT",
  diagonal_transport_baseline = "Diagonal transport",
  consistency_ablation = "No consistency",
  geometry_ablation = "No geometry",
  recurrence_ablation = "No recurrence"
)

pair_family_labels <- c(
  "TC-IM" = "TC -> IM",
  "TC-PT" = "TC -> PT",
  "Null" = "Null"
)

domain_labels <- c(
  TC = "TC",
  IM = "IM",
  PT = "PT"
)

# Factor-level policy:
#   - Bind semantic fields to explicit factor levels before plotting.
#   - Use the matching named color vector with scale_color_manual() or
#     scale_fill_manual().
#   - Do not rely on row order, alphabetical order, or ggplot defaults.
baseline_method_levels <- c(
  "stride_reference",
  "balanced_ot_baseline",
  "uot_baseline",
  "partial_ot_baseline",
  "diagonal_transport_baseline"
)

ablation_method_levels <- c(
  "stride_reference",
  "consistency_ablation",
  "geometry_ablation",
  "recurrence_ablation"
)

method_levels <- unique(c(baseline_method_levels, ablation_method_levels))
pair_family_levels <- c("Null", "TC -> IM", "TC -> PT")
pair_family_formal_levels <- c("TC-IM", "TC-PT")
domain_levels <- c("TC", "IM", "PT")

display_scaling <- tibble::tribble(
  ~metric_name, ~display_multiplier, ~axis_label,
  "A_MAE_active", 1e3, "Active relation MAE (x10^3)",
  "d_MAE", 1e3, "Source-open profile MAE (x10^3)",
  "e_MAE", 1e3, "Target-open profile MAE (x10^3)",
  "A_MSE_active", 1e6, "Active relation MSE (x10^6)",
  "d_MSE", 1e3, "Source-open profile MSE (x10^3)",
  "e_MSE", 1e3, "Target-open profile MSE (x10^3)"
)

# Color policy:
#   - Derive discrete colors from dittoSeq::dittoColors().
#   - Reuse the same named colors across all panels.
#   - Do not create panel-local colors for the same semantic object.
#   - Missing method slots are shown as light gray NA marks and are not added
#     as a legend category.
ditto_colors <- dittoSeq::dittoColors()

method_colors <- c(
  stride_reference = ditto_colors[[1]],
  balanced_ot_baseline = ditto_colors[[2]],
  uot_baseline = ditto_colors[[3]],
  partial_ot_baseline = ditto_colors[[4]],
  diagonal_transport_baseline = ditto_colors[[5]],
  consistency_ablation = ditto_colors[[6]],
  geometry_ablation = ditto_colors[[7]],
  recurrence_ablation = ditto_colors[[8]]
)
method_colors <- method_colors[method_levels]

pair_family_colors <- c(
  "Null" = "grey75",
  "TC -> IM" = ditto_colors[[9]],
  "TC -> PT" = ditto_colors[[10]]
)
pair_family_colors <- pair_family_colors[pair_family_levels]

domain_colors <- c(
  TC = ditto_colors[[11]],
  IM = ditto_colors[[12]],
  PT = ditto_colors[[13]]
)
domain_colors <- domain_colors[domain_levels]

prevalence_colors <- c(
  "Patient prevalence" = ggsci::pal_jama()(2)[[1]],
  "ROI prevalence" = ggsci::pal_jama()(2)[[2]]
)

# Expected ggplot binding pattern:
#   method_name <- factor(method_name, levels = baseline_method_levels)
#   method_name <- factor(method_name, levels = ablation_method_levels)
#   pair_family_label <- factor(pair_family_label, levels = pair_family_levels)
#   domain_label <- factor(domain_label, levels = domain_levels)
#   scale_color_manual(values = method_colors, labels = method_labels)
#   scale_fill_manual(values = pair_family_colors)
#   scale_fill_manual(values = domain_colors)

na_color <- "grey85"

base_theme <- theme_classic(base_size = 8) +
  theme(
    legend.title = element_blank(),
    strip.background = element_blank(),
    panel.grid = element_blank()
  )


###--Main Figure 2 Panel B: Community Annotation Overview--###

# Input:
#   DESC/tables/community_cell_subtype_row_fractions.csv
#   DESC/tables/community_domain_distribution.csv
#   DESC/tables/community_patient_occurrence_summary.csv
# Field meaning:
#   community_id identifies shared atlas communities.
#   Cell-subtype columns store fraction within community.
#   domain_label and fraction_within_community define TC/IM/PT annotation.
#   patient_prevalence, roi_prevalence, and n_patients_present support
#   prevalence side annotations.
# Result intent:
#   Show community-level annotation by cell-subpopulation composition,
#   tissue-domain distribution, and patient/ROI prevalence.
# Visualization:
#   Integrated annotation heatmap with cell-subtype fractions as the main body,
#   TC/IM/PT stacked annotation, and patient/ROI prevalence bars.
# Axes/value definition:
#   y-axis: communities.
#   x-axis: cell subpopulations.
#   Fill: fraction within community.
#   Domain annotation fill: TC, IM, PT with domain_colors.
# Output PDF:
#   main2_panelB_community_annotation_overview.pdf
#
# Data read and processing:
panelB_cell_fractions <- readr::read_csv(
  file.path(DESC, "tables", "community_cell_subtype_row_fractions.csv"),
  show_col_types = FALSE
)
panelB_domain_distribution <- readr::read_csv(
  file.path(DESC, "tables", "community_domain_distribution.csv"),
  show_col_types = FALSE
) %>%
  mutate(domain_label = factor(domain_label, levels = domain_levels))
panelB_patient_occurrence <- readr::read_csv(
  file.path(DESC, "tables", "community_patient_occurrence_summary.csv"),
  show_col_types = FALSE
)

panelB_cell_subtype_cols <- setdiff(colnames(panelB_cell_fractions), "community_id")
panelB_row_order <- panelB_cell_fractions %>%
  mutate(
    tumor_fraction = TC_CAIX + TC_EpCAM + TC_Ki67 + TC_VEGF
  ) %>%
  left_join(
    panelB_patient_occurrence %>%
      select(community_id, patient_prevalence, roi_prevalence, n_patients_present),
    by = "community_id"
  ) %>%
  arrange(desc(tumor_fraction), desc(patient_prevalence), community_id) %>%
  mutate(community_label = paste0("C", community_id))

panelB_fraction_mat <- panelB_cell_fractions %>%
  semi_join(panelB_row_order, by = "community_id") %>%
  slice(match(panelB_row_order$community_id, community_id)) %>%
  select(all_of(panelB_cell_subtype_cols)) %>%
  as.matrix()
rownames(panelB_fraction_mat) <- panelB_row_order$community_label

panelB_domain_mat <- panelB_domain_distribution %>%
  filter(community_id %in% panelB_row_order$community_id) %>%
  select(community_id, domain_label, fraction_within_community) %>%
  tidyr::complete(
    community_id = panelB_row_order$community_id,
    domain_label = factor(domain_levels, levels = domain_levels),
    fill = list(fraction_within_community = 0)
  ) %>%
  arrange(match(community_id, panelB_row_order$community_id), domain_label) %>%
  tidyr::pivot_wider(
    names_from = domain_label,
    values_from = fraction_within_community,
    values_fill = 0
  ) %>%
  select(all_of(domain_levels)) %>%
  as.matrix()
rownames(panelB_domain_mat) <- panelB_row_order$community_label

panelB_prevalence_mat <- panelB_row_order %>%
  transmute(
    `Patient prevalence` = patient_prevalence,
    `ROI prevalence` = roi_prevalence
  ) %>%
  as.matrix()
rownames(panelB_prevalence_mat) <- panelB_row_order$community_label
#
# Visualization block:
panelB_fraction_col_fun <- circlize::colorRamp2(
  c(0, 0.25, 0.5, max(1, max(panelB_fraction_mat, na.rm = TRUE))),
  c("#f7f7f7", "#fdd0a2", "#fb6a4a", "#a50f15")
)

panelB_right_annotation <- rowAnnotation(
  `Domain fraction` = anno_barplot(
    panelB_domain_mat,
    bar_width = 0.9,
    gp = gpar(fill = domain_colors[colnames(panelB_domain_mat)], col = NA),
    axis_param = list(
      side = "bottom",
      at = c(0, 0.5, 1),
      labels = c("0", "0.5", "1"),
      gp = gpar(fontsize = 6)
    ),
    width = unit(2.2, "cm")
  ),
  `Prevalence` = anno_barplot(
    panelB_prevalence_mat,
    beside = TRUE,
    bar_width = 0.8,
    gp = gpar(fill = prevalence_colors[colnames(panelB_prevalence_mat)], col = NA),
    axis_param = list(
      side = "bottom",
      at = c(0, 0.5, 1),
      labels = c("0", "0.5", "1"),
      gp = gpar(fontsize = 6)
    ),
    width = unit(2.6, "cm")
  ),
  annotation_name_gp = gpar(fontsize = 7),
  annotation_name_rot = 90
)

panelB_heatmap <- Heatmap(
  panelB_fraction_mat,
  name = "Fraction",
  col = panelB_fraction_col_fun,
  cluster_rows = FALSE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_gp = gpar(fontsize = 7),
  column_names_gp = gpar(fontsize = 6),
  column_names_rot = 45,
  row_title = "Community",
  column_title = "Cell subtype fraction within community",
  column_title_gp = gpar(fontsize = 9, fontface = "bold"),
  heatmap_legend_param = list(
    title = "Cell subtype\nfraction",
    title_gp = gpar(fontsize = 8),
    labels_gp = gpar(fontsize = 7)
  ),
  right_annotation = panelB_right_annotation,
  rect_gp = gpar(col = "grey92", lwd = 0.25)
)
#
# PDF export:
pdf(
  file.path(FIG_DIR, "main2_panelB_community_annotation_overview.pdf"),
  width = 10.8,
  height = 7.2,
  useDingbats = FALSE
)
draw(
  panelB_heatmap,
  column_title = "Community annotation overview",
  column_title_gp = gpar(fontsize = 11, fontface = "bold"),
  heatmap_legend_side = "right",
  annotation_legend_side = "right",
  annotation_legend_list = list(
    Legend(
      labels = domain_levels,
      title = "Domain",
      legend_gp = gpar(fill = domain_colors[domain_levels])
    ),
    Legend(
      labels = c("Patient", "ROI"),
      title = "Prevalence bars",
      legend_gp = gpar(fill = prevalence_colors)
    )
  )
)
dev.off()


###--Main Figure 2 Panel C: Overall Raw Relation Summaries--###

# Input:
#   B0/block0_metric_summary.csv
#   B1/block1_family_statistical_supplement.csv
# Field meaning:
#   summary_name is mapped through metric_labels.
#   Block 0 provides Null versus TC -> IM empirical-null summaries.
#   Block 1 provides TC -> IM versus TC -> PT paired cohort summaries.
#   tc_im_median, tc_pt_median, real_value, null_reference, empirical_p_value,
#   wilcoxon_p_value, bh_q_value, and q_pass support values and labels.
# Result intent:
#   Show cohort-level overall relation summaries for Null, TC -> IM, and
#   TC -> PT without patient-level detail.
# Visualization:
#   Cohort-level grouped point or bar plot.
# Axes/value definition:
#   x-axis: metric label.
#   y-axis: raw summary value.
#   color/fill: Null, TC -> IM, TC -> PT using pair_family_colors.
#   Required metrics: self_retention, depletion, off_diagonal_remodeling,
#   emergence; scale == "burden_weighted".
# Output PDF:
#   main2_panelC_overall_raw_relation_summaries.pdf
#
# Data read and processing:
panelC_metrics <- c(
  "self_retention",
  "depletion",
  "off_diagonal_remodeling",
  "emergence"
)
panelC_block0 <- readr::read_csv(
  file.path(B0, "block0_metric_summary.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    cohort_stat == "median",
    summary_name %in% panelC_metrics
  )
panelC_block1 <- readr::read_csv(
  file.path(B1, "block1_family_statistical_supplement.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    summary_name %in% panelC_metrics
  )

panelC_plot_data <- bind_rows(
  panelC_block0 %>%
    transmute(
      summary_name,
      group = "Null",
      value = null_reference,
      p_value = empirical_p_value,
      q_value = NA_real_
    ),
  panelC_block0 %>%
    transmute(
      summary_name,
      group = "TC -> IM",
      value = real_value,
      p_value = empirical_p_value,
      q_value = NA_real_
    ),
  panelC_block1 %>%
    transmute(
      summary_name,
      group = "TC -> PT",
      value = tc_pt_median,
      p_value = wilcoxon_p_value,
      q_value = bh_q_value
    )
) %>%
  mutate(
    metric_label = factor(
      unname(metric_labels[summary_name]),
      levels = unname(metric_labels[panelC_metrics])
    ),
    group = factor(group, levels = pair_family_levels),
    group_y = factor(as.character(group), levels = rev(pair_family_levels))
  )

panelC_pair_family_colors <- c(
  "Null" = "#d9d9d9",
  "TC -> IM" = ggsci::pal_jama()(3)[[1]],
  "TC -> PT" = ggsci::pal_jama()(3)[[2]]
)
panelC_pair_family_colors <- panelC_pair_family_colors[pair_family_levels]

panelC_annotation_x <- panelC_plot_data %>%
  group_by(summary_name, metric_label) %>%
  summarise(
    x_min = min(value, na.rm = TRUE),
    x_max = max(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    x_range = pmax(x_max - pmin(x_min, 0), abs(x_max) * 0.25, 0.02),
    x_tick = x_range * 0.08,
    x_empirical = x_max + x_range * 0.16,
    x_bh = x_max + x_range * 0.36
  )

panelC_stat_annotations <- bind_rows(
  panelC_block0 %>%
    transmute(
      summary_name,
      contrast_label = "emp. p",
      stat_value = empirical_p_value,
      group_start = "Null",
      group_end = "TC -> IM"
    ),
  panelC_block1 %>%
    transmute(
      summary_name,
      contrast_label = "BH q",
      stat_value = bh_q_value,
      group_start = "TC -> IM",
      group_end = "TC -> PT"
    )
) %>%
  left_join(panelC_annotation_x, by = "summary_name") %>%
  mutate(
    stat_star = case_when(
      is.na(stat_value) ~ "",
      stat_value < 0.001 ~ "***",
      stat_value < 0.01 ~ "**",
      stat_value <= 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    stat_label = paste(contrast_label, stat_star),
    x_position = if_else(contrast_label == "emp. p", x_empirical, x_bh),
    x_lower = x_position - x_tick,
    group_start_y = factor(group_start, levels = rev(pair_family_levels)),
    group_end_y = factor(group_end, levels = rev(pair_family_levels)),
    label_y = group_end_y
  )
#
# Visualization block:
panelC_plot <- ggplot(
  panelC_plot_data,
  aes(x = value, y = group_y, fill = group)
) +
  geom_col(
    width = 0.58,
    color = "grey25",
    linewidth = 0.18
  ) +
  geom_point(
    aes(color = group),
    size = 1.15,
    stroke = 0
  ) +
  geom_segment(
    data = panelC_stat_annotations,
    aes(x = x_position, xend = x_position, y = group_start_y, yend = group_end_y),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey25"
  ) +
  geom_segment(
    data = panelC_stat_annotations,
    aes(x = x_lower, xend = x_position, y = group_start_y, yend = group_start_y),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey25"
  ) +
  geom_segment(
    data = panelC_stat_annotations,
    aes(x = x_lower, xend = x_position, y = group_end_y, yend = group_end_y),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey25"
  ) +
  geom_text(
    data = panelC_stat_annotations,
    aes(x = x_position + x_tick * 0.45, y = label_y, label = stat_label),
    inherit.aes = FALSE,
    size = 2.0,
    color = "grey15",
    hjust = 0
  ) +
  facet_wrap(~ metric_label, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = panelC_pair_family_colors, drop = FALSE) +
  scale_color_manual(values = panelC_pair_family_colors, drop = FALSE) +
  scale_y_discrete(expand = expansion(add = 0.18)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.24))) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Raw cohort median",
    y = NULL,
    fill = NULL,
    color = NULL
  ) +
  base_theme +
  theme(
    legend.position = "top",
    panel.spacing.x = unit(0.55, "lines"),
    strip.text = element_text(size = 7),
    plot.margin = margin(5.5, 24, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "main2_panelC_overall_raw_relation_summaries.pdf"),
  plot = panelC_plot,
  width = 10.2,
  height = 2.35,
  device = cairo_pdf
)


###--Main Figure 2 Panel D: Tumor-Dominant Community Relation Behavior--###

# Input:
#   DESC/tables/community_cell_subtype_row_fractions.csv
#   B1/block1_source_community_comparison.csv
#   B1/block1_source_community_statistical_supplement.csv
#   DESC/tables/community_patient_occurrence_summary.csv
# Field meaning:
#   Tumor-dominant rule uses TC_CAIX + TC_EpCAM + TC_Ki67 + TC_VEGF > 0.5.
#   source_community_id identifies source communities.
#   summary_name selects source-side metrics.
#   tc_im_value/tc_pt_value store patient-level paired values.
#   tc_im_median/tc_pt_median store cohort medians.
#   bh_q_value and q_pass provide support labels.
# Result intent:
#   Show tumor-dominant source communities and their TC -> IM versus TC -> PT
#   source-side relation behavior.
# Visualization:
#   Hybrid panel: community-level dumbbell plot for all tumor-dominant
#   communities plus selected paired patient examples.
# Axes/value definition:
#   Dumbbell rows: tumor-dominant source communities.
#   Dumbbell x-axis: patient-level median raw value with IQR.
#   Dumbbell facets: Source retention and Source-open mass.
#   Boxplot x-axis: Source retention and Source-open mass.
#   Example y-axis: raw metric value.
#   Boxplot facets: tumor-dominant communities; rows use
#   comparison_status == "estimable".
# Output PDFs:
#   main2_panelD_tumor_dominant_community_heatmap.pdf
#   main2_panelD_tumor_dominant_patient_examples.pdf
#
# Data read and processing:
panelD_cell_fractions <- readr::read_csv(
  file.path(DESC, "tables", "community_cell_subtype_row_fractions.csv"),
  show_col_types = FALSE
)
panelD_source_comparison <- readr::read_csv(
  file.path(B1, "block1_source_community_comparison.csv"),
  show_col_types = FALSE
)
panelD_source_stats <- readr::read_csv(
  file.path(B1, "block1_source_community_statistical_supplement.csv"),
  show_col_types = FALSE
)
panelD_patient_occurrence <- readr::read_csv(
  file.path(DESC, "tables", "community_patient_occurrence_summary.csv"),
  show_col_types = FALSE
)

panelD_tumor_communities <- panelD_cell_fractions %>%
  mutate(tumor_fraction = TC_CAIX + TC_EpCAM + TC_Ki67 + TC_VEGF) %>%
  filter(tumor_fraction > 0.5) %>%
  left_join(
    panelD_patient_occurrence %>%
      select(community_id, patient_prevalence, roi_prevalence),
    by = "community_id"
  ) %>%
  arrange(community_id)

panelD_expected_communities <- c(0, 1, 3, 6, 10, 11, 12, 16, 17)
if (!identical(panelD_tumor_communities$community_id, panelD_expected_communities)) {
  warning(
    "Tumor-dominant communities differ from expected set: ",
    paste(panelD_tumor_communities$community_id, collapse = ", ")
  )
}

panelD_metrics <- c("self_retention", "depletion")
panelD_pair_stats <- panelD_source_stats %>%
  filter(
    source_community_id %in% panelD_tumor_communities$community_id,
    summary_name %in% panelD_metrics
  ) %>%
  select(
    source_community_id,
    summary_name,
    n_estimable,
    tc_im_median,
    tc_pt_median,
    median_delta,
    wilcoxon_p_value,
    bh_q_value,
    q_pass
  ) %>%
  mutate(
    significance_label = case_when(
      is.na(bh_q_value) ~ "ns",
      bh_q_value < 0.001 ~ "***",
      bh_q_value < 0.01 ~ "**",
      bh_q_value <= 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    significance_text = paste0(significance_label, "  BH q=", scales::pvalue(bh_q_value, accuracy = 0.001)),
    metric_label = factor(
      unname(metric_labels[summary_name]),
      levels = unname(metric_labels[panelD_metrics])
    ),
    community_label = factor(
      paste0("C", source_community_id),
      levels = paste0("C", rev(panelD_tumor_communities$community_id))
    )
  )

panelD_dumbbell_y_order <- panelD_pair_stats %>%
  arrange(metric_label, median_delta) %>%
  mutate(community_metric_label = paste(community_label, metric_label, sep = "___"))

panelD_dumbbell_y_labels <- setNames(
  as.character(panelD_dumbbell_y_order$community_label),
  panelD_dumbbell_y_order$community_metric_label
)

panelD_dumbbell_data <- panelD_pair_stats %>%
  mutate(community_metric_label = paste(community_label, metric_label, sep = "___")) %>%
  pivot_longer(
    cols = c(tc_im_median, tc_pt_median),
    names_to = "pair_family",
    values_to = "value"
  ) %>%
  mutate(
    pair_family = recode(
      pair_family,
      tc_im_median = "TC -> IM",
      tc_pt_median = "TC -> PT"
    ),
    pair_family = factor(pair_family, levels = pair_family_levels),
    community_metric_label = factor(
      community_metric_label,
      levels = panelD_dumbbell_y_order$community_metric_label
    )
  )

panelD_dumbbell_segments <- panelD_dumbbell_data %>%
  mutate(pair_key = recode(as.character(pair_family), "TC -> IM" = "tc_im", "TC -> PT" = "tc_pt")) %>%
  select(source_community_id, summary_name, metric_label, community_metric_label, pair_key, value) %>%
  pivot_wider(names_from = pair_key, values_from = value)

panelD_community_facet_order <- panelD_pair_stats %>%
  filter(summary_name == "self_retention") %>%
  arrange(desc(median_delta)) %>%
  transmute(community_facet = paste0("C", source_community_id)) %>%
  pull(community_facet)

panelD_example_stats <- panelD_pair_stats %>%
  mutate(
    community_facet = factor(
      paste0("C", source_community_id),
      levels = panelD_community_facet_order
    ),
    metric_x = match(summary_name, panelD_metrics)
  )

panelD_patient_examples <- panelD_source_comparison %>%
  filter(
    source_community_id %in% panelD_tumor_communities$community_id,
    summary_name %in% panelD_metrics,
    comparison_status == "estimable"
  ) %>%
  select(patient_id, source_community_id, summary_name, tc_im_value, tc_pt_value) %>%
  filter(!is.na(tc_im_value), !is.na(tc_pt_value)) %>%
  pivot_longer(
    cols = c(tc_im_value, tc_pt_value),
    names_to = "pair_family",
    values_to = "value"
  ) %>%
  mutate(
    pair_family = recode(
      pair_family,
      tc_im_value = "TC -> IM",
      tc_pt_value = "TC -> PT"
    ),
    pair_family = factor(pair_family, levels = pair_family_levels),
    metric_label = factor(
      unname(metric_labels[summary_name]),
      levels = unname(metric_labels[panelD_metrics])
    ),
    metric_x = match(summary_name, panelD_metrics),
    community_facet = factor(
      paste0("C", source_community_id),
      levels = panelD_community_facet_order
    ),
    point_x = metric_x + if_else(as.character(pair_family) == "TC -> IM", -0.18, 0.18),
    patient_id = factor(patient_id)
  )

panelD_patient_pair_segments <- panelD_patient_examples %>%
  select(community_facet, metric_label, patient_id, pair_family, point_x, value) %>%
  mutate(pair_key = recode(as.character(pair_family), "TC -> IM" = "tc_im", "TC -> PT" = "tc_pt")) %>%
  select(-pair_family) %>%
  pivot_wider(
    names_from = pair_key,
    values_from = c(point_x, value)
  ) %>%
  filter(!is.na(value_tc_im), !is.na(value_tc_pt))

panelD_example_medians <- panelD_patient_examples %>%
  group_by(community_facet, metric_label, metric_x, pair_family, point_x) %>%
  summarise(value = median(value, na.rm = TRUE), .groups = "drop")

panelD_facet_annotation_y <- panelD_patient_examples %>%
  group_by(community_facet) %>%
  summarise(
    facet_y_min = min(value, na.rm = TRUE),
    facet_y_max = max(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    facet_y_range = pmax(facet_y_max - facet_y_min, 0.01),
    facet_bracket_base = facet_y_max + facet_y_range * 0.07,
    facet_label_base = facet_y_max + facet_y_range * 0.14
  )

panelD_example_annotations <- panelD_patient_examples %>%
  group_by(community_facet, metric_label, metric_x) %>%
  summarise(.groups = "drop") %>%
  left_join(panelD_facet_annotation_y, by = "community_facet") %>%
  mutate(
    y_bracket = facet_bracket_base + (metric_x - 1) * facet_y_range * 0.06,
    y_label = facet_label_base + (metric_x - 1) * facet_y_range * 0.06,
    x_start = metric_x - 0.18,
    x_end = metric_x + 0.18,
    x_mid = metric_x
  ) %>%
  left_join(
    panelD_example_stats %>%
      select(source_community_id, summary_name, community_facet, metric_label, significance_label),
    by = c("community_facet", "metric_label")
  )
#
# Visualization block:
panelD_heatmap_plot <- ggplot(
  panelD_dumbbell_data,
  aes(x = value, y = community_metric_label)
) +
  geom_segment(
    data = panelD_dumbbell_segments,
    aes(x = tc_im, xend = tc_pt, y = community_metric_label, yend = community_metric_label),
    inherit.aes = FALSE,
    color = "grey70",
    linewidth = 0.35
  ) +
  geom_point(
    aes(fill = pair_family, color = pair_family),
    shape = 21,
    size = 2.0,
    stroke = 0.25
  ) +
  facet_wrap(~ metric_label, scales = "free", nrow = 1) +
  scale_fill_manual(values = pair_family_colors, drop = FALSE) +
  scale_color_manual(values = pair_family_colors, drop = FALSE) +
  scale_y_discrete(labels = panelD_dumbbell_y_labels) +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.08))) +
  coord_cartesian(clip = "off") +
  labs(
    x = "Raw cohort median",
    y = "Tumor-dominant community",
    fill = NULL,
    color = NULL
  ) +
  base_theme +
  theme(
    legend.position = "top",
    panel.spacing.x = unit(1.0, "lines"),
    plot.margin = margin(5.5, 8, 5.5, 5.5)
  )

panelD_examples_plot <- ggplot(
  panelD_patient_examples,
  aes(x = metric_x, y = value)
) +
  geom_segment(
    data = panelD_patient_pair_segments,
    aes(
      x = point_x_tc_im,
      xend = point_x_tc_pt,
      y = value_tc_im,
      yend = value_tc_pt
    ),
    inherit.aes = FALSE,
    color = "grey78",
    linewidth = 0.16,
    alpha = 0.55
  ) +
  geom_boxplot(
    aes(group = interaction(metric_x, pair_family), fill = pair_family),
    position = position_dodge(width = 0.62),
    width = 0.42,
    outlier.shape = NA,
    alpha = 0.55,
    color = "grey25",
    linewidth = 0.25
  ) +
  geom_point(
    aes(x = point_x, color = pair_family),
    position = position_jitter(width = 0.012, height = 0),
    size = 0.45,
    alpha = 0.35
  ) +
  geom_point(
    data = panelD_example_medians,
    aes(x = point_x, y = value, fill = pair_family, color = pair_family),
    inherit.aes = FALSE,
    shape = 23,
    size = 1.4,
    stroke = 0.2
  ) +
  geom_segment(
    data = panelD_example_annotations,
    aes(x = x_start, xend = x_end, y = y_bracket, yend = y_bracket),
    inherit.aes = FALSE,
    linewidth = 0.25,
    color = "grey20"
  ) +
  geom_text(
    data = panelD_example_annotations,
    aes(x = x_mid, y = y_label, label = significance_label),
    inherit.aes = FALSE,
    size = 2.5,
    color = "grey10"
  ) +
  facet_wrap(~ community_facet, scales = "free_y", nrow = 3) +
  scale_fill_manual(values = pair_family_colors, drop = FALSE) +
  scale_color_manual(values = pair_family_colors, drop = FALSE) +
  scale_x_continuous(
    breaks = seq_along(panelD_metrics),
    labels = unname(metric_labels[panelD_metrics]),
    limits = c(0.55, length(panelD_metrics) + 0.45)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.22))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Raw patient value") +
  base_theme +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
    panel.spacing = unit(0.7, "lines"),
    strip.text = element_text(size = 7),
    plot.margin = margin(5.5, 8, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "main2_panelD_tumor_dominant_community_heatmap.pdf"),
  plot = panelD_heatmap_plot,
  width = 7.3,
  height = 4.0,
  device = cairo_pdf
)
ggsave(
  filename = file.path(FIG_DIR, "main2_panelD_tumor_dominant_patient_examples.pdf"),
  plot = panelD_examples_plot,
  width = 8.8,
  height = 5.2,
  device = cairo_pdf
)


###--Main Figure 2 Panel E: Semi-Synthetic Benchmark Against Transport Baselines--###

# Input:
#   B3/a_benchmark/raw/a_benchmark/condition_summary.csv
#   B3/a_benchmark/raw/a_benchmark/patient_metrics.csv
#   B3/de_benchmark/raw/de_benchmark/condition_summary.csv
#   B3/de_benchmark/raw/de_benchmark/patient_metrics.csv
# Field meaning:
#   method_name is mapped through method_labels.
#   metric_name is mapped through metric_labels and display_scaling.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show controlled hidden-truth recovery errors for STRIDE and transport
#   baselines across relation and open-channel metrics.
# Visualization:
#   Single-row faceted lower-is-better horizontal point-range plot.
# Axes/value definition:
#   x-axis: error value, with display scaling in facet labels.
#   y-axis: method.
#   facets: F_L1_total, A_MAE_active, offdiag_mass_abs_error, g_L1_total,
#   d_MAE, e_L1_total, e_MAE.
#   balanced_ot_baseline is shown as a light-gray NA slot in open-channel
#   facets where absent from formal outputs.
# Output PDF:
#   main2_panelE_transport_baseline_benchmark.pdf
#
# Data read and processing:
panelE_metrics_relation <- c("F_L1_total", "A_MAE_active", "offdiag_mass_abs_error")
panelE_metrics_open <- c("g_L1_total", "d_MAE", "e_L1_total", "e_MAE")
panelE_metrics <- c(panelE_metrics_relation, panelE_metrics_open)

panelE_summary <- bind_rows(
  readr::read_csv(
    file.path(B3, "a_benchmark", "raw", "a_benchmark", "condition_summary.csv"),
    show_col_types = FALSE
  ) %>%
    mutate(panel_channel = "Relation") %>%
    filter(metric_name %in% panelE_metrics_relation),
  readr::read_csv(
    file.path(B3, "de_benchmark", "raw", "de_benchmark", "condition_summary.csv"),
    show_col_types = FALSE
  ) %>%
    mutate(panel_channel = "Open") %>%
    filter(metric_name %in% panelE_metrics_open)
) %>%
  filter(method_name %in% baseline_method_levels) %>%
  left_join(display_scaling, by = "metric_name") %>%
  mutate(
    panel_channel = factor(panel_channel, levels = c("Relation", "Open")),
    display_multiplier = coalesce(display_multiplier, 1),
    metric_facet = if_else(
      is.na(axis_label),
      unname(metric_labels[metric_name]),
      axis_label
    ),
    metric_facet = factor(
      metric_facet,
      levels = if_else(
        panelE_metrics %in% display_scaling$metric_name,
        display_scaling$axis_label[match(panelE_metrics, display_scaling$metric_name)],
        unname(metric_labels[panelE_metrics])
      )
    ),
    method_name = factor(method_name, levels = baseline_method_levels),
    method_label = factor(
      unname(method_labels[as.character(method_name)]),
      levels = unname(method_labels[baseline_method_levels])
    ),
    mean_display = mean_value * display_multiplier,
    ci_lower_display = ci_lower * display_multiplier,
    ci_upper_display = ci_upper * display_multiplier,
    is_missing = FALSE
  )

panelE_summary_complete <- tidyr::expand_grid(
  metric_name = panelE_metrics,
  method_name = baseline_method_levels
) %>%
  left_join(
    panelE_summary,
    by = c("metric_name", "method_name")
  ) %>%
  mutate(
    method_name = factor(method_name, levels = baseline_method_levels),
    method_label = factor(
      unname(method_labels[as.character(method_name)]),
      levels = rev(unname(method_labels[baseline_method_levels]))
    ),
    panel_channel = coalesce(
      as.character(panel_channel),
      if_else(metric_name %in% panelE_metrics_relation, "Relation", "Open")
    ),
    panel_channel = factor(panel_channel, levels = c("Relation", "Open")),
    display_multiplier = coalesce(
      display_multiplier,
      display_scaling$display_multiplier[match(metric_name, display_scaling$metric_name)],
      1
    ),
    metric_facet = coalesce(
      as.character(metric_facet),
      if_else(
        metric_name %in% display_scaling$metric_name,
        display_scaling$axis_label[match(metric_name, display_scaling$metric_name)],
        unname(metric_labels[metric_name])
      )
    ),
    metric_facet = factor(
      metric_facet,
      levels = if_else(
        panelE_metrics %in% display_scaling$metric_name,
        display_scaling$axis_label[match(panelE_metrics, display_scaling$metric_name)],
        unname(metric_labels[panelE_metrics])
      )
    ),
    is_missing = is.na(mean_value)
  )

panelE_patient_metrics <- bind_rows(
  readr::read_csv(
    file.path(B3, "a_benchmark", "raw", "a_benchmark", "patient_metrics.csv"),
    show_col_types = FALSE
  ) %>%
    filter(metric_name %in% panelE_metrics_relation),
  readr::read_csv(
    file.path(B3, "de_benchmark", "raw", "de_benchmark", "patient_metrics.csv"),
    show_col_types = FALSE
  ) %>%
    filter(metric_name %in% panelE_metrics_open)
) %>%
  filter(method_name %in% baseline_method_levels) %>%
  group_by(rerun_id, method_name, metric_name) %>%
  summarise(reported_value = mean(reported_value, na.rm = TRUE), .groups = "drop")

panelE_reference_stats <- panelE_patient_metrics %>%
  filter(method_name == "stride_reference") %>%
  select(rerun_id, metric_name, stride_value = reported_value)
panelE_stat_tests <- panelE_patient_metrics %>%
  filter(method_name != "stride_reference") %>%
  inner_join(panelE_reference_stats, by = c("rerun_id", "metric_name")) %>%
  group_by(metric_name, method_name) %>%
  summarise(
    n_pairs = n(),
    p_value = if (n() > 1 && any(reported_value != stride_value)) {
      wilcox.test(reported_value, stride_value, paired = TRUE, exact = FALSE)$p.value
    } else {
      NA_real_
    },
    .groups = "drop"
  ) %>%
  mutate(
    q_value = p.adjust(p_value, method = "BH"),
    significance_label = case_when(
      is.na(q_value) ~ "",
      q_value < 0.001 ~ "***",
      q_value < 0.01 ~ "**",
      q_value <= 0.05 ~ "*",
      TRUE ~ ""
    )
  )

panelE_summary_complete <- panelE_summary_complete %>%
  left_join(
    panelE_stat_tests %>%
      select(metric_name, method_name, q_value, significance_label),
    by = c("metric_name", "method_name")
  )

panelE_annotation_x <- panelE_summary_complete %>%
  filter(!is_missing) %>%
  group_by(metric_name, metric_facet) %>%
  summarise(
    x_min = min(ci_lower_display, mean_display, na.rm = TRUE),
    x_max = max(ci_upper_display, mean_display, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    x_range = pmax(x_max - x_min, 0.01),
    x_label = x_max + x_range * 0.10
  )

panelE_significance_annotations <- panelE_summary_complete %>%
  filter(
    !is_missing,
    as.character(method_name) != "stride_reference",
    !is.na(significance_label),
    significance_label != ""
  ) %>%
  left_join(panelE_annotation_x, by = c("metric_name", "metric_facet"))

panelE_na_slots <- panelE_summary_complete %>%
  filter(is_missing) %>%
  left_join(panelE_annotation_x, by = c("metric_name", "metric_facet"))
#
# Visualization block:
panelE_plot <- ggplot(
  panelE_summary_complete,
  aes(x = mean_display, y = method_label, color = method_name)
) +
  geom_vline(xintercept = 0, color = "grey90", linewidth = 0.2) +
  geom_errorbarh(
    data = panelE_summary_complete %>% filter(!is_missing),
    aes(xmin = ci_lower_display, xmax = ci_upper_display),
    height = 0.18,
    linewidth = 0.28
  ) +
  geom_point(
    data = panelE_summary_complete %>% filter(!is_missing),
    aes(fill = method_name),
    shape = 21,
    size = 1.8,
    stroke = 0.25
  ) +
  geom_text(
    data = panelE_na_slots,
    aes(x = coalesce(x_label, 0), y = method_label, label = "NA"),
    color = na_color,
    size = 2.1,
    inherit.aes = FALSE
  ) +
  geom_text(
    data = panelE_significance_annotations,
    aes(x = x_label, y = method_label, label = significance_label),
    inherit.aes = FALSE,
    hjust = 0,
    size = 2.3,
    color = "grey10"
  ) +
  facet_wrap(~ metric_facet, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = method_colors[baseline_method_levels],
    breaks = baseline_method_levels,
    labels = method_labels[baseline_method_levels],
    drop = FALSE,
    na.translate = FALSE
  ) +
  scale_fill_manual(
    values = method_colors[baseline_method_levels],
    breaks = baseline_method_levels,
    labels = method_labels[baseline_method_levels],
    drop = FALSE,
    na.translate = FALSE
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.24))) +
  coord_cartesian(clip = "off") +
  labs(x = "Error, lower is better", y = NULL) +
  base_theme +
  theme(
    legend.position = "top",
    axis.text.y = element_text(size = 6.5),
    strip.text = element_text(size = 6.3),
    panel.spacing.x = unit(0.55, "lines"),
    plot.margin = margin(5.5, 18, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "main2_panelE_transport_baseline_benchmark.pdf"),
  plot = panelE_plot,
  width = 13.2,
  height = 2.8,
  device = cairo_pdf
)


###--Main Figure 2 Panel F: Semi-Synthetic Objective-Component Ablation--###

# Input:
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv
#   B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv
#   B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv
# Field meaning:
#   evaluation_family identifies the ablation family.
#   method_name identifies stride_reference or the ablation arm.
#   metric_name is mapped through metric_labels and display_scaling.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show objective-component ablation performance using lower-is-better error
#   metrics comparable to Main Figure 2E where possible.
# Visualization:
#   Single-row faceted lower-is-better horizontal point-range plot.
# Axes/value definition:
#   x-axis: error value, with display scaling in facet labels.
#   y-axis: reference and ablation arms.
#   facets: F_L1_total, A_MAE_active, offdiag_mass_abs_error, g_L1_total,
#   d_MAE, e_L1_total, e_MAE.
# Output PDF:
#   main2_panelF_objective_component_ablation.pdf
#
# Data read and processing:
panelF_metrics <- c(
  "F_L1_total",
  "A_MAE_active",
  "offdiag_mass_abs_error",
  "g_L1_total",
  "d_MAE",
  "e_L1_total",
  "e_MAE"
)

panelF_method_labels <- c(
  stride_reference = "STRIDE",
  consistency_ablation = "w/o consistency",
  geometry_ablation = "w/o geometry",
  recurrence_ablation = "w/o recurrence"
)

panelF_summary_all <- bind_rows(
  readr::read_csv(
    file.path(B3, "subbag_consistency_ablation", "raw", "subbag_consistency_ablation", "condition_summary.csv"),
    show_col_types = FALSE
  ),
  readr::read_csv(
    file.path(B3, "geometry_ablation", "raw", "geometry_ablation", "condition_summary.csv"),
    show_col_types = FALSE
  ),
  readr::read_csv(
    file.path(B3, "recurrence_ablation", "raw", "recurrence_ablation", "condition_summary.csv"),
    show_col_types = FALSE
  )
) %>%
  filter(
    metric_name %in% panelF_metrics,
    method_name %in% ablation_method_levels
  )

panelF_reference_summary <- panelF_summary_all %>%
  filter(method_name == "stride_reference") %>%
  group_by(metric_name, method_name) %>%
  summarise(
    mean_value = mean(mean_value, na.rm = TRUE),
    ci_lower = mean(ci_lower, na.rm = TRUE),
    ci_upper = mean(ci_upper, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(evaluation_family = "shared_reference")

panelF_ablation_summary <- panelF_summary_all %>%
  filter(method_name != "stride_reference") %>%
  select(evaluation_family, method_name, metric_name, mean_value, ci_lower, ci_upper)

panelF_summary <- bind_rows(panelF_reference_summary, panelF_ablation_summary) %>%
  left_join(display_scaling, by = "metric_name") %>%
  mutate(
    display_multiplier = coalesce(display_multiplier, 1),
    metric_facet = if_else(
      is.na(axis_label),
      unname(metric_labels[metric_name]),
      axis_label
    ),
    metric_facet = factor(
      metric_facet,
      levels = if_else(
        panelF_metrics %in% display_scaling$metric_name,
        display_scaling$axis_label[match(panelF_metrics, display_scaling$metric_name)],
        unname(metric_labels[panelF_metrics])
      )
    ),
    method_name = factor(method_name, levels = ablation_method_levels),
    method_label = factor(
      unname(panelF_method_labels[as.character(method_name)]),
      levels = rev(unname(panelF_method_labels[ablation_method_levels]))
    ),
    mean_display = mean_value * display_multiplier,
    ci_lower_display = ci_lower * display_multiplier,
    ci_upper_display = ci_upper * display_multiplier
  )

panelF_patient_metrics <- bind_rows(
  readr::read_csv(
    file.path(B3, "subbag_consistency_ablation", "raw", "subbag_consistency_ablation", "patient_metrics.csv"),
    show_col_types = FALSE
  ),
  readr::read_csv(
    file.path(B3, "geometry_ablation", "raw", "geometry_ablation", "patient_metrics.csv"),
    show_col_types = FALSE
  ),
  readr::read_csv(
    file.path(B3, "recurrence_ablation", "raw", "recurrence_ablation", "patient_metrics.csv"),
    show_col_types = FALSE
  )
) %>%
  filter(
    metric_name %in% panelF_metrics,
    method_name %in% ablation_method_levels
  ) %>%
  group_by(rerun_id, evaluation_family, method_name, metric_name) %>%
  summarise(reported_value = mean(reported_value, na.rm = TRUE), .groups = "drop")

panelF_reference_stats <- panelF_patient_metrics %>%
  filter(method_name == "stride_reference") %>%
  select(rerun_id, evaluation_family, metric_name, stride_value = reported_value)
panelF_stat_tests <- panelF_patient_metrics %>%
  filter(method_name != "stride_reference") %>%
  inner_join(panelF_reference_stats, by = c("rerun_id", "evaluation_family", "metric_name")) %>%
  group_by(evaluation_family, metric_name, method_name) %>%
  summarise(
    n_pairs = n(),
    p_value = if (n() > 1 && any(reported_value != stride_value)) {
      wilcox.test(reported_value, stride_value, paired = TRUE, exact = FALSE)$p.value
    } else {
      NA_real_
    },
    .groups = "drop"
  ) %>%
  mutate(
    q_value = p.adjust(p_value, method = "BH"),
    significance_label = case_when(
      is.na(q_value) ~ "",
      q_value < 0.001 ~ "***",
      q_value < 0.01 ~ "**",
      q_value <= 0.05 ~ "*",
      TRUE ~ ""
    )
  )

panelF_summary <- panelF_summary %>%
  left_join(
    panelF_stat_tests %>%
      select(evaluation_family, metric_name, method_name, q_value, significance_label),
    by = c("evaluation_family", "metric_name", "method_name")
  )

panelF_annotation_x <- panelF_summary %>%
  group_by(metric_name, metric_facet) %>%
  summarise(
    x_min = min(ci_lower_display, mean_display, na.rm = TRUE),
    x_max = max(ci_upper_display, mean_display, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    x_range = pmax(x_max - x_min, 0.01),
    x_label = x_max + x_range * 0.10
  )

panelF_significance_annotations <- panelF_summary %>%
  filter(
    as.character(method_name) != "stride_reference",
    !is.na(significance_label),
    significance_label != ""
  ) %>%
  left_join(panelF_annotation_x, by = c("metric_name", "metric_facet"))
#
# Visualization block:
panelF_plot <- ggplot(
  panelF_summary,
  aes(x = mean_display, y = method_label, color = method_name)
) +
  geom_vline(xintercept = 0, color = "grey90", linewidth = 0.2) +
  geom_errorbarh(
    aes(xmin = ci_lower_display, xmax = ci_upper_display),
    height = 0.18,
    linewidth = 0.28
  ) +
  geom_point(
    aes(fill = method_name),
    shape = 21,
    size = 1.8,
    stroke = 0.25
  ) +
  geom_text(
    data = panelF_significance_annotations,
    aes(x = x_label, y = method_label, label = significance_label),
    inherit.aes = FALSE,
    hjust = 0,
    size = 2.3,
    color = "grey10"
  ) +
  facet_wrap(~ metric_facet, scales = "free_x", nrow = 1) +
  scale_color_manual(
    values = method_colors[ablation_method_levels],
    breaks = ablation_method_levels,
    labels = panelF_method_labels[ablation_method_levels],
    drop = FALSE,
    na.translate = FALSE
  ) +
  scale_fill_manual(
    values = method_colors[ablation_method_levels],
    breaks = ablation_method_levels,
    labels = panelF_method_labels[ablation_method_levels],
    drop = FALSE,
    na.translate = FALSE
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.24))) +
  coord_cartesian(clip = "off") +
  labs(x = "Error, lower is better", y = NULL) +
  base_theme +
  theme(
    legend.position = "top",
    axis.text.y = element_text(size = 6.5),
    strip.text = element_text(size = 6.3),
    panel.spacing.x = unit(0.55, "lines"),
    plot.margin = margin(5.5, 18, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "main2_panelF_objective_component_ablation.pdf"),
  plot = panelF_plot,
  width = 13.2,
  height = 2.7,
  device = cairo_pdf
)


###--Supplementary Figure 1 Panel A: Community Distribution Across Tissue Domains--###

# Input:
#   DESC/tables/community_domain_distribution.csv
# Field meaning:
#   community_id identifies communities.
#   domain_label identifies TC, IM, or PT.
#   fraction_within_community is the stacked-bar value.
# Result intent:
#   Show TC/IM/PT distribution profile for each community.
# Visualization:
#   Stacked bar plot.
# Axes/value definition:
#   x-axis: communities.
#   y-axis: fraction within community.
#   fill: TC, IM, PT using domain_colors.
# Output PDF:
#   sf1_panelA_community_domain_distribution.pdf
#
# Data read and processing:
sf1A_domain <- readr::read_csv(
  file.path(DESC, "tables", "community_domain_distribution.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    community_label = paste0("C", community_id),
    domain_label = factor(domain_label, levels = domain_levels)
  )

sf1A_community_order <- sf1A_domain %>%
  filter(domain_label == "TC") %>%
  arrange(desc(fraction_within_community), community_id) %>%
  pull(community_label)

sf1A_domain <- sf1A_domain %>%
  mutate(community_label = factor(community_label, levels = sf1A_community_order))
#
# Visualization block:
sf1A_plot <- ggplot(
  sf1A_domain,
  aes(x = community_label, y = fraction_within_community, fill = domain_label)
) +
  geom_col(width = 0.82, color = "grey35", linewidth = 0.12) +
  scale_fill_manual(values = domain_colors, drop = FALSE) +
  scale_y_continuous(
    labels = scales::label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = "Community",
    y = "Fraction within community",
    fill = NULL
  ) +
  base_theme +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.2)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "sf1_panelA_community_domain_distribution.pdf"),
  plot = sf1A_plot,
  width = 7.4,
  height = 3.2,
  device = cairo_pdf
)


###--Supplementary Figure 1 Panel B: Selected Community Fractions Across Tissue Domains--###

# Input:
#   STAGE0_H5AD
# Field meaning:
#   /obs/patient_id/categories and /obs/patient_id/codes identify patients.
#   /obs/compartment/categories and /obs/compartment/codes identify TC/IM/PT.
#   /obs/proto_id identifies the shared community id.
# Result intent:
#   Show patient-level tissue-domain composition for the community set used by
#   Main Figure 2D and SF1-G.
# Visualization:
#   Faceted boxplot with patient-level jitter and pairwise brackets.
# Axes/value definition:
#   x-axis: TC, IM, PT tissue domains.
#   y-axis: community fraction within each patient-domain; each facet uses a
#   free y-axis to show low-abundance communities.
#   facets: selected communities C0, C1, C2, C3, C6, C10, C11, C12, C14,
#   C16, C17, and C23.
#   statistics: paired Wilcoxon tests for displayed TC-IM, TC-PT, and IM-PT
#   comparisons; BH adjustment across 36 displayed comparisons.
# Output PDF:
#   sf1_panelB_selected_community_fraction_by_tissue.pdf
#
# Data read and processing:
sf1B_selected_communities <- c(0, 1, 2, 3, 6, 10, 11, 12, 14, 16, 17, 23)
sf1B_selected_labels <- paste0("C", sf1B_selected_communities)

sf1B_patient_categories <- rhdf5::h5read(STAGE0_H5AD, "/obs/patient_id/categories")
sf1B_patient_codes <- rhdf5::h5read(STAGE0_H5AD, "/obs/patient_id/codes")
sf1B_domain_categories <- rhdf5::h5read(STAGE0_H5AD, "/obs/compartment/categories")
sf1B_domain_codes <- rhdf5::h5read(STAGE0_H5AD, "/obs/compartment/codes")
sf1B_community_ids <- as.integer(rhdf5::h5read(STAGE0_H5AD, "/obs/proto_id"))

sf1B_cell_frame <- tibble::tibble(
  patient_id = sf1B_patient_categories[sf1B_patient_codes + 1L],
  domain_label = sf1B_domain_categories[sf1B_domain_codes + 1L],
  community_id = sf1B_community_ids
)

sf1B_domain_totals <- sf1B_cell_frame %>%
  count(patient_id, domain_label, name = "domain_total_cells")

sf1B_full_grid <- tidyr::expand_grid(
  sf1B_domain_totals,
  community_id = sf1B_selected_communities
)

sf1B_plot_data <- sf1B_cell_frame %>%
  filter(community_id %in% sf1B_selected_communities) %>%
  count(patient_id, domain_label, community_id, name = "community_cells") %>%
  right_join(
    sf1B_full_grid,
    by = c("patient_id", "domain_label", "community_id")
  ) %>%
  mutate(
    community_cells = tidyr::replace_na(community_cells, 0L),
    community_fraction = community_cells / domain_total_cells,
    domain_label = factor(domain_label, levels = domain_levels),
    community_label = factor(
      paste0("C", community_id),
      levels = sf1B_selected_labels
    ),
    patient_id = factor(patient_id)
  )

sf1B_wide <- sf1B_plot_data %>%
  select(patient_id, community_id, domain_label, community_fraction) %>%
  tidyr::pivot_wider(
    names_from = domain_label,
    values_from = community_fraction
  )

sf1B_pairwise_tests <- bind_rows(
  sf1B_wide %>%
    group_by(community_id) %>%
    summarise(
      p_value = wilcox.test(TC, IM, paired = TRUE, exact = FALSE)$p.value,
      .groups = "drop"
    ) %>%
    mutate(contrast = "TC-IM", xmin = 1, xmax = 2, bracket_rank = 1),
  sf1B_wide %>%
    group_by(community_id) %>%
    summarise(
      p_value = wilcox.test(TC, PT, paired = TRUE, exact = FALSE)$p.value,
      .groups = "drop"
    ) %>%
    mutate(contrast = "TC-PT", xmin = 1, xmax = 3, bracket_rank = 3),
  sf1B_wide %>%
    group_by(community_id) %>%
    summarise(
      p_value = wilcox.test(IM, PT, paired = TRUE, exact = FALSE)$p.value,
      .groups = "drop"
    ) %>%
    mutate(contrast = "IM-PT", xmin = 2, xmax = 3, bracket_rank = 2)
) %>%
  mutate(
    bh_q_value = p.adjust(p_value, method = "BH"),
    stat_label = case_when(
      bh_q_value < 0.001 ~ "***",
      bh_q_value < 0.01 ~ "**",
      bh_q_value <= 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    community_label = factor(
      paste0("C", community_id),
      levels = sf1B_selected_labels
    )
  )

sf1B_brackets <- sf1B_pairwise_tests %>%
  left_join(
    sf1B_plot_data %>%
      group_by(community_label) %>%
      summarise(panel_max = max(community_fraction, na.rm = TRUE), .groups = "drop"),
    by = "community_label"
  ) %>%
  mutate(
    y_step = pmax(panel_max * 0.11, 0.012),
    y_position = panel_max + bracket_rank * y_step,
    y_tick = y_step * 0.20,
    label_y = y_position + y_tick * 0.20
  )
#
# Visualization block:
sf1B_plot <- ggplot(
  sf1B_plot_data,
  aes(x = domain_label, y = community_fraction, color = domain_label, fill = domain_label)
) +
  geom_boxplot(
    width = 0.52,
    outlier.shape = NA,
    alpha = 0.48,
    color = "grey25",
    linewidth = 0.25
  ) +
  geom_point(
    position = position_jitter(width = 0.08, height = 0),
    size = 0.45,
    alpha = 0.42
  ) +
  geom_segment(
    data = sf1B_brackets,
    aes(x = xmin, xend = xmax, y = y_position, yend = y_position),
    inherit.aes = FALSE,
    linewidth = 0.2,
    color = "grey20"
  ) +
  geom_segment(
    data = sf1B_brackets,
    aes(x = xmin, xend = xmin, y = y_position, yend = y_position - y_tick),
    inherit.aes = FALSE,
    linewidth = 0.2,
    color = "grey20"
  ) +
  geom_segment(
    data = sf1B_brackets,
    aes(x = xmax, xend = xmax, y = y_position, yend = y_position - y_tick),
    inherit.aes = FALSE,
    linewidth = 0.2,
    color = "grey20"
  ) +
  geom_text(
    data = sf1B_brackets,
    aes(x = (xmin + xmax) / 2, y = label_y, label = stat_label),
    inherit.aes = FALSE,
    size = 1.8,
    color = "grey15",
    vjust = 0
  ) +
  facet_wrap(~ community_label, scales = "free_y", ncol = 4) +
  scale_color_manual(values = domain_colors, drop = FALSE) +
  scale_fill_manual(values = domain_colors, drop = FALSE) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 1),
    expand = expansion(mult = c(0.04, 0.24))
  ) +
  coord_cartesian(clip = "off") +
  labs(
    x = NULL,
    y = "Community fraction within patient-domain",
    color = NULL,
    fill = NULL
  ) +
  base_theme +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 7),
    panel.spacing.x = unit(0.65, "lines"),
    panel.spacing.y = unit(0.85, "lines"),
    plot.margin = margin(5.5, 8, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "sf1_panelB_selected_community_fraction_by_tissue.pdf"),
  plot = sf1B_plot,
  width = 7.8,
  height = 6.4,
  device = cairo_pdf
)


###--Supplementary Figure 1 Panel C: Representative Spatial Examples--###

# Manual placeholder:
#   This panel is assembled manually in Adobe Illustrator from existing
#   representative spatial overlay assets. It is retained here to keep
#   Supplementary Figure 1 lettering stable during figure assembly.
# Input for manual assembly:
#   DESC/tables/representative_overlay_selection.csv
#   DESC/figures/representative_spatial_overlays/*.svg
# Field meaning:
#   community_id, patient_id, domain_label, fov_id, community_fraction_in_roi,
#   community_total_cells, and overlay_path define candidate overlay entries.
# Result intent:
#   Provide representative spatial context for selected communities.
# Visualization:
#   Manual multi-panel spatial overlay montage.
# Axes/value definition:
#   No quantitative axes.
# Output PDF:
#   No R-generated PDF. Assemble this panel manually.
#
# Data read and processing:
# Manual placeholder only.
#
# Visualization block:
# Manual placeholder only.
#
# PDF export:
# Manual placeholder only.

###--Supplementary Figure 1 Panel D: Patient-Level Overall Relation Summaries--###

# Input:
#   B0/block0_patient_calibration.csv
#   B1/block1_confirmatory_family_comparison.csv
#   B0/block0_metric_summary.csv
#   B1/block1_family_statistical_supplement.csv
# Field meaning:
#   patient_id identifies paired patient values.
#   summary_name is mapped through metric_labels.
#   real_value and null_reference support Block 0 patient-level null display.
#   tc_im_value and tc_pt_value support Block 1 patient-level paired display.
# Result intent:
#   Provide patient-level support for Main Figure 2C cohort-level patterns.
# Visualization:
#   Patient-level paired or jitter plot matching Main Figure 2C.
# Axes/value definition:
#   x-axis: Null, TC -> IM, TC -> PT.
#   y-axis: raw metric value.
#   facets: Source retention, Source-open mass, Off-diagonal remodeling,
#   Target-open mass.
#   Connect patient-level TC -> IM and TC -> PT values.
#   Statistics: Null vs TC -> IM uses Block 0 empirical p-values; TC -> IM vs
#   TC -> PT uses Block 1 BH-adjusted Wilcoxon q-values. No Null vs TC -> PT
#   test is displayed.
# Output PDF:
#   sf1_panelD_patient_level_overall_relation_summaries.pdf
#
# Data read and processing:
sf1D_metrics <- c(
  "self_retention",
  "depletion",
  "off_diagonal_remodeling",
  "emergence"
)

sf1D_block0_patient <- readr::read_csv(
  file.path(B0, "block0_patient_calibration.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    summary_name %in% sf1D_metrics
  )

sf1D_block1_patient <- readr::read_csv(
  file.path(B1, "block1_confirmatory_family_comparison.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    summary_name %in% sf1D_metrics,
    comparison_status == "estimable"
  )

sf1D_block0_stats <- readr::read_csv(
  file.path(B0, "block0_metric_summary.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    cohort_stat == "median",
    summary_name %in% sf1D_metrics
  )

sf1D_block1_stats <- readr::read_csv(
  file.path(B1, "block1_family_statistical_supplement.csv"),
  show_col_types = FALSE
) %>%
  filter(
    scale == "burden_weighted",
    summary_name %in% sf1D_metrics
  )

sf1D_plot_data <- bind_rows(
  sf1D_block0_patient %>%
    transmute(
      patient_id,
      summary_name,
      group = "Null",
      value = null_reference
    ),
  sf1D_block1_patient %>%
    transmute(
      patient_id,
      summary_name,
      group = "TC -> IM",
      value = tc_im_value
    ),
  sf1D_block1_patient %>%
    transmute(
      patient_id,
      summary_name,
      group = "TC -> PT",
      value = tc_pt_value
    )
) %>%
  mutate(
    metric_label = factor(
      unname(metric_labels[summary_name]),
      levels = unname(metric_labels[sf1D_metrics])
    ),
    group = factor(group, levels = pair_family_levels),
    patient_id = factor(patient_id)
  )

sf1D_pair_segments <- sf1D_plot_data %>%
  filter(group %in% c("TC -> IM", "TC -> PT")) %>%
  mutate(pair_key = if_else(as.character(group) == "TC -> IM", "tc_im", "tc_pt")) %>%
  select(patient_id, metric_label, pair_key, value) %>%
  pivot_wider(names_from = pair_key, values_from = value) %>%
  filter(!is.na(tc_im), !is.na(tc_pt))

sf1D_stat_annotations <- bind_rows(
  sf1D_block0_stats %>%
    transmute(
      summary_name,
      contrast_label = "emp. p",
      stat_value = empirical_p_value,
      xmin = 1,
      xmax = 2,
      bracket_rank = 1
    ),
  sf1D_block1_stats %>%
    transmute(
      summary_name,
      contrast_label = "BH q",
      stat_value = bh_q_value,
      xmin = 2,
      xmax = 3,
      bracket_rank = 2
    )
) %>%
  mutate(
    metric_label = factor(
      unname(metric_labels[summary_name]),
      levels = unname(metric_labels[sf1D_metrics])
    ),
    stat_star = case_when(
      is.na(stat_value) ~ "NA",
      stat_value < 0.001 ~ "***",
      stat_value < 0.01 ~ "**",
      stat_value <= 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    stat_label = paste0(
      contrast_label,
      " ",
      stat_star,
      " (",
      scales::pvalue(stat_value, accuracy = 0.001),
      ")"
    )
  ) %>%
  left_join(
    sf1D_plot_data %>%
      group_by(metric_label) %>%
      summarise(
        panel_min = min(value, na.rm = TRUE),
        panel_max = max(value, na.rm = TRUE),
        .groups = "drop"
      ),
    by = "metric_label"
  ) %>%
  mutate(
    panel_range = pmax(panel_max - panel_min, abs(panel_max) * 0.10, 0.01),
    y_step = panel_range * 0.11,
    y_position = panel_max + bracket_rank * y_step,
    y_tick = y_step * 0.20,
    label_y = y_position + y_tick * 0.22
  )
#
# Visualization block:
sf1D_plot <- ggplot(
  sf1D_plot_data,
  aes(x = group, y = value, color = group, fill = group)
) +
  geom_segment(
    data = sf1D_pair_segments,
    aes(x = "TC -> IM", xend = "TC -> PT", y = tc_im, yend = tc_pt),
    inherit.aes = FALSE,
    color = "grey76",
    linewidth = 0.2,
    alpha = 0.65
  ) +
  geom_boxplot(
    width = 0.52,
    outlier.shape = NA,
    alpha = 0.52,
    color = "grey25",
    linewidth = 0.25
  ) +
  geom_point(
    position = position_jitter(width = 0.08, height = 0),
    size = 0.65,
    alpha = 0.42
  ) +
  geom_segment(
    data = sf1D_stat_annotations,
    aes(x = xmin, xend = xmax, y = y_position, yend = y_position),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey20"
  ) +
  geom_segment(
    data = sf1D_stat_annotations,
    aes(x = xmin, xend = xmin, y = y_position, yend = y_position - y_tick),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey20"
  ) +
  geom_segment(
    data = sf1D_stat_annotations,
    aes(x = xmax, xend = xmax, y = y_position, yend = y_position - y_tick),
    inherit.aes = FALSE,
    linewidth = 0.24,
    color = "grey20"
  ) +
  geom_text(
    data = sf1D_stat_annotations,
    aes(x = (xmin + xmax) / 2, y = label_y, label = stat_label),
    inherit.aes = FALSE,
    size = 1.85,
    color = "grey15",
    vjust = 0
  ) +
  facet_wrap(~ metric_label, scales = "free_y", nrow = 1) +
  scale_color_manual(values = pair_family_colors, drop = FALSE) +
  scale_fill_manual(values = pair_family_colors, drop = FALSE) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.28))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Raw patient value", color = NULL, fill = NULL) +
  base_theme +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1),
    strip.text = element_text(size = 7),
    panel.spacing.x = unit(0.65, "lines"),
    plot.margin = margin(5.5, 12, 5.5, 5.5)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "sf1_panelD_patient_level_overall_relation_summaries.pdf"),
  plot = sf1D_plot,
  width = 10.2,
  height = 3.1,
  device = cairo_pdf
)


###--Supplementary Figure 1 Panel E: Raw Extended Relation Matrices--###

# Input:
#   B1/block1_cohort_relation_comparison.csv
#   B1/block1_relation_element_statistical_supplement.csv
# Field meaning:
#   component and relation_axis identify A, d, and e elements.
#   source_community_id and target_community_id define matrix coordinates.
#   tc_im_value and tc_pt_value define raw relation values.
#   bh_q_value and q_pass optionally support cell outlines.
# Result intent:
#   Show the full raw relation object for TC -> IM and TC -> PT.
# Visualization:
#   Two extended-matrix heatmaps, one for TC -> IM and one for TC -> PT.
# Axes/value definition:
#   Infer K from observed community identifiers.
#   Display the complete (K + 1) x (K + 1) extended relation matrix:
#   template_A in the community x community block, template_d as the
#   source-open column, template_e as the target-open row, and the bottom-right
#   source-open/target-open intersection as blank.
#   Use one display-only log10(value + 1e-4) scale for all non-blank entries.
#   Display transformation changes colors only; raw relation values are not
#   modified or re-exported.
# Output PDFs:
#   sf1_panelE_raw_extended_relation_matrix_tc_im.pdf
#   sf1_panelE_raw_extended_relation_matrix_tc_pt.pdf
#
# Data read and processing:
sf1E_relation <- readr::read_csv(
  file.path(B1, "block1_cohort_relation_comparison.csv"),
  show_col_types = FALSE
)

sf1E_community_ids <- sort(unique(c(
  sf1E_relation$source_community_id[!is.na(sf1E_relation$source_community_id)],
  sf1E_relation$target_community_id[!is.na(sf1E_relation$target_community_id)]
)))
sf1E_k <- length(sf1E_community_ids)
sf1E_community_labels <- paste0("C", sf1E_community_ids)
sf1E_log_epsilon <- 1e-4
sf1E_log_limits <- c(-4, 0)
sf1E_log_legend_at <- seq(sf1E_log_limits[[1]], sf1E_log_limits[[2]], by = 1)
sf1E_log_legend_labels <- c("1e-4", "1e-3", "1e-2", "1e-1", "1")
sf1E_relation_col_fun <- circlize::colorRamp2(
  c(sf1E_log_limits[[1]], mean(sf1E_log_limits), sf1E_log_limits[[2]]),
  c("#f7fbff", "#6baed6", "#08306b")
)
sf1E_row_split <- factor(
  c(rep("Source community", sf1E_k), "target-open e"),
  levels = c("Source community", "target-open e")
)
sf1E_column_split <- factor(
  c(rep("Target community", sf1E_k), "source-open d"),
  levels = c("Target community", "source-open d")
)

for (sf1E_pair in c("tc_im", "tc_pt")) {
  sf1E_value_col <- paste0(sf1E_pair, "_value")
  sf1E_pair_label <- if_else(sf1E_pair == "tc_im", "TC -> IM", "TC -> PT")
  sf1E_output <- if_else(
    sf1E_pair == "tc_im",
    "sf1_panelE_raw_extended_relation_matrix_tc_im.pdf",
    "sf1_panelE_raw_extended_relation_matrix_tc_pt.pdf"
  )

  sf1E_A_mat <- matrix(
    NA_real_,
    nrow = sf1E_k,
    ncol = sf1E_k,
    dimnames = list(sf1E_community_labels, sf1E_community_labels)
  )
  sf1E_d_mat <- matrix(
    NA_real_,
    nrow = sf1E_k,
    ncol = 1,
    dimnames = list(sf1E_community_labels, "d")
  )
  sf1E_e_mat <- matrix(
    NA_real_,
    nrow = 1,
    ncol = sf1E_k,
    dimnames = list("e", sf1E_community_labels)
  )

  sf1E_A_rows <- sf1E_relation %>%
    filter(component == "template_A", relation_axis == "source_target")
  for (sf1E_i in seq_len(nrow(sf1E_A_rows))) {
    sf1E_row_idx <- match(sf1E_A_rows$source_community_id[[sf1E_i]], sf1E_community_ids)
    sf1E_col_idx <- match(sf1E_A_rows$target_community_id[[sf1E_i]], sf1E_community_ids)
    sf1E_A_mat[sf1E_row_idx, sf1E_col_idx] <- sf1E_A_rows[[sf1E_value_col]][[sf1E_i]]
  }

  sf1E_d_rows <- sf1E_relation %>% filter(component == "template_d")
  for (sf1E_i in seq_len(nrow(sf1E_d_rows))) {
    sf1E_row_idx <- match(sf1E_d_rows$source_community_id[[sf1E_i]], sf1E_community_ids)
    sf1E_d_mat[sf1E_row_idx, 1] <- sf1E_d_rows[[sf1E_value_col]][[sf1E_i]]
  }

  sf1E_e_rows <- sf1E_relation %>% filter(component == "template_e")
  for (sf1E_i in seq_len(nrow(sf1E_e_rows))) {
    sf1E_col_idx <- match(sf1E_e_rows$target_community_id[[sf1E_i]], sf1E_community_ids)
    sf1E_e_mat[1, sf1E_col_idx] <- sf1E_e_rows[[sf1E_value_col]][[sf1E_i]]
  }

  sf1E_extended_raw_mat <- matrix(
    NA_real_,
    nrow = sf1E_k + 1,
    ncol = sf1E_k + 1,
    dimnames = list(
      c(sf1E_community_labels, "target-open e"),
      c(sf1E_community_labels, "source-open d")
    )
  )
  sf1E_extended_raw_mat[seq_len(sf1E_k), seq_len(sf1E_k)] <- sf1E_A_mat
  sf1E_extended_raw_mat[seq_len(sf1E_k), sf1E_k + 1] <- sf1E_d_mat[, 1]
  sf1E_extended_raw_mat[sf1E_k + 1, seq_len(sf1E_k)] <- sf1E_e_mat[1, ]
  sf1E_extended_display_mat <- log10(sf1E_extended_raw_mat + sf1E_log_epsilon)

  sf1E_relation_heatmap <- Heatmap(
    sf1E_extended_display_mat,
    name = "Relation weight",
    col = sf1E_relation_col_fun,
    na_col = "white",
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    row_split = sf1E_row_split,
    column_split = sf1E_column_split,
    show_row_names = TRUE,
    show_column_names = TRUE,
    row_names_gp = gpar(fontsize = 5.0),
    column_names_gp = gpar(fontsize = 5.0),
    column_names_rot = 45,
    column_title = paste0(sf1E_pair_label, " relation object"),
    column_title_gp = gpar(fontsize = 10, fontface = "bold"),
    row_title_gp = gpar(fontsize = 7, fontface = "bold"),
    column_gap = unit(2.0, "mm"),
    row_gap = unit(2.0, "mm"),
    rect_gp = gpar(col = "grey90", lwd = 0.25),
    heatmap_legend_param = list(
      title = "Relation weight\nlog10(x + 1e-4)",
      at = sf1E_log_legend_at,
      labels = sf1E_log_legend_labels,
      title_gp = gpar(fontsize = 8),
      labels_gp = gpar(fontsize = 7)
    )
  )
#
# Visualization block:
  pdf(
    file.path(FIG_DIR, sf1E_output),
    width = 8.6,
    height = 7.0,
    useDingbats = FALSE
  )
  draw(
    sf1E_relation_heatmap,
    heatmap_legend_side = "right"
  )
  dev.off()
}
#
# PDF export:
# Completed inside pair-specific loop above.


###--Supplementary Figure 1 Panel F: Community-Level Relation Summary Dumbbell Plots--###

# Input:
#   B1/block1_source_community_statistical_supplement.csv
#   B1/block1_target_community_statistical_supplement.csv
# Field meaning:
#   source_community_id and target_community_id identify community rows.
#   self_retention, depletion, off_diagonal_remodeling, and
#   matched_incoming_burden define recommended visible metrics.
#   tc_im_median, tc_pt_median, bh_q_value, and q_pass support cohort median
#   dumbbell values and weak support styling.
# Result intent:
#   Provide a readable marginal view of source-side and target-side community
#   relation summaries, comparing TC -> IM against TC -> PT within each metric.
# Visualization:
#   Four standalone metric-wise dumbbell plots.
# Axes/value definition:
#   y-axis: communities, fixed to the same C0-C24 order across all outputs.
#   x-axis: raw cohort median for one metric.
#   points: TC -> IM and TC -> PT.
#   segments: paired contrast between the two transfer families.
#   q_pass: weak styling cue only; no new statistical tests.
# Output PDFs:
#   sf1_panelF_source_retention_dumbbell.pdf
#   sf1_panelF_off_diagonal_remodeling_dumbbell.pdf
#   sf1_panelF_source_open_mass_dumbbell.pdf
#   sf1_panelF_matched_incoming_mass_dumbbell.pdf
#
# Data read and processing:
sf1F_source_stats <- readr::read_csv(
  file.path(B1, "block1_source_community_statistical_supplement.csv"),
  show_col_types = FALSE
)
sf1F_target_stats <- readr::read_csv(
  file.path(B1, "block1_target_community_statistical_supplement.csv"),
  show_col_types = FALSE
)

sf1F_source_metrics <- c("self_retention", "off_diagonal_remodeling", "depletion")
sf1F_target_metrics <- c("matched_incoming_burden")
sf1F_metric_order <- c(sf1F_source_metrics, sf1F_target_metrics)
sf1F_metric_levels <- unname(metric_labels[sf1F_metric_order])
sf1F_metric_output_files <- c(
  self_retention = "sf1_panelF_source_retention_dumbbell.pdf",
  off_diagonal_remodeling = "sf1_panelF_off_diagonal_remodeling_dumbbell.pdf",
  depletion = "sf1_panelF_source_open_mass_dumbbell.pdf",
  matched_incoming_burden = "sf1_panelF_matched_incoming_mass_dumbbell.pdf"
)
sf1F_pair_family_levels <- c("TC -> IM", "TC -> PT")
sf1F_pair_family_colors <- pair_family_colors[sf1F_pair_family_levels]

sf1F_source_plot <- sf1F_source_stats %>%
  filter(summary_name %in% sf1F_source_metrics) %>%
  transmute(
    community_id = source_community_id,
    summary_name,
    metric_label = unname(metric_labels[summary_name]),
    tc_im_median,
    tc_pt_median,
    bh_q_value,
    q_pass,
    side = "Source"
  )

sf1F_target_plot <- sf1F_target_stats %>%
  filter(summary_name %in% sf1F_target_metrics) %>%
  transmute(
    community_id = target_community_id,
    summary_name,
    metric_label = unname(metric_labels[summary_name]),
    tc_im_median,
    tc_pt_median,
    bh_q_value,
    q_pass,
    side = "Target"
  )

sf1F_community_ids <- sort(unique(c(
  sf1F_source_plot$community_id,
  sf1F_target_plot$community_id
)))
sf1F_community_axis_levels <- paste0("C", rev(sf1F_community_ids))

sf1F_plot_long <- bind_rows(sf1F_source_plot, sf1F_target_plot) %>%
  pivot_longer(
    cols = c(tc_im_median, tc_pt_median),
    names_to = "pair_family",
    values_to = "value"
  ) %>%
  mutate(
    pair_family = recode(
      pair_family,
      tc_im_median = "TC -> IM",
      tc_pt_median = "TC -> PT"
    ),
    pair_family = factor(pair_family, levels = sf1F_pair_family_levels),
    metric_label = factor(metric_label, levels = sf1F_metric_levels),
    community_label = factor(paste0("C", community_id), levels = sf1F_community_axis_levels),
    support_status = if_else(q_pass, "q <= 0.05", "not q-supported")
  )

sf1F_segments <- bind_rows(sf1F_source_plot, sf1F_target_plot) %>%
  transmute(
    community_id,
    summary_name,
    metric_label = factor(metric_label, levels = sf1F_metric_levels),
    community_label = factor(paste0("C", community_id), levels = sf1F_community_axis_levels),
    tc_im = tc_im_median,
    tc_pt = tc_pt_median,
    q_pass,
    support_status = if_else(q_pass, "q <= 0.05", "not q-supported")
  )
#
# Visualization block:
for (sf1F_metric_name in sf1F_metric_order) {
  sf1F_metric_label <- unname(metric_labels[[sf1F_metric_name]])
  sf1F_metric_points <- sf1F_plot_long %>%
    filter(summary_name == sf1F_metric_name, !is.na(value))
  sf1F_metric_segments <- sf1F_segments %>%
    filter(summary_name == sf1F_metric_name, !is.na(tc_im), !is.na(tc_pt))

  sf1F_metric_plot <- ggplot() +
    geom_segment(
      data = sf1F_metric_segments,
      aes(
        x = tc_im,
        xend = tc_pt,
        y = community_label,
        yend = community_label,
        alpha = support_status
      ),
      color = "grey45",
      linewidth = 0.38,
      lineend = "round"
    ) +
    geom_point(
      data = sf1F_metric_points,
      aes(x = value, y = community_label, fill = pair_family, color = pair_family),
      shape = 21,
      size = 2.1,
      stroke = 0.25
    ) +
    scale_alpha_manual(
      values = c("q <= 0.05" = 0.9, "not q-supported" = 0.28),
      breaks = c("q <= 0.05", "not q-supported"),
      name = NULL,
      drop = FALSE
    ) +
    scale_fill_manual(values = sf1F_pair_family_colors, drop = FALSE) +
    scale_color_manual(values = sf1F_pair_family_colors, drop = FALSE) +
    scale_y_discrete(drop = FALSE) +
    scale_x_continuous(expand = expansion(mult = c(0.04, 0.10))) +
    labs(
      title = sf1F_metric_label,
      x = "Raw cohort median",
      y = "Community",
      fill = NULL,
      color = NULL
    ) +
    base_theme +
    theme(
      legend.position = "top",
      axis.text.y = element_text(size = 5.8),
      panel.grid.major.y = element_line(color = "grey92", linewidth = 0.18),
      panel.grid.major.x = element_line(color = "grey94", linewidth = 0.16),
      plot.title = element_text(face = "bold", size = 9, hjust = 0),
      plot.margin = margin(5.5, 8, 5.5, 5.5)
    ) +
    guides(
      alpha = guide_legend(order = 1, override.aes = list(color = "grey45", linewidth = 0.6)),
      fill = guide_legend(order = 2, override.aes = list(size = 2.4)),
      color = "none"
    )

  ggsave(
    filename = file.path(FIG_DIR, sf1F_metric_output_files[[sf1F_metric_name]]),
    plot = sf1F_metric_plot,
    width = 4.8,
    height = 6.2,
    device = cairo_pdf
  )
}


###--Supplementary Figure 1 Panel G: PT-Rich Target Incoming Burden Examples--###

# Input:
#   B1/block1_target_community_comparison.csv
#   B1/block1_target_community_statistical_supplement.csv
#   DESC/tables/community_domain_distribution.csv
# Field meaning:
#   patient_id identifies paired patient values.
#   target_community_id identifies PT-rich target examples inferred from
#   community composition.
#   summary_name selects matched incoming mass.
#   tc_im_value and tc_pt_value define paired raw values.
#   q fields from the target statistical supplement support displayed labels.
# Result intent:
#   Provide the target-side patient-level complement to Main Figure 2D's
#   tumor-dominant source-side community examples.
# Visualization:
#   Boxplot plus paired patient points for all PT-rich target examples.
# Axes/value definition:
#   x-axis: TC -> IM and TC -> PT.
#   y-axis: raw matched incoming mass.
#   facets: PT-rich target communities.
#   connect paired patient points.
#   target examples are inferred by domain_label == "PT" and
#   fraction_within_community > 0.5.
# Output PDF:
#   sf1_panelG_pt_rich_target_patient_examples.pdf
#
# Data read and processing:
sf1G_target_comparison <- readr::read_csv(
  file.path(B1, "block1_target_community_comparison.csv"),
  show_col_types = FALSE
)
sf1G_target_stats <- readr::read_csv(
  file.path(B1, "block1_target_community_statistical_supplement.csv"),
  show_col_types = FALSE
)
sf1G_domain_distribution <- readr::read_csv(
  file.path(DESC, "tables", "community_domain_distribution.csv"),
  show_col_types = FALSE
)

sf1G_target_communities <- sf1G_domain_distribution %>%
  filter(
    domain_label == "PT",
    fraction_within_community > 0.5
  ) %>%
  arrange(community_id) %>%
  pull(community_id)

sf1G_target_metric <- "matched_incoming_burden"
sf1G_pair_family_levels <- c("TC -> IM", "TC -> PT")
sf1G_pair_family_colors <- pair_family_colors[sf1G_pair_family_levels]

sf1G_stats <- sf1G_target_stats %>%
  filter(
    target_community_id %in% sf1G_target_communities,
    summary_name == sf1G_target_metric
  ) %>%
  mutate(
    community_facet = factor(
      paste0("C", target_community_id),
      levels = paste0("C", sf1G_target_communities)
    ),
    significance_label = case_when(
      is.na(bh_q_value) ~ "ns",
      bh_q_value < 0.001 ~ "***",
      bh_q_value < 0.01 ~ "**",
      bh_q_value <= 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

sf1G_patient_examples <- sf1G_target_comparison %>%
  filter(
    target_community_id %in% sf1G_target_communities,
    summary_name == sf1G_target_metric,
    comparison_status == "estimable"
  ) %>%
  select(patient_id, target_community_id, summary_name, tc_im_value, tc_pt_value) %>%
  filter(!is.na(tc_im_value), !is.na(tc_pt_value)) %>%
  pivot_longer(
    cols = c(tc_im_value, tc_pt_value),
    names_to = "pair_family",
    values_to = "value"
  ) %>%
  mutate(
    pair_family = recode(
      pair_family,
      tc_im_value = "TC -> IM",
      tc_pt_value = "TC -> PT"
    ),
    pair_family = factor(pair_family, levels = sf1G_pair_family_levels),
    community_facet = factor(
      paste0("C", target_community_id),
      levels = paste0("C", sf1G_target_communities)
    ),
    patient_id = factor(patient_id)
  )

sf1G_pair_segments <- sf1G_patient_examples %>%
  mutate(pair_key = if_else(as.character(pair_family) == "TC -> IM", "tc_im", "tc_pt")) %>%
  select(community_facet, patient_id, pair_key, value) %>%
  pivot_wider(names_from = pair_key, values_from = value) %>%
  filter(!is.na(tc_im), !is.na(tc_pt))

sf1G_annotations <- sf1G_patient_examples %>%
  group_by(community_facet) %>%
  summarise(
    y_min = min(value, na.rm = TRUE),
    y_max = max(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(sf1G_stats %>% select(community_facet, significance_label, bh_q_value), by = "community_facet") %>%
  mutate(
    y_range = pmax(y_max - y_min, 0.01),
    y_bracket = y_max + y_range * 0.08,
    y_label = y_max + y_range * 0.16,
    label = paste0(significance_label, "  BH q=", scales::pvalue(bh_q_value, accuracy = 0.001))
  )
#
# Visualization block:
sf1G_plot <- ggplot(
  sf1G_patient_examples,
  aes(x = pair_family, y = value, color = pair_family, fill = pair_family)
) +
  geom_segment(
    data = sf1G_pair_segments,
    aes(x = "TC -> IM", xend = "TC -> PT", y = tc_im, yend = tc_pt),
    inherit.aes = FALSE,
    color = "grey76",
    linewidth = 0.22,
    alpha = 0.65
  ) +
  geom_boxplot(
    width = 0.48,
    outlier.shape = NA,
    alpha = 0.55,
    color = "grey25",
    linewidth = 0.25
  ) +
  geom_point(
    position = position_jitter(width = 0.055, height = 0),
    size = 0.72,
    alpha = 0.45
  ) +
  geom_segment(
    data = sf1G_annotations,
    aes(x = 1, xend = 2, y = y_bracket, yend = y_bracket),
    inherit.aes = FALSE,
    linewidth = 0.25,
    color = "grey20"
  ) +
  geom_text(
    data = sf1G_annotations,
    aes(x = 1.5, y = y_label, label = label),
    inherit.aes = FALSE,
    size = 2.2,
    color = "grey10"
  ) +
  facet_wrap(~ community_facet, scales = "free_y", ncol = 4) +
  scale_color_manual(values = sf1G_pair_family_colors, drop = FALSE) +
  scale_fill_manual(values = sf1G_pair_family_colors, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.22))) +
  labs(
    x = NULL,
    y = "Matched incoming mass",
    color = NULL,
    fill = NULL
  ) +
  base_theme +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 7),
    panel.spacing = unit(0.7, "lines"),
    axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1)
  )
#
# PDF export:
ggsave(
  filename = file.path(FIG_DIR, "sf1_panelG_pt_rich_target_patient_examples.pdf"),
  plot = sf1G_plot,
  width = 7.8,
  height = 4.8,
  device = cairo_pdf
)


###--Supplementary Figure 2 Panel A: Generator Rerun Consistency--###

# Input:
#   B3/generator_validation/raw/generator_validation/rerun_stability.csv
# Field meaning:
#   validation_object_id identifies target representation.
#   metric_name == "rerun variability" selects the diagnostic metric.
#   reported_value defines the plotted value.
# Result intent:
#   Show rerun variability for generator target summaries across repeated
#   train-test splits.
# Visualization:
#   Compact point or bar plot.
# Axes/value definition:
#   x-axis: validation object label.
#   y-axis: rerun variability.
#   validation objects: community_space_target_fraction and
#   identity_projected_target_fraction.
# Output PDF:
#   sf2_panelA_generator_rerun_consistency.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO


###--Supplementary Figure 2 Panel B: Synthetic-Real Target Agreement--###

# Input:
#   B3/generator_validation/raw/generator_validation/object_scores.csv
# Field meaning:
#   rerun_id identifies repeated split.
#   validation_object_id identifies target representation.
#   metric_name selects Pearson correlation, MAE, or JS divergence.
#   reported_value defines the plotted value.
# Result intent:
#   Show rerun-level numerical agreement between synthetic targets and held-out
#   real target profiles.
# Visualization:
#   Rerun-level dot plot with median or point-range summary.
# Axes/value definition:
#   x-axis: metric.
#   y-axis: reported value.
#   color or facet: validation object label.
#   omit MSE from the visual layer.
# Output PDF:
#   sf2_panelB_synthetic_real_target_agreement.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO


###--Supplementary Figure 2 Panel C: Relation Benchmark Supplemental Error--###

# Input:
#   B3/a_benchmark/raw/a_benchmark/condition_summary.csv
#   B3/a_benchmark/raw/a_benchmark/patient_metrics.csv
# Field meaning:
#   method_name is mapped through method_labels.
#   metric_name == "A_MSE_active" selects active relation MSE.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show supplemental active relation MSE in the relation benchmark.
# Visualization:
#   Lower-is-better point-range or bar plot.
# Axes/value definition:
#   x-axis: method.
#   y-axis: Active relation MSE (x10^6).
#   methods: STRIDE, Balanced OT, Unbalanced OT, Partial OT,
#   Diagonal transport.
# Output PDF:
#   sf2_panelC_relation_benchmark_supplemental_error.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO


###--Supplementary Figure 2 Panel D: Open Benchmark Supplemental Errors--###

# Input:
#   B3/de_benchmark/raw/de_benchmark/condition_summary.csv
#   B3/de_benchmark/raw/de_benchmark/patient_metrics.csv
# Field meaning:
#   method_name is mapped through method_labels.
#   metric_name selects source-open mass error, target-open mass error,
#   source-open profile MSE, and target-open profile MSE.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show supplemental open-channel mass and profile errors in the open
#   benchmark.
# Visualization:
#   Faceted lower-is-better point-range or bar plot.
# Axes/value definition:
#   x-axis: method.
#   y-axis: error value with display scaling in facet labels.
#   facets: depletion_mass_abs_error, emergence_mass_abs_error, d_MSE, e_MSE.
#   Only methods present in de_benchmark are displayed; do not add a missing
#   balanced_ot_baseline NA slot in this supplementary panel.
# Output PDF:
#   sf2_panelD_open_benchmark_supplemental_errors.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO


###--Supplementary Figure 2 Panel E: Ablation Supplemental Relation and Mass Errors--###

# Input:
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv
#   B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv
#   B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv
# Field meaning:
#   evaluation_family identifies the ablation family.
#   method_name identifies stride_reference or the ablation arm.
#   metric_name selects A_MSE_active, depletion_mass_abs_error, and
#   emergence_mass_abs_error.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show supplemental relation and open-mass errors for objective-component
#   ablations.
# Visualization:
#   Faceted lower-is-better point-range or bar plot.
# Axes/value definition:
#   x-axis: method or ablation arm.
#   y-axis: error value with display scaling in facet labels.
#   facets: Active relation MSE, Source-open mass error, Target-open mass error.
# Output PDF:
#   sf2_panelE_ablation_supplemental_relation_and_mass_errors.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO


###--Supplementary Figure 2 Panel F: Ablation Supplemental Open-Profile MSE--###

# Input:
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/condition_summary.csv
#   B3/geometry_ablation/raw/geometry_ablation/condition_summary.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/condition_summary.csv
#   B3/subbag_consistency_ablation/raw/subbag_consistency_ablation/patient_metrics.csv
#   B3/geometry_ablation/raw/geometry_ablation/patient_metrics.csv
#   B3/recurrence_ablation/raw/recurrence_ablation/patient_metrics.csv
# Field meaning:
#   evaluation_family identifies the ablation family.
#   method_name identifies stride_reference or the ablation arm.
#   metric_name selects d_MSE and e_MSE.
#   mean_value, ci_lower, and ci_upper define summary point-ranges.
#   reported_value supports rerun-level statistical comparisons.
# Result intent:
#   Show supplemental open-profile MSEs for objective-component ablations.
# Visualization:
#   Faceted lower-is-better point-range or bar plot.
# Axes/value definition:
#   x-axis: method or ablation arm.
#   y-axis: error value.
#   facets: Source-open profile MSE (x10^3), Target-open profile MSE (x10^3).
# Output PDF:
#   sf2_panelF_ablation_supplemental_open_profile_mse.pdf
#
# Data read and processing:
# TODO
#
# Visualization block:
# TODO
#
# PDF export:
# TODO
