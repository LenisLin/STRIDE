"""STRIDE plotting namespace.

Plotting functions return matplotlib Figure objects when `save` is not
supplied. When `save` is supplied, they export a PDF figure, close it, and
return None. PDF export is local file output only. They do not fit models,
compute statistical tests, write h5ad/result payloads, or mutate scientific
inputs.
"""
from __future__ import annotations

from ._association import augmented_relation_association_bubble_plot
from ._cohort import cohort_relation_heatmap
from ._descriptive import (
    community_annotation_heatmap,
    community_fraction_comparison,
    fov_composition_heatmap,
)
from ._programs import (
    relation_program_rank_elbow_plot,
    relation_program_score_boxplot,
    relation_program_structure_heatmap,
)

__all__ = (
    "community_annotation_heatmap",
    "fov_composition_heatmap",
    "community_fraction_comparison",
    "cohort_relation_heatmap",
    "augmented_relation_association_bubble_plot",
    "relation_program_rank_elbow_plot",
    "relation_program_score_boxplot",
    "relation_program_structure_heatmap",
)
