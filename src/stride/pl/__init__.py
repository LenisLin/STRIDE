"""STRIDE plotting namespace.

Plotting functions return matplotlib Figure objects when `save` is not
supplied. When `save` is supplied, they export a PDF figure, close it, and
return None. PDF export is local file output only. They do not fit models,
compute statistical tests, write h5ad/result payloads, or mutate scientific
inputs.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_MODULES = {
    "community_annotation_heatmap": "._descriptive",
    "fov_composition_heatmap": "._descriptive",
    "community_fraction_comparison": "._descriptive",
    "cohort_relation_heatmap": "._cohort",
    "augmented_relation_association_bubble_plot": "._association",
    "relation_program_rank_elbow_plot": "._programs",
    "relation_program_score_boxplot": "._programs",
    "relation_program_structure_heatmap": "._programs",
}


def __getattr__(name: str) -> Any:
    """Load one plotting implementation on first public attribute access."""
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public functions to interactive completion tools."""
    return sorted(set(globals()) | set(__all__))
