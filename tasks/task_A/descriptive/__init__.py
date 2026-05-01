"""Task A descriptive atlas public surface."""
from __future__ import annotations

from .atlas import write_task_a_descriptive_atlas
from .contracts import (
    ATLAS_MANIFEST_FILENAME,
    ATLAS_OUTPUT_INDEX_FILENAME,
    DEFAULT_MAX_OVERLAY_COMMUNITIES,
    DescriptiveAtlasContractError,
)

__all__ = [
    "ATLAS_MANIFEST_FILENAME",
    "ATLAS_OUTPUT_INDEX_FILENAME",
    "DEFAULT_MAX_OVERLAY_COMMUNITIES",
    "DescriptiveAtlasContractError",
    "write_task_a_descriptive_atlas",
]
