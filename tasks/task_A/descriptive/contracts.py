"""Input contracts for the Task A descriptive atlas."""
from __future__ import annotations

from typing import Any

ATLAS_MANIFEST_FILENAME = "task_a_descriptive_atlas_manifest.json"
ATLAS_OUTPUT_INDEX_FILENAME = "task_a_descriptive_atlas_output_index.csv"
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"
OVERLAY_DIRNAME = "representative_spatial_overlays"
DEFAULT_MAX_OVERLAY_COMMUNITIES = 8
ATLAS_ROLE = "descriptive_only"
STAGE0_FIELD_MAPPING: dict[str, str] = {
    "patient_id_key": "patient_id",
    "fov_key": "roi_id",
    "domain_key": "compartment",
    "cell_subtype_key": "cell_type",
    "state_id_key": "proto_id",
}


class DescriptiveAtlasContractError(ValueError):
    """Raised when the Stage0 surface cannot support the descriptive atlas."""


def require_fields(payload: dict[str, Any], *, required_fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise DescriptiveAtlasContractError(f"{label} is missing required fields: {missing}")


def copy_stage0_field_mapping() -> dict[str, str]:
    return dict(STAGE0_FIELD_MAPPING)


__all__ = [
    "ATLAS_MANIFEST_FILENAME",
    "ATLAS_OUTPUT_INDEX_FILENAME",
    "ATLAS_ROLE",
    "DEFAULT_MAX_OVERLAY_COMMUNITIES",
    "DescriptiveAtlasContractError",
    "FIGURES_DIRNAME",
    "OVERLAY_DIRNAME",
    "STAGE0_FIELD_MAPPING",
    "TABLES_DIRNAME",
    "copy_stage0_field_mapping",
    "require_fields",
]
