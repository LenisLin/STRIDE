"""Internal canonical field names for STRIDE AnnData assembly.

This module is shared implementation schema, not a public API namespace.
"""
from __future__ import annotations

STRIDE_UNS_KEY = "stride"
STRIDE_CONFIG_KEY = "config"
STRIDE_FOV_METADATA_KEY = "fov_metadata"

OBS_PATIENT_KEY = "patient_id"
OBS_TIMEPOINT_KEY = "timepoint"
OBS_FOV_KEY = "fov_id"
OBS_DOMAIN_KEY = "domain_label"
OBS_CELL_TYPE_KEY = "cell_subtype_label"

OBSM_SPATIAL_KEY = "spatial"

ALLOWED_INPUT_MASS_MODES = ("fraction", "density")
