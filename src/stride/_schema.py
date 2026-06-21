"""Internal canonical field names for STRIDE AnnData assembly.

This module is shared implementation schema, not a public API namespace.
"""
from __future__ import annotations

STRIDE_UNS_KEY = "stride"
STRIDE_CONFIG_KEY = "config"
STRIDE_FOV_METADATA_KEY = "fov_metadata"
STRIDE_FOV_OBSERVATIONS_KEY = "fov_observations"
STRIDE_RELATIONS_KEY = "relations"
STRIDE_RELATION_IDS_KEY = "relation_ids"

OBS_PATIENT_KEY = "patient_id"
OBS_TIMEPOINT_KEY = "timepoint"
OBS_FOV_KEY = "fov_id"
OBS_DOMAIN_KEY = "domain_label"
OBS_CELL_TYPE_KEY = "cell_subtype_label"
OBS_STATE_ID_KEY = "state_id"

OBSM_SPATIAL_KEY = "spatial"
OBSM_LOCAL_STATE_FEATURES_KEY = "local_state_features"

UNS_STATE_FEATURE_METADATA_KEY = "state_feature_metadata"
UNS_STATE_CENTROIDS_KEY = "state_centroids"
UNS_COST_MATRIX_KEY = "cost_matrix"
UNS_COST_SCALE_KEY = "cost_scale"

# Density community observations require a later observation-schema extension.
ALLOWED_COMMUNITY_MODES = ("fraction",)
