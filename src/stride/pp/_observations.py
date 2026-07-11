"""FOV-level community fraction observation construction for STRIDE .pp."""
from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from anndata import AnnData

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_STATE_ID_KEY,
    OBS_TIMEPOINT_KEY,
    STRIDE_CONFIG_KEY,
    STRIDE_FOV_OBSERVATIONS_KEY,
    STRIDE_UNS_KEY,
)
from stride.errors import ContractError

# Numerical tolerance for fraction-simplex and cache-equivalence checks.
_OBSERVATION_ATOL = 1e-12
_FOV_GROUP_KEYS = [OBS_PATIENT_KEY, OBS_TIMEPOINT_KEY, OBS_FOV_KEY]
_FOV_METADATA_COLUMNS = [*_FOV_GROUP_KEYS, OBS_DOMAIN_KEY]


def build_fov_observations(adata: AnnData) -> AnnData:
    """Build or validate FOV-level shared-state fraction observations.

    Reads `community_mode = "fraction"` and `n_states` from
    `adata.uns["stride"]["config"]`. Aggregates `adata.obs["state_id"]`
    by `(patient_id, timepoint, fov_id)` into `community_composition`.

    Writes `adata.uns["stride"]["fov_observations"]` with:
    - `community_composition`: `[n_fov, n_states]`
    - `metadata`: `patient_id`, `timepoint`, `fov_id`, `domain_label`

    Existing valid FOV observation cache is reused with a warning.
    """
    _community_mode, n_states = _read_observation_config(adata)
    state_id = _state_ids(adata, n_states)
    expected = _build_expected_fov_observations(adata, state_id, n_states)

    stride_uns = adata.uns[STRIDE_UNS_KEY]
    if STRIDE_FOV_OBSERVATIONS_KEY in stride_uns:
        _validate_existing_fov_observations(
            stride_uns[STRIDE_FOV_OBSERVATIONS_KEY],
            expected,
            n_states,
        )
        warnings.warn(
            "adata.uns['stride']['fov_observations'] already exists; "
            "reusing existing FOV observations",
            UserWarning,
            stacklevel=2,
        )
        return adata

    stride_uns[STRIDE_FOV_OBSERVATIONS_KEY] = expected
    return adata


def _read_observation_config(adata: AnnData) -> tuple[str, int]:
    """Read FOV-observation config values used by this step."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    community_mode = config.get("community_mode")
    if community_mode != "fraction":
        raise ContractError("config['community_mode'] must be 'fraction'")

    value = config.get("n_states")
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ContractError("config['n_states'] must be a positive integer")
    if int(value) <= 0:
        raise ContractError("config['n_states'] must be a positive integer")
    return str(community_mode), int(value)


def _state_ids(adata: AnnData, n_states: int) -> np.ndarray:
    """Return integer shared-state assignments for all cells."""
    if OBS_STATE_ID_KEY not in adata.obs:
        raise ContractError("adata.obs['state_id'] is required")
    state_id = np.asarray(adata.obs[OBS_STATE_ID_KEY])
    if state_id.shape[0] != adata.n_obs:
        raise ContractError("adata.obs['state_id'] length must align to adata.n_obs")
    try:
        state_id_float = np.asarray(state_id, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.obs['state_id'] must be integer-compatible") from exc
    if not np.isfinite(state_id_float).all():
        raise ContractError("adata.obs['state_id'] contains NaN/Inf")
    if not np.equal(state_id_float, np.floor(state_id_float)).all():
        raise ContractError("adata.obs['state_id'] must be integer-compatible")
    state_id_int = state_id_float.astype(int, copy=False)
    if ((state_id_int < 0) | (state_id_int >= n_states)).any():
        raise ContractError("adata.obs['state_id'] values must be in [0, n_states - 1]")
    return state_id_int


def _build_expected_fov_observations(
    adata: AnnData,
    state_id: np.ndarray,
    n_states: int,
) -> dict[str, object]:
    """Aggregate cells into the canonical equal-mass FOV observation points.

    Each output row is a within-FOV state fraction on the shared K-state axis.
    Cell count affects only that row's composition; it does not become an FOV
    mass, so downstream bags continue to treat observed FOVs equally.
    """
    missing = [column for column in _FOV_METADATA_COLUMNS if column not in adata.obs]
    if missing:
        raise ContractError(
            "adata.obs is missing FOV observation columns: " + ", ".join(missing)
        )

    obs = _validate_metadata_frame(adata.obs.loc[:, _FOV_METADATA_COLUMNS].copy())
    obs[OBS_STATE_ID_KEY] = state_id
    grouped = obs.groupby(_FOV_GROUP_KEYS, sort=True, observed=True, dropna=False)
    domain_counts = grouped[OBS_DOMAIN_KEY].nunique(dropna=False)
    if (domain_counts != 1).any():
        raise ContractError(
            "each patient_id/timepoint/fov_id group must map to exactly one domain_label"
        )

    metadata = grouped[_FOV_METADATA_COLUMNS].first().reset_index(drop=True)
    metadata = _validate_metadata_frame(metadata)
    group_codes = grouped.ngroup().to_numpy(dtype=int, copy=False)
    matrix = np.zeros((metadata.shape[0], n_states), dtype=float)
    # Vectorized cell counting preserves the lexicographically sorted group
    # order produced by `groupby(sort=True)` and the original state-id axis.
    np.add.at(matrix, (group_codes, state_id), 1.0)
    totals = matrix.sum(axis=1)
    if (totals <= 0.0).any():
        raise ContractError("FOV observation groups must contain at least one cell")
    matrix /= totals[:, None]
    matrix = _validate_community_composition(
        matrix,
        n_fov=metadata.shape[0],
        n_states=n_states,
    )
    return {
        "community_composition": matrix,
        "metadata": metadata,
    }


def _validate_metadata_frame(metadata: object) -> pd.DataFrame:
    """Return normalized FOV metadata used by fresh builds and cache checks."""
    if not isinstance(metadata, pd.DataFrame):
        raise ContractError("fov_observations['metadata'] must be a pandas DataFrame")
    if list(metadata.columns) != _FOV_METADATA_COLUMNS:
        raise ContractError(
            "fov_observations['metadata'] columns must be "
            "patient_id, timepoint, fov_id, domain_label"
        )
    if metadata.isna().any(axis=None):
        raise ContractError("fov_observations['metadata'] contains missing values")

    normalized = metadata.reset_index(drop=True).copy()
    for column in _FOV_METADATA_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip()
        if (normalized[column] == "").any():
            raise ContractError(
                f"fov_observations['metadata'] column {column!r} "
                "contains empty string values"
            )
    return normalized


def _validate_community_composition(
    value: object,
    *,
    n_fov: int,
    n_states: int,
) -> np.ndarray:
    """Return a valid FOV-by-state fraction matrix."""
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "fov_observations['community_composition'] contains NaN/Inf"
        ) from exc
    if matrix.ndim != 2:
        raise ContractError(
            "fov_observations['community_composition'] must be a 2D matrix"
        )
    if matrix.shape != (n_fov, n_states):
        raise ContractError(
            "fov_observations['community_composition'] shape must be "
            "[n_fov, n_states]"
        )
    if not np.isfinite(matrix).all():
        raise ContractError(
            "fov_observations['community_composition'] contains NaN/Inf"
        )
    if (matrix < 0.0).any():
        raise ContractError(
            "fov_observations['community_composition'] must be nonnegative"
        )
    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=_OBSERVATION_ATOL,
    ):
        raise ContractError(
            "fov_observations['community_composition'] rows must sum to 1"
        )
    return matrix


def _validate_existing_fov_observations(
    existing: object,
    expected: Mapping[str, object],
    n_states: int,
) -> None:
    """Validate that an existing cache matches current state aggregation."""
    if not isinstance(existing, Mapping):
        raise ContractError("adata.uns['stride']['fov_observations'] must be a mapping")
    if set(existing.keys()) != {"community_composition", "metadata"}:
        raise ContractError(
            "adata.uns['stride']['fov_observations'] must contain "
            "community_composition and metadata"
        )

    expected_metadata = _validate_metadata_frame(expected["metadata"])
    metadata = _validate_metadata_frame(existing["metadata"])
    matrix = _validate_community_composition(
        existing["community_composition"],
        n_fov=expected_metadata.shape[0],
        n_states=n_states,
    )
    expected_matrix = _validate_community_composition(
        expected["community_composition"],
        n_fov=expected_metadata.shape[0],
        n_states=n_states,
    )

    if not metadata.equals(expected_metadata):
        raise ContractError(
            "adata.uns['stride']['fov_observations'] does not match current "
            "state aggregation"
        )
    if not np.allclose(
        matrix,
        expected_matrix,
        rtol=0.0,
        atol=_OBSERVATION_ATOL,
    ):
        raise ContractError(
            "adata.uns['stride']['fov_observations'] does not match current "
            "state aggregation"
        )
