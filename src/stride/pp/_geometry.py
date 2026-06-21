"""Shared-state cost geometry construction for STRIDE .pp."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist

from stride._schema import (
    STRIDE_CONFIG_KEY,
    STRIDE_UNS_KEY,
    UNS_COST_MATRIX_KEY,
    UNS_COST_SCALE_KEY,
    UNS_STATE_CENTROIDS_KEY,
)
from stride.errors import ContractError

# Numerical tolerance for symmetry, zero diagonal, and geometry-equivalent pairs.
_GEOMETRY_ATOL = 1e-12


def build_state_geometry(adata: AnnData, *, metric: str = "euclidean") -> AnnData:
    """Build or validate shared-state cost geometry for downstream fitting.

    Reads `n_states` from `adata.uns["stride"]["config"]` and requires
    `adata.uns["state_centroids"]` to align to that state axis when a new
    geometry matrix is built.

    When geometry slots are absent, builds `cost_matrix` from
    `state_centroids` and writes canonical `cost_scale` as the median
    positive upper-triangle cost. The default metric is Euclidean; other
    built-in `scipy.spatial.distance.cdist` metrics may be supplied by name.

    Existing valid precomputed geometry is reused with a warning.
    A lone `cost_matrix` is completed with canonical `cost_scale`.
    """
    n_states = _read_n_states(adata)
    has_matrix = UNS_COST_MATRIX_KEY in adata.uns
    has_scale = UNS_COST_SCALE_KEY in adata.uns

    if has_matrix and has_scale:
        cost_matrix = _validate_cost_matrix(adata.uns[UNS_COST_MATRIX_KEY], n_states)
        expected_scale = _canonical_cost_scale(cost_matrix)
        _validate_existing_cost_scale(adata.uns[UNS_COST_SCALE_KEY], expected_scale)
        _warn_zero_offdiag_pairs(cost_matrix)
        warnings.warn(
            "adata.uns['cost_matrix'] and adata.uns['cost_scale'] already exist; "
            "using existing precomputed geometry",
            UserWarning,
            stacklevel=2,
        )
        return adata

    if has_matrix:
        cost_matrix = _validate_cost_matrix(adata.uns[UNS_COST_MATRIX_KEY], n_states)
        cost_scale = _canonical_cost_scale(cost_matrix)
        _warn_zero_offdiag_pairs(cost_matrix)
        adata.uns[UNS_COST_SCALE_KEY] = cost_scale
        warnings.warn(
            "adata.uns['cost_matrix'] exists without adata.uns['cost_scale']; "
            "computed cost_scale from existing precomputed geometry",
            UserWarning,
            stacklevel=2,
        )
        return adata

    if has_scale:
        raise ContractError(
            "shared-state geometry is partial; adata.uns['cost_matrix'] is required "
            "when adata.uns['cost_scale'] exists"
        )

    centroids = _state_centroids(adata, n_states)
    cost_matrix = _validate_cost_matrix(_build_cost_matrix(centroids, metric=metric), n_states)
    cost_scale = _canonical_cost_scale(cost_matrix)
    _warn_zero_offdiag_pairs(cost_matrix)
    adata.uns[UNS_COST_MATRIX_KEY] = cost_matrix
    adata.uns[UNS_COST_SCALE_KEY] = cost_scale
    return adata


def _read_n_states(adata: AnnData) -> int:
    """Read the declared shared-state count from STRIDE config."""
    stride_uns = adata.uns.get(STRIDE_UNS_KEY)
    if not isinstance(stride_uns, Mapping):
        raise ContractError("adata.uns['stride'] must be a mapping")
    config = stride_uns.get(STRIDE_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise ContractError("adata.uns['stride']['config'] must be a mapping")

    value = config.get("n_states")
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ContractError("config['n_states'] must be a positive integer")
    if int(value) <= 0:
        raise ContractError("config['n_states'] must be a positive integer")
    return int(value)


def _state_centroids(adata: AnnData, n_states: int) -> np.ndarray:
    """Return shared-state centroids required by geometry construction."""
    if UNS_STATE_CENTROIDS_KEY not in adata.uns:
        raise ContractError("adata.uns['state_centroids'] is required")
    try:
        centroids = np.asarray(adata.uns[UNS_STATE_CENTROIDS_KEY], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.uns['state_centroids'] contains NaN/Inf") from exc
    if centroids.ndim != 2:
        raise ContractError("adata.uns['state_centroids'] must be a 2D matrix")
    if not np.isfinite(centroids).all():
        raise ContractError("adata.uns['state_centroids'] contains NaN/Inf")
    if centroids.shape[0] != n_states:
        raise ContractError("adata.uns['state_centroids'] row count must match config['n_states']")
    return centroids


def _build_cost_matrix(centroids: np.ndarray, *, metric: str) -> np.ndarray:
    """Build pairwise distances from shared-state centroids."""
    if not isinstance(metric, str) or metric.strip() == "":
        raise ContractError("metric must be a non-empty scipy cdist metric name")
    try:
        return cdist(centroids, centroids, metric=metric).astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"metric {metric!r} is not a supported scipy cdist metric") from exc


def _validate_cost_matrix(value: object, n_states: int) -> np.ndarray:
    """Validate and return a cost matrix aligned to the shared state axis."""
    try:
        cost_matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.uns['cost_matrix'] contains NaN/Inf") from exc
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ContractError("adata.uns['cost_matrix'] must be a 2D square matrix")
    if cost_matrix.shape != (n_states, n_states):
        raise ContractError("adata.uns['cost_matrix'] shape must be [n_states, n_states]")
    if not np.isfinite(cost_matrix).all():
        raise ContractError("adata.uns['cost_matrix'] contains NaN/Inf")
    if (cost_matrix < 0.0).any():
        raise ContractError("adata.uns['cost_matrix'] must be nonnegative")
    if not np.allclose(cost_matrix, cost_matrix.T, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise ContractError("adata.uns['cost_matrix'] must be symmetric")
    if not np.allclose(np.diag(cost_matrix), 0.0, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise ContractError("adata.uns['cost_matrix'] diagonal must be zero")
    return cost_matrix


def _upper_offdiag_mask(n_states: int) -> np.ndarray:
    """Return the upper-triangle state-pair mask."""
    return np.triu(np.ones((n_states, n_states), dtype=bool), k=1)


def _canonical_cost_scale(cost_matrix: np.ndarray) -> float:
    """Return the median positive upper-triangle off-diagonal cost."""
    upper_values = cost_matrix[_upper_offdiag_mask(cost_matrix.shape[0])]
    positive_costs = upper_values[upper_values > _GEOMETRY_ATOL]
    if positive_costs.size == 0:
        raise ContractError("adata.uns['cost_matrix'] must contain a positive off-diagonal cost")
    return float(np.median(positive_costs))


def _warn_zero_offdiag_pairs(cost_matrix: np.ndarray) -> None:
    """Warn when distinct state pairs are geometry-equivalent."""
    upper_values = cost_matrix[_upper_offdiag_mask(cost_matrix.shape[0])]
    n_zero_pairs = int(np.count_nonzero(np.abs(upper_values) <= _GEOMETRY_ATOL))
    if n_zero_pairs == 0:
        return
    warnings.warn(
        "adata.uns['cost_matrix'] contains "
        f"{n_zero_pairs} zero off-diagonal state pairs; "
        "the corresponding states are geometry-equivalent",
        UserWarning,
        stacklevel=3,
    )


def _validate_existing_cost_scale(value: object, expected_scale: float) -> None:
    """Validate that an existing scale matches canonical geometry scaling."""
    try:
        scale_array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive") from exc
    if scale_array.shape != ():
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive")
    scale = float(scale_array)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ContractError("adata.uns['cost_scale'] must be finite and strictly positive")
    if not np.isclose(scale, expected_scale, rtol=0.0, atol=_GEOMETRY_ATOL):
        raise ContractError(
            "adata.uns['cost_scale'] must match the median positive off-diagonal cost "
            "of adata.uns['cost_matrix']"
        )
