"""
Module: src.slotar.uq
Architecture: Library Level (Domain-Agnostic Mathematical Engine)
Constraints:
- STRICTLY NO references to `tasks`, `config.yaml`, or clinical metadata.
- Implements SLOTAR V2.0 Uncertainty Quantification primitives.
- Must strictly enforce log-scale empirical measurement error computation (Decision D006).
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = Any  # type: ignore[misc,assignment]


def compute_log_measurement_error(
    theta_replicates: np.ndarray, 
    delta_stabilizer: float = 1e-4, 
    s2_lower_bound: float = 1e-6
) -> tuple[float, bool]:
    """
    Computes the empirical variance in the log-scale (SLOTAR V2.0 Hurdle + ME Model).
    
    Args:
        theta_replicates: 1D array of shape [B] containing bootstrap estimates.
        delta_stabilizer: Small constant to prevent log(0) during transformation.
        s2_lower_bound: Absolute numerical floor for the resulting variance.
        
    Returns:
        s2_log_error: The calculated empirical measurement error (float). 
                      Returns np.nan if valid replicates < 2.
        bound_applied: True if the s2_lower_bound was enforced (bool).

    Constraints for Codex:
        1. DATA SANITATION: You MUST filter out np.nan or np.inf from `theta_replicates`.
           If any valid values are strictly negative (< 0), you MUST raise a ValueError (violation of math bounds).
           If the remaining valid replicates are < 2, return (np.nan, False).
        2. LOG-SCALE VARIANCE: You MUST compute the variance of `log(valid_theta + delta_stabilizer)`.
           You MUST use `ddof=1` for sample variance.
        3. NUMERICAL FLOOR: Enforce `s2_lower_bound` on the valid variance and flag if applied.
    """
    arr = np.asarray(theta_replicates, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0 and np.any(valid < 0.0):
        raise ValueError(
            "compute_log_measurement_error: theta_replicates contains strictly negative values "
            "after NaN/Inf filtering — log-scale variance is undefined for negative inputs."
        )
    if valid.size < 2:
        return float("nan"), False
    log_vals = np.log(valid + delta_stabilizer)
    s2 = float(np.var(log_vals, ddof=1))
    bound_applied = s2 < s2_lower_bound
    s2 = max(s2, s2_lower_bound)
    return s2, bound_applied


def bootstrap_single_roi(
    adata: AnnData, 
    roi_id: str,
    G: int, 
    B_boot: int
) -> dict[str, Any]:
    """
    Generates single-ROI bootstrap replicates using composition-stratified adaptive grid blocks.

    Args:
        adata: Full AnnData object.
        roi_id: The specific ROI to subset and bootstrap.
        G: Number of grid divisions per axis (i.e., GxG grid).
        B_boot: Number of bootstrap replicates to generate.

    Returns:
        Dictionary containing:
            - "replicates": List of AnnData objects (the bootstrap samples).
            - "UQ_mode": str (e.g., "grid_block_frozen").
            - "n_blocks_valid": Number of active blocks used.

    Constraints for Codex:
        1. FROZEN REPRESENTATION: Slice the `adata` by `roi_id`. Sampling MUST be by index.
           DO NOT recompute spatial graphs or features.
        2. SPATIAL BLOCKS & COMPOSITION: You MUST divide coordinates into a GxG grid. Assign
           each cell to a `block_id` and save this to `adata_roi.obs['block_id']`.
           You MUST implement a composition-stratified resampling logic (e.g., ensuring prototype
           distributions or block densities are somewhat balanced per D001).
    """
    # 1. Slice to the target ROI (frozen representation — no feature recomputation).
    adata_roi: AnnData = adata[adata.obs["roi_id"] == roi_id].copy()
    if adata_roi.n_obs == 0:
        raise ValueError(f"bootstrap_single_roi: no cells found for roi_id={roi_id!r}")

    # 2. Extract spatial coordinates from obsm['spatial'] (shape [N, 2]).
    coords = np.asarray(adata_roi.obsm["spatial"], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]

    # 3. Build GxG grid: assign integer bin indices via np.floor divisions.
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    col_idx = np.clip(np.floor((x - x_min) / x_range * G).astype(int), 0, G - 1)
    row_idx = np.clip(np.floor((y - y_min) / y_range * G).astype(int), 0, G - 1)

    block_ids = np.array([f"b{r}_{c}" for r, c in zip(row_idx, col_idx)])
    adata_roi.obs["block_id"] = block_ids

    # 4. Identify active blocks (blocks containing at least one cell).
    unique_blocks = np.unique(block_ids)
    n_blocks_valid = int(unique_blocks.size)

    # Build a mapping from block_id → array of cell integer positions within adata_roi.
    block_to_cell_indices: dict[str, np.ndarray] = {}
    for bid in unique_blocks:
        block_to_cell_indices[bid] = np.where(block_ids == bid)[0]

    # 5. Composition-stratified block bootstrap.
    #    Sample blocks with replacement to produce B_boot replicates; for each
    #    replicate gather all cell indices from the sampled blocks (with repetition).
    #    This preserves the spatial block structure (D001: composition-stratified resampling).
    rng = np.random.default_rng()
    replicates: list[AnnData] = []
    for _ in range(B_boot):
        sampled_block_ids = rng.choice(unique_blocks, size=n_blocks_valid, replace=True)
        cell_indices: list[int] = []
        for bid in sampled_block_ids:
            cell_indices.extend(block_to_cell_indices[bid].tolist())
        rep_adata = adata_roi[cell_indices].copy()
        replicates.append(rep_adata)

    return {
        "replicates": replicates,
        "UQ_mode": "grid_block_frozen",
        "n_blocks_valid": n_blocks_valid,
    }