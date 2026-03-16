"""
Module: tasks.task_A.arm3.block_partition

Phase 1 and Phase 2 of the Arm-3 pipeline: frozen grid partition and
full-coverage density vector construction.

Responsibilities:
- Assign each physical cell in the frozen Stage-0 artifact to an axis-aligned
  grid block, using spatial coordinates from obsm['spatial'].
- Enumerate the complete ROI block universe for every ROI, including blocks
  with zero physical cells (zero-cell blocks are mandatory in the block universe).
- Compute per-block prototype count summaries (roi_block_summary), ensuring
  zero-cell blocks are represented with count_k* = 0.
- Build full-coverage density vectors: n_k / sum_b(Area_b) where the area
  denominator spans ALL blocks in the ROI envelope, including zero-cell blocks.

Design constraints:
- Density area uses geometric grid area only (block_size_units^2 * coord_to_mm2).
- Do NOT use uns['roi_areas'] or cell_area_sum as the density area denominator.
- All grid blocks in the ROI envelope are valid in Arm-3 v1; no sparse-block
  or background filtering; no N_MIN_BLOCK validity gate.
- Zero-cell blocks must remain present in the ROI block universe and block summaries.
- block_id encodes (roi_id, grid_col, grid_row) as "{roi_id}::{col}::{row}".
- Snapping uses strict mathematical floor/ceil (see below); no heuristic margins.

Grid envelope snapping formula (strictly enforced):
    x_min_grid = floor(x_min / block_size_units) * block_size_units
    x_max_grid = ceil(x_max  / block_size_units) * block_size_units
    y_min_grid = floor(y_min / block_size_units) * block_size_units
    y_max_grid = ceil(y_max  / block_size_units) * block_size_units
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import K_FULL


def build_grid_partition(
    spatial_xy: np.ndarray,
    roi_ids: np.ndarray,
    proto_ids: np.ndarray,
    block_size_units: float,
    coord_to_mm2: float,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Assign each physical cell to a grid block and enumerate the full ROI block universe.

    For each ROI, computes the axis-aligned bounding box from spatial_xy, snaps
    the grid origin outward using strict floor/ceil, enumerates all grid cells
    in the resulting envelope (the ROI block universe), then maps each physical
    cell to its containing grid block.

    The returned per-cell DataFrame does NOT include rows for zero-cell blocks;
    zero-cell blocks are recorded in the returned roi_block_universe dict.
    Pass roi_block_universe to compute_roi_block_summary to ensure zero-cell
    blocks appear in all downstream summaries.

    Parameters
    ----------
    spatial_xy : np.ndarray, shape (N_cells, 2)
        Spatial coordinates from obsm['spatial'] in the frozen Stage-0 artifact.
        Units assumed to be micrometres (1 coord unit = 1 µm), consistent with
        COORD_TO_MM2 = 1e-6.
    roi_ids : np.ndarray, shape (N_cells,), dtype str or object
        Per-cell roi_id values decoded from obs/roi_id/codes + categories.
    proto_ids : np.ndarray, shape (N_cells,), dtype int
        Per-cell prototype assignments from obs/proto_id (range 0..K_FULL-1).
    block_size_units : float
        Grid cell side length in coordinate units. Use DEFAULT_BLOCK_SIZE_UNITS.
    coord_to_mm2 : float
        Conversion factor: (coord_unit)^2 -> mm^2. Use COORD_TO_MM2 = 1e-6.

    Returns
    -------
    block_frame : pd.DataFrame
        One row per physical cell with columns:
            cell_idx        int    original 0-based cell index
            roi_id          str    ROI identity
            block_id        str    "{roi_id}::{col_idx}::{row_idx}"
            proto_id        int    prototype assignment
            block_area_mm2  float  block_size_units^2 * coord_to_mm2 (constant)
        Zero-cell blocks are absent from this frame.

    roi_block_universe : dict[str, list[str]]
        Maps roi_id -> complete sorted list of block_ids in that ROI's envelope.
        Ordered by (col_idx, row_idx). Includes all grid cells in the snapped
        bounding box, including zero-cell blocks.
    """
    if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
        raise ValueError(
            f"spatial_xy must have shape (N_cells, 2), got {spatial_xy.shape}"
        )
    n_cells = spatial_xy.shape[0]
    if roi_ids.shape[0] != n_cells:
        raise ValueError("roi_ids length must match spatial_xy row count")
    if proto_ids.shape[0] != n_cells:
        raise ValueError("proto_ids length must match spatial_xy row count")
    if block_size_units <= 0.0:
        raise ValueError(f"block_size_units must be positive, got {block_size_units}")
    if coord_to_mm2 <= 0.0:
        raise ValueError(f"coord_to_mm2 must be positive, got {coord_to_mm2}")

    block_area_mm2 = block_size_units ** 2 * coord_to_mm2
    roi_ids_str = np.asarray(roi_ids, dtype=object)  # ensure object dtype for str comparison

    unique_rois = sorted(set(roi_ids_str.tolist()))
    if not unique_rois:
        raise ValueError("roi_ids contains no valid ROI identifiers")

    cell_records: list[dict] = []
    roi_block_universe: dict[str, list[str]] = {}

    for roi_id in unique_rois:
        cell_mask = roi_ids_str == roi_id
        cell_indices = np.flatnonzero(cell_mask)
        xy = spatial_xy[cell_mask]  # shape (n_roi_cells, 2)
        pids = proto_ids[cell_mask]

        x_min = float(xy[:, 0].min())
        x_max = float(xy[:, 0].max())
        y_min = float(xy[:, 1].min())
        y_max = float(xy[:, 1].max())

        # Strict mathematical snapping — no heuristic margins
        x_min_grid = np.floor(x_min / block_size_units) * block_size_units
        x_max_grid = np.ceil(x_max / block_size_units) * block_size_units
        y_min_grid = np.floor(y_min / block_size_units) * block_size_units
        y_max_grid = np.ceil(y_max / block_size_units) * block_size_units

        # Number of blocks along each axis
        # Use round() to avoid floating-point near-integer errors in the division
        n_cols = max(1, int(round((x_max_grid - x_min_grid) / block_size_units)))
        n_rows = max(1, int(round((y_max_grid - y_min_grid) / block_size_units)))

        # Enumerate the complete ROI block universe (column-major order)
        block_ids_ordered: list[str] = []
        for col_idx in range(n_cols):
            for row_idx in range(n_rows):
                block_ids_ordered.append(f"{roi_id}::{col_idx}::{row_idx}")
        roi_block_universe[str(roi_id)] = block_ids_ordered

        # Map each cell to its block by computing grid column and row indices
        col_indices = np.floor(
            (xy[:, 0] - x_min_grid) / block_size_units
        ).astype(int)
        row_indices = np.floor(
            (xy[:, 1] - y_min_grid) / block_size_units
        ).astype(int)
        # Clip to valid range (guards against floating-point edge cases at boundary)
        col_indices = np.clip(col_indices, 0, n_cols - 1)
        row_indices = np.clip(row_indices, 0, n_rows - 1)

        for i in range(len(cell_indices)):
            cell_records.append(
                {
                    "cell_idx": int(cell_indices[i]),
                    "roi_id": str(roi_id),
                    "block_id": f"{roi_id}::{int(col_indices[i])}::{int(row_indices[i])}",
                    "proto_id": int(pids[i]),
                    "block_area_mm2": block_area_mm2,
                }
            )

    block_frame = pd.DataFrame.from_records(cell_records)
    if not block_frame.empty:
        block_frame["cell_idx"] = block_frame["cell_idx"].astype(int)
        block_frame["proto_id"] = block_frame["proto_id"].astype(int)
        block_frame["block_area_mm2"] = block_frame["block_area_mm2"].astype(float)

    return block_frame, roi_block_universe


def compute_roi_block_summary(
    block_frame: pd.DataFrame,
    roi_block_universe: dict[str, list[str]],
    k_full: int,
    block_area_mm2: float,
) -> dict[str, pd.DataFrame]:
    """
    Build the complete per-block prototype count summary for every ROI.

    All blocks in the ROI block universe are represented in the output, including
    blocks with zero physical cells. Zero-cell blocks have count_k* = 0.0 and
    nonzero block_area_mm2.

    Implementation strategy to preserve zero-cell blocks:
    1. Build the full block ID list from roi_block_universe (includes zero-cell blocks).
    2. Accumulate observed counts from block_frame using numpy scatter.
    3. Construct the output DataFrame from the full list, filling missing blocks with 0.

    Parameters
    ----------
    block_frame : pd.DataFrame
        Per-cell block assignments from build_grid_partition.
    roi_block_universe : dict[str, list[str]]
        Maps roi_id -> complete list of block_ids in that ROI's envelope.
    k_full : int
        Number of prototypes on the shared prototype axis.
    block_area_mm2 : float
        Geometric area per block in mm^2 (constant for uniform grid).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by roi_id. Each DataFrame has one row per block:
            block_id        str
            block_area_mm2  float
            count_k0 .. count_k{K-1}  float
        Zero-cell blocks are present with all count_k* = 0.0.
    """
    if k_full <= 0:
        raise ValueError(f"k_full must be a positive integer, got {k_full}")

    count_col_names = [f"count_k{k}" for k in range(k_full)]
    summary: dict[str, pd.DataFrame] = {}

    for roi_id, block_ids in roi_block_universe.items():
        n_blocks = len(block_ids)
        # Map block_id -> integer row index for O(1) scatter
        block_to_idx: dict[str, int] = {bid: i for i, bid in enumerate(block_ids)}

        # Initialise count array; shape (n_blocks, k_full)
        counts = np.zeros((n_blocks, k_full), dtype=float)

        # Select cells belonging to this ROI
        roi_mask = block_frame["roi_id"].astype(str) == str(roi_id)
        if roi_mask.any():
            roi_cells = block_frame.loc[roi_mask, ["block_id", "proto_id"]]
            block_row_indices = roi_cells["block_id"].map(block_to_idx).to_numpy()
            proto_col_indices = roi_cells["proto_id"].to_numpy()

            # Validate indices to catch any upstream inconsistency
            valid = (
                (block_row_indices >= 0)
                & (block_row_indices < n_blocks)
                & (proto_col_indices >= 0)
                & (proto_col_indices < k_full)
            )
            if not valid.all():
                n_invalid = int((~valid).sum())
                raise ValueError(
                    f"ROI {roi_id!r}: {n_invalid} cells have block_id or proto_id outside "
                    "the expected universe — check build_grid_partition output"
                )
            np.add.at(counts, (block_row_indices, proto_col_indices), 1.0)

        # Build output DataFrame from the full block universe (including zero-cell blocks)
        df = pd.DataFrame({"block_id": block_ids, "block_area_mm2": block_area_mm2})
        for k, col_name in enumerate(count_col_names):
            df[col_name] = counts[:, k]

        summary[str(roi_id)] = df

    return summary


def build_full_coverage_density_vectors(
    roi_block_summary: dict[str, pd.DataFrame],
    k_full: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Build full-coverage density vectors from the complete ROI block universe.

    Area denominator uses ALL blocks in the ROI block universe, including zero-cell
    blocks. This ensures density area equals the true geometric footprint of the ROI
    envelope. Do NOT use uns['roi_areas'] or cell_area_sum.

    Density formula for ROI r and prototype k:
        a_k^dens = sum_b(n_{b,k}) / sum_b(Area_b)
    where b ranges over the complete block universe for ROI r.

    Parameters
    ----------
    roi_block_summary : dict[str, pd.DataFrame]
        Output of compute_roi_block_summary. Must include zero-cell blocks.
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    roi_density_vectors : dict[str, np.ndarray]
        Keyed by roi_id. Shape (K,): sum_b(n_{b,k}) / sum_b(Area_b).
        Units: cells / mm^2.
    roi_total_areas : dict[str, float]
        Keyed by roi_id. sum_b(Area_b) over all blocks in ROI envelope (mm^2).
    """
    if k_full <= 0:
        raise ValueError(f"k_full must be a positive integer, got {k_full}")

    count_col_names = [f"count_k{k}" for k in range(k_full)]
    roi_density_vectors: dict[str, np.ndarray] = {}
    roi_total_areas: dict[str, float] = {}

    for roi_id, df in roi_block_summary.items():
        total_area = float(df["block_area_mm2"].sum())
        if total_area <= 0.0:
            raise ValueError(
                f"ROI {roi_id!r}: total block area is zero or negative ({total_area}). "
                "Check roi_block_summary contains at least one block."
            )

        # Sum prototype counts across all blocks (including zero-cell blocks)
        total_counts = df[count_col_names].sum(axis=0).to_numpy(dtype=float)
        density = total_counts / total_area  # units: cells / mm^2

        roi_density_vectors[str(roi_id)] = density
        roi_total_areas[str(roi_id)] = total_area

    return roi_density_vectors, roi_total_areas
