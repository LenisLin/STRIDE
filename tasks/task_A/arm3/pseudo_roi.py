"""
Module: tasks.task_A.arm3.pseudo_roi

Phase 5 of the Arm-3 pipeline: block bootstrap pseudo-ROI construction.

Responsibilities:
- Sample blocks with replacement from a ROI's frozen block universe to reach a
  target coverage fraction.
- Reconstruct pseudo-ROI density vectors from sampled blocks (with repetition).
- Run a full bootstrap pass over all pairs at a given coverage level, generating
  per-replicate density tensors and a per-replicate audit frame.

Design constraints:
- Sampled block IDs may include zero-cell blocks; zero-cell blocks contribute
  area (block_area_mm2) but zero prototype counts to the pseudo-ROI.
- Side A and side B resample independently from their own frozen block universes.
- Calibration is NOT performed on pseudo-ROIs; frozen lambda_dens and tau are
  broadcast from the full-coverage calibration phase.
- The frozen block universe (roi_block_universe) passed here must include
  zero-cell blocks; do not pre-filter it before calling these functions.
- No sparse-block filtering; no N_MIN_BLOCK validity gate.

Side A / side B bootstrap independence rule:
- Base replicate seed = rng_seed + replicate_idx
- Side A uses np.random.default_rng(base_seed)
- Side B uses np.random.default_rng(base_seed + SIDE_B_SEED_OFFSET)
- SIDE_B_SEED_OFFSET is an explicit named constant (1_000_000).
- This guarantees independent RNG streams that never share state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named constant: side-B seed offset.
# Side A and side B must be independently seeded within each replicate.
# Using a large fixed offset (1_000_000) ensures no overlap between the
# A and B RNG sequences for any realistic replicate count (<<= 1_000_000).
# ---------------------------------------------------------------------------

SIDE_B_SEED_OFFSET: int = 1_000_000


def sample_blocks_to_coverage(
    available_block_ids: list[str],
    target_coverage: float,
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample block IDs with replacement to reach a target coverage count.

    n_sampled = max(1, floor(target_coverage * len(available_block_ids))).

    available_block_ids is the complete frozen ROI block universe including
    zero-cell blocks. Sampled block IDs may include zero-cell blocks; each
    sampled zero-cell block contributes block_area_mm2 to pseudo-ROI area and
    zero counts to all prototype slots.

    Parameters
    ----------
    available_block_ids : list[str]
        Complete frozen ROI block universe (roi_block_universe[roi_id]).
        Must include zero-cell blocks.
    target_coverage : float
        Fraction of total blocks to sample, e.g. 0.75 for 75%.
        Must be in (0.0, 1.0]. Under the current Arm-3 contract, the active
        reduced bootstrap ladder is 0.75 / 0.50 / 0.25.
    rng : np.random.Generator
        Seeded random generator. Caller is responsible for seeding per replicate
        to ensure reproducibility.

    Returns
    -------
    list[str]
        Sampled block IDs (may contain repeats). Length =
        max(1, floor(target_coverage * len(available_block_ids))).
    """
    n_total = len(available_block_ids)
    if n_total == 0:
        raise ValueError("sample_blocks_to_coverage: available_block_ids is empty")
    n_sampled = max(1, int(np.floor(target_coverage * n_total)))
    idx = rng.integers(0, n_total, size=n_sampled)
    return [available_block_ids[i] for i in idx]


def build_pseudo_roi_density(
    roi_block_df: pd.DataFrame,
    sampled_block_ids: list[str],
    k_full: int,
) -> tuple[np.ndarray, float]:
    """
    Reconstruct pseudo-ROI density vector from sampled blocks (with repetition).

    Pseudo-ROI reconstruction formula:
        n_k^pseudo = sum over sampled blocks of count_k (with repetition)
        A^pseudo   = sum over sampled blocks of block_area_mm2 (with repetition)
        a_k^dens   = n_k^pseudo / A^pseudo

    Zero-cell blocks in sampled_block_ids contribute block_area_mm2 to A^pseudo
    and zero to n_k^pseudo for all k. They must not be excluded.

    Parameters
    ----------
    roi_block_df : pd.DataFrame
        One entry from roi_block_summary (output of compute_roi_block_summary).
        Must include zero-cell blocks. Must have columns block_id, block_area_mm2,
        count_k0 .. count_k{K-1}. Index is arbitrary (not required to be block_id).
    sampled_block_ids : list[str]
        Block IDs selected by sample_blocks_to_coverage (may contain repeats).
    k_full : int
        Prototype axis dimension.

    Returns
    -------
    density_vector : np.ndarray, shape (K,)
        n_k^pseudo / A^pseudo for each prototype k. Units: cells / mm^2.
    total_pseudo_area : float
        A^pseudo in mm^2 (sum of block_area_mm2 over sampled blocks with repetition).
    """
    count_cols = [f"count_k{k}" for k in range(k_full)]

    # Build a block_id -> integer row-index lookup for fast scatter
    block_to_idx: dict[str, int] = {
        bid: i for i, bid in enumerate(roi_block_df["block_id"].tolist())
    }

    areas: np.ndarray = roi_block_df["block_area_mm2"].to_numpy(dtype=float)
    counts: np.ndarray = roi_block_df[count_cols].to_numpy(dtype=float)  # (n_blocks, K)

    total_counts = np.zeros(k_full, dtype=float)
    total_area = 0.0

    for bid in sampled_block_ids:
        row_idx = block_to_idx[bid]
        total_area += areas[row_idx]
        total_counts += counts[row_idx]

    if total_area <= 0.0:
        raise ValueError(
            f"build_pseudo_roi_density: total_pseudo_area is {total_area} — "
            "all sampled blocks have zero area, which should not be possible"
        )

    density_vector = total_counts / total_area
    return density_vector, float(total_area)


def run_bootstrap_pass(
    roi_block_summary: dict[str, pd.DataFrame],
    pair_meta: pd.DataFrame,
    coverage: float,
    n_reps: int,
    k_full: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run one full bootstrap pass for a given coverage level.

    For each replicate i (0-based):
        base_seed = rng_seed + i
        side A uses np.random.default_rng(base_seed)
        side B uses np.random.default_rng(base_seed + SIDE_B_SEED_OFFSET)

    Side A and side B are independently seeded and never share RNG state.
    This guarantees reproducible and statistically independent samples on
    each side within every replicate.

    Sampled block IDs may include zero-cell blocks in both side A and side B
    samples. Zero-cell blocks contribute area but zero counts.

    Parameters
    ----------
    roi_block_summary : dict[str, pd.DataFrame]
        Frozen block partition from compute_roi_block_summary.
        Must include zero-cell blocks for every ROI.
    pair_meta : pd.DataFrame
        Ordered pair metadata restricted to the relevant direction subset
        (e.g. anchor directions only). Required columns:
            roi_a, roi_b, pair_id, patient_id, pair_family, compartment_a.
    coverage : float
        Target coverage fraction (0.0, 1.0].
    n_reps : int
        Number of bootstrap replicates per pair.
    k_full : int
        Prototype axis dimension.
    rng_seed : int
        Base seed. Replicate i uses rng_seed + i for side A,
        rng_seed + i + SIDE_B_SEED_OFFSET for side B.

    Returns
    -------
    A_reps : np.ndarray, shape (n_reps, N_pairs, K)
        Pseudo-ROI density tensors for side A across all replicates.
    B_reps : np.ndarray, shape (n_reps, N_pairs, K)
        Pseudo-ROI density tensors for side B across all replicates.
    pseudo_meta : pd.DataFrame
        One row per (replicate_id, pair_id) with columns:
            replicate_id, pair_id, coverage,
            pseudo_area_a_mm2, pseudo_area_b_mm2,
            n_blocks_sampled_a, n_blocks_sampled_b.
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError(
            f"run_bootstrap_pass: coverage must be in (0, 1], got {coverage!r}"
        )
    if n_reps <= 0:
        raise ValueError(
            f"run_bootstrap_pass: n_reps must be a positive integer, got {n_reps!r}"
        )

    pair_meta = pair_meta.reset_index(drop=True)
    n_pairs = len(pair_meta)
    count_cols = [f"count_k{k}" for k in range(k_full)]

    # Pre-cache per-ROI arrays for efficient numpy indexing.
    # Avoids rebuilding block_to_idx and numpy arrays in the inner loop.
    roi_cache: dict[str, dict] = {}
    for roi_id, df in roi_block_summary.items():
        block_ids_list = df["block_id"].tolist()
        areas = df["block_area_mm2"].to_numpy(dtype=float)
        counts = df[count_cols].to_numpy(dtype=float)  # (n_blocks, K)
        roi_cache[str(roi_id)] = {
            "block_ids": block_ids_list,
            "n_blocks": len(block_ids_list),
            "areas": areas,
            "counts": counts,
        }

    A_reps = np.zeros((n_reps, n_pairs, k_full), dtype=float)
    B_reps = np.zeros((n_reps, n_pairs, k_full), dtype=float)
    audit_rows: list[dict] = []

    for rep_idx in range(n_reps):
        base_seed = rng_seed + rep_idx
        rng_a = np.random.default_rng(base_seed)
        rng_b = np.random.default_rng(base_seed + SIDE_B_SEED_OFFSET)

        for pair_idx in range(n_pairs):
            row = pair_meta.iloc[pair_idx]
            roi_a = str(row["roi_a"])
            roi_b = str(row["roi_b"])
            pair_id = str(row["pair_id"])

            cache_a = roi_cache[roi_a]
            cache_b = roi_cache[roi_b]

            # --- side A ---
            n_a = cache_a["n_blocks"]
            n_sampled_a = max(1, int(np.floor(coverage * n_a)))
            idx_a = rng_a.integers(0, n_a, size=n_sampled_a)
            total_counts_a = cache_a["counts"][idx_a].sum(axis=0)   # (K,)
            total_area_a = float(cache_a["areas"][idx_a].sum())

            # --- side B ---
            n_b = cache_b["n_blocks"]
            n_sampled_b = max(1, int(np.floor(coverage * n_b)))
            idx_b = rng_b.integers(0, n_b, size=n_sampled_b)
            total_counts_b = cache_b["counts"][idx_b].sum(axis=0)   # (K,)
            total_area_b = float(cache_b["areas"][idx_b].sum())

            if total_area_a <= 0.0 or total_area_b <= 0.0:
                raise RuntimeError(
                    f"run_bootstrap_pass: zero pseudo-area at rep={rep_idx}, "
                    f"pair_idx={pair_idx} (roi_a={roi_a!r}, roi_b={roi_b!r}). "
                    "All blocks have zero area — check roi_block_summary."
                )

            A_reps[rep_idx, pair_idx] = total_counts_a / total_area_a
            B_reps[rep_idx, pair_idx] = total_counts_b / total_area_b

            audit_rows.append(
                {
                    "replicate_id": rep_idx,
                    "pair_id": pair_id,
                    "coverage": coverage,
                    "pseudo_area_a_mm2": total_area_a,
                    "pseudo_area_b_mm2": total_area_b,
                    "n_blocks_sampled_a": n_sampled_a,
                    "n_blocks_sampled_b": n_sampled_b,
                }
            )

    pseudo_meta = pd.DataFrame.from_records(audit_rows)
    return A_reps, B_reps, pseudo_meta
