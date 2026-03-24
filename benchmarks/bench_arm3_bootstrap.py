"""
Benchmark: Arm3 calibration + bootstrap-heavy path.

Measures wall time, peak RSS, and rows/sec for:
  1. calibrate_lambda_dens (3 families)
  2. calibrate_tau_by_compartment (3 compartments, return_plan=True)
  3. run_bootstrap_pass at n_reps=100 across coverage levels (0.75, 0.50, 0.25)

Compares two modes for run_bootstrap_pass:
  Mode A — current:  roi_cache pre-built (default implementation)
  Mode B — no-cache: per-pair DataFrame lookup (simulates pre-cache implementation)

Attributes roi_cache speedup: (Mode B - Mode A) / Mode B × 100%

Usage:
    python benchmarks/bench_arm3_bootstrap.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from slotar.uot import UOTSolveConfig, precompute_logKernels
from tasks.task_A.arm3.calibrate import calibrate_lambda_dens, calibrate_tau_by_compartment
from tasks.task_A.arm3.pseudo_roi import run_bootstrap_pass, SIDE_B_SEED_OFFSET
from tasks.task_A.arm3.constants import K_FULL, COVERAGE_LEVELS, ARM3_PAIR_FAMILIES

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

EPS_SCHEDULE = [1.0, 0.5, 0.1]
LAMBDA_GRID = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
TARGET_ALPHA = 0.05
N_REPS = 100
N_BLOCKS_PER_ROI = 50
RNG_SEED = 42
N_REPEATS_BENCH = 2

# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------
#
# 12 ROIs (2 patients × 3 compartments × 2 ROIs each), N_BLOCKS_PER_ROI blocks each.
# This matches the helpers_task_a_fixture.py ROI_SPECS structure.

_ROI_SPECS = (
    ("P01", "TC", "P01_TC_01"),
    ("P01", "TC", "P01_TC_02"),
    ("P01", "IM", "P01_IM_01"),
    ("P01", "IM", "P01_IM_02"),
    ("P01", "PT", "P01_PT_01"),
    ("P01", "PT", "P01_PT_02"),
    ("P02", "TC", "P02_TC_01"),
    ("P02", "TC", "P02_TC_02"),
    ("P02", "IM", "P02_IM_01"),
    ("P02", "IM", "P02_IM_02"),
    ("P02", "PT", "P02_PT_01"),
    ("P02", "PT", "P02_PT_02"),
)


def _build_fixtures(rng: np.random.Generator):
    """Build synthetic roi_block_summary, roi_density_vectors, pair_meta, and maps."""
    count_cols = [f"count_k{k}" for k in range(K_FULL)]
    roi_block_summary: dict[str, pd.DataFrame] = {}
    roi_density_vectors: dict[str, np.ndarray] = {}
    roi_compartment_map: dict[str, str] = {}
    roi_patient_map: dict[str, str] = {}

    for patient_id, compartment, roi_id in _ROI_SPECS:
        roi_compartment_map[roi_id] = compartment
        roi_patient_map[roi_id] = patient_id
        block_ids = [f"{roi_id}_blk{b:03d}" for b in range(N_BLOCKS_PER_ROI)]
        areas = rng.uniform(0.01, 0.05, size=N_BLOCKS_PER_ROI)
        counts = rng.integers(0, 20, size=(N_BLOCKS_PER_ROI, K_FULL)).astype(float)
        df = pd.DataFrame({"block_id": block_ids, "block_area_mm2": areas})
        for k, col in enumerate(count_cols):
            df[col] = counts[:, k]
        roi_block_summary[roi_id] = df
        total_area = float(areas.sum())
        roi_density_vectors[roi_id] = counts.sum(axis=0) / total_area

    # Build pair_meta: all 6 ordered cross-compartment directions within each patient
    pair_records = []
    for patient_id in ("P01", "P02"):
        patient_rois = {(comp, roi_id) for _, comp, roi_id in _ROI_SPECS if _ == patient_id
                        for _, comp2, roi_id in _ROI_SPECS if comp2 == comp}

        # Simplified: build from specs directly
        comp_rois: dict[str, list[str]] = {}
        for pid, comp, roi_id in _ROI_SPECS:
            if pid == patient_id:
                comp_rois.setdefault(comp, []).append(roi_id)

        direction_to_family = {
            ("TC", "IM"): "TC-IM", ("IM", "TC"): "TC-IM",
            ("IM", "PT"): "IM-PT", ("PT", "IM"): "IM-PT",
            ("TC", "PT"): "TC-PT", ("PT", "TC"): "TC-PT",
        }
        for (ca, cb), family in direction_to_family.items():
            for roi_a in comp_rois.get(ca, []):
                for roi_b in comp_rois.get(cb, []):
                    pair_records.append({
                        "pair_id": f"A3::{patient_id}::{roi_a}::{roi_b}",
                        "patient_id": patient_id,
                        "roi_a": roi_a,
                        "roi_b": roi_b,
                        "pair_family": family,
                        "compartment_a": ca,
                        "compartment_b": cb,
                    })
    pair_meta = pd.DataFrame.from_records(pair_records)
    return roi_block_summary, roi_density_vectors, roi_compartment_map, roi_patient_map, pair_meta


def _build_kernels_and_cfg():
    C = np.abs(np.arange(K_FULL, dtype=float)[:, None] - np.arange(K_FULL, dtype=float)[None, :])
    kernels = precompute_logKernels(C, EPS_SCHEDULE, s_C=1.0)
    cfg = UOTSolveConfig(eps_schedule=EPS_SCHEDULE, max_iter=500, tol=1e-6)
    scaled_cost = C.copy()
    return kernels, cfg, scaled_cost


# ---------------------------------------------------------------------------
# No-cache variant of run_bootstrap_pass (simulates pre-optimization baseline)
# ---------------------------------------------------------------------------


def _run_bootstrap_pass_no_cache(
    roi_block_summary: dict[str, pd.DataFrame],
    pair_meta: pd.DataFrame,
    coverage: float,
    n_reps: int,
    k_full: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Equivalent to run_bootstrap_pass but without pre-built roi_cache.
    Performs per-pair DataFrame lookup on every replicate iteration.
    This simulates the pre-optimization code path.
    """
    count_cols = [f"count_k{k}" for k in range(k_full)]
    pair_meta = pair_meta.reset_index(drop=True)
    n_pairs = len(pair_meta)
    A_reps = np.zeros((n_reps, n_pairs, k_full), dtype=float)
    B_reps = np.zeros((n_reps, n_pairs, k_full), dtype=float)
    audit_rows = []

    for rep_idx in range(n_reps):
        base_seed = rng_seed + rep_idx
        rng_a = np.random.default_rng(base_seed)
        rng_b = np.random.default_rng(base_seed + SIDE_B_SEED_OFFSET)

        for pair_idx in range(n_pairs):
            row = pair_meta.iloc[pair_idx]
            roi_a = str(row["roi_a"])
            roi_b = str(row["roi_b"])
            pair_id = str(row["pair_id"])

            # --- no-cache: look up DataFrame per-pair, per-rep ---
            df_a = roi_block_summary[roi_a]
            areas_a = df_a["block_area_mm2"].to_numpy(dtype=float)
            counts_a = df_a[count_cols].to_numpy(dtype=float)
            n_a = len(areas_a)

            df_b = roi_block_summary[roi_b]
            areas_b = df_b["block_area_mm2"].to_numpy(dtype=float)
            counts_b = df_b[count_cols].to_numpy(dtype=float)
            n_b = len(areas_b)

            n_sampled_a = max(1, int(np.floor(coverage * n_a)))
            idx_a = rng_a.integers(0, n_a, size=n_sampled_a)
            total_counts_a = counts_a[idx_a].sum(axis=0)
            total_area_a = float(areas_a[idx_a].sum())

            n_sampled_b = max(1, int(np.floor(coverage * n_b)))
            idx_b = rng_b.integers(0, n_b, size=n_sampled_b)
            total_counts_b = counts_b[idx_b].sum(axis=0)
            total_area_b = float(areas_b[idx_b].sum())

            A_reps[rep_idx, pair_idx] = total_counts_a / total_area_a
            B_reps[rep_idx, pair_idx] = total_counts_b / total_area_b
            audit_rows.append({
                "replicate_id": rep_idx,
                "pair_id": pair_id,
                "coverage": coverage,
                "pseudo_area_a_mm2": total_area_a,
                "pseudo_area_b_mm2": total_area_b,
                "n_blocks_sampled_a": n_sampled_a,
                "n_blocks_sampled_b": n_sampled_b,
            })

    return A_reps, B_reps, pd.DataFrame.from_records(audit_rows)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Benchmark 3: Arm3 calibration + bootstrap-heavy path")
    print(f"  K={K_FULL}, N_BLOCKS_PER_ROI={N_BLOCKS_PER_ROI}, n_reps={N_REPS}")
    print(f"  eps_schedule={EPS_SCHEDULE}, lambda_grid size={len(LAMBDA_GRID)}")
    print("=" * 72)

    rng = np.random.default_rng(RNG_SEED)
    roi_block_summary, roi_density_vectors, roi_compartment_map, roi_patient_map, pair_meta = _build_fixtures(rng)
    kernels, cfg, scaled_cost = _build_kernels_and_cfg()

    print(f"\nFixture: {len(_ROI_SPECS)} ROIs, {len(pair_meta)} ordered pairs\n")

    # --- Step 1: calibrate_lambda_dens ---
    print("--- Step 1: calibrate_lambda_dens (3 families) ---")
    calib_elapsed: list[float] = []
    for _ in range(N_REPEATS_BENCH):
        t0 = time.perf_counter()
        lambda_dens = calibrate_lambda_dens(
            roi_density_vectors=roi_density_vectors,
            pair_meta=pair_meta,
            k_full=K_FULL,
            lambda_grid=LAMBDA_GRID,
            uot_cfg=cfg,
            kernels=kernels,
            target_alpha=TARGET_ALPHA,
        )
        calib_elapsed.append((time.perf_counter() - t0) * 1000.0)
    calib_ms = float(np.median(calib_elapsed))
    print(f"  lambda_dens={lambda_dens}")
    print(f"  Time: {calib_ms:.1f} ms (median of {N_REPEATS_BENCH} repeats)\n")

    # --- Step 2: calibrate_tau_by_compartment ---
    print("--- Step 2: calibrate_tau_by_compartment (3 compartments, return_plan=True) ---")
    tau_elapsed: list[float] = []
    for _ in range(N_REPEATS_BENCH):
        t0 = time.perf_counter()
        tau_by_compartment = calibrate_tau_by_compartment(
            roi_density_vectors=roi_density_vectors,
            roi_compartment_map=roi_compartment_map,
            roi_patient_map=roi_patient_map,
            k_full=K_FULL,
            scaled_cost_matrix=scaled_cost,
            frozen_lambdas=lambda_dens,
            uot_cfg=cfg,
            kernels=kernels,
        )
        tau_elapsed.append((time.perf_counter() - t0) * 1000.0)
    tau_ms = float(np.median(tau_elapsed))
    print(f"  tau_by_compartment={tau_by_compartment}")
    print(f"  Time: {tau_ms:.1f} ms (median of {N_REPEATS_BENCH} repeats)")
    print(f"  Note: dense [K, K] plan materialization dominates this phase.\n")

    # --- Step 3: run_bootstrap_pass per coverage level ---
    print("--- Step 3: run_bootstrap_pass at n_reps={} ---".format(N_REPS))
    print(f"  {'coverage':>10}  {'mode':>10}  {'wall_ms':>9}  {'rss_mb':>8}  {'rows/s':>10}")
    print("  " + "-" * 58)

    anchor_pair_meta = pair_meta[pair_meta["pair_family"].isin(("TC-IM", "TC-PT"))].reset_index(drop=True)
    n_anchor = len(anchor_pair_meta)

    cache_times: dict[float, float] = {}
    nocache_times: dict[float, float] = {}

    for coverage in COVERAGE_LEVELS:
        for label, fn in [("cached", run_bootstrap_pass), ("no-cache", _run_bootstrap_pass_no_cache)]:
            elapsed_list: list[float] = []
            rss_list: list[float] = []
            for _ in range(N_REPEATS_BENCH):
                tracemalloc.start()
                t0 = time.perf_counter()
                fn(
                    roi_block_summary=roi_block_summary,
                    pair_meta=anchor_pair_meta,
                    coverage=coverage,
                    n_reps=N_REPS,
                    k_full=K_FULL,
                    rng_seed=RNG_SEED,
                )
                ms = (time.perf_counter() - t0) * 1000.0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                elapsed_list.append(ms)
                rss_list.append(peak / 1024 / 1024)

            med_ms = float(np.median(elapsed_list))
            med_rss = float(np.median(rss_list))
            rows_per_s = (N_REPS * n_anchor) / (med_ms / 1000.0)
            print(f"  {coverage:>10.2f}  {label:>10}  {med_ms:>9.1f}  {med_rss:>8.2f}  {rows_per_s:>10.0f}")
            if label == "cached":
                cache_times[coverage] = med_ms
            else:
                nocache_times[coverage] = med_ms

    # Attribution summary
    print()
    print("--- Attribution: ROI reference caching (pseudo_roi.py roi_cache) ---")
    print(f"  {'coverage':>10}  {'Δms (saved)':>12}  {'speedup_%':>11}")
    for coverage in COVERAGE_LEVELS:
        nc_ms = nocache_times[coverage]
        c_ms = cache_times[coverage]
        delta = nc_ms - c_ms
        speedup = delta / nc_ms * 100.0 if nc_ms > 0 else 0.0
        print(f"  {coverage:>10.2f}  {delta:>12.1f}  {speedup:>10.1f}%")

    total_calib_ms = calib_ms + tau_ms
    print()
    print("--- Overall Arm3 attribution ---")
    print(f"  calibrate_lambda_dens:          {calib_ms:.1f} ms")
    print(f"  calibrate_tau_by_compartment:   {tau_ms:.1f} ms  (dense plan path — K×K per pair)")
    print(f"  run_bootstrap_pass (75%, cache):{cache_times[0.75]:.1f} ms  for {N_REPS} reps × {n_anchor} pairs")
    print()
    print("Interpretation:")
    print("  - roi_cache pre-builds block_id→numpy arrays once, avoiding per-rep DataFrame")
    print("    slices. Speedup is proportional to n_reps × n_pairs.")
    print("  - calibrate_tau_by_compartment materialises dense [K, K] plans; this cost is")
    print("    fixed (not bootstrap-repeated) and outside the caching refactor.")
    print("  - Remaining slow paths: dense plan materialisation in tau calibration is the")
    print("    largest non-cached overhead and would require algorithmic changes to reduce.")


if __name__ == "__main__":
    main()
