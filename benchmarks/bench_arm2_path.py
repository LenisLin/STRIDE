"""
Benchmark: TaskA Arm2 formal/post-hoc path.

Measures wall time, peak RSS, and rows/sec for:
  1. calibrate_joint_lambda per pair family (3 families, N varies per family)
  2. batched_uot_solve on the full 48-pair Arm2 set

Attributes calibrate_joint_lambda overhead as a fraction of total time.

Usage:
    python benchmarks/bench_arm2_path.py
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

from slotar.uot import UOTSolveConfig, batched_uot_solve, calibrate_joint_lambda, precompute_logKernels
from tests.helpers_task_a_fixture import (
    K_FULL,
    ROI_SPECS,
    ARM2_ORDERED_PAIR_SPECS,
    expected_arm2_pair_records,
    expected_roi_vectors,
)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

EPS_SCHEDULE = [1.0, 0.5, 0.1]
LAMBDA_GRID = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
TARGET_ALPHA = 0.05
N_REPEATS = 3

# ---------------------------------------------------------------------------
# Build fixture data
# ---------------------------------------------------------------------------


def _build_arm2_fixture():
    """Return roi_vectors, pair_meta aligned to Arm2 pair spec."""
    roi_vectors = expected_roi_vectors(K_FULL)
    pair_records = expected_arm2_pair_records()
    pair_meta = pd.DataFrame.from_records(pair_records)
    return roi_vectors, pair_meta


def _build_kernels():
    C = np.abs(np.arange(K_FULL, dtype=float)[:, None] - np.arange(K_FULL, dtype=float)[None, :])
    return precompute_logKernels(C, EPS_SCHEDULE, s_C=1.0)


def _build_cfg() -> UOTSolveConfig:
    return UOTSolveConfig(eps_schedule=EPS_SCHEDULE, max_iter=500, tol=1e-6)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _bench_calibrate(roi_vectors, pair_meta, kernels, cfg) -> dict:
    """Time calibrate_joint_lambda for each of the 3 pair families."""
    family_times: dict[str, float] = {}
    for pair_family in ("TC-IM", "IM-PT", "TC-PT"):
        family_rows = pair_meta[pair_meta["pair_family"] == pair_family]
        A = np.stack([roi_vectors[str(r)] for r in family_rows["roi_a"]], axis=0).astype(float)
        B = np.stack([roi_vectors[str(r)] for r in family_rows["roi_b"]], axis=0).astype(float)
        elapsed: list[float] = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            calibrate_joint_lambda(A=A, B=B, lambda_grid=LAMBDA_GRID, kernels=kernels, cfg=cfg, target_alpha=TARGET_ALPHA)
            elapsed.append((time.perf_counter() - t0) * 1000.0)
        family_times[pair_family] = float(np.median(elapsed))
    return family_times


def _bench_solve(roi_vectors, pair_meta, kernels, cfg) -> dict:
    """Time batched_uot_solve on the full 48-pair set."""
    A = np.stack([roi_vectors[str(r)] for r in pair_meta["roi_a"]], axis=0).astype(float)
    B = np.stack([roi_vectors[str(r)] for r in pair_meta["roi_b"]], axis=0).astype(float)
    n = A.shape[0]
    lambda_pl = np.full(n, 1.0, dtype=float)

    elapsed: list[float] = []
    peak_rss: list[float] = []
    for _ in range(N_REPEATS):
        tracemalloc.start()
        t0 = time.perf_counter()
        batched_uot_solve(A=A, B=B, lambda_pl=lambda_pl, kernels=kernels, cfg=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed.append(ms)
        peak_rss.append(peak / 1024 / 1024)

    return {
        "n_pairs": n,
        "solve_ms": float(np.median(elapsed)),
        "peak_rss_mb": float(np.median(peak_rss)),
        "rows_per_sec": n / (float(np.median(elapsed)) / 1000.0),
    }


def main() -> None:
    print("=" * 72)
    print("Benchmark 2: TaskA Arm2 formal/post-hoc path")
    print(f"  K={K_FULL}, eps_schedule={EPS_SCHEDULE}, lambda_grid size={len(LAMBDA_GRID)}, repeats={N_REPEATS}")
    print("=" * 72)

    roi_vectors, pair_meta = _build_arm2_fixture()
    kernels = _build_kernels()
    cfg = _build_cfg()

    print(f"\nFixture: {len(pair_meta)} ordered pairs across 3 pair families, {len(roi_vectors)} ROIs\n")

    # 1. calibrate_joint_lambda per family
    print("--- Step 1: calibrate_joint_lambda per pair family ---")
    family_times = _bench_calibrate(roi_vectors, pair_meta, kernels, cfg)
    total_calib_ms = sum(family_times.values())
    for fam, ms in family_times.items():
        n_pairs = len(pair_meta[pair_meta["pair_family"] == fam])
        print(f"  {fam}: {ms:.1f} ms  ({n_pairs} pairs, {n_pairs * len(LAMBDA_GRID)} solver calls)")
    print(f"  Total calibration: {total_calib_ms:.1f} ms")

    # 2. batched_uot_solve
    print("\n--- Step 2: batched_uot_solve (full 48-pair set, one lambda per family) ---")
    solve_result = _bench_solve(roi_vectors, pair_meta, kernels, cfg)
    print(f"  {solve_result['n_pairs']} pairs: {solve_result['solve_ms']:.1f} ms  "
          f"| {solve_result['rows_per_sec']:.0f} rows/s  "
          f"| peak RSS {solve_result['peak_rss_mb']:.2f} MB")

    total_ms = total_calib_ms + solve_result["solve_ms"]

    print("\n--- Attribution summary ---")
    print(f"  calibrate_joint_lambda: {total_calib_ms:.1f} ms  ({total_calib_ms / total_ms * 100:.0f}% of total)")
    print(f"  batched_uot_solve:      {solve_result['solve_ms']:.1f} ms  ({solve_result['solve_ms'] / total_ms * 100:.0f}% of total)")
    print(f"  Total Arm2 path:        {total_ms:.1f} ms")
    print()
    print("Interpretation:")
    print("  - calibrate_joint_lambda runs N_pairs × len(lambda_grid) UOT solves per family.")
    print("  - The post-hoc analysis_compute.py recompute layer is a single batched_uot_solve")
    print("    (shown above). Overhead relative to calibration scales with lambda_grid size.")
    print("  - Remaining slow paths: any step not shown here (e.g., IO, clinical analysis)")
    print("    is outside the scope of this refactor.")


if __name__ == "__main__":
    main()
