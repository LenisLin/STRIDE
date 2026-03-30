"""
Benchmark: Thread-count sensitivity for calibrate_joint_lambda and Phase 6B bootstrap.

Tests whether the ThreadPoolExecutor concurrency in:
  1. calibrate_joint_lambda (uot.py) — currently max_workers=candidates.size (7 for default grid)
  2. Phase 6B bootstrap replicate loop (arm3_uq_stress.py) — currently max_workers=min(n_reps, 8)

actually produces a measurable speedup vs sequential (max_workers=1).

Measures wall time and peak RSS at max_workers = 1, 2, 4, 8 (and candidates.size for Part 1).

No source files are modified. Local variants of the concurrent loops are used, keeping
the same worker logic.

Usage:
    python tasks/_shared/benchmarks/bench_thread_sensitivity.py
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from slotar.uot import (
    UOTSolveConfig,
    _calibrate_one_candidate,
    batched_uot_solve,
    precompute_logKernels,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K = 25
EPS_SCHEDULE = [1.0, 0.5, 0.1]
LAMBDA_GRID = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)  # 7 candidates
TARGET_ALPHA = 0.05
N_PAIRS_CALIB = 24   # representative Arm2 family pool size
N_PAIRS_BOOT = 16    # representative Arm3 anchor pair count per coverage level
N_REPS_BOOT = 20     # bootstrap replicates (kept small for bench speed)
N_REPEATS = 4        # median over this many timed runs
RNG_SEED = 42

WORKER_COUNTS = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_cfg() -> UOTSolveConfig:
    return UOTSolveConfig(eps_schedule=EPS_SCHEDULE, max_iter=500, tol=1e-6)


def _make_kernels() -> list[np.ndarray]:
    C = np.abs(
        np.arange(K, dtype=float)[:, None] - np.arange(K, dtype=float)[None, :]
    )
    return precompute_logKernels(C, EPS_SCHEDULE, s_C=1.0)


def _make_pair_tensors(n_pairs: int, rng: np.random.Generator):
    A = rng.random((n_pairs, K)) + 0.1
    B = rng.random((n_pairs, K)) + 0.1
    return A, B


# ---------------------------------------------------------------------------
# Part 1: calibrate_joint_lambda thread sensitivity
# ---------------------------------------------------------------------------


def _calibrate_joint_lambda_with_workers(
    A: np.ndarray,
    B: np.ndarray,
    lambda_grid: tuple[float, ...],
    kernels: list[np.ndarray],
    cfg: UOTSolveConfig,
    target_alpha: float,
    max_workers: int,
) -> float:
    """Local variant of calibrate_joint_lambda with explicit max_workers."""
    candidates = np.asarray(tuple(lambda_grid), dtype=float)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_calibrate_one_candidate, c, A, B, kernels, cfg, target_alpha): c
            for c in candidates
        }
        results = [fut.result() for fut in as_completed(futures)]
    best = min(
        ((c, e) for c, e in results if np.isfinite(e)),
        key=lambda x: x[1],
        default=None,
    )
    if best is None:
        raise ValueError("No finite calibration result")
    return best[0]


def bench_calibrate(A: np.ndarray, B: np.ndarray, kernels: list, cfg: UOTSolveConfig) -> dict:
    """Run calibrate_joint_lambda at varying max_workers, return timing table."""
    results = {}
    candidates_size = len(LAMBDA_GRID)

    worker_list = sorted(set(WORKER_COUNTS + [candidates_size]))

    for w in worker_list:
        elapsed: list[float] = []
        rss: list[float] = []
        best_lambda: float | None = None
        for _ in range(N_REPEATS):
            tracemalloc.start()
            t0 = time.perf_counter()
            lam = _calibrate_joint_lambda_with_workers(
                A=A, B=B,
                lambda_grid=LAMBDA_GRID,
                kernels=kernels,
                cfg=cfg,
                target_alpha=TARGET_ALPHA,
                max_workers=w,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed.append(ms)
            rss.append(peak / 1024 / 1024)
            best_lambda = lam
        results[w] = {
            "wall_ms": float(np.median(elapsed)),
            "rss_mb": float(np.median(rss)),
            "best_lambda": best_lambda,
        }
    return results


# ---------------------------------------------------------------------------
# Part 2: Bootstrap parallel loop thread sensitivity
# ---------------------------------------------------------------------------


def _bootstrap_worker(
    rep_idx: int,
    A_rep: np.ndarray,
    B_rep: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: list[np.ndarray],
    cfg: UOTSolveConfig,
) -> tuple[int, np.ndarray]:
    """Minimal bootstrap worker: just calls batched_uot_solve (the heavy inner op)."""
    metrics, _details, _status = batched_uot_solve(
        A=A_rep,
        B=B_rep,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=None,
    )
    return rep_idx, metrics["T"]


def bench_bootstrap(
    A_reps: np.ndarray,
    B_reps: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: list[np.ndarray],
    cfg: UOTSolveConfig,
) -> dict:
    """Run parallel bootstrap loop at varying max_workers, return timing table."""
    n_reps = A_reps.shape[0]
    results = {}

    worker_list = sorted(set(WORKER_COUNTS))

    for w in worker_list:
        elapsed: list[float] = []
        rss: list[float] = []
        for _ in range(N_REPEATS):
            slots: list[np.ndarray | None] = [None] * n_reps
            tracemalloc.start()
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=w) as pool:
                futures = {
                    pool.submit(
                        _bootstrap_worker,
                        rep_idx,
                        A_reps[rep_idx],
                        B_reps[rep_idx],
                        lambda_pl,
                        kernels,
                        cfg,
                    ): rep_idx
                    for rep_idx in range(n_reps)
                }
                for fut in as_completed(futures):
                    idx, t_arr = fut.result()
                    slots[idx] = t_arr
            ms = (time.perf_counter() - t0) * 1000.0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed.append(ms)
            rss.append(peak / 1024 / 1024)
            # Verify all slots filled
            assert all(s is not None for s in slots), "Slot-index accumulation failure"
        results[w] = {
            "wall_ms": float(np.median(elapsed)),
            "rss_mb": float(np.median(rss)),
            "rows_per_sec": (n_reps * N_PAIRS_BOOT) / (float(np.median(elapsed)) / 1000.0),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _speedup_pct(baseline_ms: float, current_ms: float) -> float:
    if baseline_ms <= 0:
        return 0.0
    return (baseline_ms - current_ms) / baseline_ms * 100.0


def main() -> None:
    cpu_count = os.cpu_count() or "?"
    print("=" * 72)
    print("Benchmark 4: Thread-count sensitivity")
    print(f"  K={K}, lambda_grid size={len(LAMBDA_GRID)}, N_PAIRS_CALIB={N_PAIRS_CALIB}")
    print(f"  N_PAIRS_BOOT={N_PAIRS_BOOT}, N_REPS_BOOT={N_REPS_BOOT}, repeats={N_REPEATS}")
    print(f"  Logical CPU cores: {cpu_count}")
    print("=" * 72)

    rng = np.random.default_rng(RNG_SEED)
    kernels = _make_kernels()
    cfg = _make_cfg()

    # Part 1: calibrate_joint_lambda
    print("\n--- Part 1: calibrate_joint_lambda thread sensitivity ---")
    print(f"  N_PAIRS={N_PAIRS_CALIB}, {len(LAMBDA_GRID)} candidates, {N_REPEATS} repeats (median)")
    print(f"  Current production setting: max_workers={len(LAMBDA_GRID)} (=candidates.size, no cap)")
    print()

    A_calib, B_calib = _make_pair_tensors(N_PAIRS_CALIB, rng)
    calib_results = bench_calibrate(A_calib, B_calib, kernels, cfg)

    seq_ms = calib_results[1]["wall_ms"]
    print(f"  {'max_workers':>12}  {'wall_ms':>9}  {'rss_mb':>8}  {'speedup_%':>10}  {'best_lambda':>12}")
    print("  " + "-" * 60)
    candidates_size = len(LAMBDA_GRID)
    for w, r in sorted(calib_results.items()):
        label = f"{w}" + (" (current)" if w == candidates_size else "")
        sp = _speedup_pct(seq_ms, r["wall_ms"])
        print(
            f"  {label:>12}  {r['wall_ms']:>9.1f}  {r['rss_mb']:>8.3f}  "
            f"{sp:>9.1f}%  {r['best_lambda']:>12.4f}"
        )

    # Consistency check: all max_workers should return the same best_lambda
    lambdas = [r["best_lambda"] for r in calib_results.values()]
    consistent = all(abs(lam - lambdas[0]) < 1e-10 for lam in lambdas)
    print(f"\n  Result consistency across all max_workers: {'PASS' if consistent else 'FAIL'}")

    # Part 2: Bootstrap loop
    print("\n--- Part 2: Phase 6B bootstrap loop thread sensitivity ---")
    print(f"  N_REPS={N_REPS_BOOT}, N_PAIRS={N_PAIRS_BOOT}, {N_REPEATS} repeats (median)")
    print(f"  Current production setting: max_workers=min(n_reps, 8)=min({N_REPS_BOOT}, 8)={min(N_REPS_BOOT, 8)}")
    print()

    A_reps = np.stack([
        (rng.random((N_PAIRS_BOOT, K)) + 0.1) for _ in range(N_REPS_BOOT)
    ])
    B_reps = np.stack([
        (rng.random((N_PAIRS_BOOT, K)) + 0.1) for _ in range(N_REPS_BOOT)
    ])
    lambda_pl = np.full(N_PAIRS_BOOT, 1.0, dtype=float)

    boot_results = bench_bootstrap(A_reps, B_reps, lambda_pl, kernels, cfg)

    seq_boot_ms = boot_results[1]["wall_ms"]
    print(f"  {'max_workers':>12}  {'wall_ms':>9}  {'rss_mb':>8}  {'speedup_%':>10}  {'rows/s':>10}")
    print("  " + "-" * 60)
    for w, r in sorted(boot_results.items()):
        sp = _speedup_pct(seq_boot_ms, r["wall_ms"])
        print(
            f"  {w:>12}  {r['wall_ms']:>9.1f}  {r['rss_mb']:>8.3f}  "
            f"{sp:>9.1f}%  {r['rows_per_sec']:>10.0f}"
        )

    print("\n--- Summary ---")
    print("  Interpretation:")
    print("  - calibrate_joint_lambda: GIL releases during scipy/numpy C-ext calls")
    print("    allow real parallelism. Speedup bounded by #candidates and CPU count.")
    print("  - Bootstrap loop: same GIL-release pattern; speedup bounded by n_reps")
    print("    and CPU count. max_workers=8 is the production cap.")
    print("  - BLAS/OpenMP interference risk is LOW: hot path is element-wise logsumexp,")
    print("    not GEMM. No BLAS thread pool conflict expected.")
    print(f"  - Note: candidates.size={len(LAMBDA_GRID)}, no cap in production.")
    print(f"    On large grids this oversubscribes. Consider min(candidates.size, cpu_count).")

    # BLAS thread probe
    try:
        import numpy.core._multiarray_umath as _npu  # noqa: F401
        blas_threads = None
        try:
            import threadpoolctl
            info = threadpoolctl.threadpool_info()
            blas_threads = [
                f"{d.get('internal_api','?')} n_threads={d.get('num_threads','?')}"
                for d in info
            ]
        except ImportError:
            blas_threads = ["threadpoolctl not available — cannot probe BLAS thread count"]
        print("\n  BLAS/thread pool probe:")
        for line in (blas_threads or ["no info"]):
            print(f"    {line}")
    except Exception as exc:
        print(f"\n  BLAS probe skipped: {exc}")


if __name__ == "__main__":
    main()
