"""
Benchmark: Solver-only batched UOT microbenchmark.

Measures wall time, peak RSS, and rows/sec for batched_uot_solve across
varying batch sizes N, with K=25 prototypes (Task A canonical dimension).

Compares two modes:
  Mode A — current (chunked extraction):  _EXTRACTION_TARGET_PLAN_ELEMENTS = 250_000
  Mode B — no-chunking baseline:          _EXTRACTION_TARGET_PLAN_ELEMENTS = 10**9

Separately times the solve phase and extraction phase to attribute cost.

Usage:
    python benchmarks/bench_uot_solver.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import slotar.uot as uot_mod
from slotar.uot import UOTSolveConfig, batched_uot_solve, precompute_logKernels

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

K = 25
EPS_SCHEDULE = [1.0, 0.5, 0.1]
BATCH_SIZES = [1, 10, 50, 100, 250, 500, 1_000]
N_REPEATS = 2
LAMBDA_VAL = 1.0
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tensors(n: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random non-negative [N, K] mass tensors and cost kernels."""
    A = rng.random((n, k)) + 0.1
    B = rng.random((n, k)) + 0.1
    lambda_pl = np.full(n, LAMBDA_VAL, dtype=float)
    C = np.abs(np.arange(k, dtype=float)[:, None] - np.arange(k, dtype=float)[None, :])
    kernels = precompute_logKernels(C, EPS_SCHEDULE, s_C=1.0)
    return A, B, lambda_pl, kernels


def _make_cfg() -> UOTSolveConfig:
    return UOTSolveConfig(eps_schedule=EPS_SCHEDULE, max_iter=500, tol=1e-6)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class _PhaseTimer:
    """Thin context manager for wall-time measurement inside benchmark runs."""
    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def _run_benchmark_single(n: int, chunk_limit: int) -> dict[str, float]:
    """Run one (N, chunk_limit) benchmark, return timing + RSS metrics."""
    rng = np.random.default_rng(RNG_SEED)
    A, B, lambda_pl, kernels = _make_tensors(n, K, rng)
    cfg = _make_cfg()

    # Patch the module-level constant to control chunking
    orig = uot_mod._EXTRACTION_TARGET_PLAN_ELEMENTS
    uot_mod._EXTRACTION_TARGET_PLAN_ELEMENTS = chunk_limit

    results: list[dict[str, float]] = []
    for _ in range(N_REPEATS):
        # Patch the two internal functions to get separate phase timings.
        # We do this by wrapping with a lightweight timer that records the
        # cumulative time inside each phase.
        solve_timer = _PhaseTimer()
        extract_timer = _PhaseTimer()

        orig_solve = uot_mod._batched_log_sinkhorn_eps_scaling
        orig_extract = uot_mod._extract_batched_metrics

        def _timed_solve(*args, **kwargs):
            with solve_timer:
                return orig_solve(*args, **kwargs)

        def _timed_extract(*args, **kwargs):
            with extract_timer:
                return orig_extract(*args, **kwargs)

        uot_mod._batched_log_sinkhorn_eps_scaling = _timed_solve
        uot_mod._extract_batched_metrics = _timed_extract

        tracemalloc.start()
        t0 = time.perf_counter()
        batched_uot_solve(
            A=A,
            B=B,
            lambda_pl=lambda_pl,
            kernels=kernels,
            cfg=cfg,
            return_plan=True,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        uot_mod._batched_log_sinkhorn_eps_scaling = orig_solve
        uot_mod._extract_batched_metrics = orig_extract

        results.append({
            "total_ms": total_ms,
            "solve_ms": solve_timer.elapsed_ms,
            "extract_ms": extract_timer.elapsed_ms,
            "peak_rss_mb": peak / 1024 / 1024,
        })

    uot_mod._EXTRACTION_TARGET_PLAN_ELEMENTS = orig

    # Report median over repeats
    return {
        "total_ms": float(np.median([r["total_ms"] for r in results])),
        "solve_ms": float(np.median([r["solve_ms"] for r in results])),
        "extract_ms": float(np.median([r["extract_ms"] for r in results])),
        "peak_rss_mb": float(np.median([r["peak_rss_mb"] for r in results])),
        "rows_per_sec": n / (float(np.median([r["total_ms"] for r in results])) / 1000.0),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Benchmark 1: Solver-only batched UOT microbenchmark")
    print(f"  K={K}, eps_schedule={EPS_SCHEDULE}, return_plan=True, repeats={N_REPEATS}")
    print("=" * 72)

    header = f"{'N':>6}  {'mode':>12}  {'total_ms':>9}  {'solve_ms':>9}  {'extract_ms':>11}  {'rss_mb':>7}  {'rows/s':>10}"
    print(header)
    print("-" * len(header))

    attribution_rows: list[dict] = []

    for n in BATCH_SIZES:
        for label, chunk_limit in [("chunked", 250_000), ("no-chunk", 10**9)]:
            r = _run_benchmark_single(n, chunk_limit)
            print(
                f"{n:>6}  {label:>12}  {r['total_ms']:>9.1f}  {r['solve_ms']:>9.1f}  "
                f"{r['extract_ms']:>11.1f}  {r['peak_rss_mb']:>7.2f}  {r['rows_per_sec']:>10.0f}"
            )
            attribution_rows.append({"n": n, "mode": label, **r})

    # Attribution summary
    print()
    print("--- Attribution: Chunked Extraction Improvement ---")
    print(f"{'N':>6}  {'Δtotal_ms':>10}  {'Δextract_ms':>12}  {'speedup_%':>10}")
    chunked = {r["n"]: r for r in attribution_rows if r["mode"] == "chunked"}
    nochunk = {r["n"]: r for r in attribution_rows if r["mode"] == "no-chunk"}
    for n in BATCH_SIZES:
        if n < 250:
            continue
        c = chunked[n]
        nc = nochunk[n]
        delta_total = nc["total_ms"] - c["total_ms"]
        delta_extract = nc["extract_ms"] - c["extract_ms"]
        speedup = delta_total / nc["total_ms"] * 100 if nc["total_ms"] > 0 else 0.0
        print(f"{n:>6}  {delta_total:>10.1f}  {delta_extract:>12.1f}  {speedup:>9.1f}%")

    print()
    print("Interpretation:")
    print("  - Δtotal_ms and Δextract_ms > 0 → chunked extraction is faster")
    print("  - Peak RSS reduction reflects memory savings from not materialising")
    print("    the full dense [N, K, K] plan slab before writing to output arrays")
    print("  - Remaining time in 'solve_ms' is outside this refactor's scope")


if __name__ == "__main__":
    main()
