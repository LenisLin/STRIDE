"""
Module: tasks.task_A.arm3_uq_stress

Top-level orchestration runner for Task A Arm-3: density-primary coverage-reduction
and UQ stress test on the frozen Stage-0 artifact.

This module orchestrates the 8-phase Arm-3 execution sequence:

    Phase 0 — Constants and frozen manifest verification
    Phase 1 — Frozen grid partition (block_partition.build_grid_partition)
    Phase 2 — Full-coverage density reference (block_partition.build_full_coverage_density_vectors)
    Phase 3 — Pair universe and anchor-direction subset
    Phase 4 — Full-coverage calibration
               lambda_dens: IMPLEMENTED (calibrate.calibrate_lambda_dens)
               tau:         IMPLEMENTED (calibrate.calibrate_tau_by_compartment)
    Phase 5 — Pseudo-ROI bootstrap                    [IMPLEMENTED]
    Phase 6 — Arm-3 inference                         [IMPLEMENTED]
    Phase 7 — Continuous retention summary            [IMPLEMENTED]
    Phase 8 — Prototype summary and descriptive memo  [IMPLEMENTED]

Current tranche: Phase 0–8. The runner writes Phase 0–3 validation, Phase 4
calibration, Phase 5–6 inference packages, Phase 7 summaries, and the Phase 8
prototype summary / descriptive memo outputs to result_root.

Design constraints:
- Arm-3 stays entirely in the Task layer. Nothing is moved into src/slotar/.
- Arm-3 is density-primary; count tensors from common.assemble_tensors are not
  used for density-mode inference.
- Full-coverage COUNT tensors (from roi_block_summary) are built separately in
  Phase 5/6 solely for support mask construction; they are not passed to the
  UOT solver.
- This module can now be invoked through `tasks/task_A/pipeline.py` when
  `enabled_arms` includes `A3_uq_stress`.
- result_root is accepted as an argument; no output path is hard-coded here.
- Stage-0 spatial coordinates are read directly via h5py to avoid loading the
  full artifact into memory (follows arm2/analysis_io.py HDF5 style).
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .arm2_spatial_gradient import ORDERED_PAIR_SPECS, PAIR_FAMILIES
from .arm3.block_partition import (
    build_full_coverage_density_vectors,
    build_grid_partition,
    compute_roi_block_summary,
)
from .arm3.calibrate import calibrate_lambda_dens, calibrate_tau_by_compartment
from .arm3.constants import (
    ARM3_ANCHOR_DIRECTIONS,
    ARM3_NAME,
    COORD_TO_MM2,
    COVERAGE_LEVELS,
    DEFAULT_BLOCK_SIZE_UNITS,
    DEFAULT_N_REPS,
    DEFAULT_RNG_SEED,
    DENSITY_EPS,
    K_FULL,
)
from .arm3.inference import (
    assemble_density_tensors,
    broadcast_frozen_lambda,
    broadcast_frozen_tau,
    compute_arm3_density_metrics,
    compute_floor_dominated_flags,
    freeze_support_masks,
    run_uot_batch_with_events,
)
from .arm3.output import (
    build_arm3_memo,
    build_prototype_stability_table,
    write_phase8_outputs,
)
from .arm3.pseudo_roi import run_bootstrap_pass
from .arm3.retention import (
    build_prototype_contrast_table,
    compute_contrast_degradation_summary,
    compute_degradation_summary,
)
from .common import resolve_task_a_mass_mode, run_balanced_ot_batch
from slotar.uot import UOTSolveConfig, precompute_logKernels

# ---------------------------------------------------------------------------
# Phase 0–3 output file names
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "arm3_phase0_manifest.json"
_BLOCK_FRAME_FILENAME = "arm3_phase1_block_frame.parquet"
_BLOCK_UNIVERSE_FILENAME = "arm3_phase1_roi_block_universe.json"
_BLOCK_SUMMARY_FILENAME = "arm3_phase1_roi_block_summary.parquet"
_DENSITY_REFERENCE_FILENAME = "arm3_phase2_roi_density_reference.parquet"
_PAIR_META_FULL_FILENAME = "arm3_phase3_pair_meta_full.parquet"
_PAIR_META_ANCHOR_FILENAME = "arm3_phase3_pair_meta_anchor.parquet"

# ---------------------------------------------------------------------------
# Phase 4 output file names
# ---------------------------------------------------------------------------

_LAMBDA_DENS_FILENAME = "arm3_phase4_lambda_dens.json"
_TAU_BY_COMPARTMENT_FILENAME = "arm3_phase4_tau_by_compartment.json"
_CALIBRATION_RECORD_FILENAME = "arm3_phase4_calibration_record.json"
_TIMING_SUMMARY_FILENAME = "arm3_runtime_timing.json"

# ---------------------------------------------------------------------------
# Phase 5/6 output file names
# ---------------------------------------------------------------------------

_PSEUDO_ROI_AUDIT_FILENAME = "arm3_phase5_pseudo_roi_audit.parquet"
_FULL_COVERAGE_RESULTS_FILENAME = "arm3_phase6_full_coverage_results.parquet"
_BOOTSTRAP_RESULTS_FILENAME = "arm3_phase6_bootstrap_results.parquet"
_BALANCED_OT_RESULTS_FILENAME = "arm3_phase6_balanced_ot_results.parquet"
_SUPPORT_MASK_AUDIT_FILENAME = "arm3_phase6_support_mask_audit.parquet"
_METRIC_SUMMARY_ANCHOR_FILENAME = "arm3_phase6_metric_summary_anchor.parquet"

# Patch 2: prototype-level event parquets (written by Phase 6, consumed by Phase 8)
_PROTO_EVENTS_FULL_FILENAME = "arm3_phase6_prototype_events_full.parquet"
_PROTO_EVENTS_BOOTSTRAP_FILENAME = "arm3_phase6_prototype_events_bootstrap.parquet"

# ---------------------------------------------------------------------------
# Phase 7 output file names
# ---------------------------------------------------------------------------

_PHASE7_DEGRADATION_FILENAME = "arm3_phase7_degradation_summary"   # .parquet + .csv
# Patch 2A: contrast-based sign-consistency summary
_PHASE7_CONTRAST_FILENAME = "arm3_phase7_contrast_summary"          # .parquet + .csv

# ---------------------------------------------------------------------------
# Phase 8 output file names
# ---------------------------------------------------------------------------

# Patch 2B: prototype contrast prep table for Phase 8
_PHASE8_PROTO_CONTRAST_FILENAME = "arm3_phase8_prototype_contrast_prep"  # .parquet

ARM3_LAMBDA_MODE = "pair_specific_joint"
ARM3_TAU_MODE = "task_fixed_by_compartment"


@contextmanager
def _timed_section(timing_seconds: dict[str, float], name: str):
    started = perf_counter()
    try:
        yield
    finally:
        timing_seconds[name] = timing_seconds.get(name, 0.0) + (perf_counter() - started)


def _coverage_key(coverage: float) -> str:
    return f"coverage_{int(round(float(coverage) * 100.0)):02d}"


def _write_timing_summary(
    *,
    result_root: Path,
    timing_seconds: dict[str, float],
    n_anchor_pairs: int,
    n_reps: int,
    coverage_levels: tuple[float, ...],
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_instrumentation_added": True,
        "n_anchor_pairs": int(n_anchor_pairs),
        "n_reps": int(n_reps),
        "coverage_levels": [float(value) for value in coverage_levels],
        "timing_seconds": {
            key: float(value)
            for key, value in sorted(timing_seconds.items())
        },
    }
    with (result_root / _TIMING_SUMMARY_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------------------
# Public runner (Phase 0–4)
# ---------------------------------------------------------------------------


def run_arm3(
    stage0_path: Path | str,
    config: dict[str, Any],
    result_root: Path | str,
) -> pd.DataFrame:
    """
    Run the Arm-3 Phase 0–8 pipeline and write validation, calibration,
    inference packages, Phase 7 summaries, Phase 8 summary outputs, and a
    shared full-coverage pair-level metrics surface.

    Accepts the frozen Stage-0 artifact path, task config, and an externally
    provided result root. Writes Phase 0–3 validation, Phase 4 calibration,
    Phase 5–6 inference files, Phase 7 summaries, and the Phase 8 prototype
    summary / descriptive memo outputs to result_root.

    Parameters
    ----------
    stage0_path : Path | str
        Path to the frozen Stage-0 .h5ad artifact. Canonical location:
        /mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad
    config : dict[str, Any]
        Task config dict. Required key: data.k_full.
        Optional keys:
            uot_params.*             (eps_schedule, max_iter, tol, eta_floor, n_min_proto)
            arm3.block_size_units    (overrides DEFAULT_BLOCK_SIZE_UNITS)
            arm3.lambda_grid         (falls back to arm2.lambda_grid)
            arm3.target_alpha        (falls back to arm2.target_alpha or 0.05)
            arm3.tau_q               (default 0.5)
            arm3.n_reps              (default DEFAULT_N_REPS=100)
            arm3.rng_seed            (default DEFAULT_RNG_SEED=42)
    result_root : Path | str
        Root directory for output files. Must be config/CLI-provided.
        Must point to external result location; not the repository tree.

    Returns
    -------
    pd.DataFrame
        Arm-3 full-coverage anchor-pair metrics narrowed to the shared Task-A
        operator surface plus Arm-3-specific columns.
    """
    stage0_path = Path(stage0_path)
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    total_started = perf_counter()
    timing_seconds: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Phase 0: Constants and manifest
    # -----------------------------------------------------------------------
    k_full = int(config["data"]["k_full"])
    if k_full != K_FULL:
        raise ValueError(
            f"config data.k_full={k_full} does not match locked K_FULL={K_FULL}"
        )

    arm3_cfg = config.get("arm3", {})
    block_size_units = float(arm3_cfg.get("block_size_units", DEFAULT_BLOCK_SIZE_UNITS))
    block_area_mm2 = block_size_units ** 2 * COORD_TO_MM2
    mass_mode = resolve_task_a_mass_mode(config, ARM3_NAME)

    print(f"[Arm-3 Phase 0] ARM3_NAME={ARM3_NAME}, K_FULL={K_FULL}, "
          f"block_size_units={block_size_units}, COORD_TO_MM2={COORD_TO_MM2}")

    # -----------------------------------------------------------------------
    # Phase 1: Load spatial inputs and build frozen grid partition
    # -----------------------------------------------------------------------
    print(f"[Arm-3 Phase 1] Loading Stage-0 spatial inputs from {stage0_path}")
    with _timed_section(timing_seconds, "phase1.load_stage0_inputs_seconds"):
        spatial_xy, roi_ids_arr, proto_ids_arr, roi_table, cost_matrix, s_C, proto_labels = (
            _load_arm3_stage0_inputs(stage0_path, k_full)
        )
    print(f"[Arm-3 Phase 1] Prototype labels loaded: {proto_labels[:5]}{'...' if len(proto_labels) > 5 else ''}")

    print(f"[Arm-3 Phase 1] {spatial_xy.shape[0]:,} cells, "
          f"{len(roi_table['roi_id'].unique()):,} ROIs")

    with _timed_section(timing_seconds, "phase1.build_grid_partition_seconds"):
        block_frame, roi_block_universe = build_grid_partition(
            spatial_xy=spatial_xy,
            roi_ids=roi_ids_arr,
            proto_ids=proto_ids_arr,
            block_size_units=block_size_units,
            coord_to_mm2=COORD_TO_MM2,
        )

    n_total_blocks = sum(len(v) for v in roi_block_universe.values())
    print(f"[Arm-3 Phase 1] Grid partition: {n_total_blocks:,} total blocks "
          f"across {len(roi_block_universe):,} ROIs")

    with _timed_section(timing_seconds, "phase1.compute_roi_block_summary_seconds"):
        roi_block_summary = compute_roi_block_summary(
            block_frame=block_frame,
            roi_block_universe=roi_block_universe,
            k_full=k_full,
            block_area_mm2=block_area_mm2,
        )

    # Validate zero-cell blocks survived aggregation
    for roi_id, df in roi_block_summary.items():
        expected_n = len(roi_block_universe[roi_id])
        if len(df) != expected_n:
            raise RuntimeError(
                f"ROI {roi_id!r}: roi_block_summary has {len(df)} rows but "
                f"roi_block_universe has {expected_n} blocks — zero-cell blocks lost"
            )

    # -----------------------------------------------------------------------
    # Phase 2: Build full-coverage density reference from all blocks
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 2] Building full-coverage density vectors")
    with _timed_section(timing_seconds, "phase2.build_density_reference_seconds"):
        roi_density_vectors, roi_total_areas = build_full_coverage_density_vectors(
            roi_block_summary=roi_block_summary,
            k_full=k_full,
        )
    print(f"[Arm-3 Phase 2] Density vectors built for {len(roi_density_vectors):,} ROIs")

    # -----------------------------------------------------------------------
    # Phase 3: Pair universe and anchor-direction subset
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 3] Building pair universe")
    with _timed_section(timing_seconds, "phase3.build_pair_universe_seconds"):
        pair_meta_full, pair_meta_anchor = _build_pair_universe(roi_table)
    print(f"[Arm-3 Phase 3] Full pair universe: {len(pair_meta_full):,} rows; "
          f"Anchor subset: {len(pair_meta_anchor):,} rows")

    # Validate all ROIs referenced in pair_meta are present in density vectors
    all_pair_rois = set(pair_meta_full["roi_a"].astype(str)) | set(pair_meta_full["roi_b"].astype(str))
    missing_rois = sorted(all_pair_rois - set(roi_density_vectors))
    if missing_rois:
        raise ValueError(
            f"Pair universe references {len(missing_rois)} ROI(s) absent from "
            f"density vectors: {missing_rois[:5]}{'...' if len(missing_rois) > 5 else ''}"
        )

    # -----------------------------------------------------------------------
    # Write Phase 0–3 validation package
    # -----------------------------------------------------------------------
    print(f"[Arm-3] Writing Phase 0–3 validation package to {result_root}")
    with _timed_section(timing_seconds, "phase0_3.write_outputs_seconds"):
        _write_phase0_3_outputs(
            result_root=result_root,
            stage0_path=stage0_path,
            block_size_units=block_size_units,
            block_area_mm2=block_area_mm2,
            spatial_xy=spatial_xy,
            block_frame=block_frame,
            roi_block_universe=roi_block_universe,
            roi_block_summary=roi_block_summary,
            roi_density_vectors=roi_density_vectors,
            roi_total_areas=roi_total_areas,
            pair_meta_full=pair_meta_full,
            pair_meta_anchor=pair_meta_anchor,
        )
    print("[Arm-3] Phase 0–3 validation package written.")

    # -----------------------------------------------------------------------
    # Phase 4 pre-calibration integrity checks
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 4] Running pre-calibration integrity checks")
    _check_block_universe_integrity(roi_block_universe)
    _check_pair_universe_integrity(
        pair_meta_full=pair_meta_full,
        pair_meta_anchor=pair_meta_anchor,
        roi_density_vectors=roi_density_vectors,
    )
    print("[Arm-3 Phase 4] Integrity checks passed")

    # -----------------------------------------------------------------------
    # Phase 4: Full-coverage calibration
    # -----------------------------------------------------------------------

    # Build UOT config and kernels from config + frozen cost geometry
    with _timed_section(timing_seconds, "phase4.build_uot_runtime_seconds"):
        uot_cfg = _build_uot_cfg(config)
        kernels = precompute_logKernels(cost_matrix, uot_cfg.eps_schedule, s_C=float(s_C))
    scaled_cost_matrix = np.asarray(cost_matrix, dtype=float) / float(s_C)

    # Resolve lambda grid: arm3.lambda_grid → arm2.lambda_grid → error
    arm2_cfg = config.get("arm2", {})
    raw_lambda_grid = arm3_cfg.get("lambda_grid", arm2_cfg.get("lambda_grid"))
    if raw_lambda_grid is None:
        raise ValueError(
            "Phase 4 calibration requires a lambda grid. "
            "Set config key 'arm3.lambda_grid' or 'arm2.lambda_grid'."
        )
    lambda_grid: tuple[float, ...] = tuple(float(v) for v in raw_lambda_grid)
    target_alpha = float(
        arm3_cfg.get("target_alpha", arm2_cfg.get("target_alpha", 0.05))
    )

    print(
        f"[Arm-3 Phase 4] Calibrating lambda_dens per pair family "
        f"(grid={lambda_grid}, target_alpha={target_alpha})"
    )
    with _timed_section(timing_seconds, "phase4.calibrate_lambda_seconds"):
        lambda_dens = calibrate_lambda_dens(
            roi_density_vectors=roi_density_vectors,
            pair_meta=pair_meta_full,
            k_full=k_full,
            lambda_grid=lambda_grid,
            uot_cfg=uot_cfg,
            kernels=kernels,
            target_alpha=target_alpha,
        )
    print(f"[Arm-3 Phase 4] lambda_dens calibrated: {lambda_dens}")

    # Build roi_compartment_map and roi_patient_map for tau calibration
    roi_compartment_map: dict[str, str] = dict(
        zip(
            roi_table["roi_id"].astype(str).tolist(),
            roi_table["compartment"].astype(str).tolist(),
        )
    )
    roi_patient_map: dict[str, str] = dict(
        zip(
            roi_table["roi_id"].astype(str).tolist(),
            roi_table["patient_id"].astype(str).tolist(),
        )
    )

    # tau_q: quantile level for the Pi-weighted cost distribution calibration.
    # Default 0.5 (median). Override via config key 'arm3.tau_q'.
    tau_q = float(arm3_cfg.get("tau_q", 0.5))

    print(f"[Arm-3 Phase 4] Calibrating tau per compartment (tau_q={tau_q})")
    with _timed_section(timing_seconds, "phase4.calibrate_tau_seconds"):
        tau_by_compartment = calibrate_tau_by_compartment(
            roi_density_vectors=roi_density_vectors,
            roi_compartment_map=roi_compartment_map,
            roi_patient_map=roi_patient_map,
            k_full=k_full,
            scaled_cost_matrix=scaled_cost_matrix,
            frozen_lambdas=lambda_dens,
            uot_cfg=uot_cfg,
            kernels=kernels,
            tau_q=tau_q,
        )
    print(f"[Arm-3 Phase 4] tau_by_compartment calibrated: {tau_by_compartment}")

    pair_meta_anchor_runtime = pair_meta_anchor.copy()
    pair_meta_anchor_runtime["arm"] = ARM3_NAME
    pair_meta_anchor_runtime["mass_mode"] = mass_mode
    pair_meta_anchor_runtime["lambda_mode"] = ARM3_LAMBDA_MODE
    pair_meta_anchor_runtime["tau_mode"] = ARM3_TAU_MODE
    pair_meta_anchor_runtime["group_mode"] = "provided"
    pair_meta_anchor_runtime["drift_mode"] = "unavailable"

    # Write Phase 4 outputs
    print(f"[Arm-3 Phase 4] Writing calibration outputs to {result_root}")
    with _timed_section(timing_seconds, "phase4.write_outputs_seconds"):
        _write_phase4_outputs(
            result_root=result_root,
            lambda_dens=lambda_dens,
            lambda_grid=lambda_grid,
            target_alpha=target_alpha,
            tau_by_compartment=tau_by_compartment,
            tau_q=tau_q,
        )
    print("[Arm-3 Phase 4] Calibration outputs written.")

    # -----------------------------------------------------------------------
    # Phase 5/6 setup
    # -----------------------------------------------------------------------

    n_reps = int(arm3_cfg.get("n_reps", DEFAULT_N_REPS))
    rng_seed = int(arm3_cfg.get("rng_seed", DEFAULT_RNG_SEED))

    print(
        f"[Arm-3 Phase 5] Bootstrap parameters: "
        f"coverage_levels={COVERAGE_LEVELS}, n_reps={n_reps}, rng_seed={rng_seed}"
    )

    # --- Build full-coverage density tensors for anchor pairs ---
    with _timed_section(timing_seconds, "phase5_6.assemble_full_density_tensors_seconds"):
        A_full, B_full = assemble_density_tensors(
            roi_density_vectors, pair_meta_anchor_runtime, k_full
        )
    n_anchor_pairs = len(pair_meta_anchor_runtime)
    print(
        f"[Arm-3 Phase 5/6] Anchor pairs: {n_anchor_pairs}, "
        f"tensor shape: {A_full.shape}"
    )

    # --- Build full-coverage COUNT tensors for support mask construction ---
    # Count vectors are the sum of count_k* columns across all blocks per ROI.
    # These are NOT used as UOT inputs; they are used solely for support masks.
    count_col_names = [f"count_k{k}" for k in range(k_full)]
    roi_count_vectors: dict[str, np.ndarray] = {
        roi_id: df[count_col_names].sum(axis=0).to_numpy(dtype=float)
        for roi_id, df in roi_block_summary.items()
    }
    with _timed_section(timing_seconds, "phase5_6.assemble_support_count_tensors_seconds"):
        A_count, B_count = assemble_density_tensors(
            roi_count_vectors, pair_meta_anchor_runtime, k_full
        )

    # --- Freeze support masks from COUNT tensors (n_min_proto is a COUNT threshold) ---
    with _timed_section(timing_seconds, "phase5_6.freeze_support_masks_seconds"):
        support_masks = freeze_support_masks(
            A_count=A_count,
            B_count=B_count,
            n_min_proto=uot_cfg.n_min_proto,
            k_full=k_full,
        )

    # Self-check: support mask shape must match anchor pair count × K_FULL
    assert support_masks.shape == (n_anchor_pairs, k_full), (
        f"Support mask shape mismatch: got {support_masks.shape}, "
        f"expected ({n_anchor_pairs}, {k_full})"
    )
    print(
        f"[Arm-3 Phase 5/6] Support masks frozen from COUNT tensors; "
        f"shape={support_masks.shape}, "
        f"mean_supported_fraction={support_masks.mean():.3f}"
    )

    # --- Broadcast frozen lambda and tau for anchor pairs ---
    with _timed_section(timing_seconds, "phase5_6.broadcast_calibration_seconds"):
        lambda_arr = broadcast_frozen_lambda(pair_meta_anchor_runtime, lambda_dens)
        tau_arr = broadcast_frozen_tau(pair_meta_anchor_runtime, tau_by_compartment)

    # Lambda/tau broadcast integrity check
    if np.any(~np.isfinite(lambda_arr)):
        raise RuntimeError("broadcast_frozen_lambda: produced non-finite lambda values")
    if np.any(~np.isfinite(tau_arr)):
        raise RuntimeError("broadcast_frozen_tau: produced non-finite tau values")
    print(
        f"[Arm-3 Phase 5/6] lambda broadcast: min={lambda_arr.min():.4g}, "
        f"max={lambda_arr.max():.4g}; "
        f"tau broadcast: min={tau_arr.min():.4g}, max={tau_arr.max():.4g}"
    )

    # -----------------------------------------------------------------------
    # Phase 6A: Full-coverage reference UOT + Balanced OT + metrics
    # Patch 1: use run_uot_batch_with_events (single solver call) to obtain
    # both the scalar DataFrame and prototype-level event tensors.
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 6] Running full-coverage reference UOT (with prototype events)")
    with _timed_section(timing_seconds, "phase6.full_coverage_uot_with_events_seconds"):
        df_full, T_k_full, B_k_full, D_k_full = run_uot_batch_with_events(
            A=A_full,
            B=B_full,
            lambda_pl=lambda_arr,
            kernels=kernels,
            uot_cfg=uot_cfg,
            pair_meta=pair_meta_anchor_runtime,
            tau_external=tau_arr,
            external_support_mask=support_masks,
        )
        df_full = compute_arm3_density_metrics(df_full, A_full, B_full)

    # Shape-only Balanced OT on full-coverage: L1-normalise first
    A_shape_full = A_full / (A_full.sum(axis=1, keepdims=True) + DENSITY_EPS)
    B_shape_full = B_full / (B_full.sum(axis=1, keepdims=True) + DENSITY_EPS)

    # Self-check: Balanced OT input must be L1-normalised (sum <= 1 + eps)
    assert np.all(A_shape_full.sum(axis=1) <= 1.0 + 1e-9), (
        "Balanced OT self-check failed: A_shape_full rows are not L1-normalised"
    )
    with _timed_section(timing_seconds, "phase6.full_coverage_balanced_ot_seconds"):
        bot_costs_full = run_balanced_ot_batch(
            A_shape_full, B_shape_full, scaled_cost_matrix, n_min_proto=0.0
        )

    # Rename columns for Phase 6 output schema
    with _timed_section(timing_seconds, "phase6.full_coverage_schema_projection_seconds"):
        df_full_out = _select_full_coverage_columns(df_full, lambda_arr, tau_arr)

    # -----------------------------------------------------------------------
    # Phase 5: Bootstrap per coverage level + Phase 6 inference
    # -----------------------------------------------------------------------
    all_bootstrap_results: list[pd.DataFrame] = []
    all_bot_rows: list[pd.DataFrame] = []
    all_pseudo_audit: list[pd.DataFrame] = []
    # Patch 1: accumulate prototype event rows across all replicates and coverages
    all_proto_boot_rows: list[pd.DataFrame] = []

    for coverage in COVERAGE_LEVELS:
        coverage_key = _coverage_key(coverage)
        print(
            f"[Arm-3 Phase 5] Bootstrap coverage={coverage:.0%} "
            f"({n_reps} replicates × {n_anchor_pairs} pairs)"
        )

        with _timed_section(timing_seconds, f"phase5.{coverage_key}.bootstrap_pass_seconds"):
            A_reps, B_reps, pseudo_meta = run_bootstrap_pass(
                roi_block_summary=roi_block_summary,
                pair_meta=pair_meta_anchor,
                coverage=coverage,
                n_reps=n_reps,
                k_full=k_full,
                rng_seed=rng_seed,
            )

        # Self-check: sampled block counts must match n_sampled formula
        # (verified implicitly by the pseudo_meta n_blocks_sampled_* columns)
        all_pseudo_audit.append(pseudo_meta)

        # Phase 6B: UOT + metrics for each replicate
        # Patch 1: use run_uot_batch_with_events (single solver call per replicate)
        for rep_idx in range(n_reps):
            A_rep = A_reps[rep_idx]   # (N_anchor, K)
            B_rep = B_reps[rep_idx]   # (N_anchor, K)

            with _timed_section(timing_seconds, f"phase6.{coverage_key}.uot_with_events_seconds"):
                df_rep, T_k_rep, B_k_rep, D_k_rep = run_uot_batch_with_events(
                    A=A_rep,
                    B=B_rep,
                    lambda_pl=lambda_arr,
                    kernels=kernels,
                    uot_cfg=uot_cfg,
                    pair_meta=pair_meta_anchor_runtime,
                    tau_external=tau_arr,
                    external_support_mask=support_masks,
                )
                df_rep = compute_arm3_density_metrics(df_rep, A_rep, B_rep)

            # Prototype events for this replicate (long-format, one row per pair×prototype)
            with _timed_section(timing_seconds, f"phase6.{coverage_key}.prototype_event_rows_seconds"):
                all_proto_boot_rows.append(
                    _build_prototype_events_df(
                        pair_meta=pair_meta_anchor_runtime,
                        T_k=T_k_rep, B_k=B_k_rep, D_k=D_k_rep,
                        proto_labels=proto_labels,
                        coverage=coverage,
                        replicate_id=rep_idx,
                    )
                )

            # Floor-dominated flags using frozen support masks
            floor_dom = compute_floor_dominated_flags(
                A_dens=A_rep,
                support_masks=support_masks,
                eta_floor=uot_cfg.eta_floor,
            )
            df_rep["floor_dominated"] = floor_dom
            df_rep["replicate_id"] = rep_idx
            df_rep["coverage"] = coverage
            df_rep["lambda_dens"] = lambda_arr
            df_rep["tau"] = tau_arr

            all_bootstrap_results.append(df_rep)

            # Balanced OT comparator: L1-normalise before passing
            A_shape_rep = A_rep / (A_rep.sum(axis=1, keepdims=True) + DENSITY_EPS)
            B_shape_rep = B_rep / (B_rep.sum(axis=1, keepdims=True) + DENSITY_EPS)

            # Self-check: L1-normalised input
            assert np.all(A_shape_rep.sum(axis=1) <= 1.0 + 1e-9), (
                f"Balanced OT self-check failed at coverage={coverage}, rep={rep_idx}: "
                "A_shape_rep rows are not L1-normalised"
            )

            with _timed_section(timing_seconds, f"phase6.{coverage_key}.balanced_ot_seconds"):
                bot_costs = run_balanced_ot_batch(
                    A_shape_rep, B_shape_rep, scaled_cost_matrix, n_min_proto=0.0
                )
            bot_row = pair_meta_anchor_runtime[
                ["pair_id", "patient_id", "pair_type", "pair_family",
                 "compartment_a", "compartment_b"]
            ].copy()
            bot_row["replicate_id"] = rep_idx
            bot_row["coverage"] = coverage
            bot_row["balanced_ot_cost"] = bot_costs
            bot_row["comparator_type"] = "shape_only_bootstrap"
            all_bot_rows.append(bot_row)

        print(
            f"[Arm-3 Phase 6] Coverage={coverage:.0%}: "
            f"{n_reps} replicates processed"
        )

    # Full-coverage Balanced OT reference row.
    # coverage=1.0 is stored only as a baseline marker for audit compatibility;
    # it is not part of the reduced bootstrap loop.
    bot_row_full = pair_meta_anchor_runtime[
        ["pair_id", "patient_id", "pair_type", "pair_family",
         "compartment_a", "compartment_b"]
    ].copy()
    bot_row_full["replicate_id"] = -1
    bot_row_full["coverage"] = 1.0
    bot_row_full["balanced_ot_cost"] = bot_costs_full
    bot_row_full["comparator_type"] = "shape_only_full_coverage"
    all_bot_rows.append(bot_row_full)

    # -----------------------------------------------------------------------
    # Concat and build audit tables
    # -----------------------------------------------------------------------
    with _timed_section(timing_seconds, "phase6.concat_bootstrap_tables_seconds"):
        df_bootstrap_all = pd.concat(all_bootstrap_results, ignore_index=True)
        df_bot_all = pd.concat(all_bot_rows, ignore_index=True)
        df_pseudo_audit_all = pd.concat(all_pseudo_audit, ignore_index=True)

    # Support mask audit table
    with _timed_section(timing_seconds, "phase6.build_support_mask_audit_seconds"):
        support_mask_audit = _build_support_mask_audit(pair_meta_anchor_runtime, support_masks, k_full)

    # Anchor metric summary
    # Self-check: must contain only TC->IM and TC->PT
    anchor_types_present = set(df_bootstrap_all["pair_type"].astype(str).unique())
    unexpected_anchor = anchor_types_present - set(ARM3_ANCHOR_DIRECTIONS)
    if unexpected_anchor:
        raise RuntimeError(
            f"Anchor summary self-check failed: unexpected pair_type values "
            f"{sorted(unexpected_anchor)} in bootstrap results"
        )
    with _timed_section(timing_seconds, "phase6.build_metric_summary_seconds"):
        df_metric_summary = _build_metric_summary_anchor(df_bootstrap_all, df_full_out)

    # -----------------------------------------------------------------------
    # Write Phase 5/6 outputs
    # -----------------------------------------------------------------------
    print(f"[Arm-3 Phase 5/6] Writing outputs to {result_root}")
    with _timed_section(timing_seconds, "phase5_6.write_outputs_seconds"):
        _write_phase5_6_outputs(
            result_root=result_root,
            df_pseudo_audit=df_pseudo_audit_all,
            df_full_coverage=df_full_out,
            df_bootstrap=df_bootstrap_all,
            df_bot=df_bot_all,
            df_support_mask_audit=support_mask_audit,
            df_metric_summary=df_metric_summary,
        )
    print("[Arm-3 Phase 5/6] Outputs written.")

    # -----------------------------------------------------------------------
    # Patch 1: Prototype event parquets (Phase 6 supplementary outputs)
    # -----------------------------------------------------------------------

    # Self-check 1: Prototype label integrity
    for k in range(k_full):
        lbl = proto_labels[k]
        if not isinstance(lbl, str) or not lbl:
            raise RuntimeError(
                f"Prototype label integrity check failed: prototype_k={k} has "
                f"invalid label {lbl!r}"
            )
    print(
        f"[Arm-3 Phase 6] Prototype label integrity check passed "
        f"({k_full} labels; first 5: {proto_labels[:5]})"
    )

    # Build full-coverage prototype event table (sentinel replicate_id=-1)
    with _timed_section(timing_seconds, "phase6.full_coverage_prototype_event_rows_seconds"):
        df_proto_events_full = _build_prototype_events_df(
            pair_meta=pair_meta_anchor_runtime,
            T_k=T_k_full, B_k=B_k_full, D_k=D_k_full,
            proto_labels=proto_labels,
            coverage=1.0,
            replicate_id=-1,
        )

    # Self-check 2: Event mass nonnegativity on ok rows (full-coverage)
    ok_mask_full = df_full["uot_status"] == "ok"
    ok_idx_full = np.flatnonzero(ok_mask_full.to_numpy())
    if ok_idx_full.size > 0:
        for _arr_name, _arr in [
            ("T_k_full", T_k_full), ("B_k_full", B_k_full), ("D_k_full", D_k_full)
        ]:
            _ok_vals = _arr[ok_idx_full]
            _n_neg = int(np.sum(_ok_vals < 0))
            _n_nan = int(np.sum(np.isnan(_ok_vals)))
            if _n_neg > 0:
                raise RuntimeError(
                    f"Event mass nonnegativity check failed: {_arr_name} has "
                    f"{_n_neg} negative values on ok rows"
                )
            if _n_nan > 0:
                raise RuntimeError(
                    f"Event mass NaN check failed: {_arr_name} has "
                    f"{_n_nan} NaN values on ok rows"
                )
    print(
        f"[Arm-3 Phase 6] Event mass nonnegativity check passed "
        f"(full-coverage, {ok_idx_full.size} ok rows)"
    )

    # Concatenate bootstrap prototype events
    with _timed_section(timing_seconds, "phase6.concat_prototype_event_tables_seconds"):
        df_proto_events_bootstrap = (
            pd.concat(all_proto_boot_rows, ignore_index=True)
            if all_proto_boot_rows
            else pd.DataFrame(
                columns=[
                    "pair_id", "patient_id", "pair_type", "pair_family",
                    "coverage", "replicate_id", "prototype_k", "prototype_label",
                    "T_mass", "B_mass", "D_mass",
                ]
            )
        )

    # Write prototype event parquets
    _proto_full_path = result_root / _PROTO_EVENTS_FULL_FILENAME
    _proto_boot_path = result_root / _PROTO_EVENTS_BOOTSTRAP_FILENAME
    with _timed_section(timing_seconds, "phase6.write_prototype_event_outputs_seconds"):
        df_proto_events_full.to_parquet(_proto_full_path, index=False)
        df_proto_events_bootstrap.to_parquet(_proto_boot_path, index=False)
    print(
        f"[Arm-3 Phase 6] Prototype event parquets written: "
        f"{len(df_proto_events_full)} full-coverage rows, "
        f"{len(df_proto_events_bootstrap)} bootstrap rows."
    )

    # -----------------------------------------------------------------------
    # Phase 7 — Continuous retention summary
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 7] Computing continuous retention summary...")

    # Filter to anchor directions only (both frames are already anchor-only per
    # Phase 6 self-check, but we guard explicitly here for safety).
    anchor_set = set(ARM3_ANCHOR_DIRECTIONS)
    df_full_anchor = df_full_out[
        df_full_out["pair_type"].astype(str).isin(anchor_set)
    ].copy()
    df_boot_anchor = df_bootstrap_all[
        df_bootstrap_all["pair_type"].astype(str).isin(anchor_set)
    ].copy()

    _monitored_quantities = ["U_abs_dens", "Q_src_dens"]

    with _timed_section(timing_seconds, "phase7.compute_degradation_summary_seconds"):
        df_degradation = compute_degradation_summary(
            df_full_cov=df_full_anchor,
            df_reduced=df_boot_anchor,
            monitored_quantities=_monitored_quantities,
            pair_id_col="pair_id",
            patient_id_col="patient_id",
        )

    # Write Phase 7 outputs
    _deg_parquet = result_root / f"{_PHASE7_DEGRADATION_FILENAME}.parquet"
    _deg_csv = result_root / f"{_PHASE7_DEGRADATION_FILENAME}.csv"
    with _timed_section(timing_seconds, "phase7.write_degradation_outputs_seconds"):
        df_degradation.to_parquet(_deg_parquet, index=False)
        df_degradation.to_csv(_deg_csv, index=False)
    print(
        f"[Arm-3 Phase 7] Degradation summary written "
        f"({len(df_degradation)} rows, {df_degradation['patient_id'].nunique()} patients)."
    )

    # -------------------------------------------------------------------
    # Patch 2A: Contrast-based sign-consistency summary (Phase 7)
    # -------------------------------------------------------------------
    # Self-check 3: Contrast completeness — verified inside
    # compute_contrast_degradation_summary (drops patients missing either
    # TC->IM or TC->PT and warns; never silently fabricates a contrast).
    #
    # Self-check 4: Sign-consistency audit — n_zero_reference_sign is
    # counted, printed, and stored in the output DataFrame per
    # (contrast_name, coverage) group.
    print("[Arm-3 Phase 7] Computing contrast-based sign-consistency summary...")
    with _timed_section(timing_seconds, "phase7.compute_contrast_summary_seconds"):
        df_contrast = compute_contrast_degradation_summary(
            df_full_cov=df_full_anchor,
            df_reduced=df_boot_anchor,
            patient_id_col="patient_id",
        )

    # Audit report: print zero-reference counts per contrast × coverage
    if not df_contrast.empty:
        audit_grp = (
            df_contrast
            .drop_duplicates(subset=["coverage", "contrast_name"])
            [["coverage", "contrast_name", "n_evaluable", "n_zero_reference_sign"]]
        )
        for _, row in audit_grp.iterrows():
            print(
                f"[Phase 7 contrast audit] contrast={row['contrast_name']}, "
                f"coverage={row['coverage']:.0%}: "
                f"n_evaluable={int(row['n_evaluable'])}, "
                f"n_zero_reference_sign={int(row['n_zero_reference_sign'])} "
                f"(non-evaluable patients excluded from pi_c)"
            )

    # Write contrast summary
    _con_parquet = result_root / f"{_PHASE7_CONTRAST_FILENAME}.parquet"
    _con_csv = result_root / f"{_PHASE7_CONTRAST_FILENAME}.csv"
    with _timed_section(timing_seconds, "phase7.write_contrast_outputs_seconds"):
        df_contrast.to_parquet(_con_parquet, index=False)
        df_contrast.to_csv(_con_csv, index=False)
    print(
        f"[Arm-3 Phase 7] Contrast summary written "
        f"({len(df_contrast)} rows)."
    )

    # -------------------------------------------------------------------
    # Patch 2B: Prototype contrast prep table (Phase 8 data path)
    # -------------------------------------------------------------------
    print("[Arm-3 Phase 7/8 prep] Building prototype contrast table for Phase 8...")
    with _timed_section(timing_seconds, "phase7_8.build_prototype_contrast_seconds"):
        df_proto_contrast = build_prototype_contrast_table(
            df_proto_events_full=df_proto_events_full,
            df_proto_events_bootstrap=df_proto_events_bootstrap,
        )
    _proto_con_path = result_root / f"{_PHASE8_PROTO_CONTRAST_FILENAME}.parquet"
    with _timed_section(timing_seconds, "phase7_8.write_prototype_contrast_seconds"):
        df_proto_contrast.to_parquet(_proto_con_path, index=False)
    print(
        f"[Arm-3 Phase 8 prep] Prototype contrast table written "
        f"({len(df_proto_contrast)} rows, "
        f"{df_proto_contrast['prototype_k'].nunique() if not df_proto_contrast.empty else 0} "
        f"unique prototypes)."
    )

    # -----------------------------------------------------------------------
    # Phase 8 — Prototype summary table and descriptive memo
    # -----------------------------------------------------------------------
    print("[Arm-3 Phase 8] Building prototype summary table and descriptive memo...")
    with _timed_section(timing_seconds, "phase8.finalize_outputs_seconds"):
        df_phase8, _ = finalize_arm3_phase8(
            result_root=result_root,
            df_degradation=df_degradation,
            df_contrast=df_contrast,
            df_proto_contrast=df_proto_contrast,
        )
    print(
        f"[Arm-3 Phase 8] Outputs written "
        f"({len(df_phase8)} prototype-by-coverage rows)."
    )

    timing_seconds["total_runtime_seconds"] = perf_counter() - total_started
    _write_timing_summary(
        result_root=result_root,
        timing_seconds=timing_seconds,
        n_anchor_pairs=n_anchor_pairs,
        n_reps=n_reps,
        coverage_levels=COVERAGE_LEVELS,
    )
    return df_full_out


def finalize_arm3_phase8(
    result_root: Path | str,
    df_degradation: pd.DataFrame | None = None,
    df_contrast: pd.DataFrame | None = None,
    df_proto_contrast: pd.DataFrame | None = None,
    calibration_record: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Finalize Phase 8 from in-memory tables or by loading existing artifacts.

    This function is reusable for:
    - the live `run_arm3(...)` path after Phase 7 / prep tables are built, and
    - a Phase-8-only closure pass on an existing result root.
    """
    result_root = Path(result_root)

    if df_degradation is None:
        df_degradation = pd.read_parquet(
            result_root / f"{_PHASE7_DEGRADATION_FILENAME}.parquet"
        )
    else:
        df_degradation = df_degradation.copy()

    if df_contrast is None:
        df_contrast = pd.read_parquet(
            result_root / f"{_PHASE7_CONTRAST_FILENAME}.parquet"
        )
    else:
        df_contrast = df_contrast.copy()

    if df_proto_contrast is None:
        df_proto_contrast = pd.read_parquet(
            result_root / f"{_PHASE8_PROTO_CONTRAST_FILENAME}.parquet"
        )
    else:
        df_proto_contrast = df_proto_contrast.copy()

    if calibration_record is None:
        with open(result_root / _CALIBRATION_RECORD_FILENAME, "r", encoding="utf-8") as fh:
            calibration_record = json.load(fh)
    else:
        calibration_record = dict(calibration_record)

    df_prototype_stability = build_prototype_stability_table(df_proto_contrast)
    memo_text = build_arm3_memo(
        result_root=result_root,
        df_degradation=df_degradation,
        df_contrast=df_contrast,
        df_prototype_stability=df_prototype_stability,
        calibration_record=calibration_record,
    )
    write_phase8_outputs(
        result_root=result_root,
        df_prototype_stability=df_prototype_stability,
        memo_text=memo_text,
    )
    return df_prototype_stability, memo_text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_arm3_stage0_inputs(
    stage0_path: Path,
    k_full: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, float, list[str]]:
    """
    Load the minimal set of Stage-0 fields needed for Phase 0–4.

    Reads directly via h5py to avoid loading the full artifact into memory.
    Follows the HDF5 categorical layout style from arm2/analysis_io.py.

    Returns
    -------
    spatial_xy : np.ndarray, shape (N_cells, 2)
    roi_ids : np.ndarray, shape (N_cells,), dtype object (str)
    proto_ids : np.ndarray, shape (N_cells,), dtype int
    roi_table : pd.DataFrame
        One row per unique ROI with columns: roi_id, patient_id, compartment.
    cost_matrix : np.ndarray, shape (k_full, k_full)
        Frozen cost matrix from uns/cost_matrix; used for kernel precomputation.
    s_C : float
        Cost-scale divisor from uns/s_C; used for kernel precomputation.
    proto_labels : list[str]
        Prototype name strings of length k_full, read from obs/proto_id/categories.
        Falls back to ["proto_0", ..., "proto_{k_full-1}"] if the path is absent.
    """
    with h5py.File(stage0_path, "r") as f:
        # ---- spatial coordinates ----------------------------------------
        if "obsm/spatial" in f:
            spatial_xy = np.asarray(f["obsm/spatial"], dtype=float)
        elif "obsm/X_spatial" in f:
            spatial_xy = np.asarray(f["obsm/X_spatial"], dtype=float)
        else:
            raise KeyError(
                "Stage-0 artifact is missing obsm/spatial (also tried obsm/X_spatial). "
                "Cannot build Arm-3 grid partition without spatial coordinates."
            )
        if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
            raise ValueError(
                f"obsm/spatial must have shape (N, 2), got {spatial_xy.shape}"
            )

        # ---- prototype assignments ----------------------------------------
        proto_ids = np.asarray(f["obs/proto_id"], dtype=int)

        # ---- roi_id (categorical) ----------------------------------------
        roi_codes = np.asarray(f["obs/roi_id/codes"], dtype=int)
        roi_cats = _decode_h5_strings(f["obs/roi_id/categories"][()])

        # ---- patient_id (categorical) ------------------------------------
        patient_codes = np.asarray(f["obs/patient_id/codes"], dtype=int)
        patient_cats = _decode_h5_strings(f["obs/patient_id/categories"][()])

        # ---- compartment (categorical) -----------------------------------
        compartment_codes = np.asarray(f["obs/compartment/codes"], dtype=int)
        compartment_cats = _decode_h5_strings(f["obs/compartment/categories"][()])

        # ---- cost_matrix and s_C (needed for Phase 4 kernels) -----------
        cost_matrix_shape = f["uns/cost_matrix"].shape
        if cost_matrix_shape != (k_full, k_full):
            raise ValueError(
                f"Stage-0 cost_matrix shape {cost_matrix_shape} does not match "
                f"config k_full={k_full}; expected ({k_full}, {k_full})"
            )
        cost_matrix = np.asarray(f["uns/cost_matrix"], dtype=float)

        s_C_raw = f["uns/s_C"]
        s_C = float(np.asarray(s_C_raw).flat[0])
        if not np.isfinite(s_C) or s_C <= 0.0:
            raise ValueError(
                f"Stage-0 uns/s_C must be finite and strictly positive, got {s_C}"
            )

        # ---- prototype label strings (Patch 1) ---------------------------
        # Read obs/proto_id/categories if present; fall back to generic names.
        try:
            proto_cats_raw = f["obs/proto_id/categories"][()]
            proto_labels: list[str] = list(_decode_h5_strings(proto_cats_raw))
        except (KeyError, OSError):
            proto_labels = [f"proto_{k}" for k in range(k_full)]

    n_cells = spatial_xy.shape[0]

    # Validate index alignment across all cell-level arrays
    if proto_ids.shape[0] != n_cells:
        raise ValueError("proto_id length does not match spatial_xy row count")
    if roi_codes.shape[0] != n_cells:
        raise ValueError("roi_id/codes length does not match spatial_xy row count")

    # Decode per-cell categorical arrays to string arrays
    roi_ids = np.array([
        str(roi_cats[c]) if 0 <= c < len(roi_cats) else "__invalid__"
        for c in roi_codes
    ], dtype=object)
    patient_ids_per_cell = np.array([
        str(patient_cats[c]) if 0 <= c < len(patient_cats) else "__invalid__"
        for c in patient_codes
    ], dtype=object)
    compartments_per_cell = np.array([
        str(compartment_cats[c]) if 0 <= c < len(compartment_cats) else "__invalid__"
        for c in compartment_codes
    ], dtype=object)

    # Validate prototype range
    valid_proto = (proto_ids >= 0) & (proto_ids < k_full)
    if not valid_proto.any():
        raise ValueError("Stage-0 artifact contains no valid prototype assignments")

    # Build ROI-level table: one row per unique ROI (deduped from per-cell data)
    roi_df = pd.DataFrame({
        "roi_id": roi_ids,
        "patient_id": patient_ids_per_cell,
        "compartment": compartments_per_cell,
    })
    valid_mask = (
        (roi_ids != "__invalid__")
        & (patient_ids_per_cell != "__invalid__")
        & (compartments_per_cell != "__invalid__")
        & valid_proto
    )
    roi_table = (
        roi_df[valid_mask]
        .drop_duplicates(subset=["roi_id"])
        .sort_values("roi_id")
        .reset_index(drop=True)
    )
    roi_table["roi_id"] = roi_table["roi_id"].astype(str)
    roi_table["patient_id"] = roi_table["patient_id"].astype(str)
    roi_table["compartment"] = roi_table["compartment"].astype(str)

    if roi_table.empty:
        raise ValueError("No valid ROIs found in Stage-0 artifact")

    # Validate proto_labels length matches k_full
    if len(proto_labels) != k_full:
        proto_labels = [f"proto_{k}" for k in range(k_full)]

    return spatial_xy, roi_ids, proto_ids, roi_table, cost_matrix, s_C, proto_labels


def _decode_h5_strings(values: np.ndarray) -> np.ndarray:
    """Decode byte-heavy string arrays from h5ad categorical storage."""
    return np.asarray(
        [
            v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in values
        ],
        dtype=object,
    )


def _build_pair_universe(
    roi_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full ordered pair universe and the anchor-direction subset.

    Phase 3 of the Arm-3 pipeline.

    Reuses ORDERED_PAIR_SPECS from arm2_spatial_gradient to enumerate all six
    ordered directions, then filters to ARM3_ANCHOR_DIRECTIONS for the primary
    anchor subset. This follows the same generation logic as
    arm2_spatial_gradient.generate_cross_compartment_pairs but operates on the
    roi_table DataFrame from the HDF5 read (no AnnData required).

    Anchor directions (TC->IM, TC->PT) are the only ones that enter primary
    Arm-3 degradation summaries. Reverse and exploratory directions remain in
    pair_meta_full for audit use.

    Parameters
    ----------
    roi_table : pd.DataFrame
        One row per unique ROI with columns: roi_id, patient_id, compartment.

    Returns
    -------
    pair_meta_full : pd.DataFrame
        All six ordered directions with pair_id, patient_id, compartment_a,
        compartment_b, pair_family, pair_type, roi_a, roi_b.
    pair_meta_anchor : pd.DataFrame
        Subset where pair_type in ARM3_ANCHOR_DIRECTIONS.
    """
    records: list[dict[str, Any]] = []

    for patient_id, patient_df in roi_table.groupby("patient_id", sort=True):
        roi_by_compartment: dict[str, list[str]] = {
            compartment: sorted(group["roi_id"].astype(str).tolist())
            for compartment, group in patient_df.groupby("compartment", sort=True)
        }

        for pair_type, pair_family, source_comp, target_comp in ORDERED_PAIR_SPECS:
            source_rois = roi_by_compartment.get(source_comp, [])
            target_rois = roi_by_compartment.get(target_comp, [])
            if not source_rois or not target_rois:
                continue

            for roi_a in source_rois:
                for roi_b in target_rois:
                    pair_id = (
                        f"{ARM3_NAME}::{pair_type}::{patient_id}::{roi_a}::{roi_b}"
                    )
                    records.append(
                        {
                            "pair_id": pair_id,
                            "patient_group_id": pair_id,
                            "patient_id": str(patient_id),
                            "compartment": source_comp,
                            "patient_id_a": str(patient_id),
                            "patient_id_b": str(patient_id),
                            "compartment_a": source_comp,
                            "compartment_b": target_comp,
                            "same_patient": True,
                            "same_compartment": False,
                            "pair_type": pair_type,
                            "pair_family": pair_family,
                            "roi_a": roi_a,
                            "roi_b": roi_b,
                        }
                    )

    pair_meta_full = pd.DataFrame.from_records(records)

    if pair_meta_full.empty:
        raise ValueError(
            "Arm-3 pair universe is empty — no eligible within-patient "
            "cross-compartment ROI pairs found in roi_table"
        )

    # Anchor subset: TC->IM and TC->PT only
    anchor_mask = pair_meta_full["pair_type"].isin(ARM3_ANCHOR_DIRECTIONS)
    pair_meta_anchor = pair_meta_full[anchor_mask].reset_index(drop=True)

    if pair_meta_anchor.empty:
        raise ValueError(
            f"Anchor-direction subset is empty — no rows with pair_type in "
            f"{ARM3_ANCHOR_DIRECTIONS} found in the full pair universe"
        )

    return pair_meta_full.reset_index(drop=True), pair_meta_anchor


# ---------------------------------------------------------------------------
# Phase 4 integrity self-checks
# ---------------------------------------------------------------------------


def _check_block_universe_integrity(
    roi_block_universe: dict[str, list[str]],
) -> None:
    """
    Verify that each ROI's block count matches its snapped-envelope grid dimensions.

    For each ROI, parses block_id values of the form "{roi_id}::{col}::{row}",
    infers the grid dimensions as (max_col - min_col + 1) x (max_row - min_row + 1),
    and asserts the full Cartesian product is present. Uses range-safe arithmetic
    so that non-zero-based grids are handled correctly. Does not rely only on
    occupied blocks.

    Raises RuntimeError on any integrity violation.
    """
    for roi_id, block_ids in roi_block_universe.items():
        if not block_ids:
            raise RuntimeError(
                f"ROI {roi_id!r}: block universe is empty"
            )

        cols: list[int] = []
        rows: list[int] = []
        for bid in block_ids:
            parts = bid.split("::")
            if len(parts) != 3:
                raise RuntimeError(
                    f"ROI {roi_id!r}: malformed block_id {bid!r}; "
                    "expected format '{roi_id}::{col}::{row}'"
                )
            try:
                cols.append(int(parts[1]))
                rows.append(int(parts[2]))
            except ValueError as exc:
                raise RuntimeError(
                    f"ROI {roi_id!r}: non-integer grid index in block_id {bid!r}"
                ) from exc

        n_cols = max(cols) - min(cols) + 1
        n_rows = max(rows) - min(rows) + 1
        expected = n_cols * n_rows
        actual = len(block_ids)

        if actual != expected:
            raise RuntimeError(
                f"ROI {roi_id!r}: block universe has {actual} entries but "
                f"grid dimensions {n_cols}×{n_rows} imply {expected} blocks — "
                "zero-cell blocks may have been lost or duplicated"
            )

        # Verify all (col, row) pairs are distinct
        coord_set = set(zip(cols, rows))
        if len(coord_set) != actual:
            raise RuntimeError(
                f"ROI {roi_id!r}: duplicate (col, row) pairs in block universe "
                f"({actual - len(coord_set)} duplicates)"
            )


def _check_pair_universe_integrity(
    pair_meta_full: pd.DataFrame,
    pair_meta_anchor: pd.DataFrame,
    roi_density_vectors: dict[str, np.ndarray],
) -> None:
    """
    Verify the pair universe and anchor subset are internally consistent.

    Checks:
    1. All six ordered directions are present in pair_meta_full.
    2. Anchor subset contains only TC->IM and TC->PT (ARM3_ANCHOR_DIRECTIONS).
    3. Every roi_a / roi_b referenced in pair_meta_full exists in roi_density_vectors.

    Raises RuntimeError on any integrity violation.
    """
    # Check 1: all six ordered directions are present
    expected_directions = {spec[0] for spec in ORDERED_PAIR_SPECS}
    actual_directions = set(pair_meta_full["pair_type"].astype(str).unique())
    missing_directions = expected_directions - actual_directions
    if missing_directions:
        raise RuntimeError(
            f"Pair universe is missing ordered directions: {sorted(missing_directions)}"
        )

    # Check 2: anchor subset contains only the two locked directions
    anchor_types = set(pair_meta_anchor["pair_type"].astype(str).unique())
    unexpected = anchor_types - set(ARM3_ANCHOR_DIRECTIONS)
    if unexpected:
        raise RuntimeError(
            f"Anchor subset contains unexpected pair_type values: {sorted(unexpected)}"
        )
    missing_anchor = set(ARM3_ANCHOR_DIRECTIONS) - anchor_types
    if missing_anchor:
        raise RuntimeError(
            f"Anchor subset is missing expected anchor directions: {sorted(missing_anchor)}"
        )

    # Check 3: every roi_a / roi_b in pair_meta_full exists in density reference
    all_pair_rois = (
        set(pair_meta_full["roi_a"].astype(str))
        | set(pair_meta_full["roi_b"].astype(str))
    )
    missing_rois = sorted(all_pair_rois - set(roi_density_vectors))
    if missing_rois:
        raise RuntimeError(
            f"Pair universe references {len(missing_rois)} ROI(s) absent from "
            f"the full-coverage density reference: "
            f"{missing_rois[:5]}{'...' if len(missing_rois) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# UOT config builder
# ---------------------------------------------------------------------------


def _build_uot_cfg(config: dict[str, Any]) -> UOTSolveConfig:
    """
    Build UOTSolveConfig for Phase 4 from the task config dict.

    Uses config section 'uot_params' with defaults matching Arm-2 config.yaml.
    tau_mode is set to 'pi_weighted_q25' (the UOTSolveConfig default).
    """
    uot_params = config.get("uot_params", {})
    eps_schedule = tuple(
        float(v) for v in uot_params.get("eps_schedule", [1.0, 0.5, 0.1])
    )
    max_iter = int(uot_params.get("max_iter", 2000))
    tol = float(uot_params.get("tol", 1e-6))
    eta_floor = float(uot_params.get("eta_floor", 1e-12))
    n_min_proto = float(uot_params.get("n_min_proto", 0.0))
    return UOTSolveConfig(
        eps_schedule=eps_schedule,
        max_iter=max_iter,
        tol=tol,
        eta_floor=eta_floor,
        n_min_proto=n_min_proto,
    )


# ---------------------------------------------------------------------------
# Phase 0–3 output writer (unchanged from Phase 0–3 tranche)
# ---------------------------------------------------------------------------


def _write_phase0_3_outputs(
    result_root: Path,
    stage0_path: Path,
    block_size_units: float,
    block_area_mm2: float,
    spatial_xy: np.ndarray,
    block_frame: pd.DataFrame,
    roi_block_universe: dict[str, list[str]],
    roi_block_summary: dict[str, pd.DataFrame],
    roi_density_vectors: dict[str, np.ndarray],
    roi_total_areas: dict[str, float],
    pair_meta_full: pd.DataFrame,
    pair_meta_anchor: pd.DataFrame,
) -> None:
    """Write the Phase 0–3 validation package to result_root."""

    # ---- Phase 0: manifest -----------------------------------------------
    n_total_blocks = sum(len(v) for v in roi_block_universe.values())
    manifest = {
        "arm3_name": ARM3_NAME,
        "k_full": K_FULL,
        "block_size_units": block_size_units,
        "coord_to_mm2": COORD_TO_MM2,
        "block_area_mm2": block_area_mm2,
        "anchor_directions": list(ARM3_ANCHOR_DIRECTIONS),
        "stage0_path": str(stage0_path),
        "n_cells": int(spatial_xy.shape[0]),
        "n_rois": len(roi_block_universe),
        "n_total_blocks": n_total_blocks,
        "spatial_x_range": [float(spatial_xy[:, 0].min()), float(spatial_xy[:, 0].max())],
        "spatial_y_range": [float(spatial_xy[:, 1].min()), float(spatial_xy[:, 1].max())],
        "n_pairs_full": len(pair_meta_full),
        "n_pairs_anchor": len(pair_meta_anchor),
        "run_utc": datetime.now(tz=timezone.utc).isoformat(),
        "phase_implemented": "0-8",
        "phase_4_plus": "IMPLEMENTED",
    }
    with open(result_root / _MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # ---- Phase 1a: block_frame -------------------------------------------
    block_frame.to_parquet(result_root / _BLOCK_FRAME_FILENAME, index=False)

    # ---- Phase 1b: roi_block_universe (JSON; compact) --------------------
    with open(result_root / _BLOCK_UNIVERSE_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(roi_block_universe, fh)

    # ---- Phase 1c: roi_block_summary (stacked long format) ---------------
    stacked_parts: list[pd.DataFrame] = []
    for roi_id, df in roi_block_summary.items():
        part = df.copy()
        part.insert(0, "roi_id", roi_id)
        stacked_parts.append(part)
    stacked_summary = pd.concat(stacked_parts, ignore_index=True)
    stacked_summary.to_parquet(result_root / _BLOCK_SUMMARY_FILENAME, index=False)

    # ---- Phase 2: density reference -------------------------------------
    density_records: list[dict] = []
    for roi_id, density_vec in roi_density_vectors.items():
        row: dict[str, Any] = {
            "roi_id": roi_id,
            "total_area_mm2": roi_total_areas[roi_id],
        }
        for k, col_name in enumerate([f"density_k{k}" for k in range(K_FULL)]):
            row[col_name] = float(density_vec[k])
        density_records.append(row)
    df_density = (
        pd.DataFrame.from_records(density_records)
        .sort_values("roi_id")
        .reset_index(drop=True)
    )
    df_density.to_parquet(result_root / _DENSITY_REFERENCE_FILENAME, index=False)

    # ---- Phase 3: pair meta tables --------------------------------------
    pair_meta_full.to_parquet(result_root / _PAIR_META_FULL_FILENAME, index=False)
    pair_meta_anchor.to_parquet(result_root / _PAIR_META_ANCHOR_FILENAME, index=False)


# ---------------------------------------------------------------------------
# Phase 4 output writer
# ---------------------------------------------------------------------------


def _select_full_coverage_columns(
    df: pd.DataFrame,
    lambda_arr: np.ndarray,
    tau_arr: np.ndarray,
) -> pd.DataFrame:
    """
    Build the Phase 6 full-coverage output DataFrame with the required schema.

    Selects and renames columns to match the arm3_phase6_full_coverage_results schema.
    """
    df = df.copy()
    df["lambda_dens"] = lambda_arr
    df["tau"] = tau_arr

    required = [
        "patient_group_id", "pair_id", "arm", "patient_id", "compartment",
        "patient_id_a", "patient_id_b",
        "compartment_a", "compartment_b",
        "same_patient", "same_compartment",
        "pair_type", "roi_a", "roi_b",
        "lambda_pl", "lambda_mode", "tau_mode", "mass_mode",
        "uot_status", "bypass_reason", "mass_pruned_ratio", "n_min_proto_used",
        "S0", "S1", "scale_ratio", "U",
        "T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau",
        "pair_family",
        "lambda_dens",
        "U_abs_dens", "S_src", "S_tgt", "Delta_scale",
        "Q_src_dens", "Q_tgt_dens",
    ]
    if "density_active_pruned_ratio" in df.columns:
        required.insert(required.index("n_min_proto_used") + 1, "density_active_pruned_ratio")
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"_select_full_coverage_columns: df is missing required output columns: "
            f"{missing_cols}"
        )
    return df[required].reset_index(drop=True)


def _build_support_mask_audit(
    pair_meta_anchor: pd.DataFrame,
    support_masks: np.ndarray,
    k_full: int,
) -> pd.DataFrame:
    """
    Build the support mask audit table.

    Columns: pair_id, n_supported_proto, support_fraction.
    """
    n_supported = support_masks.sum(axis=1).astype(int)
    support_fraction = n_supported.astype(float) / k_full
    audit = pair_meta_anchor[["pair_id"]].copy().reset_index(drop=True)
    audit["n_supported_proto"] = n_supported
    audit["support_fraction"] = support_fraction
    return audit


def _build_metric_summary_anchor(
    df_bootstrap: pd.DataFrame,
    df_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build arm3_phase6_metric_summary_anchor.parquet.

    Restricted to anchor directions (TC->IM, TC->PT).
    Grouped by pair_type and coverage marker.
    Summarises U_abs_dens, Q_src_dens, Q_tgt_dens, scale_ratio, floor_dominated.

    Full-coverage rows use coverage=1.0 as a baseline marker for grouping only.
    The reduced bootstrap ladder remains limited to 0.75 / 0.50 / 0.25.
    """
    # Add a baseline coverage marker for stacking with reduced bootstrap rows.
    # coverage=1.0 here denotes the frozen full-coverage reference only.
    df_full_copy = df_full.copy()
    df_full_copy["coverage"] = 1.0
    df_full_copy["floor_dominated"] = False  # no floor-dominated concept for full-coverage baseline

    # Select columns present in both frames
    shared_cols = [
        "pair_id", "pair_type", "coverage",
        "U_abs_dens", "Q_src_dens", "Q_tgt_dens", "scale_ratio",
    ]
    boot_cols = shared_cols + ["floor_dominated"]

    # Only include columns that exist in df_bootstrap
    boot_avail = [c for c in boot_cols if c in df_bootstrap.columns]
    full_avail = [c for c in boot_cols if c in df_full_copy.columns]

    df_stack = pd.concat(
        [
            df_bootstrap[boot_avail].copy(),
            df_full_copy[full_avail].copy(),
        ],
        ignore_index=True,
    )

    # Self-check: only anchor directions
    non_anchor = set(df_stack["pair_type"].astype(str).unique()) - set(ARM3_ANCHOR_DIRECTIONS)
    if non_anchor:
        raise RuntimeError(
            f"_build_metric_summary_anchor: non-anchor pair_type found: {sorted(non_anchor)}"
        )

    metric_cols = ["U_abs_dens", "Q_src_dens", "Q_tgt_dens", "scale_ratio"]

    # Mean, std, median per (pair_type, coverage)
    agg_records: list[dict] = []
    for (pair_type, coverage), grp in df_stack.groupby(["pair_type", "coverage"], sort=True):
        rec: dict[str, Any] = {"pair_type": pair_type, "coverage": coverage}
        n = len(grp)
        rec["n_rows"] = n
        for col in metric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            rec[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
            rec[f"{col}_median"] = float(vals.median()) if len(vals) > 0 else float("nan")
        if "floor_dominated" in grp.columns:
            rec["floor_dominated_rate"] = float(grp["floor_dominated"].mean())
        agg_records.append(rec)

    return pd.DataFrame.from_records(agg_records)


def _write_phase5_6_outputs(
    result_root: Path,
    df_pseudo_audit: pd.DataFrame,
    df_full_coverage: pd.DataFrame,
    df_bootstrap: pd.DataFrame,
    df_bot: pd.DataFrame,
    df_support_mask_audit: pd.DataFrame,
    df_metric_summary: pd.DataFrame,
) -> None:
    """Write Phase 5/6 output parquet files to result_root."""

    df_pseudo_audit.to_parquet(
        result_root / _PSEUDO_ROI_AUDIT_FILENAME, index=False
    )
    df_full_coverage.to_parquet(
        result_root / _FULL_COVERAGE_RESULTS_FILENAME, index=False
    )
    df_bootstrap.to_parquet(
        result_root / _BOOTSTRAP_RESULTS_FILENAME, index=False
    )
    df_bot.to_parquet(
        result_root / _BALANCED_OT_RESULTS_FILENAME, index=False
    )
    df_support_mask_audit.to_parquet(
        result_root / _SUPPORT_MASK_AUDIT_FILENAME, index=False
    )
    df_metric_summary.to_parquet(
        result_root / _METRIC_SUMMARY_ANCHOR_FILENAME, index=False
    )


def _write_phase4_outputs(
    result_root: Path,
    lambda_dens: dict[str, float],
    lambda_grid: tuple[float, ...],
    target_alpha: float,
    tau_by_compartment: dict[str, float],
    tau_q: float,
) -> None:
    """
    Write Phase 4 calibration outputs to result_root.

    Always writes:
        arm3_phase4_lambda_dens.json
        arm3_phase4_tau_by_compartment.json
        arm3_phase4_calibration_record.json
    """
    # arm3_phase4_lambda_dens.json
    with open(result_root / _LAMBDA_DENS_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(lambda_dens, fh, indent=2)

    # arm3_phase4_tau_by_compartment.json
    with open(result_root / _TAU_BY_COMPARTMENT_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(tau_by_compartment, fh, indent=2)

    # arm3_phase4_calibration_record.json
    outputs_written = [
        _LAMBDA_DENS_FILENAME,
        _TAU_BY_COMPARTMENT_FILENAME,
        _CALIBRATION_RECORD_FILENAME,
    ]
    record: dict[str, Any] = {
        "phase": 4,
        "arm3_name": ARM3_NAME,
        "lambda_dens_calibrated": True,
        "lambda_dens": lambda_dens,
        "lambda_grid_used": list(lambda_grid),
        "target_alpha": target_alpha,
        "tau_by_compartment_calibrated": True,
        "tau_by_compartment": tau_by_compartment,
        "tau_q": tau_q,
        "tau_calibration_rule": "pi_weighted_pooled_cost_quantile_across_within_patient_same_compartment_pairs",
        "run_utc": datetime.now(tz=timezone.utc).isoformat(),
        "outputs_written": outputs_written,
        "phase_5_plus": "IMPLEMENTED",
    }
    with open(result_root / _CALIBRATION_RECORD_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)


# ---------------------------------------------------------------------------
# Patch 1: Prototype event table builder
# ---------------------------------------------------------------------------


def _build_prototype_events_df(
    pair_meta: pd.DataFrame,
    T_k: np.ndarray,
    B_k: np.ndarray,
    D_k: np.ndarray,
    proto_labels: list[str],
    coverage: float,
    replicate_id: int,
) -> pd.DataFrame:
    """
    Build a long-format prototype event DataFrame for one inference batch.

    Vectorised expansion: for each pair i in [0, N) and each prototype k in
    [0, K), one row is emitted with the event masses T_mass, B_mass, D_mass.

    Parameters
    ----------
    pair_meta : pd.DataFrame
        Anchor pair metadata (N rows). Required columns: pair_id, patient_id,
        pair_type, pair_family.
    T_k, B_k, D_k : np.ndarray, shape (N, K)
        Per-prototype event mass tensors from run_uot_batch_with_events.
        NaN on non-ok rows.
    proto_labels : list[str]
        Prototype name strings of length K.
    coverage : float
        Coverage level: 1.0 for full-coverage reference, 0.75/0.50/0.25 for
        bootstrap replicates.
    replicate_id : int
        Replicate index.  Sentinel value -1 for the full-coverage reference.

    Returns
    -------
    pd.DataFrame with columns:
        pair_id, patient_id, pair_type, pair_family,
        coverage, replicate_id, prototype_k, prototype_label,
        T_mass, B_mass, D_mass.
    """
    n_pairs, k_full = T_k.shape
    pair_meta_r = pair_meta.reset_index(drop=True)

    # Vectorised long-format expansion: all K prototypes for each pair
    pair_indices = np.repeat(np.arange(n_pairs), k_full)   # [0,0,...,0, 1,1,...]
    k_indices = np.tile(np.arange(k_full), n_pairs)         # [0,1,...,K-1, 0,1,...]

    proto_labels_arr = np.asarray(proto_labels, dtype=object)

    df = pd.DataFrame({
        "pair_id":        pair_meta_r["pair_id"].to_numpy()[pair_indices],
        "patient_id":     pair_meta_r["patient_id"].to_numpy()[pair_indices],
        "pair_type":      pair_meta_r["pair_type"].to_numpy()[pair_indices],
        "pair_family":    pair_meta_r["pair_family"].to_numpy()[pair_indices],
        "coverage":       float(coverage),
        "replicate_id":   int(replicate_id),
        "prototype_k":    k_indices,
        "prototype_label": proto_labels_arr[k_indices],
        "T_mass":         T_k.ravel(),
        "B_mass":         B_k.ravel(),
        "D_mass":         D_k.ravel(),
    })
    return df
