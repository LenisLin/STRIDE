"""
Module: src.slotar.contracts
Architecture: Library Level (Domain-Agnostic Mathematical Engine)
Constraints:
- STRICTLY NO references to `tasks`, `config.yaml`, or clinical metadata.
- Implements SLOTAR V2.0 data contracts. Validation functions must be used strictly 
  as gatekeepers before any heavy mathematical operations.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)

try:
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = Any  # type: ignore[misc,assignment]

# ---- Canonical column/field names (Locked per V2.0 conventions) ----
REQUIRED_OBS_COLS: tuple[str, ...] = ("patient_id", "timepoint", "roi_id", "compartment")
OPTIONAL_OBS_COLS: tuple[str, ...] = ("cell_type", "proto_id", "block_id")

REQUIRED_OBSM_KEYS: tuple[str, ...] = ("spatial",)
OPTIONAL_OBSM_KEYS: tuple[str, ...] = ("community_features",)

CANONICAL_COST_SCALE_KEY = "s_C"
COST_SCALE_ALIASES: tuple[str, ...] = ("global_cost_scale",)

REQUIRED_UNS_KEYS: tuple[str, ...] = ("roi_areas",)
OPTIONAL_UNS_KEYS: tuple[str, ...] = ("scaler_params", CANONICAL_COST_SCALE_KEY, *COST_SCALE_ALIASES)

MICRO_METRICS: tuple[str, ...] = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")
CANONICAL_UOT_STATUSES: tuple[str, ...] = (
    "ok",
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)
CANONICAL_BYPASS_REASONS: tuple[str, ...] = (
    "S0_zero",
    "S1_zero",
    "empty_support_after_prune",
    "uot_numerical_failure",
)

class DataContractError(ValueError):
    """Raised when a declared SLOTAR programmer-level contract is violated."""

def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataContractError(f"{where}: missing required columns: {missing}")


def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise DataContractError(f"{where}: missing required keys: {missing}")


def validate_adata_inputs(
    adata: AnnData,
    *,
    require_cell_type: bool = False,
    require_representation: bool = False,
    require_prototypes: bool = False,
    require_cost_scale: bool = False,
    require_cost_matrix: bool = False,
) -> None:
    """
    Validates AnnData against SLOTAR data_contracts.
    This function is intentionally modular: callers choose strictness via flags.
    """
    # obs
    _require_keys(adata.obs, REQUIRED_OBS_COLS, where="adata.obs")
    if require_cell_type and "cell_type" not in adata.obs.columns:
        raise DataContractError("adata.obs: missing required column 'cell_type'")

    # obsm
    _require_keys(adata.obsm, REQUIRED_OBSM_KEYS, where="adata.obsm")
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise DataContractError("adata.obsm['spatial'] must have shape [n, 2]")
    if not np.isfinite(spatial).all():
        raise DataContractError("adata.obsm['spatial'] contains NaN/Inf (must fail-fast)")

    if require_representation:
        _require_keys(adata.obsm, ("community_features",), where="adata.obsm")
        feats = np.asarray(adata.obsm["community_features"])
        if feats.ndim != 2 or feats.shape[0] != spatial.shape[0]:
            raise DataContractError("adata.obsm['community_features'] must have shape [n, d]")

    if require_prototypes and "proto_id" not in adata.obs.columns:
        raise DataContractError("adata.obs: missing required column 'proto_id'")

    # uns
    _require_keys(adata.uns, REQUIRED_UNS_KEYS, where="adata.uns")
    roi_areas = adata.uns.get("roi_areas")
    if not isinstance(roi_areas, Mapping):
        raise DataContractError("adata.uns['roi_areas'] must be a mapping roi_id -> area_mm2")

    if require_cost_scale:
        if CANONICAL_COST_SCALE_KEY in adata.uns:
            pass
        elif any(k in adata.uns for k in COST_SCALE_ALIASES):
            pass
        else:
            raise DataContractError(
                f"adata.uns: missing cost scale key (expected '{CANONICAL_COST_SCALE_KEY}' "
                f"or aliases {list(COST_SCALE_ALIASES)})"
            )

    if require_cost_matrix:
        if "cost_matrix" not in adata.uns:
            raise DataContractError("adata.uns: missing required key 'cost_matrix'")
        try:
            cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataContractError("adata.uns['cost_matrix'] must be numeric / array-like") from exc

        if cost_matrix.ndim != 2:
            raise DataContractError("adata.uns['cost_matrix'] must be 2D")
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise DataContractError("adata.uns['cost_matrix'] must be square")
        if not np.isfinite(cost_matrix).all():
            raise DataContractError("adata.uns['cost_matrix'] contains NaN/Inf")


def validate_uot_inputs(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
) -> None:
    """
    Validates programmer-level invariants for batched_uot_solve inputs.
    Data-level degeneracies (empty mass/support per item) are expected to be handled
    via status codes inside the batched solver (batch-isolated), not by raising here.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    lambda_pl = np.asarray(lambda_pl, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise DataContractError("A and B must be 2D arrays of shape [N, K]")
    if A.shape != B.shape:
        raise DataContractError(f"A and B shape mismatch: {A.shape} vs {B.shape}")

    N, K = A.shape
    if lambda_pl.shape != (N,):
        raise DataContractError(f"lambda_pl must have shape {(N,)}, got {lambda_pl.shape}")

    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise DataContractError("A/B contain NaN/Inf")
    if (A < 0).any() or (B < 0).any():
        raise DataContractError("A/B must be non-negative")

    if not np.isfinite(lambda_pl).all() or (lambda_pl <= 0).any():
        raise DataContractError("lambda_pl must be finite and strictly positive")

    if len(kernels) == 0:
        raise DataContractError("kernels must be a non-empty epsilon schedule")
    for i, logK in enumerate(kernels):
        logK = np.asarray(logK, dtype=float)
        if logK.shape != (K, K):
            raise DataContractError(f"kernels[{i}] must have shape {(K, K)}, got {logK.shape}")
        if not np.isfinite(logK).all():
            raise DataContractError(f"kernels[{i}] contains NaN/Inf")


def validate_metrics_table(df: pd.DataFrame) -> None:
    """
    Enforces output contracts at the table level (schema + bypass rules).
    """
    _require_columns(df, ("patient_group_id", "uot_status"), where="metrics table")

    # Primary key integrity
    if df["patient_group_id"].isna().any():
        raise DataContractError("metrics table: patient_group_id contains NA")
    if df["patient_group_id"].duplicated().any():
        raise DataContractError("metrics table: patient_group_id contains duplicates")

    # Vocabulary contract for status and bypass audit fields
    status_vals = df["uot_status"].astype("string")
    bad_status = ~(status_vals.isna() | status_vals.isin(CANONICAL_UOT_STATUSES))
    if bad_status.any():
        invalid = sorted(status_vals[bad_status].dropna().unique().tolist())
        raise DataContractError(f"metrics table: invalid uot_status values: {invalid}")

    if "bypass_reason" in df.columns:
        reason_vals = df["bypass_reason"].astype("string")
        bad_reason = ~(reason_vals.isna() | reason_vals.isin(CANONICAL_BYPASS_REASONS))
        if bad_reason.any():
            invalid = sorted(reason_vals[bad_reason].dropna().unique().tolist())
            raise DataContractError(f"metrics table: invalid bypass_reason values: {invalid}")

    # Bypass contract: if uot_status != ok => all micro metrics must be NaN (if present)
    if "uot_status" in df.columns:
        bypass = df["uot_status"].astype("string") != "ok"
        for m in MICRO_METRICS:
            if m in df.columns:
                bad = (~df.loc[bypass, m].isna()).any()
                if bad:
                    raise DataContractError(
                        f"metrics table: rows with uot_status!=ok must have NaN for '{m}'"
                    )


def validate_events_table(df: pd.DataFrame) -> None:
    """
    Minimal events schema validation. Tighten as your event schema locks.
    """
    _require_columns(df, ("patient_group_id", "event_type"), where="events table")
    if df["patient_group_id"].isna().any():
        raise DataContractError("events table: patient_group_id contains NA")
