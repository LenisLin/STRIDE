"""Compatibility table validators for migration-era task outputs."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..errors import ContractError, DataContractError
from .status import CANONICAL_BYPASS_REASONS, CANONICAL_UOT_STATUSES

MICRO_METRICS: tuple[str, ...] = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ContractError(f"{where}: missing required columns: {missing}")


def validate_metrics_table(df: pd.DataFrame) -> None:
    """Validate the narrow legacy metrics-table surface."""
    _require_columns(df, ("patient_group_id", "uot_status"), where="metrics table")

    if df["patient_group_id"].isna().any():
        raise ContractError("metrics table: patient_group_id contains NA")
    if df["patient_group_id"].duplicated().any():
        raise ContractError("metrics table: patient_group_id contains duplicates")

    status_vals = df["uot_status"].astype("string")
    bad_status = ~(status_vals.isna() | status_vals.isin(CANONICAL_UOT_STATUSES))
    if bad_status.any():
        invalid = sorted(status_vals[bad_status].dropna().unique().tolist())
        raise ContractError(f"metrics table: invalid uot_status values: {invalid}")

    if "bypass_reason" in df.columns:
        reason_vals = df["bypass_reason"].astype("string")
        bad_reason = ~(reason_vals.isna() | reason_vals.isin(CANONICAL_BYPASS_REASONS))
        if bad_reason.any():
            invalid = sorted(reason_vals[bad_reason].dropna().unique().tolist())
            raise ContractError(f"metrics table: invalid bypass_reason values: {invalid}")

    bypass = df["uot_status"].astype("string") != "ok"
    for metric_name in MICRO_METRICS:
        if metric_name in df.columns and (~df.loc[bypass, metric_name].isna()).any():
            raise ContractError(
                f"metrics table: rows with uot_status!=ok must have NaN for '{metric_name}'"
            )


def validate_events_table(df: pd.DataFrame) -> None:
    """Validate the minimal legacy event-table schema."""
    _require_columns(df, ("patient_group_id", "event_type"), where="events table")
    if df["patient_group_id"].isna().any():
        raise ContractError("events table: patient_group_id contains NA")


__all__ = [
    "ContractError",
    "DataContractError",
    "MICRO_METRICS",
    "validate_events_table",
    "validate_metrics_table",
]
