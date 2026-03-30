"""Compatibility bridge exports for older `slotar.io.bridge` imports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...errors import DataContractError
from ...io.r_export import EVENTS_FILENAME, META_FILENAME, METRICS_FILENAME, write_r_handover


def save_for_r(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: str | Path,
    meta_audit: dict[str, Any],
    aux_tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Compatibility alias for `write_r_handover(...)`."""
    return write_r_handover(
        metrics_df=metrics_df,
        events_df=events_df,
        output_dir=output_dir,
        meta_audit=meta_audit,
        aux_tables=aux_tables,
    )


__all__ = [
    "DataContractError",
    "EVENTS_FILENAME",
    "META_FILENAME",
    "METRICS_FILENAME",
    "save_for_r",
    "write_r_handover",
]
