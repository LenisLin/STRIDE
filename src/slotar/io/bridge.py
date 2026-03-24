"""
Module: src.slotar.io.bridge
Architecture: Library Level (I/O & Export)
Constraints:
- STRICTLY NO `yaml` imports or config parsing.
- Paths and audit metadata must be explicitly provided by the caller (tasks layer).
- Must enforce output data contracts before writing to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..contracts import DataContractError, validate_events_table, validate_metrics_table

METRICS_FILENAME = "metrics_.csv"
EVENTS_FILENAME = "events_.parquet"
META_FILENAME = "meta_.json"


def _validate_aux_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise DataContractError("aux_tables keys must be non-empty strings")
    if normalized in {"metrics", "events", "meta"}:
        raise DataContractError(f"aux_tables key {normalized!r} collides with a canonical artifact name")
    return normalized


def save_for_r(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: str | Path,
    meta_audit: dict[str, Any],
    aux_tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """
    Export canonical SLOTAR bridge artifacts to disk.
    """
    validate_metrics_table(metrics_df)
    validate_events_table(events_df)

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / METRICS_FILENAME
    events_path = out_dir / EVENTS_FILENAME
    meta_path = out_dir / META_FILENAME

    metrics_df.to_csv(metrics_path, index=False)
    events_df.to_parquet(events_path, index=False)

    written_paths: dict[str, Path] = {
        "metrics": metrics_path,
        "events": events_path,
        "meta": meta_path,
    }
    aux_payload: dict[str, str] = {}
    if aux_tables is not None:
        for name, table in aux_tables.items():
            normalized_name = _validate_aux_name(name)
            if not isinstance(table, pd.DataFrame):
                raise DataContractError(f"aux_tables[{normalized_name!r}] must be a pandas DataFrame")
            aux_path = out_dir / f"aux_{normalized_name}.csv"
            table.to_csv(aux_path, index=False)
            written_paths[f"aux:{normalized_name}"] = aux_path
            aux_payload[normalized_name] = aux_path.name

    payload = dict(meta_audit)
    payload.setdefault("schema_version", "v2.0")
    payload["artifacts"] = {
        "metrics": metrics_path.name,
        "events": events_path.name,
        "meta": meta_path.name,
    }
    if aux_payload:
        payload["aux_tables"] = aux_payload

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return written_paths
