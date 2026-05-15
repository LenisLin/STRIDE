"""Block 1 artifact writers and output guards."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from stride.errors import ContractError

from .schemas import PROHIBITED_OUTPUT_MARKERS


def _validate_exact_columns(frame: pd.DataFrame, *, columns: Sequence[str], label: str) -> None:
    expected = tuple(columns)
    observed = tuple(str(column) for column in frame.columns)
    if observed != expected:
        raise ContractError(f"{label} columns do not match the schema: expected={expected}, observed={observed}")


def _reject_existing_output(path: Path) -> None:
    if path.exists():
        raise ContractError(f"Block 1 output already exists and will not be overwritten: {path}")


def write_block1_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[str],
) -> Path:
    """Validate exact columns and atomically write an R-friendly CSV."""
    _validate_exact_columns(frame, columns=columns, label=str(path.name))
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_output(resolved_path)
    with tempfile.NamedTemporaryFile(
        dir=resolved_path.parent,
        prefix=f".{resolved_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False, columns=list(columns))
        _reject_existing_output(resolved_path)
        temp_path.replace(resolved_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved_path


def write_block1_json(
    payload: Mapping[str, object],
    path: Path,
    *,
    required_keys: Sequence[str],
) -> Path:
    """Validate required manifest keys and atomically write JSON."""
    missing = tuple(key for key in required_keys if key not in payload)
    if missing:
        raise ContractError(f"{path.name} is missing required manifest keys: {missing}")
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_output(resolved_path)
    with tempfile.NamedTemporaryFile(
        dir=resolved_path.parent,
        prefix=f".{resolved_path.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        _reject_existing_output(resolved_path)
        temp_path.replace(resolved_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved_path


def validate_no_forbidden_block1_outputs(output_dir: Path) -> None:
    """Guard against accidental p-value, FDR, figure, or annotation artifacts."""
    resolved_dir = Path(output_dir)
    if not resolved_dir.exists():
        return
    forbidden_hits: list[str] = []
    for path in resolved_dir.rglob("*"):
        if not path.is_file():
            continue
        lowercase_name = path.name.lower()
        if any(marker in lowercase_name for marker in PROHIBITED_OUTPUT_MARKERS):
            forbidden_hits.append(str(path))
    if forbidden_hits:
        raise ContractError(f"Block 1 emitted forbidden outputs: {tuple(sorted(forbidden_hits))}")


__all__ = [
    "validate_no_forbidden_block1_outputs",
    "write_block1_csv",
    "write_block1_json",
]
