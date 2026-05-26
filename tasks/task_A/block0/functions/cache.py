"""Fit-cache persistence for Task A Block 0 execution.

The cache stores realized full-STRIDE patient fit payloads (`A`, `d`, `e`,
`mu_minus`, `mu_plus`) from real `TC-IM` and empirical-null fits. It does not
derive calibration metrics; analysis commands consume this cache later.
"""
from __future__ import annotations

import csv
import hashlib
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stride.errors import ContractError

from .schemas import (
    FIT_CACHE_FILENAME,
    FIT_CACHE_INDEX_COLUMNS,
    FIT_CACHE_INDEX_FILENAME,
    FIT_LABEL_NULL,
    FIT_LABEL_REAL,
    Block0FitRecord,
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a written artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_block0_fit_cache(
    records: Sequence[Block0FitRecord],
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Persist fit records as compact arrays plus a human-auditable index."""
    ordered_records = _validate_and_order_records(tuple(records))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = _records_to_arrays(ordered_records)
    index_rows = _records_to_index_rows(ordered_records)
    cache_path = output_path / FIT_CACHE_FILENAME
    index_path = output_path / FIT_CACHE_INDEX_FILENAME

    with tempfile.TemporaryDirectory(
        dir=output_path,
        prefix=".block0_fit_cache_",
        suffix=".tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_cache_path = temp_dir / FIT_CACHE_FILENAME
        temp_index_path = temp_dir / FIT_CACHE_INDEX_FILENAME

        with temp_cache_path.open("wb") as handle:
            np.savez(
                handle,
                A=arrays["A"],
                d=arrays["d"],
                e=arrays["e"],
                source_burden=arrays["source_burden"],
                target_burden=arrays["target_burden"],
            )
        _write_index_csv(temp_index_path, index_rows)
        read_block0_fit_cache(temp_cache_path, temp_index_path)

        temp_cache_path.replace(cache_path)
        temp_index_path.replace(index_path)

    return {
        "fit_cache_path": cache_path,
        "fit_cache_index_path": index_path,
        "fit_cache_sha256": sha256_file(cache_path),
        "fit_cache_index_sha256": sha256_file(index_path),
        "record_count": len(ordered_records),
        "k_states": int(arrays["d"].shape[1]),
        "patient_count": len({record.patient_id for record in ordered_records}),
    }


def read_block0_fit_cache(
    fit_cache_path: str | Path,
    fit_cache_index_path: str | Path,
) -> tuple[Block0FitRecord, ...]:
    """Load persisted fit records from an execution cache."""
    index_rows = _read_index_csv(Path(fit_cache_index_path))
    with np.load(Path(fit_cache_path), allow_pickle=False) as payload:
        required = ("A", "d", "e", "source_burden", "target_burden")
        missing = tuple(name for name in required if name not in payload)
        if missing:
            raise ContractError(f"Block 0 fit cache is missing arrays: {missing}")
        arrays = {name: np.asarray(payload[name], dtype=float) for name in required}

    n_records = len(index_rows)
    if any(array.shape[0] != n_records for array in arrays.values()):
        raise ContractError("Block 0 fit cache arrays must align with index rows")

    records: list[Block0FitRecord] = []
    for row_index, row in enumerate(index_rows):
        records.append(
            Block0FitRecord(
                patient_id=str(row["patient_id"]),
                fit_label=str(row["fit_label"]),
                permutation_index=_parse_permutation_index(row["permutation_index"]),
                fit_status=str(row["fit_status"]),
                A=arrays["A"][row_index],
                d=arrays["d"][row_index],
                e=arrays["e"][row_index],
                source_burden=arrays["source_burden"][row_index],
                d_weights=arrays["source_burden"][row_index],
                e_weights=arrays["target_burden"][row_index],
            )
        )
    return _validate_and_order_records(tuple(records))


def _validate_and_order_records(records: tuple[Block0FitRecord, ...]) -> tuple[Block0FitRecord, ...]:
    if not records:
        raise ContractError("Block 0 fit cache requires at least one fit record")
    by_patient: dict[str, dict[str, list[Block0FitRecord]]] = defaultdict(
        lambda: {FIT_LABEL_REAL: [], FIT_LABEL_NULL: []}
    )
    k_states: int | None = None
    for record in records:
        _validate_record_arrays(record)
        record_k = int(np.asarray(record.d, dtype=float).shape[0])
        if k_states is None:
            k_states = record_k
        elif record_k != k_states:
            raise ContractError("Block 0 fit cache records must share one K-state basis")
        by_patient[str(record.patient_id)][str(record.fit_label)].append(record)

    expected_indices: tuple[int, ...] | None = None
    for patient_id, grouped in by_patient.items():
        if len(grouped[FIT_LABEL_REAL]) != 1:
            raise ContractError(f"Block 0 patient {patient_id!r} must have one real record")
        observed_indices = tuple(
            sorted(int(record.permutation_index) for record in grouped[FIT_LABEL_NULL])
        )
        if not observed_indices:
            raise ContractError(f"Block 0 patient {patient_id!r} has no null records")
        patient_expected = tuple(range(max(observed_indices) + 1))
        if observed_indices != patient_expected:
            raise ContractError(
                f"Block 0 patient {patient_id!r} null records must cover {patient_expected}"
            )
        if expected_indices is None:
            expected_indices = patient_expected
        elif observed_indices != expected_indices:
            raise ContractError("Block 0 fit cache patients must share null permutation indices")

    ordered: list[Block0FitRecord] = []
    for patient_id in sorted(by_patient):
        ordered.extend(sorted(by_patient[patient_id][FIT_LABEL_REAL], key=lambda record: record.patient_id))
    for permutation_index in expected_indices or ():
        for patient_id in sorted(by_patient):
            match = [
                record
                for record in by_patient[patient_id][FIT_LABEL_NULL]
                if int(record.permutation_index) == int(permutation_index)
            ]
            ordered.extend(match)
    return tuple(ordered)


def _validate_record_arrays(record: Block0FitRecord) -> None:
    A = _finite_array(record.A, name="A")
    d = _finite_array(record.d, name="d")
    e = _finite_array(record.e, name="e")
    source = _finite_array(record.source_burden, name="source_burden")
    target = _finite_array(record.e_weights, name="target_burden")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ContractError("Block 0 fit cache A must be a square matrix")
    expected_shape = (A.shape[0],)
    if d.shape != expected_shape or e.shape != expected_shape:
        raise ContractError("Block 0 fit cache d/e vectors must match A dimensions")
    if source.shape != expected_shape or target.shape != expected_shape:
        raise ContractError("Block 0 fit cache burden vectors must match A dimensions")


def _finite_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ContractError(f"Block 0 fit cache {name} must contain finite values")
    return array


def _records_to_arrays(records: tuple[Block0FitRecord, ...]) -> dict[str, np.ndarray]:
    return {
        "A": np.stack([np.asarray(record.A, dtype=float) for record in records], axis=0),
        "d": np.stack([np.asarray(record.d, dtype=float) for record in records], axis=0),
        "e": np.stack([np.asarray(record.e, dtype=float) for record in records], axis=0),
        "source_burden": np.stack(
            [np.asarray(record.source_burden, dtype=float) for record in records],
            axis=0,
        ),
        "target_burden": np.stack(
            [np.asarray(record.e_weights, dtype=float) for record in records],
            axis=0,
        ),
    }


def _records_to_index_rows(records: tuple[Block0FitRecord, ...]) -> list[dict[str, object]]:
    return [
        {
            "record_id": record_id,
            "fit_label": record.fit_label,
            "permutation_index": "" if record.permutation_index is None else int(record.permutation_index),
            "patient_id": record.patient_id,
            "fit_status": record.fit_status,
        }
        for record_id, record in enumerate(records)
    ]


def _write_index_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_CACHE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIT_CACHE_INDEX_COLUMNS:
            raise ContractError("Block 0 fit cache index columns do not match the schema")
        rows = [dict(row) for row in reader]
    for expected_record_id, row in enumerate(rows):
        if int(row["record_id"]) != expected_record_id:
            raise ContractError("Block 0 fit cache index record_id must be sequential")
    return rows


def _parse_permutation_index(value: object) -> int | None:
    if value is None or str(value) == "":
        return None
    return int(value)


__all__ = [
    "read_block0_fit_cache",
    "sha256_file",
    "write_block0_fit_cache",
]
