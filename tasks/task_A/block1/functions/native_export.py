"""Task-local relation export records used by Block 1 analysis."""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stride.errors import ContractError


@dataclass(frozen=True)
class Block1RelationExportManifest:
    schema_version: str
    writer: str
    manifest_path: Path | None
    patient_index_path: Path | None
    patient_arrays_path: Path | None
    cohort_index_path: Path | None
    cohort_arrays_path: Path | None
    patient_index_sha256: str | None
    patient_arrays_sha256: str | None
    cohort_index_sha256: str | None
    cohort_arrays_sha256: str | None
    fit_status: str
    implementation_tier: str
    patient_count: int
    patient_record_count: int
    cohort_record_count: int
    k_states: int
    cohort_fit_status: str
    manifest_sha256: str | None = None


@dataclass(frozen=True)
class PatientRelationRecord:
    record_id: int
    patient_id: str
    fit_status: str
    implementation_tier: str
    k_states: int
    A: np.ndarray | None = None
    d: np.ndarray | None = None
    e: np.ndarray | None = None
    mu_minus: np.ndarray | None = None
    mu_plus: np.ndarray | None = None
    status_reason: str | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.fit_status == "ok"


@dataclass(frozen=True)
class CohortRelationRecord:
    record_id: int
    cohort_relation_id: str
    fit_status: str
    support_n_patients: int
    support_patient_ids: tuple[str, ...]
    dispersion: float | None
    k_states: int
    template_A: np.ndarray | None = None
    template_d: np.ndarray | None = None
    template_e: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.fit_status == "ok"


@dataclass(frozen=True)
class Block1RelationExport:
    manifest: Block1RelationExportManifest
    state_ids: tuple[int, ...]
    patient_records: tuple[PatientRelationRecord, ...]
    cohort_records: tuple[CohortRelationRecord, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stride_native_relation_export(manifest_path: str | Path) -> Block1RelationExport:
    """Read a legacy native export into TaskA-local record classes."""
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest_payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    manifest_dir = resolved_manifest_path.parent
    patient_index_path = _resolve_relative_path(manifest_payload.get("patient_index_path"), manifest_dir)
    patient_arrays_path = _resolve_relative_path(manifest_payload.get("patient_arrays_path"), manifest_dir)
    cohort_index_path = _resolve_relative_path(manifest_payload.get("cohort_index_path"), manifest_dir)
    cohort_arrays_path = _resolve_relative_path(manifest_payload.get("cohort_arrays_path"), manifest_dir)
    if patient_index_path is None or patient_arrays_path is None:
        raise ContractError("Block 1 legacy native export requires patient index and arrays")

    patient_records, state_ids = _read_patient_records(patient_index_path, patient_arrays_path)
    cohort_records = (
        ()
        if cohort_index_path is None or cohort_arrays_path is None
        else _read_cohort_records(cohort_index_path, cohort_arrays_path)
    )
    manifest = Block1RelationExportManifest(
        schema_version=str(manifest_payload["schema_version"]),
        writer=str(manifest_payload["writer"]),
        manifest_path=resolved_manifest_path,
        patient_index_path=patient_index_path,
        patient_arrays_path=patient_arrays_path,
        cohort_index_path=cohort_index_path,
        cohort_arrays_path=cohort_arrays_path,
        patient_index_sha256=manifest_payload.get("patient_index_sha256"),
        patient_arrays_sha256=manifest_payload.get("patient_arrays_sha256"),
        cohort_index_sha256=manifest_payload.get("cohort_index_sha256"),
        cohort_arrays_sha256=manifest_payload.get("cohort_arrays_sha256"),
        fit_status=str(manifest_payload["fit_status"]),
        implementation_tier=str(manifest_payload["implementation_tier"]),
        patient_count=int(manifest_payload["patient_count"]),
        patient_record_count=int(manifest_payload["patient_record_count"]),
        cohort_record_count=int(manifest_payload["cohort_record_count"]),
        k_states=int(manifest_payload["k_states"]),
        cohort_fit_status=str(manifest_payload["cohort_fit_status"]),
        manifest_sha256=sha256_file(resolved_manifest_path),
    )
    return Block1RelationExport(
        manifest=manifest,
        state_ids=state_ids,
        patient_records=patient_records,
        cohort_records=cohort_records,
    )


def _resolve_relative_path(value: object, manifest_dir: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return (manifest_dir / candidate).resolve()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_loads(value: str) -> dict[str, Any]:
    if value == "":
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ContractError("Block 1 native export JSON payload must be an object")
    return dict(loaded)


def _optional_array_slot(value: object, *, n_slots: int, label: str) -> int | None:
    stripped = str(value).strip()
    if stripped == "":
        return None
    slot = int(float(stripped))
    if slot < 0 or slot >= int(n_slots):
        raise ContractError(f"{label} array_slot is out of bounds")
    return slot


def _optional_float(value: object) -> float | None:
    stripped = str(value).strip()
    if stripped == "":
        return None
    return float(stripped)


def _read_patient_records(
    index_path: Path,
    arrays_path: Path,
) -> tuple[tuple[PatientRelationRecord, ...], tuple[int, ...]]:
    rows = _read_csv_rows(index_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        state_ids = tuple(int(value) for value in np.asarray(archive["state_ids"], dtype=int))
        records: list[PatientRelationRecord] = []
        for row in rows:
            slot = _optional_array_slot(row.get("array_slot", ""), n_slots=archive["A"].shape[0], label="patient")
            records.append(
                PatientRelationRecord(
                    record_id=int(row["record_id"]),
                    patient_id=str(row["patient_id"]),
                    fit_status=str(row["fit_status"]),
                    implementation_tier=str(row["implementation_tier"]),
                    k_states=int(row["k_states"]),
                    A=None if slot is None else np.asarray(archive["A"][slot], dtype=float),
                    d=None if slot is None else np.asarray(archive["d"][slot], dtype=float),
                    e=None if slot is None else np.asarray(archive["e"][slot], dtype=float),
                    mu_minus=None if slot is None else np.asarray(archive["mu_minus"][slot], dtype=float),
                    mu_plus=None if slot is None else np.asarray(archive["mu_plus"][slot], dtype=float),
                    status_reason=str(row.get("status_reason") or "") or None,
                    audit=_json_loads(str(row.get("audit_json", ""))),
                    diagnostics=_json_loads(str(row.get("diagnostics_json", ""))),
                )
            )
    return tuple(records), state_ids


def _read_cohort_records(index_path: Path, arrays_path: Path) -> tuple[CohortRelationRecord, ...]:
    rows = _read_csv_rows(index_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        records: list[CohortRelationRecord] = []
        for row in rows:
            slot = _optional_array_slot(row.get("array_slot", ""), n_slots=archive["template_A"].shape[0], label="cohort")
            support_ids = tuple(str(value) for value in json.loads(str(row["support_patient_ids_json"])))
            records.append(
                CohortRelationRecord(
                    record_id=int(row["record_id"]),
                    cohort_relation_id=str(row["cohort_relation_id"]),
                    fit_status=str(row["fit_status"]),
                    support_n_patients=int(row["support_n_patients"]),
                    support_patient_ids=support_ids,
                    dispersion=_optional_float(row.get("dispersion", "")),
                    k_states=int(row["k_states"]),
                    template_A=None if slot is None else np.asarray(archive["template_A"][slot], dtype=float),
                    template_d=None if slot is None else np.asarray(archive["template_d"][slot], dtype=float),
                    template_e=None if slot is None else np.asarray(archive["template_e"][slot], dtype=float),
                    metadata=_json_loads(str(row.get("metadata_json", ""))),
                )
            )
    return tuple(records)


__all__ = [
    "CohortRelationRecord",
    "Block1RelationExport",
    "Block1RelationExportManifest",
    "PatientRelationRecord",
    "read_stride_native_relation_export",
    "sha256_file",
]
