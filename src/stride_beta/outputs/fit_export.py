"""Task-agnostic native relation export for STRIDE fit results."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ContractError
from ..latent import CohortRelation
from .fit_result import PatientRelationResult, STRIDEFitResult


STRIDE_NATIVE_RELATION_EXPORT_VERSION = "stride_native_relation_export_v1"
STRIDE_NATIVE_RELATION_EXPORT_WRITER = "stride.outputs.fit_export"
MANIFEST_FILENAME = "stride_native_relation_manifest.json"
PATIENT_INDEX_FILENAME = "stride_patient_relation_index.csv"
PATIENT_ARRAYS_FILENAME = "stride_patient_relation_arrays.npz"
COHORT_INDEX_FILENAME = "stride_cohort_relation_index.csv"
COHORT_ARRAYS_FILENAME = "stride_cohort_relation_arrays.npz"

PATIENT_INDEX_COLUMNS: tuple[str, ...] = (
    "record_id",
    "patient_id",
    "fit_status",
    "implementation_tier",
    "k_states",
    "array_slot",
    "status_reason",
    "audit_json",
    "diagnostics_json",
)
COHORT_INDEX_COLUMNS: tuple[str, ...] = (
    "record_id",
    "cohort_relation_id",
    "fit_status",
    "support_n_patients",
    "support_patient_ids_json",
    "dispersion",
    "k_states",
    "array_slot",
    "metadata_json",
)


def _native_relation_export_target_paths(output_path: Path) -> tuple[Path, ...]:
    return (
        output_path / MANIFEST_FILENAME,
        output_path / PATIENT_INDEX_FILENAME,
        output_path / PATIENT_ARRAYS_FILENAME,
        output_path / COHORT_INDEX_FILENAME,
        output_path / COHORT_ARRAYS_FILENAME,
    )


def _reject_existing_native_relation_targets(output_path: Path) -> None:
    existing = tuple(path for path in _native_relation_export_target_paths(output_path) if path.exists())
    if existing:
        raise ContractError(
            "Native relation export target already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest for one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: str) -> dict[str, Any]:
    if value == "":
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ContractError("JSON payload must deserialize to an object")
    return dict(loaded)


def _optional_array_slot(value: object, *, n_slots: int, label: str) -> int | None:
    stripped = str(value).strip()
    if stripped == "":
        return None
    try:
        slot = int(stripped)
    except ValueError:
        try:
            numeric_slot = float(stripped)
        except ValueError as exc:
            raise ContractError(f"{label} array_slot must be an integer") from exc
        if not numeric_slot.is_integer():
            raise ContractError(f"{label} array_slot must be an integer")
        slot = int(numeric_slot)
    if slot < 0 or slot >= int(n_slots):
        raise ContractError(
            f"{label} array_slot {slot} is out of bounds for {int(n_slots)} realized arrays"
        )
    return slot


def _optional_float(value: str) -> float | None:
    stripped = str(value).strip()
    if stripped == "":
        return None
    return float(stripped)


def _normalize_status_reason(
    patient_result: PatientRelationResult | None = None,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> str | None:
    if diagnostics is not None and "defer_reason" in diagnostics:
        reason = str(diagnostics["defer_reason"]).strip()
        return reason or None
    if patient_result is not None and "defer_reason" in patient_result.diagnostics:
        reason = str(patient_result.diagnostics["defer_reason"]).strip()
        return reason or None
    return None


def _finite_vector(value: object, *, expected_size: int, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (expected_size,):
        raise ContractError(
            f"{field_name} must have shape {(expected_size,)}, got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ContractError(f"{field_name} must contain finite values")
    return vector


def _finite_matrix(value: object, *, expected_size: int, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (expected_size, expected_size):
        raise ContractError(
            f"{field_name} must have shape {(expected_size, expected_size)}, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ContractError(f"{field_name} must contain finite values")
    return matrix


def _optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


@dataclass(frozen=True)
class NativeRelationExportManifest:
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

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "writer": self.writer,
            "patient_index_path": None if self.patient_index_path is None else str(self.patient_index_path),
            "patient_arrays_path": None if self.patient_arrays_path is None else str(self.patient_arrays_path),
            "cohort_index_path": None if self.cohort_index_path is None else str(self.cohort_index_path),
            "cohort_arrays_path": None if self.cohort_arrays_path is None else str(self.cohort_arrays_path),
            "patient_index_sha256": self.patient_index_sha256,
            "patient_arrays_sha256": self.patient_arrays_sha256,
            "cohort_index_sha256": self.cohort_index_sha256,
            "cohort_arrays_sha256": self.cohort_arrays_sha256,
            "fit_status": self.fit_status,
            "implementation_tier": self.implementation_tier,
            "patient_count": self.patient_count,
            "patient_record_count": self.patient_record_count,
            "cohort_record_count": self.cohort_record_count,
            "k_states": self.k_states,
            "cohort_fit_status": self.cohort_fit_status,
        }


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
class NativeRelationExport:
    manifest: NativeRelationExportManifest
    state_ids: tuple[int, ...]
    patient_records: tuple[PatientRelationRecord, ...]
    cohort_records: tuple[CohortRelationRecord, ...]


def _resolve_shared_state_ids(patient_results: Sequence[PatientRelationResult]) -> tuple[int, ...]:
    shared_state_ids: tuple[int, ...] | None = None
    for patient_result in patient_results:
        if patient_result.fit_status != "ok":
            continue
        if patient_result.state_ids is None:
            raise ContractError("ok patient results require state_ids for native relation export")
        current_state_ids = tuple(int(state_id) for state_id in patient_result.state_ids)
        if shared_state_ids is None:
            shared_state_ids = current_state_ids
            continue
        if current_state_ids != shared_state_ids:
            raise ContractError("Patient native relation export requires one shared state axis")
    return shared_state_ids or ()


def _cohort_record_from_relation(
    relation: CohortRelation,
    *,
    record_id: int,
    expected_k_states: int,
) -> CohortRelationRecord:
    if relation.fit_status != "ok":
        return CohortRelationRecord(
            record_id=record_id,
            cohort_relation_id=str(relation.cohort_id),
            fit_status=str(relation.fit_status),
            support_n_patients=len(relation.support_patient_ids),
            support_patient_ids=tuple(str(patient_id) for patient_id in relation.support_patient_ids),
            dispersion=None if relation.dispersion is None else float(relation.dispersion),
            k_states=expected_k_states,
            template_A=None,
            template_d=None,
            template_e=None,
            metadata=dict(relation.metadata),
        )

    template_A = _finite_matrix(
        relation.A,
        expected_size=expected_k_states,
        field_name="CohortRelation.A",
    )
    template_d = _finite_vector(
        relation.d,
        expected_size=expected_k_states,
        field_name="CohortRelation.d",
    )
    template_e = _finite_vector(
        relation.e,
        expected_size=expected_k_states,
        field_name="CohortRelation.e",
    )
    return CohortRelationRecord(
        record_id=record_id,
        cohort_relation_id=str(relation.cohort_id),
        fit_status=str(relation.fit_status),
        support_n_patients=len(relation.support_patient_ids),
        support_patient_ids=tuple(str(patient_id) for patient_id in relation.support_patient_ids),
        dispersion=None if relation.dispersion is None else float(relation.dispersion),
        k_states=expected_k_states,
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        metadata=dict(relation.metadata),
    )


def _patient_record_from_result(
    patient_result: PatientRelationResult,
    *,
    record_id: int,
    expected_k_states: int,
) -> PatientRelationRecord:
    if patient_result.fit_status != "ok":
        return PatientRelationRecord(
            record_id=record_id,
            patient_id=str(patient_result.patient_id),
            fit_status=str(patient_result.fit_status),
            implementation_tier=str(patient_result.implementation_tier),
            k_states=expected_k_states,
            status_reason=_normalize_status_reason(patient_result),
            audit=(
                {}
                if patient_result.audit is None
                else {
                    "patient_id": str(patient_result.audit.patient_id),
                    "relation_status": str(patient_result.audit.relation_status),
                    "observation_fit_status": patient_result.audit.observation_fit_status,
                    "metadata": dict(patient_result.audit.metadata),
                }
            ),
            diagnostics=dict(patient_result.diagnostics),
        )

    A = _finite_matrix(patient_result.A, expected_size=expected_k_states, field_name="A")
    d = _finite_vector(patient_result.d, expected_size=expected_k_states, field_name="d")
    e = _finite_vector(patient_result.e, expected_size=expected_k_states, field_name="e")
    mu_minus = _finite_vector(
        patient_result.mu_minus,
        expected_size=expected_k_states,
        field_name="mu_minus",
    )
    mu_plus = _finite_vector(
        patient_result.mu_plus,
        expected_size=expected_k_states,
        field_name="mu_plus",
    )
    return PatientRelationRecord(
        record_id=record_id,
        patient_id=str(patient_result.patient_id),
        fit_status="ok",
        implementation_tier=str(patient_result.implementation_tier),
        k_states=expected_k_states,
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        status_reason=None,
        audit=(
            {}
            if patient_result.audit is None
            else {
                "patient_id": str(patient_result.audit.patient_id),
                "relation_status": str(patient_result.audit.relation_status),
                "observation_fit_status": patient_result.audit.observation_fit_status,
                "metadata": dict(patient_result.audit.metadata),
            }
        ),
        diagnostics=dict(patient_result.diagnostics),
    )


def validate_stride_native_relation_export(export: NativeRelationExport) -> None:
    """Validate native relation export schema, alignment, and status semantics."""
    manifest = export.manifest
    state_ids = tuple(int(state_id) for state_id in export.state_ids)
    if manifest.k_states != len(state_ids):
        raise ContractError("Native relation export manifest must agree with the shared state axis")
    if manifest.patient_record_count != len(export.patient_records):
        raise ContractError("Native relation export patient record count mismatch")
    if manifest.cohort_record_count != len(export.cohort_records):
        raise ContractError("Native relation export cohort record count mismatch")
    if manifest.patient_count != len({record.patient_id for record in export.patient_records}):
        raise ContractError("Native relation export patient_count must match unique patient ids")

    for expected_record_id, record in enumerate(export.patient_records):
        if record.record_id != expected_record_id:
            raise ContractError("Patient relation export record_id must be sequential")
        if record.k_states != len(state_ids):
            raise ContractError("Patient relation export must use the shared state axis")
        if record.fit_status == "ok":
            if any(array is None for array in (record.A, record.d, record.e, record.mu_minus, record.mu_plus)):
                raise ContractError("ok patient relation export records require realized arrays")
            _finite_matrix(record.A, expected_size=record.k_states, field_name="patient A")
            _finite_vector(record.d, expected_size=record.k_states, field_name="patient d")
            _finite_vector(record.e, expected_size=record.k_states, field_name="patient e")
            _finite_vector(
                record.mu_minus,
                expected_size=record.k_states,
                field_name="patient mu_minus",
            )
            _finite_vector(
                record.mu_plus,
                expected_size=record.k_states,
                field_name="patient mu_plus",
            )
        elif any(array is not None for array in (record.A, record.d, record.e, record.mu_minus, record.mu_plus)):
            raise ContractError("Non-ok patient relation export records must not carry realized arrays")

    for expected_record_id, record in enumerate(export.cohort_records):
        if record.record_id != expected_record_id:
            raise ContractError("Cohort relation export record_id must be sequential")
        if record.k_states != len(state_ids):
            raise ContractError("Cohort relation export must use the shared state axis")
        if record.fit_status == "ok":
            if any(array is None for array in (record.template_A, record.template_d, record.template_e)):
                raise ContractError("ok cohort relation export records require realized arrays")
            _finite_matrix(
                record.template_A,
                expected_size=record.k_states,
                field_name="cohort template_A",
            )
            _finite_vector(
                record.template_d,
                expected_size=record.k_states,
                field_name="cohort template_d",
            )
            _finite_vector(
                record.template_e,
                expected_size=record.k_states,
                field_name="cohort template_e",
            )
        elif any(array is not None for array in (record.template_A, record.template_d, record.template_e)):
            raise ContractError("Non-ok cohort relation export records must not carry realized arrays")

    if manifest.cohort_fit_status == "ok" and len(export.cohort_records) == 0:
        raise ContractError("ok cohort native relation export requires at least one cohort record")


def _write_csv(path: Path, *, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_relative_path(value: str | None, *, manifest_dir: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (manifest_dir / candidate).resolve()


def _build_native_relation_export(
    fit_result: STRIDEFitResult,
    *,
    export_version: str,
) -> NativeRelationExport:
    shared_state_ids = _resolve_shared_state_ids(fit_result.patient_results)
    k_states = len(shared_state_ids)
    patient_records = tuple(
        _patient_record_from_result(
            patient_result,
            record_id=record_id,
            expected_k_states=k_states,
        )
        for record_id, patient_result in enumerate(fit_result.patient_results)
    )
    cohort_records = (
        (
            _cohort_record_from_relation(
                fit_result.cohort_relation,
                record_id=0,
                expected_k_states=k_states,
            ),
        )
        if fit_result.cohort_relation.fit_status == "ok"
        else ()
    )
    return NativeRelationExport(
        manifest=NativeRelationExportManifest(
            schema_version=export_version,
            writer=STRIDE_NATIVE_RELATION_EXPORT_WRITER,
            manifest_path=None,
            patient_index_path=Path(PATIENT_INDEX_FILENAME),
            patient_arrays_path=Path(PATIENT_ARRAYS_FILENAME),
            cohort_index_path=Path(COHORT_INDEX_FILENAME),
            cohort_arrays_path=Path(COHORT_ARRAYS_FILENAME),
            patient_index_sha256=None,
            patient_arrays_sha256=None,
            cohort_index_sha256=None,
            cohort_arrays_sha256=None,
            fit_status=str(fit_result.fit_status),
            implementation_tier=str(fit_result.implementation_tier),
            patient_count=len(fit_result.patient_ids),
            patient_record_count=len(patient_records),
            cohort_record_count=len(cohort_records),
            k_states=k_states,
            cohort_fit_status=str(fit_result.cohort_relation.fit_status),
        ),
        state_ids=shared_state_ids,
        patient_records=patient_records,
        cohort_records=cohort_records,
    )


def _patient_relation_index_rows(
    patient_records: Sequence[PatientRelationRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_id, record in enumerate(patient_records):
        rows.append(
            {
                "record_id": record_id,
                "patient_id": record.patient_id,
                "fit_status": record.fit_status,
                "implementation_tier": record.implementation_tier,
                "k_states": record.k_states,
                "array_slot": "",
                "status_reason": "" if record.status_reason is None else record.status_reason,
                "audit_json": _json_dumps(dict(record.audit)),
                "diagnostics_json": _json_dumps(dict(record.diagnostics)),
            }
        )
    patient_slot = 0
    for row, record in zip(rows, patient_records, strict=True):
        if record.is_ok:
            row["array_slot"] = patient_slot
            patient_slot += 1
    return rows


def _cohort_relation_index_rows(
    cohort_records: Sequence[CohortRelationRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cohort_slot = 0
    for record_id, record in enumerate(cohort_records):
        row = {
            "record_id": record_id,
            "cohort_relation_id": record.cohort_relation_id,
            "fit_status": record.fit_status,
            "support_n_patients": int(record.support_n_patients),
            "support_patient_ids_json": _json_dumps(list(record.support_patient_ids)),
            "dispersion": (
                ""
                if record.dispersion is None
                else float(record.dispersion)
            ),
            "k_states": record.k_states,
            "array_slot": "",
            "metadata_json": _json_dumps(dict(record.metadata)),
        }
        if record.is_ok:
            row["array_slot"] = cohort_slot
            cohort_slot += 1
        rows.append(row)
    return rows


def _patient_relation_arrays(
    patient_records: Sequence[PatientRelationRecord],
    *,
    k_states: int,
    state_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    ok_records = [record for record in patient_records if record.is_ok]
    return {
        "A": (
            np.stack([record.A for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states, k_states), dtype=float)
        ),
        "d": (
            np.stack([record.d for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
        "e": (
            np.stack([record.e for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
        "mu_minus": (
            np.stack([record.mu_minus for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
        "mu_plus": (
            np.stack([record.mu_plus for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
        "state_ids": np.asarray(tuple(state_ids), dtype=int),
    }


def _cohort_relation_arrays(
    cohort_records: Sequence[CohortRelationRecord],
    *,
    k_states: int,
) -> dict[str, np.ndarray]:
    ok_records = [record for record in cohort_records if record.is_ok]
    return {
        "template_A": (
            np.stack([record.template_A for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states, k_states), dtype=float)
        ),
        "template_d": (
            np.stack([record.template_d for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
        "template_e": (
            np.stack([record.template_e for record in ok_records], axis=0)
            if ok_records
            else np.empty((0, k_states), dtype=float)
        ),
    }


def write_stride_native_relation_export(
    fit_result: STRIDEFitResult,
    output_dir: Path,
    *,
    export_version: str = STRIDE_NATIVE_RELATION_EXPORT_VERSION,
) -> NativeRelationExportManifest:
    """Write generic native relation files from package-returned fit outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _reject_existing_native_relation_targets(output_path)

    export = _build_native_relation_export(fit_result, export_version=export_version)
    validate_stride_native_relation_export(export)
    patient_rows = _patient_relation_index_rows(export.patient_records)
    cohort_rows = _cohort_relation_index_rows(export.cohort_records)
    patient_arrays = _patient_relation_arrays(
        export.patient_records,
        k_states=export.manifest.k_states,
        state_ids=export.state_ids,
    )
    cohort_arrays = _cohort_relation_arrays(
        export.cohort_records,
        k_states=export.manifest.k_states,
    )

    with tempfile.TemporaryDirectory(
        dir=output_path,
        prefix=".stride_native_relation_export_",
        suffix=".tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        patient_index_path = temp_dir / PATIENT_INDEX_FILENAME
        patient_arrays_path = temp_dir / PATIENT_ARRAYS_FILENAME
        cohort_index_path = temp_dir / COHORT_INDEX_FILENAME
        cohort_arrays_path = temp_dir / COHORT_ARRAYS_FILENAME
        manifest_path = temp_dir / MANIFEST_FILENAME

        _write_csv(patient_index_path, columns=PATIENT_INDEX_COLUMNS, rows=patient_rows)
        with patient_arrays_path.open("wb") as handle:
            np.savez(handle, **patient_arrays)
        _write_csv(cohort_index_path, columns=COHORT_INDEX_COLUMNS, rows=cohort_rows)
        with cohort_arrays_path.open("wb") as handle:
            np.savez(handle, **cohort_arrays)

        manifest = NativeRelationExportManifest(
            schema_version=export_version,
            writer=STRIDE_NATIVE_RELATION_EXPORT_WRITER,
            manifest_path=output_path / MANIFEST_FILENAME,
            patient_index_path=output_path / PATIENT_INDEX_FILENAME,
            patient_arrays_path=output_path / PATIENT_ARRAYS_FILENAME,
            cohort_index_path=output_path / COHORT_INDEX_FILENAME,
            cohort_arrays_path=output_path / COHORT_ARRAYS_FILENAME,
            patient_index_sha256=sha256_file(patient_index_path),
            patient_arrays_sha256=sha256_file(patient_arrays_path),
            cohort_index_sha256=sha256_file(cohort_index_path),
            cohort_arrays_sha256=sha256_file(cohort_arrays_path),
            fit_status=str(fit_result.fit_status),
            implementation_tier=str(fit_result.implementation_tier),
            patient_count=export.manifest.patient_count,
            patient_record_count=export.manifest.patient_record_count,
            cohort_record_count=export.manifest.cohort_record_count,
            k_states=export.manifest.k_states,
            cohort_fit_status=str(fit_result.cohort_relation.fit_status),
        )
        manifest_path.write_text(
            json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        _reject_existing_native_relation_targets(output_path)
        (temp_dir / PATIENT_INDEX_FILENAME).replace(output_path / PATIENT_INDEX_FILENAME)
        (temp_dir / PATIENT_ARRAYS_FILENAME).replace(output_path / PATIENT_ARRAYS_FILENAME)
        (temp_dir / COHORT_INDEX_FILENAME).replace(output_path / COHORT_INDEX_FILENAME)
        (temp_dir / COHORT_ARRAYS_FILENAME).replace(output_path / COHORT_ARRAYS_FILENAME)
        manifest_path.replace(output_path / MANIFEST_FILENAME)

    resolved_manifest_path = output_path / MANIFEST_FILENAME
    return NativeRelationExportManifest(
        schema_version=manifest.schema_version,
        writer=manifest.writer,
        manifest_path=resolved_manifest_path,
        patient_index_path=output_path / PATIENT_INDEX_FILENAME,
        patient_arrays_path=output_path / PATIENT_ARRAYS_FILENAME,
        cohort_index_path=output_path / COHORT_INDEX_FILENAME,
        cohort_arrays_path=output_path / COHORT_ARRAYS_FILENAME,
        patient_index_sha256=manifest.patient_index_sha256,
        patient_arrays_sha256=manifest.patient_arrays_sha256,
        cohort_index_sha256=manifest.cohort_index_sha256,
        cohort_arrays_sha256=manifest.cohort_arrays_sha256,
        fit_status=manifest.fit_status,
        implementation_tier=manifest.implementation_tier,
        patient_count=manifest.patient_count,
        patient_record_count=manifest.patient_record_count,
        cohort_record_count=manifest.cohort_record_count,
        k_states=manifest.k_states,
        cohort_fit_status=manifest.cohort_fit_status,
        manifest_sha256=sha256_file(resolved_manifest_path),
    )


def _read_native_relation_manifest(resolved_manifest_path: Path) -> NativeRelationExportManifest:
    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    manifest_dir = resolved_manifest_path.parent
    manifest = NativeRelationExportManifest(
        schema_version=str(payload["schema_version"]),
        writer=str(payload["writer"]),
        manifest_path=resolved_manifest_path,
        patient_index_path=_resolve_relative_path(
            payload.get("patient_index_path"), manifest_dir=manifest_dir
        ),
        patient_arrays_path=_resolve_relative_path(
            payload.get("patient_arrays_path"), manifest_dir=manifest_dir
        ),
        cohort_index_path=_resolve_relative_path(
            payload.get("cohort_index_path"), manifest_dir=manifest_dir
        ),
        cohort_arrays_path=_resolve_relative_path(
            payload.get("cohort_arrays_path"), manifest_dir=manifest_dir
        ),
        patient_index_sha256=payload.get("patient_index_sha256"),
        patient_arrays_sha256=payload.get("patient_arrays_sha256"),
        cohort_index_sha256=payload.get("cohort_index_sha256"),
        cohort_arrays_sha256=payload.get("cohort_arrays_sha256"),
        fit_status=str(payload["fit_status"]),
        implementation_tier=str(payload["implementation_tier"]),
        patient_count=int(payload["patient_count"]),
        patient_record_count=int(payload["patient_record_count"]),
        cohort_record_count=int(payload["cohort_record_count"]),
        k_states=int(payload["k_states"]),
        cohort_fit_status=str(payload["cohort_fit_status"]),
        manifest_sha256=sha256_file(resolved_manifest_path),
    )
    if manifest.schema_version != STRIDE_NATIVE_RELATION_EXPORT_VERSION:
        raise ContractError(
            "Native relation export manifest schema_version does not match "
            f"{STRIDE_NATIVE_RELATION_EXPORT_VERSION!r}"
        )
    if manifest.writer != STRIDE_NATIVE_RELATION_EXPORT_WRITER:
        raise ContractError("Native relation export manifest writer does not match the schema")
    return manifest


def _validate_native_relation_manifest_hashes(manifest: NativeRelationExportManifest) -> None:
    hash_checks = (
        (manifest.patient_index_path, manifest.patient_index_sha256, "patient index"),
        (manifest.patient_arrays_path, manifest.patient_arrays_sha256, "patient arrays"),
        (manifest.cohort_index_path, manifest.cohort_index_sha256, "cohort index"),
        (manifest.cohort_arrays_path, manifest.cohort_arrays_sha256, "cohort arrays"),
    )
    for path, expected_hash, label in hash_checks:
        if path is None or expected_hash is None:
            raise ContractError(f"Native relation export {label} hash metadata is required")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ContractError(f"Native relation export {label} SHA-256 mismatch")


def _read_index_rows(path: Path, *, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ContractError(f"Native relation export {label} columns do not match the schema")
        return [dict(row) for row in reader]


def _read_patient_arrays(path: Path) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        state_ids = tuple(int(state_id) for state_id in np.asarray(payload["state_ids"], dtype=int).tolist())
        arrays = {
            "A": np.asarray(payload["A"], dtype=float),
            "d": np.asarray(payload["d"], dtype=float),
            "e": np.asarray(payload["e"], dtype=float),
            "mu_minus": np.asarray(payload["mu_minus"], dtype=float),
            "mu_plus": np.asarray(payload["mu_plus"], dtype=float),
        }
    return state_ids, arrays


def _read_cohort_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "template_A": np.asarray(payload["template_A"], dtype=float),
            "template_d": np.asarray(payload["template_d"], dtype=float),
            "template_e": np.asarray(payload["template_e"], dtype=float),
        }


def _patient_records_from_rows(
    patient_rows: Sequence[Mapping[str, str]],
    patient_arrays: Mapping[str, np.ndarray],
) -> tuple[PatientRelationRecord, ...]:
    records: list[PatientRelationRecord] = []
    for row in patient_rows:
        array_slot = _optional_array_slot(
            row["array_slot"],
            n_slots=int(patient_arrays["A"].shape[0]),
            label="patient relation export",
        )
        records.append(
            PatientRelationRecord(
                record_id=int(row["record_id"]),
                patient_id=str(row["patient_id"]),
                fit_status=str(row["fit_status"]),
                implementation_tier=str(row["implementation_tier"]),
                k_states=int(row["k_states"]),
                A=None if array_slot is None else patient_arrays["A"][array_slot],
                d=None if array_slot is None else patient_arrays["d"][array_slot],
                e=None if array_slot is None else patient_arrays["e"][array_slot],
                mu_minus=None if array_slot is None else patient_arrays["mu_minus"][array_slot],
                mu_plus=None if array_slot is None else patient_arrays["mu_plus"][array_slot],
                status_reason=(
                    None
                    if str(row["status_reason"]).strip() == ""
                    else str(row["status_reason"]).strip()
                ),
                audit=_json_loads(str(row["audit_json"])),
                diagnostics=_json_loads(str(row["diagnostics_json"])),
            )
        )
    return tuple(records)


def _cohort_records_from_rows(
    cohort_rows: Sequence[Mapping[str, str]],
    cohort_arrays: Mapping[str, np.ndarray],
) -> tuple[CohortRelationRecord, ...]:
    records: list[CohortRelationRecord] = []
    for row in cohort_rows:
        array_slot = _optional_array_slot(
            row["array_slot"],
            n_slots=int(cohort_arrays["template_A"].shape[0]),
            label="cohort relation export",
        )
        support_patient_ids_json = json.loads(str(row["support_patient_ids_json"]))
        if not isinstance(support_patient_ids_json, list):
            raise ContractError("Cohort relation export support_patient_ids_json must be a list")
        records.append(
            CohortRelationRecord(
                record_id=int(row["record_id"]),
                cohort_relation_id=str(row["cohort_relation_id"]),
                fit_status=str(row["fit_status"]),
                support_n_patients=int(row["support_n_patients"]),
                support_patient_ids=tuple(str(patient_id) for patient_id in support_patient_ids_json),
                dispersion=_optional_float(str(row["dispersion"])),
                k_states=int(row["k_states"]),
                template_A=None if array_slot is None else cohort_arrays["template_A"][array_slot],
                template_d=None if array_slot is None else cohort_arrays["template_d"][array_slot],
                template_e=None if array_slot is None else cohort_arrays["template_e"][array_slot],
                metadata=_json_loads(str(row["metadata_json"])),
            )
        )
    return tuple(records)


def read_stride_native_relation_export(
    manifest_path: Path,
    *,
    validate_hashes: bool = True,
) -> NativeRelationExport:
    """Read generic native relation files and return typed in-memory records."""
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = _read_native_relation_manifest(resolved_manifest_path)

    if validate_hashes:
        _validate_native_relation_manifest_hashes(manifest)

    if manifest.patient_index_path is None or manifest.patient_arrays_path is None:
        raise ContractError("Native relation export requires patient artifact paths")
    if manifest.cohort_index_path is None or manifest.cohort_arrays_path is None:
        raise ContractError("Native relation export requires cohort artifact paths")

    patient_rows = _read_index_rows(
        manifest.patient_index_path,
        columns=PATIENT_INDEX_COLUMNS,
        label="patient index",
    )
    cohort_rows = _read_index_rows(
        manifest.cohort_index_path,
        columns=COHORT_INDEX_COLUMNS,
        label="cohort index",
    )
    state_ids, patient_arrays = _read_patient_arrays(manifest.patient_arrays_path)
    cohort_arrays = _read_cohort_arrays(manifest.cohort_arrays_path)

    export = NativeRelationExport(
        manifest=manifest,
        state_ids=state_ids,
        patient_records=_patient_records_from_rows(patient_rows, patient_arrays),
        cohort_records=_cohort_records_from_rows(cohort_rows, cohort_arrays),
    )
    validate_stride_native_relation_export(export)
    return export


__all__ = [
    "COHORT_ARRAYS_FILENAME",
    "COHORT_INDEX_FILENAME",
    "CohortRelationRecord",
    "MANIFEST_FILENAME",
    "NativeRelationExport",
    "NativeRelationExportManifest",
    "PATIENT_ARRAYS_FILENAME",
    "PATIENT_INDEX_FILENAME",
    "PatientRelationRecord",
    "STRIDE_NATIVE_RELATION_EXPORT_VERSION",
    "read_stride_native_relation_export",
    "sha256_file",
    "validate_stride_native_relation_export",
    "write_stride_native_relation_export",
]
