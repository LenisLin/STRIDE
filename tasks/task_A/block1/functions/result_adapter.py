"""Adapters from live `.tl.FitResult` objects to Block 1 execute artifacts."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from stride.errors import ContractError
from stride.tl import FitResult

from ...workflows.fit_adapter import (
    derive_task_a_patient_side_compositions,
    extract_task_a_relations,
)
from .native_export import (
    Block1RelationExport,
    Block1RelationExportManifest,
    CohortRelationRecord,
    PatientRelationRecord,
)

BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME = "block1_stride_tl_fit_manifest.json"
BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME = "block1_stride_tl_fit_arrays.npz"


def block1_arrays_from_fit_result(result: FitResult) -> dict[str, object]:
    """Extract the minimum Block 1 array payload available from `.tl.FitResult`."""
    extracted = extract_task_a_relations(result)
    payload: dict[str, object] = {
        "relation_ids": tuple(relation.relation_id for relation in extracted),
        "relations": {},
    }
    relations: dict[str, object] = {}
    for relation in extracted:
        relations[relation.relation_id] = {
            "patient_ids": relation.patient_ids,
            "A": relation.A.copy(),
            "d": relation.d.copy(),
            "e": relation.e.copy(),
            "cohort": relation.cohort,
        }
    payload["relations"] = relations
    return payload


def write_block1_native_fit_result(
    result: FitResult,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write a small TaskA-native manifest and array archive for `.tl.FitResult`."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME
    arrays_path = output_path / BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME
    existing = tuple(path for path in (manifest_path, arrays_path) if path.exists())
    if existing:
        raise ContractError(
            "Block 1 native FitResult export already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )

    arrays_payload = block1_arrays_from_fit_result(result)
    relations = arrays_payload["relations"]
    if not isinstance(relations, dict):
        raise ContractError("Block 1 native FitResult adapter produced invalid relations payload")

    npz_payload: dict[str, np.ndarray] = {}
    relation_records: list[dict[str, Any]] = []
    patient_count_set: set[str] = set()
    npz_payload["state_ids"] = np.arange(int(result.n_states), dtype=int)
    for relation_id, relation_payload in relations.items():
        if not isinstance(relation_payload, dict):
            raise ContractError("Block 1 native FitResult relation payload must be a mapping")
        patient_ids = tuple(str(value) for value in relation_payload["patient_ids"])
        patient_count_set.update(patient_ids)
        safe_relation_id = str(relation_id).replace("/", "_")
        npz_payload[f"{safe_relation_id}__A"] = np.asarray(relation_payload["A"], dtype=float)
        npz_payload[f"{safe_relation_id}__d"] = np.asarray(relation_payload["d"], dtype=float)
        npz_payload[f"{safe_relation_id}__e"] = np.asarray(relation_payload["e"], dtype=float)
        relation_records.append(
            {
                "relation_id": str(relation_id),
                "patient_ids": list(patient_ids),
                "patient_count": len(patient_ids),
                "A_key": f"{safe_relation_id}__A",
                "d_key": f"{safe_relation_id}__d",
                "e_key": f"{safe_relation_id}__e",
                "cohort_present": result.relations[str(relation_id)].cohort is not None,
            }
        )

    with tempfile.TemporaryDirectory(
        dir=output_path,
        prefix=".block1_stride_tl_fit_",
        suffix=".tmp",
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_arrays_path = temp_dir / BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME
        temp_manifest_path = temp_dir / BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME
        with temp_arrays_path.open("wb") as handle:
            np.savez(handle, **npz_payload)
        manifest = {
            "schema_version": "task_a_block1_stride_tl_fit_export_v1",
            "fit_surface": "stride.tl.fit",
            "source": result.source,
            "target": result.target,
            "n_states": int(result.n_states),
            "relation_count": int(len(result.relation_ids)),
            "relation_ids": list(result.relation_ids),
            "patient_count": len(patient_count_set),
            "relations": relation_records,
            "arrays_path": str(arrays_path),
            "arrays_sha256": sha256_file(temp_arrays_path),
        }
        temp_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_arrays_path.replace(arrays_path)
        temp_manifest_path.replace(manifest_path)

    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "arrays_path": arrays_path,
        "arrays_sha256": sha256_file(arrays_path),
        "patient_count": len(patient_count_set),
        "patient_record_count": sum(record["patient_count"] for record in relation_records),
        "cohort_record_count": sum(1 for record in relation_records if record["cohort_present"]),
        "k_states": int(result.n_states),
    }


def read_block1_stride_tl_native_export(
    manifest_path: str | Path,
    *,
    source_adata: Any,
    manifest_sha256: str | None = None,
) -> Block1RelationExport:
    """Read a TaskA-native `.tl.fit` export as the Block 1 analysis record shape."""
    resolved_manifest_path = Path(manifest_path).resolve()
    if manifest_sha256 is not None and sha256_file(resolved_manifest_path) != manifest_sha256:
        raise ContractError("Block 1 native FitResult manifest SHA-256 mismatch")
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("schema_version")) != "task_a_block1_stride_tl_fit_export_v1":
        raise ContractError("Block 1 analyze expected a task_a_block1_stride_tl_fit_export_v1 manifest")
    if str(manifest.get("fit_surface")) != "stride.tl.fit":
        raise ContractError("Block 1 analyze expected fit_surface='stride.tl.fit'")

    arrays_path = Path(str(manifest["arrays_path"])).resolve()
    if sha256_file(arrays_path) != str(manifest["arrays_sha256"]):
        raise ContractError("Block 1 native FitResult arrays SHA-256 mismatch")

    patient_records: list[PatientRelationRecord] = []
    cohort_records: list[CohortRelationRecord] = []
    relation_records = list(manifest.get("relations", ()))
    if len(relation_records) != 1:
        raise ContractError("Block 1 task-native export expects exactly one relation per family")
    fitted_patient_ids = tuple(
        str(patient_id)
        for relation_payload in relation_records
        if isinstance(relation_payload, dict)
        for patient_id in relation_payload.get("patient_ids", ())
    )
    burdens = derive_task_a_patient_side_compositions(
        source_adata,
        required_patient_ids=tuple(dict.fromkeys(fitted_patient_ids)),
    )

    with np.load(arrays_path, allow_pickle=False) as archive:
        for relation_index, relation_payload in enumerate(relation_records):
            if not isinstance(relation_payload, dict):
                raise ContractError("Block 1 relation manifest entries must be mappings")
            patient_ids = tuple(str(value) for value in relation_payload["patient_ids"])
            A = np.asarray(archive[str(relation_payload["A_key"])], dtype=float)
            d = np.asarray(archive[str(relation_payload["d_key"])], dtype=float)
            e = np.asarray(archive[str(relation_payload["e_key"])], dtype=float)
            if A.ndim != 3 or d.ndim != 2 or e.ndim != 2:
                raise ContractError("Block 1 task-native arrays must be A[P,K,K], d[P,K], e[P,K]")
            if A.shape[0] != len(patient_ids) or d.shape[0] != len(patient_ids) or e.shape[0] != len(patient_ids):
                raise ContractError("Block 1 task-native arrays do not align with patient_ids")
            if A.shape[1] != A.shape[2] or d.shape[1] != A.shape[1] or e.shape[1] != A.shape[1]:
                raise ContractError("Block 1 task-native arrays must share one K-state axis")
            k_states = int(A.shape[1])
            for patient_index, patient_id in enumerate(patient_ids):
                if patient_id not in burdens:
                    raise ContractError(
                        f"Block 1 source AnnData lacks burden vectors for patient {patient_id!r}"
                    )
                mu_minus, mu_plus = burdens[patient_id]
                patient_records.append(
                    PatientRelationRecord(
                        record_id=len(patient_records),
                        patient_id=patient_id,
                        fit_status="ok",
                        implementation_tier="canonical_stride_tl",
                        k_states=k_states,
                        A=A[patient_index].copy(),
                        d=d[patient_index].copy(),
                        e=e[patient_index].copy(),
                        mu_minus=np.asarray(mu_minus, dtype=float).copy(),
                        mu_plus=np.asarray(mu_plus, dtype=float).copy(),
                    )
                )
            cohort_records.append(
                CohortRelationRecord(
                    record_id=relation_index,
                    cohort_relation_id=str(relation_payload["relation_id"]),
                    fit_status="ok" if bool(relation_payload.get("cohort_present", False)) else "not_available",
                    support_n_patients=len(patient_ids),
                    support_patient_ids=patient_ids,
                    dispersion=None,
                    k_states=k_states,
                    template_A=np.mean(A, axis=0, dtype=float),
                    template_d=np.mean(d, axis=0, dtype=float),
                    template_e=np.mean(e, axis=0, dtype=float),
                    metadata={"fit_surface": "stride.tl.fit"},
                )
            )

    export_manifest = Block1RelationExportManifest(
        schema_version=str(manifest["schema_version"]),
        writer="tasks.task_A.block1.functions.result_adapter",
        manifest_path=resolved_manifest_path,
        patient_index_path=None,
        patient_arrays_path=arrays_path,
        cohort_index_path=None,
        cohort_arrays_path=arrays_path,
        patient_index_sha256=None,
        patient_arrays_sha256=str(manifest["arrays_sha256"]),
        cohort_index_sha256=None,
        cohort_arrays_sha256=str(manifest["arrays_sha256"]),
        fit_status="ok",
        implementation_tier="canonical_stride_tl",
        patient_count=int(manifest["patient_count"]),
        patient_record_count=len(patient_records),
        cohort_record_count=len(cohort_records),
        k_states=int(manifest["n_states"]),
        cohort_fit_status="ok" if cohort_records else "not_available",
        manifest_sha256=sha256_file(resolved_manifest_path),
    )
    state_ids = tuple(range(int(manifest["n_states"])))
    return Block1RelationExport(
        manifest=export_manifest,
        state_ids=state_ids,
        patient_records=tuple(patient_records),
        cohort_records=tuple(cohort_records),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BLOCK1_NATIVE_RESULT_ARRAYS_FILENAME",
    "BLOCK1_NATIVE_RESULT_MANIFEST_FILENAME",
    "block1_arrays_from_fit_result",
    "read_block1_stride_tl_native_export",
    "sha256_file",
    "write_block1_native_fit_result",
]
