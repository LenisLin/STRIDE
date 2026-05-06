"""Build an objective Task A result packet for human review.

The packet is intentionally non-interpretive. It mirrors currently available
Task A artifact surfaces into a single repo-local directory, records
provenance, and explicitly lists missing downstream surfaces instead of
inventing them. Block 3 is currently deferred from this packet because the
active Block 3 engineering surface has been removed pending a clean rebuild.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from stride.errors import ContractError

from .block0.schemas import (
    BLOCK0_ANALYSIS_SPEC_VERSION,
    CALIBRATION_MANIFEST_FILENAME,
    CALIBRATION_READY_STATUS,
    DIAGNOSTIC_READINESS_STATUS,
    MANIFEST_REQUIRED_FIELDS as BLOCK0_CALIBRATION_MANIFEST_REQUIRED_FIELDS,
    METRIC_SUMMARY_COLUMNS as BLOCK0_METRIC_SUMMARY_COLUMNS,
    METRIC_SUMMARY_FILENAME,
    NULL_FAMILY as BLOCK0_NULL_FAMILY,
    PATIENT_CALIBRATION_COLUMNS as BLOCK0_PATIENT_CALIBRATION_COLUMNS,
    PATIENT_CALIBRATION_FILENAME,
    REAL_FAMILY as BLOCK0_REAL_FAMILY,
)
from .block0.writers import validate_block0_frame_columns
from .block2.review import write_block2_review_surface


PACKET_ROLE = "objective_task_a_result_packet"
PACKET_SPEC_VERSION = "task_a_result_packet_v1"
PACKET_MANIFEST_FILENAME = "task_a_result_packet_manifest.json"
PACKET_INDEX_FILENAME = "task_a_result_packet_index.csv"
HUMAN_INDEX_FILENAME = "RESULTS_INDEX.md"
BLOCK1_CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "artifact_contracts.md"
BLOCK2_CONTRACT_PATH = BLOCK1_CONTRACT_PATH
BLOCK3_PACKET_DEFERRED_MESSAGE = (
    "Block 3 packet integration is deferred / non-authority / pending clean bridge spec"
)
BLOCK0_CALIBRATION_CONTEXT = "calibration_context"
BLOCK0_ALLOWED_READINESS_STATUSES = {
    CALIBRATION_READY_STATUS,
    DIAGNOSTIC_READINESS_STATUS,
}
BLOCK0_NONEMPTY_SOURCE_FIELDS = (
    "source_execution_manifest_path",
    "source_fit_cache_path",
    "source_fit_cache_index_path",
    "source_fit_cache_sha256",
    "source_fit_cache_index_sha256",
    "patient_calibration_path",
    "metric_summary_path",
)

INDEX_COLUMNS: tuple[str, ...] = (
    "layer",
    "artifact_name",
    "expected_relative_path",
    "packet_relative_path",
    "source_path",
    "artifact_kind",
    "artifact_status",
    "contract_alignment",
    "implementation_tier",
    "evidence_lineage",
    "format",
    "n_rows",
    "n_columns",
    "observed_columns",
    "rows_represent",
    "columns_represent",
    "claim_scope",
    "review_role",
    "analysis_level",
    "family_surface_role",
    "is_proof_carrying",
    "proof_carrying_status",
    "source_workflow",
    "source_manifest_or_bundle",
    "sha256",
    "notes",
    "review_rank",
)

LAYER_ORDER: tuple[str, ...] = ("atlas", "block0", "block1", "block2", "block3")
STEP3_DEFERRED_LAYERS: tuple[str, ...] = ("block3",)


@dataclass(frozen=True)
class TaskAResultPacket:
    packet_root: Path
    manifest_path: Path
    index_path: Path
    human_index_path: Path
    layer_manifest_paths: dict[str, Path]
    layer_review_index_paths: dict[str, Path]


@dataclass(frozen=True)
class ArtifactPlan:
    layer: str
    artifact_name: str
    expected_relative_path: str
    packet_relative_path: str | None
    source_path: Path | None
    artifact_kind: str
    artifact_status: str
    contract_alignment: str
    format: str
    rows_represent: str
    columns_represent: str
    claim_scope: str
    proof_carrying_status: str
    source_workflow: str
    source_manifest_or_bundle: str
    notes: str
    review_rank: int
    review_role: str = "provenance"
    analysis_level: str = "mixed"
    family_surface_role: str = "not_applicable"


def _included_layers(*, include_block3: bool) -> tuple[str, ...]:
    _ = include_block3
    return tuple(layer for layer in LAYER_ORDER if layer != "block3")


def _deferred_layers(*, include_block3: bool) -> tuple[str, ...]:
    _ = include_block3
    return STEP3_DEFERRED_LAYERS


def _surface_lineage_summary(
    *,
    atlas_manifest: Path,
    prepare_manifest: Path,
    block0_calibration_manifest: Path,
    block1_bundle: Path | None,
    block2_manifest: Path | None,
) -> dict[str, dict[str, Any]]:
    atlas_payload = _load_json_dict(atlas_manifest, label="Task A atlas manifest")
    prepare_payload = _load_json_dict(prepare_manifest, label="Task A prepare manifest")
    block0_payload = _load_block0_calibration_manifest(block0_calibration_manifest)
    lineage: dict[str, dict[str, Any]] = {
        "atlas": {
            "atlas_role": str(atlas_payload.get("atlas_role", "")),
            "claim_scope": str(atlas_payload.get("claim_scope", "")),
            "stage0_h5ad": str(atlas_payload.get("stage0_h5ad", "")),
            "output_index": str(atlas_payload.get("output_index", "")),
        },
        "prepare": {
            "implementation_tier": str(prepare_payload.get("implementation_tier", "")),
            "evidence_lineage": str(prepare_payload.get("evidence_lineage", "")),
            "fit_surface": str(prepare_payload.get("fit_surface", "")),
        },
        "block0": {
            "implementation_tier": "canonical_full",
            "evidence_lineage": BLOCK0_CALIBRATION_CONTEXT,
            "fit_surface": "fit_stride",
            "analysis_spec_version": str(block0_payload["analysis_spec_version"]),
            "readiness_status": str(block0_payload["readiness_status"]),
            "real_family": str(block0_payload["real_family"]),
            "null_family": str(block0_payload["null_family"]),
        },
    }
    if block1_bundle is not None:
        block1_payload = _load_json_dict(block1_bundle, label="Task A Block 1 bundle")
        lineage["block1"] = {
            "implementation_tier": str(block1_payload.get("implementation_tier", "")),
            "evidence_lineage": str(block1_payload.get("evidence_lineage", "")),
            "fit_surface": str(block1_payload.get("fit_surface", "")),
            "cohort_recurrence_fit_status": str(
                block1_payload.get("cohort_recurrence_fit_status", "")
            ),
        }
    if block2_manifest is not None:
        block2_payload = _load_json_dict(block2_manifest, label="Task A Block 2 manifest")
        lineage["block2"] = {
            "implementation_tier": str(block2_payload.get("implementation_tier", "")),
            "evidence_lineage": str(block2_payload.get("evidence_lineage", "")),
            "fit_surface": str(block2_payload.get("fit_surface", "")),
            "upstream_block1_cohort_recurrence_fit_status": str(
                block2_payload.get("block1_cohort_recurrence_fit_status", "")
            ),
        }
    return lineage


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_json_dict(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {resolved}")
    return payload


def _artifact_lineage_from_json(path: Path | None) -> dict[str, str]:
    if path is None or path.suffix.lower() != ".json" or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    implementation_tier = str(payload.get("implementation_tier", ""))
    evidence_lineage = str(payload.get("evidence_lineage", ""))
    if implementation_tier == "" and evidence_lineage == "":
        return {}
    return {
        "implementation_tier": implementation_tier,
        "evidence_lineage": evidence_lineage,
    }


def _resolve_artifact_lineage(
    *,
    layer: str,
    relative_path: str,
    source_path: Path | None,
    surface_lineage: dict[str, dict[str, Any]],
) -> dict[str, str]:
    json_lineage = _artifact_lineage_from_json(source_path)
    if json_lineage:
        return json_lineage

    normalized = relative_path.lstrip("./")
    if "provenance/prepare/" in normalized or Path(normalized).name in {
        "task_a_prepare_manifest.json",
        "task_a_stride_mapping.json",
        "task_a_core_fit_dry_run.csv",
    }:
        lineage = surface_lineage.get("prepare", {})
    else:
        lineage = surface_lineage.get(layer, {})

    return {
        "implementation_tier": str(lineage.get("implementation_tier", "")),
        "evidence_lineage": str(lineage.get("evidence_lineage", "")),
    }


def _require_fields(payload: dict[str, Any], *, required_fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ContractError(f"{label} is missing required fields: {missing}")


def _require_nonempty_fields(payload: dict[str, Any], *, fields: tuple[str, ...], label: str) -> None:
    empty = [field for field in fields if str(payload.get(field, "")).strip() == ""]
    if empty:
        raise ContractError(f"{label} has empty required provenance fields: {empty}")


def _load_block0_calibration_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json_dict(path, label="Block 0 calibration manifest")
    _require_fields(
        payload,
        required_fields=BLOCK0_CALIBRATION_MANIFEST_REQUIRED_FIELDS,
        label="Block 0 calibration manifest",
    )
    _require_nonempty_fields(
        payload,
        fields=BLOCK0_NONEMPTY_SOURCE_FIELDS,
        label="Block 0 calibration manifest",
    )

    expected_values = {
        "analysis_spec_version": BLOCK0_ANALYSIS_SPEC_VERSION,
        "real_family": BLOCK0_REAL_FAMILY,
        "null_family": BLOCK0_NULL_FAMILY,
    }
    for field_name, expected_value in expected_values.items():
        observed_value = str(payload.get(field_name, ""))
        if observed_value != expected_value:
            raise ContractError(
                "Block 0 calibration manifest has incompatible "
                f"{field_name}: expected {expected_value!r}, got {observed_value!r}"
            )

    readiness_status = str(payload.get("readiness_status", ""))
    if readiness_status not in BLOCK0_ALLOWED_READINESS_STATUSES:
        raise ContractError(
            "Block 0 calibration manifest has incompatible readiness_status: "
            f"expected one of {sorted(BLOCK0_ALLOWED_READINESS_STATUSES)}, got {readiness_status!r}"
        )
    return payload


def _validate_block0_calibration_table(
    path: Path,
    *,
    expected_columns: tuple[str, ...],
    label: str,
) -> None:
    validate_block0_frame_columns(
        pd.read_csv(path, nrows=0),
        expected_columns,
        label=label,
    )


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_signature(path: str | Path) -> tuple[int, int, str]:
    resolved = _resolve_path(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0, ""
        row_count = sum(1 for _ in reader)
    return row_count, len(header), "|".join(str(column) for column in header)


def _json_signature(path: str | Path) -> tuple[int | str, int | str, str]:
    payload = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        keys = list(payload.keys())
        return 1, len(keys), "|".join(str(key) for key in keys)
    if isinstance(payload, list):
        keys: list[str] = []
        if payload and isinstance(payload[0], dict):
            keys = list(payload[0].keys())
        return len(payload), len(keys), "|".join(str(key) for key in keys)
    return "", "", ""


def _observe_artifact(path: str | Path) -> tuple[int | str, int | str, str]:
    suffix = _resolve_path(path).suffix.lower()
    if suffix == ".csv":
        return _csv_signature(path)
    if suffix == ".json":
        return _json_signature(path)
    return "", "", ""


def _proof_carrying_bool(status: str) -> bool:
    return status in {"all", "partial"}


def _metadata(
    *,
    artifact_kind: str,
    format_name: str,
    rows_represent: str,
    columns_represent: str,
    claim_scope: str,
    proof_carrying_status: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "format": format_name,
        "rows_represent": rows_represent,
        "columns_represent": columns_represent,
        "claim_scope": claim_scope,
        "proof_carrying_status": proof_carrying_status,
        "notes": notes,
    }


def _atlas_metadata(relative_path: str, *, category: str, artifact_kind: str, format_name: str) -> dict[str, Any]:
    name = Path(relative_path).name
    if name == "task_a_descriptive_atlas_manifest.json":
        return _metadata(
            artifact_kind="manifest",
            format_name="json",
            rows_represent="Single JSON object for one descriptive atlas export.",
            columns_represent="Top-level keys record atlas role, Stage 0 field keys, counts, and the atlas output index path.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "task_a_descriptive_atlas_output_index.csv":
        return _metadata(
            artifact_kind="index",
            format_name="csv",
            rows_represent="One row per atlas table or figure written by the descriptive-atlas workflow.",
            columns_represent="Columns identify each artifact's relative path, kind, category, format, and short description.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_cell_subtype_counts.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent atlas communities.",
            columns_represent="Columns represent cell-subtype count totals within each community.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_cell_subtype_row_fractions.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent atlas communities.",
            columns_represent="Columns represent within-community cell-subtype fractions.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_domain_distribution.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent community-by-domain summary rows.",
            columns_represent="Columns record cell counts and within-community / within-domain fractions across TC, IM, and PT.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_domain_roi_prevalence.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent community-by-domain ROI prevalence summaries.",
            columns_represent="Columns record positive ROI counts, total ROI counts, and ROI prevalence by domain.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_patient_occurrence_summary.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent atlas communities.",
            columns_represent="Columns summarize patient prevalence, ROI prevalence, and positive-patient cell burden per community.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "community_patient_occurrence_matrix.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent atlas communities.",
            columns_represent="Columns represent patients in a binary community-by-patient occurrence matrix.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if name == "representative_overlay_selection.csv":
        return _metadata(
            artifact_kind="table",
            format_name="csv",
            rows_represent="Rows represent one deterministic ROI selection per displayed overlay community.",
            columns_represent="Columns record patient/domain/ROI identity, community burden in the ROI, and the linked overlay path.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    if artifact_kind == "figure" or category == "representative_spatial_overlays":
        return _metadata(
            artifact_kind="figure",
            format_name=format_name,
            rows_represent="Not tabular; each file is one atlas figure.",
            columns_represent="SVG graphical elements only.",
            claim_scope="descriptive",
            proof_carrying_status="none",
        )
    return _metadata(
        artifact_kind=artifact_kind,
        format_name=format_name,
        rows_represent="Atlas artifact surface.",
        columns_represent="See the source atlas index and the artifact's native schema.",
        claim_scope="descriptive",
        proof_carrying_status="none",
    )


def _prepare_provenance_plans(
    *,
    prepare_manifest_path: Path,
    layer: str,
    packet_dir: str,
    review_rank_base: int,
    label_suffix: str,
) -> list[ArtifactPlan]:
    prepare_payload = _load_json_dict(prepare_manifest_path, label=f"{layer} prepare manifest")
    _require_fields(
        prepare_payload,
        required_fields=("mapping_manifest", "core_fit_dry_run"),
        label=f"{layer} prepare manifest",
    )
    mapping_path = _resolve_path(prepare_payload["mapping_manifest"])
    core_fit_path = _resolve_path(prepare_payload["core_fit_dry_run"])
    plans = [
        ArtifactPlan(
            layer=layer,
            artifact_name="task_a_prepare_manifest.json",
            expected_relative_path="task_a_prepare_manifest.json",
            packet_relative_path=f"{packet_dir}/task_a_prepare_manifest{label_suffix}.json",
            source_path=prepare_manifest_path,
            artifact_kind="manifest",
            artifact_status="available",
            contract_alignment="current_contract_available",
            format="json",
            rows_represent="Single JSON object for one Task A prepare run.",
            columns_represent="Top-level keys record Step 1 provenance, readiness state, pair families, and linked mapping/dry-run files.",
            claim_scope="provenance",
            proof_carrying_status="none",
            source_workflow="prepare_task_a_stage0_mapping",
            source_manifest_or_bundle=str(prepare_manifest_path),
            notes=f"Mirrored as {layer} provenance from {prepare_manifest_path.parent.name}.",
            review_rank=review_rank_base,
            review_role="provenance",
            analysis_level="run_level",
        ),
        ArtifactPlan(
            layer=layer,
            artifact_name="task_a_stride_mapping.json",
            expected_relative_path="task_a_stride_mapping.json",
            packet_relative_path=f"{packet_dir}/task_a_stride_mapping{label_suffix}.json",
            source_path=mapping_path,
            artifact_kind="manifest",
            artifact_status="available",
            contract_alignment="current_contract_available",
            format="json",
            rows_represent="Single JSON object for one Task A Stage 0 to STRIDE mapping summary.",
            columns_represent="Top-level keys record field mappings, patient ids, family summaries, and real-data crosswalk entries.",
            claim_scope="provenance",
            proof_carrying_status="none",
            source_workflow="prepare_task_a_stage0_mapping",
            source_manifest_or_bundle=str(prepare_manifest_path),
            notes=f"Prepare provenance linked from {prepare_manifest_path.name}.",
            review_rank=review_rank_base + 1,
            review_role="provenance",
            analysis_level="run_level",
        ),
        ArtifactPlan(
            layer=layer,
            artifact_name="task_a_core_fit_dry_run.csv",
            expected_relative_path="task_a_core_fit_dry_run.csv",
            packet_relative_path=f"{packet_dir}/task_a_core_fit_dry_run{label_suffix}.csv",
            source_path=core_fit_path,
            artifact_kind="table",
            artifact_status="available",
            contract_alignment="current_contract_available",
            format="csv",
            rows_represent="Rows represent patient-by-pair-family dry-run fit realization records.",
            columns_represent="Columns record fit status, defer reason, bridge realization, and source/target domain labels.",
            claim_scope="provenance",
            proof_carrying_status="none",
            source_workflow="prepare_task_a_stage0_mapping",
            source_manifest_or_bundle=str(prepare_manifest_path),
            notes="Step 1 dry-run realization context mirrored for provenance.",
            review_rank=review_rank_base + 2,
            review_role="provenance",
            analysis_level="patient_level",
        ),
    ]
    return plans


def _collect_atlas_plans(atlas_manifest_path: Path) -> list[ArtifactPlan]:
    atlas_payload = _load_json_dict(atlas_manifest_path, label="Task A atlas manifest")
    _require_fields(
        atlas_payload,
        required_fields=("output_index",),
        label="Task A atlas manifest",
    )
    atlas_root = atlas_manifest_path.parent
    output_index_path = _resolve_path(atlas_payload["output_index"])
    output_index = pd.read_csv(output_index_path)
    required_index_columns = {"relative_path", "artifact_kind", "category", "format", "description"}
    missing_columns = required_index_columns - set(output_index.columns.astype(str))
    if missing_columns:
        raise ContractError(
            "Task A atlas output index is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    plans: list[ArtifactPlan] = []
    for row in output_index.to_dict(orient="records"):
        relative_path = str(row["relative_path"])
        source_path = _resolve_path(atlas_root / relative_path)
        metadata = _atlas_metadata(
            relative_path,
            category=str(row["category"]),
            artifact_kind=str(row["artifact_kind"]),
            format_name=str(row["format"]),
        )
        plans.append(
            ArtifactPlan(
                layer="atlas",
                artifact_name=Path(relative_path).name,
                expected_relative_path=relative_path,
                packet_relative_path=f"atlas/bundle/{relative_path}",
                source_path=source_path,
                artifact_kind=str(metadata["artifact_kind"]),
                artifact_status="available",
                contract_alignment="current_contract_available",
                format=str(metadata["format"]),
                rows_represent=str(metadata["rows_represent"]),
                columns_represent=str(metadata["columns_represent"]),
                claim_scope=str(metadata["claim_scope"]),
                proof_carrying_status=str(metadata["proof_carrying_status"]),
                source_workflow="write_task_a_descriptive_atlas",
                source_manifest_or_bundle=str(atlas_manifest_path),
                notes=f"{row['description']}",
                review_rank={
                    "task_a_descriptive_atlas_manifest.json": 10,
                    "task_a_descriptive_atlas_output_index.csv": 11,
                    "tables/community_cell_subtype_row_fractions.csv": 20,
                    "tables/community_cell_subtype_counts.csv": 21,
                    "tables/community_domain_distribution.csv": 22,
                    "tables/community_domain_roi_prevalence.csv": 23,
                    "tables/community_patient_occurrence_summary.csv": 24,
                    "tables/community_patient_occurrence_matrix.csv": 25,
                    "tables/representative_overlay_selection.csv": 26,
                }.get(relative_path, 50),
                review_role="descriptive",
                analysis_level=(
                    "run_level"
                    if relative_path in {"task_a_descriptive_atlas_manifest.json", "task_a_descriptive_atlas_output_index.csv"}
                    else "community_level"
                ),
            )
        )

    return plans

def _collect_block0_calibration_plans(
    *,
    block0_calibration_manifest_path: Path,
    prepare_manifest_path: Path,
) -> list[ArtifactPlan]:
    payload = _load_block0_calibration_manifest(block0_calibration_manifest_path)
    patient_calibration_path = _resolve_path(payload["patient_calibration_path"])
    metric_summary_path = _resolve_path(payload["metric_summary_path"])
    source_execution_manifest_path = str(payload["source_execution_manifest_path"])
    source_fit_cache_path = str(payload["source_fit_cache_path"])
    source_fit_cache_index_path = str(payload["source_fit_cache_index_path"])
    source_fit_cache_sha256 = str(payload["source_fit_cache_sha256"])
    source_fit_cache_index_sha256 = str(payload["source_fit_cache_index_sha256"])
    _validate_block0_calibration_table(
        patient_calibration_path,
        expected_columns=BLOCK0_PATIENT_CALIBRATION_COLUMNS,
        label=PATIENT_CALIBRATION_FILENAME,
    )
    _validate_block0_calibration_table(
        metric_summary_path,
        expected_columns=BLOCK0_METRIC_SUMMARY_COLUMNS,
        label=METRIC_SUMMARY_FILENAME,
    )
    common_notes = (
        "Cache-derived Block 0 calibration context. Source execution/cache paths and hashes "
        "are recorded in the calibration manifest for provenance; raw fit cache artifacts are "
        "not mirrored into the result packet by default. "
        f"source_execution_manifest_path={source_execution_manifest_path}; "
        f"source_fit_cache_path={source_fit_cache_path}; "
        f"source_fit_cache_index_path={source_fit_cache_index_path}; "
        f"source_fit_cache_sha256={source_fit_cache_sha256}; "
        f"source_fit_cache_index_sha256={source_fit_cache_index_sha256}."
    )

    plans = [
        ArtifactPlan(
            layer="block0",
            artifact_name=CALIBRATION_MANIFEST_FILENAME,
            expected_relative_path=CALIBRATION_MANIFEST_FILENAME,
            packet_relative_path=f"block0/calibration/{CALIBRATION_MANIFEST_FILENAME}",
            source_path=block0_calibration_manifest_path,
            artifact_kind="manifest",
            artifact_status="available",
            contract_alignment="current_calibration_manifest",
            format="json",
            rows_represent="Single JSON object for one Block 0 cache-derived calibration analysis.",
            columns_represent=(
                "Top-level keys record calibration provenance, fixed family-summary analysis spec, "
                "real/null families, readiness status, derived table paths, and source cache paths/hashes."
            ),
            claim_scope=BLOCK0_CALIBRATION_CONTEXT,
            proof_carrying_status="none",
            source_workflow="tasks.task_A.block0.analyze",
            source_manifest_or_bundle=str(block0_calibration_manifest_path),
            notes=common_notes,
            review_rank=10,
            review_role="calibration",
            analysis_level="run_level",
            family_surface_role=BLOCK0_CALIBRATION_CONTEXT,
        ),
        ArtifactPlan(
            layer="block0",
            artifact_name=PATIENT_CALIBRATION_FILENAME,
            expected_relative_path=PATIENT_CALIBRATION_FILENAME,
            packet_relative_path=f"block0/calibration/{PATIENT_CALIBRATION_FILENAME}",
            source_path=patient_calibration_path,
            artifact_kind="table",
            artifact_status="available",
            contract_alignment="current_calibration_manifest",
            format="csv",
            rows_represent=(
                "Rows represent patient_id x summary_name x scale x reference_stat "
                "family-summary calibration records derived from the fit cache."
            ),
            columns_represent=(
                "Columns record patient identity, family-summary name/role, eligible axis, scale, "
                "expected tail, real/null reference values, empirical p-values, opposite-tail diagnostic fraction, "
                "effect sizes, and readiness."
            ),
            claim_scope=BLOCK0_CALIBRATION_CONTEXT,
            proof_carrying_status="none",
            source_workflow="tasks.task_A.block0.analyze",
            source_manifest_or_bundle=str(block0_calibration_manifest_path),
            notes="Mirrored from patient_calibration_path in the Block 0 calibration manifest.",
            review_rank=11,
            review_role="calibration",
            analysis_level="patient_level",
            family_surface_role=BLOCK0_CALIBRATION_CONTEXT,
        ),
        ArtifactPlan(
            layer="block0",
            artifact_name=METRIC_SUMMARY_FILENAME,
            expected_relative_path=METRIC_SUMMARY_FILENAME,
            packet_relative_path=f"block0/calibration/{METRIC_SUMMARY_FILENAME}",
            source_path=metric_summary_path,
            artifact_kind="table",
            artifact_status="available",
            contract_alignment="current_calibration_manifest",
            format="csv",
            rows_represent=(
                "Rows represent summary_name x scale x cohort_stat cohort-level "
                "family-summary calibration departures derived from the fit cache."
            ),
            columns_represent=(
                "Columns record family-summary identity and role, eligible axis, expected tail, "
                "real/null reference values, empirical p-values, opposite-tail diagnostic fraction, "
                "patient delta direction counts, effect sizes, and readiness."
            ),
            claim_scope=BLOCK0_CALIBRATION_CONTEXT,
            proof_carrying_status="none",
            source_workflow="tasks.task_A.block0.analyze",
            source_manifest_or_bundle=str(block0_calibration_manifest_path),
            notes="Mirrored from metric_summary_path in the Block 0 calibration manifest.",
            review_rank=12,
            review_role="calibration",
            analysis_level="cohort_level",
            family_surface_role=BLOCK0_CALIBRATION_CONTEXT,
        ),
    ]

    plans.extend(
        _prepare_provenance_plans(
            prepare_manifest_path=prepare_manifest_path,
            layer="block0",
            packet_dir="block0/provenance/prepare",
            review_rank_base=20,
            label_suffix="",
        )
    )
    return plans


BLOCK1_EXPECTED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "relative_path": "block1_bundle.json",
        "artifact_kind": "bundle",
        "format": "json",
        "rows_represent": "Single JSON object for one Block 1 bundle.",
        "columns_represent": "Top-level keys record Block 1 provenance, linked summary paths, contract versions, and proof/supportive summary roles.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 10,
    },
    {
        "relative_path": "block1_workflow_manifest.json",
        "artifact_kind": "manifest",
        "format": "json",
        "rows_represent": "Single JSON object for one compact Block 1 pointer manifest.",
        "columns_represent": "Top-level keys point to the Block 1 bundle's summary, comparison, and correspondence surfaces.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 11,
    },
    {
        "relative_path": "block1_stage0_mapping.json",
        "artifact_kind": "manifest",
        "format": "json",
        "rows_represent": "Single JSON object for one Block 1 Stage 0 mapping summary.",
        "columns_represent": "Top-level keys record field mappings, patient ids, family summaries, and real-data crosswalk entries.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 12,
    },
    {
        "relative_path": "block1_core_fit_dry_run.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient-by-pair-family Block 1 dry-run fit realization records.",
        "columns_represent": "Columns record fit status, defer reason, bridge realization, and source/target domain labels.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 13,
    },
    {
        "relative_path": "block1_recurrence_summary.json",
        "artifact_kind": "manifest",
        "format": "json",
        "rows_represent": "Single JSON object summarizing canonical cohort-level recurrence support across confirmatory Block 1 pair families.",
        "columns_represent": "Top-level keys record recurrence fit status, family counts, used-patient counts, and per-pair-family recurrence metadata.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 14,
    },
    {
        "relative_path": "block1_recurrence_families.json",
        "artifact_kind": "table",
        "format": "json",
        "rows_represent": "Each JSON row represents one canonical cohort-level recurrence family linked to a Block 1 pair family.",
        "columns_represent": "Fields record recurrence family identity, support, members, and template A/d/e arrays.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 15,
    },
    {
        "relative_path": "block1_recurrence_embeddings.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient-level canonical recurrence embeddings linked to each Block 1 pair family.",
        "columns_represent": "Columns record pair family, patient id, embedding fit status, recurrence inclusion, and low-dimensional coordinates.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 16,
    },
    {
        "relative_path": "block1_family_summary.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x summary_name x scale family-summary records.",
        "columns_represent": "Columns record summary value, summary role, eligible axis/count, and burden total.",
        "claim_scope": "confirmatory",
        "proof_carrying_status": "partial",
        "review_rank": 20,
    },
    {
        "relative_path": "block1_source_community_summary.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x source_community_id source-community summary records.",
        "columns_represent": "Columns record source burden/weight, self-retention, depletion, remodeling, and top-target values.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 21,
    },
    {
        "relative_path": "block1_target_community_summary.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x target_community_id target-community summary records.",
        "columns_represent": "Columns record target burden/weight, incoming matched quantities, and emergence quantities.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 22,
    },
    {
        "relative_path": "block1_confirmatory_family_comparison.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x summary_name x scale paired TC-IM versus TC-PT comparison records.",
        "columns_represent": "Columns record paired family values, deltas, comparison status, and contrast direction.",
        "claim_scope": "confirmatory",
        "proof_carrying_status": "partial",
        "review_rank": 23,
    },
    {
        "relative_path": "block1_exploratory_source_community_comparison.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x source_community_id x summary_name paired exploratory source-community comparison records.",
        "columns_represent": "Columns record TC-IM / TC-PT values, deltas, and comparison status.",
        "claim_scope": "exploratory",
        "proof_carrying_status": "none",
        "review_rank": 24,
    },
    {
        "relative_path": "block1_exploratory_target_community_comparison.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x target_community_id x summary_name paired exploratory target-community comparison records.",
        "columns_represent": "Columns record TC-IM / TC-PT values, deltas, and comparison status.",
        "claim_scope": "exploratory",
        "proof_carrying_status": "none",
        "review_rank": 25,
    },
    {
        "relative_path": "community_correspondence/block1_community_correspondence_manifest.json",
        "artifact_kind": "manifest",
        "format": "json",
        "rows_represent": "Single JSON object for one Block 1 community-correspondence packet.",
        "columns_represent": "Top-level keys record correspondence provenance, Stage 0 keys, linked summary paths, and the packet output index.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 30,
    },
    {
        "relative_path": "community_correspondence/block1_community_correspondence_index.csv",
        "artifact_kind": "index",
        "format": "csv",
        "rows_represent": "One row per community-correspondence table or referenced Block 1 surface.",
        "columns_represent": "Columns identify each packet-relative path, artifact kind, category, format, and short description.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 31,
    },
    {
        "relative_path": "community_correspondence/tables/community_cell_subtype_counts.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent communities observed in Stage 0 / Block 1 correspondence.",
        "columns_represent": "Columns represent cell-subtype counts within each community.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 32,
    },
    {
        "relative_path": "community_correspondence/tables/community_cell_subtype_row_fractions.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent communities observed in Stage 0 / Block 1 correspondence.",
        "columns_represent": "Columns represent within-community cell-subtype fractions.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 33,
    },
    {
        "relative_path": "community_correspondence/tables/source_community_major_targets.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x source_community_id x target_rank major-target records.",
        "columns_represent": "Columns record source burden/weight plus ranked target-community ids and operator values.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 34,
    },
    {
        "relative_path": "community_correspondence/tables/source_community_burden_components.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x source_community_id source burden-component records.",
        "columns_represent": "Columns record matched, self-retention, remodeling, and depletion burdens plus source burden/weight.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 35,
    },
    {
        "relative_path": "community_correspondence/tables/target_community_burden_components.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent patient_id x pair_family x target_community_id target burden-component records.",
        "columns_represent": "Columns record incoming matched quantities, emergence quantities, and target burden/weight.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 36,
    },
    {
        "relative_path": "community_correspondence/tables/community_id_crosswalk.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent community ids across the configured and observed community axis.",
        "columns_represent": "Columns record Stage 0 presence plus source-summary and target-summary observation counts by community id.",
        "claim_scope": "supportive",
        "proof_carrying_status": "none",
        "review_rank": 37,
    },
)


def _block1_plan_from_spec(
    *,
    spec: dict[str, Any],
    source_path: Path | None,
    artifact_status: str,
    source_manifest_or_bundle: str,
    notes: str,
) -> ArtifactPlan:
    relative_path = str(spec["relative_path"])
    return ArtifactPlan(
        layer="block1",
        artifact_name=Path(relative_path).name,
        expected_relative_path=relative_path,
        packet_relative_path=f"block1/bundle/{relative_path}" if source_path is not None else None,
        source_path=source_path,
        artifact_kind=str(spec["artifact_kind"]),
        artifact_status=artifact_status,
        contract_alignment=(
            "current_contract_available" if source_path is not None else "current_contract_missing"
        ),
        format=str(spec["format"]),
        rows_represent=str(spec["rows_represent"]),
        columns_represent=str(spec["columns_represent"]),
        claim_scope=str(spec["claim_scope"]),
        proof_carrying_status=str(spec["proof_carrying_status"]),
        source_workflow="write_task_a_block1_bundle",
        source_manifest_or_bundle=source_manifest_or_bundle,
        notes=notes,
        review_rank=int(spec["review_rank"]),
        review_role=(
            "proof_carrying"
            if str(spec["proof_carrying_status"]) in {"all", "partial"}
            else ("provenance" if str(spec["claim_scope"]) == "provenance" else "supportive")
        ),
        analysis_level=(
            "patient_level"
            if any(
                token in relative_path
                for token in (
                    "family_summary",
                    "source_community_summary",
                    "target_community_summary",
                    "recurrence_embeddings",
                    "confirmatory_family_comparison",
                    "exploratory_source_community_comparison",
                    "exploratory_target_community_comparison",
                    "source_community_major_targets",
                    "source_community_burden_components",
                    "target_community_burden_components",
                )
            )
            else (
                "cohort_level"
                if "recurrence_" in relative_path
                else ("community_level" if "community_" in relative_path else "run_level")
            )
        ),
        family_surface_role=(
            "comparison_surface"
            if "comparison" in relative_path or "family_summary" in relative_path
            else "not_applicable"
        ),
    )


def _collect_block1_plans(block1_bundle_path: Path | None) -> list[ArtifactPlan]:
    if block1_bundle_path is None:
        return [
            _block1_plan_from_spec(
                spec=spec,
                source_path=None,
                artifact_status="missing_on_disk",
                source_manifest_or_bundle=str(BLOCK1_CONTRACT_PATH),
                notes="Expected by the frozen Block 1 contract; no live Block 1 bundle was supplied or found on disk.",
            )
            for spec in BLOCK1_EXPECTED_ARTIFACTS
        ]

    block1_payload = _load_json_dict(block1_bundle_path, label="Task A Block 1 bundle")
    output_root = _resolve_path(block1_payload.get("output_dir", block1_bundle_path.parent))
    path_map: dict[str, Path] = {
        "block1_bundle.json": block1_bundle_path,
        "block1_workflow_manifest.json": output_root / "block1_workflow_manifest.json",
        "block1_stage0_mapping.json": _resolve_path(block1_payload["mapping_manifest_path"]),
        "block1_core_fit_dry_run.csv": _resolve_path(block1_payload["core_fit_dry_run_path"]),
        "block1_recurrence_summary.json": _resolve_path(block1_payload["recurrence_summary_path"]),
        "block1_recurrence_families.json": _resolve_path(block1_payload["recurrence_families_path"]),
        "block1_recurrence_embeddings.csv": _resolve_path(block1_payload["recurrence_embeddings_path"]),
        "block1_family_summary.csv": _resolve_path(block1_payload["family_summary_path"]),
        "block1_source_community_summary.csv": _resolve_path(block1_payload["source_community_summary_path"]),
        "block1_target_community_summary.csv": _resolve_path(block1_payload["target_community_summary_path"]),
        "block1_confirmatory_family_comparison.csv": _resolve_path(block1_payload["confirmatory_family_comparison_path"]),
        "block1_exploratory_source_community_comparison.csv": _resolve_path(
            block1_payload["exploratory_source_community_comparison_path"]
        ),
        "block1_exploratory_target_community_comparison.csv": _resolve_path(
            block1_payload["exploratory_target_community_comparison_path"]
        ),
        "community_correspondence/block1_community_correspondence_manifest.json": _resolve_path(
            block1_payload["community_correspondence_manifest_path"]
        ),
        "community_correspondence/block1_community_correspondence_index.csv": _resolve_path(
            block1_payload["community_correspondence_index_path"]
        ),
        "community_correspondence/tables/community_cell_subtype_counts.csv": output_root
        / "community_correspondence"
        / "tables"
        / "community_cell_subtype_counts.csv",
        "community_correspondence/tables/community_cell_subtype_row_fractions.csv": output_root
        / "community_correspondence"
        / "tables"
        / "community_cell_subtype_row_fractions.csv",
        "community_correspondence/tables/source_community_major_targets.csv": output_root
        / "community_correspondence"
        / "tables"
        / "source_community_major_targets.csv",
        "community_correspondence/tables/source_community_burden_components.csv": output_root
        / "community_correspondence"
        / "tables"
        / "source_community_burden_components.csv",
        "community_correspondence/tables/target_community_burden_components.csv": output_root
        / "community_correspondence"
        / "tables"
        / "target_community_burden_components.csv",
        "community_correspondence/tables/community_id_crosswalk.csv": output_root
        / "community_correspondence"
        / "tables"
        / "community_id_crosswalk.csv",
    }
    plans: list[ArtifactPlan] = []
    for spec in BLOCK1_EXPECTED_ARTIFACTS:
        relative_path = str(spec["relative_path"])
        source_path = path_map.get(relative_path)
        if source_path is not None and source_path.exists():
            plans.append(
                _block1_plan_from_spec(
                    spec=spec,
                    source_path=source_path.resolve(),
                    artifact_status="available",
                    source_manifest_or_bundle=str(block1_bundle_path),
                    notes="Mirrored from the supplied live Block 1 bundle.",
                )
            )
        else:
            plans.append(
                _block1_plan_from_spec(
                    spec=spec,
                    source_path=None,
                    artifact_status="missing_on_disk",
                    source_manifest_or_bundle=str(block1_bundle_path),
                    notes="Expected by the supplied Block 1 bundle surface, but the file was not present on disk.",
                )
            )
    return plans


BLOCK2_EXPECTED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "relative_path": "block2_bounded_audit_manifest.json",
        "artifact_kind": "manifest",
        "format": "json",
        "rows_represent": "Single JSON object for one Block 2 robustness run.",
        "columns_represent": "Top-level keys record Block 2 provenance, linked Block 1 inputs, linked Block 2 outputs, route scope, and replicate counts.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 10,
    },
    {
        "relative_path": "block2_bounded_audit_summary.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent one frozen Block 1 finding summarized across the primary Block 2 routes.",
        "columns_represent": "Columns record scope, finding priority, overall robustness call, and worst-case primary-route support quantities.",
        "claim_scope": "robustness",
        "proof_carrying_status": "partial",
        "review_rank": 20,
    },
    {
        "relative_path": "block2_family_robustness.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent route-level family robustness summaries for the frozen Block 1 family findings.",
        "columns_represent": "Columns record route identity, full-data direction, recovery rates, estimability, support fractions, and the route-level robustness call.",
        "claim_scope": "robustness",
        "proof_carrying_status": "all",
        "review_rank": 21,
    },
    {
        "relative_path": "block2_source_community_robustness.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent route-level source-community robustness summaries for the scoped Block 1 source findings.",
        "columns_represent": "Columns record route identity, community id, full-data direction, recovery rates, estimability, support fractions, rank stability, and the route-level robustness call.",
        "claim_scope": "robustness",
        "proof_carrying_status": "partial",
        "review_rank": 22,
    },
    {
        "relative_path": "block2_target_community_robustness.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent route-level target-community robustness summaries for the scoped Block 1 target findings.",
        "columns_represent": "Columns record route identity, community id, full-data direction, recovery rates, estimability, support fractions, rank stability, and the route-level robustness call.",
        "claim_scope": "robustness",
        "proof_carrying_status": "partial",
        "review_rank": 23,
    },
    {
        "relative_path": "block2_replicate_manifest.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent one attempted Block 2 perturbation replicate.",
        "columns_represent": "Columns record route identity, retained patient/ROI/cell counts, encoded perturbation membership, and any replicate-level failure reason.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 24,
    },
    {
        "relative_path": "block2_contract_audit.csv",
        "artifact_kind": "table",
        "format": "csv",
        "rows_represent": "Rows represent one Block 2 provenance or contract check.",
        "columns_represent": "Columns record the check name, pass/fail state, and supporting detail.",
        "claim_scope": "provenance",
        "proof_carrying_status": "none",
        "review_rank": 25,
    },
)


def _block2_plan_from_spec(
    *,
    spec: dict[str, Any],
    source_path: Path | None,
    artifact_status: str,
    source_manifest_or_bundle: str,
    notes: str,
) -> ArtifactPlan:
    relative_path = str(spec["relative_path"])
    return ArtifactPlan(
        layer="block2",
        artifact_name=Path(relative_path).name,
        expected_relative_path=relative_path,
        packet_relative_path=f"block2/bundle/{relative_path}" if source_path is not None else None,
        source_path=source_path,
        artifact_kind=str(spec["artifact_kind"]),
        artifact_status=artifact_status,
        contract_alignment=(
            "current_contract_available" if source_path is not None else "current_contract_missing"
        ),
        format=str(spec["format"]),
        rows_represent=str(spec["rows_represent"]),
        columns_represent=str(spec["columns_represent"]),
        claim_scope=str(spec["claim_scope"]),
        proof_carrying_status=str(spec["proof_carrying_status"]),
        source_workflow="write_block2_bundle",
        source_manifest_or_bundle=source_manifest_or_bundle,
        notes=notes,
        review_rank=int(spec["review_rank"]),
        review_role=(
            "proof_carrying"
            if str(spec["proof_carrying_status"]) in {"all", "partial"}
            else "provenance"
        ),
        analysis_level=(
            "replicate"
            if relative_path == "block2_replicate_manifest.csv"
            else (
                "source_community"
                if relative_path == "block2_source_community_robustness.csv"
                else (
                    "target_community"
                    if relative_path == "block2_target_community_robustness.csv"
                    else (
                        "family"
                        if relative_path == "block2_family_robustness.csv"
                        else "cohort_decision"
                    )
                )
            )
        ),
        family_surface_role=(
            "comparison_surface"
            if relative_path in {
                "block2_bounded_audit_summary.csv",
                "block2_family_robustness.csv",
            }
            else "not_applicable"
        ),
    )


def _collect_block2_plans(block2_manifest_path: Path | None) -> list[ArtifactPlan]:
    if block2_manifest_path is None:
        return [
            _block2_plan_from_spec(
                spec=spec,
                source_path=None,
                artifact_status="missing_on_disk",
                source_manifest_or_bundle=str(BLOCK2_CONTRACT_PATH),
                notes="Expected by the frozen Block 2 contract; no live Block 2 manifest was supplied or found on disk.",
            )
            for spec in BLOCK2_EXPECTED_ARTIFACTS
        ]

    block2_payload = _load_json_dict(block2_manifest_path, label="Task A Block 2 manifest")
    _require_fields(
        block2_payload,
        required_fields=(
            "summary_path",
            "contract_path",
            "replicate_manifest_path",
            "family_robustness_path",
            "source_community_robustness_path",
            "target_community_robustness_path",
        ),
        label="Task A Block 2 manifest",
    )
    path_map: dict[str, Path] = {
        "block2_bounded_audit_manifest.json": block2_manifest_path,
        "block2_bounded_audit_summary.csv": _resolve_path(block2_payload["summary_path"]),
        "block2_contract_audit.csv": _resolve_path(block2_payload["contract_path"]),
        "block2_replicate_manifest.csv": _resolve_path(block2_payload["replicate_manifest_path"]),
        "block2_family_robustness.csv": _resolve_path(block2_payload["family_robustness_path"]),
        "block2_source_community_robustness.csv": _resolve_path(
            block2_payload["source_community_robustness_path"]
        ),
        "block2_target_community_robustness.csv": _resolve_path(
            block2_payload["target_community_robustness_path"]
        ),
    }

    plans: list[ArtifactPlan] = []
    for spec in BLOCK2_EXPECTED_ARTIFACTS:
        relative_path = str(spec["relative_path"])
        source_path = path_map.get(relative_path)
        if source_path is not None and source_path.exists():
            plans.append(
                _block2_plan_from_spec(
                    spec=spec,
                    source_path=source_path.resolve(),
                    artifact_status="available",
                    source_manifest_or_bundle=str(block2_manifest_path),
                    notes="Mirrored from the supplied live Block 2 robustness manifest.",
                )
            )
        else:
            plans.append(
                _block2_plan_from_spec(
                    spec=spec,
                    source_path=None,
                    artifact_status="missing_on_disk",
                    source_manifest_or_bundle=str(block2_manifest_path),
                    notes="Expected by the supplied Block 2 manifest surface, but the file was not present on disk.",
                )
            )
    return plans


BLOCK3_EXPECTED_ARTIFACTS: tuple[dict[str, Any], ...] = ()


def _collect_block3_plans(block3_manifest_path: Path | None) -> list[ArtifactPlan]:
    _ = block3_manifest_path
    return []


def _materialize_plan(
    plan: ArtifactPlan,
    *,
    packet_root: Path,
    surface_lineage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    packet_relative_path = plan.packet_relative_path or ""
    source_path = str(plan.source_path) if plan.source_path is not None else ""
    n_rows: int | str = ""
    n_columns: int | str = ""
    observed_columns = ""
    sha256 = ""

    if plan.source_path is not None:
        if plan.packet_relative_path is None:
            raise ContractError(
                f"Artifact plan for available file is missing a packet_relative_path: {plan.artifact_name}"
            )
        resolved_source = plan.source_path.resolve()
        if not resolved_source.exists():
            raise FileNotFoundError(f"Source artifact is missing: {resolved_source}")
        destination = packet_root / plan.packet_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_source, destination)
        n_rows, n_columns, observed_columns = _observe_artifact(destination)
        sha256 = _hash_file(destination)

    artifact_lineage = _resolve_artifact_lineage(
        layer=plan.layer,
        relative_path=packet_relative_path or plan.expected_relative_path,
        source_path=plan.source_path.resolve() if plan.source_path is not None else None,
        surface_lineage=surface_lineage,
    )

    return {
        "layer": plan.layer,
        "artifact_name": plan.artifact_name,
        "expected_relative_path": plan.expected_relative_path,
        "packet_relative_path": packet_relative_path,
        "source_path": source_path,
        "artifact_kind": plan.artifact_kind,
        "artifact_status": plan.artifact_status,
        "contract_alignment": plan.contract_alignment,
        "implementation_tier": artifact_lineage["implementation_tier"],
        "evidence_lineage": artifact_lineage["evidence_lineage"],
        "format": plan.format,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "observed_columns": observed_columns,
        "rows_represent": plan.rows_represent,
        "columns_represent": plan.columns_represent,
        "claim_scope": plan.claim_scope,
        "review_role": plan.review_role,
        "analysis_level": plan.analysis_level,
        "family_surface_role": plan.family_surface_role,
        "is_proof_carrying": _proof_carrying_bool(plan.proof_carrying_status),
        "proof_carrying_status": plan.proof_carrying_status,
        "source_workflow": plan.source_workflow,
        "source_manifest_or_bundle": plan.source_manifest_or_bundle,
        "sha256": sha256,
        "notes": plan.notes,
        "review_rank": plan.review_rank,
    }


def _record_existing_packet_artifact(
    *,
    packet_root: Path,
    layer: str,
    artifact_name: str,
    expected_relative_path: str,
    packet_relative_path: str,
    artifact_kind: str,
    contract_alignment: str,
    format_name: str,
    rows_represent: str,
    columns_represent: str,
    claim_scope: str,
    review_role: str,
    analysis_level: str,
    family_surface_role: str,
    proof_carrying_status: str,
    source_workflow: str,
    source_manifest_or_bundle: str,
    notes: str,
    review_rank: int,
    surface_lineage: dict[str, dict[str, Any]],
    source_path: str = "",
) -> dict[str, Any]:
    destination = packet_root / packet_relative_path
    if not destination.exists():
        raise FileNotFoundError(f"Packet artifact is missing before indexing: {destination}")
    n_rows, n_columns, observed_columns = _observe_artifact(destination)
    resolved_source_path = (
        Path(source_path).expanduser().resolve()
        if str(source_path).strip() != ""
        else None
    )
    artifact_lineage = _resolve_artifact_lineage(
        layer=layer,
        relative_path=packet_relative_path or expected_relative_path,
        source_path=resolved_source_path,
        surface_lineage=surface_lineage,
    )
    return {
        "layer": layer,
        "artifact_name": artifact_name,
        "expected_relative_path": expected_relative_path,
        "packet_relative_path": packet_relative_path,
        "source_path": source_path,
        "artifact_kind": artifact_kind,
        "artifact_status": "available",
        "contract_alignment": contract_alignment,
        "implementation_tier": artifact_lineage["implementation_tier"],
        "evidence_lineage": artifact_lineage["evidence_lineage"],
        "format": format_name,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "observed_columns": observed_columns,
        "rows_represent": rows_represent,
        "columns_represent": columns_represent,
        "claim_scope": claim_scope,
        "review_role": review_role,
        "analysis_level": analysis_level,
        "family_surface_role": family_surface_role,
        "is_proof_carrying": _proof_carrying_bool(proof_carrying_status),
        "proof_carrying_status": proof_carrying_status,
        "source_workflow": source_workflow,
        "source_manifest_or_bundle": source_manifest_or_bundle,
        "sha256": _hash_file(destination),
        "notes": notes,
        "review_rank": review_rank,
    }


def _write_layer_review_indexes(
    index_df: pd.DataFrame,
    *,
    packet_root: Path,
    layers: tuple[str, ...],
) -> dict[str, Path]:
    review_paths: dict[str, Path] = {}
    columns = [
        "review_rank",
        "artifact_name",
        "expected_relative_path",
        "packet_relative_path",
        "artifact_status",
        "artifact_kind",
        "implementation_tier",
        "evidence_lineage",
        "claim_scope",
        "review_role",
        "analysis_level",
        "family_surface_role",
        "proof_carrying_status",
        "rows_represent",
        "columns_represent",
        "notes",
    ]
    for layer in layers:
        layer_df = (
            index_df.loc[index_df["layer"] == layer, columns]
            .sort_values(["review_rank", "artifact_name"], kind="mergesort")
            .reset_index(drop=True)
        )
        review_path = packet_root / layer / f"{layer}_review_index.csv"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        layer_df.to_csv(review_path, index=False)
        review_paths[layer] = review_path
    return review_paths


def _write_layer_manifests(
    index_df: pd.DataFrame,
    *,
    packet_root: Path,
    layer_review_index_paths: dict[str, Path],
    layers: tuple[str, ...],
    surface_lineage: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    layer_manifest_paths: dict[str, Path] = {}
    for layer in layers:
        layer_df = index_df.loc[index_df["layer"] == layer].copy()
        artifact_counts = {
            "n_artifacts": int(len(layer_df)),
            "n_available_artifacts": int((layer_df["artifact_status"] == "available").sum()),
            "n_missing_artifacts": int((layer_df["artifact_status"] == "missing_on_disk").sum()),
        }
        payload = {
            "workflow_name": "write_task_a_result_packet",
            "packet_role": PACKET_ROLE,
            "layer": layer,
            "scientific_interpretation_allowed": False,
            "review_index_path": str(layer_review_index_paths[layer]),
            "artifact_counts": artifact_counts,
            "claim_scopes_present": sorted(
                {str(value) for value in layer_df["claim_scope"].astype(str).tolist()}
            ),
            "review_roles_present": sorted(
                {str(value) for value in layer_df["review_role"].astype(str).tolist()}
            ),
            "analysis_levels_present": sorted(
                {str(value) for value in layer_df["analysis_level"].astype(str).tolist()}
            ),
            "family_surface_roles_present": sorted(
                {str(value) for value in layer_df["family_surface_role"].astype(str).tolist()}
            ),
            "artifact_statuses_present": sorted(
                {str(value) for value in layer_df["artifact_status"].astype(str).tolist()}
            ),
            "implementation_tiers_present": sorted(
                {str(value) for value in layer_df["implementation_tier"].astype(str).tolist()}
            ),
            "evidence_lineages_present": sorted(
                {str(value) for value in layer_df["evidence_lineage"].astype(str).tolist()}
            ),
            "source_workflows": sorted(
                {str(value) for value in layer_df["source_workflow"].astype(str).tolist()}
            ),
        }
        if layer in surface_lineage:
            payload["surface_lineage"] = surface_lineage[layer]
        if layer == "atlas":
            payload["notes"] = (
                "Descriptive-only context layer mirrored from the audited atlas bundle and labeled "
                "as part of the canonical rerun packet."
            )
        elif layer == "block0":
            schema_variants = sorted(
                {str(value) for value in layer_df["contract_alignment"].astype(str).tolist()}
            )
            payload["notes"] = (
                "Available Block 0 calibration files are mirrored exactly from the cache-derived analysis surface. "
                f"Observed Block 0 schema/alignment states: {schema_variants}. "
                "This layer records calibration context only; raw fit cache artifacts are referenced by source "
                "paths and hashes in the mirrored manifest rather than copied into the packet."
            )
        elif layer == "block1":
            payload["notes"] = (
                "Block 1 rows describe the frozen discovery-layer contract surface on the canonical full STRIDE "
                "path, including the cohort-level recurrence exports added by the Step 3 rerun."
            )
        elif layer == "block2":
            if "write_block2_review_surface" in set(layer_df["source_workflow"].astype(str).tolist()):
                payload["notes"] = (
                    "Block 2 rows describe the frozen robustness-layer contract surface and packet-local review-only "
                    "reorganizations. Raw files remain mirrored faithfully, while the review tables clarify route "
                    "coverage, call semantics, and primary review order without changing scientific contents. "
                    "This layer is attached to the canonical Block 1 rerun."
                )
            else:
                payload["notes"] = (
                    "Block 2 rows describe the frozen robustness-layer contract surface over the canonical Block 1 "
                    "rerun. Available files are mirrored when supplied; otherwise missing surfaces remain explicit."
                )
        else:
            payload["notes"] = (
                "Block 3 rows describe packet-local mirrors of the supplied method-validation bundle surface. "
                "Available files are mirrored when supplied; otherwise missing surfaces remain explicit. This "
                "packet layer remains non-authority and does not upgrade mirrored Block 3 wording into "
                "scientific authority."
            )
        manifest_path = packet_root / layer / f"{layer}_layer_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        layer_manifest_paths[layer] = manifest_path
    return layer_manifest_paths


def _write_human_index(
    index_df: pd.DataFrame,
    *,
    packet_root: Path,
    layers: tuple[str, ...],
    surface_lineage: dict[str, dict[str, Any]],
) -> Path:
    atlas_df = index_df.loc[index_df["layer"] == "atlas"].copy()
    block0_df = index_df.loc[index_df["layer"] == "block0"].copy()
    block1_df = index_df.loc[index_df["layer"] == "block1"].copy()
    block2_df = index_df.loc[index_df["layer"] == "block2"].copy()
    inspect_first = [
        "atlas/bundle/task_a_descriptive_atlas_manifest.json",
        "atlas/bundle/tables/community_cell_subtype_row_fractions.csv",
        "atlas/bundle/tables/community_domain_distribution.csv",
        "atlas/bundle/tables/community_patient_occurrence_summary.csv",
        "block0/calibration/block0_calibration_manifest.json",
        "block0/calibration/block0_patient_calibration.csv",
        "block0/calibration/block0_metric_summary.csv",
        "block1/block1_review_index.csv",
        "block2/BLOCK2_RESULTS_INDEX.md",
        "block2/review/block2_primary_finding_review_table.csv",
        "block2/review/block2_route_summary.csv",
        "block2/bundle/block2_bounded_audit_summary.csv",
        "block2/block2_review_index.csv",
    ]
    lines = [
        "# Task A Objective Result Packet",
        "",
        "This packet mirrors the canonical Task A rerun surface through Block 2 without biological interpretation.",
        "Block 3 is explicitly deferred from this packet because the active Block 3 engineering surface has been removed pending rebuild.",
        "Proxy-era Block 0 outputs are not accepted or repackaged here.",
        "",
        "## Current Surface Status",
        f"- Included layers: {', '.join(layers)}",
        (
            f"- Deferred layers: {', '.join(_deferred_layers(include_block3='block3' in layers))}"
            if _deferred_layers(include_block3="block3" in layers)
            else "- Deferred layers: none"
        ),
        f"- Atlas available artifacts: {int((atlas_df['artifact_status'] == 'available').sum())}",
        f"- Block 0 available artifacts: {int((block0_df['artifact_status'] == 'available').sum())}",
        f"- Block 1 available artifacts: {int((block1_df['artifact_status'] == 'available').sum())}",
        f"- Block 1 missing artifacts: {int((block1_df['artifact_status'] == 'missing_on_disk').sum())}",
        f"- Block 2 available artifacts: {int((block2_df['artifact_status'] == 'available').sum())}",
        f"- Block 2 missing artifacts: {int((block2_df['artifact_status'] == 'missing_on_disk').sum())}",
        "",
        "## Surface Lineage",
        f"- Prepare: `{surface_lineage.get('prepare', {}).get('implementation_tier', '')}` / `{surface_lineage.get('prepare', {}).get('evidence_lineage', '')}`",
        f"- Block 0: `{surface_lineage.get('block0', {}).get('implementation_tier', '')}` / `{surface_lineage.get('block0', {}).get('evidence_lineage', '')}`",
        f"- Block 1: `{surface_lineage.get('block1', {}).get('implementation_tier', '')}` / `{surface_lineage.get('block1', {}).get('evidence_lineage', '')}`",
        f"- Block 2: `{surface_lineage.get('block2', {}).get('implementation_tier', '')}` / `{surface_lineage.get('block2', {}).get('evidence_lineage', '')}`",
        "",
        "## Inspect First",
    ]
    for relative_path in inspect_first:
        candidate = packet_root / relative_path
        if candidate.exists():
            lines.append(f"- `{relative_path}`")
    lines.extend(
        [
            "",
            "## Atlas",
            f"- Review index: `atlas/atlas_review_index.csv`",
            f"- Layer manifest: `atlas/atlas_layer_manifest.json`",
            "",
            "## Block 0",
            f"- Review index: `block0/block0_review_index.csv`",
            f"- Layer manifest: `block0/block0_layer_manifest.json`",
            "",
            "## Block 1",
            f"- Review index: `block1/block1_review_index.csv`",
            f"- Layer manifest: `block1/block1_layer_manifest.json`",
            "",
            "## Block 2",
            f"- Review index: `block2/block2_review_index.csv`",
            f"- Layer manifest: `block2/block2_layer_manifest.json`",
            "",
            "## Missing Block 1 Surfaces",
        ]
    )
    missing_block1 = (
        block1_df.loc[block1_df["artifact_status"] == "missing_on_disk", "expected_relative_path"]
        .astype(str)
        .tolist()
    )
    if not missing_block1:
        lines.append("- None.")
    else:
        for expected_path in missing_block1:
            lines.append(f"- `{expected_path}`")
    lines.extend(["", "## Missing Block 2 Surfaces"])
    missing_block2 = (
        block2_df.loc[block2_df["artifact_status"] == "missing_on_disk", "expected_relative_path"]
        .astype(str)
        .tolist()
    )
    if not missing_block2:
        lines.append("- None.")
    else:
        for expected_path in missing_block2:
            lines.append(f"- `{expected_path}`")
    human_index_path = packet_root / HUMAN_INDEX_FILENAME
    human_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return human_index_path


def validate_task_a_result_packet(packet_manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = _resolve_path(packet_manifest_path)
    manifest_payload = _load_json_dict(manifest_path, label="Task A result packet manifest")
    _require_fields(
        manifest_payload,
        required_fields=(
            "packet_root",
            "central_index_path",
            "layer_manifest_paths",
            "human_index_path",
            "included_layers",
            "deferred_layers",
        ),
        label="Task A result packet manifest",
    )
    packet_root = _resolve_path(manifest_payload["packet_root"])
    index_path = _resolve_path(manifest_payload["central_index_path"])
    if not index_path.exists():
        raise ContractError(f"Task A result packet central index is missing: {index_path}")
    index_df = pd.read_csv(index_path, keep_default_na=False)
    checks: list[dict[str, Any]] = []
    included_layers = tuple(str(layer) for layer in manifest_payload["included_layers"])
    deferred_layers = tuple(str(layer) for layer in manifest_payload["deferred_layers"])
    if "block3" in included_layers and "block3" in deferred_layers:
        raise ContractError("Task A result packet cannot both include and defer block3")
    if "block3" not in included_layers and "block3" not in deferred_layers:
        raise ContractError("Task A result packet must either include or defer block3 explicitly")
    if tuple(index_df["layer"].astype(str).unique().tolist()) != included_layers:
        raise ContractError(
            "Task A result packet included_layers do not match the indexed layer order"
        )

    for field_name in ("human_index_path",):
        candidate = _resolve_path(manifest_payload[field_name])
        passed = candidate.exists()
        checks.append({"check": field_name, "passed": passed, "detail": str(candidate)})
        if not passed:
            raise ContractError(f"Task A result packet is missing required file: {candidate}")

    for layer, path_str in dict(manifest_payload["layer_manifest_paths"]).items():
        candidate = _resolve_path(path_str)
        passed = candidate.exists()
        checks.append({"check": f"layer_manifest:{layer}", "passed": passed, "detail": str(candidate)})
        if not passed:
            raise ContractError(f"Task A result packet layer manifest is missing: {candidate}")

    for row in index_df.to_dict(orient="records"):
        artifact_status = str(row["artifact_status"])
        packet_relative_path = str(row["packet_relative_path"]).strip()
        if artifact_status == "available":
            if not packet_relative_path:
                raise ContractError(
                    f"Available packet artifact is missing packet_relative_path: {row['artifact_name']}"
                )
            packet_path = packet_root / packet_relative_path
            if not packet_path.exists():
                raise ContractError(f"Referenced packet artifact is missing: {packet_path}")
            observed_hash = _hash_file(packet_path)
            expected_hash = str(row["sha256"]).strip()
            if expected_hash and expected_hash != observed_hash:
                raise ContractError(
                    f"Checksum mismatch for packet artifact {packet_path}: {expected_hash} != {observed_hash}"
                )
        elif artifact_status == "missing_on_disk" and packet_relative_path:
            raise ContractError(
                f"Missing artifact unexpectedly has packet_relative_path populated: {packet_relative_path}"
            )
        checks.append(
            {
                "check": f"artifact:{row['layer']}:{row['artifact_name']}",
                "passed": True,
                "detail": packet_relative_path or str(row["expected_relative_path"]),
            }
        )
    return checks


def write_task_a_result_packet(
    *,
    atlas_manifest_path: str | Path,
    prepare_manifest_path: str | Path,
    block0_calibration_manifest_path: str | Path,
    output_dir: str | Path,
    block1_bundle_path: str | Path | None = None,
    block2_manifest_path: str | Path | None = None,
    block3_manifest_path: str | Path | None = None,
) -> TaskAResultPacket:
    if block3_manifest_path is not None:
        raise ContractError(BLOCK3_PACKET_DEFERRED_MESSAGE)

    atlas_manifest = _resolve_path(atlas_manifest_path)
    prepare_manifest = _resolve_path(prepare_manifest_path)
    block0_calibration_manifest = _resolve_path(block0_calibration_manifest_path)
    block1_bundle = None if block1_bundle_path is None else _resolve_path(block1_bundle_path)
    block2_manifest = None if block2_manifest_path is None else _resolve_path(block2_manifest_path)
    layers = _included_layers(include_block3=False)

    packet_root = _resolve_path(output_dir)
    packet_root.mkdir(parents=True, exist_ok=True)
    surface_lineage = _surface_lineage_summary(
        atlas_manifest=atlas_manifest,
        prepare_manifest=prepare_manifest,
        block0_calibration_manifest=block0_calibration_manifest,
        block1_bundle=block1_bundle,
        block2_manifest=block2_manifest,
    )

    plans: list[ArtifactPlan] = []
    plans.extend(_collect_atlas_plans(atlas_manifest))
    plans.extend(
        _collect_block0_calibration_plans(
            block0_calibration_manifest_path=block0_calibration_manifest,
            prepare_manifest_path=prepare_manifest,
        )
    )
    block1_plans = _collect_block1_plans(block1_bundle)
    block2_plans = _collect_block2_plans(block2_manifest)
    block3_plans: list[ArtifactPlan] = []
    plans.extend(block1_plans)
    plans.extend(block2_plans)
    plans.extend(block3_plans)

    records = [
        _materialize_plan(
            plan,
            packet_root=packet_root,
            surface_lineage=surface_lineage,
        )
        for plan in plans
    ]
    block2_review_surface = None
    if block2_manifest is not None and all(plan.artifact_status == "available" for plan in block2_plans):
        block2_review_surface = write_block2_review_surface(
            block2_manifest_path=block2_manifest,
            output_dir=packet_root / "block2",
        )
        records.extend(
            [
                _record_existing_packet_artifact(
                    packet_root=packet_root,
                    layer="block2",
                    artifact_name=artifact.artifact_name,
                    expected_relative_path=artifact.packet_relative_path.removeprefix("block2/"),
                    packet_relative_path=artifact.packet_relative_path,
                    artifact_kind=artifact.artifact_kind,
                    contract_alignment="current_contract_review_surface",
                    format_name=artifact.format,
                    rows_represent=artifact.rows_represent,
                    columns_represent=artifact.columns_represent,
                    claim_scope=artifact.claim_scope,
                    review_role=artifact.review_role,
                    analysis_level=artifact.analysis_level,
                    family_surface_role=artifact.family_surface_role,
                    proof_carrying_status=artifact.proof_carrying_status,
                    source_workflow=artifact.source_workflow,
                    source_manifest_or_bundle=artifact.source_manifest_or_bundle,
                    notes=artifact.notes,
                    review_rank=artifact.review_rank,
                    surface_lineage=surface_lineage,
                    source_path=artifact.source_path,
                )
                for artifact in block2_review_surface.artifacts
            ]
        )
    index_df = pd.DataFrame.from_records(records, columns=INDEX_COLUMNS)
    layer_order = {layer: idx for idx, layer in enumerate(layers)}
    index_df = (
        index_df.assign(_layer_order=index_df["layer"].map(layer_order))
        .sort_values(["_layer_order", "review_rank", "artifact_name"], kind="mergesort")
        .drop(columns="_layer_order")
        .reset_index(drop=True)
    )

    index_path = packet_root / PACKET_INDEX_FILENAME
    index_df.to_csv(index_path, index=False)

    layer_review_index_paths = _write_layer_review_indexes(
        index_df,
        packet_root=packet_root,
        layers=layers,
    )
    layer_manifest_paths = _write_layer_manifests(
        index_df,
        packet_root=packet_root,
        layer_review_index_paths=layer_review_index_paths,
        layers=layers,
        surface_lineage=surface_lineage,
    )
    human_index_path = _write_human_index(
        index_df,
        packet_root=packet_root,
        layers=layers,
        surface_lineage=surface_lineage,
    )

    manifest_payload = {
        "workflow_name": "write_task_a_result_packet",
        "packet_role": PACKET_ROLE,
        "packet_spec_version": PACKET_SPEC_VERSION,
        "scientific_interpretation_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "packet_root": str(packet_root),
        "central_index_path": str(index_path),
        "human_index_path": str(human_index_path),
        "layer_manifest_paths": {layer: str(path) for layer, path in sorted(layer_manifest_paths.items())},
        "layer_review_index_paths": {
            layer: str(path) for layer, path in sorted(layer_review_index_paths.items())
        },
        "included_layers": list(layers),
        "deferred_layers": list(_deferred_layers(include_block3=False)),
        "surface_lineage": surface_lineage,
        "input_sources": {
            "atlas_manifest_path": str(atlas_manifest),
            "prepare_manifest_path": str(prepare_manifest),
            "block0_calibration_manifest_path": str(block0_calibration_manifest),
            "block1_bundle_path": str(block1_bundle) if block1_bundle is not None else None,
            "block2_manifest_path": str(block2_manifest) if block2_manifest is not None else None,
            "block3_manifest_path": None,
        },
        "block1_surface_policy": "faithful_current_state",
        "block2_surface_policy": "faithful_current_state",
        "block3_surface_policy": "deferred_non_authority_pending_clean_bridge_spec",
        "historical_proxy_output_policy": (
            "proxy-history Block 0 artifacts are rejected and must not be relabeled as canonical rerun evidence"
        ),
        "artifact_counts": {
            "total": int(len(index_df)),
            "available": int((index_df["artifact_status"] == "available").sum()),
            "missing_on_disk": int((index_df["artifact_status"] == "missing_on_disk").sum()),
            "by_layer": {
                layer: {
                    "total": int((index_df["layer"] == layer).sum()),
                    "available": int(
                        ((index_df["layer"] == layer) & (index_df["artifact_status"] == "available")).sum()
                    ),
                    "missing_on_disk": int(
                        ((index_df["layer"] == layer) & (index_df["artifact_status"] == "missing_on_disk")).sum()
                    ),
                }
                for layer in layers
            },
        },
    }
    manifest_path = packet_root / PACKET_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    validate_task_a_result_packet(manifest_path)
    return TaskAResultPacket(
        packet_root=packet_root,
        manifest_path=manifest_path,
        index_path=index_path,
        human_index_path=human_index_path,
        layer_manifest_paths=layer_manifest_paths,
        layer_review_index_paths=layer_review_index_paths,
    )


__all__ = [
    "HUMAN_INDEX_FILENAME",
    "PACKET_INDEX_FILENAME",
    "PACKET_MANIFEST_FILENAME",
    "TaskAResultPacket",
    "validate_task_a_result_packet",
    "write_task_a_result_packet",
]
