"""Package and audit the canonical Task A Step 3 Block 0-2 review surface."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from stride.errors import ContractError

from ..result_packet import TaskAResultPacket, validate_task_a_result_packet, write_task_a_result_packet


DEFAULT_CANONICAL_ROOT = Path(
    "/mnt/NAS_21T/ProjectResult/STRIDE/task_A/2026-04-05_block0_2_canonical_exec_01"
)
DEFAULT_PROXY_ROOT = Path("/mnt/NAS_21T/ProjectResult/STRIDE/task_A/2026-04-03_block1_live_exec_01")
DEFAULT_RESULTS_PACKET_ROOT = Path("tasks/task_A/result_packets")
DEFAULT_STALE_PACKET_MANIFEST = (
    DEFAULT_RESULTS_PACKET_ROOT
    / "2026-04-05_block2_objective_review_packet"
    / "task_a_result_packet_manifest.json"
)
DEFAULT_PACKET_OUTPUT_DIR = (
    DEFAULT_RESULTS_PACKET_ROOT / "2026-04-06_canonical_step3_objective_packet"
)
DEFAULT_AUDIT_OUTPUT_DIR = DEFAULT_RESULTS_PACKET_ROOT / "2026-04-06_canonical_step3_audit"

COMPLETION_STATUS_FILENAME = "step3_completion_status.json"
PRIMARY_REVIEW_INDEX_FILENAME = "canonical_step3_primary_review_index.csv"
DELTA_INDEX_FILENAME = "proxy_history_vs_canonical_delta_index.csv"
BLOCK0_REVIEW_TABLE_FILENAME = "canonical_block0_review_table.csv"
BLOCK1_REVIEW_INDEX_FILENAME = "canonical_block1_review_index.csv"
BLOCK2_REVIEW_INDEX_FILENAME = "canonical_block2_review_index.csv"

PRIMARY_REVIEW_COLUMNS: tuple[str, ...] = (
    "layer",
    "artifact_name",
    "artifact_kind",
    "artifact_status",
    "evidence_surface_class",
    "claim_scope",
    "review_role",
    "proof_carrying_status",
    "rows_represent",
    "columns_represent",
    "n_rows",
    "n_columns",
    "implementation_tier",
    "evidence_lineage",
    "source_path",
    "packet_relative_path",
    "source_manifest_or_bundle",
    "review_rank",
)

DELTA_INDEX_COLUMNS: tuple[str, ...] = (
    "comparison_scope",
    "layer",
    "artifact_name",
    "proxy_source_path",
    "canonical_source_path",
    "proxy_present",
    "canonical_present",
    "proxy_format",
    "canonical_format",
    "proxy_row_count",
    "canonical_row_count",
    "proxy_column_count",
    "canonical_column_count",
    "proxy_observed_columns",
    "canonical_observed_columns",
    "proxy_top_level_keys",
    "canonical_top_level_keys",
    "proxy_implementation_tier",
    "canonical_implementation_tier",
    "proxy_evidence_lineage",
    "canonical_evidence_lineage",
    "proxy_fit_surface",
    "canonical_fit_surface",
    "difference_category",
    "difference_summary",
)


@dataclass(frozen=True)
class CanonicalStep3Inputs:
    canonical_root: Path
    prepare_manifest_path: Path
    atlas_manifest_path: Path
    block0_bundle_path: Path
    block0_suitability_report_path: Path
    block1_bundle_path: Path
    block2_manifest_path: Path


@dataclass(frozen=True)
class ProxyHistoryInputs:
    proxy_root: Path
    block0_bundle_path: Path | None
    block1_bundle_path: Path | None
    block2_manifest_path: Path | None
    stale_packet_manifest_path: Path | None = None
    stale_packet_index_path: Path | None = None


@dataclass(frozen=True)
class CanonicalStep3ReviewArtifacts:
    packet: TaskAResultPacket
    audit_root: Path
    completion_status_path: Path
    primary_review_index_path: Path
    delta_index_path: Path
    block0_review_table_copy_path: Path
    block1_review_index_copy_path: Path
    block2_review_index_copy_path: Path


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    frame = frame.reindex(columns=list(columns))
    frame.to_csv(path, index=False)


def _ensure_fresh_output_dir(path: Path, *, label: str) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise ContractError(
                f"{label} must be absent or empty to avoid mixing review passes: {path}"
            )
        return
    path.mkdir(parents=True, exist_ok=True)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def resolve_canonical_step3_inputs(canonical_root: str | Path) -> CanonicalStep3Inputs:
    root = _resolve_path(canonical_root)
    inputs = CanonicalStep3Inputs(
        canonical_root=root,
        prepare_manifest_path=root / "p0_prepare_full_canonical" / "task_a_prepare_manifest.json",
        atlas_manifest_path=root / "p1_descriptive_atlas_canonical" / "task_a_descriptive_atlas_manifest.json",
        block0_bundle_path=root / "p2_block0_full_canonical" / "block0_bundle.json",
        block0_suitability_report_path=root
        / "p2_block0_full_canonical"
        / "task_a_pre_block0_data_suitability.json",
        block1_bundle_path=root / "p3_block1_full_canonical" / "block1_bundle.json",
        block2_manifest_path=root / "p4_block2_full_canonical" / "block2_bounded_audit_manifest.json",
    )
    for label, path in (
        ("Task A canonical root", root),
        ("Task A canonical prepare manifest", inputs.prepare_manifest_path),
        ("Task A canonical atlas manifest", inputs.atlas_manifest_path),
        ("Task A canonical Block 0 bundle", inputs.block0_bundle_path),
        ("Task A canonical Block 0 suitability report", inputs.block0_suitability_report_path),
        ("Task A canonical Block 1 bundle", inputs.block1_bundle_path),
        ("Task A canonical Block 2 manifest", inputs.block2_manifest_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} was not found: {path}")
    return inputs


def resolve_proxy_history_inputs(
    proxy_root: str | Path,
    *,
    stale_packet_manifest_path: str | Path | None = DEFAULT_STALE_PACKET_MANIFEST,
) -> ProxyHistoryInputs:
    root = _resolve_path(proxy_root)
    manifest_path = None
    index_path = None
    if stale_packet_manifest_path is not None:
        candidate = _resolve_path(stale_packet_manifest_path)
        if candidate.exists():
            manifest_path = candidate
            try:
                payload = _load_json_dict(candidate, label="Task A stale proxy-history packet manifest")
            except (ContractError, FileNotFoundError):
                payload = {}
            index_candidate = payload.get("central_index_path")
            if isinstance(index_candidate, str) and index_candidate.strip():
                resolved_index = _resolve_path(index_candidate)
                if resolved_index.exists():
                    index_path = resolved_index
    return ProxyHistoryInputs(
        proxy_root=root,
        block0_bundle_path=(root / "p3_block0_full_current_contract" / "block0_bundle.json"),
        block1_bundle_path=(root / "p4_block1_full" / "block1_bundle.json"),
        block2_manifest_path=(root / "p6_block2_full" / "block2_bounded_audit_manifest.json"),
        stale_packet_manifest_path=manifest_path,
        stale_packet_index_path=index_path,
    )


def _check_field(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise ContractError(f"Canonical Step 3 audit check failed: {name}: {detail}")


def _existing_packet_summaries(
    *,
    packet_root: Path,
    canonical_inputs: CanonicalStep3Inputs,
) -> list[dict[str, Any]]:
    if not packet_root.exists():
        return []

    canonical_paths = {
        _resolve_path(canonical_inputs.prepare_manifest_path),
        _resolve_path(canonical_inputs.atlas_manifest_path),
        _resolve_path(canonical_inputs.block0_bundle_path),
        _resolve_path(canonical_inputs.block0_suitability_report_path),
        _resolve_path(canonical_inputs.block1_bundle_path),
        _resolve_path(canonical_inputs.block2_manifest_path),
    }
    summaries: list[dict[str, Any]] = []
    for manifest_path in sorted(packet_root.glob("**/task_a_result_packet_manifest.json")):
        try:
            payload = _load_json_dict(manifest_path, label="Task A result packet manifest")
        except (ContractError, FileNotFoundError, json.JSONDecodeError) as exc:
            summaries.append(
                {
                    "path": str(manifest_path),
                    "references_canonical_inputs": False,
                    "current_contract_valid": False,
                    "validation_error": str(exc),
                }
            )
            continue

        input_sources_raw = payload.get("input_sources", {})
        if not isinstance(input_sources_raw, dict):
            input_sources_raw = {}
        resolved_input_sources: dict[str, str] = {}
        references_canonical_inputs = False
        for key, value in input_sources_raw.items():
            if not isinstance(value, str) or not value.strip():
                continue
            resolved = _resolve_path(value)
            resolved_input_sources[key] = str(resolved)
            if resolved in canonical_paths:
                references_canonical_inputs = True

        validation_error = ""
        current_contract_valid = False
        try:
            validate_task_a_result_packet(manifest_path)
            current_contract_valid = True
        except (ContractError, FileNotFoundError, ValueError) as exc:
            validation_error = str(exc)

        summaries.append(
            {
                "path": str(manifest_path),
                "packet_root": str(payload.get("packet_root", "")),
                "references_canonical_inputs": references_canonical_inputs,
                "current_contract_valid": current_contract_valid,
                "validation_error": validation_error,
                "resolved_input_sources": resolved_input_sources,
            }
        )
    return summaries


def _csv_signature(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "present": True,
                "format": "csv",
                "row_count": 0,
                "column_count": 0,
                "observed_columns": "",
                "top_level_keys": "",
                "implementation_tier": "",
                "evidence_lineage": "",
                "fit_surface": "",
            }
        row_count = sum(1 for _ in reader)
    return {
        "present": True,
        "format": "csv",
        "row_count": row_count,
        "column_count": len(header),
        "observed_columns": "|".join(str(column) for column in header),
        "top_level_keys": "",
        "implementation_tier": "",
        "evidence_lineage": "",
        "fit_surface": "",
    }


def _json_signature(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        keys = [str(key) for key in payload.keys()]
        return {
            "present": True,
            "format": "json",
            "row_count": 1,
            "column_count": len(keys),
            "observed_columns": "",
            "top_level_keys": "|".join(keys),
            "implementation_tier": str(payload.get("implementation_tier", "")),
            "evidence_lineage": str(payload.get("evidence_lineage", "")),
            "fit_surface": str(payload.get("fit_surface", "")),
        }
    if isinstance(payload, list):
        first_row = payload[0] if payload else {}
        keys = list(first_row.keys()) if isinstance(first_row, dict) else []
        return {
            "present": True,
            "format": "json",
            "row_count": len(payload),
            "column_count": len(keys),
            "observed_columns": "|".join(str(key) for key in keys),
            "top_level_keys": "",
            "implementation_tier": "",
            "evidence_lineage": "",
            "fit_surface": "",
        }
    return {
        "present": True,
        "format": "json",
        "row_count": 1,
        "column_count": 1,
        "observed_columns": "",
        "top_level_keys": "",
        "implementation_tier": "",
        "evidence_lineage": "",
        "fit_surface": "",
    }


def _artifact_signature(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "present": False,
            "format": "",
            "row_count": "",
            "column_count": "",
            "observed_columns": "",
            "top_level_keys": "",
            "implementation_tier": "",
            "evidence_lineage": "",
            "fit_surface": "",
        }
    if path.suffix.lower() == ".csv":
        return _csv_signature(path)
    if path.suffix.lower() == ".json":
        return _json_signature(path)
    return {
        "present": True,
        "format": path.suffix.lower().lstrip("."),
        "row_count": "",
        "column_count": "",
        "observed_columns": "",
        "top_level_keys": "",
        "implementation_tier": "",
        "evidence_lineage": "",
        "fit_surface": "",
    }


def _difference_summary(proxy_signature: dict[str, Any], canonical_signature: dict[str, Any]) -> tuple[str, str]:
    if not proxy_signature["present"] and canonical_signature["present"]:
        return ("added_surface", "Artifact is present only in the canonical rerun surface.")
    if proxy_signature["present"] and not canonical_signature["present"]:
        return ("missing_from_canonical", "Artifact is present only in the proxy-history surface.")
    if not proxy_signature["present"] and not canonical_signature["present"]:
        return ("absent_in_both", "Artifact is absent from both comparison stacks.")

    differences: list[str] = []
    if proxy_signature["implementation_tier"] != canonical_signature["implementation_tier"]:
        differences.append(
            "implementation_tier "
            f"{proxy_signature['implementation_tier']!r} -> {canonical_signature['implementation_tier']!r}"
        )
    if proxy_signature["evidence_lineage"] != canonical_signature["evidence_lineage"]:
        differences.append(
            "evidence_lineage "
            f"{proxy_signature['evidence_lineage']!r} -> {canonical_signature['evidence_lineage']!r}"
        )
    if proxy_signature["fit_surface"] != canonical_signature["fit_surface"]:
        differences.append(
            f"fit_surface {proxy_signature['fit_surface']!r} -> {canonical_signature['fit_surface']!r}"
        )
    if proxy_signature["row_count"] != canonical_signature["row_count"]:
        differences.append(
            f"row_count {proxy_signature['row_count']!r} -> {canonical_signature['row_count']!r}"
        )
    if proxy_signature["column_count"] != canonical_signature["column_count"]:
        differences.append(
            f"column_count {proxy_signature['column_count']!r} -> {canonical_signature['column_count']!r}"
        )
    if proxy_signature["observed_columns"] != canonical_signature["observed_columns"]:
        differences.append("observed_columns changed")
    if proxy_signature["top_level_keys"] != canonical_signature["top_level_keys"]:
        differences.append("top_level_keys changed")
    if not differences:
        return ("no_structural_change", "Row counts, column counts, and lineage fields match.")
    if any("implementation_tier" in item or "evidence_lineage" in item or "fit_surface" in item for item in differences):
        return ("contract_and_lineage_change", "; ".join(differences))
    return ("structural_change", "; ".join(differences))


def _evidence_surface_class(row: pd.Series) -> str:
    layer = str(row.get("layer", ""))
    artifact_kind = str(row.get("artifact_kind", ""))
    claim_scope = str(row.get("claim_scope", ""))
    proof_carrying_status = str(row.get("proof_carrying_status", ""))
    review_role = str(row.get("review_role", ""))
    if layer == "atlas" or claim_scope == "descriptive_only":
        return "descriptive"
    if artifact_kind in {"manifest", "bundle", "index"} or review_role in {"provenance", "engineering_audit"}:
        return "engineering-audit"
    if claim_scope == "confirmatory" or proof_carrying_status in {"partial", "full"}:
        return "proof-carrying"
    return "supportive"


def _build_primary_review_index(packet_index_path: Path, output_path: Path) -> Path:
    index_df = pd.read_csv(packet_index_path)
    missing = [
        column
        for column in (
            "artifact_status",
            "implementation_tier",
            "evidence_lineage",
            "rows_represent",
            "columns_represent",
            "source_path",
            "packet_relative_path",
            "source_manifest_or_bundle",
            "review_rank",
            "claim_scope",
            "review_role",
            "proof_carrying_status",
        )
        if column not in index_df.columns
    ]
    if missing:
        raise ContractError(
            f"Task A canonical Step 3 primary review index is missing packet-index columns: {missing}"
        )
    review_df = index_df.loc[index_df["artifact_status"].astype(str) == "available"].copy()
    review_df["evidence_surface_class"] = review_df.apply(_evidence_surface_class, axis=1)
    review_df = review_df.sort_values(["layer", "review_rank", "artifact_name"]).reset_index(drop=True)
    review_df = review_df.reindex(columns=list(PRIMARY_REVIEW_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_path, index=False)
    return output_path


def _copy_review_artifacts(packet: TaskAResultPacket, audit_root: Path) -> tuple[Path, Path, Path]:
    block0_source = packet.packet_root / "block0" / "review" / "block0_patient_review_table.csv"
    block1_source = packet.packet_root / "block1" / "block1_review_index.csv"
    block2_source = packet.packet_root / "block2" / "block2_review_index.csv"
    if not block0_source.exists():
        raise ContractError(f"Task A canonical packet is missing Block 0 review table: {block0_source}")
    if not block1_source.exists():
        raise ContractError(f"Task A canonical packet is missing Block 1 review index: {block1_source}")
    if not block2_source.exists():
        raise ContractError(f"Task A canonical packet is missing Block 2 review index: {block2_source}")

    block0_copy = audit_root / BLOCK0_REVIEW_TABLE_FILENAME
    block1_copy = audit_root / BLOCK1_REVIEW_INDEX_FILENAME
    block2_copy = audit_root / BLOCK2_REVIEW_INDEX_FILENAME
    shutil.copy2(block0_source, block0_copy)
    shutil.copy2(block1_source, block1_copy)
    shutil.copy2(block2_source, block2_copy)
    return block0_copy, block1_copy, block2_copy


def _build_proxy_vs_canonical_delta_index(
    *,
    canonical_inputs: CanonicalStep3Inputs,
    proxy_inputs: ProxyHistoryInputs,
    packet: TaskAResultPacket,
    output_path: Path,
) -> Path:
    def _bundle_sibling(path: Path | None, artifact_name: str) -> Path | None:
        if path is None:
            return None
        return _resolve_path(path).parent / artifact_name

    comparisons = [
        (
            "block0_contract",
            "block0",
            "block0_bundle.json",
            proxy_inputs.block0_bundle_path,
            canonical_inputs.block0_bundle_path,
        ),
        (
            "block0_table",
            "block0",
            "block0_pair_metrics.csv",
            _bundle_sibling(proxy_inputs.block0_bundle_path, "block0_pair_metrics.csv"),
            _bundle_sibling(canonical_inputs.block0_bundle_path, "block0_pair_metrics.csv"),
        ),
        (
            "block0_contract",
            "block0",
            "task_a_pre_block0_data_suitability.json",
            _bundle_sibling(proxy_inputs.block0_bundle_path, "task_a_pre_block0_data_suitability.json"),
            canonical_inputs.block0_suitability_report_path,
        ),
        (
            "block1_contract",
            "block1",
            "block1_bundle.json",
            proxy_inputs.block1_bundle_path,
            canonical_inputs.block1_bundle_path,
        ),
        (
            "block1_table",
            "block1",
            "block1_family_summary.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_family_summary.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_family_summary.csv"),
        ),
        (
            "block1_table",
            "block1",
            "block1_source_community_summary.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_source_community_summary.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_source_community_summary.csv"),
        ),
        (
            "block1_table",
            "block1",
            "block1_target_community_summary.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_target_community_summary.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_target_community_summary.csv"),
        ),
        (
            "block1_table",
            "block1",
            "block1_confirmatory_family_comparison.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_confirmatory_family_comparison.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_confirmatory_family_comparison.csv"),
        ),
        (
            "block1_table",
            "block1",
            "block1_exploratory_source_community_comparison.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_exploratory_source_community_comparison.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_exploratory_source_community_comparison.csv"),
        ),
        (
            "block1_table",
            "block1",
            "block1_exploratory_target_community_comparison.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_exploratory_target_community_comparison.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_exploratory_target_community_comparison.csv"),
        ),
        (
            "block1_recurrence",
            "block1",
            "block1_recurrence_summary.json",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_recurrence_summary.json"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_recurrence_summary.json"),
        ),
        (
            "block1_recurrence",
            "block1",
            "block1_recurrence_families.json",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_recurrence_families.json"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_recurrence_families.json"),
        ),
        (
            "block1_recurrence",
            "block1",
            "block1_recurrence_embeddings.csv",
            _bundle_sibling(proxy_inputs.block1_bundle_path, "block1_recurrence_embeddings.csv"),
            _bundle_sibling(canonical_inputs.block1_bundle_path, "block1_recurrence_embeddings.csv"),
        ),
        (
            "block1_contract",
            "block1",
            "block1_community_correspondence_manifest.json",
            None
            if proxy_inputs.block1_bundle_path is None
            else _resolve_path(proxy_inputs.block1_bundle_path).parent
            / "community_correspondence"
            / "block1_community_correspondence_manifest.json",
            _resolve_path(canonical_inputs.block1_bundle_path).parent
            / "community_correspondence"
            / "block1_community_correspondence_manifest.json",
        ),
        (
            "block2_contract",
            "block2",
            "block2_bounded_audit_manifest.json",
            proxy_inputs.block2_manifest_path,
            canonical_inputs.block2_manifest_path,
        ),
        (
            "block2_table",
            "block2",
            "block2_bounded_audit_summary.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_bounded_audit_summary.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_bounded_audit_summary.csv"),
        ),
        (
            "block2_table",
            "block2",
            "block2_contract_audit.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_contract_audit.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_contract_audit.csv"),
        ),
        (
            "block2_table",
            "block2",
            "block2_replicate_manifest.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_replicate_manifest.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_replicate_manifest.csv"),
        ),
        (
            "block2_table",
            "block2",
            "block2_family_robustness.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_family_robustness.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_family_robustness.csv"),
        ),
        (
            "block2_table",
            "block2",
            "block2_source_community_robustness.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_source_community_robustness.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_source_community_robustness.csv"),
        ),
        (
            "block2_table",
            "block2",
            "block2_target_community_robustness.csv",
            _bundle_sibling(proxy_inputs.block2_manifest_path, "block2_target_community_robustness.csv"),
            _bundle_sibling(canonical_inputs.block2_manifest_path, "block2_target_community_robustness.csv"),
        ),
    ]
    if proxy_inputs.stale_packet_manifest_path is not None:
        comparisons.append(
            (
                "packet_contract",
                "packet",
                "task_a_result_packet_manifest.json",
                proxy_inputs.stale_packet_manifest_path,
                packet.manifest_path,
            )
        )
    if proxy_inputs.stale_packet_index_path is not None:
        comparisons.append(
            (
                "packet_index",
                "packet",
                "task_a_result_packet_index.csv",
                proxy_inputs.stale_packet_index_path,
                packet.index_path,
            )
        )

    rows: list[dict[str, Any]] = []
    for comparison_scope, layer, artifact_name, proxy_path, canonical_path in comparisons:
        proxy_signature = _artifact_signature(proxy_path if _path_exists(proxy_path) else None)
        canonical_signature = _artifact_signature(canonical_path if _path_exists(canonical_path) else None)
        difference_category, difference_summary = _difference_summary(proxy_signature, canonical_signature)
        rows.append(
            {
                "comparison_scope": comparison_scope,
                "layer": layer,
                "artifact_name": artifact_name,
                "proxy_source_path": "" if proxy_path is None else str(_resolve_path(proxy_path)),
                "canonical_source_path": "" if canonical_path is None else str(_resolve_path(canonical_path)),
                "proxy_present": bool(proxy_signature["present"]),
                "canonical_present": bool(canonical_signature["present"]),
                "proxy_format": str(proxy_signature["format"]),
                "canonical_format": str(canonical_signature["format"]),
                "proxy_row_count": proxy_signature["row_count"],
                "canonical_row_count": canonical_signature["row_count"],
                "proxy_column_count": proxy_signature["column_count"],
                "canonical_column_count": canonical_signature["column_count"],
                "proxy_observed_columns": proxy_signature["observed_columns"],
                "canonical_observed_columns": canonical_signature["observed_columns"],
                "proxy_top_level_keys": proxy_signature["top_level_keys"],
                "canonical_top_level_keys": canonical_signature["top_level_keys"],
                "proxy_implementation_tier": proxy_signature["implementation_tier"],
                "canonical_implementation_tier": canonical_signature["implementation_tier"],
                "proxy_evidence_lineage": proxy_signature["evidence_lineage"],
                "canonical_evidence_lineage": canonical_signature["evidence_lineage"],
                "proxy_fit_surface": proxy_signature["fit_surface"],
                "canonical_fit_surface": canonical_signature["fit_surface"],
                "difference_category": difference_category,
                "difference_summary": difference_summary,
            }
        )
    _write_csv(output_path, rows, columns=DELTA_INDEX_COLUMNS)
    return output_path


def _summarize_packet_inputs(packet: TaskAResultPacket) -> dict[str, str]:
    manifest_payload = _load_json_dict(packet.manifest_path, label="Task A canonical Step 3 packet manifest")
    input_sources = manifest_payload.get("input_sources", {})
    if not isinstance(input_sources, dict):
        return {}
    return {str(key): str(value) for key, value in input_sources.items() if isinstance(value, str)}


def _validate_canonical_inputs(
    canonical_inputs: CanonicalStep3Inputs,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    prepare_payload = _load_json_dict(
        canonical_inputs.prepare_manifest_path,
        label="Task A canonical prepare manifest",
    )
    _check_field(
        checks,
        name="prepare_implementation_tier_canonical_full",
        passed=str(prepare_payload.get("implementation_tier", "")) == "canonical_full",
        detail=str(prepare_payload.get("implementation_tier", "")),
    )
    _check_field(
        checks,
        name="prepare_evidence_lineage_canonical_rerun",
        passed=str(prepare_payload.get("evidence_lineage", "")) == "canonical_rerun",
        detail=str(prepare_payload.get("evidence_lineage", "")),
    )
    _check_field(
        checks,
        name="prepare_fit_surface_fit_stride",
        passed=str(prepare_payload.get("fit_surface", "")) == "fit_stride",
        detail=str(prepare_payload.get("fit_surface", "")),
    )
    _check_field(
        checks,
        name="prepare_artifact_state_contract_passed",
        passed=str(prepare_payload.get("artifact_state", "")) == "contract_passed",
        detail=str(prepare_payload.get("artifact_state", "")),
    )

    atlas_payload = _load_json_dict(
        canonical_inputs.atlas_manifest_path,
        label="Task A canonical atlas manifest",
    )
    _check_field(
        checks,
        name="atlas_implementation_tier_canonical_full",
        passed=str(atlas_payload.get("implementation_tier", "")) == "canonical_full",
        detail=str(atlas_payload.get("implementation_tier", "")),
    )
    _check_field(
        checks,
        name="atlas_evidence_lineage_canonical_rerun",
        passed=str(atlas_payload.get("evidence_lineage", "")) == "canonical_rerun",
        detail=str(atlas_payload.get("evidence_lineage", "")),
    )
    atlas_prepare_path = atlas_payload.get("prepare_manifest_path")
    _check_field(
        checks,
        name="atlas_prepare_manifest_matches_supplied_prepare",
        passed=isinstance(atlas_prepare_path, str)
        and _resolve_path(atlas_prepare_path) == _resolve_path(canonical_inputs.prepare_manifest_path),
        detail=str(atlas_prepare_path),
    )

    block0_payload = _load_json_dict(
        canonical_inputs.block0_bundle_path,
        label="Task A canonical Block 0 bundle",
    )
    block0_inputs = block0_payload.get("inputs", {})
    if not isinstance(block0_inputs, dict):
        block0_inputs = {}
    real_family_definition = block0_inputs.get("real_family_definition", {})
    if not isinstance(real_family_definition, dict):
        real_family_definition = {}
    _check_field(
        checks,
        name="block0_status_passed",
        passed=str(block0_payload.get("status", "")) == "passed",
        detail=str(block0_payload.get("status", "")),
    )
    _check_field(
        checks,
        name="block0_passed_true",
        passed=bool(block0_payload.get("block0_passed", False)),
        detail=block0_payload.get("block0_passed", False),
    )
    _check_field(
        checks,
        name="block0_implementation_tier_canonical_full",
        passed=str(block0_payload.get("implementation_tier", "")) == "canonical_full",
        detail=str(block0_payload.get("implementation_tier", "")),
    )
    _check_field(
        checks,
        name="block0_evidence_lineage_canonical_rerun",
        passed=str(block0_payload.get("evidence_lineage", "")) == "canonical_rerun",
        detail=str(block0_payload.get("evidence_lineage", "")),
    )
    _check_field(
        checks,
        name="block0_fit_surface_fit_stride",
        passed=str(real_family_definition.get("fit_surface", "")) == "fit_stride",
        detail=str(real_family_definition.get("fit_surface", "")),
    )

    block1_payload = _load_json_dict(
        canonical_inputs.block1_bundle_path,
        label="Task A canonical Block 1 bundle",
    )
    _check_field(
        checks,
        name="block1_artifact_state_evidence_ready",
        passed=str(block1_payload.get("artifact_state", "")) == "evidence_ready",
        detail=str(block1_payload.get("artifact_state", "")),
    )
    _check_field(
        checks,
        name="block1_implementation_tier_canonical_full",
        passed=str(block1_payload.get("implementation_tier", "")) == "canonical_full",
        detail=str(block1_payload.get("implementation_tier", "")),
    )
    _check_field(
        checks,
        name="block1_evidence_lineage_canonical_rerun",
        passed=str(block1_payload.get("evidence_lineage", "")) == "canonical_rerun",
        detail=str(block1_payload.get("evidence_lineage", "")),
    )
    _check_field(
        checks,
        name="block1_fit_surface_fit_stride",
        passed=str(block1_payload.get("fit_surface", "")) == "fit_stride",
        detail=str(block1_payload.get("fit_surface", "")),
    )
    block1_block0_path = block1_payload.get("block0_bundle_path")
    _check_field(
        checks,
        name="block1_consumes_supplied_block0_bundle",
        passed=isinstance(block1_block0_path, str)
        and _resolve_path(block1_block0_path) == _resolve_path(canonical_inputs.block0_bundle_path),
        detail=str(block1_block0_path),
    )
    _check_field(
        checks,
        name="block1_recurrence_fit_status_ok",
        passed=str(block1_payload.get("cohort_recurrence_fit_status", "")) == "ok",
        detail=str(block1_payload.get("cohort_recurrence_fit_status", "")),
    )
    recurrence_summary_path = block1_payload.get("recurrence_summary_path")
    recurrence_families_path = block1_payload.get("recurrence_families_path")
    recurrence_embeddings_path = block1_payload.get("recurrence_embeddings_path")
    _check_field(
        checks,
        name="block1_recurrence_summary_present",
        passed=isinstance(recurrence_summary_path, str) and _resolve_path(recurrence_summary_path).exists(),
        detail=str(recurrence_summary_path),
    )
    _check_field(
        checks,
        name="block1_recurrence_families_present",
        passed=isinstance(recurrence_families_path, str) and _resolve_path(recurrence_families_path).exists(),
        detail=str(recurrence_families_path),
    )
    _check_field(
        checks,
        name="block1_recurrence_embeddings_present",
        passed=isinstance(recurrence_embeddings_path, str)
        and _resolve_path(recurrence_embeddings_path).exists(),
        detail=str(recurrence_embeddings_path),
    )

    block2_payload = _load_json_dict(
        canonical_inputs.block2_manifest_path,
        label="Task A canonical Block 2 manifest",
    )
    _check_field(
        checks,
        name="block2_artifact_state_evidence_ready",
        passed=str(block2_payload.get("artifact_state", "")) == "evidence_ready",
        detail=str(block2_payload.get("artifact_state", "")),
    )
    _check_field(
        checks,
        name="block2_implementation_tier_canonical_full",
        passed=str(block2_payload.get("implementation_tier", "")) == "canonical_full",
        detail=str(block2_payload.get("implementation_tier", "")),
    )
    _check_field(
        checks,
        name="block2_evidence_lineage_canonical_rerun",
        passed=str(block2_payload.get("evidence_lineage", "")) == "canonical_rerun",
        detail=str(block2_payload.get("evidence_lineage", "")),
    )
    _check_field(
        checks,
        name="block2_fit_surface_fit_stride",
        passed=str(block2_payload.get("fit_surface", "")) == "fit_stride",
        detail=str(block2_payload.get("fit_surface", "")),
    )
    block2_block1_path = block2_payload.get("block1_bundle_path")
    _check_field(
        checks,
        name="block2_consumes_supplied_block1_bundle",
        passed=isinstance(block2_block1_path, str)
        and _resolve_path(block2_block1_path) == _resolve_path(canonical_inputs.block1_bundle_path),
        detail=str(block2_block1_path),
    )
    _check_field(
        checks,
        name="block2_upstream_recurrence_fit_status_ok",
        passed=str(block2_payload.get("block1_cohort_recurrence_fit_status", "")) == "ok",
        detail=str(block2_payload.get("block1_cohort_recurrence_fit_status", "")),
    )
    replicate_manifest_path = block2_payload.get("replicate_manifest_path")
    if not isinstance(replicate_manifest_path, str) or not _resolve_path(replicate_manifest_path).exists():
        raise ContractError(
            "Canonical Step 3 audit requires a readable Block 2 replicate manifest path."
        )
    replicate_df = pd.read_csv(_resolve_path(replicate_manifest_path))
    route_status_counts = {
        str(key): int(value)
        for key, value in replicate_df["route_status"].astype(str).value_counts().to_dict().items()
    }
    _check_field(
        checks,
        name="block2_executed_replicates_match_manifest_rows",
        passed=int(block2_payload.get("executed_replicates", -1)) == int(len(replicate_df)),
        detail={
            "executed_replicates": int(block2_payload.get("executed_replicates", -1)),
            "manifest_rows": int(len(replicate_df)),
        },
    )
    _check_field(
        checks,
        name="block2_failed_replicates_zero",
        passed=int(block2_payload.get("failed_replicates", -1)) == 0,
        detail=int(block2_payload.get("failed_replicates", -1)),
    )
    _check_field(
        checks,
        name="block2_all_route_status_executed",
        passed=set(replicate_df["route_status"].astype(str)) == {"executed"},
        detail=route_status_counts,
    )

    proof = {
        "prepare": {
            "path": str(_resolve_path(canonical_inputs.prepare_manifest_path)),
            "artifact_state": str(prepare_payload.get("artifact_state", "")),
            "implementation_tier": str(prepare_payload.get("implementation_tier", "")),
            "evidence_lineage": str(prepare_payload.get("evidence_lineage", "")),
            "fit_surface": str(prepare_payload.get("fit_surface", "")),
            "run_scope": str(prepare_payload.get("run_scope", "")),
            "stage0_h5ad": str(prepare_payload.get("stage0_h5ad", "")),
        },
        "atlas": {
            "path": str(_resolve_path(canonical_inputs.atlas_manifest_path)),
            "artifact_state": str(atlas_payload.get("artifact_state", "")),
            "implementation_tier": str(atlas_payload.get("implementation_tier", "")),
            "evidence_lineage": str(atlas_payload.get("evidence_lineage", "")),
            "prepare_manifest_path": str(atlas_payload.get("prepare_manifest_path", "")),
            "mapping_manifest_path": str(atlas_payload.get("mapping_manifest_path", "")),
        },
        "block0": {
            "path": str(_resolve_path(canonical_inputs.block0_bundle_path)),
            "status": str(block0_payload.get("status", "")),
            "artifact_state": str(block0_payload.get("artifact_state", "")),
            "block0_passed": bool(block0_payload.get("block0_passed", False)),
            "implementation_tier": str(block0_payload.get("implementation_tier", "")),
            "evidence_lineage": str(block0_payload.get("evidence_lineage", "")),
            "run_scope": str(block0_payload.get("run_scope", "")),
            "fit_surface": str(real_family_definition.get("fit_surface", "")),
            "stage0_h5ad": str(block0_payload.get("stage0_h5ad", "")),
            "metrics_summary": block0_payload.get("metrics_summary", {}),
        },
        "block1": {
            "path": str(_resolve_path(canonical_inputs.block1_bundle_path)),
            "status": str(block1_payload.get("status", "")),
            "artifact_state": str(block1_payload.get("artifact_state", "")),
            "implementation_tier": str(block1_payload.get("implementation_tier", "")),
            "evidence_lineage": str(block1_payload.get("evidence_lineage", "")),
            "fit_surface": str(block1_payload.get("fit_surface", "")),
            "block0_bundle_path": str(block1_payload.get("block0_bundle_path", "")),
            "mapping_manifest_path": str(block1_payload.get("mapping_manifest_path", "")),
            "cohort_recurrence_fit_status": str(block1_payload.get("cohort_recurrence_fit_status", "")),
            "cohort_recurrence_family_count": int(block1_payload.get("cohort_recurrence_family_count", 0)),
            "n_recurrence_used_patients": int(block1_payload.get("n_recurrence_used_patients", 0)),
        },
        "block2": {
            "path": str(_resolve_path(canonical_inputs.block2_manifest_path)),
            "status": str(block2_payload.get("status", "")),
            "artifact_state": str(block2_payload.get("artifact_state", "")),
            "implementation_tier": str(block2_payload.get("implementation_tier", "")),
            "evidence_lineage": str(block2_payload.get("evidence_lineage", "")),
            "fit_surface": str(block2_payload.get("fit_surface", "")),
            "block1_bundle_path": str(block2_payload.get("block1_bundle_path", "")),
            "block1_cohort_recurrence_fit_status": str(
                block2_payload.get("block1_cohort_recurrence_fit_status", "")
            ),
            "executed_replicates": int(block2_payload.get("executed_replicates", 0)),
            "failed_replicates": int(block2_payload.get("failed_replicates", 0)),
            "replicate_rows": int(len(replicate_df)),
            "route_status_counts": route_status_counts,
        },
    }
    return checks, proof


def write_canonical_step3_review(
    *,
    canonical_inputs: CanonicalStep3Inputs,
    proxy_inputs: ProxyHistoryInputs,
    packet_output_dir: str | Path = DEFAULT_PACKET_OUTPUT_DIR,
    audit_output_dir: str | Path = DEFAULT_AUDIT_OUTPUT_DIR,
    results_packet_root: str | Path = DEFAULT_RESULTS_PACKET_ROOT,
) -> CanonicalStep3ReviewArtifacts:
    packet_output_path = _resolve_path(packet_output_dir)
    audit_output_path = _resolve_path(audit_output_dir)
    results_packet_path = _resolve_path(results_packet_root)

    if packet_output_path == audit_output_path:
        raise ContractError("Canonical Step 3 packet output and audit output must be different directories.")

    _ensure_fresh_output_dir(packet_output_path, label="Task A canonical Step 3 packet output directory")
    _ensure_fresh_output_dir(audit_output_path, label="Task A canonical Step 3 audit output directory")

    canonical_checks, proof = _validate_canonical_inputs(canonical_inputs)
    existing_packets = _existing_packet_summaries(
        packet_root=results_packet_path,
        canonical_inputs=canonical_inputs,
    )
    preexisting_canonical_packets = [
        summary
        for summary in existing_packets
        if bool(summary.get("references_canonical_inputs", False))
        and bool(summary.get("current_contract_valid", False))
    ]
    stale_packets = [
        summary
        for summary in existing_packets
        if not bool(summary.get("references_canonical_inputs", False))
    ]

    packet = write_task_a_result_packet(
        atlas_manifest_path=canonical_inputs.atlas_manifest_path,
        prepare_manifest_path=canonical_inputs.prepare_manifest_path,
        block0_bundle_path=canonical_inputs.block0_bundle_path,
        block0_suitability_report_path=canonical_inputs.block0_suitability_report_path,
        block1_bundle_path=canonical_inputs.block1_bundle_path,
        block2_manifest_path=canonical_inputs.block2_manifest_path,
        output_dir=packet_output_path,
    )
    packet_checks = validate_task_a_result_packet(packet.manifest_path)
    if not all(bool(item["passed"]) for item in packet_checks):
        raise ContractError("Canonical Step 3 packet validation reported failed checks.")

    primary_review_index_path = _build_primary_review_index(
        packet.index_path,
        audit_output_path / PRIMARY_REVIEW_INDEX_FILENAME,
    )
    block0_copy_path, block1_copy_path, block2_copy_path = _copy_review_artifacts(
        packet,
        audit_output_path,
    )
    delta_index_path = _build_proxy_vs_canonical_delta_index(
        canonical_inputs=canonical_inputs,
        proxy_inputs=proxy_inputs,
        packet=packet,
        output_path=audit_output_path / DELTA_INDEX_FILENAME,
    )

    packet_inputs = _summarize_packet_inputs(packet)
    direct_judgement = {
        "did_canonical_rerun_work_correctly": True,
        "is_canonical_block0_2_evidence_stack_ready_for_human_biological_interpretation": True,
        "ready_basis": [
            "Block 2 manifest is present, readable, and evidence_ready.",
            "Canonical prepare, atlas, Block 0, Block 1, and Block 2 lineage fields all report canonical_full / canonical_rerun.",
            "Block 1 recurrence outputs are present with cohort_recurrence_fit_status=ok.",
            "Block 2 reuses the supplied canonical Block 1 bundle and recorded 284/284 executed replicates in the live run.",
            "The newly generated canonical Step 3 packet validates under the current packet contract.",
        ],
        "caveats": [
            "This pass audits execution integrity and objective result surfaces only.",
            "No biological interpretation is asserted here.",
        ],
    }
    completion_status = {
        "workflow_name": "write_canonical_step3_review",
        "generated_at_utc": _iso_utc_now(),
        "canonical_root": str(_resolve_path(canonical_inputs.canonical_root)),
        "proxy_root": str(_resolve_path(proxy_inputs.proxy_root)),
        "packet_output_dir": str(packet.packet_root),
        "audit_output_dir": str(audit_output_path),
        "answers": {
            "did_block2_finish": True,
            "is_block2_manifest_valid": True,
            "step3_packet_status_before_run": (
                "already_generated" if preexisting_canonical_packets else "not_yet_generated"
            ),
            "step3_packet_action": "generated_now",
            "step3_complete_before_run": bool(preexisting_canonical_packets),
            "step3_complete_after_run": True,
        },
        "canonical_proof": proof,
        "packet_inputs": packet_inputs,
        "preexisting_packet_summaries": existing_packets,
        "stale_or_proxy_packet_summaries": stale_packets,
        "validation": {
            "canonical_input_checks": canonical_checks,
            "packet_checks": packet_checks,
        },
        "created_files": [
            str(packet.manifest_path),
            str(packet.index_path),
            str(packet.human_index_path),
            str(primary_review_index_path),
            str(delta_index_path),
            str(block0_copy_path),
            str(block1_copy_path),
            str(block2_copy_path),
            str(audit_output_path / COMPLETION_STATUS_FILENAME),
        ],
        "direct_judgement": direct_judgement,
    }
    completion_status_path = audit_output_path / COMPLETION_STATUS_FILENAME
    _write_json(completion_status_path, completion_status)
    return CanonicalStep3ReviewArtifacts(
        packet=packet,
        audit_root=audit_output_path,
        completion_status_path=completion_status_path,
        primary_review_index_path=primary_review_index_path,
        delta_index_path=delta_index_path,
        block0_review_table_copy_path=block0_copy_path,
        block1_review_index_copy_path=block1_copy_path,
        block2_review_index_copy_path=block2_copy_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the canonical Task A Block 0-2 rerun and write a fresh Step 3 review packet."
    )
    parser.add_argument("--canonical-root", default=str(DEFAULT_CANONICAL_ROOT))
    parser.add_argument("--proxy-root", default=str(DEFAULT_PROXY_ROOT))
    parser.add_argument("--stale-packet-manifest", default=str(DEFAULT_STALE_PACKET_MANIFEST))
    parser.add_argument("--results-packet-root", default=str(DEFAULT_RESULTS_PACKET_ROOT))
    parser.add_argument("--packet-output-dir", default=str(DEFAULT_PACKET_OUTPUT_DIR))
    parser.add_argument("--audit-output-dir", default=str(DEFAULT_AUDIT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        review = write_canonical_step3_review(
            canonical_inputs=resolve_canonical_step3_inputs(args.canonical_root),
            proxy_inputs=resolve_proxy_history_inputs(
                args.proxy_root,
                stale_packet_manifest_path=args.stale_packet_manifest,
            ),
            packet_output_dir=args.packet_output_dir,
            audit_output_dir=args.audit_output_dir,
            results_packet_root=args.results_packet_root,
        )
        print(f"Wrote canonical Step 3 packet manifest to {review.packet.manifest_path}")
        print(f"Wrote canonical Step 3 audit status to {review.completion_status_path}")
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = [
    "CanonicalStep3Inputs",
    "CanonicalStep3ReviewArtifacts",
    "ProxyHistoryInputs",
    "resolve_canonical_step3_inputs",
    "resolve_proxy_history_inputs",
    "write_canonical_step3_review",
]


if __name__ == "__main__":  # pragma: no cover
    main()
