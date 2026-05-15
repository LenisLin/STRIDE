"""Block 1 analyze surface over generic native relation exports."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from stride.errors import ContractError
from stride.outputs.fit_export import (
    NativeRelationExport,
    read_stride_native_relation_export,
    sha256_file,
)

from .functions.comparisons import (
    COHORT_RELATION_COMPARISON_COLUMNS,
    CONFIRMATORY_FAMILY_COMPARISON_FILENAME,
    DESCRIPTIVE_SOURCE_COMMUNITY_COMPARISON_FILENAME,
    DESCRIPTIVE_TARGET_COMMUNITY_COMPARISON_FILENAME,
    FAMILY_COMPARISON_COLUMNS,
    PAIRED_COMPARISON_CONTRACT_VERSION,
    SOURCE_COMMUNITY_COMPARISON_COLUMNS,
    TARGET_COMMUNITY_COMPARISON_COLUMNS,
    build_block1_cohort_relation_comparison_frame,
    build_block1_comparison_frames,
)
from .functions.schemas import (
    ANALYSIS_MANIFEST_REQUIRED_FIELDS,
    BLOCK1_ANALYSIS_MANIFEST_FILENAME,
    BLOCK1_COHORT_RELATION_COMPARISON_FILENAME,
    BLOCK1_CONFIRMATORY_FAMILY_COMPARISON_FILENAME,
    BLOCK1_EXECUTE_MANIFEST_FILENAME,
    BLOCK1_FAMILY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_FAMILY_SUMMARY_FILENAME,
    BLOCK1_RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_COMPARISON_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_SUMMARY_FILENAME,
    BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION,
    BLOCK1_TARGET_COMMUNITY_COMPARISON_FILENAME,
    BLOCK1_TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_TARGET_COMMUNITY_SUMMARY_FILENAME,
    Block1ExecuteManifest,
    Block1FamilyExportRecord,
    COHORT_RELATION_COMPARISON_COLUMNS as SCHEMA_COHORT_RELATION_COMPARISON_COLUMNS,
    CONFIRMATORY_PAIR_FAMILIES,
    EXECUTE_MANIFEST_REQUIRED_FIELDS,
    FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS,
    RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS,
    RUN_SCOPE_FULL_COHORT,
    SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
    STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA,
    STATISTICAL_SUPPLEMENT_Q_ALPHA,
    TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
    validate_block1_family_contract,
)
from .functions.statistics import (
    BH_POLICY,
    SIGN_TEST_METHOD,
    WILCOXON_METHOD,
    build_family_statistical_supplement,
    build_relation_element_statistical_supplement,
    build_source_community_statistical_supplement,
    build_target_community_statistical_supplement,
)
from .functions.summaries import (
    FAMILY_SUMMARY_COLUMNS,
    SOURCE_SUMMARY_COLUMNS,
    SUMMARY_CONTRACT_VERSION,
    TARGET_SUMMARY_COLUMNS,
    build_block1_summary_frames,
)
from .functions.writers import (
    validate_no_forbidden_block1_outputs,
    write_block1_csv,
    write_block1_json,
)
from ..config import TaskAOrderedPairFamilySpec


FIT_STATUS_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "pair_family",
    "fit_status",
    "status_reason",
    "implementation_tier",
    "native_export_manifest_path",
)

BLOCK1_ANALYSIS_OUTPUT_FILENAMES: tuple[str, ...] = (
    BLOCK1_FAMILY_SUMMARY_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_SUMMARY_FILENAME,
    BLOCK1_TARGET_COMMUNITY_SUMMARY_FILENAME,
    BLOCK1_CONFIRMATORY_FAMILY_COMPARISON_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_COMPARISON_FILENAME,
    BLOCK1_TARGET_COMMUNITY_COMPARISON_FILENAME,
    BLOCK1_COHORT_RELATION_COMPARISON_FILENAME,
    BLOCK1_FAMILY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_FILENAME,
    BLOCK1_ANALYSIS_MANIFEST_FILENAME,
)


def load_block1_execute_manifest(path: Path) -> Block1ExecuteManifest:
    """Load and validate the execute manifest without reading external task layers."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = tuple(key for key in EXECUTE_MANIFEST_REQUIRED_FIELDS if key not in payload)
    if missing:
        raise ContractError(f"{manifest_path.name} is missing required keys: {missing}")
    if str(payload["phase"]) != "execute":
        raise ContractError("Block 1 analyze requires an execute-phase manifest")

    family_exports: list[Block1FamilyExportRecord] = []
    for item in payload["family_exports"]:
        if not isinstance(item, Mapping):
            raise ContractError("Block 1 family_exports entries must be mappings")
        family_exports.append(
            Block1FamilyExportRecord(
                pair_family=str(item["pair_family"]),
                source_domain=str(item["source_domain"]),
                target_domain=str(item["target_domain"]),
                claim_role=str(item["claim_role"]),
                native_export_manifest_path=Path(str(item["native_export_manifest_path"])).resolve(),
                native_export_manifest_sha256=str(item["native_export_manifest_sha256"]),
                fit_status=str(item["fit_status"]),
                cohort_fit_status=str(item["cohort_fit_status"]),
                patient_count=int(item["patient_count"]),
                patient_record_count=int(item["patient_record_count"]),
                cohort_record_count=int(item["cohort_record_count"]),
                k_states=int(item["k_states"]),
            )
        )
    return Block1ExecuteManifest(
        task_name=str(payload["task_name"]),
        phase=str(payload["phase"]),
        config_path=Path(str(payload["config_path"])),
        config_fingerprint=str(payload["config_fingerprint"]),
        stage0_h5ad=Path(str(payload["stage0_h5ad"])),
        run_scope=str(payload["run_scope"]),
        patient_ids=tuple(str(patient_id) for patient_id in payload["patient_ids"]),
        confirmatory_pair_families=tuple(str(name) for name in payload["confirmatory_pair_families"]),
        family_exports=tuple(family_exports),
        readiness_status=str(payload["readiness_status"]),
        scientific_interpretation_allowed=bool(payload["scientific_interpretation_allowed"]),
        prohibited_outputs=tuple(str(item) for item in payload["prohibited_outputs"]),
    )


def load_block1_native_exports(
    execute_manifest: Block1ExecuteManifest,
) -> dict[str, NativeRelationExport]:
    """Read generic native exports keyed by pair_family and validate hashes."""
    exports: dict[str, NativeRelationExport] = {}
    for family_export in execute_manifest.family_exports:
        observed_hash = sha256_file(family_export.native_export_manifest_path)
        if observed_hash != family_export.native_export_manifest_sha256:
            raise ContractError(
                f"Block 1 native export manifest SHA-256 mismatch for {family_export.pair_family!r}"
            )
        exports[family_export.pair_family] = read_stride_native_relation_export(
            family_export.native_export_manifest_path
        )
    return exports


def require_block1_execute_status_for_analysis(
    execute_manifest: Block1ExecuteManifest,
    native_exports: Mapping[str, NativeRelationExport],
) -> None:
    """Reject formal analysis from non-ok execute/native status context."""
    if execute_manifest.run_scope != RUN_SCOPE_FULL_COHORT:
        return

    non_ok_reasons: list[str] = []
    for family_export in execute_manifest.family_exports:
        export = native_exports[family_export.pair_family]
        if family_export.fit_status != "ok":
            non_ok_reasons.append(
                f"{family_export.pair_family}:execute_fit_status={family_export.fit_status}"
            )
        if export.manifest.fit_status != "ok":
            non_ok_reasons.append(
                f"{family_export.pair_family}:native_fit_status={export.manifest.fit_status}"
            )
        if export.manifest.cohort_fit_status != "ok":
            non_ok_reasons.append(
                f"{family_export.pair_family}:cohort_fit_status={export.manifest.cohort_fit_status}"
            )
        for patient_record in export.patient_records:
            if patient_record.fit_status != "ok":
                non_ok_reasons.append(
                    f"{family_export.pair_family}:{patient_record.patient_id}="
                    f"{patient_record.fit_status}"
                )
        if not export.cohort_records:
            non_ok_reasons.append(f"{family_export.pair_family}:missing_cohort_record")
        for cohort_record in export.cohort_records:
            if cohort_record.fit_status != "ok":
                non_ok_reasons.append(
                    f"{family_export.pair_family}:{cohort_record.recurrence_family_id}="
                    f"{cohort_record.fit_status}"
                )

    if non_ok_reasons:
        raise ContractError(
            "Block 1 formal analyze cannot continue from non-ok execute/native "
            "status context: "
            + "; ".join(non_ok_reasons)
        )


def build_block1_fit_status_frame(
    execute_manifest: Block1ExecuteManifest,
    native_exports: Mapping[str, NativeRelationExport],
) -> pd.DataFrame:
    """Build internal patient/family fit status context for comparisons."""
    records: list[dict[str, object]] = []
    for family_export in execute_manifest.family_exports:
        export = native_exports[family_export.pair_family]
        for patient_record in export.patient_records:
            records.append(
                {
                    "patient_id": str(patient_record.patient_id),
                    "pair_family": family_export.pair_family,
                    "fit_status": str(patient_record.fit_status),
                    "status_reason": (
                        patient_record.status_reason
                        if patient_record.status_reason is not None
                        else str(patient_record.diagnostics.get("defer_reason", "")).strip() or None
                    ),
                    "implementation_tier": str(patient_record.implementation_tier),
                    "native_export_manifest_path": str(family_export.native_export_manifest_path),
                }
            )
    if not records:
        return pd.DataFrame(columns=list(FIT_STATUS_COLUMNS))
    return (
        pd.DataFrame.from_records(records)
        .loc[:, list(FIT_STATUS_COLUMNS)]
        .sort_values(["patient_id", "pair_family"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_block1_analysis_manifest_payload(
    *,
    execute_manifest_path: Path,
    execute_manifest_sha256: str,
    output_paths: Mapping[str, Path],
    run_scope: str,
    native_exports: Mapping[str, NativeRelationExport],
) -> dict[str, object]:
    """Build the Block 1 analysis manifest."""
    return {
        "task_name": "block1_real_data_discovery",
        "phase": "analyze",
        "source_execute_manifest_path": str(execute_manifest_path),
        "source_execute_manifest_sha256": execute_manifest_sha256,
        "run_scope": run_scope,
        "readiness_status": "diagnostic" if run_scope != "full_cohort" else "evidence_ready",
        "summary_contract_version": SUMMARY_CONTRACT_VERSION,
        "comparison_contract_version": PAIRED_COMPARISON_CONTRACT_VERSION,
        "input_native_exports": [
            {
                "pair_family": pair_family,
                "manifest_path": str(export.manifest.manifest_path),
                "manifest_sha256": export.manifest.manifest_sha256,
                "fit_status": export.manifest.fit_status,
                "cohort_fit_status": export.manifest.cohort_fit_status,
            }
            for pair_family, export in native_exports.items()
        ],
        "family_summary_path": str(output_paths["family_summary_path"]),
        "source_community_summary_path": str(output_paths["source_community_summary_path"]),
        "target_community_summary_path": str(output_paths["target_community_summary_path"]),
        "confirmatory_family_comparison_path": str(output_paths["confirmatory_family_comparison_path"]),
        "source_community_comparison_path": str(output_paths["source_community_comparison_path"]),
        "target_community_comparison_path": str(output_paths["target_community_comparison_path"]),
        "cohort_relation_comparison_path": str(output_paths["cohort_relation_comparison_path"]),
        "family_statistical_supplement_path": str(output_paths["family_statistical_supplement_path"]),
        "source_community_statistical_supplement_path": str(
            output_paths["source_community_statistical_supplement_path"]
        ),
        "target_community_statistical_supplement_path": str(
            output_paths["target_community_statistical_supplement_path"]
        ),
        "relation_element_statistical_supplement_path": str(
            output_paths["relation_element_statistical_supplement_path"]
        ),
        "statistical_supplement_contract_version": BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION,
        "statistical_method": WILCOXON_METHOD,
        "sign_test_method": SIGN_TEST_METHOD,
        "multiple_testing_policy": BH_POLICY,
        "effect_floor_abs_median_delta": STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA,
        "statistical_supplement_q_alpha": STATISTICAL_SUPPLEMENT_Q_ALPHA,
        "scientific_interpretation_allowed": False,
        "emits_p_values": True,
        "emits_figures": False,
        "emits_annotations": False,
    }


def _reject_existing_block1_analysis_outputs(
    output_dir: Path,
    filenames: Sequence[str] = BLOCK1_ANALYSIS_OUTPUT_FILENAMES,
) -> None:
    output_path = Path(output_dir)
    existing = tuple(output_path / filename for filename in filenames if (output_path / filename).exists())
    if existing:
        raise ContractError(
            "Block 1 analyze output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )


def _resolve_confirmatory_pair_families(
    execute_manifest: Block1ExecuteManifest,
) -> tuple[TaskAOrderedPairFamilySpec, ...]:
    if tuple(execute_manifest.confirmatory_pair_families) != CONFIRMATORY_PAIR_FAMILIES:
        raise ContractError("Block 1 execute manifest families do not match the frozen contract")
    if tuple(record.pair_family for record in execute_manifest.family_exports) != CONFIRMATORY_PAIR_FAMILIES:
        raise ContractError("Block 1 execute manifest family_exports do not match frozen order")
    for family_export in execute_manifest.family_exports:
        validate_block1_family_contract(
            family_export.pair_family,
            family_export.source_domain,
            family_export.target_domain,
            family_export.claim_role,
        )
    return tuple(
        TaskAOrderedPairFamilySpec(
            name=family_export.pair_family,
            source_domain=family_export.source_domain,
            target_domain=family_export.target_domain,
            claim_role=family_export.claim_role,
            pair_types=(),
        )
        for family_export in execute_manifest.family_exports
    )


def run_block1_analyze(
    *,
    execute_manifest_path: Path,
    output_dir: Path,
) -> Path:
    """Read execute artifacts, write summary/comparison CSVs, and write manifest."""
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _reject_existing_block1_analysis_outputs(resolved_output_dir)
    execute_manifest = load_block1_execute_manifest(execute_manifest_path)
    pair_families = _resolve_confirmatory_pair_families(execute_manifest)
    native_exports = load_block1_native_exports(execute_manifest)
    require_block1_execute_status_for_analysis(execute_manifest, native_exports)
    fit_status_df = build_block1_fit_status_frame(execute_manifest, native_exports)

    family_summary_df, source_summary_df, target_summary_df = build_block1_summary_frames(
        native_exports=native_exports,
        pair_families=pair_families,
    )
    family_comparison_df, source_comparison_df, target_comparison_df = build_block1_comparison_frames(
        fit_status_df=fit_status_df,
        family_summary_df=family_summary_df,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
        patient_ids=execute_manifest.patient_ids,
    )
    cohort_relation_comparison_df = build_block1_cohort_relation_comparison_frame(
        native_exports=native_exports
    )
    family_statistical_supplement_df = build_family_statistical_supplement(
        family_comparison_df
    )
    source_statistical_supplement_df = build_source_community_statistical_supplement(
        source_comparison_df
    )
    target_statistical_supplement_df = build_target_community_statistical_supplement(
        target_comparison_df
    )
    relation_element_statistical_supplement_df = build_relation_element_statistical_supplement(
        native_exports=native_exports,
        cohort_relation_comparison_df=cohort_relation_comparison_df,
    )

    output_paths = {
        "family_summary_path": write_block1_csv(
            family_summary_df,
            resolved_output_dir / BLOCK1_FAMILY_SUMMARY_FILENAME,
            columns=FAMILY_SUMMARY_COLUMNS,
        ),
        "source_community_summary_path": write_block1_csv(
            source_summary_df,
            resolved_output_dir / BLOCK1_SOURCE_COMMUNITY_SUMMARY_FILENAME,
            columns=SOURCE_SUMMARY_COLUMNS,
        ),
        "target_community_summary_path": write_block1_csv(
            target_summary_df,
            resolved_output_dir / BLOCK1_TARGET_COMMUNITY_SUMMARY_FILENAME,
            columns=TARGET_SUMMARY_COLUMNS,
        ),
        "confirmatory_family_comparison_path": write_block1_csv(
            family_comparison_df,
            resolved_output_dir / BLOCK1_CONFIRMATORY_FAMILY_COMPARISON_FILENAME,
            columns=FAMILY_COMPARISON_COLUMNS,
        ),
        "source_community_comparison_path": write_block1_csv(
            source_comparison_df,
            resolved_output_dir / BLOCK1_SOURCE_COMMUNITY_COMPARISON_FILENAME,
            columns=SOURCE_COMMUNITY_COMPARISON_COLUMNS,
        ),
        "target_community_comparison_path": write_block1_csv(
            target_comparison_df,
            resolved_output_dir / BLOCK1_TARGET_COMMUNITY_COMPARISON_FILENAME,
            columns=TARGET_COMMUNITY_COMPARISON_COLUMNS,
        ),
        "cohort_relation_comparison_path": write_block1_csv(
            cohort_relation_comparison_df,
            resolved_output_dir / BLOCK1_COHORT_RELATION_COMPARISON_FILENAME,
            columns=COHORT_RELATION_COMPARISON_COLUMNS,
        ),
        "family_statistical_supplement_path": write_block1_csv(
            family_statistical_supplement_df,
            resolved_output_dir / BLOCK1_FAMILY_STATISTICAL_SUPPLEMENT_FILENAME,
            columns=FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS,
        ),
        "source_community_statistical_supplement_path": write_block1_csv(
            source_statistical_supplement_df,
            resolved_output_dir / BLOCK1_SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
            columns=SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
        ),
        "target_community_statistical_supplement_path": write_block1_csv(
            target_statistical_supplement_df,
            resolved_output_dir / BLOCK1_TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME,
            columns=TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS,
        ),
        "relation_element_statistical_supplement_path": write_block1_csv(
            relation_element_statistical_supplement_df,
            resolved_output_dir / BLOCK1_RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_FILENAME,
            columns=RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS,
        ),
    }
    analysis_manifest_payload = build_block1_analysis_manifest_payload(
        execute_manifest_path=Path(execute_manifest_path).resolve(),
        execute_manifest_sha256=sha256_file(Path(execute_manifest_path).resolve()),
        output_paths=output_paths,
        run_scope=execute_manifest.run_scope,
        native_exports=native_exports,
    )
    analysis_manifest_path = write_block1_json(
        analysis_manifest_payload,
        resolved_output_dir / BLOCK1_ANALYSIS_MANIFEST_FILENAME,
        required_keys=ANALYSIS_MANIFEST_REQUIRED_FIELDS,
    )
    validate_no_forbidden_block1_outputs(resolved_output_dir)
    return analysis_manifest_path


__all__ = [
    "FIT_STATUS_COLUMNS",
    "build_block1_analysis_manifest_payload",
    "build_block1_fit_status_frame",
    "load_block1_execute_manifest",
    "load_block1_native_exports",
    "require_block1_execute_status_for_analysis",
    "run_block1_analyze",
]
