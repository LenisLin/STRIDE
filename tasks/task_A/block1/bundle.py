"""Write the frozen Task A Block 1 bundle surface.

This module consumes a passed Block 0 bundle plus the Stage 0 h5ad and emits
the Block 1 mapping, dry-run, summary tables, bundle, and workflow manifest.
The implementation stays task-local and keeps the legacy block identifier for
compatibility, while the scientific role is the new real-data biological
discovery layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stride.errors import ContractError
from stride.outputs.fit_result import STRIDEFitResult

from ..block0.bundle import PASSED_STATUS, require_block0_passed_contract
from ..config import load_task_a_config_bundle
from ..contracts import EVIDENCE_READY_STATE, TaskAStage0StrideMappingSummary
from .comparisons import (
    CONFIRMATORY_FAMILY_COMPARISON_FILENAME,
    EXPLORATORY_SOURCE_COMMUNITY_COMPARISON_FILENAME,
    EXPLORATORY_TARGET_COMMUNITY_COMPARISON_FILENAME,
    PAIRED_COMPARISON_CONTRACT_VERSION,
    build_block1_comparison_frames,
)
from .correspondence import write_block1_community_correspondence_packet
from .summaries import (
    FAMILY_SUMMARY_FILENAME,
    FAMILY_SUMMARY_SCALES,
    PROOF_CARRYING_SUMMARY_NAMES,
    SOURCE_COMMUNITY_SUMMARY_FILENAME,
    SOURCE_ELIGIBILITY_RULE,
    SUMMARY_CONTRACT_VERSION,
    SUPPORTIVE_SUMMARY_NAMES,
    TARGET_COMMUNITY_SUMMARY_FILENAME,
    TARGET_ELIGIBILITY_RULE,
    build_block1_summary_frames,
)
from ..workflows.stride_adapter import (
    describe_task_a_stage0_stride_mapping,
    run_task_a_family_core_fit_dry_run,
)


BLOCK_NAME = "block1_continuity_backbone"
SCIENTIFIC_ROLE = "real_data_biological_discovery"
MAPPING_FILENAME = "block1_stage0_mapping.json"
CORE_FIT_DRY_RUN_FILENAME = "block1_core_fit_dry_run.csv"
RECURRENCE_SUMMARY_FILENAME = "block1_recurrence_summary.json"
RECURRENCE_FAMILIES_FILENAME = "block1_recurrence_families.json"
RECURRENCE_EMBEDDINGS_FILENAME = "block1_recurrence_embeddings.csv"
BUNDLE_FILENAME = "block1_bundle.json"
WORKFLOW_MANIFEST_FILENAME = "block1_workflow_manifest.json"


@dataclass(frozen=True)
class TaskABlock1Bundle:
    block: str
    scientific_role: str
    status: str
    artifact_state: str
    implementation_tier: str
    evidence_lineage: str
    fit_surface: str
    block0_bundle_path: Path
    block0_gate_status: str
    config_fingerprint: str
    config_path: Path
    stage0_h5ad: Path
    output_dir: Path
    mapping_manifest_path: Path
    core_fit_dry_run_path: Path
    recurrence_summary_path: Path
    recurrence_families_path: Path
    recurrence_embeddings_path: Path
    family_summary_path: Path
    source_community_summary_path: Path
    target_community_summary_path: Path
    confirmatory_family_comparison_path: Path
    exploratory_source_community_comparison_path: Path
    exploratory_target_community_comparison_path: Path
    community_correspondence_manifest_path: Path
    community_correspondence_index_path: Path
    bundle_path: Path
    pair_families: tuple[str, ...]
    confirmatory_pair_families: tuple[str, ...]
    summary_contract_version: str
    paired_comparison_contract_version: str
    proof_carrying_family_summaries: tuple[str, ...]
    supportive_family_summaries: tuple[str, ...]
    family_summary_scales: tuple[str, ...]
    source_eligibility_rule: str
    target_eligibility_rule: str
    fit_result_counts: dict[str, int]
    cohort_recurrence_fit_status: str
    cohort_recurrence_fit_status_by_pair_family: dict[str, str]
    cohort_recurrence_family_count: int
    cohort_recurrence_family_count_by_pair_family: dict[str, int]
    n_recurrence_used_patients: int
    n_recurrence_used_patients_by_pair_family: dict[str, int]
    stage0_mapping: TaskAStage0StrideMappingSummary

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "scientific_role": self.scientific_role,
            "status": self.status,
            "artifact_state": self.artifact_state,
            "implementation_tier": self.implementation_tier,
            "evidence_lineage": self.evidence_lineage,
            "fit_surface": self.fit_surface,
            "block0_bundle_path": str(self.block0_bundle_path),
            "block0_gate_status": self.block0_gate_status,
            "config_fingerprint": self.config_fingerprint,
            "config_path": str(self.config_path),
            "stage0_h5ad": str(self.stage0_h5ad),
            "output_dir": str(self.output_dir),
            "mapping_manifest_path": str(self.mapping_manifest_path),
            "core_fit_dry_run_path": str(self.core_fit_dry_run_path),
            "recurrence_summary_path": str(self.recurrence_summary_path),
            "recurrence_families_path": str(self.recurrence_families_path),
            "recurrence_embeddings_path": str(self.recurrence_embeddings_path),
            "family_summary_path": str(self.family_summary_path),
            "source_community_summary_path": str(self.source_community_summary_path),
            "target_community_summary_path": str(self.target_community_summary_path),
            "confirmatory_family_comparison_path": str(self.confirmatory_family_comparison_path),
            "exploratory_source_community_comparison_path": str(
                self.exploratory_source_community_comparison_path
            ),
            "exploratory_target_community_comparison_path": str(
                self.exploratory_target_community_comparison_path
            ),
            "community_correspondence_manifest_path": str(
                self.community_correspondence_manifest_path
            ),
            "community_correspondence_index_path": str(self.community_correspondence_index_path),
            "bundle_path": str(self.bundle_path),
            "pair_families": list(self.pair_families),
            "confirmatory_pair_families": list(self.confirmatory_pair_families),
            "summary_contract_version": self.summary_contract_version,
            "paired_comparison_contract_version": self.paired_comparison_contract_version,
            "proof_carrying_family_summaries": list(self.proof_carrying_family_summaries),
            "supportive_family_summaries": list(self.supportive_family_summaries),
            "family_summary_scales": list(self.family_summary_scales),
            "source_eligibility_rule": self.source_eligibility_rule,
            "target_eligibility_rule": self.target_eligibility_rule,
            "fit_result_counts": dict(sorted(self.fit_result_counts.items())),
            "cohort_recurrence_fit_status": self.cohort_recurrence_fit_status,
            "cohort_recurrence_fit_status_by_pair_family": dict(
                sorted(self.cohort_recurrence_fit_status_by_pair_family.items())
            ),
            "cohort_recurrence_family_count": self.cohort_recurrence_family_count,
            "cohort_recurrence_family_count_by_pair_family": dict(
                sorted(self.cohort_recurrence_family_count_by_pair_family.items())
            ),
            "n_recurrence_used_patients": self.n_recurrence_used_patients,
            "n_recurrence_used_patients_by_pair_family": dict(
                sorted(self.n_recurrence_used_patients_by_pair_family.items())
            ),
            "stage0_mapping": self.stage0_mapping.to_json_dict(),
        }

    def to_workflow_manifest_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "scientific_role": self.scientific_role,
            "status": self.status,
            "artifact_state": self.artifact_state,
            "implementation_tier": self.implementation_tier,
            "evidence_lineage": self.evidence_lineage,
            "fit_surface": self.fit_surface,
            "block0_bundle_path": str(self.block0_bundle_path),
            "block0_gate_status": self.block0_gate_status,
            "config_fingerprint": self.config_fingerprint,
            "bundle_path": str(self.bundle_path),
            "core_fit_dry_run_path": str(self.core_fit_dry_run_path),
            "mapping_manifest_path": str(self.mapping_manifest_path),
            "recurrence_summary_path": str(self.recurrence_summary_path),
            "recurrence_families_path": str(self.recurrence_families_path),
            "recurrence_embeddings_path": str(self.recurrence_embeddings_path),
            "family_summary_path": str(self.family_summary_path),
            "source_community_summary_path": str(self.source_community_summary_path),
            "target_community_summary_path": str(self.target_community_summary_path),
            "confirmatory_family_comparison_path": str(self.confirmatory_family_comparison_path),
            "exploratory_source_community_comparison_path": str(
                self.exploratory_source_community_comparison_path
            ),
            "exploratory_target_community_comparison_path": str(
                self.exploratory_target_community_comparison_path
            ),
            "community_correspondence_manifest_path": str(
                self.community_correspondence_manifest_path
            ),
            "community_correspondence_index_path": str(self.community_correspondence_index_path),
            "summary_contract_version": self.summary_contract_version,
            "paired_comparison_contract_version": self.paired_comparison_contract_version,
            "proof_carrying_family_summaries": list(self.proof_carrying_family_summaries),
            "supportive_family_summaries": list(self.supportive_family_summaries),
            "family_summary_scales": list(self.family_summary_scales),
            "cohort_recurrence_fit_status": self.cohort_recurrence_fit_status,
            "cohort_recurrence_fit_status_by_pair_family": dict(
                sorted(self.cohort_recurrence_fit_status_by_pair_family.items())
            ),
            "cohort_recurrence_family_count": self.cohort_recurrence_family_count,
            "cohort_recurrence_family_count_by_pair_family": dict(
                sorted(self.cohort_recurrence_family_count_by_pair_family.items())
            ),
            "n_recurrence_used_patients": self.n_recurrence_used_patients,
            "n_recurrence_used_patients_by_pair_family": dict(
                sorted(self.n_recurrence_used_patients_by_pair_family.items())
            ),
        }


def _serialize_recurrence_array(value: object | None) -> list[float] | list[list[float]] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array.tolist()


def _used_recurrence_patient_ids(result: STRIDEFitResult) -> tuple[str, ...]:
    recurrence = result.recurrence
    used_ids = recurrence.used_patient_ids if recurrence.used_patient_ids else recurrence.patient_ids
    return tuple(str(patient_id) for patient_id in used_ids)


def _write_recurrence_artifacts(
    *,
    fit_results: dict[str, STRIDEFitResult],
    output_root: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    recurrence_status_by_pair_family: dict[str, str] = {}
    recurrence_family_count_by_pair_family: dict[str, int] = {}
    recurrence_used_patient_count_by_pair_family: dict[str, int] = {}
    unique_used_patient_ids: set[str] = set()

    for pair_family, fit_result in sorted(fit_results.items()):
        recurrence = fit_result.recurrence
        used_patient_ids = _used_recurrence_patient_ids(fit_result)
        unique_used_patient_ids.update(used_patient_ids)
        recurrence_status_by_pair_family[pair_family] = str(recurrence.fit_status)
        recurrence_family_count_by_pair_family[pair_family] = int(len(recurrence.families))
        recurrence_used_patient_count_by_pair_family[pair_family] = int(len(used_patient_ids))

        summary_rows.append(
            {
                "pair_family": pair_family,
                "implementation_tier": str(fit_result.implementation_tier),
                "fit_surface": "fit_stride",
                "cohort_recurrence_fit_status": str(recurrence.fit_status),
                "n_recurrence_families": int(len(recurrence.families)),
                "n_recurrence_used_patients": int(len(used_patient_ids)),
                "recurrence_unit": str(recurrence.recurrence_unit),
                "basis_dim": (
                    int(recurrence.parameters.basis_dim)
                    if recurrence.parameters is not None
                    else None
                ),
                "patient_ids": list(str(patient_id) for patient_id in recurrence.patient_ids),
                "used_patient_ids": list(used_patient_ids),
                "family_ids": [str(family.family_id) for family in recurrence.families],
                "metadata": dict(recurrence.metadata),
            }
        )
        for family in recurrence.families:
            family_rows.append(
                {
                    "pair_family": pair_family,
                    "family_id": str(family.family_id),
                    "fit_status": str(family.fit_status),
                    "support_n_patients": int(family.support_n_patients),
                    "within_family_dispersion": (
                        None
                        if family.within_family_dispersion is None
                        else float(family.within_family_dispersion)
                    ),
                    "member_patient_ids": [str(patient_id) for patient_id in family.member_patient_ids],
                    "template_A": _serialize_recurrence_array(family.template_A),
                    "template_d": _serialize_recurrence_array(family.template_d),
                    "template_e": _serialize_recurrence_array(family.template_e),
                }
            )
        basis_dim = (
            int(recurrence.parameters.basis_dim)
            if recurrence.parameters is not None
            else 0
        )
        for embedding in recurrence.embeddings:
            row: dict[str, Any] = {
                "pair_family": pair_family,
                "patient_id": str(embedding.patient_id),
                "fit_status": str(embedding.fit_status),
                "used_for_recurrence": str(embedding.patient_id) in set(used_patient_ids),
            }
            coordinates = np.asarray(embedding.coordinates, dtype=float).reshape(-1)
            n_coordinates = max(basis_dim, int(coordinates.shape[0]))
            for idx in range(n_coordinates):
                value = float(coordinates[idx]) if idx < coordinates.shape[0] else np.nan
                row[f"coord_{idx + 1}"] = value
            embedding_rows.append(row)

    recurrence_summary_path = output_root / RECURRENCE_SUMMARY_FILENAME
    recurrence_families_path = output_root / RECURRENCE_FAMILIES_FILENAME
    recurrence_embeddings_path = output_root / RECURRENCE_EMBEDDINGS_FILENAME

    overall_recurrence_status = (
        "ok"
        if recurrence_status_by_pair_family
        and all(status == "ok" for status in recurrence_status_by_pair_family.values())
        else "deferred"
    )
    recurrence_summary_payload = {
        "fit_surface": "fit_stride",
        "implementation_tier": "canonical_full",
        "evidence_lineage": "canonical_rerun",
        "cohort_recurrence_fit_status": overall_recurrence_status,
        "cohort_recurrence_fit_status_by_pair_family": dict(
            sorted(recurrence_status_by_pair_family.items())
        ),
        "cohort_recurrence_family_count": int(sum(recurrence_family_count_by_pair_family.values())),
        "cohort_recurrence_family_count_by_pair_family": dict(
            sorted(recurrence_family_count_by_pair_family.items())
        ),
        "n_recurrence_used_patients": int(len(unique_used_patient_ids)),
        "n_recurrence_used_patients_by_pair_family": dict(
            sorted(recurrence_used_patient_count_by_pair_family.items())
        ),
        "pair_families": summary_rows,
    }
    recurrence_summary_path.write_text(
        json.dumps(recurrence_summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    recurrence_families_path.write_text(
        json.dumps(family_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(embedding_rows).to_csv(recurrence_embeddings_path, index=False)
    return (
        recurrence_summary_path,
        recurrence_families_path,
        recurrence_embeddings_path,
        recurrence_summary_payload,
    )


def _build_block1_fit_metadata(config_bundle: Any) -> dict[str, Any]:
    return {
        "task_block": BLOCK_NAME,
        "task_block1_target_alpha": float(config_bundle.block1.target_alpha),
        "task_block1_lambda_grid": list(config_bundle.block1.lambda_grid),
    }


def write_task_a_block1_bundle(
    *,
    config_path: str | Path,
    data_path: str | Path,
    block0_bundle_path: str | Path,
    output_dir: str | Path,
) -> TaskABlock1Bundle:
    config_bundle = load_task_a_config_bundle(config_path)
    if BLOCK_NAME not in config_bundle.enabled_blocks:
        raise ContractError(
            f"Task A config does not enable {BLOCK_NAME}; enabled_blocks={list(config_bundle.enabled_blocks)}"
        )
    block0_contract = require_block0_passed_contract(
        block0_bundle_path,
        config_path=config_bundle.config_path,
        data_path=data_path,
    )
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stage0_mapping = describe_task_a_stage0_stride_mapping(
        data_path,
        config_bundle=config_bundle,
    )
    dry_run_df, dry_run_results = run_task_a_family_core_fit_dry_run(
        data_path,
        config_bundle=config_bundle,
        pair_families=config_bundle.ordered_proxy.confirmatory_pair_families,
        fit_metadata=_build_block1_fit_metadata(config_bundle),
    )
    family_summary_df, source_summary_df, target_summary_df = build_block1_summary_frames(
        fit_results=dry_run_results,
        pair_families=config_bundle.ordered_proxy.confirmatory_pair_families,
    )
    (
        family_comparison_df,
        source_comparison_df,
        target_comparison_df,
    ) = build_block1_comparison_frames(
        dry_run_df=dry_run_df,
        family_summary_df=family_summary_df,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
        patient_ids=stage0_mapping.patient_ids,
    )

    mapping_path = output_root / MAPPING_FILENAME
    mapping_path.write_text(
        json.dumps(stage0_mapping.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    dry_run_path = output_root / CORE_FIT_DRY_RUN_FILENAME
    dry_run_df.to_csv(dry_run_path, index=False)
    (
        recurrence_summary_path,
        recurrence_families_path,
        recurrence_embeddings_path,
        recurrence_summary_payload,
    ) = _write_recurrence_artifacts(
        fit_results=dry_run_results,
        output_root=output_root,
    )
    family_summary_path = output_root / FAMILY_SUMMARY_FILENAME
    family_summary_df.to_csv(family_summary_path, index=False)
    source_summary_path = output_root / SOURCE_COMMUNITY_SUMMARY_FILENAME
    source_summary_df.to_csv(source_summary_path, index=False)
    target_summary_path = output_root / TARGET_COMMUNITY_SUMMARY_FILENAME
    target_summary_df.to_csv(target_summary_path, index=False)
    family_comparison_path = output_root / CONFIRMATORY_FAMILY_COMPARISON_FILENAME
    family_comparison_df.to_csv(family_comparison_path, index=False)
    source_comparison_path = output_root / EXPLORATORY_SOURCE_COMMUNITY_COMPARISON_FILENAME
    source_comparison_df.to_csv(source_comparison_path, index=False)
    target_comparison_path = output_root / EXPLORATORY_TARGET_COMMUNITY_COMPARISON_FILENAME
    target_comparison_df.to_csv(target_comparison_path, index=False)
    (
        community_correspondence_manifest_path,
        community_correspondence_index_path,
    ) = write_block1_community_correspondence_packet(
        config_path=config_bundle.config_path,
        data_path=data_path,
        output_dir=output_root,
        stage0_mapping=stage0_mapping,
        source_summary_path=source_summary_path,
        target_summary_path=target_summary_path,
        confirmatory_family_comparison_path=family_comparison_path,
        exploratory_source_comparison_path=source_comparison_path,
        exploratory_target_comparison_path=target_comparison_path,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
    )

    bundle = TaskABlock1Bundle(
        block=BLOCK_NAME,
        scientific_role=SCIENTIFIC_ROLE,
        status="active",
        artifact_state=EVIDENCE_READY_STATE,
        implementation_tier="canonical_full",
        evidence_lineage="canonical_rerun",
        fit_surface="fit_stride",
        block0_bundle_path=block0_contract.bundle_path,
        block0_gate_status=PASSED_STATUS,
        config_fingerprint=config_bundle.config_fingerprint,
        config_path=config_bundle.config_path,
        stage0_h5ad=Path(data_path).expanduser().resolve(),
        output_dir=output_root,
        mapping_manifest_path=mapping_path,
        core_fit_dry_run_path=dry_run_path,
        recurrence_summary_path=recurrence_summary_path,
        recurrence_families_path=recurrence_families_path,
        recurrence_embeddings_path=recurrence_embeddings_path,
        family_summary_path=family_summary_path,
        source_community_summary_path=source_summary_path,
        target_community_summary_path=target_summary_path,
        confirmatory_family_comparison_path=family_comparison_path,
        exploratory_source_community_comparison_path=source_comparison_path,
        exploratory_target_community_comparison_path=target_comparison_path,
        community_correspondence_manifest_path=community_correspondence_manifest_path,
        community_correspondence_index_path=community_correspondence_index_path,
        bundle_path=output_root / BUNDLE_FILENAME,
        pair_families=tuple(
            family.name for family in config_bundle.ordered_proxy.confirmatory_pair_families
        ),
        confirmatory_pair_families=tuple(
            family.name for family in config_bundle.ordered_proxy.confirmatory_pair_families
        ),
        summary_contract_version=SUMMARY_CONTRACT_VERSION,
        paired_comparison_contract_version=PAIRED_COMPARISON_CONTRACT_VERSION,
        proof_carrying_family_summaries=PROOF_CARRYING_SUMMARY_NAMES,
        supportive_family_summaries=SUPPORTIVE_SUMMARY_NAMES,
        family_summary_scales=FAMILY_SUMMARY_SCALES,
        source_eligibility_rule=SOURCE_ELIGIBILITY_RULE,
        target_eligibility_rule=TARGET_ELIGIBILITY_RULE,
        fit_result_counts={
            family_name: len(result.patient_results)
            for family_name, result in dry_run_results.items()
        },
        cohort_recurrence_fit_status=str(recurrence_summary_payload["cohort_recurrence_fit_status"]),
        cohort_recurrence_fit_status_by_pair_family={
            str(key): str(value)
            for key, value in dict(
                recurrence_summary_payload["cohort_recurrence_fit_status_by_pair_family"]
            ).items()
        },
        cohort_recurrence_family_count=int(
            recurrence_summary_payload["cohort_recurrence_family_count"]
        ),
        cohort_recurrence_family_count_by_pair_family={
            str(key): int(value)
            for key, value in dict(
                recurrence_summary_payload["cohort_recurrence_family_count_by_pair_family"]
            ).items()
        },
        n_recurrence_used_patients=int(
            recurrence_summary_payload["n_recurrence_used_patients"]
        ),
        n_recurrence_used_patients_by_pair_family={
            str(key): int(value)
            for key, value in dict(
                recurrence_summary_payload["n_recurrence_used_patients_by_pair_family"]
            ).items()
        },
        stage0_mapping=stage0_mapping,
    )
    bundle.bundle_path.write_text(
        json.dumps(bundle.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / WORKFLOW_MANIFEST_FILENAME).write_text(
        json.dumps(
            bundle.to_workflow_manifest_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bundle


__all__ = [
    "BLOCK_NAME",
    "BUNDLE_FILENAME",
    "CORE_FIT_DRY_RUN_FILENAME",
    "MAPPING_FILENAME",
    "RECURRENCE_EMBEDDINGS_FILENAME",
    "RECURRENCE_FAMILIES_FILENAME",
    "RECURRENCE_SUMMARY_FILENAME",
    "SCIENTIFIC_ROLE",
    "TaskABlock1Bundle",
    "WORKFLOW_MANIFEST_FILENAME",
    "write_task_a_block1_bundle",
]
