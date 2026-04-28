"""Write the frozen Task A Block 2 robustness surface.

This module consumes an evidence-ready Block 1 bundle, re-estimates the frozen
Block 1 comparison surfaces under task-local perturbations, and emits
proof-carrying Block 2 robustness summaries plus a compatibility-named top
summary/manifest surface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from stride.errors import ContractError

from ..block0.bundle import PASSED_STATUS, require_block0_passed_contract
from ..block1.comparisons import PAIRED_COMPARISON_CONTRACT_VERSION
from ..block1.summaries import (
    SOURCE_ELIGIBILITY_RULE,
    SUMMARY_CONTRACT_VERSION,
    TARGET_ELIGIBILITY_RULE,
)
from ..config import (
    compute_task_a_config_fingerprint,
    load_raw_task_a_config,
    load_task_a_config_bundle,
)
from ..contracts import EVIDENCE_READY_STATE, coerce_task_a_artifact_state
from ..workflows.stride_adapter import load_task_a_dataset_handle
from .robustness import (
    REPLICATE_STATUS_FAILED,
    REPLICATE_STATUS_EXECUTED,
    REPLICATE_STATUS_PENDING,
    build_anchor_findings,
    build_block2_summary,
    build_replicate_assessment_rows,
    build_replicate_manifest,
    run_block1_reestimate_for_block2,
    summarize_assessments_by_route,
)


BLOCK_NAME = "block2_bounded_audit"
SUMMARY_FILENAME = "block2_bounded_audit_summary.csv"
CONTRACT_FILENAME = "block2_contract_audit.csv"
MANIFEST_FILENAME = "block2_bounded_audit_manifest.json"
REPLICATE_MANIFEST_FILENAME = "block2_replicate_manifest.csv"
FAMILY_ROBUSTNESS_FILENAME = "block2_family_robustness.csv"
SOURCE_ROBUSTNESS_FILENAME = "block2_source_community_robustness.csv"
TARGET_ROBUSTNESS_FILENAME = "block2_target_community_robustness.csv"
SCIENTIFIC_ROLE = "robustness_over_block1_summaries"
CLAIM_SCOPE = "block1_summary_robustness"
RECURRENCE_STATUS_VALUES: frozenset[str] = frozenset({"ok", "deferred", "failed"})
RESUME_META_FILENAME = ".block2_resume_meta.json"
RESUME_MANIFEST_FILENAME = ".block2_resume_replicate_manifest.csv"
RESUME_ASSESSMENTS_FILENAME = ".block2_resume_assessments.json"
REPLICATE_RESUME_KEY_COLUMNS: tuple[str, ...] = (
    "route_name",
    "route_group",
    "replicate_index",
    "replicate_label",
    "selection_seed",
    "patient_subset_json",
    "dropped_roi_ids_json",
    "route_note",
)
REPLICATE_RESUME_MUTABLE_COLUMNS: tuple[str, ...] = (
    "route_status",
    "failure_reason",
    "n_patients_retained",
    "n_rois_retained",
    "n_cells_retained",
)


@dataclass(frozen=True)
class TaskABlock1BundleContract:
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
    recurrence_summary_path: Path | None
    recurrence_families_path: Path | None
    recurrence_embeddings_path: Path | None
    family_summary_path: Path
    source_community_summary_path: Path
    target_community_summary_path: Path
    confirmatory_family_comparison_path: Path
    exploratory_source_community_comparison_path: Path
    exploratory_target_community_comparison_path: Path
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

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "TaskABlock1BundleContract":
        block = str(payload["block"])
        status = str(payload.get("status", "unknown"))
        return cls(
            block=block,
            scientific_role=str(payload.get("scientific_role", "")),
            status=status,
            artifact_state=coerce_task_a_artifact_state(
                artifact_state=None if payload.get("artifact_state") in (None, "") else str(payload["artifact_state"]),
                legacy_status=status,
                block=block,
            ),
            implementation_tier=str(payload.get("implementation_tier", "")),
            evidence_lineage=str(payload.get("evidence_lineage", "")),
            fit_surface=str(payload.get("fit_surface", "")),
            block0_bundle_path=Path(str(payload["block0_bundle_path"])).expanduser().resolve(),
            block0_gate_status=str(payload.get("block0_gate_status", "unknown")),
            config_fingerprint=str(payload.get("config_fingerprint", "")),
            config_path=Path(str(payload["config_path"])).expanduser().resolve(),
            stage0_h5ad=Path(str(payload["stage0_h5ad"])).expanduser().resolve(),
            output_dir=Path(str(payload["output_dir"])).expanduser().resolve(),
            mapping_manifest_path=Path(str(payload["mapping_manifest_path"])).expanduser().resolve(),
            core_fit_dry_run_path=Path(str(payload["core_fit_dry_run_path"])).expanduser().resolve(),
            recurrence_summary_path=(
                Path(str(payload["recurrence_summary_path"])).expanduser().resolve()
                if payload.get("recurrence_summary_path") not in (None, "")
                else None
            ),
            recurrence_families_path=(
                Path(str(payload["recurrence_families_path"])).expanduser().resolve()
                if payload.get("recurrence_families_path") not in (None, "")
                else None
            ),
            recurrence_embeddings_path=(
                Path(str(payload["recurrence_embeddings_path"])).expanduser().resolve()
                if payload.get("recurrence_embeddings_path") not in (None, "")
                else None
            ),
            family_summary_path=Path(str(payload["family_summary_path"])).expanduser().resolve(),
            source_community_summary_path=Path(str(payload["source_community_summary_path"])).expanduser().resolve(),
            target_community_summary_path=Path(str(payload["target_community_summary_path"])).expanduser().resolve(),
            confirmatory_family_comparison_path=Path(str(payload["confirmatory_family_comparison_path"])).expanduser().resolve(),
            exploratory_source_community_comparison_path=Path(
                str(payload["exploratory_source_community_comparison_path"])
            ).expanduser().resolve(),
            exploratory_target_community_comparison_path=Path(
                str(payload["exploratory_target_community_comparison_path"])
            ).expanduser().resolve(),
            bundle_path=Path(str(payload["bundle_path"])).expanduser().resolve(),
            pair_families=tuple(str(name) for name in payload.get("pair_families", [])),
            confirmatory_pair_families=tuple(str(name) for name in payload.get("confirmatory_pair_families", [])),
            summary_contract_version=str(payload.get("summary_contract_version", "")),
            paired_comparison_contract_version=str(
                payload.get("paired_comparison_contract_version", "")
            ),
            proof_carrying_family_summaries=tuple(
                str(name) for name in payload.get("proof_carrying_family_summaries", [])
            ),
            supportive_family_summaries=tuple(
                str(name) for name in payload.get("supportive_family_summaries", [])
            ),
            family_summary_scales=tuple(str(name) for name in payload.get("family_summary_scales", [])),
            source_eligibility_rule=str(payload.get("source_eligibility_rule", "")),
            target_eligibility_rule=str(payload.get("target_eligibility_rule", "")),
            fit_result_counts={str(key): int(value) for key, value in dict(payload.get("fit_result_counts", {})).items()},
            cohort_recurrence_fit_status=str(payload.get("cohort_recurrence_fit_status", "")),
            cohort_recurrence_fit_status_by_pair_family={
                str(key): str(value)
                for key, value in dict(
                    payload.get("cohort_recurrence_fit_status_by_pair_family", {})
                ).items()
            },
            cohort_recurrence_family_count=int(payload.get("cohort_recurrence_family_count", 0)),
            cohort_recurrence_family_count_by_pair_family={
                str(key): int(value)
                for key, value in dict(
                    payload.get("cohort_recurrence_family_count_by_pair_family", {})
                ).items()
            },
            n_recurrence_used_patients=int(payload.get("n_recurrence_used_patients", 0)),
            n_recurrence_used_patients_by_pair_family={
                str(key): int(value)
                for key, value in dict(
                    payload.get("n_recurrence_used_patients_by_pair_family", {})
                ).items()
            },
        )


def load_block1_bundle_contract(bundle_path: str | Path) -> TaskABlock1BundleContract:
    path = Path(bundle_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = TaskABlock1BundleContract.from_json_dict(payload)
    if contract.bundle_path == path:
        return contract
    return TaskABlock1BundleContract(
        block=contract.block,
        scientific_role=contract.scientific_role,
        status=contract.status,
        artifact_state=contract.artifact_state,
        implementation_tier=contract.implementation_tier,
        evidence_lineage=contract.evidence_lineage,
        fit_surface=contract.fit_surface,
        block0_bundle_path=contract.block0_bundle_path,
        block0_gate_status=contract.block0_gate_status,
        config_fingerprint=contract.config_fingerprint,
        config_path=contract.config_path,
        stage0_h5ad=contract.stage0_h5ad,
        output_dir=contract.output_dir,
        mapping_manifest_path=contract.mapping_manifest_path,
        core_fit_dry_run_path=contract.core_fit_dry_run_path,
        recurrence_summary_path=contract.recurrence_summary_path,
        recurrence_families_path=contract.recurrence_families_path,
        recurrence_embeddings_path=contract.recurrence_embeddings_path,
        family_summary_path=contract.family_summary_path,
        source_community_summary_path=contract.source_community_summary_path,
        target_community_summary_path=contract.target_community_summary_path,
        confirmatory_family_comparison_path=contract.confirmatory_family_comparison_path,
        exploratory_source_community_comparison_path=contract.exploratory_source_community_comparison_path,
        exploratory_target_community_comparison_path=contract.exploratory_target_community_comparison_path,
        bundle_path=path,
        pair_families=contract.pair_families,
        confirmatory_pair_families=contract.confirmatory_pair_families,
        summary_contract_version=contract.summary_contract_version,
        paired_comparison_contract_version=contract.paired_comparison_contract_version,
        proof_carrying_family_summaries=contract.proof_carrying_family_summaries,
        supportive_family_summaries=contract.supportive_family_summaries,
        family_summary_scales=contract.family_summary_scales,
        source_eligibility_rule=contract.source_eligibility_rule,
        target_eligibility_rule=contract.target_eligibility_rule,
        fit_result_counts=contract.fit_result_counts,
        cohort_recurrence_fit_status=contract.cohort_recurrence_fit_status,
        cohort_recurrence_fit_status_by_pair_family=contract.cohort_recurrence_fit_status_by_pair_family,
        cohort_recurrence_family_count=contract.cohort_recurrence_family_count,
        cohort_recurrence_family_count_by_pair_family=contract.cohort_recurrence_family_count_by_pair_family,
        n_recurrence_used_patients=contract.n_recurrence_used_patients,
        n_recurrence_used_patients_by_pair_family=contract.n_recurrence_used_patients_by_pair_family,
    )


def validate_block1_bundle_contract(contract: TaskABlock1BundleContract) -> None:
    if contract.block != "block1_continuity_backbone":
        raise ContractError(
            f"Block 2 requires a block1_continuity_backbone bundle, got {contract.block!r}"
        )
    if contract.scientific_role != "real_data_biological_discovery":
        raise ContractError(
            "Block 2 requires the frozen Block 1 scientific role "
            "'real_data_biological_discovery'"
        )
    if contract.implementation_tier != "canonical_full":
        raise ContractError(
            "Block 2 requires canonical-full Block 1 outputs, "
            f"got implementation_tier={contract.implementation_tier!r}"
        )
    if contract.evidence_lineage != "canonical_rerun":
        raise ContractError(
            "Block 2 requires canonical-rerun Block 1 outputs, "
            f"got evidence_lineage={contract.evidence_lineage!r}"
        )
    if contract.fit_surface != "fit_stride":
        raise ContractError(
            "Block 2 requires Block 1 to be produced through fit_stride, "
            f"got fit_surface={contract.fit_surface!r}"
        )
    if contract.artifact_state != EVIDENCE_READY_STATE:
        raise ContractError(
            "Block 2 requires an evidence-ready Block 1 bundle, "
            f"got artifact_state {contract.artifact_state!r}"
        )
    require_block0_passed_contract(
        contract.block0_bundle_path,
        config_path=contract.config_path,
        data_path=contract.stage0_h5ad,
    )
    if contract.block0_gate_status != PASSED_STATUS:
        raise ContractError("Block 2 requires Block 0 pass provenance in the Block 1 bundle")
    if tuple(contract.confirmatory_pair_families) != ("TC-IM", "TC-PT"):
        raise ContractError(
            "Block 2 requires Block 1 confirmatory_pair_families to remain frozen as "
            "('TC-IM', 'TC-PT')"
        )
    if contract.summary_contract_version != SUMMARY_CONTRACT_VERSION:
        raise ContractError(
            "Block 2 requires the frozen Block 1 summary contract version to remain "
            f"{SUMMARY_CONTRACT_VERSION!r}"
        )
    if contract.paired_comparison_contract_version != PAIRED_COMPARISON_CONTRACT_VERSION:
        raise ContractError(
            "Block 2 requires the frozen Block 1 paired comparison contract version "
            f"to remain {PAIRED_COMPARISON_CONTRACT_VERSION!r}"
        )
    for label, path in (
        ("recurrence_summary_path", contract.recurrence_summary_path),
        ("recurrence_families_path", contract.recurrence_families_path),
        ("recurrence_embeddings_path", contract.recurrence_embeddings_path),
    ):
        if path is None or not path.exists():
            raise ContractError(
                "Block 2 requires canonical Block 1 recurrence exports to exist; "
                f"missing {label}"
            )
    if contract.cohort_recurrence_fit_status not in RECURRENCE_STATUS_VALUES:
        raise ContractError(
            "Block 2 requires Block 1 to report a valid cohort_recurrence_fit_status, "
            f"got {contract.cohort_recurrence_fit_status!r}"
        )
    expected_pair_families = tuple(contract.confirmatory_pair_families)
    if not contract.cohort_recurrence_fit_status_by_pair_family:
        raise ContractError(
            "Block 2 requires Block 1 to report cohort_recurrence_fit_status_by_pair_family"
        )
    if tuple(sorted(contract.cohort_recurrence_fit_status_by_pair_family)) != tuple(
        sorted(expected_pair_families)
    ):
        raise ContractError(
            "Block 2 requires Block 1 recurrence fit-status coverage for every confirmatory pair family"
        )
    if not contract.cohort_recurrence_family_count_by_pair_family:
        raise ContractError(
            "Block 2 requires Block 1 to report cohort_recurrence_family_count_by_pair_family"
        )
    if tuple(sorted(contract.cohort_recurrence_family_count_by_pair_family)) != tuple(
        sorted(expected_pair_families)
    ):
        raise ContractError(
            "Block 2 requires Block 1 recurrence family-count coverage for every confirmatory pair family"
        )
    if not contract.n_recurrence_used_patients_by_pair_family:
        raise ContractError(
            "Block 2 requires Block 1 to report n_recurrence_used_patients_by_pair_family"
        )
    if tuple(sorted(contract.n_recurrence_used_patients_by_pair_family)) != tuple(
        sorted(expected_pair_families)
    ):
        raise ContractError(
            "Block 2 requires Block 1 recurrence patient-count coverage for every confirmatory pair family"
        )
    if tuple(contract.family_summary_scales) != ("burden_weighted", "community_mean"):
        raise ContractError(
            "Block 2 requires the frozen Block 1 family summary scales to remain "
            "('burden_weighted', 'community_mean')"
        )
    if tuple(contract.proof_carrying_family_summaries) != ("self_retention", "depletion"):
        raise ContractError(
            "Block 2 requires the frozen Block 1 proof-carrying family summaries to remain "
            "('self_retention', 'depletion')"
        )
    if tuple(contract.supportive_family_summaries) != ("off_diagonal_remodeling", "emergence"):
        raise ContractError(
            "Block 2 requires the frozen Block 1 supportive family summaries to remain "
            "('off_diagonal_remodeling', 'emergence')"
        )
    if contract.source_eligibility_rule != SOURCE_ELIGIBILITY_RULE:
        raise ContractError(
            "Block 2 requires the frozen Block 1 source eligibility rule to remain "
            f"{SOURCE_ELIGIBILITY_RULE!r}"
        )
    if contract.target_eligibility_rule != TARGET_ELIGIBILITY_RULE:
        raise ContractError(
            "Block 2 requires the frozen Block 1 target eligibility rule to remain "
            f"{TARGET_ELIGIBILITY_RULE!r}"
        )


def _contract_checks(
    contract: TaskABlock1BundleContract,
    *,
    config_enabled: bool,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "check": "block1_bundle_exists",
                "passed": contract.bundle_path.exists(),
                "detail": str(contract.bundle_path),
            },
            {
                "check": "block1_stage0_h5ad_exists",
                "passed": contract.stage0_h5ad.exists(),
                "detail": str(contract.stage0_h5ad),
            },
            {
                "check": "block1_family_summary_exists",
                "passed": contract.family_summary_path.exists(),
                "detail": str(contract.family_summary_path),
            },
            {
                "check": "block1_source_community_summary_exists",
                "passed": contract.source_community_summary_path.exists(),
                "detail": str(contract.source_community_summary_path),
            },
            {
                "check": "block1_target_community_summary_exists",
                "passed": contract.target_community_summary_path.exists(),
                "detail": str(contract.target_community_summary_path),
            },
            {
                "check": "block1_confirmatory_family_comparison_exists",
                "passed": contract.confirmatory_family_comparison_path.exists(),
                "detail": str(contract.confirmatory_family_comparison_path),
            },
            {
                "check": "block1_exploratory_source_community_comparison_exists",
                "passed": contract.exploratory_source_community_comparison_path.exists(),
                "detail": str(contract.exploratory_source_community_comparison_path),
            },
            {
                "check": "block1_exploratory_target_community_comparison_exists",
                "passed": contract.exploratory_target_community_comparison_path.exists(),
                "detail": str(contract.exploratory_target_community_comparison_path),
            },
            {
                "check": "block1_bundle_artifact_state_evidence_ready",
                "passed": contract.artifact_state == EVIDENCE_READY_STATE,
                "detail": contract.artifact_state,
            },
            {
                "check": "block1_bundle_implementation_tier_canonical_full",
                "passed": contract.implementation_tier == "canonical_full",
                "detail": contract.implementation_tier,
            },
            {
                "check": "block1_bundle_evidence_lineage_canonical_rerun",
                "passed": contract.evidence_lineage == "canonical_rerun",
                "detail": contract.evidence_lineage,
            },
            {
                "check": "block1_bundle_fit_surface_fit_stride",
                "passed": contract.fit_surface == "fit_stride",
                "detail": contract.fit_surface,
            },
            {
                "check": "block0_gate_passed",
                "passed": contract.block0_gate_status == PASSED_STATUS,
                "detail": contract.block0_gate_status,
            },
            {
                "check": "block1_recurrence_summary_exists",
                "passed": contract.recurrence_summary_path is not None
                and contract.recurrence_summary_path.exists(),
                "detail": (
                    str(contract.recurrence_summary_path)
                    if contract.recurrence_summary_path is not None
                    else ""
                ),
            },
            {
                "check": "block1_recurrence_families_exists",
                "passed": contract.recurrence_families_path is not None
                and contract.recurrence_families_path.exists(),
                "detail": (
                    str(contract.recurrence_families_path)
                    if contract.recurrence_families_path is not None
                    else ""
                ),
            },
            {
                "check": "block1_recurrence_embeddings_exists",
                "passed": contract.recurrence_embeddings_path is not None
                and contract.recurrence_embeddings_path.exists(),
                "detail": (
                    str(contract.recurrence_embeddings_path)
                    if contract.recurrence_embeddings_path is not None
                    else ""
                ),
            },
            {
                "check": "block1_cohort_recurrence_fit_status_valid",
                "passed": contract.cohort_recurrence_fit_status in RECURRENCE_STATUS_VALUES,
                "detail": contract.cohort_recurrence_fit_status,
            },
            {
                "check": "block1_recurrence_status_by_pair_family_complete",
                "passed": tuple(sorted(contract.cohort_recurrence_fit_status_by_pair_family))
                == tuple(sorted(contract.confirmatory_pair_families)),
                "detail": json.dumps(
                    dict(sorted(contract.cohort_recurrence_fit_status_by_pair_family.items()))
                ),
            },
            {
                "check": "block1_recurrence_family_count_by_pair_family_complete",
                "passed": tuple(sorted(contract.cohort_recurrence_family_count_by_pair_family))
                == tuple(sorted(contract.confirmatory_pair_families)),
                "detail": json.dumps(
                    dict(sorted(contract.cohort_recurrence_family_count_by_pair_family.items()))
                ),
            },
            {
                "check": "block1_n_recurrence_used_patients_by_pair_family_complete",
                "passed": tuple(sorted(contract.n_recurrence_used_patients_by_pair_family))
                == tuple(sorted(contract.confirmatory_pair_families)),
                "detail": json.dumps(
                    dict(sorted(contract.n_recurrence_used_patients_by_pair_family.items()))
                ),
            },
            {
                "check": "confirmatory_pair_families_scoped",
                "passed": tuple(contract.confirmatory_pair_families) == ("TC-IM", "TC-PT"),
                "detail": json.dumps(list(contract.confirmatory_pair_families)),
            },
            {
                "check": "summary_contract_version_frozen",
                "passed": contract.summary_contract_version == SUMMARY_CONTRACT_VERSION,
                "detail": contract.summary_contract_version,
            },
            {
                "check": "paired_comparison_contract_version_frozen",
                "passed": contract.paired_comparison_contract_version
                == PAIRED_COMPARISON_CONTRACT_VERSION,
                "detail": contract.paired_comparison_contract_version,
            },
            {
                "check": "proof_carrying_family_summaries_frozen",
                "passed": tuple(contract.proof_carrying_family_summaries) == ("self_retention", "depletion"),
                "detail": json.dumps(list(contract.proof_carrying_family_summaries)),
            },
            {
                "check": "supportive_family_summaries_frozen",
                "passed": tuple(contract.supportive_family_summaries) == ("off_diagonal_remodeling", "emergence"),
                "detail": json.dumps(list(contract.supportive_family_summaries)),
            },
            {
                "check": "source_eligibility_rule_frozen",
                "passed": contract.source_eligibility_rule == SOURCE_ELIGIBILITY_RULE,
                "detail": contract.source_eligibility_rule,
            },
            {
                "check": "target_eligibility_rule_frozen",
                "passed": contract.target_eligibility_rule == TARGET_ELIGIBILITY_RULE,
                "detail": contract.target_eligibility_rule,
            },
            {
                "check": "config_enables_block2",
                "passed": bool(config_enabled),
                "detail": str(config_enabled),
            },
        ]
    )


def _filter_stage0_subset(
    adata: Any,
    *,
    patient_ids: tuple[str, ...],
    dropped_roi_ids: tuple[str, ...],
) -> Any:
    subset = adata[
        adata.obs["patient_id"].astype(str).isin(set(patient_ids)).to_numpy(dtype=bool)
    ].copy()
    if not dropped_roi_ids:
        return subset
    return subset[
        ~subset.obs["roi_id"].astype(str).isin(set(dropped_roi_ids)).to_numpy(dtype=bool)
    ].copy()


def _config_matches_block1_provenance(
    contract: TaskABlock1BundleContract,
    *,
    current_fingerprint: str,
) -> bool:
    if contract.config_fingerprint == current_fingerprint:
        return True
    current_raw_config = load_raw_task_a_config(contract.config_path)
    block1_compat_config = dict(current_raw_config)
    block1_compat_config.pop("block2", None)
    block1_compat_config["enabled_blocks"] = [
        str(block_name)
        for block_name in current_raw_config.get("enabled_blocks", [])
        if str(block_name) != BLOCK_NAME
    ]
    block1_compat_fingerprint = compute_task_a_config_fingerprint(block1_compat_config)
    return contract.config_fingerprint == block1_compat_fingerprint


def _resume_paths(output_root: Path) -> dict[str, Path]:
    return {
        "meta": output_root / RESUME_META_FILENAME,
        "manifest": output_root / RESUME_MANIFEST_FILENAME,
        "assessments": output_root / RESUME_ASSESSMENTS_FILENAME,
    }


def _clear_resume_checkpoint(output_root: Path) -> None:
    for path in _resume_paths(output_root).values():
        if path.exists():
            path.unlink()


def _write_resume_checkpoint(
    *,
    output_root: Path,
    block1_bundle_path: Path,
    config_fingerprint: str,
    replicate_manifest_df: pd.DataFrame,
    assessment_rows: list[dict[str, Any]],
) -> None:
    paths = _resume_paths(output_root)
    paths["meta"].write_text(
        json.dumps(
            {
                "block1_bundle_path": str(block1_bundle_path),
                "config_fingerprint": str(config_fingerprint),
                "n_replicates": int(replicate_manifest_df.shape[0]),
                "n_assessment_rows": int(len(assessment_rows)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    replicate_manifest_df.to_csv(paths["manifest"], index=False)
    paths["assessments"].write_text(
        json.dumps(assessment_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_resume_checkpoint(
    *,
    output_root: Path,
    block1_bundle_path: Path,
    config_fingerprint: str,
    replicate_manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]] | None:
    paths = _resume_paths(output_root)
    if not all(path.exists() for path in paths.values()):
        return None

    resume_meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if str(resume_meta.get("block1_bundle_path", "")) != str(block1_bundle_path):
        raise ContractError(
            "Existing Block 2 resume checkpoint was created from a different Block 1 bundle. "
            "Clear the output directory or rerun without --resume."
        )
    if str(resume_meta.get("config_fingerprint", "")) != str(config_fingerprint):
        raise ContractError(
            "Existing Block 2 resume checkpoint was created from a different Task A config fingerprint. "
            "Clear the output directory or rerun without --resume."
        )

    checkpoint_manifest_df = pd.read_csv(paths["manifest"], keep_default_na=False)
    required_columns = set(REPLICATE_RESUME_KEY_COLUMNS) | set(REPLICATE_RESUME_MUTABLE_COLUMNS)
    missing_columns = sorted(required_columns - set(checkpoint_manifest_df.columns))
    if missing_columns:
        raise ContractError(
            "Existing Block 2 resume checkpoint is missing required columns: "
            f"{missing_columns}"
        )

    current_keys = replicate_manifest_df.loc[:, list(REPLICATE_RESUME_KEY_COLUMNS)].reset_index(drop=True)
    checkpoint_keys = checkpoint_manifest_df.loc[:, list(REPLICATE_RESUME_KEY_COLUMNS)].reset_index(drop=True)
    if not current_keys.equals(checkpoint_keys):
        raise ContractError(
            "Existing Block 2 resume checkpoint does not match the current replicate manifest. "
            "Clear the output directory or rerun without --resume."
        )

    assessment_rows = json.loads(paths["assessments"].read_text(encoding="utf-8"))
    if not isinstance(assessment_rows, list):
        raise ContractError("Existing Block 2 resume assessments payload must be a JSON list")

    restored_manifest_df = replicate_manifest_df.copy()
    for column_name in REPLICATE_RESUME_MUTABLE_COLUMNS:
        restored_manifest_df[column_name] = checkpoint_manifest_df[column_name].to_numpy()
    return restored_manifest_df, [dict(row) for row in assessment_rows]


def write_block2_bundle(
    *,
    block1_bundle_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> Path:
    contract = load_block1_bundle_contract(block1_bundle_path)
    config_bundle = load_task_a_config_bundle(contract.config_path)
    if not _config_matches_block1_provenance(
        contract,
        current_fingerprint=config_bundle.config_fingerprint,
    ):
        raise ContractError(
            "Block 1 bundle provenance mismatch: config content changed after the "
            "bundle was produced outside the Block 2-only config surface"
        )
    if BLOCK_NAME not in config_bundle.enabled_blocks:
        raise ContractError(
            f"Task A config does not enable {BLOCK_NAME}; enabled_blocks={list(config_bundle.enabled_blocks)}"
        )
    validate_block1_bundle_contract(contract)

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_family_df = pd.read_csv(contract.confirmatory_family_comparison_path)
    full_source_comparison_df = pd.read_csv(contract.exploratory_source_community_comparison_path)
    full_target_comparison_df = pd.read_csv(contract.exploratory_target_community_comparison_path)
    full_source_summary_df = pd.read_csv(contract.source_community_summary_path)
    anchor_df = build_anchor_findings(
        full_family_df=full_family_df,
        full_source_comparison_df=full_source_comparison_df,
        full_target_comparison_df=full_target_comparison_df,
        full_source_summary_df=full_source_summary_df,
        block2_config=config_bundle.block2,
    )

    stage0_adata = load_task_a_dataset_handle(contract.stage0_h5ad).adata
    replicate_manifest_df = build_replicate_manifest(
        stage0_adata,
        config_bundle=config_bundle,
    )
    assessment_rows: list[dict[str, Any]] = []
    if resume:
        restored = _load_resume_checkpoint(
            output_root=output_root,
            block1_bundle_path=contract.bundle_path,
            config_fingerprint=config_bundle.config_fingerprint,
            replicate_manifest_df=replicate_manifest_df,
        )
        if restored is None:
            _write_resume_checkpoint(
                output_root=output_root,
                block1_bundle_path=contract.bundle_path,
                config_fingerprint=config_bundle.config_fingerprint,
                replicate_manifest_df=replicate_manifest_df,
                assessment_rows=assessment_rows,
            )
        else:
            replicate_manifest_df, assessment_rows = restored
    else:
        _clear_resume_checkpoint(output_root)

    for idx, replicate_row in replicate_manifest_df.iterrows():
        if str(replicate_row["route_status"]) != REPLICATE_STATUS_PENDING:
            continue

        patient_subset = tuple(json.loads(str(replicate_row["patient_subset_json"])))
        dropped_roi_ids = tuple(json.loads(str(replicate_row["dropped_roi_ids_json"])))
        subset = _filter_stage0_subset(
            stage0_adata,
            patient_ids=patient_subset,
            dropped_roi_ids=dropped_roi_ids,
        )
        replicate_manifest_df.loc[idx, "n_patients_retained"] = int(
            subset.obs["patient_id"].astype(str).nunique()
        )
        replicate_manifest_df.loc[idx, "n_rois_retained"] = int(
            subset.obs.loc[:, ["patient_id", "roi_id"]].drop_duplicates().shape[0]
        )
        replicate_manifest_df.loc[idx, "n_cells_retained"] = int(subset.n_obs)
        if int(subset.n_obs) <= 0:
            replicate_manifest_df.loc[idx, "route_status"] = REPLICATE_STATUS_FAILED
            replicate_manifest_df.loc[idx, "failure_reason"] = "Perturbation retained zero cells."
            if resume:
                _write_resume_checkpoint(
                    output_root=output_root,
                    block1_bundle_path=contract.bundle_path,
                    config_fingerprint=config_bundle.config_fingerprint,
                    replicate_manifest_df=replicate_manifest_df,
                    assessment_rows=assessment_rows,
                )
            continue

        try:
            family_comparison_df, source_comparison_df, target_comparison_df, source_summary_df = (
                run_block1_reestimate_for_block2(
                    subset,
                    config_bundle=config_bundle,
                    route_name=str(replicate_row["route_name"]),
                    replicate_index=int(replicate_row["replicate_index"]),
                    selection_seed=int(replicate_row["selection_seed"]),
                )
            )
        except (ContractError, ValueError) as exc:
            replicate_manifest_df.loc[idx, "route_status"] = REPLICATE_STATUS_FAILED
            replicate_manifest_df.loc[idx, "failure_reason"] = str(exc)
            if resume:
                _write_resume_checkpoint(
                    output_root=output_root,
                    block1_bundle_path=contract.bundle_path,
                    config_fingerprint=config_bundle.config_fingerprint,
                    replicate_manifest_df=replicate_manifest_df,
                    assessment_rows=assessment_rows,
                )
            continue

        assessment_rows.extend(
            build_replicate_assessment_rows(
                anchor_df=anchor_df,
                family_comparison_df=family_comparison_df,
                source_comparison_df=source_comparison_df,
                target_comparison_df=target_comparison_df,
                source_summary_df=source_summary_df,
                route_name=str(replicate_row["route_name"]),
                route_group=str(replicate_row["route_group"]),
                replicate_index=int(replicate_row["replicate_index"]),
            )
        )
        replicate_manifest_df.loc[idx, "route_status"] = REPLICATE_STATUS_EXECUTED
        replicate_manifest_df.loc[idx, "failure_reason"] = ""
        if resume:
            _write_resume_checkpoint(
                output_root=output_root,
                block1_bundle_path=contract.bundle_path,
                config_fingerprint=config_bundle.config_fingerprint,
                replicate_manifest_df=replicate_manifest_df,
                assessment_rows=assessment_rows,
            )

    assessment_df = pd.DataFrame.from_records(assessment_rows)
    family_robustness_df, source_robustness_df, target_robustness_df = summarize_assessments_by_route(
        anchor_df=anchor_df,
        assessment_df=assessment_df,
        replicate_manifest_df=replicate_manifest_df,
        block2_config=config_bundle.block2,
    )
    summary_df = build_block2_summary(
        family_robustness_df=family_robustness_df,
        source_robustness_df=source_robustness_df,
        target_robustness_df=target_robustness_df,
    )

    summary_path = output_root / SUMMARY_FILENAME
    summary_df.to_csv(summary_path, index=False)
    contract_path = output_root / CONTRACT_FILENAME
    _contract_checks(
        contract,
        config_enabled=True,
    ).to_csv(contract_path, index=False)
    replicate_manifest_path = output_root / REPLICATE_MANIFEST_FILENAME
    replicate_manifest_df.to_csv(replicate_manifest_path, index=False)
    family_robustness_path = output_root / FAMILY_ROBUSTNESS_FILENAME
    family_robustness_df.to_csv(family_robustness_path, index=False)
    source_robustness_path = output_root / SOURCE_ROBUSTNESS_FILENAME
    source_robustness_df.to_csv(source_robustness_path, index=False)
    target_robustness_path = output_root / TARGET_ROBUSTNESS_FILENAME
    target_robustness_df.to_csv(target_robustness_path, index=False)

    manifest_path = output_root / MANIFEST_FILENAME
    manifest_payload = {
        "block": BLOCK_NAME,
        "scientific_role": SCIENTIFIC_ROLE,
        "status": "active",
        "artifact_state": EVIDENCE_READY_STATE,
        "scientific_interpretation_allowed": False,
        "claim_scope": CLAIM_SCOPE,
        "implementation_tier": "canonical_full",
        "evidence_lineage": "canonical_rerun",
        "fit_surface": "fit_stride",
        "config_path": str(config_bundle.config_path),
        "config_fingerprint": config_bundle.config_fingerprint,
        "output_dir": str(output_root),
        "bundle_path": str(manifest_path),
        "block1_bundle_path": str(contract.bundle_path),
        "block0_bundle_path": str(contract.block0_bundle_path),
        "block1_stage0_mapping_path": str(contract.mapping_manifest_path),
        "block1_core_fit_dry_run_path": str(contract.core_fit_dry_run_path),
        "block1_recurrence_summary_path": (
            str(contract.recurrence_summary_path)
            if contract.recurrence_summary_path is not None
            else None
        ),
        "block1_recurrence_families_path": (
            str(contract.recurrence_families_path)
            if contract.recurrence_families_path is not None
            else None
        ),
        "block1_recurrence_embeddings_path": (
            str(contract.recurrence_embeddings_path)
            if contract.recurrence_embeddings_path is not None
            else None
        ),
        "block1_family_summary_path": str(contract.family_summary_path),
        "block1_source_community_summary_path": str(contract.source_community_summary_path),
        "block1_target_community_summary_path": str(contract.target_community_summary_path),
        "block1_confirmatory_family_comparison_path": str(
            contract.confirmatory_family_comparison_path
        ),
        "block1_exploratory_source_community_comparison_path": str(
            contract.exploratory_source_community_comparison_path
        ),
        "block1_exploratory_target_community_comparison_path": str(
            contract.exploratory_target_community_comparison_path
        ),
        "summary_path": str(summary_path),
        "contract_path": str(contract_path),
        "replicate_manifest_path": str(replicate_manifest_path),
        "family_robustness_path": str(family_robustness_path),
        "source_community_robustness_path": str(source_robustness_path),
        "target_community_robustness_path": str(target_robustness_path),
        "summary_rows": int(summary_df.shape[0]),
        "replicate_rows": int(replicate_manifest_df.shape[0]),
        "block1_cohort_recurrence_fit_status": contract.cohort_recurrence_fit_status,
        "block1_cohort_recurrence_fit_status_by_pair_family": dict(
            sorted(contract.cohort_recurrence_fit_status_by_pair_family.items())
        ),
        "block1_cohort_recurrence_family_count": int(contract.cohort_recurrence_family_count),
        "block1_cohort_recurrence_family_count_by_pair_family": dict(
            sorted(contract.cohort_recurrence_family_count_by_pair_family.items())
        ),
        "block1_n_recurrence_used_patients": int(contract.n_recurrence_used_patients),
        "block1_n_recurrence_used_patients_by_pair_family": dict(
            sorted(contract.n_recurrence_used_patients_by_pair_family.items())
        ),
        "primary_routes": list(config_bundle.block2.primary_routes),
        "primary_source_communities": list(config_bundle.block2.primary_source_communities),
        "primary_target_communities": list(config_bundle.block2.primary_target_communities),
        "executed_replicates": int(
            (replicate_manifest_df["route_status"].astype(str) == REPLICATE_STATUS_EXECUTED).sum()
        ),
        "failed_replicates": int(
            (replicate_manifest_df["route_status"].astype(str) != REPLICATE_STATUS_EXECUTED).sum()
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _clear_resume_checkpoint(output_root)
    return manifest_path


__all__ = [
    "BLOCK_NAME",
    "CLAIM_SCOPE",
    "CONTRACT_FILENAME",
    "FAMILY_ROBUSTNESS_FILENAME",
    "MANIFEST_FILENAME",
    "REPLICATE_MANIFEST_FILENAME",
    "SCIENTIFIC_ROLE",
    "SOURCE_ROBUSTNESS_FILENAME",
    "SUMMARY_FILENAME",
    "TARGET_ROBUSTNESS_FILENAME",
    "TaskABlock1BundleContract",
    "build_block2_summary",
    "load_block1_bundle_contract",
    "validate_block1_bundle_contract",
    "write_block2_bundle",
]
