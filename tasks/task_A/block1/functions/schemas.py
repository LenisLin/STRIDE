"""Schemas and filenames for the live Task A Block 1 execute/analyze surface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stride.errors import ContractError

BLOCK1_LIVE_ID = "block1_real_data_discovery"
CONFIRMATORY_PAIR_FAMILIES: tuple[str, str] = ("TC-IM", "TC-PT")
FROZEN_CONFIRMATORY_FAMILY_CONTRACT: dict[str, tuple[str, str, str]] = {
    "TC-IM": ("TC", "IM", "confirmatory"),
    "TC-PT": ("TC", "PT", "confirmatory"),
}
RUN_SCOPE_FULL_COHORT = "full_cohort"
RUN_SCOPE_PATIENT_SUBSET = "patient_subset"
READINESS_STATUS_EVIDENCE_READY = "evidence_ready"
READINESS_STATUS_DIAGNOSTIC = "diagnostic"

BLOCK1_EXECUTE_MANIFEST_FILENAME = "block1_execute_manifest.json"
BLOCK1_ANALYSIS_MANIFEST_FILENAME = "block1_analysis_manifest.json"
BLOCK1_FAMILY_SUMMARY_FILENAME = "block1_family_summary.csv"
BLOCK1_SOURCE_COMMUNITY_SUMMARY_FILENAME = "block1_source_community_summary.csv"
BLOCK1_TARGET_COMMUNITY_SUMMARY_FILENAME = "block1_target_community_summary.csv"
BLOCK1_CONFIRMATORY_FAMILY_COMPARISON_FILENAME = "block1_confirmatory_family_comparison.csv"
BLOCK1_SOURCE_COMMUNITY_COMPARISON_FILENAME = "block1_source_community_comparison.csv"
BLOCK1_TARGET_COMMUNITY_COMPARISON_FILENAME = "block1_target_community_comparison.csv"
BLOCK1_COHORT_RELATION_COMPARISON_FILENAME = "block1_cohort_relation_comparison.csv"
BLOCK1_FAMILY_STATISTICAL_SUPPLEMENT_FILENAME = "block1_family_statistical_supplement.csv"
BLOCK1_SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME = (
    "block1_source_community_statistical_supplement.csv"
)
BLOCK1_TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME = (
    "block1_target_community_statistical_supplement.csv"
)
BLOCK1_RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_FILENAME = (
    "block1_relation_element_statistical_supplement.csv"
)
BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION = "task_a_block1_statistical_supplement_v1"
STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA = 0.05
STATISTICAL_SUPPLEMENT_Q_ALPHA = 0.05

EXECUTE_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_name",
    "phase",
    "config_path",
    "config_fingerprint",
    "stage0_h5ad",
    "run_scope",
    "patient_ids",
    "confirmatory_pair_families",
    "family_exports",
    "readiness_status",
    "scientific_interpretation_allowed",
    "prohibited_outputs",
)
ANALYSIS_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_name",
    "phase",
    "source_execute_manifest_path",
    "source_execute_manifest_sha256",
    "run_scope",
    "readiness_status",
    "summary_contract_version",
    "comparison_contract_version",
    "input_native_exports",
    "family_summary_path",
    "source_community_summary_path",
    "target_community_summary_path",
    "confirmatory_family_comparison_path",
    "source_community_comparison_path",
    "target_community_comparison_path",
    "cohort_relation_comparison_path",
    "family_statistical_supplement_path",
    "source_community_statistical_supplement_path",
    "target_community_statistical_supplement_path",
    "relation_element_statistical_supplement_path",
    "statistical_supplement_contract_version",
    "statistical_method",
    "sign_test_method",
    "multiple_testing_policy",
    "effect_floor_abs_median_delta",
    "statistical_supplement_q_alpha",
    "scientific_interpretation_allowed",
    "emits_p_values",
    "emits_figures",
    "emits_annotations",
)

PROHIBITED_OUTPUT_MARKERS: tuple[str, ...] = (
    "p_value",
    "p-values",
    "fdr",
    "figure",
    "annotation",
)

COHORT_RELATION_COMPARISON_COLUMNS: tuple[str, ...] = (
    "component",
    "relation_axis",
    "source_community_id",
    "target_community_id",
    "tc_im_value",
    "tc_pt_value",
    "delta_tc_im_minus_tc_pt",
    "contrast_direction",
    "tc_im_support_n_patients",
    "tc_pt_support_n_patients",
    "tc_im_within_family_dispersion",
    "tc_pt_within_family_dispersion",
    "comparison_scope_role",
)

_STATISTICAL_COMMON_COLUMNS: tuple[str, ...] = (
    "statistical_supplement_contract_version",
    "statistical_surface",
    "pair_family_left",
    "pair_family_right",
    "n_patients",
    "n_estimable",
    "n_nonzero_delta",
    "support_positive_n",
    "support_negative_n",
    "support_zero_n",
    "support_positive_fraction",
    "support_negative_fraction",
    "tc_im_median",
    "tc_pt_median",
    "median_delta",
    "mean_delta",
    "abs_median_delta",
    "contrast_direction",
    "wilcoxon_p_value",
    "sign_test_p_value",
    "statistical_test",
    "sign_test",
    "bh_scope",
    "multiple_testing_policy",
    "bh_q_value",
    "q_alpha",
    "q_pass",
    "effect_floor_abs_median_delta",
    "effect_floor_pass",
    "review_candidate",
)

FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS: tuple[str, ...] = (
    "statistical_supplement_contract_version",
    "statistical_surface",
    "summary_name",
    "summary_role",
    "scale",
    "eligible_entity_axis",
    *_STATISTICAL_COMMON_COLUMNS[2:],
    "comparison_scope_role",
)
SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS: tuple[str, ...] = (
    "statistical_supplement_contract_version",
    "statistical_surface",
    "source_community_id",
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    *_STATISTICAL_COMMON_COLUMNS[2:],
    "comparison_scope_role",
)
TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS: tuple[str, ...] = (
    "statistical_supplement_contract_version",
    "statistical_surface",
    "target_community_id",
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    *_STATISTICAL_COMMON_COLUMNS[2:],
    "comparison_scope_role",
)
RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS: tuple[str, ...] = (
    "statistical_supplement_contract_version",
    "statistical_surface",
    "component",
    "relation_axis",
    "source_community_id",
    "target_community_id",
    *_STATISTICAL_COMMON_COLUMNS[2:],
    "cohort_tc_im_value",
    "cohort_tc_pt_value",
    "cohort_delta_tc_im_minus_tc_pt",
    "cohort_comparison_scope_role",
)


@dataclass(frozen=True)
class Block1RunRequest:
    task_config_path: Path
    stage0_h5ad_path: Path
    output_dir: Path
    patient_ids: tuple[str, ...]
    run_scope: str
    device: object | None = None


@dataclass(frozen=True)
class Block1FamilyExportRecord:
    pair_family: str
    source_domain: str
    target_domain: str
    claim_role: str
    native_export_manifest_path: Path
    native_export_manifest_sha256: str
    fit_status: str
    cohort_fit_status: str
    patient_count: int
    patient_record_count: int
    cohort_record_count: int
    k_states: int
    fit_surface: str = "stride.tl.fit"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "pair_family": self.pair_family,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "claim_role": self.claim_role,
            "fit_surface": self.fit_surface,
            "native_export_manifest_path": str(self.native_export_manifest_path),
            "native_export_manifest_sha256": self.native_export_manifest_sha256,
            "fit_status": self.fit_status,
            "cohort_fit_status": self.cohort_fit_status,
            "patient_count": self.patient_count,
            "patient_record_count": self.patient_record_count,
            "cohort_record_count": self.cohort_record_count,
            "k_states": self.k_states,
        }


@dataclass(frozen=True)
class Block1ExecuteManifest:
    task_name: str
    phase: str
    config_path: Path
    config_fingerprint: str
    stage0_h5ad: Path
    run_scope: str
    patient_ids: tuple[str, ...]
    confirmatory_pair_families: tuple[str, ...]
    family_exports: tuple[Block1FamilyExportRecord, ...]
    readiness_status: str
    scientific_interpretation_allowed: bool
    prohibited_outputs: tuple[str, ...]


def validate_block1_family_contract(
    name: str,
    source_domain: str,
    target_domain: str,
    claim_role: str,
) -> None:
    """Validate frozen Block 1 confirmatory family domain semantics."""
    family_name = str(name)
    expected = FROZEN_CONFIRMATORY_FAMILY_CONTRACT.get(family_name)
    if expected is None:
        raise ContractError(f"Block 1 family {family_name!r} is outside the frozen confirmatory contract")
    observed = (str(source_domain), str(target_domain), str(claim_role))
    if observed != expected:
        raise ContractError(
            "Block 1 family contract mismatch for "
            f"{family_name!r}: expected source_domain={expected[0]!r}, "
            f"target_domain={expected[1]!r}, claim_role={expected[2]!r}; "
            f"observed source_domain={observed[0]!r}, "
            f"target_domain={observed[1]!r}, claim_role={observed[2]!r}"
        )


__all__ = [
    "ANALYSIS_MANIFEST_REQUIRED_FIELDS",
    "BLOCK1_ANALYSIS_MANIFEST_FILENAME",
    "BLOCK1_COHORT_RELATION_COMPARISON_FILENAME",
    "BLOCK1_CONFIRMATORY_FAMILY_COMPARISON_FILENAME",
    "BLOCK1_EXECUTE_MANIFEST_FILENAME",
    "BLOCK1_FAMILY_STATISTICAL_SUPPLEMENT_FILENAME",
    "BLOCK1_FAMILY_SUMMARY_FILENAME",
    "BLOCK1_LIVE_ID",
    "BLOCK1_RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_FILENAME",
    "BLOCK1_SOURCE_COMMUNITY_COMPARISON_FILENAME",
    "BLOCK1_SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME",
    "BLOCK1_SOURCE_COMMUNITY_SUMMARY_FILENAME",
    "BLOCK1_STATISTICAL_SUPPLEMENT_CONTRACT_VERSION",
    "BLOCK1_TARGET_COMMUNITY_COMPARISON_FILENAME",
    "BLOCK1_TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_FILENAME",
    "BLOCK1_TARGET_COMMUNITY_SUMMARY_FILENAME",
    "COHORT_RELATION_COMPARISON_COLUMNS",
    "CONFIRMATORY_PAIR_FAMILIES",
    "EXECUTE_MANIFEST_REQUIRED_FIELDS",
    "FAMILY_STATISTICAL_SUPPLEMENT_COLUMNS",
    "FROZEN_CONFIRMATORY_FAMILY_CONTRACT",
    "PROHIBITED_OUTPUT_MARKERS",
    "READINESS_STATUS_DIAGNOSTIC",
    "READINESS_STATUS_EVIDENCE_READY",
    "RELATION_ELEMENT_STATISTICAL_SUPPLEMENT_COLUMNS",
    "RUN_SCOPE_FULL_COHORT",
    "RUN_SCOPE_PATIENT_SUBSET",
    "SOURCE_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS",
    "STATISTICAL_SUPPLEMENT_EFFECT_FLOOR_ABS_MEDIAN_DELTA",
    "STATISTICAL_SUPPLEMENT_Q_ALPHA",
    "TARGET_COMMUNITY_STATISTICAL_SUPPLEMENT_COLUMNS",
    "Block1ExecuteManifest",
    "Block1FamilyExportRecord",
    "Block1RunRequest",
    "validate_block1_family_contract",
]
