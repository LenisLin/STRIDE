"""Task A Block 0 calibration schema and vocabulary skeleton.

Block 0 asks whether real `TC-IM` STRIDE relation structure departs from a
within-patient count-preserving FOV domain-label permutation null. Schemas define the allowed
Task A config, Stage 0 h5ad, output dir, permutation count, master seed, and
optional selector surface plus the formal calibration outputs. They do not
define biology interpretation, pass/fail gates, downstream execution
authorization, or inputs from prepare, descriptive-atlas, old suitability, or
passed-bundle artifacts. See `tasks/task_A/README.md`,
`tasks/task_A/contracts/artifact_contracts.md`, and
`tasks/task_A/contracts/design_freeze.py`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from stride.errors import ContractError


BLOCK_NAME = "block0_calibration"
EXECUTION_NAME = "block0_execution_cache"
REAL_FAMILY = "TC-IM"
NULL_FAMILY = "TC-IM_within_patient_domain_label_permutation_null"
SOURCE_DOMAIN = "TC"
TARGET_DOMAIN = "IM"
FULL_COHORT_SCOPE = "full_cohort"
PATIENT_SUBSET_SCOPE = "patient_subset"
DEMO_SUBSET_SCOPE = "demo_subset"
FULL_CALIBRATION_N_PERMUTATIONS = 199
MIN_N_PERMUTATIONS = 2
CALIBRATION_READY_STATUS = "calibration_ready"
DIAGNOSTIC_READINESS_STATUS = "diagnostic"
FAILED_READINESS_STATUS = "failed"
BLOCK0_ANALYSIS_SPEC_VERSION = "block1_family_summary_calibration_v1"
P_VALUE_CORRECTION = "plus_one"
FIT_LABEL_REAL = "real"
FIT_LABEL_NULL = "null"
SUMMARY_NAMES: tuple[str, ...] = (
    "self_retention",
    "depletion",
    "off_diagonal_remodeling",
    "emergence",
)
REFERENCE_STATS: tuple[str, ...] = ("median", "mean")
FAMILY_SUMMARY_SCALES: tuple[str, ...] = ("burden_weighted", "community_mean")
SUMMARY_ROLES: dict[str, str] = {
    "self_retention": "proof_carrying",
    "depletion": "proof_carrying",
    "off_diagonal_remodeling": "diagnostic_supportive",
    "emergence": "supportive",
}
SUMMARY_EXPECTED_TAILS: dict[str, str] = {
    "self_retention": "left",
    "depletion": "right",
    "off_diagonal_remodeling": "right",
    "emergence": "right",
}
EFFECT_RATIO_STATUS_ESTIMABLE = "estimable"
EFFECT_RATIO_STATUS_NOT_ESTIMABLE = "not_estimable"
CALIBRATION_MANIFEST_FILENAME = "block0_calibration_manifest.json"
PATIENT_CALIBRATION_FILENAME = "block0_patient_calibration.csv"
METRIC_SUMMARY_FILENAME = "block0_metric_summary.csv"
EXECUTION_MANIFEST_FILENAME = "block0_execution_manifest.json"
EXECUTION_PROGRESS_FILENAME = "block0_execution_progress.jsonl"
FIT_CACHE_FILENAME = "block0_fit_cache.npz"
FIT_CACHE_INDEX_FILENAME = "block0_fit_cache_index.csv"
FIT_CACHE_SCHEMA_VERSION = "block0_fit_cache_v1"
FIT_CACHE_INDEX_COLUMNS: tuple[str, ...] = (
    "record_id",
    "fit_label",
    "permutation_index",
    "patient_id",
    "fit_status",
)
EXECUTION_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_name",
    "config_path",
    "stage0_h5ad",
    "run_scope",
    "n_permutations",
    "master_seed",
    "seed_derivation_policy",
    "real_family",
    "null_family",
    "permutation_policy",
    "fit_status",
    "readiness_status",
    "patient_count",
    "record_count",
    "k_states",
    "fit_cache_schema_version",
    "fit_cache_path",
    "fit_cache_index_path",
    "fit_cache_sha256",
    "fit_cache_index_sha256",
    "progress_path",
)

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_name",
    "config_path",
    "stage0_h5ad",
    "run_scope",
    "n_permutations",
    "master_seed",
    "seed_derivation_policy",
    "real_family",
    "null_family",
    "permutation_policy",
    "summary_roles",
    "fit_status",
    "readiness_status",
    "analysis_spec_version",
    "source_execution_manifest_path",
    "source_fit_cache_path",
    "source_fit_cache_index_path",
    "source_fit_cache_sha256",
    "source_fit_cache_index_sha256",
    "patient_calibration_path",
    "metric_summary_path",
)
_PATIENT_BASE_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "run_scope",
    "real_family",
    "null_family",
    "n_permutations",
    "real_fit_status",
    "null_fit_status",
)
PATIENT_CALIBRATION_COLUMNS: tuple[str, ...] = (
    *_PATIENT_BASE_COLUMNS,
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "scale",
    "reference_stat",
    "expected_tail",
    "real_value",
    "null_reference",
    "empirical_p_value",
    "primary_tail_fraction",
    "opposite_tail_fraction",
    "effect_delta",
    "effect_ratio",
    "effect_ratio_status",
    "readiness_status",
)
METRIC_SUMMARY_COLUMNS: tuple[str, ...] = (
    "summary_name",
    "summary_role",
    "eligible_entity_axis",
    "scale",
    "cohort_stat",
    "expected_tail",
    "real_value",
    "null_reference",
    "empirical_p_value",
    "primary_tail_fraction",
    "opposite_tail_fraction",
    "effect_delta",
    "effect_ratio",
    "effect_ratio_status",
    "n_patient_delta_positive",
    "n_patient_delta_negative",
    "n_patient_delta_zero",
    "readiness_status",
)


def _validated_csv_row(
    values: Mapping[str, object],
    *,
    expected_columns: tuple[str, ...],
    row_label: str,
) -> dict[str, object]:
    """Return a CSV row in canonical order after exact schema validation."""
    missing = tuple(column for column in expected_columns if column not in values)
    extra = tuple(column for column in values if column not in expected_columns)
    if missing or extra:
        raise ContractError(
            f"{row_label} does not match the Block 0 schema; "
            f"missing={missing}, extra={extra}"
        )
    return {column: values[column] for column in expected_columns}


@dataclass(frozen=True)
class Block0RunConfig:
    """Resolved run configuration for the future Block 0 calibration runner."""

    config_path: Path
    data_path: Path
    output_dir: Path
    run_scope: str
    n_permutations: int
    master_seed: int
    patient_ids: tuple[str, ...] | None = None
    demo_subset_name: str | None = None

    def __post_init__(self) -> None:
        if self.n_permutations < MIN_N_PERMUTATIONS:
            raise ContractError(
                f"Task A Block 0 requires n_permutations >= {MIN_N_PERMUTATIONS}"
            )
        if self.run_scope not in {
            FULL_COHORT_SCOPE,
            PATIENT_SUBSET_SCOPE,
            DEMO_SUBSET_SCOPE,
        }:
            raise ContractError(f"Unsupported Block 0 run_scope: {self.run_scope!r}")
        if self.patient_ids is not None and len(self.patient_ids) == 0:
            raise ContractError("Block 0 patient_ids selector must not be empty")

    @property
    def readiness_status(self) -> str:
        """Return the readiness class implied by scope and permutation count."""
        if self.run_scope == FULL_COHORT_SCOPE and self.n_permutations == FULL_CALIBRATION_N_PERMUTATIONS:
            return CALIBRATION_READY_STATUS
        return DIAGNOSTIC_READINESS_STATUS


@dataclass(frozen=True)
class Block0PatientDomainCounts:
    """Within-patient `TC`/`IM` FOV count structure preserved by the null."""

    patient_id: str
    n_TC: int
    n_IM: int
    fov_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if str(self.patient_id).strip() == "":
            raise ContractError("Block0PatientDomainCounts.patient_id must be non-empty")
        if self.n_TC < 0 or self.n_IM < 0:
            raise ContractError("Block 0 patient domain counts require non-negative counts")
        if self.n_TC == 0 or self.n_IM == 0:
            raise ContractError("Block 0 patient domain counts require both TC and IM observations")
        if len(self.fov_ids) != self.n_TC + self.n_IM:
            raise ContractError("Block 0 patient domain counts must match the FOV id list length")


@dataclass(frozen=True)
class Block0DomainLabelPermutationAssignment:
    """Single FOV label assignment in one within-patient empirical-null permutation."""

    permutation_index: int
    patient_id: str
    fov_id: str
    original_domain_label: str
    permuted_domain_label: str
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.permutation_index, bool)
            or not isinstance(self.permutation_index, int)
            or self.permutation_index < 0
        ):
            raise ContractError("Block 0 permutation assignments require a non-negative index")
        if str(self.patient_id).strip() == "":
            raise ContractError("Block 0 permutation assignments require a non-empty patient_id")
        if str(self.fov_id).strip() == "":
            raise ContractError("Block 0 permutation assignments require a non-empty fov_id")
        allowed_labels = {SOURCE_DOMAIN, TARGET_DOMAIN}
        if self.original_domain_label not in allowed_labels:
            raise ContractError("Block 0 original_domain_label must be TC or IM")
        if self.permuted_domain_label not in allowed_labels:
            raise ContractError("Block 0 permuted_domain_label must be TC or IM")


@dataclass(frozen=True)
class Block0PatientCalibrationRow:
    """Patient-level diagnostic heterogeneity row matching the artifact contract."""

    values: Mapping[str, object]

    def to_csv_row(self) -> dict[str, object]:
        return _validated_csv_row(
            self.values,
            expected_columns=PATIENT_CALIBRATION_COLUMNS,
            row_label="Block0PatientCalibrationRow",
        )


@dataclass(frozen=True)
class Block0MetricSummaryRow:
    """Cohort-level calibration summary row, not an interpretation decision."""

    values: Mapping[str, object]

    def to_csv_row(self) -> dict[str, object]:
        return _validated_csv_row(
            self.values,
            expected_columns=METRIC_SUMMARY_COLUMNS,
            row_label="Block0MetricSummaryRow",
        )


@dataclass(frozen=True)
class Block0FitRecord:
    """Lightweight patient-level fit payload consumed by Block 0 calibration metrics."""

    patient_id: str
    fit_label: str
    A: object
    d: object
    e: object
    source_burden: object
    d_weights: object
    e_weights: object
    permutation_index: int | None = None
    fit_status: str = "ok"

    def __post_init__(self) -> None:
        if str(self.patient_id).strip() == "":
            raise ContractError("Block0FitRecord.patient_id must be non-empty")
        if self.fit_label not in {FIT_LABEL_REAL, FIT_LABEL_NULL}:
            raise ContractError(f"Unsupported Block0FitRecord.fit_label: {self.fit_label!r}")
        if self.fit_status != "ok":
            raise ContractError("Block0FitRecord currently requires fit_status='ok'")
        if self.fit_label == FIT_LABEL_REAL and self.permutation_index is not None:
            raise ContractError("Real Block0FitRecord must not carry permutation_index")
        if self.fit_label == FIT_LABEL_NULL:
            if (
                self.permutation_index is None
                or isinstance(self.permutation_index, bool)
                or not isinstance(self.permutation_index, int)
                or int(self.permutation_index) < 0
            ):
                raise ContractError("Null Block0FitRecord requires a non-negative permutation_index")
        for field_name in ("A", "d", "e", "source_burden", "d_weights", "e_weights"):
            if getattr(self, field_name) is None:
                raise ContractError(f"Block0FitRecord.{field_name} must be present")


__all__ = [
    "BLOCK_NAME",
    "BLOCK0_ANALYSIS_SPEC_VERSION",
    "CALIBRATION_MANIFEST_FILENAME",
    "CALIBRATION_READY_STATUS",
    "DEMO_SUBSET_SCOPE",
    "DIAGNOSTIC_READINESS_STATUS",
    "EFFECT_RATIO_STATUS_ESTIMABLE",
    "EFFECT_RATIO_STATUS_NOT_ESTIMABLE",
    "EXECUTION_MANIFEST_FILENAME",
    "EXECUTION_MANIFEST_REQUIRED_FIELDS",
    "EXECUTION_NAME",
    "EXECUTION_PROGRESS_FILENAME",
    "FAILED_READINESS_STATUS",
    "FIT_CACHE_FILENAME",
    "FIT_CACHE_INDEX_COLUMNS",
    "FIT_CACHE_INDEX_FILENAME",
    "FIT_CACHE_SCHEMA_VERSION",
    "FIT_LABEL_NULL",
    "FIT_LABEL_REAL",
    "FULL_CALIBRATION_N_PERMUTATIONS",
    "FULL_COHORT_SCOPE",
    "FAMILY_SUMMARY_SCALES",
    "MANIFEST_REQUIRED_FIELDS",
    "METRIC_SUMMARY_COLUMNS",
    "METRIC_SUMMARY_FILENAME",
    "MIN_N_PERMUTATIONS",
    "NULL_FAMILY",
    "PATIENT_CALIBRATION_COLUMNS",
    "PATIENT_CALIBRATION_FILENAME",
    "PATIENT_SUBSET_SCOPE",
    "P_VALUE_CORRECTION",
    "REAL_FAMILY",
    "REFERENCE_STATS",
    "SOURCE_DOMAIN",
    "SUMMARY_EXPECTED_TAILS",
    "SUMMARY_NAMES",
    "SUMMARY_ROLES",
    "TARGET_DOMAIN",
    "Block0DomainLabelPermutationAssignment",
    "Block0FitRecord",
    "Block0MetricSummaryRow",
    "Block0PatientCalibrationRow",
    "Block0PatientDomainCounts",
    "Block0RunConfig",
]
