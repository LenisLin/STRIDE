"""Bootstrap and uncertainty support helpers for STRIDE output analysis."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.longitudinal import resolve_fov_key
from ..errors import ContractError

try:
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = Any  # type: ignore[misc,assignment]


_ALLOWED_UNCERTAINTY_STATUSES: tuple[str, ...] = ("ok", "deferred", "failed")
_ALLOWED_REALIZED_FIT_STATUSES: tuple[str, ...] = ("ok", "deferred", "failed")


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if int(value) <= 0:
        raise ContractError(f"{field_name} must be a positive integer")


def _validate_uncertainty_status(value: str, *, field_name: str) -> None:
    if value not in _ALLOWED_UNCERTAINTY_STATUSES:
        raise ContractError(
            f"{field_name} must be one of {_ALLOWED_UNCERTAINTY_STATUSES}, got {value!r}"
        )


def _validate_realized_fit_status(value: str, *, field_name: str) -> None:
    if value not in _ALLOWED_REALIZED_FIT_STATUSES:
        raise ContractError(
            f"{field_name} must be one of {_ALLOWED_REALIZED_FIT_STATUSES}, got {value!r}"
        )


def _validate_scalar_or_nan(value: float, *, field_name: str) -> None:
    scalar = float(value)
    if not np.isfinite(scalar) and not np.isnan(scalar):
        raise ContractError(f"{field_name} must be finite or NaN")


def _mean_or_nan(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float), dtype=float))


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for generic bootstrap and stability routines."""

    n_replicates: int = 100
    random_state: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_replicates, field_name="BootstrapConfig.n_replicates")


@dataclass(frozen=True)
class PatientBootstrapConfig:
    """Explicit patient-level bootstrap configuration for STRIDE bridge uncertainty."""

    n_boot: int = 100
    random_state: int | None = None
    preserve_domain_stratification: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_boot, field_name="PatientBootstrapConfig.n_boot")


@dataclass(frozen=True)
class BootstrapArraySummary:
    """Elementwise bootstrap summary for one realized STRIDE bridge array."""

    mean: np.ndarray
    std: np.ndarray
    nonzero_frequency: np.ndarray
    mean_abs_deviation: float
    max_abs_deviation: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        std = np.asarray(self.std, dtype=float)
        nonzero_frequency = np.asarray(self.nonzero_frequency, dtype=float)
        if mean.shape != std.shape or mean.shape != nonzero_frequency.shape:
            raise ContractError(
                "BootstrapArraySummary.mean/std/nonzero_frequency must share one shape"
            )
        if not np.isfinite(mean).all():
            raise ContractError("BootstrapArraySummary.mean must be finite")
        if not np.all(np.isfinite(std) | np.isnan(std)):
            raise ContractError("BootstrapArraySummary.std must be finite or NaN")
        if not np.isfinite(nonzero_frequency).all():
            raise ContractError("BootstrapArraySummary.nonzero_frequency must be finite")
        if np.any((nonzero_frequency < 0.0) | (nonzero_frequency > 1.0)):
            raise ContractError(
                "BootstrapArraySummary.nonzero_frequency must lie in the closed interval [0, 1]"
            )
        if not np.isfinite(float(self.mean_abs_deviation)) or float(self.mean_abs_deviation) < 0.0:
            raise ContractError(
                "BootstrapArraySummary.mean_abs_deviation must be finite and non-negative"
            )
        if not np.isfinite(float(self.max_abs_deviation)) or float(self.max_abs_deviation) < 0.0:
            raise ContractError(
                "BootstrapArraySummary.max_abs_deviation must be finite and non-negative"
            )

    @property
    def mean_element_std(self) -> float:
        """Return the mean elementwise bootstrap standard deviation."""
        return float(np.nanmean(np.asarray(self.std, dtype=float)))

    @property
    def mean_nonzero_frequency(self) -> float:
        """Return the mean elementwise support frequency across bootstrap replicates."""
        return float(np.mean(np.asarray(self.nonzero_frequency, dtype=float), dtype=float))


@dataclass(frozen=True)
class PatientBootstrapUncertaintyResult:
    """Patient-level bootstrap uncertainty bundle for the realized STRIDE bridge."""

    patient_id: str
    realized_fit_status: str
    uncertainty_status: str
    eligible: bool
    n_boot: int
    bootstrap_seed: int | None = None
    replicate_statuses: tuple[str, ...] = ()
    replicate_diagnostics: tuple[Mapping[str, Any], ...] = ()
    A_summary: BootstrapArraySummary | None = None
    d_summary: BootstrapArraySummary | None = None
    e_summary: BootstrapArraySummary | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patient_id = str(self.patient_id).strip()
        if patient_id == "":
            raise ContractError("PatientBootstrapUncertaintyResult.patient_id must be non-empty")
        _validate_realized_fit_status(
            self.realized_fit_status,
            field_name="PatientBootstrapUncertaintyResult.realized_fit_status",
        )
        _validate_uncertainty_status(
            self.uncertainty_status,
            field_name="PatientBootstrapUncertaintyResult.uncertainty_status",
        )
        _validate_positive_int(self.n_boot, field_name="PatientBootstrapUncertaintyResult.n_boot")
        if len(self.replicate_statuses) != self.n_boot:
            raise ContractError(
                "PatientBootstrapUncertaintyResult.replicate_statuses must have length n_boot"
            )
        if len(self.replicate_diagnostics) != self.n_boot:
            raise ContractError(
                "PatientBootstrapUncertaintyResult.replicate_diagnostics must have length n_boot"
            )
        for status in self.replicate_statuses:
            _validate_uncertainty_status(
                str(status),
                field_name="PatientBootstrapUncertaintyResult.replicate_status",
            )

        has_any_summary = any(
            summary is not None for summary in (self.A_summary, self.d_summary, self.e_summary)
        )
        has_all_summaries = all(
            summary is not None for summary in (self.A_summary, self.d_summary, self.e_summary)
        )
        if has_any_summary and not has_all_summaries:
            raise ContractError(
                "PatientBootstrapUncertaintyResult must provide A_summary, d_summary, and e_summary together"
            )
        if self.uncertainty_status != "ok" and has_any_summary:
            raise ContractError(
                "Non-ok PatientBootstrapUncertaintyResult objects must not carry bootstrap summaries"
            )
        if self.realized_fit_status != "ok":
            if self.uncertainty_status != "deferred":
                raise ContractError(
                    "Non-ok realized_fit_status requires uncertainty_status='deferred'"
                )
            if self.eligible:
                raise ContractError(
                    "Non-ok realized_fit_status requires eligible=False"
                )
            if has_any_summary:
                raise ContractError(
                    "Non-ok realized_fit_status must not carry bootstrap summaries"
                )
        else:
            if not self.eligible:
                raise ContractError("realized_fit_status='ok' requires eligible=True")

        if has_all_summaries:
            if self.A_summary.mean.ndim != 2 or self.A_summary.mean.shape[0] != self.A_summary.mean.shape[1]:
                raise ContractError("PatientBootstrapUncertaintyResult.A_summary must be square [K, K]")
            if self.d_summary.mean.ndim != 1 or self.e_summary.mean.ndim != 1:
                raise ContractError("PatientBootstrapUncertaintyResult d/e summaries must be 1D [K]")
            n_states = self.A_summary.mean.shape[0]
            if self.d_summary.mean.shape != (n_states,) or self.e_summary.mean.shape != (n_states,):
                raise ContractError(
                    "PatientBootstrapUncertaintyResult A/d/e summaries must align to one shared K-state axis"
                )

        if self.uncertainty_status == "ok":
            if not has_all_summaries:
                raise ContractError(
                    "PatientBootstrapUncertaintyResult.uncertainty_status='ok' requires summaries"
                )
            if self.n_ok == 0:
                raise ContractError(
                    "PatientBootstrapUncertaintyResult.uncertainty_status='ok' requires at least one ok replicate"
                )
        elif self.n_ok > 0:
            raise ContractError(
                "Non-ok PatientBootstrapUncertaintyResult objects must not report ok replicates"
            )
        elif self.uncertainty_status == "failed" and self.n_failed == 0:
            raise ContractError(
                "PatientBootstrapUncertaintyResult.uncertainty_status='failed' requires at least one failed replicate"
            )

    @property
    def status_counts(self) -> dict[str, int]:
        """Return bootstrap replicate counts by status."""
        counts = Counter(str(status) for status in self.replicate_statuses)
        return {status: int(counts.get(status, 0)) for status in _ALLOWED_UNCERTAINTY_STATUSES}

    @property
    def n_ok(self) -> int:
        """Return the number of successful bootstrap replicates."""
        return int(self.status_counts["ok"])

    @property
    def n_deferred(self) -> int:
        """Return the number of deferred bootstrap replicates."""
        return int(self.status_counts["deferred"])

    @property
    def n_failed(self) -> int:
        """Return the number of failed bootstrap replicates."""
        return int(self.status_counts["failed"])

    @property
    def success_rate(self) -> float:
        """Return the fraction of bootstrap replicates that realized bridge outputs."""
        return float(self.n_ok / self.n_boot)


@dataclass(frozen=True)
class CohortBootstrapUncertaintySummary:
    """Cohort-level robustness summary aggregated over patient bootstrap results."""

    uncertainty_status: str
    n_patients: int
    n_eligible_patients: int
    n_realized_patients: int
    patient_status_counts: Mapping[str, int]
    mean_patient_success_rate: float
    mean_patient_A_mean_element_std: float
    mean_patient_d_mean_element_std: float
    mean_patient_e_mean_element_std: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_uncertainty_status(
            self.uncertainty_status,
            field_name="CohortBootstrapUncertaintySummary.uncertainty_status",
        )
        for field_name, value in (
            ("CohortBootstrapUncertaintySummary.n_patients", self.n_patients),
            ("CohortBootstrapUncertaintySummary.n_eligible_patients", self.n_eligible_patients),
            ("CohortBootstrapUncertaintySummary.n_realized_patients", self.n_realized_patients),
        ):
            if int(value) < 0:
                raise ContractError(f"{field_name} must be non-negative")
        if self.n_eligible_patients > self.n_patients:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.n_eligible_patients must not exceed n_patients"
            )
        if self.n_realized_patients > self.n_eligible_patients:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.n_realized_patients must not exceed n_eligible_patients"
            )
        invalid_status_keys = sorted(
            str(status)
            for status in self.patient_status_counts
            if str(status) not in _ALLOWED_UNCERTAINTY_STATUSES
        )
        if invalid_status_keys:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.patient_status_counts contains invalid statuses: "
                f"{invalid_status_keys}"
            )
        if sum(int(count) for count in self.patient_status_counts.values()) != self.n_patients:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.patient_status_counts must sum to n_patients"
            )
        for status, count in self.patient_status_counts.items():
            if int(count) < 0:
                raise ContractError(
                    f"CohortBootstrapUncertaintySummary.patient_status_counts[{status!r}] must be non-negative"
                )
        for field_name, value in (
            ("mean_patient_success_rate", self.mean_patient_success_rate),
            ("mean_patient_A_mean_element_std", self.mean_patient_A_mean_element_std),
            ("mean_patient_d_mean_element_std", self.mean_patient_d_mean_element_std),
            ("mean_patient_e_mean_element_std", self.mean_patient_e_mean_element_std),
        ):
            _validate_scalar_or_nan(float(value), field_name=field_name)
        if np.isfinite(float(self.mean_patient_success_rate)) and not (
            0.0 <= float(self.mean_patient_success_rate) <= 1.0
        ):
            raise ContractError(
                "CohortBootstrapUncertaintySummary.mean_patient_success_rate must lie in [0, 1] or be NaN"
            )
        for field_name, value in (
            ("mean_patient_A_mean_element_std", self.mean_patient_A_mean_element_std),
            ("mean_patient_d_mean_element_std", self.mean_patient_d_mean_element_std),
            ("mean_patient_e_mean_element_std", self.mean_patient_e_mean_element_std),
        ):
            if np.isfinite(float(value)) and float(value) < 0.0:
                raise ContractError(f"{field_name} must be non-negative or NaN")
        if self.uncertainty_status == "ok" and self.n_realized_patients == 0:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.uncertainty_status='ok' requires realized patients"
            )
        if self.uncertainty_status == "failed" and self.n_realized_patients != 0:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.uncertainty_status='failed' must not report realized patients"
            )
        if self.uncertainty_status == "deferred" and self.n_realized_patients != 0:
            raise ContractError(
                "CohortBootstrapUncertaintySummary.uncertainty_status='deferred' must not report realized patients"
            )


@dataclass(frozen=True)
class STRIDEBootstrapUncertaintyResult:
    """Top-level STRIDE bootstrap uncertainty surface attached to ``STRIDEFitResult``."""

    config: PatientBootstrapConfig
    patient_results: tuple[PatientBootstrapUncertaintyResult, ...]
    cohort_summary: CohortBootstrapUncertaintySummary
    uncertainty_mode: str = "patient_bridge_bootstrap_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for patient_result in self.patient_results:
            if patient_result.n_boot != self.config.n_boot:
                raise ContractError(
                    "STRIDEBootstrapUncertaintyResult.patient_results must align with config.n_boot"
                )
        patient_ids = tuple(result.patient_id for result in self.patient_results)
        if len(set(patient_ids)) != len(patient_ids):
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.patient_results must have unique patient_id values"
            )
        if len(self.patient_results) != self.cohort_summary.n_patients:
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.patient_results must align with cohort_summary.n_patients"
            )
        status_counts = Counter(result.uncertainty_status for result in self.patient_results)
        normalized_counts = {
            status: int(status_counts.get(status, 0))
            for status in _ALLOWED_UNCERTAINTY_STATUSES
        }
        summary_counts = {
            str(status): int(count)
            for status, count in dict(self.cohort_summary.patient_status_counts).items()
        }
        if summary_counts != normalized_counts:
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.cohort_summary.patient_status_counts must match patient_results"
            )
        n_eligible = sum(int(result.eligible) for result in self.patient_results)
        if n_eligible != self.cohort_summary.n_eligible_patients:
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.cohort_summary.n_eligible_patients must match patient_results"
            )
        n_realized = sum(int(result.uncertainty_status == "ok") for result in self.patient_results)
        if n_realized != self.cohort_summary.n_realized_patients:
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.cohort_summary.n_realized_patients must match patient_results"
            )
        if n_realized > 0:
            expected_status = "ok"
        elif n_eligible > 0 and summary_counts["failed"] > 0:
            expected_status = "failed"
        else:
            expected_status = "deferred"
        if self.cohort_summary.uncertainty_status != expected_status:
            raise ContractError(
                "STRIDEBootstrapUncertaintyResult.cohort_summary.uncertainty_status is incoherent "
                "with patient_results"
            )

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """Return ordered patient identifiers for attached uncertainty results."""
        return tuple(result.patient_id for result in self.patient_results)


def bootstrap_observation_measures(
    state_matrix: np.ndarray,
    *,
    config: BootstrapConfig | None = None,
) -> tuple[np.ndarray, ...]:
    """Bootstrap stacked observation-layer measures row-wise."""
    resolved_config = config or BootstrapConfig()
    matrix = np.asarray(state_matrix, dtype=float)
    rng = np.random.default_rng(resolved_config.random_state)
    return tuple(
        matrix[rng.choice(matrix.shape[0], size=matrix.shape[0], replace=True)].copy()
        for _ in range(resolved_config.n_replicates)
    )


def bootstrap_patient_relations(
    relations: Sequence[np.ndarray],
    *,
    config: BootstrapConfig | None = None,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Bootstrap a sequence of patient-level arrays by resampling patients."""
    resolved_config = config or BootstrapConfig()
    arrays = tuple(np.asarray(relation, dtype=float) for relation in relations)
    rng = np.random.default_rng(resolved_config.random_state)
    return tuple(
        tuple(arrays[idx].copy() for idx in rng.choice(len(arrays), size=len(arrays), replace=True))
        for _ in range(resolved_config.n_replicates)
    )


def summarize_stability(values: np.ndarray) -> dict[str, float]:
    """Summarize bootstrap stability with mean, sd, and coefficient of variation."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")
    cv = float(sd / mean) if np.isfinite(mean) and mean != 0.0 and np.isfinite(sd) else float("nan")
    return {"mean": mean, "sd": sd, "cv": cv}


def summarize_bootstrap_array(
    replicates: np.ndarray,
    *,
    reference: np.ndarray,
    nonzero_atol: float = 1e-12,
) -> BootstrapArraySummary:
    """Compute an auditable elementwise summary for bootstrap bridge outputs."""
    replicate_array = np.asarray(replicates, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if replicate_array.ndim < 1:
        raise ContractError("replicates must expose a leading bootstrap axis")
    if replicate_array.shape[0] == 0:
        raise ContractError("replicates must contain at least one bootstrap draw")
    if replicate_array.shape[1:] != reference_array.shape:
        raise ContractError(
            "replicates must align with reference after the leading bootstrap axis"
        )
    if not np.isfinite(replicate_array).all():
        raise ContractError("replicates must be finite")
    if not np.isfinite(reference_array).all():
        raise ContractError("reference must be finite")

    mean = np.mean(replicate_array, axis=0, dtype=float)
    std = (
        np.std(replicate_array, axis=0, ddof=1, dtype=float)
        if replicate_array.shape[0] > 1
        else np.full(reference_array.shape, np.nan, dtype=float)
    )
    nonzero_frequency = np.mean(
        np.abs(replicate_array) > float(nonzero_atol),
        axis=0,
        dtype=float,
    )
    abs_delta = np.abs(replicate_array - reference_array[None, ...])
    return BootstrapArraySummary(
        mean=mean,
        std=std,
        nonzero_frequency=nonzero_frequency,
        mean_abs_deviation=float(np.mean(abs_delta, dtype=float)),
        max_abs_deviation=float(np.max(abs_delta)),
    )


def build_cohort_bootstrap_summary(
    patient_results: Sequence[PatientBootstrapUncertaintyResult],
) -> CohortBootstrapUncertaintySummary:
    """Aggregate patient bootstrap uncertainty objects into one cohort summary."""
    results = tuple(patient_results)
    status_counts = Counter(result.uncertainty_status for result in results)
    eligible_results = [result for result in results if result.eligible]
    realized_results = [result for result in results if result.uncertainty_status == "ok"]

    if realized_results:
        uncertainty_status = "ok"
    elif eligible_results and any(result.uncertainty_status == "failed" for result in eligible_results):
        uncertainty_status = "failed"
    else:
        uncertainty_status = "deferred"

    return CohortBootstrapUncertaintySummary(
        uncertainty_status=uncertainty_status,
        n_patients=len(results),
        n_eligible_patients=len(eligible_results),
        n_realized_patients=len(realized_results),
        patient_status_counts={
            status: int(status_counts.get(status, 0))
            for status in _ALLOWED_UNCERTAINTY_STATUSES
        },
        mean_patient_success_rate=_mean_or_nan(
            [result.success_rate for result in eligible_results]
        ),
        mean_patient_A_mean_element_std=_mean_or_nan(
            [result.A_summary.mean_element_std for result in realized_results if result.A_summary is not None]
        ),
        mean_patient_d_mean_element_std=_mean_or_nan(
            [result.d_summary.mean_element_std for result in realized_results if result.d_summary is not None]
        ),
        mean_patient_e_mean_element_std=_mean_or_nan(
            [result.e_summary.mean_element_std for result in realized_results if result.e_summary is not None]
        ),
        diagnostics={
            "eligible_patient_ids": tuple(result.patient_id for result in eligible_results),
            "realized_patient_ids": tuple(result.patient_id for result in realized_results),
        },
    )


def estimate_log_measurement_error(
    theta_replicates: np.ndarray,
    delta_stabilizer: float = 1e-4,
    s2_lower_bound: float = 1e-6,
) -> tuple[float, bool]:
    """Compute empirical log-scale measurement error from bootstrap replicates."""
    arr = np.asarray(theta_replicates, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size > 0 and np.any(valid < 0.0):
        raise ContractError(
            "estimate_log_measurement_error: theta_replicates contains strictly negative values "
            "after NaN/Inf filtering"
        )
    if valid.size < 2:
        return float("nan"), False
    log_vals = np.log(valid + delta_stabilizer)
    s2 = float(np.var(log_vals, ddof=1))
    bound_applied = s2 < s2_lower_bound
    s2 = max(s2, s2_lower_bound)
    return s2, bound_applied


def bootstrap_observation_unit(
    adata: AnnData,
    observation_id: str,
    G: int,
    B_boot: int,
    observation_key: str | None = None,
) -> dict[str, Any]:
    """Generate bootstrap replicates for one FOV/ROI observation unit."""
    active_observation_key = observation_key or resolve_fov_key(adata)
    adata_unit: AnnData = adata[adata.obs[active_observation_key] == observation_id].copy()
    if adata_unit.n_obs == 0:
        raise ContractError(
            f"bootstrap_observation_unit: no cells found for {active_observation_key}={observation_id!r}"
        )

    coords = np.asarray(adata_unit.obsm["spatial"], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    col_idx = np.clip(np.floor((x - x_min) / x_range * G).astype(int), 0, G - 1)
    row_idx = np.clip(np.floor((y - y_min) / y_range * G).astype(int), 0, G - 1)

    block_ids = np.array([f"b{r}_{c}" for r, c in zip(row_idx, col_idx)])
    adata_unit.obs["block_id"] = block_ids

    unique_blocks = np.unique(block_ids)
    n_blocks_valid = int(unique_blocks.size)

    block_to_cell_indices: dict[str, np.ndarray] = {}
    for block_id in unique_blocks:
        block_to_cell_indices[block_id] = np.where(block_ids == block_id)[0]

    rng = np.random.default_rng()
    replicates: list[AnnData] = []
    for _ in range(B_boot):
        sampled_block_ids = rng.choice(unique_blocks, size=n_blocks_valid, replace=True)
        cell_indices: list[int] = []
        for block_id in sampled_block_ids:
            cell_indices.extend(block_to_cell_indices[block_id].tolist())
        replicates.append(adata_unit[cell_indices].copy())

    return {
        "replicates": replicates,
        "uncertainty_mode": "grid_block_frozen",
        "UQ_mode": "grid_block_frozen",
        "n_blocks_valid": n_blocks_valid,
    }


__all__ = [
    "BootstrapArraySummary",
    "BootstrapConfig",
    "CohortBootstrapUncertaintySummary",
    "PatientBootstrapConfig",
    "PatientBootstrapUncertaintyResult",
    "STRIDEBootstrapUncertaintyResult",
    "bootstrap_observation_measures",
    "bootstrap_observation_unit",
    "bootstrap_patient_relations",
    "build_cohort_bootstrap_summary",
    "estimate_log_measurement_error",
    "summarize_bootstrap_array",
    "summarize_stability",
]
