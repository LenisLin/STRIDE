"""Task A mapping and dry-run summary dataclasses.

These payloads define the task-local JSON and CSV schemas for the Step 3
canonical rerun surfaces. They cover Step 1 mapping, canonical ``fit_stride``
dry-run provenance, and the recurrence-context fields that Block 1 now
require, without redefining the STRIDE core outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskAStage0FieldMapping:
    patient_id_key: str
    timepoint_key: str
    fov_key: str
    domain_key: str
    cell_subtype_key: str
    state_id_key: str
    state_ids: tuple[int, ...]
    n_states: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "patient_id_key": self.patient_id_key,
            "timepoint_key": self.timepoint_key,
            "fov_key": self.fov_key,
            "domain_key": self.domain_key,
            "cell_subtype_key": self.cell_subtype_key,
            "state_id_key": self.state_id_key,
            "state_ids": list(self.state_ids),
            "n_states": self.n_states,
        }


@dataclass(frozen=True)
class TaskARealDataCrosswalk:
    """Records the exact real-data crosswalk between raw Stage 0 field names
    and stable STRIDE canonical keys, plus derived observation-layer semantics.

    Each entry records ``raw_key -> canonical_key`` with a ``mapping_type``
    of ``"direct"`` (identity), ``"alias"`` (accepted migration-era alias),
    or ``"derived"`` (computed at adapter time from an existing field).
    """

    # --- obs-layer field crosswalk ---
    patient_id_raw: str = "patient_id"
    patient_id_canonical: str = "patient_id"
    patient_id_mapping: str = "direct"

    fov_raw: str = "roi_id"
    fov_canonical: str = "fov_id"
    fov_mapping: str = "alias"

    domain_raw: str = "compartment"
    domain_canonical: str = "domain_label"
    domain_mapping: str = "alias"

    ordered_group_source: str = "compartment"
    ordered_group_mapping: str = "derived"
    ordered_group_note: str = (
        "FovObservation.timepoint is set from compartment-derived ordered "
        "groups for two-group family slicing; raw timepoint is inert metadata"
    )

    timepoint_raw: str = "timepoint"
    timepoint_raw_observed_values: tuple[str, ...] = ("0",)
    timepoint_inert: bool = True
    timepoint_note: str = (
        "Raw timepoint carries only a single inert value in the current "
        "cohort; it is retained as Stage 0 metadata but not used for "
        "ordered-group derivation"
    )

    cell_subtype_raw: str = "cell_type"
    cell_subtype_canonical: str = "cell_subtype_label"
    cell_subtype_mapping: str = "alias"

    state_id_raw: str = "proto_id"
    state_id_canonical: str = "state_id"
    state_id_mapping: str = "alias"

    # --- obsm / uns-layer crosswalk ---
    spatial_key: str = "spatial"
    spatial_mapping: str = "direct"

    feature_raw: str = "community_features"
    feature_canonical: str = "local_state_features"
    feature_mapping: str = "alias"

    centroids_raw: str = "prototype_centroids"
    centroids_canonical: str = "state_centroids"
    centroids_mapping: str = "alias"

    cost_scale_raw: str = "s_C"
    cost_scale_canonical: str = "cost_scale"
    cost_scale_mapping: str = "alias"

    cost_matrix_key: str = "cost_matrix"
    cost_matrix_mapping: str = "direct"

    # --- derived observation-layer semantics ---
    mass_value: float = 1.0
    mass_mode: str = "uniform"
    mass_note: str = "Observation-layer mass is uniform at adapter time"

    # --- explicitly unmapped / deferred surfaces ---
    unmapped_obs_fields: tuple[str, ...] = ("block_id", "cell_area")
    deferred_downstream_fields: tuple[str, ...] = (
        "comparison_id",
        "count_stratum_key",
        "real_fit_status",
        "null_fit_status",
    )

    def to_json_dict(self) -> dict[str, Any]:
        entries: list[dict[str, str | float | bool | list[str]]] = [
            {"raw": self.patient_id_raw, "canonical": self.patient_id_canonical, "type": self.patient_id_mapping, "layer": "obs"},
            {"raw": self.fov_raw, "canonical": self.fov_canonical, "type": self.fov_mapping, "layer": "obs"},
            {"raw": self.domain_raw, "canonical": self.domain_canonical, "type": self.domain_mapping, "layer": "obs"},
            {"raw": self.ordered_group_source, "canonical": "FovObservation.timepoint", "type": self.ordered_group_mapping, "layer": "obs", "note": self.ordered_group_note},
            {"raw": self.timepoint_raw, "canonical": "timepoint", "type": "inert_metadata", "layer": "obs", "observed_values": list(self.timepoint_raw_observed_values), "note": self.timepoint_note},
            {"raw": self.cell_subtype_raw, "canonical": self.cell_subtype_canonical, "type": self.cell_subtype_mapping, "layer": "obs"},
            {"raw": self.state_id_raw, "canonical": self.state_id_canonical, "type": self.state_id_mapping, "layer": "obs"},
            {"raw": self.spatial_key, "canonical": self.spatial_key, "type": self.spatial_mapping, "layer": "obsm"},
            {"raw": self.feature_raw, "canonical": self.feature_canonical, "type": self.feature_mapping, "layer": "obsm"},
            {"raw": self.centroids_raw, "canonical": self.centroids_canonical, "type": self.centroids_mapping, "layer": "uns"},
            {"raw": self.cost_scale_raw, "canonical": self.cost_scale_canonical, "type": self.cost_scale_mapping, "layer": "uns"},
            {"raw": self.cost_matrix_key, "canonical": self.cost_matrix_key, "type": self.cost_matrix_mapping, "layer": "uns"},
        ]
        return {
            "crosswalk": entries,
            "derived_semantics": {
                "mass": self.mass_value,
                "mass_mode": self.mass_mode,
                "note": self.mass_note,
            },
            "unmapped_obs_fields": list(self.unmapped_obs_fields),
            "deferred_downstream_fields": list(self.deferred_downstream_fields),
        }


@dataclass(frozen=True)
class TaskAFamilyStrideMappingSummary:
    pair_family: str
    source_domain: str
    target_domain: str
    claim_role: str
    pair_types: tuple[str, ...]
    ordered_group_labels: tuple[str, str]
    eligible_patients: tuple[str, ...]
    skipped_patients: tuple[str, ...]
    n_observations: int
    n_source_observations: int
    n_target_observations: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pair_family": self.pair_family,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "claim_role": self.claim_role,
            "pair_types": list(self.pair_types),
            "ordered_group_labels": list(self.ordered_group_labels),
            "eligible_patients": list(self.eligible_patients),
            "skipped_patients": list(self.skipped_patients),
            "n_observations": self.n_observations,
            "n_source_observations": self.n_source_observations,
            "n_target_observations": self.n_target_observations,
        }


@dataclass(frozen=True)
class TaskAStage0StrideMappingSummary:
    field_mapping: TaskAStage0FieldMapping
    patient_ids: tuple[str, ...]
    family_summaries: tuple[TaskAFamilyStrideMappingSummary, ...]
    real_data_crosswalk: TaskARealDataCrosswalk | None = None

    def to_json_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "field_mapping": self.field_mapping.to_json_dict(),
            "patient_ids": list(self.patient_ids),
            "family_summaries": [summary.to_json_dict() for summary in self.family_summaries],
        }
        if self.real_data_crosswalk is not None:
            result["real_data_crosswalk"] = self.real_data_crosswalk.to_json_dict()
        return result


@dataclass(frozen=True)
class TaskACoreFitDryRunRecord:
    """One patient-level row from a canonical ``fit_stride`` Task A dry run."""

    pair_family: str
    claim_role: str
    patient_id: str
    implementation_tier: str
    fit_surface: str
    fit_status: str
    bridge_realized: bool
    defer_reason: str | None
    uncertainty_status: str | None
    cohort_recurrence_fit_status: str
    n_recurrence_families: int
    n_recurrence_used_patients: int
    source_domain: str
    target_domain: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pair_family": self.pair_family,
            "claim_role": self.claim_role,
            "patient_id": self.patient_id,
            "implementation_tier": self.implementation_tier,
            "fit_surface": self.fit_surface,
            "fit_status": self.fit_status,
            "bridge_realized": self.bridge_realized,
            "defer_reason": self.defer_reason,
            "uncertainty_status": self.uncertainty_status,
            "cohort_recurrence_fit_status": self.cohort_recurrence_fit_status,
            "n_recurrence_families": self.n_recurrence_families,
            "n_recurrence_used_patients": self.n_recurrence_used_patients,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
        }


__all__ = [
    "TaskACoreFitDryRunRecord",
    "TaskAFamilyStrideMappingSummary",
    "TaskARealDataCrosswalk",
    "TaskAStage0FieldMapping",
    "TaskAStage0StrideMappingSummary",
]
