"""Cohort-level recurrence contracts built from STRIDE patient relations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ContractError
from .operators import PatientRelation, validate_patient_relation

_ALLOWED_RECURRENCE_STATUSES = frozenset({"ok", "deferred", "failed"})


def _require_recurrence_status(value: object, *, field_name: str) -> str:
    status = str(value)
    if status not in _ALLOWED_RECURRENCE_STATUSES:
        raise ContractError(
            f"{field_name} must be one of {tuple(sorted(_ALLOWED_RECURRENCE_STATUSES))}, "
            f"got {status!r}"
        )
    return status


def _normalize_patient_ids(patient_ids: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(patient_id).strip() for patient_id in patient_ids)
    if any(patient_id == "" for patient_id in normalized):
        raise ContractError(f"{field_name} must contain non-empty patient identifiers")
    if len(set(normalized)) != len(normalized):
        raise ContractError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class RecurrenceParameters:
    """Shared low-dimensional recurrence parameters across patients."""

    basis_dim: int
    loadings: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRecurrenceEmbedding:
    """Low-dimensional embedding for one patient relation."""

    patient_id: str
    coordinates: np.ndarray
    fit_status: str = "ok"

    def __post_init__(self) -> None:
        patient_id = str(self.patient_id).strip()
        if patient_id == "":
            raise ContractError("PatientRecurrenceEmbedding.patient_id must be non-empty")
        status = _require_recurrence_status(
            self.fit_status,
            field_name="PatientRecurrenceEmbedding.fit_status",
        )
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 1:
            raise ContractError("PatientRecurrenceEmbedding.coordinates must be a 1D array")
        if status == "ok" and not np.isfinite(coordinates).all():
            raise ContractError("ok PatientRecurrenceEmbedding.coordinates must be finite")
        object.__setattr__(self, "patient_id", patient_id)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "fit_status", status)


@dataclass(frozen=True)
class RecurrenceFamily:
    """One recurrence-family summary on the shared state axis."""

    family_id: str
    template_A: np.ndarray
    template_d: np.ndarray
    template_e: np.ndarray
    support_n_patients: int
    within_family_dispersion: float | None = None
    fit_status: str = "ok"
    member_patient_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        family_id = str(self.family_id).strip()
        if family_id == "":
            raise ContractError("RecurrenceFamily.family_id must be non-empty")
        status = _require_recurrence_status(
            self.fit_status,
            field_name="RecurrenceFamily.fit_status",
        )
        if (
            isinstance(self.support_n_patients, bool)
            or not isinstance(self.support_n_patients, int)
            or int(self.support_n_patients) < 0
        ):
            raise ContractError("RecurrenceFamily.support_n_patients must be a non-negative int")
        member_patient_ids = _normalize_patient_ids(
            self.member_patient_ids,
            field_name="RecurrenceFamily.member_patient_ids",
        )
        if int(self.support_n_patients) != len(member_patient_ids):
            raise ContractError(
                "RecurrenceFamily.support_n_patients must match member_patient_ids length"
            )
        validate_patient_relation(A=self.template_A, d=self.template_d, e=self.template_e)
        if self.within_family_dispersion is not None:
            dispersion = float(self.within_family_dispersion)
            if not np.isfinite(dispersion) or dispersion < 0.0:
                raise ContractError(
                    "RecurrenceFamily.within_family_dispersion must be finite and non-negative"
                )
            object.__setattr__(self, "within_family_dispersion", dispersion)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "fit_status", status)
        object.__setattr__(self, "member_patient_ids", member_patient_ids)


@dataclass(frozen=True)
class RecurrenceResult:
    """Container for cohort-level recurrence outputs.

    The result holds the cohort members that were analyzed plus any learned
    recurrence families, low-dimensional embeddings, and fit metadata.
    """

    patient_ids: tuple[str, ...]
    families: tuple[RecurrenceFamily, ...]
    fit_status: str
    used_patient_ids: tuple[str, ...] = ()
    recurrence_unit: str = "patient"
    parameters: RecurrenceParameters | None = None
    embeddings: tuple[PatientRecurrenceEmbedding, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patient_ids = _normalize_patient_ids(
            self.patient_ids,
            field_name="RecurrenceResult.patient_ids",
        )
        used_patient_ids = _normalize_patient_ids(
            self.used_patient_ids,
            field_name="RecurrenceResult.used_patient_ids",
        )
        invalid_used = tuple(patient_id for patient_id in used_patient_ids if patient_id not in patient_ids)
        if invalid_used:
            raise ContractError("RecurrenceResult.used_patient_ids must be a subset of patient_ids")
        status = _require_recurrence_status(
            self.fit_status,
            field_name="RecurrenceResult.fit_status",
        )
        families = tuple(self.families)
        for family in families:
            if not isinstance(family, RecurrenceFamily):
                raise ContractError("RecurrenceResult.families must contain RecurrenceFamily objects")
            invalid_members = tuple(
                patient_id for patient_id in family.member_patient_ids if patient_id not in patient_ids
            )
            if invalid_members:
                raise ContractError(
                    "RecurrenceFamily.member_patient_ids must be a subset of RecurrenceResult.patient_ids"
                )
            if status == "ok" and family.fit_status != "ok":
                raise ContractError("ok RecurrenceResult must not contain non-ok families")
            if status != "ok" and family.fit_status == "ok":
                raise ContractError("non-ok RecurrenceResult must not contain ok families")
        embeddings = tuple(self.embeddings)
        if embeddings:
            embedding_ids = tuple(embedding.patient_id for embedding in embeddings)
            if embedding_ids != patient_ids:
                raise ContractError("RecurrenceResult.embeddings must align with patient_ids")
            if status == "ok" and any(embedding.fit_status != "ok" for embedding in embeddings):
                raise ContractError("ok RecurrenceResult must carry ok embeddings when provided")
            if status != "ok" and any(embedding.fit_status == "ok" for embedding in embeddings):
                raise ContractError("non-ok RecurrenceResult must not carry ok embeddings")
        object.__setattr__(self, "patient_ids", patient_ids)
        object.__setattr__(self, "used_patient_ids", used_patient_ids)
        object.__setattr__(self, "fit_status", status)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "embeddings", embeddings)


@dataclass(frozen=True)
class RecurrenceConfig:
    """Configuration for the current cohort-level recurrence interface."""

    recurrence_unit: str = "patient"
    min_support_n_patients: int = 1
    basis_dim: int = 2
    mode: str = "template_mean_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _vectorize_relation_arrays(
    A: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    A_arr = np.asarray(A, dtype=float)
    d_arr = np.asarray(d, dtype=float).reshape(-1)
    e_arr = np.asarray(e, dtype=float).reshape(-1)
    return np.concatenate([A_arr.reshape(-1), d_arr, e_arr]).astype(float, copy=False)


def vectorize_patient_relation(relation: PatientRelation) -> np.ndarray:
    """Vectorize one patient relation for cohort-level recurrence comparisons."""
    return _vectorize_relation_arrays(relation.A, relation.d, relation.e)


def _deferred_embeddings(
    patient_ids: Sequence[str],
    *,
    basis_dim: int,
) -> tuple[PatientRecurrenceEmbedding, ...]:
    return tuple(
        PatientRecurrenceEmbedding(
            patient_id=str(patient_id),
            coordinates=np.full(int(basis_dim), np.nan, dtype=float),
            fit_status="deferred",
        )
        for patient_id in patient_ids
    )


def _compute_embeddings(
    patient_ids: Sequence[str],
    relation_matrix: np.ndarray,
    *,
    basis_dim: int,
) -> tuple[tuple[PatientRecurrenceEmbedding, ...], np.ndarray]:
    n_patients, n_features = relation_matrix.shape
    resolved_basis_dim = max(1, int(basis_dim))
    coordinates = np.zeros((n_patients, resolved_basis_dim), dtype=float)
    loadings = np.zeros((resolved_basis_dim, n_features), dtype=float)
    centered = relation_matrix - np.mean(relation_matrix, axis=0, dtype=float)
    if n_patients >= 2 and np.any(np.abs(centered) > 1e-12):
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        active_dim = min(resolved_basis_dim, singular_values.shape[0])
        coordinates[:, :active_dim] = U[:, :active_dim] * singular_values[:active_dim]
        loadings[:active_dim, :] = Vt[:active_dim, :]
    embeddings = tuple(
        PatientRecurrenceEmbedding(
            patient_id=str(patient_id),
            coordinates=coordinates[idx],
            fit_status="ok",
        )
        for idx, patient_id in enumerate(patient_ids)
    )
    return embeddings, loadings


def _project_template_rows(
    template_A: np.ndarray,
    template_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    A_arr = np.asarray(template_A, dtype=float).copy()
    d_arr = np.asarray(template_d, dtype=float).reshape(-1).copy()
    row_totals = np.sum(A_arr, axis=1, dtype=float) + d_arr
    for row_idx, row_total in enumerate(row_totals):
        if row_total <= 0.0:
            A_arr[row_idx, :] = 0.0
            d_arr[row_idx] = 1.0
            continue
        if not np.isclose(row_total, 1.0, rtol=0.0, atol=1e-8):
            A_arr[row_idx, :] /= row_total
            d_arr[row_idx] /= row_total
    return A_arr, d_arr


def validate_recurrence_inputs(relations: Sequence[PatientRelation]) -> None:
    """Validate model-layer patient relations before recurrence estimation."""
    if len(relations) == 0:
        raise ContractError("relations must contain at least one patient relation")
    for relation in relations:
        validate_patient_relation(
            A=relation.A,
            d=relation.d,
            e=relation.e,
            mu_minus=relation.mu_minus,
            mu_plus=relation.mu_plus,
            state_ids=relation.state_ids,
        )


def build_recurrence_result(
    patient_ids: Sequence[str],
    families: Sequence[RecurrenceFamily] = (),
    *,
    fit_status: str = "ok",
    used_patient_ids: Sequence[str] | None = None,
    recurrence_unit: str = "patient",
    parameters: RecurrenceParameters | None = None,
    embeddings: Sequence[PatientRecurrenceEmbedding] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RecurrenceResult:
    """Build a recurrence result from already-assembled family-level objects."""
    for family in families:
        validate_patient_relation(A=family.template_A, d=family.template_d, e=family.template_e)
    return RecurrenceResult(
        patient_ids=tuple(str(patient_id) for patient_id in patient_ids),
        families=tuple(families),
        fit_status=str(fit_status),
        used_patient_ids=tuple(
            str(patient_id) for patient_id in (used_patient_ids if used_patient_ids is not None else patient_ids)
        ),
        recurrence_unit=str(recurrence_unit),
        parameters=parameters,
        embeddings=tuple(embeddings),
        metadata=dict(metadata or {}),
    )


def summarize_recurrence_support(result: RecurrenceResult) -> dict[str, int]:
    """Summarize support size by recurrence family."""
    return {family.family_id: int(family.support_n_patients) for family in result.families}


def build_deferred_recurrence_result(
    patient_ids: Sequence[str],
    *,
    used_patient_ids: Sequence[str] = (),
    config: RecurrenceConfig | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RecurrenceResult:
    """Build an explicit deferred recurrence payload for the canonical namespace."""
    resolved_config = config or RecurrenceConfig(mode="deferred")
    resolved_patient_ids = tuple(str(patient_id) for patient_id in patient_ids)
    resolved_used_ids = tuple(str(patient_id) for patient_id in used_patient_ids)
    return RecurrenceResult(
        patient_ids=resolved_patient_ids,
        families=(),
        fit_status="deferred",
        used_patient_ids=resolved_used_ids,
        recurrence_unit=resolved_config.recurrence_unit,
        parameters=RecurrenceParameters(
            basis_dim=resolved_config.basis_dim,
            loadings=None,
            metadata={"mode": resolved_config.mode},
        ),
        embeddings=_deferred_embeddings(
            resolved_patient_ids,
            basis_dim=resolved_config.basis_dim,
        ),
        metadata={
            "mode": resolved_config.mode,
            "message": (
                message
                or "Canonical cohort-level recurrence estimation remains deferred."
            ),
            **dict(resolved_config.metadata),
            **dict(metadata or {}),
        },
    )


def estimate_recurrence(
    relations: Sequence[PatientRelation],
    config: RecurrenceConfig | None = None,
) -> RecurrenceResult:
    """Estimate a conservative first-pass cohort template over patient relations."""
    validate_recurrence_inputs(relations)
    resolved_config = config or RecurrenceConfig()
    patient_ids = tuple(relation.patient_id for relation in relations)
    required_support = max(2, int(resolved_config.min_support_n_patients))
    if len(relations) < required_support:
        return build_deferred_recurrence_result(
            patient_ids,
            used_patient_ids=patient_ids,
            config=resolved_config,
            message=(
                "Canonical cohort-level recurrence estimation remains deferred because it "
                f"requires at least {required_support} realized patients on the shared state axis."
            ),
        )

    relation_matrix = np.vstack([vectorize_patient_relation(relation) for relation in relations]).astype(
        float,
        copy=False,
    )
    template_A = np.mean(
        np.stack([np.asarray(relation.A, dtype=float) for relation in relations], axis=0),
        axis=0,
        dtype=float,
    )
    template_d = np.mean(
        np.stack([np.asarray(relation.d, dtype=float) for relation in relations], axis=0),
        axis=0,
        dtype=float,
    )
    template_e = np.mean(
        np.stack([np.asarray(relation.e, dtype=float) for relation in relations], axis=0),
        axis=0,
        dtype=float,
    )
    template_A, template_d = _project_template_rows(template_A, template_d)
    validate_patient_relation(A=template_A, d=template_d, e=template_e)

    embeddings, loadings = _compute_embeddings(
        patient_ids,
        relation_matrix,
        basis_dim=resolved_config.basis_dim,
    )
    centered = relation_matrix - np.mean(relation_matrix, axis=0, dtype=float)
    within_family_dispersion = float(
        np.mean(np.linalg.norm(centered, axis=1), dtype=float)
    )
    family = RecurrenceFamily(
        family_id="family_0",
        template_A=template_A,
        template_d=template_d,
        template_e=template_e,
        support_n_patients=len(relations),
        within_family_dispersion=within_family_dispersion,
        fit_status="ok",
        member_patient_ids=patient_ids,
    )
    return RecurrenceResult(
        patient_ids=patient_ids,
        families=(family,),
        fit_status="ok",
        used_patient_ids=patient_ids,
        recurrence_unit=resolved_config.recurrence_unit,
        parameters=RecurrenceParameters(
            basis_dim=resolved_config.basis_dim,
            loadings=loadings,
            metadata={
                "mode": resolved_config.mode,
                "feature_vector_size": int(relation_matrix.shape[1]),
            },
        ),
        embeddings=embeddings,
        metadata={
            "mode": resolved_config.mode,
            "message": (
                "Estimated a conservative first-pass cohort template over realized "
                "patient relations on the shared state axis."
            ),
            "n_used_patients": len(patient_ids),
            **dict(resolved_config.metadata),
        },
    )


__all__ = [
    "PatientRecurrenceEmbedding",
    "RecurrenceConfig",
    "RecurrenceFamily",
    "RecurrenceParameters",
    "RecurrenceResult",
    "build_deferred_recurrence_result",
    "build_recurrence_result",
    "estimate_recurrence",
    "summarize_recurrence_support",
    "validate_recurrence_inputs",
    "vectorize_patient_relation",
]
