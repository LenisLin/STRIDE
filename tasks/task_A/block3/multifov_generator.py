"""Multi-FOV train-template generator for Task A Block 3.

This module is implementation-local. It builds the shared hidden patient truth
and generated source/target FOV observations used by the live Block 3 3B/3C
surface. Hidden template identities and truth arrays are returned only to the
execution/scoring layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL = 1e-12


@dataclass(frozen=True)
class RawTemplate:
    patient_id: str
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    d: np.ndarray
    v: np.ndarray
    identified_rows: np.ndarray


@dataclass(frozen=True)
class Template:
    patient_id: str
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    d: np.ndarray
    s: np.ndarray
    row_imputed_mask: np.ndarray
    open_mass_train: float

    @property
    def B(self) -> np.ndarray:
        return np.concatenate([self.A, self.d[:, None]], axis=1)


@dataclass(frozen=True)
class PatientTruth:
    patient_id: str
    sampled_template_patient_id: str
    medoid_template_patient_id: str
    x: np.ndarray
    A: np.ndarray
    d: np.ndarray
    e: np.ndarray
    y_truth: np.ndarray
    s_true: np.ndarray
    row_imputed_mask: np.ndarray


@dataclass(frozen=True)
class GeneratedPatient:
    truth: PatientTruth
    source_fovs: np.ndarray
    target_fovs: np.ndarray
    endpoint_x_obs: np.ndarray
    endpoint_y_obs: np.ndarray
    endpoint_closure_l1: float


def normalize_simplex(values: np.ndarray, *, eps: float = TOL) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    total = float(np.sum(vector, dtype=float))
    if not np.isfinite(total) or total <= eps:
        raise ValueError("simplex vector must have positive finite mass")
    return vector / total


def sample_split(
    patient_ids: tuple[str, ...],
    *,
    n_test: int = 8,
    seed: int = 17,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if n_test <= 0 or n_test >= len(patient_ids):
        raise ValueError("n_test must be positive and smaller than patient count")
    rng = np.random.default_rng(seed)
    shuffled = tuple(str(item) for item in rng.permutation(tuple(patient_ids)).tolist())
    test = tuple(sorted(shuffled[:n_test]))
    train = tuple(sorted(patient_id for patient_id in patient_ids if patient_id not in set(test)))
    return train, test


def _clip_tiny_negative(values: np.ndarray, *, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(arr < -eps):
        raise ValueError("generator produced negative residual mass beyond tolerance")
    return np.where(arr < 0.0, 0.0, arr)


def build_raw_template_from_train_pair(
    *,
    patient_id: str,
    x: np.ndarray,
    y: np.ndarray,
    C_norm: np.ndarray,
    tau: float = 2.0,
    eps: float = TOL,
) -> RawTemplate:
    x_vec = normalize_simplex(x, eps=eps)
    y_vec = normalize_simplex(y, eps=eps)
    K = x_vec.size
    if C_norm.shape != (K, K):
        raise ValueError("C_norm shape does not match train endpoint dimension")

    r_minus = np.maximum(x_vec - y_vec, 0.0)
    r_plus = np.maximum(y_vec - x_vec, 0.0)
    rho = float(np.sum(r_minus, dtype=float))
    A = np.eye(K, dtype=float)
    d = np.zeros(K, dtype=float)
    v = np.zeros(K, dtype=float)
    identified = x_vec > eps
    if rho <= eps:
        return RawTemplate(
            patient_id=patient_id,
            x=x_vec,
            y=y_vec,
            A=A,
            d=d,
            v=v,
            identified_rows=identified,
        )

    R0 = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i != j:
                R0[i, j] = r_minus[i] * r_plus[j] / rho
    gate = np.exp(-np.asarray(C_norm, dtype=float) / float(tau))
    np.fill_diagonal(gate, 0.0)
    R = gate * R0
    u = _clip_tiny_negative(r_minus - np.sum(R, axis=1, dtype=float), eps=eps)
    v = _clip_tiny_negative(r_plus - np.sum(R, axis=0, dtype=float), eps=eps)

    T = R.copy()
    for i in range(K):
        T[i, i] = min(float(x_vec[i]), float(y_vec[i]))

    A = np.zeros((K, K), dtype=float)
    d = np.zeros(K, dtype=float)
    for i in range(K):
        if identified[i]:
            A[i, :] = T[i, :] / x_vec[i]
            d[i] = u[i] / x_vec[i]
    return RawTemplate(
        patient_id=patient_id,
        x=x_vec,
        y=y_vec,
        A=A,
        d=d,
        v=v,
        identified_rows=identified,
    )


def _row_pool(raw_templates: list[RawTemplate], *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if not raw_templates:
        raise ValueError("template bank requires at least one train template")
    K = raw_templates[0].x.size
    pooled_A = np.eye(K, dtype=float)
    pooled_d = np.zeros(K, dtype=float)
    for i in range(K):
        numerator_A = np.zeros(K, dtype=float)
        numerator_d = 0.0
        denominator = 0.0
        for template in raw_templates:
            if bool(template.identified_rows[i]):
                weight = float(template.x[i])
                numerator_A += weight * template.A[i, :]
                numerator_d += weight * float(template.d[i])
                denominator += weight
        if denominator > eps:
            pooled_A[i, :] = numerator_A / denominator
            pooled_d[i] = numerator_d / denominator
            row_total = float(np.sum(pooled_A[i, :], dtype=float) + pooled_d[i])
            if row_total > eps:
                pooled_A[i, :] /= row_total
                pooled_d[i] /= row_total
    return pooled_A, pooled_d


def build_template_bank(
    *,
    train_patient_ids: tuple[str, ...],
    source_endpoints: dict[str, np.ndarray],
    target_endpoints: dict[str, np.ndarray],
    C_norm: np.ndarray,
    tau: float = 2.0,
    eps: float = TOL,
) -> tuple[Template, ...]:
    raw_templates = [
        build_raw_template_from_train_pair(
            patient_id=patient_id,
            x=source_endpoints[patient_id],
            y=target_endpoints[patient_id],
            C_norm=C_norm,
            tau=tau,
            eps=eps,
        )
        for patient_id in train_patient_ids
    ]
    pooled_A, pooled_d = _row_pool(raw_templates, eps=eps)
    total_v = np.sum(np.vstack([raw.v for raw in raw_templates]), axis=0, dtype=float)
    emergence_fallback = (
        normalize_simplex(total_v, eps=eps)
        if float(np.sum(total_v, dtype=float)) > eps
        else np.full(raw_templates[0].x.size, 1.0 / raw_templates[0].x.size)
    )

    templates: list[Template] = []
    for raw in raw_templates:
        A = np.asarray(raw.A, dtype=float).copy()
        d = np.asarray(raw.d, dtype=float).copy()
        row_imputed = ~raw.identified_rows
        for row_index, was_missing in enumerate(row_imputed):
            if bool(was_missing):
                A[row_index, :] = pooled_A[row_index, :]
                d[row_index] = pooled_d[row_index]
        v_mass = float(np.sum(raw.v, dtype=float))
        s = normalize_simplex(raw.v, eps=eps) if v_mass > eps else emergence_fallback.copy()
        templates.append(
            Template(
                patient_id=raw.patient_id,
                x=raw.x,
                y=raw.y,
                A=A,
                d=d,
                s=s,
                row_imputed_mask=row_imputed.astype(bool),
                open_mass_train=float(np.sum(raw.x * d, dtype=float)),
            )
        )
    return tuple(templates)


def select_template_medoid(
    templates: tuple[Template, ...],
    *,
    alpha_s: float = 1.0,
) -> Template:
    if not templates:
        raise ValueError("cannot select medoid from an empty template bank")
    distances: list[float] = []
    for left in templates:
        total = 0.0
        for right in templates:
            total += float(np.sum(np.abs(left.B - right.B), dtype=float))
            total += float(alpha_s) * float(np.sum(np.abs(left.s - right.s), dtype=float))
        distances.append(total)
    return templates[int(np.argmin(np.asarray(distances, dtype=float)))]


def synthesize_patient_truth(
    *,
    patient_id: str,
    test_x: np.ndarray,
    template_bank: tuple[Template, ...],
    medoid: Template,
    rng: np.random.Generator,
    lambda_individual: float = 0.10,
    eps: float = TOL,
) -> PatientTruth:
    x = normalize_simplex(test_x, eps=eps)
    if not 0.0 <= float(lambda_individual) <= 1.0:
        raise ValueError("lambda_individual must lie in [0, 1]")
    individual = template_bank[int(rng.integers(0, len(template_bank)))]
    lam = float(lambda_individual)
    B = (1.0 - lam) * medoid.B + lam * individual.B
    A = np.asarray(B[:, :-1], dtype=float)
    d = np.asarray(B[:, -1], dtype=float)
    s_true = normalize_simplex((1.0 - lam) * medoid.s + lam * individual.s, eps=eps)
    open_mass = float(np.sum(x * d, dtype=float))
    e = open_mass * s_true
    y_truth = normalize_simplex(x @ A + e, eps=eps)
    return PatientTruth(
        patient_id=patient_id,
        sampled_template_patient_id=individual.patient_id,
        medoid_template_patient_id=medoid.patient_id,
        x=x,
        A=A,
        d=d,
        e=e,
        y_truth=y_truth,
        s_true=s_true,
        row_imputed_mask=(medoid.row_imputed_mask | individual.row_imputed_mask),
    )


def generate_patient_multifov(
    *,
    truth: PatientTruth,
    real_source_fovs: np.ndarray,
    n_target_fovs: int,
    eta: float = 0.3,
    eps: float = TOL,
) -> GeneratedPatient:
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    source_real = np.asarray(real_source_fovs, dtype=float)
    if source_real.ndim != 2 or source_real.shape[0] == 0:
        raise ValueError("real_source_fovs must be a non-empty [n_fov, K] matrix")
    if n_target_fovs <= 0:
        raise ValueError("n_target_fovs must be positive")
    source_fovs = np.vstack(
        [
            normalize_simplex((1.0 - float(eta)) * truth.x + float(eta) * row, eps=eps)
            for row in source_real
        ]
    )
    target_rows = []
    for index in range(n_target_fovs):
        anchor = source_fovs[index % source_fovs.shape[0]]
        target_rows.append(normalize_simplex(anchor @ truth.A + truth.e, eps=eps))
    target_fovs = np.vstack(target_rows)
    endpoint_x = normalize_simplex(np.mean(source_fovs, axis=0, dtype=float), eps=eps)
    endpoint_y = normalize_simplex(np.mean(target_fovs, axis=0, dtype=float), eps=eps)
    closure = float(np.sum(np.abs(endpoint_y - truth.y_truth), dtype=float))
    return GeneratedPatient(
        truth=truth,
        source_fovs=source_fovs,
        target_fovs=target_fovs,
        endpoint_x_obs=endpoint_x,
        endpoint_y_obs=endpoint_y,
        endpoint_closure_l1=closure,
    )


def compute_patient_diagnostics(
    *,
    generated: GeneratedPatient,
    C_norm: np.ndarray,
) -> dict[str, object]:
    truth = generated.truth
    x = np.asarray(truth.x, dtype=float)
    A = np.asarray(truth.A, dtype=float)
    d = np.asarray(truth.d, dtype=float)
    e = np.asarray(truth.e, dtype=float)
    F = x[:, None] * A
    offdiag = ~np.eye(A.shape[0], dtype=bool)
    retained = float(np.sum(np.diag(F), dtype=float))
    offdiag_mass = float(np.sum(F[offdiag], dtype=float))
    open_mass = float(np.sum(x * d, dtype=float))
    offdiag_cost_numerator = float(
        np.sum(F[offdiag] * np.asarray(C_norm, dtype=float)[offdiag], dtype=float)
    )
    locality = None if offdiag_mass <= 0.0 else offdiag_cost_numerator / offdiag_mass
    n_expected_blocks = min(int(generated.source_fovs.shape[0]), int(generated.target_fovs.shape[0]))
    row_accounting = np.sum(A, axis=1, dtype=float) + d
    ordering = (
        "retained_gt_offdiag_gt_open"
        if retained > offdiag_mass > open_mass
        else "outside_target_order"
    )
    return {
        "patient_id": truth.patient_id,
        "sampled_template_patient_id": truth.sampled_template_patient_id,
        "medoid_template_patient_id": truth.medoid_template_patient_id,
        "row_accounting_max_abs_error": float(np.max(np.abs(row_accounting - 1.0))),
        "truth_y_simplex_error": float(abs(float(np.sum(truth.y_truth, dtype=float)) - 1.0)),
        "source_fov_simplex_max_error": float(
            np.max(np.abs(np.sum(generated.source_fovs, axis=1, dtype=float) - 1.0))
        ),
        "target_fov_simplex_max_error": float(
            np.max(np.abs(np.sum(generated.target_fovs, axis=1, dtype=float) - 1.0))
        ),
        "retained_diagonal_mass": retained,
        "offdiag_remodeling_mass": offdiag_mass,
        "open_mass": open_mass,
        "emergence_mass": float(np.sum(e, dtype=float)),
        "burden_ordering_status": ordering,
        "endpoint_closure_l1": generated.endpoint_closure_l1,
        "imputed_row_source_mass": float(np.sum(x * truth.row_imputed_mask.astype(float), dtype=float)),
        "n_source_fovs": int(generated.source_fovs.shape[0]),
        "n_target_fovs": int(generated.target_fovs.shape[0]),
        "n_expected_evidence_blocks": n_expected_blocks,
        "has_multiple_expected_evidence_blocks": bool(n_expected_blocks >= 2),
        "offdiag_geometry_locality": locality,
        "truth_finite": bool(np.isfinite(A).all() and np.isfinite(d).all() and np.isfinite(e).all()),
    }
