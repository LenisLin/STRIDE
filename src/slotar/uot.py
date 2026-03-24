"""
Module: src.slotar.uot
Architecture: Library Level (Domain-Agnostic Mathematical Engine)
Constraints:
- STRICTLY NO references to `tasks`, `config.yaml`, or clinical metadata.
- Implements Batched Unbalanced Optimal Transport (Decision D005).
- Solves the compacted valid batch jointly via batched masked log-domain
  epsilon-scaling while preserving the external [N, K] tensor contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .contracts import DataContractError, validate_uot_inputs
from .exceptions import (
    ERR_UOT_EMPTY_MASS_SOURCE,
    ERR_UOT_EMPTY_MASS_TARGET,
    ERR_UOT_EMPTY_SUPPORT,
    ERR_UOT_NUMERICAL,
)
from .utils import _active_mask_from_combined_mass

STATUS_OK: str = "ok"
MICRO_METRICS: tuple[str, ...] = ("T", "D_pos", "B_pos", "d_rel", "b_rel", "M", "R", "tau")
_EXTRACTION_TARGET_PLAN_ELEMENTS = 250_000


@dataclass(frozen=True)
class UOTSolveConfig:
    """
    Numerical configuration for Batched UOT solve.
    """

    eps_schedule: Sequence[float]
    max_iter: int = 2000
    tol: float = 1e-6
    eta_floor: float = 1e-12
    n_min_proto: float = 0.0  # Maintained as float to support both density and count thresholds
    tau_q: float = 0.25
    tau_mode: str = "pi_weighted_q25"


@dataclass(frozen=True)
class _WorkingBatch:
    row_idx: np.ndarray
    A: np.ndarray
    B: np.ndarray
    mask: np.ndarray
    lambda_pl: np.ndarray
    tau: np.ndarray | None


def precompute_logKernels(C: np.ndarray, eps_schedule: Sequence[float], s_C: float = 1.0) -> list[np.ndarray]:
    """
    Precompute log-kernels (-C / eps) for the epsilon scaling schedule.
    """
    C = np.asarray(C, dtype=float)
    eps_arr = np.asarray(tuple(eps_schedule), dtype=float)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise DataContractError("C must be a square matrix of shape [K, K]")
    if not np.isfinite(C).all():
        raise DataContractError("C contains NaN/Inf")
    if eps_arr.ndim != 1 or eps_arr.size == 0:
        raise DataContractError("eps_schedule must be a non-empty 1D sequence")
    if not np.isfinite(eps_arr).all() or (eps_arr <= 0).any():
        raise DataContractError("eps_schedule values must be finite and strictly positive")
    if not np.isfinite(s_C) or s_C <= 0:
        raise DataContractError("s_C must be finite and strictly positive")

    scaled_C = C / float(s_C)
    return [-(scaled_C / eps) for eps in eps_arr]


def _calibrate_one_candidate(
    candidate: float,
    A: np.ndarray,
    B: np.ndarray,
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    target_alpha: float,
) -> tuple[float, float]:
    """
    Evaluate one lambda candidate and return (candidate, error).

    Returns (candidate, inf) if the candidate produces no valid solutions or a
    non-finite median unmatched ratio. Thread-safe: batched_uot_solve allocates
    all working arrays internally and shares no mutable state between calls.
    """
    lambda_pl = np.full(A.shape[0], float(candidate), dtype=float)
    metrics, _details, status = batched_uot_solve(
        A=A,
        B=B,
        lambda_pl=lambda_pl,
        kernels=kernels,
        cfg=cfg,
        tau_external=None,
    )
    ok_mask = status == STATUS_OK
    if not np.any(ok_mask):
        return float(candidate), np.inf

    denom = (
        metrics["T"][ok_mask]
        + metrics["B_pos"][ok_mask]
        + metrics["D_pos"][ok_mask]
    )
    unmatched_ratio = np.divide(
        metrics["B_pos"][ok_mask] + metrics["D_pos"][ok_mask],
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=denom > 0.0,
    )
    family_median = float(np.nanmedian(unmatched_ratio))
    if not np.isfinite(family_median):
        return float(candidate), np.inf

    return float(candidate), abs(family_median - float(target_alpha))


def calibrate_joint_lambda(
    A: np.ndarray,
    B: np.ndarray,
    lambda_grid: Sequence[float],
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    target_alpha: float = 0.05,
) -> float:
    """
    Calibrate one shared lambda on a task-provided pair pool.

    Scans the candidate grid in parallel (one thread per candidate) using
    ThreadPoolExecutor. batched_uot_solve is stateless and safe for concurrent
    invocation. The candidate with the minimum |median_unmatched - target_alpha|
    is returned.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    candidates = np.asarray(tuple(lambda_grid), dtype=float)

    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
        raise DataContractError("A and B must be 2D arrays with the same shape")
    if A.shape[0] == 0:
        raise ValueError("Calibration requires at least one pair row")
    if candidates.ndim != 1 or candidates.size == 0:
        raise DataContractError("lambda_grid must be a non-empty 1D sequence")
    if not np.isfinite(candidates).all() or (candidates <= 0.0).any():
        raise DataContractError("lambda_grid values must be finite and strictly positive")
    if not np.isfinite(target_alpha):
        raise DataContractError("target_alpha must be finite")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_calibrate_one_candidate, c, A, B, kernels, cfg, target_alpha): c
            for c in candidates
        }
        results = [fut.result() for fut in as_completed(futures)]

    best = min(
        ((c, e) for c, e in results if np.isfinite(e)),
        key=lambda x: x[1],
        default=None,
    )
    if best is None:
        raise ValueError("No lambda candidate produced a finite calibration summary")
    return best[0]


def _nan_metrics(n_items: int) -> dict[str, np.ndarray]:
    return {name: np.full(n_items, np.nan, dtype=float) for name in MICRO_METRICS}


def _nan_details(
    n_items: int,
    n_proto: int,
    *,
    return_plan: bool,
) -> dict[str, np.ndarray]:
    source_transport = np.full((n_items, n_proto), np.nan, dtype=float)
    target_transport = np.full((n_items, n_proto), np.nan, dtype=float)
    destruction = np.full((n_items, n_proto), np.nan, dtype=float)
    creation = np.full((n_items, n_proto), np.nan, dtype=float)

    details: dict[str, np.ndarray] = {
        "source_transport_marginal": source_transport,
        "target_transport_marginal": target_transport,
        "source_marginal": source_transport,
        "target_marginal": target_transport,
        "T_k": source_transport,
        "D_k": destruction,
        "B_k": creation,
    }
    if return_plan:
        plan = np.full((n_items, n_proto, n_proto), np.nan, dtype=float)
        details["Pi"] = plan
        details["plan"] = plan
    return details


def _scaled_cost_from_kernel(log_kernel: np.ndarray, eps: float) -> np.ndarray:
    return -np.asarray(log_kernel, dtype=float) * float(eps)


def _validate_external_support_mask(
    external_support_mask: np.ndarray | None,
    *,
    n_items: int,
    n_proto: int,
) -> np.ndarray | None:
    if external_support_mask is None:
        return None

    mask_arr = np.asarray(external_support_mask)
    if mask_arr.shape != (n_items, n_proto):
        raise DataContractError(
            "external_support_mask must have shape "
            f"{(n_items, n_proto)}, got {mask_arr.shape}"
        )

    if mask_arr.dtype == np.bool_ or np.issubdtype(mask_arr.dtype, np.bool_):
        return mask_arr.astype(bool, copy=False)

    try:
        numeric_mask = np.asarray(mask_arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            "external_support_mask must have boolean semantics (bool or 0/1-valued)"
        ) from exc

    if not np.isfinite(numeric_mask).all() or not np.all((numeric_mask == 0.0) | (numeric_mask == 1.0)):
        raise DataContractError(
            "external_support_mask must have boolean semantics (bool or 0/1-valued)"
        )
    return numeric_mask.astype(bool)


def _validate_runtime_inputs(
    *,
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    n_items: int,
    n_proto: int,
    tau_external: np.ndarray | None,
    external_support_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(kernels) != len(tuple(cfg.eps_schedule)):
        raise DataContractError("kernels length must match cfg.eps_schedule length exactly")

    support_mask_arr = _validate_external_support_mask(
        external_support_mask,
        n_items=n_items,
        n_proto=n_proto,
    )

    if tau_external is None:
        return None, support_mask_arr

    tau_arr = np.asarray(tau_external, dtype=float)
    if tau_arr.shape != (n_items,):
        raise DataContractError(f"tau_external must have shape {(n_items,)}, got {tau_arr.shape}")
    return tau_arr, support_mask_arr


def _subset_optional(array: np.ndarray | None, keep: np.ndarray) -> np.ndarray | None:
    if array is None:
        return None
    return array[keep]


def _subset_working_batch(batch: _WorkingBatch, keep: np.ndarray) -> _WorkingBatch:
    return _WorkingBatch(
        row_idx=batch.row_idx[keep],
        A=batch.A[keep],
        B=batch.B[keep],
        mask=batch.mask[keep],
        lambda_pl=batch.lambda_pl[keep],
        tau=_subset_optional(batch.tau, keep),
    )


def _screen_batch(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    cfg: UOTSolveConfig,
    tau_external: np.ndarray | None,
    external_support_mask: np.ndarray | None,
    status: np.ndarray,
) -> _WorkingBatch | None:
    with np.errstate(over="ignore", invalid="ignore"):
        source_mass = np.sum(A, axis=1, dtype=float)
        target_mass = np.sum(B, axis=1, dtype=float)
        if external_support_mask is None:
            active_mask = _active_mask_from_combined_mass(A + B, cfg.n_min_proto)
        else:
            active_mask = external_support_mask
        active_source_mass = np.sum(np.where(active_mask, A, 0.0), axis=1, dtype=float)
        active_target_mass = np.sum(np.where(active_mask, B, 0.0), axis=1, dtype=float)

    numerical = ~np.isfinite(source_mass) | ~np.isfinite(target_mass)
    status[numerical] = ERR_UOT_NUMERICAL

    source_empty = (status == STATUS_OK) & (source_mass <= 0.0)
    status[source_empty] = ERR_UOT_EMPTY_MASS_SOURCE

    target_empty = (status == STATUS_OK) & (target_mass <= 0.0)
    status[target_empty] = ERR_UOT_EMPTY_MASS_TARGET

    active_nonfinite = (status == STATUS_OK) & (
        ~np.isfinite(active_source_mass) | ~np.isfinite(active_target_mass)
    )
    status[active_nonfinite] = ERR_UOT_NUMERICAL

    no_active_support = np.count_nonzero(active_mask, axis=1) == 0
    empty_support = (status == STATUS_OK) & (
        no_active_support | (active_source_mass <= 0.0) | (active_target_mass <= 0.0)
    )
    status[empty_support] = ERR_UOT_EMPTY_SUPPORT

    row_idx = np.flatnonzero(status == STATUS_OK)
    if row_idx.size == 0:
        return None

    return _WorkingBatch(
        row_idx=row_idx,
        A=A[row_idx],
        B=B[row_idx],
        mask=active_mask[row_idx],
        lambda_pl=lambda_pl[row_idx],
        tau=_subset_optional(tau_external, row_idx),
    )


def _log_mass_tensor(mass: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    log_mass = np.full(mass.shape, -np.inf, dtype=float)
    with np.errstate(divide="ignore"):
        log_mass[positive_mask] = np.log(mass[positive_mask])
    return log_mass


def _invalid_scaling_rows(
    log_u: np.ndarray,
    log_v: np.ndarray,
    mask: np.ndarray,
    source_positive: np.ndarray,
    target_positive: np.ndarray,
) -> np.ndarray:
    bad_u = (source_positive & ~np.isfinite(log_u)) | (mask & (np.isnan(log_u) | np.isposinf(log_u)))
    bad_v = (target_positive & ~np.isfinite(log_v)) | (mask & (np.isnan(log_v) | np.isposinf(log_v)))
    return np.any(bad_u | bad_v, axis=1)


def _rowwise_max_update(
    log_u_prev: np.ndarray,
    log_u_next: np.ndarray,
    log_v_prev: np.ndarray,
    log_v_next: np.ndarray,
    source_positive: np.ndarray,
    target_positive: np.ndarray,
) -> np.ndarray:
    delta_u = np.zeros(log_u_prev.shape, dtype=float)
    delta_v = np.zeros(log_v_prev.shape, dtype=float)

    valid_u = source_positive & np.isfinite(log_u_prev) & np.isfinite(log_u_next)
    valid_v = target_positive & np.isfinite(log_v_prev) & np.isfinite(log_v_next)

    delta_u[valid_u] = np.abs(log_u_next[valid_u] - log_u_prev[valid_u])
    delta_v[valid_v] = np.abs(log_v_next[valid_v] - log_v_prev[valid_v])

    return np.maximum(np.max(delta_u, axis=1), np.max(delta_v, axis=1))


def _batched_log_sinkhorn_eps_scaling(
    batch: _WorkingBatch,
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    status: np.ndarray,
) -> tuple[_WorkingBatch, np.ndarray, np.ndarray, np.ndarray, float] | None:
    source_positive = batch.mask & (batch.A > 0.0)
    target_positive = batch.mask & (batch.B > 0.0)
    log_a = _log_mass_tensor(batch.A, source_positive)
    log_b = _log_mass_tensor(batch.B, target_positive)
    log_u = np.where(batch.mask, 0.0, -np.inf)
    log_v = np.where(batch.mask, 0.0, -np.inf)

    last_log_kernel: np.ndarray | None = None
    last_eps: float | None = None

    for eps, log_kernel in zip(cfg.eps_schedule, kernels, strict=True):
        if batch.row_idx.size == 0:
            return None

        last_eps = float(eps)
        last_log_kernel = np.asarray(log_kernel, dtype=float)
        theta = batch.lambda_pl[:, None] / (batch.lambda_pl[:, None] + last_eps)
        row_delta = np.full(batch.row_idx.shape, np.inf, dtype=float)
        converged = False

        for _ in range(cfg.max_iter):
            with np.errstate(invalid="ignore"):
                log_kv = logsumexp(last_log_kernel[None, :, :] + log_v[:, None, :], axis=2)
                log_u_next = theta * (log_a - log_kv)
                log_u_next = np.where(batch.mask, log_u_next, -np.inf)

                log_ktu = logsumexp(last_log_kernel.T[None, :, :] + log_u_next[:, None, :], axis=2)
                log_v_next = theta * (log_b - log_ktu)
                log_v_next = np.where(batch.mask, log_v_next, -np.inf)

            invalid_rows = _invalid_scaling_rows(
                log_u=log_u_next,
                log_v=log_v_next,
                mask=batch.mask,
                source_positive=source_positive,
                target_positive=target_positive,
            )
            if np.any(invalid_rows):
                status[batch.row_idx[invalid_rows]] = ERR_UOT_NUMERICAL
                keep = ~invalid_rows
                batch = _subset_working_batch(batch, keep)
                log_a = log_a[keep]
                log_b = log_b[keep]
                source_positive = source_positive[keep]
                target_positive = target_positive[keep]
                log_u = log_u[keep]
                log_v = log_v[keep]
                log_u_next = log_u_next[keep]
                log_v_next = log_v_next[keep]
                theta = theta[keep]
                if batch.row_idx.size == 0:
                    return None

            row_delta = _rowwise_max_update(
                log_u_prev=log_u,
                log_u_next=log_u_next,
                log_v_prev=log_v,
                log_v_next=log_v_next,
                source_positive=source_positive,
                target_positive=target_positive,
            )
            log_u = log_u_next
            log_v = log_v_next

            if np.all(row_delta <= cfg.tol):
                converged = True
                break

        if not converged:
            not_converged = row_delta > cfg.tol
            if np.any(not_converged):
                status[batch.row_idx[not_converged]] = ERR_UOT_NUMERICAL
                keep = ~not_converged
                batch = _subset_working_batch(batch, keep)
                log_a = log_a[keep]
                log_b = log_b[keep]
                source_positive = source_positive[keep]
                target_positive = target_positive[keep]
                log_u = log_u[keep]
                log_v = log_v[keep]
                if batch.row_idx.size == 0:
                    return None

    if last_log_kernel is None or last_eps is None or batch.row_idx.size == 0:
        return None

    return batch, log_u, log_v, last_log_kernel, last_eps


def _empty_extracted_details(
    n_proto: int,
    *,
    return_plan: bool,
) -> dict[str, np.ndarray]:
    details = {
        "source_transport_marginal": np.empty((0, n_proto), dtype=float),
        "target_transport_marginal": np.empty((0, n_proto), dtype=float),
        "D_k": np.empty((0, n_proto), dtype=float),
        "B_k": np.empty((0, n_proto), dtype=float),
    }
    if return_plan:
        details["Pi"] = np.empty((0, n_proto, n_proto), dtype=float)
    return details


def _extraction_chunk_size(n_proto: int) -> int:
    plan_elements = max(int(n_proto) * int(n_proto), 1)
    return max(1, int(_EXTRACTION_TARGET_PLAN_ELEMENTS // plan_elements))


def _extract_batched_metrics(
    batch: _WorkingBatch,
    log_u: np.ndarray,
    log_v: np.ndarray,
    last_log_kernel: np.ndarray,
    last_eps: float,
    status: np.ndarray,
    return_plan: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_proto = int(batch.A.shape[1])
    chunk_size = _extraction_chunk_size(n_proto)
    final_cost = _scaled_cost_from_kernel(last_log_kernel, last_eps)

    row_chunks: list[np.ndarray] = []
    metric_chunks = {name: [] for name in MICRO_METRICS}
    detail_chunks: dict[str, list[np.ndarray]] = {
        "source_transport_marginal": [],
        "target_transport_marginal": [],
        "D_k": [],
        "B_k": [],
    }
    if return_plan:
        detail_chunks["Pi"] = []

    for start in range(0, batch.row_idx.size, chunk_size):
        stop = min(start + chunk_size, batch.row_idx.size)
        row_idx_chunk = batch.row_idx[start:stop]
        log_u_chunk = log_u[start:stop]
        log_v_chunk = log_v[start:stop]
        mask_chunk = batch.mask[start:stop]
        a_chunk = batch.A[start:stop]
        b_chunk = batch.B[start:stop]
        tau_chunk = None if batch.tau is None else batch.tau[start:stop]

        log_pi_chunk = log_u_chunk[:, :, None] + last_log_kernel[None, :, :] + log_v_chunk[:, None, :]
        row_shift = np.max(log_pi_chunk, axis=(1, 2))
        finite_rows = np.isfinite(row_shift)
        if not np.all(finite_rows):
            status[row_idx_chunk[~finite_rows]] = ERR_UOT_NUMERICAL
        if not np.any(finite_rows):
            continue

        row_idx_chunk = row_idx_chunk[finite_rows]
        log_pi_chunk = log_pi_chunk[finite_rows]
        row_shift = row_shift[finite_rows]
        mask_chunk = mask_chunk[finite_rows]
        a_chunk = a_chunk[finite_rows]
        b_chunk = b_chunk[finite_rows]
        if tau_chunk is not None:
            tau_chunk = tau_chunk[finite_rows]

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            plan_scaled = np.exp(log_pi_chunk - row_shift[:, None, None])
            scale = np.exp(row_shift)

        transport_scaled = np.sum(plan_scaled, axis=(1, 2), dtype=float)
        pre_scaled = np.sum(plan_scaled, axis=2, dtype=float)
        post_scaled = np.sum(plan_scaled, axis=1, dtype=float)
        transport_mass = scale * transport_scaled
        source_transport_marginal = scale[:, None] * pre_scaled
        target_transport_marginal = scale[:, None] * post_scaled

        a_masked = np.where(mask_chunk, a_chunk, 0.0)
        b_masked = np.where(mask_chunk, b_chunk, 0.0)
        source_mass = np.sum(a_masked, axis=1, dtype=float)
        target_mass = np.sum(b_masked, axis=1, dtype=float)

        destruction_by_proto = np.maximum(a_masked - source_transport_marginal, 0.0)
        creation_by_proto = np.maximum(b_masked - target_transport_marginal, 0.0)
        destruction = np.sum(destruction_by_proto, axis=1, dtype=float)
        creation = np.sum(creation_by_proto, axis=1, dtype=float)
        d_rel = destruction / source_mass
        b_rel = creation / target_mass
        metric_m = np.sum(final_cost[None, :, :] * plan_scaled, axis=(1, 2), dtype=float) / transport_scaled

        tau_metric = np.full(row_idx_chunk.size, np.nan, dtype=float)
        retention = np.full(row_idx_chunk.size, np.nan, dtype=float)
        if tau_chunk is not None:
            finite_tau = np.isfinite(tau_chunk)
            tau_metric[finite_tau] = tau_chunk[finite_tau]
            if np.any(finite_tau):
                retained_mass = np.sum(
                    plan_scaled * (final_cost[None, :, :] <= tau_chunk[:, None, None]),
                    axis=(1, 2),
                    dtype=float,
                )
                retention[finite_tau] = retained_mass[finite_tau] / transport_scaled[finite_tau]

        core_finite = (
            np.isfinite(transport_mass)
            & (transport_mass > 0.0)
            & np.isfinite(destruction)
            & np.isfinite(creation)
            & np.isfinite(d_rel)
            & np.isfinite(b_rel)
            & np.isfinite(metric_m)
        )
        tau_finite_or_missing = np.ones(row_idx_chunk.size, dtype=bool)
        if tau_chunk is not None:
            tau_finite_or_missing = ~np.isfinite(tau_chunk) | np.isfinite(retention)

        valid_rows = core_finite & tau_finite_or_missing
        if not np.all(valid_rows):
            status[row_idx_chunk[~valid_rows]] = ERR_UOT_NUMERICAL
        if not np.any(valid_rows):
            continue

        row_chunks.append(row_idx_chunk[valid_rows])
        metric_chunks["T"].append(transport_mass[valid_rows])
        metric_chunks["D_pos"].append(destruction[valid_rows])
        metric_chunks["B_pos"].append(creation[valid_rows])
        metric_chunks["d_rel"].append(d_rel[valid_rows])
        metric_chunks["b_rel"].append(b_rel[valid_rows])
        metric_chunks["M"].append(metric_m[valid_rows])
        metric_chunks["R"].append(retention[valid_rows])
        metric_chunks["tau"].append(tau_metric[valid_rows])
        detail_chunks["source_transport_marginal"].append(source_transport_marginal[valid_rows])
        detail_chunks["target_transport_marginal"].append(target_transport_marginal[valid_rows])
        detail_chunks["D_k"].append(destruction_by_proto[valid_rows])
        detail_chunks["B_k"].append(creation_by_proto[valid_rows])
        if return_plan:
            detail_chunks["Pi"].append(scale[valid_rows, None, None] * plan_scaled[valid_rows])

    if not row_chunks:
        return np.empty(0, dtype=int), _nan_metrics(0), _empty_extracted_details(n_proto, return_plan=return_plan)

    row_idx = np.concatenate(row_chunks)
    metrics = {name: np.concatenate(chunks, axis=0) for name, chunks in metric_chunks.items()}
    details = {name: np.concatenate(chunks, axis=0) for name, chunks in detail_chunks.items()}
    return row_idx, metrics, details


def batched_uot_solve(
    A: np.ndarray,
    B: np.ndarray,
    lambda_pl: np.ndarray,
    kernels: Sequence[np.ndarray],
    cfg: UOTSolveConfig,
    tau_external: np.ndarray | None = None,
    external_support_mask: np.ndarray | None = None,
    return_plan: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """
    Solve entropic unbalanced OT under the batched [N, K] contract.

    Args:
        A: Source mass tensor of shape [N, K]. Must be non-negative.
        B: Target mass tensor of shape [N, K]. Must be non-negative.
        lambda_pl: Regularization parameters of shape [N].
        kernels: List of precomputed log-kernels (-C/eps), each of shape [K, K].
        cfg: UOTSolveConfig object containing numerical parameters.
        tau_external: Optional externally calibrated retention thresholds of shape [N].
            If omitted, `tau` and `R` are returned as NaN for otherwise-successful rows.
        external_support_mask: Optional boolean semantic support mask of shape [N, K].
            When provided, this mask is authoritative and is not recomputed from A+B.
        return_plan: If True, include the dense transport plan `Pi` in the details output.

    Returns:
        metrics: Dictionary containing output 1D tensors (e.g., 'T', 'D_pos', 'B_pos') each of shape [N].
        details: Dictionary containing exact per-prototype transport marginals and event tensors.
        status: Object array of shape [N] containing "ok" or pure error strings.

    Programmer-level input contracts are validated up front via validate_uot_inputs(...).
    Per-item data degeneracies are isolated by status codes and NaN metric padding.
    """
    validate_uot_inputs(A=A, B=B, lambda_pl=lambda_pl, kernels=kernels)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    lambda_arr = np.asarray(lambda_pl, dtype=float)
    n_items, n_proto = A.shape
    tau_arr, support_mask_arr = _validate_runtime_inputs(
        kernels=kernels,
        cfg=cfg,
        n_items=n_items,
        n_proto=n_proto,
        tau_external=tau_external,
        external_support_mask=external_support_mask,
    )

    metrics = _nan_metrics(n_items)
    details = _nan_details(n_items, n_proto, return_plan=return_plan)
    status = np.full(n_items, STATUS_OK, dtype=object)

    batch = _screen_batch(
        A=A,
        B=B,
        lambda_pl=lambda_arr,
        cfg=cfg,
        tau_external=tau_arr,
        external_support_mask=support_mask_arr,
        status=status,
    )
    if batch is None:
        return metrics, details, status

    solve_state = _batched_log_sinkhorn_eps_scaling(
        batch=batch,
        kernels=kernels,
        cfg=cfg,
        status=status,
    )
    if solve_state is None:
        return metrics, details, status

    solved_batch, log_u, log_v, last_log_kernel, last_eps = solve_state
    row_idx, row_metrics, row_details = _extract_batched_metrics(
        batch=solved_batch,
        log_u=log_u,
        log_v=log_v,
        last_log_kernel=last_log_kernel,
        last_eps=last_eps,
        status=status,
        return_plan=return_plan,
    )
    for metric_name, values in row_metrics.items():
        metrics[metric_name][row_idx] = values
    if row_idx.size > 0:
        details["source_transport_marginal"][row_idx] = row_details["source_transport_marginal"]
        details["target_transport_marginal"][row_idx] = row_details["target_transport_marginal"]
        details["D_k"][row_idx] = row_details["D_k"]
        details["B_k"][row_idx] = row_details["B_k"]
        if return_plan:
            details["Pi"][row_idx] = row_details["Pi"]

    return metrics, details, status
