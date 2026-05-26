"""Shared open-profile and relation/open metric builders for Block 3 stages."""
from __future__ import annotations

import numpy as np

from . import execution as shared
from .contracts import Block3MetricName, Block3PatientMetricRow, MetricStatus


def _metric_order() -> tuple[Block3MetricName, ...]:
    return (
        Block3MetricName.F_L1_TOTAL,
        Block3MetricName.G_L1_TOTAL,
        Block3MetricName.E_L1_TOTAL,
        Block3MetricName.OFFDIAG_MASS_ABS_ERROR,
        Block3MetricName.DEPLETION_MASS_ABS_ERROR,
        Block3MetricName.EMERGENCE_MASS_ABS_ERROR,
        Block3MetricName.OFFDIAG_RATIO,
        Block3MetricName.DEPLETION_CAPTURE,
        Block3MetricName.EMERGENCE_CAPTURE,
        Block3MetricName.ENDPOINT_Y_MAE,
        Block3MetricName.A_MAE_ACTIVE,
        Block3MetricName.A_MSE_ACTIVE,
        Block3MetricName.TARGET_RECALL_AT_K,
        Block3MetricName.OPEN_SUPPORT_F1,
        Block3MetricName.D_MAE,
        Block3MetricName.D_MSE,
        Block3MetricName.E_MAE,
        Block3MetricName.E_MSE,
    )


def build_mass_level_metric_rows(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    truth: shared.Block3PatientTruth,
    output: shared.Block3MethodOutput,
    open_mass_scale: float | None = None,
) -> list[Block3PatientMetricRow]:
    """Build the full mass-level Block 3 metric set used by 3B and 3C."""

    if output.fit_status != "ok" or output.A is None or output.d is None or output.e is None:
        return [
            shared._make_patient_metric_row(
                rerun_id=rerun_id,
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                evaluation_family=evaluation_family,
                method_name=method_name,
                metric_name=metric_name,
                value=None,
                status=MetricStatus.NOT_ESTIMABLE,
                patient_id=truth.patient_id,
                open_mass_scale=open_mass_scale,
            )
            for metric_name in _metric_order()
        ]

    x = np.asarray(truth.x, dtype=float)
    A_true = np.asarray(truth.A, dtype=float)
    d_true = np.asarray(truth.d, dtype=float)
    e_true = np.asarray(truth.e, dtype=float)
    A_hat = np.asarray(output.A, dtype=float)
    d_hat = np.asarray(output.d, dtype=float)
    e_hat = np.asarray(output.e, dtype=float)
    F_true = x[:, None] * A_true
    F_hat = x[:, None] * A_hat
    g_true = x * d_true
    g_hat = x * d_hat
    offdiag_mask = ~np.eye(A_true.shape[0], dtype=bool)
    true_offdiag = float(np.sum(F_true[offdiag_mask], dtype=float))
    pred_offdiag = float(np.sum(F_hat[offdiag_mask], dtype=float))
    true_depletion_mass = float(np.sum(g_true, dtype=float))
    pred_depletion_mass = float(np.sum(g_hat, dtype=float))
    true_emergence_mass = float(np.sum(e_true, dtype=float))
    pred_emergence_mass = float(np.sum(e_hat, dtype=float))
    active = x > shared._TOL
    endpoint_pred = shared._normalize_probabilities(x @ A_hat + e_hat)
    endpoint_target = np.asarray(truth.y_endpoint if truth.y_endpoint is not None else truth.y, dtype=float)
    recall = shared._target_recall_at_k(x, A_true, A_hat)
    values = {
        Block3MetricName.F_L1_TOTAL: float(np.sum(np.abs(F_hat - F_true), dtype=float)),
        Block3MetricName.G_L1_TOTAL: float(np.sum(np.abs(g_hat - g_true), dtype=float)),
        Block3MetricName.E_L1_TOTAL: float(np.sum(np.abs(e_hat - e_true), dtype=float)),
        Block3MetricName.OFFDIAG_MASS_ABS_ERROR: abs(pred_offdiag - true_offdiag),
        Block3MetricName.DEPLETION_MASS_ABS_ERROR: abs(pred_depletion_mass - true_depletion_mass),
        Block3MetricName.EMERGENCE_MASS_ABS_ERROR: abs(pred_emergence_mass - true_emergence_mass),
        Block3MetricName.OFFDIAG_RATIO: pred_offdiag / true_offdiag
        if true_offdiag > shared._TOL
        else None,
        Block3MetricName.DEPLETION_CAPTURE: pred_depletion_mass / true_depletion_mass
        if true_depletion_mass > shared._TOL
        else None,
        Block3MetricName.EMERGENCE_CAPTURE: pred_emergence_mass / true_emergence_mass
        if true_emergence_mass > shared._TOL
        else None,
        Block3MetricName.ENDPOINT_Y_MAE: float(np.mean(np.abs(endpoint_target - endpoint_pred))),
        Block3MetricName.A_MAE_ACTIVE: float(np.mean(np.abs(F_hat[active] - F_true[active])))
        if np.any(active)
        else 0.0,
        Block3MetricName.A_MSE_ACTIVE: float(np.mean(np.square(F_hat[active] - F_true[active])))
        if np.any(active)
        else 0.0,
        Block3MetricName.TARGET_RECALL_AT_K: recall,
        Block3MetricName.OPEN_SUPPORT_F1: shared._open_support_f1(g_true, g_hat, e_true, e_hat),
        Block3MetricName.D_MAE: float(np.mean(np.abs(g_hat - g_true))),
        Block3MetricName.D_MSE: float(np.mean(np.square(g_hat - g_true))),
        Block3MetricName.E_MAE: float(np.mean(np.abs(e_hat - e_true))),
        Block3MetricName.E_MSE: float(np.mean(np.square(e_hat - e_true))),
    }
    rows: list[Block3PatientMetricRow] = []
    for metric_name in _metric_order():
        value = values[metric_name]
        rows.append(
            shared._make_patient_metric_row(
                rerun_id=rerun_id,
                subexperiment_id=subexperiment_id,
                condition_id=condition_id,
                evaluation_family=evaluation_family,
                method_name=method_name,
                metric_name=metric_name,
                value=value,
                status=MetricStatus.NOT_APPLICABLE if value is None else MetricStatus.REPORTED,
                patient_id=truth.patient_id,
                open_mass_scale=open_mass_scale,
            )
        )
    return rows


def build_open_metric_rows(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    truth: shared.Block3PatientTruth,
    output: shared.Block3MethodOutput,
    open_mass_scale: float | None = None,
) -> list[Block3PatientMetricRow]:
    """Build open-profile recovery metrics shared by `3B-2` and `3C`."""

    metric_rows: list[Block3PatientMetricRow] = []
    if output.fit_status != "ok" or output.A is None or output.d is None or output.e is None:
        for metric_name in (
            Block3MetricName.OPEN_SUPPORT_F1,
            Block3MetricName.D_MAE,
            Block3MetricName.E_MAE,
            Block3MetricName.D_MSE,
            Block3MetricName.E_MSE,
        ):
            metric_rows.append(
                shared._make_patient_metric_row(
                    rerun_id=rerun_id,
                    subexperiment_id=subexperiment_id,
                    condition_id=condition_id,
                    evaluation_family=evaluation_family,
                    method_name=method_name,
                    metric_name=metric_name,
                    value=None,
                    status=MetricStatus.NOT_ESTIMABLE,
                    patient_id=truth.patient_id,
                    open_mass_scale=open_mass_scale,
                )
            )
        return metric_rows

    pred_depletion = np.asarray(truth.x, dtype=float) * np.asarray(output.d, dtype=float)
    pred_emergence = np.asarray(output.e, dtype=float)
    true_depletion = np.asarray(truth.x, dtype=float) * np.asarray(truth.d, dtype=float)
    true_emergence = np.asarray(truth.e, dtype=float)
    metric_rows.append(
        shared._make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.OPEN_SUPPORT_F1,
            value=None
            if truth.open_mass <= shared._TOL
            else shared._open_support_f1(true_depletion, pred_depletion, true_emergence, pred_emergence),
            status=MetricStatus.NOT_APPLICABLE if truth.open_mass <= shared._TOL else MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        shared._make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.D_MAE,
            value=float(np.mean(np.abs(true_depletion - pred_depletion))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        shared._make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.E_MAE,
            value=float(np.mean(np.abs(true_emergence - pred_emergence))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        shared._make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.D_MSE,
            value=float(np.mean(np.square(true_depletion - pred_depletion))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    metric_rows.append(
        shared._make_patient_metric_row(
            rerun_id=rerun_id,
            subexperiment_id=subexperiment_id,
            condition_id=condition_id,
            evaluation_family=evaluation_family,
            method_name=method_name,
            metric_name=Block3MetricName.E_MSE,
            value=float(np.mean(np.square(true_emergence - pred_emergence))),
            status=MetricStatus.REPORTED,
            patient_id=truth.patient_id,
            open_mass_scale=open_mass_scale,
        )
    )
    return metric_rows


def build_relation_and_open_metric_rows(
    *,
    rerun_id: str,
    subexperiment_id: str,
    condition_id: str,
    evaluation_family: str,
    method_name: str,
    truth: shared.Block3PatientTruth,
    output: shared.Block3MethodOutput,
) -> list[Block3PatientMetricRow]:
    """Build the full `3C` relation/open/mass metric row set."""

    return build_mass_level_metric_rows(
        rerun_id=rerun_id,
        subexperiment_id=subexperiment_id,
        condition_id=condition_id,
        evaluation_family=evaluation_family,
        method_name=method_name,
        truth=truth,
        output=output,
    )


__all__ = [
    "build_mass_level_metric_rows",
    "build_open_metric_rows",
    "build_relation_and_open_metric_rows",
]
