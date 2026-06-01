"""Compact provenance assembly from detached STRIDE loss ledgers."""
from __future__ import annotations

from typing import Any

from ..losses.assembly import ABLATION_TERM_HANDLING, LossLedger
from ..optimize.config import TrainConfig
from ..optimize.result import OptimizerRunInfo
from ..outputs.provenance import (
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    STRIDEFitProvenance,
    build_stride_fit_provenance,
)

try:  # pragma: no cover - depends on optional runtime
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _scalar(value: Any) -> float:
    if torch is not None and torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _obs_component_payload(ledger: LossLedger) -> dict[str, Any]:
    component = ledger.components["obs"]
    return {
        "raw": _scalar(component.raw),
        "scale": _scalar(component.scale),
        "normalized": _scalar(component.normalized),
        "floor_used": bool(component.floor_used),
    }


def _open_component_payload(ledger: LossLedger) -> dict[str, Any]:
    component = ledger.components["open"]
    return {
        "raw": _scalar(component.raw),
        "normalized": _scalar(component.normalized),
    }


def _geometry_component_payload(ledger: LossLedger) -> dict[str, Any]:
    component = ledger.components["geometry"]
    return {
        "raw": _scalar(component.raw),
        "scale": _scalar(component.scale),
        "normalized": _scalar(component.normalized),
        "effective": _scalar(component.effective_normalized),
        "floor_used": bool(component.floor_used),
    }


def _subbag_consistency_status(ledger: LossLedger) -> str:
    statuses = tuple(record.status for record in ledger.consistency_patients.values())
    if any(status == "ok" for status in statuses):
        return "ok"
    if statuses:
        return str(statuses[0])
    return "insufficient_blocks"


def _subbag_consistency_component_payload(ledger: LossLedger) -> dict[str, Any]:
    component = ledger.components["consistency"]
    return {
        "raw": _scalar(component.raw),
        "effective": _scalar(component.effective_normalized),
        "status": _subbag_consistency_status(ledger),
    }


def _recurrence_component_payload(ledger: LossLedger) -> dict[str, Any]:
    return {
        "raw": _scalar(ledger.recurrence.raw),
        "cohort_scaled": _scalar(ledger.recurrence.cohort_scaled),
    }


def _optimizer_payload(
    train_config: TrainConfig,
    *,
    optimizer_run_info: OptimizerRunInfo,
) -> dict[str, Any]:
    schedule = train_config.schedule
    warmup = schedule.warmup_stage
    main = schedule.main_stage
    cosine = schedule.cosine
    return {
        "framework": "torch",
        "algorithm": "AdamW",
        "weight_decay": 0.0,
        "protocol_name": str(schedule.protocol_name),
        "exit_flag": str(optimizer_run_info.optimizer_exit_flag or optimizer_run_info.reason or "unknown"),
        "warmup": {
            "steps": int(warmup.max_steps),
            "lr": float(warmup.lr),
            "scheduler_policy": str(warmup.scheduler_policy),
            "early_stop": "not_allowed",
        },
        "main": {
            "min_steps": int(main.min_steps),
            "max_steps": int(main.max_steps),
            "lr": float(main.lr),
            "scheduler_policy": str(main.scheduler_policy),
            "early_stop": str(schedule.early_stop_eligibility_policy),
        },
        "cosine": {
            "T_max": int(cosine.T_max),
            "eta_min": float(cosine.eta_min),
        },
        "early_stop_thresholds": {
            "min_relative_improvement": float(train_config.min_relative_improvement),
            "convergence_tol": float(train_config.convergence_tol),
            "patience": int(train_config.patience),
        },
    }


def build_successful_provenance(
    ledger: LossLedger,
    *,
    train_config: TrainConfig,
    optimizer_run_info: OptimizerRunInfo,
) -> STRIDEFitProvenance:
    """Build validated compact provenance for a successful training run."""
    initialization = ledger.initialization
    state_geometry = dict(ledger.metadata.get("state_geometry", {}))
    payload: dict[str, Any] = {
        "provenance_schema_version": STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
        "objective_contract_version": str(ledger.metadata["objective_contract_version"]),
        "random_seed": train_config.seed,
        "objective_constants": dict(ledger.metadata["objective_constants"]),
        "objective_scale_initialization": {
            "policy": "identity_plus_small_open",
            "delta_init": float(initialization.delta_init),
            "K": int(initialization.K),
            "dtype": str(initialization.dtype),
        },
        "optimizer_start_initialization": dict(ledger.metadata["optimizer_start_initialization"]),
        "loss": {
            "total": _scalar(ledger.total),
            "fit": _scalar(ledger.fit),
            "prior": _scalar(ledger.prior),
            "cohort": _scalar(ledger.cohort),
            "components": {
                "obs": _obs_component_payload(ledger),
                "open": _open_component_payload(ledger),
                "geometry": _geometry_component_payload(ledger),
                "subbag_consistency": _subbag_consistency_component_payload(ledger),
                "recurrence": _recurrence_component_payload(ledger),
            },
        },
        "e_bounds": list(ledger.metadata.get("e_bounds", (0.0, 1.0))),
        "post_reconstruction_form": ledger.metadata.get(
            "post_reconstruction_form",
            "normalize(q_minus @ A + e)",
        ),
        "observation_comparison_plan": dict(ledger.metadata["observation_comparison_plan"]),
        "observation_discrepancy": dict(ledger.metadata["observation_discrepancy"]),
        "state_geometry": {
            "normalization": state_geometry.get("normalization", "C_norm = C_raw / s_C"),
            "s_C": float(state_geometry["s_C"]),
        },
        "optimizer": {
            **_optimizer_payload(train_config, optimizer_run_info=optimizer_run_info),
        },
        "recurrence": {
            "support_n_patients": int(ledger.recurrence.support_n_patients),
            "dispersion": _scalar(ledger.recurrence.dispersion),
        },
        "detailed_optimizer_trace": bool(train_config.detailed_trace),
    }
    if train_config.detailed_trace:
        payload["optimizer_trace_ref"] = "STRIDEFitResult.diagnostics.optimizer_trace"
    if ledger.ablation_mode != "none":
        payload.update(
            {
                "ablation_mode": ledger.ablation_mode,
                "ablation_term_handling": ledger.ablation_term_handling or ABLATION_TERM_HANDLING,
                "ablation_denominator_policy": ledger.metadata["ablation_denominator_policy"],
            }
        )
    return build_stride_fit_provenance(payload)


__all__ = ["build_successful_provenance"]
