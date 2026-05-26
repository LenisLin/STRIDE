"""Assemble the STRIDE v1 three-block differentiable objective."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..errors import ContractError
from ..geometry.state_geometry import StateGeometry
from ..observation.balanced_sinkhorn import BalancedSinkhornDivergenceConfig
from ._constants import (
    ABLATION_MODES,
    ABLATION_TERM_HANDLING,
    EPSILON_NORM,
    GEOMETRY_EFFECTIVE_WEIGHT,
    NUMERICAL_MIN_MASS,
    OBJECTIVE_CONTRACT_VERSION,
    OFFDIAG_INIT_MASS,
    RHO_SUBBAG,
    S_COHORT,
)
from ._context import ObjectiveContext, _validate_objective_context
from ._initialization import (
    EvidenceBlock,
    FovCostScale,
    ScaleInit,
    compute_init_fov_cost_scale,
    identity_plus_small_open_initialization,
)
from ._parameters import (
    ADEState,
    LogitState,
    parameters_from_unconstrained,
    post_reconstruct,
    unconstrained_from_constrained,
    unconstrained_from_initialization,
    _validate_parameters,
)
from ._totals import (
    CohortLossLedger,
    ConsistencyPatientLedger,
    LossComponent,
    LossLedger,
    LossTotals,
    ObservationBlockLedger,
    assemble_loss_totals,
)
from .cohort import compute_recurrence_raw
from .fit import compute_consistency_raw_from_block_losses, compute_observation_raw
from .prior import compute_geometry_raw, compute_open_raw


def _observation_comparison_plan_metadata(
    evidence_blocks: Sequence[EvidenceBlock],
) -> dict[str, Any]:
    block_counts: Counter[str] = Counter()
    policies: set[str] = set()
    for block in evidence_blocks:
        if not isinstance(block, EvidenceBlock):
            raise ContractError("evidence_blocks must contain EvidenceBlock objects")
        patient_id = str(block.patient_id).strip()
        if patient_id == "":
            raise ContractError("EvidenceBlock.patient_id must be non-empty")
        metadata = dict(block.metadata)
        policy = str(metadata.get("block_construction_policy", "")).strip()
        if policy == "":
            raise ContractError(
                "EvidenceBlock.metadata['block_construction_policy'] is required"
            )
        block_counts[patient_id] += 1
        policies.add(policy)
    if len(policies) != 1:
        raise ContractError("evidence blocks must share one block_construction_policy")
    return {
        "resolved_by": "task_layer",
        "n_evidence_blocks": len(evidence_blocks),
        "domain_policy": "observation_layer_only",
        "block_construction_policy": next(iter(policies)),
        "n_blocks_by_patient": dict(sorted(block_counts.items())),
    }


def compute_loss_ledger(
    params: ADEState,
    evidence_blocks: Sequence[EvidenceBlock],
    geometry: StateGeometry,
    *,
    objective_weights: Sequence[float] = (1.0, 1.0, 1.0),
    epsilon_norm: float = EPSILON_NORM,
    ablation_mode: str = "none",
    config: BalancedSinkhornDivergenceConfig | None = None,
    objective_context: ObjectiveContext | None = None,
) -> LossLedger:
    """Evaluate the canonical loss assembly objective and normalization ledger."""
    _, _, _, patient_ids = _validate_parameters(params)
    if objective_context is None:
        objective_context = ObjectiveContext.build(
            params=params,
            evidence_blocks=evidence_blocks,
            geometry=geometry,
            epsilon_norm=epsilon_norm,
            config=config,
        )
    else:
        _validate_objective_context(
            objective_context,
            params=params,
            evidence_blocks=evidence_blocks,
            geometry=geometry,
            epsilon_norm=epsilon_norm,
            config=config,
        )
    init = objective_context.initialization
    fov_cost_scales = objective_context.fov_cost_scales

    current_obs = compute_observation_raw(
        params,
        evidence_blocks,
        geometry,
        fov_cost_scales=fov_cost_scales,
        config=config,
        observed_self_ground_cost_cache=objective_context.observation_ground_cost_cache,
    )
    baseline_obs = objective_context.baseline_obs
    obs_scale = objective_context.obs_scale
    current_normalized_block_losses = current_obs.block_values / obs_scale
    block_patient_ids = tuple(block.patient_id for block in evidence_blocks)
    consistency_raw, consistency_records = compute_consistency_raw_from_block_losses(
        patient_ids=patient_ids,
        block_patient_ids=block_patient_ids,
        normalized_block_losses=current_normalized_block_losses,
    )

    recurrence = compute_recurrence_raw(params)
    totals = assemble_loss_totals(
        raw_components={
            "obs": current_obs.raw,
            "open": compute_open_raw(params),
            "geometry": compute_geometry_raw(params, geometry),
            "consistency": consistency_raw,
            "recurrence": recurrence.raw,
        },
        baseline_components={
            "obs": baseline_obs.raw,
            "geometry": objective_context.baseline_geometry_raw,
        },
        objective_weights=objective_weights,
        epsilon_norm=epsilon_norm,
        ablation_mode=ablation_mode,
    )
    observation_records = tuple(
        ObservationBlockLedger(
            block_id=record.block_id,
            patient_id=record.patient_id,
            raw=record.raw,
            normalized=current_normalized_block_losses[index],
            status=record.status,
            fov_cost_scale=record.fov_cost_scale,
            fov_cost_scale_floor_used=record.fov_cost_scale_floor_used,
            metadata=record.metadata,
            warnings=record.warnings,
        )
        for index, record in enumerate(current_obs.block_records)
    )
    observation_metadata = dict(current_obs.metadata)
    observation_discrepancy = {
        key: observation_metadata[key]
        for key in (
            "operator_version",
            "backend",
            "dtype",
            "inner_epsilon_schedule",
            "outer_epsilon_schedule",
            "max_iter",
            "tol",
            "warning_tol",
        )
        if key in observation_metadata
    }
    state_geometry = dict(observation_metadata.get("state_geometry", {}))
    metadata = {
        "objective_contract_version": OBJECTIVE_CONTRACT_VERSION,
        "objective_constants": {
            "rho_subbag": RHO_SUBBAG,
            "geometry_effective_weight": GEOMETRY_EFFECTIVE_WEIGHT,
            "s_cohort": S_COHORT,
            "epsilon_norm": EPSILON_NORM,
        },
        "optimizer_start_initialization": {
            "policy": "offdiag_seeded_identity_plus_small_open",
            "delta_init": init.delta_init,
            "offdiag_init_mass": OFFDIAG_INIT_MASS,
            "numerical_min_mass": NUMERICAL_MIN_MASS,
            "K": init.K,
            "dtype": init.dtype,
        },
        "e_bounds": (0.0, 1.0),
        "post_reconstruction_form": "normalize(q_minus @ A + e)",
        "observation_comparison_plan": _observation_comparison_plan_metadata(evidence_blocks),
        "observation_discrepancy": observation_discrepancy,
        "state_geometry": state_geometry,
        "fov_cost_scales": [
            {
                "s_G_init": item.value,
                "s_G_init_floor_used": item.floor_used,
                "positive_cost_count": item.positive_cost_count,
            }
            for item in fov_cost_scales
        ],
        "ablation_denominator_policy": totals.ablation_denominator_policy,
    }
    return LossLedger(
        total=totals.total,
        fit=totals.fit,
        prior=totals.prior,
        cohort=totals.cohort,
        components=totals.components,
        objective_weights=totals.objective_weights,
        epsilon_norm=totals.epsilon_norm,
        rho_subbag=totals.rho_subbag,
        geometry_effective_weight=totals.geometry_effective_weight,
        s_cohort=totals.s_cohort,
        offdiag_init_mass=OFFDIAG_INIT_MASS,
        numerical_min_mass=NUMERICAL_MIN_MASS,
        initialization=init,
        observation_blocks=observation_records,
        consistency_patients=consistency_records,
        recurrence=recurrence,
        ablation_mode=totals.ablation_mode,
        ablation_term_handling=totals.ablation_term_handling,
        metadata=metadata,
    )


__all__ = [
    "ABLATION_MODES",
    "ABLATION_TERM_HANDLING",
    "EPSILON_NORM",
    "GEOMETRY_EFFECTIVE_WEIGHT",
    "NUMERICAL_MIN_MASS",
    "OBJECTIVE_CONTRACT_VERSION",
    "OFFDIAG_INIT_MASS",
    "RHO_SUBBAG",
    "S_COHORT",
    "ADEState",
    "CohortLossLedger",
    "ConsistencyPatientLedger",
    "EvidenceBlock",
    "FovCostScale",
    "LogitState",
    "LossComponent",
    "LossLedger",
    "LossTotals",
    "ObservationBlockLedger",
    "ObjectiveContext",
    "ScaleInit",
    "assemble_loss_totals",
    "compute_consistency_raw_from_block_losses",
    "compute_geometry_raw",
    "compute_init_fov_cost_scale",
    "compute_loss_ledger",
    "compute_open_raw",
    "compute_recurrence_raw",
    "identity_plus_small_open_initialization",
    "parameters_from_unconstrained",
    "post_reconstruct",
    "unconstrained_from_constrained",
    "unconstrained_from_initialization",
]
