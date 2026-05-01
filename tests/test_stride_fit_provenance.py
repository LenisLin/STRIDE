from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stride.errors import ContractError
from stride.latent.recurrence import RecurrenceFamily, RecurrenceResult
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult
from stride.outputs.provenance import (
    STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
    build_stride_fit_provenance,
    validate_stride_fit_provenance,
)


def _loss_component(raw: float) -> dict[str, object]:
    return {
        "raw": raw,
        "scale": 1.0,
        "normalized": raw,
        "floor_used": False,
    }


def _minimal_provenance_payload(
    *,
    K: int = 2,
    support_n_patients: int = 2,
    n_evidence_blocks: int = 2,
) -> dict[str, object]:
    return {
        "provenance_schema_version": STRIDE_FIT_PROVENANCE_SCHEMA_VERSION,
        "alpha": 0.5,
        "random_seed": 17,
        "initialization": {
            "policy": "identity_plus_small_open",
            "delta_init": 0.05,
            "K": K,
            "dtype": "float64",
        },
        "loss": {
            "total": 0.325,
            "local": 0.2,
            "regularization": 0.45,
            "epsilon_norm": 0.01,
            "local_denominator": 3,
            "regularization_denominator": 2,
            "components": {
                "obs": _loss_component(0.1),
                "open": _loss_component(0.2),
                "geometry": _loss_component(0.3),
                "consistency": _loss_component(0.4),
                "recurrence": _loss_component(0.5),
            },
        },
        "e_bounds": [0.0, 1.0],
        "post_reconstruction_form": "normalize(q_minus @ A + e)",
        "observation_comparison_plan": {
            "resolved_by": "task_layer",
            "n_evidence_blocks": n_evidence_blocks,
            "domain_policy": "observation_layer_only",
        },
        "observation_discrepancy": {
            "operator_version": "D_obs^BalancedSinkhornDivergence-v1",
            "backend": "torch",
            "dtype": "float64",
            "inner_epsilon_schedule": [0.5, 0.2, 0.1],
            "outer_epsilon_schedule": [0.5, 0.2, 0.1],
            "max_iter": 1000,
            "tol": 1e-6,
            "warning_tol": 1e-4,
        },
        "state_geometry": {
            "normalization": "C_norm = C_raw / s_C",
            "s_C": 1.0,
        },
        "optimizer": {
            "framework": "torch",
            "algorithm": "AdamW",
            "weight_decay": 0.0,
            "scheduler_policy": "none",
        },
        "recurrence": {
            "support_n_patients": support_n_patients,
            "dispersion": 0.0,
        },
        "detailed_optimizer_trace": False,
    }


def _ok_patient_result() -> PatientBridgeResult:
    return PatientBridgeResult(
        patient_id="p1",
        fit_status="ok",
        A=np.asarray([[0.7, 0.1], [0.2, 0.6]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.1, 0.2], dtype=float),
        state_ids=(0, 1),
        implementation_tier="canonical_full",
    )


def _recurrence_family(
    *,
    support_n_patients: int = 1,
    dispersion: float | None = 0.0,
) -> RecurrenceFamily:
    return RecurrenceFamily(
        family_id="cohort_consensus",
        template_A=np.asarray([[0.7, 0.1], [0.2, 0.6]], dtype=float),
        template_d=np.asarray([0.2, 0.2], dtype=float),
        template_e=np.asarray([0.1, 0.2], dtype=float),
        support_n_patients=support_n_patients,
        within_family_dispersion=dispersion,
        member_patient_ids=("p1",),
    )


def _set_loss_values(
    payload: dict[str, object],
    *,
    local: float,
    regularization: float,
    total: float,
) -> None:
    loss = dict(payload["loss"])
    loss["local"] = local
    loss["regularization"] = regularization
    loss["total"] = total
    payload["loss"] = loss


def test_valid_minimal_stride_fit_provenance() -> None:
    provenance = build_stride_fit_provenance(_minimal_provenance_payload())

    assert provenance.provenance_schema_version == STRIDE_FIT_PROVENANCE_SCHEMA_VERSION
    assert provenance.initialization["policy"] == "identity_plus_small_open"
    validate_stride_fit_provenance(provenance)


def test_stride_fit_provenance_rejects_missing_required_field() -> None:
    payload = _minimal_provenance_payload()
    del payload["loss"]

    with pytest.raises(ContractError, match="missing required provenance field 'loss'"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_forbidden_status_like_fields() -> None:
    payload = _minimal_provenance_payload()
    payload["patient_status"] = {"p1": "ok"}

    with pytest.raises(ContractError, match="must not carry forbidden provenance field 'patient_status'"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_optional_diagnostics_are_allowlisted() -> None:
    payload = _minimal_provenance_payload()
    payload["alpha_sensitivity"] = {"grid": [0.25, 0.5, 1.0]}
    payload["legacy_observation_diagnostics"] = {"source": "legacy_bridge"}
    payload["optimizer_trace_ref"] = "artifacts/optimizer_trace.json"

    validate_stride_fit_provenance(payload)

    payload["optimizer_debug"] = {"step_count": 12}
    with pytest.raises(ContractError, match="unexpected provenance field 'optimizer_debug'"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_allows_valid_ablation_extension() -> None:
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = "geometry"
    payload["ablation_term_handling"] = "remove"
    payload["ablation_denominator_policy"] = "fixed_denominator_no_reweighting"
    _set_loss_values(payload, local=0.1, regularization=0.45, total=0.275)

    provenance = build_stride_fit_provenance(payload)

    assert provenance.ablation_mode == "geometry"
    assert provenance.ablation_term_handling == "remove"
    assert provenance.ablation_denominator_policy == "fixed_denominator_no_reweighting"
    validate_stride_fit_provenance(provenance)


@pytest.mark.parametrize(
    ("ablation_mode", "local", "regularization", "total"),
    [
        ("geometry", 0.1, 0.45, 0.275),
        ("recurrence", 0.2, 0.2, 0.2),
        ("consistency", 0.2, 0.25, 0.225),
    ],
)
def test_stride_fit_provenance_ablation_arithmetic_uses_fixed_denominator_effective_terms(
    ablation_mode: str,
    local: float,
    regularization: float,
    total: float,
) -> None:
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = ablation_mode
    payload["ablation_term_handling"] = "remove"
    payload["ablation_denominator_policy"] = "fixed_denominator_no_reweighting"
    _set_loss_values(payload, local=local, regularization=regularization, total=total)

    validate_stride_fit_provenance(payload)


@pytest.mark.parametrize(
    ("ablation_mode", "wrong_field", "wrong_value"),
    [
        ("geometry", "local", 0.2),
        ("recurrence", "regularization", 0.45),
        ("consistency", "regularization", 0.45),
    ],
)
def test_stride_fit_provenance_ablation_arithmetic_rejects_non_effective_terms(
    ablation_mode: str,
    wrong_field: str,
    wrong_value: float,
) -> None:
    expected_values = {
        "geometry": (0.1, 0.45, 0.275),
        "recurrence": (0.2, 0.2, 0.2),
        "consistency": (0.2, 0.25, 0.225),
    }
    local, regularization, total = expected_values[ablation_mode]
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = ablation_mode
    payload["ablation_term_handling"] = "remove"
    payload["ablation_denominator_policy"] = "fixed_denominator_no_reweighting"
    _set_loss_values(payload, local=local, regularization=regularization, total=total)
    loss = dict(payload["loss"])
    loss[wrong_field] = wrong_value
    payload["loss"] = loss

    with pytest.raises(ContractError, match=f"loss.{wrong_field}.*formula"):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("ablation_mode", "open_channel"),
        ("ablation_term_handling", "mask"),
        ("ablation_denominator_policy", "renormalize"),
    ],
)
def test_stride_fit_provenance_rejects_invalid_ablation_extension_values(
    field_name: str,
    bad_value: str,
) -> None:
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = "geometry"
    payload["ablation_term_handling"] = "remove"
    payload["ablation_denominator_policy"] = "fixed_denominator_no_reweighting"
    payload[field_name] = bad_value

    with pytest.raises(ContractError, match=field_name):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize(
    "field_names",
    [
        ("ablation_mode",),
        ("ablation_mode", "ablation_term_handling"),
        ("ablation_term_handling", "ablation_denominator_policy"),
    ],
)
def test_stride_fit_provenance_rejects_partial_ablation_extension(
    field_names: tuple[str, ...],
) -> None:
    values = {
        "ablation_mode": "geometry",
        "ablation_term_handling": "remove",
        "ablation_denominator_policy": "fixed_denominator_no_reweighting",
    }
    payload = _minimal_provenance_payload()
    for field_name in field_names:
        payload[field_name] = values[field_name]

    with pytest.raises(ContractError, match="ablation provenance extension fields"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_allows_reference_compatibility_ablation_none_label() -> None:
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = "none"

    provenance = build_stride_fit_provenance(payload)

    assert provenance.ablation_mode == "none"
    assert provenance.ablation_term_handling is None
    assert provenance.ablation_denominator_policy is None
    validate_stride_fit_provenance(provenance)


@pytest.mark.parametrize(
    "extra_fields",
    [
        {"ablation_term_handling": "remove"},
        {"ablation_denominator_policy": "fixed_denominator_no_reweighting"},
        {
            "ablation_term_handling": "remove",
            "ablation_denominator_policy": "fixed_denominator_no_reweighting",
        },
    ],
)
def test_stride_fit_provenance_rejects_ablation_none_with_extension_fields(
    extra_fields: dict[str, str],
) -> None:
    payload = _minimal_provenance_payload()
    payload["ablation_mode"] = "none"
    payload.update(extra_fields)

    with pytest.raises(ContractError, match="ablation_mode.*none"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_allows_ordinary_reference_without_ablation_extension() -> None:
    payload = _minimal_provenance_payload()

    validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_nonpositive_evidence_blocks() -> None:
    payload = _minimal_provenance_payload(n_evidence_blocks=0)

    with pytest.raises(ContractError, match="n_evidence_blocks.*positive"):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("local_denominator", 4),
        ("regularization_denominator", 3),
    ],
)
def test_stride_fit_provenance_rejects_wrong_loss_denominator(
    field_name: str,
    bad_value: int,
) -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    loss[field_name] = bad_value
    payload["loss"] = loss

    with pytest.raises(ContractError, match=field_name):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("local", 0.21),
        ("regularization", 0.46),
        ("total", 0.34),
    ],
)
def test_stride_fit_provenance_rejects_loss_arithmetic_mismatch(
    field_name: str,
    bad_value: float,
) -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    loss[field_name] = bad_value
    payload["loss"] = loss

    with pytest.raises(ContractError, match=f"loss.{field_name}.*formula"):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize("field_name", ["total", "local", "regularization"])
def test_stride_fit_provenance_rejects_negative_loss_totals(field_name: str) -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    loss[field_name] = -0.1
    payload["loss"] = loss

    with pytest.raises(ContractError, match=f"loss.{field_name}.*non-negative"):
        validate_stride_fit_provenance(payload)


@pytest.mark.parametrize("component_name", ["obs", "open", "geometry", "consistency", "recurrence"])
@pytest.mark.parametrize("field_name", ["raw", "normalized"])
def test_stride_fit_provenance_rejects_negative_loss_component_values(
    component_name: str,
    field_name: str,
) -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components[component_name])
    component[field_name] = -0.1
    components[component_name] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match=f"loss.components.{component_name}.{field_name}.*non-negative"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_zero_loss_component_scale() -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components["geometry"])
    component["scale"] = 0.0
    components["geometry"] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match="loss.components.geometry.scale.*positive"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_requires_open_component_scale_one() -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components["open"])
    component["scale"] = 0.5
    components["open"] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match="loss.components.open.scale.*1.0"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_requires_open_component_floor_unused() -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components["open"])
    component["floor_used"] = True
    components["open"] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match="loss.components.open.floor_used.*False"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_component_normalized_mismatch() -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components["geometry"])
    component["scale"] = 2.0
    component["normalized"] = 0.3
    components["geometry"] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match="loss.components.geometry.normalized.*raw / scale"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_open_normalized_mismatch() -> None:
    payload = _minimal_provenance_payload()
    loss = dict(payload["loss"])
    components = dict(loss["components"])
    component = dict(components["open"])
    component["normalized"] = 0.25
    components["open"] = component
    loss["components"] = components
    payload["loss"] = loss

    with pytest.raises(ContractError, match="loss.components.open.normalized.*raw / scale"):
        validate_stride_fit_provenance(payload)


def test_stride_fit_provenance_rejects_initialization_delta_mismatch() -> None:
    payload = _minimal_provenance_payload()
    initialization = dict(payload["initialization"])
    initialization["delta_init"] = 0.04
    payload["initialization"] = initialization

    with pytest.raises(ContractError, match="initialization.delta_init.*min"):
        validate_stride_fit_provenance(payload)


def test_ok_canonical_full_stride_fit_result_requires_provenance() -> None:
    patient_result = _ok_patient_result()

    with pytest.raises(ContractError, match="requires compact successful-fit provenance"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(patient_result,),
            recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status="ok"),
            fit_status="ok",
            implementation_tier="canonical_full",
        )

    result = STRIDEFitResult(
        patient_inputs=(SimpleNamespace(patient_id="p1"),),
        patient_results=(patient_result,),
        recurrence=RecurrenceResult(
            patient_ids=("p1",),
            families=(_recurrence_family(support_n_patients=1, dispersion=0.0),),
            fit_status="ok",
            used_patient_ids=("p1",),
        ),
        fit_status="ok",
        implementation_tier="canonical_full",
        provenance=_minimal_provenance_payload(support_n_patients=1),
    )
    assert result.provenance is not None


def test_ok_canonical_full_stride_fit_result_rejects_missing_recurrence_family() -> None:
    with pytest.raises(ContractError, match="requires a single cohort recurrence family"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1),
        )


@pytest.mark.parametrize("fit_status", ["deferred", "failed"])
def test_non_ok_stride_fit_result_rejects_successful_fit_provenance(fit_status: str) -> None:
    with pytest.raises(ContractError, match="Only fit_status='ok'.*compact successful-fit provenance"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(
                PatientBridgeResult(
                    patient_id="p1",
                    fit_status=fit_status,
                    implementation_tier="canonical_full",
                ),
            ),
            recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status=fit_status),
            fit_status=fit_status,
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_provenance_k_mismatch() -> None:
    with pytest.raises(ContractError, match="initialization.K must match patient relation state dimension"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(K=3, support_n_patients=1),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_recurrence_support_mismatch() -> None:
    with pytest.raises(ContractError, match="recurrence.support_n_patients must match recurrence.used_patient_ids"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=2),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_recurrence_family_support_mismatch() -> None:
    with pytest.raises(ContractError, match="support_n_patients"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(_recurrence_family(support_n_patients=2),),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_recurrence_family_dispersion_missing() -> None:
    with pytest.raises(ContractError, match="within_family_dispersion"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(_recurrence_family(support_n_patients=1, dispersion=None),),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_recurrence_family_dispersion_mismatch() -> None:
    with pytest.raises(ContractError, match="recurrence.dispersion"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(_recurrence_family(support_n_patients=1, dispersion=0.25),),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_multiple_recurrence_families() -> None:
    with pytest.raises(ContractError, match="requires a single cohort recurrence family"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(
                    _recurrence_family(support_n_patients=1),
                    _recurrence_family(support_n_patients=1),
                ),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1),
            metadata={"n_evidence_blocks": 2},
        )


def test_ok_canonical_full_stride_fit_result_rejects_nonpositive_evidence_blocks() -> None:
    with pytest.raises(ContractError, match="must be positive"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1, n_evidence_blocks=2),
            metadata={"n_evidence_blocks": 0},
        )


def test_ok_canonical_full_stride_fit_result_rejects_non_integer_metadata_evidence_blocks() -> None:
    with pytest.raises(ContractError, match="positive integer"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1, n_evidence_blocks=2),
            metadata={"n_evidence_blocks": "2"},
        )


def test_ok_canonical_full_stride_fit_result_rejects_evidence_block_count_mismatch() -> None:
    with pytest.raises(ContractError, match="n_evidence_blocks must match"):
        STRIDEFitResult(
            patient_inputs=(SimpleNamespace(patient_id="p1"),),
            patient_results=(_ok_patient_result(),),
            recurrence=RecurrenceResult(
                patient_ids=("p1",),
                families=(),
                fit_status="ok",
                used_patient_ids=("p1",),
            ),
            fit_status="ok",
            implementation_tier="canonical_full",
            provenance=_minimal_provenance_payload(support_n_patients=1, n_evidence_blocks=2),
            metadata={"n_evidence_blocks": 3},
        )


def test_deferred_canonical_full_stride_fit_result_does_not_require_provenance() -> None:
    result = STRIDEFitResult(
        patient_inputs=(SimpleNamespace(patient_id="p1"),),
        patient_results=(
            PatientBridgeResult(
                patient_id="p1",
                fit_status="deferred",
                implementation_tier="canonical_full",
            ),
        ),
        recurrence=RecurrenceResult(patient_ids=("p1",), families=(), fit_status="deferred"),
        fit_status="deferred",
        implementation_tier="canonical_full",
    )

    assert result.provenance is None
