from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

import stride.pp
import stride.tl as tl
import stride.tl._run as run_module
from stride.errors import ContractError
from stride.io import build_adata
from stride.pp import (
    build_fov_observations,
    build_local_features,
    build_state_basis,
    build_state_geometry,
    validate_ready,
)
from stride.tl import CohortResult, FitResult, RelationResult, fit
from stride.tl._losses import LossLedger
from stride.tl._parameters import RelationParameters
from stride.tl._train import TrainingResult, TrainingRunInfo


def _ready_adata() -> AnnData:
    adata = AnnData(X=np.ones((1, 1)))
    adata.uns["cost_matrix"] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    adata.uns["cost_scale"] = 1.0
    adata.uns["stride"] = {
        "config": {
            "source": "pre",
            "target": "post",
            "community_mode": "fraction",
            "n_states": 2,
            "relations": np.asarray([["TC", "IM"], ["PT", "IM"]], dtype=object),
            "relation_ids": ["pre_TC_to_post_IM", "pre_PT_to_post_IM"],
        },
        "fov_observations": {
            "community_composition": np.asarray(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                    [0.4, 0.6],
                ],
                dtype=float,
            ),
            "metadata": pd.DataFrame(
                [
                    {
                        "patient_id": "p1",
                        "timepoint": "pre",
                        "fov_id": "f1",
                        "domain_label": "TC",
                    },
                    {
                        "patient_id": "p1",
                        "timepoint": "post",
                        "fov_id": "f2",
                        "domain_label": "IM",
                    },
                    {
                        "patient_id": "p2",
                        "timepoint": "pre",
                        "fov_id": "f3",
                        "domain_label": "PT",
                    },
                ]
            ),
        },
    }
    return adata


def _all_skip_adata() -> AnnData:
    adata = _ready_adata()
    adata.uns["stride"]["config"]["relations"] = np.asarray([["TC", "IM"]], dtype=object)
    adata.uns["stride"]["config"]["relation_ids"] = ["pre_TC_to_post_IM"]
    adata.uns["stride"]["fov_observations"]["metadata"] = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "timepoint": "pre",
                "fov_id": "f1",
                "domain_label": "TC",
            },
            {
                "patient_id": "p2",
                "timepoint": "post",
                "fov_id": "f2",
                "domain_label": "IM",
            },
        ]
    )
    adata.uns["stride"]["fov_observations"]["community_composition"] = np.asarray(
        [[0.8, 0.2], [0.3, 0.7]],
        dtype=float,
    )
    return adata


def _build_pp_ready_adata() -> AnnData:
    cell = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(1, 9)],
            "patient": ["p1"] * 8,
            "time": ["pre", "pre", "pre", "pre", "post", "post", "post", "post"],
            "fov": ["f1", "f1", "f1", "f1", "f2", "f2", "f2", "f2"],
            "domain": ["TC", "TC", "TC", "TC", "IM", "IM", "IM", "IM"],
            "cell_type": ["a", "a", "b", "c", "b", "b", "a", "c"],
            "x": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            "y": [0.0] * 8,
        }
    )
    adata = build_adata(
        X=np.ones((cell.shape[0], 2), dtype=float),
        var=["g1", "g2"],
        cell=cell,
        cell_id="cell_id",
        patient="patient",
        time="time",
        fov_id="fov",
        domain="domain",
        cell_type="cell_type",
        x="x",
        y="y",
        source="pre",
        target="post",
        time_order=["pre", "post"],
        relations=[("TC", "IM")],
        community_mode="fraction",
        n_states=3,
        k_neighbors=2,
    )
    build_local_features(adata)
    build_state_basis(adata)
    build_state_geometry(adata)
    build_fov_observations(adata)
    validate_ready(adata)
    return adata


def _fake_training_result(
    patient_ids: tuple[str, ...],
    *,
    n_states: int = 2,
) -> TrainingResult:
    P = len(patient_ids)
    A = torch.eye(n_states, dtype=torch.float64).unsqueeze(0).repeat(P, 1, 1) * 0.9
    d = torch.full((P, n_states), 0.1, dtype=torch.float64)
    e = torch.full((P, n_states), 0.05, dtype=torch.float64)
    total = torch.tensor(1.0, dtype=torch.float64)
    return TrainingResult(
        parameters=RelationParameters(patient_ids=patient_ids, A=A, d=d, e=e),
        loss_ledger=LossLedger(
            total=total,
            fit=total,
            prior=total,
            cohort=total,
            components={
                "obs_raw": total,
                "obs_normalized": total,
                "open_raw": total,
                "geometry_raw": total,
                "geometry_normalized": total,
                "geometry_effective": total,
                "consistency_raw": torch.tensor(0.0, dtype=torch.float64),
                "recurrence_raw": torch.tensor(0.0, dtype=torch.float64),
            },
            metadata={
                "objective_contract_version": "stride_full_estimator_three_block_v1",
                "objective_constants": {
                    "rho_subbag": 1.0,
                    "geometry_effective_weight": 0.01,
                    "s_cohort": 0.01,
                    "epsilon_norm": 0.01,
                },
                "loss_scales": {
                    "obs_scale": total,
                    "obs_scale_floor_used": False,
                    "geometry_scale": total,
                    "geometry_scale_floor_used": False,
                },
                "observation_discrepancy": {
                    "operator_version": "D_obs^BalancedSinkhornDivergence-v1",
                    "backend": "torch",
                    "dtype": "float64",
                    "inner_epsilon_schedule": [0.5, 0.2, 0.1],
                    "outer_epsilon_schedule": [0.5, 0.2, 0.1],
                    "max_iter": 100,
                    "tol": 1e-6,
                    "warning_tol": 1e-4,
                },
                "state_geometry": {
                    "normalization": "C_norm = C_raw / s_C",
                    "s_C": 1.0,
                },
            },
        ),
        run_info=TrainingRunInfo(
            optimizer_exit_flag="max_steps_exhausted_finite",
            random_seed=None,
        ),
    )


def test_tl_fit_public_entry_is_exported() -> None:
    assert callable(fit)
    assert CohortResult is not None
    assert RelationResult is not None
    assert tl.__all__ == ("fit", "FitResult", "RelationResult", "CohortResult")


def test_read_fit_slots_returns_fixed_pp_ready_payload() -> None:
    slots = run_module._read_fit_slots(_ready_adata())

    assert slots["source"] == "pre"
    assert slots["target"] == "post"
    assert slots["n_states"] == 2
    assert slots["metadata"].shape[0] == slots["community_composition"].shape[0]


def test_iter_declared_relations_preserves_relation_id_order() -> None:
    records = run_module._iter_declared_relations(_ready_adata().uns["stride"]["config"])

    assert [record["relation_id"] for record in records] == [
        "pre_TC_to_post_IM",
        "pre_PT_to_post_IM",
    ]
    assert records[0]["source_domain"] == "TC"
    assert records[1]["source_domain"] == "PT"


def test_iter_declared_relations_rejects_bare_string_relation_ids() -> None:
    config = dict(_ready_adata().uns["stride"]["config"])
    config["relation_ids"] = "pre_TC_to_post_IM"

    with pytest.raises(ContractError, match="must be a sequence, not a string"):
        run_module._iter_declared_relations(config)


def test_iter_declared_relations_rejects_duplicate_relation_ids() -> None:
    config = dict(_ready_adata().uns["stride"]["config"])
    config["relation_ids"] = ["duplicate", "duplicate"]

    with pytest.raises(ContractError, match="duplicate"):
        run_module._iter_declared_relations(config)


def test_iter_declared_relations_rejects_relation_id_length_mismatch() -> None:
    config = dict(_ready_adata().uns["stride"]["config"])
    config["relation_ids"] = ["pre_TC_to_post_IM"]

    with pytest.raises(ContractError, match="length must match"):
        run_module._iter_declared_relations(config)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source", None, "must not be None"),
        ("target", np.nan, "contains a missing value"),
        ("target", pd.NA, "contains a missing value"),
        ("relation_ids", [None, "pre_PT_to_post_IM"], "must not be None"),
        ("relations", np.asarray([[["TC"], "IM"], ["PT", "IM"]], dtype=object), "scalar"),
        (
            "relations",
            np.asarray([["TC", {"domain": "IM"}], ["PT", "IM"]], dtype=object),
            "scalar",
        ),
    ],
)
def test_iter_declared_relations_rejects_non_scalar_or_missing_identifiers(
    field: str,
    value: object,
    match: str,
) -> None:
    config = dict(_ready_adata().uns["stride"]["config"])
    config[field] = value

    with pytest.raises(ContractError, match=match):
        run_module._iter_declared_relations(config)


def test_fit_does_not_call_pp_validate_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_validate_ready(_adata: AnnData) -> None:
        raise AssertionError("validate_ready should not be called by stride.tl.fit")

    monkeypatch.setattr(stride.pp, "validate_ready", fail_validate_ready)
    monkeypatch.setattr(
        run_module,
        "train_relation",
        lambda relation, _cost_matrix, _cost_scale, *, device: _fake_training_result(
            relation.patient_ids
        ),
    )

    with pytest.warns(UserWarning) as warning_records:
        result = fit(_ready_adata())

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "skipped patients missing source or target support" in message
        for message in warning_messages
    )
    assert any("no eligible patients" in message for message in warning_messages)
    assert isinstance(result, FitResult)


def test_fit_orchestrates_realized_relations_in_declared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_calls: list[str] = []

    def fake_train_relation(relation, _cost_matrix, _cost_scale, *, device):
        train_calls.append(str(relation.relation_id))
        n_states = int(relation.blocks[0].source_bag.shape[1])
        return _fake_training_result(relation.patient_ids, n_states=n_states)

    monkeypatch.setattr(run_module, "train_relation", fake_train_relation)

    with pytest.warns(UserWarning) as warning_records:
        result = fit(_ready_adata())

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "skipped patients missing source or target support" in message
        for message in warning_messages
    )
    assert any("no eligible patients" in message for message in warning_messages)
    assert isinstance(result, FitResult)
    assert train_calls == ["pre_TC_to_post_IM"]
    assert result.relation_ids == ("pre_TC_to_post_IM",)
    assert tuple(result.relations) == ("pre_TC_to_post_IM",)


def test_fit_default_cuda_device_falls_back_to_cpu_when_cuda_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[object] = []

    def fake_train_relation(relation, _cost_matrix, _cost_scale, *, device):
        devices.append(device)
        n_states = int(relation.blocks[0].source_bag.shape[1])
        return _fake_training_result(relation.patient_ids, n_states=n_states)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run_module, "train_relation", fake_train_relation)

    with pytest.warns(UserWarning):
        fit(_ready_adata())

    assert devices == [torch.device("cpu")]


def test_fit_passes_public_device_to_relation_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[object] = []

    def fake_train_relation(relation, _cost_matrix, _cost_scale, *, device):
        devices.append(device)
        n_states = int(relation.blocks[0].source_bag.shape[1])
        return _fake_training_result(relation.patient_ids, n_states=n_states)

    monkeypatch.setattr(run_module, "train_relation", fake_train_relation)

    with pytest.warns(UserWarning):
        fit(_ready_adata(), device="cpu")

    assert devices == [torch.device("cpu")]


def test_fit_records_skip_warning_for_empty_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_module,
        "train_relation",
        lambda relation, _cost_matrix, _cost_scale, *, device: _fake_training_result(
            relation.patient_ids
        ),
    )

    with pytest.warns(UserWarning) as warning_records:
        result = fit(_ready_adata())

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "skipped patients missing source or target support" in message
        for message in warning_messages
    )
    assert any("no eligible patients" in message for message in warning_messages)
    assert result.warnings[0]["code"] == "relation_skipped_no_eligible_patients"
    assert result.warnings[0]["relation_id"] == "pre_PT_to_post_IM"
    assert result.warnings[0]["skipped_patient_ids"] == ("p1", "p2")
    assert result.warnings[0]["support_counts"] == {
        "p1": {"source": 0, "target": 1},
        "p2": {"source": 1, "target": 0},
    }


def test_fit_raises_contract_error_when_all_relations_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_module,
        "train_relation",
        lambda relation, _cost_matrix, _cost_scale, *, device: _fake_training_result(
            relation.patient_ids
        ),
    )

    with (
        pytest.warns(UserWarning) as warning_records,
        pytest.raises(
            ContractError,
            match="no realized",
        ),
    ):
        fit(_all_skip_adata())

    warning_messages = [str(record.message) for record in warning_records]
    assert any(
        "skipped patients missing source or target support" in message
        for message in warning_messages
    )
    assert any("no eligible patients" in message for message in warning_messages)


def test_fit_consumes_io_pp_validated_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_calls: list[str] = []

    def fake_train_relation(relation, _cost_matrix, _cost_scale, *, device):
        train_calls.append(str(relation.relation_id))
        n_states = int(relation.blocks[0].source_bag.shape[1])
        return _fake_training_result(relation.patient_ids, n_states=n_states)

    monkeypatch.setattr(run_module, "train_relation", fake_train_relation)

    result = fit(_build_pp_ready_adata())

    assert isinstance(result, FitResult)
    assert train_calls == ["pre_TC_to_post_IM"]
    assert result.relation_ids == ("pre_TC_to_post_IM",)
    relation = result.relations["pre_TC_to_post_IM"]
    assert isinstance(relation, RelationResult)
    assert relation.patient_ids == ("p1",)
    assert relation.A.shape[0] == len(relation.patient_ids)
    assert relation.cohort is not None
    assert isinstance(relation.cohort, CohortResult)
