from __future__ import annotations

# ruff: noqa: E402, I001

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stride.errors import ContractError
from stride.latent.recurrence import RecurrenceResult
from stride.outputs.fit_result import PatientBridgeResult, STRIDEFitResult

ANNDATA_AVAILABLE = importlib.util.find_spec("anndata") is not None
pytestmark = pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")


def _write_config(path: Path) -> Path:
    from tests.helpers_task_a_fixture import K_FULL

    config = {
        "task_name": "Task A block1 summaries",
        "enabled_blocks": ["block0_locality_gate", "block1_continuity_backbone"],
        "data": {"mass_mode": "uniform", "k_full": K_FULL},
        "block0": {"random_seed": 7},
        "block1": {
            "target_alpha": 0.05,
            "lambda_grid": [0.05, 0.1, 0.5, 1.0],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run_block1(tmp_path: Path) -> dict[str, object]:
    from tasks.task_A.workflows.run_block1 import run_block1_workflow
    from tests.helpers_task_a_fixture import write_passed_block0_bundle, write_task_a_fixture

    fixture_path = write_task_a_fixture(tmp_path / "fixture.h5ad")
    config_path = _write_config(tmp_path / "config.yaml")
    block0_bundle_path = write_passed_block0_bundle(
        tmp_path / "block0" / "block0_bundle.json",
        config_path=config_path,
        data_path=fixture_path,
    )
    bundle_path = run_block1_workflow(
        config_path=str(config_path),
        data_path=str(fixture_path),
        block0_bundle=str(block0_bundle_path),
        output_dir=str(tmp_path / "block1"),
    )
    return json.loads(Path(bundle_path).read_text(encoding="utf-8"))


def _make_ok_patient_result(
    *,
    patient_id: str,
    state_ids: tuple[int, int],
) -> PatientBridgeResult:
    A = np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float)
    d = np.asarray([0.2, 0.2], dtype=float)
    e = np.asarray([0.1, 0.0], dtype=float)
    mu_minus = np.asarray([0.6, 0.4], dtype=float)
    mu_plus = np.sum(A, axis=0, dtype=float) + e
    return PatientBridgeResult(
        patient_id=patient_id,
        fit_status="ok",
        A=A,
        d=d,
        e=e,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        state_ids=state_ids,
        auxiliary={
            "matched_transition_burden": A * mu_minus[:, None],
            "source_unmatched_burden": d * mu_minus,
            "target_unmatched_burden": e * float(np.sum(mu_minus, dtype=float)),
        },
    )


def _make_fit_result(*patient_results: PatientBridgeResult) -> STRIDEFitResult:
    patient_ids = tuple(result.patient_id for result in patient_results)
    return STRIDEFitResult(
        patient_inputs=tuple(SimpleNamespace(patient_id=patient_id) for patient_id in patient_ids),
        patient_results=tuple(patient_results),
        recurrence=RecurrenceResult(patient_ids=patient_ids, families=(), fit_status="ok"),
        fit_status="ok",
    )


def _make_dry_run_row(
    *,
    patient_id: str,
    pair_family: str,
    fit_status: str,
    defer_reason: str | None = None,
) -> dict[str, object]:
    target_domain = {"TC-IM": "IM", "TC-PT": "PT"}[pair_family]
    return {
        "pair_family": pair_family,
        "claim_role": "confirmatory",
        "patient_id": patient_id,
        "implementation_tier": "canonical_full",
        "fit_surface": "fit_stride",
        "fit_status": fit_status,
        "bridge_realized": fit_status == "ok",
        "defer_reason": defer_reason,
        "uncertainty_status": None,
        "cohort_recurrence_fit_status": "ok",
        "n_recurrence_families": 0,
        "n_recurrence_used_patients": 2,
        "source_domain": "TC",
        "target_domain": target_domain,
    }


def _make_family_summary_rows(
    *,
    patient_id: str,
    pair_family: str,
    adjustment: float,
) -> list[dict[str, object]]:
    target_domain = {"TC-IM": "IM", "TC-PT": "PT"}[pair_family]
    base_values = {
        "self_retention": 0.6,
        "depletion": 0.2,
        "off_diagonal_remodeling": 0.2,
        "emergence": 0.1,
    }
    summary_role = {
        "self_retention": "proof_carrying",
        "depletion": "proof_carrying",
        "off_diagonal_remodeling": "diagnostic_supportive",
        "emergence": "supportive",
    }
    eligible_axis = {
        "self_retention": "source",
        "depletion": "source",
        "off_diagonal_remodeling": "source",
        "emergence": "target",
    }
    rows: list[dict[str, object]] = []
    for summary_name, base_value in base_values.items():
        for scale, scale_adjustment in (("burden_weighted", 0.0), ("community_mean", 0.05)):
            rows.append(
                {
                    "patient_id": patient_id,
                    "pair_family": pair_family,
                    "claim_role": "confirmatory",
                    "source_domain": "TC",
                    "target_domain": target_domain,
                    "summary_name": summary_name,
                    "summary_role": summary_role[summary_name],
                    "scale": scale,
                    "value": base_value + adjustment + scale_adjustment,
                    "eligible_entity_axis": eligible_axis[summary_name],
                    "eligible_entity_count": 2,
                    "burden_total": 1.0 + adjustment,
                }
            )
    return rows


def test_block1_source_summary_rows_partition_source_mass(tmp_path: Path) -> None:
    payload = _run_block1(tmp_path)
    source_summary = pd.read_csv(payload["source_community_summary_path"])

    total = (
        source_summary["self_retention"].astype(float)
        + source_summary["depletion"].astype(float)
        + source_summary["off_diagonal_remodeling"].astype(float)
    )
    assert total.to_list() == pytest.approx([1.0] * len(total))


def test_block1_family_summary_stays_paired_on_confirmatory_families(tmp_path: Path) -> None:
    payload = _run_block1(tmp_path)
    family_summary = pd.read_csv(payload["family_summary_path"])

    patient_pair_counts = (
        family_summary.groupby("patient_id", sort=True)["pair_family"].nunique().to_dict()
    )
    assert patient_pair_counts == {"P01": 2, "P02": 2}
    assert set(family_summary["pair_family"].astype(str)) == {"TC-IM", "TC-PT"}
    assert set(family_summary["scale"].astype(str)) == {"burden_weighted", "community_mean"}


def test_block1_bundle_keeps_legacy_block_id_while_exposing_new_summary_contract(tmp_path: Path) -> None:
    payload = _run_block1(tmp_path)

    assert payload["block"] == "block1_continuity_backbone"
    assert payload["scientific_role"] == "real_data_biological_discovery"
    assert payload["implementation_tier"] == "canonical_full"
    assert payload["evidence_lineage"] == "canonical_rerun"
    assert payload["fit_surface"] == "fit_stride"
    assert Path(payload["family_summary_path"]).name == "block1_family_summary.csv"
    assert Path(payload["source_community_summary_path"]).name == "block1_source_community_summary.csv"
    assert Path(payload["target_community_summary_path"]).name == "block1_target_community_summary.csv"
    assert Path(payload["recurrence_summary_path"]).name == "block1_recurrence_summary.json"
    assert Path(payload["recurrence_families_path"]).name == "block1_recurrence_families.json"
    assert Path(payload["recurrence_embeddings_path"]).name == "block1_recurrence_embeddings.csv"


def test_build_block1_summary_frames_are_sorted_and_column_stable() -> None:
    from tasks.task_A.block1.summaries import (
        FAMILY_SUMMARY_COLUMNS,
        SOURCE_SUMMARY_COLUMNS,
        TARGET_SUMMARY_COLUMNS,
        build_block1_summary_frames,
    )
    from tasks.task_A.config import TaskAOrderedPairFamilySpec

    pair_families = (
        TaskAOrderedPairFamilySpec(
            name="TC-PT",
            source_domain="TC",
            target_domain="PT",
            claim_role="confirmatory",
            pair_types=("TC->PT", "PT->TC"),
        ),
        TaskAOrderedPairFamilySpec(
            name="TC-IM",
            source_domain="TC",
            target_domain="IM",
            claim_role="confirmatory",
            pair_types=("TC->IM", "IM->TC"),
        ),
    )
    fit_results = {
        "TC-PT": _make_fit_result(
            _make_ok_patient_result(patient_id="P02", state_ids=(9, 4)),
            _make_ok_patient_result(patient_id="P01", state_ids=(9, 4)),
        ),
        "TC-IM": _make_fit_result(
            _make_ok_patient_result(patient_id="P02", state_ids=(7, 3)),
            _make_ok_patient_result(patient_id="P01", state_ids=(7, 3)),
        ),
    }

    family_frame, source_frame, target_frame = build_block1_summary_frames(
        fit_results=fit_results,
        pair_families=pair_families,
    )

    assert tuple(family_frame.columns) == FAMILY_SUMMARY_COLUMNS
    assert tuple(source_frame.columns) == SOURCE_SUMMARY_COLUMNS
    assert tuple(target_frame.columns) == TARGET_SUMMARY_COLUMNS

    family_order = list(
        family_frame.loc[:, ["patient_id", "pair_family", "summary_name", "scale"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert family_order == sorted(family_order)

    source_order = list(
        source_frame.loc[:, ["patient_id", "pair_family", "source_community_id"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert source_order == sorted(source_order)

    target_order = list(
        target_frame.loc[:, ["patient_id", "pair_family", "target_community_id"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert target_order == sorted(target_order)


def test_build_block1_summary_frames_require_realized_auxiliary_burdens() -> None:
    from tasks.task_A.block1.summaries import build_block1_summary_frames
    from tasks.task_A.config import TaskAOrderedPairFamilySpec

    family_spec = TaskAOrderedPairFamilySpec(
        name="TC-IM",
        source_domain="TC",
        target_domain="IM",
        claim_role="confirmatory",
        pair_types=("TC->IM", "IM->TC"),
    )
    patient_result = PatientBridgeResult(
        patient_id="P01",
        fit_status="ok",
        A=np.asarray([[0.6, 0.2], [0.1, 0.7]], dtype=float),
        d=np.asarray([0.2, 0.2], dtype=float),
        e=np.asarray([0.1, 0.0], dtype=float),
        mu_minus=np.asarray([0.6, 0.4], dtype=float),
        mu_plus=np.asarray([0.8, 0.9], dtype=float),
        state_ids=(7, 3),
        auxiliary={
            "source_unmatched_burden": np.asarray([0.12, 0.08], dtype=float),
            "target_unmatched_burden": np.asarray([0.1, 0.0], dtype=float),
        },
    )

    with pytest.raises(ContractError, match="matched_transition_burden"):
        build_block1_summary_frames(
            fit_results={"TC-IM": _make_fit_result(patient_result)},
            pair_families=(family_spec,),
        )


def test_build_block1_comparison_frames_preserve_confirmatory_scope_and_statuses() -> None:
    from tasks.task_A.block1.comparisons import (
        FAMILY_COMPARISON_COLUMNS,
        SOURCE_COMMUNITY_COMPARISON_COLUMNS,
        TARGET_COMMUNITY_COMPARISON_COLUMNS,
        build_block1_comparison_frames,
    )

    dry_run_df = pd.DataFrame(
        [
            _make_dry_run_row(patient_id="P01", pair_family="TC-IM", fit_status="ok"),
            _make_dry_run_row(patient_id="P01", pair_family="TC-PT", fit_status="ok"),
            _make_dry_run_row(
                patient_id="P02",
                pair_family="TC-IM",
                fit_status="deferred",
                defer_reason="fixture_left_deferred",
            ),
            _make_dry_run_row(patient_id="P02", pair_family="TC-PT", fit_status="ok"),
            _make_dry_run_row(patient_id="P03", pair_family="TC-IM", fit_status="ok"),
        ]
    )
    family_summary_df = pd.DataFrame(
        _make_family_summary_rows(patient_id="P01", pair_family="TC-IM", adjustment=0.20)
        + _make_family_summary_rows(patient_id="P01", pair_family="TC-PT", adjustment=0.05)
        + _make_family_summary_rows(patient_id="P02", pair_family="TC-PT", adjustment=0.15)
        + _make_family_summary_rows(patient_id="P03", pair_family="TC-IM", adjustment=0.25)
    )
    source_summary_df = pd.DataFrame(
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "source_community_id": 7,
                "source_burden": 0.7,
                "source_weight": 0.7,
                "self_retention": 0.8,
                "depletion": 0.1,
                "off_diagonal_remodeling": 0.1,
                "self_retention_burden": 0.56,
                "depletion_burden": 0.07,
                "off_diagonal_burden": 0.07,
                "top_target_1_id": 8,
                "top_target_1_value": 0.1,
                "top_target_2_id": None,
                "top_target_2_value": 0.0,
                "top_target_3_id": None,
                "top_target_3_value": 0.0,
            },
            {
                "patient_id": "P01",
                "pair_family": "TC-PT",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "PT",
                "source_community_id": 7,
                "source_burden": 0.6,
                "source_weight": 0.6,
                "self_retention": 0.5,
                "depletion": 0.2,
                "off_diagonal_remodeling": 0.3,
                "self_retention_burden": 0.30,
                "depletion_burden": 0.12,
                "off_diagonal_burden": 0.18,
                "top_target_1_id": 9,
                "top_target_1_value": 0.3,
                "top_target_2_id": None,
                "top_target_2_value": 0.0,
                "top_target_3_id": None,
                "top_target_3_value": 0.0,
            },
            {
                "patient_id": "P02",
                "pair_family": "TC-PT",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "PT",
                "source_community_id": 7,
                "source_burden": 0.4,
                "source_weight": 0.4,
                "self_retention": 0.4,
                "depletion": 0.3,
                "off_diagonal_remodeling": 0.3,
                "self_retention_burden": 0.16,
                "depletion_burden": 0.12,
                "off_diagonal_burden": 0.12,
                "top_target_1_id": 9,
                "top_target_1_value": 0.3,
                "top_target_2_id": None,
                "top_target_2_value": 0.0,
                "top_target_3_id": None,
                "top_target_3_value": 0.0,
            },
            {
                "patient_id": "P03",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "source_community_id": 7,
                "source_burden": 0.5,
                "source_weight": 0.5,
                "self_retention": 0.6,
                "depletion": 0.2,
                "off_diagonal_remodeling": 0.2,
                "self_retention_burden": 0.30,
                "depletion_burden": 0.10,
                "off_diagonal_burden": 0.10,
                "top_target_1_id": 8,
                "top_target_1_value": 0.2,
                "top_target_2_id": None,
                "top_target_2_value": 0.0,
                "top_target_3_id": None,
                "top_target_3_value": 0.0,
            },
        ]
    )
    target_summary_df = pd.DataFrame(
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "target_community_id": 8,
                "target_burden": 0.5,
                "target_weight": 0.5,
                "incoming_matched_operator": 0.9,
                "incoming_matched_burden": 0.45,
                "emergence_tendency": 0.1,
                "emergence_burden": 0.05,
            },
            {
                "patient_id": "P01",
                "pair_family": "TC-PT",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "PT",
                "target_community_id": 8,
                "target_burden": 0.4,
                "target_weight": 0.4,
                "incoming_matched_operator": 0.7,
                "incoming_matched_burden": 0.28,
                "emergence_tendency": 0.2,
                "emergence_burden": 0.08,
            },
            {
                "patient_id": "P02",
                "pair_family": "TC-PT",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "PT",
                "target_community_id": 8,
                "target_burden": 0.6,
                "target_weight": 0.6,
                "incoming_matched_operator": 0.6,
                "incoming_matched_burden": 0.36,
                "emergence_tendency": 0.4,
                "emergence_burden": 0.24,
            },
        ]
    )

    family_frame, source_frame, target_frame = build_block1_comparison_frames(
        dry_run_df=dry_run_df,
        family_summary_df=family_summary_df,
        source_summary_df=source_summary_df,
        target_summary_df=target_summary_df,
        patient_ids=("P01", "P02", "P03"),
    )

    assert tuple(family_frame.columns) == FAMILY_COMPARISON_COLUMNS
    assert tuple(source_frame.columns) == SOURCE_COMMUNITY_COMPARISON_COLUMNS
    assert tuple(target_frame.columns) == TARGET_COMMUNITY_COMPARISON_COLUMNS

    p01_family = family_frame.loc[
        (family_frame["patient_id"] == "P01")
        & (family_frame["summary_name"] == "self_retention")
        & (family_frame["scale"] == "burden_weighted")
    ].iloc[0]
    assert p01_family["comparison_status"] == "estimable"
    assert p01_family["delta_tc_im_minus_tc_pt"] == pytest.approx(0.15)
    assert p01_family["contrast_direction"] == "tc_im_gt_tc_pt"

    p02_family = family_frame.loc[
        (family_frame["patient_id"] == "P02")
        & (family_frame["summary_name"] == "self_retention")
        & (family_frame["scale"] == "burden_weighted")
    ].iloc[0]
    assert p02_family["comparison_status"] == "deferred"
    assert p02_family["tc_im_fit_status"] == "deferred"
    assert p02_family["tc_pt_fit_status"] == "ok"
    assert p02_family["tc_im_defer_reason"] == "fixture_left_deferred"

    p03_family = family_frame.loc[
        (family_frame["patient_id"] == "P03")
        & (family_frame["summary_name"] == "self_retention")
        & (family_frame["scale"] == "burden_weighted")
    ].iloc[0]
    assert p03_family["comparison_status"] == "missing"
    assert p03_family["tc_im_fit_status"] == "ok"
    assert p03_family["tc_pt_fit_status"] == "missing"

    p01_source = source_frame.loc[
        (source_frame["patient_id"] == "P01")
        & (source_frame["source_community_id"] == 7)
        & (source_frame["summary_name"] == "self_retention")
    ].iloc[0]
    assert p01_source["comparison_status"] == "estimable"
    assert p01_source["delta_tc_im_minus_tc_pt"] == pytest.approx(0.3)
    assert p01_source["comparison_scope_role"] == "exploratory_supportive"

    p02_source = source_frame.loc[
        (source_frame["patient_id"] == "P02")
        & (source_frame["source_community_id"] == 7)
        & (source_frame["summary_name"] == "self_retention")
    ].iloc[0]
    assert p02_source["comparison_status"] == "deferred"

    p03_source = source_frame.loc[
        (source_frame["patient_id"] == "P03")
        & (source_frame["source_community_id"] == 7)
        & (source_frame["summary_name"] == "self_retention")
    ].iloc[0]
    assert p03_source["comparison_status"] == "missing"

    p01_target = target_frame.loc[
        (target_frame["patient_id"] == "P01")
        & (target_frame["target_community_id"] == 8)
        & (target_frame["summary_name"] == "incoming_matched_operator")
    ].iloc[0]
    assert p01_target["comparison_status"] == "estimable"
    assert p01_target["delta_tc_im_minus_tc_pt"] == pytest.approx(0.2)
    assert p01_target["comparison_scope_role"] == "exploratory_supportive"


def test_build_block1_comparison_frames_require_family_rows_for_ok_fits() -> None:
    from tasks.task_A.block1.comparisons import build_block1_comparison_frames

    dry_run_df = pd.DataFrame(
        [
            _make_dry_run_row(patient_id="P01", pair_family="TC-IM", fit_status="ok"),
            _make_dry_run_row(patient_id="P01", pair_family="TC-PT", fit_status="ok"),
        ]
    )
    family_summary_df = pd.DataFrame(
        [
            {
                "patient_id": "P01",
                "pair_family": "TC-IM",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "IM",
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "scale": "burden_weighted",
                "value": 0.6,
                "eligible_entity_axis": "source",
                "eligible_entity_count": 2,
                "burden_total": 1.0,
            },
            {
                "patient_id": "P01",
                "pair_family": "TC-PT",
                "claim_role": "confirmatory",
                "source_domain": "TC",
                "target_domain": "PT",
                "summary_name": "self_retention",
                "summary_role": "proof_carrying",
                "scale": "burden_weighted",
                "value": 0.4,
                "eligible_entity_axis": "source",
                "eligible_entity_count": 2,
                "burden_total": 1.0,
            },
        ]
    )

    with pytest.raises(ContractError, match="missing an expected summary row"):
        build_block1_comparison_frames(
            dry_run_df=dry_run_df,
            family_summary_df=family_summary_df,
            source_summary_df=pd.DataFrame(),
            target_summary_df=pd.DataFrame(),
            patient_ids=("P01",),
        )
