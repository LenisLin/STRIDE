from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

import stride
import stride.pl as pl
from stride.errors import ContractError
from stride.tl import CohortResult, FitResult, RelationResult


def _cohort(relation_id: str = "pre_TC_to_post_IM") -> CohortResult:
    return CohortResult(
        relation_id=relation_id,
        patient_ids=("p1", "p2"),
        template_A=np.asarray([[0.70, 0.20], [0.15, 0.75]], dtype=float),
        template_d=np.asarray([0.10, 0.10], dtype=float),
        template_e=np.asarray([0.05, 0.20], dtype=float),
        support_n_patients=2,
        dispersion=0.04,
    )


def _relation(relation_id: str = "pre_TC_to_post_IM") -> RelationResult:
    cohort = _cohort(relation_id)
    return RelationResult(
        relation_id=relation_id,
        patient_ids=cohort.patient_ids,
        A=np.stack([cohort.template_A, cohort.template_A]),
        d=np.stack([cohort.template_d, cohort.template_d]),
        e=np.stack([cohort.template_e, cohort.template_e]),
        support={},
        cohort=cohort,
    )


def _fit() -> FitResult:
    first = _relation("pre_TC_to_post_IM")
    second = _relation("pre_PT_to_post_IM")
    return FitResult(
        relations={first.relation_id: first, second.relation_id: second},
        relation_ids=(first.relation_id, second.relation_id),
        source="pre",
        target="post",
        n_states=2,
    )


def test_cohort_relation_heatmap_is_pl_only_public_api() -> None:
    assert pl.cohort_relation_heatmap is not None
    assert "cohort_relation_heatmap" in pl.__all__
    assert "cohort_relation_heatmap" not in stride.__all__


def test_cohort_relation_heatmap_accepts_cohort_result() -> None:
    fig = pl.cohort_relation_heatmap(_cohort(), state_labels=("C0", "C1"))

    assert isinstance(fig, Figure)
    heatmap_axes = [ax for ax in fig.axes if ax.images]
    assert len(heatmap_axes) == 1
    ax = heatmap_axes[0]
    assert ax.get_xlabel() == "Target community"
    assert ax.get_ylabel() == "Source community"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["C0", "C1", "source open d"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["C0", "C1", "target open e"]
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {90.0}
    image_array = ax.images[0].get_array()
    assert image_array.shape == (3, 3)
    assert np.ma.is_masked(image_array[-1, -1])
    assert ax.get_title() == "Cohort relation template: pre_TC_to_post_IM"
    assert fig._suptitle is None


def test_cohort_relation_heatmap_accepts_relation_result() -> None:
    fig = pl.cohort_relation_heatmap(_relation())

    assert isinstance(fig, Figure)
    heatmap_axes = [ax for ax in fig.axes if ax.images]
    assert len(heatmap_axes) == 1
    assert heatmap_axes[0].images[0].get_array().shape == (3, 3)


def test_cohort_relation_heatmap_resolves_fit_result_relation_id() -> None:
    fig = pl.cohort_relation_heatmap(_fit(), relation_id="pre_PT_to_post_IM")

    assert isinstance(fig, Figure)
    assert len([ax for ax in fig.axes if ax.images]) == 1
    assert fig._suptitle is None


def test_cohort_relation_heatmap_defaults_to_all_fit_relations() -> None:
    fig = pl.cohort_relation_heatmap(_fit())

    assert isinstance(fig, Figure)
    assert fig._suptitle is None
    heatmap_axes = [ax for ax in fig.axes if ax.images]
    assert len(heatmap_axes) == 2
    titles = [ax.get_title() for ax in heatmap_axes]
    assert "pre_TC_to_post_IM" in titles[0]
    assert "pre_PT_to_post_IM" in titles[1]
    assert "support n=" not in titles[0]
    assert "| n=2 | disp=0.04" in titles[0]
    assert any(ax.get_ylabel() == "Template value" for ax in fig.axes)


def test_cohort_relation_heatmap_rejects_unknown_relation_id() -> None:
    with pytest.raises(ContractError, match="unknown relation_id"):
        pl.cohort_relation_heatmap(_fit(), relation_id="missing")


def test_cohort_relation_heatmap_rejects_missing_cohort_in_fit() -> None:
    first = _relation("pre_TC_to_post_IM")
    second = _relation("pre_PT_to_post_IM")
    second = RelationResult(
        relation_id=second.relation_id,
        patient_ids=second.patient_ids,
        A=second.A,
        d=second.d,
        e=second.e,
        support=second.support,
        cohort=None,
    )
    fit = FitResult(
        relations={first.relation_id: first, second.relation_id: second},
        relation_ids=(first.relation_id, second.relation_id),
        source="pre",
        target="post",
        n_states=2,
    )

    with pytest.raises(ContractError, match="cohort"):
        pl.cohort_relation_heatmap(fit)


def test_cohort_relation_heatmap_rejects_missing_cohort() -> None:
    relation = _relation()
    relation = RelationResult(
        relation_id=relation.relation_id,
        patient_ids=relation.patient_ids,
        A=relation.A,
        d=relation.d,
        e=relation.e,
        support=relation.support,
        cohort=None,
    )

    with pytest.raises(ContractError, match="cohort"):
        pl.cohort_relation_heatmap(relation)


def test_cohort_relation_heatmap_rejects_invalid_shapes() -> None:
    cohort = _cohort()
    invalid = CohortResult(
        relation_id=cohort.relation_id,
        patient_ids=cohort.patient_ids,
        template_A=np.ones((2, 3), dtype=float),
        template_d=cohort.template_d,
        template_e=cohort.template_e,
        support_n_patients=cohort.support_n_patients,
        dispersion=cohort.dispersion,
    )

    with pytest.raises(ContractError, match="template_A"):
        pl.cohort_relation_heatmap(invalid)


def test_cohort_relation_heatmap_saves_pdf_with_tight_bbox(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        calls.append(kwargs)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    returned = pl.cohort_relation_heatmap(_cohort(), save=tmp_path / "cohort.pdf")

    assert returned is None
    assert calls[0]["bbox_inches"] == "tight"
    assert calls[0]["pad_inches"] == 0.08
