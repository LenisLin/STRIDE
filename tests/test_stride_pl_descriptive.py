from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.legend import Legend

import stride
import stride.pl as pl
import stride.pl._utils as pl_utils
from stride.errors import ContractError
from stride.pl._utils import BIO_PASTEL_PALETTE

pytestmark = pytest.mark.usefixtures("_pycomplexheatmap_available")


@pytest.fixture
def _pycomplexheatmap_available(monkeypatch):
    calls: list[dict[str, object]] = []
    module = types.ModuleType("PyComplexHeatmap")

    class FakeClusterMapPlotter:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.ax = plt.gca()
            ax = plt.gca()
            data = kwargs["data"]
            ax.imshow(data.to_numpy(dtype=float), aspect="auto")
            ax.set_xticks(np.arange(data.shape[1]), labels=list(data.columns))
            if kwargs.get("show_rownames", True):
                ax.set_yticks(np.arange(data.shape[0]), labels=list(data.index))
            else:
                ax.set_yticks([])

    def fake_heatmap_annotation(**kwargs):
        return {"kind": "HeatmapAnnotation", **kwargs}

    def fake_anno_simple(values, **kwargs):
        return {"kind": "anno_simple", "values": values, **kwargs}

    def fake_anno_barplot(values, **kwargs):
        return {"kind": "anno_barplot", "values": values, **kwargs}

    def fake_composite(**kwargs):
        calls.append({"kind": "composite", **kwargs})
        return plt.gca(), []

    module.ClusterMapPlotter = FakeClusterMapPlotter
    module.HeatmapAnnotation = fake_heatmap_annotation
    module.anno_simple = fake_anno_simple
    module.anno_barplot = fake_anno_barplot
    module.composite = fake_composite
    monkeypatch.setitem(sys.modules, "PyComplexHeatmap", module)
    return calls


def _pl_adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "patient_id": [
                "p1",
                "p1",
                "p1",
                "p1",
                "p2",
                "p2",
                "p2",
                "p2",
                "p1",
                "p2",
            ],
            "timepoint": [
                "pre",
                "pre",
                "post",
                "post",
                "pre",
                "pre",
                "post",
                "post",
                "pre",
                "post",
            ],
            "fov_id": ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4", "f5", "f6"],
            "domain_label": ["TC", "TC", "IM", "IM", "PT", "PT", "IM", "IM", "TC", "IM"],
            "cell_subtype_label": ["T", "B", "T", "M", "B", "M", "T", "M", "B", "T"],
            "state_id": [0, 1, 1, 2, 0, 2, 1, 2, 0, 2],
        },
        index=[f"c{i}" for i in range(10)],
    )
    adata = AnnData(X=np.ones((obs.shape[0], 2), dtype=float), obs=obs)
    adata.uns["stride"] = {
        "config": {
            "source": "pre",
            "target": "post",
            "community_mode": "fraction",
            "n_states": 4,
            "relations": np.asarray([["TC", "IM"], ["PT", "IM"]], dtype=object),
            "relation_ids": ["pre_TC_to_post_IM", "pre_PT_to_post_IM"],
        },
        "fov_observations": {
            "community_composition": np.asarray(
                [
                    [0.50, 0.50, 0.00, 0.00],
                    [0.00, 0.50, 0.50, 0.00],
                    [0.25, 0.25, 0.50, 0.00],
                    [0.10, 0.20, 0.70, 0.00],
                    [1.00, 0.00, 0.00, 0.00],
                ],
                dtype=float,
            ),
            "metadata": pd.DataFrame(
                {
                    "patient_id": ["p1", "p1", "p2", "p2", "p1"],
                    "timepoint": ["pre", "post", "pre", "post", "pre"],
                    "fov_id": ["f1", "f2", "f3", "f4", "f5"],
                    "domain_label": ["TC", "IM", "PT", "IM", "TC"],
                }
            ),
        },
    }
    return adata


def _snapshot_adata(adata: AnnData) -> tuple[pd.DataFrame, dict[str, object]]:
    return adata.obs.copy(deep=True), copy.deepcopy(dict(adata.uns))


def _assert_adata_unchanged(
    adata: AnnData,
    snapshot: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    obs_before, uns_before = snapshot
    pd.testing.assert_frame_equal(adata.obs, obs_before)
    config = adata.uns["stride"]["config"]
    config_before = uns_before["stride"]["config"]
    assert config["source"] == config_before["source"]
    assert config["target"] == config_before["target"]
    assert config["community_mode"] == config_before["community_mode"]
    assert config["n_states"] == config_before["n_states"]
    assert config["relation_ids"] == config_before["relation_ids"]
    np.testing.assert_array_equal(config["relations"], config_before["relations"])
    np.testing.assert_allclose(
        adata.uns["stride"]["fov_observations"]["community_composition"],
        uns_before["stride"]["fov_observations"]["community_composition"],
    )
    pd.testing.assert_frame_equal(
        adata.uns["stride"]["fov_observations"]["metadata"],
        uns_before["stride"]["fov_observations"]["metadata"],
    )


def test_pl_import_and_root_api_boundary() -> None:
    assert pl.community_annotation_heatmap is not None
    assert pl.fov_composition_heatmap is not None
    assert pl.community_fraction_comparison is not None
    assert "community_annotation_heatmap" not in stride.__all__


def test_palette_contracts_are_available() -> None:
    assert {"teal", "magenta", "blue", "light_blue"}.issubset(BIO_PASTEL_PALETTE)
    assert {
        "source": BIO_PASTEL_PALETTE["navy"],
        "target": BIO_PASTEL_PALETTE["magenta"],
    } == pl_utils.SOURCE_TARGET_PALETTE
    assert pl_utils._group_color_map(["treated", "control"]) == pl_utils._group_color_map(
        ["control", "treated"]
    )


def test_community_annotation_heatmap_uses_complex_heatmap_and_preserves_input(
    _pycomplexheatmap_available,
) -> None:
    adata = _pl_adata()
    snapshot = _snapshot_adata(adata)

    fig = pl.community_annotation_heatmap(adata)

    assert isinstance(fig, Figure)
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == [
        "C0",
        "C1",
        "C2",
        "C3",
    ]
    assert fig.axes[0].get_xlabel() == "Cell subtype"
    assert fig.axes[0].get_ylabel() == "Community"
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["B", "M", "T"]
    assert any(ax.get_xlabel() == "Patient prevalence" for ax in fig.axes)
    assert any(ax.get_xlabel() == "FOV prevalence" for ax in fig.axes)
    assert any(ax.get_ylabel() == "Cell subtype fraction" for ax in fig.axes)
    legends = [artist for ax in fig.axes for artist in ax.get_children() if isinstance(artist, Legend)]
    assert any(legend.get_title().get_text() == "Domain" for legend in legends)
    assert not _pycomplexheatmap_available
    _assert_adata_unchanged(adata, snapshot)


def test_community_annotation_heatmap_saves_pdf(tmp_path) -> None:
    adata = _pl_adata()
    path = tmp_path / "annotation.pdf"

    returned = pl.community_annotation_heatmap(adata, save=path)

    assert returned is None
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_community_annotation_heatmap_rejects_missing_obs_column() -> None:
    adata = _pl_adata()
    del adata.obs["cell_subtype_label"]

    with pytest.raises(ContractError, match="cell_subtype_label"):
        pl.community_annotation_heatmap(adata)


def test_fov_composition_heatmap_uses_complex_heatmap_and_does_not_sort_slot(
    _pycomplexheatmap_available,
) -> None:
    adata = _pl_adata()
    original_matrix = adata.uns["stride"]["fov_observations"][
        "community_composition"
    ].copy()
    original_metadata = adata.uns["stride"]["fov_observations"]["metadata"].copy()

    fig = pl.fov_composition_heatmap(adata)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "Community"
    assert fig.axes[0].get_ylabel() == "FOV"
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["C0", "C1", "C2", "C3"]
    assert not [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert any(ax.get_ylabel() == "Community fraction" for ax in fig.axes)
    legends = [artist for ax in fig.axes for artist in ax.get_children() if isinstance(artist, Legend)]
    assert any(legend.get_title().get_text() == "Time" for legend in legends)
    assert not _pycomplexheatmap_available
    np.testing.assert_allclose(
        adata.uns["stride"]["fov_observations"]["community_composition"],
        original_matrix,
    )
    pd.testing.assert_frame_equal(
        adata.uns["stride"]["fov_observations"]["metadata"],
        original_metadata,
    )


def test_fov_composition_heatmap_saves_pdf(tmp_path) -> None:
    path = tmp_path / "nested" / "fov.pdf"

    returned = pl.fov_composition_heatmap(_pl_adata(), save=path)

    assert returned is None
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_fov_composition_heatmap_rejects_mismatched_rows() -> None:
    adata = _pl_adata()
    adata.uns["stride"]["fov_observations"]["metadata"] = adata.uns["stride"][
        "fov_observations"
    ]["metadata"].iloc[:-1].copy()

    with pytest.raises(ContractError, match="row count"):
        pl.fov_composition_heatmap(adata)


def test_community_fraction_comparison_fov_and_cell_routes_run() -> None:
    adata = _pl_adata()

    fov_fig = pl.community_fraction_comparison(adata)
    cell_fig = pl.community_fraction_comparison(adata, scale="cell_state_fraction")

    assert isinstance(fov_fig, Figure)
    assert fov_fig.axes[0].get_ylabel() == "Community fraction (FOV mean)"
    assert isinstance(cell_fig, Figure)
    assert cell_fig.axes[0].get_ylabel() == "Community fraction (cell-level)"


def test_community_fraction_comparison_group_and_external_stats_run() -> None:
    adata = _pl_adata()
    groups = {"p1": "treated", "p2": "control"}
    stats = pd.DataFrame(
        {
            "relation_id": ["pre_TC_to_post_IM"],
            "group": ["treated"],
            "community_id": [0],
            "x1": ["source"],
            "x2": ["target"],
            "q_value": [0.01],
        }
    )

    fig = pl.community_fraction_comparison(
        adata,
        group_labels=groups,
        stats=stats,
        show_ns=False,
    )

    assert isinstance(fig, Figure)
    assert any(text.get_text() == "q=0.01 **" for ax in fig.axes for text in ax.texts)


def test_non_pdf_save_path_raises_contract_error(tmp_path) -> None:
    with pytest.raises(ContractError, match="PDF"):
        pl.fov_composition_heatmap(_pl_adata(), save=tmp_path / "plot.png")


def test_pycomplexheatmap_is_declared_as_required_dependency() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert '"PyComplexHeatmap' in pyproject
    assert '"statannotations' in pyproject


def test_fov_composition_heatmap_rejects_invalid_config_n_states() -> None:
    adata = _pl_adata()
    adata.uns["stride"]["config"]["n_states"] = "bad"

    with pytest.raises(ContractError, match="n_states"):
        pl.fov_composition_heatmap(adata)


def test_community_fraction_comparison_rejects_relation_without_support() -> None:
    adata = _pl_adata()
    adata.uns["stride"]["config"]["relations"] = np.asarray([["XX", "YY"]], dtype=object)
    adata.uns["stride"]["config"]["relation_ids"] = ["unsupported"]

    with pytest.raises(ContractError, match="no plottable rows"):
        pl.community_fraction_comparison(adata)


def test_community_fraction_comparison_accepts_custom_stats_mapping_keys() -> None:
    adata = _pl_adata()
    stats = pd.DataFrame(
        {
            "relation_id": ["pre_TC_to_post_IM"],
            "community_id": [0],
            "left_side": ["source"],
            "right_side": ["target"],
            "label": ["baseline"],
        }
    )

    fig = pl.community_fraction_comparison(
        adata,
        stats=stats,
        stats_x1_key="left_side",
        stats_x2_key="right_side",
    )

    assert isinstance(fig, Figure)
    assert any(text.get_text() == "baseline" for ax in fig.axes for text in ax.texts)


def test_descriptive_plots_do_not_mutate_anndata() -> None:
    for plotter in (
        pl.community_annotation_heatmap,
        pl.fov_composition_heatmap,
        pl.community_fraction_comparison,
    ):
        adata = _pl_adata()
        snapshot = _snapshot_adata(adata)

        plotter(adata)

        _assert_adata_unchanged(adata, snapshot)


def test_public_plot_saves_use_tight_bbox(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []
    original_savefig = Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        calls.append(kwargs)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy_savefig)

    pl.community_annotation_heatmap(_pl_adata(), save=tmp_path / "annotation.pdf")
    pl.fov_composition_heatmap(_pl_adata(), save=tmp_path / "fov.pdf")
    pl.community_fraction_comparison(_pl_adata(), save=tmp_path / "fraction.pdf")

    assert len(calls) == 3
    assert all(call["bbox_inches"] == "tight" for call in calls)
    assert all(call["pad_inches"] == 0.08 for call in calls)
