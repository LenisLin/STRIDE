from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slotar.io.bridge import DataContractError, save_for_r


def _valid_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_group_id": "g1",
                "uot_status": "ok",
                "T": 1.0,
                "D_pos": 0.1,
                "B_pos": 0.2,
                "d_rel": 0.1,
                "b_rel": 0.2,
                "M": 0.3,
                "R": float("nan"),
                "tau": float("nan"),
            }
        ]
    )


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_group_id": "g1",
                "event_type": "remodeling",
                "source_proto": 0,
                "target_proto": 1,
            }
        ]
    )


def test_save_for_r_writes_canonical_bridge_artifacts(tmp_path: Path) -> None:
    paths = save_for_r(
        metrics_df=_valid_metrics(),
        events_df=_valid_events(),
        output_dir=tmp_path / "bridge_out",
        meta_audit={"run_id": "task-a-demo"},
        aux_tables={"baseline_summary": pd.DataFrame({"row_id": [1], "value": [2.0]})},
    )

    assert paths["metrics"].name == "metrics_.csv"
    assert paths["events"].name == "events_.parquet"
    assert paths["meta"].name == "meta_.json"
    assert paths["metrics"].exists()
    assert paths["events"].exists()
    assert paths["meta"].exists()
    assert paths["aux:baseline_summary"].exists()

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["schema_version"] == "v2.0"
    assert meta["run_id"] == "task-a-demo"
    assert meta["artifacts"]["metrics"] == "metrics_.csv"
    assert meta["artifacts"]["events"] == "events_.parquet"
    assert meta["aux_tables"]["baseline_summary"] == "aux_baseline_summary.csv"


def test_save_for_r_preserves_existing_schema_version(tmp_path: Path) -> None:
    paths = save_for_r(
        metrics_df=_valid_metrics(),
        events_df=_valid_events(),
        output_dir=tmp_path / "bridge_out",
        meta_audit={"schema_version": "custom-vX"},
    )
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["schema_version"] == "custom-vX"


def test_save_for_r_enforces_contracts_before_writing(tmp_path: Path) -> None:
    bad_metrics = pd.DataFrame({"uot_status": ["ok"]})
    with pytest.raises(DataContractError, match="patient_group_id"):
        save_for_r(
            metrics_df=bad_metrics,
            events_df=_valid_events(),
            output_dir=tmp_path / "bridge_out",
            meta_audit={},
        )
