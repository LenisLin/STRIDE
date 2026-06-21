from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stride._schema import (
    OBS_DOMAIN_KEY,
    OBS_FOV_KEY,
    OBS_PATIENT_KEY,
    OBS_TIMEPOINT_KEY,
)
from stride.errors import ContractError
from stride.tl._resolve import (
    _build_evidence_blocks,
    _partition_subbags,
    resolve_relation,
)


def _metadata(rows: list[dict[str, str]], *, index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows, index=index)


def test_resolve_relation_builds_patient_blocks() -> None:
    metadata = _metadata(
        [
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "fov_b",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "fov_a",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "fov_c",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "fov_d",
            },
        ]
    )
    community_composition = np.arange(12, dtype=float).reshape(4, 3)

    relation = resolve_relation(
        relation_id="TC_to_IM",
        source_timepoint="pre",
        target_timepoint="post",
        source_domain="TC",
        target_domain="IM",
        metadata=metadata,
        community_composition=community_composition,
    )

    assert relation.patient_ids == ("p1",)
    assert len(relation.blocks) == 2
    assert relation.support_counts == {"p1": {"source": 2, "target": 2}}
    assert relation.metadata["n_eligible_patients"] == 1
    assert relation.metadata["n_skipped_patients"] == 0


def test_resolve_relation_uses_row_positions_not_dataframe_index() -> None:
    metadata = _metadata(
        [
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "source_b",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "target",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "source_a",
            },
        ],
        index=[10, 20, 30],
    )
    community_composition = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    relation = resolve_relation(
        relation_id="TC_to_IM",
        source_timepoint="pre",
        target_timepoint="post",
        source_domain="TC",
        target_domain="IM",
        metadata=metadata,
        community_composition=community_composition,
    )

    block = relation.blocks[0]
    np.testing.assert_array_equal(block.source_bag, np.array([[0.5, 0.5], [1.0, 0.0]]))
    np.testing.assert_array_equal(block.target_bag, np.array([[0.0, 1.0]]))
    assert block.metadata["source_row_indices"] == (2, 0)
    assert block.metadata["target_row_indices"] == (1,)


def test_resolve_relation_skips_patient_missing_one_side() -> None:
    metadata = _metadata(
        [
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "p1_source",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "p1_target",
            },
            {
                OBS_PATIENT_KEY: "p2",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "p2_source",
            },
        ]
    )
    community_composition = np.eye(3, dtype=float)

    with pytest.warns(UserWarning, match="skipped patients missing source or target support"):
        relation = resolve_relation(
            relation_id="TC_to_IM",
            source_timepoint="pre",
            target_timepoint="post",
            source_domain="TC",
            target_domain="IM",
            metadata=metadata,
            community_composition=community_composition,
        )

    assert relation.patient_ids == ("p1",)
    assert relation.skipped_patient_ids == ("p2",)
    assert relation.support_counts["p2"] == {"source": 1, "target": 0}


def test_resolve_relation_warns_and_returns_empty_when_no_eligible_patient() -> None:
    metadata = _metadata(
        [
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "p1_source",
            },
            {
                OBS_PATIENT_KEY: "p2",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "p2_target",
            },
        ]
    )
    community_composition = np.eye(2, dtype=float)

    with pytest.warns(UserWarning) as records:
        relation = resolve_relation(
            relation_id="TC_to_IM",
            source_timepoint="pre",
            target_timepoint="post",
            source_domain="TC",
            target_domain="IM",
            metadata=metadata,
            community_composition=community_composition,
        )

    assert relation.patient_ids == ()
    assert relation.blocks == ()
    assert relation.skipped_patient_ids == ("p1", "p2")
    messages = [str(record.message) for record in records]
    assert any("skipped patients missing source or target support" in msg for msg in messages)
    assert any("has no eligible patients" in msg for msg in messages)


def test_partition_subbags_balanced_nonempty() -> None:
    assert _partition_subbags((0, 1, 2, 3, 4), 2) == ((0, 1, 2), (3, 4))


@pytest.mark.parametrize("n_parts", [0, True, 4])
def test_partition_subbags_rejects_invalid_count(n_parts: int | bool) -> None:
    with pytest.raises(ContractError):
        _partition_subbags((0, 1, 2), n_parts)


@pytest.mark.parametrize(
    ("source_rows", "target_rows"),
    [
        ((), (1,)),
        ((0,), ()),
    ],
)
def test_build_evidence_blocks_rejects_empty_side(
    source_rows: tuple[int, ...],
    target_rows: tuple[int, ...],
) -> None:
    metadata = _metadata(
        [
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "pre",
                OBS_DOMAIN_KEY: "TC",
                OBS_FOV_KEY: "source",
            },
            {
                OBS_PATIENT_KEY: "p1",
                OBS_TIMEPOINT_KEY: "post",
                OBS_DOMAIN_KEY: "IM",
                OBS_FOV_KEY: "target",
            },
        ]
    )

    with pytest.raises(ContractError, match="non-empty source and target"):
        _build_evidence_blocks(
            patient_id="p1",
            source_rows=source_rows,
            target_rows=target_rows,
            metadata=metadata,
            community_composition=np.eye(2, dtype=float),
        )


def test_resolve_relation_does_not_import_stride_beta() -> None:
    source = Path("src/stride/tl/_resolve.py").read_text(encoding="utf-8")

    assert "stride_beta" not in source
