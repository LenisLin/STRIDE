"""Native output carriers for Task A Block 3b internal Phase 3.

Task and purpose:
    Define small serialization helpers for method-native outputs used by the
    `3B-1 A benchmark` and `3B-2 d/e benchmark` architecture scaffold.

Relevant document anchors:
    - docs/task_A_spec.md §4.5.5-§4.5.6 and §5.1 Phase 3
    - docs/task_A_block3_redesign_v1_1.md §4.3, §5.5, §5.6

Expected inputs and outputs:
    Inputs are numpy-compatible native arrays for STRIDE `A/d/e` outputs and
    plan-based comparator `P` outputs. Outputs are JSON strings or `None` for
    flat raw artifact records, including optional `P_json`.

Internal Phase 3 boundary:
    This module supports only internal scaffold artifacts and typed carriers.
    It is not a public API, workflow entrypoint, runner, review command, or
    result-packet bridge authority.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NativeMethodArrays:
    """Container for STRIDE-native arrays and optional plan diagnostics.

    Inputs are numpy arrays already produced by method execution. Outputs are
    consumed by artifact serializers as `A/d/e`, optional native `P`, and
    optional JSON-compatible metadata. Missing arrays indicate non-estimable
    method status rather than a numerical zero.
    """

    A: np.ndarray | None = None
    d: np.ndarray | None = None
    e: np.ndarray | None = None
    P: np.ndarray | None = None
    metadata: dict[str, object] | None = None


def array_to_json(array: np.ndarray | None) -> str | None:
    """Serialize an optional array input to compact JSON or `None` output."""

    if array is None:
        return None
    return json.dumps(np.asarray(array, dtype=float).tolist(), separators=(",", ":"))


def plan_to_json(plan: np.ndarray | None) -> str | None:
    """Serialize optional native plan `P` for `method_native_output_store`."""

    return array_to_json(plan)


def metadata_to_json(metadata: dict[str, object] | None) -> str | None:
    """Serialize optional method metadata for flat artifact storage."""

    if metadata is None:
        return None
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
