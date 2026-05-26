"""Task A Block 3 internal rebuild package.

Role:
    Provide the narrow package-level import surface for the internal Block 3
    rebuild stack.

Authority anchors:
    - docs/task_A/spec.md §4.5.2, §4.5.5, §4.5.6, §5.1 Phase 3
    - docs/task_A/block3/scientific_contract.md §4.1-§4.4, §5.5, §5.6

Local boundary:
    - This module exposes only registry-first contract objects that other
      modules may import safely.
    - It does not expose execution, raw-bundle, or review-surface writers as a
      package-level stable API.
    - It does not reopen the public Block 3 runner, review workflow, or packet
      bridge.

Primary contents:
    - Frozen enum exports used across Block 3 callers.
    - The live registry accessor for the internal rebuild package.

Why this module exists:
    The rest of the Block 3 package contains execution and artifact-writing
    helpers whose interfaces may evolve as the internal Phase 3 surface matures.
    Keeping `__all__` small lets downstream imports depend on the frozen
    vocabulary and registry entrypoint without implicitly depending on the
    implementation modules.
"""
from __future__ import annotations

from importlib import import_module

# Export only the frozen vocabulary and the live registry accessor. Execution,
# raw-bundle, and review writers remain internal implementation surfaces.
__all__ = [
    "Block3MethodName",
    "Block3MetricName",
    "Block3SubexperimentId",
    "MetricStatus",
    "get_live_block3_registry",
]


def __getattr__(name: str) -> object:
    """Lazily expose the frozen vocabulary without loading execution helpers."""

    if name in {
        "Block3MethodName",
        "Block3MetricName",
        "Block3SubexperimentId",
        "MetricStatus",
    }:
        return getattr(import_module(".contracts", __name__), name)
    if name == "get_live_block3_registry":
        return getattr(import_module(".registry", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
