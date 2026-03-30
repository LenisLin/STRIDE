"""Ablation hooks for recurrence, emergence, and transport-first comparators."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for one ablation sweep."""

    enabled: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def run_ablation_suite(
    baseline_metrics: Mapping[str, float],
    *,
    ablations: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Collect baseline and ablation metrics into one comparable payload."""
    return {
        "baseline": {str(key): float(value) for key, value in baseline_metrics.items()},
        **{
            str(name): {str(key): float(value) for key, value in metrics.items()}
            for name, metrics in ablations.items()
        },
    }


def compare_to_ablation(
    suite: Mapping[str, Mapping[str, float]],
    *,
    baseline_name: str = "baseline",
) -> dict[str, dict[str, float]]:
    """Compute metric deltas relative to the named baseline entry."""
    baseline = suite[baseline_name]
    comparison: dict[str, dict[str, float]] = {}
    for name, metrics in suite.items():
        if name == baseline_name:
            continue
        comparison[name] = {
            key: float(metrics[key]) - float(baseline[key])
            for key in baseline.keys()
            if key in metrics
        }
    return comparison


__all__ = ["AblationConfig", "compare_to_ablation", "run_ablation_suite"]

