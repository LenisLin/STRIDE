"""Write frozen semisynthetic benchmark artifacts for Task A.

This export surface writes a manifest CSV and companion contract JSON for the
deterministic semi-synthetic benchmark. It does not participate in the
real-data execution graph.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import CONTRACT_PASSED_STATE
from ..semisynthetic.benchmark import build_same_marginals_world_pair, build_semisynthetic_manifest


def write_semisynthetic_artifacts(
    *,
    output_root: Path,
    manifest_filename: str,
    n_patients: int,
    seed: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_semisynthetic_manifest(n_patients=n_patients, seed=seed)
    manifest_path = output_root / manifest_filename
    manifest.to_csv(manifest_path, index=False)

    stronger, weaker = build_same_marginals_world_pair()
    (output_root / "task_a_semisynthetic_contract.json").write_text(
        json.dumps(
            {
                "artifact_state": CONTRACT_PASSED_STATE,
                "n_patients": int(n_patients),
                "seed": int(seed),
                "same_marginals_pair_family": stronger.pair_family,
                "stronger_continuity_score": float(stronger.relation_matrix.trace() / stronger.relation_matrix.sum()),
                "weaker_continuity_score": float(weaker.relation_matrix.trace() / weaker.relation_matrix.sum()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["write_semisynthetic_artifacts"]
