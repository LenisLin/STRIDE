from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from slotar.contracts import COST_SCALE_ALIASES, validate_adata_inputs
from slotar.uot import UOTSolveConfig, precompute_logKernels

from .common import (
    TaskARoiReferenceBundle,
    build_task_a_roi_reference_bundle_from_adata,
    validate_task_a_mass_mode_surface,
)

try:
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover - optional until a runtime path needs it
    ad = None  # type: ignore[assignment]


TASK_A_METRICS_FILENAME = "task_A_metrics.parquet"
TASK_A_RUN_MANIFEST_FILENAME = "task_a_run_manifest.json"
TASK_A_MANIFEST_SCHEMA_VERSION = "task_a_run_manifest.v1"
ARM_ARTIFACT_SUBDIRS: dict[str, str] = {
    "A1_baseline": "arm1_baseline",
    "A1_broken_reference": "arm1_broken_reference",
    "A2_cross_compartment": "arm2_cross_compartment",
    "A3_uq_stress": "arm3_uq_stress",
}
ANALYSIS_SUBDIR = "analysis"
FOCUSED_ANALYSIS_SUBDIR = "focused"
BIOINFORMED_ANALYSIS_SUBDIR = "bioinformed"


@dataclass(frozen=True)
class TaskAArtifactLayout:
    """Canonical artifact layout for one Task-A formal run."""

    run_root: Path
    metrics_parquet: Path
    manifest_path: Path
    arm_artifact_roots: dict[str, Path]
    arm_analysis_roots: dict[str, Path]


@dataclass(frozen=True)
class TaskARunManifest:
    """Serializable formal-run contract used by post-hoc Task-A tools."""

    schema_version: str
    task_name: str
    config_path: Path
    stage0_h5ad: Path
    run_root: Path
    metrics_parquet: Path
    enabled_arms: tuple[str, ...]
    arm_artifact_roots: dict[str, Path]
    arm_analysis_roots: dict[str, Path]

    @classmethod
    def from_runtime(cls, runtime: "TaskARuntimeBundle") -> "TaskARunManifest":
        return cls(
            schema_version=TASK_A_MANIFEST_SCHEMA_VERSION,
            task_name=str(runtime.config.get("task_name", "Task A")),
            config_path=runtime.config_path,
            stage0_h5ad=runtime.data_path,
            run_root=runtime.artifact_layout.run_root,
            metrics_parquet=runtime.artifact_layout.metrics_parquet,
            enabled_arms=runtime.enabled_arms,
            arm_artifact_roots=runtime.artifact_layout.arm_artifact_roots,
            arm_analysis_roots=runtime.artifact_layout.arm_analysis_roots,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "config_path": str(self.config_path),
            "stage0_h5ad": str(self.stage0_h5ad),
            "run_root": str(self.run_root),
            "metrics_parquet": str(self.metrics_parquet),
            "enabled_arms": list(self.enabled_arms),
            "arm_artifact_roots": {
                arm_name: str(path)
                for arm_name, path in sorted(self.arm_artifact_roots.items())
            },
            "arm_analysis_roots": {
                arm_name: str(path)
                for arm_name, path in sorted(self.arm_analysis_roots.items())
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "TaskARunManifest":
        enabled_arms_raw = payload.get("enabled_arms", [])
        if isinstance(enabled_arms_raw, (str, bytes)) or not isinstance(enabled_arms_raw, Sequence):
            raise ValueError("Task-A run manifest enabled_arms must be a sequence of strings")

        return cls(
            schema_version=str(payload.get("schema_version", "")),
            task_name=str(payload.get("task_name", "Task A")),
            config_path=Path(str(payload["config_path"])).expanduser().resolve(),
            stage0_h5ad=Path(str(payload["stage0_h5ad"])).expanduser().resolve(),
            run_root=Path(str(payload["run_root"])).expanduser().resolve(),
            metrics_parquet=Path(str(payload["metrics_parquet"])).expanduser().resolve(),
            enabled_arms=tuple(str(arm_name) for arm_name in enabled_arms_raw),
            arm_artifact_roots={
                str(arm_name): Path(str(path)).expanduser().resolve()
                for arm_name, path in dict(payload.get("arm_artifact_roots", {})).items()
            },
            arm_analysis_roots={
                str(arm_name): Path(str(path)).expanduser().resolve()
                for arm_name, path in dict(payload.get("arm_analysis_roots", {})).items()
            },
        )


@dataclass(slots=True)
class TaskARuntimeBundle:
    """Shared Task-A runtime assets reused across enabled arms in one run."""

    config_path: Path
    data_path: Path
    config: dict[str, Any]
    enabled_arms: tuple[str, ...]
    artifact_layout: TaskAArtifactLayout
    adata: Any | None
    uot_cfg: UOTSolveConfig | None
    kernels: tuple[np.ndarray, ...] | None
    roi_references: TaskARoiReferenceBundle | None


def load_task_a_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Task-A config must deserialize to a mapping")
    return loaded


def load_enabled_arms(config: Mapping[str, Any]) -> tuple[str, ...]:
    enabled_arms = config.get("enabled_arms", [])
    if isinstance(enabled_arms, (str, bytes)) or not isinstance(enabled_arms, Sequence):
        raise ValueError("enabled_arms must be a list-like of strings")

    enabled_arms_list = list(enabled_arms)
    if any(not isinstance(arm_name, str) for arm_name in enabled_arms_list):
        raise ValueError("enabled_arms must be a list-like of strings")
    if not enabled_arms_list:
        raise ValueError("enabled_arms must contain at least one supported Task-A arm")
    if len(set(enabled_arms_list)) != len(enabled_arms_list):
        raise ValueError("enabled_arms must not contain duplicates")
    return tuple(enabled_arms_list)


def build_task_a_uot_config(config: Mapping[str, Any]) -> UOTSolveConfig:
    uot_params = config["uot_params"]
    return UOTSolveConfig(
        eps_schedule=tuple(float(eps) for eps in uot_params["eps_schedule"]),
        max_iter=int(uot_params["max_iter"]),
        tol=float(uot_params["tol"]),
        eta_floor=float(uot_params["eta_floor"]),
        n_min_proto=float(uot_params["n_min_proto"]),
        tau_q=float(uot_params.get("tau_q", 0.25)),
        tau_mode=str(uot_params.get("tau_mode", "external_fixed_by_task")),
    )


def resolve_task_a_artifact_layout(
    run_root: str | Path,
    enabled_arms: Sequence[str],
) -> TaskAArtifactLayout:
    resolved_run_root = Path(run_root).expanduser().resolve()
    enabled = tuple(str(arm_name) for arm_name in enabled_arms)

    if len(enabled) == 1:
        arm_artifact_roots = {enabled[0]: resolved_run_root}
    else:
        arm_artifact_roots = {
            arm_name: resolved_run_root / ARM_ARTIFACT_SUBDIRS.get(arm_name, arm_name)
            for arm_name in enabled
        }

    return TaskAArtifactLayout(
        run_root=resolved_run_root,
        metrics_parquet=resolved_run_root / TASK_A_METRICS_FILENAME,
        manifest_path=resolved_run_root / TASK_A_RUN_MANIFEST_FILENAME,
        arm_artifact_roots=arm_artifact_roots,
        arm_analysis_roots={
            arm_name: arm_root / ANALYSIS_SUBDIR
            for arm_name, arm_root in arm_artifact_roots.items()
        },
    )


def resolve_task_a_manifest_path(path_or_root: str | Path) -> Path:
    candidate = Path(path_or_root).expanduser().resolve()
    if candidate.suffix.lower() == ".json":
        return candidate
    return candidate / TASK_A_RUN_MANIFEST_FILENAME


def load_task_a_run_manifest(path_or_root: str | Path) -> TaskARunManifest:
    manifest_path = resolve_task_a_manifest_path(path_or_root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = TaskARunManifest.from_json_dict(payload)
    if manifest.schema_version != TASK_A_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Task-A run manifest schema_version: "
            f"{manifest.schema_version!r}"
        )
    return manifest


def write_task_a_run_manifest(runtime: TaskARuntimeBundle) -> Path:
    manifest = TaskARunManifest.from_runtime(runtime)
    runtime.artifact_layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.artifact_layout.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_json_dict(), handle, indent=2, sort_keys=True)
    return runtime.artifact_layout.manifest_path


def resolve_task_a_arm_artifact_root(
    contract: TaskARunManifest | TaskAArtifactLayout,
    arm_name: str,
) -> Path:
    try:
        return contract.arm_artifact_roots[arm_name]
    except KeyError as exc:
        raise ValueError(f"Task-A contract does not define arm artifact root for {arm_name!r}") from exc


def resolve_task_a_arm_analysis_root(
    contract: TaskARunManifest | TaskAArtifactLayout,
    arm_name: str,
) -> Path:
    try:
        return contract.arm_analysis_roots[arm_name]
    except KeyError as exc:
        raise ValueError(f"Task-A contract does not define arm analysis root for {arm_name!r}") from exc


def resolve_task_a_arm_focused_output_dir(
    contract: TaskARunManifest | TaskAArtifactLayout,
    arm_name: str,
) -> Path:
    return resolve_task_a_arm_analysis_root(contract, arm_name) / FOCUSED_ANALYSIS_SUBDIR


def resolve_task_a_arm_bioinformed_output_dir(
    contract: TaskARunManifest | TaskAArtifactLayout,
    arm_name: str,
) -> Path:
    return resolve_task_a_arm_analysis_root(contract, arm_name) / BIOINFORMED_ANALYSIS_SUBDIR


def prepare_task_a_runtime(
    config_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    supported_arms: Collection[str],
    arm3_operator_name: str,
) -> TaskARuntimeBundle:
    config_path_resolved = Path(config_path).expanduser().resolve()
    data_path_resolved = Path(data_path).expanduser().resolve()
    config = load_task_a_config(config_path_resolved)
    enabled_arms = load_enabled_arms(config)
    unsupported = [arm_name for arm_name in enabled_arms if arm_name not in supported_arms]
    if unsupported:
        raise NotImplementedError(
            "Patch-2 only supports enabled_arms drawn from "
            f"{sorted(supported_arms)}, got {list(enabled_arms)!r}"
        )

    validate_task_a_mass_mode_surface(config, enabled_arms)
    artifact_layout = resolve_task_a_artifact_layout(output_dir, enabled_arms)
    artifact_layout.run_root.mkdir(parents=True, exist_ok=True)

    adata = None
    uot_cfg = None
    kernels = None
    roi_references = None
    if any(str(arm_name) != arm3_operator_name for arm_name in enabled_arms):
        if ad is None:  # pragma: no cover - exercised only when a runtime path needs AnnData
            raise ModuleNotFoundError("anndata is required for Task-A AnnData-backed runtime paths")

        adata = ad.read_h5ad(data_path_resolved)
        validate_adata_inputs(
            adata,
            require_prototypes=True,
            require_cost_scale=True,
            require_cost_matrix=True,
        )

        k_full = int(config["data"]["k_full"])
        cost_matrix = np.asarray(adata.uns["cost_matrix"], dtype=float)
        if cost_matrix.shape != (k_full, k_full):
            raise ValueError(
                "Task-A cost_matrix shape must match the declared shared prototype axis: "
                f"expected {(k_full, k_full)}, got {cost_matrix.shape}"
            )

        uot_cfg = build_task_a_uot_config(config)
        kernels = tuple(
            precompute_logKernels(
                cost_matrix,
                uot_cfg.eps_schedule,
                s_C=_resolve_cost_scale(adata),
            )
        )
        roi_references = build_task_a_roi_reference_bundle_from_adata(
            adata,
            k_full=k_full,
        )

    return TaskARuntimeBundle(
        config_path=config_path_resolved,
        data_path=data_path_resolved,
        config=config,
        enabled_arms=enabled_arms,
        artifact_layout=artifact_layout,
        adata=adata,
        uot_cfg=uot_cfg,
        kernels=kernels,
        roi_references=roi_references,
    )


def _resolve_cost_scale(adata_obj: Any) -> float:
    if "s_C" in adata_obj.uns:
        return float(adata_obj.uns["s_C"])
    for alias in COST_SCALE_ALIASES:
        if alias in adata_obj.uns:
            return float(adata_obj.uns[alias])
    raise ValueError("Validated AnnData is missing a usable SLOTAR cost scale")
