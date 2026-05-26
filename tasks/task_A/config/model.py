"""Typed Task A config loading and surface validation.

This module freezes the task-local config schema, block selection rules, and
uniform mass-mode requirement. It does not define block algorithms or change
the STRIDE core API.

For the active Block 3 architecture, canonical reruns remain the outer
statistical unit and patients are nested within a generator realization.
Comments and validation rules here must not imply IID `patient x rerun`
treatment. Retired Block 3 assessment-policy thresholds are intentionally not
part of the frozen config surface.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TASK_A_BLOCKS: frozenset[str] = frozenset(
    {
        "block0_locality_gate",
        "block1_real_data_discovery",
    }
)
REMOVED_TASK_A_TOP_LEVEL_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "enabled_arms",
        "arm" + "1",
        "arm" + "2",
        "arm" + "3",
        "A" + "1_baseline",
        "A" + "1_broken_reference",
        "A" + "2_cross_compartment",
        "A" + "3_uq_stress",
    }
)
REMOVED_TASK_A_DATA_CONFIG_KEYS: frozenset[str] = frozenset({"mass_mode_by_arm"})
SUPPORTED_TASK_A_MASS_MODES: tuple[str, ...] = ("uniform",)
REMOVED_TASK_A_BLOCK3_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "strong_support_ratio_threshold",
        "partial_support_ratio_threshold",
        "strong_effect_ratio_threshold",
        "partial_effect_ratio_threshold",
    }
)


@dataclass(frozen=True)
class TaskAOrderedPairFamilySpec:
    name: str
    source_domain: str
    target_domain: str
    claim_role: str
    pair_types: tuple[str, ...]

    @property
    def ordered_group_labels(self) -> tuple[str, str]:
        return (self.source_domain, self.target_domain)


@dataclass(frozen=True)
class TaskAOrderedProxySpec:
    domains: tuple[str, ...]
    pair_families: tuple[TaskAOrderedPairFamilySpec, ...]

    @property
    def confirmatory_pair_families(self) -> tuple[TaskAOrderedPairFamilySpec, ...]:
        return tuple(family for family in self.pair_families if family.claim_role == "confirmatory")

    @property
    def audit_pair_families(self) -> tuple[TaskAOrderedPairFamilySpec, ...]:
        return tuple(family for family in self.pair_families if family.claim_role != "confirmatory")


@dataclass(frozen=True)
class TaskAStage0Config:
    stage0_h5ad: str | None
    build_missing_basis: bool
    default_basis_neighbors: int
    default_geometry_neighbors: int


@dataclass(frozen=True)
class TaskADataConfig:
    mass_mode: str
    k_full: int


@dataclass(frozen=True)
class TaskABlock1Config:
    target_alpha: float
    lambda_grid: tuple[float, ...]


@dataclass(frozen=True)
class TaskABlock3Config:
    """Frozen Block 3 config surface.

    The active Round 2 surface exposes only registry-first architecture
    controls and review community selection. Assessment-policy thresholds are
    retired and must not be reintroduced here.
    """

    master_seed: int
    enabled_subexperiments: tuple[str, ...]
    benchmark_pair_family: str
    review_primary_source_communities: tuple[int, ...]
    review_secondary_source_communities: tuple[int, ...]
    review_primary_target_communities: tuple[int, ...]
    review_secondary_target_communities: tuple[int, ...]


@dataclass(frozen=True)
class TaskAExportConfig:
    mapping_manifest_filename: str
    prepare_manifest_filename: str
    core_fit_dry_run_filename: str


@dataclass(frozen=True)
class TaskABenchmarkConfig:
    default_n_patients: int
    default_seed: int


@dataclass(frozen=True)
class TaskAConfigBundle:
    config_path: Path
    raw_config: dict[str, Any]
    config_fingerprint: str
    enabled_blocks: tuple[str, ...]
    ordered_proxy: TaskAOrderedProxySpec
    data: TaskADataConfig
    block1: TaskABlock1Config
    block3: TaskABlock3Config
    stage0: TaskAStage0Config
    exports: TaskAExportConfig
    benchmarks: TaskABenchmarkConfig

    @property
    def ordered_pair_family_names(self) -> tuple[str, ...]:
        return tuple(family.name for family in self.ordered_proxy.pair_families)


DEFAULT_ORDERED_PAIR_FAMILIES: tuple[TaskAOrderedPairFamilySpec, ...] = (
    TaskAOrderedPairFamilySpec(
        name="TC-IM",
        source_domain="TC",
        target_domain="IM",
        claim_role="confirmatory",
        pair_types=("TC->IM", "IM->TC"),
    ),
    TaskAOrderedPairFamilySpec(
        name="TC-PT",
        source_domain="TC",
        target_domain="PT",
        claim_role="confirmatory",
        pair_types=("TC->PT", "PT->TC"),
    ),
    TaskAOrderedPairFamilySpec(
        name="IM-PT",
        source_domain="IM",
        target_domain="PT",
        claim_role="audit_only",
        pair_types=("IM->PT", "PT->IM"),
    ),
)

DEFAULT_BLOCK1_TARGET_ALPHA = 0.05
DEFAULT_BLOCK1_LAMBDA_GRID: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_BLOCK3_MASTER_SEED = 17
DEFAULT_BLOCK3_REVIEW_PRIMARY_SOURCE_COMMUNITIES: tuple[int, ...] = (1, 10, 11, 12)
DEFAULT_BLOCK3_REVIEW_SECONDARY_SOURCE_COMMUNITIES: tuple[int, ...] = (0, 3, 6, 16, 17)
DEFAULT_BLOCK3_REVIEW_PRIMARY_TARGET_COMMUNITIES: tuple[int, ...] = (23, 20, 2, 22)
DEFAULT_BLOCK3_REVIEW_SECONDARY_TARGET_COMMUNITIES: tuple[int, ...] = (14, 13, 21)
DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS: tuple[str, ...] = ("3A", "3B", "3C")
DEFAULT_BLOCK3_BENCHMARK_PAIR_FAMILY = "TC-IM"


def load_raw_task_a_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Task-A config must deserialize to a mapping")
    return loaded


def compute_task_a_config_fingerprint(config: Mapping[str, Any]) -> str:
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_enabled_blocks(config: Mapping[str, Any]) -> tuple[str, ...]:
    enabled_blocks = config.get("enabled_blocks", [])
    if isinstance(enabled_blocks, (str, bytes)) or not isinstance(enabled_blocks, Sequence):
        raise ValueError("enabled_blocks must be a list-like of strings")

    enabled_blocks_list = [str(block_name) for block_name in enabled_blocks]
    if not enabled_blocks_list:
        raise ValueError("enabled_blocks must contain at least one supported Task-A block")
    if len(set(enabled_blocks_list)) != len(enabled_blocks_list):
        raise ValueError("enabled_blocks must not contain duplicates")
    unsupported = sorted(set(enabled_blocks_list) - SUPPORTED_TASK_A_BLOCKS)
    if unsupported:
        raise ValueError(
            "enabled_blocks contains unsupported Task-A blocks: "
            f"{unsupported}"
        )
    return tuple(enabled_blocks_list)


def validate_task_a_config_surface(config: Mapping[str, Any]) -> None:
    removed_keys = sorted(key for key in REMOVED_TASK_A_TOP_LEVEL_CONFIG_KEYS if key in config)
    if removed_keys:
        raise ValueError(
            "Task-A config still uses removed keys: "
            f"{removed_keys}. Rewrite the config around the Stage 0 and block-local surfaces."
        )

    data_cfg_raw = config.get("data", {})
    if data_cfg_raw is None:
        data_cfg_raw = {}
    if not isinstance(data_cfg_raw, Mapping):
        raise ValueError("Task-A config key 'data' must be a mapping")

    removed_data_keys = sorted(key for key in REMOVED_TASK_A_DATA_CONFIG_KEYS if key in data_cfg_raw)
    if removed_data_keys:
        raise ValueError(
            "Task-A data config still uses removed keys: "
            f"{removed_data_keys}. Rewrite the config around the Stage 0 and block-local surfaces."
        )

    if "mass_mode" not in data_cfg_raw:
        raise ValueError("Task-A config is missing data.mass_mode")
    if "k_full" not in data_cfg_raw:
        raise ValueError("Task-A config is missing data.k_full")
    mass_mode = str(data_cfg_raw.get("mass_mode", "")).strip()
    if mass_mode not in SUPPORTED_TASK_A_MASS_MODES:
        raise ValueError(
            "Task-A Step 1 requires data.mass_mode='uniform' to keep observation-layer "
            f"mass semantics honest; got {mass_mode!r}"
        )
    load_enabled_blocks(config)


def _coerce_pair_family_specs(config: Mapping[str, Any]) -> TaskAOrderedProxySpec:
    ordered_proxy_raw = config.get("ordered_proxy", {})
    if ordered_proxy_raw is None:
        ordered_proxy_raw = {}
    if not isinstance(ordered_proxy_raw, Mapping):
        raise ValueError("Task-A config key 'ordered_proxy' must be a mapping")

    domains_raw = ordered_proxy_raw.get("domains", ("TC", "IM", "PT"))
    if isinstance(domains_raw, (str, bytes)) or not isinstance(domains_raw, Sequence):
        raise ValueError("ordered_proxy.domains must be a sequence of strings")
    domains = tuple(str(domain).strip() for domain in domains_raw)
    if any(domain == "" for domain in domains):
        raise ValueError("ordered_proxy.domains must not contain empty values")

    pair_families_raw = ordered_proxy_raw.get("pair_families")
    if pair_families_raw is None:
        return TaskAOrderedProxySpec(
            domains=domains,
            pair_families=DEFAULT_ORDERED_PAIR_FAMILIES,
        )
    if isinstance(pair_families_raw, (str, bytes)) or not isinstance(pair_families_raw, Sequence):
        raise ValueError("ordered_proxy.pair_families must be a sequence of mappings")

    pair_families: list[TaskAOrderedPairFamilySpec] = []
    for item in pair_families_raw:
        if not isinstance(item, Mapping):
            raise ValueError("ordered_proxy.pair_families must contain mappings")
        pair_types_raw = item.get("pair_types", ())
        if isinstance(pair_types_raw, (str, bytes)) or not isinstance(pair_types_raw, Sequence):
            raise ValueError("ordered_proxy.pair_families[].pair_types must be a sequence")
        spec = TaskAOrderedPairFamilySpec(
            name=str(item.get("name", "")).strip(),
            source_domain=str(item.get("source_domain", "")).strip(),
            target_domain=str(item.get("target_domain", "")).strip(),
            claim_role=str(item.get("claim_role", "audit_only")).strip() or "audit_only",
            pair_types=tuple(str(pair_type).strip() for pair_type in pair_types_raw),
        )
        if not spec.name or not spec.source_domain or not spec.target_domain:
            raise ValueError("ordered_proxy.pair_families entries must define name/source_domain/target_domain")
        if not spec.pair_types:
            raise ValueError(f"ordered_proxy.pair_families[{spec.name!r}] must define at least one pair_type")
        pair_families.append(spec)

    names = [family.name for family in pair_families]
    if len(set(names)) != len(names):
        raise ValueError("ordered_proxy.pair_families names must be unique")
    return TaskAOrderedProxySpec(domains=domains, pair_families=tuple(pair_families))


def _coerce_stage0_config(config: Mapping[str, Any]) -> TaskAStage0Config:
    stage0_raw = config.get("stage0", {})
    if stage0_raw is None:
        stage0_raw = {}
    if not isinstance(stage0_raw, Mapping):
        raise ValueError("Task-A config key 'stage0' must be a mapping")
    stage0_h5ad = stage0_raw.get("stage0_h5ad")
    return TaskAStage0Config(
        stage0_h5ad=None if stage0_h5ad in (None, "") else str(stage0_h5ad),
        build_missing_basis=bool(stage0_raw.get("build_missing_basis", False)),
        default_basis_neighbors=int(stage0_raw.get("default_basis_neighbors", 20)),
        default_geometry_neighbors=int(stage0_raw.get("default_geometry_neighbors", 5)),
    )


def _coerce_data_config(config: Mapping[str, Any]) -> TaskADataConfig:
    data_raw = config.get("data", {})
    if data_raw is None:
        data_raw = {}
    if not isinstance(data_raw, Mapping):
        raise ValueError("Task-A config key 'data' must be a mapping")

    mass_mode = str(data_raw.get("mass_mode", "")).strip()
    if mass_mode not in SUPPORTED_TASK_A_MASS_MODES:
        raise ValueError(
            "Task-A Step 1 requires data.mass_mode='uniform' to keep observation-layer "
            f"mass semantics honest; got {mass_mode!r}"
        )
    if "k_full" not in data_raw:
        raise ValueError("Task-A config is missing data.k_full")
    return TaskADataConfig(
        mass_mode=mass_mode,
        k_full=int(data_raw["k_full"]),
    )


def _coerce_block1_config(config: Mapping[str, Any]) -> TaskABlock1Config:
    block1_raw = config.get("block1", {})
    if block1_raw is None:
        block1_raw = {}
    if not isinstance(block1_raw, Mapping):
        raise ValueError("Task-A config key 'block1' must be a mapping")

    try:
        target_alpha = float(block1_raw.get("target_alpha", DEFAULT_BLOCK1_TARGET_ALPHA))
    except (TypeError, ValueError) as exc:
        raise ValueError("Task-A config key 'block1.target_alpha' must be a finite float") from exc
    if not math.isfinite(target_alpha):
        raise ValueError("Task-A config key 'block1.target_alpha' must be a finite float")
    if not (0.0 < target_alpha < 1.0):
        raise ValueError("Task-A config key 'block1.target_alpha' must lie strictly between 0 and 1")

    lambda_grid_raw = block1_raw.get("lambda_grid", DEFAULT_BLOCK1_LAMBDA_GRID)
    if isinstance(lambda_grid_raw, (str, bytes)) or not isinstance(lambda_grid_raw, Sequence):
        raise ValueError("Task-A config key 'block1.lambda_grid' must be a sequence of positive finite floats")
    try:
        lambda_grid = tuple(float(value) for value in lambda_grid_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Task-A config key 'block1.lambda_grid' must be a sequence of positive finite floats"
        ) from exc
    if not lambda_grid:
        raise ValueError("Task-A config key 'block1.lambda_grid' must not be empty")
    if any((not math.isfinite(value)) or value <= 0.0 for value in lambda_grid):
        raise ValueError("Task-A config key 'block1.lambda_grid' must contain only positive finite floats")
    return TaskABlock1Config(
        target_alpha=target_alpha,
        lambda_grid=lambda_grid,
    )


def _coerce_int_sequence(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Task-A config key {field_name!r} must be a sequence of integers")
    try:
        resolved = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Task-A config key {field_name!r} must be a sequence of integers") from exc
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Task-A config key {field_name!r} must not contain duplicates")
    return resolved


def _coerce_block3_subexperiment_sequence(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    supported_subexperiments = frozenset(DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Task-A config key {field_name!r} must be a sequence of subexperiment ids")
    resolved = tuple(str(item).strip() for item in value)
    if not resolved:
        raise ValueError(f"Task-A config key {field_name!r} must not be empty")
    if any(not item for item in resolved):
        raise ValueError(f"Task-A config key {field_name!r} must not contain empty subexperiment ids")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Task-A config key {field_name!r} must not contain duplicates")
    unsupported = sorted(set(resolved) - supported_subexperiments)
    if unsupported:
        raise ValueError(
            f"Task-A config key {field_name!r} contains unsupported Block 3 subexperiments: {unsupported}"
        )
    return resolved


def _coerce_block3_benchmark_pair_family(
    block3_raw: Mapping[str, Any],
    *,
    ordered_proxy: TaskAOrderedProxySpec,
) -> str:
    if "benchmark_pair_family" not in block3_raw:
        raise ValueError("Task-A config is missing block3.benchmark_pair_family")
    benchmark_pair_family = str(block3_raw.get("benchmark_pair_family", "")).strip()
    if not benchmark_pair_family:
        raise ValueError("Task-A config key 'block3.benchmark_pair_family' must not be empty")

    ordered_family_names = tuple(family.name for family in ordered_proxy.pair_families)
    if benchmark_pair_family not in ordered_family_names:
        raise ValueError(
            "Task-A config key 'block3.benchmark_pair_family' must name an ordered pair family; "
            f"got {benchmark_pair_family!r}, available families are {ordered_family_names!r}"
        )
    if benchmark_pair_family != DEFAULT_BLOCK3_BENCHMARK_PAIR_FAMILY:
        raise ValueError(
            "Task-A Block 3 currently accepts only "
            f"block3.benchmark_pair_family={DEFAULT_BLOCK3_BENCHMARK_PAIR_FAMILY!r}; "
            f"got {benchmark_pair_family!r}"
        )
    return benchmark_pair_family


def _coerce_block3_config(
    config: Mapping[str, Any],
    *,
    ordered_proxy: TaskAOrderedProxySpec,
) -> TaskABlock3Config:
    block3_raw = config.get("block3", {})
    if block3_raw is None:
        block3_raw = {}
    if not isinstance(block3_raw, Mapping):
        raise ValueError("Task-A config key 'block3' must be a mapping")

    removed_block3_keys = sorted(key for key in REMOVED_TASK_A_BLOCK3_CONFIG_KEYS if key in block3_raw)
    if removed_block3_keys:
        raise ValueError(
            "Task-A config key 'block3' still uses retired assessment-policy fields: "
            f"{removed_block3_keys}. The active Block 3 architecture now freezes raw metric and review/export routing only."
        )

    block3_config = TaskABlock3Config(
        master_seed=int(block3_raw.get("master_seed", DEFAULT_BLOCK3_MASTER_SEED)),
        enabled_subexperiments=_coerce_block3_subexperiment_sequence(
            block3_raw.get("enabled_subexperiments", DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS),
            field_name="block3.enabled_subexperiments",
        ),
        benchmark_pair_family=_coerce_block3_benchmark_pair_family(
            block3_raw,
            ordered_proxy=ordered_proxy,
        ),
        review_primary_source_communities=_coerce_int_sequence(
            block3_raw.get(
                "review_primary_source_communities",
                DEFAULT_BLOCK3_REVIEW_PRIMARY_SOURCE_COMMUNITIES,
            ),
            field_name="block3.review_primary_source_communities",
        ),
        review_secondary_source_communities=_coerce_int_sequence(
            block3_raw.get(
                "review_secondary_source_communities",
                DEFAULT_BLOCK3_REVIEW_SECONDARY_SOURCE_COMMUNITIES,
            ),
            field_name="block3.review_secondary_source_communities",
        ),
        review_primary_target_communities=_coerce_int_sequence(
            block3_raw.get(
                "review_primary_target_communities",
                DEFAULT_BLOCK3_REVIEW_PRIMARY_TARGET_COMMUNITIES,
            ),
            field_name="block3.review_primary_target_communities",
        ),
        review_secondary_target_communities=_coerce_int_sequence(
            block3_raw.get(
                "review_secondary_target_communities",
                DEFAULT_BLOCK3_REVIEW_SECONDARY_TARGET_COMMUNITIES,
            ),
            field_name="block3.review_secondary_target_communities",
        ),
    )
    return block3_config


def _coerce_export_config(config: Mapping[str, Any]) -> TaskAExportConfig:
    exports_raw = config.get("exports", {})
    if exports_raw is None:
        exports_raw = {}
    if not isinstance(exports_raw, Mapping):
        raise ValueError("Task-A config key 'exports' must be a mapping")
    return TaskAExportConfig(
        mapping_manifest_filename=str(exports_raw.get("mapping_manifest_filename", "task_a_stride_mapping.json")),
        prepare_manifest_filename=str(exports_raw.get("prepare_manifest_filename", "task_a_prepare_manifest.json")),
        core_fit_dry_run_filename=str(exports_raw.get("core_fit_dry_run_filename", "task_a_core_fit_dry_run.csv")),
    )


def _coerce_benchmark_config(config: Mapping[str, Any]) -> TaskABenchmarkConfig:
    benchmarks_raw = config.get("benchmarks", {})
    if benchmarks_raw is None:
        benchmarks_raw = {}
    if not isinstance(benchmarks_raw, Mapping):
        raise ValueError("Task-A config key 'benchmarks' must be a mapping")
    return TaskABenchmarkConfig(
        default_n_patients=int(benchmarks_raw.get("default_n_patients", 6)),
        default_seed=int(benchmarks_raw.get("default_seed", 0)),
    )


def load_task_a_config_bundle(config_path: str | Path) -> TaskAConfigBundle:
    resolved_path = Path(config_path).expanduser().resolve()
    raw_config = load_raw_task_a_config(resolved_path)
    validate_task_a_config_surface(raw_config)
    ordered_proxy = _coerce_pair_family_specs(raw_config)
    return TaskAConfigBundle(
        config_path=resolved_path,
        raw_config=raw_config,
        config_fingerprint=compute_task_a_config_fingerprint(raw_config),
        enabled_blocks=load_enabled_blocks(raw_config),
        ordered_proxy=ordered_proxy,
        data=_coerce_data_config(raw_config),
        block1=_coerce_block1_config(raw_config),
        block3=_coerce_block3_config(raw_config, ordered_proxy=ordered_proxy),
        stage0=_coerce_stage0_config(raw_config),
        exports=_coerce_export_config(raw_config),
        benchmarks=_coerce_benchmark_config(raw_config),
    )


__all__ = [
    "DEFAULT_ORDERED_PAIR_FAMILIES",
    "SUPPORTED_TASK_A_BLOCKS",
    "TaskABenchmarkConfig",
    "TaskABlock1Config",
    "TaskABlock3Config",
    "TaskAConfigBundle",
    "TaskADataConfig",
    "TaskAExportConfig",
    "TaskAOrderedPairFamilySpec",
    "TaskAOrderedProxySpec",
    "TaskAStage0Config",
    "SUPPORTED_TASK_A_MASS_MODES",
    "DEFAULT_BLOCK1_LAMBDA_GRID",
    "DEFAULT_BLOCK1_TARGET_ALPHA",
    "DEFAULT_BLOCK3_MASTER_SEED",
    "DEFAULT_BLOCK3_REVIEW_PRIMARY_SOURCE_COMMUNITIES",
    "DEFAULT_BLOCK3_REVIEW_SECONDARY_SOURCE_COMMUNITIES",
    "DEFAULT_BLOCK3_REVIEW_PRIMARY_TARGET_COMMUNITIES",
    "DEFAULT_BLOCK3_REVIEW_SECONDARY_TARGET_COMMUNITIES",
    "DEFAULT_BLOCK3_ENABLED_SUBEXPERIMENTS",
    "DEFAULT_BLOCK3_BENCHMARK_PAIR_FAMILY",
    "compute_task_a_config_fingerprint",
    "load_enabled_blocks",
    "load_raw_task_a_config",
    "load_task_a_config_bundle",
    "validate_task_a_config_surface",
]
