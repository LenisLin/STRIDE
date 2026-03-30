from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tasks.task_A.arm2.analysis_contract import ARM_NAME, Arm2FocusedPaths
from tasks.task_A.arm2.analysis_response import (
    build_corrected_output_package_from_existing_dir,
    build_corrected_output_package_from_legacy_package,
    can_rebuild_from_existing_focused_dir,
    write_corrected_output_package,
)
from tasks.task_A.runtime_contract import (
    TASK_A_METRICS_FILENAME,
    load_task_a_run_manifest,
    resolve_task_a_arm_focused_output_dir,
)

DEFAULT_TASK_A_ROOT = Path("/mnt/NAS_21T/ProjectResult/SLOTAR/task_A")
DEFAULT_ARM2_RUN_ROOT = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment"
DEFAULT_STAGE0_PATH = Path("/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_OUTPUT_DIR = DEFAULT_ARM2_RUN_ROOT / "analysis" / "focused"
FOCUSED_OUTPUT_SUBDIR = "focused"
DEFAULT_ARM2_CANDIDATES: tuple[Path, ...] = (
    DEFAULT_ARM2_RUN_ROOT / TASK_A_METRICS_FILENAME,
    DEFAULT_TASK_A_ROOT / TASK_A_METRICS_FILENAME,
    DEFAULT_ARM2_RUN_ROOT / "task_A_arm2_metrics.parquet",
)


def _read_arm_column(path: Path) -> set[str]:
    try:
        arms = pd.read_parquet(path, columns=["arm"])["arm"].astype(str)
    except (KeyError, ValueError, OSError):
        df = pd.read_parquet(path)
        if "arm" not in df.columns:
            return set()
        arms = df["arm"].astype(str)
    return set(arms.dropna().unique().tolist())


def _validate_arm2_parquet(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Arm-II parquet does not exist: {resolved}")
    if ARM_NAME not in _read_arm_column(resolved):
        raise ValueError(f"Parquet does not contain arm={ARM_NAME!r}: {resolved}")
    return resolved


def discover_arm2_parquet(explicit_path: str | None, search_root: Path) -> Path:
    if explicit_path is not None:
        return _validate_arm2_parquet(Path(explicit_path))

    for path in DEFAULT_ARM2_CANDIDATES:
        if path.exists() and ARM_NAME in _read_arm_column(path):
            return path.resolve()

    discovered: list[Path] = []
    if search_root.exists():
        for path in sorted(search_root.rglob("*.parquet")):
            if "analysis" in path.parts:
                continue
            resolved = path.resolve()
            if ARM_NAME in _read_arm_column(resolved):
                discovered.append(resolved)

    if not discovered:
        raise FileNotFoundError(
            "No Arm-II parquet containing arm='A2_cross_compartment' was found under "
            f"{search_root}. Pass --input-parquet explicitly."
        )
    if len(discovered) > 1:
        raise FileExistsError(
            "Multiple Arm-II parquets were found and no manifest/default candidate resolved the ambiguity. "
            f"Pass --input-parquet explicitly. Found: {discovered}"
        )
    return discovered[0]


def _resolve_focused_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.name == FOCUSED_OUTPUT_SUBDIR:
        return resolved
    return resolved / FOCUSED_OUTPUT_SUBDIR


def _load_manifest_from_args(args: argparse.Namespace):
    if args.task_a_manifest is not None:
        return load_task_a_run_manifest(args.task_a_manifest)
    if args.task_a_run_root is not None:
        return load_task_a_run_manifest(args.task_a_run_root)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the operative Arm-II real-data mirror rebuild workflow.",
    )
    parser.add_argument(
        "--task-a-manifest",
        default=None,
        help="Optional Task-A run manifest JSON. When provided, it is the primary source of runtime/artifact paths.",
    )
    parser.add_argument(
        "--task-a-run-root",
        default=None,
        help="Optional Task-A formal run root containing task_a_run_manifest.json.",
    )
    parser.add_argument(
        "--input-parquet",
        default=None,
        help="Optional explicit Arm-II parquet path. Defaults to manifest resolution, then deterministic candidate search.",
    )
    parser.add_argument(
        "--search-root",
        default=str(DEFAULT_TASK_A_ROOT),
        help="Root directory used for Arm-II parquet fallback discovery when no manifest is supplied.",
    )
    parser.add_argument(
        "--stage0-h5ad",
        default=None,
        help="Frozen Stage-0 artifact used by the Arm-II real-data mirror pipeline. Defaults to the Task-A manifest when provided.",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Task-A config used by the Arm-II real-data mirror pipeline. Defaults to the Task-A manifest when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Analysis directory whose compatibility `focused/` package will be replaced by the proposal-aligned real-data mirror outputs. Defaults to the Task-A manifest arm analysis root when provided.",
    )
    parser.add_argument(
        "--prototype-view-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional downstream subset for outputs 06 and 07 only.",
    )
    return parser.parse_args(argv)


def build_paths_from_args(args: argparse.Namespace) -> Arm2FocusedPaths:
    prototype_view_ids = None
    if args.prototype_view_ids:
        prototype_view_ids = tuple(sorted({int(proto_id) for proto_id in args.prototype_view_ids}))

    manifest = _load_manifest_from_args(args)
    if manifest is not None:
        arm2_metrics_parquet = (
            _validate_arm2_parquet(Path(args.input_parquet))
            if args.input_parquet is not None
            else _validate_arm2_parquet(manifest.metrics_parquet)
        )
        stage0_h5ad = (
            Path(args.stage0_h5ad).expanduser().resolve()
            if args.stage0_h5ad is not None
            else manifest.stage0_h5ad
        )
        task_config = (
            Path(args.task_config).expanduser().resolve()
            if args.task_config is not None
            else manifest.config_path
        )
        output_dir = _resolve_focused_output_dir(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else resolve_task_a_arm_focused_output_dir(manifest, ARM_NAME)
        )
    else:
        arm2_metrics_parquet = discover_arm2_parquet(
            explicit_path=args.input_parquet,
            search_root=Path(args.search_root).expanduser().resolve(),
        )
        stage0_h5ad = Path(args.stage0_h5ad or DEFAULT_STAGE0_PATH).expanduser().resolve()
        task_config = Path(args.task_config or DEFAULT_CONFIG_PATH).expanduser().resolve()
        output_dir = _resolve_focused_output_dir(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else DEFAULT_OUTPUT_DIR
        )

    return Arm2FocusedPaths(
        arm2_metrics_parquet=arm2_metrics_parquet,
        stage0_h5ad=stage0_h5ad,
        task_config=task_config,
        output_dir=output_dir,
        prototype_view_ids=prototype_view_ids,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = build_paths_from_args(args)
    if can_rebuild_from_existing_focused_dir(paths.output_dir):
        package = build_corrected_output_package_from_existing_dir(
            paths.output_dir,
            stage0_h5ad=paths.stage0_h5ad,
            task_config=paths.task_config,
        )
    else:
        from tasks.task_A.analyze_arm2_focused import run_focused_analysis

        legacy_package = run_focused_analysis(paths)
        package = build_corrected_output_package_from_legacy_package(legacy_package)
    if paths.output_dir.exists():
        shutil.rmtree(paths.output_dir)
    write_corrected_output_package(package, paths.output_dir)


if __name__ == "__main__":
    main()
