from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from tasks.task_A.arm2.analysis_bioinformed import (
    run_bioinformed_analysis,
    write_bioinformed_output_package,
)
from tasks.task_A.arm2.analysis_contract import ARM_NAME, Arm2FocusedPaths
from tasks.task_A.runtime_contract import (
    TASK_A_METRICS_FILENAME,
    load_task_a_run_manifest,
    resolve_task_a_arm_bioinformed_output_dir,
)

DEFAULT_TASK_A_ROOT = Path("/mnt/NAS_21T/ProjectResult/SLOTAR/task_A")
DEFAULT_ARM2_RUN_ROOT = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment"
DEFAULT_ARM2_PATH = DEFAULT_ARM2_RUN_ROOT / TASK_A_METRICS_FILENAME
DEFAULT_STAGE0_PATH = Path("/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_OUTPUT_DIR = DEFAULT_ARM2_RUN_ROOT / "analysis" / "bioinformed"


def _load_manifest_from_args(args: argparse.Namespace):
    if args.task_a_manifest is not None:
        return load_task_a_run_manifest(args.task_a_manifest)
    if args.task_a_run_root is not None:
        return load_task_a_run_manifest(args.task_a_run_root)
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Arm-II bounded residual audit extraction workflow.",
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
        "--arm2-metrics-parquet",
        default=None,
        help="Path to the Arm-II metrics parquet. Defaults to manifest resolution when available.",
    )
    parser.add_argument(
        "--stage0-h5ad",
        default=None,
        help="Path to the frozen Stage-0 .h5ad artifact. Defaults to manifest resolution when available.",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Path to the Task-A config YAML. Defaults to manifest resolution when available.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the Arm-II bounded audit package will be written. Defaults to the manifest arm analysis root when available.",
    )
    return parser.parse_args(argv)


def build_paths_from_args(args: argparse.Namespace) -> Arm2FocusedPaths:
    manifest = _load_manifest_from_args(args)
    if manifest is not None:
        arm2_metrics_parquet = (
            Path(args.arm2_metrics_parquet).expanduser().resolve()
            if args.arm2_metrics_parquet is not None
            else manifest.metrics_parquet
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
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else resolve_task_a_arm_bioinformed_output_dir(manifest, ARM_NAME)
        )
    else:
        arm2_metrics_parquet = Path(args.arm2_metrics_parquet or DEFAULT_ARM2_PATH).expanduser().resolve()
        stage0_h5ad = Path(args.stage0_h5ad or DEFAULT_STAGE0_PATH).expanduser().resolve()
        task_config = Path(args.task_config or DEFAULT_CONFIG_PATH).expanduser().resolve()
        output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()

    return Arm2FocusedPaths(
        arm2_metrics_parquet=arm2_metrics_parquet,
        stage0_h5ad=stage0_h5ad,
        task_config=task_config,
        output_dir=output_dir,
        prototype_view_ids=None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = build_paths_from_args(args)
    tables_by_filename = run_bioinformed_analysis(paths)
    write_bioinformed_output_package(tables_by_filename, paths.output_dir)


if __name__ == "__main__":
    main()
