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
from tasks.task_A.arm2.analysis_contract import Arm2FocusedPaths

DEFAULT_TASK_A_ROOT = Path("/mnt/NAS_21T/ProjectResult/SLOTAR/task_A")
DEFAULT_ARM2_PATH = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "task_A_metrics.parquet"
DEFAULT_STAGE0_PATH = Path("/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_OUTPUT_DIR = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "analysis" / "bioinformed"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Arm-II biologically informed extraction workflow.",
    )
    parser.add_argument(
        "--arm2-metrics-parquet",
        default=str(DEFAULT_ARM2_PATH),
        help="Path to the Arm-II metrics parquet.",
    )
    parser.add_argument(
        "--stage0-h5ad",
        default=str(DEFAULT_STAGE0_PATH),
        help="Path to the frozen Stage-0 .h5ad artifact.",
    )
    parser.add_argument(
        "--task-config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the Task-A config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the Arm-II bioinformed package will be written.",
    )
    return parser.parse_args(argv)


def build_paths_from_args(args: argparse.Namespace) -> Arm2FocusedPaths:
    return Arm2FocusedPaths(
        arm2_metrics_parquet=Path(args.arm2_metrics_parquet).expanduser().resolve(),
        stage0_h5ad=Path(args.stage0_h5ad).expanduser().resolve(),
        task_config=Path(args.task_config).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        prototype_view_ids=None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = build_paths_from_args(args)
    tables_by_filename = run_bioinformed_analysis(paths)
    write_bioinformed_output_package(tables_by_filename, paths.output_dir)


if __name__ == "__main__":
    main()
