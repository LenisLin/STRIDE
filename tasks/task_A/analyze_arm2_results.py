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

from tasks.task_A.analyze_arm2_focused import run_focused_analysis
from tasks.task_A.arm2.analysis_contract import ARM_NAME, Arm2FocusedPaths
from tasks.task_A.arm2.analysis_output import write_output_package

DEFAULT_TASK_A_ROOT = Path("/mnt/NAS_21T/ProjectResult/SLOTAR/task_A")
DEFAULT_OUTPUT_DIR = DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "analysis"
FOCUSED_OUTPUT_SUBDIR = "focused"
DEFAULT_STAGE0_PATH = Path("/mnt/NAS_21T/ProjectData/SLOTAR/task_A_stage0/task_A_stage0_k25.h5ad")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_ARM2_CANDIDATES: tuple[Path, ...] = (
    DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "task_A_metrics.parquet",
    DEFAULT_TASK_A_ROOT / "arm2_cross_compartment" / "task_A_arm2_metrics.parquet",
    DEFAULT_TASK_A_ROOT / "task_A_metrics.parquet",
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


def discover_arm2_parquet(explicit_path: str | None, search_root: Path) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Arm-II parquet does not exist: {path}")
        if ARM_NAME not in _read_arm_column(path):
            raise ValueError(f"Parquet does not contain arm={ARM_NAME!r}: {path}")
        return path

    candidates: list[Path] = []
    for path in DEFAULT_ARM2_CANDIDATES:
        if path.exists():
            candidates.append(path.resolve())

    if search_root.exists():
        for path in search_root.rglob("*.parquet"):
            if "analysis" in path.parts:
                continue
            resolved = path.resolve()
            if resolved not in candidates and ARM_NAME in _read_arm_column(resolved):
                candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(
            "No Arm-II parquet containing arm='A2_cross_compartment' was found under "
            f"{search_root}. Pass --input-parquet explicitly."
        )

    preferred = [path.resolve() for path in DEFAULT_ARM2_CANDIDATES if path.exists()]
    for path in preferred:
        if path in candidates:
            return path

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_focused_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.name == FOCUSED_OUTPUT_SUBDIR:
        return resolved
    return resolved / FOCUSED_OUTPUT_SUBDIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the operative Arm-II focused end-to-end analysis workflow.",
    )
    parser.add_argument(
        "--input-parquet",
        default=None,
        help="Optional explicit Arm-II parquet path. Defaults to auto-discovery under ProjectResult.",
    )
    parser.add_argument(
        "--search-root",
        default=str(DEFAULT_TASK_A_ROOT),
        help="Root directory used for Arm-II parquet auto-discovery.",
    )
    parser.add_argument(
        "--stage0-h5ad",
        default=str(DEFAULT_STAGE0_PATH),
        help="Frozen Stage-0 artifact used by the focused Arm-II pipeline.",
    )
    parser.add_argument(
        "--task-config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Task-A config used by the focused Arm-II pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Analysis directory whose `focused/` package will be replaced.",
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

    arm2_path = discover_arm2_parquet(
        explicit_path=args.input_parquet,
        search_root=Path(args.search_root).expanduser().resolve(),
    )
    return Arm2FocusedPaths(
        arm2_metrics_parquet=arm2_path,
        stage0_h5ad=Path(args.stage0_h5ad).expanduser().resolve(),
        task_config=Path(args.task_config).expanduser().resolve(),
        output_dir=_resolve_focused_output_dir(Path(args.output_dir)),
        prototype_view_ids=prototype_view_ids,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = build_paths_from_args(args)
    package = run_focused_analysis(paths)
    if paths.output_dir.exists():
        shutil.rmtree(paths.output_dir)
    write_output_package(package, paths.output_dir)


if __name__ == "__main__":
    main()
