"""Internal Task A Block 3 CLI.

This module is intentionally package-local. The command surface uses semantic
experiment names and keeps numbered labels as emitted `subexperiment_id`
metadata only.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from .registry import BLOCK3_EXPERIMENT_NAME_TO_SUBEXPERIMENT_ID


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tasks.task_A.block3")
    parser.add_argument(
        "experiment_name",
        choices=tuple(BLOCK3_EXPERIMENT_NAME_TO_SUBEXPERIMENT_ID),
        help="Semantic Block 3 experiment name.",
    )
    parser.add_argument("--task-config", required=True, help="Task A config YAML.")
    parser.add_argument("--stage0-h5ad", required=True, help="Stage 0 h5ad input.")
    parser.add_argument("--output-dir", required=True, help="Output root for raw/review artifacts.")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device forwarded to STRIDE refits and UOT runtime, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the semantic route and validate the CLI surface without executing the experiment.",
    )
    parser.add_argument(
        "--max-reruns",
        type=int,
        default=None,
        help="Internal smoke-test limit for generated reruns; formal runs omit this flag.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Internal smoke-test limit for held-out test patients; formal runs omit this flag.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    subexperiment_id = BLOCK3_EXPERIMENT_NAME_TO_SUBEXPERIMENT_ID[args.experiment_name]
    if args.dry_run:
        print(f"{args.experiment_name}\t{subexperiment_id}")
        return 0
    from .execution import execute_internal_block3_experiment

    result = execute_internal_block3_experiment(
        experiment_name=args.experiment_name,
        task_config_path=args.task_config,
        stage0_h5ad=args.stage0_h5ad,
        output_dir=args.output_dir,
        device=args.device,
        max_reruns=args.max_reruns,
        n_test=args.n_test,
    )
    print(str(result.review_manifest_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
