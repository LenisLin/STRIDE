"""Task-local wrapper for the Block 1 compatibility-named discovery surface.

This wrapper consumes the Task A config, Stage 0 h5ad, and a passed Block 0
bundle, then writes the frozen Block 1 real-data biological discovery
artifacts. It does not implement Block 0 or Block 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

from ..block1.bundle import BLOCK_NAME, write_task_a_block1_bundle


def run_block1_workflow(
    *,
    config_path: str | Path,
    data_path: str | Path,
    block0_bundle: str | Path,
    output_dir: str | Path,
) -> Path:
    bundle = write_task_a_block1_bundle(
        config_path=config_path,
        data_path=data_path,
        block0_bundle_path=block0_bundle,
        output_dir=output_dir,
    )
    return bundle.bundle_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task-A Block 1 through the task-local workflow wrapper.")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--stage0-h5ad", required=True)
    parser.add_argument("--block0-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        run_block1_workflow(
            config_path=args.task_config,
            data_path=args.stage0_h5ad,
            block0_bundle=args.block0_bundle,
            output_dir=args.output_dir,
        )
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["BLOCK_NAME", "run_block1_workflow"]


if __name__ == "__main__":  # pragma: no cover
    main()
