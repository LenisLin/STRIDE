"""Task-local wrapper for the compatibility-named Block 2 robustness surface.

This wrapper consumes an existing evidence-ready Block 1 bundle, perturbs the
frozen Stage 0 cohort under task-local robustness routes, re-estimates the same
Block 1 summary/comparison objects, and writes proof-carrying Block 2
robustness tables plus the compatibility-named top summary/manifest surface.
It does not compare baselines or estimate true emergence/disappearance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

from ..block2.bundle import write_block2_bundle


def run_block2_workflow(
    *,
    block1_bundle: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    return write_block2_bundle(
        block1_bundle_path=block1_bundle,
        output_dir=output_path,
        resume=resume,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task-A Block 2 over an existing Block 1 bundle.")
    parser.add_argument("--block1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted Block 2 run from checkpoint files in the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        run_block2_workflow(
            block1_bundle=args.block1_bundle,
            output_dir=args.output_dir,
            resume=bool(args.resume),
        )
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["run_block2_workflow"]


if __name__ == "__main__":  # pragma: no cover
    main()
