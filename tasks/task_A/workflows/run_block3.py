"""Deprecated workflow wrapper for the removed Task A Block 3 engineering surface."""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

DEPRECATION_MESSAGE = (
    "Task A run_block3 is retired from the active path. "
    "The public Block 3 runner is retired, while the internal non-authority "
    "Phase 3 package remains on disk under tasks/task_A/block3/. "
    "Public execution must be rebuilt from the docs authority before it can "
    "run again."
)


def run_block3_workflow(
    *,
    block2_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    """Fail fast because the active Block 3 runner has been removed.

    The live scientific contract remains in the docs hierarchy. The public
    workflow entrypoint is retired, while the on-disk ``tasks/task_A/block3``
    package remains an internal non-authority Phase 3 implementation surface.
    """

    raise ContractError(DEPRECATION_MESSAGE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated. The Task A Block 3 engineering runner has been "
            "removed from the active path."
        )
    )
    parser.add_argument("--block2-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        run_block3_workflow(
            block2_manifest=args.block2_manifest,
            output_dir=args.output_dir,
        )
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["run_block3_workflow"]


if __name__ == "__main__":  # pragma: no cover
    main()
