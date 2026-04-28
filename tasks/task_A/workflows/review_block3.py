"""Deprecated workflow wrapper for the retired Task A Block 3 review surface."""
from __future__ import annotations

import argparse
from pathlib import Path

from stride.errors import ContractError

DEPRECATION_MESSAGE = (
    "Task A review_block3 is retired from the active path. "
    "Block 3 review/packet integration remains deferred / non-authority / "
    "pending clean bridge spec."
)


def write_block3_review(
    *,
    block3_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    """Fail fast because the active review workflow has been retired."""

    raise ContractError(DEPRECATION_MESSAGE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated. The packet-local Task A Block 3 review workflow is "
            "retired from the active path."
        )
    )
    parser.add_argument("--block3-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        write_block3_review(
            block3_manifest=args.block3_manifest,
            output_dir=args.output_dir,
        )
    except (ContractError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["write_block3_review"]


if __name__ == "__main__":  # pragma: no cover
    main()
