"""Command-line interface for py-chargemol."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from py_chargemol import __version__
from py_chargemol.qdr.run import LOG_FILE, OUTPUT_FILE, run_qdr


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="py-chargemol",
        description="Post-process Chargemol DDEC6 atomic population analyses.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    qdr = commands.add_parser(
        "qdr",
        help="compute quadrupole-dipole resorbed (QDR) point charges",
        description=(
            "Compute quadrupole-dipole resorbed (QDR) point charges from the "
            "DDEC6_net_atomic_charges.xyz and "
            "overlap_populations_and_contact_exchanges.txt files that Chargemol "
            f"writes, and save them to {OUTPUT_FILE}."
        ),
    )
    qdr.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        metavar="DIR",
        help="directory with the Chargemol output (default: current directory)",
    )
    qdr.add_argument(
        "--charges", type=Path, help="net atomic charge file (default: found in DIR)"
    )
    qdr.add_argument(
        "--overlaps",
        type=Path,
        help="overlap population file (default: found in DIR)",
    )
    qdr.add_argument(
        "--output",
        type=Path,
        help=f"QDR point charge file (default: DIR/{OUTPUT_FILE})",
    )
    qdr.add_argument("--log", type=Path, help=f"log file (default: DIR/{LOG_FILE})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = build_parser().parse_args(argv)
    return _qdr(arguments)


def _qdr(arguments: argparse.Namespace) -> int:
    directory: Path = arguments.directory
    log_path: Path = arguments.log or directory / LOG_FILE
    try:
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    except OSError as error:
        return _error(f"cannot write the log file: {error}")
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger = logging.getLogger("py_chargemol")
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    try:
        result = run_qdr(
            directory,
            charges_file=arguments.charges,
            overlaps_file=arguments.overlaps,
            output_file=arguments.output,
        )
    except (OSError, ValueError) as error:
        package_logger.error("%s. Program will terminate.", error)
        return _error(str(error))
    finally:
        package_logger.removeHandler(handler)
        handler.close()

    if result is None:
        print("A single atom has no bonds to resorb multipoles into; nothing written.")
    else:
        output = arguments.output or directory / OUTPUT_FILE
        print(f"Wrote QDR point charges for {len(result.charges)} atoms to {output}")
    return 0


def _error(message: str) -> int:
    print(f"py-chargemol qdr: error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
