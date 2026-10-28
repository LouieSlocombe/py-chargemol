"""Shared fixtures: the worked examples that ship with the reference QDR program."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from py_chargemol.ddec6 import (
    DDEC6Analysis,
    OverlapPairs,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)

WORKED_EXAMPLES = (
    Path(__file__).resolve().parents[1] / "QDR_program_04_09_2025" / "worked_examples"
)

CHARGEMOL_OUTPUTS = ("DDEC6", "overlap_populations")
"""Name prefixes of the Chargemol files that are the QDR program's input."""


@pytest.fixture(scope="session")
def worked_examples() -> Path:
    """Directory holding one subdirectory per worked example."""
    return WORKED_EXAMPLES


@pytest.fixture(scope="session")
def load_example(
    worked_examples: Path,
) -> Callable[[str], tuple[DDEC6Analysis, OverlapPairs]]:
    """Read a worked example's net atomic charges and overlap populations."""

    def load(name: str) -> tuple[DDEC6Analysis, OverlapPairs]:
        directory = worked_examples / name
        charges = next(directory.glob("DDEC6*net_atomic_charges.xyz"))
        analysis = read_ddec6_net_atomic_charges(charges)
        overlaps = directory / "overlap_populations_and_contact_exchanges.txt"
        return analysis, read_overlap_populations(overlaps)

    return load


@pytest.fixture
def copy_example(worked_examples: Path, tmp_path: Path) -> Callable[[str], Path]:
    """Copy a worked example's Chargemol output, without the QDR results, to tmp."""

    def copy(name: str) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        for path in (worked_examples / name).iterdir():
            if path.name.startswith(CHARGEMOL_OUTPUTS):
                shutil.copy(path, directory)
        return directory

    return copy
