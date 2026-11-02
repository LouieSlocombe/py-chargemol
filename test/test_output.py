"""Tests for the QDR point charge file writer."""

from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pytest

from py_chargemol import __version__
from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.qdr import compute_qdr_charges, format_qdr_point_charges
from py_chargemol.qdr.output import fortran_es


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.220446049250313e-16, "    2.2204460E-016"),
        (0.0, "    0.0000000E+000"),
        (1234.5, "    1.2345000E+003"),
        (-1e-300, "   -1.0000000E-300"),
        (float("nan"), "               nan"),
        (float("inf"), "               inf"),
    ],
)
def test_fortran_es_matches_the_es18_7e3_descriptor(
    value: float, expected: str
) -> None:
    assert fortran_es(value) == expected


@pytest.fixture
def formamide_text(
    load_example: Callable[[str], tuple[DDEC6Analysis, OverlapPairs]],
) -> str:
    analysis, pairs = load_example("formamide_VASP")
    result = compute_qdr_charges(analysis, pairs)
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    return format_qdr_point_charges(analysis, result, None, timestamp)


def test_names_the_port_and_its_version(formamide_text: str) -> None:
    assert (
        "Two-pass QDR computed using QDR program version April 9, 2025 "
        f"(py-chargemol {__version__})\n"
    ) in formamide_text


def test_ends_with_the_timestamp(formamide_text: str) -> None:
    assert formamide_text.endswith("\n 2026/01/02  03:04:05\n")


def test_periodic_systems_have_no_total_moments(formamide_text: str) -> None:
    assert "Since the system is non-periodic" not in formamide_text


def test_residual_dipole_rows_include_the_magnitude(formamide_text: str) -> None:
    lines = formamide_text.splitlines()
    first = lines[lines.index("<atomic dipoles>") + 1]

    *components, magnitude = (float(value) for value in first.split())
    assert len(first) == 3 * 13 + 20
    assert magnitude == pytest.approx(np.linalg.norm(components), abs=2e-6)
