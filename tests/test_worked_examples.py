"""Regression tests: the port reproduces the QDR program's worked examples.

Each example's Chargemol input is run through the command-line interface, and the
output is compared with the ``QDR-DDEC6_point_charges.xyz`` the Fortran program wrote,
line by line. The text around the numbers must match exactly. The numbers may differ
by one unit in their last printed digit, since the two programs round independently.
"""

import re
import shutil
from pathlib import Path

import pytest

from py_chargemol.__main__ import main

EXAMPLES = [
    "ALPO5",
    "BN_nanotube",
    "CO2_CCSD_augccpVTZ",
    "CO_molecule",
    "H2_CCSD_def2QZVPPD",
    "formamide_VASP",
]

NUMBER = re.compile(r" *-?\d+\.\d+(?:E[-+]\d+)?")
DATE = re.compile(r" \d{4}/\d{2}/\d{2}  \d{2}:\d{2}:\d{2}")
VERSION = "Two-pass QDR computed using QDR program version April 9, 2025"


def copy_inputs(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if path.name.startswith(("DDEC6", "overlap_populations")):
            shutil.copy(path, destination)


def assert_numbers_close(value: str, expected: str, where: str) -> None:
    if "E" in expected:  # the residual norm, which is round-off
        assert float(value) < 1e-10, where
        assert float(expected) < 1e-10, where
        return
    decimals = len(expected.rpartition(".")[2])
    assert abs(float(value) - float(expected)) <= 1.5 * 10.0**-decimals, where


@pytest.mark.parametrize("example", EXAMPLES)
def test_reproduces_the_reference_output(
    worked_examples: Path, tmp_path: Path, example: str
) -> None:
    copy_inputs(worked_examples / example, tmp_path)

    assert main(["qdr", str(tmp_path)]) == 0

    ours = (tmp_path / "QDR-DDEC6_point_charges.xyz").read_text().splitlines()
    reference_file = worked_examples / example / "QDR-DDEC6_point_charges.xyz"
    reference = reference_file.read_text().splitlines()
    assert len(ours) == len(reference)
    for number, (line, expected) in enumerate(zip(ours, reference, strict=True), 1):
        where = f"line {number}: {line!r} != {expected!r}"
        if expected.startswith(VERSION):
            assert line.startswith(f"{VERSION} (py-chargemol "), where
        elif DATE.fullmatch(expected):
            assert DATE.fullmatch(line), where
        else:
            assert NUMBER.sub("#", line) == NUMBER.sub("#", expected), where
            for value, expected_value in zip(
                NUMBER.findall(line), NUMBER.findall(expected), strict=True
            ):
                assert_numbers_close(value, expected_value, where)


def test_writes_a_log_file(worked_examples: Path, tmp_path: Path) -> None:
    copy_inputs(worked_examples / "formamide_VASP", tmp_path)

    assert main(["qdr", str(tmp_path)]) == 0

    log = (tmp_path / "QDR_charges_computation_logfile.txt").read_text()
    assert "natoms: 6" in log
    assert "vector3:     28.345890     0.000000     0.000000" in log
    assert "num_overlap_values: 30" in log
    assert "total multipole moments are skipped" in log
    assert "Finished QDR program in" in log


def test_logs_molecular_moments(worked_examples: Path, tmp_path: Path) -> None:
    copy_inputs(worked_examples / "CO_molecule", tmp_path)

    assert main(["qdr", str(tmp_path)]) == 0

    log = (tmp_path / "QDR_charges_computation_logfile.txt").read_text()
    assert "Nonperiodic system" in log
    assert "center_of_mass:   0.000000  0.000000  1.223688" in log
    assert "dipole_magnitude:      0.039840" in log
