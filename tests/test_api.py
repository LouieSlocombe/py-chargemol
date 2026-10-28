"""Tests for the public Python API and its file-free calculation results."""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from py_chargemol import (
    DDEC6Analysis,
    OverlapPairs,
    QDRResult,
    calculate_qdr,
    compute_qdr_charges,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
    total_multipole_moments,
)

type Loader = Callable[[str], tuple[DDEC6Analysis, OverlapPairs]]
type Copier = Callable[[str], Path]

EXAMPLES = [
    "ALPO5",
    "BN_nanotube",
    "CO2_CCSD_augccpVTZ",
    "CO_molecule",
    "H2_CCSD_def2QZVPPD",
    "formamide_VASP",
]
CHARGES_FILE = "DDEC6_even_tempered_net_atomic_charges.xyz"
OVERLAPS_FILE = "overlap_populations_and_contact_exchanges.txt"


def assert_same_result(actual: QDRResult, expected: QDRResult) -> None:
    """Both public entry points return the same numerical calculation."""
    np.testing.assert_array_equal(actual.charges, expected.charges)
    np.testing.assert_array_equal(actual.atomic_dipoles, expected.atomic_dipoles)
    np.testing.assert_array_equal(
        actual.atomic_quadrupoles, expected.atomic_quadrupoles
    )
    assert actual.cg_iterations == expected.cg_iterations
    assert actual.residual_norm == expected.residual_norm


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("existing_outputs", [False, True])
def test_calculate_returns_results_without_writing_files(
    copy_example: Copier,
    load_example: Loader,
    example: str,
    existing_outputs: bool,
) -> None:
    directory = copy_example(example)
    if existing_outputs:
        for name in (
            "QDR-DDEC6_point_charges.xyz",
            "QDR_charges_computation_logfile.txt",
        ):
            (directory / name).write_bytes(b"Existing results must remain intact.\n")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    result = calculate_qdr(directory)

    assert isinstance(result, QDRResult)
    assert_same_result(result, compute_qdr_charges(*load_example(example)))
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_calculate_defaults_to_current_directory(
    copy_example: Copier, load_example: Loader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(copy_example("CO_molecule"))

    result = calculate_qdr()

    assert_same_result(result, compute_qdr_charges(*load_example("CO_molecule")))


@pytest.mark.parametrize("relative_paths", [False, True])
def test_calculate_accepts_explicit_paths(
    copy_example: Copier,
    load_example: Loader,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_paths: bool,
) -> None:
    directory = copy_example("CO_molecule")
    monkeypatch.chdir(tmp_path)
    charges, overlaps = directory / CHARGES_FILE, directory / OVERLAPS_FILE
    if relative_paths:
        charges, overlaps = (
            charges.relative_to(tmp_path),
            overlaps.relative_to(tmp_path),
        )

    result = calculate_qdr(
        tmp_path / "unused", charges_file=charges, overlaps_file=overlaps
    )

    assert_same_result(result, compute_qdr_charges(*load_example("CO_molecule")))


@pytest.mark.parametrize("charges_name", ["DDEC6_net_atomic_charges.xyz", CHARGES_FILE])
@pytest.mark.parametrize("overlaps_name", [OVERLAPS_FILE, "overlap_populations.txt"])
def test_calculate_discovers_current_and_legacy_file_names(
    copy_example: Copier,
    load_example: Loader,
    charges_name: str,
    overlaps_name: str,
) -> None:
    directory = copy_example("CO_molecule")
    (directory / CHARGES_FILE).rename(directory / charges_name)
    (directory / OVERLAPS_FILE).rename(directory / overlaps_name)

    result = calculate_qdr(str(directory))

    assert_same_result(result, compute_qdr_charges(*load_example("CO_molecule")))


def test_calculate_prefers_current_file_names(copy_example: Copier) -> None:
    directory = copy_example("CO_molecule")
    expected = calculate_qdr(directory)
    (directory / CHARGES_FILE).rename(directory / "DDEC6_net_atomic_charges.xyz")
    (directory / CHARGES_FILE).write_text("invalid legacy charge file")
    (directory / "overlap_populations.txt").write_text("invalid legacy overlap file")

    assert_same_result(calculate_qdr(directory), expected)


@pytest.mark.parametrize("name", [CHARGES_FILE, OVERLAPS_FILE])
def test_calculate_raises_for_missing_inputs(copy_example: Copier, name: str) -> None:
    directory = copy_example("CO_molecule")
    (directory / name).unlink()

    with pytest.raises(FileNotFoundError, match="could not find"):
        calculate_qdr(directory)


@pytest.mark.parametrize("name", [CHARGES_FILE, OVERLAPS_FILE])
def test_calculate_raises_for_malformed_inputs(copy_example: Copier, name: str) -> None:
    directory = copy_example("CO_molecule")
    (directory / name).write_text("\n")

    with pytest.raises(ValueError, match="the file is empty"):
        calculate_qdr(directory)


def test_calculate_passes_solver_settings(
    copy_example: Copier, load_example: Loader
) -> None:
    directory = copy_example("ALPO5")

    result = calculate_qdr(directory, max_cg_iterations=1, tolerance=1e-3)

    expected = compute_qdr_charges(
        *load_example("ALPO5"), max_cg_iterations=1, tolerance=1e-3
    )
    assert_same_result(result, expected)
    assert result.cg_iterations == 1


@pytest.fixture
def single_atom_inputs(copy_example: Copier) -> Path:
    """A carbon atom retaining the multipoles it had in the CO example."""
    directory = copy_example("CO_molecule")
    charges = directory / CHARGES_FILE
    lines = charges.read_text().splitlines()
    lines[0] = "1"
    carbon = [
        line
        for line in lines
        if line.split()[:1] != ["O"] and line.split()[:2] != ["2", "O"]
    ]
    charges.write_text("\n".join(carbon) + "\n")
    (directory / OVERLAPS_FILE).unlink()
    return directory


def test_calculate_isolated_atom_needs_no_overlap_file(
    single_atom_inputs: Path,
) -> None:
    analysis = read_ddec6_net_atomic_charges(single_atom_inputs / CHARGES_FILE)

    result = calculate_qdr(single_atom_inputs)

    np.testing.assert_array_equal(result.charges, analysis.net_charges)
    np.testing.assert_array_equal(result.atomic_dipoles, analysis.atomic_dipoles)
    np.testing.assert_array_equal(
        result.atomic_quadrupoles, analysis.atomic_quadrupoles
    )
    assert result.cg_iterations == 0
    assert result.residual_norm == 0.0
    assert result.moments is not None


def test_calculate_isolated_atom_honors_explicit_missing_overlap_file(
    single_atom_inputs: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        calculate_qdr(
            single_atom_inputs, overlaps_file=single_atom_inputs / "missing.txt"
        )


def test_calculate_isolated_atom_honors_explicit_invalid_overlap_file(
    single_atom_inputs: Path,
) -> None:
    overlaps = single_atom_inputs / "invalid.txt"
    overlaps.write_text("\n")

    with pytest.raises(ValueError, match="the file is empty"):
        calculate_qdr(single_atom_inputs, overlaps_file=overlaps)


def test_calculate_periodic_single_atom_uses_self_image_bonds(
    single_atom_inputs: Path,
) -> None:
    charges = single_atom_inputs / CHARGES_FILE
    text = charges.read_text().replace(
        "Nonperiodic system",
        "jmolscript: unitcell [{2 0 0}, {0 2 0}, {0 0 2}]",
    )
    charges.write_text(text)
    overlaps = single_atom_inputs / OVERLAPS_FILE
    overlaps.write_text(
        "2\nOverlap populations\natom1 atom2 tx ty tz population\n"
        "1 1 0 0 1 0.6\n1 1 0 0 -1 0.6\n"
    )
    analysis = read_ddec6_net_atomic_charges(charges)

    result = calculate_qdr(single_atom_inputs)

    expected = compute_qdr_charges(analysis, read_overlap_populations(overlaps))
    assert_same_result(result, expected)
    np.testing.assert_array_equal(result.charges, analysis.net_charges)
    assert not np.array_equal(result.atomic_dipoles, analysis.atomic_dipoles)
    assert not np.array_equal(result.atomic_quadrupoles, analysis.atomic_quadrupoles)
    assert result.cg_iterations > 0
    assert result.moments is None


@pytest.mark.parametrize("example", EXAMPLES)
def test_result_contains_total_moments_only_for_molecules(
    load_example: Loader, example: str
) -> None:
    analysis, pairs = load_example(example)

    result = compute_qdr_charges(analysis, pairs)

    if analysis.is_periodic:
        assert result.moments is None
        return
    expected = total_multipole_moments(
        analysis.atomic_numbers,
        analysis.positions,
        result.charges,
        result.atomic_dipoles,
        result.atomic_quadrupoles,
    )
    assert result.moments is not None
    np.testing.assert_array_equal(
        result.moments.center_of_mass, expected.center_of_mass
    )
    for name in ("charges", "with_dipoles", "with_quadrupoles"):
        actual_order, expected_order = (
            getattr(result.moments, name),
            getattr(expected, name),
        )
        np.testing.assert_array_equal(actual_order.dipole, expected_order.dipole)
        np.testing.assert_array_equal(
            actual_order.quadrupole, expected_order.quadrupole
        )


@pytest.mark.parametrize(
    "atomic_numbers",
    [
        np.array([], dtype=np.int64),
        np.array([[6, 8]], dtype=np.int64),
        np.array([6.0, 8.0]),
        np.array([False, True]),
        np.array([0, 8]),
        np.array([-1, 8]),
        np.array([110, 8]),
    ],
)
def test_solver_rejects_invalid_atomic_numbers(
    load_example: Loader, atomic_numbers: npt.NDArray[np.generic]
) -> None:
    analysis, pairs = load_example("CO_molecule")
    invalid = dataclasses.replace(
        analysis, atomic_numbers=cast("npt.NDArray[np.int64]", atomic_numbers)
    )

    with pytest.raises(ValueError, match="atomic_numbers must be a nonempty integer"):
        compute_qdr_charges(invalid, pairs)


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5])
def test_solver_rejects_invalid_iteration_limits(
    load_example: Loader, iterations: float
) -> None:
    with pytest.raises(
        ValueError, match="max_cg_iterations must be a positive integer"
    ):
        compute_qdr_charges(
            *load_example("CO_molecule"), max_cg_iterations=cast("int", iterations)
        )


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_solver_rejects_invalid_tolerances(
    load_example: Loader, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="tolerance must be finite and nonnegative"):
        compute_qdr_charges(*load_example("CO_molecule"), tolerance=tolerance)


def test_solver_allows_zero_tolerance(load_example: Loader) -> None:
    result = compute_qdr_charges(*load_example("CO_molecule"), tolerance=0.0)

    assert np.isfinite(result.charges).all()
    assert result.cg_iterations > 0
