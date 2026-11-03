"""Interoperability of newly generated DDEC files with the existing QDR API."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from py_chargemol.ddec6 import (
    DDEC6Analysis,
    OverlapPairs,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)
from py_chargemol.density.analysis import DDECResult
from py_chargemol.density.bonds import BondOrderResult
from py_chargemol.density.diagnostics import SpinGradientDiagnostics
from py_chargemol.density.output import write_density_results
from py_chargemol.density.references import NSHELLS, SCALE
from py_chargemol.density.spin import SpinResult
from py_chargemol.qdr.run import calculate_qdr

type Loader = Callable[[str], tuple[DDEC6Analysis, OverlapPairs]]


@pytest.fixture(params=["CO_molecule", "formamide_VASP"])
def result(load_example: Loader, request: pytest.FixtureRequest) -> DDECResult:
    analysis, pairs = load_example(request.param)
    atoms = analysis.natoms
    contacts = pairs.populations * 0.9
    bonds = pairs.populations * 1.2
    return DDECResult(
        analysis=analysis,
        bond_result=BondOrderResult(
            pairs,
            bonds,
            contacts,
            contacts,
            np.bincount(pairs.atom1, weights=bonds, minlength=atoms),
            np.bincount(pairs.atom1, weights=contacts, minlength=atoms),
            np.zeros(1),
            np.zeros((atoms, 100)),
            1,
            np.zeros(atoms),
        ),
        _radial_moments=np.ones((atoms, 3)),
        hirshfeld_charges=analysis.net_charges * 0.5,
        iterations=7,
        normalization=1.0,
        method="ddec6",
        spin=None,
        cm5_charges=analysis.net_charges * 0.8,
        cloud_r_squared=np.full(atoms, 0.9),
        _radial_shell_weights=np.full((atoms, NSHELLS), 0.25),
    )


def test_charge_and_overlap_files_roundtrip(result: DDECResult, tmp_path: Path) -> None:
    paths = write_density_results(result, tmp_path)
    assert paths["charges"].name == "DDEC6_net_atomic_charges.xyz"
    assert paths["overlaps"].name == "overlap_populations_and_contact_exchanges.txt"
    assert "qdr" not in paths
    assert "spin" not in paths
    assert "atomic_densities" not in paths
    parsed = read_ddec6_net_atomic_charges(paths["charges"])
    for field in (
        "atomic_numbers",
        "positions",
        "net_charges",
        "atomic_dipoles",
        "atomic_quadrupoles",
        "cloud_a",
        "cloud_b",
        "lattice_vectors",
    ):
        np.testing.assert_allclose(
            getattr(parsed, field), getattr(result.analysis, field), atol=1e-11
        )
    pairs = read_overlap_populations(paths["overlaps"])
    for field in ("atom1", "atom2", "translations", "populations"):
        np.testing.assert_allclose(
            getattr(pairs, field), getattr(result.overlaps, field), atol=1e-11
        )
    for path in paths.values():
        assert path.is_file()
        assert b"\r" not in path.read_bytes()
    np.testing.assert_allclose(
        calculate_qdr(tmp_path).charges, result.compute_qdr().charges, atol=1e-11
    )
    assert ("Full_quadrupole" in paths["charges"].read_text()) == (
        not result.analysis.is_periodic
    )


def test_optional_atomic_density_output(result: DDECResult, tmp_path: Path) -> None:
    densities = np.arange(result.analysis.natoms * NSHELLS, dtype=float).reshape(
        -1, NSHELLS
    )
    result = replace(
        result,
        bond_result=replace(result.bonds, corrected_spherical_average=densities),
    )
    paths = write_density_results(result, tmp_path, atomic_densities=True)
    path = paths["atomic_densities"]
    assert path.name == "spherically_averaged_atomic_densities.txt"
    lines = path.read_text().splitlines()
    assert lines[0].split() == [str(result.analysis.natoms), "atoms"]
    assert lines[2].split() == ["100", "radial", "shells"]
    assert "jmolscript" not in lines[1]
    assert ("Nonperiodic" in lines[1]) == (not result.analysis.is_periodic)
    headings = [
        index
        for index, line in enumerate(lines)
        if "in atomic units printed below." in line
    ]
    assert len(headings) == 3
    blocks = []
    for start, end in zip(headings, [*headings[1:], len(lines)], strict=True):
        values = np.fromstring(" ".join(lines[start + 1 : end]), sep=" ")
        blocks.append(values.reshape(-1, NSHELLS))
    np.testing.assert_array_equal(blocks[0], densities)
    np.testing.assert_array_equal(blocks[1], result.radial_shell_weights)
    # Integrating all radial shells equals the analytic spherical integral.
    powers = np.arange(2, 8)
    expected = 4 * np.pi * (NSHELLS / SCALE) ** powers / powers
    np.testing.assert_allclose(blocks[2].sum(axis=1), expected, rtol=1e-9)


def test_skipped_bond_analysis_omits_dependent_files(
    result: DDECResult, tmp_path: Path
) -> None:
    result = replace(
        result, bond_result=None, _radial_moments=None, _radial_shell_weights=None
    )
    paths = write_density_results(result, tmp_path)
    assert set(paths) == {"charges", "cm5", "hirshfeld"}
    for qdr in (False, True):
        destination = tmp_path / ("qdr" if qdr else "atomic_densities")
        with pytest.raises(ValueError, match="compute_bos=True"):
            write_density_results(
                result, destination, qdr=qdr, atomic_densities=not qdr
            )
        assert not destination.exists()


def test_bond_output_reports_density_correction_residual(
    result: DDECResult, tmp_path: Path
) -> None:
    result = replace(
        result,
        bond_result=replace(
            result.bonds,
            density_correction_iterations=40,
            population_residual=np.full(result.analysis.natoms, 0.05),
        ),
    )
    paths = write_density_results(result, tmp_path)
    text = paths["bond_orders"].read_text()
    assert "Number of density correction iterations = 40" in text
    assert "population residual after density correction = 5.000000000000e-02" in text


def test_optional_qdr_and_spin_output(result: DDECResult, tmp_path: Path) -> None:
    atoms = result.analysis.natoms
    spin = SpinResult(
        np.full((atoms, 3), 0.1),
        np.zeros((atoms, 100, 3)),
        np.zeros((1, 3)),
        7,
        True,
        0.0,
    )
    paths = write_density_results(
        replace(
            result,
            spin=spin,
            spin_gradients=SpinGradientDiagnostics(0.2, 0.3, 0.1),
        ),
        tmp_path,
        qdr=True,
    )
    assert paths["qdr"].name == "QDR-DDEC6_point_charges.xyz"
    assert paths["spin"].name == "DDEC6_atomic_spin_moments.xyz"
    assert len(paths["spin"].read_text().splitlines()[2].split()) == 7
    assert "QDR" in paths["qdr"].read_text()
    assert "Directional contribution = 1.000000000000e-01" in (
        paths["charges"].read_text()
    )


def test_ddec3_names_and_qdr_rejection(result: DDECResult, tmp_path: Path) -> None:
    result = replace(result, method="ddec3")
    paths = write_density_results(result, tmp_path)
    assert paths["charges"].name == "DDEC3_net_atomic_charges.xyz"
    assert paths["bond_orders"].name == "DDEC3_bond_orders.xyz"
    destination = tmp_path / "should-not-be-created"
    with pytest.raises(ValueError, match="requires a DDEC6"):
        write_density_results(result, destination, qdr=True)
    assert not destination.exists()


def test_unknown_method_does_not_write_files(
    result: DDECResult, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="method must be"):
        write_density_results(replace(result, method="unknown"), tmp_path)
    assert list(tmp_path.iterdir()) == []
