"""DDEC6 results transform with the density they partition.

Rotating or mirroring a density with its atoms, rolling a periodic density by
whole grid points with its atoms, or listing the atoms in another order is the
same physical system. Charges, spin moments, bond orders, and the other
per-atom scalars must not change. Dipoles must turn as vectors and quadrupoles
as rank-2 tensors, and every property must follow its atom.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from py_chargemol import DensityGrid, compute_ddec6
from py_chargemol.density.analysis import DDECResult

TOLERANCE = 1e-12
"""Largest change allowed, in e·bohr^l: rounding is about 1e-14."""

FIT_TOLERANCE = 1e-10
"""Largest change in the tail fit, whose logarithms of near-zero densities
amplify rounding to about 5e-12."""

_SHAPE = (24, 24, 24)
_LATTICE = np.array([[7.0, 0.0, 0.0], [1.2, 6.6, 0.0], [0.7, 0.9, 7.3]])
_NUMBERS = np.array([8, 1, 1])
_POSITIONS = np.array([[1.0, 1.4, 0.6], [2.6, 1.1, 0.9], [0.6, 2.9, 0.4]])


def _periodic_gaussian(
    points: np.ndarray, centre: np.ndarray, exponent: float
) -> np.ndarray:
    images = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ _LATTICE
    values = np.zeros(len(points))
    for image in images:
        values += np.exp(-exponent * np.sum((points - centre - image) ** 2, axis=1))
    return values


@pytest.fixture(scope="module")
def water_cell() -> DensityGrid:
    """A spin-polarised water-like density in a skew periodic cell.

    Bond and lone-pair lobes give every atom a dipole and a quadrupole, and
    the spin sits mostly on, and slightly off, the oxygen.
    """
    steps = _LATTICE / np.array(_SHAPE)[:, None]
    points = np.indices(_SHAPE).reshape(3, -1).T @ steps
    volume = abs(np.linalg.det(steps))
    oxygen, first, second = _POSITIONS
    # Centre, electrons, Gaussian exponent, and fraction of them spin up.
    lobes = [
        (oxygen, 7.0, 1.6, 0.1),
        (first, 0.7, 1.1, 0.0),
        (second, 0.7, 1.1, 0.0),
        (0.6 * oxygen + 0.4 * first, 0.8, 1.4, 0.05),
        (0.5 * oxygen + 0.5 * second, 0.5, 1.3, 0.0),
        (oxygen + np.array([-0.3, -0.2, 0.25]), 0.3, 2.0, 0.6),
    ]
    density = np.zeros(len(points))
    spin = np.zeros(len(points))
    for centre, electrons, exponent, polarised in lobes:
        lobe = _periodic_gaussian(points, centre, exponent)
        lobe *= electrons / (lobe.sum() * volume)
        density += lobe
        spin += polarised * lobe
    return DensityGrid(
        atomic_numbers=_NUMBERS,
        positions=_POSITIONS,
        density=density.reshape(_SHAPE),
        origin=np.zeros(3),
        steps=steps,
        periodic=(True, True, True),
        spin_density=spin.reshape(_SHAPE),
    )


@pytest.fixture(scope="module")
def reference(water_cell: DensityGrid) -> DDECResult:
    return compute_ddec6(water_cell)


def _orthogonal(seed: int, *, proper: bool) -> np.ndarray:
    """A random rotation, or a rotation followed by a mirror."""
    matrix, upper = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    matrix *= np.sign(np.diag(upper))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix if proper else matrix @ np.diag([1.0, 1.0, -1.0])


def _tensors(quadrupoles: np.ndarray) -> np.ndarray:
    """Symmetric 3-by-3 tensors from ``xx, xy, xz, yy, yz, zz`` rows."""
    tensors = np.zeros((len(quadrupoles), 3, 3))
    rows, columns = np.triu_indices(3)
    tensors[:, rows, columns] = quadrupoles
    tensors[:, columns, rows] = quadrupoles
    return tensors


def _scalars(result: DDECResult) -> dict[str, np.ndarray]:
    assert result.spin is not None
    return {
        "net charges": result.charges,
        "Hirshfeld charges": result.hirshfeld_charges,
        "CM5 charges": result.cm5_charges,
        "spin moments": result.spin.populations,
        "summed bond orders": result.bonds.summed_bond_orders,
        "radial moments": result.radial_moments,
    }


def _assert_transformed(
    result: DDECResult,
    reference: DDECResult,
    *,
    rotation: np.ndarray | None = None,
    order: np.ndarray | None = None,
) -> None:
    """``result`` is ``reference`` turned by ``rotation`` with atoms reordered."""
    rotation = np.eye(3) if rotation is None else rotation
    order = np.arange(len(reference.charges)) if order is None else order
    expected = _scalars(reference)
    for name, values in _scalars(result).items():
        np.testing.assert_allclose(
            values, expected[name][order], rtol=0, atol=TOLERANCE, err_msg=name
        )
    for name in ("cloud_a", "cloud_b"):
        np.testing.assert_allclose(
            getattr(result.analysis, name),
            getattr(reference.analysis, name)[order],
            rtol=0,
            atol=FIT_TOLERANCE,
            err_msg=name,
        )
    np.testing.assert_allclose(
        result.analysis.atomic_dipoles,
        (reference.analysis.atomic_dipoles @ rotation.T)[order],
        rtol=0,
        atol=TOLERANCE,
    )
    turned = rotation @ _tensors(reference.analysis.atomic_quadrupoles) @ rotation.T
    np.testing.assert_allclose(
        _tensors(result.analysis.atomic_quadrupoles),
        turned[order],
        rtol=0,
        atol=TOLERANCE,
    )


def test_reference_has_moments_to_transform(reference: DDECResult) -> None:
    # Otherwise the tests below could pass by comparing zeros.
    assert np.all(np.linalg.norm(reference.analysis.atomic_dipoles, axis=1) > 1e-3)
    assert np.all(np.abs(reference.analysis.atomic_quadrupoles).max(axis=1) > 1e-3)
    assert reference.spin is not None
    assert reference.spin.populations[0] > 0.5


@pytest.mark.parametrize("proper", [True, False], ids=["rotation", "mirror"])
def test_results_turn_with_the_density(
    water_cell: DensityGrid, reference: DDECResult, proper: bool
) -> None:
    rotation = _orthogonal(0, proper=proper)
    turned = replace(
        water_cell,
        positions=water_cell.positions @ rotation.T,
        origin=water_cell.origin @ rotation.T,
        steps=water_cell.steps @ rotation.T,
    )
    _assert_transformed(compute_ddec6(turned), reference, rotation=rotation)


def test_results_ignore_whole_point_translations_across_the_cell(
    water_cell: DensityGrid, reference: DDECResult
) -> None:
    shift = (20, 13, 17)
    moved = water_cell.positions + np.asarray(shift) @ water_cell.steps
    fractional = moved @ np.linalg.inv(water_cell.lattice_vectors)
    # Some atoms leave the cell and come back in through the opposite face.
    assert np.any(fractional >= 1)
    fractional -= np.floor(fractional)
    assert water_cell.spin_density is not None
    rolled = replace(
        water_cell,
        positions=fractional @ water_cell.lattice_vectors,
        density=np.roll(water_cell.density, shift, axis=(0, 1, 2)),
        spin_density=np.roll(water_cell.spin_density, shift, axis=(0, 1, 2)),
    )
    _assert_transformed(compute_ddec6(rolled), reference)


def test_results_follow_their_atoms_through_a_reordering(
    water_cell: DensityGrid, reference: DDECResult
) -> None:
    order = np.array([2, 0, 1])
    assert water_cell.core_electrons is not None
    reordered = replace(
        water_cell,
        atomic_numbers=water_cell.atomic_numbers[order],
        positions=water_cell.positions[order],
        core_electrons=water_cell.core_electrons[order],
    )
    _assert_transformed(compute_ddec6(reordered), reference, order=order)


def test_independent_periodic_images_preserve_bonds_and_qdr(
    water_cell: DensityGrid, reference: DDECResult
) -> None:
    images = np.array([[10, -8, 6], [-9, 7, -5], [4, -3, 11]])
    positions = water_cell.positions + images @ water_cell.lattice_vectors
    unwrapped = replace(water_cell, positions=positions)
    result = compute_ddec6(unwrapped)
    _assert_transformed(result, reference)
    np.testing.assert_array_equal(result.analysis.positions, positions)
    np.testing.assert_array_equal(unwrapped.positions, positions)
    original_pairs, pairs = reference.overlaps, result.overlaps
    np.testing.assert_array_equal(pairs.atom1, original_pairs.atom1)
    np.testing.assert_array_equal(pairs.atom2, original_pairs.atom2)
    np.testing.assert_array_equal(
        pairs.translations,
        original_pairs.translations + images[pairs.atom1] - images[pairs.atom2],
    )
    np.testing.assert_allclose(
        positions[pairs.atom2]
        + pairs.translations @ water_cell.lattice_vectors
        - positions[pairs.atom1],
        water_cell.positions[original_pairs.atom2]
        + original_pairs.translations @ water_cell.lattice_vectors
        - water_cell.positions[original_pairs.atom1],
        rtol=0,
        atol=TOLERANCE,
    )
    for actual, expected in (
        (pairs.populations, original_pairs.populations),
        (result.bonds.bond_orders, reference.bonds.bond_orders),
        (result.bonds.contact_exchanges, reference.bonds.contact_exchanges),
        (result.bonds.spin_polarization_terms, reference.bonds.spin_polarization_terms),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
    actual_qdr, expected_qdr = result.compute_qdr(), reference.compute_qdr()
    for actual, expected in (
        (actual_qdr.charges, expected_qdr.charges),
        (actual_qdr.atomic_dipoles, expected_qdr.atomic_dipoles),
        (actual_qdr.atomic_quadrupoles, expected_qdr.atomic_quadrupoles),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
