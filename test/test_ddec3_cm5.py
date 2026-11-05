"""DDEC3 iteration and CM5 charge-correction regressions."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from py_chargemol.density.cm5 import compute_cm5
from py_chargemol.density.ddec3 import _reshape, partition_ddec3
from py_chargemol.density.partition import partition_ddec6
from py_chargemol.density.references import reference_density
from py_chargemol.units import BOHR_PER_ANGSTROM


@dataclass(frozen=True)
class _Sample:
    indices: np.ndarray
    shells: np.ndarray


def test_ddec3_symmetric_reference_density_conserves_population() -> None:
    reference = reference_density(1)
    sample = _Sample(np.arange(100), np.arange(100))
    result = partition_ddec3(
        atomic_numbers=np.array([1, 1]),
        density=2 * reference,
        samples=[sample, sample],
        pixel_volume=1 / float(reference.sum()),
        core_electrons=np.zeros(2),
        core_population=np.zeros(2),
        occupancy_correction=np.zeros(2),
    )
    np.testing.assert_allclose(result.charges, [0, 0], atol=1e-12)
    np.testing.assert_allclose(result.hirshfeld_charges, [0, 0], atol=1e-12)
    np.testing.assert_allclose(result.weights[0], result.weights[1])
    assert result.iterations == 10
    assert result.normalization == pytest.approx(1)


def test_ddec3_rejects_electronless_input() -> None:
    with pytest.raises(ValueError, match="positive number of valence"):
        partition_ddec3(
            atomic_numbers=np.array([1]),
            density=np.zeros(1),
            samples=[],
            pixel_volume=1,
            core_electrons=np.zeros(1),
            core_population=np.zeros(1),
            occupancy_correction=np.zeros(1),
            net_charge=1,
        )


@pytest.mark.parametrize("method", ["ddec3", "ddec6"])
def test_bare_proton_next_to_a_separated_hydrogen_has_no_density(method: str) -> None:
    partition = partition_ddec3 if method == "ddec3" else partition_ddec6
    result = partition(
        atomic_numbers=np.array([1, 1]),
        density=np.array([0.0, 1.0]),
        samples=[
            _Sample(np.array([0]), np.array([0])),
            _Sample(np.array([1]), np.array([0])),
        ],
        pixel_volume=1,
        core_electrons=np.zeros(2),
        core_population=np.zeros(2),
        occupancy_correction=np.zeros(2),
        net_charge=1,
    )
    np.testing.assert_allclose(result.charges, [1, 0], atol=1e-12)
    np.testing.assert_array_equal(result.weights[0], np.zeros(100))


def test_ddec3_tail_reshaping_controls_an_overly_diffuse_distribution() -> None:
    density = np.linspace(0.1, 1, 100)
    points = np.ones(100)
    points[20:30] = 0
    result = _reshape(density, points, np.zeros(100))
    assert np.all(result >= 0)
    assert result[1] < result[0]
    assert result[-1] < density[-1]
    assert result @ points == pytest.approx(density @ points, abs=1e-12)


@pytest.mark.parametrize(
    ("first_density", "outer_shell", "message"),
    [
        (1.5, 10, "did not converge within 2000 iterations"),
        (1.99, 15, "nonnegative valence populations"),
    ],
)
def test_ddec3_rejects_unresolved_core_population_constraints(
    first_density: float, outer_shell: int, message: str
) -> None:
    # A core-rich Li site cannot surrender its two core electrons to H. On an
    # inadequate two-point grid, that constraint can stall or falsely converge.
    with pytest.raises(ValueError, match=message):
        partition_ddec3(
            atomic_numbers=np.array([3, 1]),
            density=np.array([first_density, 4 - first_density]),
            samples=[
                _Sample(np.array([0, 1]), np.array([0, outer_shell])),
                _Sample(np.array([0, 1]), np.array([outer_shell, 0])),
            ],
            pixel_volume=1,
            core_electrons=np.array([2, 0]),
            core_population=np.array([2, 0]),
            occupancy_correction=np.zeros(2),
        )


def test_periodic_cm5_matches_shipped_nacl_reference() -> None:
    # VASP5_NaCl_bulk_example: the source program prints Hirshfeld 0.204737
    # and CM5 0.420172, each to six decimal places.
    cell = (
        5.65
        * BOHR_PER_ANGSTROM
        * np.array(
            [[0, 0.500787, 0.500787], [0.500787, 0, 0.500787], [0.500787, 0.500787, 0]]
        )
    )
    result = compute_cm5(
        atomic_numbers=np.array([11, 17]),
        positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ cell,
        hirshfeld_charges=np.array([0.204737, -0.204737]),
        lattice_vectors=cell,
        periodic=(True, True, True),
    )
    np.testing.assert_allclose(result, [0.420172, -0.420172], atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("first", "second", "coefficient", "radius_sum"),
    [
        (1, 6, 0.0502, 1.07),
        (1, 7, 0.1747, 1.03),
        (1, 8, 0.1671, 0.96),
        (6, 7, 0.0556, 1.46),
        (6, 8, 0.0234, 1.39),
        (7, 8, -0.0346, 1.35),
    ],
)
def test_cm5_pair_overrides_are_antisymmetric(
    first: int, second: int, coefficient: float, radius_sum: float
) -> None:
    positions = np.array([[0, 0, 0], [2, 0, 0]]) * BOHR_PER_ANGSTROM
    correction = coefficient * np.exp(-2.474 * (2 - radius_sum))
    for numbers, expected in (
        ([first, second], [correction, -correction]),
        ([second, first], [-correction, correction]),
    ):
        result = compute_cm5(
            atomic_numbers=np.array(numbers),
            positions=positions,
            hirshfeld_charges=np.array([0.1, 0.2]),
        )
        np.testing.assert_allclose(result, np.array(expected) + np.array([0.1, 0.2]))
        assert result.sum() == pytest.approx(0.3)


def test_equal_elements_have_no_cm5_correction() -> None:
    charges = np.array([0.3, -0.3])
    result = compute_cm5(
        atomic_numbers=np.array([6, 6]),
        positions=np.array([[0, 0, 0], [2, 0, 0]]),
        hirshfeld_charges=charges,
    )
    np.testing.assert_array_equal(result, charges)
    result[:] = 0
    np.testing.assert_array_equal(charges, [0.3, -0.3])


def test_periodic_cm5_is_invariant_under_a_triclinic_basis_change() -> None:
    arguments: dict[str, Any] = {
        "atomic_numbers": np.array([1, 8]),
        "positions": np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]]),
        "hirshfeld_charges": np.zeros(2),
        "periodic": (True, True, True),
    }
    cubic = compute_cm5(**arguments, lattice_vectors=4 * np.eye(3))
    # An integer shear with determinant one describes the same periodic system.
    skew = compute_cm5(
        **arguments,
        lattice_vectors=4 * np.array([[1, 0, 0], [0, 1, 0], [10, 10, 1]]),
    )
    # The rectangular image domains retain different exponentially small tails.
    np.testing.assert_allclose(skew, cubic, rtol=0, atol=1e-9)


@pytest.mark.parametrize("periodic", [(True, True, True), (True, False, True)])
def test_cm5_is_invariant_under_independent_atom_images(
    periodic: tuple[bool, bool, bool],
) -> None:
    cell = np.array([[7.0, 0, 0], [1.2, 6.6, 0], [0.7, 0.9, 7.3]])
    positions = np.array([[1.0, 1.4, 0.6], [2.6, 1.1, 0.9]])
    images = np.array([[10, -8, 6], [-9, 7, -5]]) * periodic
    unwrapped = positions + images @ cell
    arguments: dict[str, Any] = {
        "atomic_numbers": np.array([1, 8]),
        "hirshfeld_charges": np.array([0.2, -0.2]),
        "lattice_vectors": cell,
        "periodic": periodic,
    }
    expected = compute_cm5(positions=positions, **arguments)
    actual = compute_cm5(positions=unwrapped, **arguments)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(unwrapped, positions + images @ cell)


def test_cm5_keeps_nonperiodic_atom_separations() -> None:
    cell = np.diag([7.0, 8.0, 9.0])
    # The y separation spans ten cell vectors, but y is not periodic.
    charges = np.array([0.2, -0.2])
    actual = compute_cm5(
        atomic_numbers=np.array([1, 8]),
        positions=np.array([[1.0, 1.0, 1.0], [2.0, 81.0, 1.0]]),
        hirshfeld_charges=charges,
        lattice_vectors=cell,
        periodic=(True, False, True),
    )
    np.testing.assert_allclose(actual, charges, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("atomic_numbers", np.array([0]), "integers in 1-109"),
        ("positions", np.zeros((2, 3)), "must match atomic_numbers"),
        ("positions", np.full((1, 3), np.nan), "must be finite"),
        ("periodic", (True,), "three booleans"),
        ("periodic", (True, False, False), "nonsingular lattice"),
        ("lattice_vectors", np.zeros((2, 3)), "finite 3 by 3"),
    ],
)
def test_cm5_validates_geometry(name: str, value: object, message: str) -> None:
    arguments: dict[str, Any] = {
        "atomic_numbers": np.array([1]),
        "positions": np.zeros((1, 3)),
        "hirshfeld_charges": np.zeros(1),
    }
    arguments[name] = value
    with pytest.raises(ValueError, match=message):
        compute_cm5(**arguments)
