"""Tests for the total multipole moment analysis."""

from pathlib import Path

import numpy as np
import pytest

from py_chargemol.ddec6 import read_ddec6_net_atomic_charges
from py_chargemol.multipoles import (
    MultipoleMoment,
    center_of_mass,
    total_multipole_moments,
)

CARBON, OXYGEN = 6, 8


def test_center_of_mass_of_carbon_monoxide(worked_examples: Path) -> None:
    """The QDR program logs the CO centre of mass as z = 1.223688 bohr."""
    analysis = read_ddec6_net_atomic_charges(
        worked_examples / "CO_molecule" / "DDEC6_even_tempered_net_atomic_charges.xyz"
    )

    origin = center_of_mass(analysis.atomic_numbers, analysis.positions)

    np.testing.assert_allclose(origin, [0.0, 0.0, 1.223688], atol=5e-7)


def single_atom(
    charge: float = 0.0,
    dipole: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quadrupole: tuple[float, ...] = (0.0,) * 6,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    return {
        "atomic_numbers": np.array([CARBON]),
        "positions": np.array([position]),
        "charges": np.array([charge]),
        "atomic_dipoles": np.array([dipole]),
        "atomic_quadrupoles": np.array([quadrupole]),
    }


def test_point_charges_give_charge_moments() -> None:
    """Charges +q and -q at z = +a and -a about the origin."""
    q, a = 0.3, 1.5
    moments = total_multipole_moments(
        atomic_numbers=np.array([OXYGEN, OXYGEN]),
        positions=np.array([[0.0, 0.0, a], [0.0, 0.0, -a]]),
        charges=np.array([q, -q]),
        atomic_dipoles=np.zeros((2, 3)),
        atomic_quadrupoles=np.zeros((2, 6)),
    )

    np.testing.assert_allclose(moments.center_of_mass, 0.0)
    np.testing.assert_allclose(moments.charges.dipole, [0.0, 0.0, 2 * q * a])
    np.testing.assert_allclose(moments.charges.quadrupole, 0.0)
    assert moments.charges.dipole_magnitude == pytest.approx(2 * q * a)


def test_a_single_atom_is_its_own_origin() -> None:
    moments = total_multipole_moments(**single_atom(charge=2.0, position=(1, 2, 3)))

    np.testing.assert_allclose(moments.center_of_mass, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(moments.charges.dipole, 0.0)
    np.testing.assert_allclose(moments.charges.quadrupole, 0.0)


def test_charge_quadrupole_components_about_the_centre_of_mass() -> None:
    """A charge on an equal-mass pair, displaced (1, 2, 3) from their midpoint."""
    moments = total_multipole_moments(
        atomic_numbers=np.array([CARBON, CARBON]),
        positions=np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        charges=np.array([0.5, 0.0]),
        atomic_dipoles=np.zeros((2, 3)),
        atomic_quadrupoles=np.zeros((2, 6)),
    )

    np.testing.assert_allclose(moments.charges.dipole, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        moments.charges.quadrupole, 0.5 * np.array([2.0, 3.0, 6.0, -3.0, 13.0])
    )


def test_atomic_dipoles_add_to_the_dipole() -> None:
    moments = total_multipole_moments(**single_atom(dipole=(0.1, -0.2, 0.3)))

    np.testing.assert_allclose(moments.charges.dipole, 0.0)
    np.testing.assert_allclose(moments.with_dipoles.dipole, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(moments.with_quadrupoles.dipole, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(moments.with_dipoles.quadrupole, 0.0)


def test_displaced_atomic_dipoles_add_to_the_quadrupole() -> None:
    """A dipole d at r contributes the symmetrised r d + d r to sum q r r."""
    moments = total_multipole_moments(
        atomic_numbers=np.array([CARBON, CARBON]),
        positions=np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
        charges=np.zeros(2),
        atomic_dipoles=np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
        atomic_quadrupoles=np.zeros((2, 6)),
    )

    x, y, z, dx, dy, dz = 1.0, 2.0, 3.0, 0.1, 0.2, 0.3
    np.testing.assert_allclose(
        moments.with_dipoles.quadrupole,
        [
            x * dy + y * dx,
            x * dz + z * dx,
            y * dz + z * dy,
            2 * (x * dx - y * dy),
            2 * (2 * z * dz - x * dx - y * dy),
        ],
    )


def test_atomic_quadrupoles_add_to_the_full_quadrupole() -> None:
    quadrupole = (0.5, 0.1, -0.2, 0.25, 0.3, -0.75)
    moments = total_multipole_moments(**single_atom(quadrupole=quadrupole))

    np.testing.assert_allclose(moments.with_dipoles.quadrupole, 0.0)
    np.testing.assert_allclose(
        moments.with_quadrupoles.quadrupole, [0.1, -0.2, 0.3, 0.25, -2.25]
    )
    # The tensor rebuilt from the five components is the atomic quadrupole itself.
    xx, xy, xz, yy, yz, zz = quadrupole
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    np.testing.assert_allclose(moments.with_quadrupoles.quadrupole_tensor, tensor)
    np.testing.assert_allclose(
        moments.with_quadrupoles.quadrupole_eigenvalues, np.linalg.eigvalsh(tensor)
    )


def test_eigenvalues_are_ascending_and_traceless() -> None:
    moment = MultipoleMoment(
        dipole=np.zeros(3), quadrupole=np.array([0.3, -0.1, 0.2, 1.0, -2.0])
    )

    eigenvalues = moment.quadrupole_eigenvalues

    assert np.all(np.diff(eigenvalues) >= 0)
    assert eigenvalues.sum() == pytest.approx(0.0, abs=1e-15)


def test_linear_molecule_has_a_degenerate_pair() -> None:
    """The QDR program reports -1.0075, 0.5037, 0.5037 for CO's full quadrupole."""
    moment = MultipoleMoment(
        dipole=np.zeros(3), quadrupole=np.array([0.0, 0.0, 0.0, 0.0, -3.0224])
    )

    np.testing.assert_allclose(
        moment.quadrupole_eigenvalues, [-1.0075, 0.5037, 0.5037], atol=1e-4
    )
