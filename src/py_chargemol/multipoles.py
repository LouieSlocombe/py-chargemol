"""Total dipole and quadrupole moments of a molecule from its atomic multipoles.

The moments are computed about the centre of mass, at three orders of the
distributed multipole expansion: the point charges alone, the point charges with the
atomic dipoles, and with the atomic quadrupoles too. Quadrupoles are reported, as by
Chargemol and the QDR program, as the five components ``xy, xz, yz, x^2 - y^2,
3z^2 - r^2`` of ``sum q r r``, so a point charge at ``(x, y, z)`` contributes
``q (x y, x z, y z, x^2 - y^2, 3 z^2 - r^2)``.

This is a port of ``module_total_multipole_moment_analysis.f08`` from T. A. Manz's
QDR program. The moments are only meaningful for nonperiodic systems.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from py_chargemol.elements import ATOMIC_WEIGHTS

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class MultipoleMoment:
    """Total dipole and quadrupole moment at one order of the expansion.

    Attributes:
        dipole: Dipole moment in e·bohr, shape ``(3,)``.
        quadrupole: Quadrupole components ``xy, xz, yz, x^2 - y^2, 3z^2 - r^2`` in
            e·bohr², shape ``(5,)``.
    """

    dipole: FloatArray
    quadrupole: FloatArray

    @property
    def dipole_magnitude(self) -> float:
        """Magnitude of the dipole moment in e·bohr."""
        return float(np.linalg.norm(self.dipole))

    @property
    def quadrupole_tensor(self) -> FloatArray:
        """The traceless tensor ``sum q (r r - r^2 I / 3)``, shape ``(3, 3)``."""
        xy, xz, yz, x2_minus_y2, three_z2_minus_r2 = self.quadrupole
        zz = three_z2_minus_r2 / 3.0
        return np.array(
            [
                [0.5 * (x2_minus_y2 - zz), xy, xz],
                [xy, 0.5 * (-x2_minus_y2 - zz), yz],
                [xz, yz, zz],
            ]
        )

    @property
    def quadrupole_eigenvalues(self) -> FloatArray:
        """Eigenvalues of the traceless quadrupole tensor, in ascending order."""
        return np.linalg.eigvalsh(self.quadrupole_tensor)


@dataclass(frozen=True, slots=True, eq=False)
class MultipoleMoments:
    """Total moments about the centre of mass at each order of the expansion.

    Attributes:
        center_of_mass: Origin of the moments in bohr, shape ``(3,)``.
        charges: Moments of the point charges alone.
        with_dipoles: Moments of the point charges and atomic dipoles.
        with_quadrupoles: Moments of the point charges, atomic dipoles, and atomic
            quadrupoles. Its dipole equals that of ``with_dipoles``.
    """

    center_of_mass: FloatArray
    charges: MultipoleMoment
    with_dipoles: MultipoleMoment
    with_quadrupoles: MultipoleMoment


def center_of_mass(atomic_numbers: IntArray, positions: FloatArray) -> FloatArray:
    """Return the centre of mass using standard atomic weights.

    Args:
        atomic_numbers: Atomic numbers, shape ``(n,)``.
        positions: Cartesian positions, shape ``(n, 3)``.

    Returns:
        The centre of mass, in the units of ``positions``.
    """
    masses = np.asarray(ATOMIC_WEIGHTS)[atomic_numbers - 1]
    return np.asarray(masses @ positions / masses.sum(), dtype=np.float64)


def total_multipole_moments(
    atomic_numbers: IntArray,
    positions: FloatArray,
    charges: FloatArray,
    atomic_dipoles: FloatArray,
    atomic_quadrupoles: FloatArray,
) -> MultipoleMoments:
    """Sum atomic multipoles into total moments about the centre of mass.

    Args:
        atomic_numbers: Atomic numbers, shape ``(n,)``.
        positions: Cartesian positions in bohr, shape ``(n, 3)``.
        charges: Atomic point charges in e, shape ``(n,)``.
        atomic_dipoles: Atomic dipoles in e·bohr, shape ``(n, 3)``.
        atomic_quadrupoles: Traceless atomic quadrupoles in e·bohr², ordered
            ``xx, xy, xz, yy, yz, zz``, shape ``(n, 6)``.

    Returns:
        The total moments at each order of the expansion.
    """
    origin = center_of_mass(atomic_numbers, positions)
    x, y, z = (positions - origin).T
    dx, dy, dz = atomic_dipoles.T
    qxx, qxy, qxz, qyy, qyz, qzz = atomic_quadrupoles.T

    charge_dipole = charges @ (positions - origin)
    charge_quadrupole = charges @ np.column_stack(
        [x * y, x * z, y * z, x**2 - y**2, 3 * z * z - (x**2 + y**2 + z**2)]
    )
    dipole = charge_dipole + atomic_dipoles.sum(axis=0)
    dipole_quadrupole = charge_quadrupole + np.sum(
        [
            x * dy + y * dx,
            x * dz + z * dx,
            y * dz + z * dy,
            2.0 * (x * dx - y * dy),
            2.0 * (2.0 * z * dz - x * dx - y * dy),
        ],
        axis=1,
    )
    full_quadrupole = dipole_quadrupole + np.sum(
        [qxy, qxz, qyz, qxx - qyy, 3.0 * qzz], axis=1
    )
    return MultipoleMoments(
        center_of_mass=origin,
        charges=MultipoleMoment(charge_dipole, charge_quadrupole),
        with_dipoles=MultipoleMoment(dipole, dipole_quadrupole),
        with_quadrupoles=MultipoleMoment(dipole.copy(), full_quadrupole),
    )
