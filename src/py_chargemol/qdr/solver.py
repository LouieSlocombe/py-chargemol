"""Quadrupole-dipole resorption of DDEC6 atomic multipoles into point charges.

QDR moves part of each atom's quadrupole and dipole onto the point charges of its
bonded neighbours, so that a model truncated at monopole order reproduces more of
the electrostatic potential. Each bond ``j`` from atom ``A`` carries a weight
``w_j = 2 tanh(2 p_j)`` from its overlap population ``p_j``, a length ``r_j``, and a
unit direction ``u_j``.

The method makes two passes, each a quadrupole step followed by a dipole step. In
every step each atom solves a small symmetric positive definite system
``(I + B diag(c) B^T) y = t`` over its own bonds, with conjugate gradients:

- Quadrupole step: ``B = w [1, ux ux, ux uy, ux uz, uy uy, uy uz, uz uz]``,
  ``c = [-1/3, 1, 2, 2, 1, 2, 1]`` and ``t = -w u^T Q_A u``. The solution moves
  charge ``y w / r^2`` from the partner atom onto ``A``, adds ``y w u / r`` to the
  dipole of ``A``, and adds ``y w (u u^T - I/3)`` to its quadrupole.
- Dipole step: ``B = w u``, ``c = [1, 1, 1]`` and ``t = -w u . d_A``. The solution
  moves charge ``q = y w / r`` from the partner atom onto ``A`` and adds
  ``q r u / 2`` to the dipoles of both atoms.

This is a port of ``module_QDR_point_charges.f08`` from T. A. Manz's QDR program
(April 9, 2025). The Fortran solves one atom per OpenMP thread. Here every atom's
conjugate gradient runs at once on NumPy arrays, with per-atom step lengths and
convergence, which gives the same iterates.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.elements import ATOMIC_WEIGHTS
from py_chargemol.multipoles import MultipoleMoments, total_multipole_moments

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_PASSES: Final = 2
"""Number of quadrupole-then-dipole resorption passes."""

_ONE_THIRD: Final = 1.0 / 3.0
_QUADRUPOLE_COEFFICIENTS: Final = np.array([-_ONE_THIRD, 1, 2, 2, 1, 2, 1], float)
_DIPOLE_COEFFICIENTS: Final = np.ones(3)
_IDENTITY_COMPONENTS: Final = np.array([1, 0, 0, 1, 0, 1], float)
"""The identity tensor in ``xx, xy, xz, yy, yz, zz`` order."""


@dataclass(frozen=True, slots=True, eq=False)
class QDRResult:
    """QDR point charges and the atomic multipoles they leave behind.

    Attributes:
        charges: QDR point charges in e, shape ``(n,)``.
        atomic_dipoles: Residual atomic dipoles in e·bohr, shape ``(n, 3)``.
        atomic_quadrupoles: Residual traceless atomic quadrupoles in e·bohr², ordered
            ``xx, xy, xz, yy, yz, zz``, shape ``(n, 6)``.
        cg_iterations: Most conjugate gradient iterations any atom took in any step.
        residual_norm: Largest L2 norm, over the steps, of the final conjugate
            gradient residual of all atoms together.
        moments: Total molecular moments about the centre of mass, or ``None``
            for periodic systems. Includes charges alone, with residual dipoles,
            and with residual quadrupoles.
    """

    charges: FloatArray
    atomic_dipoles: FloatArray
    atomic_quadrupoles: FloatArray
    cg_iterations: int
    residual_norm: float
    moments: MultipoleMoments | None = None


def compute_qdr_charges(
    analysis: DDEC6Analysis,
    pairs: OverlapPairs,
    *,
    max_cg_iterations: int = 10,
    tolerance: float = 1e-12,
) -> QDRResult:
    """Compute QDR point charges from in-memory DDEC6 data without file I/O.

    Input arrays are not modified. A system with no overlap pairs retains its
    original charges and multipoles, including a single isolated atom.

    Args:
        analysis: DDEC6 net atomic charges, atomic multipoles, and geometry.
        pairs: Bond overlap populations, listing each bond from both ends.
        max_cg_iterations: Conjugate gradient iteration limit for each atom's
            system in each step.
        tolerance: An atom's conjugate gradient stops once the L2 norm of its
            residual is at most this.

    Returns:
        The QDR point charges, residual atomic multipoles, solver statistics,
        and total moments for nonperiodic systems.

    Raises:
        ValueError: If atomic numbers or solver settings are invalid, a pair
            names an atom outside the analysis, or a bond has zero length.
    """
    if (
        analysis.atomic_numbers.ndim != 1
        or analysis.natoms == 0
        or analysis.atomic_numbers.dtype.kind not in "iu"
        or np.any(
            (analysis.atomic_numbers < 1)
            | (analysis.atomic_numbers > len(ATOMIC_WEIGHTS))
        )
    ):
        msg = (
            "atomic_numbers must be a nonempty integer array "
            f"in 1-{len(ATOMIC_WEIGHTS)}"
        )
        raise ValueError(msg)
    _validate_iteration_limit(max_cg_iterations)
    if not np.isfinite(tolerance) or tolerance < 0:
        msg = "tolerance must be finite and nonnegative"
        raise ValueError(msg)

    resorption = _Resorption(
        bonds=_Bonds.from_pairs(analysis, pairs),
        charges=analysis.net_charges.copy(),
        dipoles=analysis.atomic_dipoles.copy(),
        quadrupoles=analysis.atomic_quadrupoles.copy(),
        max_iterations=max_cg_iterations,
        tolerance=tolerance,
    )
    if len(pairs):
        for _ in range(_PASSES):
            resorption.resorb_quadrupoles()
            resorption.resorb_dipoles()

    moments = None
    if not analysis.is_periodic:
        moments = total_multipole_moments(
            analysis.atomic_numbers,
            analysis.positions,
            resorption.charges,
            resorption.dipoles,
            resorption.quadrupoles,
        )

    return QDRResult(
        charges=resorption.charges,
        atomic_dipoles=resorption.dipoles,
        atomic_quadrupoles=resorption.quadrupoles,
        cg_iterations=resorption.cg_iterations,
        residual_norm=resorption.residual_norm,
        moments=moments,
    )


def _validate_iteration_limit(value: object) -> None:
    """Check runtime callers, including NumPy integers and untyped Python code."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        msg = "max_cg_iterations must be a positive integer"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class _Bonds:
    """Bond geometry, sorted by the atom that owns each linear system."""

    natoms: int
    owner: IntArray
    partner: IntArray
    weight: FloatArray
    length: FloatArray
    direction: FloatArray
    direction_products: FloatArray
    """``u u^T`` in ``xx, xy, xz, yy, yz, zz`` order, shape ``(m, 6)``."""

    @classmethod
    def from_pairs(cls, analysis: DDEC6Analysis, pairs: OverlapPairs) -> _Bonds:
        """Compute the bonds' weights and geometry, grouped by their first atom."""
        natoms = analysis.natoms
        for name, atoms in (("atom1", pairs.atom1), ("atom2", pairs.atom2)):
            if np.any((atoms < 0) | (atoms >= natoms)):
                msg = f"overlap pair {name} indices must lie in 0-{natoms - 1}"
                raise ValueError(msg)

        order = np.argsort(pairs.atom1, kind="stable")
        owner, partner = pairs.atom1[order], pairs.atom2[order]
        vectors = (
            analysis.positions[partner]
            - analysis.positions[owner]
            + pairs.translations[order] @ analysis.lattice_vectors
        )
        length = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
        if np.any(length == 0.0):
            msg = "an overlap pair joins an atom to itself in the same unit cell"
            raise ValueError(msg)
        direction = vectors / length[:, None]
        ux, uy, uz = direction.T
        return cls(
            natoms=natoms,
            owner=owner,
            partner=partner,
            weight=2.0 * np.tanh(2.0 * pairs.populations[order]),
            length=length,
            direction=direction,
            direction_products=np.column_stack(
                [ux * ux, ux * uy, ux * uz, uy * uy, uy * uz, uz * uz]
            ),
        )

    def owner_sum(self, values: FloatArray) -> FloatArray:
        """Sum per-bond values, shape ``(m,)`` or ``(m, k)``, over each owner atom."""
        return _scatter_add(self.owner, values, self.natoms)

    def partner_sum(self, values: FloatArray) -> FloatArray:
        """Sum per-bond values over each partner atom."""
        return _scatter_add(self.partner, values, self.natoms)


@dataclass(slots=True, eq=False)
class _Resorption:
    """The multipoles being resorbed, updated in place step by step."""

    bonds: _Bonds
    charges: FloatArray
    dipoles: FloatArray
    quadrupoles: FloatArray
    max_iterations: int
    tolerance: float
    cg_iterations: int = 0
    residual_norm: float = 0.0

    def resorb_quadrupoles(self) -> None:
        """Move part of each atomic quadrupole onto the neighbours' charges."""
        bonds, w = self.bonds, self.bonds.weight
        basis = w[:, None] * np.column_stack(
            [np.ones_like(w), bonds.direction_products]
        )
        rhs = -np.einsum(
            "jk,k,jk->j",
            basis[:, 1:],
            _QUADRUPOLE_COEFFICIENTS[1:],
            self.quadrupoles[bonds.owner],
        )
        yw = self._solve(basis, _QUADRUPOLE_COEFFICIENTS, rhs) * w

        self.dipoles += bonds.owner_sum((yw / bonds.length)[:, None] * bonds.direction)
        self.quadrupoles += bonds.owner_sum(
            yw[:, None] * (bonds.direction_products - _ONE_THIRD * _IDENTITY_COMPONENTS)
        )
        self._transfer_charge(yw / bonds.length**2)

    def resorb_dipoles(self) -> None:
        """Move part of each atomic dipole onto the neighbours' charges."""
        bonds = self.bonds
        basis = bonds.weight[:, None] * bonds.direction
        rhs = -np.einsum("jk,jk->j", basis, self.dipoles[bonds.owner])
        y = self._solve(basis, _DIPOLE_COEFFICIENTS, rhs)

        transferred = y * bonds.weight / bonds.length
        self._transfer_charge(transferred)
        shift = (0.5 * transferred * bonds.length)[:, None] * bonds.direction
        self.dipoles += bonds.owner_sum(shift) + bonds.partner_sum(shift)

    def _transfer_charge(self, amount: FloatArray) -> None:
        """Move ``amount`` of charge along each bond from its partner to its owner."""
        self.charges += self.bonds.owner_sum(amount) - self.bonds.partner_sum(amount)

    def _solve(
        self, basis: FloatArray, coefficients: FloatArray, rhs: FloatArray
    ) -> FloatArray:
        """Solve every atom's ``(I + B diag(c) B^T) y = t`` by conjugate gradients.

        Each atom's system couples only that atom's bonds, so all of them iterate
        together with per-atom step lengths. An atom stops once its residual norm
        is at most the tolerance, checked before each iteration as in the Fortran.
        The iteration count and residual norm statistics are updated.
        """
        bonds, owner = self.bonds, self.bonds.owner

        def matvec(p: FloatArray) -> FloatArray:
            projections = bonds.owner_sum(basis * p[:, None]) * coefficients
            result: FloatArray = p + np.einsum("jk,jk->j", basis, projections[owner])
            return result

        y = np.zeros_like(rhs)
        residual = rhs.copy()
        search = residual.copy()
        residual_squared = bonds.owner_sum(residual * residual)
        iterations = np.zeros(bonds.natoms, np.int64)
        for _ in range(self.max_iterations):
            active = np.sqrt(residual_squared) > self.tolerance
            if not active.any():
                break
            product = matvec(search)
            alpha = _divide_where(
                residual_squared, bonds.owner_sum(search * product), active
            )
            y += alpha[owner] * search
            residual -= alpha[owner] * product
            new_residual_squared = bonds.owner_sum(residual * residual)
            beta = _divide_where(new_residual_squared, residual_squared, active)
            search = np.where(active[owner], residual + beta[owner] * search, search)
            residual_squared = np.where(active, new_residual_squared, residual_squared)
            iterations += active

        self.cg_iterations = max(self.cg_iterations, int(iterations.max()))
        self.residual_norm = max(
            self.residual_norm, float(np.sqrt(residual_squared.sum()))
        )
        return y


def _divide_where(
    numerator: FloatArray, denominator: FloatArray, where: npt.NDArray[np.bool]
) -> FloatArray:
    """Divide where ``where`` holds, and give zero elsewhere."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=where)


def _scatter_add(index: IntArray, values: FloatArray, size: int) -> FloatArray:
    """Sum rows of ``values`` into ``size`` bins by ``index``."""
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=size)
    return np.column_stack(
        [np.bincount(index, weights=column, minlength=size) for column in values.T]
    )
