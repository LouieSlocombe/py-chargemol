"""Compute QDR charges directly from Python sequences or NumPy arrays."""

import numpy as np
import numpy.typing as npt

from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.elements import ATOMIC_WEIGHTS
from py_chargemol.qdr.solver import QDRResult, compute_qdr_charges

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


def compute_qdr_from_arrays(  # noqa: PLR0913 -- Each input names a physical array.
    *,
    atomic_numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    net_charges: npt.ArrayLike,
    atomic_dipoles: npt.ArrayLike,
    atomic_quadrupoles: npt.ArrayLike,
    atom1: npt.ArrayLike,
    atom2: npt.ArrayLike,
    populations: npt.ArrayLike,
    translations: npt.ArrayLike | None = None,
    lattice_vectors: npt.ArrayLike | None = None,
    max_cg_iterations: int = 10,
    tolerance: float = 1e-12,
) -> QDRResult:
    """Compute QDR charges from in-memory data without reading or writing files.

    Inputs may be Python sequences or NumPy arrays and are never modified. All
    quantities use atomic units. Each bond must appear once from each end, with
    opposite lattice translations; this function does not add reverse pairs.

    Args:
        atomic_numbers: Atomic numbers from 1 to 109, shape ``(n,)``.
        positions: Cartesian positions in bohr, shape ``(n, 3)``.
        net_charges: DDEC6 net atomic charges in e, shape ``(n,)``.
        atomic_dipoles: Atomic dipoles in e·bohr, shape ``(n, 3)``.
        atomic_quadrupoles: Traceless atomic quadrupoles in e·bohr², ordered
            ``xx, xy, xz, yy, yz, zz``, shape ``(n, 6)``.
        atom1: Zero-based owner atom indices of directed bonds, shape ``(m,)``.
        atom2: Zero-based partner atom indices of directed bonds, shape ``(m,)``.
        populations: Bond overlap populations in e, shape ``(m,)``.
        translations: Integer lattice translations of each partner atom,
            shape ``(m, 3)``. Defaults to zero for every pair.
        lattice_vectors: Lattice vectors in bohr as rows, shape ``(3, 3)``.
            Defaults to a nonperiodic system. A zero row is nonperiodic.
        max_cg_iterations: Positive conjugate gradient iteration limit per step.
        tolerance: Finite nonnegative residual tolerance per atom.

    Returns:
        QDR point charges, residual atomic multipoles, solver statistics, and
        total molecular moments (``None`` for periodic systems).

    Raises:
        ValueError: If inputs have inconsistent shapes, contain nonfinite values
            or invalid integers, name nonexistent atoms, or form zero-length bonds.
    """
    numbers = _integer_array(atomic_numbers, "atomic_numbers", (None,))
    natoms = len(numbers)
    if natoms == 0:
        msg = "atomic_numbers must contain at least one atom"
        raise ValueError(msg)
    if np.any((numbers < 1) | (numbers > len(ATOMIC_WEIGHTS))):
        msg = f"atomic_numbers must lie in 1-{len(ATOMIC_WEIGHTS)}"
        raise ValueError(msg)

    first = _integer_array(atom1, "atom1", (None,))
    npairs = len(first)
    pairs = OverlapPairs(
        atom1=first,
        atom2=_integer_array(atom2, "atom2", (npairs,)),
        translations=(
            np.zeros((npairs, 3), dtype=np.int64)
            if translations is None
            else _integer_array(translations, "translations", (npairs, 3))
        ),
        populations=_float_array(populations, "populations", (npairs,)),
    )
    analysis = DDEC6Analysis(
        atomic_numbers=numbers,
        positions=_float_array(positions, "positions", (natoms, 3)),
        net_charges=_float_array(net_charges, "net_charges", (natoms,)),
        atomic_dipoles=_float_array(atomic_dipoles, "atomic_dipoles", (natoms, 3)),
        atomic_quadrupoles=_float_array(
            atomic_quadrupoles, "atomic_quadrupoles", (natoms, 6)
        ),
        # Cloud parameters are unused by the solver. These placeholders stay
        # internal and are never exported as an analysis or an output file.
        cloud_a=np.zeros(natoms),
        cloud_b=np.zeros(natoms),
        lattice_vectors=(
            np.zeros((3, 3))
            if lattice_vectors is None
            else _float_array(lattice_vectors, "lattice_vectors", (3, 3))
        ),
    )
    return compute_qdr_charges(
        analysis,
        pairs,
        max_cg_iterations=max_cg_iterations,
        tolerance=tolerance,
    )


def _array(
    values: npt.ArrayLike, name: str, shape: tuple[int | None, ...]
) -> npt.NDArray[np.generic]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a numeric array"
        raise ValueError(msg) from error
    if len(array.shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        msg = f"{name} must have shape {shape}, got {array.shape}"
        raise ValueError(msg)
    return array


def _float_array(
    values: npt.ArrayLike, name: str, shape: tuple[int, ...]
) -> FloatArray:
    array = _array(values, name, shape)
    if np.iscomplexobj(array):
        msg = f"{name} must contain real numbers"
        raise ValueError(msg)
    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"{name} must contain real numbers"
        raise ValueError(msg) from error
    if not np.all(np.isfinite(result)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return result


def _integer_array(
    values: npt.ArrayLike, name: str, shape: tuple[int | None, ...]
) -> IntArray:
    array = _array(values, name, shape)
    valid = array.dtype.kind == "i"
    if array.dtype.kind == "u":
        valid = bool(np.all(np.less_equal(array, np.iinfo(np.int64).max)))
    elif array.dtype.kind == "f":
        valid = bool(
            np.all(np.isfinite(array))
            and np.all(array == np.trunc(array))
            and np.all(np.greater_equal(array, np.float64(-(2**63))))
            and np.all(np.less(array, np.float64(2**63)))
        )
    if not valid:
        msg = f"{name} must contain integers representable as int64"
        raise ValueError(msg)
    return np.array(array, dtype=np.int64, copy=True)
