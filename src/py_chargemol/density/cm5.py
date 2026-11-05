"""CM5 corrections to Hirshfeld charges, using Chargemol's 2017 parameters."""

from collections.abc import Sequence
from itertools import product

import numpy as np
import numpy.typing as npt

from py_chargemol.density.grid import integration_extent
from py_chargemol.units import BOHR_PER_ANGSTROM

type FloatArray = npt.NDArray[np.float64]

# Parameters of Marenich et al.'s CM5 model, as distributed with Chargemol 3.5.
_RADII = np.array(
    [
        0.32,
        0.37,
        1.3,
        0.99,
        0.84,
        0.75,
        0.71,
        0.64,
        0.6,
        0.62,
        1.6,
        1.4,
        1.24,
        1.14,
        1.09,
        1.04,
        1.0,
        1.01,
        2.0,
        1.74,
        1.59,
        1.48,
        1.44,
        1.3,
        1.29,
        1.24,
        1.18,
        1.17,
        1.22,
        1.2,
        1.23,
        1.2,
        1.2,
        1.18,
        1.17,
        1.16,
        2.15,
        1.9,
        1.76,
        1.64,
        1.56,
        1.46,
        1.38,
        1.36,
        1.34,
        1.3,
        1.36,
        1.4,
        1.42,
        1.4,
        1.4,
        1.37,
        1.36,
        1.36,
        2.38,
        2.06,
        1.94,
        1.84,
        1.9,
        1.88,
        1.86,
        1.85,
        1.83,
        1.82,
        1.81,
        1.8,
        1.79,
        1.77,
        1.77,
        1.78,
        1.74,
        1.64,
        1.58,
        1.5,
        1.41,
        1.36,
        1.32,
        1.3,
        1.3,
        1.32,
        1.44,
        1.45,
        1.5,
        1.42,
        1.48,
        1.46,
        2.42,
        2.11,
        2.01,
        1.9,
        1.84,
        1.83,
        1.8,
        1.8,
        1.73,
        1.68,
        1.68,
        1.68,
        1.65,
        1.67,
        1.73,
        1.76,
        1.61,
        1.57,
        1.49,
        1.43,
        1.41,
        1.34,
        1.29,
    ]
)
_DZ = np.array(
    [
        0.0056,
        -0.1543,
        0.0,
        0.0333,
        -0.103,
        -0.0446,
        -0.1072,
        -0.0802,
        -0.0629,
        -0.1088,
        0.0184,
        0.0,
        -0.0726,
        -0.079,
        -0.0756,
        -0.0565,
        -0.0444,
        -0.0767,
        0.013,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0512,
        -0.0557,
        -0.0533,
        -0.0399,
        -0.0313,
        -0.0541,
        0.0092,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0361,
        -0.0393,
        -0.0376,
        -0.0281,
        -0.022,
        -0.0381,
        0.0065,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0255,
        -0.0277,
        -0.0265,
        -0.0198,
        -0.0155,
        -0.0269,
        0.0046,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)
_ALPHA = 2.474
_DIMENSIONS = 3
_SPECIAL = {
    (1, 6): 0.0502,
    (1, 7): 0.1747,
    (1, 8): 0.1671,
    (6, 7): 0.0556,
    (6, 8): 0.0234,
    (7, 8): -0.0346,
}


def _translations(cell: FloatArray, periodic: Sequence[bool]) -> FloatArray:
    if not any(periodic):
        return np.zeros((1, 3))
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths == 0) or abs(np.linalg.det(cell)) < np.finfo(float).tiny:
        msg = "periodic CM5 requires three nonsingular lattice vectors"
        raise ValueError(msg)
    # The original program sums a rectangular set of translated cells whose
    # extent is determined by a 10 angstrom interaction radius.
    limits = integration_extent(cell, radius=10 * BOHR_PER_ANGSTROM)
    limits *= np.asarray(periodic)
    return np.array(list(product(*(range(-int(n), int(n) + 1) for n in limits)))) @ cell


def compute_cm5(
    *,
    atomic_numbers: npt.NDArray[np.int64],
    positions: FloatArray,
    hirshfeld_charges: FloatArray,
    lattice_vectors: FloatArray | None = None,
    periodic: Sequence[bool] = (False, False, False),
) -> FloatArray:
    """Return CM5 charges from Hirshfeld charges and nuclear positions.

    Positions and optional lattice vectors are in bohr. The net charge is
    conserved because each pair's correction is antisymmetric. CM5 uses no
    additional density partitioning; its input must be Hirshfeld charges.
    """
    numbers = np.asarray(atomic_numbers)
    coordinates = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(hirshfeld_charges, dtype=np.float64)
    if (
        numbers.ndim != 1
        or not numbers.size
        or np.any(numbers < 1)
        or np.any(numbers > len(_RADII))
        or np.any(numbers != np.trunc(numbers))
    ):
        msg = "atomic_numbers must contain integers in 1-109"
        raise ValueError(msg)
    if coordinates.shape != (len(numbers), 3) or charges.shape != (len(numbers),):
        msg = "positions and hirshfeld_charges must match atomic_numbers"
        raise ValueError(msg)
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(charges)):
        msg = "positions and hirshfeld_charges must be finite"
        raise ValueError(msg)
    if len(periodic) != _DIMENSIONS or any(
        not isinstance(value, (bool, np.bool_)) for value in periodic
    ):
        msg = "periodic must contain three booleans"
        raise ValueError(msg)
    cell = (
        np.zeros((3, 3))
        if lattice_vectors is None
        else np.asarray(lattice_vectors, dtype=np.float64)
    )
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        msg = "lattice_vectors must be a finite 3 by 3 array"
        raise ValueError(msg)
    translations = _translations(cell, periodic)
    elements, kinds = np.unique(numbers.astype(np.int64), return_inverse=True)
    coefficients = np.array(
        [[_coefficient(int(a), int(b)) for b in elements] for a in elements]
    )
    radii = _RADII[elements - 1][kinds]
    result = charges.copy()
    # Each atom's corrections with all later atoms and their periodic images.
    for first in range(len(numbers) - 1):
        others = slice(first + 1, None)
        distances = (
            np.linalg.norm(
                coordinates[others, None, :] + translations - coordinates[first],
                axis=2,
            )
            / BOHR_PER_ANGSTROM
        )
        corrections = coefficients[kinds[first], kinds[others]] * np.exp(
            -_ALPHA * (distances - radii[first] - radii[others, None])
        ).sum(axis=1)
        result[first] += corrections.sum()
        result[others] -= corrections
    return result


def _coefficient(first: int, second: int) -> float:
    """Return the CM5 charge-transfer coefficient for two atomic numbers."""
    coefficient = _SPECIAL.get((first, second))
    if coefficient is not None:
        return coefficient
    reverse = _SPECIAL.get((second, first))
    if reverse is not None:
        return -reverse
    return float(_DZ[first - 1] - _DZ[second - 1])
