"""Compiled array kernels for radial density accumulation.

Keep the point order and arithmetic grouping of the NumPy implementation.
In particular, repeated grid indices are periodic images, so these scatter
updates must remain serial.
"""

import numpy as np

from py_chargemol._jit import jit


@jit
def add_radial_weights(
    radial: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    result: np.ndarray,
) -> None:
    """Add one atom's radial weights to its grid points in sample order."""
    for point in range(len(indices)):
        result[indices[point]] += radial[shells[point]]


@jit
def core_shell_sums(  # noqa: PLR0913, PLR0917 -- Array inputs for one atom.
    radial: np.ndarray,
    density: np.ndarray,
    inverse: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather core density and accumulate its occupied shell counts and sums.

    Return the point values so the caller can retain NumPy's pairwise sum for
    the total population, which controls the core iteration's stopping rule.
    """
    values = np.empty(len(indices))
    counts = sampled.copy()
    sums = np.zeros(len(radial))
    for point in range(len(indices)):
        index = indices[point]
        shell = shells[point]
        local_inverse = inverse[index]
        value = radial[shell] * density[index] * local_inverse
        values[point] = value
        sums[shell] += value
        if local_inverse == 0:
            counts[shell] -= 1
    return values, counts, sums


@jit
def partition_shell_sums(  # noqa: PLR0913, PLR0917 -- Array inputs for one atom.
    radial: np.ndarray,
    density: np.ndarray,
    inverse: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate all four partition shell statistics in a single point pass."""
    counts = sampled.copy()
    assigned = np.zeros(len(radial))
    weighted_points = np.zeros(len(radial))
    weighted_density = np.zeros(len(radial))
    for point in range(len(indices)):
        index = indices[point]
        shell = shells[point]
        local_inverse = inverse[index]
        fraction = radial[shell] * local_inverse
        value = fraction * density[index]
        assigned[shell] += value
        weighted_points[shell] += fraction
        weighted_density[shell] += fraction * value
        if local_inverse == 0:
            counts[shell] -= 1
    return counts, assigned, weighted_points, weighted_density
