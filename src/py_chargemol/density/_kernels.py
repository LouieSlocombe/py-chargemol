"""Compiled array kernels for radial density accumulation.

Keep the point order and arithmetic grouping of the NumPy implementation.
In particular, repeated grid indices are periodic images, so these scatter
updates must remain serial.
"""

import numpy as np

from py_chargemol._jit import jit


@jit
def assigned_core_values(
    radial: np.ndarray,
    density: np.ndarray,
    inverse: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
) -> np.ndarray:
    """Gather core assignments, retaining NumPy's later population reductions."""
    values = np.empty(len(indices))
    for point in range(len(indices)):
        index = indices[point]
        values[point] = radial[shells[point]] * density[index] * inverse[index]
    return values


@jit
def add_corrected_core(
    values: np.ndarray,
    indices: np.ndarray,
    factor: float,
    threshold: float,
    result: np.ndarray,
) -> None:
    """Correct and scatter core assignments in periodic-image sample order."""
    for point in range(len(indices)):
        significant = values[point] if values[point] > threshold else 0.0
        corrected = significant / np.sqrt(1.0 - 2.0 * factor * significant**2)
        result[indices[point]] += corrected


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


@jit
def reference_shell_sums(  # noqa: PLR0913, PLR0917 -- Array inputs for one atom.
    conditioned: np.ndarray,
    density: np.ndarray,
    total: np.ndarray,
    reference_total: np.ndarray,
    conditioned_total: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    zero: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse DDEC3 reference ratios and tail statistics without point temporaries."""
    ratio_sum = np.zeros(len(conditioned))
    numerator = np.zeros(len(conditioned))
    denominator = np.zeros(len(conditioned))
    for point in range(len(indices)):
        index = indices[point]
        shell = shells[point]
        valid = total[index] > zero
        reference = reference_total[index]
        ratio = density[index] / reference if reference > zero**2 else 1.0
        ratio_sum[shell] += ratio * valid
        root = np.sqrt(conditioned_total[index]) if valid else 0.0
        value = 0.0
        if valid and conditioned_total[index] > zero**2:
            value = conditioned[shell] / root
        numerator[shell] += value
        denominator[shell] += root
    return ratio_sum, numerator, denominator
