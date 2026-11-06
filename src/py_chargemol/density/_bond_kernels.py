"""Fused numeric loops for optional bond-integration acceleration."""

import numpy as np

from py_chargemol import _jit


@_jit.jit
def spherical_shell_sums(  # noqa: PLR0913, PLR0917 -- Array-only JIT kernel.
    radial: np.ndarray,
    density: np.ndarray,
    inverse: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather a bond density and count its occupied points in one shell pass."""
    sums = np.zeros(len(radial))
    counts = sampled.copy()
    for sample_point in range(len(indices)):
        point = indices[sample_point]
        shell = shells[sample_point]
        local_inverse = inverse[point]
        sums[shell] += density[point] * radial[shell] * local_inverse
        if local_inverse == 0:
            counts[shell] -= 1
    return sums, counts


@_jit.jit
def add_exchange_vectors(
    radial: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    total: np.ndarray,
) -> None:
    """Scatter exchange components in sample order, retaining periodic images."""
    for sample_point in range(len(indices)):
        point = indices[sample_point]
        shell = shells[sample_point]
        for component in range(radial.shape[1]):
            total[point, component] += radial[shell, component]


@_jit.jit
def minimum_shell_density(
    total: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    minimum: np.ndarray,
) -> None:
    """Reduce the total density over each atom's shells without gathered arrays."""
    for sample_point in range(len(indices)):
        point = indices[sample_point]
        shell = shells[sample_point]
        minimum[shell] = np.minimum(minimum[shell], total[point, 0])


@_jit.jit
def exchange_values(  # noqa: PLR0913, PLR0917 -- Array-only JIT kernel.
    radial: np.ndarray,
    density: np.ndarray,
    total: np.ndarray,
    squared_total: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Form per-point exchanges, leaving the final pairwise sum to NumPy.

    Exchange vectors have one, two or four components. Their short NumPy
    reductions add components in order, which is retained here.
    """
    values = np.zeros(len(indices))
    for sample_point in range(len(indices)):
        point = indices[sample_point]
        denominator = squared_total[point]
        if denominator > threshold:
            shell = shells[sample_point]
            product = 0.0
            for component in range(radial.shape[1]):
                local = radial[shell, component]
                product += local * (total[point, component] - local)
            values[sample_point] = density[point] * product / denominator
    return values


@_jit.jit
def pair_shell_sums(  # noqa: PLR0913, PLR0917 -- Array-only JIT kernel.
    separation: np.ndarray,
    indices: np.ndarray,
    shells: np.ndarray,
    rows: np.ndarray,
    selected: np.ndarray,
    point_factors: np.ndarray,
    scale: float,
    zero: float,
    shell_count: int,
) -> np.ndarray:
    """Accumulate four point factors by shell pair in a single traversal.

    Repeated grid indices represent periodic images and contribute separately.
    Coordinate arithmetic and accumulation retain the NumPy implementation's
    evaluation order; shell-level contractions remain outside this kernel.
    """
    sums = np.zeros((4, shell_count**2))
    for sample_point in selected:
        dx = rows[0, sample_point] - separation[0]
        dy = rows[1, sample_point] - separation[1]
        dz = rows[2, sample_point] - separation[2]
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        other_shell = int(np.ceil(scale * distance + zero)) - 1
        if other_shell < shell_count:
            pair = int(shells[sample_point]) * shell_count + other_shell
            point = indices[sample_point]
            for component in range(4):
                sums[component, pair] += point_factors[component, point]
    return sums
