"""Fused numeric loops for optional bond-integration acceleration."""

import numpy as np

from py_chargemol import _jit


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
