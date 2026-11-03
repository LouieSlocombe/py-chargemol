"""Population-preserving radial constraints for DDEC6 weighting functions.

These apply the constraints in Chargemol's ``module_reshaping_functions``.
The scalar normalization constraint is piecewise linear, so Newton steps solve
it exactly in a few evaluations, where Chargemol bisects.
"""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

_TOLERANCE = 1e-10
_NEWTON_ITERATIONS = 100


def _limit_density(
    density: FloatArray, direction: FloatArray, weights: FloatArray, decay: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Limit each shell by its predecessor, with the derivative along ``direction``.

    A limited shell inherits its predecessor's derivative times the decay factor.
    """
    result = np.maximum(density, 0.0)
    slope = direction.copy()
    for shell in range(1, len(result)):
        if weights[shell] > 0 and result[shell - 1] > 0:
            limit = result[shell - 1] * decay[shell]
            if limit < result[shell]:
                result[shell] = limit
                slope[shell] = slope[shell - 1] * decay[shell]
    return result, slope


def _restore_population(
    density: FloatArray, weights: FloatArray, decay: FloatArray
) -> tuple[FloatArray, bool]:
    population = float(density @ weights)
    root_density = np.sqrt(density)
    weighted_points = float(root_density @ weights)
    if min(population, weighted_points) < _TOLERANCE:
        return density.copy(), False
    result, slope = _limit_density(density, root_density, weights, decay)
    error = population - float(result @ weights)
    if error < _TOLERANCE:
        return result, False

    # Each limited shell is a minimum of linear functions of the coefficient, so
    # the population error is convex, decreasing and piecewise linear. Newton
    # steps from zero rise to the root without overshooting it, and land on it
    # exactly from its own linear piece.
    coefficient = 0.0
    for _ in range(_NEWTON_ITERATIONS):
        gradient = float(slope @ weights)
        if gradient <= 0:
            msg = "the radial-density population constraint has no solution"
            raise ValueError(msg)
        coefficient += error / gradient
        result, slope = _limit_density(
            density + coefficient * root_density, root_density, weights, decay
        )
        error = population - float(result @ weights)
        if abs(error) < _TOLERANCE:
            return result, True
    msg = "radial-density population constraint did not converge"
    raise ValueError(msg)


def monotonic_decay(density: FloatArray, weights: FloatArray) -> FloatArray:
    """Enforce radial monotonicity while preserving the integrated population."""
    result, _ = _restore_population(density, weights, np.ones_like(density))
    return result


def tail_decay(
    density: FloatArray,
    weights: FloatArray,
    first_exponent: FloatArray,
    second_exponent: FloatArray,
) -> FloatArray:
    """Constrain diffuse and contracted buried tails using fixed DDEC6 exponents.

    As in the reference algorithm, the lower-tail correction is reached only
    when enforcing the upper-tail bound required population restoration.
    """
    result, restored = _restore_population(density, weights, first_exponent)
    if not restored:
        return result
    for shell in range(1, len(result)):
        result[shell] = max(result[shell], result[shell - 1] * second_exponent[shell])
    population = float(result @ weights)
    if abs(population) > _TOLERANCE:
        result *= float(density @ weights) / population
    return result
