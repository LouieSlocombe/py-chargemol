"""Population-preserving radial constraints for DDEC6 weighting functions.

These apply the constraints in Chargemol's ``module_reshaping_functions``.
Bisection solves the scalar normalization constraint without requiring SciPy.
"""

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

_TOLERANCE = 1e-10


def _limit_density(
    density: FloatArray, weights: FloatArray, decay: FloatArray
) -> FloatArray:
    result = np.maximum(density, 0.0)
    for shell in range(1, len(result)):
        if weights[shell] > 0 and result[shell - 1] > 0:
            result[shell] = min(result[shell], result[shell - 1] * decay[shell])
    return result


def _restore_population(
    density: FloatArray, weights: FloatArray, decay: FloatArray
) -> tuple[FloatArray, bool]:
    population = float(density @ weights)
    root_density = np.sqrt(density)
    weighted_points = float(root_density @ weights)
    if min(population, weighted_points) < _TOLERANCE:
        return density.copy(), False
    result = _limit_density(density, weights, decay)
    error = population - float(result @ weights)
    if error < _TOLERANCE:
        return result, False

    lower = 0.0
    upper = error / weighted_points
    for _ in range(50):
        result = _limit_density(density + upper * root_density, weights, decay)
        error = population - float(result @ weights)
        if error <= 0:
            break
        lower = upper
        upper = 2 * lower + error / weighted_points
    else:
        msg = "could not bracket the radial-density population constraint"
        raise ValueError(msg)

    for _ in range(80):
        coefficient = (lower + upper) / 2
        result = _limit_density(density + coefficient * root_density, weights, decay)
        error = population - float(result @ weights)
        if abs(error) < _TOLERANCE:
            return result, True
        if error > 0:
            lower = coefficient
        else:
            upper = coefficient
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
