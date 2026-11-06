"""Serial shell reductions and periodic-image scatters for spin iterations."""

import numpy as np

from py_chargemol._jit import jit


@jit
def accumulate_spin(  # noqa: PLR0913, PLR0917 -- Array inputs for one atom.
    points: np.ndarray,
    shells: np.ndarray,
    fractions: np.ndarray,
    lagrange: np.ndarray,
    assigned: np.ndarray,
    sums: np.ndarray,
    ypsilon: np.ndarray,
    trial: np.ndarray,
) -> None:
    """Accumulate shell spins and both grid fields in sample order.

    ``sums`` starts at zero. Repeated grid indices are distinct periodic images;
    each component retains the point order of NumPy's unbuffered scatters.
    """
    for component in range(len(assigned)):
        for sample in range(len(points)):
            point = points[sample]
            value = assigned[component, sample]
            sums[shells[sample], component] += value
            ypsilon[component, point] += lagrange[component, sample] * fractions[sample]
            trial[component, point] += value


@jit
def subtract_spin_theta(  # noqa: PLR0913, PLR0917 -- Array inputs for one atom.
    points: np.ndarray,
    shells: np.ndarray,
    fractions: np.ndarray,
    proportional_theta: np.ndarray,
    shell_theta: np.ndarray,
    ypsilon: np.ndarray,
) -> None:
    """Subtract the shell-dependent spin constraint without point temporaries."""
    for component in range(len(proportional_theta)):
        for sample in range(len(points)):
            theta = shell_theta[component, shells[sample]]
            value = 0.5 * (theta + proportional_theta[component, sample])
            ypsilon[component, points[sample]] -= value * fractions[sample]
