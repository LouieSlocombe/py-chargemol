"""Noncollinear spin-gradient diagnostics from the 2017 Chargemol analysis."""

from dataclasses import dataclass

import numpy as np

from py_chargemol.density.grid import ZERO, DensityGrid


@dataclass(frozen=True, slots=True)
class SpinGradientDiagnostics:
    """Integrated squared gradients of spin magnitude, vector, and direction.

    Values use atomic units, with the same forward differences and integration
    domain as Chargemol's ``check_noncollinear_XC_functional``. The directional
    contribution is the nonnegative difference between total and magnitude.
    """

    magnitude: float
    total: float
    directional: float


def compute_spin_gradient_diagnostics(grid: DensityGrid) -> SpinGradientDiagnostics:
    """Measure magnitude and directional changes of a noncollinear spin grid.

    Each forward difference wraps in periodic directions. Terminal planes in
    nonperiodic directions and points with negligible local spin are excluded
    from the common integration domain. Scalar or absent spin raises
    ``ValueError`` because these diagnostics require Cartesian spin vectors.
    """
    spin = grid.spin_density
    if spin is None or spin.shape != (*grid.density.shape, 3):
        msg = "spin-gradient diagnostics require a three-component spin density"
        raise ValueError(msg)
    magnitude = np.linalg.norm(spin, axis=-1)
    active = magnitude > ZERO
    for axis, periodic in enumerate(grid.periodic):
        if not periodic:
            terminal = [slice(None)] * 3
            terminal[axis] = slice(-1, None)
            active[tuple(terminal)] = False
    inverse = np.linalg.inv(grid.steps.T)
    magnitude_weight = _gradient_integral(magnitude, inverse, active, grid.pixel_volume)
    total_weight = sum(
        _gradient_integral(spin[..., component], inverse, active, grid.pixel_volume)
        for component in range(3)
    )
    return SpinGradientDiagnostics(
        magnitude_weight,
        total_weight,
        max(total_weight - magnitude_weight, 0.0),
    )


def _gradient_integral(
    values: np.ndarray, inverse: np.ndarray, active: np.ndarray, volume: float
) -> float:
    differences = np.stack(
        [np.roll(values, -1, axis=axis) - values for axis in range(3)], axis=-1
    )
    gradients = differences[active] @ inverse
    return float(np.sum(gradients**2) * volume)
