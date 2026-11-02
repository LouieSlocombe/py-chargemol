"""Collinear and noncollinear DDEC atomic spin partitioning.

This implements the spin-function and spin-moment iterations in the September
2017 Chargemol source by Thomas A. Manz and Nidia Gabaldon Limas. Both scalar
and vector spin densities use the original tabulated angular functions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from py_chargemol.ddec6 import FloatArray, IntArray
from py_chargemol.density.bonds import AtomicSampleLike

_ZERO = 1e-10
_XI_THRESHOLD = 0.001
_XI_ZERO = 1e-6
_LOOKUP_POINTS = 10000
_MIN_ITERATIONS = 7


@dataclass(frozen=True, slots=True, eq=False)
class SpinResult:
    """Atomic spins, radial spin densities, and convergence diagnostics."""

    populations: FloatArray
    spherical_average: FloatArray
    reconstructed_density: FloatArray
    iterations: int
    converged: bool
    max_change: float


@dataclass(frozen=True, slots=True, eq=False)
class _LocalSpin:
    points: IntArray
    shells: IntArray
    fractions: FloatArray
    density: FloatArray
    proportional_spin: FloatArray
    proportional_theta: FloatArray
    shell_counts: IntArray


def _calculate_xi(tau: FloatArray) -> FloatArray:
    result = np.empty_like(tau)
    low = tau < _XI_THRESHOLD
    high = tau > 1.0 - _XI_THRESHOLD
    middle = ~(low | high)
    result[low] = 4.0 * np.pi * np.maximum(tau[low], 0.0) / 3.0
    deviation = np.minimum(tau[high], 1.0) - 1.0
    result[high] = 2.0 * np.pi + deviation * 49.956 + deviation**2 * 8370.6
    values = tau[middle]
    result[middle] = np.pi * (
        (1.0 / values**2 - 1.0) * np.log((1.0 - values) / (1.0 + values)) + 2.0 / values
    )
    return result


def _xi_derivative(tau: FloatArray) -> FloatArray:
    result = np.empty_like(tau)
    low = tau < _XI_THRESHOLD
    high = tau > 1.0 - _XI_THRESHOLD
    middle = ~(low | high)
    result[low] = 4.0 * np.pi / 3.0
    result[high] = 50.0
    values = tau[middle]
    result[middle] = (
        2.0
        * np.pi
        * (np.log((1.0 + values) / (1.0 - values)) - 2.0 * values)
        / values**3
    )
    return result


def _calculate_inverse_xi(values: FloatArray) -> FloatArray:
    result = np.empty_like(values)
    low = values < _XI_ZERO
    high = values > 2.0 * np.pi - _XI_ZERO
    middle = ~(low | high)
    result[low] = 3.0 * np.maximum(values[low], 0.0) / (4.0 * np.pi)
    result[high] = 1.0
    selected = values[middle]
    estimate = (selected - 0.036835 * selected**2) / (
        4.178319 - 0.136129 * selected + 0.038148 * selected**2
    )
    derivative = _xi_derivative(estimate)
    # Chargemol keeps this initial derivative fixed for all five updates.
    for _ in range(5):
        estimate += (selected - _calculate_xi(estimate)) / derivative
    result[middle] = estimate
    return result


@lru_cache(maxsize=1)
def _lookup_tables() -> tuple[FloatArray, FloatArray, FloatArray]:
    fractions = np.arange(_LOOKUP_POINTS + 1, dtype=np.float64) / _LOOKUP_POINTS
    xi = _calculate_xi(fractions)
    inverse = _calculate_inverse_xi(fractions * (2.0 * np.pi))
    return fractions, xi, inverse


def _theta(density: FloatArray, spin: FloatArray) -> FloatArray:
    magnitude = np.linalg.norm(spin, axis=1)
    active = (magnitude > _ZERO) & (density > _ZERO)
    result = np.zeros_like(spin)
    fractions, lookup, _ = _lookup_tables()
    xi = np.interp(magnitude[active] / density[active], fractions, lookup)
    result[active] = spin[active] * (xi / (2.0 * magnitude[active]))[:, np.newaxis]
    return result


def _local_spin_data(
    density: FloatArray,
    spin: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    samples: Sequence[AtomicSampleLike],
) -> list[_LocalSpin]:
    local: list[_LocalSpin] = []
    for atom, sample in enumerate(samples):
        active = (pseudodensity[sample.indices] > _ZERO) & (
            density[sample.indices] > _ZERO
        )
        points = sample.indices[active]
        shells = sample.shells[active]
        fraction = weights[atom, shells] / pseudodensity[points]
        assigned_density = fraction * density[points]
        proportional = fraction[:, np.newaxis] * spin[points]
        local.append(
            _LocalSpin(
                points,
                shells,
                fraction,
                assigned_density,
                proportional,
                _theta(assigned_density, proportional),
                np.bincount(shells, minlength=100),
            )
        )
    return local


def partition_spin(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    density: FloatArray,
    spin_density: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    spherical_average: FloatArray,
    samples: Sequence[AtomicSampleLike],
    voxel_volume: float,
    occupancy_correction: FloatArray | None = None,
    max_iterations: int = 500,
    tolerance: float = 5e-5,
) -> SpinResult:
    """Partition a scalar or Cartesian-vector spin density into atomic spins.

    Densities are flattened in the same order as each sample's indices. The
    supplied radial weights and spherical density are from the converged DDEC
    charge partition, before bond-density correction. ``occupancy_correction``
    adds analytically integrated on-site spin populations, when present.

    Scalar input has shape ``(N,)``; vector input has shape ``(N, 3)``. The
    corresponding population shapes are ``(n,)`` and ``(n, 3)``, and radial
    spin shapes are ``(n, 100)`` and ``(n, 100, 3)``. Inspect ``converged`` when
    overriding the original iteration limit or convergence tolerance.
    """
    spin = np.asarray(spin_density, dtype=np.float64)
    scalar = spin.ndim == 1
    if scalar:
        spin = spin[:, np.newaxis]
    if spin.shape not in ((len(density), 1), (len(density), 3)):
        msg = "spin_density must have shape (N,) or (N, 3)"
        raise ValueError(msg)
    if not np.all(np.isfinite(spin)):
        msg = "spin_density must contain finite values"
        raise ValueError(msg)
    if max_iterations <= 0:
        msg = "max_iterations must be a positive integer"
        raise ValueError(msg)
    if not np.isfinite(tolerance) or tolerance <= 0:
        msg = "tolerance must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        msg = "voxel_volume must be finite and positive"
        raise ValueError(msg)
    natoms = len(samples)
    if weights.shape != (natoms, 100) or spherical_average.shape != weights.shape:
        msg = "weights and spherical_average must have shape (n, 100)"
        raise ValueError(msg)
    components = spin.shape[1]
    correction = np.zeros((natoms, components))
    if occupancy_correction is not None:
        supplied = np.asarray(occupancy_correction, dtype=np.float64)
        if scalar and supplied.shape == (natoms,):
            supplied = supplied[:, np.newaxis]
        if supplied.shape != correction.shape or not np.all(np.isfinite(supplied)):
            msg = "occupancy_correction must match the atomic spin population shape"
            raise ValueError(msg)
        correction = supplied
    local_data = _local_spin_data(density, spin, weights, pseudodensity, samples)
    radial_spin = np.zeros((natoms, 100, components))
    populations = np.zeros((natoms, components))
    ypsilon = np.zeros_like(spin)
    omega = np.zeros_like(spin)
    trial = np.zeros_like(spin)
    fractions, _, inverse_lookup = _lookup_tables()
    converged = False
    max_change = 0.0
    for iteration in range(1, max_iterations + 1):
        sums = np.zeros_like(radial_spin)
        new_ypsilon = np.zeros_like(spin)
        new_trial = np.zeros_like(spin)
        for atom, local in enumerate(local_data):
            if iteration == 1:
                assigned_spin = local.proportional_spin
                lagrange = local.proportional_theta
            else:
                theta = _theta(
                    spherical_average[atom, local.shells],
                    radial_spin[atom, local.shells],
                )
                lagrange = (
                    ypsilon[local.points]
                    - omega[local.points]
                    + np.pi
                    * (spin[local.points] - trial[local.points])
                    / density[local.points, None]
                    + 0.5 * (theta + local.proportional_theta)
                )
                magnitude = np.maximum(np.linalg.norm(lagrange, axis=1), _ZERO**2)
                inverse = np.interp(magnitude / np.pi, fractions, inverse_lookup)
                assigned_magnitude = local.density * inverse
                if not scalar:
                    assigned_magnitude = np.maximum(assigned_magnitude, _ZERO**2)
                assigned_spin = lagrange * (assigned_magnitude / magnitude)[:, None]
            np.add.at(sums[atom], local.shells, assigned_spin)
            np.add.at(new_ypsilon, local.points, lagrange * local.fractions[:, None])
            np.add.at(new_trial, local.points, assigned_spin)
            np.divide(
                sums[atom],
                local.shell_counts[:, None],
                out=radial_spin[atom],
                where=local.shell_counts[:, None] > 0,
            )
        ypsilon, trial = new_ypsilon, new_trial
        new_populations = sums.sum(axis=1) * voxel_volume + correction
        max_change = float(np.max(np.abs(new_populations - populations), initial=0))
        populations = new_populations
        zero_scalar_spin = (
            scalar
            and max(float(np.max(np.abs(populations), initial=0)), max_change)
            < tolerance
        )
        if (max_change < tolerance and iteration >= _MIN_ITERATIONS) or (
            zero_scalar_spin and iteration > 1
        ):
            converged = True
            break
        omega.fill(0)
        for atom, local in enumerate(local_data):
            theta = _theta(
                spherical_average[atom, local.shells], radial_spin[atom, local.shells]
            )
            np.add.at(
                omega,
                local.points,
                0.5 * (theta + local.proportional_theta) * local.fractions[:, None],
            )
    return SpinResult(
        populations[:, 0] if scalar else populations,
        radial_spin[..., 0] if scalar else radial_spin,
        trial[:, 0] if scalar else trial,
        iteration,
        converged,
        max_change,
    )
