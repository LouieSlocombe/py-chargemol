"""Core-electron partitioning and grid integration corrections.

The equations follow Chargemol's September 2017 core iterator by Thomas A.
Manz and Nidia Gabaldon Limas, evaluated on NumPy arrays.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from py_chargemol import _jit
from py_chargemol.density._kernels import (
    add_corrected_core,
    assigned_core_values,
    core_shell_sums,
)
from py_chargemol.density.convergence import DEFAULT_CONVERGENCE, ConvergenceOptions
from py_chargemol.density.errors import CoreConvergenceError
from py_chargemol.density.grid import (
    NSHELLS,
    SCALE,
    ZERO,
    AtomicSample,
    occupied_counts,
    reciprocal,
    shell_counts,
    sum_radial_weights,
)

__all__ = [
    "CoreConvergenceError",
    "CoreResult",
    "dominant_occupancy_correction",
    "partition_core",
    "sum_radial_weights",
]

_MINIMUM_CORE_ITERATIONS = 6
_MAX_EXTRAPOLATION = 1000.0
_MAX_REFINEMENT_FRACTION = 0.01


@dataclass(frozen=True)
class CoreResult:
    """Corrected core density and its atom-centred partition."""

    density: np.ndarray
    populations: np.ndarray
    weights: np.ndarray
    pseudodensity: np.ndarray


def partition_core(  # noqa: PLR0913 -- Core inputs and optional convergence settings.
    density: np.ndarray,
    core_electrons: np.ndarray,
    neutral: np.ndarray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
    *,
    convergence: ConvergenceOptions = DEFAULT_CONVERGENCE,
) -> CoreResult:
    """Fit core weights and correct the density to the specified occupations."""
    tolerance = convergence.charge_tolerance
    density = np.minimum(np.asarray(density).ravel(), 30.0 / pixel_volume)
    weights = neutral.copy()
    weights[core_electrons == 0] = 0
    populations = np.zeros(len(samples))
    if not np.any(core_electrons):
        return CoreResult(np.zeros_like(density), populations, weights, density * 0)
    sampled = shell_counts(samples)
    pseudo = sum_radial_weights(weights, samples, len(density))
    totals, counts, averages = _local_averages(
        density, weights, pseudo, samples, sampled
    )
    for iteration in range(convergence.weights_max_iterations):
        updated = totals * pixel_volume
        change = np.max(np.abs(updated - populations))
        populations = updated
        if iteration + 1 >= _MINIMUM_CORE_ITERATIONS and change < tolerance:
            break
        weights = _decaying_weights(averages, counts)
        pseudo = sum_radial_weights(weights, samples, len(density))
        totals, counts, averages = _local_averages(
            density, weights, pseudo, samples, sampled
        )
    else:
        change = np.max(np.abs(totals * pixel_volume - populations))
        warnings.warn(
            f"core weights did not converge within {convergence.weights_max_iterations}"
            f" iterations (last population change={change:.8g}, "
            f"tolerance={tolerance:.8g})",
            RuntimeWarning,
            stacklevel=2,
        )
    populations = totals * pixel_volume

    previous_weights = weights
    previous_residual = core_electrons - populations
    for iteration in range(convergence.core_max_iterations):
        correction = core_electrons - populations
        adjusted = _corrected_density(
            density, weights, pseudo, samples, correction, pixel_volume=pixel_volume
        )
        totals, counts, averages = _local_averages(
            adjusted, weights, pseudo, samples, sampled
        )
        populations = totals * pixel_volume
        residual = core_electrons - populations
        if np.max(np.abs(residual)) < tolerance:
            return CoreResult(adjusted, populations, weights, pseudo)
        if iteration + 1 >= _MINIMUM_CORE_ITERATIONS:
            step = _tail_step(previous_residual, correction, residual, tolerance)
            if step:
                # Extrapolate both fields that produced the measured occupations.
                trial = _check_trial(
                    _extrapolate(density, adjusted, step),
                    _decaying_weights(
                        _extrapolate(previous_weights, weights, step), counts
                    ),
                    samples,
                    pixel_volume,
                    core_electrons,
                    tolerance=tolerance,
                    ordinary_density=(
                        adjusted
                        if iteration + 1 >= DEFAULT_CONVERGENCE.core_max_iterations
                        else None
                    ),
                )
                if trial is not None:
                    return trial
        density = adjusted
        previous_weights, previous_residual = weights, correction
        weights = _decaying_weights(averages, counts)
        pseudo = sum_radial_weights(weights, samples, len(density))
    msg = "core density could not be corrected to the requested electron populations"
    raise CoreConvergenceError(
        msg, core_electrons - populations, convergence.core_max_iterations, tolerance
    )


def _corrected_density(  # noqa: PLR0913 -- One correction of the sampled core grid.
    density: np.ndarray,
    weights: np.ndarray,
    pseudo: np.ndarray,
    samples: Sequence[AtomicSample],
    correction: np.ndarray,
    *,
    pixel_volume: float,
) -> np.ndarray:
    """Adjust each atom's assigned core density and sum it back onto the grid."""
    adjusted = np.zeros_like(density)
    inverse = reciprocal(pseudo)
    for atom, (sample, delta) in enumerate(zip(samples, correction, strict=True)):
        values = _assigned(density, weights[atom], inverse, sample)
        sum_cubes = np.sum(values**3)
        largest = np.max(values, initial=0.0)
        factor = 0.0
        if sum_cubes > ZERO**3 and largest > ZERO:
            factor = min(delta / (sum_cubes * pixel_volume), 0.25 / largest**2)
        if _jit.ENABLED:
            add_corrected_core(values, sample.indices, factor, ZERO**2, adjusted)
        else:
            significant = np.where(values > ZERO**2, values, 0.0)
            corrected = significant / np.sqrt(1.0 - 2.0 * factor * significant**2)
            np.add.at(adjusted, sample.indices, corrected)
    return adjusted


def _tail_step(
    previous: np.ndarray, current: np.ndarray, residual: np.ndarray, tolerance: float
) -> float:
    """Estimate how many steps remain along a linearly contracting residual.

    For a single geometric mode, residual + step * change vanishes. Fit that
    scalar by least squares and require the preceding pair of residuals to
    predict convergence with the same step before evaluating another grid.
    """
    change = residual - current
    squared_change = float(np.dot(change, change))
    projection = -float(np.dot(residual, change))
    if not projection > squared_change > 0:
        return 0.0
    step = min(projection, _MAX_EXTRAPOLATION * squared_change) / squared_change
    predicted = max(
        np.max(np.abs(residual + step * change)),
        np.max(np.abs(current + step * (current - previous))),
    )
    return step if predicted < tolerance else 0.0


def _extrapolate(previous: np.ndarray, current: np.ndarray, step: float) -> np.ndarray:
    """Continue the latest update using one work array for a full grid."""
    trial: np.ndarray = current - previous
    trial *= step
    trial += current
    return trial


def _check_trial(  # noqa: PLR0913 -- Reintegrate trial fields at the requested tolerance.
    density: np.ndarray,
    weights: np.ndarray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
    core_electrons: np.ndarray,
    *,
    tolerance: float,
    ordinary_density: np.ndarray | None,
) -> CoreResult | None:
    """Independently verify the extrapolated fields before accepting a result."""
    pseudo = sum_radial_weights(weights, samples, len(density))
    np.maximum(density, 0, out=density)
    density[pseudo <= ZERO] = 0
    inverse = reciprocal(pseudo)

    def integrate(field: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.sum(_assigned(field, radial, inverse, sample)) * pixel_volume
                for radial, sample in zip(weights, samples, strict=True)
            ]
        )

    populations = integrate(density)
    residual = core_electrons - populations
    if ordinary_density is not None and np.max(np.abs(residual)) >= tolerance:
        # Density and radial weights need not approach the limit at exactly
        # the same rate. Once the ordinary 40-step budget has been spent,
        # allow a 1% refinement along the density update at fixed trial weights,
        # where populations are linear. Then reintegrate the nonnegative grid.
        delta = density - ordinary_density
        response = integrate(delta)
        norm_squared = float(np.dot(response, response))
        projection = float(np.dot(residual, response))
        if (
            norm_squared == 0
            or abs(projection) > _MAX_REFINEMENT_FRACTION * norm_squared
        ):
            return None
        delta *= projection / norm_squared
        density += delta
        np.maximum(density, 0, out=density)
        density[pseudo <= ZERO] = 0
        populations = integrate(density)
    if np.max(np.abs(core_electrons - populations)) < tolerance:
        return CoreResult(density, populations, weights, pseudo)
    return None


def dominant_occupancy_correction(
    difference: np.ndarray,
    neutral: np.ndarray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
) -> np.ndarray:
    """Assign PAW valence-grid integration corrections to dominant atoms."""
    difference = difference.ravel()
    maximum = np.zeros(len(difference))
    owners = np.full(len(difference), -1, dtype=np.int64)
    for atom, (radial, sample) in enumerate(zip(neutral, samples, strict=True)):
        # Reduce images of one atom before selecting it, so a remote image
        # cannot overwrite that atom's nearer image through repeated indexing.
        weight = np.zeros(len(difference))
        np.maximum.at(weight, sample.indices, radial[sample.shells])
        higher = weight > maximum
        maximum[higher] = weight[higher]
        owners[higher] = atom
    present = owners >= 0
    return pixel_volume * np.bincount(
        owners[present], weights=difference[present], minlength=len(samples)
    )


def _assigned(
    density: np.ndarray, radial: np.ndarray, inverse: np.ndarray, sample: AtomicSample
) -> np.ndarray:
    """One atom's share of the density at its points, given the reciprocal."""
    if _jit.ENABLED:
        return assigned_core_values(
            radial, density, inverse, sample.indices, sample.shells
        )
    return np.asarray(
        radial[sample.shells] * density[sample.indices] * inverse[sample.indices]
    )


def _local_averages(
    density: np.ndarray,
    weights: np.ndarray,
    pseudo: np.ndarray,
    samples: Sequence[AtomicSample],
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each atom's assigned total, and its shell counts and averages.

    ``sampled`` holds every atom's ``shell_counts``. Works one atom at a
    time, so no array per sample point outlives its atom.
    """
    totals = np.zeros(len(samples))
    counts = np.zeros_like(weights)
    averages = np.zeros_like(weights)
    inverse = reciprocal(pseudo)
    for atom, sample in enumerate(samples):
        if _jit.ENABLED:
            values, counts[atom], sums = core_shell_sums(
                weights[atom],
                density,
                inverse,
                sample.indices,
                sample.shells,
                sampled[atom],
            )
        else:
            local_inverse = inverse[sample.indices]
            # Widened once: the stored uint8 shells would be cast at every use.
            shells = sample.shells.astype(np.intp)
            values = weights[atom][shells] * density[sample.indices] * local_inverse
            counts[atom] = occupied_counts(sampled[atom], shells, local_inverse)
            sums = np.bincount(shells, weights=values, minlength=NSHELLS)
        totals[atom] = np.sum(values)
        averages[atom] = np.divide(
            sums, counts[atom], out=np.zeros(NSHELLS), where=counts[atom] > 0
        )
    return totals, counts, averages


def _decaying_weights(averages: np.ndarray, counts: np.ndarray) -> np.ndarray:
    weights = averages.copy()
    decay = np.exp(-2.0 / SCALE)
    for shell in range(1, NSHELLS):
        restrict = (counts[:, shell] > 0) & (weights[:, shell - 1] > 0)
        weights[restrict, shell] = np.minimum(
            weights[restrict, shell], weights[restrict, shell - 1] * decay
        )
    np.maximum(weights, 0.0, out=weights)
    return weights
