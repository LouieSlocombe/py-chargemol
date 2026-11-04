"""Core-electron partitioning and grid integration corrections.

The equations follow Chargemol's September 2017 core iterator by Thomas A.
Manz and Nidia Gabaldon Limas, evaluated on NumPy arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from py_chargemol.density.grid import (
    NSHELLS,
    SCALE,
    ZERO,
    AtomicSample,
    reciprocal,
    sum_radial_weights,
)

__all__ = [
    "CoreResult",
    "dominant_occupancy_correction",
    "partition_core",
    "sum_radial_weights",
]

_MINIMUM_CORE_ITERATIONS = 6
_CHARGE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class CoreResult:
    """Corrected core density and its atom-centred partition."""

    density: np.ndarray
    populations: np.ndarray
    weights: np.ndarray
    pseudodensity: np.ndarray


def partition_core(
    density: np.ndarray,
    core_electrons: np.ndarray,
    neutral: np.ndarray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
) -> CoreResult:
    """Fit core weights and correct the density to the specified occupations."""
    density = np.minimum(np.asarray(density).ravel(), 30.0 / pixel_volume)
    weights = neutral.copy()
    weights[core_electrons == 0] = 0
    populations = np.zeros(len(samples))
    if not np.any(core_electrons):
        return CoreResult(np.zeros_like(density), populations, weights, density * 0)
    pseudo = sum_radial_weights(weights, samples, len(density))
    densities = [density[sample.indices] for sample in samples]
    local, counts, averages = _local_averages(densities, weights, pseudo, samples)
    for iteration in range(200):
        updated = np.array([np.sum(value) for value in local]) * pixel_volume
        change = np.max(np.abs(updated - populations))
        populations = updated
        if iteration + 1 >= _MINIMUM_CORE_ITERATIONS and change < _CHARGE_TOLERANCE:
            break
        weights = _decaying_weights(averages, counts)
        pseudo = sum_radial_weights(weights, samples, len(density))
        local, counts, averages = _local_averages(densities, weights, pseudo, samples)
    populations = np.array([np.sum(value) for value in local]) * pixel_volume

    for _ in range(40):
        correction = core_electrons - populations
        adjusted = np.zeros_like(density)
        for sample, values, delta in zip(samples, local, correction, strict=True):
            sum_cubes = np.sum(values**3)
            largest = np.max(values, initial=0.0)
            factor = 0.0
            if sum_cubes > ZERO**3 and largest > ZERO:
                factor = min(delta / (sum_cubes * pixel_volume), 0.25 / largest**2)
            significant = np.where(values > ZERO**2, values, 0.0)
            corrected = significant / np.sqrt(1.0 - 2.0 * factor * significant**2)
            np.add.at(adjusted, sample.indices, corrected)
        density = adjusted
        densities = [density[sample.indices] for sample in samples]
        local, counts, averages = _local_averages(densities, weights, pseudo, samples)
        populations = np.array([np.sum(value) for value in local]) * pixel_volume
        if np.max(np.abs(core_electrons - populations)) < _CHARGE_TOLERANCE:
            break
        weights = _decaying_weights(averages, counts)
        pseudo = sum_radial_weights(weights, samples, len(density))
        local, counts, averages = _local_averages(densities, weights, pseudo, samples)
    if np.max(np.abs(core_electrons - populations)) >= _CHARGE_TOLERANCE:
        msg = (
            "core density could not be corrected to the requested electron populations"
        )
        raise ValueError(msg)
    return CoreResult(density, populations, weights, pseudo)


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


def _local_averages(
    densities: Sequence[np.ndarray],
    weights: np.ndarray,
    pseudo: np.ndarray,
    samples: Sequence[AtomicSample],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    local = []
    counts = np.zeros_like(weights)
    averages = np.zeros_like(weights)
    inverse = reciprocal(pseudo)
    for atom, (sample, density) in enumerate(zip(samples, densities, strict=True)):
        local_inverse = inverse[sample.indices]
        # Widened once: the stored uint8 shells would be cast at every use.
        shells = sample.shells.astype(np.intp)
        values = weights[atom][shells] * density * local_inverse
        local.append(values)
        counts[atom] = np.bincount(shells, weights=local_inverse > 0, minlength=NSHELLS)
        sums = np.bincount(shells, weights=values, minlength=NSHELLS)
        averages[atom] = np.divide(
            sums, counts[atom], out=np.zeros(NSHELLS), where=counts[atom] > 0
        )
    return local, counts, averages


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
