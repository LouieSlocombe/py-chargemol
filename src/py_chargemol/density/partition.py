"""The seven charge-partitioning steps of the DDEC6 method in NumPy.

This follows Chargemol 3.5's reference-ion conditioning, weighted spherical
averaging, buried-tail constraints, and nonnegative valence populations.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from py_chargemol.density.grid import (
    NSHELLS,
    SCALE,
    occupied_counts,
    reciprocal,
    shell_counts,
)
from py_chargemol.density.grid import (
    RadialSample as AtomicSample,
)
from py_chargemol.density.grid import (
    sum_radial_weights as pseudodensity,
)
from py_chargemol.density.references import reference_densities
from py_chargemol.density.reshaping import monotonic_decay, tail_decay

__all__ = [
    "AtomicSample",
    "FloatArray",
    "IntArray",
    "PartitionResult",
    "partition_ddec6",
    "pseudodensity",
]

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_ZERO = 1e-10
_CHARGE_TOLERANCE = 1e-5
_FIRST_ITERATION = 4
_PARTITION_STEPS = 7
_TAIL_CONSTRAINT_THRESHOLD = 0.01


@dataclass(frozen=True)
class PartitionResult:
    """DDEC6 populations and the weighting functions needed for other analyses."""

    charges: FloatArray
    weights: FloatArray
    pseudodensity: FloatArray
    spherical_average: FloatArray
    weighted_spherical_average: FloatArray
    reference_charges: FloatArray
    hirshfeld_charges: FloatArray
    normalization: float
    iterations: int


def _shell_statistics(  # noqa: PLR0913, PLR0917 -- Radial sums of one partition step.
    weights: FloatArray,
    samples: Sequence[AtomicSample],
    density: FloatArray,
    total: FloatArray,
    counts: FloatArray,
    threshold: float = _ZERO,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Shell counts and sums of each atom's assigned density.

    ``counts`` holds every atom's ``shell_counts``; points where ``total``
    does not exceed ``threshold`` are left out of the returned counts.
    """
    shape = (len(samples), NSHELLS)
    points, assigned, weighted_points, weighted_density = (
        np.zeros(shape) for _ in range(4)
    )
    inverse = reciprocal(total, threshold)
    # One atom at a time, so no array per sample point outlives its atom.
    for atom, sample in enumerate(samples):
        # Points below the threshold contribute zeros rather than being removed.
        local_inverse = inverse[sample.indices]
        # Widened once: the stored uint8 shells would be cast at every use.
        shells = sample.shells.astype(np.intp)
        fractions = weights[atom][shells] * local_inverse
        local = fractions * density[sample.indices]
        points[atom] = occupied_counts(counts[atom], shells, local_inverse)
        assigned[atom] = np.bincount(shells, weights=local, minlength=NSHELLS)
        weighted_points[atom] = np.bincount(
            shells, weights=fractions, minlength=NSHELLS
        )
        weighted_density[atom] = np.bincount(
            shells, weights=fractions * local, minlength=NSHELLS
        )
    return points, assigned, weighted_points, weighted_density


def _normalize_population(
    assigned: FloatArray,
    pixel_volume: float,
    core_population: FloatArray,
    occupancy_correction: FloatArray,
    valence_electrons: float,
) -> tuple[FloatArray, float]:
    populations = (
        assigned.sum(axis=1) * pixel_volume + occupancy_correction - core_population
    )
    total = float(populations.sum())
    if total <= _ZERO:
        msg = "the density grid has no assignable valence electrons"
        raise ValueError(msg)
    normalization = valence_electrons / total
    return populations * normalization, normalization


def _reference_steps(  # noqa: PLR0913, PLR0917 -- Physical partitioning inputs.
    atomic_numbers: IntArray,
    samples: Sequence[AtomicSample],
    density: FloatArray,
    size: int,
    pixel_volume: float,
    core_electrons: FloatArray,
    core_population: FloatArray,
    occupancy_correction: FloatArray,
    valence_electrons: float,
    counts: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    reference_charges = np.zeros(len(samples))
    weights = reference_densities(atomic_numbers, reference_charges)
    hirshfeld = np.zeros(len(samples))
    charges = np.zeros(len(samples))
    for step in range(2):
        total = pseudodensity(weights, samples, size)
        localized_total = pseudodensity(weights**4, samples, size)
        _, assigned, _, _ = _shell_statistics(weights, samples, density, total, counts)
        populations, normalization = _normalize_population(
            assigned,
            pixel_volume,
            core_population,
            occupancy_correction,
            valence_electrons,
        )
        localized_population = np.zeros(len(samples))
        for atom, sample in enumerate(samples):
            local_density = density[sample.indices]
            localized = np.divide(
                weights[atom, sample.shells] ** 4 * local_density,
                localized_total[sample.indices],
                out=np.zeros(len(local_density)),
                where=total[sample.indices] > _ZERO,
            )
            localized_population[atom] = pixel_volume * np.sum(localized)
        localized_population += occupancy_correction - core_electrons
        localized_population *= normalization
        charges = atomic_numbers - core_electrons - populations
        localized_charges = atomic_numbers - core_electrons - localized_population
        if step == 0:
            hirshfeld = charges.copy()
        reference_charges = (2 * localized_charges + charges) / 3
        weights = reference_densities(atomic_numbers, reference_charges)
    return weights, reference_charges, hirshfeld, charges


def _condition_references(  # noqa: PLR0913, PLR0917 -- Physical partitioning inputs.
    weights: FloatArray,
    atomic_numbers: IntArray,
    samples: Sequence[AtomicSample],
    density: FloatArray,
    size: int,
    pixel_volume: float,
    core_electrons: FloatArray,
    core_population: FloatArray,
    occupancy_correction: FloatArray,
    valence_electrons: float,
    reference_charges: FloatArray,
    counts: FloatArray,
) -> FloatArray:
    total = pseudodensity(weights, samples, size)
    points, assigned, _, _ = _shell_statistics(
        weights, samples, density, total, counts, _ZERO**2
    )
    sigma = np.divide(assigned, points, out=np.zeros_like(assigned), where=points > 0)
    population = assigned.sum(axis=1) * pixel_volume
    _, normalization = _normalize_population(
        assigned, pixel_volume, core_population, occupancy_correction, valence_electrons
    )
    target = (
        (atomic_numbers - core_electrons - reference_charges) / normalization
        + core_population
        - occupancy_correction
    )
    for atom in range(len(samples)):
        if population[atom] > _CHARGE_TOLERANCE:
            sigma[atom] = np.maximum(sigma[atom] * target[atom] / population[atom], 0)
        sigma[atom] = monotonic_decay(sigma[atom], points[atom] * pixel_volume)
    return sigma


def _tail_exponents(
    conditioned: FloatArray, samples: Sequence[AtomicSample], size: int
) -> tuple[FloatArray, FloatArray]:
    total = pseudodensity(conditioned, samples, size)
    tau = np.ones_like(conditioned)
    for atom, sample in enumerate(samples):
        local_total = total[sample.indices]
        valid = local_total > _ZERO**2
        roots = np.sqrt(local_total, out=np.zeros(len(local_total)), where=valid)
        ratios = np.divide(
            conditioned[atom, sample.shells],
            roots,
            out=np.zeros(len(roots)),
            where=valid,
        )
        denominator = np.bincount(sample.shells, weights=roots, minlength=NSHELLS)
        numerator = np.bincount(sample.shells, weights=ratios, minlength=NSHELLS)
        np.divide(numerator, denominator, out=tau[atom], where=denominator > 0)
    first = np.exp(-1.75 * (1 - tau**2) / SCALE)
    constraint = (1 - tau**2) / 2.5
    second = np.zeros_like(constraint)
    active = constraint > _TAIL_CONSTRAINT_THRESHOLD
    second[active] = np.exp(-1 / (SCALE * constraint[active]))
    return first, second


def partition_ddec6(  # noqa: PLR0913 -- Explicit physical inputs to the partitioner.
    *,
    atomic_numbers: IntArray,
    density: FloatArray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
    core_electrons: FloatArray,
    core_population: FloatArray,
    occupancy_correction: FloatArray,
    net_charge: float = 0.0,
) -> PartitionResult:
    """Partition a prepared total-density grid into DDEC6 atomic populations.

    The grid and core partition have already been validated and corrected by
    the density workflow. All radial arrays use ``(natoms, 100)`` ordering.
    Core populations and occupancy corrections are those used by Chargemol's
    valence normalization; neither is inferred from an atom's atomic number.
    """
    valence_electrons = float(np.sum(atomic_numbers - core_electrons) - net_charge)
    if valence_electrons <= 0:
        msg = "DDEC6 partitioning requires a positive number of valence electrons"
        raise ValueError(msg)
    counts = shell_counts(samples)
    weights, reference_charges, hirshfeld, charges = _reference_steps(
        atomic_numbers,
        samples,
        density,
        density.size,
        pixel_volume,
        core_electrons,
        core_population,
        occupancy_correction,
        valence_electrons,
        counts,
    )
    weights = _condition_references(
        weights,
        atomic_numbers,
        samples,
        density,
        density.size,
        pixel_volume,
        core_electrons,
        core_population,
        occupancy_correction,
        valence_electrons,
        reference_charges,
        counts,
    )
    first, second = _tail_exponents(weights, samples, density.size)
    completed_steps = 3
    update_kappa = False
    max_change = 1.0
    renormalization = np.ones(len(samples))
    for iteration in range(_FIRST_ITERATION, 501):
        total = pseudodensity(weights, samples, density.size)
        points, assigned, weighted_points, weighted_density = _shell_statistics(
            weights, samples, density, total, counts
        )
        spherical = np.divide(
            assigned, points, out=np.zeros_like(assigned), where=points > 0
        )
        weighted = np.divide(
            assigned - weighted_density + 0.2 * weighted_points * spherical,
            points - 0.8 * weighted_points,
            out=np.zeros_like(assigned),
            where=points > 0,
        )
        slope = pixel_volume * (assigned - weighted_density).sum(axis=1)
        populations, normalization = _normalize_population(
            assigned,
            pixel_volume,
            core_population,
            occupancy_correction,
            valence_electrons,
        )
        old_change = max_change
        previous_charges = charges
        charges = atomic_numbers - core_electrons - populations
        max_change = float(np.max(np.abs(charges - previous_charges)))
        if iteration > _FIRST_ITERATION:
            if np.any(populations < -_CHARGE_TOLERANCE):
                update_kappa = True
            if update_kappa and max(max_change, old_change) < _CHARGE_TOLERANCE:
                update_kappa = False
                renormalization.fill(1)
        if update_kappa:
            weights /= renormalization[:, None]
            active = slope > _CHARGE_TOLERANCE / 100
            renormalization[active] = np.maximum(
                1,
                renormalization[active] * np.exp(-populations[active] / slope[active]),
            )
            renormalization[~active] = 1
            weights *= renormalization[:, None]
        else:
            completed_steps += 1
            if completed_steps == _PARTITION_STEPS:
                if np.any(populations < -_CHARGE_TOLERANCE):
                    msg = "DDEC6 could not assign nonnegative valence populations"
                    raise ValueError(msg)
                return PartitionResult(
                    charges=charges,
                    weights=weights,
                    pseudodensity=total,
                    spherical_average=spherical,
                    weighted_spherical_average=weighted,
                    reference_charges=reference_charges,
                    hirshfeld_charges=hirshfeld,
                    normalization=normalization,
                    iterations=iteration,
                )
            weights = np.array(
                [
                    tail_decay(row, point_row * pixel_volume, first_row, second_row)
                    for row, point_row, first_row, second_row in zip(
                        weighted, points, first, second, strict=True
                    )
                ]
            )
    msg = "DDEC6 atomic charges did not converge within 500 iterations"
    raise ValueError(msg)
