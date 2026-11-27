"""Iterative DDEC3 charge partitioning from the 2017 Chargemol algorithm."""

from collections.abc import Sequence

import numpy as np

from py_chargemol import _jit
from py_chargemol.density._kernels import reference_shell_sums
from py_chargemol.density.convergence import DEFAULT_CONVERGENCE, ConvergenceOptions
from py_chargemol.density.errors import DDEC3ConvergenceError
from py_chargemol.density.grid import NSHELLS, SCALE, shell_counts
from py_chargemol.density.partition import (
    AtomicSample,
    FloatArray,
    IntArray,
    PartitionResult,
    _normalize_population,
    _shell_statistics,
    pseudodensity,
)
from py_chargemol.density.references import reference_densities

_ZERO = 1e-10
_REFERENCE_WEIGHT = 3 / 14
_FIRST_RESHAPE_ITERATION = 4
_MINIMUM_ITERATIONS = 10
_MAX_ITERATIONS = 2000


def _reference_averages(  # noqa: PLR0913, PLR0917 -- Physical radial averages.
    references: FloatArray,
    conditioned: FloatArray,
    samples: Sequence[AtomicSample],
    density: FloatArray,
    total: FloatArray,
    points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    size = len(total)
    reference_total = pseudodensity(references, samples, size)
    conditioned_total = pseudodensity(conditioned, samples, size)
    average_ratio = np.ones_like(references)
    tau = np.ones_like(references)
    # One atom at a time, so no array per sample point outlives its atom.
    for atom, sample in enumerate(samples):
        if _jit.ENABLED:
            ratio_sum, numerator, denominator = reference_shell_sums(
                conditioned[atom],
                density,
                total,
                reference_total,
                conditioned_total,
                sample.indices,
                sample.shells,
                _ZERO,
            )
            np.divide(
                ratio_sum, points[atom], out=average_ratio[atom], where=points[atom] > 0
            )
            np.divide(numerator, denominator, out=tau[atom], where=points[atom] > 0)
            continue
        # Points outside the partition contribute zeros rather than being removed.
        valid = total[sample.indices] > _ZERO
        shells = sample.shells
        local_reference = reference_total[sample.indices]
        ratio = np.divide(
            density[sample.indices],
            local_reference,
            out=np.ones(len(shells)),
            where=local_reference > _ZERO**2,
        )
        ratio_sum = np.bincount(shells, weights=ratio * valid, minlength=NSHELLS)
        np.divide(
            ratio_sum, points[atom], out=average_ratio[atom], where=points[atom] > 0
        )
        local_conditioned = conditioned_total[sample.indices]
        roots = np.sqrt(local_conditioned, out=np.zeros(len(shells)), where=valid)
        numerator_values = np.divide(
            conditioned[atom, shells],
            roots,
            out=np.zeros(len(shells)),
            where=valid & (local_conditioned > _ZERO**2),
        )
        numerator = np.bincount(shells, weights=numerator_values, minlength=NSHELLS)
        denominator = np.bincount(shells, weights=roots, minlength=NSHELLS)
        np.divide(numerator, denominator, out=tau[atom], where=points[atom] > 0)
    return average_ratio, tau


def _reshape(sigma: FloatArray, points: FloatArray, tau: FloatArray) -> FloatArray:
    result = sigma.copy()
    population = float(sigma @ points)
    roots = np.sqrt(sigma)
    weighted_points = float(roots @ points)
    # The caller skips atoms with negligible population or weighted points.
    exponents = np.exp(-1.75 * (1 - tau**2) / SCALE)
    for _ in range(30):
        result[0] = max(result[0], 0)
        for shell in range(1, NSHELLS):
            result[shell] = max(result[shell], 0)
            if points[shell] > 0 and result[shell - 1] > 0:
                result[shell] = min(result[shell], result[shell - 1] * exponents[shell])
        coefficient = (population - float(result @ points)) / weighted_points
        if abs(coefficient) <= _ZERO:
            break
        result += coefficient * roots
    return result


def partition_ddec3(  # noqa: PLR0913 -- Explicit physical inputs to the partitioner.
    *,
    atomic_numbers: IntArray,
    density: FloatArray,
    samples: Sequence[AtomicSample],
    pixel_volume: float,
    core_electrons: FloatArray,
    core_population: FloatArray,
    occupancy_correction: FloatArray,
    net_charge: float = 0.0,
    convergence: ConvergenceOptions = DEFAULT_CONVERGENCE,
) -> PartitionResult:
    """Partition a prepared total-density grid into DDEC3 atomic populations.

    Inputs have the same atomic-unit conventions as ``partition_ddec6``. Four
    consecutive charge and spherical-density changes must fall below the
    charge tolerance, after at least ten iterations. An unresolved partition
    raises ``DDEC3ConvergenceError``.
    """
    tolerance = convergence.charge_tolerance
    valence_electrons = float(np.sum(atomic_numbers - core_electrons) - net_charge)
    if valence_electrons <= 0:
        msg = (
            "DDEC3 partitioning requires a positive number of valence electrons, "
            f"got {valence_electrons:g} for net_charge={net_charge:g}"
        )
        raise ValueError(msg)
    references = reference_densities(atomic_numbers, np.zeros(len(samples)))
    weights = references.copy()
    conditioned = references.copy()
    spherical = np.zeros_like(weights)
    populations = np.zeros(len(samples))
    reference_charges = np.zeros(len(samples))
    hirshfeld = np.zeros(len(samples))
    renormalization = np.ones(len(samples))
    # Convergence requires four consecutive small changes on every atom.
    changes = np.zeros((4, len(samples)))
    counts = shell_counts(samples)
    for iteration in range(1, _MAX_ITERATIONS + 1):
        total = pseudodensity(weights, samples, density.size)
        points, assigned, _, weighted_density = _shell_statistics(
            weights, samples, density, total, counts
        )
        ratios, tau = _reference_averages(
            references, conditioned, samples, density, total, points
        )
        old_spherical = spherical
        spherical = np.divide(
            assigned, points, out=np.zeros_like(assigned), where=points > 0
        )
        old_population = populations
        populations, normalization = _normalize_population(
            assigned,
            pixel_volume,
            core_population,
            occupancy_correction,
            valence_electrons,
        )
        charges = atomic_numbers - core_electrons - populations
        changes[(iteration - 1) % len(changes)] = np.maximum(
            np.abs(populations - old_population),
            np.max(np.abs(spherical - old_spherical), axis=1),
        )
        if iteration == 1:
            hirshfeld = charges.copy()
        if iteration >= _MINIMUM_ITERATIONS and np.max(changes) < tolerance:
            if np.any(populations < -tolerance):
                msg = "DDEC3 could not assign nonnegative valence populations"
                raise DDEC3ConvergenceError(
                    msg, np.minimum(populations, 0.0), iteration, tolerance
                )
            return PartitionResult(
                charges=charges,
                weights=weights,
                pseudodensity=total,
                spherical_average=spherical,
                weighted_spherical_average=spherical.copy(),
                reference_charges=reference_charges,
                hirshfeld_charges=hirshfeld,
                normalization=normalization,
                iterations=iteration,
            )
        conditioned = references * ratios
        sigma = spherical ** (1 - _REFERENCE_WEIGHT) * conditioned**_REFERENCE_WEIGHT
        weights = sigma.copy()
        reference_charges = charges.copy()
        references = reference_densities(atomic_numbers, reference_charges)
        if iteration >= _FIRST_RESHAPE_ITERATION:
            slope = pixel_volume * (assigned - weighted_density).sum(axis=1)
            for atom in range(len(samples)):
                # The reference implementation skips both reshaping and kappa
                # adjustment when this atom's radial integration is negligible.
                if (
                    min(
                        float(np.sqrt(sigma[atom]) @ points[atom]),
                        float(sigma[atom] @ points[atom]),
                    )
                    < _ZERO
                ):
                    continue
                weights[atom] = _reshape(sigma[atom], points[atom], tau[atom])
                if slope[atom] > tolerance / 100:
                    renormalization[atom] = max(
                        1, renormalization[atom] - 0.2 * populations[atom] / slope[atom]
                    )
                else:
                    renormalization[atom] = 1
                weights[atom] *= renormalization[atom]
    msg = f"DDEC3 atomic charges did not converge within {_MAX_ITERATIONS} iterations"
    raise DDEC3ConvergenceError(
        msg, np.max(changes, axis=0), _MAX_ITERATIONS, tolerance
    )
