"""DDEC overlap populations, contact exchanges, and effective bond orders.

The equations follow the September 2017 Chargemol bond-analysis modules by
Thomas A. Manz and Nidia Gabaldon Limas. All integrations use electron densities
in electrons/bohr³ and the same 100 radial shells within five angstroms used by
the DDEC partition. No external executable is called.
"""

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from py_chargemol.ddec6 import FloatArray, IntArray, OverlapPairs
from py_chargemol.density.grid import (
    CUTOFF,
    NSHELLS,
    SCALE,
    ZERO,
    RadialSample,
    integration_extent,
    nearest_grid_points,
    reciprocal,
)

_CHARGE_TOLERANCE = 1e-5


class AtomicSampleLike(RadialSample, Protocol):
    """An atom-centred grid, including periodic images of grid points."""

    @property
    def displacements(self) -> FloatArray:
        """Unwrapped Cartesian displacement from the reference atom."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class BondOrderResult:
    """Directed bond data and diagnostics from a density partition.

    ``bond_orders``, ``contact_exchanges``, and ``spin_polarization_terms`` have
    the same directed-pair ordering as ``overlaps``. Each physical pair occurs
    in both directions. ``summed_bond_orders`` includes contact exchange from
    pairs smaller than the screening threshold, as in Chargemol.
    """

    overlaps: OverlapPairs
    bond_orders: FloatArray
    contact_exchanges: FloatArray
    spin_polarization_terms: FloatArray
    summed_bond_orders: FloatArray
    summed_contact_exchanges: FloatArray
    corrected_density: FloatArray
    corrected_spherical_average: FloatArray
    density_correction_iterations: int
    population_residual: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class _DensityCorrection:
    density: FloatArray
    spherical_average: FloatArray
    shell_populations: FloatArray
    iterations: int
    residual: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class _PairSample:
    indices: IntArray
    shells: IntArray
    displacements: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class _PairDomains:
    centers: IntArray
    extent: IntArray
    shape: IntArray
    coordinates: tuple[IntArray, ...]


def _pair_domains(
    positions: FloatArray,
    lattice_vectors: FloatArray,
    samples: Sequence[AtomicSampleLike],
    steps: FloatArray,
    origin: FloatArray,
) -> _PairDomains:
    inverse = np.linalg.inv(steps)
    extent = integration_extent(steps)
    centers = (positions - origin) @ inverse
    coordinates = tuple(
        nearest_grid_points(sample.displacements @ inverse + centers[atom])
        for atom, sample in enumerate(samples)
    )
    shape = nearest_grid_points(np.diag(lattice_vectors @ inverse))
    return _PairDomains(nearest_grid_points(centers), extent, shape, coordinates)


def _shared_domain(  # noqa: PLR0913, PLR0917
    domain: _PairDomains,
    atom_i: int,
    atom_j: int,
    shift: IntArray,
    separation: FloatArray,
    sample: AtomicSampleLike,
) -> AtomicSampleLike:
    center_i = domain.centers[atom_i]
    center_j = domain.centers[atom_j] + shift * domain.shape
    middle = 0.5 * (center_i + center_j)
    multiplier = np.sqrt(max(1.0 - np.sum(separation**2) / (4.0 * CUTOFF**2), 0.0))
    lower = np.maximum(
        -domain.extent + np.maximum(center_i, center_j),
        nearest_grid_points(-domain.extent * multiplier + middle) - 2,
    )
    upper = np.minimum(
        domain.extent + np.minimum(center_i, center_j),
        nearest_grid_points(domain.extent * multiplier + middle) + 2,
    )
    coordinates = domain.coordinates[atom_i]
    active = np.all((coordinates >= lower) & (coordinates <= upper), axis=1)
    return _PairSample(
        sample.indices[active], sample.shells[active], sample.displacements[active]
    )


def _spherical_average(
    density: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    samples: Sequence[AtomicSampleLike],
    voxel_volume: float,
) -> tuple[FloatArray, FloatArray]:
    averages = np.zeros_like(weights)
    shell_populations = np.zeros_like(weights)
    inverse = reciprocal(pseudodensity)
    for atom, sample in enumerate(samples):
        local_inverse = inverse[sample.indices]
        shells = sample.shells
        assigned = density[sample.indices] * weights[atom][shells] * local_inverse
        sums = np.bincount(shells, weights=assigned, minlength=NSHELLS)
        counts = np.bincount(shells, weights=local_inverse > 0, minlength=NSHELLS)
        np.divide(sums, counts, out=averages[atom], where=counts > 0)
        shell_populations[atom] = sums * voxel_volume
    return averages, shell_populations


def _correct_density(  # noqa: PLR0913, PLR0917
    density: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    samples: Sequence[AtomicSampleLike],
    populations: FloatArray,
    voxel_volume: float,
    near_nucleus_indices: Sequence[IntArray] | None,
) -> _DensityCorrection:
    corrected = density.copy()
    averages, shell_populations = _spherical_average(
        corrected, weights, pseudodensity, samples, voxel_volume
    )
    residual = populations - shell_populations.sum(axis=1)
    if near_nucleus_indices is None:
        return _DensityCorrection(corrected, averages, shell_populations, 0, residual)
    iterations = 0
    for iterations in range(1, 41):  # noqa: B007 - return the iteration count
        residual = populations - shell_populations.sum(axis=1)
        for atom, points in enumerate(near_nucleus_indices):
            local_density = corrected[points]
            maximum = float(np.max(local_density, initial=0))
            if maximum <= ZERO:
                continue
            sum_cubes = float(np.sum(local_density**3))
            factor = residual[atom] / (sum_cubes * voxel_volume)
            factor = min(factor, 0.25 / maximum**2)
            active = local_density > ZERO
            corrected[points[active]] = local_density[active] / np.sqrt(
                1.0 - 2.0 * factor * local_density[active] ** 2
            )
        averages, shell_populations = _spherical_average(
            corrected, weights, pseudodensity, samples, voxel_volume
        )
        if np.sum(np.abs(residual)) < _CHARGE_TOLERANCE:
            break
    return _DensityCorrection(
        corrected,
        averages,
        shell_populations,
        iterations,
        populations - shell_populations.sum(axis=1),
    )


def _exchange_vectors(
    averages: FloatArray,
    samples: Sequence[AtomicSampleLike],
    grid_size: int,
    spherical_spin: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    vectors = averages[..., np.newaxis]
    if spherical_spin is not None:
        spin = np.asarray(spherical_spin, dtype=np.float64)
        if spin.ndim == 2:  # noqa: PLR2004
            spin = spin[..., np.newaxis]
        if spin.shape[:2] != averages.shape or spin.shape[2:] not in ((1,), (3,)):
            msg = "spherical_spin must have shape (n, 100), (n, 100, 1), or (n, 100, 3)"
            raise ValueError(msg)
        if not np.all(np.isfinite(spin)):
            msg = "spherical_spin must contain finite values"
            raise ValueError(msg)
        vectors = np.concatenate((vectors, spin), axis=2)
    total_vectors = np.zeros((grid_size, vectors.shape[2]))
    for atom, sample in enumerate(samples):
        np.add.at(total_vectors, sample.indices, vectors[atom, sample.shells])
    minimum_density = np.full_like(averages, 1e9)
    for atom, sample in enumerate(samples):
        np.minimum.at(
            minimum_density[atom], sample.shells, total_vectors[sample.indices, 0]
        )
    return vectors, total_vectors, minimum_density


@lru_cache(maxsize=201)
def _overlap_estimate_grid(separation: int) -> tuple[IntArray, IntArray, FloatArray]:
    """Precompute cylindrical quadrature for the original pair screening test."""
    radial_i: list[int] = []
    radial_j: list[int] = []
    measures: list[float] = []
    for x in range(separation - NSHELLS, NSHELLS + 1):
        maximum = NSHELLS**2 - max(x**2, (separation - x) ** 2)
        for height in range(int(np.ceil(np.sqrt(maximum))) + 1):
            shell_i = int(np.floor(np.hypot(x, height) + 0.5))
            shell_j = int(np.floor(np.hypot(separation - x, height) + 0.5))
            if shell_i < NSHELLS and shell_j < NSHELLS:
                radial_i.append(shell_i)
                radial_j.append(shell_j)
                measures.append(4.0 * np.pi * height / SCALE**3)
    return (
        np.asarray(radial_i, dtype=np.int64),
        np.asarray(radial_j, dtype=np.int64),
        np.asarray(measures, dtype=np.float64),
    )


def _significant_pair(  # noqa: PLR0913, PLR0917
    atom_i: int,
    atom_j: int,
    distance: float,
    averages: FloatArray,
    outside: FloatArray,
    minimum_density: FloatArray,
    pair_cutoff: float,
) -> bool:
    separation = int(np.ceil(SCALE * distance + ZERO))
    upper_bound = 1.0
    for shell in range(1, separation):
        other = separation - shell
        if shell <= NSHELLS and other <= NSHELLS:
            upper_bound = min(
                upper_bound,
                2.0 * (outside[atom_i, shell - 1] + outside[atom_j, other - 1]),
            )
    if upper_bound < pair_cutoff or separation > 2 * NSHELLS:
        return False
    shells_i, shells_j, measures = _overlap_estimate_grid(separation)
    density_i = averages[atom_i, shells_i]
    density_j = averages[atom_j, shells_j]
    denominator = np.maximum(
        np.maximum(
            minimum_density[atom_i, shells_i], minimum_density[atom_j, shells_j]
        ),
        density_i + density_j,
    )
    active = denominator > ZERO
    estimate = np.sum(
        measures[active] * density_i[active] * density_j[active] / denominator[active]
    )
    return bool(estimate > 0.5 * pair_cutoff)


def _candidate_pairs(
    positions: FloatArray,
    lattice_vectors: FloatArray,
    periodic: tuple[bool, bool, bool],
) -> Iterator[tuple[int, int, IntArray, FloatArray]]:
    limits = np.zeros(3, dtype=np.int64)
    if any(periodic):
        inverse = np.linalg.inv(lattice_vectors)
        limits = np.where(
            periodic, np.ceil(2.0 * CUTOFF * np.linalg.norm(inverse, axis=0)) + 1, 0
        ).astype(np.int64)
    translations = list(itertools.product(*(range(-k, k + 1) for k in limits)))
    for atom_i in range(len(positions)):
        for atom_j in range(atom_i, len(positions)):
            for translation in translations:
                if atom_i == atom_j and translation <= (0, 0, 0):
                    continue
                shift = np.asarray(translation, dtype=np.int64)
                separation = (
                    positions[atom_j] + shift @ lattice_vectors - positions[atom_i]
                )
                if np.linalg.norm(separation) <= 2.0 * CUTOFF:
                    yield atom_i, atom_j, shift, separation


def _pair_integrals(  # noqa: PLR0913, PLR0917
    atom_i: int,
    atom_j: int,
    separation: FloatArray,
    sample: AtomicSampleLike,
    density: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    vectors: FloatArray,
    total_vectors: FloatArray,
    voxel_volume: float,
) -> FloatArray:
    distance_j = np.linalg.norm(sample.displacements - separation, axis=1)
    shells_j = (np.ceil(SCALE * distance_j + ZERO) - 1).astype(np.int64)
    active = shells_j < NSHELLS
    points = sample.indices[active]
    shells_i = sample.shells[active]
    shells_j = shells_j[active]
    total = total_vectors[points]
    denominator = np.sum(total**2, axis=1)
    active_density = (denominator > ZERO**2) & (pseudodensity[points] > ZERO)
    points = points[active_density]
    shells_i = shells_i[active_density]
    shells_j = shells_j[active_density]
    denominator = denominator[active_density]
    prefactor = 2.0 * voxel_volume * density[points]
    dot_products = np.sum(vectors[atom_i, shells_i] * vectors[atom_j, shells_j], axis=1)
    ratio = dot_products / denominator
    integrals = np.asarray(
        [
            np.sum(prefactor * ratio),
            np.sum(prefactor * dot_products / total[active_density, 0] ** 2),
            np.sum(
                prefactor
                * weights[atom_i, shells_i]
                * weights[atom_j, shells_j]
                / pseudodensity[points] ** 2
            ),
            (10.0 / 3.0) * np.sum(prefactor * ratio**2),
        ],
        dtype=np.float64,
    )
    return np.maximum(integrals, 0.0)


def _summed_exchange(
    density: FloatArray,
    samples: Sequence[AtomicSampleLike],
    vectors: FloatArray,
    total_vectors: FloatArray,
    voxel_volume: float,
) -> FloatArray:
    summed = np.zeros(len(samples))
    for atom, sample in enumerate(samples):
        total = total_vectors[sample.indices]
        denominator = np.sum(total**2, axis=1)
        local = vectors[atom, sample.shells]
        product = np.sum(local * (total - local), axis=1)
        exchange = np.divide(
            density[sample.indices] * product,
            denominator,
            out=np.zeros(len(denominator)),
            where=denominator > ZERO**2,
        )
        summed[atom] = 2.0 * voxel_volume * np.sum(exchange)
    return summed


def _final_bond_orders(
    atom_i: IntArray,
    atom_j: IntArray,
    integrals: FloatArray,
    summed_exchange: FloatArray,
    populations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    contact = integrals[:, 0]
    squared = np.zeros(len(populations))
    np.add.at(squared, atom_i, contact**2)
    np.add.at(squared, atom_j, contact**2)
    active = (squared[atom_i] > ZERO) & (squared[atom_j] > ZERO)
    coordination = np.zeros(len(contact))
    coordination[active] = (
        1.0
        - np.tanh(
            (
                summed_exchange[atom_i[active]] ** 2 / squared[atom_i[active]]
                + summed_exchange[atom_j[active]] ** 2 / squared[atom_j[active]]
                - 2.0
            )
            / 26.0
        )
        ** 2
    )
    expansion = coordination * np.minimum(integrals[:, 3] + contact**2 / 6.0, contact)
    scaling_sum = np.zeros(len(populations))
    np.add.at(scaling_sum, atom_i, expansion)
    np.add.at(scaling_sum, atom_j, expansion)
    scale = np.ones(len(populations))
    np.divide(
        populations - 0.5 * summed_exchange,
        scaling_sum,
        out=scale,
        where=scaling_sum > ZERO,
    )
    correction = expansion * np.minimum(1.0, np.minimum(scale[atom_i], scale[atom_j]))
    summed_bonds = summed_exchange.copy()
    np.add.at(summed_bonds, atom_i, correction)
    np.add.at(summed_bonds, atom_j, correction)
    return contact + correction, summed_bonds


def compute_bond_orders(  # noqa: PLR0913
    *,
    positions: FloatArray,
    lattice_vectors: FloatArray,
    periodic: tuple[bool, bool, bool],
    density: FloatArray,
    weights: FloatArray,
    pseudodensity: FloatArray,
    samples: Sequence[AtomicSampleLike],
    populations: FloatArray,
    voxel_volume: float,
    near_nucleus_indices: Sequence[IntArray] | None = None,
    spherical_spin: FloatArray | None = None,
    pair_cutoff: float = 0.001,
    grid_steps: FloatArray | None = None,
    grid_origin: FloatArray | None = None,
) -> BondOrderResult:
    """Integrate DDEC overlaps and bond orders from a converged partition.

    ``density`` and ``pseudodensity`` are flattened all-electron and weighting
    densities, respectively. ``weights`` is the final ``(n, 100)`` DDEC radial
    weighting density; ``populations`` contains the target all-electron atomic
    populations (atomic number minus net charge). Each atomic sample includes
    every grid image within five angstroms of its reference atom.

    Supply ``near_nucleus_indices`` with the 5x5x5 grid stencil around each
    nucleus to apply Chargemol's iterative population correction. Omitting it
    integrates the provided density unchanged. Optional ``spherical_spin``
    contains the converged spherical atomic spin densities: one component for
    collinear spin, or three Cartesian components for noncollinear spin.

    ``pair_cutoff`` controls the original radial upper-bound and cylindrical
    overlap-estimate screening. Returned overlap populations are not truncated
    by that value: every screened-in pair is retained for QDR calculations.
    Supplying ``grid_steps`` and ``grid_origin`` also reproduces Chargemol's
    shared-parallelepiped integration bounds for each pair. Omitting them uses
    the full intersection of the atomic integration domains.
    """
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        msg = "voxel_volume must be finite and positive"
        raise ValueError(msg)
    if not np.isfinite(pair_cutoff) or pair_cutoff < 0:
        msg = "pair_cutoff must be finite and nonnegative"
        raise ValueError(msg)
    natoms = len(samples)
    if weights.shape != (natoms, NSHELLS) or populations.shape != (natoms,):
        msg = "weights must have shape (n, 100) and populations shape (n,)"
        raise ValueError(msg)
    if near_nucleus_indices is not None and len(near_nucleus_indices) != natoms:
        msg = "near_nucleus_indices must contain one stencil per atom"
        raise ValueError(msg)
    if (grid_steps is None) != (grid_origin is None):
        msg = "grid_steps and grid_origin must be supplied together"
        raise ValueError(msg)
    domains = None
    if grid_steps is not None and grid_origin is not None:
        domains = _pair_domains(
            positions, lattice_vectors, samples, grid_steps, grid_origin
        )
    correction = _correct_density(
        density,
        weights,
        pseudodensity,
        samples,
        populations,
        voxel_volume,
        near_nucleus_indices,
    )
    vectors, total_vectors, minimum_density = _exchange_vectors(
        correction.spherical_average, samples, len(density), spherical_spin
    )
    outside = np.cumsum(correction.shell_populations[:, ::-1], axis=1)[:, ::-1]
    summed_exchange = _summed_exchange(
        correction.density, samples, vectors, total_vectors, voxel_volume
    )
    first: list[int] = []
    second: list[int] = []
    shifts: list[IntArray] = []
    integral_rows: list[FloatArray] = []
    for atom_i, atom_j, shift, separation in _candidate_pairs(
        positions, lattice_vectors, periodic
    ):
        if not _significant_pair(
            atom_i,
            atom_j,
            float(np.linalg.norm(separation)),
            correction.spherical_average,
            outside,
            minimum_density,
            pair_cutoff,
        ):
            continue
        first.append(atom_i)
        second.append(atom_j)
        shifts.append(shift)
        sample = samples[atom_i]
        if domains is not None:
            sample = _shared_domain(domains, atom_i, atom_j, shift, separation, sample)
        integral_rows.append(
            _pair_integrals(
                atom_i,
                atom_j,
                separation,
                sample,
                correction.density,
                weights,
                pseudodensity,
                vectors,
                total_vectors,
                voxel_volume,
            )
        )
    first_atoms = np.asarray(first, dtype=np.int64)
    second_atoms = np.asarray(second, dtype=np.int64)
    translations = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    integrals = np.asarray(integral_rows, dtype=np.float64).reshape(-1, 4)
    bond_orders, summed_bonds = _final_bond_orders(
        first_atoms, second_atoms, integrals, summed_exchange, populations
    )
    # Interleave the two directions, matching Chargemol's overlap output.
    directed_first = np.column_stack((first_atoms, second_atoms)).reshape(-1)
    directed_second = np.column_stack((second_atoms, first_atoms)).reshape(-1)
    directed_shifts = np.stack((translations, -translations), axis=1).reshape(-1, 3)
    return BondOrderResult(
        overlaps=OverlapPairs(
            directed_first,
            directed_second,
            directed_shifts,
            np.repeat(integrals[:, 2], 2),
        ),
        bond_orders=np.repeat(bond_orders, 2),
        contact_exchanges=np.repeat(integrals[:, 0], 2),
        spin_polarization_terms=np.repeat(integrals[:, 1], 2),
        summed_bond_orders=summed_bonds,
        summed_contact_exchanges=summed_exchange,
        corrected_density=correction.density,
        corrected_spherical_average=correction.spherical_average,
        density_correction_iterations=correction.iterations,
        population_residual=correction.residual,
    )
