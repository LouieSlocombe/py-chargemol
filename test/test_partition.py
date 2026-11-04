"""Conservation and symmetry checks for density-based DDEC6 partitioning."""

import numpy as np
import pytest

from py_chargemol.density.grid import AtomicSample, DensityGrid, atomic_samples
from py_chargemol.density.partition import partition_ddec6, pseudodensity


@pytest.fixture
def hydrogen_grid() -> DensityGrid:
    shape = (25, 25, 25)
    origin = np.full(3, -6.0)
    steps = np.eye(3) * 0.5
    points = np.indices(shape).reshape(3, -1).T @ steps + origin
    positions = np.array([[-0.7, 0, 0], [0.7, 0, 0]])
    density = np.zeros(len(points))
    for position in positions:
        density += np.exp(-np.sum((points - position) ** 2, axis=1)) / np.pi**1.5
    return DensityGrid(
        atomic_numbers=np.ones(2, dtype=np.int64),
        positions=positions,
        density=density.reshape(shape),
        origin=origin,
        steps=steps,
    )


def test_symmetric_density_partitions_equally_without_external_files(
    hydrogen_grid: DensityGrid,
) -> None:
    samples = atomic_samples(hydrogen_grid)
    result = partition_ddec6(
        atomic_numbers=hydrogen_grid.atomic_numbers,
        density=hydrogen_grid.density.ravel(),
        samples=samples,
        pixel_volume=hydrogen_grid.pixel_volume,
        core_electrons=np.zeros(2),
        core_population=np.zeros(2),
        occupancy_correction=np.zeros(2),
    )
    np.testing.assert_allclose(result.charges, [0, 0], atol=1e-12)
    np.testing.assert_allclose(result.hirshfeld_charges, [0, 0], atol=1e-12)
    np.testing.assert_allclose(result.weights[0], result.weights[1], atol=1e-12)
    assert result.iterations == 7
    assert result.normalization == pytest.approx(1, abs=1e-8)
    np.testing.assert_allclose(
        pseudodensity(result.weights, samples, hydrogen_grid.density.size),
        result.pseudodensity,
    )
    assigned_population = 0.0
    density = hydrogen_grid.density.ravel()
    for atom, sample in enumerate(samples):
        assigned_population += hydrogen_grid.pixel_volume * np.sum(
            result.weights[atom, sample.shells]
            / result.pseudodensity[sample.indices]
            * density[sample.indices]
        )
    assert assigned_population == pytest.approx(2.0, abs=1e-8)


def test_nonzero_net_charge_is_conserved(hydrogen_grid: DensityGrid) -> None:
    result = partition_ddec6(
        atomic_numbers=hydrogen_grid.atomic_numbers,
        density=hydrogen_grid.density.ravel() / 2,
        samples=atomic_samples(hydrogen_grid),
        pixel_volume=hydrogen_grid.pixel_volume,
        core_electrons=np.zeros(2),
        core_population=np.zeros(2),
        occupancy_correction=np.zeros(2),
        net_charge=1,
    )
    np.testing.assert_allclose(result.charges, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize(
    ("charge", "zero_density", "message"),
    [(2.0, False, "positive number"), (0.0, True, "no assignable valence")],
)
def test_empty_electron_populations_are_rejected(
    hydrogen_grid: DensityGrid, charge: float, zero_density: bool, message: str
) -> None:
    density = hydrogen_grid.density.ravel()
    with pytest.raises(ValueError, match=message):
        partition_ddec6(
            atomic_numbers=hydrogen_grid.atomic_numbers,
            density=np.zeros_like(density) if zero_density else density,
            samples=atomic_samples(hydrogen_grid),
            pixel_volume=hydrogen_grid.pixel_volume,
            core_electrons=np.zeros(2),
            core_population=np.zeros(2),
            occupancy_correction=np.zeros(2),
            net_charge=charge,
        )


def _lithium_hydride_samples(single_point: bool = False) -> list[AtomicSample]:
    shells = [[10], [0]] if single_point else [[0, 10], [10, 0]]
    return [
        AtomicSample(
            indices=np.arange(len(row)),
            shells=np.array(row),
            coordinates=np.zeros((len(row), 3), dtype=np.int32),
            steps=np.eye(3),
            offset=np.zeros(3),
        )
        for row in shells
    ]


def test_nonnegative_valence_constraint_reaches_ionic_limit() -> None:
    result = partition_ddec6(
        atomic_numbers=np.array([3, 1]),
        density=np.array([1.9, 2.1]),
        samples=_lithium_hydride_samples(),
        pixel_volume=1,
        core_electrons=np.array([2.0, 0.0]),
        core_population=np.array([2.0, 0.0]),
        occupancy_correction=np.zeros(2),
    )
    assert result.iterations > 7
    np.testing.assert_allclose(result.charges, [1, -1], atol=1e-12)


@pytest.mark.parametrize(
    ("single_point", "density", "message"),
    [
        (False, [1.5, 2.5], "did not converge within 500"),
        (True, [4.0], "nonnegative valence populations"),
    ],
)
def test_inconsistent_core_partition_is_not_returned_as_a_valid_result(
    single_point: bool, density: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        partition_ddec6(
            atomic_numbers=np.array([3, 1]),
            density=np.array(density),
            samples=_lithium_hydride_samples(single_point),
            pixel_volume=1,
            core_electrons=np.array([2.0, 0.0]),
            core_population=np.array([2.0, 0.0]),
            occupancy_correction=np.zeros(2),
        )
