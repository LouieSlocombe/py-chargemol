"""Core population correction and dominant-atom grid assignment."""

import numpy as np
import pytest

from py_chargemol.density.core import (
    dominant_occupancy_correction,
    partition_core,
    sum_radial_weights,
)
from py_chargemol.density.grid import AtomicSample, DensityGrid, atomic_samples
from py_chargemol.density.references import reference_densities


def _sample(indices: list[int], shells: list[int]) -> AtomicSample:
    return AtomicSample(
        indices=np.array(indices),
        shells=np.array(shells),
        displacements=np.zeros((len(indices), 3)),
        coordinates=np.zeros((len(indices), 3), dtype=np.int32),
    )


def test_radial_weights_accumulate_repeated_periodic_images() -> None:
    samples = [_sample([0, 1, 0], [0, 1, 2]), _sample([0, 1], [0, 1])]
    radial = np.zeros((2, 100))
    radial[0, :3] = [1, 2, 3]
    radial[1, :2] = [4, 5]
    np.testing.assert_array_equal(sum_radial_weights(radial, samples, 3), [8, 7, 0])


@pytest.mark.parametrize("initial_population", [0.6, 2.0, 5.0])
def test_core_correction_recovers_requested_electron_count(
    initial_population: float,
) -> None:
    samples = [_sample([0, 1, 2], [0, 1, 2])]
    weights = np.zeros((1, 100))
    weights[0, :3] = [3, 2, 1]
    density = np.array([0.5, 0.3, 0.2]) * initial_population
    original = density.copy()
    result = partition_core(density, np.array([2]), weights, samples, 1.0)
    np.testing.assert_allclose(result.populations, [2], atol=1e-5)
    assert result.density.sum() == pytest.approx(2, abs=1e-5)
    assert np.all(result.density >= 0)
    np.testing.assert_array_equal(density, original)
    np.testing.assert_array_equal(
        result.pseudodensity, sum_radial_weights(result.weights, samples, 3)
    )


def test_two_separated_cores_recover_individual_populations() -> None:
    samples = [_sample([0, 1], [0, 1]), _sample([2, 3], [0, 1])]
    weights = np.zeros((2, 100))
    weights[:, :2] = [2, 1]
    result = partition_core(
        np.array([0.7, 0.3, 2.0, 1.0]), np.array([2, 4]), weights, samples, 1
    )
    np.testing.assert_allclose(result.populations, [2, 4], atol=1e-5)


def test_atoms_without_core_electrons_have_zero_core_density() -> None:
    result = partition_core(
        np.ones(3), np.zeros(1), np.ones((1, 100)), [_sample([0, 1, 2], [0, 1, 2])], 1
    )
    np.testing.assert_array_equal(result.density, 0)
    np.testing.assert_array_equal(result.populations, 0)
    np.testing.assert_array_equal(result.weights, 0)
    np.testing.assert_array_equal(result.pseudodensity, 0)


def test_missing_core_support_fails_instead_of_returning_wrong_populations() -> None:
    with pytest.raises(ValueError, match="could not be corrected"):
        partition_core(
            np.zeros(2), np.array([2]), np.ones((1, 100)), [_sample([0, 1], [0, 1])], 1
        )


def test_dominant_correction_uses_nearest_image_and_preserves_ties() -> None:
    samples = [_sample([0, 0, 1], [0, 1, 2]), _sample([0, 1, 2], [0, 1, 2])]
    weights = np.zeros((2, 100))
    weights[0, :3] = [5, 1, 4]
    weights[1, :3] = [3, 4, 2]
    difference = np.array([0.2, -0.1, 0.6, 100.0])
    result = dominant_occupancy_correction(difference, weights, samples, 0.5)
    # Atom0 owns point0 through its near image and wins the equal-weight tie at1.
    # Atom1 owns point2; point3 has no atom and is not assigned.
    np.testing.assert_allclose(result, [0.05, 0.3])


def test_unresolvable_overlapping_cores_fail_with_a_clear_error() -> None:
    shape = (21, 21, 21)
    origin = np.full(3, -1.0)
    steps = 0.1 * np.eye(3)
    positions = np.array([[0, 0, 0], [0.05, 0, 0]])
    points = np.indices(shape).reshape(3, -1).T @ steps + origin
    density = np.zeros(len(points))
    for position, population, exponent in zip(
        positions, [2, 10], [30, 100], strict=True
    ):
        density += (
            population
            * (exponent / np.pi) ** 1.5
            * np.exp(-exponent * np.sum((points - position) ** 2, axis=1))
        )
    grid = DensityGrid(
        atomic_numbers=np.array([3, 11]),
        positions=positions,
        density=np.zeros(shape),
        origin=origin,
        steps=steps,
    )
    with pytest.raises(ValueError, match="core density could not be corrected"):
        partition_core(
            density,
            np.array([2, 10]),
            reference_densities(grid.atomic_numbers, np.zeros(2)),
            atomic_samples(grid),
            grid.pixel_volume,
        )
