"""Slow charge exchange between two overlapping model cores."""

import numpy as np
import pytest

from py_chargemol.density import core
from py_chargemol.density.convergence import ConvergenceOptions
from py_chargemol.density.core import CoreConvergenceError, partition_core
from py_chargemol.density.grid import (
    AtomicSample,
    DensityGrid,
    atomic_samples,
    reciprocal,
)
from py_chargemol.density.references import reference_densities


def _core_pair(
    separation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[AtomicSample, ...], float]:
    grid = DensityGrid(
        atomic_numbers=np.array([3, 11]),
        positions=np.array([[0, 0, 0], [separation, 0, 0]]),
        density=np.zeros((13, 13, 13)),
        origin=np.full(3, -6 * 0.14),
        steps=0.14 * np.eye(3),
    )
    points = np.indices(grid.density.shape).reshape(3, -1).T @ grid.steps + grid.origin
    requested = np.array([2, 10])
    density = np.zeros(len(points))
    for center, count, exponent in zip(
        grid.positions, requested, [30, 100], strict=True
    ):
        radius_squared = np.sum((points - center) ** 2, axis=1)
        density += (
            count * (exponent / np.pi) ** 1.5 * np.exp(-exponent * radius_squared)
        )
    neutral = reference_densities(grid.atomic_numbers, np.zeros(2))
    return density, requested, neutral, atomic_samples(grid), grid.pixel_volume


@pytest.mark.parametrize("separation", [0.098, 0.095, 0.095125])
def test_short_contact_core_pair_converges_with_default_limit(
    separation: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without extrapolation these grids take 152, 339 and 385 corrections.
    # After 40, the two residuals still have opposite signs and exceed 1e-4 e.
    inputs = _core_pair(separation)
    _, requested, _, samples, pixel_volume = inputs
    with pytest.warns(RuntimeWarning, match="core weights did not converge"):
        result = partition_core(*inputs)

    # Repartition the returned grid to check the populations independently.
    assert np.all(result.weights >= 0)
    assert np.all(result.density >= 0)
    inverse = reciprocal(result.pseudodensity)
    populations = np.array(
        [
            np.sum(
                radial[sample.shells]
                * result.density[sample.indices]
                * inverse[sample.indices]
            )
            * pixel_volume
            for radial, sample in zip(result.weights, samples, strict=True)
        ]
    )
    np.testing.assert_allclose(populations, requested, atol=1e-5, rtol=0)
    np.testing.assert_allclose(populations, result.populations, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result.density.sum() * pixel_volume, populations.sum(), atol=1e-12, rtol=0
    )

    # Compare the spatial distribution as well as the populations. Too early
    # a refinement can satisfy the populations with a distorted core shape.
    monkeypatch.setattr(core, "_tail_step", lambda *_args: 0.0)
    with pytest.warns(RuntimeWarning, match="core weights did not converge"):
        reference = partition_core(
            *inputs,
            convergence=ConvergenceOptions(core_max_iterations=500),
        )
    assert np.abs(result.density - reference.density).sum() * pixel_volume < 0.003


def test_core_pair_can_retry_after_an_extrapolation_is_rejected() -> None:
    inputs = _core_pair(0.099375)
    with (
        pytest.warns(RuntimeWarning, match="core weights did not converge"),
        pytest.raises(CoreConvergenceError, match="iterations=40"),
    ):
        partition_core(*inputs)
    with pytest.warns(RuntimeWarning, match="core weights did not converge"):
        result = partition_core(
            *inputs, convergence=ConvergenceOptions(core_max_iterations=90)
        )
    np.testing.assert_allclose(result.populations, [2, 10], atol=1e-5, rtol=0)
