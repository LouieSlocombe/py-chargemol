"""Accelerated spin scatters preserve periodic images and convergence."""

import inspect

import numpy as np
import pytest

from py_chargemol import _jit
from py_chargemol.density import _spin_kernels, spin
from py_chargemol.density.grid import NSHELLS, AtomicSample


@pytest.fixture(params=["python", "compiled"])
def accelerated_spin(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.param == "compiled" and not _jit.ENABLED:
        pytest.skip("Numba is not enabled")
    if request.param == "python":
        for name in ("accumulate_spin", "subtract_spin_theta"):
            for module in (spin, _spin_kernels):
                monkeypatch.setattr(module, name, inspect.unwrap(getattr(module, name)))


@pytest.mark.usefixtures("accelerated_spin")
@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_spin_scatters_preserve_order_and_strided_arrays(
    components: int, dtype: type[np.int32 | np.int64], empty: bool
) -> None:
    rng = np.random.default_rng(103)
    points: np.ndarray = (np.arange(258, dtype=dtype) % 7)[::-2]
    shells: np.ndarray = (np.arange(258) % NSHELLS).astype(
        np.uint8 if dtype == np.int32 else np.int64
    )[::-2]
    if empty:
        points, shells = points[:0], shells[:0]
    size = len(points)
    fractions = rng.uniform(0, 1, size=2 * size)[::2]
    lagrange = rng.uniform(-1, 1, size=(components, 2 * size))[:, ::2]
    assigned = rng.choice([1e16, -1e16, 0.25, -0.5], size=(components, 2 * size))[
        :, ::2
    ]
    sums = np.zeros((NSHELLS, 2 * components))[:, ::2]
    ypsilon = rng.uniform(-1, 1, size=(components, 18))[:, ::2]
    trial = rng.uniform(-1, 1, size=(components, 18))[:, ::2]
    expected_sums = sums.copy()
    expected_ypsilon, expected_trial = ypsilon.copy(), trial.copy()
    for component in range(components):
        expected_sums[:, component] = np.bincount(
            shells, weights=assigned[component], minlength=NSHELLS
        )
        np.add.at(expected_ypsilon[component], points, lagrange[component] * fractions)
        np.add.at(expected_trial[component], points, assigned[component])
    _spin_kernels.accumulate_spin(
        points, shells, fractions, lagrange, assigned, sums, ypsilon, trial
    )
    np.testing.assert_array_equal(sums, expected_sums)
    np.testing.assert_array_equal(ypsilon, expected_ypsilon)
    np.testing.assert_array_equal(trial, expected_trial)

    proportional = rng.uniform(-1, 1, size=(components, 2 * size))[:, ::2]
    theta = rng.uniform(-1, 1, size=(components, 2 * NSHELLS))[:, ::2]
    for component in range(components):
        np.subtract.at(
            expected_ypsilon[component],
            points,
            0.5 * (theta[component, shells] + proportional[component]) * fractions,
        )
    _spin_kernels.subtract_spin_theta(
        points, shells, fractions, proportional, theta, ypsilon
    )
    np.testing.assert_array_equal(ypsilon, expected_ypsilon)


@pytest.mark.usefixtures("accelerated_spin")
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("max_iterations", [1, 500])
def test_spin_partition_matches_numpy_through_convergence(
    monkeypatch: pytest.MonkeyPatch, vector: bool, max_iterations: int
) -> None:
    samples: list[AtomicSample] = []
    weights = np.exp(-np.arange(NSHELLS)[None, :] * np.array([[0.03], [0.05], [0.1]]))
    density = np.zeros(31)
    spherical_average = np.zeros_like(weights)
    for atom in range(3):
        # Duplicate indices stand for periodic images; the third atom is empty.
        points = (np.arange(124, dtype=np.int32) % len(density))[::-2]
        shells = ((np.arange(124) + 7 * atom) % NSHELLS).astype(np.uint8)[::-2]
        if atom == 2:
            points, shells = points[:0], shells[:0]
        samples.append(
            AtomicSample(
                indices=points,
                shells=shells,
                coordinates=np.zeros((len(points), 3), dtype=np.int32),
                steps=np.eye(3),
                offset=np.zeros(3),
            )
        )
        np.add.at(density, points, weights[atom, shells])
        counts = np.bincount(shells, minlength=NSHELLS)
        spherical_average[atom, counts > 0] = weights[atom, counts > 0]
    spin_density = density * np.linspace(-0.2, 0.5, len(density))
    if vector:
        spin_density = spin_density[:, None] * np.array([0.36, 0.48, 0.8])
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(_jit, "ENABLED", enabled)
        results.append(
            spin.partition_spin(
                density=density,
                spin_density=spin_density,
                weights=weights,
                pseudodensity=density,
                spherical_average=spherical_average,
                samples=samples,
                voxel_volume=0.1,
                max_iterations=max_iterations,
            )
        )
    expected, result = results
    for actual, reference in (
        (result.populations, expected.populations),
        (result.spherical_average, expected.spherical_average),
        (result.reconstructed_density, expected.reconstructed_density),
    ):
        np.testing.assert_array_equal(actual, reference)
    assert result.converged == expected.converged == (max_iterations > 1)
    assert result.iterations == expected.iterations
    assert result.max_change == expected.max_change
