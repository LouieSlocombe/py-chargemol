"""Compiled density reductions agree with the NumPy reference paths."""

import inspect

import numpy as np
import pytest

from py_chargemol import _jit
from py_chargemol.density import _kernels, core, ddec3, grid, partition
from py_chargemol.density.core import _local_averages
from py_chargemol.density.grid import AtomicSample, shell_counts, sum_radial_weights
from py_chargemol.density.partition import _shell_statistics


@pytest.fixture(params=["python", "compiled"])
def accelerated_kernels(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.param == "compiled" and not _jit.ENABLED:
        pytest.skip("Numba is not enabled")
    if request.param == "python":
        # Execute the same numerical bodies under coverage without requiring
        # Numba, and independently test their compiled forms when installed.
        for module, name in (
            (core, "core_shell_sums"),
            (core, "assigned_core_values"),
            (core, "add_corrected_core"),
            (_kernels, "add_corrected_core"),
            (_kernels, "reference_shell_sums"),
            (ddec3, "reference_shell_sums"),
            (partition, "partition_shell_sums"),
            (grid, "add_radial_weights"),
        ):
            monkeypatch.setattr(module, name, inspect.unwrap(getattr(module, name)))


def _samples(dtype: type[np.int32 | np.int64], empty: bool) -> list[AtomicSample]:
    indices = np.arange(129, dtype=dtype) % 7
    shells = (np.arange(129) % 5).astype(np.uint8 if dtype == np.int32 else np.int64)
    if empty:
        indices, shells = indices[:0], shells[:0]
    return [
        AtomicSample(
            # Reversing also exercises noncontiguous sample arrays.
            indices=indices[::direction],
            shells=shells[::direction],
            coordinates=np.zeros((len(indices), 3), dtype=np.int32),
            steps=np.eye(3),
            offset=np.zeros(3),
        )
        for direction in (1, -1)
    ]


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_radial_scatter_preserves_periodic_images_and_point_order(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.int32 | np.int64],
    empty: bool,
) -> None:
    samples = _samples(dtype, empty)
    weights = np.zeros((2, 200))[:, ::2]
    weights[:, :5] = [1e16, 1e-16, 0.25, 0.0, 1.0]
    original = weights.copy()
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected = sum_radial_weights(weights, samples, 9)
    monkeypatch.setattr(_jit, "ENABLED", True)
    result = sum_radial_weights(weights, samples, 9)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[7:], 0)
    np.testing.assert_array_equal(weights, original)


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("threshold", [1e-10, 1e-20])
def test_shell_reductions_preserve_thresholds_and_population_sums(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.int32 | np.int64],
    empty: bool,
    threshold: float,
) -> None:
    samples = _samples(dtype, empty)
    weights = np.zeros((2, 200))[:, ::2]
    weights[:, :5] = [1e-12, 0.25, 0.5, 1.0, 3.0]
    density = np.array([1e20, 1e-5, 0, 0.25, 1.0, 0.5, 2.0])
    pseudo = np.array([0.0, 1e-25, 1e-20, 1e-10, 0.5, 1.0, 2.0])
    counts = shell_counts(samples)
    original = counts.copy()
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected_core = _local_averages(density, weights, pseudo, samples, counts)
    expected_partition = _shell_statistics(
        weights, samples, density, pseudo, counts, threshold
    )
    monkeypatch.setattr(_jit, "ENABLED", True)
    actual_core = _local_averages(density, weights, pseudo, samples, counts)
    actual_partition = _shell_statistics(
        weights, samples, density, pseudo, counts, threshold
    )
    for actual, expected in zip(
        (*actual_core, *actual_partition),
        (*expected_core, *expected_partition),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(counts, original)


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_core_correction_preserves_thresholds_and_image_order(
    monkeypatch: pytest.MonkeyPatch, dtype: type[np.int32 | np.int64], empty: bool
) -> None:
    radial = np.linspace(0.1, 1, 200)[::2]
    density = np.array([0, 1e-20, 0.5, 2, 4, 1e-10, 1e-21])
    inverse = np.array([0, 1, 2, 0.5, 0.25, 1, 1])
    for sample in _samples(dtype, empty):
        monkeypatch.setattr(_jit, "ENABLED", False)
        expected = core._assigned(density, radial, inverse, sample)  # noqa: SLF001
        monkeypatch.setattr(_jit, "ENABLED", True)
        values = core._assigned(density, radial, inverse, sample)  # noqa: SLF001
        np.testing.assert_array_equal(values, expected)
        for factor in (-0.05, 0.0, 0.01):
            result = np.zeros(18)[::2]
            corrected = np.where(expected > 1e-20, expected, 0)
            corrected /= np.sqrt(1 - 2 * factor * corrected**2)
            reference = result.copy()
            np.add.at(reference, sample.indices, corrected)
            _kernels.add_corrected_core(values, sample.indices, factor, 1e-20, result)
            np.testing.assert_array_equal(result, reference)


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("population", [0.6, 2.0, 5.0])
def test_complete_core_correction_matches_numpy(
    monkeypatch: pytest.MonkeyPatch, population: float
) -> None:
    sample = AtomicSample(
        indices=np.array([0, 1, 2], dtype=np.int32),
        shells=np.array([0, 1, 2], dtype=np.uint8),
        coordinates=np.zeros((3, 3), dtype=np.int32),
        steps=np.eye(3),
        offset=np.zeros(3),
    )
    weights = np.zeros((1, 100))
    weights[0, :3] = [3, 2, 1]
    density = np.array([0.5, 0.3, 0.2]) * population
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected = core.partition_core(density, np.array([2]), weights, [sample], 1)
    monkeypatch.setattr(_jit, "ENABLED", True)
    actual = core.partition_core(density, np.array([2]), weights, [sample], 1)
    for name in ("density", "populations", "weights", "pseudodensity"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_ddec3_reference_reductions_preserve_thresholds(
    dtype: type[np.int32 | np.int64], empty: bool
) -> None:
    zero = 1e-10
    density = np.array([1, 2, 3, 0, 1e-20, 1e-10, 4.0])
    total = np.array([0, zero, np.nextafter(zero, np.inf), 1, 1, 1, 2])
    reference = np.array([0, 1, 1e-20, np.nextafter(1e-20, np.inf), 2, 3, 1])
    conditioned = np.array([0, 1, 0, 1e-21, 1e-20, 2, 3])
    radial = np.linspace(0.1, 1, 200)[::2]
    for sample in _samples(dtype, empty):
        points, shells = sample.indices, sample.shells
        valid = total[points] > zero
        ratios = np.divide(
            density[points],
            reference[points],
            out=np.ones(len(points)),
            where=reference[points] > zero**2,
        )
        roots = np.sqrt(conditioned[points], out=np.zeros(len(points)), where=valid)
        numerator = np.divide(
            radial[shells],
            roots,
            out=np.zeros(len(points)),
            where=valid & (conditioned[points] > zero**2),
        )
        expected = tuple(
            np.bincount(shells, weights=values, minlength=100)
            for values in (ratios * valid, numerator, roots)
        )
        actual = _kernels.reference_shell_sums(
            radial, density, total, reference, conditioned, points, shells, zero
        )
        for result, target in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(result, target)


@pytest.mark.usefixtures("accelerated_kernels")
@pytest.mark.parametrize("empty", [False, True])
def test_ddec3_reference_averages_match_numpy(
    monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    samples = _samples(np.int32, empty)
    references = np.tile(np.linspace(0.1, 1, 100), (2, 1))
    conditioned = references * 0.5
    density = np.linspace(0, 3, 7)
    total = np.linspace(0, 2, 7)
    counts = shell_counts(samples)
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected = ddec3._reference_averages(  # noqa: SLF001
        references, conditioned, samples, density, total, counts
    )
    monkeypatch.setattr(_jit, "ENABLED", True)
    actual = ddec3._reference_averages(  # noqa: SLF001
        references, conditioned, samples, density, total, counts
    )
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, target)
