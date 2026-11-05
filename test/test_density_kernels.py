"""Compiled density reductions agree with the NumPy reference paths."""

import inspect

import numpy as np
import pytest

from py_chargemol import _jit
from py_chargemol.density import core, grid, partition
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
