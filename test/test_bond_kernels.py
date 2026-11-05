"""Accelerated pair quadrature preserves shell boundaries and periodic images."""

from dataclasses import dataclass

import numpy as np
import pytest

from py_chargemol import _jit
from py_chargemol.density._bond_kernels import pair_shell_sums
from py_chargemol.density.bonds import _pair_integrals, _PairFields
from py_chargemol.density.grid import NSHELLS, SCALE, ZERO


@dataclass(frozen=True)
class Sample:
    indices: np.ndarray
    shells: np.ndarray
    displacements: np.ndarray


@pytest.mark.parametrize("index_type", ["int32", "int64"])
@pytest.mark.parametrize("selection", ["all", "empty", "strided"])
def test_pair_quadrature_matches_numpy(
    monkeypatch: pytest.MonkeyPatch, index_type: str, selection: str
) -> None:
    rng = np.random.default_rng(153)
    # Include the nucleus, shell boundaries and the cutoff, plus general points.
    boundary = (np.array([1.0, 50.0, 100.0]) - ZERO) / SCALE
    radius = np.concatenate(
        ([0.0], np.nextafter(boundary, 0), boundary, np.nextafter(boundary, np.inf))
    )
    rows = rng.uniform(-7, 7, size=(3, 80))[:, ::2]
    rows[:, : len(radius)] = 0
    rows[0, : len(radius)] = radius
    indices = np.tile(np.arange(10, dtype=index_type), 4)
    shells = rng.integers(0, NSHELLS, size=80, dtype=np.uint8)[::2]
    sample = Sample(indices, shells, rows.T)
    selected = np.arange(len(indices), dtype=np.intp)
    if selection == "empty":
        selected = selected[:0]
    elif selection == "strided":
        selected = selected[::2]
    factors = rng.uniform(0, 1, size=(4, 20))[:, ::2]
    fields = _PairFields(
        weights=rng.uniform(0, 1, size=(2, NSHELLS)),
        vectors=rng.uniform(-1, 1, size=(2, NSHELLS, 4)),
        point_factors=factors,
    )
    separation = np.zeros(3)
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected = _pair_integrals(0, 1, separation, sample, rows, selected, fields)
    monkeypatch.setattr(_jit, "ENABLED", True)
    actual = _pair_integrals(0, 1, separation, sample, rows, selected, fields)
    np.testing.assert_array_equal(actual, expected)


def test_repeated_grid_indices_are_individual_image_contributions() -> None:
    sums = pair_shell_sums(
        np.zeros(3),
        np.array([0, 0, 0], dtype=np.int32),
        np.array([3, 3, 4], dtype=np.uint8),
        np.zeros((3, 3)),
        np.arange(3),
        np.array([[2.0], [3.0], [4.0], [5.0]]),
        SCALE,
        ZERO,
        NSHELLS,
    )
    np.testing.assert_array_equal(sums[:, 3 * NSHELLS], [4, 6, 8, 10])
    np.testing.assert_array_equal(sums[:, 4 * NSHELLS], [2, 3, 4, 5])
    np.testing.assert_array_equal(sums.sum(axis=1), [6, 9, 12, 15])
