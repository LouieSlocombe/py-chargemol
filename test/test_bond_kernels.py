"""Accelerated bond preparation and quadrature preserve the NumPy results."""

import inspect
from dataclasses import dataclass

import numpy as np
import pytest

from py_chargemol import _jit
from py_chargemol.density import bonds
from py_chargemol.density._bond_kernels import pair_shell_sums
from py_chargemol.density.bonds import (
    _exchange_vectors,
    _pair_integrals,
    _PairFields,
    _spherical_average,
    _summed_exchange,
    compute_bond_orders,
)
from py_chargemol.density.grid import NSHELLS, SCALE, ZERO, shell_counts


@dataclass(frozen=True)
class Sample:
    indices: np.ndarray
    shells: np.ndarray
    displacements: np.ndarray


@pytest.fixture(params=["python", "compiled"])
def accelerated_bond_kernels(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.param == "compiled" and not _jit.ENABLED:
        pytest.skip("Numba is not enabled")
    if request.param == "python":
        for name in (
            "spherical_shell_sums",
            "add_exchange_vectors",
            "minimum_shell_density",
            "exchange_values",
            "pair_shell_sums",
        ):
            monkeypatch.setattr(bonds, name, inspect.unwrap(getattr(bonds, name)))


@pytest.mark.usefixtures("accelerated_bond_kernels")
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


@pytest.mark.usefixtures("accelerated_bond_kernels")
@pytest.mark.parametrize("components", [1, 2, 4])
@pytest.mark.parametrize("index_type", [np.int32, np.int64])
@pytest.mark.parametrize("selection", ["all", "empty", "strided"])
def test_bond_preparation_matches_numpy(
    monkeypatch: pytest.MonkeyPatch,
    components: int,
    index_type: type[np.int32 | np.int64],
    selection: str,
) -> None:
    rng = np.random.default_rng(154)
    # Uneven, repeated indices are separate periodic-image contributions.
    indices = (np.arange(129) % 9).astype(index_type)
    shells = (np.arange(129) % 7).astype(np.uint8)
    if selection == "empty":
        indices, shells = indices[:0], shells[:0]
    elif selection == "strided":
        indices, shells = indices[::-2], shells[::-2]
    samples = [
        Sample(indices[::direction], shells[::direction], np.zeros((len(indices), 3)))
        for direction in (1, -1)
    ]
    weights = rng.uniform(0.1, 1, size=(2, 2 * NSHELLS))[:, ::2]
    density = rng.uniform(0.1, 1, size=20)[::2]
    pseudo = np.array([0, ZERO / 2, ZERO, np.nextafter(ZERO, np.inf), 1] * 2)
    counts = shell_counts(samples)
    original = counts.copy()
    spin = (
        None
        if components == 1
        else rng.uniform(-0.1, 0.1, size=(2, NSHELLS, 2 * (components - 1)))[..., ::2]
    )
    monkeypatch.setattr(_jit, "ENABLED", False)
    expected_average, expected_population = _spherical_average(
        density, weights, pseudo, samples, 0.125, counts
    )
    expected_vectors = _exchange_vectors(weights, samples, len(density), spin)
    radial, total, _ = expected_vectors
    # Exercise both sides of the strict exchange-denominator threshold.
    squared = np.sum(total**2, axis=1)
    squared[:3] = [0, ZERO**2, np.nextafter(ZERO**2, np.inf)]
    expected_exchange = _summed_exchange(
        density, samples, radial, total, squared, 0.125
    )
    monkeypatch.setattr(_jit, "ENABLED", True)
    actual_average, actual_population = _spherical_average(
        density, weights, pseudo, samples, 0.125, counts
    )
    actual_vectors = _exchange_vectors(weights, samples, len(density), spin)
    actual_exchange = _summed_exchange(density, samples, radial, total, squared, 0.125)
    for actual, expected in zip(
        (actual_average, actual_population, *actual_vectors, actual_exchange),
        (expected_average, expected_population, *expected_vectors, expected_exchange),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(counts, original)


@pytest.mark.usefixtures("accelerated_bond_kernels")
@pytest.mark.parametrize("components", [1, 2, 4])
def test_accelerated_bond_results_match_numpy(
    monkeypatch: pytest.MonkeyPatch, components: int
) -> None:
    positions = np.array([[-0.7, 0, 0], [0.7, 0, 0]])
    xyz = np.stack(
        np.meshgrid(*(np.linspace(-2, 2, 7) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    weights = np.tile(np.exp(-(((np.arange(NSHELLS) + 0.5) / SCALE) ** 2)), (2, 1))
    samples = [
        Sample(
            np.arange(len(xyz), dtype=np.int32),
            (np.ceil(SCALE * np.linalg.norm(xyz - position, axis=1) + ZERO) - 1).astype(
                np.uint8
            ),
            xyz - position,
        )
        for position in positions
    ]
    density = weights[0, samples[0].shells] + weights[1, samples[1].shells]
    spin = (
        None
        if components == 1
        else np.repeat(weights[..., None] * 0.1, components - 1, axis=2)
    )
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(_jit, "ENABLED", enabled)
        outputs.append(
            compute_bond_orders(
                positions=positions,
                lattice_vectors=np.eye(3) * 5,
                periodic=(False, False, False),
                density=density,
                weights=weights,
                pseudodensity=density,
                samples=samples,
                populations=np.ones(2),
                voxel_volume=(2 / 3) ** 3,
                near_nucleus_indices=[sample.indices[:5] for sample in samples],
                spherical_spin=spin,
                pair_cutoff=0,
            )
        )
    expected, actual = outputs
    assert len(expected.overlaps) > 0
    assert expected.density_correction_iterations > 0
    for name in (
        "bond_orders",
        "contact_exchanges",
        "spin_polarization_terms",
        "summed_bond_orders",
        "summed_contact_exchanges",
        "corrected_density",
        "corrected_spherical_average",
        "population_residual",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert (
        actual.density_correction_iterations == expected.density_correction_iterations
    )
    for name in ("atom1", "atom2", "translations", "populations"):
        np.testing.assert_array_equal(
            getattr(actual.overlaps, name), getattr(expected.overlaps, name)
        )


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
