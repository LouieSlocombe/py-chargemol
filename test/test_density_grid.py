"""Regular-grid validation and atom/image geometry invariants."""

from typing import Any

import numpy as np
import pytest

from py_chargemol.density import grid as grid_module
from py_chargemol.density.grid import (
    CUTOFF,
    AtomicSample,
    DensityGrid,
    atomic_samples,
    nuclear_stencils,
    sample_index_type,
)


def _grid(**overrides: Any) -> DensityGrid:
    arguments: dict[str, Any] = {
        "atomic_numbers": np.array([1]),
        "positions": np.zeros((1, 3)),
        "density": np.ones((3, 3, 3)),
        "origin": np.zeros(3),
        "steps": np.eye(3),
    }
    arguments.update(overrides)
    return DensityGrid(**arguments)


def test_grid_copies_arrays_and_reports_cell_geometry() -> None:
    density = np.ones((2, 3, 4))
    positions = np.zeros((1, 3))
    grid = _grid(density=density, positions=positions, steps=np.diag([0.5, 1, 2]))
    density[:] = 5
    positions[:] = 9
    np.testing.assert_array_equal(grid.density, np.ones((2, 3, 4)))
    np.testing.assert_array_equal(grid.positions, np.zeros((1, 3)))
    np.testing.assert_array_equal(grid.core_electrons, [0])
    np.testing.assert_array_equal(grid.occupancy_correction, [0])
    np.testing.assert_array_equal(grid.lattice_vectors, np.diag([1, 3, 8]))
    assert grid.natoms == 1
    assert grid.pixel_volume == pytest.approx(1)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("atomic_numbers", np.array([]), "nonempty one-dimensional"),
        ("atomic_numbers", np.array([[1]]), "nonempty one-dimensional"),
        ("atomic_numbers", [[1], [1, 2]], "finite integers"),
        ("atomic_numbers", np.array([True]), "finite integers"),
        ("atomic_numbers", np.array(["H"]), "finite integers"),
        ("atomic_numbers", np.array([1.1]), "finite integers"),
        ("atomic_numbers", np.array([np.nan]), "finite integers"),
        ("atomic_numbers", np.array([0]), "lie in 1-109"),
        ("atomic_numbers", np.array([110]), "lie in 1-109"),
        ("positions", np.zeros((1, 2)), "positions.*shape"),
        ("positions", np.full((1, 3), np.inf), "positions.*finite real"),
        ("origin", np.ones(3, dtype=complex), "origin.*finite real"),
        ("steps", np.full((3, 3), "invalid"), "steps.*finite real"),
        ("steps", np.zeros((3, 3)), "nonsingular"),
        ("density", np.ones((2, 2)), "nonempty three-dimensional"),
        ("density", np.ones((0, 2, 2)), "nonempty three-dimensional"),
        ("density", np.full((3, 3, 3), np.nan), "density.*finite real"),
        ("density", -np.ones((3, 3, 3)), "total electron density.*nonnegative"),
        ("periodic", (True,), "three booleans"),
        ("periodic", (0, 0, 1), "three booleans"),
        ("net_charge", np.inf, "net_charge.*finite"),
        ("core_electrons", np.array([-1.0]), "between zero"),
        ("core_electrons", np.array([2.0]), "between zero"),
        ("core_electrons", np.array([0.5]), "integers.*missing core"),
        ("core_density", -np.ones((3, 3, 3)), "core_density.*nonnegative"),
        ("pseudodensity", np.ones((3, 3, 2)), "pseudodensity.*shape"),
        ("spin_density", np.ones((3, 3, 3, 2)), "spin_density must match"),
        ("spin_density", np.full((3, 3, 3), np.nan), "spin_density.*finite"),
        ("atomic_corrections", np.zeros((1, 10)), "atomic_corrections.*shape"),
        (
            "wavefunction_radial_moments",
            np.zeros(5),
            "wavefunction_radial_moments.*shape",
        ),
    ],
)
def test_grid_rejects_invalid_inputs(name: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _grid(**{name: value})


def test_grid_rejects_coincident_nuclei() -> None:
    with pytest.raises(ValueError, match="distinct positions"):
        _grid(atomic_numbers=np.array([1, 1]), positions=np.zeros((2, 3)))


def test_analytic_wavefunction_radial_moments_require_a_single_centre() -> None:
    with pytest.raises(ValueError, match="single atomic centre"):
        _grid(
            atomic_numbers=np.array([1, 1]),
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            wavefunction_radial_moments=np.ones(6),
        )
    moments = np.arange(6, dtype=float)
    grid = _grid(wavefunction_radial_moments=moments)
    moments[:] = 0
    np.testing.assert_array_equal(grid.wavefunction_radial_moments, np.arange(6))


@pytest.mark.parametrize("components", [False, True])
def test_grid_accepts_explicit_core_spin_and_analytic_corrections(
    components: bool,
) -> None:
    spin_shape = (3, 3, 3, 3) if components else (3, 3, 3)
    grid = _grid(
        density=-np.ones((3, 3, 3)),
        core_density=2 * np.ones((3, 3, 3)),
        core_electrons=np.array([0.5]),
        pseudodensity=np.ones((3, 3, 3)),
        spin_density=np.zeros(spin_shape),
        occupancy_correction=np.array([0.1]),
        atomic_corrections=np.zeros((1, 11)),
    )
    assert grid.spin_density is not None
    assert grid.spin_density.shape == spin_shape
    np.testing.assert_array_equal(grid.occupancy_correction, [0.1])


def _translations(sample: AtomicSample, grid: DensityGrid) -> np.ndarray:
    return np.asarray(-(sample.coordinates // np.asarray(grid.density.shape)))


def test_nonperiodic_sample_contains_each_grid_point_once() -> None:
    grid = _grid()
    (sample,) = atomic_samples(grid)
    np.testing.assert_array_equal(np.sort(sample.indices), np.arange(grid.density.size))
    np.testing.assert_array_equal(_translations(sample, grid), np.zeros((27, 3)))
    assert sample.shells.min() == 0
    assert np.linalg.norm(sample.displacements, axis=1).max() < CUTOFF


def test_periodic_sample_tracks_distinct_images_of_repeated_points() -> None:
    grid = _grid(
        density=np.ones((2, 3, 3)), steps=2 * np.eye(3), periodic=(True, False, False)
    )
    (sample,) = atomic_samples(grid)
    assert len(sample.indices) > grid.density.size
    assert len(np.unique(sample.indices)) == grid.density.size
    folded = np.column_stack(np.unravel_index(sample.indices, grid.density.shape))
    points = folded @ grid.steps + grid.origin
    translations = _translations(sample, grid)
    images = grid.positions[0] + translations @ grid.lattice_vectors
    np.testing.assert_allclose(sample.displacements, points - images)
    np.testing.assert_array_equal(translations[:, 1:], 0)
    assert len(set(zip(sample.indices, map(tuple, translations), strict=True))) == len(
        sample.indices
    )
    # Bond integration bounds rely on this lexicographic order.
    order = np.lexsort(sample.coordinates.T[::-1])
    np.testing.assert_array_equal(order, np.arange(len(order)))


def test_skew_grid_samples_are_invariant_under_cartesian_rotation() -> None:
    steps = np.array([[1.1, 0.2, 0], [0, 1.3, 0.3], [0.4, 0, 0.9]])
    position = np.array([[0.2, 0.3, 0.4]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    (first,) = atomic_samples(
        _grid(steps=steps, positions=position, periodic=(True, False, True))
    )
    (second,) = atomic_samples(
        _grid(
            steps=steps @ rotation,
            positions=position @ rotation,
            periodic=(True, False, True),
        )
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.shells, second.shells)
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_allclose(first.displacements @ rotation, second.displacements)


def test_samples_store_narrow_types_and_derive_their_displacements() -> None:
    steps = np.array([[1.1, 0.2, 0], [0, 1.3, 0.3], [0.4, 0, 0.9]])
    grid = _grid(
        density=np.ones((4, 5, 6)),
        steps=steps,
        positions=np.array([[0.7, 1.9, 0.4]]),
        origin=np.array([-0.3, 0.1, 0.2]),
        periodic=(True, False, True),
    )
    (sample,) = atomic_samples(grid)
    assert sample.indices.dtype == np.int32
    assert sample.shells.dtype == np.uint8
    assert sample.coordinates.dtype == np.int32
    # The displacements are the bits the broadcast over the box once stored.
    lower = sample.coordinates.min(axis=0)
    axes = [
        np.arange(lo, hi + 1)
        for lo, hi in zip(lower, sample.coordinates.max(axis=0), strict=True)
    ]
    x, y, z = (np.multiply.outer(s, a) for a, s in zip(axes, steps, strict=True))
    components = (
        x[:, :, None, None]
        + y[:, None, :, None]
        + z[:, None, None, :]
        + (grid.origin - grid.positions[0])[:, None, None, None]
    )
    box = (sample.coordinates - lower).T
    np.testing.assert_array_equal(
        sample.displacements, components[:, box[0], box[1], box[2]].T
    )
    some = sample.coordinates[::3]
    np.testing.assert_array_equal(sample.cartesian(some), sample.displacements[::3])


def test_sample_indices_widen_only_for_grids_beyond_int32() -> None:
    assert sample_index_type(2**31 - 1) is np.int32
    assert sample_index_type(2**31) is np.int64


def test_nuclear_stencils_respect_boundaries_and_periodic_images() -> None:
    (corner,) = nuclear_stencils(_grid())
    assert len(corner) == 27
    (periodic,) = nuclear_stencils(
        _grid(density=np.ones((2, 2, 2)), periodic=(True, True, True))
    )
    assert len(periodic) == 125
    assert len(np.unique(periodic)) == 8
    (negative_half,) = nuclear_stencils(_grid(positions=np.array([[-0.5, 0, 0]])))
    assert len(negative_half) == 18


def test_atom_outside_molecular_grid_has_no_sampled_points() -> None:
    (sample,) = atomic_samples(_grid(positions=np.full((1, 3), 100.0)))
    assert sample.indices.size == 0
    assert sample.indices.dtype == np.int32
    assert sample.shells.dtype == np.uint8
    assert sample.coordinates.shape == (0, 3)
    assert sample.coordinates.dtype == np.int32


def test_samples_do_not_depend_on_the_slabs_they_are_measured_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(
        atomic_numbers=np.array([1, 8]),
        density=np.ones((4, 5, 6)),
        steps=np.array([[1.1, 0.2, 0], [0, 1.3, 0.3], [0.4, 0, 0.9]]),
        positions=np.array([[0.7, 1.9, 0.4], [2.1, 0.3, 3.3]]),
        origin=np.array([-0.3, 0.1, 0.2]),
        periodic=(True, False, True),
    )
    whole = atomic_samples(grid)
    # One first-axis plane per slab.
    monkeypatch.setattr(grid_module, "_SLAB_POINTS", 1)
    planes = atomic_samples(grid)
    for expected, actual in zip(whole, planes, strict=True):
        for name in ("indices", "shells", "coordinates", "offset"):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
            assert getattr(actual, name).dtype == getattr(expected, name).dtype
