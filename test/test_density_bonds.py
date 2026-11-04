"""Independent quadrature and invariance checks for density-derived bond orders."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import pytest

from py_chargemol.ddec6 import FloatArray, IntArray
from py_chargemol.density.analysis import compute_ddec6
from py_chargemol.density.bonds import (
    BondOrderResult,
    _overlap_estimate_grid,
    compute_bond_orders,
)
from py_chargemol.density.grid import NSHELLS, SCALE
from py_chargemol.density.spin import _lookup_tables, partition_spin
from py_chargemol.density.workflow import read_density
from py_chargemol.units import BOHR_PER_ANGSTROM


def test_ozone_triplet_reproduces_original_spin_and_bond_analysis() -> None:
    directory = Path(__file__).parent / "density_examples"
    expected = json.loads((directory / "ozone_triplet_PW91.json").read_text())
    result = compute_ddec6(read_density(directory / "ozone_triplet_PW91.wfx"))
    assert result.spin is not None
    assert result.spin.converged
    for actual, key in (
        (result.charges, "net_charges"),
        (result.analysis.atomic_dipoles, "atomic_dipoles"),
        (result.analysis.atomic_quadrupoles, "atomic_quadrupoles"),
        (result.spin.populations, "atomic_spins"),
        (result.bonds.summed_bond_orders, "summed_bond_orders"),
    ):
        np.testing.assert_allclose(actual, expected[key], rtol=0, atol=5e-7)
    np.testing.assert_array_equal(result.overlaps.atom1, expected["atom1"])
    np.testing.assert_array_equal(result.overlaps.atom2, expected["atom2"])
    np.testing.assert_array_equal(result.overlaps.translations, 0)
    np.testing.assert_allclose(
        result.overlaps.populations, expected["overlap_populations"], rtol=0, atol=6e-11
    )
    np.testing.assert_allclose(
        result.bonds.bond_orders, expected["bond_orders"], rtol=0, atol=5e-5
    )
    polarization = np.sqrt(
        result.bonds.spin_polarization_terms / result.bonds.contact_exchanges - 1
    )
    np.testing.assert_allclose(
        polarization, expected["bond_spin_polarizations"], rtol=0, atol=5e-5
    )


@dataclass(frozen=True)
class Sample:
    indices: IntArray
    shells: IntArray
    displacements: FloatArray


class Inputs(TypedDict):
    positions: FloatArray
    lattice_vectors: FloatArray
    periodic: tuple[bool, bool, bool]
    density: FloatArray
    weights: FloatArray
    pseudodensity: FloatArray
    samples: list[Sample]
    populations: FloatArray
    voxel_volume: float
    near_nucleus_indices: NotRequired[list[IntArray]]
    spherical_spin: NotRequired[FloatArray]
    pair_cutoff: NotRequired[float]
    grid_steps: NotRequired[FloatArray]
    grid_origin: NotRequired[FloatArray]


@pytest.fixture
def molecule() -> Inputs:
    positions = np.asarray([[-0.7, 0, 0], [0.7, 0, 0]])
    grid = np.stack(
        np.meshgrid(*(np.arange(-4.0, 4.1, 0.5) for _ in range(3)), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    radius = 5 * BOHR_PER_ANGSTROM
    shells_r = (np.arange(100) + 0.5) * radius / 100
    weights = np.tile(np.exp(-(shells_r**2)), (2, 1))
    samples: list[Sample] = []
    density = np.zeros(len(grid))
    volume = 0.5**3
    for atom, position in enumerate(positions):
        displacement = grid - position
        shells = (
            np.ceil(100 / radius * np.linalg.norm(displacement, axis=1) + 1e-10) - 1
        ).astype(np.int64)
        indices = np.arange(len(grid), dtype=np.int64)
        sample = Sample(indices, shells, displacement)
        samples.append(sample)
        weights[atom] /= np.sum(weights[atom, shells]) * volume
        density += weights[atom, shells]
    return {
        "positions": positions,
        "lattice_vectors": np.eye(3) * 8.5,
        "periodic": (False, False, False),
        "density": density,
        "weights": weights,
        "pseudodensity": density.copy(),
        "samples": samples,
        "populations": np.ones(2),
        "voxel_volume": volume,
    }


def expected_pair(
    inputs: Inputs, spin_directions: FloatArray
) -> tuple[float, float, float]:
    assigned = np.asarray(
        [
            inputs["weights"][atom, sample.shells]
            for atom, sample in enumerate(inputs["samples"])
        ]
    )
    spin = assigned[..., np.newaxis] * spin_directions[:, np.newaxis, :]
    density = inputs["density"]
    dot_product = assigned[0] * assigned[1] + np.sum(spin[0] * spin[1], axis=1)
    total_square = density**2 + np.sum((spin[0] + spin[1]) ** 2, axis=1)
    factor = 2 * inputs["voxel_volume"] * density
    contact = float(np.sum(factor * dot_product / total_square))
    second_order = float((10 / 3) * np.sum(factor * (dot_product / total_square) ** 2))
    expansion = min(second_order + contact**2 / 6, contact)
    bond_order = contact + min(expansion, 1 - contact / 2)
    polarization = float(np.sum(factor * dot_product / density**2))
    return contact, bond_order, polarization


def check_pair(result: BondOrderResult, expected: tuple[float, float, float]) -> None:
    contact, bond_order, polarization = expected
    np.testing.assert_array_equal(result.overlaps.atom1, [0, 1])
    np.testing.assert_array_equal(result.overlaps.atom2, [1, 0])
    np.testing.assert_array_equal(result.overlaps.translations, np.zeros((2, 3)))
    np.testing.assert_allclose(result.contact_exchanges, contact, atol=1e-14)
    np.testing.assert_allclose(result.bond_orders, bond_order, atol=1e-14)
    np.testing.assert_allclose(result.spin_polarization_terms, polarization, atol=1e-14)
    np.testing.assert_allclose(result.summed_contact_exchanges, contact, atol=1e-14)
    np.testing.assert_allclose(result.summed_bond_orders, bond_order, atol=1e-14)


def test_unpolarized_two_atom_quadrature(molecule: Inputs) -> None:
    result = compute_bond_orders(**molecule)
    expected = expected_pair(molecule, np.zeros((2, 3)))
    check_pair(result, expected)
    np.testing.assert_allclose(result.overlaps.populations, expected[0], atol=1e-14)
    # The original algorithm excludes weighting densities below 1e-10.
    np.testing.assert_allclose(result.population_residual, 0, atol=1e-9)
    np.testing.assert_array_equal(result.corrected_density, molecule["density"])
    assert result.density_correction_iterations == 0


@pytest.mark.parametrize(
    "directions",
    [
        np.asarray([[0.5], [0.5]]),
        np.asarray([[0.5], [-0.5]]),
        np.asarray([[0.5, 0, 0], [0, 0.3, 0.4]]),
    ],
)
def test_spin_exchange_quadrature(molecule: Inputs, directions: FloatArray) -> None:
    molecule["spherical_spin"] = molecule["weights"][..., None] * directions[:, None, :]
    result = compute_bond_orders(**molecule)
    check_pair(result, expected_pair(molecule, directions))
    unpolarized_contact = expected_pair(molecule, np.zeros((2, 3)))[0]
    np.testing.assert_allclose(
        result.overlaps.populations, unpolarized_contact, atol=1e-14
    )


def test_collinear_flat_spin_input(molecule: Inputs) -> None:
    molecule["spherical_spin"] = molecule["weights"] * 0.5
    check_pair(
        compute_bond_orders(**molecule), expected_pair(molecule, np.full((2, 1), 0.5))
    )


def test_density_correction_recovers_atomic_populations(molecule: Inputs) -> None:
    molecule["density"] = molecule["density"] * 0.97
    molecule["near_nucleus_indices"] = [
        sample.indices[np.linalg.norm(sample.displacements, axis=1) <= 1.0]
        for sample in molecule["samples"]
    ]
    original = molecule["density"].copy()
    result = compute_bond_orders(**molecule)
    assert result.density_correction_iterations > 1
    np.testing.assert_allclose(result.population_residual, 0, atol=1e-5)
    np.testing.assert_array_equal(molecule["density"], original)
    assert result.corrected_density.sum() > original.sum()


def test_screening_and_single_atom_have_no_pairs(molecule: Inputs) -> None:
    molecule["pair_cutoff"] = 2.0
    result = compute_bond_orders(**molecule)
    assert len(result.overlaps) == 0
    assert result.overlaps.translations.shape == (0, 3)
    np.testing.assert_array_equal(
        result.summed_bond_orders, result.summed_contact_exchanges
    )
    molecule["positions"] = molecule["positions"][:1]
    molecule["weights"] = molecule["weights"][:1]
    molecule["samples"] = molecule["samples"][:1]
    molecule["populations"] = molecule["populations"][:1]
    result = compute_bond_orders(**molecule)
    assert len(result.overlaps) == 0
    np.testing.assert_allclose(result.summed_bond_orders, 0, atol=1e-14)


@pytest.mark.parametrize("volume", [0.0, -1.0, np.nan, np.inf])
def test_invalid_voxel_volume(molecule: Inputs, volume: float) -> None:
    molecule["voxel_volume"] = volume
    with pytest.raises(ValueError, match="voxel_volume"):
        compute_bond_orders(**molecule)


@pytest.mark.parametrize("cutoff", [-1.0, np.nan, np.inf])
def test_invalid_pair_cutoff(molecule: Inputs, cutoff: float) -> None:
    molecule["pair_cutoff"] = cutoff
    with pytest.raises(ValueError, match="pair_cutoff"):
        compute_bond_orders(**molecule)


@pytest.mark.parametrize("shape", [(2, 99), (2, 100, 2), (1, 100, 3)])
def test_invalid_spin_shape(molecule: Inputs, shape: tuple[int, ...]) -> None:
    molecule["spherical_spin"] = np.zeros(shape)
    with pytest.raises(ValueError, match="spherical_spin"):
        compute_bond_orders(**molecule)


def test_nonfinite_spin(molecule: Inputs) -> None:
    molecule["spherical_spin"] = np.full((2, 100), np.nan)
    with pytest.raises(ValueError, match="finite"):
        compute_bond_orders(**molecule)


def test_periodic_self_image_pair() -> None:
    radius = 5 * BOHR_PER_ANGSTROM
    lattice = np.diag([3.0, 8.5, 8.5])
    grid = np.stack(
        np.meshgrid(
            np.arange(-1.5, 1.5, 0.5),
            np.arange(-4, 4.1, 0.5),
            np.arange(-4, 4.1, 0.5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    displacements = np.concatenate(
        [grid + np.asarray([3 * shift, 0, 0]) for shift in range(-3, 4)]
    )
    points = np.tile(np.arange(len(grid), dtype=np.int64), 7)
    shells = (
        np.ceil(100 / radius * np.linalg.norm(displacements, axis=1) + 1e-10) - 1
    ).astype(np.int64)
    active = shells < 100
    sample = Sample(points[active], shells[active], displacements[active])
    weights = np.exp(-(((np.arange(100) + 0.5) * radius / 100) ** 2)).reshape(1, 100)
    weights /= np.sum(weights[0, sample.shells]) * 0.5**3
    density = np.bincount(
        sample.indices, weights=weights[0, sample.shells], minlength=len(grid)
    )
    result = compute_bond_orders(
        positions=np.zeros((1, 3)),
        lattice_vectors=lattice,
        periodic=(True, False, False),
        density=density,
        weights=weights,
        pseudodensity=density,
        samples=[sample],
        populations=np.ones(1),
        voxel_volume=0.5**3,
    )
    np.testing.assert_array_equal(result.overlaps.atom1, 0)
    np.testing.assert_array_equal(result.overlaps.atom2, 0)
    np.testing.assert_array_equal(result.overlaps.translations, [[1, 0, 0], [-1, 0, 0]])
    distance = np.linalg.norm(sample.displacements - [3.0, 0, 0], axis=1)
    other_shell = (np.ceil(100 / radius * distance + 1e-10) - 1).astype(np.int64)
    active = (other_shell < 100) & (density[sample.indices] > 1e-10)
    expected = (
        2
        * 0.5**3
        * np.sum(
            weights[0, sample.shells[active]]
            * weights[0, other_shell[active]]
            / density[sample.indices[active]]
        )
    )
    np.testing.assert_allclose(result.overlaps.populations, expected, atol=1e-14)
    np.testing.assert_allclose(result.contact_exchanges, expected, atol=1e-14)
    assert result.summed_bond_orders[0] >= result.bond_orders.sum()


@pytest.mark.parametrize("separation", [1, 2, 37, 99, 100, 101, 150, 199, 200])
def test_overlap_estimate_grid_matches_the_original_loops(separation: int) -> None:
    # The screening quadrature as first written, one point at a time.
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
    expected = (
        np.asarray(radial_i, dtype=np.int64),
        np.asarray(radial_j, dtype=np.int64),
        np.asarray(measures, dtype=np.float64),
    )
    for actual, reference in zip(
        _overlap_estimate_grid(separation), expected, strict=True
    ):
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference)


def test_spin_lookup_against_original_fortran() -> None:
    # Recorded from unmodified module_spin_functions.f08, September 2017.
    selected = np.asarray([1, 10, 11, 100, 1000, 5000, 9999, 10000])
    expected = np.asarray(
        [
            [0.00041887902047863906, 0.00015000000000000001],
            [0.004188791254969183, 0.0014999993353256864],
            [0.004607670116526535, 0.0016499991137788027],
            [0.04188873984120541, 0.014999325025755222],
            [0.4197203889856269, 0.14932759254828618],
            [2.2121937286895648, 0.6729707477798602],
            [6.278273413179587, 0.9999873959423646],
            [6.283185307179586, 1.0],
        ]
    )
    _, xi, inverse = _lookup_tables()
    np.testing.assert_allclose(xi[selected], expected[:, 0], atol=1e-14)
    np.testing.assert_allclose(inverse[selected], expected[:, 1], atol=5e-11)


@pytest.mark.parametrize("polarization", [0.0, -0.5, 0.5, 1.0])
def test_uniform_spin_population(molecule: Inputs, polarization: float) -> None:
    spherical_density = compute_bond_orders(**molecule).corrected_spherical_average
    result = partition_spin(
        density=molecule["density"],
        spin_density=polarization * molecule["density"],
        weights=molecule["weights"],
        pseudodensity=molecule["pseudodensity"],
        spherical_average=spherical_density,
        samples=molecule["samples"],
        voxel_volume=molecule["voxel_volume"],
    )
    assert result.converged
    assert result.iterations == (2 if polarization == 0 else 7)
    assert result.populations.shape == (2,)
    assert result.spherical_average.shape == (2, 100)
    np.testing.assert_allclose(result.populations, polarization, atol=2e-8)
    np.testing.assert_allclose(
        result.spherical_average, spherical_density * polarization, atol=2e-8
    )


def test_spin_scalar_and_rotated_vector_agree(molecule: Inputs) -> None:
    spherical_density = compute_bond_orders(**molecule).corrected_spherical_average
    local = np.asarray(
        [
            molecule["weights"][atom, sample.shells]
            for atom, sample in enumerate(molecule["samples"])
        ]
    )
    scalar_density = 0.6 * local[0] - 0.3 * local[1]
    direction = np.asarray([0.36, 0.48, 0.8])
    results = [
        partition_spin(
            density=molecule["density"],
            spin_density=spin,
            weights=molecule["weights"],
            pseudodensity=molecule["pseudodensity"],
            spherical_average=spherical_density,
            samples=molecule["samples"],
            voxel_volume=molecule["voxel_volume"],
        )
        for spin in [scalar_density, scalar_density[:, None] * direction]
    ]
    scalar, vector = results
    assert scalar.converged
    assert vector.converged
    np.testing.assert_allclose(
        vector.populations, scalar.populations[:, None] * direction, atol=1e-9
    )
    np.testing.assert_allclose(
        vector.spherical_average,
        scalar.spherical_average[..., None] * direction,
        atol=1e-9,
    )
    np.testing.assert_allclose(np.sum(scalar.populations), 0.3, atol=5e-5)


def test_spin_occupancy_and_iteration_limit(molecule: Inputs) -> None:
    spherical_density = compute_bond_orders(**molecule).corrected_spherical_average
    result = partition_spin(
        density=molecule["density"],
        spin_density=0.5 * molecule["density"],
        weights=molecule["weights"],
        pseudodensity=molecule["pseudodensity"],
        spherical_average=spherical_density,
        samples=molecule["samples"],
        voxel_volume=molecule["voxel_volume"],
        occupancy_correction=np.asarray([0.1, -0.05]),
        max_iterations=1,
    )
    assert not result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(result.populations, [0.6, 0.45], atol=1e-9)


def test_bond_input_shape_and_geometry_validation(molecule: Inputs) -> None:
    original = molecule["weights"]
    molecule["weights"] = original[:, :99]
    with pytest.raises(ValueError, match="weights"):
        compute_bond_orders(**molecule)
    molecule["weights"] = original
    molecule["near_nucleus_indices"] = []
    with pytest.raises(ValueError, match="one stencil per atom"):
        compute_bond_orders(**molecule)
    del molecule["near_nucleus_indices"]
    molecule["grid_steps"] = np.eye(3) * 0.5
    with pytest.raises(ValueError, match="supplied together"):
        compute_bond_orders(**molecule)
    molecule["grid_origin"] = np.full(3, -4.0)
    result = compute_bond_orders(**molecule)
    check_pair(result, expected_pair(molecule, np.zeros((2, 3))))


def test_no_nuclear_density_reports_uncorrected_population() -> None:
    density = np.full(1, 1e-20)
    result = compute_bond_orders(
        positions=np.zeros((1, 3)),
        lattice_vectors=np.eye(3),
        periodic=(False, False, False),
        density=density,
        weights=np.ones((1, 100)),
        pseudodensity=np.ones(1),
        samples=[
            Sample(
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros((1, 3)),
            )
        ],
        populations=np.ones(1),
        voxel_volume=1.0,
        near_nucleus_indices=[np.zeros(1, dtype=np.int64)],
    )
    np.testing.assert_array_equal(result.corrected_density, density)
    np.testing.assert_allclose(result.population_residual, 1.0)
    assert result.density_correction_iterations == 40


class SpinInputs(TypedDict):
    density: FloatArray
    spin_density: FloatArray
    weights: FloatArray
    pseudodensity: FloatArray
    spherical_average: FloatArray
    samples: list[Sample]
    voxel_volume: float
    occupancy_correction: NotRequired[FloatArray]
    max_iterations: NotRequired[int]
    tolerance: NotRequired[float]


def test_spin_invalid_inputs_are_rejected(molecule: Inputs) -> None:
    arguments: SpinInputs = {
        "density": molecule["density"],
        "spin_density": np.zeros((1, 2)),
        "weights": molecule["weights"],
        "pseudodensity": molecule["pseudodensity"],
        "spherical_average": compute_bond_orders(
            **molecule
        ).corrected_spherical_average,
        "samples": molecule["samples"],
        "voxel_volume": molecule["voxel_volume"],
    }
    with pytest.raises(ValueError, match="spin_density must have shape"):
        partition_spin(**arguments)
    arguments["spin_density"] = molecule["density"] * np.nan
    with pytest.raises(ValueError, match="finite values"):
        partition_spin(**arguments)
    arguments["spin_density"] = molecule["density"] * 0.5
    arguments["max_iterations"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        partition_spin(**arguments)
    del arguments["max_iterations"]
    arguments["tolerance"] = np.nan
    with pytest.raises(ValueError, match="tolerance"):
        partition_spin(**arguments)
    del arguments["tolerance"]
    arguments["voxel_volume"] = 0
    with pytest.raises(ValueError, match="voxel_volume"):
        partition_spin(**arguments)
    arguments["voxel_volume"] = molecule["voxel_volume"]
    arguments["spherical_average"] = np.zeros((2, 99))
    with pytest.raises(ValueError, match="spherical_average"):
        partition_spin(**arguments)
    arguments["spherical_average"] = compute_bond_orders(
        **molecule
    ).corrected_spherical_average
    arguments["occupancy_correction"] = np.zeros((1, 3))
    with pytest.raises(ValueError, match="occupancy_correction"):
        partition_spin(**arguments)
