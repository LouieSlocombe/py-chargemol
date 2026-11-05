"""End-to-end analyses when every WFX electron is integrated analytically."""

from pathlib import Path

import numpy as np
import pytest

from py_chargemol import DensityGrid, compute_ddec, read_wfx, write_density_results
from py_chargemol.density.grid import NSHELLS, AtomicSample
from py_chargemol.density.spin import partition_spin


def _tight_wavefunction(  # noqa: PLR0913 -- Independent wavefunction cases.
    path: Path,
    *,
    exponent: float = 6,
    primitive: int = 1,
    spin: str = "Alpha",
    number: int = 1,
    occupation: float = 1,
    periodic: bool = False,
) -> Path:
    coefficient = (2 * exponent / np.pi) ** 0.75
    if primitive == 2:
        coefficient *= np.sqrt(4 * exponent)
    if periodic:
        overlap = sum(
            np.exp(-exponent * (2 * image) ** 2 / 2) for image in range(-10, 11)
        )
        coefficient /= overlap**1.5
    fields = {
        "Keywords": "GTO",
        "Number of Nuclei": "1",
        "Number of Occupied Molecular Orbitals": "1",
        "Net Charge": str(number - round(occupation)),
        "Atomic Numbers": str(number),
        "Nuclear Charges": str(number),
        "Nuclear Cartesian Coordinates": "0 0 0",
        "Number of Primitives": "1",
        "Primitive Centers": "1",
        "Primitive Exponents": str(exponent),
        "Primitive Types": str(primitive),
        "Molecular Orbital Occupation Numbers": str(occupation),
        "Molecular Orbital Spin Types": spin,
        "Molecular Orbital Primitive Coefficients": (
            f"<MO Number>\n1\n</MO Number>\n{coefficient:.16e}"
        ),
    }
    if periodic:
        fields["Number of Translation Vectors"] = "3"
        fields["Translation Vectors"] = "2 0 0\n0 2 0\n0 0 2"
    path.write_text(
        "\n".join(f"<{name}>\n{value}\n</{name}>" for name, value in fields.items())
    )
    return path


@pytest.mark.parametrize("method", ["ddec3", "ddec6"])
@pytest.mark.parametrize("exponent", [3, 6])
@pytest.mark.parametrize("primitive", [1, 2], ids=["s", "p"])
@pytest.mark.parametrize(("spin", "population"), [("Alpha", 1), ("Beta", -1)])
def test_fully_analytic_atom_retains_multipoles_and_spin(  # noqa: PLR0913, PLR0917
    tmp_path: Path,
    method: str,
    exponent: float,
    primitive: int,
    spin: str,
    population: float,
) -> None:
    grid = read_wfx(
        _tight_wavefunction(
            tmp_path / "atom.wfx", exponent=exponent, primitive=primitive, spin=spin
        ),
        grid_spacing=0.5,
    )
    np.testing.assert_array_equal(grid.density, 0)
    result = compute_ddec(grid, method=method)
    np.testing.assert_allclose(result.charges, 0, atol=1e-14)
    np.testing.assert_allclose(result.hirshfeld_charges, 0, atol=1e-14)
    np.testing.assert_allclose(result.analysis.atomic_dipoles, 0, atol=1e-14)
    quadrupole = (
        np.zeros(6)
        if primitive == 1
        else np.array([-2, 0, 0, 1, 0, 1]) / (6 * exponent)
    )
    np.testing.assert_allclose(
        result.analysis.atomic_quadrupoles[0], quadrupole, atol=1e-14
    )
    assert result.spin is not None
    assert result.spin.converged
    np.testing.assert_allclose(result.spin.populations, population, atol=1e-9)
    # Bond-density correction stops at 1e-5 electrons, after spin partitioning.
    assert (
        np.max(np.abs(result.spin.spherical_average) - result.atomic_densities) < 1e-4
    )
    assert len(result.overlaps) == 0
    assert np.all(np.isfinite(result.radial_moments))
    assert np.all(result.radial_moments > 0)
    if method == "ddec6":
        np.testing.assert_allclose(result.compute_qdr().charges, 0, atol=1e-14)


@pytest.mark.parametrize("method", ["ddec3", "ddec6"])
def test_analytic_periodic_atom_writes_bonds_and_results(
    tmp_path: Path, method: str
) -> None:
    grid = read_wfx(
        _tight_wavefunction(tmp_path / "periodic.wfx", exponent=3, periodic=True),
        grid_spacing=0.5,
    )
    result = compute_ddec(grid, method=method)
    assert result.moments is None
    np.testing.assert_allclose(result.charges, 0, atol=1e-14)
    assert result.spin is not None
    np.testing.assert_allclose(result.spin.populations, 1, atol=1e-5)
    assert len(result.overlaps) > 0
    assert np.all(np.isfinite(result.bonds.bond_orders))
    paths = write_density_results(result, tmp_path / method, qdr=method == "ddec6")
    assert paths["charges"].is_file()
    assert paths["bond_orders"].is_file()


@pytest.mark.parametrize("occupation", [1.0, 1.0005])
def test_analytic_occupancy_rounding_preserves_charge_and_spin(
    tmp_path: Path, occupation: float
) -> None:
    grid = read_wfx(
        _tight_wavefunction(tmp_path / "rounding.wfx", occupation=occupation),
        grid_spacing=0.5,
    )
    result = compute_ddec(grid)
    np.testing.assert_array_equal(result.charges, [0])
    assert result.spin is not None
    np.testing.assert_allclose(result.spin.populations, occupation, atol=1e-9)


@pytest.mark.parametrize("method", ["ddec3", "ddec6"])
def test_analytic_charged_atom_can_skip_spin_and_bonds(
    tmp_path: Path, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = read_wfx(
        _tight_wavefunction(tmp_path / "ion.wfx", number=2), grid_spacing=0.5
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("charge-only analysis must skip spin and bonds")

    for name in ("partition_spin", "compute_bond_orders"):
        monkeypatch.setattr(f"py_chargemol.density.analysis.{name}", unexpected)
    result = compute_ddec(grid, method=method, compute_bos=False, compute_spin=False)
    np.testing.assert_allclose(result.charges, [1], atol=1e-14)
    assert result.spin is None
    assert result.bond_result is None


def test_analytic_closed_shell_atom_has_no_spin(tmp_path: Path) -> None:
    grid = read_wfx(
        _tight_wavefunction(
            tmp_path / "helium.wfx", number=2, occupation=2, spin="Alpha and Beta"
        ),
        grid_spacing=0.5,
    )
    result = compute_ddec(grid)
    np.testing.assert_array_equal(result.charges, [0])
    assert result.spin is None


def test_core_only_vector_spin_retains_its_integral() -> None:
    density = np.zeros((3, 3, 3))
    core = np.ones_like(density) / 13.5
    direction = np.array([0.0, 0.6, 0.8])
    grid = DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.ones((1, 3)),
        density=density,
        origin=np.zeros(3),
        steps=np.eye(3),
        core_electrons=np.array([1]),
        core_density=core,
        spin_density=0.25 * core[..., None] * direction,
    )
    result = compute_ddec(grid)
    assert result.spin is not None
    np.testing.assert_allclose(result.spin.populations[0], 0.5 * direction, atol=1e-6)
    assert result.spin_gradients is not None


@pytest.mark.parametrize("method", ["ddec3", "ddec6"])
def test_electronless_atom_still_fails(method: str) -> None:
    grid = DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        density=np.zeros((1, 1, 1)),
        origin=np.zeros(3),
        steps=np.eye(3),
        net_charge=1,
    )
    with pytest.raises(ValueError, match="positive number of valence electrons"):
        compute_ddec(grid, method=method)


def test_spherical_spin_includes_zero_density_nodes_in_its_average() -> None:
    grid = DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        density=np.array([[[0.0, 1.0]]]),
        origin=np.zeros(3),
        steps=np.eye(3),
    )
    # Two points in one shell; the zero-density node contributes zero spin.
    sample = AtomicSample(
        indices=np.array([0, 1]),
        shells=np.array([0, 0]),
        coordinates=np.array([[0, 0, 0], [0, 0, 1]]),
        steps=grid.steps,
        offset=np.zeros(3),
    )
    spherical = np.zeros((1, NSHELLS))
    spherical[0, 0] = 0.5
    result = partition_spin(
        density=grid.density.ravel(),
        spin_density=grid.density.ravel(),
        weights=np.ones((1, NSHELLS)),
        pseudodensity=np.ones(2),
        spherical_average=spherical,
        samples=[sample],
        voxel_volume=1,
    )
    np.testing.assert_allclose(result.spherical_average, spherical)
    np.testing.assert_allclose(result.populations, [1])
