"""Gaussian WFX density quadrature and analytic core correction tests."""

import re
from pathlib import Path

import numpy as np
import pytest

from py_chargemol.density.wfx import read_wfx


def wavefunction(
    *,
    alpha: float = 0.5,
    primitive_type: int = 1,
    spin: str = "Alpha",
    periodic: int = 0,
) -> str:
    coefficient = (2 * alpha / np.pi) ** 0.75
    if primitive_type == 2:
        coefficient *= np.sqrt(4 * alpha)
    if periodic:
        overlap = sum(np.exp(-alpha * (6 * image) ** 2 / 2) for image in range(-10, 11))
        coefficient /= overlap ** (periodic / 2)
    fields = {
        "Keywords": "GTO",
        "Number of Nuclei": "1",
        "Number of Occupied Molecular Orbitals": "1",
        "Net Charge": "0",
        "Atomic Numbers": "1",
        "Nuclear Charges": "1",
        "Nuclear Cartesian Coordinates": "0 0 0",
        "Number of Primitives": "1",
        "Primitive Centers": "1",
        "Primitive Exponents": str(alpha),
        "Primitive Types": str(primitive_type),
        "Molecular Orbital Occupation Numbers": "1",
        "Molecular Orbital Spin Types": spin,
        "Molecular Orbital Primitive Coefficients": (
            f"<MO Number>\n1\n</MO Number>\n{coefficient:.16e}"
        ),
    }
    if periodic:
        fields["Number of Translation Vectors"] = str(periodic)
        fields["Translation Vectors"] = "\n".join(
            ["6 0 0", "0 6 0", "0 0 6"][:periodic]
        )
    return "\n".join(f"<{name}>\n{value}\n</{name}>" for name, value in fields.items())


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_numeric_gaussian_integrates_to_orbital_occupation(tmp_path: Path) -> None:
    grid = read_wfx(write(tmp_path / "hydrogen.wfx", wavefunction()), grid_spacing=0.5)
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(1, abs=1e-9)
    np.testing.assert_array_equal(grid.core_electrons, [0])
    assert grid.spin_density is not None
    np.testing.assert_array_equal(grid.density, grid.spin_density)
    assert grid.atomic_corrections is not None
    np.testing.assert_array_equal(grid.atomic_corrections, np.zeros((1, 11)))
    assert grid.wavefunction_radial_moments is not None
    np.testing.assert_allclose(
        grid.wavefunction_radial_moments,
        [2 / np.sqrt(np.pi), 1, 2 / np.sqrt(np.pi), 1.5, 4 / np.sqrt(np.pi), 3.75],
    )


def test_tight_gaussian_is_integrated_analytically(tmp_path: Path) -> None:
    grid = read_wfx(
        write(tmp_path / "core.wfx", wavefunction(alpha=3)), grid_spacing=0.5
    )
    assert grid.core_electrons is not None
    np.testing.assert_allclose(grid.core_electrons, [1])
    np.testing.assert_array_equal(grid.density, 0)
    assert grid.core_density is not None
    assert grid.core_density.max() > 0
    assert grid.spin_density is not None
    np.testing.assert_array_equal(grid.spin_density, grid.core_density)
    assert grid.atomic_corrections is not None
    assert (
        grid.atomic_corrections[0, 1] + grid.spin_density.sum() * grid.pixel_volume
    ) == pytest.approx(1)
    np.testing.assert_array_equal(grid.atomic_corrections[0, 2:5], 0)
    np.testing.assert_allclose(grid.atomic_corrections[0, 5:8], [-1 / 12] * 3)


def test_p_orbital_analytic_multipoles_and_beta_spin(tmp_path: Path) -> None:
    grid = read_wfx(
        write(tmp_path / "p.wfx", wavefunction(alpha=3, primitive_type=2, spin="Beta")),
        grid_spacing=0.5,
    )
    assert grid.core_electrons is not None
    np.testing.assert_allclose(grid.core_electrons, [1])
    assert grid.spin_density is not None
    assert grid.core_density is not None
    np.testing.assert_array_equal(grid.spin_density, -grid.core_density)
    assert grid.atomic_corrections is not None
    assert (
        grid.atomic_corrections[0, 1] + grid.spin_density.sum() * grid.pixel_volume
    ) == pytest.approx(-1)
    np.testing.assert_allclose(
        grid.atomic_corrections[0, 5:8], [-1 / 4, -1 / 12, -1 / 12]
    )
    assert grid.wavefunction_radial_moments is not None
    np.testing.assert_allclose(
        grid.wavefunction_radial_moments,
        [
            4 * np.sqrt(6) / (3 * np.sqrt(np.pi)),
            1,
            8 / (3 * np.sqrt(6 * np.pi)),
            5 / 12,
            8 / (np.sqrt(np.pi) * 6**1.5),
            35 / 144,
        ],
    )


@pytest.mark.parametrize("periodic", [1, 2, 3])
def test_periodic_gaussian_images_preserve_electron_count(
    tmp_path: Path, periodic: int
) -> None:
    grid = read_wfx(
        write(tmp_path / "periodic.wfx", wavefunction(periodic=periodic)),
        grid_spacing=0.5,
    )
    assert grid.periodic == (True, periodic > 1, periodic > 2)
    np.testing.assert_allclose(
        grid.lattice_vectors[:periodic], np.eye(3)[:periodic] * 6
    )
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(1, abs=1e-8)


@pytest.mark.parametrize(
    "vectors",
    [
        ["8 0 0", f"4 {4 * np.sqrt(3):.16f} 0"],
        [f"{8 / np.sqrt(2):.16f} {8 / np.sqrt(2):.16f} 0"],
    ],
)
@pytest.mark.parametrize("alpha", [1.0, 3.0])
def test_oblique_cells_preserve_electron_count(
    tmp_path: Path, vectors: list[str], alpha: float
) -> None:
    # Voxels off the Cartesian axes take the general, point-by-point quadrature.
    text = wavefunction(alpha=alpha) + (
        f"\n<Number of Translation Vectors>\n{len(vectors)}\n"
        "</Number of Translation Vectors>\n"
        "<Translation Vectors>\n" + "\n".join(vectors) + "\n</Translation Vectors>"
    )
    grid = read_wfx(write(tmp_path / "oblique.wfx", text), grid_spacing=0.5)
    assert np.any(grid.steps != np.diag(np.diag(grid.steps)))
    assert grid.core_electrons is not None
    electrons = grid.density.sum() * grid.pixel_volume + grid.core_electrons.sum()
    assert electrons == pytest.approx(1, abs=1e-9)
    np.testing.assert_allclose(grid.core_electrons, [1.0 if alpha > 2.5 else 0.0])


@pytest.mark.parametrize("spacing", [0, -1, float("nan"), float("inf")])
def test_invalid_spacing(tmp_path: Path, spacing: float) -> None:
    with pytest.raises(ValueError, match="grid_spacing"):
        read_wfx(tmp_path / "unused.wfx", grid_spacing=spacing)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("<Keywords>\nGTO", "<Keywords>\nSTO", "only Gaussian"),
        ("<Number of Nuclei>\n1", "<Number of Nuclei>\n0", "counts must be positive"),
        ("<Primitive Centers>\n1", "<Primitive Centers>\n2", "invalid primitive"),
        ("<Primitive Types>\n1", "<Primitive Types>\n57", "invalid primitive"),
        ("<Primitive Types>\n1", "<Primitive Types>\n1.5", "must contain integers"),
        ("<Primitive Exponents>\n0.5", "<Primitive Exponents>\nnan", "finite values"),
        (
            "<Primitive Exponents>\n0.5",
            "<Primitive Exponents>\n-1",
            "invalid primitive",
        ),
        (
            "<Molecular Orbital Spin Types>\nAlpha",
            "<Molecular Orbital Spin Types>\nunknown",
            "spin types",
        ),
        (
            "<Molecular Orbital Occupation Numbers>\n1",
            "<Molecular Orbital Occupation Numbers>\n2",
            "occupations do not match",
        ),
        ("<MO Number>\n1", "<MO Number>\n2", "coefficient block"),
        ("</MO Number>\n", "</MO Number>\n0 ", "invalid primitive coefficients"),
    ],
)
def test_invalid_wavefunctions(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = write(tmp_path / "invalid.wfx", wavefunction().replace(old, new))
    with pytest.raises(ValueError, match=message):
        read_wfx(path, grid_spacing=0.5)


def test_missing_required_section(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing WFX section"):
        read_wfx(write(tmp_path / "empty.wfx", ""))


def test_rejects_unnormalized_orbitals(tmp_path: Path) -> None:
    text = wavefunction().replace("</MO Number>\n", "</MO Number>\n2")
    with pytest.raises(ValueError, match="unnormalized primitive coefficients"):
        read_wfx(write(tmp_path / "unnormalized.wfx", text), grid_spacing=0.5)


@pytest.mark.parametrize("edf", [False, True])
def test_ecp_core_reconstruction_and_edf(tmp_path: Path, edf: bool) -> None:
    text = wavefunction(spin="Alpha and Beta").replace(
        "<Atomic Numbers>\n1", "<Atomic Numbers>\n3"
    )
    if edf:
        fields = {
            "Number of EDF Primitives": "1",
            "EDF Primitive Centers": "1",
            "EDF Primitive Types": "1",
            "EDF Primitive Exponents": "3",
            "EDF Primitive Coefficients": str(2 * (3 / np.pi) ** 1.5),
        }
        text += "\n" + "\n".join(
            f"<{name}>\n{value}\n</{name}>" for name, value in fields.items()
        )
    grid = read_wfx(write(tmp_path / "lithium.wfx", text), grid_spacing=0.5)
    assert grid.core_electrons is not None
    np.testing.assert_array_equal(grid.core_electrons, [2])
    assert grid.core_density is not None
    assert grid.core_density.max() > 0
    assert grid.spin_density is None
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(1, abs=1e-9)
    assert grid.wavefunction_radial_moments is not None
    # These analytic moments include the one explicit electron, excluding ECP/EDF.
    assert grid.wavefunction_radial_moments[1] == pytest.approx(1)
    assert grid.wavefunction_radial_moments[3] == pytest.approx(1.5)


@pytest.mark.parametrize("ghost_first", [False, True])
def test_ghost_basis_centers_do_not_create_atoms(
    tmp_path: Path, ghost_first: bool
) -> None:
    text = wavefunction(alpha=6)
    coefficient = (12 / np.pi) ** 0.75 / 2
    replacements = {
        "Number of Nuclei": "2",
        "Number of Primitives": "2",
        "Atomic Numbers": "0 1" if ghost_first else "1 0",
        "Nuclear Charges": "0 1" if ghost_first else "1 0",
        "Nuclear Cartesian Coordinates": "0 0 0\n0 0 0",
        "Primitive Centers": "1 2",
        "Primitive Exponents": "6 6",
        "Primitive Types": "1 1",
        "Molecular Orbital Primitive Coefficients": (
            f"<MO Number>\n1\n</MO Number>\n{coefficient:.16e} {coefficient:.16e}"
        ),
    }

    for name, value in replacements.items():
        text = re.sub(
            rf"<{name}>.*?</{name}>",
            f"<{name}>\n{value}\n</{name}>",
            text,
            flags=re.DOTALL,
        )
    grid = read_wfx(write(tmp_path / "ghost.wfx", text), grid_spacing=0.3)
    assert grid.natoms == 1
    assert grid.wavefunction_radial_moments is None
    assert grid.core_electrons is not None
    np.testing.assert_allclose(grid.core_electrons, [0.75])
    assert grid.atomic_corrections is not None
    assert grid.spin_density is not None
    assert (
        grid.atomic_corrections[0, 1] + grid.spin_density.sum() * grid.pixel_volume
    ) == pytest.approx(1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            wavefunction(periodic=1).replace(
                "Translation Vectors>\n1", "Translation Vectors>\n4"
            ),
            "translation vectors must be 0-3",
        ),
        (wavefunction(periodic=1).replace("6 0 0", "0 0 0"), "must be nonzero"),
        (wavefunction(periodic=2).replace("0 6 0", "6 0 0"), "linearly independent"),
        (wavefunction(periodic=3).replace("0 6 0", "6 0 0"), "linearly independent"),
    ],
)
def test_invalid_periodic_wavefunctions(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_wfx(write(tmp_path / "invalid.wfx", text), grid_spacing=0.5)


def test_widely_separated_periodic_images_are_negligible(tmp_path: Path) -> None:
    text = wavefunction(alpha=3, periodic=1).replace("6 0 0", "100 0 0")
    grid = read_wfx(write(tmp_path / "distant.wfx", text), grid_spacing=0.5)
    assert grid.core_electrons is not None
    np.testing.assert_allclose(grid.core_electrons, [1])


def test_mixed_cartesian_primitives_preserve_cross_terms(tmp_path: Path) -> None:
    text = wavefunction()
    coefficient = (1 / np.pi) ** 0.75
    replacements = {
        "Number of Primitives": "2",
        "Primitive Centers": "1 1",
        "Primitive Exponents": "0.5 0.5",
        "Primitive Types": "1 2",
        "Molecular Orbital Primitive Coefficients": (
            "<MO Number>\n1\n</MO Number>\n"
            f"{coefficient / np.sqrt(2):.16e} {coefficient:.16e}"
        ),
    }
    for name, value in replacements.items():
        text = re.sub(
            rf"<{name}>.*?</{name}>",
            f"<{name}>\n{value}\n</{name}>",
            text,
            flags=re.DOTALL,
        )
    grid = read_wfx(write(tmp_path / "mixed.wfx", text), grid_spacing=0.3)
    x = grid.origin[0] + np.arange(grid.density.shape[0]) * grid.steps[0, 0]
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(1, abs=1e-9)
    assert np.sum(grid.density * x[:, None, None]) * grid.pixel_volume == pytest.approx(
        1 / np.sqrt(2), abs=1e-9
    )
    assert grid.wavefunction_radial_moments is not None
    np.testing.assert_allclose(
        grid.wavefunction_radial_moments[[1, 3, 5]], [1, 2, 6.25]
    )


def test_high_angular_momentum_skips_isolated_atom_radial_diagnostic(
    tmp_path: Path,
) -> None:
    coefficient = (1 / np.pi) ** 0.75
    text = wavefunction(primitive_type=36).replace(
        f"{coefficient:.16e}", f"{coefficient * np.sqrt(32 / 945):.16e}"
    )
    grid = read_wfx(write(tmp_path / "h_orbital.wfx", text), grid_spacing=0.3)
    assert grid.wavefunction_radial_moments is None


def test_radial_diagnostic_includes_same_parity_cross_terms(tmp_path: Path) -> None:
    coefficient = (1 / np.pi) ** 0.75 / 2
    text = wavefunction()
    replacements = {
        "Number of Primitives": "2",
        "Primitive Centers": "1 1",
        "Primitive Exponents": "0.5 0.5",
        "Primitive Types": "1 1",
        "Molecular Orbital Primitive Coefficients": (
            f"<MO Number>\n1\n</MO Number>\n{coefficient:.16e} {coefficient:.16e}"
        ),
    }
    for name, value in replacements.items():
        text = re.sub(
            rf"<{name}>.*?</{name}>",
            f"<{name}>\n{value}\n</{name}>",
            text,
            flags=re.DOTALL,
        )
    grid = read_wfx(write(tmp_path / "duplicated.wfx", text), grid_spacing=0.3)
    assert grid.wavefunction_radial_moments is not None
    np.testing.assert_allclose(
        grid.wavefunction_radial_moments[[1, 3, 5]], [1, 1.5, 3.75]
    )


def test_diffuse_gaussian_uses_coarse_grid_with_conserved_interpolation(
    tmp_path: Path,
) -> None:
    grid = read_wfx(
        write(tmp_path / "diffuse.wfx", wavefunction(alpha=0.005)), grid_spacing=0.3
    )
    # A diffuse Gaussian extends beyond the atom-centred box; interpolation must
    # remain positive and cannot create electrons outside that finite domain.
    assert np.all(grid.density >= 0)
    assert 0 < grid.density.sum() * grid.pixel_volume < 1
