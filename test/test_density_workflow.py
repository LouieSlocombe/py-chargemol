"""Density discovery, legacy job controls, and signed-spin input handling."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from py_chargemol.ddec6 import StrPath
from py_chargemol.density.grid import DensityGrid
from py_chargemol.density.workflow import (
    DensityFormat,
    DensityKind,
    read_density,
    read_job_control,
)
from py_chargemol.units import BOHR_PER_ANGSTROM


def cube(values: str = "1 2 3 4 5 6 7 8") -> str:
    return (
        "density\nlast index fastest\n2 0 0 0\n"
        "2 1 0 0\n2 0 1 0\n2 0 0 1\n"
        "8 0 0.25 0.5 0.75\n1 0 0.75 0.5 0.25\n" + values + "\n"
    )


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_job_control_core_counts_and_explicit_overrides(tmp_path: Path) -> None:
    write(tmp_path / "valence_density.cube", cube())
    path = write(
        tmp_path / "job_control.txt",
        "<periodicity along A, B, and C vectors>\n.true. .false. .true.\n"
        "</periodicity along A, B, and C vectors>\n"
        "<net charge>1.5D0</net charge>\n"
        "<number of core electrons>8 2</number of core electrons>\n"
        "<input filename>calculation_label</input filename>\n"
        "<charge type>DDEC3</charge type>\n"
        "<atomic densities directory complete path>/does/not/exist"
        "</atomic densities directory complete path>\n",
    )
    controls = read_job_control(path)
    assert controls.charge_type == "ddec3"
    assert controls.core_electrons == {8: 2}
    grid = read_density(path)
    assert grid.periodic == (True, False, True)
    assert grid.net_charge == 1.5
    np.testing.assert_array_equal(grid.core_electrons, [2, 0])
    grid = read_density(
        tmp_path,
        periodic=(False, False, False),
        net_charge=0,
        core_electrons=[0, 0],
    )
    assert grid.periodic == (False, False, False)
    assert grid.net_charge == 0
    np.testing.assert_array_equal(grid.core_electrons, 0)


@pytest.mark.parametrize(("text", "expected"), [(".true.", True), ("f", False)])
def test_job_control_output_switches(tmp_path: Path, text: str, expected: bool) -> None:
    control = read_job_control(
        write(
            tmp_path / "job_control.txt",
            f"<print atomic densities>{text}</print atomic densities>\n"
            f"<compute BOs>{text}</compute BOs>\n",
        )
    )
    assert control.print_atomic_densities is expected
    assert control.compute_bos is expected


def test_partial_core_counts_preserve_original_element_defaults(tmp_path: Path) -> None:
    write(tmp_path / "valence_density.cube", cube())
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>1 0</number of core electrons>",
    )
    np.testing.assert_array_equal(read_density(tmp_path).core_electrons, [2, 0])


def test_total_only_cube_ignores_job_core_counts_and_rejects_explicit_nonzero(
    tmp_path: Path,
) -> None:
    write(tmp_path / "total_density.cube", cube())
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>",
    )
    np.testing.assert_array_equal(read_density(tmp_path).core_electrons, [0, 0])
    np.testing.assert_array_equal(
        read_density(tmp_path, core_electrons=[0, 0]).core_electrons, [0, 0]
    )
    with pytest.raises(ValueError, match="already include core electrons"):
        read_density(tmp_path, core_electrons=[2, 0])


def test_valence_cube_discovers_total_and_signed_spin(tmp_path: Path) -> None:
    write(tmp_path / "valence_density.cube", cube())
    write(tmp_path / "total_density.cube", cube("3 4 5 6 7 8 9 10"))
    write(tmp_path / "spin_density.cube", cube("-1 1 -1 1 -1 1 -1 1"))
    grid = read_density(tmp_path, core_electrons=[2, 0])
    assert grid.core_density is not None
    np.testing.assert_array_equal(grid.density, 0.03)
    np.testing.assert_allclose(
        grid.core_density, np.arange(3, 11).reshape(2, 2, 2) - 0.03
    )
    np.testing.assert_array_equal(
        grid.spin_density, np.tile([-1, 1], 4).reshape(2, 2, 2)
    )


@pytest.mark.parametrize(
    ("units", "length_scale", "density_scale"),
    [
        ("onetep", 1.0, BOHR_PER_ANGSTROM**-3),
        ("e_per_angs3", BOHR_PER_ANGSTROM, BOHR_PER_ANGSTROM**-3),
        ("e_per_bohr3", 1.0, 1.0),
    ],
)
def test_job_control_cube_units_include_signed_spin(
    tmp_path: Path, units: str, length_scale: float, density_scale: float
) -> None:
    write(tmp_path / "total_density.cube", cube())
    write(tmp_path / "spin_density.cube", cube("-1 1 -1 1 -1 1 -1 1"))
    controls = write(
        tmp_path / "job_control.txt",
        f"<density format>{units}</density format>",
    )
    assert read_job_control(controls).density_format == units
    grid = read_density(tmp_path)
    assert grid.spin_density is not None
    np.testing.assert_allclose(grid.steps, np.eye(3) * length_scale)
    np.testing.assert_allclose(
        grid.positions,
        np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]]) * length_scale,
    )
    np.testing.assert_allclose(
        grid.density, np.arange(1, 9).reshape(2, 2, 2) * density_scale
    )
    np.testing.assert_allclose(
        grid.spin_density,
        np.tile([-1, 1], 4).reshape(2, 2, 2) * density_scale,
    )


def test_vector_spin_cubes_require_complete_matching_geometry(tmp_path: Path) -> None:
    write(tmp_path / "total_density.cube", cube())
    write(tmp_path / "spin_density_x.cube", cube("0 0 0 0 0 0 0 0"))
    with pytest.raises(FileNotFoundError, match="all three"):
        read_density(tmp_path)
    write(tmp_path / "spin_density_y.cube", cube("1 1 1 1 1 1 1 1"))
    write(tmp_path / "spin_density_z.cube", cube("-1 -1 -1 -1 -1 -1 -1 -1"))
    grid = read_density(tmp_path)
    assert grid.spin_density is not None
    assert grid.spin_density.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(
        grid.spin_density[0, 0, 0], np.array([0, 1, -1]) / 2**0.5
    )
    write(tmp_path / "spin_density_y.cube", cube().replace("0.25", "0.35"))
    with pytest.raises(ValueError, match="must match"):
        read_density(tmp_path)


def test_cube_spin_density_is_bounded_before_core_iteration(tmp_path: Path) -> None:
    write(tmp_path / "total_density.cube", cube("0 1e-12 0.1 0.2 1 2 3 4"))
    write(tmp_path / "spin_density.cube", cube("1 -1 1 -1 1e-12 -1 1 -1"))
    grid = read_density(tmp_path)
    np.testing.assert_array_equal(
        grid.spin_density, np.array([0, 0, 0.1, -0.2, 0, -1, 1, -1]).reshape(2, 2, 2)
    )


def test_valence_spin_uses_reconstructed_core_density(tmp_path: Path) -> None:
    write(tmp_path / "valence_density.cube", cube("0 0 0 0 0 0 0 0"))
    write(tmp_path / "spin_density.cube", cube("1 1 1 1 1 1 1 1"))
    grid = read_density(tmp_path, core_electrons=[2, 0])
    assert grid.spin_density is not None
    assert np.any(grid.spin_density > 0)
    assert np.all(grid.spin_density <= 1)
    # This point is sqrt(0.375) bohr from oxygen, shell 7 in the original
    # 100-shell reference table (core_008_008_002_005_100.txt).
    assert grid.spin_density[0, 0, 1] == pytest.approx(0.031892989, abs=1e-12)


def test_named_input_and_ambiguous_auto_discovery(tmp_path: Path) -> None:
    write(tmp_path / "first.cube", cube())
    write(tmp_path / "second.cube", cube())
    with pytest.raises(ValueError, match="multiple density inputs"):
        read_density(tmp_path)
    write(tmp_path / "job_control.txt", "<input filename>second.cube</input filename>")
    read_density(tmp_path)
    write(tmp_path / "job_control.txt", "<input filename>missing.cube</input filename>")
    with pytest.raises(FileNotFoundError, match="input file does not exist"):
        read_density(tmp_path)
    # An explicitly selected source file takes precedence over job-control discovery.
    read_density(tmp_path / "first.cube")


def test_job_control_can_reference_parent_input(tmp_path: Path) -> None:
    write(tmp_path / "total.cube", cube())
    calculation = tmp_path / "analysis"
    calculation.mkdir()
    write(
        calculation / "job_control.txt",
        "<input filename>../total.cube</input filename>",
    )
    grid = read_density(calculation)
    np.testing.assert_array_equal(grid.atomic_numbers, [8, 1])


def test_vasp_core_counts_from_job_control_without_potcar(tmp_path: Path) -> None:
    text = (
        "O H\n1\n2 0 0\n0 2 0\n0 0 2\nO H\n1 1\nDirect\n"
        "0 0 0\n0.5 0.5 0.5\n\n2 2 2\n1 1 1 1 1 1 1 1\n"
    )
    write(tmp_path / "AECCAR2", text)
    write(tmp_path / "CHGCAR", text)
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>",
    )
    grid = read_density(tmp_path)
    np.testing.assert_array_equal(grid.atomic_numbers, [8, 1])
    np.testing.assert_array_equal(grid.core_electrons, [2, 0])
    assert grid.periodic == (True, True, True)


def test_xsf_core_counts_from_job_control(tmp_path: Path) -> None:
    text = (
        "CRYSTAL\nPRIMVEC\n1 0 0\n0 1 0\n0 0 1\n"
        "PRIMCOORD\n1 1\nO 0.25 0.5 0.75\n"
        "BEGIN_BLOCK_DATAGRID_3D\nexample\nBEGIN_DATAGRID_3D_density\n"
        "3 3 3\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        + " ".join(["0.1"] * 27)
        + "\nEND_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"
    )
    write(tmp_path / "density.XSF", text)
    with pytest.raises(ValueError, match="requires explicit core_electrons"):
        read_density(tmp_path)
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>",
    )
    grid = read_density(tmp_path)
    np.testing.assert_array_equal(grid.core_electrons, [2])
    assert grid.density.shape == (2, 2, 2)


@pytest.mark.parametrize("components", [2, 4])
def test_xsf_spin_is_bounded_by_electron_density(
    tmp_path: Path, components: int
) -> None:
    text = "ATOMS\nH 0 0 0\nBEGIN_BLOCK_DATAGRID_3D\nspin\n"
    for index, value in enumerate([0.4, -0.2, 0.3, -0.4][:components], start=1):
        text += (
            f"BEGIN_DATAGRID_3D_rho:spin_{index}\n3 3 3\n0 0 0\n"
            "1 0 0\n0 1 0\n0 0 1\n"
            + " ".join([str(value)] * 27)
            + "\nEND_DATAGRID_3D\n"
        )
    text += "END_BLOCK_DATAGRID_3D\n"
    grid = read_density(write(tmp_path / "spin.xsf", text), core_electrons=[0])
    assert grid.spin_density is not None
    magnitude = (
        np.abs(grid.spin_density)
        if components == 2
        else np.linalg.norm(grid.spin_density, axis=-1)
    )
    np.testing.assert_allclose(magnitude, 0.2, atol=1e-16)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<net charge>bad</net charge>", "finite number"),
        ("<net charge>nan</net charge>", "finite number"),
        ("<net charge>1 2</net charge>", "finite number"),
        ("<net charge>1", "closing tag"),
        ("<net charge>1</net charge><net charge>2</net charge>", "duplicate"),
        (
            (
                "<periodicity along A, B, and C vectors>true false"
                "</periodicity along A, B, and C vectors>"
            ),
            "three true/false",
        ),
        ("<number of core electrons>8 10</number of core electrons>", "invalid"),
        (
            "<number of core electrons>8</number of core electrons>",
            "atomic number and count",
        ),
        ("<input filename>\n</input filename>", "one filename"),
        (
            "<print atomic densities>yes</print atomic densities>",
            "one true/false",
        ),
        ("<compute BOs>true false</compute BOs>", "one true/false"),
    ],
)
def test_malformed_job_controls(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_job_control(write(tmp_path / "job_control.txt", text))


def test_missing_or_unrecognized_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_density(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="no auto density input"):
        read_density(tmp_path)
    source = write(tmp_path / "unknown.txt", cube())
    with pytest.raises(ValueError, match="cannot infer"):
        read_density(source)
    assert read_density(source, format="cube").natoms == 2


def test_valence_kind_requires_core_metadata(tmp_path: Path) -> None:
    write(tmp_path / "valence_density.cube", cube())
    with pytest.raises(ValueError, match="explicit core_electrons"):
        read_density(tmp_path)


def test_invalid_format_and_density_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="format must be"):
        read_density(tmp_path, format=cast("DensityFormat", "unknown"))
    with pytest.raises(ValueError, match="density_kind must be"):
        read_density(tmp_path, density_kind=cast("DensityKind", "unknown"))


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("valence_density.cube", "short\n", "cube atom header"),
        ("density.xsf", "PRIMCOORD\n", "XSF atom header"),
        ("density.xsf", "CRYSTAL\n", "missing XSF atomic"),
        ("density.xsf", "ATOMS\n8 0 0 0\n", "3D density grid"),
        ("AECCAR2", "short\n", "VASP species/count header"),
    ],
)
def test_malformed_geometry_for_per_element_core_counts(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    write(tmp_path / name, text)
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>\n! core count\n8 2\n</number of core electrons>",
    )
    with pytest.raises(ValueError, match=message):
        read_density(tmp_path)


def test_xsf_atoms_block_core_mapping_and_explicit_format(tmp_path: Path) -> None:
    text = (
        "ATOMS\n\n8 0 0 0\n\nBEGIN_BLOCK_DATAGRID_3D\nexample\n"
        "BEGIN_DATAGRID_3D_density\n3 3 3\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        + " ".join(["0.1"] * 27)
        + "\nEND_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"
    )
    write(tmp_path / "density.xsf", text)
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>",
    )
    grid = read_density(tmp_path, format="xsf", periodic=(True, False, False))
    assert grid.periodic == (True, False, False)
    np.testing.assert_array_equal(grid.core_electrons, [2])


def test_vasp4_core_mapping_uses_potcar_species(tmp_path: Path) -> None:
    text = (
        "oxygen\n1\n1 0 0\n0 1 0\n0 0 1\n1\nDirect\n0 0 0\n\n2 2 2\n1 1 1 1 1 1 1 1\n"
    )
    write(tmp_path / "AECCAR2", text)
    write(tmp_path / "CHGCAR", text)
    write(tmp_path / "POTCAR", "VRHFIN =O: s2p4\nZVAL = 6.0\nEnd of Dataset\n")
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>",
    )
    grid = read_density(tmp_path, format="vasp")
    np.testing.assert_array_equal(grid.core_electrons, [2])


def test_wfx_metadata_inference_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write(tmp_path / "density.wfx", "test WFX dispatch")
    expected = DensityGrid(
        atomic_numbers=np.asarray([8]),
        positions=np.zeros((1, 3)),
        density=np.ones((2, 2, 2)),
        steps=np.eye(3),
        origin=np.zeros(3),
        net_charge=1,
    )

    def reader(path: StrPath) -> DensityGrid:
        assert Path(path) == source
        return expected

    monkeypatch.setattr("py_chargemol.density.wfx.read_wfx", reader)
    assert read_density(tmp_path).net_charge == 1
    grid = read_density(tmp_path, core_electrons=[0], net_charge=0)
    assert grid.net_charge == 0
    np.testing.assert_array_equal(grid.core_electrons, [0])
    with pytest.raises(ValueError, match="analytic core counts"):
        read_density(tmp_path, core_electrons=[2])
    with pytest.raises(ValueError, match="translation vectors"):
        read_density(tmp_path, periodic=(True, False, False))
    assert read_density(tmp_path, periodic=(False, False, False)) is expected
    write(
        tmp_path / "job_control.txt",
        "<number of core electrons>8 2</number of core electrons>\n"
        "<periodicity along A, B, and C vectors>true true true"
        "</periodicity along A, B, and C vectors>",
    )
    assert read_density(tmp_path) is expected
