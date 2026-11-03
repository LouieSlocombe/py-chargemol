"""Density format, units, orientation, and PAW integration metadata tests."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from py_chargemol.density.analysis import compute_ddec6
from py_chargemol.density.readers import read_cube, read_vasp, read_xsf
from py_chargemol.units import BOHR_PER_ANGSTROM


def cube_text(values: str = "0 1 2 3 4 5 6 7", *, count: int = 2) -> str:
    return (
        "test density\nouter X inner Z\n1 0 0 0\n"
        f"{count} 1 0 0\n{count} 0 1 0\n{count} 0 0 1\n"
        f"8 0 0.25 0.5 0.75\n{values}\n"
    )


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_cube_order_geometry_and_fortran_exponents(tmp_path: Path) -> None:
    path = write(tmp_path / "total.cube", cube_text("0 1D0 2 3 4 5 6 7"))
    grid = read_cube(path, periodic=(True, False, False), net_charge=1)

    np.testing.assert_array_equal(grid.density, np.arange(8).reshape((2, 2, 2)))
    np.testing.assert_array_equal(grid.positions, [[0.25, 0.5, 0.75]])
    np.testing.assert_array_equal(grid.atomic_numbers, [8])
    np.testing.assert_array_equal(grid.core_electrons, [0])
    assert grid.periodic == (True, False, False)
    assert grid.net_charge == 1
    assert grid.pixel_volume == 1


def test_cube_rows_with_short_final_lines(tmp_path: Path) -> None:
    rows = [list(map(str, range(start, start + 8))) for start in range(0, 32, 8)]
    values = "\n".join(" ".join(row[:6]) + "\n" + " ".join(row[6:]) for row in rows)
    text = cube_text(values).replace("2 0 0 1", "8 0 0 1")
    grid = read_cube(write(tmp_path / "rows.cube", text))
    np.testing.assert_array_equal(grid.density, np.arange(32).reshape((2, 2, 8)))


def test_fortran_exponents_across_blank_and_multiple_lines(tmp_path: Path) -> None:
    values = "0D0 1d0\n\n2.0D+00 3E0\n4d0 5.D0\n6D0 0.7d1"
    grid = read_cube(write(tmp_path / "fortran.cube", cube_text(values)))
    np.testing.assert_array_equal(grid.density, np.arange(8).reshape((2, 2, 2)))


def test_cube_negative_counts_and_independent_density_units(tmp_path: Path) -> None:
    path = write(tmp_path / "angstrom.cube", cube_text(count=-2))
    grid = read_cube(path, density_unit="angstrom^-3")

    np.testing.assert_allclose(grid.steps, np.eye(3) * BOHR_PER_ANGSTROM)
    np.testing.assert_allclose(
        grid.positions, [[0.25, 0.5, 0.75]] * np.array(BOHR_PER_ANGSTROM)
    )
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(28)
    override = read_cube(path, length_unit="bohr")
    np.testing.assert_array_equal(override.steps, np.eye(3))


def test_cube_valence_and_core_subtraction(tmp_path: Path) -> None:
    valence = write(tmp_path / "valence.cube", cube_text("-1 1 1 1 1 1 1 1"))
    total = write(tmp_path / "total.cube", cube_text("2 2 2 2 2 2 2 2"))
    grid = read_cube(
        valence, density_kind="valence", core_electrons=[2], total_density_file=total
    )

    assert grid.density[0, 0, 0] == 0
    assert grid.core_density is not None
    assert grid.core_density[0, 0, 0] == 2
    assert grid.core_density[1, 1, 1] == pytest.approx(1.97)
    np.testing.assert_array_equal(grid.core_electrons, [2])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "unexpected end"),
        (cube_text().replace("1 0 0 0", "1 0 0", 1), "atom count and origin"),
        (cube_text().replace("1 0 0 0", "-1 0 0 0", 1), "scalar density cube"),
        (cube_text().replace("1 0 0 0", "1 0 0 0 2", 1), "scalar density cube"),
        (cube_text().replace("2 1 0 0", "2 1 0"), "grid axis"),
        (cube_text().replace("2 1 0 0", "-2 1 0 0"), "consistent sign"),
        (cube_text().replace("8 0 0.25 0.5 0.75", "8 0 0.25 0.5"), "five fields"),
        (cube_text("nan 1 2 3 4 5 6 7"), "nonfinite"),
        (cube_text("no 1 2 3 4 5 6 7"), "invalid numeric"),
        (cube_text("0 1 2"), "unexpected end"),
        (cube_text("0 1 2 3 4 5 6 7 8"), "too many values"),
        (cube_text("0\n1 2 3 4 5 6 7 8"), "too many values"),
        (cube_text("0 1 2 3 4 5 6 7 no"), "invalid numeric"),
        (cube_text() + "another dataset\n", "unexpected data"),
        (cube_text("0\n1 2 3 4 5 6 7") + "another dataset\n", "unexpected data"),
    ],
)
def test_malformed_cubes(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_cube(write(tmp_path / "invalid.cube", text))


def test_cube_core_metadata_and_options_are_explicit(tmp_path: Path) -> None:
    path = write(tmp_path / "density.cube", cube_text())
    with pytest.raises(ValueError, match="explicit core_electrons"):
        read_cube(path, density_kind="valence")
    with pytest.raises(ValueError, match="only meaningful"):
        read_cube(path, total_density_file=path)
    with pytest.raises(ValueError, match="density_kind"):
        read_cube(path, density_kind=cast("object", "unknown"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported cube units"):
        read_cube(path, length_unit=cast("object", "nm"))  # type: ignore[arg-type]
    other = write(tmp_path / "other.cube", cube_text().replace("0.25", "0.35"))
    with pytest.raises(ValueError, match="must match"):
        read_cube(
            path, density_kind="valence", core_electrons=[2], total_density_file=other
        )


def vasp_text(
    *, symbols: bool = True, values: str = "0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08"
) -> str:
    return (
        "oxygen\n1\n1 0 0\n0 1 0\n0 0 1\n"
        + ("O\n" if symbols else "")
        + "1\nDirect\n0.25 0.5 0.75\n\n2 2 2\n"
        + values
        + "\n"
    )


@pytest.fixture
def vasp_inputs(tmp_path: Path) -> Path:
    write(tmp_path / "AECCAR2", vasp_text())
    write(tmp_path / "CHGCAR", vasp_text(values="6 6 6 6 6 6 6 6"))
    write(tmp_path / "AECCAR0", vasp_text(values="2 2 2 2 2 2 2 2"))
    write(
        tmp_path / "POTCAR", "PAW O\n6\nVRHFIN =O: s2p4\nZVAL = 6.0\nEnd of Dataset\n"
    )
    return tmp_path


@pytest.mark.parametrize("symbols", [True, False])
def test_vasp_density_order_counts_and_volume(vasp_inputs: Path, symbols: bool) -> None:
    if not symbols:
        for name in ("AECCAR2", "AECCAR0", "CHGCAR"):
            path = vasp_inputs / name
            path.write_text(path.read_text().replace("O\n", ""))
    grid = read_vasp(vasp_inputs)
    volume = BOHR_PER_ANGSTROM**3

    np.testing.assert_allclose(
        grid.density * volume, np.arange(1, 9).reshape((2, 2, 2), order="F") / 100
    )
    np.testing.assert_allclose(
        grid.positions, np.array([[0.25, 0.5, 0.75]]) * BOHR_PER_ANGSTROM
    )
    np.testing.assert_array_equal(grid.core_electrons, [2])
    assert grid.core_density is not None
    assert grid.core_density.sum() * grid.pixel_volume == pytest.approx(2)
    assert grid.pseudodensity is not None
    assert grid.pseudodensity.sum() * grid.pixel_volume == pytest.approx(6)
    assert grid.net_charge == pytest.approx(0)
    assert grid.periodic == (True, True, True)


def test_vasp_clipping_preserves_pseudodensity(vasp_inputs: Path) -> None:
    write(vasp_inputs / "AECCAR2", vasp_text(values="-1 4 4 4 4 4 4 4"))
    grid = read_vasp(vasp_inputs)
    assert grid.density[0, 0, 0] == 0
    assert grid.density.max() * grid.pixel_volume == pytest.approx(0.03)
    assert grid.pseudodensity is not None
    assert grid.pseudodensity.sum() * grid.pixel_volume == pytest.approx(6)


def test_vasp_explicit_core_and_missing_core_density(vasp_inputs: Path) -> None:
    (vasp_inputs / "POTCAR").unlink()
    (vasp_inputs / "AECCAR0").unlink()
    (vasp_inputs / "CHGCAR").rename(vasp_inputs / "CHG")
    grid = read_vasp(vasp_inputs, core_electrons=[2], net_charge=1)
    assert grid.core_density is None
    assert grid.net_charge == 1


@pytest.mark.parametrize("name", ["AECCAR0", "CHGCAR"])
def test_vasp_rejects_mismatching_geometry(vasp_inputs: Path, name: str) -> None:
    path = vasp_inputs / name
    path.write_text(path.read_text().replace("0.25", "0.5"))
    with pytest.raises(ValueError, match="must match"):
        read_vasp(vasp_inputs)


def test_vasp_requires_core_and_charge_metadata(vasp_inputs: Path) -> None:
    (vasp_inputs / "CHGCAR").unlink()
    with pytest.raises(FileNotFoundError, match="CHGCAR or CHG"):
        read_vasp(vasp_inputs)
    (vasp_inputs / "POTCAR").unlink()
    with pytest.raises(ValueError, match="POTCAR or explicit"):
        read_vasp(vasp_inputs)


@pytest.mark.parametrize("scale", ["2", "-8", "2 2 2"])
def test_vasp_scale_and_cartesian_selective_coordinates(
    vasp_inputs: Path, scale: str
) -> None:
    for name in ("AECCAR2", "AECCAR0", "CHGCAR"):
        path = vasp_inputs / name
        text = (
            path.read_text()
            .replace("oxygen\n1\n", f"oxygen\n{scale}\n")
            .replace("Direct", "Selective dynamics\nCartesian")
        )
        path.write_text(text)
    grid = read_vasp(vasp_inputs)
    np.testing.assert_allclose(
        grid.positions, np.array([[0.5, 1, 1.5]]) * BOHR_PER_ANGSTROM
    )
    assert grid.pixel_volume * 8 == pytest.approx(8 * BOHR_PER_ANGSTROM**3)


def xsf_text(name: str = "density", value: float = 0.1) -> str:
    return (
        "# Comment\nCRYSTAL\nPRIMVEC\n1 0 0\n0 1 0\n0 0 1\n"
        "PRIMCOORD\n1 1\nO 0.25 0.5 0.75\n"
        "BEGIN_BLOCK_DATAGRID_3D\nexample\n"
        f"BEGIN_DATAGRID_3D_{name}\n3 3 3\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        + " ".join([str(value)] * 27)
        + "\nEND_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"
    )


def test_xsf_periodic_endpoint_planes_and_units(tmp_path: Path) -> None:
    grid = read_xsf(
        write(tmp_path / "density.xsf", xsf_text()),
        core_electrons=[2],
        density_unit="angstrom^-3",
    )
    assert grid.density.shape == (2, 2, 2)
    assert grid.density.sum() * grid.pixel_volume == pytest.approx(0.1)
    assert grid.periodic == (True, True, True)
    np.testing.assert_allclose(
        grid.positions, np.array([[0.25, 0.5, 0.75]]) * BOHR_PER_ANGSTROM
    )


def test_xsf_block_stops_at_the_line_that_fills_it(tmp_path: Path) -> None:
    text = xsf_text().replace("\n0.1 ", "\n0.1\n", 1)
    grid = read_xsf(write(tmp_path / "density.xsf", text), core_electrons=[2])
    np.testing.assert_array_equal(grid.density, np.full((2, 2, 2), 0.1))


def test_xsf_collinear_spin_channels(tmp_path: Path) -> None:
    text = xsf_text("rho:spin_1", 0.3)
    second = xsf_text("rho:spin_2", 0.1).split("BEGIN_DATAGRID_3D_", 1)[1]
    text += "BEGIN_DATAGRID_3D_" + second
    grid = read_xsf(write(tmp_path / "spin.xsf", text), core_electrons=[2])
    np.testing.assert_allclose(grid.density, 0.4)
    assert grid.spin_density is not None
    np.testing.assert_allclose(grid.spin_density, 0.2)


@pytest.mark.parametrize("keyword", ["ATOMS", "POLYMER", "SLAB"])
def test_xsf_lower_dimensional_structures(tmp_path: Path, keyword: str) -> None:
    text = xsf_text().replace("CRYSTAL", keyword)
    if keyword == "ATOMS":
        text = (
            text.replace("ATOMS\n", "")
            .replace("PRIMCOORD\n1 1", "ATOMS")
            .replace("O 0.25", "8 0.25")
        )
    grid = read_xsf(write(tmp_path / "density.xsf", text), core_electrons=[2])
    assert grid.periodic == (keyword != "ATOMS", keyword == "SLAB", False)


def add_vasp_spin(directory: Path, blocks: list[str]) -> None:
    path = directory / "CHGCAR"
    text = path.read_text() + "augmentation occupancies 1 2\n0 0\n"
    text += "".join(f"\n2 2 2\n{values}\n" for values in blocks)
    path.write_text(text)


def test_vasp_collinear_spin_preserves_signed_integrals(vasp_inputs: Path) -> None:
    add_vasp_spin(vasp_inputs, ["0.2 -0.2 0.2 -0.2 0.2 -0.2 0.2 -0.2"])
    grid = read_vasp(vasp_inputs)
    assert grid.spin_density is not None
    assert grid.spin_density.shape == grid.density.shape
    assert np.maximum(grid.spin_density, 0).sum() * grid.pixel_volume == pytest.approx(
        0.1
    )
    assert -np.minimum(grid.spin_density, 0).sum() * grid.pixel_volume == pytest.approx(
        0.1
    )


def test_vasp_spin_reconstructs_missing_core(vasp_inputs: Path) -> None:
    (vasp_inputs / "AECCAR0").unlink()
    add_vasp_spin(vasp_inputs, ["0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2"])
    grid = read_vasp(vasp_inputs)
    assert grid.core_density is not None
    assert grid.spin_density is not None
    assert np.all(np.abs(grid.spin_density) <= grid.density + grid.core_density)
    assert 0 < grid.spin_density.sum() * grid.pixel_volume <= 0.2


@pytest.mark.parametrize("axis", [None, "1 0 0", "0 0 0", "0 1", "unrelated INCAR"])
def test_vasp_noncollinear_spin_and_saxis(vasp_inputs: Path, axis: str | None) -> None:
    add_vasp_spin(vasp_inputs, [" ".join([str(value)] * 8) for value in [0.1, 0, 0.2]])
    if axis is not None:
        write(
            vasp_inputs / "INCAR",
            axis if axis == "unrelated INCAR" else f"SAXIS = {axis} # axis",
        )
    if axis in {"0 0 0", "0 1"}:
        with pytest.raises(ValueError, match="SAXIS"):
            read_vasp(vasp_inputs)
        return
    grid = read_vasp(vasp_inputs)
    assert grid.spin_density is not None
    expected = [0.2, 0, -0.1] if axis == "1 0 0" else [0.1, 0, 0.2]
    np.testing.assert_allclose(
        grid.spin_density.sum(axis=(0, 1, 2)) * grid.pixel_volume, expected, atol=1e-15
    )


def test_vasp_incomplete_noncollinear_blocks_are_rejected(vasp_inputs: Path) -> None:
    add_vasp_spin(vasp_inputs, ["0 0 0 0 0 0 0 0"] * 2)
    with pytest.raises(ValueError, match="zero, one, or three"):
        read_vasp(vasp_inputs)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no POTCAR datasets"),
        ("VRHFIN =O: s2p4", "VRHFIN and ZVAL"),
        ("VRHFIN =O: s2p4\nZVAL = 6.5", "ZVAL must be an integer"),
        ("VRHFIN =O: s2p4\nZVAL = 9", "ZVAL must be an integer"),
    ],
)
def test_invalid_potcar_metadata(vasp_inputs: Path, text: str, message: str) -> None:
    write(vasp_inputs / "POTCAR", text)
    with pytest.raises(ValueError, match=message):
        read_vasp(vasp_inputs)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("oxygen\n1", "oxygen\n1 2", "invalid VASP lattice"),
        ("oxygen\n1", "oxygen\n0", "scale must be positive"),
        ("1 0 0", "0 0 0", "singular VASP lattice"),
        ("O\n", "H\n", "element order"),
        ("O\n1\n", "O\n0\n", "atom counts"),
        ("Direct", "Unknown", "Direct or Cartesian"),
        ("0.25 0.5 0.75", "0.25 0.5", "three coordinates"),
        ("2 2 2", "2 2", "three integer grid dimensions"),
        ("2 2 2", "2 0 2", "dimensions must be positive"),
    ],
)
def test_invalid_vasp_headers(
    vasp_inputs: Path, old: str, new: str, message: str
) -> None:
    path = vasp_inputs / "AECCAR2"
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        read_vasp(vasp_inputs)


def test_vasp4_requires_potcar_element_identities(vasp_inputs: Path) -> None:
    (vasp_inputs / "POTCAR").unlink()
    write(vasp_inputs / "AECCAR2", vasp_text(symbols=False))
    with pytest.raises(ValueError, match="atom counts"):
        read_vasp(vasp_inputs, core_electrons=[2])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("CRYSTAL", "atomic coordinates and a 3D density grid"),
        (xsf_text().replace("3 3 3", "1 3 3"), "at least two points"),
        (xsf_text().replace("PRIMVEC\n1 0 0", "PRIMVEC\n2 0 0"), "vectors must match"),
    ],
)
def test_invalid_xsf_headers(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_xsf(write(tmp_path / "invalid.xsf", text), core_electrons=[2])


def test_xsf_noncollinear_components(tmp_path: Path) -> None:
    text = xsf_text("rho:spin_1", 0.3)
    for name, value in [
        ("rho:spin_2", 0.1),
        ("rho:spin_3", 0.02),
        ("rho:spin_4", 0.03),
    ]:
        text += (
            "BEGIN_DATAGRID_3D_"
            + xsf_text(name, value).split("BEGIN_DATAGRID_3D_", 1)[1]
        )
    grid = read_xsf(write(tmp_path / "spin.xsf", text), core_electrons=[2])
    assert grid.spin_density is not None
    np.testing.assert_allclose(
        grid.spin_density, np.broadcast_to([0.04, -0.06, 0.2], grid.spin_density.shape)
    )


def test_xsf_rejects_ambiguous_grids_and_units(tmp_path: Path) -> None:
    text = (
        xsf_text()
        + "BEGIN_DATAGRID_3D_"
        + xsf_text("extra").split("BEGIN_DATAGRID_3D_", 1)[1]
    )
    with pytest.raises(ValueError, match="two/four rho:spin grids"):
        read_xsf(write(tmp_path / "ambiguous.xsf", text), core_electrons=[2])
    with pytest.raises(ValueError, match="unsupported XSF density units"):
        read_xsf(
            write(tmp_path / "units.xsf", xsf_text()),
            core_electrons=[2],
            density_unit="nm^-3",  # type: ignore[arg-type]
        )


def test_xsf_rejects_duplicate_grid_labels(tmp_path: Path) -> None:
    text = (
        xsf_text(value=0.1)
        + "BEGIN_DATAGRID_3D_"
        + xsf_text(value=0.2).split("BEGIN_DATAGRID_3D_", 1)[1]
    )
    with pytest.raises(ValueError, match="duplicate XSF density grid"):
        read_xsf(write(tmp_path / "duplicate.xsf", text), core_electrons=[2])


def test_cube_optional_dataset_count_one(tmp_path: Path) -> None:
    text = cube_text().replace("1 0 0 0", "1 0 0 0 1", 1)
    grid = read_cube(write(tmp_path / "one.cube", text))
    assert grid.density.sum() == 28


def test_cube_valence_cap_preserves_total_and_exposes_lost_electrons(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "density.cube", cube_text("1 1 1 1 1 1 1 1"))
    total = read_cube(path)
    valence = read_cube(path, density_kind="valence", core_electrons=[2])
    np.testing.assert_array_equal(total.density, 1)
    np.testing.assert_array_equal(valence.density, 0.03)
    assert valence.pseudodensity is None
    with pytest.raises(ValueError, match=r"density integrates to.*expected 6"):
        compute_ddec6(valence)
    combined = read_cube(
        path, density_kind="valence", core_electrons=[2], total_density_file=path
    )
    assert combined.core_density is not None
    np.testing.assert_allclose(combined.density + combined.core_density, total.density)


def test_valence_cube_singular_voxel_vectors(tmp_path: Path) -> None:
    path = write(tmp_path / "singular.cube", cube_text().replace("2 1 0 0", "2 0 0 0"))
    with pytest.raises(ValueError, match="nonsingular grid"):
        read_cube(path, density_kind="valence", core_electrons=[2])
