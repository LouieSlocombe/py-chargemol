"""Tests for the Chargemol DDEC6 file readers."""

from pathlib import Path

import numpy as np
import pytest

from py_chargemol.ddec6 import read_ddec6_net_atomic_charges, read_overlap_populations
from py_chargemol.units import BOHR_PER_ANGSTROM


@pytest.fixture
def co_charges(worked_examples: Path) -> Path:
    co_molecule = worked_examples / "CO_molecule"
    return co_molecule / "DDEC6_even_tempered_net_atomic_charges.xyz"


@pytest.fixture
def co_lines(co_charges: Path) -> list[str]:
    return co_charges.read_text().splitlines()


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_reads_a_nonperiodic_molecule(co_charges: Path) -> None:
    analysis = read_ddec6_net_atomic_charges(co_charges)

    assert analysis.natoms == 2
    np.testing.assert_array_equal(analysis.atomic_numbers, [6, 8])
    np.testing.assert_allclose(
        analysis.positions,
        np.array([[0.0, 0.0, 0.004975], [0.0, 0.0, 1.129925]]) * BOHR_PER_ANGSTROM,
    )
    np.testing.assert_array_equal(analysis.net_charges, [0.132034, -0.132034])
    np.testing.assert_array_equal(analysis.atomic_dipoles[:, 2], [0.312981, 0.007543])
    np.testing.assert_array_equal(analysis.cloud_a, [0.450606, 0.621995])
    np.testing.assert_array_equal(analysis.cloud_b, [2.069311, 2.344336])
    np.testing.assert_array_equal(analysis.lattice_vectors, np.zeros((3, 3)))
    assert analysis.periodic == (False, False, False)
    assert not analysis.is_periodic


def test_quadrupoles_become_traceless_cartesian_tensors(co_charges: Path) -> None:
    """Q(3z^2-R^2) = -1.407389 for carbon gives zz = Q/3 and xx = yy = -zz/2."""
    quadrupoles = read_ddec6_net_atomic_charges(co_charges).atomic_quadrupoles

    zz = -1.407389 / 3.0
    np.testing.assert_allclose(quadrupoles[0], [-zz / 2, 0.0, 0.0, -zz / 2, 0.0, zz])
    np.testing.assert_allclose(quadrupoles[:, [0, 3, 5]].sum(axis=1), 0.0, atol=1e-15)


def test_quadrupole_components_map_to_tensor_elements(tmp_path: Path) -> None:
    lines = [
        "    1",
        "Nonperiodic system",
        "C      0.000000     0.000000     0.000000     0.000000",
        *[f"header line {i}" for i in range(14)],
        "    1 C  0.0 0.0 0.0 0.0  0.1 0.2 0.3 0.4  1.0 2.0 3.0 4.0 6.0  0.0 0.0 0.0",
        "cloud header",
        "cloud columns",
        "    1 C  0.0 0.0 0.0  0.5 2.0 0.99",
    ]
    analysis = read_ddec6_net_atomic_charges(write(tmp_path / "charges.xyz", lines))

    np.testing.assert_array_equal(analysis.atomic_dipoles, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        analysis.atomic_quadrupoles, [[1.0, 1.0, 2.0, -3.0, 3.0, 2.0]]
    )


def test_reads_a_periodic_cell(worked_examples: Path) -> None:
    analysis = read_ddec6_net_atomic_charges(
        worked_examples / "formamide_VASP" / "DDEC6_net_atomic_charges.xyz"
    )

    assert analysis.natoms == 6
    np.testing.assert_array_equal(analysis.atomic_numbers, [1, 1, 1, 6, 7, 8])
    np.testing.assert_allclose(
        analysis.lattice_vectors,
        np.array([[0, 15, 0], [0, 0, 15], [15, 0, 0]]) * BOHR_PER_ANGSTROM,
    )
    assert analysis.periodic == (True, True, True)
    assert analysis.is_periodic


def test_a_zero_lattice_vector_is_a_nonperiodic_direction(
    tmp_path: Path, co_lines: list[str]
) -> None:
    co_lines[1] = (
        'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell '
        "[{ 1.0D+01 0.0 0.0 }, { 0.0 0.0 0.0 }, { 0.0 0.0 .5e1 }]"
    )
    analysis = read_ddec6_net_atomic_charges(write(tmp_path / "c.xyz", co_lines))

    assert analysis.periodic == (True, False, True)
    np.testing.assert_allclose(analysis.lattice_vectors[0, 0], 10 * BOHR_PER_ANGSTROM)
    np.testing.assert_allclose(analysis.lattice_vectors[2, 2], 5 * BOHR_PER_ANGSTROM)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["", "  "], "the file is empty"),
        (["two", "Nonperiodic system"], "expected a count on the first line"),
        (["-1", "Nonperiodic system"], "the count on the first line is negative"),
        (["0", "Nonperiodic system"], "the file lists no atoms"),
        (["2"], "missing the periodicity line"),
        (["2", "Periodic system"], "expected 'Nonperiodic system' or a jmolscript"),
        (["2", "unitcell [{ 1 2 3 }, { 4 5 6 }]"], "or a jmolscript unit cell"),
        (["2", "Nonperiodic system", "C 0 0 0 0"], "before all 2 rows of atomic"),
    ],
)
def test_malformed_headers_are_rejected(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_ddec6_net_atomic_charges(write(tmp_path / "charges.xyz", lines))


def test_unknown_elements_are_rejected(tmp_path: Path, co_lines: list[str]) -> None:
    co_lines[2] = co_lines[2].replace("C ", "Q ")

    with pytest.raises(ValueError, match="unknown atomic symbol 'Q'"):
        read_ddec6_net_atomic_charges(write(tmp_path / "c.xyz", co_lines))


def test_unreadable_numbers_are_rejected(tmp_path: Path, co_lines: list[str]) -> None:
    co_lines[2] = co_lines[2].replace("0.132034", "0.13x")

    with pytest.raises(ValueError, match="could not read columns 2-5"):
        read_ddec6_net_atomic_charges(write(tmp_path / "c.xyz", co_lines))


def test_misaligned_tables_are_rejected(tmp_path: Path, co_lines: list[str]) -> None:
    """An extra header line shifts the multipole table, which the labels catch."""
    co_lines.insert(6, "An unexpected extra header line")

    with pytest.raises(ValueError, match="expected row 1 \\(C\\) of the atomic multi"):
        read_ddec6_net_atomic_charges(write(tmp_path / "c.xyz", co_lines))


def test_mislabelled_cloud_rows_are_rejected(
    tmp_path: Path, co_lines: list[str]
) -> None:
    row = next(i for i, line in enumerate(co_lines) if "0.450606" in line)
    co_lines[row] = co_lines[row].replace(" C ", " O ")

    with pytest.raises(ValueError, match="row 1 \\(C\\) of the electron cloud"):
        read_ddec6_net_atomic_charges(write(tmp_path / "c.xyz", co_lines))


def test_reads_overlap_populations(worked_examples: Path) -> None:
    co_molecule = worked_examples / "CO_molecule"
    pairs = read_overlap_populations(
        co_molecule / "overlap_populations_and_contact_exchanges.txt"
    )

    assert len(pairs) == 2
    np.testing.assert_array_equal(pairs.atom1, [0, 1])
    np.testing.assert_array_equal(pairs.atom2, [1, 0])
    np.testing.assert_array_equal(pairs.translations, np.zeros((2, 3)))
    np.testing.assert_array_equal(pairs.populations, [1.438144, 1.438144])


def test_reads_periodic_image_translations(worked_examples: Path) -> None:
    pairs = read_overlap_populations(
        worked_examples
        / "formamide_VASP"
        / "overlap_populations_and_contact_exchanges.txt"
    )

    assert len(pairs) == 30
    np.testing.assert_array_equal(pairs.translations[:2], [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(pairs.populations[:2], [0.000294, 0.000294])


def test_reads_the_older_overlap_layout(tmp_path: Path) -> None:
    """overlap_populations.txt lacks the contact exchange column."""
    lines = ["1", "Nonperiodic system", "header", "  1  2  0  0  1  0.25", ""]

    pairs = read_overlap_populations(write(tmp_path / "overlap_populations.txt", lines))

    np.testing.assert_array_equal(pairs.translations, [[0, 0, 1]])
    np.testing.assert_array_equal(pairs.populations, [0.25])


def test_reads_an_empty_pair_list(tmp_path: Path) -> None:
    pairs = read_overlap_populations(
        write(tmp_path / "overlaps.txt", ["0", "Nonperiodic system", "header"])
    )

    assert len(pairs) == 0
    assert pairs.translations.shape == (0, 3)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "the file is empty"),
        (["x"], "expected a count on the first line"),
        (["2", "Nonperiodic", "header", "1 2 0 0 0 0.5"], "before all 2 rows"),
        (["1", "Nonperiodic", "header", "1 2 0 0 0.5 0.5"], "columns 1-5"),
        (["1", "Nonperiodic", "header", "0 2 0 0 0 0.5"], "must be positive"),
    ],
)
def test_malformed_overlap_files_are_rejected(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        read_overlap_populations(write(tmp_path / "overlaps.txt", lines))
