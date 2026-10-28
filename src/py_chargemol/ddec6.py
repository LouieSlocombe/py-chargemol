"""Readers for the Chargemol DDEC6 output files that the QDR method needs.

Two files are read:

- ``DDEC6_net_atomic_charges.xyz`` (or ``DDEC6_even_tempered_net_atomic_charges.xyz``),
  with the net atomic charges, atomic dipoles and quadrupoles, and electron cloud
  penetration parameters.
- ``overlap_populations_and_contact_exchanges.txt`` (or ``overlap_populations.txt``),
  with the bond overlap populations between pairs of atoms.

The layout handling follows the QDR program's Fortran list-directed reads, which skip
blank lines. Unlike the QDR program, the atomic multipole table is found by its title
rather than at a fixed line, because the September 2017 Chargemol release lists one
computational parameter fewer above it than later versions. All quantities are
converted to atomic units (bohr) on reading.
"""

import itertools
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from .elements import symbol_to_number
from .units import BOHR_PER_ANGSTROM

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type StrPath = str | os.PathLike[str]

_MULTIPOLE_TABLE_TITLE: Final = "The following XYZ coordinates are in angstroms."
"""Start of the title line of the atomic multipole table."""

_HEADER_LINES_BEFORE_MULTIPOLES: Final = 2
"""Non-blank lines from the multipole table's title to its first row."""

_HEADER_LINES_BEFORE_CLOUD: Final = 2
"""Non-blank lines between the multipole table and the electron cloud table."""

_MULTIPOLE_COLUMNS: Final = 18
"""Atom number, symbol, position, charge, dipole and its magnitude, five quadrupole
components, and three quadrupole eigenvalues."""

_CLOUD_COLUMNS: Final = 8
"""Atom number, symbol, position, ``a``, ``b``, and the squared correlation."""

_HEADER_LINES_BEFORE_OVERLAPS: Final = 2
"""Non-blank lines between the pair count and the first overlap population."""

_CELL_NUMBERS: Final = 9
"""Numbers in a jmolscript unit cell: three lattice vectors."""

_NUMBER: Final = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")

_OVERFLOW: Final = re.compile(r"\*+")
"""A Fortran fixed-width field too narrow for its integer, which prints as asterisks."""

_ONE_THIRD: Final = 1.0 / 3.0


@dataclass(frozen=True, slots=True, eq=False)
class DDEC6Analysis:
    """Per-atom results of a Chargemol DDEC6 analysis, in atomic units.

    Attributes:
        atomic_numbers: Atomic numbers, shape ``(n,)``.
        positions: Cartesian positions in bohr, shape ``(n, 3)``.
        net_charges: DDEC6 net atomic charges in e, shape ``(n,)``.
        atomic_dipoles: Atomic dipoles in e·bohr, shape ``(n, 3)``.
        atomic_quadrupoles: Traceless Cartesian atomic quadrupoles in e·bohr²,
            ordered ``xx, xy, xz, yy, yz, zz``, shape ``(n, 6)``.
        cloud_a: Parameter ``a`` of the ``exp(a - b r)`` fit to each atom's spherically
            averaged electron density tail, shape ``(n,)``.
        cloud_b: Parameter ``b`` of that fit, in inverse bohr, shape ``(n,)``.
        lattice_vectors: Unit cell vectors in bohr as rows, shape ``(3, 3)``. A row
            of zeros marks a nonperiodic direction.
    """

    atomic_numbers: IntArray
    positions: FloatArray
    net_charges: FloatArray
    atomic_dipoles: FloatArray
    atomic_quadrupoles: FloatArray
    cloud_a: FloatArray
    cloud_b: FloatArray
    lattice_vectors: FloatArray

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atomic_numbers)

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        """Whether each lattice direction is periodic."""
        a, b, c = (bool(np.any(vector != 0.0)) for vector in self.lattice_vectors)
        return a, b, c

    @property
    def is_periodic(self) -> bool:
        """Whether any lattice direction is periodic."""
        return any(self.periodic)


@dataclass(frozen=True, slots=True, eq=False)
class OverlapPairs:
    """Directed atom pairs with their DDEC6 bond overlap populations.

    Each bond appears once from each end. The second atom sits in the periodic image
    of the unit cell shifted by ``translations`` lattice vectors.

    Attributes:
        atom1: Zero-based index of the first atom, shape ``(m,)``.
        atom2: Zero-based index of the second atom, shape ``(m,)``.
        translations: Lattice translations of the second atom, shape ``(m, 3)``.
        populations: Bond overlap populations in e, shape ``(m,)``.
    """

    atom1: IntArray
    atom2: IntArray
    translations: IntArray
    populations: FloatArray

    def __len__(self) -> int:
        """Return the number of directed pairs."""
        return len(self.atom1)


def read_ddec6_net_atomic_charges(path: StrPath) -> DDEC6Analysis:
    """Read a Chargemol ``DDEC6_net_atomic_charges.xyz`` file.

    Args:
        path: The file to read. ``DDEC6_even_tempered_net_atomic_charges.xyz`` has
            the same layout and can be read too.

    Returns:
        The per-atom analysis in atomic units.

    Raises:
        ValueError: If the file is empty or does not have the expected layout.
    """
    all_lines = _read_lines(path)
    natoms = _parse_count(all_lines[0], path)
    if natoms is None:
        natoms = _table_length(all_lines, skip=2)
    if natoms == 0:
        msg = f"{path}: the file lists no atoms"
        raise ValueError(msg)
    if len(all_lines) == 1:
        msg = f"{path}: missing the periodicity line after the atom count"
        raise ValueError(msg)
    lattice_vectors = _parse_lattice(all_lines[1], path) * BOHR_PER_ANGSTROM

    # Fortran list-directed reads skip blank records, so count non-blank lines only.
    body = [line for line in itertools.islice(all_lines, 2, None) if line.strip()]
    coordinate_rows = _block(body, 0, natoms, "atomic coordinates", path)
    start = _find_title(body, natoms, _MULTIPOLE_TABLE_TITLE, "atomic multipoles", path)
    start += _HEADER_LINES_BEFORE_MULTIPOLES
    multipole_rows = _block(body, start, natoms, "atomic multipoles", path)
    start += natoms + _HEADER_LINES_BEFORE_CLOUD
    cloud_rows = _block(body, start, natoms, "electron cloud parameters", path)

    symbols = [row.split(maxsplit=1)[0] for row in coordinate_rows]
    try:
        atomic_numbers = np.array([symbol_to_number(s) for s in symbols], np.int64)
    except ValueError as error:
        msg = f"{path}: {error}"
        raise ValueError(msg) from None
    coordinates = _table(coordinate_rows, range(1, 5), np.float64, path)
    _check_rows(multipole_rows, symbols, _MULTIPOLE_COLUMNS, "atomic multipoles", path)
    multipoles = _table(multipole_rows, range(6, 15), np.float64, path)
    _check_rows(cloud_rows, symbols, _CLOUD_COLUMNS, "electron cloud parameters", path)
    cloud = _table(cloud_rows, range(5, 7), np.float64, path)

    return DDEC6Analysis(
        atomic_numbers=atomic_numbers,
        positions=coordinates[:, :3] * BOHR_PER_ANGSTROM,
        net_charges=coordinates[:, 3].copy(),
        atomic_dipoles=multipoles[:, 0:3].copy(),
        atomic_quadrupoles=_traceless_tensor_components(multipoles[:, 4:9]),
        cloud_a=cloud[:, 0].copy(),
        cloud_b=cloud[:, 1].copy(),
        lattice_vectors=lattice_vectors,
    )


def read_overlap_populations(path: StrPath) -> OverlapPairs:
    """Read a Chargemol ``overlap_populations_and_contact_exchanges.txt`` file.

    Args:
        path: The file to read. The older ``overlap_populations.txt`` has the same
            leading columns and can be read too.

    Returns:
        The directed atom pairs, with zero-based atom indices.

    Raises:
        ValueError: If the file is empty or does not have the expected layout.
    """
    all_lines = _read_lines(path)
    lines = [line for line in all_lines if line.strip()]
    count = _parse_count(lines[0], path)
    if count is None:
        count = _table_length(all_lines, skip=1 + _HEADER_LINES_BEFORE_OVERLAPS)
    rows = _block(lines, 1 + _HEADER_LINES_BEFORE_OVERLAPS, count, "overlaps", path)
    if count == 0:
        return OverlapPairs(
            atom1=np.empty(0, np.int64),
            atom2=np.empty(0, np.int64),
            translations=np.empty((0, 3), np.int64),
            populations=np.empty(0, np.float64),
        )
    indices = _table(rows, range(5), np.int64, path)
    populations = _table(rows, range(5, 6), np.float64, path)[:, 0]
    if np.any(indices[:, :2] < 1):
        msg = f"{path}: atom numbers in the overlap populations must be positive"
        raise ValueError(msg)
    return OverlapPairs(
        atom1=indices[:, 0] - 1,
        atom2=indices[:, 1] - 1,
        translations=indices[:, 2:5].copy(),
        populations=populations,
    )


def _read_lines(path: StrPath) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        msg = f"{path}: the file is empty"
        raise ValueError(msg)
    return lines


def _parse_count(line: str, path: StrPath) -> int | None:
    """Parse the count on a file's first line.

    Chargemol writes the count in a five-character field, which Fortran fills with
    asterisks once the count reaches 100,000.

    Returns:
        The count, or None if it overflowed its field.
    """
    try:
        field = line.split(maxsplit=1)[0]
        if _OVERFLOW.fullmatch(field):
            return None
        count = int(field)
    except IndexError, ValueError:
        msg = f"{path}: expected a count on the first line, found {line.strip()!r}"
        raise ValueError(msg) from None
    if count < 0:
        msg = f"{path}: the count on the first line is negative"
        raise ValueError(msg)
    return count


def _table_length(lines: Sequence[str], skip: int) -> int:
    """Count the rows of the table that follows the first ``skip`` non-blank lines.

    Chargemol ends each table with a blank line, so the table runs up to the next one.
    """
    nonblank = (i for i, line in enumerate(lines) if line.strip())
    start = next(itertools.islice(nonblank, skip, None), len(lines))
    return sum(1 for _ in itertools.takewhile(str.strip, lines[start:]))


def _parse_lattice(line: str, path: StrPath) -> FloatArray:
    """Parse the unit cell from the second line of a Chargemol xyz file, in Å."""
    text = line.strip()
    if text.lower() == "nonperiodic system":
        return np.zeros((3, 3))
    _, bracket, cell = text.partition("[{")
    numbers = _NUMBER.findall(cell.partition("]")[0])
    if not bracket or len(numbers) != _CELL_NUMBERS:
        msg = (
            f"{path}: expected 'Nonperiodic system' or a jmolscript unit cell on "
            f"the second line, found {text!r}"
        )
        raise ValueError(msg)
    values = [float(n.replace("D", "E").replace("d", "e")) for n in numbers]
    return np.array(values).reshape(3, 3)


def _block(
    lines: Sequence[str], start: int, count: int, what: str, path: StrPath
) -> Sequence[str]:
    rows = lines[start : start + count]
    if len(rows) < count:
        msg = f"{path}: the file ends before all {count} rows of {what}"
        raise ValueError(msg)
    return rows


def _table[T: np.generic](
    rows: Sequence[str], columns: range, dtype: type[T], path: StrPath
) -> npt.NDArray[T]:
    """Parse whitespace-separated columns from rows, one row per table row."""
    try:
        return np.loadtxt(rows, dtype=dtype, usecols=columns, ndmin=2)
    except ValueError as error:
        first, last = columns.start + 1, columns.stop
        msg = f"{path}: could not read columns {first}-{last}: {error}"
        raise ValueError(msg) from None


def _find_title(
    lines: Sequence[str], start: int, title: str, what: str, path: StrPath
) -> int:
    """Find the first line from ``start`` on that begins with ``title``."""
    found = (
        index
        for index, line in enumerate(lines[start:], start=start)
        if line.lstrip().startswith(title)
    )
    index = next(found, None)
    if index is None:
        msg = f"{path}: missing the title of the {what} table, {title!r}"
        raise ValueError(msg)
    return index


def _check_rows(
    rows: Sequence[str], symbols: Sequence[str], columns: int, what: str, path: StrPath
) -> None:
    """Check that table rows are numbered 1..n, name the same elements, and are whole.

    Fortran prints a number too wide for its field as asterisks with no leading space,
    which joins it to the previous column and shifts the columns after it.
    """
    for number, (row, symbol) in enumerate(zip(rows, symbols, strict=True), start=1):
        fields = row.split()
        if fields[:2] != [str(number), symbol]:
            msg = (
                f"{path}: expected row {number} ({symbol}) of the {what}, "
                f"found {row.strip()!r}"
            )
            raise ValueError(msg)
        if len(fields) != columns:
            msg = (
                f"{path}: expected {columns} columns in row {number} of the {what}, "
                f"found {len(fields)}"
            )
            raise ValueError(msg)


def _traceless_tensor_components(spherical: FloatArray) -> FloatArray:
    """Convert Chargemol's quadrupole components to a traceless Cartesian tensor.

    Args:
        spherical: Columns ``Qxy, Qxz, Qyz, Q(x^2-y^2), Q(3z^2-R^2)``.

    Returns:
        Columns ``xx, xy, xz, yy, yz, zz`` of the traceless tensor.
    """
    xy, xz, yz, x2_minus_y2, three_z2_minus_r2 = spherical.T
    zz = _ONE_THIRD * three_z2_minus_r2
    xx = 0.5 * (x2_minus_y2 - zz)
    yy = 0.5 * (-x2_minus_y2 - zz)
    return np.column_stack([xx, xy, xz, yy, yz, zz])
