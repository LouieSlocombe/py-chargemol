"""Read electron-density grids without calling an external electronic-structure tool.

Coordinates and densities returned by these readers are in bohr and electrons
per cubic bohr. Cube files store the last grid index fastest; VASP and XSF store
the first grid index fastest. VASP PAW valence densities must come from AECCAR2,
with CHGCAR supplying the electron-count integration correction.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from py_chargemol.ddec6 import StrPath
from py_chargemol.density.grid import (
    NSHELLS,
    ZERO,
    DensityGrid,
    atomic_samples,
    sum_radial_weights,
)
from py_chargemol.density.references import core_reference_density
from py_chargemol.elements import symbol_to_number
from py_chargemol.units import BOHR_PER_ANGSTROM

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_CUBE_AXIS_FIELDS = 4
_CUBE_ATOM_FIELDS = 5
_XSF_ATOM_FIELDS = 4
_DIMENSIONS = 3
_SLICE_LINES = 4096


def _bound_spin_density(grid: DensityGrid, spin: FloatArray) -> FloatArray:
    """Bound cube/XSF spin against electron density before the core iterator."""
    total = _provisional_total_density(grid)
    magnitude = (
        np.abs(spin) if spin.ndim == _DIMENSIONS else np.linalg.norm(spin, axis=-1)
    )
    scale = np.zeros_like(total)
    np.divide(total, magnitude, out=scale, where=(total >= ZERO) & (magnitude >= ZERO))
    scale = np.minimum(scale, 1.0)
    return spin * (scale if spin.ndim == _DIMENSIONS else scale[..., None])


def _provisional_total_density(grid: DensityGrid) -> FloatArray:
    if grid.core_density is not None:
        return np.asarray(grid.density + grid.core_density, dtype=np.float64)
    if not np.any(grid.core_electrons):
        return grid.density
    return np.asarray(grid.density + _reference_core_density(grid), dtype=np.float64)


def _reference_core_density(grid: DensityGrid) -> FloatArray:
    """Sample reference cores before their integrated populations are corrected."""
    radial = np.zeros((grid.natoms, NSHELLS))
    for atom, (number, count) in enumerate(
        zip(grid.atomic_numbers, np.asarray(grid.core_electrons), strict=True)
    ):
        if count:
            radial[atom] = core_reference_density(int(number), int(count))
    core = sum_radial_weights(radial, atomic_samples(grid), grid.density.size)
    return np.asarray(core.reshape(grid.density.shape), dtype=np.float64)


@dataclass(slots=True)
class _Text:
    path: Path
    lines: list[str]
    index: int = 0

    @classmethod
    def open(cls, path: StrPath) -> _Text:
        source = Path(path)
        return cls(source, source.read_text(encoding="utf-8").splitlines())

    def line(self, *, skip_empty: bool = True) -> str:
        while self.index < len(self.lines):
            line = self.lines[self.index].strip()
            self.index += 1
            if line or not skip_empty:
                return line
        msg = f"{self.path}: unexpected end of file"
        raise ValueError(msg)

    def values(self, count: int) -> FloatArray:
        values = np.empty(count)
        filled = 0
        while filled < count:
            remaining = count - filled
            lines = max(1, min(_SLICE_LINES, remaining // len(self.line().split())))
            start = self.index - 1
            fields = _fields(self.lines[start : start + lines])
            # Halve slices that run past the block so overflow is caught per line.
            while len(fields) > remaining and lines > 1:
                lines //= 2
                fields = _fields(self.lines[start : start + lines])
            chunk = _parse(fields, self.path)
            if len(chunk) > remaining:
                msg = f"{self.path}: too many values in the density block"
                raise ValueError(msg)
            values[filled : filled + len(chunk)] = chunk
            filled += len(chunk)
            self.index = min(start + lines, len(self.lines))
        return values


@dataclass(frozen=True, slots=True)
class _Grid:
    atomic_numbers: IntArray
    positions: FloatArray
    density: FloatArray
    origin: FloatArray
    steps: FloatArray
    spin_density: FloatArray | None = None


def _fields(lines: list[str]) -> list[str]:
    return " ".join(lines).replace("D", "E").replace("d", "e").split()


def _floats(fields: list[str], path: Path) -> FloatArray:
    return _parse(_fields(fields), path)


def _parse(fields: list[str], path: Path) -> FloatArray:
    try:
        values = np.fromiter(map(float, fields), np.float64, count=len(fields))
    except ValueError as error:
        msg = f"{path}: invalid numeric value: {error}"
        raise ValueError(msg) from error
    if not np.all(np.isfinite(values)):
        msg = f"{path}: density input contains nonfinite values"
        raise ValueError(msg)
    return values


def _dimensions(fields: list[str], path: Path) -> tuple[int, int, int]:
    try:
        a, b, c = (int(v) for v in fields)
    except ValueError as error:
        msg = f"{path}: expected three integer grid dimensions"
        raise ValueError(msg) from error
    if min(a, b, c) < 1:
        msg = f"{path}: grid dimensions must be positive"
        raise ValueError(msg)
    return a, b, c


def _same_grid(first: _Grid, second: _Grid, path: StrPath) -> None:
    if (
        first.density.shape != second.density.shape
        or not np.array_equal(first.atomic_numbers, second.atomic_numbers)
        or first.positions.shape != second.positions.shape
        or not np.allclose(first.positions, second.positions, rtol=0, atol=1e-8)
        or not np.allclose(first.origin, second.origin, rtol=0, atol=1e-8)
        or not np.allclose(first.steps, second.steps, rtol=0, atol=1e-8)
    ):
        msg = f"{path}: atom positions, lattice vectors, and grid must match"
        raise ValueError(msg)


def read_cube(  # noqa: PLR0913 -- File format choices and physical metadata.
    path: StrPath,
    *,
    density_kind: Literal["total", "valence"] = "total",
    core_electrons: npt.ArrayLike | None = None,
    total_density_file: StrPath | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    net_charge: float = 0.0,
    length_unit: Literal["bohr", "angstrom"] | None = None,
    density_unit: Literal["bohr^-3", "angstrom^-3"] = "bohr^-3",
) -> DensityGrid:
    """Read a scalar Gaussian cube electron-density file.

    Positive voxel counts imply bohr coordinates, negative counts angstrom;
    ``length_unit`` overrides this convention. Density units are independent of
    coordinate units and default to electrons/bohr³. Orbital and multi-dataset
    cube files are rejected. ``density_kind='valence'`` requires explicit core
    electron counts: the cube's nuclear-charge column is not reliable metadata
    for inferring a pseudopotential. Valence density is capped at 0.03 electrons
    per voxel, as in Chargemol. An optional matching total-density cube supplies
    the core density by subtraction after that cap. Total-only cubes already
    include every electron and require zero core-electron counts.
    """
    if density_kind not in {"total", "valence"}:
        msg = "density_kind must be 'total' or 'valence'"
        raise ValueError(msg)
    if density_kind == "valence" and core_electrons is None:
        msg = "valence cube input requires explicit core_electrons for every atom"
        raise ValueError(msg)
    if (
        density_kind == "total"
        and core_electrons is not None
        and np.any(np.asarray(core_electrons) != 0)
    ):
        msg = (
            "total-density cubes already include core electrons; "
            "core_electrons must be zero"
        )
        raise ValueError(msg)
    if total_density_file is not None and density_kind != "valence":
        msg = "total_density_file is only meaningful with a valence cube"
        raise ValueError(msg)
    raw = _cube(path, length_unit, density_unit)
    density = np.maximum(raw.density, 0.0)
    if density_kind == "valence":
        voxel_volume = abs(float(np.linalg.det(raw.steps)))
        if voxel_volume == 0:
            msg = f"{path}: cube voxel vectors must define a nonsingular grid"
            raise ValueError(msg)
        density = np.minimum(density, 0.03 / voxel_volume)
    core = None
    if total_density_file is not None:
        total = _cube(total_density_file, length_unit, density_unit)
        _same_grid(raw, total, total_density_file)
        core = np.maximum(total.density - density, 0.0)
    return DensityGrid(
        atomic_numbers=raw.atomic_numbers,
        positions=raw.positions,
        density=density,
        origin=raw.origin,
        steps=raw.steps,
        periodic=periodic,
        core_electrons=(
            None if core_electrons is None else np.asarray(core_electrons, dtype=float)
        ),
        core_density=core,
        net_charge=net_charge,
    )


def _cube(
    path: StrPath,
    length_unit: Literal["bohr", "angstrom"] | None,
    density_unit: Literal["bohr^-3", "angstrom^-3"],
) -> _Grid:
    text = _Text.open(path)
    text.line(skip_empty=False)
    text.line(skip_empty=False)
    fields = text.line().split()
    if len(fields) not in {4, 5}:
        msg = f"{path}: expected atom count and origin in cube header"
        raise ValueError(msg)
    natoms = int(fields[0])
    if natoms < 1 or (len(fields) == _CUBE_ATOM_FIELDS and int(fields[4]) != 1):
        msg = f"{path}: expected a scalar density cube with a positive atom count"
        raise ValueError(msg)
    origin = _floats(fields[1:4], text.path)
    axes = [text.line().split() for _ in range(3)]
    if any(len(axis) != _CUBE_AXIS_FIELDS for axis in axes):
        msg = f"{path}: expected a voxel count and three components per cube grid axis"
        raise ValueError(msg)
    counts = np.array([int(axis[0]) for axis in axes])
    if np.any(counts == 0) or not (np.all(counts > 0) or np.all(counts < 0)):
        msg = f"{path}: cube voxel counts must be nonzero and have a consistent sign"
        raise ValueError(msg)
    shape = tuple(int(v) for v in np.abs(counts))
    steps = np.stack([_floats(axis[1:], text.path) for axis in axes])
    numbers = np.empty(natoms, dtype=np.int64)
    positions = np.empty((natoms, 3))
    for i in range(natoms):
        fields = text.line().split()
        if len(fields) != _CUBE_ATOM_FIELDS:
            msg = f"{path}: expected five fields in cube atom row {i + 1}"
            raise ValueError(msg)
        numbers[i] = int(fields[0])
        positions[i] = _floats(fields[2:], text.path)
    unit = length_unit or ("angstrom" if counts[0] < 0 else "bohr")
    if unit not in {"bohr", "angstrom"} or density_unit not in {
        "bohr^-3",
        "angstrom^-3",
    }:
        msg = "unsupported cube units"
        raise ValueError(msg)
    scale = BOHR_PER_ANGSTROM if unit == "angstrom" else 1.0
    density_scale = BOHR_PER_ANGSTROM**-3 if density_unit == "angstrom^-3" else 1.0
    density = text.values(int(np.prod(np.abs(counts)))).reshape(shape) * density_scale
    if any(line.strip() for line in text.lines[text.index :]):
        msg = f"{path}: unexpected data after the scalar cube density"
        raise ValueError(msg)
    return _Grid(numbers, positions * scale, density, origin * scale, steps * scale)


def read_vasp(
    directory: StrPath,
    *,
    core_electrons: npt.ArrayLike | None = None,
    net_charge: float | None = None,
) -> DensityGrid:
    """Read VASP AECCAR2, CHGCAR/CHG, optional AECCAR0, and POTCAR.

    POTCAR supplies element identities and ZVAL core counts. It may be omitted
    only when explicit core counts and a VASP 5 element-symbol header are supplied.
    AECCAR2 is the reconstructed valence density; CHGCAR alone is insufficient
    for PAW DDEC analysis. Its pseudodensity is retained for the per-atom
    integration correction. Net charge defaults to the pseudodensity integral.
    Collinear and noncollinear spin blocks are reconstructed from CHGCAR/CHG;
    noncollinear vectors use the optional INCAR SAXIS orientation.
    """
    directory = Path(directory)
    species = None
    potcar_core = np.empty(0)
    if (directory / "POTCAR").is_file():
        species, potcar_core = _potcar(directory / "POTCAR")
    elif core_electrons is None:
        msg = "VASP input requires POTCAR or explicit core_electrons"
        raise ValueError(msg)
    valence, counts = _vasp_grid(directory / "AECCAR2", species)
    core_counts = (
        np.repeat(potcar_core, counts)
        if core_electrons is None
        else np.asarray(core_electrons, dtype=float)
    )
    pseudo_path = next(
        (
            directory / name
            for name in ("CHGCAR", "CHGCAR_noncollinear", "CHG", "CHG_noncollinear")
            if (directory / name).is_file()
        ),
        None,
    )
    if pseudo_path is None:
        msg = f"{directory}: CHGCAR or CHG is required for VASP integration corrections"
        raise FileNotFoundError(msg)
    pseudo, _ = _vasp_grid(pseudo_path, species)
    _same_grid(valence, pseudo, pseudo_path)
    core = None
    if (directory / "AECCAR0").is_file():
        core_data, _ = _vasp_grid(directory / "AECCAR0", species)
        _same_grid(valence, core_data, directory / "AECCAR0")
        core = np.maximum(core_data.density, 0.0)
    voxel_volume = abs(float(np.linalg.det(valence.steps)))
    density = np.clip(valence.density, 0.0, 0.03 / voxel_volume)
    if net_charge is None:
        net_charge = float(
            np.sum(valence.atomic_numbers)
            - np.sum(core_counts)
            - np.sum(pseudo.density) * voxel_volume
        )
    grid = DensityGrid(
        atomic_numbers=valence.atomic_numbers,
        positions=valence.positions,
        density=density,
        origin=valence.origin,
        steps=valence.steps,
        periodic=(True, True, True),
        core_electrons=core_counts,
        core_density=core,
        net_charge=net_charge,
        pseudodensity=pseudo.density,
    )
    if pseudo.spin_density is not None:
        grid.spin_density = _vasp_spin(pseudo.spin_density, grid, directory)
    return grid


def _vasp_spin(raw: FloatArray, grid: DensityGrid, directory: Path) -> FloatArray:
    if grid.core_density is None:
        grid.core_density = _reference_core_density(grid)
    total = grid.density + grid.core_density
    pseudo = np.asarray(grid.pseudodensity)
    raw = (
        raw @ _spin_rotation(directory).T
        if raw.ndim == total.ndim + 1
        else raw[..., None]
    )
    enhancement = np.minimum(
        np.abs(np.divide(total, pseudo, out=np.zeros_like(total), where=pseudo > 0)), 10
    )
    spin = raw * enhancement[..., None]
    positive = np.maximum(raw, 0).sum(axis=(0, 1, 2))
    negative = -np.minimum(raw, 0).sum(axis=(0, 1, 2))
    for iteration in range(4):
        if iteration:
            actual_positive = np.maximum(spin, 0).sum(axis=(0, 1, 2))
            actual_negative = -np.minimum(spin, 0).sum(axis=(0, 1, 2))
            positive_scale = np.divide(
                positive,
                actual_positive,
                out=np.ones_like(positive),
                where=actual_positive * grid.pixel_volume > ZERO,
            )
            negative_scale = np.divide(
                negative,
                actual_negative,
                out=np.ones_like(negative),
                where=actual_negative * grid.pixel_volume > ZERO,
            )
            spin *= np.where(spin > 0, positive_scale, negative_scale)
        magnitude = np.linalg.norm(spin, axis=-1)
        spin *= np.minimum(
            np.divide(total, magnitude, out=np.ones_like(total), where=magnitude > 0), 1
        )[..., None]
    return np.asarray(spin[..., 0] if spin.shape[-1] == 1 else spin, dtype=np.float64)


def _spin_rotation(directory: Path) -> FloatArray:
    path = directory / "INCAR"
    axis = np.array([0.0, 0.0, 1.0])
    if path.is_file():
        match = re.search(
            r"(?im)(?:^|;)\s*SAXIS\s*=\s*([^!#;\n]+)",
            path.read_text(encoding="utf-8"),
        )
        if match:
            axis = _floats(match[1].split(), path)
    if axis.shape != (3,) or np.linalg.norm(axis) == 0:
        msg = f"{path}: SAXIS must contain three finite values and be nonzero"
        raise ValueError(msg)
    alpha = np.arctan2(axis[1], axis[0])
    beta = np.arctan2(np.linalg.norm(axis[:2]), axis[2])
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    return np.array([[cb * ca, -sa, sb * ca], [cb * sa, ca, sb * sa], [-sb, 0, cb]])


def _potcar(path: Path) -> tuple[IntArray, FloatArray]:
    records = path.read_text(encoding="utf-8").split("End of Dataset")
    numbers, core = [], []
    for record in records:
        if not record.strip():
            continue
        element = re.search(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", record)
        valence = re.search(r"ZVAL\s*=\s*([-+\d.EeDd]+)", record)
        if element is None or valence is None:
            msg = f"{path}: each POTCAR dataset must contain VRHFIN and ZVAL"
            raise ValueError(msg)
        number = symbol_to_number(element[1])
        count = _floats([valence[1]], path)[0]
        if count < 0 or count > number or count != round(count):
            msg = f"{path}: ZVAL must be an integer between zero and atomic number"
            raise ValueError(msg)
        numbers.append(number)
        core.append(number - count)
    if not numbers:
        msg = f"{path}: no POTCAR datasets found"
        raise ValueError(msg)
    return np.array(numbers, dtype=np.int64), np.array(core)


def _vasp_grid(path: Path, species: IntArray | None) -> tuple[_Grid, IntArray]:
    text = _Text.open(path)
    text.line(skip_empty=False)
    scaling = _floats(text.line().split(), path)
    vectors = np.stack([_floats(text.line().split(), path) for _ in range(3)])
    if vectors.shape != (3, 3) or len(scaling) not in {1, 3}:
        msg = f"{path}: invalid VASP lattice vectors or scale"
        raise ValueError(msg)
    base_volume = abs(float(np.linalg.det(vectors)))
    if base_volume == 0:
        msg = f"{path}: singular VASP lattice"
        raise ValueError(msg)
    if len(scaling) == 1 and scaling[0] < 0:
        scaling = np.array([(-scaling[0] / base_volume) ** (1 / 3)])
    if np.any(scaling <= 0):
        msg = f"{path}: VASP scale must be positive, or one negative target volume"
        raise ValueError(msg)
    lattice = vectors * scaling * BOHR_PER_ANGSTROM
    volume = abs(float(np.linalg.det(lattice)))
    fields = text.line().split()
    if not all(field.isdigit() for field in fields):
        named_species = np.array([symbol_to_number(v) for v in fields], dtype=np.int64)
        if species is not None and not np.array_equal(species, named_species):
            msg = f"{path}: element order does not match POTCAR"
            raise ValueError(msg)
        species = named_species
        fields = text.line().split()
    counts = np.array([int(v) for v in fields], dtype=np.int64)
    if species is None or len(species) != len(counts) or np.any(counts < 1):
        msg = f"{path}: atom counts must match elements from POTCAR or header"
        raise ValueError(msg)
    mode = text.line().lower()
    if mode.startswith("s"):
        mode = text.line().lower()
    numbers = np.repeat(species, counts)
    positions = np.stack([_floats(text.line().split()[:3], path) for _ in numbers])
    if positions.shape != (len(numbers), 3):
        msg = f"{path}: expected three coordinates per atom"
        raise ValueError(msg)
    if mode.startswith("d"):
        positions = positions @ lattice
    elif mode.startswith(("c", "k")):
        positions *= scaling * BOHR_PER_ANGSTROM
    else:
        msg = f"{path}: coordinates must be Direct or Cartesian"
        raise ValueError(msg)
    shape = _dimensions(text.line().split(), path)
    density = text.values(int(np.prod(shape))).reshape(shape, order="F") / volume
    spin = _vasp_spin_blocks(text, shape, volume)
    return _Grid(
        numbers,
        positions,
        density,
        np.zeros(3),
        lattice / np.array(shape)[:, None],
        spin,
    ), counts


def _vasp_spin_blocks(
    text: _Text, shape: tuple[int, int, int], volume: float
) -> FloatArray | None:
    blocks = []
    while text.index < len(text.lines):
        fields = text.lines[text.index].split()
        text.index += 1
        if fields == [str(value) for value in shape]:
            blocks.append(
                text.values(int(np.prod(shape))).reshape(shape, order="F") / volume
            )
        elif fields[:1] == ["augmentation"]:
            text.values(int(fields[-1]))
    if len(blocks) not in {0, 1, 3}:
        msg = f"{text.path}: expected zero, one, or three VASP spin-density blocks"
        raise ValueError(msg)
    if not blocks:
        return None
    return blocks[0] if len(blocks) == 1 else np.stack(blocks, axis=-1)


def read_xsf(
    path: StrPath,
    *,
    core_electrons: npt.ArrayLike,
    net_charge: float = 0.0,
    density_unit: Literal["bohr^-3", "angstrom^-3"] = "bohr^-3",
) -> DensityGrid:
    """Read XSF scalar, collinear, or noncollinear ``rho:spin_1/2/3/4`` grids.

    XSF coordinates are in angstrom. Density values default to electrons/bohr³,
    as in Chargemol's XSF interface. The terminal grid planes are omitted because
    XSF stores both ends of each cell axis. Core counts are always explicit.
    """
    grids, periodic = _xsf_grids(path)
    spin = None
    first = next(iter(grids.values()))
    spin_names = {"rho:spin_1", "rho:spin_2"}
    if set(grids) in (spin_names, spin_names | {"rho:spin_3", "rho:spin_4"}):
        second = grids["rho:spin_2"]
        first = grids["rho:spin_1"]
        _same_grid(first, second, path)
        density = first.density + second.density
        spin = first.density - second.density
        if "rho:spin_3" in grids:
            _same_grid(first, grids["rho:spin_3"], path)
            _same_grid(first, grids["rho:spin_4"], path)
            spin = np.stack(
                [
                    2 * grids["rho:spin_3"].density,
                    -2 * grids["rho:spin_4"].density,
                    spin,
                ],
                axis=-1,
            )
    elif len(grids) == 1:
        density = first.density
    else:
        msg = f"{path}: expected one scalar XSF density or two/four rho:spin grids"
        raise ValueError(msg)
    if density_unit not in {"bohr^-3", "angstrom^-3"}:
        msg = "unsupported XSF density units"
        raise ValueError(msg)
    scale = BOHR_PER_ANGSTROM**-3 if density_unit == "angstrom^-3" else 1.0
    grid = DensityGrid(
        atomic_numbers=first.atomic_numbers,
        positions=first.positions,
        density=np.maximum(density, 0.0) * scale,
        origin=first.origin,
        steps=first.steps,
        periodic=periodic,
        core_electrons=np.asarray(core_electrons, dtype=float),
        net_charge=net_charge,
        spin_density=None if spin is None else spin * scale,
    )
    if grid.spin_density is not None:
        grid.spin_density = _bound_spin_density(grid, grid.spin_density)
    return grid


def _xsf_grids(
    path: StrPath,
) -> tuple[dict[str, _Grid], tuple[bool, bool, bool]]:
    text = _Text.open(path)
    text.lines = [line.split("#", 1)[0].strip() for line in text.lines]
    text.lines = [line for line in text.lines if line]
    periodic = (False, False, False)
    numbers: list[int] = []
    coordinates: list[FloatArray] = []
    grids: dict[str, _Grid] = {}
    primitive = None
    while text.index < len(text.lines):
        line = text.line().lower()
        if line in {"crystal", "slab", "polymer"}:
            periodic = (True, line != "polymer", line == "crystal")
        elif line == "primvec":
            primitive = np.stack(
                [_floats(text.line().split(), text.path) for _ in range(3)]
            )
        elif line in {"primcoord", "atoms"}:
            count = int(text.line().split()[0]) if line == "primcoord" else None
            while text.index < len(text.lines) and (
                count is None or len(numbers) < count
            ):
                fields = text.lines[text.index].split()
                if count is None and len(fields) < _XSF_ATOM_FIELDS:
                    break
                text.index += 1
                numbers.append(
                    int(fields[0])
                    if fields[0].isdigit()
                    else symbol_to_number(fields[0])
                )
                coordinates.append(_floats(fields[1:4], text.path))
        elif line.startswith("begin_datagrid_3d"):
            name = line.removeprefix("begin_datagrid_3d").lstrip("_")
            if name in grids:
                msg = f"{path}: duplicate XSF density grid {name!r}"
                raise ValueError(msg)
            shape = _dimensions(text.line().split(), text.path)
            if min(shape) <= 1:
                msg = f"{path}: XSF grids need at least two points on each axis"
                raise ValueError(msg)
            origin = _floats(text.line().split(), text.path) * BOHR_PER_ANGSTROM
            vectors = np.stack(
                [_floats(text.line().split(), text.path) for _ in range(3)]
            )
            if (
                primitive is not None
                and any(periodic)
                and not np.allclose(primitive, vectors, atol=1e-8, rtol=0)
            ):
                msg = (
                    f"{path}: XSF primitive vectors and density grid vectors must match"
                )
                raise ValueError(msg)
            density = text.values(int(np.prod(shape))).reshape(shape, order="F")[
                :-1, :-1, :-1
            ]
            steps = vectors * BOHR_PER_ANGSTROM / (np.array(shape) - 1)[:, None]
            grids[name] = _Grid(
                np.array(numbers, dtype=np.int64),
                np.array(coordinates) * BOHR_PER_ANGSTROM,
                density,
                origin,
                steps,
            )
    if not grids or not numbers:
        msg = f"{path}: XSF input must contain atomic coordinates and a 3D density grid"
        raise ValueError(msg)
    return grids, periodic
