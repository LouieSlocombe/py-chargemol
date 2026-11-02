"""Discover density inputs and interpret the relevant Chargemol job controls."""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from py_chargemol.ddec6 import IntArray, StrPath
from py_chargemol.density.grid import DensityGrid
from py_chargemol.density.readers import (
    _bound_spin_density,
    _cube,
    _Grid,
    _potcar,
    _same_grid,
    read_cube,
    read_vasp,
    read_xsf,
)
from py_chargemol.elements import symbol_to_number

type DensityFormat = Literal["auto", "cube", "vasp", "xsf", "wfx"]
type DensityKind = Literal["total", "valence"]

_PERIODICITY_TAG = "periodicity along a, b, and c vectors"
_VASP_FILES = {"AECCAR0", "AECCAR2", "CHGCAR", "CHG", "POTCAR"}
_SUFFIXES = {".cube": "cube", ".cub": "cube", ".wfx": "wfx", ".xsf": "xsf"}
_AXES = 3
_MAX_ATOMIC_NUMBER = 109
_ATOM_FIELDS = 4
_CUBE_UNITS: dict[
    str | None,
    tuple[Literal["bohr", "angstrom"] | None, Literal["bohr^-3", "angstrom^-3"]],
] = {
    "onetep": ("bohr", "angstrom^-3"),
    "e_per_angs3": ("angstrom", "angstrom^-3"),
}
_BOOLEANS = {
    ".true.": True,
    "true": True,
    "t": True,
    ".false.": False,
    "false": False,
    "f": False,
}
_DEFAULT_CORE_COUNTS = np.repeat([0, 2, 10, 18, 36, 54, 86], [2, 8, 8, 18, 18, 32, 23])
_SUPPORTED_TAGS = {
    _PERIODICITY_TAG,
    "net charge",
    "input filename",
    "number of core electrons",
    "charge type",
    "density format",
    "print atomic densities",
    "compute bos",
}


@dataclass(frozen=True, slots=True)
class JobControl:
    """Physical metadata read from ``job_control.txt``.

    The external atomic-density directory is deliberately unused: reference
    ions are bundled package resources. Charge method selection is exposed for
    callers, while :func:`read_density` only uses density-related settings.
    """

    periodic: tuple[bool, bool, bool] | None = None
    net_charge: float | None = None
    input_filename: str | None = None
    core_electrons: dict[int, int] = field(default_factory=dict)
    charge_type: str | None = None
    density_format: str | None = None
    print_atomic_densities: bool | None = None
    compute_bos: bool | None = None


def _control_tags(path: Path) -> dict[str, str]:
    """Read supported tags once, rejecting unmatched or repeated delimiters."""
    tags: dict[str, str] = {}
    opening: tuple[str, int] | None = None
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"<(/?)([^/<>]+)>", text):
        key = " ".join(match[2].lower().split())
        if key not in _SUPPORTED_TAGS:
            continue
        if not match[1]:
            if opening is not None:
                msg = f"{path}: missing closing tag for <{opening[0]}>"
                raise ValueError(msg)
            if key in tags:
                msg = f"{path}: duplicate <{key}> tag"
                raise ValueError(msg)
            opening = key, match.end()
        elif opening is None or key != opening[0]:
            msg = f"{path}: unexpected closing tag </{key}>"
            raise ValueError(msg)
        else:
            tags[key] = text[opening[1] : match.start()].strip()
            opening = None
    if opening is not None:
        msg = f"{path}: missing closing tag for <{opening[0]}>"
        raise ValueError(msg)
    return tags


def read_job_control(path: StrPath) -> JobControl:
    """Read optional physical metadata from a Chargemol job-control file.

    Unknown tags, including original executable and reference-density paths,
    are ignored. Duplicate or malformed supported tags raise ``ValueError``.
    Core-electron rows contain atomic number followed by core-electron count.
    """
    path = Path(path)
    tags = _control_tags(path)
    periodic = None
    if _PERIODICITY_TAG in tags:
        values = _tokens(tags[_PERIODICITY_TAG])
        if len(values) != _AXES or any(
            value.lower() not in _BOOLEANS for value in values
        ):
            msg = f"{path}: periodicity must contain three true/false values"
            raise ValueError(msg)
        periodic = (
            _BOOLEANS[values[0].lower()],
            _BOOLEANS[values[1].lower()],
            _BOOLEANS[values[2].lower()],
        )
    net_charge = None
    if "net charge" in tags:
        try:
            (value,) = _tokens(tags["net charge"])
            net_charge = float(value.replace("D", "E").replace("d", "e"))
        except ValueError as error:
            msg = f"{path}: net charge must be one finite number"
            raise ValueError(msg) from error
        if not np.isfinite(net_charge):
            msg = f"{path}: net charge must be one finite number"
            raise ValueError(msg)
    cores: dict[int, int] = {}
    for line in tags.get("number of core electrons", "").splitlines():
        fields = _tokens(line)
        if not fields:
            continue
        try:
            number, count = map(int, fields)
        except ValueError as error:
            msg = f"{path}: each core-electron row must contain atomic number and count"
            raise ValueError(msg) from error
        if (
            not 1 <= number <= _MAX_ATOMIC_NUMBER
            or not 0 <= count <= number
            or number in cores
        ):
            msg = (
                f"{path}: invalid or repeated core-electron specification for {number}"
            )
            raise ValueError(msg)
        cores[number] = count
    filename = tags.get("input filename")
    if filename is not None:
        filename = filename.strip().strip("\"'")
        if not filename or "\n" in filename:
            msg = f"{path}: input filename must contain one filename or job label"
            raise ValueError(msg)
    method = tags.get("charge type")
    units = tags.get("density format")
    return JobControl(
        periodic=periodic,
        net_charge=net_charge,
        input_filename=filename,
        core_electrons=cores,
        charge_type=None if method is None else method.lower(),
        density_format=None if units is None else units.lower(),
        print_atomic_densities=_boolean_control(path, tags, "print atomic densities"),
        compute_bos=_boolean_control(path, tags, "compute bos"),
    )


def _tokens(text: str) -> list[str]:
    return " ".join(
        line.split("!", 1)[0].split("#", 1)[0] for line in text.splitlines()
    ).split()


def _boolean_control(path: Path, tags: dict[str, str], key: str) -> bool | None:
    if key not in tags:
        return None
    values = _tokens(tags[key])
    if len(values) != 1 or values[0].lower() not in _BOOLEANS:
        msg = f"{path}: {key} must contain one true/false value"
        raise ValueError(msg)
    return _BOOLEANS[values[0].lower()]


def _file_format(path: Path) -> str:
    if path.name.upper() in _VASP_FILES:
        return "vasp"
    try:
        return _SUFFIXES[path.suffix.lower()]
    except KeyError:
        msg = f"{path}: cannot infer density format; specify cube, xsf, wfx, or vasp"
        raise ValueError(msg) from None


def _discover(directory: Path, kind: str, control: JobControl) -> Path:
    if control.input_filename is not None:
        named = directory / control.input_filename
        if named.is_file() and (kind == "auto" or _file_format(named) == kind):
            return named
        if named.suffix.lower() in _SUFFIXES and not named.exists():
            msg = f"{directory / 'job_control.txt'}: input file does not exist: {named}"
            raise FileNotFoundError(msg)
    if kind in {"auto", "vasp"} and (directory / "AECCAR2").is_file():
        return directory / "AECCAR2"
    if kind in {"auto", "cube"}:
        for name in ("valence_density.cube", "total_density.cube"):
            if (directory / name).is_file():
                return directory / name
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in _SUFFIXES
        and (kind == "auto" or _SUFFIXES[path.suffix.lower()] == kind)
        and not path.name.lower().startswith("spin_density")
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        msg = (
            f"{directory}: multiple density inputs found; choose a file explicitly: "
            + ", ".join(path.name for path in candidates)
        )
        raise ValueError(msg)
    msg = (
        f"{directory}: no {kind} density input found "
        "(expected cube, XSF, WFX, or VASP AECCAR2)"
    )
    raise FileNotFoundError(msg)


def _cube_numbers(path: Path) -> IntArray:
    with path.open(encoding="utf-8") as source:
        try:
            next(source)
            next(source)
            count = int(next(source).split()[0])
            for _ in range(3):
                next(source)
            return np.asarray(
                [int(next(source).split()[0]) for _ in range(count)], dtype=np.int64
            )
        except (StopIteration, ValueError, IndexError) as error:
            msg = f"{path}: invalid cube atom header"
            raise ValueError(msg) from error


def _xsf_numbers(path: Path) -> IntArray:
    # Read only the geometry prefix; avoid loading the potentially large grid twice.
    with path.open(encoding="utf-8") as source:
        lines = (line.split("#", 1)[0].strip() for line in source)
        for line in lines:
            if line.upper() == "PRIMCOORD":
                try:
                    header = next(value for value in lines if value)
                    count = int(header.split()[0])
                    atom_lines = [
                        next(value for value in lines if value) for _ in range(count)
                    ]
                except (StopIteration, ValueError, IndexError) as error:
                    msg = f"{path}: invalid XSF atom header"
                    raise ValueError(msg) from error
                return np.asarray(
                    [_element(row.split()[0]) for row in atom_lines], dtype=np.int64
                )
            if line.upper() == "ATOMS":
                numbers = []
                for row in lines:
                    fields = row.split()
                    if not fields:
                        continue
                    if len(fields) < _ATOM_FIELDS or fields[0].upper().startswith(
                        "BEGIN"
                    ):
                        break
                    numbers.append(_element(fields[0]))
                return np.asarray(numbers, dtype=np.int64)
    msg = f"{path}: missing XSF atomic coordinates"
    raise ValueError(msg)


def _element(value: str) -> int:
    return int(value) if value.isdigit() else symbol_to_number(value)


def _vasp_numbers(path: Path) -> IntArray:
    with (path.parent / "AECCAR2").open(encoding="utf-8") as source:
        try:
            for _ in range(5):
                next(source)
            fields = next(source).split()
            if all(value.isdigit() for value in fields):
                numbers, _ = _potcar(path.parent / "POTCAR")
            else:
                numbers = np.asarray(
                    [symbol_to_number(value) for value in fields], dtype=np.int64
                )
                fields = next(source).split()
            counts = np.asarray([int(value) for value in fields], dtype=np.int64)
            return np.repeat(numbers, counts).astype(np.int64)
        except (StopIteration, ValueError) as error:
            msg = f"{path}: invalid VASP species/count header"
            raise ValueError(msg) from error


def _cores(
    path: Path, kind: str, control: JobControl, explicit: npt.ArrayLike | None
) -> npt.ArrayLike | None:
    if explicit is not None or not control.core_electrons:
        return explicit
    if kind == "cube":
        numbers = _cube_numbers(path)
    elif kind == "xsf":
        numbers = _xsf_numbers(path)
    elif kind == "vasp":
        numbers = _vasp_numbers(path)
    else:
        return None
    if np.any((numbers < 1) | (numbers > _MAX_ATOMIC_NUMBER)):
        msg = f"{path}: atomic numbers must lie in 1-109"
        raise ValueError(msg)
    return np.asarray(
        [
            control.core_electrons.get(
                int(number), int(_DEFAULT_CORE_COUNTS[number - 1])
            )
            for number in numbers
        ]
    )


def _cube_spin(
    grid: DensityGrid,
    directory: Path,
    length_unit: Literal["bohr", "angstrom"] | None,
    density_unit: Literal["bohr^-3", "angstrom^-3"],
) -> DensityGrid:
    scalar = directory / "spin_density.cube"
    components = [directory / f"spin_density_{axis}.cube" for axis in "xyz"]
    if scalar.is_file():
        paths = [scalar]
    elif any(path.is_file() for path in components):
        if not all(path.is_file() for path in components):
            msg = (
                f"{directory}: noncollinear spin requires all three "
                "spin_density_x/y/z.cube files"
            )
            raise FileNotFoundError(msg)
        paths = components
    else:
        return grid
    geometry = _Grid(
        grid.atomic_numbers, grid.positions, grid.density, grid.origin, grid.steps
    )
    densities = []
    for path in paths:
        raw = _cube(path, length_unit, density_unit)
        _same_grid(geometry, raw, path)
        densities.append(raw.density)
    spin = densities[0] if len(densities) == 1 else np.stack(densities, axis=-1)
    return replace(grid, spin_density=_bound_spin_density(grid, spin))


def _read_wavefunction(
    path: Path,
    periodic: tuple[bool, bool, bool] | None,
    core_electrons: npt.ArrayLike | None,
) -> DensityGrid:
    from py_chargemol.density.wfx import read_wfx  # noqa: PLC0415

    grid = read_wfx(path)
    if periodic is not None and periodic != grid.periodic:
        msg = "periodic override must agree with the WFX translation vectors"
        raise ValueError(msg)
    if core_electrons is not None and not np.array_equal(
        core_electrons, np.asarray(grid.core_electrons)
    ):
        msg = "core_electrons override must agree with the WFX analytic core counts"
        raise ValueError(msg)
    return grid


def read_density(  # noqa: PLR0913
    source: StrPath,
    *,
    format: DensityFormat = "auto",  # noqa: A002 - public format-selection keyword
    periodic: tuple[bool, bool, bool] | None = None,
    net_charge: float | None = None,
    core_electrons: npt.ArrayLike | None = None,
    density_kind: DensityKind | None = None,
) -> DensityGrid:
    """Read a density file or discover one in a calculation directory.

    Optional ``job_control.txt`` supplies periodicity, net charge, input filename,
    core-electron counts by atomic number, and cube density units. Explicit
    keyword arguments take precedence. VASP and WFX infer metadata when no
    override is supplied. Cube periodicity defaults to nonperiodic, and a
    ``valence_density.cube`` requires explicit or job-control core counts.

    A matching ``total_density.cube`` beside a valence cube supplies its core
    density. Collinear or three-component spin cubes are discovered there too.
    Ambiguous unnamed inputs raise an error instead of choosing arbitrarily.
    The ``density format`` control ``onetep`` selects bohr coordinates with
    electrons/angstrom³ densities; ``e_per_angs3`` uses angstrom for both.
    WFX core counts and periodicity come from its wavefunction metadata;
    contradictory explicit overrides are rejected and job controls are ignored.
    """
    if format not in {"auto", "cube", "vasp", "xsf", "wfx"}:
        msg = "format must be auto, cube, vasp, xsf, or wfx"
        raise ValueError(msg)
    if density_kind not in {None, "total", "valence"}:
        msg = "density_kind must be total or valence"
        raise ValueError(msg)
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    directory = source if source.is_dir() else source.parent
    control_path = directory / "job_control.txt"
    control = read_job_control(control_path) if control_path.is_file() else JobControl()
    selected = (
        _discover(directory, format, control)
        if source.is_dir() or source.name == "job_control.txt"
        else source
    )
    kind = _file_format(selected) if format == "auto" else format
    effective_periodic = control.periodic if periodic is None else periodic
    effective_charge = control.net_charge if net_charge is None else net_charge
    cores = _cores(selected, kind, control, core_electrons)
    if kind == "vasp":
        grid = read_vasp(
            selected.parent, core_electrons=cores, net_charge=effective_charge
        )
    elif kind == "wfx":
        grid = _read_wavefunction(selected, periodic, core_electrons)
        effective_periodic = grid.periodic
    elif kind == "xsf":
        if cores is None:
            msg = (
                "XSF input requires explicit core_electrons or job-control core counts"
            )
            raise ValueError(msg)
        grid = read_xsf(
            selected, core_electrons=cores, net_charge=effective_charge or 0.0
        )
    else:
        length_unit, density_unit = _CUBE_UNITS.get(
            control.density_format, (None, "bohr^-3")
        )
        cube_kind = density_kind or (
            "valence" if selected.name.lower().startswith("valence") else "total"
        )
        total_file = selected.parent / "total_density.cube"
        grid = read_cube(
            selected,
            density_kind=cube_kind,
            core_electrons=core_electrons if cube_kind == "total" else cores,
            total_density_file=total_file
            if cube_kind == "valence"
            and total_file.is_file()
            and total_file != selected
            else None,
            periodic=effective_periodic or (False, False, False),
            net_charge=effective_charge or 0.0,
            length_unit=length_unit,
            density_unit=density_unit,
        )
        grid = _cube_spin(grid, selected.parent, length_unit, density_unit)
    if effective_periodic is not None and effective_periodic != grid.periodic:
        grid = replace(grid, periodic=effective_periodic)
    if effective_charge is not None and effective_charge != grid.net_charge:
        grid = replace(grid, net_charge=effective_charge)
    return grid
