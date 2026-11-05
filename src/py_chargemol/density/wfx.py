"""Gaussian WFX density grids with analytic tight-primitive corrections.

This implements the 2017 Chargemol Gaussian-pair grid construction in Python:
primitive products above exponent sum five are assigned analytically to cores;
diffuse products use nested grids, pairwise quadrature correction, and linear
interpolation. On grids whose voxels lie along x, y and z, each product is
built from one-dimensional factors, since Cartesian Gaussians are separable.
No wavefunction converter or external executable is needed.
"""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import product
from pathlib import Path

import numpy as np

from py_chargemol.ddec6 import StrPath
from py_chargemol.density.grid import (
    CUTOFF,
    NSHELLS,
    SCALE,
    DensityGrid,
    integration_extent,
    nearest_grid_points,
    wrap_periodic_positions,
)
from py_chargemol.density.references import core_reference_density

_GRID_SPACING = 0.14
_OVERLAP_TOLERANCE = 1e-12
_ANALYTIC_CUTOFF = 5.0
_COARSE_CUTOFF = 0.4
_NORMALIZATION_TOLERANCE = 0.001
_MAX_RADIAL_MOMENT_DEGREE = 8
_GHOST_NUCLEAR_CHARGE = 0.5
_ZERO = 1e-10
_PERIODIC_CUTOFF = 28.0
_DIMENSIONS = 3
_NESTED_STRIDES = (1, 2, 3, 4, 6, 8, 12)
# Each coarse grid is interpolated onto the next finer grid it divides.
_NESTED_MERGES = ((12, 6), (8, 4), (6, 3), (4, 2), (3, 1), (2, 1))
_MULTIPOLE_POWERS = np.array(
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ],
    dtype=np.int64,
)
_POWERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (3, 0, 0),
        (0, 3, 0),
        (0, 0, 3),
        (2, 1, 0),
        (2, 0, 1),
        (0, 2, 1),
        (1, 2, 0),
        (1, 0, 2),
        (0, 1, 2),
        (1, 1, 1),
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
        (3, 1, 0),
        (3, 0, 1),
        (1, 3, 0),
        (0, 3, 1),
        (1, 0, 3),
        (0, 1, 3),
        (2, 2, 0),
        (2, 0, 2),
        (0, 2, 2),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (0, 0, 5),
        (0, 1, 4),
        (0, 2, 3),
        (0, 3, 2),
        (0, 4, 1),
        (0, 5, 0),
        (1, 0, 4),
        (1, 1, 3),
        (1, 2, 2),
        (1, 3, 1),
        (1, 4, 0),
        (2, 0, 3),
        (2, 1, 2),
        (2, 2, 1),
        (2, 3, 0),
        (3, 0, 2),
        (3, 1, 1),
        (3, 2, 0),
        (4, 0, 1),
        (4, 1, 0),
        (5, 0, 0),
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class _Wavefunction:
    numbers: np.ndarray
    nuclear_charges: np.ndarray
    positions: np.ndarray
    centers: np.ndarray
    exponents: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    occupations: np.ndarray
    spin_occupations: np.ndarray
    net_charge: float
    edf_centers: np.ndarray
    edf_exponents: np.ndarray
    edf_powers: np.ndarray
    edf_coefficients: np.ndarray
    periodic_vectors: np.ndarray


@dataclass(frozen=True)
class _Sections:
    path: Path
    text: str

    def get(self, name: str, default: str | None = None) -> str:
        match = re.search(
            rf"<{re.escape(name)}>\s*(.*?)\s*</{re.escape(name)}>",
            self.text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            if default is not None:
                return default
            msg = f"{self.path}: missing WFX section <{name}>"
            raise ValueError(msg)
        return match[1]

    def numbers(self, name: str, count: int, default: str | None = None) -> np.ndarray:
        text = self.get(name, default)
        values = np.array(
            [float(v.replace("D", "E").replace("d", "e")) for v in text.split()]
        )
        if len(values) != count or not np.all(np.isfinite(values)):
            msg = f"{self.path}: <{name}> must contain {count} finite values"
            raise ValueError(msg)
        return values

    def integers(self, name: str, count: int, default: str | None = None) -> np.ndarray:
        values = self.numbers(name, count, default)
        if np.any(values != np.trunc(values)):
            msg = f"{self.path}: <{name}> must contain integers"
            raise ValueError(msg)
        return values.astype(np.int64)


def _orbital_coefficients(
    text: str, norbitals: int, nprimitives: int, path: StrPath
) -> np.ndarray:
    """Read one finite primitive-coefficient block for each molecular orbital."""
    matches = re.findall(
        r"<MO Number>\s*(\d+)\s*</MO Number>\s*([^<]*)", text, re.IGNORECASE
    )
    if len(matches) != norbitals or {int(m[0]) for m in matches} != set(
        range(1, norbitals + 1)
    ):
        msg = f"{path}: expected one coefficient block per molecular orbital"
        raise ValueError(msg)
    coefficients = np.empty((norbitals, nprimitives))
    for number, values in matches:
        array = np.array(
            [float(v.replace("D", "E").replace("d", "e")) for v in values.split()]
        )
        if len(array) != nprimitives or not np.all(np.isfinite(array)):
            msg = f"{path}: invalid primitive coefficients for MO {number}"
            raise ValueError(msg)
        coefficients[int(number) - 1] = array
    return coefficients


def _read_wavefunction(path: StrPath) -> _Wavefunction:
    source = Path(path)
    data = _Sections(source, source.read_text(encoding="utf-8"))
    if data.get("Keywords").strip().upper() != "GTO":
        msg = f"{path}: only Gaussian GTO WFX wavefunctions are supported"
        raise ValueError(msg)
    nperiodic = int(data.integers("Number of Translation Vectors", 1, "0")[0])
    if not 0 <= nperiodic <= _DIMENSIONS:
        msg = f"{path}: number of translation vectors must be 0-3"
        raise ValueError(msg)
    periodic_vectors = data.numbers("Translation Vectors", 3 * nperiodic, "").reshape(
        nperiodic, 3
    )
    natoms = int(data.integers("Number of Nuclei", 1)[0])
    nprimitives = int(data.integers("Number of Primitives", 1)[0])
    norbitals = int(data.integers("Number of Occupied Molecular Orbitals", 1)[0])
    if min(natoms, nprimitives, norbitals) < 1:
        msg = f"{path}: WFX atom, primitive, and orbital counts must be positive"
        raise ValueError(msg)
    numbers = data.integers("Atomic Numbers", natoms)
    nuclear = data.numbers("Nuclear Charges", natoms)
    positions = data.numbers("Nuclear Cartesian Coordinates", natoms * 3).reshape(
        natoms, 3
    )
    centers = data.integers("Primitive Centers", nprimitives) - 1
    exponents = data.numbers("Primitive Exponents", nprimitives)
    types = data.integers("Primitive Types", nprimitives)
    _validate_primitives(centers, exponents, types, natoms, path)
    occupations = data.numbers("Molecular Orbital Occupation Numbers", norbitals)
    spin_types = data.get("Molecular Orbital Spin Types").splitlines()
    spin_types = [value.strip().lower() for value in spin_types if value.strip()]
    if len(spin_types) != norbitals or any(
        v not in {"alpha", "beta", "alpha and beta"} for v in spin_types
    ):
        msg = f"{path}: invalid molecular orbital spin types"
        raise ValueError(msg)
    spin = occupations * np.array(
        [1 if v == "alpha" else -1 if v == "beta" else 0 for v in spin_types]
    )
    coefficients = _orbital_coefficients(
        data.get("Molecular Orbital Primitive Coefficients"),
        norbitals,
        nprimitives,
        path,
    )
    net_charge = float(data.numbers("Net Charge", 1)[0])
    if (
        np.any(occupations < 0)
        or abs(nuclear.sum() - net_charge - occupations.sum())
        > _NORMALIZATION_TOLERANCE
    ):
        msg = f"{path}: orbital occupations do not match nuclear charges and net charge"
        raise ValueError(msg)
    nedf = int(data.integers("Number of EDF Primitives", 1, "0")[0])
    edf_centers = data.integers("EDF Primitive Centers", nedf, "") - 1
    edf_exponents = data.numbers("EDF Primitive Exponents", nedf, "")
    edf_types = data.integers("EDF Primitive Types", nedf, "")
    edf_coefficients = data.numbers("EDF Primitive Coefficients", nedf, "")
    _validate_primitives(edf_centers, edf_exponents, edf_types, natoms, path)
    order = np.lexsort((exponents, centers))
    return _Wavefunction(
        numbers=numbers,
        nuclear_charges=nuclear,
        positions=positions,
        centers=centers[order],
        exponents=exponents[order],
        powers=_POWERS[types[order] - 1],
        coefficients=coefficients[:, order],
        occupations=occupations,
        spin_occupations=spin,
        net_charge=net_charge,
        edf_centers=edf_centers,
        edf_exponents=edf_exponents,
        edf_powers=_POWERS[edf_types - 1],
        edf_coefficients=edf_coefficients,
        periodic_vectors=periodic_vectors,
    )


def _validate_primitives(
    centers: np.ndarray,
    exponents: np.ndarray,
    types: np.ndarray,
    natoms: int,
    path: StrPath,
) -> None:
    if (
        np.any((centers < 0) | (centers >= natoms))
        or np.any(exponents <= 0)
        or np.any((types < 1) | (types > len(_POWERS)))
    ):
        msg = f"{path}: invalid primitive centers, exponents, or Cartesian types (1-56)"
        raise ValueError(msg)


def _axis_overlap(
    alpha: float, beta: float, left: float, right: float, degree: int
) -> np.ndarray:
    total = alpha + beta
    center = (alpha * left + beta * right) / total
    result = np.zeros((degree + 1, degree + 1))
    result[0, 0] = math.sqrt(math.pi / total)
    for j in range(1, degree + 1):
        result[0, j] = (center - right) * result[0, j - 1]
        if j > 1:
            result[0, j] += (j - 1) / (2 * total) * result[0, j - 2]
    for i in range(1, degree + 1):
        for j in range(degree + 1):
            result[i, j] = (center - left) * result[i - 1, j]
            if i > 1:
                result[i, j] += (i - 1) / (2 * total) * result[i - 2, j]
            if j > 0:
                result[i, j] += j / (2 * total) * result[i - 1, j - 1]
    return result


def _integral(tables: np.ndarray, p: np.ndarray, q: np.ndarray, factor: float) -> float:
    return float(factor * np.prod(tables[np.arange(3), p, q]))


def _integrals(tables: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the unscaled overlap of every row of ``p`` with every row of ``q``."""
    return np.asarray(np.prod(tables[np.arange(3), p[:, None], q[None, :]], axis=-1))


def _grid_spacing(alpha_sum: float) -> int:
    for spacing, denominator in ((1, 1), (2, 9 / 4), (3, 4), (4, 9), (6, 16), (8, 36)):
        if alpha_sum > _COARSE_CUTOFF / denominator:
            return spacing
    return 12


@dataclass(frozen=True)
class _Geometry:
    shape: np.ndarray
    origin: np.ndarray
    steps: np.ndarray
    periodic: tuple[bool, bool, bool]
    extent: np.ndarray

    @property
    def voxel_volume(self) -> float:
        return abs(float(np.linalg.det(self.steps)))

    @property
    def lattice(self) -> np.ndarray:
        return np.asarray(self.shape[:, None] * self.steps)

    @property
    def axis_aligned(self) -> bool:
        """Whether the voxel vectors lie along x, y and z, so Gaussians factorize."""
        return bool(np.all(self.steps == np.diag(np.diag(self.steps))))

    def axes(self, center: np.ndarray, radius: float, stride: int) -> list[np.ndarray]:
        """Return the fine-grid indices along each axis of a sphere's bounding box."""
        center_grid = (center - self.origin) @ np.linalg.inv(self.steps)
        nearest = nearest_grid_points(center_grid)
        extent = np.ceil(radius * self.extent / CUTOFF).astype(int) + 1
        lower, upper = nearest - extent, nearest + extent
        offset = stride - 2 if stride > 1 else 0
        axes = []
        for axis, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
            low = lo if self.periodic[axis] else max(lo, 0)
            high = hi if self.periodic[axis] else min(hi, self.shape[axis] - 1)
            start = low + (offset - low) % stride
            axes.append(np.arange(start, high + 1, stride))
        return axes

    def points(
        self, center: np.ndarray, radius: float, stride: int
    ) -> tuple[np.ndarray, np.ndarray]:
        axes = self.axes(center, radius, stride)
        offset = stride - 2 if stride > 1 else 0
        counts = self.shape // stride
        # Broadcast the three grid axes; rows of the returned xyz are x, y and z.
        x, y, z = (
            np.multiply.outer(step, axis)
            for axis, step in zip(axes, self.steps, strict=True)
        )
        xyz = (
            x[:, :, None, None]
            + y[:, None, :, None]
            + z[:, None, None, :]
            + self.origin[:, None, None, None]
        )
        keep = np.sum((xyz - center[:, None, None, None]) ** 2, axis=0) <= radius**2
        i, j, k = (
            ((axis - offset) // stride) % n
            for axis, n in zip(axes, counts, strict=True)
        )
        indices = (i[:, None, None] * counts[1] + j[None, :, None]) * counts[2] + k
        return indices[keep], xyz[:, keep]

    def image_shifts(self, extent: np.ndarray | None = None) -> Iterator[np.ndarray]:
        if extent is None:
            extent = integration_extent(self.lattice, radius=_PERIODIC_CUTOFF)
            extent *= np.array(self.periodic)
        lattice = self.lattice
        for image in product(*(range(-n, n + 1) for n in extent)):
            yield np.asarray(image) @ lattice


def _geometry(
    positions: np.ndarray, periodic_vectors: np.ndarray, spacing: float
) -> _Geometry:
    nperiodic = len(periodic_vectors)
    steps = np.eye(3) * spacing
    periodic = (nperiodic > 0, nperiodic > 1, nperiodic == _DIMENSIONS)
    counts = 24 * np.ceil(
        np.linalg.norm(periodic_vectors, axis=1) / (24 * spacing)
    ).astype(int)
    if np.any(counts == 0):
        msg = "WFX translation vectors must be nonzero"
        raise ValueError(msg)
    steps[:nperiodic] = periodic_vectors / counts[:, None]
    if nperiodic == 1:
        axis = periodic_vectors[0]
        direction = (
            np.cross([0, 0, 1], axis)
            if abs(axis[0]) > abs(axis[2])
            else np.cross(axis, [1, 0, 0])
        )
        steps[1] = direction * spacing / np.linalg.norm(direction)
    if 0 < nperiodic < _DIMENSIONS:
        direction = np.cross(steps[0], steps[1])
        if np.linalg.norm(direction) == 0:
            msg = "WFX translation vectors must be linearly independent"
            raise ValueError(msg)
        steps[2] = direction * spacing / np.linalg.norm(direction)
    if abs(np.linalg.det(steps)) < np.finfo(float).tiny:
        msg = "WFX translation vectors must be nonzero and linearly independent"
        raise ValueError(msg)
    extent = integration_extent(steps)
    rounded = nearest_grid_points(positions @ np.linalg.inv(steps))
    shape = (
        24 * np.ceil((rounded.max(axis=0) - rounded.min(axis=0) + 2 * extent + 11) / 24)
    ).astype(int)
    shape[:nperiodic] = counts
    origin_grid = rounded.min(axis=0) - extent - 6
    origin_grid[:nperiodic] = 0
    return _Geometry(shape, origin_grid @ steps, steps, periodic, extent)


def _interpolate(
    values: np.ndarray, target_stride: int, source_stride: int
) -> np.ndarray:
    ratio = source_stride // target_stride
    target_offset = max(target_stride - 2, 0)
    source_offset = source_stride - 2
    result = values
    for axis in range(3):
        coordinate = (
            np.arange(result.shape[axis] * ratio) * target_stride
            + target_offset
            - source_offset
        ) / source_stride
        lower = np.floor(coordinate).astype(int)
        fraction = coordinate - lower
        dims = [1, 1, 1]
        dims[axis] = len(coordinate)
        result = np.take(result, lower, axis=axis, mode="wrap") * (
            1 - fraction.reshape(dims)
        ) + np.take(result, lower + 1, axis=axis, mode="wrap") * fraction.reshape(dims)
    return result


def _single_atom_radial_moments(
    wave: _Wavefunction, density_matrix: np.ndarray
) -> np.ndarray | None:
    """Integrate explicit-electron r**m for m=-1..4 in an isolated atom."""
    if len(wave.positions) != 1 or len(wave.periodic_vectors):
        return None
    moments = np.zeros(6)
    for i, alpha in enumerate(wave.exponents):
        for j in range(i, len(wave.exponents)):
            powers = wave.powers[i] + wave.powers[j]
            if np.any(powers % 2):
                continue
            degree = int(powers.sum())
            if degree > _MAX_RADIAL_MOMENT_DEGREE:
                # The original isolated-atom diagnostic excludes H products.
                return None
            coefficient = density_matrix[i, j] * (1 if i == j else 2)
            angular = math.prod(math.gamma((power + 1) / 2) for power in powers)
            angular /= math.gamma((degree + 3) / 2)
            exponents = (degree + np.arange(-1, 5) + 3) / 2
            moments += (
                coefficient
                * angular
                * np.array([math.gamma(exponent) for exponent in exponents])
                / (alpha + wave.exponents[j]) ** exponents
            )
    return moments


def read_wfx(
    path: StrPath,
    *,
    grid_spacing: float = _GRID_SPACING,
    interpolate_diffuse: bool = True,
) -> DensityGrid:
    """Construct a molecular or periodic density grid from a Gaussian WFX wavefunction.

    Tight primitive pairs are integrated analytically to preserve core electron
    counts and atomic multipoles. ``grid_spacing`` is in bohr and defaults to
    Chargemol's 0.14 bohr. Cartesian primitive types through H are supported,
    including effective core potentials and optional electron-density functions.

    As in Chargemol, diffuse primitive pairs are evaluated on coarser nested
    grids and interpolated onto the final grid. ``interpolate_diffuse=False``
    instead evaluates every pair on the final grid. This is slower, but
    removes the interpolation error: about 2e-4 e in the ozone example's
    charges, so results then differ from Chargemol's at that level.
    """
    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        msg = "grid_spacing must be finite and positive"
        raise ValueError(msg)
    wave = _read_wavefunction(path)
    real_atoms = wave.nuclear_charges > _GHOST_NUCLEAR_CHARGE
    atom_map = np.full(len(real_atoms), -1)
    atom_map[real_atoms] = np.arange(np.count_nonzero(real_atoms))
    positions = wave.positions[real_atoms]
    geometry = _geometry(positions, wave.periodic_vectors, grid_spacing)
    # Pair-image sums use local cell centres; retain the input atom images in
    # the returned grid so its coordinates and bond translations stay intact.
    local_positions, _ = wrap_periodic_positions(
        wave.positions, geometry.lattice, geometry.periodic
    )
    wave = replace(wave, positions=local_positions)
    shape = geometry.shape
    strides = _NESTED_STRIDES if interpolate_diffuse else (1,)
    grids = {stride: np.zeros(tuple(shape // stride)) for stride in strides}
    spins = {stride: np.zeros_like(values) for stride, values in grids.items()}
    core = np.zeros(tuple(shape))
    core_counts = wave.numbers[real_atoms] - wave.nuclear_charges[real_atoms]
    corrections = np.zeros((len(positions), 11))
    total_matrix = wave.coefficients.T @ (wave.occupations[:, None] * wave.coefficients)
    spin_matrix = wave.coefficients.T @ (
        wave.spin_occupations[:, None] * wave.coefficients
    )
    groups = [
        np.flatnonzero((wave.centers == center) & (wave.exponents == alpha))
        for center, alpha in dict.fromkeys(
            zip(wave.centers.tolist(), wave.exponents.tolist(), strict=True)
        )
    ]
    construction = _Construction(
        wave,
        geometry,
        grids,
        spins,
        core,
        core_counts,
        corrections,
        total_matrix,
        spin_matrix,
        atom_map,
        nested=interpolate_diffuse,
    )
    integrated = 0.0
    pair_tolerance = _OVERLAP_TOLERANCE / (len(groups) * (len(groups) + 1) / 2)
    for index, left in enumerate(groups):
        for right in groups[index:]:
            for shift in construction.image_shifts(left, right, pair_tolerance):
                integrated += construction.accumulate(left, right, shift)
    if abs(integrated - wave.occupations.sum()) > _NORMALIZATION_TOLERANCE:
        msg = (
            f"{path}: unnormalized primitive coefficients ({integrated:.8f} electrons)"
        )
        raise ValueError(msg)
    for source, target in _NESTED_MERGES if interpolate_diffuse else ():
        grids[target] += _interpolate(grids.pop(source), target, source)
        spins[target] += _interpolate(spins.pop(source), target, source)
    _add_edf(wave, geometry, core)
    core = np.maximum(core, 0)
    valence = np.maximum(grids[1], -core)
    negative = core_counts < _ZERO
    corrections[negative, 0] = core_counts[negative]
    core_counts[negative] = 0
    # An atom whose electrons are all tight can exceed its atomic number by the
    # normalization error; keep that excess as an occupancy correction too.
    numbers = wave.numbers[real_atoms]
    excess = core_counts - numbers
    full = (excess > 0) & (excess <= _NORMALIZATION_TOLERANCE)
    corrections[full, 0] = excess[full]
    core_counts[full] = numbers[full]
    _add_missing_core(wave, geometry, core)
    return DensityGrid(
        atomic_numbers=numbers,
        positions=positions,
        density=valence,
        origin=geometry.origin,
        steps=geometry.steps,
        periodic=geometry.periodic,
        core_electrons=core_counts,
        core_density=core,
        net_charge=wave.net_charge,
        spin_density=spins[1] if np.any(wave.spin_occupations) else None,
        atomic_corrections=corrections,
        wavefunction_radial_moments=_single_atom_radial_moments(wave, total_matrix),
    )


@dataclass
class _Construction:
    wave: _Wavefunction
    geometry: _Geometry
    grids: dict[int, np.ndarray]
    spins: dict[int, np.ndarray]
    core: np.ndarray
    core_counts: np.ndarray
    corrections: np.ndarray
    total_matrix: np.ndarray
    spin_matrix: np.ndarray
    atom_map: np.ndarray
    nested: bool

    @cached_property
    def log_norms(self) -> np.ndarray:
        """Log L2 norms of monomials times exp(-alpha r²/2), for overlap bounds."""
        powers = self.wave.powers
        gamma = np.array(
            [sum(math.lgamma(float(p) + 0.5) for p in row) for row in powers]
        )
        return np.asarray(
            (gamma - (powers.sum(axis=1) + 1.5) * np.log(self.wave.exponents)) / 2
        )

    def image_shifts(
        self, left: np.ndarray, right: np.ndarray, tolerance: float
    ) -> Iterator[np.ndarray]:
        """Bound each group's omitted electron and spin sums with enough images."""
        geometry, wave = self.geometry, self.wave
        periodic = np.asarray(geometry.periodic)
        if not np.any(periodic):
            yield np.zeros(3)
            return
        rows, columns = np.meshgrid(left, right, indexing="ij")
        coefficients = np.maximum(
            np.abs(self.total_matrix[rows, columns]),
            np.abs(self.spin_matrix[rows, columns]),
        ) * np.where(rows == columns, 1, 2)
        keep = (columns >= rows) & (coefficients > 0)
        if not np.any(keep):
            return
        terms = (
            np.log(coefficients[keep])
            + self.log_norms[rows[keep]]
            + self.log_norms[columns[keep]]
        )
        largest = float(terms.max())
        log_bound = largest + math.log(float(np.exp(terms - largest).sum()))
        alpha, beta = wave.exponents[left[0]], wave.exponents[right[0]]
        decay = min(alpha, beta) / (2 * (1 + min(alpha, beta) / max(alpha, beta)))
        # Splitting the Gaussian product and applying Cauchy-Schwarz bounds
        # every overlap by its monomial norms times exp(-decay * distance²).
        reciprocal = np.linalg.inv(geometry.lattice)[:, periodic]
        lengths = np.linalg.norm(reciprocal, axis=0)
        directions = reciprocal / lengths
        gram_bound = np.abs(directions.T @ directions).sum(axis=1).max()
        gamma = decay / (gram_bound * lengths**2)
        # The Gram row-sum bound makes the radial Gaussian no larger than
        # the product of these 1D bounds along periodic fractional axes.
        full_logs = np.log(3 + np.sqrt(np.pi / gamma))
        extent = integration_extent(geometry.lattice, radius=_PERIODIC_CUTOFF)
        extent *= periodic
        budget = math.log(tolerance / np.count_nonzero(periodic))
        for index, axis in enumerate(np.flatnonzero(periodic)):
            other_logs = float(full_logs.sum() - full_logs[index])
            exponent = float(gamma[index])
            # Wrapped centres differ by less than one fractional cell. Outside
            # +/-N the shifted Gaussian tail is at most
            # 2 exp(-gamma N²) (1 + 1/(2 gamma N)); union over the periodic axes.
            while (
                log_bound
                + math.log(2)
                - exponent * float(extent[axis]) ** 2
                + math.log1p(1 / (2 * exponent * extent[axis]))
                + other_logs
                > budget
            ):
                extent[axis] += 1
        yield from geometry.image_shifts(extent)

    def accumulate(
        self, left: np.ndarray, right: np.ndarray, shift: np.ndarray
    ) -> float:
        wave, geometry = self.wave, self.geometry
        i0, j0 = int(left[0]), int(right[0])
        a, b = int(wave.centers[i0]), int(wave.centers[j0])
        alpha, beta = float(wave.exponents[i0]), float(wave.exponents[j0])
        point_a, point_b = wave.positions[a], wave.positions[b] + shift
        alpha_sum = alpha + beta
        separation2 = float(np.sum((point_a - point_b) ** 2))
        factor = math.exp(-alpha * beta * separation2 / alpha_sum)
        if factor == 0:
            return 0.0
        p, q = wave.powers[left], wave.powers[right]
        # Screening bounds and quadrupole corrections raise either power by two.
        degree = int(p.max() + q.max()) + 2
        tables = np.array(
            [
                _axis_overlap(alpha, beta, float(point_a[k]), float(point_b[k]), degree)
                for k in range(3)
            ]
        )
        # Each primitive pair (i, j >= i) of the two groups is one matrix element.
        rows, columns = np.meshgrid(left, right, indexing="ij")
        upper = columns >= rows
        multiplicity = np.where(rows == columns, 1.0, 2.0)
        total = self.total_matrix[rows, columns]
        spin = self.spin_matrix[rows, columns]
        overlaps = factor * _integrals(tables, p, q)
        coefficients = np.where(upper, total * multiplicity, 0.0)
        integrated = float(np.sum(coefficients * overlaps))
        included = upper & _included_pairs(
            tables, p, q, factor, a == b, alpha_sum, total, spin
        )
        if not np.any(included):
            return integrated
        coefficients = np.where(included, coefficients, 0.0)
        spin_coefficients = np.where(included, spin * multiplicity, 0.0)
        analytic = (
            alpha * min(wave.nuclear_charges[a], 1)
            + beta * min(wave.nuclear_charges[b], 1)
            > _ANALYTIC_CUTOFF
        )
        stride = _grid_spacing(alpha_sum) if self.nested and not analytic else 1
        center = (alpha * point_a + beta * point_b) / alpha_sum
        radius = math.sqrt(
            max(
                -math.log(_OVERLAP_TOLERANCE) / alpha_sum
                - alpha * beta * separation2 / alpha_sum**2,
                0.25,
            )
        )
        pair = _Pair(point_a, point_b, p, q, alpha_sum, factor, center, radius)
        sampler: _Box | _Points = (
            _Box(geometry, pair, stride)
            if geometry.axis_aligned
            else _Points(geometry, pair, stride)
        )
        scale = geometry.voxel_volume * stride**3
        sampled = sampler.sums() * scale
        values = sampler.values(coefficients)
        spin_values = sampler.values(spin_coefficients)
        if analytic:
            for i, j in zip(*np.nonzero(included), strict=True):
                _analytic_correction(
                    tables,
                    p[i],
                    q[j],
                    factor,
                    alpha,
                    beta,
                    float(coefficients[i, j]),
                    float(spin_coefficients[i, j]),
                    wave.nuclear_charges[a],
                    wave.nuclear_charges[b],
                    int(self.atom_map[a]),
                    int(self.atom_map[b]),
                    self.core_counts,
                    self.corrections,
                    sampled_overlap=float(sampled[i, j]),
                )
            sampler.add(self.core, values)
        else:
            absolute = sampler.sums(absolute=True) * scale
            maximum = (
                np.abs(coefficients + spin_coefficients)
                + np.abs(coefficients - spin_coefficients)
            ) / 2
            correction = np.clip(
                np.divide(
                    overlaps - sampled,
                    absolute,
                    out=np.zeros_like(absolute),
                    where=maximum * absolute > _OVERLAP_TOLERANCE,
                ),
                -0.05,
                0.05,
            )
            values += sampler.values(coefficients * correction, absolute=True)
            spin_values += sampler.values(spin_coefficients * correction, absolute=True)
            sampler.add(self.grids[stride], values)
        sampler.add(self.spins[stride], spin_values)
        return integrated


@dataclass(frozen=True)
class _Pair:
    """Two primitive groups, whose product is ``factor exp(-exponent |r - center|²)``.

    The product's Cartesian monomials are ``powers`` of the displacement from
    each group's centre. Grid points beyond ``radius`` are neglected.
    """

    left_center: np.ndarray
    right_center: np.ndarray
    left_powers: np.ndarray
    right_powers: np.ndarray
    exponent: float
    factor: float
    center: np.ndarray
    radius: float


class _Points:
    """Primitive products at each grid point within the pair's sphere."""

    def __init__(self, geometry: _Geometry, pair: _Pair, stride: int) -> None:
        self.indices, xyz = geometry.points(pair.center, pair.radius, stride)
        gaussian = np.exp(
            -pair.exponent * np.sum((xyz - pair.center[:, None]) ** 2, axis=0)
        )
        self.left = _monomials(xyz - pair.left_center[:, None], pair.left_powers)
        self.right = _monomials(xyz - pair.right_center[:, None], pair.right_powers) * (
            gaussian * pair.factor
        )

    @cached_property
    def absolute(self) -> tuple[np.ndarray, np.ndarray]:
        return np.abs(self.left), np.abs(self.right)

    def sums(self, *, absolute: bool = False) -> np.ndarray:
        """Sum each primitive product, or its magnitude, over the points."""
        left, right = self.absolute if absolute else (self.left, self.right)
        return np.asarray(left @ right.T)

    def values(self, coefficients: np.ndarray, *, absolute: bool = False) -> np.ndarray:
        """Return the coefficient-weighted sum of primitive products at each point."""
        left, right = self.absolute if absolute else (self.left, self.right)
        return np.asarray(np.einsum("in,in->n", left, coefficients @ right))

    def add(self, grid: np.ndarray, values: np.ndarray) -> None:
        np.add.at(grid.ravel(), self.indices, values)


class _Box:
    """Primitive products over the bounding box of the pair's sphere.

    On an axis-aligned grid each Cartesian Gaussian product is a product of x,
    y and z functions, so exponentials and monomials are evaluated along the
    box's three axes only. The pairs collapse onto their distinct per-axis
    power combinations, and the box is added to the grid as one block. Points
    outside the sphere are masked, which gives the same values as ``_Points``.
    """

    def __init__(self, geometry: _Geometry, pair: _Pair, stride: int) -> None:
        axes = geometry.axes(pair.center, pair.radius, stride)
        offset = stride - 2 if stride > 1 else 0
        counts = geometry.shape // stride
        # The wrapped strided index of each axis's first point (zero if none).
        self.starts = [
            int(np.sum((axis[:1] - offset) // stride % count))
            for axis, count in zip(axes, counts, strict=True)
        ]
        left_degree = int(pair.left_powers.max())
        right_degree = int(pair.right_powers.max())
        self.combinations = tuple(
            pair.left_powers[:, None, k] * (right_degree + 1)
            + pair.right_powers[None, :, k]
            for k in range(3)
        )
        self.tables: list[np.ndarray] = []
        squared: list[np.ndarray] = []
        for k, axis in enumerate(axes):
            x = geometry.steps[k, k] * axis + geometry.origin[k]
            squared.append((x - pair.center[k]) ** 2)
            gaussian = np.exp(-pair.exponent * squared[-1])
            if k == 0:
                gaussian *= pair.factor
            left = _powers(x - pair.left_center[k], left_degree)
            right = _powers(x - pair.right_center[k], right_degree) * gaussian
            self.tables.append((left[:, None] * right[None, :]).reshape(-1, len(x)))
        x2, y2, z2 = squared
        self.mask: np.ndarray = (
            x2[:, None, None] + y2[None, :, None] + z2[None, None, :] <= pair.radius**2
        )

    @cached_property
    def absolute(self) -> list[np.ndarray]:
        return [np.abs(table) for table in self.tables]

    @cached_property
    def inside(self) -> np.ndarray:
        return self.mask.astype(np.float64)

    def sums(self, *, absolute: bool = False) -> np.ndarray:
        """Sum each primitive product, or its magnitude, over the sphere."""
        x, y, z = self.absolute if absolute else self.tables
        (a, nx), (b, _), (c, _) = x.shape, y.shape, z.shape
        partial = np.matmul(y, self.inside @ z.T)
        moments = (x @ partial.reshape(nx, b * c)).reshape(a, b, c)
        return np.asarray(moments[self.combinations])

    def values(self, coefficients: np.ndarray, *, absolute: bool = False) -> np.ndarray:
        """Return the coefficient-weighted sum of primitive products in the box."""
        x, y, z = self.absolute if absolute else self.tables
        (a, nx), (b, ny), (c, nz) = x.shape, y.shape, z.shape
        tensor = np.zeros((a, b, c))
        np.add.at(tensor, self.combinations, coefficients)
        partial = y.T @ (tensor @ z)
        values: np.ndarray = (x.T @ partial.reshape(a, ny * nz)).reshape(nx, ny, nz)
        values[~self.mask] = 0
        return values

    def add(self, grid: np.ndarray, values: np.ndarray) -> None:
        """Add the box from its start, folding periodic images into the cell."""
        block = []
        for axis, (start, count) in enumerate(
            zip(self.starts, grid.shape, strict=True)
        ):
            length = values.shape[axis]
            if start + length <= count:
                block.append(slice(start, start + length))
                continue
            periods = -(-(start + length) // count)
            padding = [(0, 0)] * 3
            padding[axis] = (start, periods * count - start - length)
            padded = np.pad(values, padding)
            values = padded.reshape(
                (*padded.shape[:axis], periods, count, *padded.shape[axis + 1 :])
            ).sum(axis=axis)
            block.append(slice(None))
        grid[tuple(block)] += values


def _powers(delta: np.ndarray, degree: int) -> np.ndarray:
    """Return ``delta**k`` for k = 0 to ``degree``, by repeated multiplication."""
    terms = [np.ones_like(delta)]
    for _ in range(degree):
        terms.append(terms[-1] * delta)
    return np.stack(terms)


def _monomials(delta: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Evaluate each row of Cartesian ``powers`` at ``(3, N)`` displacements."""
    table = _powers(delta, int(powers.max()))
    return np.asarray(
        table[powers[:, 0], 0] * table[powers[:, 1], 1] * table[powers[:, 2], 2]
    )


def _included_pairs(  # noqa: PLR0913, PLR0917 -- Both Gaussian factors define the overlap bound.
    tables: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    factor: float,
    same_center: bool,
    alpha_sum: float,
    total: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    """Screen every primitive pair by its bound on the absolute overlap."""
    p, q = p[:, None], q[None, :]
    if same_center:
        p, q = p + q, np.zeros_like(q)
    if alpha_sum > 1:
        p, q = 2 * (p // 2), 2 * (q // 2)
    else:
        p, q = 2 * ((p + 1) // 2), 2 * ((q + 1) // 2)
    maximum = (np.abs(total + spin) + np.abs(total - spin)) / 2
    bound = factor * np.prod(tables[np.arange(3), p, q], axis=-1)
    return np.asarray(maximum * bound >= _OVERLAP_TOLERANCE)


def _analytic_correction(  # noqa: PLR0913, PLR0917 -- Two Gaussian centers and their atomic accumulators.
    tables: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    factor: float,
    alpha: float,
    beta: float,
    coefficient: float,
    spin_coefficient: float,
    nuclear_a: float,
    nuclear_b: float,
    atom_a: int,
    atom_b: int,
    core_counts: np.ndarray,
    corrections: np.ndarray,
    sampled_overlap: float,
) -> None:
    if nuclear_a > 0 and nuclear_b > 0:
        ratio = (beta / alpha) ** 10
        weight_a = 1 / (1 + ratio)
    else:
        weight_a = 1.0 if nuclear_a > 0 else 0.0
    overlap = _integral(tables, p, q, factor)
    for atom, weight, is_left in (
        (atom_a, weight_a, True),
        (atom_b, 1 - weight_a, False),
    ):
        if atom < 0:
            continue
        core_counts[atom] += weight * coefficient * overlap
        # Spin partitioning retains the sampled tight density. Only its
        # quadrature error belongs in the on-site analytic correction.
        corrections[atom, 1] += weight * spin_coefficient * (overlap - sampled_overlap)
        for column, extra in enumerate(_MULTIPOLE_POWERS, start=2):
            moment = _integral(
                tables, p + extra if is_left else p, q if is_left else q + extra, factor
            )
            corrections[atom, column] -= weight * coefficient * moment


def _add_edf(wave: _Wavefunction, geometry: _Geometry, core: np.ndarray) -> None:
    for center, alpha, powers, coefficient in zip(
        wave.edf_centers,
        wave.edf_exponents,
        wave.edf_powers,
        wave.edf_coefficients,
        strict=True,
    ):
        indices, xyz = geometry.points(wave.positions[center], CUTOFF, 1)
        delta = xyz - wave.positions[center][:, None]
        np.add.at(
            core.ravel(),
            indices,
            (
                coefficient
                * np.prod(delta ** powers[:, None], axis=0)
                * np.exp(-alpha * np.sum(delta**2, axis=0))
            ),
        )


def _add_missing_core(
    wave: _Wavefunction, geometry: _Geometry, core: np.ndarray
) -> None:
    if len(wave.edf_centers):
        return
    for number, nuclear, position in zip(
        wave.numbers, wave.nuclear_charges, wave.positions, strict=True
    ):
        missing = round(number - nuclear)
        if nuclear <= _GHOST_NUCLEAR_CHARGE or missing == 0:
            continue
        radial = core_reference_density(int(number), missing)
        indices, xyz = geometry.points(position, CUTOFF, 1)
        shells = (
            np.ceil(
                SCALE * np.linalg.norm(xyz - position[:, None], axis=0) + 1e-10
            ).astype(int)
            - 1
        )
        keep = shells < NSHELLS
        np.add.at(core.ravel(), indices[keep], radial[shells[keep]])
