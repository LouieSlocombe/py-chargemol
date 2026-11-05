"""Regular electron-density grids and atom-centred integration domains."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
import numpy.typing as npt

from py_chargemol.units import BOHR_PER_ANGSTROM

NSHELLS = 100
CUTOFF = 5.0 * BOHR_PER_ANGSTROM
SCALE = NSHELLS / CUTOFF
ZERO = 1e-10
_DIMENSIONS = 3
_MAX_ATOMIC_NUMBER = 109
# Box points measured at once by atomic_samples. Temporaries of 256 kB stay in
# a core's L2 cache, which makes measuring the box about twice as fast.
_SLAB_POINTS = 1 << 15


class RadialSample(Protocol):
    """Grid indices and radial shells shared by all density partitions."""

    @property
    def indices(self) -> npt.NDArray[np.integer]:
        """Flat grid indices, including repeated periodic images."""
        ...

    @property
    def shells(self) -> npt.NDArray[np.integer]:
        """Zero-based radial shell of each sampled point."""
        ...


@dataclass
class DensityGrid:
    """A regular density grid, with Cartesian lengths in bohr.

    ``density`` contains valence electrons per bohr cubed; an all-electron
    density uses zero ``core_electrons``. ``steps`` contains the three voxel
    vectors as rows. Periodic grids contain each boundary point only once.
    Nuclear positions may use any periodic cell images. Analyses retain those
    coordinates and report bond translations relative to them.
    ``pseudodensity`` optionally supplies the PAW pseudo-valence density used
    to correct the occupancies of a reconstructed valence grid.
    """

    atomic_numbers: np.ndarray
    positions: np.ndarray
    density: np.ndarray
    origin: np.ndarray
    steps: np.ndarray
    periodic: tuple[bool, bool, bool] = (False, False, False)
    core_electrons: np.ndarray | None = None
    core_density: np.ndarray | None = None
    net_charge: float = 0.0
    occupancy_correction: np.ndarray | None = None
    pseudodensity: np.ndarray | None = None
    spin_density: np.ndarray | None = None
    atomic_corrections: np.ndarray | None = None
    wavefunction_radial_moments: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Copy and validate the numerical inputs."""
        self.atomic_numbers = _atomic_numbers(self.atomic_numbers)
        self.positions = _finite(self.positions, "positions", (self.natoms, 3))
        if len({tuple(position) for position in self.positions}) != self.natoms:
            msg = "nuclei must have distinct positions"
            raise ValueError(msg)
        self.origin = _finite(self.origin, "origin", (3,))
        self.steps = _finite(self.steps, "steps", (3, 3))
        if abs(np.linalg.det(self.steps)) < np.finfo(float).tiny:
            msg = "steps must define a nonsingular three-dimensional grid"
            raise ValueError(msg)
        self.density = _finite_array(self.density, "density")
        if self.density.ndim != _DIMENSIONS or not self.density.size:
            msg = "density must be a nonempty three-dimensional array"
            raise ValueError(msg)
        self.periodic = _periodicity(self.periodic)
        if not np.isfinite(self.net_charge):
            msg = "net_charge must be finite"
            raise ValueError(msg)
        self._validate_analysis_arrays()

    def _validate_analysis_arrays(self) -> None:
        """Validate core, spin, and occupancy fields against the density grid."""
        self.core_electrons = _finite(
            np.zeros(self.natoms)
            if self.core_electrons is None
            else self.core_electrons,
            "core_electrons",
            (self.natoms,),
        )
        if np.any(self.core_electrons < 0) or np.any(
            self.core_electrons > self.atomic_numbers
        ):
            msg = "core_electrons must lie between zero and each atomic number"
            raise ValueError(msg)
        if self.core_density is None and np.any(
            self.core_electrons != np.trunc(self.core_electrons)
        ):
            msg = "core_electrons must be integers for missing core density"
            raise ValueError(msg)
        self.occupancy_correction = _finite(
            np.zeros(self.natoms)
            if self.occupancy_correction is None
            else self.occupancy_correction,
            "occupancy_correction",
            (self.natoms,),
        )
        for name in ("core_density", "pseudodensity"):
            value = getattr(self, name)
            if value is not None:
                checked = _finite(value, name, self.density.shape)
                if name == "core_density" and np.any(checked < 0):
                    msg = f"{name} must contain nonnegative values"
                    raise ValueError(msg)
                setattr(self, name, checked)
        total = (
            self.density
            if self.core_density is None
            else self.density + self.core_density
        )
        if np.any(total < 0):
            msg = "total electron density must contain nonnegative values"
            raise ValueError(msg)
        if self.atomic_corrections is not None:
            self.atomic_corrections = _finite(
                self.atomic_corrections, "atomic_corrections", (self.natoms, 11)
            )
        if self.wavefunction_radial_moments is not None:
            if self.natoms != 1:
                msg = "wavefunction radial moments require a single atomic centre"
                raise ValueError(msg)
            self.wavefunction_radial_moments = _finite(
                self.wavefunction_radial_moments, "wavefunction_radial_moments", (6,)
            )
        if self.spin_density is not None:
            shape = np.shape(self.spin_density)
            if shape not in (self.density.shape, (*self.density.shape, 3)):
                msg = (
                    "spin_density must match density, optionally with three components"
                )
                raise ValueError(msg)
            self.spin_density = _finite(self.spin_density, "spin_density", shape)

    @property
    def natoms(self) -> int:
        """Number of nuclei in the grid."""
        return len(self.atomic_numbers)

    @property
    def pixel_volume(self) -> float:
        """Volume represented by each grid point, in bohr cubed."""
        return float(abs(np.linalg.det(self.steps)))

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Full grid-cell vectors as rows, including nonperiodic directions."""
        return np.asarray(np.asarray(self.density.shape)[:, None] * self.steps)


@dataclass(frozen=True)
class AtomicSample:
    """Grid points within the radial cutoff of one reference-cell nucleus.

    Repeated indices represent different periodic images of the same grid
    point. ``coordinates`` are the unwrapped int32 grid coordinates, in
    lexicographic order; ``-(coordinates // shape)`` gives the corresponding
    atom image relative to the reference cell, when the grid point is folded
    back into that cell.

    A sample holds millions of points per atom on a fine grid, so it stores
    each point in 17 bytes: an int32 flat index (int64 only for grids beyond
    2**31 points), a uint8 shell and the int32 coordinates. Cartesian
    ``displacements`` are derived from the coordinates, the grid ``steps`` and
    the nucleus ``offset`` when read, rather than stored.
    """

    indices: np.ndarray
    shells: np.ndarray
    coordinates: np.ndarray
    steps: np.ndarray
    offset: np.ndarray

    @property
    def displacements(self) -> npt.NDArray[np.float64]:
        """Unwrapped Cartesian displacement of every point from the nucleus."""
        return self.cartesian(self.coordinates)

    def displacement_rows(self) -> npt.NDArray[np.float64]:
        """The ``displacements``, one contiguous row per component: ``(3, n)``.

        Elementwise work on one Cartesian component at a time reads these
        rows contiguously, without the transposed copy ``displacements`` makes.
        """
        return self._components(self.coordinates)

    def cartesian(self, coordinates: np.ndarray) -> npt.NDArray[np.float64]:
        """Cartesian displacement from the nucleus of unwrapped grid coordinates.

        Sums the axes in the order ``atomic_samples`` measured the shells in,
        so the displacements are the bits it would have stored. Each Cartesian
        component is built along contiguous rows, then laid out as the stored
        ``(n, 3)`` rows were, so that reductions over them sum in the same order.
        """
        return np.ascontiguousarray(self._components(coordinates).T)

    def _components(self, coordinates: np.ndarray) -> npt.NDArray[np.float64]:
        axes = coordinates.T.astype(np.float64)
        result = np.empty((_DIMENSIONS, len(coordinates)))
        for component, row in enumerate(result):
            np.multiply(axes[0], self.steps[0, component], out=row)
            np.add(row, axes[1] * self.steps[1, component], out=row)
            np.add(row, axes[2] * self.steps[2, component], out=row)
            np.add(row, self.offset[component], out=row)
        return result


def sample_index_type(size: int) -> type[np.signedinteger]:
    """The narrowest of int32 and int64 that indexes a grid of ``size`` points."""
    return np.int32 if size <= np.iinfo(np.int32).max else np.int64


def atomic_samples(grid: DensityGrid) -> tuple[AtomicSample, ...]:
    """Construct finite-radius integration domains, including periodic images."""
    inverse = np.linalg.inv(grid.steps)
    extent = integration_extent(grid.steps)
    shape = np.asarray(grid.density.shape)
    result = []
    for position in grid.positions:
        center = (position - grid.origin) @ inverse
        nearest = nearest_grid_points(center)
        lower = nearest - extent
        upper = nearest + extent
        for axis, periodic in enumerate(grid.periodic):
            if not periodic:
                lower[axis] = max(lower[axis], 0)
                upper[axis] = min(upper[axis], shape[axis] - 1)
        axes = [np.arange(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True)]
        result.append(_atomic_sample(grid, position, axes))
    return tuple(result)


def _atomic_sample(
    grid: DensityGrid, position: np.ndarray, axes: list[npt.NDArray[np.int64]]
) -> AtomicSample:
    """Sample the points of one atom's box of grid ``axes`` within the cutoff.

    The box is measured a slab of first-axis planes at a time, so that no
    temporary array spans the whole box, and its points are kept in the same
    lexicographic order.
    """
    shape = np.asarray(grid.density.shape)
    index_type = sample_index_type(grid.density.size)
    x, y, z = (
        np.multiply.outer(step, axis)
        for axis, step in zip(axes, grid.steps, strict=True)
    )
    offset = grid.origin - position
    xy = x[:, :, None] + y[:, None, :]
    wrapped = [axis % size for axis, size in zip(axes, shape, strict=True)]
    rows = max(1, _SLAB_POINTS // max(1, len(axes[1]) * len(axes[2])))
    # Start from empty parts, so a box outside a nonperiodic grid has no points.
    indices = [np.empty(0, dtype=index_type)]
    shells = [np.empty(0, dtype=np.uint8)]
    coordinates = [np.empty((0, _DIMENSIONS), dtype=np.int32)]
    for start in range(0, len(axes[0]), rows):
        slab = xy[:, start : start + rows]
        squared = _squared_component(slab[0], z[0], offset[0])
        squared += _squared_component(slab[1], z[1], offset[1])
        squared += _squared_component(slab[2], z[2], offset[2])
        slab_shells = np.ceil(SCALE * np.sqrt(squared) + ZERO).astype(np.int64) - 1
        keep = slab_shells < NSHELLS
        i, j, k = np.nonzero(keep)
        i += start
        indices.append(
            (
                (wrapped[0][i] * shape[1] + wrapped[1][j]) * shape[2] + wrapped[2][k]
            ).astype(index_type)
        )
        shells.append(slab_shells[keep].astype(np.uint8))
        coordinates.append(
            np.column_stack((axes[0][i], axes[1][j], axes[2][k])).astype(np.int32)
        )
    return AtomicSample(
        indices=np.concatenate(indices),
        shells=np.concatenate(shells),
        coordinates=np.concatenate(coordinates),
        steps=grid.steps.copy(),
        offset=offset,
    )


def _squared_component(
    xy: np.ndarray, z: np.ndarray, offset: float
) -> npt.NDArray[np.float64]:
    """Square one Cartesian component of a slab's displacements from the nucleus.

    The component is summed as ``((x + y) + z) + offset`` and the squares are
    added in component order, so each distance has the bits it would have if
    the whole box were broadcast at once.
    """
    value: npt.NDArray[np.float64] = xy[:, :, None] + z
    value += offset
    return np.square(value, out=value)


def sum_radial_weights(
    weights: npt.NDArray[np.float64], samples: Sequence[RadialSample], size: int
) -> npt.NDArray[np.float64]:
    """Accumulate radial atomic weights, including repeated periodic images."""
    result = np.zeros(size)
    for radial, sample in zip(weights, samples, strict=True):
        np.add.at(result, sample.indices, radial[sample.shells])
    return result


def shell_counts(samples: Sequence[RadialSample]) -> npt.NDArray[np.float64]:
    """Each atom's number of sampled points in each radial shell, ``(n, 100)``."""
    counts = np.zeros((len(samples), NSHELLS))
    for atom, sample in enumerate(samples):
        counts[atom] = np.bincount(sample.shells, minlength=NSHELLS)
    return counts


def occupied_counts(
    counts: npt.NDArray[np.float64],
    shells: npt.NDArray[np.integer],
    inverse: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """One atom's shell counts of the points where its gathered ``inverse`` is set.

    ``counts`` are the atom's ``shell_counts`` and ``inverse`` comes from
    ``reciprocal``, zero where its values were too small. Few points lack
    one, so subtracting them gives the counts with less work than counting
    the others again.
    """
    empty = inverse == 0
    if not empty.any():
        return counts
    return counts - np.bincount(shells[empty], minlength=NSHELLS)


def reciprocal(
    values: npt.NDArray[np.float64], threshold: float = ZERO
) -> npt.NDArray[np.float64]:
    """Return ``1 / values`` where values exceed ``threshold``, and zero elsewhere.

    Multiplying sampled weights by this grid replaces a masked division per
    sample point, which dominates the cost of each partition iteration.
    """
    result: npt.NDArray[np.float64] = np.divide(
        1.0, values, out=np.zeros_like(values), where=values > threshold
    )
    return result


def nearest_grid_points(values: np.ndarray) -> npt.NDArray[np.int64]:
    """Round grid coordinates as Fortran NINT: half integers away from zero."""
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)


def wrap_periodic_positions(
    positions: np.ndarray,
    lattice_vectors: np.ndarray,
    periodic: Sequence[bool],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return local cell positions and their original integer image offsets.

    Only periodic fractional coordinates are wrapped. Inputs are not modified,
    and ``positions = wrapped + images @ lattice_vectors``. A pair translation
    from wrapped positions becomes ``translation + images[i] - images[j]``
    when expressed using the original positions.
    """
    wrapped = np.asarray(positions, dtype=np.float64)
    images = np.zeros(wrapped.shape, dtype=np.int64)
    if any(periodic):
        fractional = wrapped @ np.linalg.inv(lattice_vectors)
        images = np.where(periodic, np.floor(fractional), 0).astype(np.int64)
        wrapped = wrapped - images @ lattice_vectors
    return wrapped, images


def integration_extent(
    steps: np.ndarray, *, radius: float = CUTOFF
) -> npt.NDArray[np.int64]:
    """Return a sphere's bounding-box half-widths in grid coordinates.

    Cartesian displacements multiply the inverse voxel matrix on the right.
    Each column's norm therefore bounds that grid coordinate over the sphere,
    including when all three voxel vectors are oblique.
    """
    reciprocal_lengths = np.linalg.norm(np.linalg.inv(steps), axis=0)
    # Retain the extra voxel on each side for consistent grid quadrature.
    extent: npt.NDArray[np.int64] = (
        np.ceil(radius * reciprocal_lengths).astype(np.int64) + 1
    )
    return extent


def nuclear_stencils(grid: DensityGrid) -> tuple[np.ndarray, ...]:
    """Return the 5x5x5 near-nucleus domains used for occupancy correction."""
    offsets = np.array(list(product(range(-2, 3), repeat=3)))
    centers = (grid.positions - grid.origin) @ np.linalg.inv(grid.steps)
    centers = nearest_grid_points(centers)
    shape = np.asarray(grid.density.shape)
    result = []
    for center in centers:
        points = center + offsets
        valid = np.ones(len(points), dtype=bool)
        for axis, periodic in enumerate(grid.periodic):
            if not periodic:
                valid &= (points[:, axis] >= 0) & (points[:, axis] < shape[axis])
        result.append(np.ravel_multi_index((points[valid] % shape).T, tuple(shape)))
    return tuple(result)


def _finite(values: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = _finite_array(values, name)
    if array.shape != shape:
        msg = f"{name} must be a finite array with shape {shape}"
        raise ValueError(msg)
    return array


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        if np.iscomplexobj(values):
            raise TypeError
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"{name} must contain finite real values"
        raise ValueError(msg) from error
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain finite real values"
        raise ValueError(msg)
    return array


def _atomic_numbers(values: np.ndarray) -> np.ndarray:
    try:
        numbers = np.asarray(values)
    except (TypeError, ValueError) as error:
        msg = "atomic_numbers must contain finite integers"
        raise ValueError(msg) from error
    if numbers.ndim != 1 or not numbers.size:
        msg = "atomic_numbers must be a nonempty one-dimensional array"
        raise ValueError(msg)
    if numbers.dtype.kind not in "iuf" or (
        not np.all(np.isfinite(numbers)) or np.any(numbers != np.trunc(numbers))
    ):
        msg = "atomic_numbers must contain finite integers"
        raise ValueError(msg)
    if np.any((numbers < 1) | (numbers > _MAX_ATOMIC_NUMBER)):
        msg = "atomic_numbers must lie in 1-109"
        raise ValueError(msg)
    return np.array(numbers, dtype=np.int64, copy=True)


def _periodicity(values: Sequence[object]) -> tuple[bool, bool, bool]:
    if len(values) != _DIMENSIONS or any(
        not isinstance(value, (bool, np.bool_)) for value in values
    ):
        msg = "periodic must contain three booleans"
        raise ValueError(msg)
    return bool(values[0]), bool(values[1]), bool(values[2])
