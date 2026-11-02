"""Regular electron-density grids and atom-centred integration domains."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from py_chargemol.units import BOHR_PER_ANGSTROM

NSHELLS = 100
CUTOFF = 5.0 * BOHR_PER_ANGSTROM
SCALE = NSHELLS / CUTOFF
ZERO = 1e-10
_DIMENSIONS = 3
_MAX_ATOMIC_NUMBER = 109


@dataclass
class DensityGrid:
    """A regular density grid, with Cartesian lengths in bohr.

    ``density`` contains valence electrons per bohr cubed; an all-electron
    density uses zero ``core_electrons``. ``steps`` contains the three voxel
    vectors as rows. Periodic grids contain each boundary point only once.
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
    point. ``translations`` gives the corresponding atom image relative to
    the reference cell, when the grid point is folded back into that cell.
    """

    indices: np.ndarray
    shells: np.ndarray
    displacements: np.ndarray
    translations: np.ndarray


def atomic_samples(grid: DensityGrid) -> tuple[AtomicSample, ...]:
    """Construct finite-radius integration domains, including periodic images."""
    inverse = np.linalg.inv(grid.steps)
    lengths = np.linalg.norm(grid.steps, axis=1)
    cosines = grid.steps @ grid.steps.T / np.outer(lengths, lengths)
    sines = np.sqrt(np.maximum(1.0 - cosines**2, 0.0))
    np.fill_diagonal(sines, 1.0)
    # Preserve the original Chargemol integration parallelepiped, including
    # its extra voxel on each side, for consistent grid quadrature.
    extent = np.ceil(CUTOFF / (sines.min(axis=1) * lengths)).astype(np.int64) + 1
    shape = np.asarray(grid.density.shape)
    result = []
    for position in grid.positions:
        center = (position - grid.origin) @ inverse
        nearest = (np.sign(center) * np.floor(np.abs(center) + 0.5)).astype(np.int64)
        lower = nearest - extent
        upper = nearest + extent
        for axis, periodic in enumerate(grid.periodic):
            if not periodic:
                lower[axis] = max(lower[axis], 0)
                upper[axis] = min(upper[axis], shape[axis] - 1)
        axes = [np.arange(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True)]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        displacement = coordinates @ grid.steps + grid.origin - position
        shells = (
            np.ceil(SCALE * np.linalg.norm(displacement, axis=1) + ZERO).astype(
                np.int64
            )
            - 1
        )
        keep = shells < NSHELLS
        coordinates = coordinates[keep]
        result.append(
            AtomicSample(
                indices=np.ravel_multi_index((coordinates % shape).T, tuple(shape)),
                shells=shells[keep],
                displacements=displacement[keep],
                translations=-(coordinates // shape),
            )
        )
    return tuple(result)


def nuclear_stencils(grid: DensityGrid) -> tuple[np.ndarray, ...]:
    """Return the 5x5x5 near-nucleus domains used for occupancy correction."""
    offsets = np.array(list(product(range(-2, 3), repeat=3)))
    centers = (grid.positions - grid.origin) @ np.linalg.inv(grid.steps)
    # Fortran NINT rounds half integers away from zero.
    centers = (np.sign(centers) * np.floor(np.abs(centers) + 0.5)).astype(np.int64)
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
