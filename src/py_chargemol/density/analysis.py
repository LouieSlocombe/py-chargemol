"""DDEC6 and DDEC3 analyses directly from an electron-density grid."""

from dataclasses import dataclass

import numpy as np

from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.density.bonds import BondOrderResult, compute_bond_orders
from py_chargemol.density.cm5 import compute_cm5
from py_chargemol.density.core import (
    CoreResult,
    dominant_occupancy_correction,
    partition_core,
)
from py_chargemol.density.ddec3 import partition_ddec3
from py_chargemol.density.diagnostics import (
    SpinGradientDiagnostics,
    compute_spin_gradient_diagnostics,
)
from py_chargemol.density.grid import (
    NSHELLS,
    SCALE,
    ZERO,
    AtomicSample,
    DensityGrid,
    atomic_samples,
    nuclear_stencils,
    reciprocal,
    sum_radial_weights,
)
from py_chargemol.density.partition import PartitionResult, partition_ddec6
from py_chargemol.density.references import core_reference_density, reference_densities
from py_chargemol.density.spin import SpinResult, partition_spin
from py_chargemol.multipoles import MultipoleMoments, total_multipole_moments
from py_chargemol.qdr.solver import QDRResult, QDRSolver, compute_qdr_charges

_DIMENSIONS = 3
_TAIL_START_SHELL = 39
_MIN_FIT_POINTS = 2


@dataclass(frozen=True)
class DDECResult:
    """Charges, multipoles, bonds and diagnostics computed from a density grid.

    ``radial_moments`` has columns for the second, third and fourth electron
    radial moments. Bond quantities follow the directed ordering in ``overlaps``.
    """

    analysis: DDEC6Analysis
    bond_result: BondOrderResult | None
    _radial_moments: np.ndarray | None
    hirshfeld_charges: np.ndarray
    iterations: int
    normalization: float
    method: str
    spin: SpinResult | None
    cm5_charges: np.ndarray
    cloud_r_squared: np.ndarray
    _radial_shell_weights: np.ndarray | None
    spin_gradients: SpinGradientDiagnostics | None = None
    wavefunction_radial_moments: np.ndarray | None = None

    @property
    def has_bond_orders(self) -> bool:
        """Whether bond orders and associated density moments were computed."""
        return self.bond_result is not None

    @property
    def bonds(self) -> BondOrderResult:
        """Bond results, requiring analysis with ``compute_bos=True``."""
        if self.bond_result is None:
            msg = "bond analysis was skipped; rerun with compute_bos=True"
            raise ValueError(msg)
        return self.bond_result

    @property
    def radial_moments(self) -> np.ndarray:
        """Second, third, and fourth electron radial moments, shape ``(n, 3)``."""
        if self._radial_moments is None:
            msg = "radial moments require bond analysis; rerun with compute_bos=True"
            raise ValueError(msg)
        return self._radial_moments

    @property
    def radial_shell_weights(self) -> np.ndarray:
        """Numerical shell integration volumes, shape ``(n, 100)``."""
        if self._radial_shell_weights is None:
            msg = "radial shell weights require bond analysis; use compute_bos=True"
            raise ValueError(msg)
        return self._radial_shell_weights

    @property
    def charges(self) -> np.ndarray:
        """DDEC net atomic charges in elementary-charge units."""
        return self.analysis.net_charges

    @property
    def overlaps(self) -> OverlapPairs:
        """Directed bond overlap populations for QDR resorption."""
        return self.bonds.overlaps

    @property
    def atomic_densities(self) -> np.ndarray:
        """Corrected spherical atomic electron densities, shape ``(n, 100)``."""
        return self.bonds.corrected_spherical_average

    @property
    def moments(self) -> MultipoleMoments | None:
        """Total molecular multipoles about the centre of mass, if nonperiodic."""
        analysis = self.analysis
        if analysis.is_periodic:
            return None
        return total_multipole_moments(
            analysis.atomic_numbers,
            analysis.positions,
            analysis.net_charges,
            analysis.atomic_dipoles,
            analysis.atomic_quadrupoles,
        )

    def compute_qdr(self, *, solver: QDRSolver = "cg") -> QDRResult:
        """Resorb the computed DDEC6 multipoles into QDR point charges.

        ``solver`` is ``"cg"`` for the QDR program's conjugate gradients, or
        ``"direct"`` for an exact closed-form solve.
        """
        if self.method != "ddec6":
            msg = "QDR resorption requires a DDEC6 analysis"
            raise ValueError(msg)
        return compute_qdr_charges(self.analysis, self.overlaps, solver=solver)


DDEC6Result = DDECResult


def compute_ddec6(
    grid: DensityGrid, *, compute_bos: bool = True, compute_spin: bool = True
) -> DDECResult:
    """Compute DDEC6 charges, atomic multipoles, spin moments and bonds."""
    return compute_ddec(
        grid, method="ddec6", compute_bos=compute_bos, compute_spin=compute_spin
    )


def compute_ddec3(
    grid: DensityGrid, *, compute_bos: bool = True, compute_spin: bool = True
) -> DDECResult:
    """Compute the older DDEC3 charge partition and its associated moments."""
    return compute_ddec(
        grid, method="ddec3", compute_bos=compute_bos, compute_spin=compute_spin
    )


def compute_ddec(
    grid: DensityGrid,
    *,
    method: str = "ddec6",
    compute_bos: bool = True,
    compute_spin: bool = True,
) -> DDECResult:
    """Compute DDEC populations, atomic multipoles, and bonds using NumPy.

    Reference ions and core densities are package resources. No Chargemol
    installation, external executable, or network access is used. Input density
    still comes from an electronic-structure calculation. ``compute_bos=False``
    skips bond orders, overlaps, corrected atomic densities, and radial moments.
    ``compute_spin=False`` skips the spin partition and spin-gradient
    diagnostics of a spin-polarized grid, which hold several arrays per sample
    point and set the peak memory of a charge-only analysis. Bond orders of a
    spin-polarized density depend on its spin partition, so skipping it
    requires ``compute_bos=False``.
    """
    if method not in ("ddec6", "ddec3"):
        msg = "method must be 'ddec6' or 'ddec3'"
        raise ValueError(msg)
    if compute_bos and not compute_spin:
        msg = (
            "bond orders need the spin partition: "
            "compute_spin=False requires compute_bos=False"
        )
        raise ValueError(msg)
    spin_gradients = (
        compute_spin_gradient_diagnostics(grid)
        if compute_spin
        and grid.spin_density is not None
        and grid.spin_density.ndim == _DIMENSIONS + 1
        else None
    )
    samples = atomic_samples(grid)
    neutral = reference_densities(grid.atomic_numbers, np.zeros(grid.natoms))
    core_counts = np.asarray(grid.core_electrons)
    occupancy = np.asarray(grid.occupancy_correction).copy()
    if grid.atomic_corrections is not None:
        occupancy += grid.atomic_corrections[:, 0]
    if grid.pseudodensity is not None:
        occupancy += dominant_occupancy_correction(
            grid.pseudodensity - grid.density, neutral, samples, grid.pixel_volume
        )
    valence_count = float(np.sum(grid.atomic_numbers - core_counts) - grid.net_charge)
    integrated = float(np.sum(grid.density) * grid.pixel_volume + np.sum(occupancy))
    if abs(valence_count - integrated) > max(0.1, 0.001 * valence_count):
        msg = (
            f"density integrates to {integrated:.8g} valence electrons; "
            f"expected {valence_count:.8g}. Check net_charge, core_electrons, "
            "density units, and grid resolution"
        )
        raise ValueError(msg)
    if grid.core_density is None:
        raw_core = np.zeros_like(neutral)
        for atom, (number, count) in enumerate(
            zip(grid.atomic_numbers, core_counts, strict=True)
        ):
            if count:
                raw_core[atom] = core_reference_density(int(number), int(count))
        core_density = sum_radial_weights(raw_core, samples, grid.density.size)
    else:
        core_density = grid.core_density.ravel()
    core = partition_core(
        core_density, core_counts, neutral, samples, grid.pixel_volume
    )
    total_density = grid.density.ravel() + core.density
    partitioner = partition_ddec6 if method == "ddec6" else partition_ddec3
    partition = partitioner(
        atomic_numbers=grid.atomic_numbers,
        density=total_density,
        samples=samples,
        pixel_volume=grid.pixel_volume,
        core_electrons=core_counts,
        core_population=core.populations,
        occupancy_correction=occupancy,
        net_charge=grid.net_charge,
    )
    dipoles, quadrupoles = _multipoles(grid, samples, total_density, partition, core)
    cloud_a, cloud_b, cloud_r_squared = _cloud_parameters(partition.spherical_average)
    analysis = DDEC6Analysis(
        atomic_numbers=grid.atomic_numbers.copy(),
        positions=grid.positions.copy(),
        net_charges=partition.charges,
        atomic_dipoles=dipoles,
        atomic_quadrupoles=quadrupoles,
        cloud_a=cloud_a,
        cloud_b=cloud_b,
        lattice_vectors=grid.lattice_vectors * np.asarray(grid.periodic)[:, None],
    )
    spin = None
    if compute_spin and grid.spin_density is not None:
        shape = (-1,) if grid.spin_density.ndim == _DIMENSIONS else (-1, 3)
        correction = (
            None if grid.atomic_corrections is None else grid.atomic_corrections[:, 1]
        )
        spin = partition_spin(
            density=total_density,
            spin_density=grid.spin_density.reshape(shape),
            weights=partition.weights,
            pseudodensity=partition.pseudodensity,
            spherical_average=partition.spherical_average,
            samples=samples,
            voxel_volume=grid.pixel_volume,
            occupancy_correction=correction,
        )
    bonds = None
    radial_moments = None
    radial_shell_weights = None
    if compute_bos:
        bonds = compute_bond_orders(
            positions=grid.positions,
            lattice_vectors=grid.lattice_vectors,
            periodic=grid.periodic,
            density=total_density,
            weights=partition.weights,
            pseudodensity=partition.pseudodensity,
            samples=samples,
            populations=grid.atomic_numbers - partition.charges,
            voxel_volume=grid.pixel_volume,
            near_nucleus_indices=nuclear_stencils(grid),
            spherical_spin=None if spin is None else spin.spherical_average,
            grid_steps=grid.steps,
            grid_origin=grid.origin,
        )
        radial_moments = _radial_moments(
            samples, bonds.corrected_density, partition, grid.pixel_volume
        )
        radial_shell_weights = np.stack(
            [
                np.bincount(
                    sample.shells,
                    weights=partition.pseudodensity[sample.indices] > ZERO,
                    minlength=NSHELLS,
                )
                * grid.pixel_volume
                for sample in samples
            ]
        )
    return DDECResult(
        analysis=analysis,
        bond_result=bonds,
        _radial_moments=radial_moments,
        hirshfeld_charges=partition.hirshfeld_charges,
        iterations=partition.iterations,
        normalization=partition.normalization,
        method=method,
        spin=spin,
        cm5_charges=compute_cm5(
            atomic_numbers=grid.atomic_numbers,
            positions=grid.positions,
            hirshfeld_charges=partition.hirshfeld_charges,
            lattice_vectors=grid.lattice_vectors,
            periodic=grid.periodic,
        ),
        cloud_r_squared=cloud_r_squared,
        _radial_shell_weights=radial_shell_weights,
        spin_gradients=spin_gradients,
        wavefunction_radial_moments=(
            None
            if grid.wavefunction_radial_moments is None
            else grid.wavefunction_radial_moments.copy()
        ),
    )


def compute_qdr_from_density(
    grid: DensityGrid, *, solver: QDRSolver = "cg"
) -> QDRResult:
    """Compute DDEC6 and then QDR point charges directly from a density grid."""
    return compute_ddec6(grid).compute_qdr(solver=solver)


def _assigned(
    radial: np.ndarray, density: np.ndarray, inverse: np.ndarray, sample: AtomicSample
) -> np.ndarray:
    """Assign density by radial weight, given the reciprocal pseudodensity."""
    return np.asarray(
        radial[sample.shells] * density[sample.indices] * inverse[sample.indices]
    )


def _multipoles(
    grid: DensityGrid,
    samples: tuple[AtomicSample, ...],
    total_density: np.ndarray,
    partition: PartitionResult,
    core: CoreResult,
) -> tuple[np.ndarray, np.ndarray]:
    dipoles = np.zeros((grid.natoms, 3))
    quadrupoles = np.zeros((grid.natoms, 6))
    total_inverse = reciprocal(partition.pseudodensity)
    core_inverse = reciprocal(core.pseudodensity)
    for atom, sample in enumerate(samples):
        total = _assigned(partition.weights[atom], total_density, total_inverse, sample)
        core_local = _assigned(core.weights[atom], core.density, core_inverse, sample)
        electron_weight = (
            (total - core_local) * partition.normalization * grid.pixel_volume
        )
        displacement = sample.displacements
        dipoles[atom] = -electron_weight @ displacement
        tensor = -(displacement.T * electron_weight) @ displacement
        if grid.atomic_corrections is not None:
            correction = grid.atomic_corrections[atom]
            dipoles[atom] += correction[2:5]
            tensor += np.array(
                [
                    [correction[5], correction[8], correction[9]],
                    [correction[8], correction[6], correction[10]],
                    [correction[9], correction[10], correction[7]],
                ]
            )
        tensor -= np.eye(3) * np.trace(tensor) / 3
        quadrupoles[atom] = tensor[np.triu_indices(3)]
    return dipoles, quadrupoles


def _cloud_parameters(
    spherical_average: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = (np.arange(NSHELLS) + 0.5) / SCALE
    intercepts = np.zeros(len(spherical_average))
    slopes = np.zeros(len(spherical_average))
    r_squared = np.zeros(len(spherical_average))
    for atom, density in enumerate(spherical_average):
        active = (np.arange(NSHELLS) >= _TAIL_START_SHELL) & (density > ZERO**1.5)
        if np.count_nonzero(active) >= _MIN_FIT_POINTS:
            x, y = radii[active], np.log(density[active])
            centered = x - np.mean(x)
            slope = float(centered @ (y - np.mean(y)) / (centered @ centered))
            intercepts[atom] = np.mean(y) - slope * np.mean(x)
            slopes[atom] = -slope
            variance = np.sum((y - np.mean(y)) ** 2)
            if variance > 0:
                r_squared[atom] = (centered @ (y - np.mean(y))) ** 2 / (
                    (centered @ centered) * variance
                )
    return intercepts, slopes, r_squared


def _radial_moments(
    samples: tuple[AtomicSample, ...],
    density: np.ndarray,
    partition: PartitionResult,
    pixel_volume: float,
) -> np.ndarray:
    moments = np.zeros((len(samples), 3))
    inverse = reciprocal(partition.pseudodensity)
    for atom, sample in enumerate(samples):
        assigned = _assigned(partition.weights[atom], density, inverse, sample)
        radii = np.linalg.norm(sample.displacements, axis=1)
        for column, power in enumerate((2, 3, 4)):
            moments[atom, column] = np.sum(assigned * radii**power) * pixel_volume
    return moments
