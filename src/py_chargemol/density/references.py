"""Bundled charge-compensated reference ions used by the DDEC methods.

The original 2017 Chargemol radial-density files are stored unmodified in
per-element ZIP archives. No external data directory or download is needed.
"""

from functools import cache
from importlib.resources import files
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt

from py_chargemol.units import BOHR_PER_ANGSTROM

type FloatArray = npt.NDArray[np.float64]

NSHELLS = 100
_MAX_ATOMIC_NUMBER = 109
_DENSITY_FLOOR = 1e-16
_ZERO = 1e-10
SCALE = NSHELLS / (5 * BOHR_PER_ANGSTROM)
SHELL_VOLUMES: FloatArray = (
    4 * np.pi * np.diff(np.arange(NSHELLS + 1, dtype=np.float64) ** 3) / (3 * SCALE**3)
)


@cache
def _archive_densities(atomic_number: int, kind: str) -> dict[int, FloatArray]:
    if not 1 <= atomic_number <= _MAX_ATOMIC_NUMBER:
        msg = "reference atomic numbers must lie in 1-109"
        raise ValueError(msg)
    resource = files("py_chargemol.density").joinpath(
        "data", f"{kind}_{atomic_number:03d}.zip"
    )
    result = {}
    with resource.open("rb") as stream, ZipFile(stream) as archive:
        for name in archive.namelist():
            text = archive.read(name).decode("ascii")
            body = text.split("density in atomic units:", maxsplit=1)[1]
            density = np.fromstring(body, sep=" ", dtype=np.float64)
            density.setflags(write=False)
            electrons = int(name.split("_")[3])
            result[electrons] = density
    return result


@cache
def _raw_densities(atomic_number: int) -> dict[int, FloatArray]:
    return {
        atomic_number - electrons: density
        for electrons, density in _archive_densities(atomic_number, "c2").items()
    }


def core_reference_density(atomic_number: int, core_electrons: int) -> FloatArray:
    """Return raw reference-core shells for pseudopotential reconstruction.

    The density is intentionally not normalized or reshaped here: Chargemol
    corrects its integrated population in the subsequent core iterator.
    """
    raw = _archive_densities(atomic_number, "core")
    if core_electrons == 0:
        return np.zeros(NSHELLS)
    if core_electrons not in raw:
        msg = (
            f"no reference core density for atomic number {atomic_number}, "
            f"{core_electrons} core electrons"
        )
        raise ValueError(msg)
    return raw[core_electrons].copy()


@cache
def _integer_density(atomic_number: int, charge: int) -> FloatArray:
    """Apply Chargemol's cusp, monotonicity, and ion-difference corrections."""
    raw = _raw_densities(atomic_number)
    # reference_density only selects existing boundary ions or adjacent ions
    # strictly inside the contiguous bundled range.
    sign = -1 if charge < 0 else 1
    previous = np.zeros(NSHELLS)
    for ion_charge in range(0, charge + sign, sign):
        electrons = atomic_number - ion_charge
        density = np.maximum(raw[ion_charge], _DENSITY_FLOOR)
        density[0] = max(
            (electrons - float(density[1:] @ SHELL_VOLUMES[1:])) / SHELL_VOLUMES[0],
            _DENSITY_FLOOR,
        )
        density = np.maximum.accumulate(density[::-1])[::-1].copy()
        density = np.maximum(
            density * electrons / (density @ SHELL_VOLUMES), _DENSITY_FLOOR
        )
        population = float(density @ SHELL_VOLUMES)
        if ion_charge != 0:
            for _ in range(100):
                difference = np.maximum.accumulate(
                    np.maximum(sign * (previous - density), 0)[::-1]
                )[::-1]
                corrected = previous - sign * difference
                shift = float((corrected - density) @ SHELL_VOLUMES)
                density = corrected * population / (population + shift)
                if abs(shift) < _DENSITY_FLOOR:
                    break
        previous = density
    previous.setflags(write=False)
    return previous


def reference_density(atomic_number: int, charge: float = 0.0) -> FloatArray:
    """Return a reference ion's 100 radial shells, in electrons per bohr cubed.

    Noninteger charges interpolate adjacent integer ions. Outside the available
    ion range, Chargemol scales the nearest reference to the desired population.
    Returned arrays are independent copies of the immutable cached references.
    """
    if not np.isfinite(charge):
        msg = "reference ion charge must be finite"
        raise ValueError(msg)
    raw = _raw_densities(atomic_number)
    lower, upper = min(raw), max(raw)
    if lower < charge < upper:
        high = int(np.ceil(charge))
        fraction = high - charge
        return fraction * _integer_density(atomic_number, high - 1) + (
            1 - fraction
        ) * _integer_density(atomic_number, high)
    boundary = lower if charge <= lower else upper
    electrons = atomic_number - boundary
    if electrons <= _ZERO:
        return np.zeros(NSHELLS)
    return (
        (atomic_number - charge) / electrons * _integer_density(atomic_number, boundary)
    )


def reference_densities(
    atomic_numbers: npt.NDArray[np.int64], charges: FloatArray
) -> FloatArray:
    """Return one row of radial reference densities per atomic number and charge."""
    return np.array(
        [
            reference_density(int(number), float(charge))
            for number, charge in zip(atomic_numbers, charges, strict=True)
        ],
        dtype=np.float64,
    )
