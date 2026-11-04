"""Write the quartz density fixture, or record its DDEC6 results again.

``python test/density_examples/make_quartz.py VASP_DIRECTORY`` reads a VASP
single point with py-chargemol's reader, averages each density over blocks of
3 x 3 x 3 grid points, and writes the averaged grids and their geometry, then
records their DDEC6 results. Without a directory, it keeps the stored grids
and records their results again: run it so after a deliberate change to the
method, and say why in the commit.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from py_chargemol import DensityGrid, compute_ddec6
from py_chargemol.density.readers import read_vasp

HERE = Path(__file__).resolve().parent
METADATA = HERE / "quartz.json"
FIELDS = ("density", "core_density", "pseudodensity")
BLOCK = 3
SOURCES = ("CHGCAR", "AECCAR0", "AECCAR2", "POTCAR")
SOURCE = {
    "description": (
        "Primitive alpha-quartz (Materials Project mp-7000), relaxed with "
        "PBE and rattled by ASE with stdev 0.05 A, seed 1"
    ),
    "code": "VASP 6.6.0",
    "settings": (
        "PBE, PAW_PBE Si 05Jan2001 and O 08Apr2002 (PBE.64), ENCUT 680 eV, "
        "PREC Accurate, LASPH, KSPACING 0.5, ISMEAR 0, SIGMA 0.05, ISPIN 2, "
        "ISYM 0, LAECHG"
    ),
    "grid": "96 x 96 x 96, averaged over blocks of 3 x 3 x 3 to 32 x 32 x 32",
    "spin": (
        "The calculation was spin-polarised but is nonmagnetic: its total "
        "magnetisation is -1.3e-4, so the spin grid is not kept."
    ),
}


def _average(values: np.ndarray) -> np.ndarray:
    """``values`` averaged over blocks of ``BLOCK`` points along each axis."""
    blocks = [size // BLOCK for size in values.shape]
    split = values.reshape(blocks[0], BLOCK, blocks[1], BLOCK, blocks[2], BLOCK)
    return np.asarray(split.mean(axis=(1, 3, 5)))


def _write_values(path: Path, values: np.ndarray) -> None:
    """Save one array in an NPZ archive compressed losslessly with LZMA."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(values))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as archive:
        archive.writestr("values.npy", buffer.getvalue())


def build(directory: Path) -> None:
    """Average a VASP single point's densities and write them and their geometry.

    Each block's average stands at its centre, so every density keeps its
    integral over the cell.

    Raises:
        ValueError: If the grid does not divide into whole blocks.
    """
    # Neutral: NELECT was the POTCARs' valence count. Left to the reader, the
    # charge would be the pseudodensity's integral, off by about 2e-7.
    grid = read_vasp(directory, net_charge=0.0)
    if any(size % BLOCK for size in grid.density.shape):
        msg = f"a grid of {grid.density.shape} does not divide into blocks of {BLOCK}"
        raise ValueError(msg)
    for name in FIELDS:
        _write_values(HERE / f"quartz_{name}.npz", _average(getattr(grid, name)))
    sha256 = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in SOURCES
    }
    metadata = {
        "atomic_numbers": grid.atomic_numbers.tolist(),
        "positions": grid.positions.tolist(),
        "origin": (grid.origin + (BLOCK - 1) / 2 * grid.steps.sum(axis=0)).tolist(),
        "steps": (grid.steps * BLOCK).tolist(),
        "core_electrons": np.asarray(grid.core_electrons).tolist(),
        "periodic": list(grid.periodic),
        "net_charge": grid.net_charge,
        "source": SOURCE,
        "sha256": sha256,
    }
    METADATA.write_text(json.dumps(metadata, indent=1) + "\n")


def load() -> DensityGrid:
    """The stored quartz grid."""
    metadata = json.loads(METADATA.read_text())
    fields = {}
    for name in FIELDS:
        with np.load(HERE / f"quartz_{name}.npz", allow_pickle=False) as archive:
            fields[name] = archive["values"]
    return DensityGrid(
        atomic_numbers=np.array(metadata["atomic_numbers"]),
        positions=np.array(metadata["positions"]),
        origin=np.array(metadata["origin"]),
        steps=np.array(metadata["steps"]),
        core_electrons=np.array(metadata["core_electrons"]),
        periodic=tuple(metadata["periodic"]),
        net_charge=metadata["net_charge"],
        **fields,
    )


def record() -> None:
    """Partition the stored grid and record its DDEC6 results."""
    result = compute_ddec6(load())
    analysis = result.analysis
    overlaps = result.overlaps
    ddec6: dict[str, Any] = {
        "net_charges": analysis.net_charges,
        "atomic_dipoles": analysis.atomic_dipoles,
        "atomic_quadrupoles": analysis.atomic_quadrupoles,
        "cloud_a": analysis.cloud_a,
        "cloud_b": analysis.cloud_b,
        "hirshfeld_charges": result.hirshfeld_charges,
        "cm5_charges": result.cm5_charges,
        "radial_moments": result.radial_moments,
        "summed_bond_orders": result.bonds.summed_bond_orders,
        "overlaps": {
            "atom1": overlaps.atom1,
            "atom2": overlaps.atom2,
            "translations": overlaps.translations,
            "populations": overlaps.populations,
            "bond_orders": result.bonds.bond_orders,
        },
    }
    metadata = json.loads(METADATA.read_text())
    metadata["DDEC6"] = json.loads(json.dumps(ddec6, default=np.ndarray.tolist))
    METADATA.write_text(json.dumps(metadata, indent=1) + "\n")


def main() -> None:
    """Build the fixture from a VASP directory if given, then record results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, nargs="?", help="VASP single point")
    arguments = parser.parse_args()
    if arguments.directory is not None:
        build(arguments.directory)
    record()


if __name__ == "__main__":
    main()
