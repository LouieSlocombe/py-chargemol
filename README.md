# py-chargemol

py-chargemol computes DDEC6 and DDEC3 atomic charges, atomic multipoles, spin
moments, bond orders, Hirshfeld and CM5 charges, and quadrupole-dipole resorbed
(QDR) point charges directly from electron densities. The numerical algorithms
are implemented in Python and NumPy, based on Chargemol 3.5 (September 26, 2017)
and T. A. Manz's April 9, 2025 QDR program.

Supply an electron density or wavefunction from an electronic-structure
calculation, or use existing Chargemol files or in-memory DDEC6 arrays for QDR.

Use DDEC6 net atomic charges to interpret electron transfer and chemical
states. QDR folds part of the atomic dipoles and quadrupoles into the charges
on bonded neighbours, weighted by their overlap populations. The resulting
point charges are intended for atom-centred force-field models of the
electrostatic potential.

## Installation

Install with Python 3.14 or newer. Pip installs NumPy (2.3.3 or newer), the
only required runtime dependency. Reference densities for elements 1–109 are
bundled, so analyses run offline without a Chargemol executable or Fortran
compiler.

```bash
python -m pip install py-chargemol
py-chargemol --version
```

For a local checkout, run `python -m pip install .` from its root. For optional
Numba acceleration of partitioning, spin, and bond calculations, run
`python -m pip install 'py-chargemol[numba]'`. Numba compiles kernels on
first use and caches them for later runs. To use NumPy with Numba installed,
set `PY_CHARGEMOL_DISABLE_JIT=1` before starting Python. See the
[first-use and repeated-run benchmarks](https://github.com/LouieSlocombe/py-chargemol/blob/main/build_tools/README.md#benchmarking-optional-numba-acceleration).

## Usage

| Starting point | Python API | Command line |
| --- | --- | --- |
| Electron density or wavefunction | [`read_density` → `compute_ddec6`](#compute-directly-from-an-electron-density) | `py-chargemol analyze` |
| Existing Chargemol output files | [`calculate_qdr`](#compute-qdr-from-chargemol-output) | `py-chargemol qdr` |
| DDEC6 arrays in memory | [`compute_qdr_from_arrays`](#compute-directly-from-arrays) | — |

Run the worked examples from the root of a
[repository checkout](https://github.com/LouieSlocombe/py-chargemol) or an
unpacked source distribution. Both include the example inputs under `test/`
and the guides under `docs/`.

### Compute directly from an electron density

To run a complete analysis, use the supplied triplet ozone wavefunction.
With the default settings, this command writes the DDEC6
charges, atomic spin moments, bond orders, and QDR charges to `results/ozone/`.

```bash
py-chargemol analyze test/density_examples/ozone_triplet_PW91.wfx --output-dir results/ozone
```

The first three atom rows in `DDEC6_net_atomic_charges.xyz` give the charges
in e. To obtain the same values in Python, use the density reader and partitioner.

```python
from py_chargemol import compute_ddec6, read_density

grid = read_density("test/density_examples/ozone_triplet_PW91.wfx")
ozone = compute_ddec6(grid)
print(ozone.charges.round(6))  # [ 0.21533  -0.107665 -0.107665]
```

For your own calculation, replace the WFX path with a density file or VASP
calculation directory. The
[density guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/docs/density.md)
covers the Python workflow, file discovery, units, core counts, output files,
and electron-count errors.

### Compute QDR from Chargemol output

Try the carbon monoxide example with the Python API.
The two values are the point charges on carbon and oxygen, in elementary-charge
units.

```python
from py_chargemol import calculate_qdr

co = calculate_qdr("test/worked_examples/CO_molecule")
print(co.charges.round(6))  # [-0.003964  0.003964]
```

For your own Chargemol directory, `py-chargemol qdr DIR` writes
`QDR-DDEC6_point_charges.xyz` and a run log there. For filenames, output
overrides, residual multipoles, and solver diagnostics, see the
[QDR guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/docs/qdr.md).

### Compute directly from arrays

Pass the DDEC6 net charges, atomic dipoles and quadrupoles, geometry, and
directed overlap populations to `compute_qdr_from_arrays`. The
[array example and argument table](https://github.com/LouieSlocombe/py-chargemol/blob/main/docs/qdr.md#compute-from-arrays)
give the units, tensor component order, and periodic-image convention.

## Development

Run `uv sync --locked` from a checkout to install the package and development
tools in editable mode. The [development guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/build_tools/README.md)
covers conda setup, local checks, optional Numba tests, and benchmarks. See
the [changelog](https://github.com/LouieSlocombe/py-chargemol/blob/main/CHANGELOG.md)
and [release guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/RELEASE.md)
for release history and publishing steps.

## Compatibility with the original programs

QDR regression tests compare the six original worked examples with the Fortran
program's output. Density tests include a spin-polarized ozone wavefunction,
NaCl with corrected triclinic integration bounds, and a quartz cell with
recorded Python results. See the [validation notes](https://github.com/LouieSlocombe/py-chargemol/blob/main/test/README.md)
for the comparisons, tolerances, and commands to reproduce them.

## Citation

Publications using these analyses must cite the original methods, as the
licenses require. For DDEC, bond-order, and spin calculations, cite the methods
listed in
[LICENSE-CHARGEMOL](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE-CHARGEMOL),
including the DDEC6 papers by T. A. Manz and N. Gabaldon Limas and the
Chargemol 3.5 program. For QDR charges, cite both the method and the program.

- A. C. Escobosa and T. A. Manz, "How well do various QM-derived net atomic
  charges reproduce the electrostatic potential surrounding a material across
  multiple geometric conformations?", RSC Advances, Vol. 15 (2025) 23223-23265,
  [doi:10.1039/D4RA07900K](https://doi.org/10.1039/D4RA07900K).
- T. A. Manz, InteractionModels program, 2025 version, ddec.sourceforge.net.

The output file keeps the Fortran program's citation text, which predates the
paper's publication.

## License

The QDR implementation retains the original terms, which
[LICENSE](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE)
reproduces. The density-analysis implementation and reference data retain
Chargemol's terms in
[LICENSE-CHARGEMOL](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE-CHARGEMOL).
Both require attribution and citation of the original methods in published
results.
