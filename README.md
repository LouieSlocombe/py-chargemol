# py-chargemol

py-chargemol computes DDEC6 and DDEC3 atomic charges, atomic multipoles, spin
moments, bond orders, Hirshfeld and CM5 charges, and quadrupole-dipole resorbed
(QDR) point charges directly from electron densities. The numerical algorithms
are implemented in Python and NumPy, based on Chargemol 3.5 (September 26, 2017)
and T. A. Manz's April 9, 2025 QDR program.

After installation, analyses run without a Chargemol executable, Fortran
compiler, external atomic-density directory, or network access. Reference
densities for neutral atoms, ions, and core electrons are bundled for elements
1–109. You can also compute QDR charges from existing Chargemol results or
in-memory DDEC6 arrays.

QDR charges are meant for atom-centred point-charge models in classical force
fields. They fold part of each atom's DDEC6 quadrupole and dipole into the point
charges of its bonded neighbours, weighted by the bond overlap populations, so a
model truncated at monopole order reproduces more of the electrostatic potential
around the material. To interpret charge transfer or the chemical states of
atoms, use the DDEC6 net atomic charges, which are not resorbed.

## Installation

Python 3.14 or newer is required. NumPy 2.3.3 or newer is the only required
runtime dependency and is installed automatically by pip.

Install the latest release from PyPI:

```bash
python -m pip install py-chargemol
py-chargemol --version
```

To install from a checkout instead, run this from the repository root:

```bash
python -m pip install .
```

For optional JIT acceleration of density partitioning, spin, and bond analysis,
install the Numba extra:

```bash
python -m pip install 'py-chargemol[numba]'
```

When Numba is installed, these kernels compile automatically on first use and
cache the compiled code on disk for later runs. The first calculation includes
compilation overhead; repeated calculations benefit most. The default install
uses NumPy, including on free-threaded Python builds. To select the NumPy path
with Numba installed, set `PY_CHARGEMOL_DISABLE_JIT=1` before starting Python.
The [benchmark tool](build_tools/README.md#benchmarking-optional-numba-acceleration)
compares first-use and repeated calculations on the shipped density examples.

See [Development](#development) for an editable install and developer tools, and the
[changelog](https://github.com/LouieSlocombe/py-chargemol/blob/main/CHANGELOG.md)
for release notes. Conda and uv are optional development tools.

You still need an electron density or wavefunction from an electronic-structure
calculation. Atomic coordinates alone do not determine a DDEC analysis:

```text
Electron-density calculation → py-chargemol DDEC6 analysis → QDR charges
```

## Usage

Choose an entry point for the data you already have:

| Starting point | Python API | Command line |
| --- | --- | --- |
| Electron density or wavefunction | [`read_density` → `compute_ddec6`](#compute-directly-from-an-electron-density) | `py-chargemol analyze` |
| Existing Chargemol output files | [`calculate_qdr`](#compute-qdr-from-chargemol-output) | `py-chargemol qdr` |
| DDEC6 arrays in memory | [`compute_qdr_from_arrays`](#compute-directly-from-arrays) | — |

<details>
<summary><strong>Electron-density analysis: Python, command line, and supported inputs</strong></summary>

### Compute directly from an electron density

For a VASP calculation:

```python
from py_chargemol import compute_ddec6, read_density, write_density_results

grid = read_density("path/to/vasp/calculation")
ddec = compute_ddec6(grid)
qdr = ddec.compute_qdr()

print(ddec.charges)
print(ddec.analysis.atomic_dipoles)
print(ddec.bonds.bond_orders)
print(qdr.charges)

# Optional: write DDEC, bond, radial-moment and QDR output files.
write_density_results(ddec, "results", qdr=True)
```

The returned `DDECResult` also exposes `hirshfeld_charges`, `cm5_charges`,
`radial_moments`, and `atomic_densities`. For nonperiodic systems, `ddec.moments`
contains total molecular multipoles; for periodic systems it is `None`.

For magnetic inputs, `ddec.spin.populations` contains atomic spin moments;
`ddec.spin` is `None` when spin data are absent or spin analysis is disabled.
Noncollinear inputs also provide `ddec.spin_gradients`. Isolated-atom WFX inputs
can provide analytic explicit-electron radial moments for powers −1 through 4
in `ddec.wavefunction_radial_moments`; this is `None` for unsupported cases.
The bond result records the remaining population residual after density
correction, also printed in the bond-order output.

The command-line equivalent is:

```bash
py-chargemol analyze path/to/vasp/calculation --output-dir results
```

The command honors `charge type`, `compute bos`, and `print atomic densities`
in `job_control.txt`; explicit command-line options take precedence.

- `--method ddec3` selects the older charge-partitioning method and disables
  QDR, which requires DDEC6. In Python, use `compute_ddec3(grid)`.
- `--no-qdr` writes the density analysis without resorption.
- `--atomic-densities` exports spherical atomic densities and radial integrals.
  In Python, use `write_density_results(ddec, "results", atomic_densities=True)`.
- `--no-bonds` skips bond orders, overlaps, radial moments, atomic-density
  exports, and QDR. In Python, use `compute_ddec6(grid, compute_bos=False)`.

For a charge-only Python calculation, use
`compute_ddec6(grid, compute_bos=False, compute_spin=False)` to also skip spin
partitioning and spin-gradient diagnostics, reducing memory use for magnetic
inputs. `compute_spin=False` requires `compute_bos=False`.

See `py-chargemol analyze --help` for charge, periodicity, and core-electron
options. Output files with matching names are overwritten.

#### Supported inputs

| Source | Inputs |
| --- | --- |
| VASP 4/5 file formats | `AECCAR2` plus `CHGCAR` or `CHG`, and `POTCAR`; optional `AECCAR0`. Explicit core counts can replace `POTCAR` when the geometry names the elements. `CHGCAR` alone is insufficient. |
| Gaussian WFX | Wavefunction with primitive coefficients, occupations, and optional effective-core-density functions. Tight Gaussian contributions are integrated analytically. |
| Gaussian cube | Total electron density, or valence density with explicit core counts; optional separate total and spin-density files. |
| XSF | Scalar valence density, spin-up/spin-down grids, or noncollinear spin components, with explicit core counts. |
| Python arrays | A `DensityGrid` containing atomic numbers, positions, density, origin, voxel vectors (`steps`), and periodicity. Lengths are in bohr and density in electrons/bohr³. |

`read_density` discovers conventional filenames and reads supported settings
from `job_control.txt`. Explicit function arguments override those settings.
For WFX inputs, core counts and periodicity come from wavefunction metadata;
conflicting explicit overrides are rejected. An external atomic-density
directory is unnecessary because the reference data are bundled. The individual
`read_cube`, `read_vasp`, `read_wfx`, and `read_xsf` functions are also public.

Like Chargemol, `read_wfx` evaluates diffuse Gaussian products on coarser
nested grids and interpolates them onto the final grid.
`read_wfx(path, interpolate_diffuse=False)` evaluates them directly on the
final grid instead. This removes the diffuse-grid interpolation error at the
cost of extra computation. It changes the ozone example's charges by about
2e-4 e relative to the default, which follows Chargemol's interpolation.

Cube coordinate units follow the signed voxel counts; electron density defaults
to electrons/bohr³. XSF coordinates are in ångströms, with density also defaulting
to electrons/bohr³. Reader options support explicit unit conversion, and
`read_density` honors the original `onetep` and `e_per_angs3` density-format
settings. Python `DensityGrid` arrays always use atomic units.

For example, a periodic valence-density cube for water with an oxygen
pseudopotential that replaces two core electrons is read with:

```python
from py_chargemol import compute_qdr_from_density, read_cube

grid = read_cube(
    "valence_density.cube",
    density_kind="valence",
    core_electrons=[2, 0, 0],  # atom order: O, H, H
    periodic=(True, True, True),
)
qdr = compute_qdr_from_density(grid)
```

Core counts must match the actual pseudopotentials. For an all-electron cube,
use `read_cube("total_density.cube")`. Set periodicity explicitly when needed.
During analysis, the density integral is checked against the expected electron
count. A discrepancy beyond the allowed tolerance raises an error; check the
charge, units, core counts, and grid resolution.

</details>

### Compute QDR from Chargemol output

From the repository root, try the carbon monoxide example with the Python API.
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

## Project information

<details>
<summary><strong>Compatibility and reference fixtures</strong></summary>

### Compatibility with the original programs

The density engine is based on the September 2017 Chargemol program. Output
files retain the scientific quantities and familiar table layouts, while
status messages and filenames use the Python interface.
Invalid densities, failed partition convergence, and negative valence
populations raise explicit errors. Quadrupole eigenvalues are computed from
the complete tensor, including analytic WFX corrections.

Triclinic integration domains use reciprocal-lattice bounds to include the full
radial cutoff sphere. This corrects omissions in the original angle-based
bounds; for the bundled NaCl example, DDEC6 and DDEC3 charges differ from the
original results by about 2.5e-5 e and 7e-6 e, respectively.

Grid validation is not identical to the original executable: the Python API
permits coarse grids for controlled numerical experiments and does not apply
all of Chargemol's negative-density rejection thresholds. Electron-count and
partition checks do not establish grid convergence; physical calculations
still require a sufficiently resolved input density. Periodic, effective-core,
and ghost-centre WFX paths have analytic and conservation tests, but were not
compared against supplied full reference calculations for those cases.

The QDR regression tests compare every worked example with the Fortran program,
allowing one unit in the last printed digit for rounding and checking tiny
solver residuals against an absolute tolerance of `1e-10`. Some implementation
details differ:

- QDR solves run across atoms using NumPy arrays instead of a separate OpenMP
  thread per atom. QDR atom indices and lattice translations use 64-bit
  integers, so separate `int8` builds for very large cells are not needed.
- The 3×3 quadrupole eigenvalues come from `numpy.linalg.eigvalsh`, not a closed
  form.
- Malformed input raises an explicit error, with file and line information
  where available. Numeric fields that overflow to asterisks are rejected,
  except for the recoverable table counts described below.
- The Fortran program skips 14 lines between the atomic coordinates and the
  multipole table. The September 2017 Chargemol release writes 13, so here the
  table is found by its title line instead, and that release's
  `DDEC6_even_tempered_net_atomic_charges.xyz` can be read too.
- Chargemol writes the atom and pair counts in five-character fields, which
  print as `*****` from 100,000 on. The Fortran program stops on these; here
  the count is taken from the number of rows in the table instead.
- The output's version line names the py-chargemol version. The log file holds
  the same information in a different layout, and the XYZ output always uses
  LF line endings.

### Reference fixtures

`test/worked_examples/` holds the Chargemol inputs and reference QDR charge
outputs from the six worked examples distributed with T. A. Manz's April 9,
2025 QDR program. These files are preserved unmodified. The regression tests
run every example and compare the output with the Fortran program's reference
output line by line.

The density engine has separate offline regression fixtures for the original
VASP NaCl and spin-polarized Gaussian ozone calculations, plus an α-quartz cell
with py-chargemol reference results. See the
[density fixture notes](https://github.com/LouieSlocombe/py-chargemol/blob/main/test/density_examples/README.md) for provenance and
regeneration instructions. NaCl retains the original Chargemol reference data
alongside corrected Python results for full-sphere integration. Numerical
reference cases also cover the
reference-ion, reshaping, and spin routines.
During development it was also checked against
the supplied CP2K water and four Gaussian ozone examples, including charged
and spin-polarized wavefunctions. Ordinary tests use only packaged fixtures;
they do not run Chargemol or require a compiler.

The 1,897 reference-ion and core-density text files used at the default radial
resolution are stored byte-for-byte in compressed, per-element package
archives. A SHA-256 manifest records their provenance. The installed wheel
includes these resources.

</details>

<details>
<summary><strong>Development and project layout</strong></summary>

### Development

Clone the repository and enter its directory:

```bash
git clone https://github.com/LouieSlocombe/py-chargemol.git
cd py-chargemol
```

With [uv](https://docs.astral.sh/uv/), install the package in editable mode and
its development tools using the committed lockfile:

```bash
uv sync --locked
```

Alternatively, create or update the `py-chargemol-dev` conda environment, which also
installs the package in editable mode with its development tools (see
[`build_tools/`](https://github.com/LouieSlocombe/py-chargemol/blob/main/build_tools/README.md)):

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

Install the Git hooks once. They then run Ruff, lockfile, workflow security,
and file hygiene checks before each commit. The hooks skip the reference
fixture data in `test/worked_examples/` and the original WFX regression input
to preserve those files.

```bash
uv run --locked prek install
```

Run the complete local checks:

```bash
uv run --locked prek run --all-files
uv run --locked mypy
uv run --locked pytest --cov
uv build --no-sources
```

`--cov` measures coverage and fails below 100%, as CI does. Leave it off to run a
subset of the tests. Ruff can apply safe lint and formatting changes with:

```bash
uv run --locked ruff check --fix .
uv run --locked ruff format .
```

In the conda environment, run the same tools without the `uv run --locked`
prefix.

#### Dependencies

`pyproject.toml` declares the oldest versions the package supports and
`uv.lock` pins the exact versions used in development and CI. CI also tests
against the oldest allowed versions, so keep lower bounds honest. Update the
conda environment files in `build_tools/` alongside `pyproject.toml`.

Dependabot refreshes `uv.lock`, the pre-commit hooks, and the GitHub Actions
once a month, grouping each into a single pull request.

#### Continuous integration

Every push to `main` and every pull request runs the hooks and mypy, tests on
Linux, macOS, and Windows with Python 3.14 and 3.15, tests the
free-threaded 3.14t build and the oldest allowed dependencies, and builds and
smoke tests the sdist and wheel. Require the **All checks passed** job in
branch protection rather than each matrix entry.

`release.yml` is configured to publish to PyPI with trusted publishing when a
GitHub release is published. Follow the
[release guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/RELEASE.md)
for preparation and publishing steps. Check that the license allows the
distribution first.

### Project layout

```text
.
├── .github/                    # CI, PyPI release, and Dependabot
├── build_tools/                # conda environments, installer and package smoke test
├── src/py_chargemol/
│   ├── ddec6.py                # readers for Chargemol's DDEC6 output
│   ├── elements.py             # element symbols and atomic weights
│   ├── multipoles.py           # total molecular multipole moments
│   ├── units.py                # unit conversions
│   ├── __main__.py             # the py-chargemol command
│   ├── density/                # independent density-to-DDEC calculations
│   │   ├── analysis.py         # density-analysis APIs and result objects
│   │   ├── workflow.py         # input discovery and job_control.txt settings
│   │   ├── grid.py             # density grids and atomic integration samples
│   │   ├── readers.py          # cube, VASP and XSF density input
│   │   ├── wfx.py              # wavefunctions and analytic Gaussian integrals
│   │   ├── core.py             # core partitioning and integration corrections
│   │   ├── references.py       # reference-ion and core-density lookup
│   │   ├── partition.py        # DDEC6 partitioning and radial constraints
│   │   ├── ddec3.py            # DDEC3 partitioning
│   │   ├── reshaping.py        # radial-density tail constraints
│   │   ├── cm5.py              # Hirshfeld-to-CM5 charge correction
│   │   ├── bonds.py            # overlaps, contact exchanges and bond orders
│   │   ├── spin.py             # collinear and noncollinear spin partitioning
│   │   ├── diagnostics.py      # noncollinear spin-gradient diagnostics
│   │   ├── output.py           # density-analysis output files
│   │   └── data/               # bundled reference-ion and core densities
│   └── qdr/
│       ├── arrays.py           # direct computation from Python and NumPy arrays
│       ├── solver.py           # the QDR resorption passes
│       ├── output.py           # QDR-DDEC6_point_charges.xyz writer
│       └── run.py              # calculate_qdr and the legacy run_qdr writer
├── test/                       # unit and worked-example regression tests
│   ├── density_examples/       # density fixtures and provenance notes
│   └── worked_examples/        # unmodified reference inputs and QDR outputs
├── CHANGELOG.md                # release notes
├── RELEASE.md                  # release preparation and publishing
├── pyproject.toml              # project metadata and tool configuration
└── uv.lock                     # pinned development dependencies
```

</details>

<details>
<summary><strong>Citation and license</strong></summary>

### Citation

Publications using these analyses must cite the original methods, as the
licenses require. For DDEC, bond-order, and spin calculations, cite the methods
listed in
[LICENSE-CHARGEMOL](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE-CHARGEMOL),
including the DDEC6 papers by T. A. Manz and N. Gabaldon Limas and the
Chargemol 3.5 program. For QDR charges, cite:

- A. C. Escobosa and T. A. Manz, "How well do various QM-derived net atomic
  charges reproduce the electrostatic potential surrounding a material across
  multiple geometric conformations?", RSC Advances, Vol. 15 (2025) 23223-23265,
  [doi:10.1039/D4RA07900K](https://doi.org/10.1039/D4RA07900K).
- T. A. Manz, InteractionModels program, 2025 version, ddec.sourceforge.net.

The output file keeps the Fortran program's citation text, which predates the
paper's publication.

### License

The QDR implementation retains the original terms, which
[LICENSE](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE)
reproduces. The density-analysis implementation and reference data retain
Chargemol's terms in
[LICENSE-CHARGEMOL](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE-CHARGEMOL).
Both require attribution and citation of the original methods in published
results.

</details>
