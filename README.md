# py-chargemol

py-chargemol computes DDEC6 and DDEC3 atomic charges, atomic multipoles, spin
moments, bond orders, Hirshfeld and CM5 charges, and quadrupole-dipole resorbed
(QDR) point charges directly from electron densities. The numerical algorithms
are implemented in Python and NumPy, based on Chargemol 3.5 (September 26, 2017)
and T. A. Manz's April 9, 2025 QDR program.

**No Chargemol executable, Fortran compiler, external atomic-density directory,
or network access is needed.** The reference atomic densities are bundled in
the Python package for elements 1–109, including neutral ions, charged ions,
and core electrons. Existing Chargemol result files and in-memory DDEC6 arrays
remain supported.

QDR charges are meant for atom-centred point-charge models in classical force
fields. They fold part of each atom's DDEC6 quadrupole and dipole into the point
charges of its bonded neighbours, weighted by the bond overlap populations, so a
model truncated at monopole order reproduces more of the electrostatic potential
around the material. To interpret charge transfer or the chemical states of
atoms, use the DDEC6 net atomic charges, which are not resorbed.

## Installation

Python 3.14 or newer is required. NumPy 2.3.3 or newer is the only runtime
dependency and is installed automatically by pip.

Install the latest release from PyPI:

```bash
python -m pip install py-chargemol
py-chargemol --version
```

To install from a checkout instead, run this from the repository root:

```bash
python -m pip install .
```

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
| Existing Chargemol output files | [`calculate_qdr`](#python) | `py-chargemol qdr` |
| DDEC6 arrays in memory | [`compute_qdr_from_arrays`](#compute-directly-from-arrays) | — |

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

`DDECResult` also exposes Hirshfeld/CM5 charges, radial moments, spherical
atomic densities, and total molecular multipoles. For magnetic inputs,
`result.spin.populations` contains atomic spin moments; noncollinear inputs
also provide integrated spin-gradient diagnostics in `result.spin_gradients`.
Isolated-atom WFX inputs provide analytic explicit-electron radial moments
for powers −1 through 4 in `result.wavefunction_radial_moments` when supported
by the original diagnostic.
The bond result records the remaining population residual after density
correction, which is also printed in the bond-order output.

The command-line equivalent is:

```bash
py-chargemol analyze path/to/vasp/calculation --output-dir results
```

`--method ddec3` selects the older charge-partitioning method; QDR requires
DDEC6. `--no-qdr` writes the density analysis without resorption. See
`py-chargemol analyze --help` for charge, periodicity, and core-electron options.
The command also honors `charge type` and `print atomic densities` in
`job_control.txt`; explicit command-line options take precedence.
Use `--atomic-densities` to export spherical atomic densities and radial integrals,
or `write_density_results(ddec, "results", atomic_densities=True)` in Python.
`--no-bonds` (or `<compute bos> .false. </compute bos>` in the job controls)
skips bond orders, radial moments, atomic-density exports, and QDR. The Python
equivalent is `compute_ddec6(grid, compute_bos=False)`.

Supported density sources:

| Source | Inputs |
| --- | --- |
| VASP 4/5 | `AECCAR2`, `CHGCAR` or `CHG`, and `POTCAR`; `AECCAR0` when available. Explicit core counts can replace POTCAR when the geometry names the elements. |
| Gaussian WFX | Wavefunction with primitive coefficients, occupations, and optional effective-core-density functions. Tight Gaussian contributions are integrated analytically. |
| Gaussian cube | Total electron density, or valence density with explicit core counts; optional separate total and spin-density files. |
| XSF | Scalar valence density, spin-up/spin-down grids, or noncollinear spin components, with explicit core counts. |
| Python arrays | A `DensityGrid` containing atomic numbers, positions, density, origin, voxel vectors, and periodicity, all in atomic units. |

`read_density` discovers conventional filenames and reads supported settings
from `job_control.txt`. Explicit function arguments override those settings.
Its atomic-densities-directory setting is unnecessary because the reference
data are bundled. The individual `read_cube`, `read_vasp`, `read_wfx`, and
`read_xsf` functions are also public.

Like Chargemol, `read_wfx` evaluates diffuse Gaussian products on coarser
nested grids and interpolates them onto the final grid.
`read_wfx(path, interpolate_diffuse=False)` evaluates them directly on the
final grid instead. That is slower but free of interpolation error, which
moves the ozone example's charges by about 2e-4 e, so results then differ
from Chargemol's at that level.

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
The density integral is checked against the electron count; inconsistent
charge, units, or core counts raise an error.

### Python

Read existing Chargemol output and return the computed arrays without writing
any files:

```python
from py_chargemol import calculate_qdr

result = calculate_qdr("path/to/chargemol/output")

print(result.charges)  # QDR point charges, e, shape (n,)
print(result.atomic_dipoles)  # residual dipoles, e·bohr, shape (n, 3)
print(result.atomic_quadrupoles)  # residual quadrupoles, e·bohr², shape (n, 6)
print(result.cg_iterations, result.residual_norm)  # solver diagnostics

if result.moments is not None:
    print(result.moments.charges.dipole_magnitude)
    print(result.moments.with_dipoles.dipole_magnitude)
    print(result.moments.with_quadrupoles.quadrupole_tensor)
```

`calculate_qdr(directory=".", *, charges_file=None, overlaps_file=None,
max_cg_iterations=10, tolerance=1e-12, solver="cg")` returns a `QDRResult`. The
directory defaults to the current directory. It discovers these input filenames:

- `DDEC6_net_atomic_charges.xyz` (or `DDEC6_even_tempered_net_atomic_charges.xyz`)
- `overlap_populations_and_contact_exchanges.txt`, `overlap_populations.txt`, or
  the September 2017 `overlap_populations.xyz`

Pass `charges_file` and `overlaps_file` to choose other input files. A
nonperiodic single atom needs no overlap file and returns its original charge
and atomic multipoles.

Like the QDR program, the default `solver="cg"` solves each atom's linear
systems by conjugate gradients. Those systems are the identity plus a matrix of
rank at most five, so `solver="direct"` solves them exactly in closed form with
a 5×5 or 3×3 system per atom. Conjugate gradients converge in at most six
iterations, so the two agree to rounding; the direct solver needs no iteration
limit or tolerance and reports zero `cg_iterations`. Every QDR entry point
accepts `solver`, including `compute_qdr_from_arrays`, `compute_qdr_from_density`,
`DDECResult.compute_qdr`, and `write_density_results` (as `qdr_solver`).

For nonperiodic systems, `result.moments` contains the total molecular moments
about the centre of mass at three levels: `charges` (point charges alone),
`with_dipoles` (including residual atomic dipoles), and `with_quadrupoles`
(including residual atomic quadrupoles too). Each provides a `dipole`,
`dipole_magnitude`, `quadrupole`, `quadrupole_tensor`, and
`quadrupole_eigenvalues`. For periodic systems, `result.moments` is `None`.

#### Compute directly from arrays

`compute_qdr_from_arrays` accepts Python lists or NumPy arrays, so an in-memory
workflow needs no intermediate files or electron-cloud parameters. This
synthetic two-atom example is runnable as written:

```python
from py_chargemol import compute_qdr_from_arrays

result = compute_qdr_from_arrays(
    atomic_numbers=[1, 1],
    positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
    net_charges=[0.1, -0.1],
    atomic_dipoles=[[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],
    atomic_quadrupoles=[[0.0] * 6, [0.0] * 6],
    atom1=[0, 1],
    atom2=[1, 0],
    populations=[0.5, 0.5],
)

print(result.charges)
print(result.atomic_dipoles)
```

The inputs describe `n` atoms and `m` directed overlap pairs:

| Argument | Shape | Meaning and units |
| --- | --- | --- |
| `atomic_numbers` | `(n,)` | Atomic numbers |
| `positions` | `(n, 3)` | Cartesian coordinates in bohr |
| `net_charges` | `(n,)` | DDEC6 net atomic charges in e |
| `atomic_dipoles` | `(n, 3)` | DDEC6 atomic dipoles in e·bohr |
| `atomic_quadrupoles` | `(n, 6)` | Traceless DDEC6 quadrupoles in e·bohr², ordered `xx, xy, xz, yy, yz, zz` |
| `atom1`, `atom2` | `(m,)` | Zero-based atom indices |
| `populations` | `(m,)` | Bond overlap populations in e |
| `translations` | `(m, 3)` | Integer lattice translations of the second atom; defaults to zero |
| `lattice_vectors` | `(3, 3)` | Cell vectors as rows, in bohr; defaults to zero for a nonperiodic system |

Each bond must appear in both directions with the same population, as in the
example. For a periodic pair `(i, j)` with translation `t`, also supply `(j, i)`
with translation `-t`. A zero cell-vector row marks a nonperiodic direction.
All arguments are keyword-only; `max_cg_iterations=10`, `tolerance=1e-12`, and
`solver="cg"` control the solver. The returned `QDRResult` has the same arrays,
diagnostics, and moments as `calculate_qdr`.

This array API performs QDR resorption using supplied DDEC6 results. Use
`compute_ddec6` to generate those results from an electron density. The synthetic
values above illustrate the API, not a physical DDEC6 analysis.

#### Work with parsed analysis objects

The readers, `DDEC6Analysis`, `OverlapPairs`, and pure `compute_qdr_charges`
function are also available from the package root:

```python
from py_chargemol import (
    DDEC6Analysis,
    OverlapPairs,
    compute_qdr_charges,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)

analysis: DDEC6Analysis = read_ddec6_net_atomic_charges("DDEC6_net_atomic_charges.xyz")
pairs: OverlapPairs = read_overlap_populations(
    "overlap_populations_and_contact_exchanges.txt"
)
result = compute_qdr_charges(analysis, pairs)
```

The readers convert Chargemol coordinates to bohr and pair indices to zero-based
indices. `compute_qdr_charges` returns the same `QDRResult`, including molecular
moments for nonperiodic systems, without writing files.

#### Write the legacy output file

Use `run_qdr` when you want to compute and write
`QDR-DDEC6_point_charges.xyz`:

```python
from py_chargemol import run_qdr

result = run_qdr("path/to/chargemol/output")
```

Progress goes to the `py_chargemol.qdr.run` Python logger. Configure Python
logging to capture it; the command-line interface creates the logfile
automatically.

### Chargemol input compatibility

The September 2017 `overlap_populations.xyz` file has the same leading columns
as the newer text files and is read directly. A development Chargemol release
is not required. The Python density workflow supplies the analysis and overlaps
directly to QDR without intermediate files.

### Command line

Run `py-chargemol qdr` in the directory that holds the Chargemol output, or pass
the directory:

```bash
py-chargemol qdr path/to/chargemol/output
```

Like the Fortran program, the command writes two files to that directory:

- `QDR-DDEC6_point_charges.xyz`: the QDR point charges as an XYZ file Jmol can
  read, followed by the residual atomic dipoles and quadrupoles, the electron
  cloud parameters, and, for molecules, the total dipole and quadrupole moments.
- `QDR_charges_computation_logfile.txt`: a log of the run.

`--charges`, `--overlaps`, `--output`, and `--log` choose other files.
`--solver direct` solves the linear systems in closed form instead of by
conjugate gradients, and the output file says so in place of the iteration
count; `py-chargemol analyze` takes the same choice as `--qdr-solver`. See
`py-chargemol qdr --help`.

## Compatibility with the original programs

The density engine implements the analyses enabled in the September 2017
Chargemol program. Output files retain the scientific quantities and familiar
table layouts, while status messages and filenames use the Python interface.
Invalid densities, failed partition convergence, and negative valence
populations raise explicit errors. Quadrupole eigenvalues are computed from
the complete tensor, including analytic WFX corrections.

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

- Each atom's conjugate gradient solve ran on its own OpenMP thread. Here they
  all run at once on NumPy arrays. Integers are 64-bit throughout, so the
  separate `int8` builds for very large cells are not needed.
- The 3×3 quadrupole eigenvalues come from `numpy.linalg.eigvalsh`, not a closed
  form.
- Malformed input raises an error that names the file and line, where the
  Fortran could read garbage or stop silently. This includes a number too wide
  for its field, which Fortran prints as asterisks joined to the column before
  it, shifting the columns after it.
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

## Reference fixtures

`test/worked_examples/` holds the Chargemol inputs and reference QDR charge
outputs from the six worked examples distributed with T. A. Manz's April 9,
2025 QDR program. These files are preserved unmodified. The regression tests
run every example and compare the output with the Fortran program's reference
output line by line.

The density engine has separate offline regression fixtures for the original
VASP NaCl and spin-polarized Gaussian ozone calculations, plus numerical
reference cases for the reference-ion, reshaping, and spin routines.
During development it was also checked against
the supplied CP2K water and four Gaussian ozone examples, including charged
and spin-polarized wavefunctions. Ordinary tests use only packaged fixtures;
they do not run Chargemol or require a compiler.

The 1,897 reference-ion and core-density text files used at the default radial
resolution are stored byte-for-byte in compressed, per-element package
archives. A SHA-256 manifest records their provenance. The installed wheel
includes these resources.

## Citation

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

## Development

Clone the repository and enter its directory:

```bash
git clone https://github.com/LouieSlocombe/py-chargemol.git
cd py-chargemol
```

With [uv](https://docs.astral.sh/uv/) 0.12 or newer, install the package in
editable mode and its development tools:

```bash
uv sync
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
uv run prek install
```

Run the complete local checks:

```bash
uv run prek run --all-files
uv run mypy
uv run pytest --cov
uv build
```

`--cov` measures coverage and fails below 100%, as CI does. Leave it off to run a
subset of the tests. Ruff can apply safe lint and formatting changes with:

```bash
uv run ruff check --fix .
uv run ruff format .
```

In the conda environment, run the same tools without `uv run`.

### Dependencies

`pyproject.toml` declares the oldest versions the package supports and
`uv.lock` pins the exact versions used in development and CI. CI also tests
against the oldest allowed versions, so keep lower bounds honest. Update the
conda environment files in `build_tools/` alongside `pyproject.toml`.

Dependabot refreshes `uv.lock`, the pre-commit hooks, and the GitHub Actions
once a month, grouping each into a single pull request.

### Continuous integration

Every push to `main` and every pull request runs the hooks and mypy, tests on
Linux, macOS, and Windows with Python 3.14 and 3.15, tests the
free-threaded 3.14t build and the oldest allowed dependencies, and builds and
smoke tests the sdist and wheel. Require the **All checks passed** job in
branch protection rather than each matrix entry.

`release.yml` can publish to PyPI with trusted publishing when a GitHub release
is published. Follow the
[release guide](https://github.com/LouieSlocombe/py-chargemol/blob/main/RELEASE.md)
for preparation and publishing steps. Check that the license allows the
distribution first.

## Project layout

```text
.
├── .github/                    # CI, PyPI release, and Dependabot
├── build_tools/                # conda environments and install script
├── src/py_chargemol/
│   ├── ddec6.py                # readers for Chargemol's DDEC6 output
│   ├── elements.py             # element symbols and atomic weights
│   ├── multipoles.py           # total molecular multipole moments
│   ├── units.py                # unit conversions
│   ├── __main__.py             # the py-chargemol command
│   ├── density/                # independent density-to-DDEC calculations
│   │   ├── readers.py          # cube, VASP and XSF density input
│   │   ├── wfx.py              # wavefunctions and analytic Gaussian integrals
│   │   ├── partition.py        # DDEC6 partitioning and radial constraints
│   │   ├── ddec3.py            # DDEC3 partitioning
│   │   ├── cm5.py              # Hirshfeld-to-CM5 charge correction
│   │   ├── bonds.py            # overlaps, contact exchanges and bond orders
│   │   ├── spin.py             # collinear and noncollinear spin partitioning
│   │   └── data/               # bundled reference-ion and core densities
│   └── qdr/
│       ├── arrays.py           # direct computation from Python and NumPy arrays
│       ├── solver.py           # the QDR resorption passes
│       ├── output.py           # QDR-DDEC6_point_charges.xyz writer
│       └── run.py              # calculate_qdr and the legacy run_qdr writer
├── test/                       # unit and worked-example regression tests
│   └── worked_examples/        # unmodified reference inputs and QDR outputs
├── CHANGELOG.md                # release notes
├── RELEASE.md                  # release preparation and publishing
├── pyproject.toml              # project metadata and tool configuration
└── uv.lock                     # pinned development dependencies
```

## License

The QDR implementation retains the original terms, which
[LICENSE](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE)
reproduces. The density-analysis implementation and reference data retain
Chargemol's terms in
[LICENSE-CHARGEMOL](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE-CHARGEMOL).
Both require
attribution and citation of the original methods in published results.
