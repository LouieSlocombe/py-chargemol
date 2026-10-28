# py-chargemol

py-chargemol computes quadrupole-dipole resorbed (QDR) point charges from a
[Chargemol](https://ddec.sourceforge.net) DDEC6 analysis. It is a typed,
NumPy-based Python port of T. A. Manz's QDR program (version April 9, 2025) and
reproduces that program's results. Use it directly with Python arrays or existing
Chargemol files; writing the reference program's output files is optional.

QDR charges are meant for atom-centred point-charge models in classical force
fields. They fold part of each atom's DDEC6 quadrupole and dipole into the point
charges of its bonded neighbours, weighted by the bond overlap populations, so a
model truncated at monopole order reproduces more of the electrostatic potential
around the material. To interpret charge transfer or the chemical states of
atoms, use the DDEC6 net atomic charges, which are not resorbed.

## Installation

Python 3.14 or newer is required. NumPy is the only runtime dependency.

Install the published package with pip:

```bash
python -m pip install py-chargemol
```

See [Development](#development) to install from a source checkout, and the
[changelog](https://github.com/LouieSlocombe/py-chargemol/blob/main/CHANGELOG.md)
for release notes.

## Usage

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
max_cg_iterations=10, tolerance=1e-12)` returns a `QDRResult`. The directory
defaults to the current directory. It discovers these input filenames:

- `DDEC6_net_atomic_charges.xyz` (or `DDEC6_even_tempered_net_atomic_charges.xyz`)
- `overlap_populations_and_contact_exchanges.txt` (or `overlap_populations.txt`)

Pass `charges_file` and `overlaps_file` to choose other input files. A
nonperiodic single atom needs no overlap file and returns its original charge
and atomic multipoles.

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
All arguments are keyword-only; `max_cg_iterations=10` and `tolerance=1e-12`
control the solver. The returned `QDRResult` has the same arrays, diagnostics,
and moments as `calculate_qdr`.

py-chargemol performs QDR resorption using supplied DDEC6 results. It does not
run Chargemol or infer DDEC6 charges, multipoles, or overlap populations from
atomic coordinates. The synthetic values above illustrate the API, not a
physical DDEC6 analysis.

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

The Chargemol release on SourceForge (September 2017) writes an
`overlap_populations.xyz` file, but not
`overlap_populations_and_contact_exchanges.txt`. The QDR user guide says a
pre-release Chargemol version must be used to generate both input files, and
explains how to request it.

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

`--charges`, `--overlaps`, `--output`, and `--log` choose other files. See
`py-chargemol qdr --help`.

## Differences from the Fortran program

The regression tests compare every worked example with the Fortran program,
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

## Reference program

[`QDR_program_04_09_2025/`](https://github.com/LouieSlocombe/py-chargemol/tree/main/QDR_program_04_09_2025)
holds the original Fortran source, user guide, and worked examples, unmodified.
The prebuilt executables
are not tracked. The regression tests run every worked example and compare the
output with the Fortran program's line by line.

## Citation

Publications that use these charges must cite the method, as the license
requires:

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

Alternatively, create the `py-chargemol-dev` conda environment, which also
installs the package in editable mode with its development tools (see
[`build_tools/`](https://github.com/LouieSlocombe/py-chargemol/blob/main/build_tools/README.md)):

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

Install the Git hooks once. They then run Ruff, lockfile, workflow security,
and file hygiene checks before each commit. The hooks skip the reference
program.

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
├── QDR_program_04_09_2025/     # the reference Fortran program, unmodified
├── build_tools/                # conda environments and install script
├── src/py_chargemol/
│   ├── ddec6.py                # readers for Chargemol's DDEC6 output
│   ├── elements.py             # element symbols and atomic weights
│   ├── multipoles.py           # total molecular multipole moments
│   ├── units.py                # unit conversions
│   ├── __main__.py             # the py-chargemol command
│   └── qdr/
│       ├── arrays.py           # direct computation from Python and NumPy arrays
│       ├── solver.py           # the QDR resorption passes
│       ├── output.py           # QDR-DDEC6_point_charges.xyz writer
│       └── run.py              # calculate_qdr and the legacy run_qdr writer
├── tests/                      # unit and worked-example regression tests
├── CHANGELOG.md                # release notes
├── RELEASE.md                  # release preparation and publishing
├── pyproject.toml              # project metadata and tool configuration
└── uv.lock                     # pinned development dependencies
```

## License

py-chargemol is a derived work of the QDR program. It is distributed under the
same terms, which
[LICENSE](https://github.com/LouieSlocombe/py-chargemol/blob/main/LICENSE)
reproduces. Among them: derived works must keep these terms, and published
results must cite the original research papers.
