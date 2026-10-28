# py-chargemol

py-chargemol computes quadrupole-dipole resorbed (QDR) point charges from a
[Chargemol](https://ddec.sourceforge.net) DDEC6 analysis. It is a typed,
NumPy-based Python port of T. A. Manz's QDR program (version April 9, 2025) and
reproduces that program's results and output files.

QDR charges are meant for atom-centred point-charge models in classical force
fields. They fold part of each atom's DDEC6 quadrupole and dipole into the point
charges of its bonded neighbours, weighted by the bond overlap populations, so a
model truncated at monopole order reproduces more of the electrostatic potential
around the material. To interpret charge transfer or the chemical states of
atoms, use the DDEC6 net atomic charges, which are not resorbed.

## Installation

Python 3.14 or newer is required. NumPy is the only runtime dependency.

With conda, create the `py-chargemol-dev` environment, which installs the
package in editable mode with its development tools (see
[`build_tools/`](build_tools/README.md)):

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

With [uv](https://docs.astral.sh/uv/) 0.12 or newer:

```bash
uv sync
```

## Usage

QDR needs two files that Chargemol writes:

- `DDEC6_net_atomic_charges.xyz` (or `DDEC6_even_tempered_net_atomic_charges.xyz`)
- `overlap_populations_and_contact_exchanges.txt` (or `overlap_populations.txt`)

Older Chargemol releases do not write the overlap populations file. The QDR user
guide explains how to get a version that does.

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

### Python

`run_qdr` does the same as the command:

```python
from py_chargemol.qdr import run_qdr

result = run_qdr("path/to/chargemol/output")
```

You can also run each step on its own:

```python
from py_chargemol.ddec6 import read_ddec6_net_atomic_charges, read_overlap_populations
from py_chargemol.multipoles import total_multipole_moments
from py_chargemol.qdr import compute_qdr_charges

analysis = read_ddec6_net_atomic_charges("DDEC6_net_atomic_charges.xyz")
pairs = read_overlap_populations("overlap_populations_and_contact_exchanges.txt")
result = compute_qdr_charges(analysis, pairs)

print(result.charges)  # QDR point charges, e
print(result.atomic_dipoles)  # residual atomic dipoles, e·bohr, shape (n, 3)
print(result.atomic_quadrupoles)  # residual traceless quadrupoles, shape (n, 6)

moments = total_multipole_moments(
    analysis.atomic_numbers,
    analysis.positions,
    result.charges,
    result.atomic_dipoles,
    result.atomic_quadrupoles,
)
print(moments.with_dipoles.dipole_magnitude)
```

Arrays are in atomic units, with one row per atom. Quadrupoles are ordered
`xx, xy, xz, yy, yz, zz`.

## Differences from the Fortran program

The numbers agree with the Fortran program's to their last printed digit on all
of its worked examples. Some implementation details differ:

- Each atom's conjugate gradient solve ran on its own OpenMP thread. Here they
  all run at once on NumPy arrays. Integers are 64-bit throughout, so the
  separate `int8` builds for very large cells are not needed.
- The 3×3 quadrupole eigenvalues come from `numpy.linalg.eigvalsh`, not a closed
  form.
- Malformed input raises an error that names the file and line, where the
  Fortran could read garbage or stop silently.
- Chargemol writes the atom and pair counts in five-character fields, which
  print as `*****` from 100,000 on. The Fortran program stops on these; here
  the count is taken from the number of rows in the table instead.
- The output's version line names the py-chargemol version. The log file holds
  the same information in a different layout, and output files always use LF
  line endings.

## Reference program

[`QDR_program_04_09_2025/`](QDR_program_04_09_2025) holds the original Fortran
source, user guide, and worked examples, unmodified. The prebuilt executables
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
Linux, macOS, and Windows with Python 3.14 and the upcoming 3.15, tests the
free-threaded 3.14t build and the oldest allowed dependencies, and builds and
smoke tests the sdist and wheel. Require the **All checks passed** job in
branch protection rather than each matrix entry.

`release.yml` can publish to PyPI with trusted publishing when a GitHub release
is published. Check that the license allows the distribution first.

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
│       ├── solver.py           # the QDR resorption passes
│       ├── output.py           # QDR-DDEC6_point_charges.xyz writer
│       └── run.py              # run_qdr, the end-to-end program
├── tests/                      # unit and worked-example regression tests
├── pyproject.toml              # project metadata and tool configuration
└── uv.lock                     # pinned development dependencies
```

## License

py-chargemol is a derived work of the QDR program. It is distributed under the
same terms, which [LICENSE](LICENSE) reproduces. Among them: derived works must
keep these terms, and published results must cite the original research papers.
