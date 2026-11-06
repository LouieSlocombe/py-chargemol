# Conda environments

## Development

With conda installed, [conda_install.sh](conda_install.sh) creates or updates the
`py-chargemol-dev` environment from [environment-dev.yml](environment-dev.yml).
The environment includes uv, prek, Ruff, mypy, and pytest, and installs
py-chargemol from this checkout in editable mode. Run these commands from the
repository root:

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

To run the script from another directory, use its absolute path. Rerunning it
updates the environment in place; it does not recreate it or prune packages
that are absent from the environment file. Set `ENV_NAME` to use a different
name:

```bash
ENV_NAME=py-chargemol-test bash build_tools/conda_install.sh
```

With the environment active, run the development tools from the repository root:

```bash
prek install                # once: run the hooks before each commit
prek run --all-files
mypy
pytest --cov                # --cov enforces 100% coverage; omit it for a subset
uv lock                     # after changing dependencies in pyproject.toml
```

Source edits take effect immediately in an editable install. After changing
package metadata such as the version in `pyproject.toml`, refresh the installed
metadata from the repository root:

```bash
python -m pip install --no-deps -e .
```

For dependency changes, rerun `conda_install.sh` after updating the environment
file so conda also resolves the new requirements.

## Using the package

Conda is optional: `python -m pip install .` installs the package and NumPy into
an existing Python 3.14+ environment. [environment.yml](environment.yml) creates
a conda environment with Python, pip, and NumPy. Create it, then install the
package with pip from the repository root:

```bash
conda env create -f build_tools/environment.yml
conda activate py-chargemol
python -m pip install .
```

These environments do not install Chargemol. py-chargemol computes DDEC
analyses and QDR charges itself using bundled reference atomic densities.
You supply an electron density or wavefunction from an electronic-structure
calculation. See the [installation requirements](../README.md#installation).

Conda resolves dependencies from these environment files; it does not use the
versions pinned in `uv.lock`. Keep their dependency constraints aligned with
[pyproject.toml](../pyproject.toml). For the locked development environment, use
the [uv workflow](../README.md#development).

## Benchmarking optional Numba acceleration

[benchmark_numba.py](benchmark_numba.py) measures the shipped quartz DDEC6,
NaCl DDEC3, and triplet-ozone DDEC6 calculations, including bonds and available
spin. It reports input loading separately from the first analysis and repeated
analyses in the same process; ozone loading includes WFX density construction.
QDR and output-file writing are excluded. Install the Numba extra, then compare
both implementations from the repository root:

```bash
uv sync --locked --extra numba
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python build_tools/benchmark_numba.py --output numba.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PY_CHARGEMOL_DISABLE_JIT=1 \
  .venv/bin/python build_tools/benchmark_numba.py --output numpy.json
```

Use `--case quartz`, `--case nacl3`, or `--case ozone` to select one input,
and `--repeat 5` to change the number of warmed runs. Compare the median repeated
analysis times as well as the first-use time. First use can load existing disk
caches; to measure compilation, run one case in a fresh process with
`NUMBA_CACHE_DIR` set to a new empty directory. When running all cases together,
later cases may reuse kernels already compiled by earlier ones. Keep thread
settings, dependencies, and inputs identical between comparisons, and run the
benchmarks sequentially without other CPU-intensive work.

`--profile-dir profiles` also saves a separate warmed `cProfile` run per case,
outside the reported timings. Inspect one with
`python -m pstats profiles/quartz.prof`. JSON output includes dependency versions,
the imported package path, JIT status, and relevant environment variables so
comparisons between checkouts can be reproduced.

The expansion of the initial four Numba kernels was measured on 6 November 2026
against commit `cf956b2`, with Numba enabled in both versions. These are complete
analysis times after warm-up, excluding the separately measured input reading,
QDR, and output writing; each is the median of three runs in one process.

| Shipped input | Grid | Previous kernels | Expanded kernels | Speedup |
| --- | --- | ---: | ---: | ---: |
| Quartz, DDEC6 | 32 × 32 × 32 | 0.562 s | 0.473 s | 1.19× |
| NaCl, DDEC3 | 56 × 56 × 56 | 10.612 s | 3.536 s | 3.00× |
| Triplet ozone, DDEC6 with spin | 168 × 192 × 168 | 2.525 s | 1.957 s | 1.29× |

The machine used an Intel Core i7-14700KF on Linux, Python 3.14.10, NumPy 2.5.4,
Numba 0.68.1, and llvmlite 0.50.0, with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. Charges, dipoles, quadrupoles, radial moments, atomic
densities, overlaps, bond orders, and available spin arrays matched exactly;
charge and spin iteration counts were unchanged. These measurements describe
the shipped fixtures, not a guarantee for other systems.

In a separate first-use check with a fresh Numba cache for each version, quartz
took 1.19 s with the previous kernels and 1.60 s with the expanded kernels.
Additional compilation can outweigh the warmed savings for a single small
calculation. Numba remains optional, and the NumPy implementation is retained.
