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
