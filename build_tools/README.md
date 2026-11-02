# Conda environments

## Development

`conda_install.sh` creates or updates the `py-chargemol-dev` environment from
`environment-dev.yml`. The environment includes uv, prek, Ruff, mypy, and pytest,
and installs py-chargemol from this checkout in editable mode. Run it from any
directory:

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

Rerunning the script updates the environment in place; it does not delete and
recreate it or prune packages that are absent from the environment file. Set
`ENV_NAME` to use a different name:

```bash
ENV_NAME=py-chargemol-test bash build_tools/conda_install.sh
```

The development tools run directly inside the environment:

```bash
prek install                # once: run the hooks before each commit
prek run --all-files
mypy
pytest --cov                # --cov enforces 100% coverage; omit it for a subset
uv lock                     # after changing dependencies in pyproject.toml
```

## Using the package

Conda is optional: `python -m pip install .` installs the package and NumPy into
an existing Python 3.14+ environment. `environment.yml` is an alternative for
creating a conda environment with only the runtime dependencies. Create it,
then install the package with pip from the repository root:

```bash
conda env create -f build_tools/environment.yml
conda activate py-chargemol
python -m pip install .
```

These environments do not install Chargemol. py-chargemol computes DDEC
analyses and QDR charges itself using bundled reference atomic densities.
You supply an electron density or wavefunction from an electronic-structure
calculation. See the [installation requirements](../README.md#installation).

Both conda environments resolve the latest versions that `pyproject.toml`
allows, not the exact versions pinned in `uv.lock`.
