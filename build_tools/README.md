# Conda environments

## Development

`conda_install.sh` creates the `py-chargemol-dev` environment from
`environment-dev.yml`. The environment includes uv, prek, Ruff, mypy, and pytest,
and installs py-chargemol from this checkout in editable mode. Run it from any
directory:

```bash
bash build_tools/conda_install.sh
conda activate py-chargemol-dev
```

The script removes and recreates the environment on every run. Set `ENV_NAME`
to use a different name:

```bash
ENV_NAME=py-chargemol-test bash build_tools/conda_install.sh
```

The development tools run directly inside the environment:

```bash
prek install                # once: run the hooks before each commit
prek run --all-files
mypy
pytest
uv lock                     # after changing dependencies in pyproject.toml
```

## Using the package

`environment.yml` holds only the runtime dependencies. Create it, then install
the package with pip from the repository root:

```bash
conda env create -f build_tools/environment.yml
conda activate py-chargemol
python -m pip install -e .
```

Both conda environments resolve the latest versions that `pyproject.toml`
allows, not the exact versions pinned in `uv.lock`.
