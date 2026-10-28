# py-chargemol

A Python port of T. A. Manz's QDR program for Chargemol DDEC6 analyses. It uses
[uv](https://docs.astral.sh/uv/) for environments, locking, and builds, a `src`
layout, Ruff, strict mypy, pytest with full branch coverage, prek hooks, and
GitHub Actions with trusted publishing to PyPI.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12 or newer.
  uv installs the Python version pinned in `.python-version` (3.14) if it is
  missing.

## Quick start

Create the environment with the package and its development tools:

```bash
uv sync
```

Check the installed command:

```bash
uv run py-chargemol --version
uv run python -m py_chargemol --version
```

## Development

Install the Git hooks once. They then run Ruff, lockfile, workflow security,
and file hygiene checks before each commit:

```bash
uv run prek install
```

Run the complete local checks:

```bash
uv run prek run --all-files
uv run mypy
uv run pytest
uv build
```

Ruff can apply safe lint and formatting changes with:

```bash
uv run ruff check --fix .
uv run ruff format .
```

The hooks use the standard `.pre-commit-config.yaml` format, so
`pre-commit` works too if you prefer it.

### Dependencies

`pyproject.toml` declares the oldest versions the package supports and
`uv.lock` pins the exact versions used in development and CI. CI also tests
against the oldest allowed versions, so keep lower bounds honest.

```bash
uv add requests                # add a runtime dependency
uv add --group dev hypothesis  # add a development tool
uv lock --upgrade              # refresh every locked version
```

Dependabot refreshes `uv.lock`, the pre-commit hooks, and the GitHub Actions
once a month, grouping each into a single pull request.

### Continuous integration

Every push to `main` and every pull request runs the hooks and mypy, tests on
Linux, macOS, and Windows with Python 3.14 and the upcoming 3.15, tests the
free-threaded 3.14t build and the oldest allowed dependencies, and builds and
smoke tests the sdist and wheel. Require the **All checks passed** job in
branch protection rather than each matrix entry.

## Releasing

1. Add the repository as a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) on PyPI, with
   workflow `release.yml` and environment `pypi`, and create a `pypi`
   environment in the repository settings.
2. Bump the version and commit it, for example `uv version --bump minor`.
3. Publish a GitHub release whose tag is the version prefixed with `v`, such as
   `v0.1.0`. The release workflow checks the tag matches, builds the
   distributions, and uploads them to PyPI with attestations.

## Conda

If you prefer Conda, see [`build_tools/`](build_tools/README.md).

## Project layout

```text
.
├── .github/
│   ├── dependabot.yml          # monthly grouped dependency updates
│   └── workflows/              # CI and PyPI release
├── build_tools/                # optional Conda setup
├── src/py_chargemol/           # installable package
├── tests/                      # behavior-focused tests
├── .pre-commit-config.yaml     # Git hooks run by prek or pre-commit
├── .python-version             # development Python version for uv
├── pyproject.toml              # project metadata and tool configuration
└── uv.lock                     # pinned development dependencies
```

## License

Distributed under the QDR program's license terms; see [LICENSE](LICENSE).
