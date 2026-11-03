# Releasing py-chargemol

The next release is **1.0.0**, with Git tag **v1.0.0**. Its release notes are in
[CHANGELOG.md](CHANGELOG.md). Publishing a GitHub release triggers the PyPI
upload; a draft release does not.

## First-release setup

Before publishing, confirm the distribution complies with the original QDR
terms in [LICENSE](LICENSE) and the Chargemol terms in
[LICENSE-CHARGEMOL](LICENSE-CHARGEMOL), including their derived-work and
source-distribution conditions.

Create the GitHub repository environment `pypi` and configure its deployment
rules for release tags. Configure a PyPI trusted publisher with these exact
values:

| Field | Value |
| --- | --- |
| Project | `py-chargemol` |
| Owner | `LouieSlocombe` |
| Repository | `py-chargemol` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

For a new PyPI project, register a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
For an existing project, add a
[trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
in its publishing settings. No long-lived PyPI token is needed by this workflow.

## Prepare and verify

1. Set the version in `pyproject.toml`, update `CHANGELOG.md`, and run `uv lock`.
   Runtime and CLI versions come from installed package metadata; there is no
   separate source version to update.
2. From the repository root, run the checks with the locked dependencies:

   ```bash
   uv sync --locked
   uv run --locked prek run --all-files --show-diff-on-failure
   uv run --locked mypy
   uv run --locked pytest --cov
   uv build --no-sources --clear
   uvx --from twine twine check --strict dist/*
   uv run --isolated --no-project --with dist/*.whl python build_tools/smoke_install.py
   uv run --isolated --no-project --with dist/*.tar.gz python build_tools/smoke_install.py
   ```

   Both smoke tests must print the proposed release version and complete a
   molecular density-to-QDR calculation, including bond orders, using bundled
   reference-ion and core data. The wheel includes `py.typed`, the density
   archives, and both licenses; the sdist also includes the lockfile and the
   tests with their unmodified worked-example fixtures under `test/`.
3. Review and commit the release changes, push them for review, and require the
   **All checks passed** CI result on the exact commit being released. CI covers
   Linux, macOS, and Windows, Python 3.14 and 3.15, free-threaded 3.14, and the
   oldest supported dependencies.

## Publish

After the release changes and first-release setup have been reviewed:

1. Tag the verified release commit as `v1.0.0` and push that tag.
2. Create a GitHub release for that tag using the 1.0.0 changelog entry as its
   notes, then publish it when ready to upload to PyPI.
3. Watch the **Release** workflow. It verifies the tag matches the package
   version, runs the full CI workflow on that tag, checks distribution metadata,
   and smoke-tests the exact wheel and sdist it uploads. The publish job waits
   for validation and build to succeed and uses the `pypi` environment.
4. Check the PyPI project page and test installation outside the checkout:

   ```bash
   uv run --isolated --no-project --with py-chargemol==1.0.0 py-chargemol --version
   ```

If validation fails, fix the issue before publishing. If a version has already
been uploaded to PyPI, prepare a new version for any replacement files.
