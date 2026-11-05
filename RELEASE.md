# Releasing py-chargemol

Release versions are set in [pyproject.toml](pyproject.toml), with release notes
in [CHANGELOG.md](CHANGELOG.md). The Git tag must be `v` followed by the package
version. The [Release workflow](.github/workflows/release.yml) is configured to
run when a GitHub release is published. Creating a tag or saving a draft release
does not trigger it.

## Publisher setup

Before publishing, confirm the distribution complies with the original QDR
terms in [LICENSE](LICENSE) and the Chargemol terms in
[LICENSE-CHARGEMOL](LICENSE-CHARGEMOL), including their derived-work and
source-distribution conditions.

Before the first release, create the GitHub repository environment `pypi` and
configure its deployment rules for release tags. Configure a PyPI trusted
publisher with these values, and confirm they still match the repository before
later releases:

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

1. Set the version in `pyproject.toml`, move the relevant `Unreleased` notes in
   `CHANGELOG.md` under that version, and run `uv lock`.
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
   **All checks passed** CI result on the exact commit being released. The
   [CI matrix](.github/workflows/ci.yml) tests Python 3.14 and 3.15 on Linux,
   macOS, and Windows, plus free-threaded Python 3.14 and the oldest allowed
   direct dependencies on Linux.

## Publish

After the release changes and publisher setup have been reviewed:

1. Tag the verified release commit as `v<version>` and push that tag. Use the
   version from `pyproject.toml`; for example, version `1.1.0` uses tag `v1.1.0`.
2. Create a GitHub release for that tag using its changelog entry as the release
   notes, then publish it when ready to upload to PyPI.
3. Watch the **Release** workflow. It is configured to verify the tag matches the
   package version, run the full CI workflow on that tag, check distribution
   metadata, and smoke-test the exact wheel and sdist it uploads. The publish
   job requires validation and build to succeed and uses the `pypi` environment.
4. Check the [PyPI project page](https://pypi.org/project/py-chargemol/) and test
   installation in an isolated environment. From the released checkout, run:

   ```bash
   release_version="$(uv version --short)"
   uv run --isolated --no-project --with "py-chargemol==$release_version" py-chargemol --version
   ```

If validation fails, fix the issue before publishing. If a version has already
been uploaded to PyPI, prepare a new version for any replacement files.
