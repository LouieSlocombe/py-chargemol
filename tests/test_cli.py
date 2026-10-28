"""Tests for the package metadata and command-line interface."""

from importlib.metadata import version

import pytest

from py_chargemol import __version__
from py_chargemol.__main__ import build_parser, main


def test_version_matches_distribution_metadata() -> None:
    """The runtime version stays in sync with package metadata."""
    assert __version__ == version("py-chargemol")


def test_parser_names_the_command() -> None:
    """The parser reports the installed console command name."""
    assert build_parser().prog == "py-chargemol"


def test_cli_without_arguments_succeeds() -> None:
    assert main([]) == 0


def test_cli_reports_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    """The --version flag prints the package version and exits cleanly."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"
