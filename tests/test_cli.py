"""Tests for the package metadata and command-line interface."""

import logging
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pytest

from py_chargemol import __version__
from py_chargemol.__main__ import build_parser, main


@pytest.fixture
def co_inputs(copy_example: Callable[[str], Path]) -> Path:
    """A directory with the CO example's Chargemol output."""
    return copy_example("CO_molecule")


def test_version_matches_distribution_metadata() -> None:
    """The runtime version stays in sync with package metadata."""
    assert __version__ == version("py-chargemol")


def test_parser_names_the_command() -> None:
    """The parser reports the installed console command name."""
    assert build_parser().prog == "py-chargemol"


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "required: command" in capsys.readouterr().err


def test_cli_reports_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    """The --version flag prints the package version and exits cleanly."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"


def test_qdr_writes_into_the_input_directory(
    co_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["qdr", str(co_inputs)]) == 0

    output = co_inputs / "QDR-DDEC6_point_charges.xyz"
    assert output.read_text().startswith("    2\nNonperiodic system\nC ")
    assert (co_inputs / "QDR_charges_computation_logfile.txt").is_file()
    assert capsys.readouterr().out == (
        f"Wrote QDR point charges for 2 atoms to {output}\n"
    )


def test_qdr_defaults_to_the_current_directory(
    co_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(co_inputs)

    assert main(["qdr"]) == 0

    assert (co_inputs / "QDR-DDEC6_point_charges.xyz").is_file()


def test_qdr_accepts_explicit_paths(co_inputs: Path, tmp_path: Path) -> None:
    charges = co_inputs / "DDEC6_even_tempered_net_atomic_charges.xyz"
    overlaps = co_inputs / "overlap_populations_and_contact_exchanges.txt"
    output, log = tmp_path / "charges.xyz", tmp_path / "qdr.log"

    exit_code = main(
        [
            "qdr",
            str(tmp_path / "unused"),
            *("--charges", str(charges), "--overlaps", str(overlaps)),
            *("--output", str(output), "--log", str(log)),
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    assert "Entering QDR program." in log.read_text()


def test_qdr_falls_back_to_older_file_names(co_inputs: Path) -> None:
    newer = co_inputs / "overlap_populations_and_contact_exchanges.txt"
    newer.rename(co_inputs / "overlap_populations.txt")

    assert main(["qdr", str(co_inputs)]) == 0

    log = (co_inputs / "QDR_charges_computation_logfile.txt").read_text()
    assert "overlap_populations.txt" in log


def test_qdr_reports_missing_inputs(
    co_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (co_inputs / "overlap_populations_and_contact_exchanges.txt").unlink()

    assert main(["qdr", str(co_inputs)]) == 1

    error = capsys.readouterr().err
    assert error.startswith("py-chargemol qdr: error: could not find ")
    assert "overlap_populations.txt" in error
    log = (co_inputs / "QDR_charges_computation_logfile.txt").read_text()
    assert "Program will terminate." in log
    assert not (co_inputs / "QDR-DDEC6_point_charges.xyz").exists()


def test_qdr_reports_malformed_inputs(
    co_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (co_inputs / "DDEC6_even_tempered_net_atomic_charges.xyz").write_text("\n")

    assert main(["qdr", str(co_inputs)]) == 1

    assert "the file is empty" in capsys.readouterr().err


def test_qdr_reports_an_unwritable_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["qdr", str(tmp_path / "missing")]) == 1

    assert "cannot write the log file" in capsys.readouterr().err


@pytest.mark.parametrize("level", [logging.NOTSET, logging.WARNING])
@pytest.mark.parametrize(("missing_overlaps", "expected_exit"), [(False, 0), (True, 1)])
def test_qdr_restores_logging_configuration(
    co_inputs: Path,
    caplog: pytest.LogCaptureFixture,
    level: int,
    missing_overlaps: bool,
    expected_exit: int,
) -> None:
    """Calling the CLI leaves the caller's logging settings intact, even on error."""
    caplog.set_level(level, logger="py_chargemol")
    package_logger = logging.getLogger("py_chargemol")
    previous_handlers = package_logger.handlers.copy()
    if missing_overlaps:
        (co_inputs / "overlap_populations_and_contact_exchanges.txt").unlink()

    assert main(["qdr", str(co_inputs)]) == expected_exit

    assert package_logger.level == level
    assert package_logger.handlers == previous_handlers


def test_qdr_skips_a_single_atom(
    co_inputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """As in the QDR program, a lone atom needs no overlap file and gets no output."""
    charges = co_inputs / "DDEC6_even_tempered_net_atomic_charges.xyz"
    lines = charges.read_text().splitlines()
    lines[0] = "    1"
    # Drop oxygen's coordinate row ("O x y z q") and its table rows ("2 O ...").
    carbon_only = [
        line
        for line in lines
        if line.split()[:1] != ["O"] and line.split()[:2] != ["2", "O"]
    ]
    charges.write_text("\n".join(carbon_only) + "\n")
    (co_inputs / "overlap_populations_and_contact_exchanges.txt").unlink()

    assert main(["qdr", str(co_inputs)]) == 0

    assert "nothing written" in capsys.readouterr().out
    assert not (co_inputs / "QDR-DDEC6_point_charges.xyz").exists()
