"""Shared fixtures: the worked examples that ship with the reference QDR program."""

from pathlib import Path

import pytest

WORKED_EXAMPLES = (
    Path(__file__).resolve().parents[1] / "QDR_program_04_09_2025" / "worked_examples"
)


@pytest.fixture
def worked_examples() -> Path:
    """Directory holding one subdirectory per worked example."""
    return WORKED_EXAMPLES
