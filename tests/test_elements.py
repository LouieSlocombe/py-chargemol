"""Tests for the element tables."""

import pytest

from py_chargemol.elements import (
    ATOMIC_WEIGHTS,
    ELEMENT_SYMBOLS,
    number_to_symbol,
    symbol_to_number,
)


def test_tables_cover_hydrogen_to_meitnerium() -> None:
    assert len(ELEMENT_SYMBOLS) == len(ATOMIC_WEIGHTS) == 109
    assert ELEMENT_SYMBOLS[0] == "H"
    assert ELEMENT_SYMBOLS[-1] == "Mt"


def test_weights_are_the_coplen_values() -> None:
    """Spot checks against the table in the Fortran source."""
    assert ATOMIC_WEIGHTS[symbol_to_number("C") - 1] == 12.0107
    assert ATOMIC_WEIGHTS[symbol_to_number("O") - 1] == 15.9994
    assert ATOMIC_WEIGHTS[symbol_to_number("Mt") - 1] == 268.0


@pytest.mark.parametrize("number", range(1, 110))
def test_symbols_and_numbers_round_trip(number: int) -> None:
    assert symbol_to_number(number_to_symbol(number)) == number


def test_x_is_an_alias_for_xenon() -> None:
    assert symbol_to_number("X") == 54
    assert number_to_symbol(54) == "Xe"


def test_surrounding_whitespace_is_ignored() -> None:
    assert symbol_to_number(" Cl ") == 17


@pytest.mark.parametrize("symbol", ["", "c", "CL", "Uue"])
def test_unknown_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(ValueError, match="unknown atomic symbol"):
        symbol_to_number(symbol)


@pytest.mark.parametrize("number", [0, 110, -1])
def test_out_of_range_numbers_are_rejected(number: int) -> None:
    with pytest.raises(ValueError, match="outside 1-109"):
        number_to_symbol(number)
