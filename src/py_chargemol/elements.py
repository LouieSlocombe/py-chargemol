"""Element symbols and standard atomic weights for hydrogen to meitnerium."""

from typing import Final

ELEMENT_SYMBOLS: Final = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
)  # fmt: skip
"""Element symbols indexed by atomic number minus one."""

ATOMIC_WEIGHTS: Final = (
    1.00794, 4.002602, 6.941, 9.012182, 10.811,
    12.0107, 14.0067, 15.9994, 18.9984032, 20.1797,
    22.989770, 24.3050, 26.981538, 28.0855, 30.973761,
    32.065, 35.453, 39.948, 39.0983, 40.078,
    44.955910, 47.867, 50.9415, 51.9961, 54.938049,
    55.845, 58.933200, 58.6934, 63.546, 65.409,
    69.723, 72.64, 74.92160, 78.96, 79.904,
    83.798, 85.4678, 87.62, 88.90585, 91.224,
    92.90638, 95.94, 98.0, 101.07, 102.90550,
    106.42, 107.8682, 112.411, 114.818, 118.710,
    121.760, 127.60, 126.90447, 131.293, 132.90545,
    137.327, 138.9055, 140.116, 140.90765, 144.24,
    145.0, 150.36, 151.964, 157.25, 158.92534,
    162.500, 164.93032, 167.259, 168.93421, 173.04,
    174.967, 178.49, 180.9479, 183.84, 186.207,
    190.23, 192.217, 195.078, 196.96655, 200.59,
    204.3833, 207.2, 208.98038, 209.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.0381,
    231.03588, 238.02891, 237.0, 244.0, 243.0,
    247.0, 247.0, 251.0, 252.0, 257.0,
    258.0, 259.0, 262.0, 261.0, 262.0,
    266.0, 264.0, 277.0, 268.0,
)  # fmt: skip
"""Standard atomic weights in daltons, indexed by atomic number minus one.

These are the natural-abundance averages from T. B. Coplen, Atomic Weights of the
Elements 1999, with the changes from the 2001 review in Chem. Int. 23, 179 (2001),
as tabulated by NIST. They are the values the QDR program uses for centres of mass.
"""

_SYMBOL_ALIASES: Final = {"X": 54}
"""Alternative spellings Chargemol writes, mapped to atomic numbers."""

_ATOMIC_NUMBERS: Final = {
    **{symbol: number for number, symbol in enumerate(ELEMENT_SYMBOLS, start=1)},
    **_SYMBOL_ALIASES,
}


def symbol_to_number(symbol: str) -> int:
    """Return the atomic number of an element symbol.

    Symbols are case-sensitive, as Chargemol writes them. ``X`` is accepted as an
    alternative spelling of xenon.

    Args:
        symbol: Element symbol such as ``"C"`` or ``"Cl"``.

    Returns:
        The atomic number, from 1 to 109.

    Raises:
        ValueError: If the symbol is not a known element.
    """
    try:
        return _ATOMIC_NUMBERS[symbol.strip()]
    except KeyError:
        msg = f"unknown atomic symbol {symbol!r}"
        raise ValueError(msg) from None


def number_to_symbol(atomic_number: int) -> str:
    """Return the element symbol for an atomic number.

    Args:
        atomic_number: Atomic number from 1 to 109.

    Returns:
        The element symbol.

    Raises:
        ValueError: If no element in the table has that atomic number.
    """
    if not 1 <= atomic_number <= len(ELEMENT_SYMBOLS):
        msg = f"atomic number {atomic_number} is outside 1-{len(ELEMENT_SYMBOLS)}"
        raise ValueError(msg)
    return ELEMENT_SYMBOLS[atomic_number - 1]
