"""Writer for ``QDR-DDEC6_point_charges.xyz``, in the QDR program's layout.

The file starts as an XYZ file of the QDR point charges, readable by Jmol, followed
by the residual atomic dipoles and quadrupoles, the electron cloud parameters, and,
for molecules, the total multipole moments. Numbers use the Fortran program's fixed
formats, so tools that parse its output can read this one too.
"""

import math
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from py_chargemol._version import __version__
from py_chargemol.ddec6 import DDEC6Analysis, StrPath
from py_chargemol.elements import number_to_symbol
from py_chargemol.multipoles import MultipoleMoment, MultipoleMoments
from py_chargemol.qdr.solver import QDRResult
from py_chargemol.units import BOHR_PER_ANGSTROM

QDR_PROGRAM_VERSION = "QDR program version April 9, 2025"
"""The version of T. A. Manz's QDR program that this package ports."""

_ADVICE = (
    "Advice on when to use these QDR point charges:",
    (
        "These QDR point charges are strictly intended to be used for constructing "
        "atom-centered point charge models used in classical force fields."
    ),
    (
        "They partly re-absorb the atomic quadrupoles and dipoles into the point "
        "charges, thereby allowing an electrostatic model truncated at monopole order "
        "(i.e., an atom-centered point charge model) to more accurately reproduce the "
        "electrostatic potential surrounding the material."
    ),
    (
        "For chemical interpretation of electron transfer and for describing the "
        "chemical states of atoms in materials, please use the DDEC6 net atomic "
        "charges that are not quadrupole-dipole resorbed."
    ),
    "",
    "Citation information:",
    (
        "citation for computer program: T. A. Manz, InteractionModels program, 2025 "
        "version, ddec.sourceforge.net"
    ),
    (
        "citation for method to compute QDR-DDEC6 charges: A. C. Escobosa and T. A. "
        "Manz, How well do various QM-derived net atomic charges reproduce the "
        "electrostatic potential surrounding a material across multiple geometric "
        "conformations?, submitted to RSC Advances, 2025."
    ),
)

_CLOUD_DESCRIPTION = (
    "The spherically averaged atomic electron density fit to a function exp(a - br) "
    "for r values having spherically averaged atomic electron density values between "
    "min_density_cloud_penetration and max_density_cloud_penetration."
)

_QUADRUPOLE_COLUMNS = (
    "    (xx-rr/3)           xy           xz    (yy-rr/3)           yz    (zz-rr/3)"
)

_PORT = f"(py-chargemol {__version__})"

_BLANK = " "
"""A blank line as Fortran list-directed output writes it."""


def fortran_es(
    value: float, width: int = 18, digits: int = 7, exponent: int = 3
) -> str:
    """Format a number like the Fortran edit descriptor ``ES18.7E3``.

    Args:
        value: The number to format.
        width: Field width; the result is right-aligned in it.
        digits: Digits after the decimal point of the mantissa.
        exponent: Digits in the exponent.

    Returns:
        The formatted number, such as ``"    2.2204460E-016"``.
    """
    if not math.isfinite(value):
        return f"{value}".rjust(width)
    mantissa, power = f"{value:.{digits}E}".split("E")
    return f"{mantissa}E{int(power):+0{exponent + 1}d}".rjust(width)


def format_qdr_point_charges(
    analysis: DDEC6Analysis,
    result: QDRResult,
    moments: MultipoleMoments | None,
    timestamp: datetime,
) -> str:
    """Lay out QDR results as the QDR program's ``QDR-DDEC6_point_charges.xyz``.

    Args:
        analysis: The DDEC6 analysis the charges were computed from.
        result: The QDR point charges and residual multipoles.
        moments: Total multipole moments to append, for nonperiodic systems.
        timestamp: Time written on the last line.

    Returns:
        The file contents.
    """
    return "\n".join(_lines(analysis, result, moments, timestamp)) + "\n"


def write_qdr_point_charges(
    path: StrPath,
    analysis: DDEC6Analysis,
    result: QDRResult,
    moments: MultipoleMoments | None,
    timestamp: datetime,
) -> None:
    """Write ``QDR-DDEC6_point_charges.xyz``; see :func:`format_qdr_point_charges`."""
    text = format_qdr_point_charges(analysis, result, moments, timestamp)
    # LF on every platform, so the file is the same wherever it is written.
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _lines(
    analysis: DDEC6Analysis,
    result: QDRResult,
    moments: MultipoleMoments | None,
    timestamp: datetime,
) -> Iterator[str]:
    angstrom = analysis.positions / BOHR_PER_ANGSTROM
    yield f"{analysis.natoms:5d}"
    if analysis.is_periodic:
        v1, v2, v3 = (
            _floats(v, 13, 6) for v in analysis.lattice_vectors / BOHR_PER_ANGSTROM
        )
        yield (
            'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell '
            f"[{{ {v1} }}, {{ {v2} }}, {{ {v3} }}]"
        )
    else:
        yield "Nonperiodic system"
    for number, position, charge in zip(
        analysis.atomic_numbers, angstrom, result.charges, strict=True
    ):
        yield f"{number_to_symbol(int(number)):<2}{_floats([*position, charge], 13, 6)}"

    yield _BLANK
    total_charge = result.charges.sum()
    yield f"The sum of net atomic charges in the unit cell is {total_charge:13.6f}"
    yield f"Two-pass QDR computed using {QDR_PROGRAM_VERSION} {_PORT}"
    yield _BLANK
    yield (
        "For each QDR pass, conjugate gradient converged to the linear equation system "
        f"solution within {result.cg_iterations:6d}  CG iterations."
    )
    yield (
        "The maximum final L2 (Euclidean) norm of the residual was "
        f"{fortran_es(result.residual_norm)}"
    )
    yield ""
    yield from _ADVICE
    yield _BLANK

    yield "The XYZ coordinates for atoms at the top of this file are in angstroms."
    yield "The following residual atomic dipoles are in atomic units."
    yield "     dipole_x     dipole_y     dipole_z    dipole_magnitude"
    yield "<atomic dipoles>"
    for dipole in result.atomic_dipoles:
        yield f"{_floats(dipole, 13, 6)}{np.linalg.norm(dipole):20.6f}"
    yield "</atomic dipoles>"
    yield _BLANK

    yield _CLOUD_DESCRIPTION
    yield "            a            b       where a and b are in atomic units"
    yield "<electron cloud parameters>"
    for a, b in zip(analysis.cloud_a, analysis.cloud_b, strict=True):
        yield _floats([a, b], 13, 6)
    yield "</electron cloud parameters>"
    yield _BLANK

    yield "The following atomic quadrupoles are in atomic units."
    yield _QUADRUPOLE_COLUMNS
    yield "<atomic traceless quadrupole cartesian tensor components>"
    for quadrupole in result.atomic_quadrupoles:
        yield _floats(quadrupole, 13, 6)
    yield "</atomic traceless quadrupole cartesian tensor components>"
    yield _BLANK

    if moments is not None:
        yield from _moment_lines(moments)
    yield f" {timestamp:%Y/%m/%d  %H:%M:%S}"


def _moment_lines(moments: MultipoleMoments) -> Iterator[str]:
    yield (
        " Since the system is non-periodic, the total dipole and quadrupole moments "
        "were computed relative to the system center of mass."
    )
    yield " Multipole moments (in atomic units) using the net atomic charges:"
    yield from _moment(
        moments.charges, "Partial_charge_dipole", "Partial_charge_quadrupole"
    )
    yield (
        " Multipole moments (in atomic units) using the net atomic charges and atomic "
        "dipoles:"
    )
    yield from _moment(moments.with_dipoles, "Dipole", "Quadrupole")
    yield (
        " Multipole moments (in atomic units) using the net atomic charges, atomic "
        "dipoles, and atomic quadrupoles:"
    )
    yield from _moment(moments.with_quadrupoles, "Dipole", "Full_quadrupole")
    yield "  "


def _moment(moment: MultipoleMoment, dipole: str, quadrupole: str) -> Iterator[str]:
    # The magnitude label of the charge-only dipole has two extra leading spaces.
    gap = "     " if dipole.startswith("Partial") else "   "
    x, y, z = moment.dipole
    yield (
        f" {dipole}_x = {x:9.4f}   {dipole}_y = {y:9.4f}   {dipole}_z = {z:9.4f}"
        f"{gap}{dipole}_magnitude = {moment.dipole_magnitude:9.4f}"
    )
    labels = ("xy", "xz", "yz", "x2minusy2", "3z2minusr2")
    yield " " + "   ".join(
        f"{quadrupole}_{label} = {value:9.4f}"
        for label, value in zip(labels, moment.quadrupole, strict=True)
    )
    eigenvalues = _floats(moment.quadrupole_eigenvalues, 9, 4)
    yield f" Eigenvalues of traceless quadrupole moment tensor = {eigenvalues}"


def _floats(values: Sequence[float] | np.ndarray, width: int, decimals: int) -> str:
    """Format numbers like the Fortran edit descriptor ``nFw.d``."""
    return "".join(f"{value:{width}.{decimals}f}" for value in values)
