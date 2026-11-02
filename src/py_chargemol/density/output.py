"""Write density analyses in the familiar Chargemol XYZ and overlap layouts."""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from py_chargemol._version import __version__
from py_chargemol.ddec6 import DDEC6Analysis, FloatArray, StrPath
from py_chargemol.density.analysis import DDECResult
from py_chargemol.density.grid import NSHELLS, SCALE
from py_chargemol.elements import number_to_symbol
from py_chargemol.qdr.output import _moment_lines, format_qdr_point_charges
from py_chargemol.units import BOHR_PER_ANGSTROM

_BOND_PRINT_CUTOFF = 0.001


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{value:.12e}" for value in values)


def _lattice(analysis: DDEC6Analysis) -> str:
    if not analysis.is_periodic:
        return "Nonperiodic system"
    rows = ", ".join(
        "{ " + _numbers(row) + " }"
        for row in analysis.lattice_vectors / BOHR_PER_ANGSTROM
    )
    return 'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell [' + rows + "]"


def _xyz(analysis: DDEC6Analysis, values: FloatArray) -> list[str]:
    lines = [str(analysis.natoms), _lattice(analysis)]
    columns = np.asarray(values).reshape(analysis.natoms, -1)
    for number, position, row in zip(
        analysis.atomic_numbers,
        analysis.positions / BOHR_PER_ANGSTROM,
        columns,
        strict=True,
    ):
        lines.append(f"{number_to_symbol(int(number))} " + _numbers((*position, *row)))
    return lines


def format_ddec_charges(result: DDECResult) -> str:
    """Format atomic charges, multipoles, and electron-cloud fitting parameters.

    The output preserves the eighteen-column multipole and eight-column cloud
    tables consumed by :func:`read_ddec6_net_atomic_charges`. Cartesian lengths
    are angstrom; charges and atomic multipoles are in atomic units.
    """
    analysis = result.analysis
    lines = _xyz(analysis, analysis.net_charges)
    lines.extend(
        [
            "",
            f"py-chargemol {__version__}; {result.method.upper()} density analysis",
            "Number of radial integration shells = 100",
            f"Number of iterations to convergence = {result.iterations}",
            f"Density normalization factor = {result.normalization:.12e}",
            "",
            (
                "The following XYZ coordinates are in angstroms. "
                "The atomic dipoles and quadrupoles are in atomic units."
            ),
            (
                "atom number, atomic symbol, x, y, z, net_charge, "
                "dipole_x, dipole_y, dipole_z, dipole_mag, Qxy, Qxz, Qyz, "
                "Q(x^2-y^2), Q(3z^2-R^2), three eigenvalues of "
                "traceless quadrupole moment tensor"
            ),
        ]
    )
    for atom, (number, position, charge, dipole, quadrupole) in enumerate(
        zip(
            analysis.atomic_numbers,
            analysis.positions / BOHR_PER_ANGSTROM,
            analysis.net_charges,
            analysis.atomic_dipoles,
            analysis.atomic_quadrupoles,
            strict=True,
        ),
        start=1,
    ):
        xx, xy, xz, yy, yz, zz = quadrupole
        tensor = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        values = (
            *position,
            charge,
            *dipole,
            np.linalg.norm(dipole),
            xy,
            xz,
            yz,
            xx - yy,
            3.0 * zz,
            *np.linalg.eigvalsh(tensor),
        )
        lines.append(f"{atom} {number_to_symbol(int(number))} " + _numbers(values))
    lines.extend(
        [
            "",
            (
                "The spherically averaged electron density is fit to "
                "exp(a - br) for r >= 2 angstrom."
            ),
            (
                "atom number, atomic symbol, x, y, z, a, b, Rsquared; "
                "a and b are in atomic units"
            ),
        ]
    )
    for atom, (number, position, intercept, slope, r_squared) in enumerate(
        zip(
            analysis.atomic_numbers,
            analysis.positions / BOHR_PER_ANGSTROM,
            analysis.cloud_a,
            analysis.cloud_b,
            result.cloud_r_squared,
            strict=True,
        ),
        start=1,
    ):
        lines.append(
            f"{atom} {number_to_symbol(int(number))} "
            + _numbers((*position, intercept, slope, r_squared))
        )
    moments = result.moments
    if moments is not None:
        lines.extend(("", *_moment_lines(moments)))
    if result.spin_gradients is not None:
        gradients = result.spin_gradients
        lines.extend(
            [
                "",
                "Integrated squared spin gradients in atomic units:",
                f"Magnitude contribution = {gradients.magnitude:.12e}",
                f"Total vector contribution = {gradients.total:.12e}",
                f"Directional contribution = {gradients.directional:.12e}",
            ]
        )
    if result.wavefunction_radial_moments is not None:
        lines.extend(
            [
                "",
                (
                    "Analytic single-centre wavefunction radial moments "
                    "(explicit electrons only), in atomic units:"
                ),
                "m = -1, 0, 1, 2, 3, 4",
                _numbers(result.wavefunction_radial_moments),
            ]
        )
    return "\n".join(lines) + "\n"


def format_overlap_populations(result: DDECResult) -> str:
    """Format directed pairs, overlaps, contact exchanges, and bond orders."""
    pairs = result.overlaps
    lines = [
        str(len(pairs)),
        _lattice(result.analysis),
        (
            "atom1, atom2, translation A, translation B, translation C, "
            "overlap population, contact exchange, bond order"
        ),
    ]
    for atom1, atom2, translation, overlap, exchange, bond in zip(
        pairs.atom1,
        pairs.atom2,
        pairs.translations,
        pairs.populations,
        result.bonds.contact_exchanges,
        result.bonds.bond_orders,
        strict=True,
    ):
        indices = " ".join(
            str(int(value)) for value in (atom1 + 1, atom2 + 1, *translation)
        )
        lines.append(indices + " " + _numbers((overlap, exchange, bond)))
    return "\n".join(lines) + "\n"


def _bond_orders(result: DDECResult) -> str:
    lines = _xyz(result.analysis, result.bonds.summed_bond_orders)
    lines.extend(
        [
            "",
            "The sum of bond orders for each atom is listed above.",
            (
                "Number of density correction iterations = "
                f"{result.bonds.density_correction_iterations}"
            ),
            (
                "Maximum atomic population residual after density correction = "
                f"{np.max(np.abs(result.bonds.population_residual)):.12e} electrons"
            ),
            "Bond orders greater than 0.001 are listed below.",
        ]
    )
    pairs = result.overlaps
    for atom in range(result.analysis.natoms):
        symbol = number_to_symbol(int(result.analysis.atomic_numbers[atom]))
        lines.extend(
            [
                "",
                (
                    f"Printing BOs for ATOM # {atom + 1} ( {symbol} ) "
                    "in the reference unit cell."
                ),
            ]
        )
        for pair in np.flatnonzero(
            (pairs.atom1 == atom) & (result.bonds.bond_orders > _BOND_PRINT_CUTOFF)
        ):
            other = int(pairs.atom2[pair])
            other_symbol = number_to_symbol(int(result.analysis.atomic_numbers[other]))
            translation = ", ".join(
                str(int(value)) for value in pairs.translations[pair]
            )
            exchange = result.bonds.contact_exchanges[pair]
            polarization = (
                0.0
                if exchange <= 0
                else np.sqrt(
                    max(
                        result.bonds.spin_polarization_terms[pair] / exchange - 1.0, 0.0
                    )
                )
            )
            lines.append(
                f"Bonded to the ({translation}) translated image of atom number "
                f"{other + 1} ( {other_symbol} ) with bond order = "
                f"{result.bonds.bond_orders[pair]:.12e} "
                f"The average spin polarization of this bonding = {polarization:.12e}"
            )
        lines.append(
            "The sum of bond orders for this atom is SBO = "
            f"{result.bonds.summed_bond_orders[atom]:.12e}"
        )
    return "\n".join(lines) + "\n"


def _radial_blocks(rows: FloatArray) -> list[str]:
    lines: list[str] = []
    for row in rows:
        lines.extend(
            "".join(f"{value:15.8E}" for value in row[start : start + 10])
            for start in range(0, NSHELLS, 10)
        )
        lines.append("")
    return lines


def format_atomic_densities(result: DDECResult) -> str:
    """Format spherical atomic densities and their numerical integration weights.

    The Chargemol layout includes one hundred shell values per atom, followed
    by the analytic shell integrals of ``r**m`` for ``m = -1, 0, ..., 4``.
    All densities, weights, and integrals use atomic units.
    """
    analysis = result.analysis
    lattice = _lattice(analysis).removeprefix(
        'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell '
    )
    lines = [
        f"{analysis.natoms:5d}  atoms",
        lattice,
        f"{NSHELLS:5d}  radial shells",
        "Spherically averaged atomic densities in atomic units printed below.",
        *_radial_blocks(result.atomic_densities),
        "",
        (
            "Radial shell integration weights (summed grid point volumes) "
            "in atomic units printed below."
        ),
        *_radial_blocks(result.radial_shell_weights),
        "",
        (
            "Average analytic radial shell moments (m = -1 to 4) "
            "in atomic units printed below."
        ),
    ]
    edges = np.arange(NSHELLS + 1) / SCALE
    powers = np.arange(2, 8)[:, None]
    moments = 4 * np.pi * np.diff(edges**powers, axis=1) / powers
    lines.extend(_radial_blocks(moments))
    return "\n".join(lines) + "\n"


def write_density_results(
    result: DDECResult,
    directory: StrPath = ".",
    *,
    qdr: bool = False,
    atomic_densities: bool = False,
) -> dict[str, Path]:
    """Write an analysis and return paths keyed by quantity.

    Outputs include the atomic charge/multipole file, directed overlap table,
    bond-order file, CM5 and Hirshfeld charges, three radial-moment files, and
    spin moments when available. If bond analysis was skipped, bond, overlap,
    and radial-moment files are omitted. ``qdr=True`` also writes resorbed charges for
    a DDEC6 analysis. ``atomic_densities=True`` writes spherical atomic densities
    with radial integration weights. Existing files with those names are replaced.
    """
    if result.method not in {"ddec6", "ddec3"}:
        msg = "result method must be ddec6 or ddec3"
        raise ValueError(msg)
    method = result.method.upper()
    texts: dict[str, tuple[str, str]] = {
        "charges": (f"{method}_net_atomic_charges.xyz", format_ddec_charges(result)),
        "cm5": (
            "CM5_net_atomic_charges.xyz",
            "\n".join(_xyz(result.analysis, result.cm5_charges)) + "\n",
        ),
        "hirshfeld": (
            "Hirshfeld_net_atomic_charges.xyz",
            "\n".join(_xyz(result.analysis, result.hirshfeld_charges)) + "\n",
        ),
    }
    if result.has_bond_orders:
        texts["overlaps"] = (
            "overlap_populations_and_contact_exchanges.txt",
            format_overlap_populations(result),
        )
        texts["bond_orders"] = (f"{method}_bond_orders.xyz", _bond_orders(result))
        for column, label in enumerate(("Rsquared", "Rcubed", "Rfourth")):
            text = "\n".join(_xyz(result.analysis, result.radial_moments[:, column]))
            texts[label] = (f"DDEC_atomic_{label}_moments.xyz", text + "\n")
    if result.spin is not None:
        text = "\n".join(_xyz(result.analysis, result.spin.populations))
        texts["spin"] = (f"{method}_atomic_spin_moments.xyz", text + "\n")
    if atomic_densities:
        texts["atomic_densities"] = (
            "spherically_averaged_atomic_densities.txt",
            format_atomic_densities(result),
        )
    if qdr:
        resorbed = result.compute_qdr()
        texts["qdr"] = (
            "QDR-DDEC6_point_charges.xyz",
            format_qdr_point_charges(
                result.analysis, resorbed, resorbed.moments, datetime.now(UTC)
            ),
        )
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, (filename, text) in texts.items():
        path = destination / filename
        path.write_text(text, encoding="utf-8", newline="\n")
        outputs[key] = path
    return outputs
