"""Run QDR on a directory of Chargemol output, as the QDR program does."""

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np

from py_chargemol import __version__
from py_chargemol.ddec6 import (
    StrPath,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)
from py_chargemol.multipoles import MultipoleMoments, total_multipole_moments
from py_chargemol.qdr.output import QDR_PROGRAM_VERSION, write_qdr_point_charges
from py_chargemol.qdr.solver import QDRResult, compute_qdr_charges

CHARGE_FILES: Final = (
    "DDEC6_net_atomic_charges.xyz",
    "DDEC6_even_tempered_net_atomic_charges.xyz",
)
"""Chargemol net atomic charge files, in the order they are looked for."""

OVERLAP_FILES: Final = (
    "overlap_populations_and_contact_exchanges.txt",
    "overlap_populations.txt",
)
"""Chargemol overlap population files, in the order they are looked for."""

OUTPUT_FILE: Final = "QDR-DDEC6_point_charges.xyz"
"""Name of the QDR point charge file."""

LOG_FILE: Final = "QDR_charges_computation_logfile.txt"
"""Name of the QDR program's log file."""

logger = logging.getLogger(__name__)


def run_qdr(
    directory: StrPath = ".",
    *,
    charges_file: StrPath | None = None,
    overlaps_file: StrPath | None = None,
    output_file: StrPath | None = None,
) -> QDRResult | None:
    """Compute QDR point charges for a Chargemol calculation and write them out.

    Progress is logged to this module's logger. As in the QDR program, a system of
    a single atom has nothing to resorb, so nothing is computed or written for one.

    Args:
        directory: Directory holding the Chargemol output files.
        charges_file: Net atomic charge file. By default, the first of
            :data:`CHARGE_FILES` found in ``directory``.
        overlaps_file: Overlap population file. By default, the first of
            :data:`OVERLAP_FILES` found in ``directory``.
        output_file: Where to write the QDR point charges. By default,
            :data:`OUTPUT_FILE` in ``directory``.

    Returns:
        The QDR results, or ``None`` for a single atom.

    Raises:
        FileNotFoundError: If an input file is missing.
        ValueError: If an input file is empty or malformed.
    """
    directory = Path(directory)
    started = time.perf_counter()
    logger.info("Entering QDR program.")
    logger.info("%s (py-chargemol %s)", QDR_PROGRAM_VERSION, __version__)
    logger.info("Copyright (c) 2025 by Thomas A. Manz. Rights reserved.")
    logger.info("%s", f"{datetime.now().astimezone():%Y/%m/%d  %H:%M:%S}")

    charges_path = _input_path(directory, charges_file, CHARGE_FILES)
    logger.info("Reading %s", charges_path)
    analysis = read_ddec6_net_atomic_charges(charges_path)
    logger.info("natoms: %d", analysis.natoms)
    if analysis.is_periodic:
        for number, vector in enumerate(analysis.lattice_vectors, start=1):
            logger.info("vector%d: %s", number, _floats(vector, "13.6f"))
    else:
        logger.info("Nonperiodic system")
    if analysis.natoms == 1:
        logger.info("A single atom has no bonds to resorb multipoles into.")
        return None

    overlaps_path = _input_path(directory, overlaps_file, OVERLAP_FILES)
    logger.info("Reading %s", overlaps_path)
    pairs = read_overlap_populations(overlaps_path)
    logger.info("num_overlap_values: %d", len(pairs))
    read = time.perf_counter()
    logger.info("Input file reading took %.6f seconds", read - started)

    logger.info("Computing the QDR point charges")
    result = compute_qdr_charges(analysis, pairs)
    solved = time.perf_counter()
    logger.info("Solved QDR linear equation system in %.6f seconds", solved - read)
    logger.info("Most conjugate gradient iterations: %d", result.cg_iterations)
    logger.info("Largest final residual norm: %.7E", result.residual_norm)

    moments = None
    if analysis.is_periodic:
        logger.info(
            "Since the system is periodic, total multipole moments are skipped."
        )
    else:
        moments = total_multipole_moments(
            analysis.atomic_numbers,
            analysis.positions,
            result.charges,
            result.atomic_dipoles,
            result.atomic_quadrupoles,
        )
        _log_moments(moments)

    output_path = directory / OUTPUT_FILE if output_file is None else Path(output_file)
    write_qdr_point_charges(
        output_path, analysis, result, moments, datetime.now().astimezone()
    )
    logger.info("Wrote %s", output_path)
    logger.info("Finished QDR program in %.6f seconds", time.perf_counter() - started)
    return result


def _input_path(
    directory: Path, given: StrPath | None, candidates: Sequence[str]
) -> Path:
    if given is not None:
        return Path(given)
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    names = " or ".join(candidates)
    msg = f"could not find {names} in {directory}"
    raise FileNotFoundError(msg)


def _log_moments(moments: MultipoleMoments) -> None:
    logger.info("center_of_mass: %s", _floats(moments.center_of_mass, "10.6f"))
    logger.info("Total multipole moments are computed about the center of mass.")
    orders = (
        ("the net atomic charges", moments.charges),
        ("the net atomic charges and atomic dipoles", moments.with_dipoles),
        (
            "the net atomic charges, atomic dipoles, and atomic quadrupoles",
            moments.with_quadrupoles,
        ),
    )
    for description, moment in orders:
        logger.info("Dipole and quadrupole moments using %s:", description)
        logger.info("dipole (x, y, z): %s", _floats(moment.dipole, "13.6f"))
        logger.info("dipole_magnitude: %13.6f", moment.dipole_magnitude)
        logger.info(
            "quadrupole (xy, xz, yz, x2minusy2, 3z2minusr2): %s",
            _floats(moment.quadrupole, "13.6f"),
        )


def _floats(values: np.ndarray, spec: str) -> str:
    return "".join(f"{value:{spec}}" for value in values)
