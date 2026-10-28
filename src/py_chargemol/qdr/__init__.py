"""Quadrupole-dipole resorbed (QDR) point charges for classical force fields.

A Python port of T. A. Manz's QDR program (April 9, 2025). :func:`calculate_qdr`
reads a Chargemol directory and returns results without writing files.
:func:`compute_qdr_from_arrays` accepts Python lists or NumPy arrays directly;
:func:`compute_qdr_charges` accepts the data classes in :mod:`py_chargemol.ddec6`.
:func:`run_qdr` also writes the original program's output file.
"""

from py_chargemol.qdr.arrays import compute_qdr_from_arrays
from py_chargemol.qdr.output import format_qdr_point_charges, write_qdr_point_charges
from py_chargemol.qdr.run import calculate_qdr, run_qdr
from py_chargemol.qdr.solver import QDRResult, compute_qdr_charges

__all__ = [
    "QDRResult",
    "calculate_qdr",
    "compute_qdr_charges",
    "compute_qdr_from_arrays",
    "format_qdr_point_charges",
    "run_qdr",
    "write_qdr_point_charges",
]
