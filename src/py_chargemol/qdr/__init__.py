"""Quadrupole-dipole resorbed (QDR) point charges for classical force fields.

A Python port of T. A. Manz's QDR program (April 9, 2025). :func:`run_qdr` reproduces
the program: it reads a directory of Chargemol DDEC6 output and writes
``QDR-DDEC6_point_charges.xyz``. :func:`compute_qdr_charges` is the calculation
itself, on data already read with :mod:`py_chargemol.ddec6`.
"""

from py_chargemol.qdr.output import format_qdr_point_charges, write_qdr_point_charges
from py_chargemol.qdr.run import run_qdr
from py_chargemol.qdr.solver import QDRResult, compute_qdr_charges

__all__ = [
    "QDRResult",
    "compute_qdr_charges",
    "format_qdr_point_charges",
    "run_qdr",
    "write_qdr_point_charges",
]
