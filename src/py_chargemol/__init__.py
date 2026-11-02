"""DDEC density partitioning and QDR charges, implemented in Python and NumPy.

Use :func:`read_density` and :func:`compute_ddec6` for density files, or
:class:`DensityGrid` for arrays. :func:`calculate_qdr` and
:func:`compute_qdr_from_arrays` also accept existing DDEC6 results. Computation
does not write files or launch external programs.
"""

from py_chargemol._version import __version__
from py_chargemol.ddec6 import (
    DDEC6Analysis,
    OverlapPairs,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)
from py_chargemol.density.analysis import (
    DDECResult,
    compute_ddec,
    compute_ddec3,
    compute_ddec6,
    compute_qdr_from_density,
)
from py_chargemol.density.grid import DensityGrid
from py_chargemol.density.output import write_density_results
from py_chargemol.density.readers import read_cube, read_vasp, read_xsf
from py_chargemol.density.wfx import read_wfx
from py_chargemol.density.workflow import read_density
from py_chargemol.multipoles import (
    MultipoleMoment,
    MultipoleMoments,
    total_multipole_moments,
)
from py_chargemol.qdr import (
    QDRResult,
    calculate_qdr,
    compute_qdr_charges,
    compute_qdr_from_arrays,
    format_qdr_point_charges,
    run_qdr,
    write_qdr_point_charges,
)

__all__ = [
    "DDEC6Analysis",
    "DDECResult",
    "DensityGrid",
    "MultipoleMoment",
    "MultipoleMoments",
    "OverlapPairs",
    "QDRResult",
    "__version__",
    "calculate_qdr",
    "compute_ddec",
    "compute_ddec3",
    "compute_ddec6",
    "compute_qdr_charges",
    "compute_qdr_from_arrays",
    "compute_qdr_from_density",
    "format_qdr_point_charges",
    "read_cube",
    "read_ddec6_net_atomic_charges",
    "read_density",
    "read_overlap_populations",
    "read_vasp",
    "read_wfx",
    "read_xsf",
    "run_qdr",
    "total_multipole_moments",
    "write_density_results",
    "write_qdr_point_charges",
]
