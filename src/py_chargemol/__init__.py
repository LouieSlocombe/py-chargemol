"""Direct Python access to DDEC6 data and QDR charges, in atomic units.

Use :func:`calculate_qdr` to read a Chargemol directory, or
:func:`compute_qdr_from_arrays` for data already in memory. Both return NumPy
arrays and molecular moments without writing output files.
"""

from py_chargemol._version import __version__
from py_chargemol.ddec6 import (
    DDEC6Analysis,
    OverlapPairs,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)
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
    "MultipoleMoment",
    "MultipoleMoments",
    "OverlapPairs",
    "QDRResult",
    "__version__",
    "calculate_qdr",
    "compute_qdr_charges",
    "compute_qdr_from_arrays",
    "format_qdr_point_charges",
    "read_ddec6_net_atomic_charges",
    "read_overlap_populations",
    "run_qdr",
    "total_multipole_moments",
    "write_qdr_point_charges",
]
