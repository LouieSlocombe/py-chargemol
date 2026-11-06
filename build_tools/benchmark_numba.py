"""Measure first-use and warmed density analyses on the shipped fixtures.

Run from a checkout with py-chargemol installed. Select the NumPy implementation
with PY_CHARGEMOL_DISABLE_JIT=1 before starting Python. Input reading is measured
separately, including wavefunction-to-grid construction for the ozone case.
"""

# ruff: noqa: INP001 -- Standalone development script.

import argparse
import cProfile
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import py_chargemol
from py_chargemol import DensityGrid, _jit, compute_ddec, read_density

_CASES = ("quartz", "nacl3", "ozone")
_FIXTURES = Path(__file__).resolve().parents[1] / "test" / "density_examples"


def _positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        msg = "repeat must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return count


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_grid(case: str) -> DensityGrid:
    if case == "ozone":
        return read_density(_FIXTURES / "ozone_triplet_PW91.wfx")
    name = "nacl" if case == "nacl3" else case
    metadata = json.loads((_FIXTURES / f"{name}.json").read_text())
    fields = {}
    for field in ("density", "core_density", "pseudodensity"):
        with np.load(_FIXTURES / f"{name}_{field}.npz", allow_pickle=False) as archive:
            fields[field] = archive["values"]
    return DensityGrid(
        atomic_numbers=np.array(metadata["atomic_numbers"]),
        positions=np.array(metadata["positions"]),
        origin=np.array(metadata["origin"]),
        steps=np.array(metadata["steps"]),
        core_electrons=np.array(metadata["core_electrons"]),
        periodic=tuple(metadata["periodic"]),
        net_charge=metadata["net_charge"],
        **fields,
    )


def _measure(case: str, repeat: int, profile_dir: Path | None) -> dict[str, object]:
    started = perf_counter()
    grid = _load_grid(case)
    input_seconds = perf_counter() - started
    method = "ddec3" if case == "nacl3" else "ddec6"
    started = perf_counter()
    result = compute_ddec(grid, method=method)
    first_seconds = perf_counter() - started
    repeated_seconds = []
    for _ in range(repeat):
        started = perf_counter()
        result = compute_ddec(grid, method=method)
        repeated_seconds.append(perf_counter() - started)
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        with cProfile.Profile() as profiler:
            compute_ddec(grid, method=method)
        profiler.dump_stats(profile_dir / f"{case}.prof")
    return {
        "case": case,
        "method": method,
        "atoms": grid.natoms,
        "grid_shape": grid.density.shape,
        "input_seconds": input_seconds,
        "first_analysis_seconds": first_seconds,
        "repeated_analysis_seconds": repeated_seconds,
        "median_repeated_analysis_seconds": median(repeated_seconds),
        "charge_iterations": result.iterations,
        "spin_iterations": None if result.spin is None else result.spin.iterations,
    }


def main() -> None:
    """Write JSON timings, optionally saving a separate warmed CPU profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *_CASES), default="all")
    parser.add_argument(
        "--repeat", type=_positive_count, default=3, help="warmed runs per case"
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--profile-dir", type=Path, help="save an extra warmed .prof run for each case"
    )
    args = parser.parse_args()
    cases = _CASES if args.case == "all" else (args.case,)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": _package_version("numba"),
        "llvmlite": _package_version("llvmlite"),
        "py_chargemol": py_chargemol.__version__,
        "package_path": py_chargemol.__file__,
        "accelerated_path_selected": _jit.ENABLED,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PY_CHARGEMOL_DISABLE_JIT",
                "NUMBA_DISABLE_JIT",
                "NUMBA_CACHE_DIR",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "timing_scope": "DDEC with bonds and available spin; excludes QDR",
        "first_use_note": (
            "First analysis includes any compilation or cache loading needed in this "
            "process. Earlier cases may already have compiled shared kernels."
        ),
        "cases": [_measure(case, args.repeat, args.profile_dir) for case in cases],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
