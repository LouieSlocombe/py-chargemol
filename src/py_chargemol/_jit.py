"""Optional, lazily imported Numba compilation for private numerical kernels.

Importing the package or using QDR does not load LLVM. Set
``PY_CHARGEMOL_DISABLE_JIT=1`` before importing to use the NumPy implementation,
even when the optional Numba extra is installed.
"""

import os
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from typing import cast


def _enabled() -> bool:
    return (
        os.environ.get("PY_CHARGEMOL_DISABLE_JIT") != "1"
        and find_spec("numba") is not None
    )


ENABLED = _enabled()


def jit[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Compile on first use, caching machine code and preserving strict arithmetic.

    Without Numba, retain the Python function for testing the kernel against its
    vectorized counterpart; production callers select their NumPy fallback.
    """
    if not ENABLED:
        return function
    compiled: Callable[P, R] | None = None

    @wraps(function)
    def dispatch(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal compiled
        if compiled is None:
            numba = import_module("numba")
            compiled = cast(
                "Callable[P, R]",
                numba.njit(cache=True, nogil=True, fastmath=False)(function),
            )
        return compiled(*args, **kwargs)

    return dispatch
