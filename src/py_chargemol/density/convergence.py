"""Per-calculation controls for DDEC charge convergence."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvergenceOptions:
    """Iteration budgets for core fitting and the DDEC charge tolerance.

    ``weights_max_iterations`` limits the initial core-weight fit.
    ``core_max_iterations`` limits the subsequent core-population correction.
    ``charge_tolerance`` sets the convergence threshold in the core fit,
    DDEC charge partition, and bond-density population correction.
    """

    core_max_iterations: int = 40
    weights_max_iterations: int = 200
    charge_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        """Require positive iteration counts and a finite, positive tolerance."""
        limits: tuple[tuple[str, object], ...] = (
            ("core_max_iterations", self.core_max_iterations),
            ("weights_max_iterations", self.weights_max_iterations),
        )
        for name, value in limits:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                msg = f"{name} must be a positive integer"
                raise ValueError(msg)
        tolerance: object = self.charge_tolerance
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not isfinite(tolerance)
            or tolerance <= 0
        ):
            msg = "charge_tolerance must be finite and positive"
            raise ValueError(msg)


DEFAULT_CONVERGENCE = ConvergenceOptions()
