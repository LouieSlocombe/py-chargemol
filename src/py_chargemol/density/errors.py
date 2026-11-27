"""Numerical convergence failures from the density partitioners."""

from typing import override

import numpy as np


class _ConvergenceError(ValueError):
    """Per-atom residuals and the iteration at which the solver stopped."""

    def __init__(
        self, message: str, residuals: np.ndarray, iterations: int, tolerance: float
    ) -> None:
        """Keep the constructor arguments so worker processes can return the error."""
        self.residuals = residuals.copy()
        self.iterations = iterations
        self.tolerance = tolerance
        magnitude = np.abs(self.residuals)
        self.worst_atoms = np.argsort(-magnitude, kind="stable")
        self.worst_atoms = self.worst_atoms[magnitude[self.worst_atoms] >= tolerance]
        super().__init__(message, self.residuals, iterations, tolerance)

    @override
    def __str__(self) -> str:
        """Report the worst zero-based atom index and residual magnitude."""
        atom = int(np.argmax(np.abs(self.residuals)))
        return (
            f"{self.args[0]} (iterations={self.iterations}, "
            f"max |residual|={abs(self.residuals[atom]):.8g} at atom index {atom}, "
            f"tolerance={self.tolerance:.8g})"
        )


class CoreConvergenceError(_ConvergenceError):
    """Core correction exhausted its iterations.

    `residuals` are the target minus the assigned electron populations.
    `worst_atoms` contains zero-based indices at or above the tolerance,
    ordered by decreasing absolute residual.
    """


class DDEC3ConvergenceError(_ConvergenceError):
    """DDEC3 failed its convergence or nonnegative-population check.

    For iteration exhaustion, each residual is the largest absolute change
    in population or spherical density for that atom over the final four
    iterations. For an invalid final partition, it is the negative part of
    the atom's valence population. `worst_atoms` uses zero-based indices.
    """
