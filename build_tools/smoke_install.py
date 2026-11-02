"""Exercise an installed package and its bundled density resources."""

# ruff: noqa: INP001 -- Standalone script executed against the installed wheel.

import numpy as np

from py_chargemol import DensityGrid, __version__, compute_ddec6
from py_chargemol.density.references import core_reference_density, reference_density


def main() -> None:
    """Check both resource families and the complete density-to-QDR pipeline."""
    if reference_density(8).shape != (100,) or core_reference_density(8, 2).shape != (
        100,
    ):
        msg = "installed package is missing its atomic reference densities"
        raise RuntimeError(msg)
    analysis = compute_ddec6(_molecular_grid())
    if analysis.bonds is None or not np.all(analysis.bonds.bond_orders > 0):
        msg = "installed package did not compute the molecular bond orders"
        raise RuntimeError(msg)
    result = analysis.compute_qdr()
    np.testing.assert_allclose(result.charges, [0.0, 0.0], atol=1e-12)
    if result.cg_iterations == 0 or not np.isfinite(result.residual_norm):
        msg = "installed package did not complete the molecular QDR solve"
        raise RuntimeError(msg)
    print(f"py-chargemol {__version__}: standalone density calculation passed")


def _molecular_grid() -> DensityGrid:
    """Build a symmetric two-electron density that exercises bonds and QDR."""
    positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    axis = np.linspace(-4.0, 4.0, 17)
    spacing = axis[1] - axis[0]
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    density = sum(
        np.exp(-np.sum((points - position) ** 2, axis=-1)) for position in positions
    )
    density *= 2 / (density.sum() * spacing**3)
    return DensityGrid(
        atomic_numbers=np.array([1, 1]),
        positions=positions,
        density=density,
        origin=np.full(3, axis[0]),
        steps=np.eye(3) * spacing,
    )


if __name__ == "__main__":
    main()
