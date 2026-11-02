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
    grid = DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        density=np.ones((1, 1, 1)),
        origin=np.zeros(3),
        steps=np.eye(3),
    )
    result = compute_ddec6(grid).compute_qdr()
    np.testing.assert_allclose(result.charges, [0.0], atol=1e-12)
    print(f"py-chargemol {__version__}: standalone density calculation passed")


if __name__ == "__main__":
    main()
