"""Physical spin-gradient limits and Cartesian finite differences."""

import numpy as np
import pytest

from py_chargemol.density.diagnostics import compute_spin_gradient_diagnostics
from py_chargemol.density.grid import DensityGrid


def spin_grid(
    spin: np.ndarray | None,
    *,
    steps: np.ndarray | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> DensityGrid:
    return DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        density=np.ones((3, 3, 3)),
        origin=np.zeros(3),
        steps=np.eye(3) if steps is None else steps,
        periodic=periodic,
        spin_density=spin,
    )


@pytest.mark.parametrize("spin", [None, np.zeros((3, 3, 3))])
def test_diagnostics_require_vector_spin(spin: np.ndarray | None) -> None:
    with pytest.raises(ValueError, match="three-component"):
        compute_spin_gradient_diagnostics(spin_grid(spin))


@pytest.mark.parametrize("value", [0, 1e-12, 1])
def test_constant_spin_has_no_gradients(value: float) -> None:
    result = compute_spin_gradient_diagnostics(spin_grid(np.full((3, 3, 3, 3), value)))
    assert (result.magnitude, result.total, result.directional) == (0, 0, 0)


def test_linear_spin_magnitude_uses_cartesian_skew_grid_gradients() -> None:
    steps = np.array([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.4]])
    cartesian = np.moveaxis(np.indices((3, 3, 3)), 0, -1) @ steps
    gradient = np.array([0.3, -0.4, 0.5])
    magnitude = 1 + cartesian @ gradient
    spin = np.zeros((3, 3, 3, 3))
    spin[..., 2] = magnitude
    grid = spin_grid(spin, steps=steps)
    result = compute_spin_gradient_diagnostics(grid)
    expected = 8 * grid.pixel_volume * float(gradient @ gradient)
    assert result.magnitude == pytest.approx(expected)
    assert result.total == pytest.approx(expected)
    assert result.directional == pytest.approx(0, abs=1e-15)


def test_periodic_rotation_changes_direction_but_not_magnitude() -> None:
    angles = 2 * np.pi * np.arange(3) / 3
    spin = np.zeros((3, 3, 3, 3))
    spin[..., 0] = np.cos(angles)[:, None, None]
    spin[..., 1] = np.sin(angles)[:, None, None]
    result = compute_spin_gradient_diagnostics(
        spin_grid(spin, periodic=(True, True, True))
    )
    assert result.magnitude == pytest.approx(0, abs=1e-28)
    # Each of 27 forward edges has a 120-degree unit-vector difference.
    assert result.total == pytest.approx(81)
    assert result.directional == pytest.approx(81)


def test_zero_spin_is_excluded_and_nonperiodic_terminal_planes_do_not_wrap() -> None:
    spin = np.zeros((3, 3, 3, 3))
    spin[1, ..., 0] = 1
    grid = spin_grid(spin, periodic=(False, True, True))
    result = compute_spin_gradient_diagnostics(grid)
    # Only the nine occupied points contribute; their forward neighbors vanish.
    assert result.magnitude == pytest.approx(9)
    assert result.total == pytest.approx(9)
    assert result.directional == 0
    spin.fill(0)
    spin[2, ..., 0] = 1
    result = compute_spin_gradient_diagnostics(
        spin_grid(spin, periodic=(False, True, True))
    )
    assert (result.magnitude, result.total, result.directional) == (0, 0, 0)
