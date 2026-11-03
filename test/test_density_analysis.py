"""End-to-end density calculations with no external executable or data path."""

import json
from pathlib import Path

import numpy as np
import pytest

from py_chargemol import (
    DensityGrid,
    compute_ddec,
    compute_ddec3,
    compute_ddec6,
    compute_qdr_from_density,
)
from py_chargemol.__main__ import main
from py_chargemol.density.output import write_density_results


@pytest.fixture
def hydrogen_density() -> DensityGrid:
    shape = (32, 32, 32)
    origin = np.full(3, -0.35 * 31 / 2)
    steps = np.eye(3) * 0.35
    coordinates = np.indices(shape).reshape(3, -1).T @ steps + origin
    positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    density = np.sum(
        [np.exp(-np.sum((coordinates - p) ** 2, axis=1)) for p in positions], axis=0
    )
    density *= 2 / (density.sum() * np.linalg.det(steps))
    return DensityGrid(
        atomic_numbers=np.array([1, 1]),
        positions=positions,
        density=density.reshape(shape),
        origin=origin,
        steps=steps,
    )


def test_density_to_qdr_conserves_electrons_and_molecular_symmetry(
    hydrogen_density: DensityGrid,
) -> None:
    result = compute_ddec6(hydrogen_density)
    np.testing.assert_allclose(result.charges, 0, atol=1e-7)
    np.testing.assert_allclose(result.cm5_charges, 0, atol=1e-7)
    np.testing.assert_allclose(result.analysis.atomic_dipoles.sum(axis=0), 0, atol=1e-7)
    assert np.all(result.radial_moments > 0)
    assert result.moments is not None
    assert (
        result.atomic_densities.shape == result.radial_shell_weights.shape == (2, 100)
    )
    assert len(result.overlaps) == 2
    qdr = result.compute_qdr()
    np.testing.assert_allclose(qdr.charges.sum(), 0, atol=1e-12)
    assert qdr.moments is not None
    np.testing.assert_allclose(
        compute_qdr_from_density(hydrogen_density).charges, qdr.charges
    )


def test_density_to_qdr_can_solve_directly(
    hydrogen_density: DensityGrid, tmp_path: Path
) -> None:
    result = compute_ddec6(hydrogen_density)
    direct = compute_qdr_from_density(hydrogen_density, solver="direct")
    assert direct.solver == "direct"
    np.testing.assert_allclose(
        direct.charges, result.compute_qdr().charges, rtol=0, atol=1e-12
    )
    paths = write_density_results(result, tmp_path, qdr=True, qdr_solver="direct")
    assert "solved directly, in closed form" in paths["qdr"].read_text()


def test_ddec3_analysis_is_available_and_does_not_masquerade_as_qdr(
    hydrogen_density: DensityGrid,
) -> None:
    result = compute_ddec3(hydrogen_density)
    np.testing.assert_allclose(result.charges, 0, atol=1e-7)
    assert result.method == "ddec3"
    with pytest.raises(ValueError, match="requires a DDEC6"):
        result.compute_qdr()


def test_density_electron_count_and_method_are_checked(
    hydrogen_density: DensityGrid,
) -> None:
    with pytest.raises(ValueError, match="method must be"):
        compute_ddec(hydrogen_density, method="invalid")
    hydrogen_density.net_charge = 1
    with pytest.raises(ValueError, match="density integrates"):
        compute_ddec6(hydrogen_density)


@pytest.mark.parametrize("vector", [False, True])
def test_spin_density_flows_through_to_spin_populations_and_bonds(
    hydrogen_density: DensityGrid,
    vector: bool,
) -> None:
    scalar = hydrogen_density.density * 0.3
    hydrogen_density.spin_density = (
        scalar[..., None] * np.array([0.0, 0.6, 0.8]) if vector else scalar
    )
    result = compute_ddec6(hydrogen_density)
    assert result.spin is not None
    assert (result.spin_gradients is not None) == vector
    expected = np.array([0.0, 0.36, 0.48]) if vector else 0.6
    np.testing.assert_allclose(result.spin.populations.sum(axis=0), expected, atol=1e-5)
    assert np.all(np.isfinite(result.bonds.bond_orders))


@pytest.mark.parametrize("method", ["ddec6", "ddec3"])
def test_charge_only_analysis_skips_bonds_and_associated_densities(
    hydrogen_density: DensityGrid,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_bond_analysis(**_kwargs: object) -> None:
        pytest.fail("compute_bos=False must not perform bond analysis")

    monkeypatch.setattr(
        "py_chargemol.density.analysis.compute_bond_orders", unexpected_bond_analysis
    )
    compute = compute_ddec6 if method == "ddec6" else compute_ddec3
    result = compute(hydrogen_density, compute_bos=False)
    np.testing.assert_allclose(result.charges, 0, atol=1e-7)
    assert result.moments is not None
    assert not result.has_bond_orders
    assert result.bond_result is None
    for field in (
        "bonds",
        "overlaps",
        "atomic_densities",
        "radial_moments",
        "radial_shell_weights",
    ):
        with pytest.raises(ValueError, match="compute_bos=True"):
            getattr(result, field)
    with pytest.raises(ValueError, match=r"compute_bos=True|requires a DDEC6"):
        result.compute_qdr()


def test_unresolved_bond_density_population_is_reported() -> None:
    density = np.zeros((7, 7, 7))
    density[3, 0, 0] = 0.95
    result = compute_ddec6(
        DensityGrid(
            atomic_numbers=np.array([1]),
            positions=np.zeros((1, 3)),
            density=density,
            origin=np.zeros(3),
            steps=np.eye(3),
            occupancy_correction=np.array([0.05]),
        )
    )
    np.testing.assert_allclose(result.charges, 0)
    assert result.bonds.density_correction_iterations == 40
    np.testing.assert_allclose(result.bonds.population_residual, 0.05)
    assert result.bonds.corrected_density.sum() == pytest.approx(0.95)


def test_analytic_atomic_multipole_corrections_are_retained(
    hydrogen_density: DensityGrid,
) -> None:
    baseline = compute_ddec6(hydrogen_density)
    correction = np.zeros((2, 11))
    correction[0, 2] = 0.2
    correction[1, 8] = 0.4
    hydrogen_density.atomic_corrections = correction
    result = compute_ddec6(hydrogen_density)
    assert result.analysis.atomic_dipoles[0, 0] == pytest.approx(
        baseline.analysis.atomic_dipoles[0, 0] + 0.2
    )
    assert result.analysis.atomic_quadrupoles[1, 1] == pytest.approx(
        baseline.analysis.atomic_quadrupoles[1, 1] + 0.4
    )


def test_single_atom_density_needs_no_overlap_or_tail_fit(tmp_path: Path) -> None:
    grid = DensityGrid(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        density=np.ones((1, 1, 1)),
        origin=np.zeros(3),
        steps=np.eye(3),
        wavefunction_radial_moments=np.arange(6, dtype=float),
    )
    result = compute_ddec6(grid)
    np.testing.assert_allclose(result.charges, 0)
    np.testing.assert_allclose(result.analysis.atomic_dipoles, 0)
    np.testing.assert_allclose(result.analysis.cloud_a, 0)
    assert len(result.overlaps) == 0
    np.testing.assert_allclose(result.compute_qdr().charges, 0)
    np.testing.assert_array_equal(result.wavefunction_radial_moments, np.arange(6))
    assert result.wavefunction_radial_moments is not grid.wavefunction_radial_moments
    paths = write_density_results(result, tmp_path)
    assert "Analytic single-centre wavefunction radial moments" in (
        paths["charges"].read_text()
    )


def test_constant_sampled_tail_has_zero_slope_and_fit_quality() -> None:
    grid = DensityGrid(
        atomic_numbers=np.array([8]),
        positions=np.zeros((1, 3)),
        density=np.ones((2, 2, 2)),
        origin=np.full(3, -3.0),
        steps=np.eye(3),
    )
    result = compute_ddec6(grid)
    np.testing.assert_allclose(result.analysis.cloud_b, 0)
    np.testing.assert_allclose(result.cloud_r_squared, 0)


def test_missing_core_is_reconstructed_from_bundled_reference_tables(
    hydrogen_density: DensityGrid,
) -> None:
    grid = DensityGrid(
        atomic_numbers=np.array([3, 1]),
        positions=hydrogen_density.positions,
        density=hydrogen_density.density,
        origin=hydrogen_density.origin,
        steps=hydrogen_density.steps,
        core_electrons=np.array([2, 0]),
    )
    result = compute_ddec6(grid)
    np.testing.assert_allclose(result.charges.sum(), 0, atol=1e-12)
    assert np.all(np.isfinite(result.analysis.atomic_dipoles))


@pytest.mark.parametrize("method", ["ddec6", "ddec3"])
def test_analyze_cli_runs_without_chargemol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    method: str,
) -> None:
    source = tmp_path / "total_density.cube"
    source.write_text(
        "single atom\ntotal density\n1 0 0 0\n1 1 0 0\n1 0 1 0\n1 0 0 1\n1 1 0 0 0\n1\n"
    )
    output = tmp_path / "results"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--method",
                method,
                "--periodic",
                "0",
                "0",
                "0",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert f"Computed {method.upper()} for 1 atoms" in capsys.readouterr().out
    assert (output / f"{method.upper()}_net_atomic_charges.xyz").is_file()
    assert (output / "QDR-DDEC6_point_charges.xyz").exists() == (method == "ddec6")


def test_analyze_cli_selects_the_qdr_solver(tmp_path: Path) -> None:
    source = tmp_path / "total_density.cube"
    source.write_text(
        "single atom\ntotal density\n1 0 0 0\n1 1 0 0\n1 0 1 0\n1 0 0 1\n1 1 0 0 0\n1\n"
    )
    output = tmp_path / "results"
    arguments = ["analyze", str(source), "--output-dir", str(output)]
    assert main([*arguments, "--qdr-solver", "direct"]) == 0
    text = (output / "QDR-DDEC6_point_charges.xyz").read_text()
    assert "solved directly, in closed form" in text
    assert "CG iterations" not in text


def test_analyze_cli_reports_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(tmp_path / "missing")]) == 1
    assert "py-chargemol analyze: error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source_kind", "options", "method", "densities", "bonds", "qdr", "control_bonds"),
    [
        ("directory", [], "DDEC3", True, True, False, True),
        (
            "control",
            ["--method", "ddec6", "--no-qdr"],
            "DDEC6",
            True,
            True,
            False,
            True,
        ),
        (
            "file",
            ["--no-atomic-densities", "--no-bonds"],
            "DDEC3",
            False,
            False,
            False,
            True,
        ),
        (
            "directory",
            ["--method", "ddec6", "--atomic-densities", "--bonds"],
            "DDEC6",
            True,
            True,
            True,
            False,
        ),
        ("directory", ["--method", "ddec6"], "DDEC6", False, False, False, False),
    ],
)
def test_analyze_cli_respects_job_controls_and_explicit_overrides(  # noqa: PLR0913, PLR0917
    tmp_path: Path,
    source_kind: str,
    options: list[str],
    method: str,
    densities: bool,
    bonds: bool,
    qdr: bool,
    control_bonds: bool,
) -> None:
    source = tmp_path / "total_density.cube"
    source.write_text(
        "single atom\ntotal density\n1 0 0 0\n1 1 0 0\n1 0 1 0\n1 0 0 1\n1 1 0 0 0\n1\n"
    )
    control = tmp_path / "job_control.txt"
    control.write_text(
        "<charge type> DDEC3 </charge type>\n"
        "<print atomic densities> .true. </print atomic densities>\n"
        f"<compute bos> {str(control_bonds).lower()} </compute bos>\n"
    )
    selected = {"directory": tmp_path, "control": control, "file": source}[source_kind]
    output = tmp_path / "results"
    assert main(["analyze", str(selected), "--output-dir", str(output), *options]) == 0
    assert (output / f"{method}_net_atomic_charges.xyz").is_file()
    assert (output / "spherically_averaged_atomic_densities.txt").exists() == densities
    assert (output / f"{method}_bond_orders.xyz").exists() == bonds
    assert (output / "QDR-DDEC6_point_charges.xyz").exists() == qdr


def test_nacl_reproduces_the_original_chargemol_calculation() -> None:
    directory = Path(__file__).parent / "density_examples"
    metadata = json.loads((directory / "nacl.json").read_text())
    fields = {}
    for name in ("density", "core_density", "pseudodensity"):
        with np.load(directory / f"nacl_{name}.npz", allow_pickle=False) as archive:
            fields[name] = archive["values"]
    grid = DensityGrid(
        atomic_numbers=np.array(metadata["atomic_numbers"]),
        positions=np.array(metadata["positions"]),
        origin=np.array(metadata["origin"]),
        steps=np.array(metadata["steps"]),
        core_electrons=np.array(metadata["core_electrons"]),
        periodic=(True, True, True),
        **fields,
    )
    result = compute_ddec6(grid)
    reference = metadata["DDEC6"]
    assert result.moments is None
    for field in (
        "net_charges",
        "atomic_dipoles",
        "atomic_quadrupoles",
        "cloud_a",
        "cloud_b",
    ):
        np.testing.assert_allclose(
            getattr(result.analysis, field), reference[field], atol=1e-6, rtol=0
        )
    expected = reference["overlaps"]
    reference_pairs = {
        (first, second, *shift): value
        for first, second, shift, value in zip(
            expected["atom1"],
            expected["atom2"],
            expected["translations"],
            expected["populations"],
            strict=True,
        )
    }
    assert len(result.overlaps) == len(reference_pairs)
    for first, second, shift, value in zip(
        result.overlaps.atom1,
        result.overlaps.atom2,
        result.overlaps.translations,
        result.overlaps.populations,
        strict=True,
    ):
        assert value == pytest.approx(reference_pairs[first, second, *shift], abs=1e-8)
