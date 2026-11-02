"""Validate the self-contained reference data and radial constraints."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

from py_chargemol.density.references import (
    SHELL_VOLUMES,
    core_reference_density,
    reference_densities,
    reference_density,
)
from py_chargemol.density.reshaping import monotonic_decay, tail_decay

_ORACLE = json.loads(
    (Path(__file__).parent / "data/density_reference.json").read_text()
)


def test_bundled_density_files_are_unmodified() -> None:
    directory = files("py_chargemol.density").joinpath("data")
    manifest = json.loads(directory.joinpath("SHA256SUMS.json").read_text())
    actual = {}
    for resource in directory.iterdir():
        if resource.name.endswith(".zip"):
            with resource.open("rb") as stream, ZipFile(stream) as archive:
                for name in archive.namelist():
                    actual[name] = hashlib.sha256(archive.read(name)).hexdigest()
    assert actual == manifest
    assert len(actual) == 1897


@pytest.mark.parametrize("case", _ORACLE["references"])
def test_reference_density_matches_fortran(case: dict[str, Any]) -> None:
    actual = reference_density(case["atomic_number"], case["charge"])
    np.testing.assert_allclose(actual, case["density"], rtol=1e-10, atol=2e-11)


@pytest.mark.parametrize("number", range(1, 110))
@pytest.mark.parametrize("charge", [-10.0, -0.4, 0.0, 0.3, 1.0])
def test_reference_normalization_and_decay(number: int, charge: float) -> None:
    actual = reference_density(number, charge)
    assert actual.shape == (100,)
    assert np.all(np.isfinite(actual))
    assert actual.min() >= -1e-12
    assert actual @ SHELL_VOLUMES == pytest.approx(number - charge, abs=1e-9)
    assert np.max(np.diff(actual)) < 1e-10


def test_fractional_reference_ions_interpolate_and_return_copies() -> None:
    neutral = reference_density(8)
    cation = reference_density(8, 1)
    actual = reference_density(8, 0.3)
    np.testing.assert_allclose(actual, 0.7 * neutral + 0.3 * cation)
    actual[:] = -1
    assert np.all(reference_density(8, 0.3) >= 0)
    np.testing.assert_allclose(
        reference_densities(np.array([8, 8]), np.array([0.0, 1.0])),
        [neutral, cation],
    )


def test_core_densities_are_raw_and_return_copies() -> None:
    actual = core_reference_density(8, 2)
    np.testing.assert_allclose(actual[:3], [307.56732, 34.90403, 7.5950193])
    actual[:] = 0
    assert core_reference_density(8, 2)[0] > 300
    np.testing.assert_array_equal(core_reference_density(1, 0), np.zeros(100))
    with pytest.raises(ValueError, match="no reference core density"):
        core_reference_density(8, 3)


@pytest.mark.parametrize("number", [0, 110])
def test_reference_number_validation(number: int) -> None:
    with pytest.raises(ValueError, match="atomic numbers must lie"):
        reference_density(number)


@pytest.mark.parametrize("charge", [np.nan, np.inf, -np.inf])
def test_reference_charge_validation(charge: float) -> None:
    with pytest.raises(ValueError, match="charge must be finite"):
        reference_density(1, charge)


@pytest.mark.parametrize("case", _ORACLE["reshaping"])
def test_radial_constraints_match_fortran(case: dict[str, Any]) -> None:
    density = np.array(case["density"])
    weights = np.array(case["weights"])
    if case["mode"] == 1:
        actual = monotonic_decay(density, weights)
    else:
        actual = tail_decay(
            density, weights, np.array(case["first"]), np.array(case["second"])
        )
    np.testing.assert_allclose(actual, case["expected"], rtol=1e-9, atol=1e-9)
    assert actual @ weights == pytest.approx(density @ weights, abs=1e-9)
    np.testing.assert_array_equal(density, case["density"])


def test_impossible_radial_constraint_reports_failure() -> None:
    with pytest.raises(ValueError, match="could not bracket"):
        tail_decay(np.ones(2), np.array([0.0, 1.0]), np.zeros(2), np.zeros(2))


def test_subtolerance_radial_tails_remain_finite() -> None:
    density = np.array(
        [
            2.5551550226868336e-12,
            4.550323233703344e-11,
            5.558677698437231e-11,
            6.101118551839302e-12,
        ]
    )
    exponents = np.array(
        [
            0.12867990424219689,
            0.8058866139918319,
            0.07652021054241964,
            0.2441784787477964,
        ]
    )
    result = tail_decay(density, np.ones(4), exponents, np.zeros(4))
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert abs(result.sum() - density.sum()) < 1e-10


def test_unattainable_absolute_population_tolerance_reports_failure() -> None:
    density = np.array(
        [140347114030734.38, 48390177297680.35, 293668492546082.25, 2183474105728412.5]
    )
    exponents = np.array(
        [
            0.1559071642819676,
            0.16671427745545564,
            0.9342648767267837,
            0.8168797001373653,
        ]
    )
    with pytest.raises(ValueError, match="population constraint did not converge"):
        tail_decay(density, np.ones(4), exponents, np.zeros(4))
