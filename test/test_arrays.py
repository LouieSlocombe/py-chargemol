"""Tests for calculating QDR charges from in-memory arrays and sequences."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.qdr.arrays import compute_qdr_from_arrays
from py_chargemol.qdr.solver import QDRResult, compute_qdr_charges

type Inputs = dict[str, npt.ArrayLike]
type Loader = Callable[[str], tuple[DDEC6Analysis, OverlapPairs]]


@pytest.fixture
def inputs() -> Inputs:
    return {
        "atomic_numbers": [1, 1],
        "positions": [[0, 0, 0], [2, 0, 0]],
        "net_charges": [0, 0],
        "atomic_dipoles": [[1, 0, 0], [0, 0, 0]],
        "atomic_quadrupoles": [[1, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0]],
        "atom1": [0, 1],
        "atom2": [1, 0],
        "populations": [0.5, 0.5],
    }


def calculate(inputs: Inputs) -> QDRResult:
    return compute_qdr_from_arrays(
        atomic_numbers=inputs["atomic_numbers"],
        positions=inputs["positions"],
        net_charges=inputs["net_charges"],
        atomic_dipoles=inputs["atomic_dipoles"],
        atomic_quadrupoles=inputs["atomic_quadrupoles"],
        atom1=inputs["atom1"],
        atom2=inputs["atom2"],
        populations=inputs["populations"],
        translations=inputs.get("translations"),
        lattice_vectors=inputs.get("lattice_vectors"),
    )


@pytest.mark.parametrize(
    "example", ["formamide_VASP", "BN_nanotube", "ALPO5", "CO_molecule"]
)
def test_matches_calculation_from_analysis(load_example: Loader, example: str) -> None:
    analysis, pairs = load_example(example)
    result = compute_qdr_from_arrays(
        atomic_numbers=analysis.atomic_numbers,
        positions=analysis.positions,
        net_charges=analysis.net_charges,
        atomic_dipoles=analysis.atomic_dipoles,
        atomic_quadrupoles=analysis.atomic_quadrupoles,
        atom1=pairs.atom1,
        atom2=pairs.atom2,
        populations=pairs.populations,
        translations=pairs.translations,
        lattice_vectors=analysis.lattice_vectors,
        max_cg_iterations=1,
        tolerance=1e-9,
    )
    expected = compute_qdr_charges(analysis, pairs, max_cg_iterations=1, tolerance=1e-9)

    np.testing.assert_array_equal(result.charges, expected.charges)
    np.testing.assert_array_equal(result.atomic_dipoles, expected.atomic_dipoles)
    np.testing.assert_array_equal(
        result.atomic_quadrupoles, expected.atomic_quadrupoles
    )
    assert result.cg_iterations == expected.cg_iterations
    assert result.residual_norm == expected.residual_norm
    if analysis.is_periodic:
        assert result.moments is None
    else:
        assert result.moments is not None
        assert expected.moments is not None
        np.testing.assert_array_equal(
            result.moments.with_quadrupoles.quadrupole,
            expected.moments.with_quadrupoles.quadrupole,
        )


def test_accepts_lists_and_integer_multipoles_without_files(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = calculate(inputs)

    assert result.charges.dtype == np.float64
    assert np.any(result.charges != 0)
    assert result.charges.sum() == pytest.approx(0)
    assert result.moments is not None
    assert list(tmp_path.iterdir()) == []


def test_inputs_and_results_do_not_share_mutable_arrays(inputs: Inputs) -> None:
    arrays = {name: np.asarray(value) for name, value in inputs.items()}
    before = {name: value.copy() for name, value in arrays.items()}

    result = calculate(dict(arrays))
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, before[name])
    result.charges[:] = 99
    result.atomic_dipoles[:] = 99
    result.atomic_quadrupoles[:] = 99
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, before[name])


def test_no_pairs_returns_original_multipoles(inputs: Inputs) -> None:
    inputs.update(atom1=[], atom2=[], populations=[])

    result = calculate(inputs)

    np.testing.assert_array_equal(result.charges, inputs["net_charges"])
    np.testing.assert_array_equal(result.atomic_dipoles, inputs["atomic_dipoles"])
    np.testing.assert_array_equal(
        result.atomic_quadrupoles, inputs["atomic_quadrupoles"]
    )
    assert result.cg_iterations == 0
    assert result.residual_norm == 0


@pytest.mark.parametrize("dtype", [np.int8, np.uint64, np.float16, np.float64])
def test_accepts_exact_integer_values(inputs: Inputs, dtype: npt.DTypeLike) -> None:
    inputs["atomic_numbers"] = np.array([1, 1], dtype=dtype)
    inputs["atom1"] = np.array([0, 1], dtype=dtype)
    inputs["atom2"] = np.array([1, 0], dtype=dtype)
    inputs["translations"] = np.zeros((2, 3), dtype=dtype)

    assert np.all(np.isfinite(calculate(inputs).charges))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("atomic_numbers", [[1, 1]]),
        ("positions", [[0, 0, 0]]),
        ("net_charges", [[0, 0]]),
        ("atomic_dipoles", [[0, 0], [0, 0]]),
        ("atomic_quadrupoles", np.zeros((2, 5))),
        ("atom1", 0),
        ("atom2", [1]),
        ("populations", [0.5]),
        ("translations", [[0, 0, 0]]),
        ("lattice_vectors", np.zeros((2, 3))),
    ],
)
def test_rejects_inconsistent_shapes(
    inputs: Inputs, field: str, value: npt.ArrayLike
) -> None:
    inputs[field] = value

    with pytest.raises(ValueError, match=rf"{field} must have shape"):
        calculate(inputs)


@pytest.mark.parametrize("numbers", [[], [0, 1], [1, 110]])
def test_rejects_missing_or_unknown_atoms(
    inputs: Inputs, numbers: npt.ArrayLike
) -> None:
    inputs["atomic_numbers"] = numbers

    with pytest.raises(ValueError, match="atomic_numbers must"):
        calculate(inputs)


@pytest.mark.parametrize("field", ["atomic_numbers", "atom1", "atom2", "translations"])
@pytest.mark.parametrize(
    "invalid", [0.5, np.nan, np.inf, -np.inf, 2**63, -(2**64), "1", True, 1j]
)
def test_rejects_invalid_integer_values(
    inputs: Inputs, field: str, invalid: complex | str
) -> None:
    inputs[field] = (
        [[invalid, invalid, invalid], [invalid, invalid, invalid]]
        if field == "translations"
        else [invalid, invalid]
    )

    with pytest.raises(ValueError, match=rf"{field} must contain integers"):
        calculate(inputs)


@pytest.mark.parametrize("invalid", [float(2**63), -float(2**64)])
def test_rejects_floating_point_integer_overflow(
    inputs: Inputs, invalid: float
) -> None:
    inputs["atom1"] = [invalid, invalid]

    with pytest.raises(ValueError, match="atom1 must contain integers"):
        calculate(inputs)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_preserves_largest_int64_integer(inputs: Inputs, dtype: npt.DTypeLike) -> None:
    inputs["atom1"] = np.array([0, np.iinfo(np.int64).max], dtype=dtype)

    with pytest.raises(ValueError, match="atom1 indices must lie"):
        calculate(inputs)


@pytest.mark.parametrize(
    "field",
    [
        "positions",
        "net_charges",
        "atomic_dipoles",
        "atomic_quadrupoles",
        "populations",
        "lattice_vectors",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_float_values(
    inputs: Inputs, field: str, invalid: float
) -> None:
    values = np.asarray(inputs.get(field, np.zeros((3, 3))), dtype=float).copy()
    values.flat[0] = invalid
    inputs[field] = values

    with pytest.raises(ValueError, match=rf"{field} must contain only finite"):
        calculate(inputs)


@pytest.mark.parametrize("invalid", ["bad", 1j, 10**1000])
def test_rejects_values_not_convertible_to_real_numbers(
    inputs: Inputs, invalid: str | complex
) -> None:
    inputs["net_charges"] = [invalid, invalid]

    with pytest.raises(ValueError, match="net_charges must contain real numbers"):
        calculate(inputs)


def test_rejects_ragged_sequences(inputs: Inputs) -> None:
    inputs["positions"] = [[0, 0], [0, 0, 0]]

    with pytest.raises(ValueError, match="positions must be a numeric array"):
        calculate(inputs)


@pytest.mark.parametrize("field", ["atom1", "atom2"])
@pytest.mark.parametrize("invalid", [-1, 2])
def test_rejects_out_of_range_atom_indices(
    inputs: Inputs, field: str, invalid: int
) -> None:
    inputs[field] = [invalid, 0]

    with pytest.raises(ValueError, match=rf"{field} indices must lie"):
        calculate(inputs)


def test_rejects_zero_length_bonds(inputs: Inputs) -> None:
    inputs["atom2"] = inputs["atom1"]

    with pytest.raises(ValueError, match="joins an atom to itself"):
        calculate(inputs)
