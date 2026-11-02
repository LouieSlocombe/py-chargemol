"""Tests for the QDR solver."""

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from py_chargemol.ddec6 import DDEC6Analysis, OverlapPairs
from py_chargemol.qdr import QDRResult, compute_qdr_charges

type FloatArray = npt.NDArray[np.float64]
type Loader = Callable[[str], tuple[DDEC6Analysis, OverlapPairs]]

EXAMPLES = ["formamide_VASP", "BN_nanotube", "ALPO5", "CO2_CCSD_augccpVTZ"]


def tensor(components: FloatArray) -> FloatArray:
    xx, xy, xz, yy, yz, zz = components
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def dense_reference(analysis: DDEC6Analysis, pairs: OverlapPairs) -> QDRResult:
    """QDR written as plainly as possible: per-atom loops and exact dense solves.

    The quadrupole system's matrix is I + G, where G is the Gram matrix of the bonds'
    weighted traceless direction tensors w (u u^T - I/3), whose Frobenius inner
    products are w_j w_k ((u_j . u_k)^2 - 1/3). The dipole system's is I + G with
    G_jk = w_j w_k u_j . u_k.
    """
    charges = analysis.net_charges.copy()
    dipoles = analysis.atomic_dipoles.copy()
    quadrupoles = analysis.atomic_quadrupoles.copy()
    vectors = (
        analysis.positions[pairs.atom2]
        - analysis.positions[pairs.atom1]
        + pairs.translations @ analysis.lattice_vectors
    )
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    weights = 2.0 * np.tanh(2.0 * pairs.populations)
    bonds_of = [np.flatnonzero(pairs.atom1 == atom) for atom in range(analysis.natoms)]

    for _ in range(2):
        y = np.zeros(len(pairs))
        for atom, bonds in enumerate(bonds_of):
            u, w = directions[bonds], weights[bonds]
            gram = np.outer(w, w) * ((u @ u.T) ** 2 - 1.0 / 3.0)
            rhs = -w * np.einsum("ja,ab,jb->j", u, tensor(quadrupoles[atom]), u)
            y[bonds] = np.linalg.solve(np.eye(len(bonds)) + gram, rhs)
            for j in bonds:
                yw = y[j] * weights[j]
                dipoles[atom] += yw / lengths[j] * directions[j]
                traceless = np.outer(directions[j], directions[j]) - np.eye(3) / 3
                quadrupoles[atom] += yw * traceless[np.triu_indices(3)]
        for k, (a, b) in enumerate(zip(pairs.atom1, pairs.atom2, strict=True)):
            moved = y[k] * weights[k] / lengths[k] ** 2
            charges[a] += moved
            charges[b] -= moved

        y = np.zeros(len(pairs))
        for atom, bonds in enumerate(bonds_of):
            wu = weights[bonds, None] * directions[bonds]
            rhs = -wu @ dipoles[atom]
            y[bonds] = np.linalg.solve(np.eye(len(bonds)) + wu @ wu.T, rhs)
        for k, (a, b) in enumerate(zip(pairs.atom1, pairs.atom2, strict=True)):
            moved = y[k] * weights[k] / lengths[k]
            charges[a] += moved
            charges[b] -= moved
            dipoles[a] += 0.5 * moved * lengths[k] * directions[k]
            dipoles[b] += 0.5 * moved * lengths[k] * directions[k]

    return QDRResult(charges, dipoles, quadrupoles, cg_iterations=0, residual_norm=0)


@pytest.mark.parametrize("example", EXAMPLES)
def test_matches_exact_dense_solves(load_example: Loader, example: str) -> None:
    """CG converges within its limit, so it agrees with direct solves."""
    analysis, pairs = load_example(example)

    result = compute_qdr_charges(analysis, pairs)
    expected = dense_reference(analysis, pairs)

    np.testing.assert_allclose(result.charges, expected.charges, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        result.atomic_dipoles, expected.atomic_dipoles, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        result.atomic_quadrupoles, expected.atomic_quadrupoles, rtol=0, atol=1e-10
    )
    assert 1 <= result.cg_iterations < 10
    assert result.residual_norm < 1e-11


@pytest.mark.parametrize("example", EXAMPLES)
def test_conserves_total_charge(load_example: Loader, example: str) -> None:
    analysis, pairs = load_example(example)

    result = compute_qdr_charges(analysis, pairs)

    assert result.charges.sum() == pytest.approx(analysis.net_charges.sum(), abs=1e-12)


def test_residual_quadrupoles_stay_traceless(load_example: Loader) -> None:
    analysis, pairs = load_example("ALPO5")

    quadrupoles = compute_qdr_charges(analysis, pairs).atomic_quadrupoles

    np.testing.assert_allclose(quadrupoles[:, [0, 3, 5]].sum(axis=1), 0, atol=1e-12)


def test_pair_order_does_not_matter(load_example: Loader) -> None:
    analysis, pairs = load_example("formamide_VASP")
    order = np.random.default_rng(0).permutation(len(pairs))
    shuffled = OverlapPairs(
        atom1=pairs.atom1[order],
        atom2=pairs.atom2[order],
        translations=pairs.translations[order],
        populations=pairs.populations[order],
    )

    result = compute_qdr_charges(analysis, shuffled)

    expected = compute_qdr_charges(analysis, pairs)
    np.testing.assert_allclose(result.charges, expected.charges, atol=1e-14)
    np.testing.assert_allclose(
        result.atomic_dipoles, expected.atomic_dipoles, atol=1e-14
    )


def test_inputs_are_not_modified(load_example: Loader) -> None:
    analysis, pairs = load_example("formamide_VASP")
    before = analysis.net_charges.copy(), analysis.atomic_dipoles.copy()

    compute_qdr_charges(analysis, pairs)

    np.testing.assert_array_equal(analysis.net_charges, before[0])
    np.testing.assert_array_equal(analysis.atomic_dipoles, before[1])


def test_no_pairs_leaves_the_ddec6_multipoles(load_example: Loader) -> None:
    analysis, pairs = load_example("formamide_VASP")
    empty = OverlapPairs(
        atom1=pairs.atom1[:0],
        atom2=pairs.atom2[:0],
        translations=pairs.translations[:0],
        populations=pairs.populations[:0],
    )

    result = compute_qdr_charges(analysis, empty)

    np.testing.assert_array_equal(result.charges, analysis.net_charges)
    np.testing.assert_array_equal(result.atomic_dipoles, analysis.atomic_dipoles)
    np.testing.assert_array_equal(
        result.atomic_quadrupoles, analysis.atomic_quadrupoles
    )
    assert result.cg_iterations == 0
    assert result.residual_norm == 0.0


def test_iteration_limit_is_respected(load_example: Loader) -> None:
    analysis, pairs = load_example("ALPO5")

    result = compute_qdr_charges(analysis, pairs, max_cg_iterations=1)

    assert result.cg_iterations == 1
    assert result.residual_norm > 1e-6


def test_loose_tolerance_stops_early(load_example: Loader) -> None:
    analysis, pairs = load_example("ALPO5")

    loose = compute_qdr_charges(analysis, pairs, tolerance=1e-3)

    assert loose.cg_iterations < compute_qdr_charges(analysis, pairs).cg_iterations
    assert loose.residual_norm > 1e-12


@pytest.mark.parametrize("field", ["atom1", "atom2"])
@pytest.mark.parametrize("index", [-1, 6])
def test_pairs_must_name_atoms_in_the_analysis(
    load_example: Loader, field: str, index: int
) -> None:
    analysis, pairs = load_example("formamide_VASP")
    atoms = getattr(pairs, field).copy()
    atoms[3] = index

    with pytest.raises(ValueError, match=f"{field} indices must lie in 0-5"):
        compute_qdr_charges(analysis, dataclasses.replace(pairs, **{field: atoms}))


def test_bonds_must_have_length(load_example: Loader) -> None:
    analysis, pairs = load_example("formamide_VASP")
    atom2 = pairs.atom2.copy()
    atom2[0] = pairs.atom1[0]
    translations = pairs.translations.copy()
    translations[0] = 0

    with pytest.raises(ValueError, match="joins an atom to itself"):
        compute_qdr_charges(
            analysis,
            dataclasses.replace(pairs, atom2=atom2, translations=translations),
        )
