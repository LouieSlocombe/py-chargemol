# QDR point charges

QDR uses DDEC6 atomic dipoles, quadrupoles, and directed bond overlap
populations to redistribute charge between bonded atoms. The resulting charges
serve as atom-centred point charges for classical force fields. Use the
original DDEC6 charges for chemical interpretation of electron transfer.

## Start with Chargemol output

Run this example from the repository root with py-chargemol installed.
The CO fixture has DDEC6 charges of +0.132034 e on carbon and −0.132034 e on
oxygen. Inspect the conversion to QDR in memory with `calculate_qdr`.

```python
from py_chargemol import calculate_qdr

result = calculate_qdr("test/worked_examples/CO_molecule")
print(result.charges.round(6))  # [-0.003964  0.003964]
print(result.atomic_dipoles)    # residual atomic dipoles, in e·bohr
```

To produce the XYZ file and run log with the CLI, give each an output path
outside the reference fixture. Run these commands from the same repository root.

```bash
mkdir -p results/co
py-chargemol qdr test/worked_examples/CO_molecule \
  --output results/co/QDR-DDEC6_point_charges.xyz \
  --log results/co/QDR_charges_computation_logfile.txt
```

`py-chargemol qdr DIR` and `calculate_qdr(DIR)` look for the first existing
file in each row below. Both use the working directory when `DIR` is omitted.

| Input | Filenames in discovery order |
| --- | --- |
| Charges, multipoles, and geometry | `DDEC6_net_atomic_charges.xyz`, `DDEC6_even_tempered_net_atomic_charges.xyz` |
| Directed overlaps | `overlap_populations_and_contact_exchanges.txt`, `overlap_populations.txt`, `overlap_populations.xyz` |

With the CLI, output and log default to `DIR`. Use `--charges` and `--overlaps`
to override input discovery. All explicit CLI paths, and Python's
`charges_file`, `overlaps_file`, and `output_file`, resolve from the current
working directory. Create parent directories for explicit output paths first.
Existing output and log files with the selected names are overwritten.

In Python, `run_qdr(DIR)` writes the XYZ file and reports progress through
the `py_chargemol.qdr.run` logger. For a one-atom input, `run_qdr` returns
`None` and the CLI writes only the log. Use `calculate_qdr` to obtain arrays
in that case. An isolated atom keeps its input charge and multipoles and
needs no overlap file. A periodic one-atom cell requires an overlap file
because an atom can bond to its periodic images.

## Compute from arrays

Use this entry point if another program already supplies DDEC6 results in
memory. The following two-atom input is synthetic.

```python
from py_chargemol import compute_qdr_from_arrays

result = compute_qdr_from_arrays(
    atomic_numbers=[1, 1],
    positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
    net_charges=[0.1, -0.1],
    atomic_dipoles=[[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],
    atomic_quadrupoles=[[0.0] * 6, [0.0] * 6],
    atom1=[0, 1],
    atom2=[1, 0],
    populations=[0.5, 0.5],
)
print(result.charges)
```

All arguments are keyword-only. For `n` atoms and `m` directed overlap pairs,
supply Python sequences or NumPy arrays with the following shapes.

| Argument | Shape | Meaning and units |
| --- | --- | --- |
| `atomic_numbers` | `(n,)` | Atomic numbers 1–109 |
| `positions` | `(n, 3)` | Cartesian coordinates in bohr |
| `net_charges` | `(n,)` | DDEC6 net atomic charges in e |
| `atomic_dipoles` | `(n, 3)` | DDEC6 atomic dipoles in e·bohr |
| `atomic_quadrupoles` | `(n, 6)` | Traceless DDEC6 quadrupoles in e·bohr², ordered `xx, xy, xz, yy, yz, zz` |
| `atom1`, `atom2` | `(m,)` | Zero-based owner and partner indices |
| `populations` | `(m,)` | Bond overlap populations in e |
| `translations` | `(m, 3)` | Integer lattice translations of the partner atom, default all zeros |
| `lattice_vectors` | `(3, 3)` | Cell vectors as rows, in bohr, default all zeros |

Supply one row from each end of a bond, with the same overlap population.
For an owner `i`, partner `j`, and translation `t`, the displacement is
`positions[j] + t @ lattice_vectors - positions[i]`. The reverse row has
owner `j`, partner `i`, and translation `-t`. A zero lattice-vector row marks
a nonperiodic direction. The overlap table on disk numbers atoms starting at 1.

When working with files in Python, use the readers to convert positions from
ångströms to bohr, unpack the quadrupole columns, and convert the pair indices.
The parsed objects can go straight to the solver.

```python
from pathlib import Path

from py_chargemol import (
    compute_qdr_charges,
    read_ddec6_net_atomic_charges,
    read_overlap_populations,
)

co = Path("test/worked_examples/CO_molecule")
analysis = read_ddec6_net_atomic_charges(co / "DDEC6_even_tempered_net_atomic_charges.xyz")
pairs = read_overlap_populations(co / "overlap_populations_and_contact_exchanges.txt")
result = compute_qdr_charges(analysis, pairs)
```

## Read the result and check the solver

Every computation entry point returns a `QDRResult`. Its `charges`,
`atomic_dipoles`, and `atomic_quadrupoles` are the resorbed point charges and
the residual atomic moments, in the same units and shapes as the corresponding
input arrays. The written XYZ starts with coordinates in ångströms and QDR
charges in e, followed by the residual multipoles and electron-cloud parameters.

For a molecule, `result.moments` gives total moments about the centre of mass
at three levels. Compare `moments.charges.dipole_magnitude` with
`moments.with_dipoles.dipole_magnitude` to see the contribution of the residual
dipoles. Use `moments.with_quadrupoles.quadrupole_tensor` for the full molecular
quadrupole. Each level exposes the dipole vector, its magnitude, the five
quadrupole components, the traceless tensor, and its eigenvalues. For periodic
systems, `result.moments` is `None`.

The default solver (`solver="cg"`) follows the original QDR program. With
`calculate_qdr`, `compute_qdr_charges`, or `compute_qdr_from_arrays`, set
`max_cg_iterations` (default 10) and `tolerance` (default `1e-12`) to control
each atomic solve. `result.cg_iterations` is the largest iteration count for
any atom in any step. The tolerance applies per atom, while
`result.residual_norm` reports the largest final L2 norm across steps with
all atoms' residuals combined.

`solver="direct"` solves a reduced 5×5 or 3×3 system per atom and reports zero
CG iterations. It agrees with converged CG to rounding in the regression
tests. Select it with `--solver direct` for the QDR CLI, `--qdr-solver direct`
for the density CLI, or `qdr_solver="direct"` in `write_density_results`.
