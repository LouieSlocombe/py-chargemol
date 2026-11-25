# Analyse an electron density

Start with an electron density or wavefunction from an electronic-structure
calculation. `read_density` loads it into a `DensityGrid`. `compute_ddec6`
partitions the electrons among the atoms and returns a `DDECResult`. Computed
arrays are available in memory before you choose which files to write.

## Run the triplet ozone example

From the repository root, load the Gaussian wavefunction and check the atomic
charges and spin populations against the printed Chargemol reference.

```python
from py_chargemol import compute_ddec6, read_density, write_density_results

grid = read_density("test/density_examples/ozone_triplet_PW91.wfx")
result = compute_ddec6(grid)
print(result.charges.round(6))          # [ 0.21533  -0.107665 -0.107665]
print(result.spin.populations.sum())    # approximately 2 unpaired electrons
paths = write_density_results(result, "results/ozone", qdr=True)
print(paths["qdr"])                     # results/ozone/QDR-DDEC6_point_charges.xyz
```

Use `result.compute_qdr().charges` to get QDR point charges in memory from
this partition. If QDR is the only quantity you need,
`compute_qdr_from_density(grid)` performs both calculations in one call.
For DDEC3, use `compute_ddec3(grid)` or `py-chargemol analyze FILE --method ddec3`.
QDR requires DDEC6 and directed overlap populations.

## Choose the input

`read_density(source)` and `py-chargemol analyze SOURCE` take an explicit
file, a calculation directory, or its `job_control.txt`. For directory
discovery, the reader tries the named `input filename` from job control, then
`AECCAR2`, `valence_density.cube`, `total_density.cube`, then a single
remaining cube, XSF, or WFX candidate. Spin cubes are companion files. If the
last step finds several candidates, pass a file path. Use `format="cube"`
or `--format cube` for a cube with an unrecognised extension.

| Source | Density and geometry | Core electrons, charge, and periodicity |
| --- | --- | --- |
| VASP 4/5 formats | `AECCAR2` for reconstructed valence, `CHGCAR` or `CHG` for the pseudo-valence grid and any spin blocks. Optional `AECCAR0` supplies the core grid. | `POTCAR` supplies species and valence counts. A VASP 5 element-symbol header plus per-atom core counts can replace it. Defaults to three periodic axes and infers net charge from the pseudo-valence integral. |
| Gaussian cube | A scalar total or valence density. Signed voxel counts set the coordinate units (positive for bohr, negative for ångströms). Density defaults to electrons/bohr³ independently of the coordinate units. | Defaults to neutral and nonperiodic. Total density includes all electrons and uses zero core counts. Valence density needs the pseudopotential core counts. |
| XSF | One scalar valence grid, two `rho:spin_1/2` grids for collinear spin, or four `rho:spin_1/2/3/4` grids for noncollinear spin. Coordinates are in ångströms, density defaults to electrons/bohr³. | Supply core counts. Charge defaults to zero. `CRYSTAL`, `SLAB`, and `POLYMER` indicate three, two, and one periodic axes respectively. The reader removes duplicate terminal grid planes. |
| Gaussian WFX | Primitive coefficients and occupations define the density. The reader handles tight Gaussian products analytically and supports effective-core-density functions. | The wavefunction supplies charge, core counts, and periodicity. Explicit core counts and periodicity must agree with the WFX data. |

For a valence cube, `read_density` infers `density_kind="valence"` when the
filename starts with `valence`. For other filenames set the kind explicitly.
A neighbouring `total_density.cube` on the same grid supplies the core density
by subtraction. For collinear spin, add `spin_density.cube`. For noncollinear
spin, supply all three files `spin_density_x.cube`, `spin_density_y.cube`, and
`spin_density_z.cube` with matching atoms, origin, voxel vectors, and shape.

Use a format-specific reader to choose units or other format options directly.
For example, the call below describes a periodic water cell with atoms in
O, H, H order and an oxygen pseudopotential that replaces two electrons.

```python
from py_chargemol import read_cube

grid = read_cube(
    "valence_density.cube",
    density_kind="valence",
    core_electrons=[2, 0, 0],
    periodic=(True, True, True),
    density_unit="bohr^-3",
)
```

`read_cube` accepts `length_unit`, `density_unit`, and an explicit
`total_density_file`. Use `read_density` for automatic companion-file
discovery. `read_xsf` also accepts `density_unit="angstrom^-3"`.
`read_wfx` defaults to a grid spacing of 0.14 bohr and interpolates diffuse
Gaussian products from coarser grids. Setting `interpolate_diffuse=False`
evaluates those products on the final grid. This costs more time and changes
the ozone example's charges by about 2 × 10⁻⁴ e.

### Use Chargemol job controls

Place `job_control.txt` alongside the input or pass it as the source. For
the water cube above, the equivalent settings include these tags.

```text
<input filename>valence_density.cube</input filename>
<periodicity along A, B, and C vectors>true true true</periodicity along A, B, and C vectors>
<number of core electrons>
8 2
1 0
</number of core electrons>
<net charge>0</net charge>
```

The core table gives **atomic number and core count per element**. The Python
`core_electrons` argument and CLI `--core-electrons` take **one count per
atom in input order**. When a job-control core table is present, unspecified
elements take Chargemol's element defaults. Check counts against the
pseudopotentials used to generate the density. Total-only cubes ignore the
job-control core table.

Explicit reader arguments or CLI options override the corresponding settings.
For cubes, the job-control `density format` value `onetep` means bohr
coordinates and electrons/Å³, while `e_per_angs3` means ångströms and
electrons/Å³. WFX always takes core counts and periodicity from the
wavefunction. The net charge can still be set with `net charge` or `--charge`.

The CLI also reads `charge type` (`ddec6` or `ddec3`), `compute bos`, and
`print atomic densities`. In Python, select the method and bond calculation
when calling `compute_ddec` and select exports when calling
`write_density_results`. `read_density` only prepares the grid.

## Supply density arrays in Python

`DensityGrid` takes lengths in bohr and densities in electrons/bohr³. For
an array of shape `(nx, ny, nz)`, the sample at `[i, j, k]` lies at
`origin + [i, j, k] @ steps`. The three rows of `steps` are voxel vectors.
For a periodic axis, include each sampled point only once. Supply
`spin_density` with the density shape for collinear spin, or append a final
axis of length three for Cartesian spin components.

The quartz fixture shows how to construct a periodic grid from a reconstructed
valence density, a core density, and a PAW pseudo-valence density. From the
repository root, run a charge-only partition and check the first silicon atom.

```python
import json
from pathlib import Path

import numpy as np
from py_chargemol import DensityGrid, compute_ddec6

fixtures = Path("test/density_examples")
meta = json.loads((fixtures / "quartz.json").read_text())
arrays = {}
for field in ("density", "core_density", "pseudodensity"):
    with np.load(fixtures / f"quartz_{field}.npz") as archive:
        arrays[field] = archive["values"]

grid = DensityGrid(
    atomic_numbers=np.array(meta["atomic_numbers"]),
    positions=np.array(meta["positions"]),
    origin=np.array(meta["origin"]),
    steps=np.array(meta["steps"]),
    core_electrons=np.array(meta["core_electrons"]),
    periodic=tuple(meta["periodic"]),
    net_charge=meta["net_charge"],
    **arrays,
)
result = compute_ddec6(grid, compute_bos=False, compute_spin=False)
print(round(result.charges[0], 6))  # 1.894978 e
```

For an all-electron array, put the full density in `density` and use the
default zero core counts. For a valence-only grid, supply `core_electrons`.
When no core-density grid is supplied, the calculation uses bundled core
reference densities for the requested element and integer core count.

## Choose results and output files

The CLI writes to the current working directory by default. `--output-dir`
selects a different directory. In Python, `write_density_results` creates
the destination and returns a dictionary of paths for this run. Files of
the same name are overwritten. Use a separate directory per calculation to
keep outputs from earlier runs with different options apart.

In the table, `METHOD` stands for `DDEC6` or `DDEC3`.

| Python quantity | Filename | When written |
| --- | --- | --- |
| `result.charges`, `result.analysis.atomic_dipoles`, `result.analysis.atomic_quadrupoles` | `METHOD_net_atomic_charges.xyz` | Always. The XYZ positions are in ångströms, the charges and atomic multipoles in atomic units. |
| `result.cm5_charges`, `result.hirshfeld_charges` | `CM5_net_atomic_charges.xyz`, `Hirshfeld_net_atomic_charges.xyz` | Always |
| `result.spin.populations` | `METHOD_atomic_spin_moments.xyz` | With spin input and spin partitioning enabled. One value per atom for collinear spin, three for noncollinear spin. |
| `result.bonds.summed_bond_orders`, `result.bonds.bond_orders` | `METHOD_bond_orders.xyz` | With bonds. The file reports each atom's sum and lists individual bond orders above 0.001. |
| `result.overlaps`, `result.bonds.contact_exchanges`, `result.bonds.bond_orders` | `overlap_populations_and_contact_exchanges.txt` | With bonds. Each row identifies an owner, partner, and lattice translation. |
| `result.radial_moments` | `DDEC_atomic_Rsquared_moments.xyz`, `DDEC_atomic_Rcubed_moments.xyz`, `DDEC_atomic_Rfourth_moments.xyz` | With bonds. The three array columns correspond to powers 2, 3, and 4. |
| `result.atomic_densities`, `result.radial_shell_weights` | `spherically_averaged_atomic_densities.txt` | With bonds and `--atomic-densities` or `atomic_densities=True`. Includes 100 shells per atom and analytic shell integrals. |
| `result.compute_qdr().charges` | `QDR-DDEC6_point_charges.xyz` | DDEC6 with bonds. On by default for the CLI, opt in with `qdr=True` for the Python writer. |

`--no-qdr` skips resorption while retaining bond analysis. `--no-bonds`
skips the pair calculations, radial moments, spherical-density exports, and
QDR. In Python, test `result.has_bond_orders` before accessing those arrays
when using `compute_bos=False`. Add `compute_spin=False` to skip the spin
partition of a magnetic density and reduce peak memory. This option requires
`compute_bos=False`.

For molecules, `result.moments` gives the total multipoles about the centre
of mass. It is `None` when any axis is periodic. Noncollinear spin inputs can
also produce `result.spin_gradients`. For supported single-centre WFX inputs,
`result.wavefunction_radial_moments` gives analytic explicit-electron moments
for powers −1 through 4.

## Diagnose an electron-count mismatch

If the error says `density integrates to ... valence electrons; expected ...`,
the expected count is `sum(atomic_numbers - core_electrons) - net_charge`.
For a plain grid the sampled integral is `grid.density.sum() * grid.pixel_volume`.
The analysis also includes occupancy corrections for PAW or analytic WFX
contributions when present. It rejects discrepancies larger than 0.1 electrons
or 0.1% of the expected count, whichever is greater.

Check the sign of the system charge (positive means electrons were removed),
the core counts and density kind, and the density units independently of the
coordinate units. For array input, also check the voxel volume and repeated
boundary planes. For a molecular grid, check that the box contains the density
tails. Assess convergence of the quantities of interest by refining the input
grid even when its integral passes the count check.
