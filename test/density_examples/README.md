# Density regression inputs

These fixtures exercise density analysis without an external Chargemol
installation or the original input downloads. Attribution for the Chargemol
fixtures is retained in [LICENSE-CHARGEMOL](../../LICENSE-CHARGEMOL).

The NaCl and quartz fixtures each contain three NPZ files: `*_density.npz`
stores the valence density, `*_core_density.npz` the core density, and
`*_pseudodensity.npz` the pseudo-valence density. Each archive uses lossless ZIP
LZMA compression and contains one NumPy array named `values`. Densities and
geometry are in atomic units, as returned by the VASP reader.

## NaCl

The NaCl fixture is derived from the VASP5 NaCl bulk example distributed with
Chargemol 3.5 (September 26, 2017). [nacl.json](nacl.json) records its geometry,
core-electron counts, source-file SHA-256 hashes, and reference DDEC6/DDEC3
results. The tests compare charges, multipoles, fitted tails, and directed
overlaps with the original reference calculations.

## Triplet ozone

`ozone_triplet_PW91.wfx` is the unmodified Gaussian triplet ozone wavefunction
from `examples_to_run/GAUSSIAN_WFX_ozone_triplet_PW91_example` in the same
Chargemol release. Its [JSON companion](ozone_triplet_PW91.json) records the
input SHA-256 hash and DDEC6 charges, multipoles, spin moments, overlaps,
individual and summed bond orders, and bond spin polarizations from the
distributed `DDEC6` output files. The regression
constructs the density directly from the WFX and checks the full calculation
at the precision printed by those reference files, including the tight
Gaussian spin contributions and their analytic residual corrections.

## Alpha-quartz

[quartz.json](quartz.json) and the three `quartz_*.npz` grids describe a
primitive alpha-quartz cell with randomly displaced atoms, computed with VASP
6.6.0 (PBE, PAW, 680 eV). [make_quartz.py](make_quartz.py) read the calculation
with the VASP reader and averaged each density over blocks of 3 × 3 × 3 points,
from 96³ to 32³, preserving each density's integral. The JSON records the
calculation settings and the SHA-256 hashes of the source CHGCAR, AECCAR0,
AECCAR2, and POTCAR. The source files are not distributed: POTCARs are licensed
with VASP. The stored fixture omits the spin grid because the calculation is
effectively nonmagnetic; the source metadata records its residual
magnetization.

Unlike the Chargemol examples, the quartz references are py-chargemol's own
results, recorded by `make_quartz.py`. This fixture detects changes in results;
it does not independently establish their accuracy. The regression uses
relative and absolute tolerances of `1e-9` and covers a low-symmetry covalent
cell, per-pair bond orders, Hirshfeld and CM5 charges, and radial moments.

After an intentional change to the method, regenerate the recorded DDEC6
results from the stored grids by running this from the repository root:

```bash
uv run --locked python test/density_examples/make_quartz.py
```

This updates the `DDEC6` section of `quartz.json` and leaves the grids and source
metadata intact. Review the changes and explain the reason in the commit.
Passing the original VASP calculation directory as an argument also rebuilds
the grids and source metadata before recording results.
