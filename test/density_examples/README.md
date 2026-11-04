# Density regression inputs

The NaCl fixture is derived from the VASP5 NaCl bulk example distributed with
Chargemol 3.5, September 26, 2017. `nacl.json` records its geometry, core counts,
the original input-file SHA-256 hashes, and reference DDEC6/DDEC3 results.

The three NPZ files store the valence, core, and pseudo-valence grids in atomic
units, as returned by the VASP reader. They use lossless ZIP LZMA compression
and contain one ordinary NumPy `values` array each. The regression requires
neither the original download nor a Fortran executable. The tests compare
computed charges, multipoles, fitted tails, and directed overlaps with the
original reference calculation.

The original program and data attribution is retained in `LICENSE-CHARGEMOL`.

`ozone_triplet_PW91.wfx` is the unmodified Gaussian triplet ozone wavefunction
from `examples_to_run/GAUSSIAN_WFX_ozone_triplet_PW91_example` in the same
Chargemol release. Its JSON companion records the input SHA-256 and DDEC6
charges, multipoles, spin moments, overlaps, summed bond orders, and bond spin
polarizations from the distributed `DDEC6` output files. The regression
constructs the density directly from the WFX and checks the full calculation
at the precision printed by those reference files, including the tight
Gaussian spin contributions and their analytic residual corrections.

`quartz.json` and the three `quartz_*.npz` grids are a rattled primitive
alpha-quartz cell, computed with VASP 6.6.0 (PBE, PAW, 680 eV). `make_quartz.py`
read it with the VASP reader and averaged each density over blocks of 3 × 3 × 3
points, from 96³ to 32³, which keeps every density's integral. The JSON records
the settings and the SHA-256 hashes of the source CHGCAR, AECCAR0, AECCAR2, and
POTCAR. The source files are not distributed: POTCARs are licensed with VASP.

Unlike the Chargemol examples, the quartz references are py-chargemol's own
results, recorded by `make_quartz.py`, with no Fortran calculation behind them.
The regression therefore catches changes rather than errors. In return it
holds results to 1e-9 instead of the six printed decimals of Chargemol's
output, and it adds a low-symmetry covalent cell, per-pair bond orders, and
Hirshfeld, CM5, and radial-moment results. After a deliberate change to the
method, run `python test/density_examples/make_quartz.py` to record the new
results, and say why in the commit.
