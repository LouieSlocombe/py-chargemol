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
