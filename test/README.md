# Numerical validation

The suite runs offline using the reference densities installed with the
package and the fixtures in this directory. Run these commands from the
repository root with the development environment installed.

| What to check | Command |
| --- | --- |
| QDR output against all six original examples | `python -m pytest test/test_worked_examples.py` |
| Periodic NaCl and quartz from stored density arrays | `python -m pytest test/test_density_analysis.py -k 'nacl or quartz'` |
| Molecular charges, spins, and bonds from the triplet ozone WFX | `python -m pytest test/test_density_bonds.py -k ozone_triplet` |
| Bundled atomic-density files and reference-ion interpolation | `python -m pytest test/test_references.py` |

## Density calculations

- **Triplet ozone.** The WFX is an original Chargemol 3.5 input. Charges,
  dipoles, quadrupoles, spins, overlaps, and bond orders are compared with
  the distributed output at the precision of the corresponding table.
- **Bulk NaCl.** The original VASP example has an oblique primitive cell.
  py-chargemol uses reciprocal-lattice bounds to cover the full integration
  sphere. This changes the sodium charge by about 2.5 × 10⁻⁵ e for DDEC6 and
  7 × 10⁻⁶ e for DDEC3 relative to Chargemol. The tests use corrected
  reference values checked against larger integration boxes. Both sets of
  results are retained.
- **Alpha-quartz.** The low-symmetry cell exercises per-pair bond orders,
  Hirshfeld and CM5 charges, and radial moments. Its reference is a recorded
  py-chargemol calculation, useful for detecting numerical changes.

For fixture provenance, tolerances, and quartz regeneration instructions, see
the [density fixture notes](density_examples/README.md). The 1,897 original
reference-ion and core-density files are checked byte for byte by SHA-256.
Their source and radial resolution are documented with the
[packaged archives](../src/py_chargemol/density/data/README.md).

The Python density engine reports failed charge-partition convergence and
negative valence populations as errors. Its grid checks permit coarse grids
for numerical experiments and differ from Chargemol's negative-density
thresholds. Check grid convergence for the physical quantities of interest.
Quadrupole eigenvalues use the complete tensor, including analytic WFX
corrections. For periodic, effective-core, and ghost-centre WFX inputs the
suite checks analytic cases and conservation. Full external reference
calculations for these cases are outside the fixture set. Development
comparisons also used the supplied CP2K water and four Gaussian ozone examples.

## QDR output compatibility

The [six worked examples](worked_examples/) contain the original Chargemol
inputs and reference QDR output from T. A. Manz's April 9, 2025 program.
The regression runs the CLI on a temporary copy and compares the output line
by line. Numeric fields allow rounding in the last printed digit. Reference
and computed solver residuals must each be below `1e-10`. The version line
identifies py-chargemol and the timestamp reflects the run.

The reader finds the multipole table by its title, accommodating the
different header lengths in the September 2017 Chargemol release and later
files. When a five-character atom or pair count overflows to `*****`, the
reader recovers the count from the table rows. Other numeric overflows
raise errors with file and line information where available.

The solver uses NumPy arrays across atoms, 64-bit pair indices and
translations, and `numpy.linalg.eigvalsh` for quadrupole eigenvalues. The
XYZ writer uses LF line endings. The log contains timings and solver
diagnostics in the Python logger's layout.
