# Changelog

## 1.0.0

First release. py-chargemol implements the density analyses of Chargemol 3.5
(September 26, 2017) and T. A. Manz's April 9, 2025 QDR program in Python and
NumPy. No Chargemol installation, Fortran compiler, or external atomic-density
directory is needed.

- Compute DDEC6 and DDEC3 net atomic charges from electron densities, with
  core-density reconstruction, atomic multipoles, collinear and noncollinear
  spin moments, bond orders, overlap populations, and Hirshfeld and CM5
  charges.
- Read VASP, Gaussian cube, XSF, and WFX inputs, or `DensityGrid` arrays in
  memory, with supported `job_control.txt` settings. Analyze them with
  `read_density` and `compute_ddec6` or the `py-chargemol analyze` command.
- Bundle the reference-ion and core-density data for elements 1–109 with their
  original attribution and SHA-256 provenance hashes.
- Compute quadrupole-dipole resorbed (QDR) point charges directly from a density
  analysis (`compute_qdr_from_density`), from existing Chargemol output
  (`calculate_qdr`), or from DDEC6 arrays (`compute_qdr_from_arrays`), without
  writing files. `QDRResult` includes residual atomic multipoles, solver
  diagnostics, and, for nonperiodic systems, total molecular multipole moments.
- Read Chargemol net atomic charges and overlap populations, including the
  September 2017 layouts and atom and pair counts that overflow their
  fixed-width fields.
- Write density-analysis, bond-order, and QDR output files with
  `write_density_results`, `run_qdr`, or the `py-chargemol qdr` command, which
  also writes a computation log.
- Reject malformed, ambiguous, or nonfinite inputs with explicit errors,
  including inconsistent electron counts and invalid QDR arrays.
- Reproduce the six QDR worked examples within rounding at the last printed
  digit, with regression tests included in the source distribution.
- Support Python 3.14 or newer, with NumPy as the only runtime dependency.

The public API is the names exported from `py_chargemol` and the `py-chargemol`
command; later releases follow semantic versioning for them. The package
retains the original QDR and Chargemol licenses, and publications must cite the
original methods.
