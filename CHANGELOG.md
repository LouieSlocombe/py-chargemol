# Changelog

## Unreleased

Faster density analysis and input reading. Results change by less than 2e-11
on the regression fixtures, mostly by floating-point rounding.

- Add `read_wfx(..., interpolate_diffuse=False)`, which evaluates diffuse
  Gaussian products directly on the final grid instead of interpolating them
  from coarser nested grids. It is slower but free of interpolation error,
  which moves the ozone example's charges by about 2e-4 e. The default still
  reproduces Chargemol.
- Add a closed-form QDR solver: `solver="direct"` on every QDR entry point,
  `--solver direct` on `py-chargemol qdr`, and `--qdr-solver direct` on
  `py-chargemol analyze`. Each atom's linear systems are the identity plus a
  matrix of rank at most five, so they are solved exactly with a 5×5 or 3×3
  system per atom instead of by conjugate gradients. Results agree with the
  default conjugate gradients to rounding, and `QDRResult.solver` records the
  choice. The default still reproduces the QDR program's output.
- Build WFX density grids about 3× faster by evaluating each pair of primitive
  groups with matrix products instead of a Python loop over primitive pairs.
- Build WFX density grids about 3× faster again on grids whose voxels lie
  along x, y and z, which include every molecular WFX grid. Cartesian
  Gaussian products are separable, so each pair is assembled from
  one-dimensional factors and added to the grid as one block. Grids with
  oblique voxels keep the point-by-point evaluation.
- Solve the DDEC6 tail-reshaping population constraint exactly with Newton
  steps instead of bisection, in a few evaluations rather than up to 130.
  Charges move by up to 2e-11, since bisection stopped within 1e-10 of the
  target population.
- Parse density blocks in cube, VASP, and XSF files about 5× faster, with the
  same values and error messages.
- Speed up DDEC6, DDEC3, core, spin, multipole, bond-order, and CM5 calculations
  by about 1.5–2×: shared sample gathers, masked instead of compacted arrays,
  reciprocal pseudodensities, shell-level spin angular functions, and
  vectorized pair screening.
- Reduce peak memory by about a quarter. Atomic integration samples store int32
  grid `coordinates`, which bond integration reuses, in place of the unused
  image `translations`.

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
