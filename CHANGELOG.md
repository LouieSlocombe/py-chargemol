# Changelog

## Unreleased

This release adds QDR and WFX calculation options, speeds up density analysis,
and reduces memory use. Performance changes preserve the default regression
results to within about 2e-11; disabling WFX interpolation is a separate,
opt-in numerical change.

### Added

- Optional Numba acceleration for core-density averages, DDEC shell statistics,
  radial-weight accumulation, and bond integration. Install
  `py-chargemol[numba]` to compile these kernels automatically on first use and
  cache them for later runs. The default installation retains the NumPy path;
  `PY_CHARGEMOL_DISABLE_JIT=1` selects it explicitly when Numba is installed.
- A direct QDR solver. Use `solver="direct"` in the QDR computation APIs,
  `qdr_solver="direct"` in `write_density_results`, `--solver direct` with
  `py-chargemol qdr`, or `--qdr-solver direct` with `py-chargemol analyze`.
  It solves a 5×5 or 3×3 linear system per atom and agrees with converged
  conjugate gradients to rounding. `QDRResult.solver` records the choice.
  Conjugate gradients remain the default for compatibility with the original
  QDR program.
- `read_wfx(..., interpolate_diffuse=False)` evaluates diffuse Gaussian
  products directly on the final grid. This is slower but avoids interpolation
  from coarser grids, changing the ozone example's charges by about 2e-4 e.
  Interpolation remains enabled by default for compatibility with Chargemol.
- `compute_spin=False` in `compute_ddec6`, `compute_ddec3`, and `compute_ddec`
  skips spin partitioning and spin-gradient diagnostics to reduce memory use.
  Charges and multipoles are unchanged. This option requires
  `compute_bos=False`, because spin-polarized bond orders need the spin
  partition.

### Changed

The performance figures below were measured during development on the tested
inputs. Each describes an individual optimization step with its own baseline.

- Speed up WFX grid construction using matrix products for primitive groups
  and separable Cartesian Gaussian factors for axis-aligned grids. Each change
  reduced grid-construction time by about a factor of three in its benchmarks.
  Oblique grids retain point-by-point evaluation.
- Parse cube, VASP, and XSF density blocks about five times faster, preserving
  values and error messages.
- Replace bisection with Newton steps for the DDEC6 tail-reshaping population
  constraint. This needs only a few evaluations instead of up to 130. Charges
  change by up to 2e-11 because the previous solver stopped within 1e-10 of
  the target population.
- Reduce repeated array operations in DDEC6, DDEC3, core, spin, multipole,
  bond-order, and CM5 calculations. Reuse shell counts between iterations and
  remove unused core-density averages.
- Reduce atomic integration sample storage from 52 to 17 bytes per point using
  compact indices, shells, and grid coordinates. Cartesian displacements are
  reconstructed when needed; flat indices use `int64` for grids beyond `2**31`
  points. Core, DDEC6, and DDEC3 partitions also process one atom at a time to
  avoid retaining every atom's gathered density. These changes preserve
  results bit for bit, with some additional computation during partitioning.
- Build atomic integration samples in slabs of grid planes, making sample
  construction 3–4.5 times faster without changing the samples. For the
  α-quartz cell at 96³, construction took 1.2 s instead of 5.4 s, and the
  charge-only peak memory after the preceding memory reductions fell from
  1.43 to 1.36 GB with spin partitioning disabled.
- Vectorize the cylindrical quadrature used in bond-pair screening and reduce
  repeated passes over each pair's grid points. Together these changes reduced
  bond-analysis time for the α-quartz cell at 96³ from 67 to 34 s while
  preserving results bit for bit.
- Sum bond integrals over pairs of radial shells. In a separate benchmark,
  α-quartz bond-analysis time fell from 38 to 27 s. The changed summation
  order affects overlap populations, contact exchanges, and bond orders by
  at most 9e-16; charges, multipoles, and spins are unchanged.

### Fixed

- Complete DDEC3 and DDEC6 analyses when every WFX electron is integrated
  analytically, retaining charges, multipoles, spin, bonds, and QDR output.
  Keep spin consistent with corrected core densities and include zero-density
  nodes in spherical spin averages.
- Include the periodic image overlaps of diffuse WFX primitives using
  exponent- and coefficient-dependent bounds, avoiding false normalization
  errors from the previous fixed cutoff.
- Preserve WFX densities for independently shifted periodic atom images,
  including oblique cells, while retaining the original output coordinates.
- Honor explicit density formats for extensionless or custom filenames named
  by `job_control.txt` when analyzing a directory.
- Preserve bond pairs, CM5 charges, and QDR results when periodic atoms are
  given in different cell images, retaining their original coordinates and
  adjusting the reported bond translations (#16).
- Include all points within the integration radius on triclinic grids, and
  include nearby periodic images in skew-cell CM5 and WFX calculations (#15).
- Accept WFX atoms whose electrons are all in tight Gaussian primitives, such
  as a hydrogen 1s with exponent 6. Analytic core counts that round just above
  the atomic number are now clamped, with the excess retained as an occupancy
  correction to conserve electrons.

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
