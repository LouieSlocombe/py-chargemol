# Changelog

## Unreleased

- Add independent Python/NumPy DDEC6 and DDEC3 density partitioning, core-density
  reconstruction, multipoles, collinear/noncollinear spin moments, bond orders,
  overlap populations, Hirshfeld/CM5 charges, and direct density-to-QDR analysis.
- Bundle reference-ion and core-density data with original attribution and
  provenance hashes; no Chargemol installation or external data path is needed.
- Read cube, VASP, XSF, and WFX inputs and add `py-chargemol analyze` with Python
  APIs for file and array workflows.
- Accept the September 2017 `overlap_populations.xyz` filename directly.
- Clarify runtime requirements, installation from a checkout, and density inputs
  required for the standalone calculation.
- Update existing conda environments in place instead of deleting them when
  running the development installer.
- Reject nonfinite values in Chargemol input files before computing QDR charges.
- Remove the bundled QDR program and keep tests and their required reference
  fixtures under `test/` (#11).
- Add `calculate_qdr` to read existing Chargemol output and return QDR charges,
  residual atomic multipoles, and solver diagnostics without writing files.
- Add `compute_qdr_from_arrays` for direct computation from Python lists or NumPy
  arrays, without intermediate files or electron-cloud parameters.
- Include total molecular multipole moments in `QDRResult.moments` for
  nonperiodic systems; periodic results use `None`.
- Export the computation functions, DDEC6 readers, and analysis and overlap-pair
  types from the package root. Preserve `run_qdr` and the command-line interface
  for workflows that write the legacy output files.

## 0.1.0

First release of the Python port of T. A. Manz's April 9, 2025 QDR program.

- Compute quadrupole-dipole resorbed DDEC6 point charges through the
  `py-chargemol qdr` command or the typed Python API.
- Read Chargemol net atomic charges and overlap populations, including the
  September 2017 atomic-charge layout and atom/pair counts that overflow their
  fixed-width fields.
- Write QDR charge files, residual atomic multipoles, electron-cloud parameters,
  and total molecular multipole moments. The command also writes a computation
  log.
- Reproduce the six reference worked examples within rounding at the last
  printed digit, with regression tests included in the source distribution.
- Support Python 3.14 or newer with NumPy as the only runtime dependency.

The package retains the original QDR license and citation requirements. The API
is in alpha and may change before version 1.0.
