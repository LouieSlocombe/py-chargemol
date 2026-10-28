# Changelog

## Unreleased

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
  fixed-width fields. QDR still requires the overlap-population input described
  in the README; the September 2017 Chargemol release does not produce it.
- Write QDR charge files, residual atomic multipoles, electron-cloud parameters,
  and total molecular multipole moments. The command also writes a computation
  log.
- Reproduce the six reference worked examples within rounding at the last
  printed digit, with regression tests included in the source distribution.
- Support Python 3.14 or newer with NumPy as the only runtime dependency.

The package retains the original QDR license and citation requirements. The API
is in alpha and may change before version 1.0.
