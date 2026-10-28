# Changelog

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
