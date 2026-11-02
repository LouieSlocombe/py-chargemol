# Reference atomic densities

These archives contain the 943 original `c2_ZZZ_ZZZ_NNN_500_100.txt` reference
ion files and the `core_ZZZ_ZZZ_NNN_500_100.txt` core-density files from the
`atomic_densities/` directory of
`chargemol_09_26_2017.zip`, distributed by Thomas A. Manz and Nidia Gabaldon Limas
at <https://sourceforge.net/projects/ddec/files/chargemol_09_26_2017.zip/download>.

The source files are preserved byte for byte, including their metadata and line
endings. Each archive groups one element's charge states. `SHA256SUMS.json`
records the SHA-256 hash of every original file. Archive timestamps are fixed
to September 26, 2017 for reproducibility.

The bundled set covers atomic numbers 1–109, a 500 pm radial cutoff, and 100
radial shells. Other resolutions and effective nuclear charges in the original
distribution are not required by this implementation. Runtime interpolation,
normalization, and monotonicity corrections are performed in Python.

The original authors' attribution, licensing terms, and citation requirements
apply to this reference data; see the repository's license notices.
