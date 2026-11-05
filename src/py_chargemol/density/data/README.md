# Reference atomic densities

These archives contain 943 reference-ion files and 954 core-density files from
the `atomic_densities/` directory of
[`chargemol_09_26_2017.zip`](https://sourceforge.net/projects/ddec/files/chargemol_09_26_2017.zip/download),
distributed by Thomas A. Manz and Nidia Gabaldon Limas. The original filenames
have the forms `c2_ZZZ_ZZZ_NNN_500_100.txt` and
`core_ZZZ_ZZZ_NNN_500_100.txt`.

The source files are preserved byte for byte, including their metadata and line
endings. The `c2_*.zip` archives group reference ions by element; the
`core_*.zip` archives group core densities by element and include the available
core-electron counts. [SHA256SUMS.json](SHA256SUMS.json) records the SHA-256 hash
of all 1,897 files. Archive timestamps are fixed to September 26, 2017 for
reproducibility. The integrity test in
[test_references.py](../../../../test/test_references.py) checks every archived
file against the manifest.

The bundled set covers atomic numbers 1–109, a 500 pm radial cutoff, and 100
radial shells. Other resolutions and effective nuclear charges in the original
distribution are not required by this implementation. Runtime interpolation,
normalization, and monotonicity corrections are performed in Python.

The original authors' attribution, licensing terms, and citation requirements
apply to this reference data; see [LICENSE-CHARGEMOL](../../../../LICENSE-CHARGEMOL).
