# Paper I release map

The authoritative manuscript source is
`quantum_revision_ultracritical/main.tex`; the separate Supplemental Material
starts from `quantum_revision_ultracritical/supplement.tex`. Both use the same
Quantum class, bibliography, and notation. The scientific scope is frozen in
the manuscript's ``Assumptions and scope'' subsection.

## Evidence chain

`run_reproduction.ps1` executes the release chain

```text
physical Gram -> SRM -> small floating-point SDP diagnostics
              -> CSV -> Figures 1, 2, and 3 -> environment manifest
```

It also regenerates the weighted-hull and critical-overlap diagnostic tables.
The generated copies of the four principal tables are placed in `reproduced/`
and compared with their archived counterparts using documented tolerances.

## Release map

- `NUMERICAL_REPRODUCIBILITY.md` defines every numerical grid and explains the
  floating-point bracket semantics.
- `REPRODUCING.md` gives the clean CPython 3.12 protocol.
- `reproduction_manifest.json` is the Paper-I profile manifest. The release
  archive includes it as `paper1/reproduction_manifest.json` and excludes the
  legacy root-level all-project manifest.
- `PAPER1_RELEASE_MANIFEST.md` identifies the frozen manuscript, numerical
  outputs, figures, and submission artifacts.
- `build_release.py` creates deterministic arXiv, Quantum, source, and GitHub
  release artifacts from a strict Paper I allowlist. Every allowlisted input
  must exist, `--main-bbl` must name the bibliography produced by the same
  clean build as the main PDF, and the output directory must be empty.

Files for multiple intervals, forest factorizations, and their dedicated tests
belong to Paper II and are excluded from the Paper I release archive.
