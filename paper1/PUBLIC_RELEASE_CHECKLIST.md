# Paper I public-release gate

## Prepared release identity

- Suggested repository name: `quantum-change-intervals`
- Release-candidate tag: `v1.5.0-paper1`
- Suggested archive title: *Quantum Change Intervals: Exact Asymptotic
  Localization with Collective Measurements - code and data*
- Archive contents: the files hashed by the `paper1` profile in
  `paper1/reproduction_manifest.json`, excluding local virtual environments,
  temporary renderings, TeX auxiliary files, and workspace-only Paper II
  material.

## Release metadata

The manuscript and upload package record the repository URL, author names, Xu
Chen's ORCID and corresponding-author email, both affiliations, funding,
contribution statement, competing-interest statement, and AI-assistance
disclosure. The repository retains its existing MIT License, with the
copyright notice synchronized to Xu Chen and Xue Ma.

Xue Ma's ORCID and a Zenodo or institutional-archive DOI may be added later;
neither item blocks the GitHub tag `v1.5.0-paper1`. Do not add an archival DOI
to the manuscript until it resolves to the immutable tagged release.

## Publication sequence

1. Update the existing repository at
   `https://github.com/chenxu8817-cyber/quantum-change-intervals` without
   rewriting its history or replacing its MIT terms.
2. Run `paper1/run_reproduction.ps1` in a clean CPython 3.12 clone. Build the
   article and Supplemental Material from the same final source snapshot, then
   inspect the source, data, figure, test, and PDF gates.
3. Generate and verify `paper1/reproduction_manifest.json` only after the
   source and release metadata are final. Confirm that it contains no absolute
   host paths.
4. Invoke `paper1/build_release.py` with the fresh `main.bbl` from the same
   clean article build and a new or empty output directory. Extract the public
   source archive, rerun its tests, and compile the arXiv archive in isolation.
5. Commit the frozen snapshot, create and push the immutable
   `v1.5.0-paper1` tag, create the GitHub Release, and compare every downloaded
   asset with the generated `SHA256SUMS` file.
6. A Zenodo or institutional archive may later preserve the tagged release.
   If a DOI is added to the manuscript, do so in a new source version rather
   than modifying the immutable tag.

## Tooling boundary

Git is available and the public remote can be updated through the configured
Git credential manager. The annotated tag is the immutable source anchor; the
GitHub Release carries the corresponding submission and source assets. Zenodo
or another archival service remains optional.
