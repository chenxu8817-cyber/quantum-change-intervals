# Paper I public-release gate

## Prepared release identity

- Suggested repository name: `quantum-change-intervals`
- Current manuscript release tag: `v1.1.0-paper1`
- Suggested archive title: *Quantum Change Intervals: Exact Asymptotic
  Localization with Collective Measurements - code and data*
- Archive contents: the files hashed by the `paper1` profile in
  `reproduction_manifest.json`, excluding local virtual environments,
  temporary renderings, TeX auxiliary files, and workspace-only Paper II
  material.

## Release metadata

The manuscript and upload package record the repository URL, author names, Xu
Chen's ORCID and corresponding-author email, both affiliations, funding,
contribution statement, competing-interest statement, and AI-assistance
disclosure. The repository retains its existing MIT License, with the
copyright notice synchronized to Xu Chen and Xue Ma.

Xue Ma's ORCID and a Zenodo or institutional-archive DOI may be added later;
neither item blocks the GitHub tag `v1.1.0-paper1`. Do not add an archival DOI
to the manuscript until it resolves to the immutable tagged release.

## Publication sequence

1. Update the existing repository at
   `https://github.com/chenxu8817-cyber/quantum-change-intervals` without
   rewriting its history or replacing its MIT terms.
2. Run `paper1/run_reproduction.ps1` in a clean CPython 3.12 clone, regenerate
   the PDF and manifest, inspect the diff, and push `main`.
3. Create and push the immutable `v1.1.0-paper1` tag. Optionally enable the
   repository in Zenodo and let Zenodo archive that exact tag.
4. If archived, record both the concept DOI and version DOI. Use the version DOI in the
   manuscript's data-and-code statement for exact reproducibility; use the
   concept DOI in general repository documentation if desired.
5. Add the version DOI to the existing data/code statement in
   `quantum_submission/content.tex`, compile `main.pdf`, and confirm that the
   DOI resolves to the archived tag.
6. Compare the Zenodo archive file hashes with
   `paper1/reproduction_manifest.json`.

## Tooling boundary

Git is available and the public remote can be updated through the configured
Git credential manager. GitHub CLI and archival credentials are not assumed;
a pushed annotated tag is the immutable release anchor. Creating a separate
GitHub release page or Zenodo DOI may require the corresponding authenticated
web/API session.
