# Quantum Change Intervals

Code, certified numerical data, tests, figures, and LaTeX sources accompanying
the manuscript *Quantum Change Intervals: Exact Asymptotic Localization with
Collective Measurements* by Xu Chen and Xue Ma.

Repository URL:
<https://github.com/chenxu8817-cyber/quantum-change-intervals>

## Scope

This release is restricted to Paper I: one nonempty contiguous interval in
which the known pure state changes from `|0>` to `|psi>` and then returns to
`|0>`, with arbitrary collective measurements. Exploratory two-interval and
fixed-m material is intentionally excluded.

## Quick start

The audited environment uses CPython 3.12.x. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv-paper1
.\.venv-paper1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paper1\Scripts\python.exe -m pip install -r requirements-lock.txt
.\paper1\run_reproduction.ps1
```

For a fast test-and-figure check that skips regeneration of the full certified
SDP grid:

```powershell
.\paper1\run_reproduction.ps1 -SkipCertifiedGrid
```

See [REPRODUCING.md](REPRODUCING.md) and
[paper1/NUMERICAL_REPRODUCIBILITY.md](paper1/NUMERICAL_REPRODUCIBILITY.md)
for the full protocol, numerical grids, tolerances, and certificate semantics.

## Main entry points

- `paper1_analytics.py`: Toeplitz limits and analytic special cases.
- `paper1_numerics.py`: fixed/growing-length and no-anomaly calculations.
- `interval_unknown_length_numerics.py`: certified unknown-length SDP grid.
- `srm_scaling.py`: full dense physical-Gram SRM scaling.
- `sdp_certification.py`: primal/dual solutions and safe feasible bounds.
- `paper1_make_figures.py`: publication Figure 1--3 generation.
- `verify_paper1_results.py`: tolerance-based comparison with reference data.
- `environment_manifest.py`: environment metadata and SHA-256 manifest.

## Archived reference data

- `certified_sdp_results.csv`: 30 unknown-length primal/dual SDP cases.
- `srm_scaling_m1.csv`: full dense physical-Gram SRM calculations through
  `n=50`.
- `paper1/paper1_fixed_and_growing_srm.csv`: fixed and growing known lengths.
- `paper1/paper1_h0_certified_sdp.csv`: fixed-prior no-anomaly cases.

Generated reruns are written to `paper1/reproduced/`, which is intentionally
ignored by Git to avoid duplicating the version-controlled reference tables.

## Manuscript source

The Quantum-class source is in `paper1/quantum_submission/`. Compile
`main.tex` with pdfLaTeX/BibTeX. For Overleaf, use pdfLaTeX and TeX Live 2025;
the current Quantum class version 6.1 is not compatible with the TeX Live 2026
`array` package at the theorem-comparison table.

## Citation and release status

Citation metadata and the GitHub repository URL are provided in
`CITATION.cff`. The immutable Paper-I release is tagged
`v1.0.0-paper1`; an archival DOI can be added after a Zenodo or institutional
deposit.

The repository is released under the MIT License. The existing remote MIT
license was retained and its copyright notice was synchronized to both
authors. See `RELEASE_NOTES.md` and `GITHUB_UPLOAD_INSTRUCTIONS.md`.
