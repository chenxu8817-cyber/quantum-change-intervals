# Quantum Change Intervals

This repository contains the code, floating-point numerical diagnostics,
tests, figures, and LaTeX sources for
*Quantum Change Intervals: Exact Asymptotic Localization with Collective
Measurements* by Xu Chen and Xue Ma.

- Preprint: [arXiv:2608.24543](https://arxiv.org/abs/2608.24543)
- Repository: <https://github.com/chenxu8817-cyber/quantum-change-intervals>

## Scope

The Paper I release concerns one nonempty contiguous interval on which a known
pure state changes from `|0>` to `|psi>` and then returns to `|0>`. The prior is
uniform over the relevant interval labels, except for the stated fixed
no-change mass, and the receiver may use an arbitrary collective
minimum-error POVM. Multi-interval models, unknown anomalous states, and
local or sequential optimization are reserved for later work.

## Reproduce the results

The final release run uses CPython 3.12.10 and the versions pinned in
`requirements-lock.txt`. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv-paper1
.\.venv-paper1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paper1\Scripts\python.exe -m pip install -r requirements-lock.txt
.\paper1\run_reproduction.ps1 `
  -PythonExe .\.venv-paper1\Scripts\python.exe
```

The workflow fixes all supported BLAS thread counts to one, runs the public
Paper I test allowlist, regenerates every release CSV and Figures 1, 2, and 3, checks the archived
scientific columns, and writes an environment and file-hash manifest. Use
`-FullWorkspaceTests` in a full repository clone to add the internal hash gate
and the two Paper II regression modules; in that mode the complete suite is
run both before and after generation.

See [REPRODUCING.md](REPRODUCING.md) and
[paper1/NUMERICAL_REPRODUCIBILITY.md](paper1/NUMERICAL_REPRODUCIBILITY.md)
for commands, grids, tolerances, and the interpretation of SDP outputs.

## Main files

- `paper1/quantum_revision_ultracritical/`: authoritative main and
  Supplemental Material sources.
- `paper1_analytics.py`: Toeplitz limits and analytic special cases.
- `paper1_numerics.py`: fixed and growing known lengths, including the
  fixed-prior no-change model.
- `interval_unknown_length_numerics.py`: 30-case unknown-length SDP grid.
- `srm_scaling.py`: dense physical-Gram SRM calculations through `n=50`.
- `sdp_certification.py`: primal and dual postprocessing for floating-point
  diagnostic brackets.
- `paper1_make_figures.py`: publication figure generator for Figures 1, 2, and 3.
- `paper1/build_release.py`: deterministic, allowlisted release packager.

The historical words `certified` and `certification` remain in some filenames
for backward compatibility. They do not denote interval-arithmetic or
exact-rational certification. The manuscript consistently treats these
calculations as floating-point diagnostics.

## Manuscript compilation

Compile `paper1/quantum_revision_ultracritical/main.tex` with pdfLaTeX and
BibTeX. Compile `supplement.tex` separately for the Supplemental Material. On
Overleaf, upload the files from the same directory, preserve the `figures/`
subdirectory, choose pdfLaTeX, and select the appropriate root file.

## Release status

The release target for this update is `v1.5.1-paper1`. The tag and GitHub
release are declared current only after the remote assets have been downloaded
and checked against the published SHA-256 list.
