# Reproducing the Paper I release

The final release environment uses CPython 3.12.10 with the exact package
versions in `requirements-lock.txt`. The manifest records the interpreter,
packages, numerical solvers, platform, BLAS configuration, and source hashes.

## Clean environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv-paper1
.\.venv-paper1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paper1\Scripts\python.exe -m pip install -r requirements-lock.txt
.\paper1\run_reproduction.ps1 `
  -PythonExe .\.venv-paper1\Scripts\python.exe
```

The runner sets

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

before importing the numerical libraries. This setting defines the official
byte-deterministic weighted-hull CSV.

## Generated data

The workflow regenerates:

- `paper1/reproduced/paper1_fixed_and_growing_srm.csv`;
- `paper1/reproduced/paper1_h0_certified_sdp.csv`;
- `paper1/reproduced/certified_sdp_results.csv`;
- `paper1/reproduced/srm_scaling_m1.csv`;
- `proofs/weighted_hull_diagnostics.csv`;
- `paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv`;
- all publication formats for Figure 1–3 in the authoritative `figures/`
  directory;
- `paper1/reproduction_manifest.json`.

The default test gate is the public Paper I allowlist. In a full repository
clone, add `-FullWorkspaceTests` to run the internal weighted-hull hash gate
and the two Paper II regression modules as a separate workspace regression
gate.

## Numerical meaning

The small SDPs report primal and dual solutions, residuals, and postprocessed
floating-point bracket endpoints. They use IEEE double precision without
directed rounding, interval arithmetic, or exact rational verification. They
are finite-size diagnostics and are not formal certificates of the exact
finite-dimensional optimum.

Detailed grids and thresholds are in
`paper1/NUMERICAL_REPRODUCIBILITY.md`.
