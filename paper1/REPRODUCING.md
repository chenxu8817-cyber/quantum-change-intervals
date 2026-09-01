# Reproducing Paper I

The pinned numerical environment uses CPython 3.12.x. The final release rerun
uses CPython 3.12.10. Do not run these commands with the workspace default
Python 3.14.

## Clean environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv-paper1
.\.venv-paper1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paper1\Scripts\python.exe -m pip install -r requirements-lock.txt
.\paper1\run_reproduction.ps1
```

The default workflow uses only the activated `.venv-paper1`; repository code
does not silently prepend `.local_pydeps`.

An existing clean Python 3.12 environment can instead be selected explicitly:

```powershell
.\paper1\run_reproduction.ps1 `
  -PythonExe .\.venv-paper1-release-py312\Scripts\python.exe
```

## Audited workspace cache

The existing `.local_pydeps` directory is a CPython 3.12 convenience cache.
It is not part of the clean-environment protocol.  To reproduce the current
workspace without reinstalling packages, opt in explicitly:

```powershell
.\paper1\run_reproduction.ps1 -UseWorkspaceCache
```

The script rejects a non-3.12 interpreter before importing compiled packages.

## Outputs

The release workflow:

1. runs the Paper-I release-test allowlist with all BLAS thread counts fixed to
   one;
2. regenerates fixed- and growing-length SRM checks;
3. regenerates the fixed-\(H_0\) primal/dual SDP grid;
4. regenerates the 30-case unknown-length primal/dual grid;
5. regenerates the exact dense \(m=1\) SRM scaling table through \(n=50\);
6. compares scientific columns with archived tables using explicit tolerances;
7. regenerates the weighted-hull and critical-SRM diagnostic tables;
8. writes Figures 1–3 directly to
   `quantum_revision_ultracritical/figures`;
9. reruns the post-generation release gate against the files that were
   actually written; and
10. records a Paper-I-only environment and SHA-256 manifest bound to the final
    main text, Supplemental Material, code, data, and figures, then verifies
    every recorded hash against the files on disk.

The default allowlist excludes the two Paper-II modules
`test_fixed_m_sdp_grid.py` and `test_m3_forest_factorization.py`. To add the
complete workspace suite as a stronger gate, run

```powershell
.\paper1\run_reproduction.ps1 `
  -PythonExe .\.venv-paper1-release-py312\Scripts\python.exe `
  -FullWorkspaceTests
```

With `-FullWorkspaceTests`, the complete suite runs both before generation and
again after the final CSV files and figures have been written. Manifest
generation and the manifest freshness check follow the second test pass.

Timing and iteration-count columns are reported but excluded from numerical
identity checks.  They are machine-dependent diagnostics, not scientific
results.

## Fast validation

To skip the 30-case SDP regeneration while editing prose:

```powershell
.\paper1\run_reproduction.ps1 -UseWorkspaceCache -SkipCertifiedGrid
```

This mode is a development check and is not the final reproduction protocol.

## Release packaging gate

After the clean PDF builds and the Paper-I profile manifest have passed their
checks, create the submission archives in a new or empty directory:

```powershell
.\.venv-paper1-release-py312\Scripts\python.exe paper1\build_release.py `
  --version v1.5.0-paper1 `
  --main-pdf <clean-main-build>\main.pdf `
  --supplement-pdf <clean-supplement-build>\supplement.pdf `
  --main-bbl <clean-main-build>\main.bbl `
  --output-dir <empty-release-directory>
```

The packager fails closed if `--main-bbl` is omitted, an allowlisted file is
missing, the output directory is nonempty, or
`paper1/reproduction_manifest.json` contains an absolute host path, a missing
hashed file, or a stale SHA-256 value. The source
archives contain only the Paper-I profile manifest; the legacy root-level
all-project manifest is excluded.
