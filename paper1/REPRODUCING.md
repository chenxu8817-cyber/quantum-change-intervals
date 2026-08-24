# Reproducing Paper I

The certified environment uses CPython 3.12.x. The current audited rerun used
3.12.10; the earlier archived runtime record used 3.12.13. Do not run these
commands with the workspace default Python 3.14.

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

## Audited workspace cache

The existing `.local_pydeps` directory is a CPython 3.12 convenience cache.
It is not part of the clean-environment protocol.  To reproduce the current
workspace without reinstalling packages, opt in explicitly:

```powershell
.\paper1\run_reproduction.ps1 -UseWorkspaceCache
```

The script rejects a non-3.12 interpreter before importing compiled packages.

## Outputs

The workflow:

1. runs the complete unit-test suite;
2. regenerates fixed- and growing-length SRM checks;
3. regenerates the fixed-\(H_0\) primal/dual SDP grid;
4. regenerates the 30-case unknown-length primal/dual grid;
5. regenerates the exact dense \(m=1\) SRM scaling table through \(n=50\);
6. compares scientific columns with archived tables using explicit tolerances;
7. records a Paper-I-only environment and SHA256 manifest.

Timing and iteration-count columns are reported but excluded from numerical
identity checks.  They are machine-dependent diagnostics, not scientific
results.

## Fast validation

To skip the 30-case certified SDP regeneration while editing prose:

```powershell
.\paper1\run_reproduction.ps1 -UseWorkspaceCache -SkipCertifiedGrid
```

This mode is a development check and is not the final reproduction protocol.
