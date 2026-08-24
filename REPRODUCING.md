# Reproducing Paper I

The authoritative clean-environment protocol is
[`paper1/REPRODUCING.md`](paper1/REPRODUCING.md). From the repository root on
Windows PowerShell, the complete audited workflow is:

```powershell
py -3.12 -m venv .venv-paper1
.\.venv-paper1\Scripts\python.exe -m pip install --upgrade pip
.\.venv-paper1\Scripts\python.exe -m pip install -r requirements-lock.txt
.\paper1\run_reproduction.ps1
```

The workflow runs the Paper-I unit tests, regenerates the one-interval
numerical grids and publication figures, verifies the archived scientific
columns and strict feasible SDP bounds, and writes a machine-readable
environment manifest. Paper-II multi-interval calculations are outside this
release.
