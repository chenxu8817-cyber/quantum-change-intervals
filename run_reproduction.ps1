param(
    [switch]$UseWorkspaceCache,
    [switch]$SkipCertifiedGrid
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv-paper1\Scripts\python.exe"

if ($UseWorkspaceCache) {
    $pythonExe = (py -3.12 -c "import sys; print(sys.executable)").Trim()
    $env:QCI_USE_LOCAL_DEPS = "1"
    $env:PYTHONPATH = Join-Path $repoRoot ".local_pydeps"
} elseif (Test-Path -LiteralPath $venvPython) {
    $pythonExe = $venvPython
    Remove-Item Env:QCI_USE_LOCAL_DEPS -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
} else {
    throw "Create .venv-paper1 with Python 3.12 and install requirements-lock.txt, or pass -UseWorkspaceCache for the audited local cache."
}

$version = (& $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($version -ne "3.12") {
    throw "Paper I requires Python 3.12.x; found $version at $pythonExe"
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$reproduced = Join-Path $PSScriptRoot "reproduced"
New-Item -ItemType Directory -Force -Path $reproduced | Out-Null

Push-Location $repoRoot
try {
    & $pythonExe -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
    & $pythonExe paper1_numerics.py
    if ($LASTEXITCODE -ne 0) { throw "Paper I numerical grids failed" }
    $sdpFigureData = Join-Path $repoRoot "certified_sdp_results.csv"
    $srmFigureData = Join-Path $repoRoot "srm_scaling_m1.csv"
    if (-not $SkipCertifiedGrid) {
        $sdpFigureData = Join-Path $reproduced "certified_sdp_results.csv"
        & $pythonExe interval_unknown_length_numerics.py --n-min 3 --n-max 7 --c 0.3 0.6 0.8 0.9 0.95 0.99 --solver CLARABEL --output $sdpFigureData
        if ($LASTEXITCODE -ne 0) { throw "certified SDP grid failed" }
    }
    $srmFigureData = Join-Path $reproduced "srm_scaling_m1.csv"
    & $pythonExe srm_scaling.py --m 1 --n 10 20 30 40 50 --c 0.5 0.8 0.9 0.95 0.99 --max-gram-gib 2 --output $srmFigureData
    if ($LASTEXITCODE -ne 0) { throw "SRM scaling grid failed" }
    if (-not $SkipCertifiedGrid) {
        & $pythonExe verify_paper1_results.py --regenerated-dir $reproduced
        if ($LASTEXITCODE -ne 0) { throw "result verification failed" }
    }
    $figureOutput = Join-Path $PSScriptRoot "figures"
    & $pythonExe paper1_make_figures.py --output-dir $figureOutput --sdp-data $sdpFigureData --srm-data $srmFigureData
    if ($LASTEXITCODE -ne 0) { throw "Paper I figure generation failed" }
    $submissionFigures = Join-Path $PSScriptRoot "quantum_submission\figures"
    New-Item -ItemType Directory -Force -Path $submissionFigures | Out-Null
    foreach ($figureName in @(
        "figure1_model_geometry.pdf",
        "figure1_model_geometry.png",
        "figure2_analytic_limits.pdf",
        "figure2_analytic_limits.png",
        "figure3_finite_size.pdf",
        "figure3_finite_size.png"
    )) {
        Copy-Item -LiteralPath (Join-Path $figureOutput $figureName) -Destination (Join-Path $submissionFigures $figureName) -Force
    }
    & $pythonExe environment_manifest.py --profile paper1 --output (Join-Path $PSScriptRoot "reproduction_manifest.json") --seed 1729
    if ($LASTEXITCODE -ne 0) { throw "manifest generation failed" }
} finally {
    Pop-Location
}
