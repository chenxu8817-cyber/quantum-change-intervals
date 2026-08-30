param(
    [string]$PythonExe,
    [switch]$UseWorkspaceCache,
    [switch]$SkipCertifiedGrid,
    [switch]$FullWorkspaceTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv-paper1\Scripts\python.exe"

if ($PythonExe -and $UseWorkspaceCache) {
    throw "Pass either -PythonExe or -UseWorkspaceCache, not both."
}

if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Python executable not found: $PythonExe"
    }
    $resolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
    Remove-Item Env:QCI_USE_LOCAL_DEPS -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
} elseif ($UseWorkspaceCache) {
    $resolvedPython = (py -3.12 -c "import sys; print(sys.executable)").Trim()
    $env:QCI_USE_LOCAL_DEPS = "1"
    $env:PYTHONPATH = Join-Path $repoRoot ".local_pydeps"
} elseif (Test-Path -LiteralPath $venvPython) {
    $resolvedPython = $venvPython
    Remove-Item Env:QCI_USE_LOCAL_DEPS -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
} else {
    throw "Create .venv-paper1 with Python 3.12 and install requirements-lock.txt, pass -PythonExe, or use -UseWorkspaceCache."
}

$version = (& $resolvedPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($version -ne "3.12") {
    throw "Paper I requires Python 3.12.x; found $version at $resolvedPython"
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$reproduced = Join-Path $PSScriptRoot "reproduced"
New-Item -ItemType Directory -Force -Path $reproduced | Out-Null

Push-Location $repoRoot
try {
    if ($FullWorkspaceTests) {
        & $resolvedPython -m unittest discover -s tests -v
    } else {
        $paper1ReleaseTests = @(
            "tests/test_environment_manifest.py",
            "tests/test_paper1_analytics.py",
            "tests/test_paper1_figures.py",
            "tests/test_paper1_numerics.py",
            "tests/test_paper1_release_packaging.py",
            "tests/test_quantum_interval_numerics.py",
            "tests/test_sdp_certification.py",
            "tests/test_srm_scaling.py",
            "tests/test_unknown_length_full_srm_probe.py",
            "tests/test_weighted_block_tail.py",
            "tests/test_weighted_hull_adaptive_blocks.py",
            "tests/test_weighted_hull_asymptotics.py",
            "tests/test_weighted_hull_continuum_certificate.py",
            "tests/test_weighted_hull_exact.py",
            "tests/test_weighted_hull_outer_ledger.py",
            "tests/test_weighted_hull_regimes.py",
            "tests/test_weighted_hull_sdp.py"
        )
        & $resolvedPython -m unittest @paper1ReleaseTests -v
    }
    if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
    $fixedFigureData = Join-Path $reproduced "paper1_fixed_and_growing_srm.csv"
    $h0Data = Join-Path $reproduced "paper1_h0_certified_sdp.csv"
    & $resolvedPython paper1_numerics.py --fixed-output $fixedFigureData --h0-output $h0Data
    if ($LASTEXITCODE -ne 0) { throw "Paper I numerical grids failed" }
    $sdpFigureData = Join-Path $repoRoot "certified_sdp_results.csv"
    $srmFigureData = Join-Path $repoRoot "srm_scaling_m1.csv"
    if (-not $SkipCertifiedGrid) {
        $sdpFigureData = Join-Path $reproduced "certified_sdp_results.csv"
        & $resolvedPython interval_unknown_length_numerics.py --n-min 3 --n-max 7 --c 0.3 0.6 0.8 0.9 0.95 0.99 --solver CLARABEL --output $sdpFigureData
        if ($LASTEXITCODE -ne 0) { throw "certified SDP grid failed" }
    }
    $srmFigureData = Join-Path $reproduced "srm_scaling_m1.csv"
    & $resolvedPython srm_scaling.py --m 1 --n 10 20 30 40 50 --c 0.5 0.8 0.9 0.95 0.99 --max-gram-gib 2 --output $srmFigureData
    if ($LASTEXITCODE -ne 0) { throw "SRM scaling grid failed" }
    if (-not $SkipCertifiedGrid) {
        & $resolvedPython verify_paper1_results.py --regenerated-dir $reproduced
        if ($LASTEXITCODE -ne 0) { throw "result verification failed" }
    }

    $weightedHullData = Join-Path $repoRoot "proofs\weighted_hull_diagnostics.csv"
    & $resolvedPython proofs\weighted_hull_continuum_outer_probe.py --output $weightedHullData --max-srm-n 48 --max-sdp-n 5
    if ($LASTEXITCODE -ne 0) { throw "weighted-hull diagnostics failed" }

    $authoritativeRoot = Join-Path $PSScriptRoot "quantum_revision_ultracritical"
    $criticalData = Join-Path $authoritativeRoot "data\critical_srm_diagnostics.csv"
    & $resolvedPython (Join-Path $authoritativeRoot "code\unknown_length_full_srm_probe.py") --n 12 18 24 32 --tau 0.25 1 4 --retained --output $criticalData
    if ($LASTEXITCODE -ne 0) { throw "critical-SRM diagnostics failed" }

    $figureOutput = Join-Path $authoritativeRoot "figures"
    & $resolvedPython paper1_make_figures.py --output-dir $figureOutput --known-data $fixedFigureData --sdp-data $sdpFigureData --srm-data $srmFigureData
    if ($LASTEXITCODE -ne 0) { throw "Paper I figure generation failed" }

    & $resolvedPython environment_manifest.py --profile paper1 --output (Join-Path $PSScriptRoot "reproduction_manifest.json") --seed 1729
    if ($LASTEXITCODE -ne 0) { throw "manifest generation failed" }

    $postGenerationTests = @(
        "tests/test_environment_manifest.py",
        "tests/test_paper1_figures.py",
        "tests/test_paper1_numerics.py",
        "tests/test_paper1_release_packaging.py",
        "tests/test_sdp_certification.py",
        "tests/test_unknown_length_full_srm_probe.py",
        "tests/test_weighted_hull_sdp.py"
    )
    & $resolvedPython -m unittest @postGenerationTests -v
    if ($LASTEXITCODE -ne 0) { throw "post-generation release tests failed" }
} finally {
    Pop-Location
}
