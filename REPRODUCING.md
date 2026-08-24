# Reproducing the numerical results

The certified environment is Python 3.12.13 with the exact versions in
requirements-lock.txt. The workspace-local .local_pydeps directory is only a
convenience cache; the lock file and manifest are the reproducibility record.

## Environment

Create a Python 3.12 environment and install the locked dependencies:

    python -m pip install -r requirements-lock.txt

For deterministic timing comparisons, set the BLAS thread counts explicitly:

    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"

Generate the machine and solver manifest:

    python environment_manifest.py

## Tests

    python -m unittest discover -s tests -v

## Small-size primal/dual SDP certification

    python interval_unknown_length_numerics.py --n-min 3 --n-max 7 --c 0.3 0.6 0.8 0.9 0.95 0.99 --solver CLARABEL --output certified_sdp_results.csv

The CSV contains independently recomputed primal and dual objectives, absolute
and relative gaps, primal completeness residuals, primal/dual PSD violations,
and complementarity residuals.

Small fixed-m multi-interval certificates are generated with:

    python fixed_m_sdp_grid.py --m 2 --n 4 5 --c 0.5 0.9 0.99 --output certified_sdp_m2.csv

    python fixed_m_sdp_grid.py --m 3 --n 6 7 --c 0.5 0.9 0.99 --output certified_sdp_m3.csv

## Resource-aware exact SRM scaling

The exact dense m=1 run through n=50 is:

    python srm_scaling.py --m 1 --n 10 20 30 40 50 --c 0.5 0.8 0.9 0.95 0.99 --max-gram-gib 2 --output srm_scaling_m1.csv

For m=2 or larger, choose smaller n. The script refuses a case before
allocation whenever the dense Gram estimate exceeds the configured limit. It
never silently substitutes an approximate matrix for the exact physical Gram.
