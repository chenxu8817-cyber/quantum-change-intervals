# Numerical certification and reproducibility report

## Scope

The numerical work is split into two non-interchangeable layers.

1. Small-size certification solves both the primal and dual minimum-error SDPs.
2. Medium-size scaling computes the exact SRM from the dense physical Gram
   matrix and refuses infeasible allocations before construction.

No approximate matrix is silently substituted for the exact physical Gram.

## Primal and dual SDP certification

The file certified_sdp_results.csv contains 30 cases:

- n = 3,...,7;
- c = 0.3, 0.6, 0.8, 0.9, 0.95, 0.99;
- solver: CLARABEL;
- both primal and dual objectives;
- independently recomputed feasibility and complementarity residuals.

Worst observed certificate diagnostics were:

| diagnostic | maximum |
|---|---:|
| absolute primal-dual gap | 6.782e-9 |
| absolute relative gap | 7.157e-8 |
| primal completeness residual, Frobenius | 2.083e-13 |
| primal PSD violation | 6.802e-10 |
| dual PSD violation | 2.822e-10 |
| complementarity residual | 6.782e-9 |

Small negative gaps at the 1e-9 scale are preserved in the CSV and are
interpreted as floating-point solver tolerance, not truncated to zero.

The same general certificate was also run directly on multi-interval models:

| file | m | n values | c values | maximum absolute gap |
|---|---:|---|---|---:|
| certified_sdp_m2.csv | 2 | 4,5 | 0.5,0.9,0.99 | 4.380e-9 |
| certified_sdp_m3.csv | 3 | 6,7 | 0.5,0.9,0.99 | 5.706e-9 |

Thus primal/dual certification is not inferred from the single-interval
experiment; it is evaluated on the physical m=2 and m=3 Gram matrices as well.

## Exact dense SRM scaling

The resource-aware exact calculations are:

| file | m | n range | largest candidate count | overlap grid |
|---|---:|---:|---:|---|
| srm_scaling_m1.csv | 1 | 10,20,30,40,50 | 1275 | 0.5,0.8,0.9,0.95,0.99 |
| srm_scaling_m2.csv | 2 | 8,10,12,14 | 1365 | 0.5,0.8,0.9,0.95,0.99 |
| srm_scaling_m3.csv | 3 | 7,8,9,10 | 462 | 0.5,0.8,0.9,0.95,0.99 |

For m=1 the exact full-candidate SRM is therefore extended through n=50.
For m=2 and m=3 the exact dense range is intentionally smaller because

    M(n,m) = binomial(n+1, 2m).

At m=2,n=50, M=249900 and the Gram alone requires about 500 GB in float64.
The scaling script rejects such a case before allocation.

## High-overlap stress tests

Every scaling table includes c=0.9,0.95,0.99 and records

    xi(c) = 1 / abs(log(c))

and n/xi. At c=0.99, xi is approximately 99.5. Consequently even n=50 has
n/xi approximately 0.503 and is not in a clean asymptotic regime. The large
finite-size deviations at high overlap are therefore reported as slow
convergence and conditioning stress, not as evidence against the theorem.

## Unit tests

The test suite checks:

- binomial candidate counts;
- bit-mask, incidence, and direct symmetric-difference agreement;
- Gram symmetry, unit diagonal, PSD, and c=0,1 endpoints;
- agreement of the common m=2 constructor with the legacy implementation;
- the general endpoint-matching dichotomy for m=1,...,4;
- SRM endpoint values and the square-root-trace lower bound;
- canonical rank reduction;
- primal/dual recovery of orthogonal, identical, and binary Helstrom cases;
- direct fixed-m multi-interval SDP integration;
- certificate output fields and residuals;
- dense-memory resource refusal;
- environment manifest completeness.
- the seven directed \(m=3\) forest-layer factorizations
  \(B_{F,\boldsymbol\delta}=U_{F,\boldsymbol\delta}V_{F,\boldsymbol\delta}\);
- exact subset-expansion reconstruction of the oriented three-block
  correction and the exclusion of simultaneous upper/lower cross edges.
- exhaustive monotone-forest checks through \(m=6\) of the separator bound
  \(\kappa(F)+z(F)\le m-1\).
- exhaustive checks through \(m=4\) of the two-sided energy-cover inequalities
  used to pay the concatenated multi-depth coefficients.

The file proofs/EnergyPayment.lean machine-checks the discrete exponent and
separator arithmetic with Lean 4 and contains no sorry or custom axiom. It
does not claim to formalize the real-analysis or Schatten-norm part of the
paper, because this workspace does not vendor Mathlib.

Run:

    python -m unittest discover -s tests -v

## Locked environment

The reproducibility files are:

- .python-version: Python 3.12.13;
- pyproject.toml: direct exact dependencies;
- requirements-lock.txt: full exact package snapshot;
- reproduction_manifest.json: machine, solver, BLAS, thread, and file hashes;
- REPRODUCING.md: commands for tests and data regeneration.
