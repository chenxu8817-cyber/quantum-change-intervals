# Paper I numerical certification

## Scope

Paper I retains only one-interval assets.  Multiple-interval CSV files,
forest tests, and the Lean combinatorial proof belong to Paper II and are not
evidence for the claims in this manuscript.

## Certified unknown-length grid

`certified_sdp_results.csv` contains 30 cases with

- \(n=3,\ldots,7\);
- \(c=0.3,0.6,0.8,0.9,0.95,0.99\);
- independent primal and dual solutions;
- signed and absolute gaps;
- raw and normalized completeness residuals;
- primal/dual PSD violations;
- complementarity residuals;
- postprocessed feasible primal and dual objectives;
- the feasible interval width, primal contraction, and dual safety shift;
- rechecked safe completeness and PSD residuals.

The completed reference verification has all 30 primal and dual statuses equal
to optimal. Its largest absolute primal--dual gap is
\(6.782\times10^{-9}\), its largest normalized Frobenius completeness residual
is \(1.759\times10^{-12}\), and its largest primal and dual PSD violations are
\(6.801\times10^{-10}\) and \(2.822\times10^{-10}\), respectively.

These raw objectives are solver diagnostics, not the reported rigorous
bracket. For the upper bound, the largest negative dual-slack eigenvalue is
repaired by an identity shift plus a scale-aware positive roundoff margin. For
the lower bound, every raw POVM element is projected onto the PSD cone,
normalized by the inverse square root of its sum, and conservatively completed
after a final contraction. The largest resulting feasible interval width is
\(8.854\times10^{-9}\). The largest safe completeness residual is
\(1.377\times10^{-15}\), every safe primal and dual PSD violation is zero,
and the smallest safe primal eigenvalue and rechecked dual slack eigenvalue
are \(3.4103\times10^{-13}\) and \(3.4102\times10^{-13}\), respectively.

## Fixed and growing known lengths

`paper1_fixed_and_growing_srm.csv` checks three schedules:

- fixed \(i=2\), compared with the elliptic/Toeplitz limit;
- \(i=\lceil\sqrt N\rceil\);
- \(i=\lfloor N/2\rfloor\),

for \(N=10,20,40,80\) and \(c=0.5,0.8,0.95\).

## No-anomaly hypothesis

`paper1_h0_certified_sdp.csv` solves weighted-prior primal and dual SDPs for

- \(c=0.5,0.8,0.95\);
- \(\pi_0=0.2,0.5\);
- \(N=3,5\);
- \(i=1,2\).

All 24 cases have optimal primal and dual statuses. The largest absolute gap
is \(2.149\times10^{-9}\), the largest normalized completeness residual is
\(4.856\times10^{-13}\), and the largest dual PSD violation is
\(6.889\times10^{-10}\).

The same feasible postprocessing is applied to these weighted-prior cases. The
largest safe interval width is \(3.883\times10^{-9}\), the largest safe
completeness residual is \(4.014\times10^{-16}\), and every safe primal and
dual PSD violation is zero. The smallest safe primal eigenvalue and safe dual
slack eigenvalue are \(2.2726\times10^{-13}\) and
\(2.2734\times10^{-13}\), respectively.

The endpoint unit tests additionally verify perfect discrimination at \(c=0\)
and largest-prior guessing at \(c=1\).

## Exact SRM scaling

`srm_scaling_m1.csv` contains exact dense physical-Gram SRM results for
\(n=10,20,30,40,50\) and \(c=0.5,0.8,0.9,0.95,0.99\).

## Verification semantics

`verify_paper1_results.py` compares regenerated and archived scientific
columns within declared tolerances.  It ignores wall-clock and solver-iteration
columns. It separately checks the raw residual thresholds, safe residual
thresholds, ordering of every feasible interval, and consistency of the
lower/upper aliases used by Figure 3(c). A SHA256 manifest records the exact
manuscript, code, tests, data, and environment metadata used in the run.

The stored verification report has passed: true: all 30 certified SDP rows and
all 25 SRM-scaling rows agree with their archived counterparts within the
declared tolerances.
