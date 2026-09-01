# Paper I numerical verification and safeguarded SDP brackets

## Scope

Paper I retains only one-interval assets.  Multiple-interval CSV files,
forest tests, and the Lean combinatorial proof belong to Paper II and are not
evidence for the claims in this manuscript.

## Unknown-length SDP grid

`certified_sdp_results.csv` contains 30 cases with

- \(n=3,\ldots,7\);
- \(c=0.3,0.6,0.8,0.9,0.95,0.99\);
- independent primal and dual solutions;
- signed and absolute gaps;
- raw and normalized completeness residuals;
- primal/dual PSD violations;
- complementarity residuals;
- postprocessed primal and dual bracket endpoints;
- the floating-point bracket width, primal contraction, and dual safety shift;
- rechecked completeness and PSD residuals.

The completed reference verification has all 30 primal and dual statuses equal
to optimal. Its largest absolute primal--dual gap is
\(6.782\times10^{-9}\), its largest normalized Frobenius completeness residual
is \(1.759\times10^{-12}\), and its largest primal and dual PSD violations are
\(6.801\times10^{-10}\) and \(2.822\times10^{-10}\), respectively.

These raw objectives are solver diagnostics, not the reported safeguarded
floating-point bracket. For the upper endpoint, the largest negative
dual-slack eigenvalue is
repaired by an identity shift plus a scale-aware positive roundoff margin. For
the lower endpoint, every raw POVM element is projected onto the PSD cone. A
singular-safe spectral contraction makes the projected sum no larger than the
identity, the PSD remainder completes one outcome, and a scale-aware mixture
with the uniform POVM makes every effect positive definite. In exact
arithmetic, exactly feasible endpoints would be lower and upper bounds by weak
duality. The implementation uses IEEE
double precision and conventional eigensolvers, without directed rounding,
interval arithmetic, or exact-rational verification; the values below are
therefore numerical diagnostics rather than formal certificates. The largest
resulting floating-point bracket width is \(1.025\times10^{-8}\). The largest
rechecked completeness residual is \(1.408\times10^{-15}\), every rechecked
primal and dual PSD violation is reported as zero at evaluator resolution, and
the smallest postprocessed primal eigenvalue and dual slack eigenvalue
are \(3.4104\times10^{-13}\) and \(3.4102\times10^{-13}\), respectively.

## Fixed and growing known lengths

`paper1_fixed_and_growing_srm.csv` checks three schedules:

- fixed \(i=2\), compared with the elliptic/Toeplitz limit;
- \(i=\lceil\sqrt N\rceil\);
- \(i=\lfloor N/2\rfloor\),

for \(N=10,20,40,80\) and \(c=0.5,0.8,0.95\).

## No-change hypothesis

`paper1_h0_certified_sdp.csv` solves weighted-prior primal and dual SDPs for

- \(c=0.5,0.8,0.95\);
- \(\pi_0=0.2,0.5\);
- \(N=3,5\);
- \(i=1,2\).

All 24 cases have optimal primal and dual statuses. The largest absolute gap
is \(2.149\times10^{-9}\), the largest normalized completeness residual is
\(4.856\times10^{-13}\), and the largest dual PSD violation is
\(6.889\times10^{-10}\).

The same safeguarded floating-point postprocessing is applied to these
weighted-prior cases. The largest numerical interval width is
\(4.393\times10^{-9}\), the largest rechecked completeness residual is
\(4.108\times10^{-16}\), and every rechecked primal and dual PSD violation is
reported as zero at evaluator resolution. The smallest postprocessed primal
eigenvalue and dual
slack eigenvalue are \(2.2724\times10^{-13}\) and
\(2.2734\times10^{-13}\), respectively.

The endpoint unit tests additionally verify perfect discrimination at \(c=0\)
and largest-prior guessing at \(c=1\).

## Exact SRM scaling

`srm_scaling_m1.csv` contains exact dense physical-Gram SRM results for
\(n=10,20,30,40,50\) and \(c=0.5,0.8,0.9,0.95,0.99\).

## Verification semantics

`verify_paper1_results.py` compares regenerated and archived scientific
columns within declared tolerances.  It ignores wall-clock and solver-iteration
columns. It separately checks the raw residual thresholds, safeguarded
floating-point residual thresholds, ordering of every numerical interval, and
consistency of the
lower/upper aliases used by Figure 3(c). A SHA256 manifest records the exact
manuscript, code, tests, data, and environment metadata used in the run.

The stored verification report records `passed: true`; all 30 SDP rows and
all 25 SRM-scaling rows agree with their archived counterparts within the
declared tolerances.
