# Final Quantum submission review

Date: 2026-08-31

This review was performed after the scientific scope and theorem set were
frozen. It evaluates submission consistency and presentation only; it adds no
new model, theorem, numerical claim, or citation.

## Overall decision

The article and Supplemental Material are ready for release packaging. No
blocking mathematical, editorial, bibliographic, numerical, or typesetting
issue remains in the reviewed snapshot.

The main article is dense but not structurally confused. Its sequence is
coherent: model and Gram geometry, fixed known length, growing known length,
unknown length, moving-overlap regimes, the no-change extension, finite-size
diagnostics, and conclusion. The Supplemental Material is necessarily longer
because it contains the sector decomposition, singular-safe dual bounds,
local square-root analysis, continuum matching, adaptive outer construction,
and the global trace theorem. The proof map at its beginning makes these
dependencies explicit.

## Proof architecture

The two proof routes in the Supplemental Material should be retained.

1. The sectorwise dual-certificate and global-trace route proves the common
   trace, SRM, and optimum law and supplies the explicit convergence rate.
2. The local square-root-diagonal route proves a stronger labelwise statement
   and provides the retained-orthogonality input used by the no-change
   extension.

They are not duplicate derivations of the same statement. Removing the first
would lose the quantitative three-criterion squeeze; removing the second would
lose the local law and its transfer consequences. The present proof overview
is a more appropriate simplification than deleting either route.

## Claim consistency

- The abstract, Introduction, theorem statement, Supplemental Material, and
  conclusion use the same hull-dominance assumptions:
  `c_n -> 1`, `c_n < 1` eventually, and `n p_1(c_n) -> infinity`.
- The conclusion is stated simultaneously for the trace benchmark, SRM, and
  optimum, under the uniform prior on all nonempty intervals. The criterion is
  presented as sufficient; no converse or unnecessary monotonicity assumption
  is claimed.
- Fixed-overlap results remain pointwise in the overlap. Moving-overlap and
  ultra-critical results are presented as sequence statements with distinct
  normalizations. The compact-continuum coefficient is consistently
  `2 lambda^2/pi^4`.
- Every no-change result is described as joint Bayes detection and exact
  localization. The fixed prior `pi_0` and the conditional uniform anomaly
  prior are stated explicitly.
- The raw weighted-SRM counterexample at `c=1` remains explicit. Its
  maximum-a-posteriori relabeling is correctly identified as classical
  postprocessing.
- Finite SDPs are consistently described as floating-point diagnostics. The
  manuscript does not call them formal certificates, exact finite-dimensional
  optima, or rigorous numerical enclosures. For `n>7`, only SRM values are
  reported.

## Originality and related work

The originality claims are concentrated on exact asymptotic Bayes laws,
Toeplitz and Foelner transfer, singular-safe comparison, and asymptotic SRM
optimality. The manuscript makes no unconditional priority claim.

The relation to permanent change points, the two-change framework,
multi-anomaly detection, quantum edge detection, and classical changed-interval
models is stated as a boundary relation, structured subensemble, or geometric
analogy as appropriate. The Mohan--Sikora--Upadhyay embedding uses
`N'=n+1`, `c_1=a`, and `c_2=b+1`; the common leading reference system preserves
the Gram matrix and covers the endpoint `a=1`.

All 28 cited references have unique keys and are used. The bibliographic audit
found no unresolved DOI, invented source, duplicate entry, or metadata conflict.

## Language and presentation

The main article and Supplemental Material were reviewed for repetitive
caveats, audit-list prose, formulaic transitions, vague claims, unnecessary
jargon, and unsupported strengthening. The remaining qualifications occur
where they define the theorem domain or the numerical evidence and are not
repeated defensively.

No systematic AI-style pattern was found. Sentences generally follow a
result-reason-scope structure, transitions are direct, and claims are calibrated
to the evidence. The sources contain no Unicode en dash or em dash. Hyphens and
LaTeX double hyphens that remain serve compound modifiers, mathematical names,
numeric ranges, or author names; they are not ornamental interruptions.

Figures 1--3 have distinct roles: physical model, analytic limiting laws, and
finite-size diagnostics. A fourth phase-diagram figure is not needed because
the unified theorem gives a sufficient sequence criterion rather than a proved
necessary-and-sufficient phase boundary.

Final status: **PASS**.
