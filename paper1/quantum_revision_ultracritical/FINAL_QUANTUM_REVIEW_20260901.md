# Final Quantum submission review

Date: 2026-09-01

This conclusion-blind review was performed after the Paper I scope and theorem
set were frozen. It assesses mathematical consistency, exposition,
reproducibility, and submission readiness. It adds no model, theorem,
numerical claim, or citation.

## Overall assessment

No blocking mathematical, bibliographic, numerical, editorial, or
typesetting issue remains in the reviewed snapshot. The 18-page article has a
clear sequence:

1. returning-interval model and exact Gram geometry;
2. fixed and growing known lengths;
3. unknown length at fixed overlap;
4. moving-overlap and hull-dominance regimes;
5. fixed-prior no-change extension; and
6. finite-size floating-point diagnostics.

The 26-page Supplemental Material is dense but proportionate to the proof
load. It contains the complete Følner and exceptional-sector arguments,
excitation compression, singular-safe dual estimates, continuum matching,
adaptive outer construction, local SRM refinement, and numerical details. Its
opening proof map and normalization table make the dependencies explicit.

## Proof architecture and concision

The two global proof routes have different domains and should both remain.
The Toeplitz/Følner route proves fixed-overlap transfer laws. The weighted-hull
and sectorwise route treats moving overlaps, for which the fixed-overlap
comparison is not uniform. Removing either route would leave a headline
regime unsupported.

The local square-root-diagonal law is a downstream strengthening, not an
independent proof of the common optimum and SRM limit. It reuses retained
orthogonality, incidence-trace, and odd-cosine estimates and is not an input to
the global trace and dual squeeze. It could be omitted without changing the
headline common-limit theorem, but retaining it in the Supplemental Material
adds a genuine labelwise statement and clarifies why finite-size local
deviations can coexist with the global law.

The article now states the moving-overlap theorem before its supporting
machinery, keeps the sector Minkowski step, Volterra repair, triangular factor,
and normalization conventions in the main proof, and points to named
Supplemental results for detailed estimates. Repeated proof outlines and scope
caveats were removed. The final Supplemental page was consolidated during PDF
inspection, reducing the file from 27 to 26 pages.

## Claim consistency

- The abstract, introduction, theorem statements, Supplemental Material, and
  conclusion use the same hull-dominance assumptions:
  `c_n -> 1`, `c_n < 1` eventually, and `n p_1(c_n) -> infinity`.
- The hull-dominance condition is sufficient. No converse or
  necessary-and-sufficient phase boundary is claimed.
- Fixed overlap, compact continuum, moving outer, and ultracritical results
  retain their distinct hypotheses, priors, and normalizations.
- The interval prior is uniform over all nonempty intervals. When `H_0` is
  included, `pi_0` is fixed and the anomaly prior is conditionally uniform.
- Every `H_0` result is identified as joint Bayes detection and exact
  localization.
- The exact `c=1` counterexample for the raw weighted SRM remains explicit.
  Maximum-a-posteriori relabeling is identified as classical postprocessing.
- Finite SDPs are described as residual-checked floating-point diagnostics,
  not formal finite-dimensional optima or rigorous enclosures. For `n>7`, the
  article reports SRM calculations without claiming to compute `P_opt`.

## Scope and originality

Paper I remains restricted to known pure reference and anomalous states, one
nonempty returning interval, uniform priors, and collective minimum-error
POVMs. Multiple intervals, unknown anomalous states, local or sequential
optima, general LOCC separation, mixed states, correlated outputs, and minimax
priors remain outside this release.

The originality claims are concentrated on exact asymptotic Bayes laws, the
Toeplitz/Følner and exceptional-sector transfer mechanism, singular-safe dual
control, the moving-overlap hull-dominance theorem, and asymptotic SRM
optimality. The article makes no unconditional priority claim.

Permanent change points, general two-change models, multi-anomaly detection,
quantum edge detection, and classical changed intervals are described as
boundary cases, structured subensembles, or geometric analogues. The
Mohan--Sikora--Upadhyay parameter embedding and the distinctions between
minimum-error, unambiguous, local, and collective tasks were checked against
the cited papers.

## Language and references

The article and Supplemental Material were reviewed for repeated caveats,
vague claims, formulaic transitions, unnecessary jargon, ornamental dashes,
and unsupported strengthening. No systematic AI-style pattern remains.
Technical compounds such as `minimum-error`, `square-root`, and
`fixed-overlap` retain necessary hyphens. The remaining negative statements
define theorem domains, decision criteria, or the status of numerical
evidence.

An independent audit checked all 28 references against DOI records,
publisher pages, official book records, or arXiv. All 27 DOI records resolve,
and the Helstrom monograph metadata are correct. No invented source, metadata
conflict, missing citation, duplicate key, or unused entry was found. The two
preprints should be checked again before any later manuscript revision because
their version status may change.

## Numerical and release gate

In the clean public release candidate, CPython 3.12.10 completed 241 tests
twice with no failure: 186 passed and 55 documented retired-interface tests
were skipped on each pass. The workflow regenerated all release CSV files and
Figures 1, 2, and 3, and all four archived-versus-regenerated scientific data
comparisons passed.

For the 30-case unknown-length SDP grid, the largest raw primal-dual gap was
`6.7818e-9` and the largest safeguarded floating-point diagnostic width was
`1.0243e-8`. For the 24-case no-change grid, the corresponding values were
`2.1485e-9` and `4.3924e-9`. Rechecked equality residuals were below
`1.5e-15`, and rechecked PSD violations were reported as zero at evaluator
resolution. These values are diagnostics, not interval-arithmetic
certificates.

The environment manifest was generated after the second test pass and then
verified file by file. The release packager independently rejects missing,
nonportable, or stale manifest entries.

Final submission-review status: **PASS**.
