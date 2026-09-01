# Final Quantum submission review

Date: 2026-09-01

This conclusion-blind review was performed after the Paper I scope and theorem
set were frozen. It assesses mathematical consistency, exposition,
reproducibility, and submission readiness. Stage 5--6 added no physical model
or headline conclusion. It made an already-used elliptic endpoint expansion
and the maximum-a-posteriori decision rule explicit.

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

The 27-page Supplemental Material is dense but proportionate to the proof
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

The article states the moving-overlap theorem before its supporting
machinery, keeps the sector Minkowski step, Volterra repair, triangular factor,
and normalization conventions in the main proof, and points to named
Supplemental results for detailed estimates. Repeated proof outlines and scope
caveats were reduced. The final Supplemental page contains only the closing
finite-size diagnostics; it remains readable and is not an accidental blank
page.

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

A post-edit heuristic scan classified both TeX sources as `Low` for detected
AI-style patterns, with score 0.265. Its vocabulary-diversity warning reflects
repeated mathematical notation and TeX commands; replacing stable technical
terms with cosmetic synonyms would reduce precision and was not attempted.

An independent audit checked all 28 references against DOI records,
publisher pages, official book records, or arXiv. All 27 DOI records resolve,
and the Helstrom monograph metadata are correct. No invented source, metadata
conflict, missing citation, duplicate key, or unused entry was found. The two
preprints were checked live on 2026-09-01: arXiv:2312.04023 remains at version
2 and arXiv:2602.11846 remains at version 1. Their status should be refreshed
before any later manuscript revision.

## Numerical and release gate

In a clean CPython 3.12.10 environment, the full workspace suite completed 258
tests twice with no failure: 204 passed and 54 documented retired-interface or
out-of-scope tests were skipped on each pass. Between the two passes, the
workflow regenerated all release CSV files and Figures 1, 2, and 3. All four
archived-versus-regenerated scientific data comparisons passed.

The 17-file public Paper I test subset then ran twice with deterministic BLAS
thread settings. Each pass executed 241 tests: 187 passed, 54 documented
retired-interface tests were skipped, and none failed.

The first extracted source candidate ran the same 241 tests twice: 186 passed,
55 documented tests were skipped, and none failed. The additional skip is the
expected extracted-archive path. Its manifest verified file by file. The
arXiv archive compiled in isolation to 18 pages with clean final labels and
references; its page-by-page extracted text matched the frozen article, and
its embedded Supplemental PDF matched the frozen SHA-256 value.

For the 30-case unknown-length SDP grid, the largest raw primal-dual gap was
`6.7818e-9` and the largest safeguarded floating-point diagnostic width was
`1.0243e-8`. For the 24-case no-change grid, the corresponding values were
`2.1485e-9` and `4.3924e-9`. Rechecked equality residuals were below
`1.5e-15`, and rechecked PSD violations were reported as zero at evaluator
resolution. These values are diagnostics, not interval-arithmetic
certificates.

The public environment manifest is regenerated only after the frozen source
and review records are final. The release packager independently rejects
missing, nonportable, or stale manifest entries.

Final submission-review status: **PASS**.
