# v1.5.0-paper1

This release freezes the polished Paper I article and Supplemental Material
for the arXiv update and submission to *Quantum*. The scientific scope and
formal theorem set are unchanged from `v1.4.0-paper1`.

Changes since `v1.4.0-paper1` include:

- a shorter main presentation that states the moving-overlap theorem before
  its supporting machinery and directs detailed boundary-layer estimates to
  named Supplemental results;
- a clearer proof map distinguishing the two complementary global asymptotic
  arguments from the downstream local SRM refinement;
- synchronized hypotheses, priors, endpoint cases, and numerical caveats
  across the abstract, introduction, theorem statements, captions, discussion,
  and conclusion;
- removal of repetitive defensive wording while retaining every scientific
  limitation needed to interpret the results;
- a fail-closed manifest freshness check that runs after all generators and
  tests and is repeated by the release packager; and
- regenerated numerical tables, figures, PDFs, submission archives, and
  independent scientific, bibliographic, language, and PDF audits.

# v1.4.0-paper1

This release freezes the submission-ready Paper I article and Supplemental
Material after the final mathematical, bibliographic, numerical, editorial,
and PDF audits. It retains the unified moving-overlap theorem introduced in
the preceding revision and aligns every public artifact with the same source
snapshot.

Changes since `v1.3.0-paper1` include:

- a shorter and clearer presentation of the two complementary proof routes,
  without changing their assumptions or conclusions;
- synchronized fixed-overlap, moving-overlap, no-change, and endpoint wording
  across the abstract, theorem statements, discussion, and conclusion;
- fully vector Figures 1, 2, and 3, including the Figure 3(c) heat map and
  color scale;
- a portable Paper I reproduction manifest with current release metadata and
  no host-specific absolute paths;
- hardened deterministic packaging for arXiv, Quantum, public source, and
  GitHub release artifacts; and
- final PDF, reference, claim, and extracted-package reproduction checks.

# v1.3.0-paper1

This reproducible release accompanies the revised manuscript *Quantum Change
Intervals: Exact Asymptotic Localization with Collective Measurements*.
It closes the moving-overlap analysis with a unified sufficient condition:
for the uniform prior on all nonempty intervals, if (c_n\to1), (c_n<1)
eventually, and (n p_1(c_n)\to\infty), then the trace benchmark, SRM, and
collective optimum all satisfy (P_X(G_n(c_n))\sim p_1(c_n)^2).  The result
joins the inner, compact-continuum, and moving-outer regimes without assuming
monotonicity.

The release also includes:

- the final 20-page article and separate 23-page Supplemental Material;
- the exact weighted-hull identity, reusable block-tail certificate, and the
  global and local proof routes used for the unified theorem;
- the singular-safe dual certificate and the no-change weighted-PGM results,
  including the exact (c=1) counterexample and MAP repair;
- all numerical tables and Figures 1--3 regenerated under CPython 3.12.10;
- residual-checked floating-point SDP diagnostics, without claiming formal
  finite-dimensional enclosures; and
- deterministic arXiv, Quantum, source, and GitHub release packaging with
  SHA-256 inventories.

# v1.2.0-paper1

This submission-ready manuscript release adds the exceptional-sector Gram
transfer theorem and its Følner comparison corollary, clarifies the joint
no-change priors and the scope of the Bayes-optimal statements, and strengthens
the theorem-to-theorem comparison with prior quantum change-point,
multi-anomaly, edge-detection, and certified-answer work.

It also includes:

- synchronized Quantum-class sources and the verified 15-page PDF;
- revised Figures 1--3, including an explicit nonorthogonal-overlap cue,
  prior-specific labels without comparison shading, and separated curve labels;
- corrected safeguarded primal normalization and explicit numerical scope for
  \(n>7\);
- the regenerated bibliography, proof audit, reference data, environment
  manifest, and reproducibility package; and
- an AI-assistance disclosure integrated into the author-contribution
  statement.

# v1.1.0-paper1

This intermediate source tag introduced the exceptional-sector transfer
framework and the expanded prior-work comparison. It was not published as a
separate GitHub Release; the submission-ready revision is
`v1.2.0-paper1`.

# v1.0.0-paper1

This immutable release accompanies the manuscript *Quantum Change Intervals:
Exact Asymptotic Localization with Collective Measurements*.

It includes:

- synchronized Quantum-class LaTeX sources and the verified 14-page PDF;
- publication figures and the Python figure-generation source;
- fixed-length, growing-length, unknown-length, and no-anomaly numerical data;
- primal/dual SDP and full dense physical-Gram SRM implementations;
- the pinned CPython 3.12 environment, unit tests, verification scripts, and
  reproduction manifest; and
- MIT license and citation metadata for Xu Chen and Xue Ma.

The Paper-I release is intentionally restricted to one nonempty contiguous
change interval with a known pure anomalous state and collective measurements.
Multiple disjoint intervals and unknown anomalous states are outside this
release.
