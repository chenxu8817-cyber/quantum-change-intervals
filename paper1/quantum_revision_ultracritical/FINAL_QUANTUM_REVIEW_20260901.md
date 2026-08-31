# Final Quantum submission review

Date: 2026-09-01

This review was performed after the Paper I scope and theorem set were frozen.
It assesses mathematical consistency, presentation, reproducibility, and
submission readiness. It adds no model, theorem, numerical claim, or citation.

## Overall assessment

No blocking mathematical, bibliographic, numerical, editorial, or
typesetting issue remains in the reviewed snapshot. The article is technically
dense, but its main line is coherent:

1. returning-interval model and exact Gram geometry;
2. fixed and growing known lengths;
3. unknown length at fixed overlap;
4. moving-overlap and hull-dominance regimes;
5. the fixed-prior no-change extension;
6. finite-size floating-point diagnostics.

The Supplemental Material is longer because it contains the complete sector
decomposition, singular-safe dual bounds, local square-root analysis,
continuum matching, adaptive outer construction, quantitative trace laws, and
diagnostic details. Its opening proof map and normalization ledger make the
dependency structure explicit. The length is justified by the number of
independent asymptotic regimes rather than by duplicated exposition.

## Proof architecture and concision

Two proof routes remain necessary, but they now have distinct stated roles.

1. The Toeplitz/Foelner and exceptional-sector route establishes the fixed-
   overlap transfer results.
2. The weighted-hull, sectorwise dual, global-trace, and adaptive-macroblock
   route treats moving overlaps, where the fixed-overlap comparison is not
   uniform.

Within the second route, the local square-root-diagonal argument is presented
as an independent stronger refinement, not as a duplicate derivation of the
common optimum/SRM limit. Removing the global-trace route would lose the
optimum upper squeeze and the explicit rate; removing the local route would
lose the labelwise result and retained-orthogonality input. The appropriate
simplification was therefore to shorten repeated proof outlines in the main
article while retaining complete proofs in the Supplemental Material.

The revision also reduced repeated contribution lists and caveats, divided the
no-change discussion into shorter logical units, normalized terminology, and
replaced checklist-like transitions with direct result-reason-scope prose.
No theorem assumption, conclusion, numerical value, citation, or scientific
qualification was removed in this editing pass.

## Claim consistency

- The abstract, Introduction, unified theorem, Supplemental Material, and
  conclusion use the same hull-dominance assumptions:
  `c_n -> 1`, `c_n < 1` eventually, and `n p_1(c_n) -> infinity`.
- This criterion is stated as sufficient. The article claims neither a
  converse nor a necessary-and-sufficient phase boundary.
- Fixed overlap, moving overlap, compact continuum, and ultra-critical
  statements retain their distinct hypotheses and normalizations. The
  continuum coefficient is consistently `2 lambda^2/pi^4`.
- The uniform prior is over all nonempty intervals. In the no-change model,
  `pi_0` is fixed and the anomaly prior is conditionally uniform.
- Every `H_0` result is identified as joint Bayes detection and exact
  localization.
- The exact `c=1` counterexample for the raw weighted SRM is explicit. The
  maximum-a-posteriori repair is described as classical postprocessing and is
  not used to obscure the counterexample.
- Finite SDPs are described only as residual-checked floating-point
  diagnostics. For sizes above the stated cutoff, the article reports SRM
  values without claiming to have computed `P_opt`.

## Scope and originality

Paper I remains restricted to known pure reference and anomalous states, one
nonempty returning interval, uniform priors, and collective minimum-error
POVMs. Multiple intervals, unknown anomalous states, local or sequential
optima, general LOCC separation, mixed states, correlated outputs, and minimax
priors remain outside this release.

The originality claims are concentrated on exact asymptotic Bayes laws, the
Toeplitz/Foelner and exceptional-sector transfer mechanism, the
singular-safe dual comparison, the moving-overlap hull-dominance theorem, and
asymptotic SRM optimality. The article makes no unconditional priority claim.

Relations to permanent change points, general two-change models,
multi-anomaly detection, quantum edge detection, and classical changed
intervals are stated respectively as boundary cases, structured
subensembles, or geometric analogies. The theorem-level comparison with the
Mohan--Sikora--Upadhyay framework uses the correct parameter embedding and
does not attribute the present asymptotic laws to earlier work.

## Language, references, and formalization scope

The article and Supplemental Material were reviewed for repeated caveats,
vague claims, formulaic transitions, unnecessary jargon, ornamental dashes,
and unsupported strengthening. No systematic AI-style pattern remains. The
qualifications that remain define theorem domains, decision criteria, or the
status of numerical evidence.

An independent reference audit verified all 28 entries against primary or
official sources. Twenty-seven DOI records resolve correctly; the remaining
Helstrom monograph was verified through the publisher. No invented source,
metadata conflict, missing citation key, duplicate key, or unused entry was
found.

A separate Lean development was used as internal proof-audit infrastructure
for selected finite-dimensional Gram, PGM, norm, growing-known-length, and
no-change components. The fixed-overlap unknown-length closure and the new
critical-scaling phase diagram are not claimed to be formally verified. The
submission therefore makes no statement that all proofs are Lean-certified.

## Numerical and release gate

The clean Python 3.12.10 regression completed 254 tests with no failure; 54
documented tests for retired interfaces were skipped. All release CSV files
and Figures 1--3 were regenerated from the current code. The small-instance
SDP tables retained zero feasible PSD violation at the evaluator's resolution
and equality residuals below `1.5e-15`; these values remain floating-point
diagnostics rather than interval-arithmetic enclosures.

Final status: **PASS**.
