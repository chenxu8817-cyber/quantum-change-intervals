# Paper I numerical verification

The authoritative numerical report is
[`paper1/NUMERICAL_REPRODUCIBILITY.md`](paper1/NUMERICAL_REPRODUCIBILITY.md).
The public Paper-I evidence consists of:

- 30 safeguarded unknown-length primal/dual SDP cases in
  `certified_sdp_results.csv` (legacy filename);
- 25 full dense one-interval SRM scaling cases through `n=50` in
  `srm_scaling_m1.csv`;
- fixed- and growing-known-length checks in
  `paper1/paper1_fixed_and_growing_srm.csv`;
- 24 fixed-prior no-change SDP cases in
  `paper1/paper1_h0_certified_sdp.csv`;
- weighted-hull continuum and outer-regime diagnostics in
  `proofs/weighted_hull_diagnostics.csv`;
- critical-overlap dense-Gram diagnostics in
  `paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv`;
- a machine-readable verification report and environment/hash manifest under
  `paper1/`.

Paper-II multi-interval tables, forest factorizations, and their dedicated
tests are intentionally not part of this release.

All bracket endpoints and residual checks use conventional IEEE double
precision. They are safeguarded floating-point diagnostics, not validated
interval-arithmetic or exact-rational certificates. Historical filenames and
APIs containing `certified` or `certify` are retained for reproducibility.
