# Paper I numerical certification

The authoritative numerical report is
[`paper1/NUMERICAL_REPRODUCIBILITY.md`](paper1/NUMERICAL_REPRODUCIBILITY.md).
The public Paper-I evidence consists of:

- 30 certified unknown-length primal/dual SDP cases in
  `certified_sdp_results.csv`;
- 25 full dense one-interval SRM scaling cases through `n=50` in
  `srm_scaling_m1.csv`;
- fixed- and growing-known-length checks in
  `paper1/paper1_fixed_and_growing_srm.csv`;
- 24 fixed-prior no-anomaly SDP cases in
  `paper1/paper1_h0_certified_sdp.csv`;
- a machine-readable verification report and environment/hash manifest under
  `paper1/`.

Paper-II multi-interval tables, forest factorizations, and their dedicated
tests are intentionally not part of this release.
