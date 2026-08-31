# Paper I release manifest

Release identifier: `v1.4.0-paper1`  
Freeze date: 2026-09-01  
Article: *Quantum Change Intervals: Exact Asymptotic Localization with
Collective Measurements*

## Scientific scope

This release concerns known pure reference and anomalous states, one nonempty
returning interval, uniform label priors, and collective minimum-error POVMs.
When a no-change hypothesis is present, its prior `pi_0` is fixed and the
anomalous labels retain their stated conditional uniform prior. Multiple
intervals, unknown anomalous states, local or sequential optima, general LOCC
separations, mixed states, correlated outputs, and minimax priors are outside
Paper I.

## Authoritative sources

| Relative path | SHA-256 |
|---|---|
| `paper1/quantum_revision_ultracritical/main.tex` | `73BAF66D7BC4DA3D157541F9401F77394C2F2E5E19B5042FE739D2FBEA474595` |
| `paper1/quantum_revision_ultracritical/content.tex` | `4BDE7FABB18A9C681DDBA57A526BEEE6018CB1DA75ED881E0A8785F203FC0907` |
| `paper1/quantum_revision_ultracritical/supplement.tex` | `BA17925EE931A5A5056372B72DD4694C9296468A8B4D5BE2F5117EF0EB0BA77E` |
| `paper1/quantum_revision_ultracritical/supplement_content.tex` | `107845FB9205A9CC965B40A8D8B8C59F46A5D27F9FC240DDF79A5D0174544A84` |
| `paper1/quantum_revision_ultracritical/references.bib` | `FAA179CC5CE32CE80EFABDF7E97C3103EE3E434A8C04989ADD72E84FA1B34930` |

The article build uses the bibliography output generated in the same clean
build:

| Relative build path | SHA-256 |
|---|---|
| `paper1/build-task12-main-20260901-final13/main.bbl` | `471D4C1B01BACD530A4C401F48203E4C298619ECCE1B53CBD3934D7EC1EFA524` |

## Frozen PDFs

| Artifact | Pages | SHA-256 |
|---|---:|---|
| Main article | 20 | `909DBC6AE2C2B8BCC892FBA1707B357077D2E3A4DEE5767C77AE9ABD897682B7` |
| Supplemental Material | 23 | `3987A16E57C2DBB52A5F1AD093746EA77325F0D86609B555019E37DB56B57C02` |

The PDFs were built with Tectonic 0.17.0. Compilation, labels, fonts, figures,
metadata, and all 43 rendered pages are covered by
`paper1/quantum_revision_ultracritical/FINAL_PDF_QA_20260901.md`.

## Numerical release tables

| Relative path | Rows | SHA-256 |
|---|---:|---|
| `paper1/paper1_fixed_and_growing_srm.csv` | 36 | `615FDBFF6A74F0B1B7CF13ABF3D8E3FB454A654C912E9EDC8E9BD66F9D7F2E8F` |
| `paper1/paper1_h0_certified_sdp.csv` | 24 | `BEFE2561CF6785C189D31C69DA3B0AD1F653BBCBD97230203C1A532819DD0EAA` |
| `certified_sdp_results.csv` | 30 | `5554006759C323B44BE7A8C2F63BA0767DEA870B35BBA8EE4F757428C269344F` |
| `srm_scaling_m1.csv` | 25 | `47CC6D80C75DB54B5FA8C7AB878447C51645933EBC01F26A922ADC12BAA8754D` |
| `proofs/weighted_hull_diagnostics.csv` | 24 | `07BA116EC35C827CA0A0BE3296F615A9E5BD45AD91D8FE4F96818DBEBE330159` |
| `paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv` | 12 | `74B0E2F86E44E7D7DAFDF0856A0E4D8BFF74C07860C24C370643B31D170822F5` |

The SDP quantities are residual-checked floating-point diagnostics. They are
not formally certified finite-dimensional optima or interval-arithmetic
enclosures.

## Figures

| Relative path | SHA-256 |
|---|---|
| `paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.pdf` | `C83DABC865F377C08DDB3AF4643F67F7F815D1A798BE093E1062D5B41343FF2B` |
| `paper1/quantum_revision_ultracritical/figures/figure2_analytic_limits.pdf` | `D0734773A6F17D74E30377C39405A0E34DAD9DB59C12F43C3FFF6EA642198233` |
| `paper1/quantum_revision_ultracritical/figures/figure3_finite_size.pdf` | `FD29DBEBC13FF0EA9D8BC0ED27F509D6C497089449EAFF22793D0E7FDACB1776` |

All three frozen article figures are vector based. The Figure 3(c) heat map and
color scale contain no raster image object.

## Reproduction environment and tests

- CPython: 3.12.10
- Random seed: 1729
- BLAS thread variables: fixed to one during release reproduction
- Clean full-workspace regression: 254 tests discovered, 200 passes, 54
  documented retired-interface skips, and no failure
- Regenerated tables: identical under the frozen comparison rules; solver
  timing and iteration columns are intentionally excluded from equality checks
- Unknown-length SDP diagnostic grid: maximum raw primal-dual gap
  `6.7817e-09`, maximum repaired diagnostic gap `1.0242e-08`, maximum equality
  residual `1.407e-15`, and zero reported PSD violation
- No-change SDP diagnostic grid: maximum raw primal-dual gap `2.1484e-09`,
  maximum repaired diagnostic gap `4.3924e-09`, maximum equality residual
  `4.107e-16`, and zero reported PSD violation
- Extracted public-source regression: 237 tests discovered, 182 passes, 55
  documented scope or retired-interface skips, and no failure
- Extracted public-source post-generation gate: 119 tests discovered, 65
  passes, 54 documented retired-interface skips, and no failure
- Extracted arXiv source: clean Tectonic 0.17.0 build to a 20-page article,
  with no undefined citation, undefined reference, multiply defined label, or
  overfull box in the final log

Package versions, solver availability, BLAS configuration, portable execution
metadata, and hashes of public release inputs are recorded in
`paper1/reproduction_manifest.json`. The Paper I archives exclude the legacy
root-level all-project manifest.

## Independent review records

The final submission and PDF gates are recorded in:

- `paper1/quantum_revision_ultracritical/FINAL_QUANTUM_REVIEW_20260901.md`;
- `paper1/quantum_revision_ultracritical/FINAL_PDF_QA_20260901.md`.

The literature audit verified all 28 references against primary or official
sources. The submission does not claim complete Lean formal verification; the
separate Lean development is internal proof-audit infrastructure with partial
coverage.

## Outer release artifacts

`paper1/build_release.py` generates the arXiv source archive, Quantum article
PDF, Supplemental Material PDF, cover letter, public source archive, and
GitHub release archive from an explicit allowlist. The source and GitHub
archives are intentional destination-named byte-for-byte aliases. Artifact
sizes and SHA-256 values are recorded in the generated inventory and
`SHA256SUMS` rather than written back here, avoiding a circular hash
dependency.

The immutable Git tag `v1.4.0-paper1` identifies the released source commit.
The tag-to-commit mapping is recorded by GitHub rather than in this tracked
file, avoiding a self-referential commit identifier.
