# Paper I release manifest

Release identifier: `v1.5.0-paper1`

Freeze date: 2026-09-01

Article: *Quantum Change Intervals: Exact Asymptotic Localization with
Collective Measurements*

## Scientific scope

This release concerns known pure reference and anomalous states, one nonempty
returning interval, uniform label priors, and collective minimum-error POVMs.
When a no-change hypothesis is present, its prior `pi_0` is fixed and the
anomalous labels have the stated conditional uniform prior. Multiple
intervals, unknown anomalous states, local or sequential optima, general LOCC
separations, mixed states, correlated outputs, and minimax priors are outside
Paper I.

The theorem set is frozen. This release changes exposition and release
engineering but does not add a model, theorem, numerical claim, citation, or
stronger conclusion relative to `v1.4.0-paper1`.

## Authoritative sources

| Relative path | SHA-256 |
|---|---|
| `paper1/quantum_revision_ultracritical/main.tex` | `FAACCE86875A07537671B92BF1BEE940B75CF8DAE78DC10F77C1003616B2B5AE` |
| `paper1/quantum_revision_ultracritical/content.tex` | `69DD78F29FFF7C4DD7200AD4C65D2104AA417ED6862FC225821ABA4189C94346` |
| `paper1/quantum_revision_ultracritical/supplement.tex` | `BA17925EE931A5A5056372B72DD4694C9296468A8B4D5BE2F5117EF0EB0BA77E` |
| `paper1/quantum_revision_ultracritical/supplement_content.tex` | `2D2510EF7DAAC9258952B3A889C9ECB1AA16843C4CF4040307FE12F21113A801` |
| `paper1/quantum_revision_ultracritical/references.bib` | `FAA179CC5CE32CE80EFABDF7E97C3103EE3E434A8C04989ADD72E84FA1B34930` |

The article build uses the bibliography output generated in the same clean
build:

| Relative build path | Bytes | SHA-256 |
|---|---:|---|
| `paper1/quantum_revision_ultracritical/build-v1.5-main-freeze/main.bbl` | 7387 | `685F2AB321CD7AD51E4E2034136B4E4E0386C5E1C6F72A1E004C413AD0107E2C` |

## Frozen PDFs

| Artifact | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| Main article | 18 | 389825 | `4FBC92F90ED5B96A6B1AE205D3610587757115ABF525B6EAEB63ED822BF042AE` |
| Supplemental Material | 26 | 231660 | `9C5FEE77C081014CA9238234580F06E3441492FCBECFFD23CED0D498552BF2E3` |

Both PDFs were built with Tectonic 0.17.0. Their final logs contain no LaTeX
error, undefined citation, undefined reference, multiply defined label,
overfull box, or fatal warning. Fonts, vector figures, metadata, and all 44
rendered pages are covered by
`paper1/quantum_revision_ultracritical/FINAL_PDF_QA_20260901.md`.

## Numerical release tables

| Relative path | Data rows | SHA-256 |
|---|---:|---|
| `paper1/paper1_fixed_and_growing_srm.csv` | 36 | `E428832DBCA3B5A93EB62D00707EC4B8E0BFC4505D6510275617506931585911` |
| `paper1/paper1_h0_certified_sdp.csv` | 24 | `CF1E49F703E47BDEC7B1099054D9528F118EAEC47A2E9707E42C540E18E298EB` |
| `certified_sdp_results.csv` | 30 | `9B7C2F0FFC954C7C631FEE149D7F824CBF9FDD8410FF4D78E6F5F999F973D656` |
| `srm_scaling_m1.csv` | 25 | `17D5803156E89DD405D15875554FDFA485C3B17462A444ECA0C4D217123287BE` |
| `proofs/weighted_hull_diagnostics.csv` | 24 | `B9C5B773D6548CE87B1808C69ECB0CFF83502C6D1F5C76BE04DB7916B8C0884B` |
| `paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv` | 12 | `74B0E2F86E44E7D7DAFDF0856A0E4D8BFF74C07860C24C370643B31D170822F5` |

All four archived-versus-regenerated scientific comparisons passed. Solver
timing and iteration fields are excluded from equality comparisons. The SDP
quantities are residual-checked floating-point diagnostics, not formally
certified finite-dimensional optima or interval-arithmetic enclosures.

For the 30-case unknown-length grid, the maximum raw primal-dual gap was
`6.781719519e-09`, the maximum safeguarded diagnostic width was
`1.024215324e-08`, the maximum rechecked equality residual was
`1.407055324e-15`, and no PSD violation was reported at evaluator resolution.
For the 24-case no-change grid, the corresponding values were
`2.148437872e-09`, `4.392378350e-09`, and `4.107450099e-16`, with no reported
PSD violation.

## Figures

| Relative path | SHA-256 |
|---|---|
| `paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.pdf` | `C83DABC865F377C08DDB3AF4643F67F7F815D1A798BE093E1062D5B41343FF2B` |
| `paper1/quantum_revision_ultracritical/figures/figure2_analytic_limits.pdf` | `D0734773A6F17D74E30377C39405A0E34DAD9DB59C12F43C3FFF6EA642198233` |
| `paper1/quantum_revision_ultracritical/figures/figure3_finite_size.pdf` | `FD29DBEBC13FF0EA9D8BC0ED27F509D6C497089449EAFF22793D0E7FDACB1776` |

The three article figures are vector based. The release also supplies 300 dpi
PNG derivatives; Figure 1 additionally has a 600 dpi TIFF derivative.

## Reproduction environment and tests

- CPython: 3.12.10 in a clean virtual environment
- Random seed: 1729
- BLAS thread variables: fixed to one during release reproduction
- Public release candidate: 241 tests run twice; each pass had 186 passes, 55
  documented retired-interface skips, and no failure
- Extracted public source archive: the same 241 tests ran twice with the same
  186-pass, 55-skip result; all four scientific data comparisons passed and
  the regenerated manifest verified file by file
- Extracted arXiv archive: clean 18-page Tectonic build with no undefined
  citation or reference, multiply defined label, overfull box, or fatal error;
  its embedded Supplemental Material has SHA-256
  `9C5FEE77C081014CA9238234580F06E3441492FCBECFFD23CED0D498552BF2E3`
- CSV files and Figures 1--3 regenerated from the current public code
- Manifest generated after the second test pass and verified file by file
- Release packager configured to reject missing, nonportable, or stale
  manifest entries
- Generated CSV and JSON files use deterministic LF line endings on every
  supported platform

Package versions, solver availability, BLAS configuration, portable execution
metadata, and public-input hashes are recorded in
`paper1/reproduction_manifest.json`. The public manifest contains no absolute
host path and excludes local environments, TeX build directories, rendered QA
pages, and Paper II or Paper III material.

## Independent review records

The final conclusion-blind scientific review, PDF inspection, semantic-diff
gate, post-polish source gate, and reference audit are recorded in:

- `paper1/quantum_revision_ultracritical/FINAL_QUANTUM_REVIEW_20260901.md`;
- `paper1/quantum_revision_ultracritical/FINAL_PDF_QA_20260901.md`;
- `paper1/release_audit/stage5_semantic_diff_audit.md`;
- `paper1/release_audit/post_polish_source_gate.md`; and
- `paper1/release_audit/final_reference_gate.md`.

The article retains both global proof routes because they cover different
asymptotic domains: Toeplitz/Følner transfer treats fixed overlap, whereas
the weighted-hull and sectorwise analysis treats moving overlap. The local
square-root-diagonal result is a downstream refinement and remains in the
Supplemental Material rather than the main proof. The reference audit verified
all 28 cited works against DOI, publisher, official book, or arXiv records.

## Outer release artifacts

`paper1/build_release.py` creates, from an explicit allowlist, the arXiv source
archive, Quantum article PDF, Supplemental Material PDF, cover letter, public
source archive, GitHub release archive, inventory, and `SHA256SUMS`. The source
and GitHub archives are intentional destination-named byte-for-byte aliases.
Artifact hashes are stored in the generated inventory and checksum file rather
than copied into this tracked document, which avoids a circular hash
dependency.

The annotated Git tag `v1.5.0-paper1` identifies the frozen source commit. The
tag-to-commit mapping and released binary assets are recorded by GitHub rather
than embedded here, avoiding a self-referential commit identifier.
