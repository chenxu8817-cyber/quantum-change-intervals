# Proof-closure audit for Paper I

This note records the claim boundary and the proof obligations that were closed
before numerical certification. It is an internal submission checklist, not a
substitute for the proofs in `quantum_submission/content.tex`.

## Scope frozen for this paper

The paper treats one contiguous returning interval
\(|0\rangle^{\otimes(a-1)}|\psi\rangle^{\otimes i}
|0\rangle^{\otimes(n-a-i+1)}\), a known pure \(|\psi\rangle\), exact
localization, arbitrary collective POVMs, uniform priors over the stated
candidate family, and the explicitly stated fixed no-change priors. It does
not claim results for an unknown anomaly state,
restricted/local measurements, a number of intervals growing with \(n\), or
general multi-anomaly patterns.

## Closed proof obligations

| Item | Closure in the manuscript |
|---|---|
| Complex overlap | A hypothesis-dependent phase is removed by diagonal-unitary Gram conjugation; all performance quantities depend only on \(c=|\langle0|\psi\rangle|\). |
| Gram kernel | The symmetric-difference formula \(c^{|I\triangle J|}\) is derived before every specialization. |
| Fixed-length spectral gap | The unique fully-excited tensor component gives \(G_{N,i}\succeq(1-r)^i I\), not merely a heuristic symbol bound. |
| Circulant approximation | The wrap-around discrepancy is written as an explicit sum of rank-\(d\) partial permutations between the upper-right and lower-left corner blocks, yielding both the trace-norm and rank bounds. |
| Circulant optimality | A Yuen--Kennedy--Lax dual certificate proves exact finite-size SRM optimality for the cyclic model. |
| Rate transfer | Square-root trace Holder continuity plus the corner trace-norm bound transfers cyclic success probabilities to the original Toeplitz family. |
| \(i=1\) | Constant-overlap symmetry and the dual certificate give the finite-\(N\) exact optimum. |
| \(i=2\) | The trigonometric symbol is reduced explicitly to a complete elliptic integral, with parameter convention stated. |
| Long known interval | A rank-one common component is separated, the residual is compared with the one-change Toeplitz matrix, and the square-root trace is squeezed with explicit rank-one bounds. |
| Exact regime | If \(i\ge N-1\), \(G_{N,i}=Q_N\) entrywise; no approximation is invoked. |
| One-change benchmark | Its Toeplitz symbol and \(p_1\) limit are stated as a proposition, so the bridge theorem is not circular. |
| Residual Toeplitz family | A formal corollary identifies the normalized residual Gram symbol and proves its common optimal/SRM limit. |
| Følner functional calculus | The proof uses the correct \(sR\) interior for a degree-\(s\), range-\(R\) polynomial and closes a single uniform \(\varepsilon\)-chain. |
| Unknown length | Short intervals are truncated at vanishing prior mass; the triangular domains are proved Følner by an explicit three-boundary-strip estimate before the local square-root diagonal lemma is applied. |
| Minimum-error upper bound | The normalized Gram-matrix upper bound is stated explicitly as Lemma 5.2 and attributed to Theorem 1 of Ref. [1]. |
| Exceptional-sector transfer | A general theorem combines vanishing deleted prior mass, operator-norm comparison of the retained Gram family, bounded comparison spectra, and local square-root diagonal convergence to transfer both the optimum and SRM limit to the complete ensemble. |
| Følner comparison transfer | The general theorem is specialized to compressions of positive \(\ell^1(\mathbb Z^d)\) convolution kernels on arbitrary finite Følner sequences. |
| Full-family SRM | The interlacing argument retains the required \(|\mathcal B_n|/M_n\) factor and is closed by the optimal-success squeeze. |
| Fixed \(H_0\) prior | The complete joint prior is defined, a vacuum-extension argument separates the prior contribution and bounds its remaining effect through the residual ensemble, and \(r=1\) is treated exactly as largest-prior guessing. Only the joint Bayes optimum is claimed for the augmented ensemble. |
| Unknown-length \(H_0\) prior | The complete joint prior and the full-family, retained-family, and residual conditional optima are defined separately before the detection-localization squeeze. Only the joint Bayes optimum is claimed for the augmented ensemble. |
| Stability | Optimal and SRM success probabilities are controlled under operator-norm Gram perturbations through Hölder continuity of the positive square root. |
| Endpoints | \(c=0\), \(c=1\), and \(r=1\) are handled directly; formulas using a spectral gap are restricted to the open interval. |

## Claims deliberately not promoted

- No “first study of two quantum change points” or “first quantum interval”
  language is used.
- The \(p_1(c)^2\) law is explained by the two endpoint coordinates, but the
  dimension language is not used as a proof.
- Finite-size SDP data corroborate the analysis; they are not presented as
  evidence for an asymptotic theorem.
- Multi-interval \(p_1\)-power laws are reserved for Paper II and do not appear
  among this paper's claims.

## Submission gate

The author, affiliation, funding, conflict-of-interest, and repository fields
have been supplied. Before submission, verify them, approve the public-release
license, add the archival identifier after deposition, and check every
bibliographic entry against the publisher record. The target-journal LaTeX
conversion must not change theorem hypotheses.
