from __future__ import annotations

import unittest

import numpy as np

from paper1_analytics import (
    fixed_length_circulant_gram,
    fixed_length_gram,
    fixed_length_limit_quadrature,
    fixed_length_symbol,
    i1_finite_optimum,
    i2_elliptic_limit,
    single_change_gram,
)
from quantum_interval_numerics import srm_quantities


class FixedLengthAnalyticTests(unittest.TestCase):
    def test_exact_long_block_regime_equals_single_change_gram(self) -> None:
        for candidate_count in range(2, 9):
            for length in (candidate_count - 1, candidate_count + 3):
                np.testing.assert_allclose(
                    fixed_length_gram(candidate_count, length, 0.73),
                    single_change_gram(candidate_count, 0.73),
                )

    def test_i1_formula_equals_direct_srm(self) -> None:
        for candidate_count in (1, 2, 5, 11):
            for r in (0.0, 0.2, 0.7, 1.0):
                gram = fixed_length_gram(candidate_count, 1, r)
                # At r=1 the Gram matrix is rank one, so dense Hermitian
                # eigensolvers can leave an O(1e-9) square-root residue in
                # the numerical nullspace.  The analytic identity is exact.
                self.assertAlmostEqual(
                    i1_finite_optimum(candidate_count, r),
                    srm_quantities(gram)["srm"],
                    delta=1e-8 if r == 1.0 else 1e-11,
                )

    def test_i2_elliptic_form_matches_toeplitz_integral(self) -> None:
        for r in (0.0, 0.1, 0.42, 0.8, 0.99):
            self.assertAlmostEqual(
                i2_elliptic_limit(r),
                fixed_length_limit_quadrature(2, r),
                places=10,
            )

    def test_symbol_has_claimed_spectral_floor(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 2001)
        for length in (1, 2, 3, 6):
            for r in (0.1, 0.6, 0.95):
                values = fixed_length_symbol(theta, length, r)
                self.assertGreaterEqual(
                    float(values.min()) + 1e-12,
                    (1.0 - r) ** length,
                )

    def test_circulant_comparison_is_a_gram_matrix(self) -> None:
        for length in (1, 2, 4):
            gram = fixed_length_circulant_gram(4 * length + 1, length, 0.61)
            np.testing.assert_allclose(gram, gram.T)
            np.testing.assert_allclose(np.diag(gram), 1.0)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -1e-11)


if __name__ == "__main__":
    unittest.main()
