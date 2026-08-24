from __future__ import annotations

import unittest

import numpy as np

from paper1_numerics import fixed_and_growing_rows, h0_gram
from sdp_certification import certify_minimum_error


class Paper1NumericalIntegrationTests(unittest.TestCase):
    def test_fixed_and_growing_rows_have_expected_schedules(self) -> None:
        rows = fixed_and_growing_rows([0.6])
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {str(row["schedule"]) for row in rows},
            {"fixed_i2", "sqrt_growth", "balanced_growth"},
        )
        self.assertTrue(all(np.isfinite(float(row["srm"])) for row in rows))

    def test_h0_orthogonal_endpoint_is_perfect(self) -> None:
        gram = h0_gram(candidate_count=4, length=2, overlap=0.0)
        priors = np.array([0.3] + [0.7 / 4] * 4)
        certificate = certify_minimum_error(gram, priors=priors)
        self.assertAlmostEqual(float(certificate["primal_objective"]), 1.0, places=7)

    def test_h0_identical_endpoint_guesses_largest_prior(self) -> None:
        gram = h0_gram(candidate_count=4, length=2, overlap=1.0)
        priors = np.array([0.3] + [0.7 / 4] * 4)
        certificate = certify_minimum_error(gram, priors=priors)
        self.assertAlmostEqual(float(certificate["dual_objective"]), 0.3, places=7)


if __name__ == "__main__":
    unittest.main()

