from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from quantum_interval_numerics import (  # noqa: E402
    endpoint_distance,
    exact_interval_gram,
    incidence_matrix,
    interval_candidates,
    interval_mask,
    p1,
    srm_quantities,
    symmetric_difference_size,
    validate_endpoint_dichotomy,
)


class CandidateAndGramTests(unittest.TestCase):
    def test_candidate_count_is_binomial(self) -> None:
        for interval_count in range(1, 4):
            for n in range(2 * interval_count - 1, 2 * interval_count + 4):
                self.assertEqual(
                    len(interval_candidates(n, interval_count)),
                    math.comb(n + 1, 2 * interval_count),
                )

    def test_minimal_domain_contains_one_candidate(self) -> None:
        for interval_count in range(1, 5):
            n = 2 * interval_count - 1
            self.assertEqual(len(interval_candidates(n, interval_count)), 1)
            self.assertEqual(interval_candidates(n - 1, interval_count), [])

    def test_mask_incidence_and_direct_symmetric_difference_agree(self) -> None:
        n = 9
        points = interval_candidates(n, 3)
        incidence = incidence_matrix(n, points)
        for row in range(0, len(points), 17):
            for col in range(0, len(points), 19):
                mask_distance = (
                    interval_mask(points[row]) ^ interval_mask(points[col])
                ).bit_count()
                incidence_distance = int(
                    np.abs(incidence[row] - incidence[col]).sum()
                )
                direct_distance = symmetric_difference_size(
                    points[row], points[col]
                )
                self.assertEqual(mask_distance, incidence_distance)
                self.assertEqual(mask_distance, direct_distance)

    def test_gram_has_expected_endpoints_and_is_psd(self) -> None:
        identity, points = exact_interval_gram(7, 2, 0.0)
        ones, _ = exact_interval_gram(7, 2, 1.0)
        middle, _ = exact_interval_gram(7, 2, 0.6)
        self.assertEqual(len(points), math.comb(8, 4))
        np.testing.assert_allclose(identity, np.eye(len(points)))
        np.testing.assert_allclose(ones, np.ones_like(ones))
        np.testing.assert_allclose(middle, middle.T)
        np.testing.assert_allclose(np.diag(middle), 1.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(middle)[0]), -1e-10)

    def test_m_two_matches_legacy_constructor(self) -> None:
        try:
            from two_unknown_intervals_numerics import exact_gram
        except ModuleNotFoundError as error:
            if error.name == "two_unknown_intervals_numerics":
                self.skipTest(
                    "Paper II compatibility module is not included in the "
                    "Paper I release"
                )
            raise

        common, common_points = exact_interval_gram(8, 2, 0.7)
        legacy, legacy_points = exact_gram(8, 0.7)
        self.assertEqual(common_points, legacy_points)
        np.testing.assert_allclose(common, legacy)

    def test_endpoint_dichotomy_holds_for_general_small_m(self) -> None:
        validate_endpoint_dichotomy(
            max_interval_count=4,
            extra_sites=3,
        )

    def test_endpoint_distance_dominates_symmetric_difference(self) -> None:
        x = (0, 2, 4, 7, 9, 10)
        y = (1, 3, 5, 6, 8, 10)
        self.assertLessEqual(
            symmetric_difference_size(x, y),
            endpoint_distance(x, y),
        )


class SrmTests(unittest.TestCase):
    def test_srm_endpoint_values(self) -> None:
        identity = np.eye(7)
        ones = np.ones((7, 7))
        self.assertAlmostEqual(srm_quantities(identity)["srm"], 1.0)
        self.assertAlmostEqual(srm_quantities(ones)["srm"], 1.0 / 7.0)

    def test_trace_lower_bound_does_not_exceed_srm(self) -> None:
        gram, _ = exact_interval_gram(8, 2, 0.8)
        quantities = srm_quantities(gram)
        self.assertLessEqual(
            quantities["trace_lower_bound"],
            quantities["srm"] + 1e-12,
        )

    def test_p1_endpoints(self) -> None:
        self.assertEqual(p1(0.0), 1.0)
        self.assertEqual(p1(1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
