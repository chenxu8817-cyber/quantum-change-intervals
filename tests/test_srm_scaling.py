from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srm_scaling import (  # noqa: E402
    estimate_dense_resources,
    run_exact_case,
)


class ResourceGuardTests(unittest.TestCase):
    def test_resource_estimate_uses_binomial_candidate_count(self) -> None:
        estimate = estimate_dense_resources(50, 2)
        self.assertEqual(estimate["candidate_count"], math.comb(51, 4))
        self.assertEqual(
            estimate["gram_bytes"],
            8 * math.comb(51, 4) ** 2,
        )

    def test_large_m_two_dense_case_is_rejected_before_allocation(self) -> None:
        with self.assertRaisesRegex(MemoryError, "dense Gram estimate"):
            run_exact_case(
                n=50,
                interval_count=2,
                overlap=0.8,
                max_gram_bytes=2_000_000_000,
            )


class ScalingRowTests(unittest.TestCase):
    def test_exact_row_reports_high_overlap_diagnostics(self) -> None:
        row = run_exact_case(
            n=8,
            interval_count=1,
            overlap=0.95,
            max_gram_bytes=2_000_000_000,
        )
        self.assertEqual(row["method"], "exact_dense_eigh")
        self.assertAlmostEqual(
            float(row["correlation_length"]),
            1.0 / abs(math.log(0.95)),
        )
        self.assertIn("condition_number_on_support", row)
        self.assertIn("target_p1_power_2m", row)
        self.assertGreaterEqual(float(row["runtime_seconds"]), 0.0)


if __name__ == "__main__":
    unittest.main()
