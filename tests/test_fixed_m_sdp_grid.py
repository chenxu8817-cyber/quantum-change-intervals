from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed_m_sdp_grid import certify_interval_case  # noqa: E402


class FixedMGridTests(unittest.TestCase):
    def test_multi_interval_case_reports_srm_and_certificate(self) -> None:
        row = certify_interval_case(
            n=4,
            interval_count=2,
            overlap=0.5,
            solver="CLARABEL",
        )
        self.assertEqual(row["m"], 2)
        self.assertEqual(row["candidate_count"], 5)
        self.assertIn("srm", row)
        self.assertIn("primal_objective", row)
        self.assertIn("dual_objective", row)
        self.assertLess(abs(float(row["absolute_gap"])), 2e-6)


if __name__ == "__main__":
    unittest.main()
