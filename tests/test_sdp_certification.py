from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sdp_certification import (  # noqa: E402
    canonical_weighted_states,
    certify_minimum_error,
)


class CanonicalStateTests(unittest.TestCase):
    def test_weighted_states_reproduce_gram_over_hypothesis_count(self) -> None:
        gram = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        )
        states, metadata = canonical_weighted_states(gram)
        np.testing.assert_allclose(states.T @ states, gram / 3.0, atol=1e-11)
        self.assertEqual(metadata["rank"], 3)

    def test_rank_reduction_removes_identical_state_nullspace(self) -> None:
        states, metadata = canonical_weighted_states(np.ones((5, 5)))
        self.assertEqual(states.shape, (1, 5))
        self.assertEqual(metadata["rank"], 1)

    def test_raw_and_weighted_spectral_metadata_are_distinct(self) -> None:
        gram = np.array([[1.0, 0.4], [0.4, 1.0]])
        priors = np.array([0.8, 0.2])
        states, metadata = canonical_weighted_states(gram, priors=priors)
        np.testing.assert_allclose(
            states.T @ states,
            np.sqrt(priors)[:, None] * gram * np.sqrt(priors)[None, :],
            atol=1e-11,
        )
        raw_spectrum = np.linalg.eigvalsh(gram)
        weighted_spectrum = np.linalg.eigvalsh(states.T @ states)
        self.assertAlmostEqual(metadata["gram_lambda_min"], raw_spectrum[0])
        self.assertAlmostEqual(metadata["gram_lambda_max"], raw_spectrum[-1])
        self.assertAlmostEqual(
            metadata["weighted_gram_lambda_min"], weighted_spectrum[0]
        )
        self.assertAlmostEqual(
            metadata["weighted_gram_lambda_max"], weighted_spectrum[-1]
        )


class SdpCertificationTests(unittest.TestCase):
    def assert_certificate_is_tight(
        self, certificate: dict[str, float | int | str], tolerance: float
    ) -> None:
        self.assertLessEqual(
            float(certificate["primal_objective"]),
            float(certificate["dual_objective"]) + tolerance,
        )
        self.assertLess(abs(float(certificate["absolute_gap"])), tolerance)
        self.assertLess(
            float(certificate["primal_equality_residual_fro"]), tolerance
        )
        self.assertLess(
            float(certificate["primal_psd_violation"]), tolerance
        )
        self.assertLess(
            float(certificate["dual_psd_violation"]), tolerance
        )
        self.assertLess(
            float(certificate["complementarity_residual"]), 10 * tolerance
        )

    def test_orthogonal_states_have_unit_success(self) -> None:
        certificate = certify_minimum_error(np.eye(3), solver="CLARABEL")
        self.assertAlmostEqual(
            float(certificate["primal_objective"]), 1.0, places=6
        )
        self.assert_certificate_is_tight(certificate, 2e-6)

    def test_identical_states_have_inverse_hypothesis_success(self) -> None:
        certificate = certify_minimum_error(
            np.ones((4, 4)), solver="CLARABEL"
        )
        self.assertAlmostEqual(
            float(certificate["dual_objective"]), 0.25, places=6
        )
        self.assert_certificate_is_tight(certificate, 2e-6)

    def test_two_state_result_matches_helstrom_formula(self) -> None:
        overlap = 0.4
        gram = np.array([[1.0, overlap], [overlap, 1.0]])
        expected = 0.5 * (1.0 + math.sqrt(1.0 - overlap**2))
        certificate = certify_minimum_error(gram, solver="CLARABEL")
        midpoint = 0.5 * (
            float(certificate["primal_objective"])
            + float(certificate["dual_objective"])
        )
        self.assertAlmostEqual(midpoint, expected, places=6)
        self.assert_certificate_is_tight(certificate, 2e-6)

    def test_postprocessed_feasible_bounds_enclose_helstrom_value(self) -> None:
        """Catches returning raw, slightly infeasible SDP objectives as bounds."""
        overlap = 0.4
        gram = np.array([[1.0, overlap], [overlap, 1.0]])
        expected = 0.5 * (1.0 + math.sqrt(1.0 - overlap**2))
        certificate = certify_minimum_error(gram, solver="CLARABEL")
        required = {
            "primal_feasible_objective",
            "dual_feasible_objective",
            "feasible_bound_gap",
            "primal_feasible_equality_residual_fro",
            "primal_feasible_psd_violation",
            "primal_feasible_min_eigenvalue",
            "dual_feasible_psd_violation",
        }
        self.assertTrue(required.issubset(certificate))
        lower = float(certificate["primal_feasible_objective"])
        upper = float(certificate["dual_feasible_objective"])
        self.assertLessEqual(lower, expected + 1e-12)
        self.assertGreaterEqual(upper, expected - 1e-12)
        self.assertLessEqual(lower, upper)
        self.assertAlmostEqual(
            float(certificate["feasible_bound_gap"]), upper - lower
        )
        self.assertLess(
            float(certificate["primal_feasible_equality_residual_fro"]),
            1e-11,
        )
        self.assertLess(
            float(certificate["primal_feasible_psd_violation"]), 1e-11
        )
        self.assertGreater(
            float(certificate["primal_feasible_min_eigenvalue"]), 0.0
        )
        self.assertLess(
            float(certificate["dual_feasible_psd_violation"]), 1e-11
        )


class CertifiedScriptIntegrationTests(unittest.TestCase):
    def test_run_case_reports_primal_dual_gap_and_residuals(self) -> None:
        from interval_unknown_length_numerics import run_case

        row = run_case(2, 0.5, solver="CLARABEL", verbose=False)
        required = {
            "primal_objective",
            "dual_objective",
            "signed_gap",
            "absolute_gap",
            "relative_signed_gap",
            "relative_gap",
            "weighted_gram_lambda_min",
            "weighted_gram_lambda_max",
            "primal_equality_residual_fro_raw",
            "primal_equality_residual_fro",
            "primal_psd_violation",
            "dual_psd_violation",
            "complementarity_residual",
            "primal_feasible_objective",
            "dual_feasible_objective",
            "feasible_bound_gap",
            "sdp_lower",
            "sdp_upper",
            "sdp_lower_minus_srm",
            "sdp_upper_minus_srm",
        }
        self.assertTrue(required.issubset(row))
        self.assertLess(float(row["absolute_gap"]), 2e-6)
        self.assertAlmostEqual(
            float(row["absolute_gap"]),
            abs(float(row["signed_gap"])),
        )
        self.assertAlmostEqual(
            float(row["sdp_lower"]),
            float(row["primal_feasible_objective"]),
        )
        self.assertAlmostEqual(
            float(row["sdp_upper"]),
            float(row["dual_feasible_objective"]),
        )
        self.assertLessEqual(float(row["sdp_lower"]), float(row["sdp_upper"]))


if __name__ == "__main__":
    unittest.main()
