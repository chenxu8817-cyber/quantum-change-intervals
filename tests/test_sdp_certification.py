from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sdp_certification as certification_module  # noqa: E402

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


class WeightedStateCertificateArtifactTests(unittest.TestCase):
    def test_direct_weighted_state_api_preserves_input_bytes(self) -> None:
        """Catches a direct solver path that mutates its caller-owned factor."""
        gram = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        )
        states, _ = canonical_weighted_states(gram, rank_tolerance=0.0)
        before_shape = states.shape
        before_dtype = states.dtype
        before_bytes = states.tobytes(order="C")

        certification_module.certify_minimum_error_from_weighted_states(
            states,
            solver="CLARABEL",
        )

        self.assertEqual(states.shape, before_shape)
        self.assertEqual(states.dtype, before_dtype)
        self.assertEqual(states.tobytes(order="C"), before_bytes)

    def test_direct_weighted_state_api_returns_recheckable_actual_artifacts(
        self,
    ) -> None:
        """Catches silently canonicalizing a second source inside the solver."""
        direct_api = getattr(
            certification_module,
            "certify_minimum_error_from_weighted_states",
            None,
        )
        self.assertTrue(
            callable(direct_api),
            "weighted-state certificate API is missing",
        )
        if not callable(direct_api):
            return

        gram = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        )
        states, _ = canonical_weighted_states(gram, rank_tolerance=0.0)
        certificate = direct_api(
            states,
            solver="CLARABEL",
        )
        required = {
            "source_weighted_states",
            "source_weighted_gram",
            "repaired_primal_povm",
            "safe_dual_operator",
            "primal_feasible_objective",
            "dual_feasible_objective",
        }
        self.assertTrue(required.issubset(certificate))

        artifact_states = np.asarray(
            certificate["source_weighted_states"], dtype=float
        )
        artifact_gram = np.asarray(
            certificate["source_weighted_gram"], dtype=float
        )
        np.testing.assert_array_equal(artifact_states, states)
        self.assertFalse(np.shares_memory(artifact_states, states))
        np.testing.assert_allclose(
            artifact_gram,
            states.T @ states,
            rtol=0.0,
            atol=8.0 * np.finfo(float).eps,
        )

        rank, hypothesis_count = states.shape
        effects = tuple(
            np.asarray(effect, dtype=float)
            for effect in certificate["repaired_primal_povm"]
        )
        self.assertEqual(len(effects), hypothesis_count)
        for effect in effects:
            self.assertEqual(effect.shape, (rank, rank))
            self.assertTrue(np.all(np.isfinite(effect)))
            np.testing.assert_allclose(
                effect, effect.T, rtol=0.0, atol=8.0e-15
            )
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0]),
                -1.0e-12,
            )
        np.testing.assert_allclose(
            sum(effects, np.zeros((rank, rank))),
            np.eye(rank),
            rtol=0.0,
            atol=1.0e-11,
        )

        rhos = [
            np.outer(states[:, index], states[:, index])
            for index in range(hypothesis_count)
        ]
        primal_objective = math.fsum(
            float(np.trace(rho @ effect))
            for rho, effect in zip(rhos, effects)
        )
        self.assertAlmostEqual(
            primal_objective,
            float(certificate["primal_feasible_objective"]),
            places=13,
        )

        dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
        self.assertEqual(dual.shape, (rank, rank))
        self.assertTrue(np.all(np.isfinite(dual)))
        np.testing.assert_allclose(dual, dual.T, rtol=0.0, atol=8.0e-15)
        self.assertAlmostEqual(
            float(np.trace(dual)),
            float(certificate["dual_feasible_objective"]),
            places=13,
        )
        self.assertGreaterEqual(
            min(
                float(
                    np.linalg.eigvalsh(
                        (dual - rho + (dual - rho).T) / 2.0
                    )[0]
                )
                for rho in rhos
            ),
            -1.0e-12,
        )
        for artifact in (artifact_states, artifact_gram, *effects, dual):
            with self.subTest(artifact_shape=artifact.shape):
                self.assertFalse(np.shares_memory(artifact, states))

    def test_legacy_gram_api_remains_scalar_serializable(self) -> None:
        """Keep existing CSV callers free of newly added matrix artifacts."""
        certificate = certify_minimum_error(
            np.array([[1.0, 0.4], [0.4, 1.0]]),
            solver="CLARABEL",
        )
        self.assertNotIn("source_weighted_states", certificate)
        self.assertNotIn("source_weighted_gram", certificate)
        self.assertNotIn("repaired_primal_povm", certificate)
        self.assertNotIn("safe_dual_operator", certificate)
        self.assertTrue(
            all(
                isinstance(value, (str, int, float, np.integer, np.floating))
                for value in certificate.values()
            )
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

    def test_primal_repair_strictly_regularizes_rank_deficient_effects(self) -> None:
        """The published strict-feasibility gate must survive zero effects."""
        measurements = [
            np.diag([1.0, 0.0]),
            np.diag([0.0, 1.0]),
            np.zeros((2, 2)),
        ]
        rhos = [
            np.diag([0.4, 0.0]),
            np.diag([0.0, 0.3]),
            np.diag([0.2, 0.1]),
        ]

        repaired, diagnostics = certification_module._safe_primal_bound(
            measurements,
            rhos,
        )

        np.testing.assert_allclose(
            sum(repaired, np.zeros((2, 2))),
            np.eye(2),
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertGreater(float(diagnostics["regularization_floor"]), 0.0)
        self.assertGreater(float(diagnostics["minimum_eigenvalue"]), 0.0)
        for effect in repaired:
            self.assertGreater(
                float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0]),
                0.0,
            )

    def test_primal_repair_does_not_invert_a_singular_projected_sum(self) -> None:
        """The release repair remains defined when the raw POVM sum is singular."""
        rhos = [np.diag([0.4, 0.0]), np.diag([0.0, 0.3])]
        for measurements in (
            [np.diag([1.0, 0.0]), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2))],
        ):
            with self.subTest(nonzero_effects=sum(np.any(m) for m in measurements)):
                repaired, diagnostics = certification_module._safe_primal_bound(
                    measurements,
                    rhos,
                )
                np.testing.assert_allclose(
                    sum(repaired, np.zeros((2, 2))),
                    np.eye(2),
                    rtol=0.0,
                    atol=1.0e-14,
                )
                self.assertGreater(
                    float(diagnostics["regularization_floor"]), 0.0
                )
                self.assertGreater(
                    float(diagnostics["minimum_eigenvalue"]), 0.0
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
