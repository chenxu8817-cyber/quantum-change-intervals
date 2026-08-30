"""Behavior tests for resource-safe weighted-hull diagnostics.

The numerical rows in this module are finite-dimensional diagnostics only.
They retain the external prior ``1 / M_n``, where
``M_n = n(n + 1) / 2``, and never promote an SRM value or an SDP midpoint to
an exact optimum.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import csv
from contextlib import contextmanager, redirect_stdout
from fractions import Fraction
import hashlib
import io
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
import warnings
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sdp_certification as certification_module  # noqa: E402

try:  # The shape-safe fallbacks make the first TDD run assertion-RED.
    import proofs.weighted_hull_continuum_outer_probe as diagnostic_probe
except ModuleNotFoundError:
    diagnostic_probe = SimpleNamespace()

from proofs import weighted_hull_finite_audit  # noqa: E402
from quantum_interval_numerics import srm_quantities  # noqa: E402
from sdp_certification import certify_minimum_error  # noqa: E402


COMPACT_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
OUTER_SCHEDULES = {
    "outer_log_log_n",
    "outer_sqrt_log_n",
    "outer_n_one_third",
}
FROZEN_CSV_SHA256 = (
    "07BA116EC35C827CA0A0BE3296F615A9E5BD45AD91D8FE4F96818DBEBE330159"
)


def _interval_labels(n: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(n) for right in range(left, n)]


def _zero_physical_interval_gram(n: int, c: float) -> np.ndarray:
    del c
    count = n * (n + 1) // 2
    return np.zeros((count, count), dtype=float)


def _zero_independent_hull_gram(n: int, c: float) -> np.ndarray:
    return _zero_physical_interval_gram(n, c)


def _zero_estimate_dense_resources(n: int) -> dict[str, int]:
    del n
    return {
        "external_M_n": 0,
        "gram_bytes": 0,
        "estimated_peak_bytes": 0,
        "eigensolver_dimension": 0,
        "eigensolver_cubic_work_units": 0,
    }


def _zero_diagnostic_row(
    n: int,
    c: float,
    *,
    schedule: str,
    max_srm_n: int = 48,
    max_sdp_n: int = 5,
    max_peak_bytes: int = 512 * 2**20,
    max_matrix_dimension: int = 1200,
    allow_srm_above_default: bool = False,
    allow_sdp_above_default: bool = False,
    solver: str = "CLARABEL",
    resource_log=None,
) -> dict[str, object]:
    del (
        c,
        max_srm_n,
        max_sdp_n,
        max_peak_bytes,
        max_matrix_dimension,
        allow_srm_above_default,
        allow_sdp_above_default,
        solver,
        resource_log,
    )
    return {
        "schema_version": "zero",
        "schedule": schedule,
        "n": n,
        "external_M_n": 0,
        "c": 0.0,
        "lambda": 0.0,
        "h": 0.0,
        "p1_squared": 1.0,
        "P_tr": 0.0,
        "P_SRM": 0.0,
        "L_safe": 0.0,
        "U_safe": 0.0,
        "strongest_certified_lower": 0.0,
        "P_tr_over_p1_squared": 0.0,
        "P_SRM_over_p1_squared": 0.0,
        "L_safe_over_p1_squared": 0.0,
        "U_safe_over_p1_squared": 0.0,
        "strongest_lower_over_p1_squared": 0.0,
        "P_tr_over_P_SRM": 0.0,
        "L_safe_over_P_SRM": 0.0,
        "P_SRM_over_U_safe": 0.0,
        "L_safe_over_U_safe": 0.0,
        "safe_gap": 0.0,
        "safe_gap_relative": 0.0,
        "primal_feasible_equality_residual_fro": 0.0,
        "primal_feasible_equality_residual_op": 0.0,
        "primal_feasible_psd_violation": 0.0,
        "primal_feasible_min_eigenvalue": 0.0,
        "dual_feasible_min_slack": 0.0,
        "dual_feasible_psd_violation": 0.0,
        "P_opt": 0.0,
        "P_opt_status": "zero",
        "srm_status": "zero",
        "sdp_status": "zero",
        "primal_status": "zero",
        "dual_status": "zero",
        "status": "zero",
        "solver": "zero",
        "cvxpy_version": "zero",
        "gram_bytes": 0,
        "estimated_peak_bytes": 0,
        "eigensolver_dimension": 0,
        "eigensolver_cubic_work_units": 0,
        "interpretation": "zero",
    }


def _zero_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("zero.csv"))
    parser.add_argument("--max-srm-n", type=int, default=0)
    parser.add_argument("--max-sdp-n", type=int, default=0)
    parser.add_argument("--max-peak-gib", type=float, default=0.0)
    parser.add_argument("--max-matrix-dimension", type=int, default=0)
    parser.add_argument("--allow-srm-above-48", action="store_true")
    parser.add_argument("--allow-sdp-above-5", action="store_true")
    return parser


def _subject(name: str, fallback):
    return getattr(diagnostic_probe, name, fallback)


def _physical_interval_gram(n: int, c: float) -> np.ndarray:
    return _subject(
        "physical_interval_gram", _zero_physical_interval_gram
    )(n, c)


def _independent_hull_gram(n: int, c: float) -> np.ndarray:
    return _subject(
        "independent_hull_gram", _zero_independent_hull_gram
    )(n, c)


def _estimate_dense_resources(n: int) -> dict[str, int]:
    return _subject(
        "estimate_dense_resources", _zero_estimate_dense_resources
    )(n)


def _diagnostic_row(n: int, c: float, **kwargs) -> dict[str, object]:
    return _subject("diagnostic_row", _zero_diagnostic_row)(n, c, **kwargs)


def _explicit_product_state_gram(n: int, c: float) -> np.ndarray:
    """Independently build all physical tensor-product interval states."""
    zero = np.array([1.0, 0.0])
    anomaly = np.array([c, math.sqrt(1.0 - c * c)])
    states = []
    for left, right in _interval_labels(n):
        state = np.array([1.0])
        for site in range(n):
            local = anomaly if left <= site <= right else zero
            state = np.kron(state, local)
        states.append(state)
    state_matrix = np.asarray(states)
    return state_matrix @ state_matrix.T


def _explicit_ge_two_coordinate_matrix(n: int, c: float) -> np.ndarray:
    """Enumerate excitation subsets without any Task 1 coordinate helper."""
    subsets = [mask for mask in range(1 << n) if mask.bit_count() >= 2]
    coordinates = np.zeros((n * (n + 1) // 2, len(subsets)))
    s = math.sqrt(1.0 - c * c)
    for row, (left, right) in enumerate(_interval_labels(n)):
        interval_mask = sum(1 << site for site in range(left, right + 1))
        length = right - left + 1
        for column, subset in enumerate(subsets):
            if subset & ~interval_mask == 0:
                weight = subset.bit_count()
                coordinates[row, column] = s**weight * c ** (
                    length - weight
                )
    return coordinates


def _explicit_theta_measurement_success(c: float) -> float:
    """Independent n=2 projective-POVM success at theta=-pi/6."""
    zero = np.array([1.0, 0.0])
    anomaly = np.array([c, math.sqrt(1.0 - c * c)])
    states = (
        np.kron(anomaly, zero),
        np.kron(anomaly, anomaly),
        np.kron(zero, anomaly),
    )
    vacuum = np.array([1.0, 0.0, 0.0, 0.0])
    left_excitation = np.array([0.0, 0.0, 1.0, 0.0])
    right_excitation = np.array([0.0, 1.0, 0.0, 0.0])
    double_excitation = np.array([0.0, 0.0, 0.0, 1.0])
    antisymmetric = (left_excitation - right_excitation) / math.sqrt(2.0)
    symmetric = (left_excitation + right_excitation) / math.sqrt(2.0)
    theta = -math.pi / 6.0
    measurement_vectors = []
    for outcome in range(3):
        phase = theta + 2.0 * math.pi * outcome / 3.0
        measurement_vectors.append(
            vacuum / math.sqrt(3.0)
            + math.sqrt(2.0 / 3.0)
            * (
                math.cos(phase) * antisymmetric
                + math.sin(phase) * symmetric
            )
        )
    effects = [
        np.outer(vector, vector) for vector in measurement_vectors
    ]
    effects[1] = effects[1] + np.outer(
        double_excitation, double_excitation
    )
    np.testing.assert_allclose(
        sum(effects), np.eye(4), rtol=0.0, atol=8.0e-16
    )
    for effect in effects:
        if float(np.linalg.eigvalsh(effect)[0]) < -8.0e-16:
            raise AssertionError("explicit theta measurement is not PSD")
    return math.fsum(
        float(state @ effect @ state)
        for state, effect in zip(states, effects)
    ) / 3.0


@contextmanager
def _patched_attribute(name: str, value: object):
    missing = object()
    original = getattr(diagnostic_probe, name, missing)
    setattr(diagnostic_probe, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(diagnostic_probe, name)
        else:
            setattr(diagnostic_probe, name, original)


class PhysicalIntervalGramTests(unittest.TestCase):
    def test_formula_matches_explicit_product_states_through_n4(self) -> None:
        for n in range(1, 5):
            for c in (0.0, 0.4, 0.75, 1.0):
                with self.subTest(n=n, c=c):
                    expected = _explicit_product_state_gram(n, c)
                    actual = _physical_interval_gram(n, c)
                    np.testing.assert_allclose(actual, expected, atol=2e-14)

    def test_physical_gram_is_psd_with_unit_diagonal(self) -> None:
        gram = _physical_interval_gram(4, 0.63)
        np.testing.assert_allclose(np.diag(gram), np.ones(10), atol=1e-14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -2e-13)

    def test_small_n_path_actively_runs_the_independent_tensor_check(
        self,
    ) -> None:
        original_checker = _subject(
            "_explicit_product_state_gram", _explicit_product_state_gram
        )

        def corrupted_checker(n: int, c: float) -> np.ndarray:
            return np.zeros_like(original_checker(n, c))

        with _patched_attribute(
            "_explicit_product_state_gram", corrupted_checker
        ):
            with self.assertRaisesRegex(AssertionError, "product-state"):
                _physical_interval_gram(3, 0.4)


class IndependentHullGramTests(unittest.TestCase):
    def test_subtracted_hull_gram_matches_explicit_ge_two_states(self) -> None:
        for n, c in ((3, 0.4), (4, 0.75), (4, 1.0)):
            with self.subTest(n=n, c=c):
                coordinates = _explicit_ge_two_coordinate_matrix(n, c)
                expected = coordinates @ coordinates.T
                actual = _independent_hull_gram(n, c)
                np.testing.assert_allclose(actual, expected, atol=3e-14)

    def test_independent_construction_ignores_task1_coordinate_helper(
        self,
    ) -> None:
        expected_coordinates = _explicit_ge_two_coordinate_matrix(4, 0.6)
        expected = expected_coordinates @ expected_coordinates.T

        def unusable_task1_helper(n: int, c: Fraction):
            del c
            count = n * (n + 1) // 2
            columns = n * (n - 1) // 2
            return [[Fraction(0)] * columns for _ in range(count)]

        original = weighted_hull_finite_audit.weighted_hull_matrix_fraction
        weighted_hull_finite_audit.weighted_hull_matrix_fraction = (
            unusable_task1_helper
        )
        try:
            actual = _independent_hull_gram(4, 0.6)
        finally:
            weighted_hull_finite_audit.weighted_hull_matrix_fraction = original
        np.testing.assert_allclose(actual, expected, atol=3e-14)


class StableHullGramTests(unittest.TestCase):
    def test_near_endpoint_hull_matches_explicit_ge_two_oracle_and_is_psd(
        self,
    ) -> None:
        n = 4
        c = 1.0 - 1.0e-8
        coordinates = _explicit_ge_two_coordinate_matrix(n, c)
        expected = coordinates @ coordinates.T
        actual = _independent_hull_gram(n, c)
        expected_scale = float(np.linalg.norm(expected, ord="fro"))
        relative_error = float(
            np.linalg.norm(actual - expected, ord="fro") / expected_scale
        )
        self.assertLess(relative_error, 5.0e-10)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(actual)[0]), -1.0e-28)

    def test_corrupted_vacuum_or_one_excitation_breaks_raw_subtraction_audit(
        self,
    ) -> None:
        for helper_name in ("_vacuum_gram", "_one_excitation_gram"):
            original = getattr(diagnostic_probe, helper_name)

            def corrupted(n: int, c: float, subject=original) -> np.ndarray:
                return np.zeros_like(subject(n, c))

            with self.subTest(helper=helper_name):
                with _patched_attribute(helper_name, corrupted):
                    with self.assertRaisesRegex(
                        AssertionError, "raw subtraction consistency"
                    ):
                        _independent_hull_gram(4, 0.6)


class DenseResourceGuardTests(unittest.TestCase):
    def test_estimate_reports_exact_gram_peak_and_cubic_work(self) -> None:
        estimate = _estimate_dense_resources(4)
        self.assertEqual(estimate["external_M_n"], 10)
        self.assertEqual(estimate["gram_bytes"], 800)
        self.assertGreaterEqual(estimate["estimated_peak_bytes"], 4_800)
        self.assertEqual(estimate["eigensolver_dimension"], 10)
        self.assertEqual(estimate["eigensolver_cubic_work_units"], 1_000)
        self.assertEqual(
            estimate.get("estimated_peak_scope"),
            "conservative_dense_linear_algebra_array_proxy_not_process_peak",
        )

    def test_resource_guard_returns_before_dense_allocation(self) -> None:
        allocation_calls: list[tuple[int, float]] = []
        original_physical = _subject(
            "physical_interval_gram", _zero_physical_interval_gram
        )

        def allocation_spy(n: int, c: float) -> np.ndarray:
            allocation_calls.append((n, c))
            return original_physical(n, c)

        logs: list[dict[str, int]] = []
        with _patched_attribute("physical_interval_gram", allocation_spy):
            row = _diagnostic_row(
                4,
                0.6,
                schedule="resource_guard_test",
                max_srm_n=4,
                max_sdp_n=0,
                max_peak_bytes=1,
                max_matrix_dimension=10,
                resource_log=logs.append,
            )
        self.assertEqual(allocation_calls, [])
        self.assertEqual(row["status"], "not_computed_resource_guard")
        self.assertEqual(row["srm_status"], "not_computed_resource_guard")
        self.assertEqual(row["sdp_status"], "not_computed_resource_guard")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["gram_bytes"], 800)

    def test_extended_cutoffs_require_explicit_opt_in_after_estimate(
        self,
    ) -> None:
        logs: list[dict[str, int]] = []
        row = _diagnostic_row(
            49,
            0.99,
            schedule="extended_srm_guard_test",
            max_srm_n=49,
            max_sdp_n=5,
            max_peak_bytes=10**12,
            max_matrix_dimension=2_000,
            allow_srm_above_default=False,
            resource_log=logs.append,
        )
        self.assertEqual(row["srm_status"], "not_computed_opt_in_required")
        self.assertEqual(row["sdp_status"], "not_computed")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["external_M_n"], 1_225)

        sdp_row = _diagnostic_row(
            6,
            0.8,
            schedule="extended_sdp_guard_test",
            max_srm_n=5,
            max_sdp_n=7,
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
            allow_sdp_above_default=False,
        )
        self.assertEqual(
            sdp_row["sdp_status"], "not_computed_opt_in_required"
        )


@unittest.skip(
    "Route B retired the public certificate/enclosure schema; covered by "
    "RouteBFallbackTests."
)
class DiagnosticRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.row = _diagnostic_row(
            3,
            0.5,
            schedule="compact_lambda",
            max_srm_n=48,
            max_sdp_n=5,
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )

    def test_row_uses_external_prior_and_records_all_core_fields(self) -> None:
        required = {
            "n",
            "external_M_n",
            "c",
            "lambda",
            "h",
            "p1_squared",
            "P_tr",
            "P_SRM",
            "P_SRM_transfer_budget",
            "P_SRM_safe_lower",
            "L_safe",
            "U_safe",
            "L_raw",
            "U_raw",
            "strongest_certified_lower",
            "safe_gap",
            "raw_safe_gap",
            "raw_safe_gap_relative",
            "primal_feasible_equality_residual_fro",
            "primal_feasible_equality_residual_op",
            "primal_feasible_psd_violation",
            "dual_feasible_psd_violation",
            "status",
            "gram_bytes",
            "estimated_peak_bytes",
            "eigensolver_dimension",
            "eigensolver_cubic_work_units",
            "estimated_peak_scope",
            "source_gram_factorization_residual_fro_relative",
            "source_gram_factorization_residual_op_relative",
            "source_gram_support_rank",
            "srm_sqrt_reconstruction_residual_fro_relative",
            "srm_sqrt_support_rank",
            "source_reconstructed_sqrt_residual_fro_relative",
            "source_reconstructed_sqrt_residual_op_relative",
            "spectral_audit_status",
            "hull_subtraction_max_abs_error",
            "hull_subtraction_max_elementwise_envelope",
            "hull_subtraction_fro_error",
            "hull_subtraction_fro_envelope",
            "hull_subtraction_audit_status",
            "hull_subtraction_audit_kind",
            "source_target_trace",
            "source_reconstructed_trace",
            "source_difference_trace_norm",
            "source_difference_fro_norm",
            "source_difference_op_norm",
            "source_difference_trace_norm_relative",
            "source_difference_fro_norm_relative",
            "source_difference_op_norm_relative",
            "source_lambda_min_target",
            "source_lambda_max_target",
            "source_lambda_min_reconstructed",
            "source_lambda_max_reconstructed",
            "source_sqrt_difference_computed_fro",
            "source_sqrt_difference_bound_trace",
            "source_sqrt_difference_bound_spectral",
            "source_sqrt_difference_bound",
            "source_transfer_delta",
            "source_transfer_budget",
            "source_transfer_status",
            "source_difference_op_norm",
            "source_difference_trace_norm_relative",
            "source_difference_fro_norm_relative",
            "source_difference_op_norm_relative",
            "source_lambda_max_target",
            "source_lambda_max_reconstructed",
            "source_reconstructed_sqrt_residual_fro_relative",
            "source_reconstructed_sqrt_residual_op_relative",
            "trusted_source_snapshot_sha256",
            "certificate_input_factor_pre_sha256",
            "certificate_input_factor_post_sha256",
            "certificate_input_factor_max_abs_error",
            "certificate_input_factor_pre_writeable",
            "certificate_input_factor_post_writeable",
            "certificate_input_factor_validation_status",
            "certificate_input_factor_validation_reason",
            "certificate_source_binding_status",
            "certificate_source_binding_reason",
            "certificate_artifact_source_states_max_abs_error",
            "certificate_artifact_source_gram_max_abs_error",
            "certificate_artifact_primal_completeness_fro",
            "certificate_artifact_primal_completeness_op",
            "certificate_artifact_primal_psd_violation",
            "certificate_artifact_primal_objective",
            "certificate_artifact_dual_symmetry_error_fro",
            "certificate_artifact_dual_min_slack",
            "certificate_artifact_dual_psd_violation",
            "certificate_artifact_dual_trace",
            "certificate_artifact_validation_status",
            "certificate_artifact_validation_reason",
            "certificate_rank",
            "certificate_expected_rank",
            "certificate_rank_tolerance",
            "certificate_probability_scale",
            "certificate_probability_tolerance",
            "certificate_residual_tolerance",
            "certificate_validation_status",
            "enclosure_kind",
            "numeric_guarantee",
            "diagnostic_grid_scope",
        }
        self.assertTrue(required.issubset(self.row))
        self.assertEqual(self.row["external_M_n"], 6)
        self.assertAlmostEqual(float(self.row["lambda"]), 1.5)
        self.assertEqual(self.row["status"], "sdp_certified")
        self.assertEqual(
            self.row["interpretation"], "finite_size_diagnostic_not_proof"
        )

    def test_trace_srm_and_safe_bounds_have_the_required_order(self) -> None:
        trace = float(self.row["P_tr"])
        srm = float(self.row["P_SRM"])
        self.assertIn("P_SRM_safe_lower", self.row)
        srm_safe = float(self.row["P_SRM_safe_lower"])
        lower = float(self.row["L_safe"])
        upper = float(self.row["U_safe"])
        probability_scale = max(
            abs(lower), abs(upper), abs(srm), 1.0 / int(self.row["external_M_n"])
        )
        tolerance = 1.0e-8 * probability_scale
        self.assertLessEqual(trace, srm + tolerance)
        self.assertLessEqual(srm, upper + tolerance)
        self.assertLessEqual(srm_safe, srm)
        self.assertLessEqual(srm_safe, upper + tolerance)
        self.assertLessEqual(lower, upper + tolerance)
        self.assertAlmostEqual(
            float(self.row["safe_gap"]), upper - lower, places=13
        )

    def test_spectral_and_hull_audits_are_residual_checked_not_directed(
        self,
    ) -> None:
        self.assertEqual(self.row.get("spectral_audit_status"), "passed")
        self.assertEqual(
            self.row.get("hull_subtraction_audit_status"), "passed"
        )
        self.assertLess(
            float(
                self.row.get(
                    "source_gram_factorization_residual_fro_relative",
                    math.inf,
                )
            ),
            1.0e-12,
        )
        self.assertLess(
            float(
                self.row.get(
                    "srm_sqrt_reconstruction_residual_fro_relative",
                    math.inf,
                )
            ),
            1.0e-12,
        )
        self.assertEqual(self.row.get("source_gram_support_rank"), 6)
        self.assertEqual(self.row.get("srm_sqrt_support_rank"), 6)
        self.assertEqual(
            self.row.get("enclosure_kind"),
            "residual_checked_floating_point_enclosure",
        )
        self.assertEqual(
            self.row.get("numeric_guarantee"),
            "ieee_double_residual_checked_not_directed_rounding_proof",
        )
        self.assertEqual(
            self.row.get("diagnostic_grid_scope"),
            "default_srm_cutoff_n48_frozen_grid_through_n32",
        )

    def test_repaired_primal_and_dual_are_feasible(self) -> None:
        self.assertLess(
            float(self.row["primal_feasible_equality_residual_fro"]),
            2e-11,
        )
        self.assertLess(
            float(self.row["primal_feasible_equality_residual_op"]),
            2e-11,
        )
        self.assertLess(
            float(self.row["primal_feasible_psd_violation"]), 2e-11
        )
        self.assertGreater(
            float(self.row["primal_feasible_min_eigenvalue"]), 0.0
        )
        self.assertLess(
            float(self.row["dual_feasible_psd_violation"]), 2e-11
        )
        self.assertGreaterEqual(
            float(self.row["dual_feasible_min_slack"]), -2e-11
        )

    def test_strongest_lower_is_max_of_safe_primal_and_srm(self) -> None:
        self.assertIn("P_SRM_safe_lower", self.row)
        expected = max(
            float(self.row["L_safe"]),
            float(self.row["P_SRM"]),
        )
        self.assertEqual(float(self.row["strongest_certified_lower"]), expected)
        self.assertEqual(float(self.row["P_SRM_transfer_budget"]), 0.0)
        self.assertEqual(
            float(self.row["P_SRM_safe_lower"]), float(self.row["P_SRM"])
        )

    def test_above_sdp_cutoff_has_no_mislabeled_optimum(self) -> None:
        row = _diagnostic_row(
            6,
            0.8,
            schedule="outer_cutoff_test",
            max_srm_n=48,
            max_sdp_n=5,
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        self.assertEqual(row["sdp_status"], "not_computed")
        self.assertEqual(row["P_opt_status"], "not_computed")
        self.assertEqual(row["P_opt"], "")
        self.assertNotEqual(row["P_opt"], row["P_SRM"])

    def test_relative_ratios_use_p1_squared_as_the_target(self) -> None:
        target = float(self.row["p1_squared"])
        for value_field, ratio_field in (
            ("P_tr", "P_tr_over_p1_squared"),
            ("P_SRM", "P_SRM_over_p1_squared"),
            ("L_safe", "L_safe_over_p1_squared"),
            ("U_safe", "U_safe_over_p1_squared"),
            (
                "strongest_certified_lower",
                "strongest_lower_over_p1_squared",
            ),
        ):
            with self.subTest(ratio_field=ratio_field):
                self.assertAlmostEqual(
                    float(self.row[ratio_field]),
                    float(self.row[value_field]) / target,
                    places=13,
                )


@unittest.skip("Route B publishes no finite-size enclosure.")
class FullRankNearSingularSdpTests(unittest.TestCase):
    OVERLAP = 1.0 - 1.0e-12

    def test_near_singular_sdp_uses_rank_zero_cutoff_and_retains_rank_three(
        self,
    ) -> None:
        calls: list[object] = []
        original = diagnostic_probe.canonical_weighted_states

        def rank_spy(*args, **kwargs):
            calls.append(kwargs.get("rank_tolerance"))
            return original(*args, **kwargs)

        with _patched_attribute("canonical_weighted_states", rank_spy):
            row = _diagnostic_row(
                2,
                self.OVERLAP,
                schedule="near_singular_full_rank",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(calls, [0.0])
        self.assertEqual(row.get("certificate_rank"), 3)
        self.assertEqual(row.get("certificate_expected_rank"), 3)
        self.assertEqual(row.get("certificate_rank_tolerance"), 0.0)
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")

    def test_near_singular_upper_encloses_pair_measurement_and_srm(self) -> None:
        c = self.OVERLAP
        row = _diagnostic_row(
            2,
            c,
            schedule="near_singular_pair_lower",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        pair_lower = (1.0 / 3.0) * (
            1.0 + math.sqrt(1.0 - c**4)
        )
        upper = float(row["U_safe"])
        srm = float(row["P_SRM"])
        with self.subTest(bound="explicit_pair_measurement"):
            self.assertGreaterEqual(upper, pair_lower)
        with self.subTest(bound="srm"):
            self.assertLessEqual(srm, upper)


@unittest.skip("Route B removed public source-transfer bound padding.")
class SourceTransferPaddingTests(unittest.TestCase):
    EXTREME_OVERLAP = 0.9999999999999987

    def test_padded_upper_encloses_explicit_theta_measurement(self) -> None:
        measurement_lower = _explicit_theta_measurement_success(
            self.EXTREME_OVERLAP
        )
        row = _diagnostic_row(
            2,
            self.EXTREME_OVERLAP,
            schedule="source_transfer_theta_measurement",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        self.assertGreaterEqual(float(row["U_safe"]), measurement_lower)
        required = (
            "L_raw",
            "U_raw",
            "source_target_trace",
            "source_reconstructed_trace",
            "source_difference_trace_norm",
            "source_difference_fro_norm",
            "source_lambda_min_target",
            "source_lambda_min_reconstructed",
            "source_sqrt_difference_computed_fro",
            "source_sqrt_difference_bound_trace",
            "source_sqrt_difference_bound_spectral",
            "source_sqrt_difference_bound",
            "source_transfer_delta",
            "source_transfer_budget",
            "P_SRM_transfer_budget",
            "P_SRM_safe_lower",
        )
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, row)
                self.assertTrue(math.isfinite(float(row[field])))
        self.assertEqual(row.get("source_transfer_status"), "passed")
        self.assertEqual(
            row.get("certificate_source_binding_status"), "passed"
        )
        sqrt_bound = min(
            float(row["source_sqrt_difference_bound_trace"]),
            float(row["source_sqrt_difference_bound_spectral"]),
        )
        self.assertEqual(
            float(row["source_sqrt_difference_bound"]), sqrt_bound
        )
        expected_delta = math.sqrt(
            2.0
            * (
                float(row["source_target_trace"])
                + float(row["source_reconstructed_trace"])
            )
        ) * sqrt_bound
        self.assertAlmostEqual(
            float(row["source_transfer_delta"]), expected_delta, places=22
        )
        self.assertGreaterEqual(
            float(row["source_transfer_budget"]), expected_delta
        )
        self.assertLessEqual(float(row["L_safe"]), float(row["L_raw"]))
        self.assertGreaterEqual(float(row["U_safe"]), float(row["U_raw"]))
        self.assertLessEqual(
            float(row["P_SRM_safe_lower"]), float(row["P_SRM"])
        )
        self.assertEqual(
            float(row["strongest_certified_lower"]),
            max(
                float(row["L_safe"]),
                float(row["P_SRM"]),
            ),
        )
        self.assertEqual(float(row["P_SRM_transfer_budget"]), 0.0)
        self.assertEqual(float(row["P_SRM_safe_lower"]), float(row["P_SRM"]))

    def test_certificate_metadata_must_bind_to_source_factorization(self) -> None:
        original = getattr(
            diagnostic_probe,
            "certify_minimum_error_from_weighted_states",
            None,
        )

        def wrong_source_minimum(*args, **kwargs):
            if not callable(original):
                return {}
            certificate = dict(original(*args, **kwargs))
            certificate["weighted_gram_lambda_min"] = (
                0.4 * float(certificate["weighted_gram_lambda_min"])
            )
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            wrong_source_minimum,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="source_metadata_binding",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_source_binding_failed",
        )
        self.assertEqual(row.get("certificate_source_binding_status"), "failed")
        self.assertEqual(row.get("L_safe"), "")
        self.assertEqual(row.get("U_safe"), "")
        self.assertEqual(row.get("status"), "srm_only_certificate_failed")

    def test_raw_and_padded_relative_gaps_are_recomputed(self) -> None:
        original = getattr(
            diagnostic_probe,
            "certify_minimum_error_from_weighted_states",
            None,
        )

        def poisoned_relative_gap(*args, **kwargs):
            if not callable(original):
                return {}
            certificate = dict(original(*args, **kwargs))
            certificate["relative_feasible_bound_gap"] = 99.0
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            poisoned_relative_gap,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="relative_gap_recomputation",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertNotEqual(row.get("safe_gap_relative"), 99.0)
        for field in ("L_raw", "U_raw", "raw_safe_gap", "raw_safe_gap_relative"):
            self.assertIn(field, row)
        raw_lower = float(row["L_raw"])
        raw_upper = float(row["U_raw"])
        raw_gap = raw_upper - raw_lower
        raw_scale = max(
            np.finfo(float).tiny,
            0.5 * (abs(raw_lower) + abs(raw_upper)),
        )
        self.assertEqual(float(row["raw_safe_gap"]), raw_gap)
        self.assertAlmostEqual(
            float(row["raw_safe_gap_relative"]),
            raw_gap / raw_scale,
            places=15,
        )
        padded_lower = float(row["L_safe"])
        padded_upper = float(row["U_safe"])
        padded_gap = padded_upper - padded_lower
        padded_scale = max(
            np.finfo(float).tiny,
            0.5 * (abs(padded_lower) + abs(padded_upper)),
        )
        self.assertEqual(float(row["safe_gap"]), padded_gap)
        self.assertAlmostEqual(
            float(row["safe_gap_relative"]),
            padded_gap / padded_scale,
            places=15,
        )


@unittest.skip("Route B uses one pinned source audit rather than bound transfer.")
class ActualSourceFactorBindingTests(unittest.TestCase):
    def test_shared_canonical_drift_cannot_change_retained_source_certificate(
        self,
    ) -> None:
        """Catches a certificate that secretly canonicalizes a second S."""
        baseline = _diagnostic_row(
            3,
            0.5,
            schedule="actual_source_baseline",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        original = certification_module.canonical_weighted_states

        def rotated_hypothesis_columns(*args, **kwargs):
            states, metadata = original(*args, **kwargs)
            rotated = np.array(states, copy=True)
            first = np.array(rotated[:, 0], copy=True)
            second = np.array(rotated[:, 1], copy=True)
            theta = 0.001
            cosine = math.cos(theta)
            sine = math.sin(theta)
            rotated[:, 0] = cosine * first + sine * second
            rotated[:, 1] = -sine * first + cosine * second
            return rotated, metadata

        certification_module.canonical_weighted_states = (
            rotated_hypothesis_columns
        )
        try:
            drifted = _diagnostic_row(
                3,
                0.5,
                schedule="actual_source_drift",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        finally:
            certification_module.canonical_weighted_states = original

        self.assertEqual(
            drifted.get("sdp_status"), "computed_safe_enclosure"
        )
        self.assertEqual(
            drifted.get("certificate_artifact_validation_status"), "passed"
        )
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    float(drifted[field]), float(baseline[field]), places=12
                )


@unittest.skip("Route B removed the duplicate post-audit certificate path.")
class TrustedSnapshotSourceAuditTests(unittest.TestCase):
    """Make the post-audit retained factor the sole transfer authority."""

    EXACT_TARGET_DUAL_UPPER = float(Fraction(1365022261, 1500000000))
    TARGET_FEASIBLE_LOWER = 0.9100138365862787

    def _row_after_post_audit_transform(
        self,
        transform,
        *,
        schedule: str,
    ) -> dict[str, object]:
        original = diagnostic_probe._spectral_reconstruction_audit
        calls = 0

        def transformed_audit(gram):
            nonlocal calls
            states, metadata, audit = original(gram)
            if states.shape == (3, 3):
                calls += 1
                states = transform(np.array(states, copy=True, order="C"))
            # Preserve stale pre-transform self-reports deliberately.  The
            # real direct solver still consumes the transformed returned S.
            return states, metadata, audit

        with _patched_attribute(
            "_spectral_reconstruction_audit", transformed_audit
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule=schedule,
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(calls, 1, "post-audit transform did not fire")
        return row

    def _assert_source_drift_fails_closed(
        self, row: dict[str, object]
    ) -> None:
        self.assertEqual(
            row.get("sdp_status"), "not_computed_source_transfer_failed"
        )
        self.assertEqual(row.get("spectral_audit_status"), "failed")
        self.assertEqual(row.get("source_transfer_status"), "failed")
        self.assertGreater(
            float(row.get("source_difference_fro_norm_relative", 0.0)),
            1.0e-10,
        )
        self.assertGreater(float(row.get("source_transfer_delta", 0.0)), 0.0)
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            with self.subTest(field=field):
                self.assertEqual(row.get(field), "")
        self.assertFalse(
            any(isinstance(value, np.ndarray) for value in row.values()),
            "fail-closed diagnostic row leaked a private matrix artifact",
        )

    def test_post_audit_scale_up_uses_snapshot_audit_or_fails_closed(
        self,
    ) -> None:
        row = self._row_after_post_audit_transform(
            lambda states: (1.0 + 2.0e-6) * states,
            schedule="post_audit_scale_up",
        )
        if row.get("L_safe") != "":
            self.assertGreater(
                float(row["L_safe"]), self.EXACT_TARGET_DUAL_UPPER
            )
        self._assert_source_drift_fails_closed(row)

    def test_post_audit_scale_down_uses_snapshot_audit_or_fails_closed(
        self,
    ) -> None:
        row = self._row_after_post_audit_transform(
            lambda states: (1.0 - 2.0e-6) * states,
            schedule="post_audit_scale_down",
        )
        if row.get("U_safe") != "":
            self.assertLess(
                float(row["U_safe"]), self.TARGET_FEASIBLE_LOWER
            )
        self._assert_source_drift_fails_closed(row)

    def test_post_audit_hypothesis_column_rotations_fail_closed(self) -> None:
        for theta in (-0.001, 0.001):
            with self.subTest(theta=theta):
                def rotate_columns(states, theta=theta):
                    first = np.array(states[:, 0], copy=True)
                    second = np.array(states[:, 1], copy=True)
                    states[:, 0] = (
                        math.cos(theta) * first + math.sin(theta) * second
                    )
                    states[:, 1] = (
                        -math.sin(theta) * first + math.cos(theta) * second
                    )
                    return states

                row = self._row_after_post_audit_transform(
                    rotate_columns,
                    schedule=f"post_audit_column_rotation_{theta}",
                )
                self._assert_source_drift_fails_closed(row)

    def test_post_audit_physical_row_rotation_is_gram_preserving(self) -> None:
        theta = 0.1

        def rotate_rows(states):
            first = np.array(states[0, :], copy=True)
            second = np.array(states[1, :], copy=True)
            states[0, :] = math.cos(theta) * first + math.sin(theta) * second
            states[1, :] = -math.sin(theta) * first + math.cos(theta) * second
            return states

        row = self._row_after_post_audit_transform(
            rotate_rows,
            schedule="post_audit_physical_row_rotation",
        )
        self.assertEqual(row.get("spectral_audit_status"), "passed")
        self.assertEqual(row.get("source_transfer_status"), "passed")
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")

    def test_unmodified_post_audit_factor_computes(self) -> None:
        row = _diagnostic_row(
            2,
            0.5,
            schedule="unmodified_post_audit_factor",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        self.assertEqual(row.get("spectral_audit_status"), "passed")
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")


@unittest.skip("Route B has one pinned source audit and one artifact pass.")
class PostAuthoritySnapshotIntegrityTests(unittest.TestCase):
    """Catch mutation after the authoritative helper has already returned."""

    N2_TARGET_FEASIBLE_LOWER = 0.9100138365862787
    N2_EXACT_DUAL_UPPER = float(Fraction(1365022261, 1500000000))
    N3_TARGET_FEASIBLE_LOWER = 0.8634848413399037

    def _row_after_post_authority_rotation(
        self,
        *,
        n: int,
        theta: float,
    ) -> tuple[dict[str, object], dict[str, object]]:
        original = diagnostic_probe._trusted_source_snapshot_audit
        original_direct = (
            diagnostic_probe.certify_minimum_error_from_weighted_states
        )
        evidence: dict[str, object] = {"calls": 0, "solver_calls": 0}

        def mutate_after_authority(gram, snapshot):
            metadata, audit = original(gram, snapshot)
            states = np.asarray(snapshot)
            if states.flags.writeable:
                return metadata, audit
            evidence["calls"] = int(evidence["calls"]) + 1
            evidence["stale_transfer_budget"] = float(
                audit["source_transfer_budget"]
            )
            states.setflags(write=True)
            first = np.array(states[:, 0], copy=True)
            second = np.array(states[:, 1], copy=True)
            states[:, 0] = (
                math.cos(theta) * first + math.sin(theta) * second
            )
            states[:, 1] = (
                -math.sin(theta) * first + math.cos(theta) * second
            )
            states.setflags(write=False)
            evidence["mutated_source"] = np.array(
                states, dtype=np.float64, copy=True, order="C"
            )
            return metadata, audit

        def count_solver_calls(*args, **kwargs):
            evidence["solver_calls"] = int(evidence["solver_calls"]) + 1
            return original_direct(*args, **kwargs)

        with _patched_attribute(
            "_trusted_source_snapshot_audit", mutate_after_authority
        ):
            with _patched_attribute(
                "certify_minimum_error_from_weighted_states",
                count_solver_calls,
            ):
                try:
                    row = _diagnostic_row(
                        n,
                        0.5,
                        schedule=f"post_authority_rotation_n{n}_{theta}",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
                except Exception as exc:
                    self.fail(
                        "post-authority mutation must fail closed, not escape: "
                        f"{exc!r}"
                    )
        self.assertEqual(evidence["calls"], 1)
        self.assertEqual(evidence["solver_calls"], 0)
        return row, evidence

    def _unsafe_stale_padding(
        self, evidence: dict[str, object]
    ) -> tuple[float, float]:
        """Reproduce the unsafe mixed-source enclosure with real artifacts."""
        source = np.asarray(evidence["mutated_source"], dtype=float)
        certificate = (
            diagnostic_probe.certify_minimum_error_from_weighted_states(
                source,
                solver="CLARABEL",
            )
        )
        artifact_source = np.asarray(
            certificate["source_weighted_states"], dtype=float
        )
        effects = tuple(
            np.asarray(effect, dtype=float)
            for effect in certificate["repaired_primal_povm"]
        )
        rhos = [
            np.outer(artifact_source[:, index], artifact_source[:, index])
            for index in range(artifact_source.shape[1])
        ]
        raw_lower = math.fsum(
            float(np.trace(rho @ effect))
            for rho, effect in zip(rhos, effects)
        )
        raw_upper = float(
            np.trace(np.asarray(certificate["safe_dual_operator"], dtype=float))
        )
        stale_budget = float(evidence["stale_transfer_budget"])
        unsafe_lower = max(
            0.0,
            float(np.nextafter(raw_lower - stale_budget, -math.inf)),
        )
        unsafe_upper = min(
            1.0,
            float(np.nextafter(raw_upper + stale_budget, math.inf)),
        )
        return unsafe_lower, unsafe_upper

    def _row_after_post_authority_identity_mutation(
        self,
        mutation,
        *,
        label: str,
    ) -> dict[str, object]:
        original = diagnostic_probe._trusted_source_snapshot_audit
        calls = 0

        def mutate_identity_after_authority(gram, snapshot):
            nonlocal calls
            metadata, audit = original(gram, snapshot)
            states = np.asarray(snapshot)
            if states.flags.writeable:
                return metadata, audit
            calls += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                mutation(states)
            return metadata, audit

        with _patched_attribute(
            "_trusted_source_snapshot_audit",
            mutate_identity_after_authority,
        ):
            try:
                row = _diagnostic_row(
                    2,
                    0.5,
                    schedule=f"post_authority_identity_{label}",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )
            except Exception as exc:
                self.fail(
                    "snapshot identity mutation must fail closed: "
                    f"{label}: {exc!r}"
                )
        self.assertEqual(calls, 1)
        return row

    def _assert_snapshot_mutation_fails_closed(
        self, row: dict[str, object]
    ) -> None:
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_trusted_source_snapshot_mutated",
        )
        self.assertEqual(
            row.get("trusted_source_snapshot_validation_status"), "failed"
        )
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            with self.subTest(field=field):
                self.assertEqual(row.get(field), "")
        self.assertFalse(
            any(isinstance(value, np.ndarray) for value in row.values()),
            "fail-closed row leaked a private source or artifact array",
        )

    def test_post_authority_rotations_fail_closed_in_both_bound_directions(
        self,
    ) -> None:
        cases = (
            (2, -0.001, "upper", self.N2_TARGET_FEASIBLE_LOWER),
            (2, 0.001, "lower", self.N2_EXACT_DUAL_UPPER),
            (3, -0.001, "upper", self.N3_TARGET_FEASIBLE_LOWER),
        )
        for n, theta, direction, oracle in cases:
            with self.subTest(n=n, theta=theta, direction=direction):
                row, evidence = self._row_after_post_authority_rotation(
                    n=n,
                    theta=theta,
                )
                unsafe_lower, unsafe_upper = self._unsafe_stale_padding(
                    evidence
                )
                if row.get("sdp_status") == "computed_safe_enclosure":
                    self.assertNotEqual(
                        row.get("trusted_source_snapshot_sha256"),
                        row.get("certificate_artifact_source_states_sha256"),
                        "the vulnerable row exposes a broken identity chain",
                    )
                if direction == "upper":
                    self.assertLess(unsafe_upper, oracle)
                    self.assertGreater(oracle - unsafe_upper, 5.0e-6)
                else:
                    self.assertGreater(unsafe_lower, oracle)
                    self.assertGreater(unsafe_lower - oracle, 5.0e-6)
                self._assert_snapshot_mutation_fails_closed(row)

    def test_post_authority_identity_fields_are_all_rechecked(self) -> None:
        def mutate_shape(states):
            states.shape = (1, states.size)

        def mutate_dtype(states):
            states.dtype = np.float32

        def mutate_layout(states):
            states.strides = (states.strides[1], states.strides[0])

        def mutate_writeable(states):
            states.setflags(write=True)

        def mutate_nonfinite(states):
            states.setflags(write=True)
            states[0, 0] = math.nan
            states.setflags(write=False)

        cases = (
            ("shape", mutate_shape, "shape"),
            ("dtype", mutate_dtype, "dtype"),
            ("layout", mutate_layout, "layout"),
            ("writeable", mutate_writeable, "writeable"),
            ("nonfinite", mutate_nonfinite, "nonfinite"),
        )
        for label, mutation, reason_fragment in cases:
            with self.subTest(identity=label):
                row = self._row_after_post_authority_identity_mutation(
                    mutation,
                    label=label,
                )
                self._assert_snapshot_mutation_fails_closed(row)
                self.assertIn(
                    reason_fragment,
                    str(
                        row.get(
                            "trusted_source_snapshot_validation_reason", ""
                        )
                    ),
                )

    def test_snapshot_identity_is_rechecked_before_and_after_consumers(
        self,
    ) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit
        original_srm = diagnostic_probe.srm_quantities
        original_direct = (
            diagnostic_probe.certify_minimum_error_from_weighted_states
        )
        original_validator = diagnostic_probe._validate_certificate_artifacts

        for phase in ("before_solver", "after_solver", "after_validator"):
            with self.subTest(phase=phase):
                captured: dict[str, np.ndarray] = {}

                def capture_snapshot(gram, snapshot):
                    metadata, audit = original_helper(gram, snapshot)
                    states = np.asarray(snapshot)
                    if not states.flags.writeable:
                        captured["snapshot"] = states
                    return metadata, audit

                def drift_captured_snapshot() -> None:
                    states = captured["snapshot"]
                    states.setflags(write=True)
                    states[0, 0] = np.nextafter(states[0, 0], math.inf)
                    states.setflags(write=False)

                def srm_then_drift(*args, **kwargs):
                    result = original_srm(*args, **kwargs)
                    drift_captured_snapshot()
                    return result

                def solve_then_drift(*args, **kwargs):
                    certificate = original_direct(*args, **kwargs)
                    drift_captured_snapshot()
                    return certificate

                def validate_then_drift(*args, **kwargs):
                    result = original_validator(*args, **kwargs)
                    drift_captured_snapshot()
                    return result

                with _patched_attribute(
                    "_trusted_source_snapshot_audit", capture_snapshot
                ):
                    try:
                        if phase == "before_solver":
                            with _patched_attribute(
                                "srm_quantities", srm_then_drift
                            ):
                                row = _diagnostic_row(
                                    2,
                                    0.5,
                                    schedule=phase,
                                    max_peak_bytes=10**8,
                                    max_matrix_dimension=100,
                                )
                        elif phase == "after_solver":
                            with _patched_attribute(
                                "certify_minimum_error_from_weighted_states",
                                solve_then_drift,
                            ):
                                row = _diagnostic_row(
                                    2,
                                    0.5,
                                    schedule=phase,
                                    max_peak_bytes=10**8,
                                    max_matrix_dimension=100,
                                )
                        else:
                            with _patched_attribute(
                                "_validate_certificate_artifacts",
                                validate_then_drift,
                            ):
                                row = _diagnostic_row(
                                    2,
                                    0.5,
                                    schedule=phase,
                                    max_peak_bytes=10**8,
                                    max_matrix_dimension=100,
                                )
                    except Exception as exc:
                        self.fail(
                            "phase mutation must fail closed, not escape: "
                            f"{phase}: {exc!r}"
                        )
                self._assert_snapshot_mutation_fails_closed(row)
                self.assertEqual(
                    row.get("trusted_source_snapshot_validation_phase"),
                    phase,
                )

    def test_artifact_sha_must_equal_independently_captured_snapshot_sha(
        self,
    ) -> None:
        original = diagnostic_probe._validate_certificate_artifacts

        def poison_recomputed_artifact_sha(*args, **kwargs):
            valid, reason, metrics = original(*args, **kwargs)
            metrics = dict(metrics)
            if valid:
                metrics["certificate_artifact_source_states_sha256"] = (
                    "0" * 64
                )
            return valid, reason, metrics

        with _patched_attribute(
            "_validate_certificate_artifacts",
            poison_recomputed_artifact_sha,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="independent_artifact_sha_binding",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_artifact_validation_failed",
        )
        self.assertEqual(
            row.get("certificate_artifact_validation_status"), "failed"
        )
        self.assertIn(
            "trusted_snapshot_sha",
            str(row.get("certificate_artifact_validation_reason", "")),
        )
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            self.assertEqual(row.get(field), "")

    def test_helper_reported_sha_must_match_caller_owned_authority(self) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit
        original_direct = (
            diagnostic_probe.certify_minimum_error_from_weighted_states
        )
        solver_calls = 0

        def forge_only_helper_sha(gram, snapshot):
            metadata, audit = original_helper(gram, snapshot)
            forged = dict(audit)
            forged["trusted_source_snapshot_sha256"] = "0" * 64
            return metadata, forged

        def count_direct_calls(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return original_direct(*args, **kwargs)

        with _patched_attribute(
            "_trusted_source_snapshot_audit", forge_only_helper_sha
        ):
            with _patched_attribute(
                "certify_minimum_error_from_weighted_states",
                count_direct_calls,
            ):
                row = _diagnostic_row(
                    2,
                    0.5,
                    schedule="forged_helper_source_sha",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )

        self.assertEqual(solver_calls, 0)
        self._assert_snapshot_mutation_fails_closed(row)
        self.assertIn(
            "helper_sha",
            str(row.get("trusted_source_snapshot_validation_reason", "")),
        )

    def test_helper_must_not_omit_caller_authority_sha(self) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit
        original_direct = (
            diagnostic_probe.certify_minimum_error_from_weighted_states
        )
        solver_calls = 0

        def omit_helper_sha(gram, snapshot):
            metadata, audit = original_helper(gram, snapshot)
            missing = dict(audit)
            missing.pop("trusted_source_snapshot_sha256")
            return metadata, missing

        def count_direct_calls(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return original_direct(*args, **kwargs)

        with _patched_attribute(
            "_trusted_source_snapshot_audit", omit_helper_sha
        ):
            with _patched_attribute(
                "certify_minimum_error_from_weighted_states",
                count_direct_calls,
            ):
                row = _diagnostic_row(
                    2,
                    0.5,
                    schedule="missing_helper_source_sha",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )

        self.assertEqual(solver_calls, 0)
        self._assert_snapshot_mutation_fails_closed(row)
        self.assertIn(
            "helper_sha",
            str(row.get("trusted_source_snapshot_validation_reason", "")),
        )

    def test_malformed_helper_return_cannot_bypass_mutation_fail_closed(
        self,
    ) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit

        class RaisingGetDict(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError("malformed audit virtual get")

        malformed_factories = (
            ("list", lambda audit: []),
            ("dict_subclass", lambda audit: RaisingGetDict(audit)),
        )
        for label, malformed_factory in malformed_factories:
            with self.subTest(malformed=label):
                def mutate_then_return_malformed_audit(gram, snapshot):
                    metadata, audit = original_helper(gram, snapshot)
                    states = np.asarray(snapshot)
                    if states.flags.writeable:
                        return metadata, malformed_factory(audit)
                    states.setflags(write=True)
                    states[0, 0] = np.nextafter(states[0, 0], math.inf)
                    states.setflags(write=False)
                    return metadata, malformed_factory(audit)

                with _patched_attribute(
                    "_trusted_source_snapshot_audit",
                    mutate_then_return_malformed_audit,
                ):
                    try:
                        row = _diagnostic_row(
                            2,
                            0.5,
                            schedule=(
                                "malformed_helper_after_source_mutation_"
                                f"{label}"
                            ),
                            max_peak_bytes=10**8,
                            max_matrix_dimension=100,
                        )
                    except Exception as exc:
                        self.fail(
                            "malformed helper output must not bypass mutation "
                            f"fail-closed handling: {label}: {exc!r}"
                        )

                self._assert_snapshot_mutation_fails_closed(row)

    def test_helper_exception_precedence_distinguishes_identity_drift(
        self,
    ) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit

        for mutate_snapshot in (False, True):
            with self.subTest(mutate_snapshot=mutate_snapshot):
                def raise_after_real_helper(gram, snapshot):
                    metadata, audit = original_helper(gram, snapshot)
                    states = np.asarray(snapshot)
                    if states.flags.writeable:
                        return metadata, audit
                    if mutate_snapshot:
                        states.setflags(write=True)
                        states[0, 0] = np.nextafter(states[0, 0], math.inf)
                        states.setflags(write=False)
                    raise RuntimeError("authoritative helper exception")

                with _patched_attribute(
                    "_trusted_source_snapshot_audit",
                    raise_after_real_helper,
                ):
                    row = _diagnostic_row(
                        2,
                        0.5,
                        schedule=f"helper_exception_mutate_{mutate_snapshot}",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
                if mutate_snapshot:
                    self._assert_snapshot_mutation_fails_closed(row)
                else:
                    self.assertEqual(
                        row.get("sdp_status"),
                        "not_computed_source_transfer_exception",
                    )
                    self.assertEqual(
                        row.get("trusted_source_snapshot_validation_status"),
                        "passed",
                    )
                    for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
                        self.assertEqual(row.get(field), "")

    def test_physical_gram_cannot_drift_after_source_authority_audit(
        self,
    ) -> None:
        original_helper = diagnostic_probe._trusted_source_snapshot_audit
        original_direct = (
            diagnostic_probe.certify_minimum_error_from_weighted_states
        )
        solver_calls = 0

        def mutate_physical_gram_after_helper(gram, snapshot):
            metadata, audit = original_helper(gram, snapshot)
            physical = np.asarray(gram)
            physical.setflags(write=True)
            physical *= 1.0 - 2.0e-6
            physical.setflags(write=False)
            return metadata, audit

        def count_direct_calls(*args, **kwargs):
            nonlocal solver_calls
            solver_calls += 1
            return original_direct(*args, **kwargs)

        with _patched_attribute(
            "_trusted_source_snapshot_audit",
            mutate_physical_gram_after_helper,
        ):
            with _patched_attribute(
                "certify_minimum_error_from_weighted_states",
                count_direct_calls,
            ):
                try:
                    row = _diagnostic_row(
                        2,
                        0.5,
                        schedule="post_authority_physical_gram_mutation",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
                except Exception as exc:
                    self.fail(
                        "physical-Gram mutation must fail closed: "
                        f"{exc!r}"
                    )

        self.assertEqual(solver_calls, 0)
        self.assertNotEqual(row.get("sdp_status"), "computed_safe_enclosure")
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            self.assertEqual(row.get(field), "")
        self.assertFalse(
            any(isinstance(value, np.ndarray) for value in row.values()),
            "fail-closed row leaked a private source or artifact array",
        )

    def test_unmodified_post_authority_snapshot_computes_with_one_sha(self) -> None:
        for n in (2, 3):
            with self.subTest(n=n):
                row = _diagnostic_row(
                    n,
                    0.5,
                    schedule=f"unmodified_post_authority_n{n}",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )
                self.assertEqual(
                    row.get("sdp_status"), "computed_safe_enclosure"
                )
                self.assertNotEqual(
                    row.get("trusted_source_snapshot_sha256"), ""
                )
                self.assertEqual(
                    row.get("trusted_source_snapshot_sha256"),
                    row.get("certificate_artifact_source_states_sha256"),
                )


@unittest.skip("Route B does not publish certificate input metadata.")
class CertificateInputMutationTests(unittest.TestCase):
    def test_unmodified_solver_input_computes_with_identity_audit(self) -> None:
        original_validator = diagnostic_probe._validate_certificate_artifacts
        storage: dict[str, np.ndarray] = {}

        def capturing_validator(snapshot, solver_input, certificate):
            storage["snapshot"] = snapshot
            storage["solver_input"] = solver_input
            storage["artifact"] = np.asarray(
                certificate["source_weighted_states"]
            )
            storage["gram"] = np.asarray(
                certificate["source_weighted_gram"]
            )
            storage["effects"] = tuple(
                np.asarray(effect)
                for effect in certificate["repaired_primal_povm"]
            )
            storage["dual"] = np.asarray(certificate["safe_dual_operator"])
            return original_validator(snapshot, solver_input, certificate)

        with _patched_attribute(
            "_validate_certificate_artifacts", capturing_validator
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="unmodified_solver_input",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertEqual(
            row.get("certificate_input_factor_validation_status"), "passed"
        )
        self.assertEqual(
            row.get("certificate_input_factor_validation_reason"), "passed"
        )
        self.assertEqual(
            row.get("certificate_input_factor_pre_writeable"), "true"
        )
        self.assertEqual(
            row.get("certificate_input_factor_post_writeable"), "true"
        )
        source_hashes = {
            str(row.get("trusted_source_snapshot_sha256")),
            str(row.get("trusted_source_snapshot_pre_sha256")),
            str(row.get("trusted_source_snapshot_post_sha256")),
            str(row.get("certificate_input_factor_pre_sha256")),
            str(row.get("certificate_input_factor_post_sha256")),
            str(row.get("certificate_artifact_source_states_sha256")),
        }
        self.assertEqual(len(source_hashes), 1)
        self.assertNotIn("", source_hashes)
        self.assertFalse(storage["snapshot"].flags.writeable)
        self.assertFalse(
            np.shares_memory(storage["snapshot"], storage["solver_input"])
        )
        artifacts = [
            ("source_states", storage["artifact"]),
            ("source_gram", storage["gram"]),
            ("dual", storage["dual"]),
        ]
        artifacts.extend(
            (f"effect_{index}", effect)
            for index, effect in enumerate(storage["effects"])
        )
        for name, artifact in artifacts:
            with self.subTest(artifact=name):
                self.assertFalse(
                    np.shares_memory(artifact, storage["snapshot"])
                )
                self.assertFalse(
                    np.shares_memory(artifact, storage["solver_input"])
                )

    def _assert_real_mutating_callee_fails_closed(
        self,
        mutate,
        *,
        schedule: str,
    ) -> dict[str, float | int | str]:
        """Exercise the real direct solver after an in-place caller mutation."""
        original = diagnostic_probe.certify_minimum_error_from_weighted_states
        calls = 0

        def mutating_direct(source_states, *args, **kwargs):
            nonlocal calls
            states = np.asarray(source_states)
            if states.shape == (3, 3):
                calls += 1
                mutate(states)
            return original(source_states, *args, **kwargs)

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            mutating_direct,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule=schedule,
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )

        self.assertEqual(calls, 1, "conditional real-solver mutant did not fire")
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_input_factor_mutated",
        )
        self.assertEqual(
            row.get("certificate_input_factor_validation_status"), "failed"
        )
        self.assertNotEqual(
            row.get("certificate_input_factor_validation_reason"), "passed"
        )
        self.assertIn(
            row.get("certificate_artifact_validation_status"),
            {"", "not_computed_certificate_input_factor_mutated"},
        )
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            with self.subTest(field=field):
                self.assertEqual(row.get(field), "")
        self.assertFalse(
            any(isinstance(value, np.ndarray) for value in row.values()),
            "fail-closed diagnostic row leaked a private matrix artifact",
        )
        for field in (
            "certificate_artifact_source_states_sha256",
            "certificate_artifact_source_gram_sha256",
            "certificate_artifact_primal_objective",
            "certificate_artifact_dual_trace",
        ):
            self.assertEqual(row.get(field), "")
        return row

    def test_in_place_hypothesis_column_rotation_fails_closed(self) -> None:
        """Catches mixing an old transfer budget with a mutated solver source."""
        for theta in (-0.001, 0.1):
            with self.subTest(theta=theta):
                def rotate_hypothesis_columns(states, theta=theta):
                    first = np.array(states[:, 0], copy=True)
                    second = np.array(states[:, 1], copy=True)
                    states[:, 0] = (
                        math.cos(theta) * first + math.sin(theta) * second
                    )
                    states[:, 1] = (
                        -math.sin(theta) * first + math.cos(theta) * second
                    )

                self._assert_real_mutating_callee_fails_closed(
                    rotate_hypothesis_columns,
                    schedule=f"mutated_hypothesis_columns_{theta}",
                )

    def test_in_place_physical_row_rotation_fails_closed(self) -> None:
        """Enforce strict source identity even when a left rotation preserves Gram."""
        theta = 0.1

        def rotate_physical_rows(states):
            first = np.array(states[0, :], copy=True)
            second = np.array(states[1, :], copy=True)
            states[0, :] = math.cos(theta) * first + math.sin(theta) * second
            states[1, :] = -math.sin(theta) * first + math.cos(theta) * second

        self._assert_real_mutating_callee_fails_closed(
            rotate_physical_rows,
            schedule="mutated_physical_rows",
        )

    def test_returned_source_artifact_must_not_alias_solver_input(self) -> None:
        """Catches a certificate returning the caller-owned solver array itself."""
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def aliasing_direct(source_states, *args, **kwargs):
            certificate = dict(original(source_states, *args, **kwargs))
            certificate["source_weighted_states"] = source_states
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            aliasing_direct,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="aliased_source_artifact",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_artifact_validation_failed",
        )
        self.assertEqual(
            row.get("certificate_artifact_validation_status"), "failed"
        )
        self.assertIn(
            "alias",
            str(row.get("certificate_artifact_validation_reason", "")),
        )
        self.assertEqual(row.get("L_safe"), "")
        self.assertEqual(row.get("U_safe"), "")

    def test_solver_input_writeable_flag_mutation_fails_closed(self) -> None:
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def changes_writeable_flag(source_states, *args, **kwargs):
            certificate = original(source_states, *args, **kwargs)
            np.asarray(source_states).setflags(write=False)
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            changes_writeable_flag,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="solver_input_writeable_flag_mutation",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_input_factor_mutated",
        )
        self.assertEqual(
            row.get("certificate_input_factor_validation_status"), "failed"
        )
        self.assertIn(
            "writeable",
            str(row.get("certificate_input_factor_validation_reason", "")),
        )
        self.assertEqual(row.get("L_safe"), "")
        self.assertEqual(row.get("U_safe"), "")

    def test_memoryview_artifacts_cannot_alias_caller_storage(self) -> None:
        """At n=1 every 1x1 artifact can algebraically equal solver_input."""
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def make_aliasing_direct(artifact_name: str):
            def aliasing_direct(source_states, *args, **kwargs):
                certificate = dict(original(source_states, *args, **kwargs))
                alias = memoryview(source_states)
                if artifact_name == "source_states":
                    certificate["source_weighted_states"] = alias
                elif artifact_name == "source_gram":
                    certificate["source_weighted_gram"] = alias
                elif artifact_name == "primal_effect":
                    certificate["repaired_primal_povm"] = (alias,)
                elif artifact_name == "dual":
                    certificate["safe_dual_operator"] = alias
                else:
                    raise AssertionError(artifact_name)
                return certificate

            return aliasing_direct

        for artifact_name in (
            "source_states",
            "source_gram",
            "primal_effect",
            "dual",
        ):
            with self.subTest(artifact=artifact_name):
                with _patched_attribute(
                    "certify_minimum_error_from_weighted_states",
                    make_aliasing_direct(artifact_name),
                ):
                    row = _diagnostic_row(
                        1,
                        0.5,
                        schedule=f"memoryview_alias_{artifact_name}",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
                self.assertEqual(
                    row.get("sdp_status"),
                    "not_computed_certificate_artifact_validation_failed",
                )
                self.assertIn(
                    "alias",
                    str(
                        row.get(
                            "certificate_artifact_validation_reason", ""
                        )
                    ),
                )
                self.assertEqual(row.get("L_safe"), "")
                self.assertEqual(row.get("U_safe"), "")


@unittest.skip("Route B primary fields are covered by RouteBFallbackTests.")
class ArtifactDerivedPrimaryFieldTests(unittest.TestCase):
    def test_tolerated_self_report_offsets_do_not_enter_primary_fields(self) -> None:
        """Catches using tolerated certificate scalars instead of actual artifacts."""
        original = diagnostic_probe.certify_minimum_error_from_weighted_states
        expected: dict[str, float] = {}
        reported: dict[str, float] = {}
        delta = 4.0e-13

        def offset_self_reports(source_states, *args, **kwargs):
            certificate = dict(original(source_states, *args, **kwargs))
            states = np.asarray(certificate["source_weighted_states"], dtype=float)
            effects = tuple(
                np.asarray(effect, dtype=float)
                for effect in certificate["repaired_primal_povm"]
            )
            rank, hypothesis_count = states.shape
            rhos = [
                np.outer(states[:, index], states[:, index])
                for index in range(hypothesis_count)
            ]
            completeness = (
                sum(effects, np.zeros((rank, rank), dtype=float)) - np.eye(rank)
            )
            primal_minimum = min(
                float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
                for effect in effects
            )
            dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
            dual_symmetric = (dual + dual.T) / 2.0
            dual_minimum_slack = min(
                float(
                    np.linalg.eigvalsh(
                        (dual_symmetric - rho + (dual_symmetric - rho).T)
                        / 2.0
                    )[0]
                )
                for rho in rhos
            )
            expected.update(
                {
                    "L_raw": math.fsum(
                        float(np.trace(rho @ effect))
                        for rho, effect in zip(rhos, effects)
                    ),
                    "U_raw": float(np.trace(dual_symmetric)),
                    "primal_feasible_equality_residual_fro": float(
                        np.linalg.norm(completeness, ord="fro")
                    ),
                    "primal_feasible_equality_residual_op": float(
                        np.linalg.norm(completeness, ord=2)
                    ),
                    "primal_feasible_psd_violation": max(
                        0.0, -primal_minimum
                    ),
                    "primal_feasible_min_eigenvalue": primal_minimum,
                    "dual_feasible_min_slack": dual_minimum_slack,
                    "dual_feasible_psd_violation": max(
                        0.0, -dual_minimum_slack
                    ),
                }
            )
            for field in (
                "primal_feasible_equality_residual_fro",
                "primal_feasible_equality_residual_op",
                "primal_feasible_psd_violation",
                "primal_feasible_min_eigenvalue",
                "dual_feasible_min_slack",
                "dual_feasible_psd_violation",
            ):
                certificate[field] = float(certificate[field]) + delta
                reported[field] = float(certificate[field])
            lower = float(certificate["primal_feasible_objective"]) + delta
            upper = float(certificate["dual_feasible_objective"]) + delta
            certificate["primal_feasible_objective"] = lower
            certificate["dual_feasible_objective"] = upper
            certificate["feasible_bound_gap"] = upper - lower
            certificate["relative_feasible_bound_gap"] = (upper - lower) / max(
                np.finfo(float).tiny,
                0.5 * (abs(lower) + abs(upper)),
            )
            reported["L_raw"] = lower
            reported["U_raw"] = upper
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            offset_self_reports,
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="artifact_derived_primary_fields",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )

        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertEqual(
            row.get("certificate_artifact_validation_status"), "passed"
        )
        for field in expected:
            with self.subTest(field=field):
                self.assertTrue(math.isfinite(float(row[field])))
                self.assertNotEqual(float(row[field]), reported[field])
        self.assertLessEqual(float(row["L_raw"]), float(row["U_raw"]))
        self.assertLessEqual(
            float(row["primal_feasible_equality_residual_op"]),
            64.0 * np.finfo(float).eps,
        )
        self.assertGreaterEqual(
            float(row["primal_feasible_min_eigenvalue"]), 0.0
        )
        self.assertGreaterEqual(float(row["dual_feasible_min_slack"]), 0.0)


@unittest.skip("Route B exposes repaired/shifted floating diagnostics, not bounds.")
class CanonicalFeasibilityRecomputationTests(unittest.TestCase):
    """Caller-side bounds must come from newly feasible artifacts."""

    @staticmethod
    def _row_with_certificate_transform(transform, schedule: str):
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def transformed(source_states, *args, **kwargs):
            certificate = dict(original(source_states, *args, **kwargs))
            return transform(certificate)

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states", transformed
        ):
            return _diagnostic_row(
                1,
                0.5,
                schedule=schedule,
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )

    def test_tolerance_size_negative_dual_slack_is_shifted_before_upper_bound(
        self,
    ) -> None:
        """Catches treating a small negative slack as an actually feasible dual."""

        def shift_dual_down(certificate):
            dual = np.asarray(
                certificate["safe_dual_operator"], dtype=float
            )
            shifted = (dual + dual.T) / 2.0 - 5.0e-10 * np.eye(
                dual.shape[0]
            )
            source = np.asarray(
                certificate["source_weighted_states"], dtype=float
            )
            rhos = [
                np.outer(source[:, index], source[:, index])
                for index in range(source.shape[1])
            ]
            minimum_slack = min(
                float(np.linalg.eigvalsh(shifted - rho)[0])
                for rho in rhos
            )
            upper = float(np.trace(shifted))
            lower = float(certificate["primal_feasible_objective"])
            gap = upper - lower
            certificate["safe_dual_operator"] = shifted
            certificate["dual_feasible_objective"] = upper
            certificate["dual_feasible_min_slack"] = minimum_slack
            certificate["dual_feasible_psd_violation"] = max(
                0.0, -minimum_slack
            )
            certificate["feasible_bound_gap"] = gap
            certificate["relative_feasible_bound_gap"] = gap / max(
                np.finfo(float).tiny,
                0.5 * (abs(lower) + abs(upper)),
            )
            return certificate

        row = self._row_with_certificate_transform(
            shift_dual_down, "repair_tolerance_size_negative_dual_slack"
        )
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertGreaterEqual(
            float(row["U_safe"]),
            float(row["strongest_certified_lower"]),
        )
        self.assertGreaterEqual(float(row["U_raw"]), 1.0)
        self.assertGreaterEqual(float(row["dual_feasible_min_slack"]), 0.0)

    def test_tolerance_size_primal_overcompleteness_is_contracted_and_filled(
        self,
    ) -> None:
        """Catches reporting an overcomplete PSD family as a feasible POVM."""

        def inflate_primal(certificate):
            source = np.asarray(
                certificate["source_weighted_states"], dtype=float
            )
            effects = tuple(
                (1.0 + 5.0e-10) * np.asarray(effect, dtype=float)
                for effect in certificate["repaired_primal_povm"]
            )
            rhos = [
                np.outer(source[:, index], source[:, index])
                for index in range(source.shape[1])
            ]
            lower = math.fsum(
                float(np.trace(rho @ effect))
                for rho, effect in zip(rhos, effects)
            )
            completeness = sum(effects) - np.eye(source.shape[0])
            upper = float(certificate["dual_feasible_objective"])
            gap = upper - lower
            certificate["repaired_primal_povm"] = effects
            certificate["primal_feasible_objective"] = lower
            certificate["primal_feasible_equality_residual_fro"] = float(
                np.linalg.norm(completeness, ord="fro")
            )
            certificate["primal_feasible_equality_residual_op"] = float(
                np.linalg.norm(completeness, ord=2)
            )
            minimum_effect_eigenvalue = min(
                float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
                for effect in effects
            )
            certificate["primal_feasible_min_eigenvalue"] = (
                minimum_effect_eigenvalue
            )
            certificate["primal_feasible_psd_violation"] = max(
                0.0, -minimum_effect_eigenvalue
            )
            certificate["feasible_bound_gap"] = gap
            certificate["relative_feasible_bound_gap"] = gap / max(
                np.finfo(float).tiny,
                0.5 * (abs(lower) + abs(upper)),
            )
            return certificate

        row = self._row_with_certificate_transform(
            inflate_primal, "repair_tolerance_size_primal_overcompleteness"
        )
        self.assertEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertLessEqual(float(row["L_raw"]), 1.0)
        self.assertLessEqual(float(row["L_safe"]), float(row["U_safe"]))
        self.assertLessEqual(
            float(row["primal_feasible_equality_residual_op"]),
            64.0 * np.finfo(float).eps,
        )


@unittest.skip("Superseded by the Route B pinned-core behavior tests.")
class CallerAuthorityBindingTests(unittest.TestCase):
    """The physical/source/artifact truth paths must be caller-owned."""

    def test_dynamic_poisoning_of_both_validator_names_cannot_set_final_bounds(
        self,
    ) -> None:
        """Catches final recomputation through a dynamically replaceable global."""
        original = diagnostic_probe._independently_validate_certificate_artifacts

        def poisoned(*args, **kwargs):
            valid, reason, metrics = original(*args, **kwargs)
            poisoned_metrics = dict(metrics)
            poisoned_metrics["certificate_artifact_primal_objective"] = 0.99
            poisoned_metrics["certificate_artifact_dual_trace"] = 0.99
            return valid, reason, poisoned_metrics

        with _patched_attribute(
            "_independently_validate_certificate_artifacts", poisoned
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="poison_both_dynamic_validator_names",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        for field in ("L_raw", "U_raw", "L_safe", "U_safe"):
            if row.get(field) != "":
                self.assertNotEqual(float(row[field]), 0.99)

    def test_alternate_psd_unit_diagonal_gram_is_rejected_against_n_c_formula(
        self,
    ) -> None:
        """Catches accepting a helper-selected ensemble with the right shape only."""
        original = diagnostic_probe._hull_gram_with_audit

        def alternate_gram(n, c):
            stable, physical, audit = original(n, c)
            del physical
            return stable, np.eye(n * (n + 1) // 2), audit

        with _patched_attribute("_hull_gram_with_audit", alternate_gram):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="alternate_psd_unit_diagonal_gram",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertNotEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertIn(
            "physical_gram_formula_mismatch",
            str(row.get("certificate_validation_reason", "")),
        )

    def test_negative_source_transfer_budget_is_rejected_by_canonical_recompute(
        self,
    ) -> None:
        """Catches trusting a helper ledger with budget < delta or budget < 0."""
        original = diagnostic_probe._trusted_source_snapshot_audit

        def negative_budget(*args, **kwargs):
            metadata, audit = original(*args, **kwargs)
            audit = dict(audit)
            audit["source_transfer_budget"] = -1.0e-12
            audit["source_transfer_status"] = "passed"
            return metadata, audit

        with _patched_attribute(
            "_trusted_source_snapshot_audit", negative_budget
        ):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="negative_source_transfer_budget",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertNotEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertIn(
            "source_transfer_ledger_mismatch",
            str(row.get("certificate_validation_reason", "")),
        )


class RouteBFallbackTests(unittest.TestCase):
    """Public Task 8 output is a floating diagnostic, never a bound claim."""

    PRIMARY_DIAGNOSTIC_FIELDS = (
        "repaired_primal_value",
        "shifted_dual_value",
        "strongest_measurement_value",
        "floating_primal_dual_span",
        "floating_primal_dual_span_relative",
        "repaired_primal_over_p1_squared",
        "shifted_dual_over_p1_squared",
        "strongest_measurement_over_p1_squared",
        "repaired_primal_over_P_SRM",
        "P_SRM_over_shifted_dual",
        "primal_completeness_residual_fro",
        "primal_completeness_residual_op",
        "primal_psd_violation",
        "primal_min_eigenvalue",
        "dual_min_slack_after_shift",
        "dual_psd_violation_after_shift",
    )

    ARTIFACT_VALIDATOR = staticmethod(
        diagnostic_probe._independently_validate_certificate_artifacts
    )

    @staticmethod
    def _refresh_artifact_self_reports(certificate: dict[str, object]) -> None:
        """Keep raw solver self-reports consistent after a test perturbation."""
        source = np.asarray(certificate["source_weighted_states"], dtype=float)
        effects = tuple(
            np.asarray(effect, dtype=float)
            for effect in certificate["repaired_primal_povm"]
        )
        rank, hypothesis_count = source.shape
        rhos = [
            np.outer(source[:, index], source[:, index])
            for index in range(hypothesis_count)
        ]
        completeness = (
            sum(effects, np.zeros((rank, rank), dtype=float)) - np.eye(rank)
        )
        primal_minimum = min(
            float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
            for effect in effects
        )
        primal_objective = math.fsum(
            float(np.trace(rho @ effect))
            for rho, effect in zip(rhos, effects)
        )
        dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
        dual = (dual + dual.T) / 2.0
        dual_minimum = min(
            float(np.linalg.eigvalsh(dual - rho)[0]) for rho in rhos
        )
        dual_trace = float(np.trace(dual))
        gap = dual_trace - primal_objective
        scale = max(
            np.finfo(float).tiny,
            0.5 * (abs(primal_objective) + abs(dual_trace)),
        )
        certificate.update(
            {
                "primal_feasible_objective": primal_objective,
                "dual_feasible_objective": dual_trace,
                "feasible_bound_gap": gap,
                "relative_feasible_bound_gap": gap / scale,
                "primal_feasible_equality_residual_fro": float(
                    np.linalg.norm(completeness, ord="fro")
                ),
                "primal_feasible_equality_residual_op": float(
                    np.linalg.norm(completeness, ord=2)
                ),
                "primal_feasible_psd_violation": max(
                    0.0, -primal_minimum
                ),
                "primal_feasible_min_eigenvalue": primal_minimum,
                "dual_feasible_min_slack": dual_minimum,
                "dual_feasible_psd_violation": max(0.0, -dual_minimum),
            }
        )

    def _row_with_artifact_transform(
        self,
        transform,
        *,
        schedule: str,
        artifact_validator=None,
    ) -> dict[str, object]:
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def transformed(source_states, *args, **kwargs):
            certificate = dict(original(source_states, *args, **kwargs))
            transform(certificate)
            self._refresh_artifact_self_reports(certificate)
            return certificate

        validator = (
            self.ARTIFACT_VALIDATOR
            if artifact_validator is None
            else artifact_validator
        )
        with _patched_attribute(
            "certify_minimum_error_from_weighted_states", transformed
        ):
            return _diagnostic_row(
                2,
                0.5,
                schedule=schedule,
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
                _canonical_artifact_validator=validator,
            )

    def test_public_sdp_row_uses_diagnostic_values_and_blanks_bound_aliases(
        self,
    ) -> None:
        """Catches publishing residual-checked floats as safe/certified bounds."""
        row = _diagnostic_row(
            2,
            0.5,
            schedule="route_b_public_semantics",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        self.assertEqual(row.get("status"), "sdp_floating_diagnostic")
        self.assertEqual(
            row.get("sdp_status"), "computed_residual_checked_diagnostic"
        )
        primal = float(row["repaired_primal_value"])
        dual = float(row["shifted_dual_value"])
        strongest = float(row["strongest_measurement_value"])
        self.assertLessEqual(primal, strongest)
        self.assertLessEqual(strongest, dual)
        self.assertEqual(strongest, max(primal, float(row["P_SRM"])))
        self.assertEqual(
            float(row["floating_primal_dual_span"]), dual - primal
        )
        self.assertEqual(row.get("P_opt"), "")
        self.assertEqual(row.get("P_opt_status"), "not_computed")
        for field in (
            "L_raw",
            "U_raw",
            "L_safe",
            "U_safe",
            "strongest_certified_lower",
            "raw_safe_gap",
            "raw_safe_gap_relative",
            "safe_gap",
            "safe_gap_relative",
            "P_SRM_safe_lower",
            "enclosure_kind",
        ):
            with self.subTest(legacy_field=field):
                self.assertEqual(row.get(field), "")
        self.assertEqual(
            row.get("legacy_bound_fields_status"), "deprecated_blank"
        )
        for field, value in row.items():
            if field.startswith("certificate_"):
                with self.subTest(certificate_field=field):
                    self.assertEqual(value, "")

    def test_deprecated_validation_facades_are_outside_public_truth_path(
        self,
    ) -> None:
        """Catches re-entering patchable double-validation or ledger facades."""

        def deprecated_bomb(*args, **kwargs):
            raise AssertionError("deprecated facade executed")

        names = (
            "_trusted_source_snapshot_audit",
            "_validate_safe_certificate",
            "_validate_certificate_artifacts",
            "_certificate_binds_to_source",
            "_artifact_metric_ledgers_match",
        )
        contexts = []
        try:
            for name in names:
                context = _patched_attribute(name, deprecated_bomb)
                context.__enter__()
                contexts.append(context)
            row = _diagnostic_row(
                2,
                0.5,
                schedule="route_b_deprecated_facades",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        finally:
            for context in reversed(contexts):
                context.__exit__(None, None, None)
        self.assertEqual(row.get("status"), "sdp_floating_diagnostic")

    def test_pinned_source_and_artifact_cores_each_execute_once(self) -> None:
        """Catches accidental duplicate source or artifact truth passes."""
        source_core = diagnostic_probe._spectral_reconstruction_audit
        artifact_core = (
            diagnostic_probe._independently_validate_certificate_artifacts
        )
        calls = {"source": 0, "artifact": 0}

        def counted_source(*args, **kwargs):
            calls["source"] += 1
            return source_core(*args, **kwargs)

        def counted_artifact(*args, **kwargs):
            calls["artifact"] += 1
            return artifact_core(*args, **kwargs)

        try:
            row = _diagnostic_row(
                2,
                0.5,
                schedule="route_b_single_truth_pass",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
                _canonical_source_builder=counted_source,
                _canonical_artifact_validator=counted_artifact,
            )
        except TypeError as exc:
            self.fail(f"single-pass dependency injection is missing: {exc}")
        self.assertEqual(row.get("status"), "sdp_floating_diagnostic")
        self.assertEqual(calls, {"source": 1, "artifact": 1})

    def test_route_b_values_have_recomputed_residuals_and_ratios(self) -> None:
        row = _diagnostic_row(
            3,
            0.5,
            schedule="route_b_residuals",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        primal = float(row["repaired_primal_value"])
        dual = float(row["shifted_dual_value"])
        strongest = float(row["strongest_measurement_value"])
        target = float(row["p1_squared"])
        self.assertLess(float(row["primal_completeness_residual_op"]), 2e-11)
        self.assertLess(float(row["primal_psd_violation"]), 2e-11)
        self.assertLess(float(row["dual_psd_violation_after_shift"]), 2e-11)
        self.assertGreaterEqual(
            float(row["dual_min_slack_after_shift"]), -2e-11
        )
        self.assertAlmostEqual(
            float(row["repaired_primal_over_p1_squared"]),
            primal / target,
            places=13,
        )
        self.assertAlmostEqual(
            float(row["shifted_dual_over_p1_squared"]),
            dual / target,
            places=13,
        )
        self.assertAlmostEqual(
            float(row["strongest_measurement_over_p1_squared"]),
            strongest / target,
            places=13,
        )

    def test_physical_gram_is_checked_directly_against_n_c(self) -> None:
        original = diagnostic_probe._hull_gram_with_audit

        def alternate_gram(n, c):
            stable, _physical, audit = original(n, c)
            return stable, np.eye(n * (n + 1) // 2), audit

        with _patched_attribute("_hull_gram_with_audit", alternate_gram):
            row = _diagnostic_row(
                2,
                0.5,
                schedule="route_b_physical_authority",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("status"),
            "not_computed_physical_gram_formula_mismatch",
        )
        self.assertIn(
            "physical_gram_formula_mismatch",
            str(row.get("artifact_recomputation_reason", "")),
        )

    def test_source_ledger_is_canonically_recomputed_and_consistent(self) -> None:
        row = _diagnostic_row(
            3,
            0.5,
            schedule="route_b_source_ledger",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        delta = float(row["source_transfer_delta"])
        budget = float(row["source_transfer_budget"])
        self.assertGreaterEqual(delta, 0.0)
        self.assertGreaterEqual(budget, delta)
        trace_norm = float(row["source_difference_trace_norm"])
        fro_norm = float(row["source_difference_fro_norm"])
        op_norm = float(row["source_difference_op_norm"])
        self.assertGreaterEqual(trace_norm + 1e-15, fro_norm)
        self.assertGreaterEqual(fro_norm + 1e-15, op_norm)
        self.assertEqual(row.get("source_transfer_status"), "passed")
        self.assertEqual(row.get("spectral_audit_status"), "passed")
        self.assertEqual(int(row["source_gram_support_rank"]), 6)
        self.assertEqual(int(row["srm_sqrt_support_rank"]), 6)
        sqrt_error = float(row["source_sqrt_difference_computed_fro"])
        sqrt_bound = float(row["source_sqrt_difference_bound"])
        self.assertLessEqual(sqrt_error, sqrt_bound + 1e-15)

    def test_negative_dual_slack_uses_rank_multiplier_and_is_shifted(self) -> None:
        evidence: dict[str, float | int] = {}

        def make_negative_slack(certificate: dict[str, object]) -> None:
            dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
            rank = dual.shape[0]
            shifted_down = (dual + dual.T) / 2.0 - 5.0e-7 * np.eye(rank)
            source = np.asarray(
                certificate["source_weighted_states"], dtype=float
            )
            rhos = [
                np.outer(source[:, index], source[:, index])
                for index in range(source.shape[1])
            ]
            raw_minimum = min(
                float(np.linalg.eigvalsh(shifted_down - rho)[0])
                for rho in rhos
            )
            evidence.update(
                {
                    "rank": rank,
                    "raw_trace": float(np.trace(shifted_down)),
                    "raw_minimum": raw_minimum,
                }
            )
            certificate["safe_dual_operator"] = shifted_down

        row = self._row_with_artifact_transform(
            make_negative_slack,
            schedule="route_b_negative_dual_slack",
        )
        self.assertEqual(
            row.get("sdp_status"), "computed_residual_checked_diagnostic"
        )
        rank = int(evidence["rank"])
        raw_trace = float(evidence["raw_trace"])
        required_shift = max(0.0, -float(evidence["raw_minimum"]))
        self.assertGreater(required_shift, 0.0)
        trace_increment = float(row["shifted_dual_value"]) - raw_trace
        self.assertAlmostEqual(
            trace_increment,
            rank * required_shift,
            delta=2e-12,
        )
        self.assertGreater(
            abs(trace_increment - required_shift), 0.5 * required_shift
        )
        self.assertGreaterEqual(
            float(row["dual_min_slack_after_shift"]), 0.0
        )
        self.assertGreaterEqual(
            float(row["shifted_dual_value"]),
            float(row["strongest_measurement_value"]),
        )
        self.assertEqual(row.get("P_opt"), "")

    def test_overcomplete_primal_is_contracted_and_filled_by_remainder(
        self,
    ) -> None:
        evidence: dict[str, float] = {}

        def inflate_effects(certificate: dict[str, object]) -> None:
            source = np.asarray(
                certificate["source_weighted_states"], dtype=float
            )
            effects = tuple(
                (1.0 + 5.0e-7) * np.asarray(effect, dtype=float)
                for effect in certificate["repaired_primal_povm"]
            )
            rhos = [
                np.outer(source[:, index], source[:, index])
                for index in range(source.shape[1])
            ]
            evidence["raw_objective"] = math.fsum(
                float(np.trace(rho @ effect))
                for rho, effect in zip(rhos, effects)
            )
            evidence["raw_overcompleteness"] = float(
                np.linalg.eigvalsh(sum(effects))[-1] - 1.0
            )
            certificate["repaired_primal_povm"] = effects

        row = self._row_with_artifact_transform(
            inflate_effects,
            schedule="route_b_overcomplete_primal",
        )
        self.assertEqual(
            row.get("sdp_status"), "computed_residual_checked_diagnostic"
        )
        self.assertGreater(evidence["raw_overcompleteness"], 0.0)
        self.assertLessEqual(
            float(row["primal_completeness_residual_fro"]), 2e-11
        )
        self.assertLessEqual(
            float(row["primal_completeness_residual_op"]), 2e-11
        )
        self.assertLessEqual(float(row["primal_psd_violation"]), 2e-11)
        self.assertLess(
            float(row["repaired_primal_value"]), evidence["raw_objective"]
        )
        self.assertLessEqual(
            float(row["repaired_primal_value"]),
            float(row["shifted_dual_value"]),
        )
        self.assertEqual(row.get("P_opt"), "")

    def test_negative_effect_is_psd_projected_before_normalization(self) -> None:
        evidence: dict[str, float] = {}

        def make_effect_indefinite(certificate: dict[str, object]) -> None:
            effects = [
                np.array(effect, dtype=float, copy=True)
                for effect in certificate["repaired_primal_povm"]
            ]
            effects[1] = (effects[1] + effects[1].T) / 2.0
            effects[1] -= 5.0e-7 * np.eye(effects[1].shape[0])
            evidence["raw_minimum"] = float(
                np.linalg.eigvalsh(effects[1])[0]
            )
            certificate["repaired_primal_povm"] = tuple(effects)

        row = self._row_with_artifact_transform(
            make_effect_indefinite,
            schedule="route_b_negative_effect_projection",
        )
        self.assertEqual(
            row.get("sdp_status"), "computed_residual_checked_diagnostic"
        )
        self.assertLess(evidence["raw_minimum"], 0.0)
        self.assertGreaterEqual(float(row["primal_min_eigenvalue"]), 0.0)
        self.assertEqual(float(row["primal_psd_violation"]), 0.0)
        self.assertLessEqual(
            float(row["primal_completeness_residual_op"]), 2e-11
        )
        self.assertLessEqual(
            float(row["repaired_primal_value"]),
            float(row["shifted_dual_value"]),
        )
        self.assertEqual(row.get("P_opt"), "")

    def test_artifact_and_exception_failures_clear_all_route_b_values(
        self,
    ) -> None:
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        def corrupt_solver_artifact(source_states, *args, **kwargs):
            certificate = dict(original(source_states, *args, **kwargs))
            effects = list(certificate["repaired_primal_povm"])
            effects[0] = np.full_like(effects[0], np.nan)
            certificate["repaired_primal_povm"] = tuple(effects)
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            corrupt_solver_artifact,
        ):
            corrupt_row = _diagnostic_row(
                2,
                0.5,
                schedule="route_b_corrupt_artifact",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )

        def no_transform(certificate: dict[str, object]) -> None:
            del certificate

        def validator_exception(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("validator exploded")

        validator_row = self._row_with_artifact_transform(
            no_transform,
            schedule="route_b_validator_exception",
            artifact_validator=validator_exception,
        )

        def solver_exception(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("solver exploded")

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states", solver_exception
        ):
            solver_row = _diagnostic_row(
                2,
                0.5,
                schedule="route_b_solver_exception",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )

        for name, row in (
            ("corrupt_artifact", corrupt_row),
            ("validator_exception", validator_row),
            ("solver_exception", solver_row),
        ):
            with self.subTest(failure=name):
                self.assertTrue(
                    str(row.get("sdp_status", "")).startswith(
                        "not_computed_"
                    )
                )
                self.assertTrue(
                    str(row.get("diagnostic_kind", "")).startswith(
                        "not_computed_"
                    )
                )
                self.assertEqual(row.get("P_opt"), "")
                for field in self.PRIMARY_DIAGNOSTIC_FIELDS:
                    self.assertEqual(row.get(field), "")


@unittest.skip("Route B removed the deprecated certificate facade namespaces.")
class OrchestrationNamespaceFailClosedTests(unittest.TestCase):
    """Adversarial return values must never become trusted row fields."""

    _CLEARED_FIELDS = (
        "L_raw",
        "U_raw",
        "L_safe",
        "U_safe",
        "raw_safe_gap",
        "raw_safe_gap_relative",
        "safe_gap",
        "safe_gap_relative",
        "primal_feasible_equality_residual_fro",
        "primal_feasible_equality_residual_op",
        "primal_feasible_psd_violation",
        "primal_feasible_min_eigenvalue",
        "dual_feasible_min_slack",
        "dual_feasible_psd_violation",
        "certificate_artifact_source_states_max_abs_error",
        "certificate_artifact_source_gram_max_abs_error",
        "certificate_artifact_primal_symmetry_error_fro",
        "certificate_artifact_primal_completeness_fro",
        "certificate_artifact_primal_completeness_op",
        "certificate_artifact_primal_psd_violation",
        "certificate_artifact_primal_min_eigenvalue",
        "certificate_artifact_primal_objective",
        "certificate_artifact_dual_symmetry_error_fro",
        "certificate_artifact_dual_min_slack",
        "certificate_artifact_dual_psd_violation",
        "certificate_artifact_dual_trace",
    )

    @staticmethod
    def _contains_array(value: object) -> bool:
        if isinstance(value, np.ndarray):
            return True
        if type(value) is dict:
            return any(
                OrchestrationNamespaceFailClosedTests._contains_array(item)
                for item in value.values()
            )
        if type(value) in (tuple, list):
            return any(
                OrchestrationNamespaceFailClosedTests._contains_array(item)
                for item in value
            )
        return False

    def _call_row_without_escape(self, schedule: str) -> dict[str, object]:
        try:
            return _diagnostic_row(
                2,
                0.5,
                schedule=schedule,
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        except Exception as exc:
            self.fail(
                "untrusted orchestration return must fail closed, not escape: "
                f"{type(exc).__name__}: {exc}"
            )

    def _assert_fail_closed(
        self,
        row: dict[str, object],
        *,
        reason_fragment: str,
    ) -> None:
        self.assertNotEqual(row.get("status"), "sdp_certified")
        self.assertNotEqual(row.get("sdp_status"), "computed_safe_enclosure")
        self.assertEqual(row.get("P_opt"), "")
        self.assertEqual(row.get("P_opt_status"), "not_computed")
        reasons = "|".join(
            str(row.get(field, ""))
            for field in (
                "certificate_validation_reason",
                "certificate_artifact_validation_reason",
                "certificate_source_binding_reason",
                "sdp_status",
                "status",
            )
        )
        self.assertIn(reason_fragment, reasons)
        for field in self._CLEARED_FIELDS:
            with self.subTest(cleared_field=field):
                self.assertEqual(row.get(field), "")
        self.assertFalse(self._contains_array(row))

    def test_poisoned_validator_objectives_cannot_create_false_enclosure(
        self,
    ) -> None:
        """The caller must not trust a validator's scalar return ledger."""
        original = diagnostic_probe._validate_certificate_artifacts
        actual: dict[str, float] = {}

        def poison_only_returned_objectives(*args, **kwargs):
            valid, reason, metrics = original(*args, **kwargs)
            actual["feasible_lower"] = float(
                metrics["certificate_artifact_primal_objective"]
            )
            poisoned = dict(metrics)
            poisoned["certificate_artifact_primal_objective"] = 0.91
            poisoned["certificate_artifact_dual_trace"] = 0.91
            return valid, reason, poisoned

        with _patched_attribute(
            "_validate_certificate_artifacts",
            poison_only_returned_objectives,
        ):
            row = self._call_row_without_escape("poisoned_artifact_objectives")

        self.assertAlmostEqual(
            actual["feasible_lower"],
            0.9100138365862797,
            places=12,
        )
        if row.get("U_safe") != "":
            self.assertLess(float(row["U_safe"]), actual["feasible_lower"])
        self._assert_fail_closed(
            row,
            reason_fragment="artifact_metrics_mismatch",
        )

    def test_helper_namespace_cannot_inject_reserved_optimum_field(self) -> None:
        original = diagnostic_probe._trusted_source_snapshot_audit

        def injecting_helper(*args, **kwargs):
            metadata, audit = original(*args, **kwargs)
            audit = dict(audit)
            audit["P_opt"] = "helper_injected_optimum"
            return metadata, audit

        with _patched_attribute(
            "_trusted_source_snapshot_audit", injecting_helper
        ):
            row = self._call_row_without_escape("helper_reserved_injection")
        self._assert_fail_closed(row, reason_fragment="source_namespace_invalid")

    def test_artifact_namespace_cannot_inject_reserved_optimum_field(self) -> None:
        original = diagnostic_probe._validate_certificate_artifacts

        def injecting_validator(*args, **kwargs):
            valid, reason, metrics = original(*args, **kwargs)
            metrics = dict(metrics)
            metrics["P_opt"] = "artifact_injected_optimum"
            return valid, reason, metrics

        with _patched_attribute(
            "_validate_certificate_artifacts", injecting_validator
        ):
            row = self._call_row_without_escape("artifact_reserved_injection")
        self._assert_fail_closed(
            row,
            reason_fragment="artifact_namespace_invalid",
        )

    def test_hull_and_validation_namespaces_reject_reserved_injection(
        self,
    ) -> None:
        original_hull = diagnostic_probe._hull_gram_with_audit
        original_validation = diagnostic_probe._validate_safe_certificate

        def injecting_hull(*args, **kwargs):
            stable, gram, audit = original_hull(*args, **kwargs)
            audit = dict(audit)
            audit["P_opt"] = "hull_injected_optimum"
            return stable, gram, audit

        def injecting_validation(*args, **kwargs):
            valid, reason, metrics = original_validation(*args, **kwargs)
            metrics = dict(metrics)
            metrics["P_opt"] = "validation_injected_optimum"
            return valid, reason, metrics

        for name, patch_name, replacement, reason in (
            (
                "hull",
                "_hull_gram_with_audit",
                injecting_hull,
                "hull_namespace_invalid",
            ),
            (
                "validation",
                "_validate_safe_certificate",
                injecting_validation,
                "validation_namespace_invalid",
            ),
        ):
            with self.subTest(namespace=name):
                with _patched_attribute(patch_name, replacement):
                    row = self._call_row_without_escape(
                        f"{name}_reserved_injection"
                    )
                self._assert_fail_closed(row, reason_fragment=reason)

    def test_source_metadata_malformed_values_fail_closed(self) -> None:
        original = diagnostic_probe._trusted_source_snapshot_audit

        class ContainsBomb(dict):
            def __contains__(self, key):
                raise RuntimeError(f"contains bomb: {key}")

        def variant_helper(variant: str):
            def malformed(*args, **kwargs):
                metadata, audit = original(*args, **kwargs)
                if variant == "none":
                    metadata = None
                elif variant == "missing":
                    metadata = dict(metadata)
                    metadata.pop("rank")
                elif variant == "nonfinite":
                    metadata = dict(metadata)
                    metadata["rank_threshold"] = math.nan
                elif variant == "raising_contains":
                    metadata = ContainsBomb(metadata)
                else:
                    raise AssertionError(variant)
                return metadata, audit

            return malformed

        for variant in ("none", "missing", "nonfinite", "raising_contains"):
            with self.subTest(variant=variant):
                with _patched_attribute(
                    "_trusted_source_snapshot_audit", variant_helper(variant)
                ):
                    row = self._call_row_without_escape(
                        f"source_metadata_{variant}"
                    )
                self._assert_fail_closed(
                    row,
                    reason_fragment="source_namespace_invalid",
                )

    def test_spectral_status_object_and_malformed_tuple_fail_closed(self) -> None:
        original = diagnostic_probe._trusted_source_snapshot_audit

        class EqualityBomb:
            def __eq__(self, other):
                raise RuntimeError(f"equality bomb: {other}")

            def __ne__(self, other):
                raise RuntimeError(f"inequality bomb: {other}")

        def bad_status(*args, **kwargs):
            metadata, audit = original(*args, **kwargs)
            audit = dict(audit)
            audit["spectral_audit_status"] = EqualityBomb()
            return metadata, audit

        def malformed_tuple(*args, **kwargs):
            metadata, _ = original(*args, **kwargs)
            return (metadata,)

        for name, replacement in (
            ("raising_status", bad_status),
            ("malformed_tuple", malformed_tuple),
        ):
            with self.subTest(variant=name):
                with _patched_attribute(
                    "_trusted_source_snapshot_audit", replacement
                ):
                    row = self._call_row_without_escape(
                        f"source_audit_{name}"
                    )
                self._assert_fail_closed(
                    row,
                    reason_fragment="source_namespace_invalid",
                )

    def test_artifact_metrics_malformed_values_fail_closed(self) -> None:
        original = diagnostic_probe._validate_certificate_artifacts

        class GetBomb(dict):
            def get(self, key, default=None):
                raise RuntimeError(f"get bomb: {key}")

        class UpdateBomb(Mapping):
            def __init__(self, payload):
                self._payload = dict(payload)

            def __getitem__(self, key):
                return self._payload[key]

            def __iter__(self):
                raise RuntimeError("iteration bomb")

            def __len__(self):
                return len(self._payload)

            def get(self, key, default=None):
                return self._payload.get(key, default)

        def variant_validator(variant: str):
            def malformed(*args, **kwargs):
                valid, reason, metrics = original(*args, **kwargs)
                if variant == "none":
                    return valid, reason, None
                if variant == "raising_get":
                    return valid, reason, GetBomb(metrics)
                if variant == "raising_iteration":
                    return valid, reason, UpdateBomb(metrics)
                if variant == "non_plain_bool":
                    return np.bool_(valid), reason, metrics
                if variant == "malformed_tuple":
                    return valid, reason
                raise AssertionError(variant)

            return malformed

        for variant in (
            "none",
            "raising_get",
            "raising_iteration",
            "non_plain_bool",
            "malformed_tuple",
        ):
            with self.subTest(variant=variant):
                with _patched_attribute(
                    "_validate_certificate_artifacts",
                    variant_validator(variant),
                ):
                    row = self._call_row_without_escape(
                        f"artifact_metrics_{variant}"
                    )
                self._assert_fail_closed(
                    row,
                    reason_fragment="artifact_namespace_invalid",
                )

    def test_certificate_container_malformed_values_fail_closed(self) -> None:
        original = diagnostic_probe.certify_minimum_error_from_weighted_states

        class ItemsBomb(dict):
            def items(self):
                raise RuntimeError("items bomb")

        class GetBomb(dict):
            def get(self, key, default=None):
                raise RuntimeError(f"get bomb: {key}")

        class ItemBomb(dict):
            def __getitem__(self, key):
                raise RuntimeError(f"getitem bomb: {key}")

        def variant_certificate(variant: str):
            def malformed(source_states, *args, **kwargs):
                certificate = dict(original(source_states, *args, **kwargs))
                if variant == "none":
                    return None
                if variant == "list":
                    return list(certificate.items())
                if variant == "raising_items":
                    return ItemsBomb(certificate)
                if variant == "raising_get":
                    return GetBomb(certificate)
                if variant == "raising_getitem":
                    return ItemBomb(certificate)
                raise AssertionError(variant)

            return malformed

        for variant in (
            "none",
            "list",
            "raising_items",
            "raising_get",
            "raising_getitem",
        ):
            with self.subTest(variant=variant):
                with _patched_attribute(
                    "certify_minimum_error_from_weighted_states",
                    variant_certificate(variant),
                ):
                    row = self._call_row_without_escape(
                        f"certificate_container_{variant}"
                    )
                self._assert_fail_closed(
                    row,
                    reason_fragment="certificate_namespace_invalid",
                )

    def test_validation_return_malformed_tuple_mapping_and_bool_fail_closed(
        self,
    ) -> None:
        original = diagnostic_probe._validate_safe_certificate

        class GetBomb(dict):
            def get(self, key, default=None):
                raise RuntimeError(f"validation get bomb: {key}")

        def variant_validation(variant: str):
            def malformed(*args, **kwargs):
                valid, reason, metrics = original(*args, **kwargs)
                if variant == "malformed_tuple":
                    return valid, reason
                if variant == "raising_get":
                    return valid, reason, GetBomb(metrics)
                if variant == "non_plain_bool":
                    return np.bool_(valid), reason, metrics
                raise AssertionError(variant)

            return malformed

        for variant in ("malformed_tuple", "raising_get", "non_plain_bool"):
            with self.subTest(variant=variant):
                with _patched_attribute(
                    "_validate_safe_certificate",
                    variant_validation(variant),
                ):
                    row = self._call_row_without_escape(
                        f"validation_return_{variant}"
                    )
                self._assert_fail_closed(
                    row,
                    reason_fragment="validation_namespace_invalid",
                )


@unittest.skip("Route B uses the single pinned artifact diagnostic pass.")
class CertificateArtifactFailClosedTests(unittest.TestCase):
    @staticmethod
    def _real_certificate_or_empty() -> dict[str, object]:
        direct_api = getattr(
            diagnostic_probe,
            "certify_minimum_error_from_weighted_states",
            None,
        )
        if not callable(direct_api):
            return {}
        gram = diagnostic_probe.physical_interval_gram(2, 0.5)
        states, _, audit = (
            diagnostic_probe._spectral_reconstruction_audit(gram)
        )
        if audit["source_transfer_status"] != "passed":
            raise AssertionError("test source audit unexpectedly failed")
        return dict(
            direct_api(
                states,
                solver="CLARABEL",
            )
        )

    def test_corrupted_actual_artifacts_fail_closed(self) -> None:
        base = self._real_certificate_or_empty()

        def rotate_source_states(certificate: dict[str, object]) -> None:
            states = np.array(
                certificate["source_weighted_states"], copy=True
            )
            first = np.array(states[:, 0], copy=True)
            second = np.array(states[:, 1], copy=True)
            theta = 0.001
            states[:, 0] = math.cos(theta) * first + math.sin(theta) * second
            states[:, 1] = -math.sin(theta) * first + math.cos(theta) * second
            certificate["source_weighted_states"] = states

        def mismatch_source_gram(certificate: dict[str, object]) -> None:
            gram = np.array(certificate["source_weighted_gram"], copy=True)
            gram[0, 1] += 1.0e-4
            gram[1, 0] += 1.0e-4
            certificate["source_weighted_gram"] = gram

        def rotate_primal_effects(certificate: dict[str, object]) -> None:
            effects = [
                np.array(effect, copy=True)
                for effect in certificate["repaired_primal_povm"]
            ]
            effects[0], effects[1] = effects[1], effects[0]
            certificate["repaired_primal_povm"] = tuple(effects)

        def nan_primal_effect(certificate: dict[str, object]) -> None:
            effects = [
                np.array(effect, copy=True)
                for effect in certificate["repaired_primal_povm"]
            ]
            effects[0][0, 0] = math.nan
            certificate["repaired_primal_povm"] = tuple(effects)

        def wrong_primal_shape(certificate: dict[str, object]) -> None:
            effects = [
                np.array(effect, copy=True)
                for effect in certificate["repaired_primal_povm"]
            ]
            effects[0] = effects[0][:-1, :]
            certificate["repaired_primal_povm"] = tuple(effects)

        def lie_about_primal_objective(certificate: dict[str, object]) -> None:
            lower = float(certificate["primal_feasible_objective"]) - 1.0e-4
            upper = float(certificate["dual_feasible_objective"])
            certificate["primal_feasible_objective"] = lower
            certificate["feasible_bound_gap"] = upper - lower
            certificate["relative_feasible_bound_gap"] = (upper - lower) / (
                0.5 * (abs(lower) + abs(upper))
            )

        def replace_dual_operator(certificate: dict[str, object]) -> None:
            dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
            certificate["safe_dual_operator"] = np.zeros_like(dual)

        def lie_about_dual_trace(certificate: dict[str, object]) -> None:
            lower = float(certificate["primal_feasible_objective"])
            upper = float(certificate["dual_feasible_objective"]) + 1.0e-4
            certificate["dual_feasible_objective"] = upper
            certificate["feasible_bound_gap"] = upper - lower
            certificate["relative_feasible_bound_gap"] = (upper - lower) / (
                0.5 * (abs(lower) + abs(upper))
            )

        def lie_about_dual_slack(certificate: dict[str, object]) -> None:
            certificate["dual_feasible_min_slack"] = (
                float(certificate["dual_feasible_min_slack"]) + 1.0e-4
            )

        corruptions = {
            "rotated_source_states": rotate_source_states,
            "source_gram_mismatch": mismatch_source_gram,
            "rotated_primal_effects": rotate_primal_effects,
            "nan_primal_effect": nan_primal_effect,
            "wrong_primal_shape": wrong_primal_shape,
            "reported_primal_objective": lie_about_primal_objective,
            "replaced_dual_operator": replace_dual_operator,
            "reported_dual_trace": lie_about_dual_trace,
            "reported_dual_slack": lie_about_dual_slack,
        }
        for name, corrupt in corruptions.items():
            with self.subTest(corruption=name):
                certificate = copy.deepcopy(base)
                if certificate:
                    corrupt(certificate)

                def corrupted_certificate(*args, **kwargs):
                    del args, kwargs
                    return copy.deepcopy(certificate)

                with _patched_attribute(
                    "certify_minimum_error_from_weighted_states",
                    corrupted_certificate,
                ):
                    row = _diagnostic_row(
                        2,
                        0.5,
                        schedule="corrupt_certificate_artifact",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
                self.assertEqual(
                    row.get("sdp_status"),
                    "not_computed_certificate_artifact_validation_failed",
                )
                self.assertEqual(
                    row.get("certificate_artifact_validation_status"),
                    "failed",
                )
                self.assertEqual(row.get("L_safe"), "")
                self.assertEqual(row.get("U_safe"), "")
                self.assertEqual(
                    row.get("status"), "srm_only_certificate_failed"
                )

    def test_artifact_validator_exceptions_fail_closed_without_leak(self) -> None:
        for exception in (
            RuntimeError("validator runtime failure"),
            np.linalg.LinAlgError("validator eigensolver failure"),
        ):
            with self.subTest(exception=type(exception).__name__):
                def exploding_validator(*args, exception=exception, **kwargs):
                    del args, kwargs
                    raise exception

                with _patched_attribute(
                    "_validate_certificate_artifacts", exploding_validator
                ):
                    try:
                        row = _diagnostic_row(
                            2,
                            0.5,
                            schedule="artifact_validator_exception",
                            max_peak_bytes=10**8,
                            max_matrix_dimension=100,
                        )
                    except Exception as exc:
                        self.fail(
                            "artifact validator exception must fail closed: "
                            f"{exc!r}"
                        )
                self.assertEqual(
                    row.get("sdp_status"),
                    "not_computed_certificate_artifact_validation_failed",
                )
                self.assertEqual(
                    row.get("certificate_artifact_validation_status"),
                    "failed",
                )
                self.assertIn(
                    "exception",
                    str(
                        row.get(
                            "certificate_artifact_validation_reason", ""
                        )
                    ),
                )
                self.assertEqual(row.get("L_safe"), "")
                self.assertEqual(row.get("U_safe"), "")
                self.assertFalse(
                    any(isinstance(value, np.ndarray) for value in row.values())
                )


class ExactRankDropTests(unittest.TestCase):
    def test_all_ones_source_is_rejected_as_exact_rank_drop(self) -> None:
        _, _, audit = diagnostic_probe._spectral_reconstruction_audit(
            np.ones((3, 3), dtype=float)
        )
        self.assertEqual(audit.get("spectral_audit_status"), "failed")
        self.assertEqual(audit.get("source_transfer_status"), "failed")

    def test_nextafter_one_may_fail_closed_without_hiding_endpoint(self) -> None:
        row = _diagnostic_row(
            2,
            float(np.nextafter(1.0, 0.0)),
            schedule="nextafter_rank_drop",
            max_peak_bytes=10**8,
            max_matrix_dimension=100,
        )
        self.assertNotEqual(row.get("status"), "sdp_certified")
        self.assertEqual(row.get("L_safe"), "")
        self.assertEqual(row.get("U_safe"), "")


class SrmDefaultCutoffBoundaryTests(unittest.TestCase):
    def test_n48_computes_without_opt_in_but_n49_requires_it(self) -> None:
        logs: list[dict[str, int | str]] = []
        n48 = _diagnostic_row(
            48,
            0.1,
            schedule="srm_default_cutoff_n48",
            max_srm_n=48,
            max_sdp_n=0,
            max_peak_bytes=512 * 2**20,
            max_matrix_dimension=1200,
            allow_srm_above_default=False,
            resource_log=logs.append,
        )
        self.assertEqual(n48.get("srm_status"), "computed")
        self.assertNotEqual(n48.get("P_SRM"), "")
        self.assertEqual(logs[0]["external_M_n"], 1176)

        logs.clear()
        n49 = _diagnostic_row(
            49,
            0.1,
            schedule="srm_default_cutoff_n49",
            max_srm_n=49,
            max_sdp_n=0,
            max_peak_bytes=10**9,
            max_matrix_dimension=1300,
            allow_srm_above_default=False,
            resource_log=logs.append,
        )
        self.assertEqual(
            n49.get("srm_status"), "not_computed_opt_in_required"
        )
        self.assertEqual(n49.get("P_SRM"), "")
        self.assertEqual(logs[0]["external_M_n"], 1225)


class AnalyticEndpointRowTests(unittest.TestCase):
    def test_endpoints_bypass_general_srm_evd_and_sdp(self) -> None:
        calls: list[str] = []
        original_srm = diagnostic_probe.srm_quantities
        original_sdp = getattr(
            diagnostic_probe,
            "certify_minimum_error_from_weighted_states",
            None,
        )

        def srm_spy(*args, **kwargs):
            calls.append("srm")
            return original_srm(*args, **kwargs)

        def sdp_spy(*args, **kwargs):
            calls.append("sdp")
            if callable(original_sdp):
                return original_sdp(*args, **kwargs)
            return {}

        with _patched_attribute("srm_quantities", srm_spy):
            with _patched_attribute(
                "certify_minimum_error_from_weighted_states", sdp_spy
            ):
                for c in (0.0, 1.0):
                    _diagnostic_row(
                        2,
                        c,
                        schedule="analytic_endpoint",
                        max_peak_bytes=10**8,
                        max_matrix_dimension=100,
                    )
        self.assertEqual(calls, [])

    def test_endpoint_probabilities_are_exact_and_not_labeled_optimum(
        self,
    ) -> None:
        for c, expected_probability in ((0.0, 1.0), (1.0, 1.0 / 3.0)):
            with self.subTest(c=c):
                row = _diagnostic_row(
                    2,
                    c,
                    schedule="analytic_endpoint",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )
                actual = {
                    field: row.get(field)
                    for field in (
                        "P_tr",
                        "P_SRM",
                        "repaired_primal_value",
                        "shifted_dual_value",
                        "strongest_measurement_value",
                        "floating_primal_dual_span",
                        "P_opt",
                        "P_opt_status",
                        "sdp_status",
                        "status",
                        "diagnostic_kind",
                        "legacy_bound_fields_status",
                    )
                }
                expected = {
                    "P_tr": expected_probability,
                    "P_SRM": expected_probability,
                    "repaired_primal_value": expected_probability,
                    "shifted_dual_value": expected_probability,
                    "strongest_measurement_value": expected_probability,
                    "floating_primal_dual_span": 0.0,
                    "P_opt": "",
                    "P_opt_status": "not_computed",
                    "sdp_status": "analytic_endpoint_exact",
                    "status": "analytic_endpoint",
                    "diagnostic_kind": "analytic_endpoint_exact",
                    "legacy_bound_fields_status": "deprecated_blank",
                }
                self.assertEqual(actual, expected)
                for field in (
                    "L_safe",
                    "U_safe",
                    "strongest_certified_lower",
                    "safe_gap",
                    "enclosure_kind",
                ):
                    self.assertEqual(row.get(field), "")


@unittest.skip("Route B publishes no certificate/enclosure fields.")
class CertificateFailClosedTests(unittest.TestCase):
    @staticmethod
    def _valid_certificate() -> dict[str, object]:
        gram = _explicit_product_state_gram(2, 0.5)
        return dict(
            certify_minimum_error(
                gram, solver="CLARABEL", rank_tolerance=0.0
            )
        )

    def _row_with_certificate(self, certificate) -> dict[str, object]:
        def fake_certificate(*args, **kwargs):
            del args, kwargs
            if isinstance(certificate, BaseException):
                raise certificate
            return dict(certificate)

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states", fake_certificate
        ):
            try:
                return _diagnostic_row(
                    2,
                    0.5,
                    schedule="invalid_certificate",
                    max_peak_bytes=10**8,
                    max_matrix_dimension=100,
                )
            except Exception as exc:  # converted into an assertion RED
                self.fail(f"certificate failure must fail closed: {exc!r}")

    def test_invalid_certificates_fail_closed_without_safe_fields(self) -> None:
        base = self._valid_certificate()
        fixtures: list[tuple[str, dict[str, object]]] = []

        rank_loss = dict(base)
        rank_loss["rank"] = 2
        fixtures.append(("rank_loss", rank_loss))

        nonfinite = dict(base)
        nonfinite["dual_feasible_objective"] = math.nan
        fixtures.append(("nonfinite", nonfinite))

        nonfinite_metadata = dict(base)
        nonfinite_metadata["weighted_gram_lambda_min"] = math.nan
        fixtures.append(("nonfinite_metadata", nonfinite_metadata))

        bad_equality = dict(base)
        bad_equality["primal_feasible_equality_residual_op"] = 1.0e-3
        fixtures.append(("bad_equality", bad_equality))

        bad_primal_psd = dict(base)
        bad_primal_psd["primal_feasible_psd_violation"] = 1.0e-3
        fixtures.append(("bad_primal_psd", bad_primal_psd))

        bad_dual_psd = dict(base)
        bad_dual_psd["dual_feasible_psd_violation"] = 1.0e-3
        fixtures.append(("bad_dual_psd", bad_dual_psd))

        bad_dual_slack = dict(base)
        bad_dual_slack["dual_feasible_min_slack"] = -1.0e-3
        fixtures.append(("bad_dual_slack", bad_dual_slack))

        reversed_bounds = dict(base)
        reversed_bounds["primal_feasible_objective"] = 0.8
        reversed_bounds["dual_feasible_objective"] = 0.7
        fixtures.append(("reversed_bounds", reversed_bounds))

        below_srm = dict(base)
        below_srm["primal_feasible_objective"] = 0.0
        below_srm["dual_feasible_objective"] = 0.0
        fixtures.append(("upper_below_srm", below_srm))

        bad_status = dict(base)
        bad_status["primal_status"] = "infeasible"
        fixtures.append(("bad_status", bad_status))

        for name, certificate in fixtures:
            with self.subTest(invalid=name):
                row = self._row_with_certificate(certificate)
                self.assertEqual(
                    row.get("sdp_status"),
                    "not_computed_certificate_validation_failed",
                )
                self.assertEqual(row.get("L_safe"), "")
                self.assertEqual(row.get("U_safe"), "")
                self.assertEqual(
                    row.get("status"), "srm_only_certificate_failed"
                )
                self.assertNotEqual(
                    row.get("certificate_validation_status"), "passed"
                )

    def test_solver_exception_is_caught_and_fails_closed(self) -> None:
        row = self._row_with_certificate(RuntimeError("solver exploded"))
        self.assertEqual(
            row.get("sdp_status"), "not_computed_certificate_exception"
        )
        self.assertEqual(row.get("L_safe"), "")
        self.assertEqual(row.get("U_safe"), "")
        self.assertEqual(row.get("status"), "srm_only_certificate_failed")


@unittest.skip("Route B publishes residual-checked diagnostics, not bounds.")
class ProbabilityScaleToleranceTests(unittest.TestCase):
    def test_srm_upper_order_uses_probability_scale_not_unit_scale(self) -> None:
        original = getattr(
            diagnostic_probe,
            "certify_minimum_error_from_weighted_states",
            None,
        )

        def subtly_invalid_certificate(
            source_states: np.ndarray, **kwargs
        ):
            if not callable(original):
                return {}
            certificate = dict(
                original(
                    source_states,
                    solver=str(kwargs.get("solver", "CLARABEL")),
                )
            )
            hypothesis_count = source_states.shape[1]
            gram = hypothesis_count * source_states.T @ source_states
            srm = float(srm_quantities(gram)["srm"])
            upper = srm - 5.0e-9
            lower = upper - 1.0e-9
            certificate.update(
                {
                    "primal_feasible_objective": lower,
                    "dual_feasible_objective": upper,
                    "feasible_bound_gap": upper - lower,
                    "relative_feasible_bound_gap": (upper - lower) / upper,
                }
            )
            return certificate

        with _patched_attribute(
            "certify_minimum_error_from_weighted_states",
            subtly_invalid_certificate,
        ):
            row = _diagnostic_row(
                4,
                0.999999,
                schedule="probability_scale_tolerance",
                max_peak_bytes=10**8,
                max_matrix_dimension=100,
            )
        self.assertEqual(
            row.get("sdp_status"),
            "not_computed_certificate_validation_failed",
        )
        expected_scale = max(
            float(row["P_SRM"]),
            1.0 / 10.0,
        )
        self.assertAlmostEqual(
            float(row.get("certificate_probability_scale", -1.0)),
            expected_scale,
            places=14,
        )
        self.assertAlmostEqual(
            float(row.get("certificate_probability_tolerance", -1.0)),
            1.0e-8 * expected_scale,
            places=20,
        )


class AnalyticOptimumFixtureTests(unittest.TestCase):
    def test_only_audited_analytic_fixtures_enclose_known_optima(self) -> None:
        overlap = 0.4
        fixtures = (
            (np.eye(3), 1.0, "orthogonal"),
            (np.ones((4, 4)), 0.25, "identical"),
            (
                np.array([[1.0, overlap], [overlap, 1.0]]),
                0.5 * (1.0 + math.sqrt(1.0 - overlap**2)),
                "two_state_helstrom",
            ),
        )
        self.assertEqual(
            {name for _, _, name in fixtures},
            {"orthogonal", "identical", "two_state_helstrom"},
        )
        for gram, optimum, name in fixtures:
            with self.subTest(fixture=name):
                certificate = certify_minimum_error(gram, solver="CLARABEL")
                lower = float(certificate["primal_feasible_objective"])
                upper = float(certificate["dual_feasible_objective"])
                self.assertLessEqual(lower, optimum + 2e-12)
                self.assertGreaterEqual(upper, optimum - 2e-12)


class CandidateFigureTests(unittest.TestCase):
    def test_bound_csv_generates_candidate_figure_without_optimum_values(
        self,
    ) -> None:
        """Catches a plotting path that cannot consume the frozen Task 8 table."""
        script = ROOT / "proofs" / "plot_task8_candidate_figure3.py"
        source = ROOT / "proofs" / "weighted_hull_diagnostics.csv"
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "candidate.pdf"
            png = Path(directory) / "candidate.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--csv",
                    str(source),
                    "--pdf",
                    str(pdf),
                    "--png",
                    str(png),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertGreater(pdf.stat().st_size, 1_000)
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertIn(FROZEN_CSV_SHA256, completed.stdout)
            semantics_lines = [
                line
                for line in completed.stdout.splitlines()
                if line.startswith("FIGURE_SEMANTICS=")
            ]
            self.assertEqual(len(semantics_lines), 1)
            semantics = semantics_lines[0].lower()
            self.assertIn("floating primal-dual span", semantics)
            self.assertIn("re-feasibilized primal / shifted dual values", semantics)
            for forbidden in ("bracket", "safe", "certified", "enclosure"):
                self.assertNotIn(forbidden, semantics)

            payload = pdf.read_bytes()
            font_sizes: list[float] = []
            for match in re.finditer(
                rb"stream\r?\n(.*?)\r?\nendstream", payload, re.DOTALL
            ):
                stream = match.group(1)
                try:
                    decoded = zlib.decompress(stream)
                except zlib.error:
                    decoded = stream
                font_sizes.extend(
                    float(value)
                    for value in re.findall(
                        rb"/[A-Za-z0-9_.+-]+\s+([0-9.]+)\s+Tf", decoded
                    )
                )
            self.assertTrue(font_sizes, "no PDF text glyph sizes were found")
            self.assertGreaterEqual(min(font_sizes), 5.0)


class CliAndCsvTests(unittest.TestCase):
    def test_direct_script_entrypoint_imports_repository_modules(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "proofs"
                    / "weighted_hull_continuum_outer_probe.py"
                ),
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertIn("--max-srm-n", completed.stdout)
        help_text = completed.stdout.lower()
        self.assertIn("floating-sdp diagnostics", help_text)
        for forbidden in ("safe-sdp", "certified", "enclosure"):
            self.assertNotIn(forbidden, help_text)

    def test_cli_has_safe_cutoffs_and_explicit_budget_opt_ins(self) -> None:
        parser = _subject("_build_parser", _zero_parser)()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.max_srm_n, 48)
        self.assertEqual(defaults.max_sdp_n, 5)
        self.assertGreater(defaults.max_peak_gib, 0.0)
        self.assertGreaterEqual(defaults.max_matrix_dimension, 1_176)
        explicit = parser.parse_args(
            [
                "--output",
                "chosen.csv",
                "--max-srm-n",
                "49",
                "--max-sdp-n",
                "7",
                "--max-peak-gib",
                "2",
                "--max-matrix-dimension",
                "1300",
                "--allow-srm-above-48",
                "--allow-sdp-above-5",
            ]
        )
        self.assertEqual(explicit.output, Path("chosen.csv"))
        self.assertTrue(explicit.allow_srm_above_48)
        self.assertTrue(explicit.allow_sdp_above_5)

    def test_default_case_grid_contains_compact_and_slow_outer_rows(
        self,
    ) -> None:
        cases = _subject("_diagnostic_cases", lambda max_srm_n: [])(48)
        self.assertEqual(max(int(case["n"]) for case in cases), 32)
        compact = {
            float(case["lambda_target"])
            for case in cases
            if case["schedule"] == "compact_lambda"
        }
        outer = {
            str(case["schedule"])
            for case in cases
            if str(case["schedule"]).startswith("outer_")
        }
        self.assertEqual(compact, set(COMPACT_LAMBDAS))
        self.assertEqual(outer, OUTER_SCHEDULES)
        for schedule in OUTER_SCHEDULES:
            schedule_ns = {
                int(case["n"])
                for case in cases
                if case["schedule"] == schedule
            }
            self.assertGreaterEqual(len(schedule_ns), 3)

    def test_csv_writer_preserves_not_computed_and_external_M_n(self) -> None:
        row = _zero_diagnostic_row(
            6,
            0.8,
            schedule="csv_fixture",
        )
        row.update(
            {
                "external_M_n": 21,
                "P_opt": "",
                "P_opt_status": "not_computed",
                "sdp_status": "not_computed",
                "status": "srm_only",
                "interpretation": "finite_size_diagnostic_not_proof",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostics.csv"
            writer = _subject("_write_diagnostics_csv", lambda path, rows: None)
            writer(output, [row])
            self.assertTrue(output.exists())
            with output.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["external_M_n"], "21")
        self.assertEqual(written[0]["P_opt"], "")
        self.assertEqual(written[0]["P_opt_status"], "not_computed")

    def test_frozen_csv_is_diagnostic_only_and_byte_deterministic(self) -> None:
        official = ROOT / "proofs" / "weighted_hull_diagnostics.csv"
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            arguments = (
                "--max-srm-n",
                "48",
                "--max-sdp-n",
                "5",
            )
            with redirect_stdout(io.StringIO()):
                diagnostic_probe.main(["--output", str(first), *arguments])
                diagnostic_probe.main(["--output", str(second), *arguments])
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
        official_bytes = official.read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_bytes, official_bytes)
        self.assertEqual(
            hashlib.sha256(official_bytes).hexdigest().upper(),
            FROZEN_CSV_SHA256,
        )
        rows = list(
            csv.DictReader(
                io.StringIO(official_bytes.decode("utf-8", errors="strict"))
            )
        )
        self.assertEqual(len(rows), 24)
        required_fields = {
            "repaired_primal_value",
            "shifted_dual_value",
            "strongest_measurement_value",
            "floating_primal_dual_span",
            "floating_primal_dual_span_relative",
            "primal_completeness_residual_fro",
            "primal_completeness_residual_op",
            "primal_psd_violation",
            "dual_min_slack_after_shift",
            "dual_psd_violation_after_shift",
            "artifact_recomputation_status",
            "artifact_recomputation_reason",
            "diagnostic_kind",
            "legacy_bound_fields_status",
            "source_transfer_delta",
            "source_transfer_budget",
            "source_transfer_status",
            "source_difference_op_norm",
            "source_difference_trace_norm_relative",
            "source_difference_fro_norm_relative",
            "source_difference_op_norm_relative",
            "source_lambda_max_target",
            "source_lambda_max_reconstructed",
            "source_reconstructed_sqrt_residual_fro_relative",
            "source_reconstructed_sqrt_residual_op_relative",
            "trusted_source_snapshot_sha256",
            "hull_subtraction_audit_kind",
        }
        self.assertTrue(required_fields.issubset(rows[0]))
        self.assertTrue(
            all(
                row["schema_version"]
                == "weighted_hull_diagnostics_v8_route_b"
                for row in rows
            )
        )
        self.assertEqual(
            sum(row["status"] == "sdp_floating_diagnostic" for row in rows),
            5,
        )
        self.assertTrue(all(row["P_opt"] == "" for row in rows))
        for row in rows:
            self.assertEqual(row["legacy_bound_fields_status"], "deprecated_blank")
            for field in (
                "L_raw",
                "U_raw",
                "L_safe",
                "U_safe",
                "strongest_certified_lower",
                "safe_gap",
                "enclosure_kind",
            ):
                self.assertEqual(row[field], "")


def run_resource_guard_mutant_red() -> unittest.result.TestResult:
    """Kill a mutation that raises resource budgets before allocation."""
    original = _subject("diagnostic_row", _zero_diagnostic_row)

    def unguarded_mutant(n: int, c: float, **kwargs):
        kwargs["max_peak_bytes"] = 10**18
        kwargs["max_matrix_dimension"] = 10**9
        return original(n, c, **kwargs)

    suite = unittest.TestSuite(
        [
            DenseResourceGuardTests(
                "test_resource_guard_returns_before_dense_allocation"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", unguarded_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_coordinate_helper_dependency_mutant_red(
) -> unittest.result.TestResult:
    """Kill replacement of independent subtraction by Task 1 coordinates."""

    def task1_coordinate_mutant(n: int, c: float) -> np.ndarray:
        coordinates = np.asarray(
            weighted_hull_finite_audit.weighted_hull_matrix_fraction(
                n, Fraction(str(c))
            ),
            dtype=float,
        )
        return coordinates @ coordinates.T

    suite = unittest.TestSuite(
        [
            IndependentHullGramTests(
                "test_independent_construction_ignores_task1_coordinate_helper"
            )
        ]
    )
    with _patched_attribute("independent_hull_gram", task1_coordinate_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_optimum_label_mutant_red() -> unittest.result.TestResult:
    """Kill copying SRM into a field advertised as an exact optimum."""
    original = _subject("diagnostic_row", _zero_diagnostic_row)

    def mislabeled_mutant(n: int, c: float, **kwargs):
        row = dict(original(n, c, **kwargs))
        row["P_opt"] = row["P_SRM"]
        row["P_opt_status"] = "computed"
        return row

    suite = unittest.TestSuite(
        [
            DiagnosticRowTests(
                "test_above_sdp_cutoff_has_no_mislabeled_optimum"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", mislabeled_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_spectral_truncation_mutant_red() -> unittest.result.TestResult:
    """Kill restoring the old positive spectral truncation threshold."""
    original = diagnostic_probe.canonical_weighted_states

    def truncated_canonical(*args, **kwargs):
        kwargs["rank_tolerance"] = 1.0e-12
        return original(*args, **kwargs)

    suite = unittest.TestSuite(
        [
            FullRankNearSingularSdpTests(
                "test_near_singular_sdp_uses_rank_zero_cutoff_and_retains_rank_three"
            )
        ]
    )
    with _patched_attribute(
        "canonical_weighted_states", truncated_canonical
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_endpoint_mutant_red() -> unittest.result.TestResult:
    """Kill routing exact endpoints through the generic floating-point path."""
    original = diagnostic_probe.diagnostic_row

    def interior_endpoint_mutant(n: int, c: float, **kwargs):
        if c == 0.0:
            c = float(np.nextafter(0.0, 1.0))
        elif c == 1.0:
            c = float(np.nextafter(1.0, 0.0))
        return original(n, c, **kwargs)

    suite = unittest.TestSuite(
        [
            AnalyticEndpointRowTests(
                "test_endpoint_probabilities_are_exact_and_not_labeled_optimum"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", interior_endpoint_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_fail_closed_mutant_red() -> unittest.result.TestResult:
    """Kill relabeling a rejected certificate as a computed enclosure."""
    original = diagnostic_probe.diagnostic_row

    def unsafe_relabel_mutant(n: int, c: float, **kwargs):
        row = dict(original(n, c, **kwargs))
        if "certificate" in str(row.get("sdp_status", "")):
            row["sdp_status"] = "computed_safe_enclosure"
            row["status"] = "sdp_certified"
            row["L_safe"] = 0.0
            row["U_safe"] = 0.0
        return row

    suite = unittest.TestSuite(
        [
            CertificateFailClosedTests(
                "test_invalid_certificates_fail_closed_without_safe_fields"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", unsafe_relabel_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_scale_tolerance_mutant_red() -> unittest.result.TestResult:
    """Kill replacing the probability scale by an unconditional unit scale."""
    original = diagnostic_probe._probability_scale

    def unit_scale_mutant(*values: float) -> float:
        return max(1.0, original(*values))

    suite = unittest.TestSuite(
        [
            ProbabilityScaleToleranceTests(
                "test_srm_upper_order_uses_probability_scale_not_unit_scale"
            )
        ]
    )
    with _patched_attribute("_probability_scale", unit_scale_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_stable_hull_mutant_red() -> unittest.result.TestResult:
    """Kill returning cancellation-prone raw subtraction near c=1."""

    def raw_subtraction_mutant(n: int, c: float) -> np.ndarray:
        return (
            diagnostic_probe.physical_interval_gram(n, c)
            - diagnostic_probe._vacuum_gram(n, c)
            - diagnostic_probe._one_excitation_gram(n, c)
        )

    suite = unittest.TestSuite(
        [
            StableHullGramTests(
                "test_near_endpoint_hull_matches_explicit_ge_two_oracle_and_is_psd"
            )
        ]
    )
    with _patched_attribute(
        "independent_hull_gram", raw_subtraction_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_raw_subtraction_path_mutant_red() -> unittest.result.TestResult:
    """Kill making the audited raw subtraction path decorative."""

    def stable_only_mutant(n: int, c: float) -> np.ndarray:
        return diagnostic_probe._stable_hull_gram(n, c)

    suite = unittest.TestSuite(
        [
            StableHullGramTests(
                "test_corrupted_vacuum_or_one_excitation_breaks_raw_subtraction_audit"
            )
        ]
    )
    with _patched_attribute("independent_hull_gram", stable_only_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_source_transfer_padding_mutant_red() -> unittest.result.TestResult:
    """Kill exposing the raw source-ensemble upper bound without padding."""
    original = diagnostic_probe.diagnostic_row

    def unpadded_upper_mutant(n: int, c: float, **kwargs):
        row = dict(original(n, c, **kwargs))
        if row.get("U_raw", "") != "":
            row["U_safe"] = row["U_raw"]
        return row

    suite = unittest.TestSuite(
        [
            SourceTransferPaddingTests(
                "test_padded_upper_encloses_explicit_theta_measurement"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", unpadded_upper_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_source_binding_mutant_red() -> unittest.result.TestResult:
    """Kill accepting a certificate whose source minimum spectrum is x0.4."""
    suite = unittest.TestSuite(
        [
            SourceTransferPaddingTests(
                "test_certificate_metadata_must_bind_to_source_factorization"
            )
        ]
    )
    with _patched_attribute(
        "_metadata_values_match", lambda expected, actual: True
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_relative_gap_copy_mutant_red() -> unittest.result.TestResult:
    """Kill copying a poisoned relative gap instead of recomputing it."""
    suite = unittest.TestSuite(
        [
            SourceTransferPaddingTests(
                "test_raw_and_padded_relative_gaps_are_recomputed"
            )
        ]
    )
    with _patched_attribute("_relative_enclosure_gap", lambda lower, upper: 99.0):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_actual_factor_drift_mutant_red() -> unittest.result.TestResult:
    """Kill recanonicalizing a second factor instead of solving on retained S."""
    direct = diagnostic_probe.certify_minimum_error_from_weighted_states

    def recanonicalizing_mutant(
        source_states: np.ndarray, **kwargs
    ) -> dict[str, object]:
        count = source_states.shape[1]
        gram = count * source_states.T @ source_states
        replacement, _ = certification_module.canonical_weighted_states(
            gram, rank_tolerance=0.0
        )
        return dict(
            direct(
                replacement,
                solver=str(kwargs.get("solver", "CLARABEL")),
            )
        )

    suite = unittest.TestSuite(
        [
            ActualSourceFactorBindingTests(
                "test_shared_canonical_drift_cannot_change_retained_source_certificate"
            )
        ]
    )
    with _patched_attribute(
        "certify_minimum_error_from_weighted_states",
        recanonicalizing_mutant,
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_certificate_input_mutation_mutant_red() -> unittest.result.TestResult:
    """Kill a conditional in-place n=2,c=.5 solver-input column rotation."""
    direct = diagnostic_probe.certify_minimum_error_from_weighted_states
    target_gram = diagnostic_probe.physical_interval_gram(2, 0.5) / 3.0

    def conditional_mutating_callee(source_states, *args, **kwargs):
        states = np.asarray(source_states)
        if states.shape == (3, 3) and np.allclose(
            states.T @ states,
            target_gram,
            rtol=0.0,
            atol=1.0e-14,
        ):
            theta = 0.1
            first = np.array(states[:, 0], copy=True)
            second = np.array(states[:, 1], copy=True)
            states[:, 0] = math.cos(theta) * first + math.sin(theta) * second
            states[:, 1] = -math.sin(theta) * first + math.cos(theta) * second
        return direct(source_states, *args, **kwargs)

    suite = unittest.TestSuite(
        [
            CertificateInputMutationTests(
                "test_unmodified_solver_input_computes_with_identity_audit"
            )
        ]
    )
    with _patched_attribute(
        "certify_minimum_error_from_weighted_states",
        conditional_mutating_callee,
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _run_post_audit_factor_mutant_red(
    transform,
) -> unittest.result.TestResult:
    """Run the real direct solver on a conditionally replaced returned S."""
    original = diagnostic_probe._spectral_reconstruction_audit

    def post_audit_mutant(gram):
        states, metadata, audit = original(gram)
        if states.shape == (3, 3):
            states = transform(np.array(states, copy=True, order="C"))
        return states, metadata, audit

    suite = unittest.TestSuite(
        [
            TrustedSnapshotSourceAuditTests(
                "test_unmodified_post_audit_factor_computes"
            )
        ]
    )
    with _patched_attribute(
        "_spectral_reconstruction_audit", post_audit_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_post_audit_scale_up_mutant_red() -> unittest.result.TestResult:
    """Kill stale transfer metadata after S is scaled up post-audit."""
    return _run_post_audit_factor_mutant_red(
        lambda states: (1.0 + 2.0e-6) * states
    )


def _retired_post_audit_scale_down_mutant_red() -> unittest.result.TestResult:
    """Kill stale transfer metadata after S is scaled down post-audit."""
    return _run_post_audit_factor_mutant_red(
        lambda states: (1.0 - 2.0e-6) * states
    )


def _retired_post_audit_column_rotation_mutant_red() -> unittest.result.TestResult:
    """Kill stale transfer metadata after post-audit hypothesis rotation."""
    def rotate(states):
        theta = 0.001
        first = np.array(states[:, 0], copy=True)
        second = np.array(states[:, 1], copy=True)
        states[:, 0] = math.cos(theta) * first + math.sin(theta) * second
        states[:, 1] = -math.sin(theta) * first + math.cos(theta) * second
        return states

    return _run_post_audit_factor_mutant_red(rotate)


def _retired_post_authority_snapshot_drift_mutant_red(
) -> unittest.result.TestResult:
    """Kill mutation of the owning readonly snapshot after authority returns."""
    original = diagnostic_probe._trusted_source_snapshot_audit

    def post_authority_mutant(gram, snapshot):
        metadata, audit = original(gram, snapshot)
        states = np.asarray(snapshot)
        if (
            states.shape in {(3, 3), (6, 6)}
            and not states.flags.writeable
        ):
            theta = -0.001
            states.setflags(write=True)
            first = np.array(states[:, 0], copy=True)
            second = np.array(states[:, 1], copy=True)
            states[:, 0] = (
                math.cos(theta) * first + math.sin(theta) * second
            )
            states[:, 1] = (
                -math.sin(theta) * first + math.cos(theta) * second
            )
            states.setflags(write=False)
        return metadata, audit

    suite = unittest.TestSuite(
        [
            PostAuthoritySnapshotIntegrityTests(
                "test_unmodified_post_authority_snapshot_computes_with_one_sha"
            )
        ]
    )
    with _patched_attribute(
        "_trusted_source_snapshot_audit", post_authority_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_forged_helper_source_sha_mutant_red() -> unittest.result.TestResult:
    """Kill a helper ledger SHA that is not the caller-owned authority SHA."""
    original = diagnostic_probe._trusted_source_snapshot_audit

    def forged_sha_mutant(gram, snapshot):
        metadata, audit = original(gram, snapshot)
        audit = dict(audit)
        audit["trusted_source_snapshot_sha256"] = "0" * 64
        return metadata, audit

    suite = unittest.TestSuite(
        [
            PostAuthoritySnapshotIntegrityTests(
                "test_unmodified_post_authority_snapshot_computes_with_one_sha"
            )
        ]
    )
    with _patched_attribute(
        "_trusted_source_snapshot_audit", forged_sha_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_missing_helper_source_sha_mutant_red() -> unittest.result.TestResult:
    """Kill omission of the mandatory helper-to-authority SHA binding."""
    original = diagnostic_probe._trusted_source_snapshot_audit

    def missing_sha_mutant(gram, snapshot):
        metadata, audit = original(gram, snapshot)
        audit = dict(audit)
        audit.pop("trusted_source_snapshot_sha256")
        return metadata, audit

    suite = unittest.TestSuite(
        [
            PostAuthoritySnapshotIntegrityTests(
                "test_unmodified_post_authority_snapshot_computes_with_one_sha"
            )
        ]
    )
    with _patched_attribute(
        "_trusted_source_snapshot_audit", missing_sha_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_physical_gram_authority_mutant_red() -> unittest.result.TestResult:
    """Kill in-place physical-Gram drift after source authority auditing."""
    original = diagnostic_probe._trusted_source_snapshot_audit

    def mutate_gram_after_helper(gram, snapshot):
        metadata, audit = original(gram, snapshot)
        physical = np.asarray(gram)
        physical.setflags(write=True)
        physical *= 1.0 - 2.0e-6
        physical.setflags(write=False)
        return metadata, audit

    suite = unittest.TestSuite(
        [
            PostAuthoritySnapshotIntegrityTests(
                "test_unmodified_post_authority_snapshot_computes_with_one_sha"
            )
        ]
    )
    with _patched_attribute(
        "_trusted_source_snapshot_audit", mutate_gram_after_helper
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_exact_rank_drop_mutant_red() -> unittest.result.TestResult:
    """Kill removing the raw-Gram strict-positive-definiteness guard."""
    suite = unittest.TestSuite(
        [
            ExactRankDropTests(
                "test_all_ones_source_is_rejected_as_exact_rank_drop"
            )
        ]
    )
    with _patched_attribute(
        "_strict_positive_definite_gram", lambda gram: True
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_srm_cutoff_boundary_mutant_red() -> unittest.result.TestResult:
    """Kill changing the default SRM cutoff comparison from > to >=."""
    suite = unittest.TestSuite(
        [
            SrmDefaultCutoffBoundaryTests(
                "test_n48_computes_without_opt_in_but_n49_requires_it"
            )
        ]
    )
    with _patched_attribute("DEFAULT_MAX_SRM_N", 47):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_route_b_bound_alias_mutant_red() -> unittest.result.TestResult:
    """Kill republishing a floating primal value under a bound alias."""
    original = diagnostic_probe.diagnostic_row

    def bound_alias_mutant(*args, **kwargs):
        row = original(*args, **kwargs)
        if row.get("repaired_primal_value") != "":
            row["L_safe"] = row["repaired_primal_value"]
        return row

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_public_sdp_row_uses_diagnostic_values_and_blanks_bound_aliases"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", bound_alias_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_route_b_duplicate_truth_pass_mutant_red() -> unittest.result.TestResult:
    """Kill reintroducing duplicate pinned source/artifact recomputation."""
    original = diagnostic_probe.diagnostic_row

    def duplicate_pass_mutant(*args, **kwargs):
        original(*args, **kwargs)
        return original(*args, **kwargs)

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_pinned_source_and_artifact_cores_each_execute_once"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", duplicate_pass_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_route_b_facade_reentry_mutant_red() -> unittest.result.TestResult:
    """Kill re-entry into a dynamically patchable deprecated validator."""
    original = diagnostic_probe.diagnostic_row

    def facade_reentry_mutant(*args, **kwargs):
        diagnostic_probe._validate_safe_certificate()
        return original(*args, **kwargs)

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_deprecated_validation_facades_are_outside_public_truth_path"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", facade_reentry_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_route_b_missing_dual_rank_multiplier_mutant_red(
) -> unittest.result.TestResult:
    """Kill replacing tr(Y)+rank*delta by tr(Y)+delta."""
    original_validator = RouteBFallbackTests.ARTIFACT_VALIDATOR

    def missing_rank_multiplier(trusted, solver_input, certificate):
        valid, reason, metrics = original_validator(
            trusted, solver_input, certificate
        )
        if not valid:
            return valid, reason, metrics
        states = np.asarray(trusted, dtype=float)
        dual = np.asarray(certificate["safe_dual_operator"], dtype=float)
        dual = (dual + dual.T) / 2.0
        rhos = [
            np.outer(states[:, index], states[:, index])
            for index in range(states.shape[1])
        ]
        raw_minimum = min(
            float(np.linalg.eigvalsh(dual - rho)[0]) for rho in rhos
        )
        delta = max(0.0, -raw_minimum)
        mutated = dict(metrics)
        mutated["certificate_artifact_dual_trace"] = float(
            np.nextafter(float(np.trace(dual)) + delta, math.inf)
        )
        return True, "passed", mutated

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_negative_dual_slack_uses_rank_multiplier_and_is_shifted"
            )
        ]
    )
    RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
        missing_rank_multiplier
    )
    try:
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
            original_validator
        )


def run_route_b_missing_primal_contraction_mutant_red(
) -> unittest.result.TestResult:
    """Kill exposing the raw overcomplete family without contraction/fill."""
    original_validator = RouteBFallbackTests.ARTIFACT_VALIDATOR

    def raw_overcomplete_metrics(trusted, solver_input, certificate):
        valid, reason, metrics = original_validator(
            trusted, solver_input, certificate
        )
        if not valid:
            return valid, reason, metrics
        states = np.asarray(trusted, dtype=float)
        rank, count = states.shape
        effects = tuple(
            np.asarray(effect, dtype=float)
            for effect in certificate["repaired_primal_povm"]
        )
        rhos = [
            np.outer(states[:, index], states[:, index])
            for index in range(count)
        ]
        completeness = sum(effects) - np.eye(rank)
        mutated = dict(metrics)
        mutated.update(
            {
                "certificate_artifact_primal_completeness_fro": float(
                    np.linalg.norm(completeness, ord="fro")
                ),
                "certificate_artifact_primal_completeness_op": float(
                    np.linalg.norm(completeness, ord=2)
                ),
                "certificate_artifact_primal_objective": math.fsum(
                    float(np.trace(rho @ effect))
                    for rho, effect in zip(rhos, effects)
                ),
            }
        )
        return True, "passed", mutated

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_overcomplete_primal_is_contracted_and_filled_by_remainder"
            )
        ]
    )
    RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
        raw_overcomplete_metrics
    )
    try:
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
            original_validator
        )


def run_route_b_missing_psd_projection_mutant_red(
) -> unittest.result.TestResult:
    """Kill normalization that omits projection of a negative effect."""
    original_validator = RouteBFallbackTests.ARTIFACT_VALIDATOR

    def raw_indefinite_metrics(trusted, solver_input, certificate):
        valid, reason, metrics = original_validator(
            trusted, solver_input, certificate
        )
        if not valid:
            return valid, reason, metrics
        effects = tuple(
            np.asarray(effect, dtype=float)
            for effect in certificate["repaired_primal_povm"]
        )
        raw_minimum = min(
            float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
            for effect in effects
        )
        mutated = dict(metrics)
        mutated["certificate_artifact_primal_min_eigenvalue"] = raw_minimum
        mutated["certificate_artifact_primal_psd_violation"] = max(
            0.0, -raw_minimum
        )
        return True, "passed", mutated

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_negative_effect_is_psd_projected_before_normalization"
            )
        ]
    )
    RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
        raw_indefinite_metrics
    )
    try:
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        RouteBFallbackTests.ARTIFACT_VALIDATOR = staticmethod(
            original_validator
        )


def run_route_b_failure_field_leak_mutant_red() -> unittest.result.TestResult:
    """Kill a failure path that leaks one primary floating diagnostic."""
    original = diagnostic_probe.diagnostic_row

    def leaking_failure_mutant(*args, **kwargs):
        row = original(*args, **kwargs)
        if str(row.get("sdp_status", "")).startswith("not_computed_"):
            row["repaired_primal_value"] = 0.5
        return row

    suite = unittest.TestSuite(
        [
            RouteBFallbackTests(
                "test_artifact_and_exception_failures_clear_all_route_b_values"
            )
        ]
    )
    with _patched_attribute("diagnostic_row", leaking_failure_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_validator_metric_trust_mutant_red() -> unittest.result.TestResult:
    """Kill accepting a poisoned validator ledger without reconciliation."""
    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_poisoned_validator_objectives_cannot_create_false_enclosure"
            )
        ]
    )
    with _patched_attribute(
        "_artifact_metric_ledgers_match",
        lambda *args, **kwargs: (True, "passed"),
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_source_reserved_field_allow_mutant_red() -> unittest.result.TestResult:
    """Kill silently stripping a reserved helper field instead of rejecting it."""
    original = diagnostic_probe._normalize_source_authority_result

    def stripping_mutant(result):
        if type(result) is tuple and len(result) == 2 and type(result[1]) is dict:
            metadata, audit = result
            audit = dict(audit)
            audit.pop("P_opt", None)
            result = (metadata, audit)
        return original(result)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_helper_namespace_cannot_inject_reserved_optimum_field"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_source_authority_result", stripping_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_artifact_reserved_field_allow_mutant_red() -> unittest.result.TestResult:
    """Kill silently stripping a reserved artifact-ledger field."""
    original = diagnostic_probe._normalize_artifact_result

    def stripping_mutant(result, **kwargs):
        if type(result) is tuple and len(result) == 3 and type(result[2]) is dict:
            valid, reason, metrics = result
            metrics = dict(metrics)
            metrics.pop("P_opt", None)
            result = (valid, reason, metrics)
        return original(result, **kwargs)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_artifact_namespace_cannot_inject_reserved_optimum_field"
            )
        ]
    )
    with _patched_attribute("_normalize_artifact_result", stripping_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_hull_reserved_field_allow_mutant_red() -> unittest.result.TestResult:
    """Kill silently stripping a reserved hull-audit field."""
    original = diagnostic_probe._normalize_hull_result

    def stripping_mutant(result, **kwargs):
        if type(result) is tuple and len(result) == 3 and type(result[2]) is dict:
            stable, gram, audit = result
            audit = dict(audit)
            audit.pop("P_opt", None)
            result = (stable, gram, audit)
        return original(result, **kwargs)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_hull_and_validation_namespaces_reject_reserved_injection"
            )
        ]
    )
    with _patched_attribute("_normalize_hull_result", stripping_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_validation_reserved_field_allow_mutant_red(
) -> unittest.result.TestResult:
    """Kill silently stripping a reserved safe-validation field."""
    original = diagnostic_probe._normalize_validation_result

    def stripping_mutant(result):
        if type(result) is tuple and len(result) == 3 and type(result[2]) is dict:
            valid, reason, metrics = result
            metrics = dict(metrics)
            metrics.pop("P_opt", None)
            result = (valid, reason, metrics)
        return original(result)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_hull_and_validation_namespaces_reject_reserved_injection"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_validation_result", stripping_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_certificate_list_coercion_mutant_red() -> unittest.result.TestResult:
    """Kill coercing an untrusted certificate list into a dict."""
    original = diagnostic_probe._normalize_certificate_namespace

    def coercing_mutant(value):
        if type(value) is list:
            value = dict(value)
        return original(value)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_certificate_container_malformed_values_fail_closed"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_certificate_namespace", coercing_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_artifact_numpy_bool_mutant_red() -> unittest.result.TestResult:
    """Kill accepting numpy.bool_ as an artifact-validation decision."""
    original = diagnostic_probe._normalize_artifact_result

    def bool_coercion_mutant(result, **kwargs):
        if type(result) is tuple and len(result) == 3:
            valid, reason, metrics = result
            if isinstance(valid, np.bool_):
                result = (bool(valid), reason, metrics)
        return original(result, **kwargs)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_artifact_metrics_malformed_values_fail_closed"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_artifact_result", bool_coercion_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_validation_numpy_bool_mutant_red() -> unittest.result.TestResult:
    """Kill accepting numpy.bool_ as a safe-validation decision."""
    original = diagnostic_probe._normalize_validation_result

    def bool_coercion_mutant(result):
        if type(result) is tuple and len(result) == 3:
            valid, reason, metrics = result
            if isinstance(valid, np.bool_):
                result = (bool(valid), reason, metrics)
        return original(result)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_validation_return_malformed_tuple_mapping_and_bool_fail_closed"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_validation_result", bool_coercion_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def _retired_nonfinite_source_metadata_mutant_red() -> unittest.result.TestResult:
    """Kill sanitizing a nonfinite source metadata scalar."""
    original = diagnostic_probe._normalize_source_authority_result

    def sanitizing_mutant(result):
        if type(result) is tuple and len(result) == 2 and type(result[0]) is dict:
            metadata, audit = result
            if not math.isfinite(float(metadata.get("rank_threshold", 0.0))):
                metadata = dict(metadata)
                metadata["rank_threshold"] = 0.0
                result = (metadata, audit)
        return original(result)

    suite = unittest.TestSuite(
        [
            OrchestrationNamespaceFailClosedTests(
                "test_source_metadata_malformed_values_fail_closed"
            )
        ]
    )
    with _patched_attribute(
        "_normalize_source_authority_result", sanitizing_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    unittest.main()
