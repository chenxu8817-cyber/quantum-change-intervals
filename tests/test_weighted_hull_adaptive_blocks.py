"""Behavior tests for the adaptive macroblock parameter ledger.

The three slow schedules at the end of this file are finite diagnostics only.
They are deliberately test-local and are not part of the theorem API.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import ceil, floor, isclose, isfinite, log, pi, sqrt
from typing import TypedDict, get_type_hints
import unittest

import proofs.weighted_hull_asymptotic_probe as adaptive_probe


REQUIRED_ADAPTIVE_KEYS = {
    "n",
    "c",
    "lambda_value",
    "h",
    "ell",
    "p1",
    "block_count",
    "minimum_block_size",
    "sizes",
    "lambda_over_B",
    "B_over_h",
    "m_p1",
    "log_m_over_ell",
}


class _ZeroAdaptiveBlockParameters(TypedDict):
    """Importable shape-safe stand-in used only for the initial RED."""


def _zero_adaptive_block_parameters(
    n: int, c: float
) -> _ZeroAdaptiveBlockParameters:
    del n, c
    return {
        "n": 0,
        "c": 0.0,
        "lambda_value": 0.0,
        "h": 0.0,
        "ell": 0.0,
        "p1": 0.0,
        "block_count": 0,
        "minimum_block_size": 0,
        "sizes": (),
        "lambda_over_B": 0.0,
        "B_over_h": 0.0,
        "m_p1": 0.0,
        "log_m_over_ell": 0.0,
    }  # type: ignore[return-value]


def _parameter_type() -> type:
    return getattr(
        adaptive_probe,
        "AdaptiveBlockParameters",
        _ZeroAdaptiveBlockParameters,
    )


def _adaptive(n: int, c: float) -> dict[str, object]:
    subject = getattr(
        adaptive_probe,
        "adaptive_block_parameters",
        _zero_adaptive_block_parameters,
    )
    return subject(n, c)


@contextmanager
def _patched_attribute(name: str, value: object):
    missing = object()
    original = getattr(adaptive_probe, name, missing)
    setattr(adaptive_probe, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(adaptive_probe, name)
        else:
            setattr(adaptive_probe, name, original)


class AdaptiveLedgerContractTests(unittest.TestCase):
    def test_public_typed_dict_has_exact_required_keys_and_types(self) -> None:
        parameter_type = _parameter_type()
        self.assertEqual(
            parameter_type.__required_keys__, frozenset(REQUIRED_ADAPTIVE_KEYS)
        )

        subject = getattr(
            adaptive_probe,
            "adaptive_block_parameters",
            _zero_adaptive_block_parameters,
        )
        self.assertIs(get_type_hints(subject)["return"], parameter_type)
        hints = get_type_hints(parameter_type)
        self.assertIs(hints["n"], int)
        self.assertIs(hints["c"], float)
        self.assertEqual(hints["sizes"], tuple[int, ...])


class AdaptiveLedgerFiniteIdentityTests(unittest.TestCase):
    def test_n100_c099_matches_hand_checked_literal_ledger(self) -> None:
        params = _adaptive(100, 0.99)

        self.assertEqual(params["n"], 100)
        self.assertEqual(params["c"], 0.99)
        self.assertAlmostEqual(params["lambda_value"], 1.0, places=13)
        self.assertAlmostEqual(params["ell"], 6.713201046722445, places=12)
        self.assertAlmostEqual(params["p1"], 0.090868349185905375, places=12)
        self.assertAlmostEqual(params["h"], 9.0868349185905375, places=11)
        self.assertEqual(params["block_count"], 6)
        self.assertEqual(params["minimum_block_size"], 16)
        self.assertEqual(params["sizes"], (17, 17, 17, 17, 16, 16))
        self.assertAlmostEqual(params["lambda_over_B"], 1.0 / 6.0, places=13)
        self.assertAlmostEqual(
            params["B_over_h"], 6.0 / 9.0868349185905375, places=12
        )
        self.assertAlmostEqual(
            params["m_p1"], 16.0 * 0.090868349185905375, places=12
        )
        self.assertAlmostEqual(
            params["log_m_over_ell"], log(16.0) / 6.713201046722445,
            places=12,
        )

    def test_exact_h_identity_and_approved_ceiling_are_independent(self) -> None:
        params = _adaptive(1000, 0.999)
        length_literal = 8.991192791684300
        probability_literal = 0.016373731663329400
        lambda_value = 1000.0 * (1.0 - 0.999)
        h_from_probability = 1000.0 * probability_literal
        h_from_compound_overlap = (
            lambda_value * 1.999 * length_literal**2 / (pi * pi)
        )
        approved_real_block_scale = (
            sqrt(h_from_probability) + lambda_value * sqrt(length_literal)
        )

        self.assertAlmostEqual(params["ell"], length_literal, places=12)
        self.assertAlmostEqual(params["p1"], probability_literal, places=13)
        self.assertAlmostEqual(params["h"], h_from_probability, places=12)
        self.assertAlmostEqual(params["h"], h_from_compound_overlap, places=12)
        self.assertGreater(approved_real_block_scale, 7.0)
        self.assertLess(approved_real_block_scale, 8.0)
        self.assertEqual(params["block_count"], 8)
        self.assertEqual(params["sizes"], (125,) * 8)

    def test_balanced_floor_and_ceiling_accounting_is_exact(self) -> None:
        params = _adaptive(101, 0.99)
        block_count = params["block_count"]
        sizes = params["sizes"]

        self.assertEqual(block_count, 6)
        self.assertEqual(len(sizes), 6)
        self.assertEqual(sum(sizes), 101)
        self.assertEqual(sizes, (17, 17, 17, 17, 17, 16))
        self.assertEqual(params["minimum_block_size"], floor(101 / 6))
        self.assertEqual(set(sizes), {floor(101 / 6), ceil(101 / 6)})

    def test_all_fields_are_finite_and_positive_on_valid_outer_fixtures(self) -> None:
        for n, c in (
            (100, 0.99),
            (10_000, 0.999),
            (1_000_000, 0.9999),
        ):
            with self.subTest(n=n, c=c):
                params = _adaptive(n, c)
                self.assertEqual(set(params), REQUIRED_ADAPTIVE_KEYS)
                self.assertGreater(params["n"], 0)
                self.assertGreater(params["c"], 0.0)
                self.assertGreater(params["block_count"], 0)
                self.assertGreater(params["minimum_block_size"], 0)
                self.assertTrue(all(size > 0 for size in params["sizes"]))
                for key in (
                    "lambda_value",
                    "h",
                    "ell",
                    "p1",
                    "lambda_over_B",
                    "B_over_h",
                    "m_p1",
                    "log_m_over_ell",
                ):
                    self.assertTrue(isfinite(params[key]), key)
                    self.assertGreater(params[key], 0.0, key)


class AdaptiveLedgerValidationTests(unittest.TestCase):
    def test_invalid_n_type_or_range_is_rejected(self) -> None:
        for invalid_n in (1, 0, -3, True, 2.0, "2"):
            with self.subTest(n=invalid_n):
                with self.assertRaisesRegex(ValueError, "n"):
                    _adaptive(invalid_n, 0.99)  # type: ignore[arg-type]

    def test_invalid_c_type_range_or_finiteness_is_rejected(self) -> None:
        for invalid_c in (
            -0.1,
            1.0,
            1.1,
            float("nan"),
            float("inf"),
            True,
            "0.99",
        ):
            with self.subTest(c=invalid_c):
                with self.assertRaisesRegex(ValueError, "c"):
                    _adaptive(100, invalid_c)  # type: ignore[arg-type]

    def test_prepartition_rejection_reports_measured_B_n_and_c(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"B=5 exceeds n=2.*c=0(?:\.0)?",
        ):
            _adaptive(2, 0.0)

    def test_nonpositive_hull_scale_is_rejected_before_division(self) -> None:
        with _patched_attribute("p1", lambda c: 0.0):
            with self.assertRaisesRegex(
                ValueError, r"positive finite hull scale.*h=0(?:\.0)?"
            ):
                _adaptive(100, 0.99)


def _schedule_values(n: int) -> tuple[tuple[str, float], ...]:
    """Return the three approved non-probative diagnostic schedules."""
    return (
        ("log_log_n", log(log(n))),
        ("sqrt_log_n", sqrt(log(n))),
        ("cube_root_n", n ** (1.0 / 3.0)),
    )


def _schedule_rows(n_values: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for n in n_values:
        for schedule, requested_lambda in _schedule_values(n):
            c = 1.0 - requested_lambda / n
            rows.append(
                {
                    "schedule": schedule,
                    "requested_lambda": requested_lambda,
                    "parameters": _adaptive(n, c),
                }
            )
    return tuple(rows)


class SlowScheduleDiagnosticTests(unittest.TestCase):
    def test_test_local_schedule_formulas_match_hand_checked_literals(self) -> None:
        actual = _schedule_values(1_000_000)
        expected = (
            ("log_log_n", 2.625791914476011),
            ("sqrt_log_n", 3.7169221888498383),
            ("cube_root_n", 99.99999999999997),
        )
        self.assertEqual(tuple(name for name, _ in actual), tuple(name for name, _ in expected))
        for (_, value), (_, literal) in zip(actual, expected):
            self.assertAlmostEqual(value, literal, places=13)

    def test_schedule_rows_obey_finite_identities_without_monotonicity_claim(self) -> None:
        rows = _schedule_rows((10_000, 1_000_000))
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["schedule"] for row in rows},
            {"log_log_n", "sqrt_log_n", "cube_root_n"},
        )
        for row in rows:
            params = row["parameters"]
            requested_lambda = row["requested_lambda"]
            n = params["n"]
            c = params["c"]
            length = params["ell"]
            probability = params["p1"]
            block_count = params["block_count"]
            minimum_size = params["minimum_block_size"]

            self.assertTrue(isfinite(requested_lambda))
            self.assertGreater(requested_lambda, 0.0)
            self.assertTrue(
                isclose(params["lambda_value"], requested_lambda, rel_tol=2e-11)
            )
            self.assertAlmostEqual(params["h"], n * probability, places=10)
            self.assertTrue(
                isclose(
                    params["h"],
                    params["lambda_value"]
                    * (1.0 + c)
                    * length**2
                    / (pi * pi),
                    rel_tol=2e-11,
                    abs_tol=0.0,
                )
            )
            self.assertEqual(
                block_count,
                ceil(sqrt(params["h"]) + params["lambda_value"] * sqrt(length)),
            )
            self.assertAlmostEqual(
                params["lambda_over_B"], params["lambda_value"] / block_count,
                places=14,
            )
            self.assertAlmostEqual(params["B_over_h"], block_count / params["h"], places=14)
            self.assertAlmostEqual(params["m_p1"], minimum_size * probability, places=14)
            self.assertAlmostEqual(
                params["log_m_over_ell"], log(minimum_size) / length, places=14
            )


def run_behavioral_zero_stub_red() -> unittest.result.TestResult:
    """Run the Task 5 behavior suite against an importable zero record."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (
        AdaptiveLedgerContractTests,
        AdaptiveLedgerFiniteIdentityTests,
        AdaptiveLedgerValidationTests,
        SlowScheduleDiagnosticTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))
    with _patched_attribute("AdaptiveBlockParameters", _ZeroAdaptiveBlockParameters):
        with _patched_attribute(
            "adaptive_block_parameters", _zero_adaptive_block_parameters
        ):
            return unittest.TextTestRunner(verbosity=2).run(suite)


def run_floor_instead_of_ceiling_mutant_red() -> unittest.result.TestResult:
    """Kill the realistic mutation ``ceil(x) -> floor(x)``."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        AdaptiveLedgerFiniteIdentityTests
    )
    with _patched_attribute("ceil", floor):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_wrong_schedule_formula_mutant_red() -> unittest.result.TestResult:
    """Kill a diagnostic schedule mutation without promoting it to a theorem."""
    original = globals()["_schedule_values"]

    def wrong_schedule_values(n: int) -> tuple[tuple[str, float], ...]:
        return (
            ("log_log_n", log(n)),
            ("sqrt_log_n", sqrt(n)),
            ("cube_root_n", sqrt(n)),
        )

    suite = unittest.TestSuite(
        [SlowScheduleDiagnosticTests("test_test_local_schedule_formulas_match_hand_checked_literals")]
    )
    try:
        globals()["_schedule_values"] = wrong_schedule_values
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals()["_schedule_values"] = original


if __name__ == "__main__":
    unittest.main()
