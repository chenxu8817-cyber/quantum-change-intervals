"""Behavior tests for the moving-outer diagnostic error-scale ledger.

The slow schedules in this module are finite diagnostics only.  They are
test-local and no assertion asks them to be monotone in ``n``.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import isclose, isfinite, log, sqrt
from typing import TypedDict, get_type_hints
import unittest

import proofs.weighted_hull_asymptotic_probe as outer_probe


REQUIRED_OUTER_KEYS = {
    "local_similarity",
    "volterra_repair",
    "log_match",
    "tail_overhead",
    "diagonal_cells",
    "one_excitation",
    "vacuum",
    "total_scale_proxy",
}


class _ZeroOuterErrorScales(TypedDict):
    local_similarity: float
    volterra_repair: float
    log_match: float
    tail_overhead: float
    diagonal_cells: float
    one_excitation: float
    vacuum: float
    total_scale_proxy: float


def _zero_outer_error_scale_proxy(n: int, c: float) -> _ZeroOuterErrorScales:
    del n, c
    return {
        "local_similarity": 0.0,
        "volterra_repair": 0.0,
        "log_match": 0.0,
        "tail_overhead": 0.0,
        "diagonal_cells": 0.0,
        "one_excitation": 0.0,
        "vacuum": 0.0,
        "total_scale_proxy": 0.0,
    }


def _scale_type() -> type:
    return getattr(outer_probe, "OuterErrorScales", _ZeroOuterErrorScales)


def _outer(n: int, c: float) -> dict[str, float]:
    subject = getattr(
        outer_probe,
        "outer_error_scale_proxy",
        _zero_outer_error_scale_proxy,
    )
    return subject(n, c)


@contextmanager
def _patched_attribute(name: str, value: object):
    missing = object()
    original = getattr(outer_probe, name, missing)
    setattr(outer_probe, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(outer_probe, name)
        else:
            setattr(outer_probe, name, original)


class OuterLedgerContractTests(unittest.TestCase):
    def test_public_typed_dict_has_exact_float_fields(self) -> None:
        scale_type = _scale_type()
        self.assertEqual(
            scale_type.__required_keys__, frozenset(REQUIRED_OUTER_KEYS)
        )
        self.assertEqual(
            set(get_type_hints(scale_type)), REQUIRED_OUTER_KEYS
        )
        for field_type in get_type_hints(scale_type).values():
            self.assertIs(field_type, float)

        subject = getattr(
            outer_probe,
            "outer_error_scale_proxy",
            _zero_outer_error_scale_proxy,
        )
        self.assertIs(get_type_hints(subject)["return"], scale_type)


class OuterLedgerLiteralTests(unittest.TestCase):
    def test_n10000_c09999_matches_hand_checked_literal_scales(self) -> None:
        scales = _outer(10_000, 0.9999)
        expected = {
            "local_similarity": 0.11111111111109888,
            "volterra_repair": 0.27773612372457074,
            "log_match": 0.37884573440146930,
            "tail_overhead": 0.59028485751075230,
            "diagonal_cells": 0.11111111111111110,
            "one_excitation": 0.07225451505209250,
            "vacuum": 0.03871513477849880,
            "total_scale_proxy": 1.5800585876895936,
        }
        self.assertEqual(set(scales), REQUIRED_OUTER_KEYS)
        for key, literal in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(scales[key], literal, places=12)

    def test_each_formula_is_recomputed_from_frozen_adaptive_literals(self) -> None:
        scales = _outer(10_000, 0.9999)
        # These literals were frozen independently in the accepted Task 5
        # fixture; this test does not call any scalar helper to derive them.
        lam = 0.9999999999998899
        hull_scale = 25.82969181745867
        length = 11.290296433659746
        block_count = 9
        minimum_size = 1111
        expected_terms = {
            "local_similarity": lam / block_count,
            "volterra_repair": log(log(minimum_size)) / log(minimum_size),
            "log_match": abs(log(minimum_size) / length - 1.0),
            "tail_overhead": sqrt(block_count / hull_scale),
            "diagonal_cells": 1.0 / block_count,
            "one_excitation": (
                log(10_000) / (sqrt(lam) * length * length)
            ),
            "vacuum": 1.0 / hull_scale,
        }
        for key, expected in expected_terms.items():
            with self.subTest(key=key):
                self.assertTrue(
                    isclose(scales[key], expected, rel_tol=2.0e-14, abs_tol=0.0)
                )
        self.assertTrue(
            isclose(
                scales["total_scale_proxy"],
                sum(expected_terms.values()),
                rel_tol=2.0e-14,
                abs_tol=0.0,
            )
        )

    def test_linear_tail_overhead_cannot_replace_the_square_root(self) -> None:
        scales = _outer(10_000, 0.9999)
        linear_mutant = 9.0 / 25.82969181745867
        self.assertAlmostEqual(scales["tail_overhead"], 0.5902848575107523)
        self.assertNotAlmostEqual(scales["tail_overhead"], linear_mutant)


class OuterLedgerDomainTests(unittest.TestCase):
    def test_invalid_adaptive_inputs_remain_rejected(self) -> None:
        for n, c in (
            (1, 0.99),
            (100, -0.01),
            (100, 1.0),
            (100, float("nan")),
        ):
            with self.subTest(n=n, c=c):
                with self.assertRaises(ValueError):
                    _outer(n, c)

    def test_small_minimum_block_is_outside_the_ledger_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside diagnostic ledger domain"):
            _outer(10, 0.9)

    def test_valid_outputs_are_finite_and_nonnegative(self) -> None:
        for n, c in (
            (100, 0.99),
            (10_000, 0.9999),
            (1_000_000, 0.99999),
        ):
            with self.subTest(n=n, c=c):
                scales = _outer(n, c)
                self.assertEqual(set(scales), REQUIRED_OUTER_KEYS)
                for key, value in scales.items():
                    self.assertTrue(isfinite(value), key)
                    self.assertGreaterEqual(value, 0.0, key)
                self.assertAlmostEqual(
                    scales["total_scale_proxy"],
                    sum(
                        value
                        for key, value in scales.items()
                        if key != "total_scale_proxy"
                    ),
                    places=14,
                )


def _schedule_values(n: int) -> tuple[tuple[str, float], ...]:
    """Return the three approved non-probative moving-outer schedules."""
    return (
        ("log_log_n", log(log(n))),
        ("sqrt_log_n", sqrt(log(n))),
        ("cube_root_n", n ** (1.0 / 3.0)),
    )


def _outer_schedule_rows(n_values: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for n in n_values:
        for schedule, requested_lambda in _schedule_values(n):
            c = 1.0 - requested_lambda / n
            rows.append(
                {
                    "schedule": schedule,
                    "n": n,
                    "requested_lambda": requested_lambda,
                    "c": c,
                    "scales": _outer(n, c),
                }
            )
    return tuple(rows)


class OuterLedgerScheduleDiagnosticTests(unittest.TestCase):
    def test_slow_schedule_rows_are_finite_identities_not_trend_claims(self) -> None:
        rows = _outer_schedule_rows((10_000, 1_000_000))
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["schedule"] for row in rows},
            {"log_log_n", "sqrt_log_n", "cube_root_n"},
        )
        for row in rows:
            scales = row["scales"]
            self.assertGreater(row["requested_lambda"], 0.0)
            for value in scales.values():
                self.assertTrue(isfinite(value))
                self.assertGreaterEqual(value, 0.0)
            self.assertTrue(
                isclose(
                    scales["total_scale_proxy"],
                    sum(
                        value
                        for key, value in scales.items()
                        if key != "total_scale_proxy"
                    ),
                    rel_tol=2.0e-14,
                    abs_tol=0.0,
                )
            )


def run_behavioral_zero_stub_red() -> unittest.result.TestResult:
    """Run all Task 6 behavior tests against an importable zero ledger."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (
        OuterLedgerContractTests,
        OuterLedgerLiteralTests,
        OuterLedgerDomainTests,
        OuterLedgerScheduleDiagnosticTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(case))
    with _patched_attribute("OuterErrorScales", _ZeroOuterErrorScales):
        with _patched_attribute(
            "outer_error_scale_proxy", _zero_outer_error_scale_proxy
        ):
            return unittest.TextTestRunner(verbosity=2).run(suite)


def run_tail_exponent_mutant_red() -> unittest.result.TestResult:
    """Kill the realistic mutation ``sqrt(B / h) -> B / h``."""
    original_subject = getattr(outer_probe, "outer_error_scale_proxy")

    def linear_tail_mutant(n: int, c: float) -> dict[str, float]:
        scales = dict(original_subject(n, c))
        parameters = outer_probe.adaptive_block_parameters(n, c)
        old_tail = scales["tail_overhead"]
        scales["tail_overhead"] = (
            parameters["block_count"] / parameters["h"]
        )
        scales["total_scale_proxy"] += scales["tail_overhead"] - old_tail
        return scales

    suite = unittest.TestSuite(
        [
            OuterLedgerLiteralTests(
                "test_linear_tail_overhead_cannot_replace_the_square_root"
            ),
            OuterLedgerLiteralTests(
                "test_each_formula_is_recomputed_from_frozen_adaptive_literals"
            ),
        ]
    )
    with _patched_attribute("outer_error_scale_proxy", linear_tail_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    unittest.main()
