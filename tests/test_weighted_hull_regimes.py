"""Behavior tests for the finite Task 7 regime diagnostics.

Python interval coordinates elsewhere in the audit code are zero-based closed
pairs, while the manuscript uses one-based endpoints.  This module has no
interval-coordinate API: ``n`` is the common site count, and every performance
quantity in the accompanying proof retains the external prior
``1 / M_n``, ``M_n = n(n + 1) / 2``.

The labels tested here are caller-selected finite threshold bands.  They are
not asymptotic classifiers and are never used as theorem evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from inspect import Parameter, Signature, getsource, signature
from math import isclose
from textwrap import dedent
from typing import Literal, Sequence, TypedDict, get_type_hints
import unittest

import proofs.weighted_hull_asymptotic_probe as regime_probe


REQUIRED_DIAGNOSTIC_KEYS = {
    "n",
    "c",
    "lambda_value",
    "h",
    "ell",
    "p1",
    "tau",
    "inner_lambda_max",
    "outer_lambda_min",
    "threshold_regime",
}


class _ZeroRegimeDiagnostics(TypedDict):
    n: int
    c: float
    lambda_value: float
    h: float
    ell: float
    p1: float
    tau: float
    inner_lambda_max: float
    outer_lambda_min: float
    threshold_regime: Literal["inner", "continuum", "outer"]


class _WrongFieldRegimeDiagnostics(TypedDict):
    """Contract mutant: ``h`` has the wrong public field type."""

    n: int
    c: float
    lambda_value: float
    h: int
    ell: float
    p1: float
    tau: float
    inner_lambda_max: float
    outer_lambda_min: float
    threshold_regime: Literal["inner", "continuum", "outer"]


EXPECTED_DIAGNOSTIC_HINTS = {
    "n": int,
    "c": float,
    "lambda_value": float,
    "h": float,
    "ell": float,
    "p1": float,
    "tau": float,
    "inner_lambda_max": float,
    "outer_lambda_min": float,
    "threshold_regime": Literal["inner", "continuum", "outer"],
}

FLOAT_DIAGNOSTIC_FIELDS = (
    "c",
    "lambda_value",
    "h",
    "ell",
    "p1",
    "tau",
    "inner_lambda_max",
    "outer_lambda_min",
)

VALID_THRESHOLD_REGIMES = ("inner", "continuum", "outer")


def _zero_regime_diagnostics(
    n: int,
    c: float,
    *,
    inner_lambda_max: float,
    outer_lambda_min: float,
) -> _ZeroRegimeDiagnostics:
    del n, c, inner_lambda_max, outer_lambda_min
    return {
        "n": 0,
        "c": 0.0,
        "lambda_value": 0.0,
        "h": 0.0,
        "ell": 0.0,
        "p1": 0.0,
        "tau": 0.0,
        "inner_lambda_max": 0.0,
        "outer_lambda_min": 0.0,
        "threshold_regime": "inner",
    }


def _zero_plateau_overlap(n: int, anchors: Sequence[int]) -> float:
    del n, anchors
    return 0.0


def _diagnostics(
    n: int,
    c: float,
    *,
    inner_lambda_max: float,
    outer_lambda_min: float,
) -> dict[str, object]:
    subject = getattr(
        regime_probe, "regime_diagnostics", _zero_regime_diagnostics
    )
    return subject(
        n,
        c,
        inner_lambda_max=inner_lambda_max,
        outer_lambda_min=outer_lambda_min,
    )


def _plateau(n: int, anchors: Sequence[int]) -> float:
    subject = getattr(regime_probe, "plateau_overlap", _zero_plateau_overlap)
    return subject(n, anchors)


def _assert_exact_runtime_diagnostic_types(
    test_case: unittest.TestCase,
    diagnostic: dict[str, object],
) -> None:
    """Enforce the exact runtime representation of every public field."""
    with test_case.subTest(runtime_field="n"):
        test_case.assertIs(type(diagnostic["n"]), int)
    for field in FLOAT_DIAGNOSTIC_FIELDS:
        with test_case.subTest(runtime_field=field):
            test_case.assertIs(type(diagnostic[field]), float)
    with test_case.subTest(runtime_field="threshold_regime"):
        label = diagnostic["threshold_regime"]
        test_case.assertIs(type(label), str)
        test_case.assertIn(label, VALID_THRESHOLD_REGIMES)


@contextmanager
def _patched_attribute(name: str, value: object):
    missing = object()
    original = getattr(regime_probe, name, missing)
    setattr(regime_probe, name, value)
    try:
        yield
    finally:
        if original is missing:
            delattr(regime_probe, name)
        else:
            setattr(regime_probe, name, original)


def _one_line_source_mutant(old: str, new: str):
    """Compile one exact one-line mutation of the live diagnostic function."""
    source = dedent(getsource(regime_probe.regime_diagnostics))
    if source.count(old) != 1:
        raise AssertionError(f"expected one mutation site for {old!r}")
    namespace: dict[str, object] = {}
    exec(
        "from __future__ import annotations\n" + source.replace(old, new),
        regime_probe.__dict__,
        namespace,
    )
    return namespace["regime_diagnostics"]


class RegimeDiagnosticContractTests(unittest.TestCase):
    def test_typed_contract_and_explicit_keyword_only_thresholds(self) -> None:
        diagnostic_type = getattr(
            regime_probe, "RegimeDiagnostics", _ZeroRegimeDiagnostics
        )
        hints = get_type_hints(diagnostic_type)
        self.assertEqual(
            diagnostic_type.__required_keys__,
            frozenset(REQUIRED_DIAGNOSTIC_KEYS),
        )
        self.assertEqual(hints, EXPECTED_DIAGNOSTIC_HINTS)

        subject = getattr(
            regime_probe, "regime_diagnostics", _zero_regime_diagnostics
        )
        expected_signature = Signature(
            parameters=(
                Parameter(
                    "n", Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
                Parameter(
                    "c", Parameter.POSITIONAL_OR_KEYWORD, annotation=float
                ),
                Parameter(
                    "inner_lambda_max",
                    Parameter.KEYWORD_ONLY,
                    annotation=float,
                ),
                Parameter(
                    "outer_lambda_min",
                    Parameter.KEYWORD_ONLY,
                    annotation=float,
                ),
            ),
            return_annotation=diagnostic_type,
        )
        subject_signature: Signature = signature(subject, eval_str=True)
        self.assertEqual(subject_signature, expected_signature)
        self.assertEqual(
            get_type_hints(subject),
            {
                "n": int,
                "c": float,
                "inner_lambda_max": float,
                "outer_lambda_min": float,
                "return": diagnostic_type,
            },
        )
        with self.assertRaises(TypeError):
            subject(8, 0.75)


class RegimeDiagnosticLiteralTests(unittest.TestCase):
    def test_n100_c099_matches_hand_checked_finite_scales(self) -> None:
        diagnostic = _diagnostics(
            100,
            0.99,
            inner_lambda_max=0.25,
            outer_lambda_min=4.0,
        )
        expected = {
            "n": 100,
            "c": 0.99,
            "lambda_value": 1.0000000000000009,
            "h": 9.086834918590371,
            "ell": 6.713201046722383,
            "p1": 0.09086834918590371,
            "tau": 21.207592441913615,
            "inner_lambda_max": 0.25,
            "outer_lambda_min": 4.0,
            "threshold_regime": "continuum",
        }
        self.assertEqual(set(diagnostic), REQUIRED_DIAGNOSTIC_KEYS)
        _assert_exact_runtime_diagnostic_types(self, diagnostic)
        for key, literal in expected.items():
            with self.subTest(key=key):
                if isinstance(literal, float):
                    self.assertTrue(
                        isclose(
                            diagnostic[key],
                            literal,
                            rel_tol=2.0e-15,
                            abs_tol=0.0,
                        )
                    )
                else:
                    self.assertEqual(diagnostic[key], literal)

    def test_threshold_bands_include_both_declared_boundaries(self) -> None:
        fixtures = (
            (0.875, "inner"),
            (0.75, "continuum"),
            (0.5, "outer"),
        )
        for c, expected_band in fixtures:
            with self.subTest(c=c):
                diagnostic = _diagnostics(
                    8,
                    c,
                    inner_lambda_max=1.0,
                    outer_lambda_min=4.0,
                )
                _assert_exact_runtime_diagnostic_types(self, diagnostic)
                self.assertEqual(
                    diagnostic["threshold_regime"], expected_band
                )

    def test_threshold_choices_change_only_the_finite_band_label(self) -> None:
        continuum = _diagnostics(
            8,
            0.75,
            inner_lambda_max=1.0,
            outer_lambda_min=3.0,
        )
        inner = _diagnostics(
            8,
            0.75,
            inner_lambda_max=2.0,
            outer_lambda_min=3.0,
        )
        _assert_exact_runtime_diagnostic_types(self, continuum)
        _assert_exact_runtime_diagnostic_types(self, inner)
        self.assertEqual(continuum["threshold_regime"], "continuum")
        self.assertEqual(inner["threshold_regime"], "inner")
        scale_keys = {
            "n", "c", "lambda_value", "h", "ell", "p1", "tau"
        }
        self.assertEqual(
            {key: continuum[key] for key in scale_keys},
            {key: inner[key] for key in scale_keys},
        )

    def test_fraction_inputs_are_normalized_to_exact_float_outputs(
        self,
    ) -> None:
        diagnostic = _diagnostics(
            8,
            Fraction(3, 4),
            inner_lambda_max=Fraction(1, 2),
            outer_lambda_min=Fraction(3, 1),
        )
        _assert_exact_runtime_diagnostic_types(self, diagnostic)
        self.assertEqual(diagnostic["n"], 8)
        self.assertEqual(diagnostic["c"], 0.75)
        self.assertEqual(diagnostic["lambda_value"], 2.0)
        self.assertEqual(diagnostic["inner_lambda_max"], 0.5)
        self.assertEqual(diagnostic["outer_lambda_min"], 3.0)
        self.assertEqual(diagnostic["threshold_regime"], "continuum")


class RegimeDiagnosticDomainTests(unittest.TestCase):
    def test_invalid_finite_scale_inputs_are_rejected(self) -> None:
        for n, c in (
            (1, 0.5),
            (8, -0.01),
            (8, 1.0),
            (8, float("nan")),
            (True, 0.5),
        ):
            with self.subTest(n=n, c=c):
                with self.assertRaises(ValueError):
                    _diagnostics(
                        n,
                        c,
                        inner_lambda_max=1.0,
                        outer_lambda_min=4.0,
                    )

    def test_thresholds_must_be_finite_positive_and_ordered(self) -> None:
        for inner_max, outer_min in (
            (1.0, 1.0),
            (2.0, 1.0),
            (-2.0, -1.0),
            (1.0, -1.0),
            (float("nan"), 4.0),
            (1.0, float("inf")),
            (True, 4.0),
        ):
            with self.subTest(
                inner_lambda_max=inner_max,
                outer_lambda_min=outer_min,
            ):
                with self.assertRaises(ValueError):
                    _diagnostics(
                        8,
                        0.75,
                        inner_lambda_max=inner_max,
                        outer_lambda_min=outer_min,
                    )

    def test_inner_threshold_is_strictly_positive_including_negative_zero(
        self,
    ) -> None:
        for inner_max in (-1.0, -0.25, 0.0, -0.0):
            with self.subTest(inner_lambda_max=inner_max):
                with self.assertRaises(ValueError):
                    _diagnostics(
                        8,
                        0.75,
                        inner_lambda_max=inner_max,
                        outer_lambda_min=4.0,
                    )


class PlateauOverlapContractTests(unittest.TestCase):
    def test_complete_signature_and_annotations(self) -> None:
        subject = getattr(
            regime_probe, "plateau_overlap", _zero_plateau_overlap
        )
        expected_signature = Signature(
            parameters=(
                Parameter(
                    "n", Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
                Parameter(
                    "anchors",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Sequence[int],
                ),
            ),
            return_annotation=float,
        )
        self.assertEqual(
            signature(subject, eval_str=True), expected_signature
        )
        self.assertEqual(
            get_type_hints(subject),
            {"n": int, "anchors": Sequence[int], "return": float},
        )


class PlateauOverlapTests(unittest.TestCase):
    def test_exact_half_open_plateau_values(self) -> None:
        anchors = (2, 4, 16, 256)
        fixtures = (
            (2, 0.75),
            (3, 0.75),
            (4, 0.9375),
            (15, 0.9375),
            (16, 0.99609375),
            (255, 0.99609375),
        )
        for n, expected in fixtures:
            with self.subTest(n=n):
                self.assertEqual(_plateau(n, anchors), expected)

    def test_n_must_lie_in_the_finite_anchor_prefix(self) -> None:
        for n in (1, 256, 300):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    _plateau(n, (2, 4, 16, 256))

    def test_anchors_must_be_strict_and_follow_the_square_recurrence(self) -> None:
        for anchors in ((2, 2, 4), (2, 5, 25), (3, 9, 80)):
            with self.subTest(anchors=anchors):
                with self.assertRaises(ValueError):
                    _plateau(3, anchors)

    def test_anchor_and_index_types_and_minimum_prefix_are_validated(self) -> None:
        fixtures = (
            (2, (2,)),
            (2, (1, 1)),
            (2.0, (2, 4)),
            (2, (2.0, 4)),
            (2, None),
        )
        for n, anchors in fixtures:
            with self.subTest(n=n, anchors=anchors):
                with self.assertRaises(ValueError):
                    _plateau(n, anchors)


def run_behavioral_zero_stub_red() -> unittest.result.TestResult:
    """Run all Task 7 behavior tests against importable shape-safe zeros."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__, fromlist=["*"])
    )
    with _patched_attribute("RegimeDiagnostics", _ZeroRegimeDiagnostics):
        with _patched_attribute("regime_diagnostics", _zero_regime_diagnostics):
            with _patched_attribute("plateau_overlap", _zero_plateau_overlap):
                return unittest.TextTestRunner(verbosity=2).run(suite)


def run_threshold_boundary_mutant_red() -> unittest.result.TestResult:
    """Kill strict comparisons at the two declared threshold boundaries."""
    original_subject = getattr(regime_probe, "regime_diagnostics")

    def strict_boundary_mutant(
        n: int,
        c: float,
        *,
        inner_lambda_max: float,
        outer_lambda_min: float,
    ) -> dict[str, object]:
        diagnostic = dict(
            original_subject(
                n,
                c,
                inner_lambda_max=inner_lambda_max,
                outer_lambda_min=outer_lambda_min,
            )
        )
        lambda_value = diagnostic["lambda_value"]
        if lambda_value == inner_lambda_max:
            diagnostic["threshold_regime"] = "continuum"
        if lambda_value == outer_lambda_min:
            diagnostic["threshold_regime"] = "continuum"
        return diagnostic

    suite = unittest.TestSuite(
        [
            RegimeDiagnosticLiteralTests(
                "test_threshold_bands_include_both_declared_boundaries"
            )
        ]
    )
    with _patched_attribute("regime_diagnostics", strict_boundary_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_negative_inner_positivity_mutant_red() -> unittest.result.TestResult:
    """Kill deletion of the strict-positive inner-threshold requirement."""
    mutant = _one_line_source_mutant(
        "if not 0.0 < inner_max < outer_min:",
        "if not inner_max < outer_min:",
    )
    suite = unittest.TestSuite(
        [
            RegimeDiagnosticDomainTests(
                "test_inner_threshold_is_strictly_positive_including_negative_zero"
            )
        ]
    )
    with _patched_attribute("regime_diagnostics", mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_wrong_typed_dict_field_mutant_red() -> unittest.result.TestResult:
    """Kill the public-contract mutation ``h: float -> h: int``."""
    suite = unittest.TestSuite(
        [
            RegimeDiagnosticContractTests(
                "test_typed_contract_and_explicit_keyword_only_thresholds"
            )
        ]
    )
    with _patched_attribute(
        "RegimeDiagnostics", _WrongFieldRegimeDiagnostics
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_extra_api_parameter_mutant_red() -> unittest.result.TestResult:
    """Kill one extra optional keyword in the public diagnostic signature."""
    original_subject = getattr(regime_probe, "regime_diagnostics")

    def extra_parameter_mutant(
        n: int,
        c: float,
        *,
        inner_lambda_max: float,
        outer_lambda_min: float,
        diagnostic_note: bool = False,
    ) -> regime_probe.RegimeDiagnostics:
        del diagnostic_note
        return original_subject(
            n,
            c,
            inner_lambda_max=inner_lambda_max,
            outer_lambda_min=outer_lambda_min,
        )

    suite = unittest.TestSuite(
        [
            RegimeDiagnosticContractTests(
                "test_typed_contract_and_explicit_keyword_only_thresholds"
            )
        ]
    )
    with _patched_attribute("regime_diagnostics", extra_parameter_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_float_n_runtime_mutant_red() -> unittest.result.TestResult:
    """Kill ``n`` being returned as its floating-point work variable."""
    mutant = _one_line_source_mutant(
        '"n": site_count,',
        '"n": site_count_float,',
    )
    suite = unittest.TestSuite(
        [
            RegimeDiagnosticLiteralTests(
                "test_n100_c099_matches_hand_checked_finite_scales"
            )
        ]
    )
    with _patched_attribute("regime_diagnostics", mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_unnormalized_fraction_outputs_mutant_red(
) -> unittest.result.TestResult:
    """Kill reuse of raw Fraction inputs in the public diagnostic record."""
    original_subject = getattr(regime_probe, "regime_diagnostics")

    def unnormalized_fraction_mutant(
        n: int,
        c: float,
        *,
        inner_lambda_max: float,
        outer_lambda_min: float,
    ) -> regime_probe.RegimeDiagnostics:
        diagnostic = dict(
            original_subject(
                n,
                c,
                inner_lambda_max=inner_lambda_max,
                outer_lambda_min=outer_lambda_min,
            )
        )
        diagnostic["c"] = c
        diagnostic["inner_lambda_max"] = inner_lambda_max
        diagnostic["outer_lambda_min"] = outer_lambda_min
        return diagnostic

    suite = unittest.TestSuite(
        [
            RegimeDiagnosticLiteralTests(
                "test_fraction_inputs_are_normalized_to_exact_float_outputs"
            )
        ]
    )
    with _patched_attribute(
        "regime_diagnostics", unnormalized_fraction_mutant
    ):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_plateau_extra_parameter_mutant_red() -> unittest.result.TestResult:
    """Kill one extra optional parameter in the plateau API."""
    original_subject = getattr(regime_probe, "plateau_overlap")

    def extra_parameter_mutant(
        n: int,
        anchors: Sequence[int],
        diagnostic_note: bool = False,
    ) -> float:
        del diagnostic_note
        return original_subject(n, anchors)

    suite = unittest.TestSuite(
        [
            PlateauOverlapContractTests(
                "test_complete_signature_and_annotations"
            )
        ]
    )
    with _patched_attribute("plateau_overlap", extra_parameter_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


def run_plateau_return_annotation_mutant_red(
) -> unittest.result.TestResult:
    """Kill ``plateau_overlap`` advertising an integer return value."""
    original_subject = getattr(regime_probe, "plateau_overlap")

    def wrong_return_mutant(
        n: int,
        anchors: Sequence[int],
    ) -> int:
        return original_subject(n, anchors)

    suite = unittest.TestSuite(
        [
            PlateauOverlapContractTests(
                "test_complete_signature_and_annotations"
            )
        ]
    )
    with _patched_attribute("plateau_overlap", wrong_return_mutant):
        return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    unittest.main()
