"""Behavior tests for compact-lambda parameters and rectangular pinching."""

from math import log, pi, sqrt
from typing import get_type_hints
import unittest
import warnings

import numpy as np

import proofs.weighted_hull_asymptotic_probe as asymptotic_probe
from proofs.weighted_hull_asymptotic_probe import (
    ContinuumParameters,
    balanced_block_sizes,
    continuum_parameters,
    ell,
    hermitian_dilation,
    p1,
    rectangular_block_pinching,
    weighted_volterra,
)


class EndpointSafeScalarTests(unittest.TestCase):
    def test_scalar_endpoints_and_direct_quadrature_sentinel(self) -> None:
        # The c=1/2 literals were independently evaluated from
        # (1/2) integral_0^{2 pi} (1 - 2 c cos t + c^2)^(-1/2) dt.
        self.assertAlmostEqual(ell(0.0), pi, places=14)
        self.assertAlmostEqual(p1(0.0), 1.0, places=14)
        self.assertEqual(p1(1.0), 0.0)
        self.assertAlmostEqual(ell(0.5), 3.3715007096251912, places=13)
        self.assertAlmostEqual(p1(0.5), 0.86378971535185334, places=13)

    def test_p1_obeys_the_exact_ell_identity_away_from_c1(self) -> None:
        # These ell literals come from the defining theta integral, evaluated
        # independently of the production elliptic-integral implementation.
        cases = (
            (0.0, pi),
            (0.2, 3.1737356949083315),
            (0.5, 3.3715007096251912),
            (0.9, 4.5610982768455406),
        )
        for c, expected_ell in cases:
            with self.subTest(c=c):
                expected_p1 = (
                    (1.0 - c * c) * expected_ell**2 / (pi * pi)
                )
                self.assertAlmostEqual(ell(c), expected_ell, places=13)
                self.assertAlmostEqual(p1(c), expected_p1, places=13)

    def test_scalar_domains_reject_endpoints_used_in_illegal_divisions(self) -> None:
        for invalid_c in (-0.1, 1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(function="ell", c=invalid_c):
                with self.assertRaises(ValueError):
                    ell(invalid_c)
        for invalid_c in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(function="p1", c=invalid_c):
                with self.assertRaises(ValueError):
                    p1(invalid_c)


class WeightedVolterraTests(unittest.TestCase):
    def test_entries_match_a_hand_written_weighted_upper_triangle(self) -> None:
        expected = np.array(
            [
                [1.0, 0.5, 0.25, 0.125],
                [0.0, 1.0, 0.5, 0.25],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(weighted_volterra(4, 0.5), expected)

    def test_c0_is_identity_without_warning_or_nonfinite_intermediate(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="raise"):
                actual = weighted_volterra(4, 0.0)

        np.testing.assert_array_equal(actual, np.eye(4))
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(caught, [])

    def test_invalid_dimension_or_overlap_is_rejected(self) -> None:
        cases = ((0, 0.5), (3, -0.1), (3, 1.1), (3, float("nan")))
        for m, c in cases:
            with self.subTest(m=m, c=c):
                with self.assertRaises(ValueError):
                    weighted_volterra(m, c)


class BalancedContinuumParameterTests(unittest.TestCase):
    def test_balanced_sizes_sum_to_n_and_differ_by_at_most_one(self) -> None:
        cases = ((10, 3, [3, 3, 4]), (12, 5, [2, 2, 2, 3, 3]), (5, 5, [1] * 5))
        for n, block_count, expected_sorted in cases:
            with self.subTest(n=n, block_count=block_count):
                sizes = balanced_block_sizes(n, block_count)
                self.assertEqual(len(sizes), block_count)
                self.assertEqual(sum(sizes), n)
                self.assertEqual(sorted(sizes), expected_sorted)
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_balanced_sizes_reject_impossible_nonempty_partitions(self) -> None:
        for n, block_count in ((0, 1), (5, 0), (5, 6)):
            with self.subTest(n=n, block_count=block_count):
                with self.assertRaises(ValueError):
                    balanced_block_sizes(n, block_count)

    def test_continuum_ledger_matches_an_independent_n100_calculation(self) -> None:
        params = continuum_parameters(100, 1.0)
        expected_sizes = (15, 15, 14, 14, 14, 14, 14)

        self.assertEqual(
            set(params),
            {
                "n",
                "lambda_value",
                "c",
                "ell",
                "p1",
                "block_count",
                "minimum_block_size",
                "sizes",
                "m_epsilon",
                "log_m_over_ell",
            },
        )
        self.assertEqual(params["n"], 100)
        self.assertEqual(params["lambda_value"], 1.0)
        self.assertAlmostEqual(params["c"], 0.99, places=15)
        self.assertAlmostEqual(params["ell"], 6.713201046722445, places=12)
        self.assertAlmostEqual(params["p1"], 0.090868349185905375, places=12)
        self.assertEqual(params["block_count"], 7)
        self.assertEqual(params["minimum_block_size"], 14)
        self.assertEqual(params["sizes"], expected_sizes)
        self.assertAlmostEqual(params["m_epsilon"], 0.14, places=14)
        self.assertAlmostEqual(
            params["log_m_over_ell"], log(14) / 6.713201046722445, places=12
        )

    def test_continuum_ledger_uses_the_declared_typed_dict(self) -> None:
        required = {
            "n",
            "lambda_value",
            "c",
            "ell",
            "p1",
            "block_count",
            "minimum_block_size",
            "sizes",
            "m_epsilon",
            "log_m_over_ell",
        }
        self.assertEqual(ContinuumParameters.__required_keys__, frozenset(required))
        self.assertIs(
            get_type_hints(continuum_parameters)["return"], ContinuumParameters
        )
        hints = get_type_hints(ContinuumParameters)
        self.assertIs(hints["n"], int)
        self.assertEqual(hints["sizes"], tuple[int, ...])

    def test_continuum_ledger_rejects_noncontinuum_inputs(self) -> None:
        for n, lambda_value in ((0, 1.0), (100, 0.0), (100, -1.0), (100, 101.0)):
            with self.subTest(n=n, lambda_value=lambda_value):
                with self.assertRaises(ValueError):
                    continuum_parameters(n, lambda_value)


class HermitianDilationTests(unittest.TestCase):
    def test_complex_rectangular_dilation_doubles_the_nuclear_norm(self) -> None:
        a = np.array([[3.0j, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=complex)
        dilation = hermitian_dilation(a)

        self.assertEqual(dilation.shape, (5, 5))
        np.testing.assert_array_equal(dilation[:2, 2:], a)
        np.testing.assert_array_equal(dilation[2:, :2], a.conjugate().T)
        np.testing.assert_array_equal(dilation[:2, :2], np.zeros((2, 2)))
        np.testing.assert_array_equal(dilation[2:, 2:], np.zeros((3, 3)))
        self.assertAlmostEqual(np.linalg.norm(a, ord="nuc"), 7.0, places=13)
        self.assertAlmostEqual(np.linalg.norm(dilation, ord="nuc"), 14.0, places=13)

    def test_dilation_rejects_nonmatrices(self) -> None:
        for invalid in (np.array([1.0, 2.0]), np.zeros((2, 2, 1))):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    hermitian_dilation(invalid)


class RectangularPinchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = np.array(
            [
                [1 + 1j, 2, 3, 4, 5],
                [6, 7 - 2j, 8, 9, 10],
                [11, 12, 13 + 3j, 14, 15],
                [16, 17, 18, 19, 20 - 1j],
            ],
            dtype=complex,
        )

    def test_two_matched_complex_blocks_are_retained_and_contractive(self) -> None:
        row_blocks = ((0, 2), (1, 3))
        col_blocks = ((0, 3), (1, 4))
        expected = np.array(
            [
                [1 + 1j, 0, 0, 4, 0],
                [0, 7 - 2j, 0, 0, 10],
                [11, 0, 0, 14, 0],
                [0, 17, 0, 0, 20 - 1j],
            ],
            dtype=complex,
        )

        retained = rectangular_block_pinching(self.a, row_blocks, col_blocks)

        np.testing.assert_array_equal(retained, expected)
        self.assertLessEqual(
            np.linalg.norm(retained, ord="nuc"),
            np.linalg.norm(self.a, ord="nuc") + 1e-12,
        )

    def test_mismatched_overlapping_or_out_of_range_blocks_are_rejected(self) -> None:
        cases = (
            (((0,), (1,)), ((0,),)),
            (((0, 1), (1, 2)), ((0,), (1,))),
            (((0,), (1,)), ((0, 2), (2, 3))),
            (((0,), (4,)), ((0,), (1,))),
            (((0,), (1,)), ((0,), (5,))),
            (((-1,), (1,)), ((0,), (1,))),
        )
        for row_blocks, col_blocks in cases:
            with self.subTest(row_blocks=row_blocks, col_blocks=col_blocks):
                with self.assertRaises(ValueError):
                    rectangular_block_pinching(self.a, row_blocks, col_blocks)

    def test_nonintegral_block_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rectangular_block_pinching(self.a, ((0.5,),), ((0,),))

    def test_arbitrary_single_entry_deletion_can_increase_nuclear_norm(self) -> None:
        full = np.ones((2, 2), dtype=float)
        deleted = np.array([[0.0, 1.0], [1.0, 1.0]])

        self.assertAlmostEqual(np.linalg.norm(full, ord="nuc"), 2.0, places=13)
        self.assertAlmostEqual(
            np.linalg.norm(deleted, ord="nuc"), sqrt(5.0), places=13
        )
        self.assertGreater(
            np.linalg.norm(deleted, ord="nuc"), np.linalg.norm(full, ord="nuc")
        )


class NonfiniteMatrixValidationTests(unittest.TestCase):
    def test_hermitian_dilation_rejects_nan_and_infinite_entries(self) -> None:
        invalid_matrices = (
            np.array([[float("nan"), 0.0]]),
            np.array([[1.0, float("inf")]]),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    hermitian_dilation(matrix)

    def test_rectangular_pinching_rejects_nan_and_infinite_entries(self) -> None:
        invalid_matrices = (
            np.array([[float("nan"), 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, float("-inf")]]),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    rectangular_block_pinching(matrix, ((0,),), ((0,),))


def _zero_ell(c: float) -> float:
    return 0.0


def _zero_p1(c: float) -> float:
    return 0.0


def _zero_weighted_volterra(m: int, c: float) -> np.ndarray:
    size = max(int(m), 0)
    return np.zeros((size, size), dtype=float)


def _zero_balanced_block_sizes(n: int, block_count: int) -> tuple[int, ...]:
    count = max(int(block_count), 0)
    return (0,) * count


def _zero_continuum_parameters(
    n: int, lambda_value: float
) -> ContinuumParameters:
    return {
        "n": int(n),
        "lambda_value": float(lambda_value),
        "c": 0.0,
        "ell": 0.0,
        "p1": 0.0,
        "block_count": 0,
        "minimum_block_size": 0,
        "sizes": (),
        "m_epsilon": 0.0,
        "log_m_over_ell": 0.0,
    }


def _zero_hermitian_dilation(a: np.ndarray) -> np.ndarray:
    array = np.asarray(a)
    if array.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    size = array.shape[0] + array.shape[1]
    return np.zeros((size, size), dtype=np.result_type(array.dtype, float))


def _zero_rectangular_block_pinching(
    a: np.ndarray,
    row_blocks: tuple[tuple[int, ...], ...],
    col_blocks: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    return np.zeros_like(np.asarray(a))


def run_behavioral_zero_stub_red() -> unittest.result.TestResult:
    """Replay the original 17-test RED against importable zero behavior stubs."""
    replacements = {
        "ell": _zero_ell,
        "p1": _zero_p1,
        "weighted_volterra": _zero_weighted_volterra,
        "balanced_block_sizes": _zero_balanced_block_sizes,
        "continuum_parameters": _zero_continuum_parameters,
        "hermitian_dilation": _zero_hermitian_dilation,
        "rectangular_block_pinching": _zero_rectangular_block_pinching,
    }
    original = {name: globals()[name] for name in replacements}
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in (
        EndpointSafeScalarTests,
        WeightedVolterraTests,
        BalancedContinuumParameterTests,
        HermitianDilationTests,
        RectangularPinchingTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    try:
        globals().update(replacements)
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals().update(original)


def run_nonfinite_validation_mutant_red() -> unittest.result.TestResult:
    """Prove the focused tests fail when finite-entry validation is removed."""
    original = asymptotic_probe._finite_numeric_matrix

    def matrix_without_finite_check(a: np.ndarray) -> np.ndarray:
        array = np.asarray(a)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("a must be a two-dimensional numeric matrix")
        return array

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        NonfiniteMatrixValidationTests
    )
    try:
        asymptotic_probe._finite_numeric_matrix = matrix_without_finite_check
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        asymptotic_probe._finite_numeric_matrix = original


if __name__ == "__main__":
    unittest.main()
