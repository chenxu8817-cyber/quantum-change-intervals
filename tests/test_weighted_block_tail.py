"""Behavior tests for the reusable singular-safe block--tail certificate.

The first TDD checkpoint keeps this module importable before the production
module exists.  Its test-local fallback has shape-safe zero behavior, so the
RED run consists only of failed assertions rather than import/runtime errors.
Once the production module is present, the fallback is unreachable.
"""

from __future__ import annotations

import io
import unittest
import warnings
from math import fsum, isfinite, sqrt
from numbers import Real

import numpy as np

try:
    import proofs.weighted_block_tail_probe as block_tail_probe
    from proofs.weighted_block_tail_probe import (
        extend_certificate,
        extend_with_budget,
        optimized_extension,
    )
except ModuleNotFoundError as error:
    if error.name != "proofs.weighted_block_tail_probe":
        raise

    block_tail_probe = None

    def _zero_square(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        local = np.asarray(q)
        tail = np.asarray(w)
        size = local.shape[0] if local.ndim == 2 else tail.size
        return np.zeros((size, size), dtype=np.result_type(local, tail, float))

    def extend_certificate(
        q: np.ndarray, w: np.ndarray, alpha: float
    ) -> np.ndarray:
        return _zero_square(q, w)

    def extend_with_budget(
        q: np.ndarray, w: np.ndarray, budget: float
    ) -> np.ndarray:
        return _zero_square(q, w)

    def optimized_extension(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        return _zero_square(q, w)


def _zero_right_suffix(n: int, a: int, c: float) -> np.ndarray:
    return np.zeros(max(int(n), 0), dtype=float)


def _zero_left_suffix(n: int, b: int, c: float) -> np.ndarray:
    return np.zeros(max(int(n), 0), dtype=float)


def _zero_split(
    n: int, anchor: int, c: float, block_start: int, block_stop: int
) -> tuple[np.ndarray, float, np.ndarray]:
    size = max(int(n), 0)
    return np.zeros(size), 0.0, np.zeros(size)


def _zero_weighted_local(
    m: int, c: float, k0: np.ndarray | None = None
) -> np.ndarray:
    size = max(int(m), 0)
    return np.zeros((size, size), dtype=float)


def _zero_tensor(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    array = np.asarray(right)
    size = array.shape[0] if array.ndim == 2 else 0
    hull_size = size * (size - 1) // 2
    return np.zeros((hull_size, hull_size), dtype=float)


def _zero_physical(n: int, c: float) -> np.ndarray:
    size = max(int(n), 0)
    return np.zeros(
        (size * (size + 1) // 2, size * (size - 1) // 2), dtype=float
    )


def _zero_global(
    n: int, c: float, block_count: int, *, return_cells: bool = False
):
    size = max(int(n), 0)
    count = max(int(block_count), 0)
    hull_size = size * (size - 1) // 2
    operator = np.zeros((hull_size, hull_size), dtype=float)
    cells = {
        (p, q): np.zeros_like(operator)
        for p in range(count)
        for q in range(p, count)
    }
    return (operator, cells) if return_cells else operator


def _zero_slack(certificate: np.ndarray, state_rows: np.ndarray) -> float:
    return 0.0


if block_tail_probe is not None:
    right_suffix = getattr(block_tail_probe, "right_suffix", _zero_right_suffix)
    left_suffix = getattr(block_tail_probe, "left_suffix", _zero_left_suffix)
    split_right_suffix = getattr(
        block_tail_probe, "split_right_suffix", _zero_split
    )
    split_left_suffix = getattr(block_tail_probe, "split_left_suffix", _zero_split)
    weighted_local_certificate = getattr(
        block_tail_probe, "weighted_local_certificate", _zero_weighted_local
    )
    tensor_cell_certificate = getattr(
        block_tail_probe, "tensor_cell_certificate", _zero_tensor
    )
    physical_hull_rows = getattr(
        block_tail_probe, "physical_hull_rows", _zero_physical
    )
    global_hull_certificate = getattr(
        block_tail_probe, "global_hull_certificate", _zero_global
    )
    minimum_certificate_slack = getattr(
        block_tail_probe, "minimum_certificate_slack", _zero_slack
    )
else:
    right_suffix = _zero_right_suffix
    left_suffix = _zero_left_suffix
    split_right_suffix = _zero_split
    split_left_suffix = _zero_split
    weighted_local_certificate = _zero_weighted_local
    tensor_cell_certificate = _zero_tensor
    physical_hull_rows = _zero_physical
    global_hull_certificate = _zero_global
    minimum_certificate_slack = _zero_slack


def _outer(vector: np.ndarray) -> np.ndarray:
    return np.outer(vector, vector.conjugate())


def _minimum_eigenvalue(matrix: np.ndarray) -> float:
    hermitian = (matrix + matrix.conjugate().T) / 2.0
    return float(np.linalg.eigvalsh(hermitian)[0])


def _assert_psd_relative(
    test: unittest.TestCase,
    slack: np.ndarray,
    certificate: np.ndarray,
    projector: np.ndarray,
) -> None:
    scale = max(
        1.0,
        float(np.linalg.norm(certificate, 2)),
        float(np.linalg.norm(projector, 2)),
    )
    test.assertGreaterEqual(_minimum_eigenvalue(slack), -1.0e-12 * scale)


def _frame_sum_suffix_certificate(m: int) -> np.ndarray:
    return np.fromfunction(
        lambda j, k: np.minimum(j, k) + 1.0, (m, m), dtype=float
    )


def _strict_pairs(n: int) -> list[tuple[int, int]]:
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def _intervals(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a, n)]


class WeightedBlockTailBudgetTests(unittest.TestCase):
    """The abstract theorem and every checked-API precondition."""

    def test_extension_coefficients_match_independent_literal(self) -> None:
        q = np.array([[2.0, 0.5], [0.5, 1.0]])
        w = np.array([1.0, -2.0])
        expected = np.array(
            [[28.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 28.0 / 3.0]]
        )
        np.testing.assert_allclose(
            extend_certificate(q, w, 3.0), expected, rtol=0.0, atol=2e-15
        )

    def test_budget_not_trace_selects_the_extension_scale(self) -> None:
        q = np.diag([0.25, 0.0])
        w = np.array([0.0, 1.0])
        # L=1 gives coefficients 2 and 2.  Using tr(Q)=1/4 instead would
        # produce coefficients 3 and 3/2, so this literal distinguishes them.
        expected = np.diag([0.5, 2.0])
        actual = extend_with_budget(q, w, 1.0)
        np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(float(np.trace(actual).real), 4.0)

    def test_exact_zero_budget_uses_tail_projector(self) -> None:
        q = np.zeros((4, 4))
        w = np.array([0.0, 0.0, 0.6, 0.8])
        expected = _outer(w)
        np.testing.assert_array_equal(extend_with_budget(q, w, 0.0), expected)
        np.testing.assert_array_equal(optimized_extension(q, w), expected)

        complex_q = np.zeros((2, 2), dtype=complex)
        complex_w = np.array([0.6 + 0.2j, -0.1j])
        # Literal conjugating outer product.  This oracle does not call
        # _outer or either production extension helper.
        complex_expected = np.array(
            [[0.4 + 0.0j, -0.02 + 0.06j], [-0.02 - 0.06j, 0.01 + 0.0j]]
        )
        np.testing.assert_allclose(
            extend_with_budget(complex_q, complex_w, 0.0),
            complex_expected,
            rtol=0.0,
            atol=1.0e-16,
        )
        np.testing.assert_allclose(
            optimized_extension(complex_q, complex_w),
            complex_expected,
            rtol=0.0,
            atol=1.0e-16,
        )

    def test_small_positive_cost_is_never_discarded(self) -> None:
        epsilon = 5.0e-15
        q = np.diag([epsilon, 0.0])
        w = np.array([0.0, 1.0])
        expected = np.diag(
            [epsilon + epsilon**0.5, 1.0 + epsilon**0.5]
        )

        actual = optimized_extension(q, w)

        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
        self.assertGreater(actual[0, 0], 0.5 * epsilon**0.5)
        x = np.array([epsilon**0.5, 0.0])
        u = x + 0.4 * w
        scale = max(1.0, np.linalg.norm(actual, 2), np.linalg.norm(_outer(u), 2))
        self.assertGreaterEqual(
            _minimum_eigenvalue(actual - _outer(u)), -1.0e-12 * scale
        )

        minimum_positive = np.nextafter(0.0, 1.0)
        subnormal_q = np.diag([minimum_positive, 0.0])
        subnormal_w = np.array([0.0, 1.0])
        subnormal = extend_with_budget(
            subnormal_q, subnormal_w, minimum_positive
        )
        self.assertTrue(np.isfinite(subnormal).all())
        self.assertGreaterEqual(_minimum_eigenvalue(subnormal), 0.0)
        self.assertLessEqual(
            float(np.trace(subnormal).real),
            (minimum_positive**0.5 + 1.0) ** 2,
        )

    def test_singular_q_is_accepted_and_dominates_compatible_vectors(self) -> None:
        q = np.diag([0.25, 0.0]).astype(complex)
        w = np.array([0.0, 0.6j])
        certificate = extend_with_budget(q, w, 1.0)
        for gamma in (0.0, 0.25, 1.0):
            with self.subTest(gamma=gamma):
                x = np.array([0.5, 0.0])
                vector = x + gamma * w
                scale = max(
                    1.0,
                    float(np.linalg.norm(certificate, 2)),
                    float(np.linalg.norm(_outer(vector), 2)),
                )
                self.assertGreaterEqual(
                    _minimum_eigenvalue(certificate - _outer(vector)),
                    -1.0e-12 * scale,
                )

    def test_optimized_rank_one_literal_uses_one_validated_representative(self) -> None:
        a = np.array([-0.02593438201999467, -0.537509346741146])
        q = np.outer(a, a)
        w = np.array([0.0, 0.0])
        original = block_tail_probe._validated_positive_operator
        calls = 0

        def counted(value: np.ndarray) -> tuple[np.ndarray, float, float]:
            nonlocal calls
            calls += 1
            return original(value)

        block_tail_probe._validated_positive_operator = counted
        try:
            certificate = block_tail_probe.optimized_extension(q, w)
        finally:
            block_tail_probe._validated_positive_operator = original

        self.assertEqual(calls, 1)
        self.assertGreaterEqual(_minimum_eigenvalue(certificate), -1.0e-15)
        self.assertLessEqual(
            float(np.trace(certificate).real),
            (sqrt(float(np.trace(q).real)) + 1.0) ** 2,
        )
        for gamma in (0.0, 0.3, 1.0):
            vector = a + gamma * w
            _assert_psd_relative(
                self, certificate - _outer(vector), certificate, _outer(vector)
            )

    def test_explicit_budget_path_validates_q_once(self) -> None:
        q = np.diag([0.25, 0.0])
        w = np.array([0.0, 1.0])
        original = block_tail_probe._validated_positive_operator
        calls = 0

        def counted(value: np.ndarray) -> tuple[np.ndarray, float, float]:
            nonlocal calls
            calls += 1
            return original(value)

        block_tail_probe._validated_positive_operator = counted
        try:
            block_tail_probe.extend_with_budget(q, w, 1.0)
        finally:
            block_tail_probe._validated_positive_operator = original
        self.assertEqual(calls, 1)

    def test_deterministic_rank_deficient_scan_has_no_false_rejection(self) -> None:
        summary = run_deterministic_rank_deficient_scan()
        self.assertEqual(summary["cases"], 40)
        self.assertEqual(summary["false_rejections"], 0)
        self.assertGreaterEqual(summary["minimum_relative_slack"], -5.0e-14)
        self.assertLessEqual(summary["maximum_trace_excess"], 5.0e-13)

    def test_negative_budget_reports_measured_violation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"budget must be nonnegative; measured budget=-0\.25",
        ):
            extend_with_budget(np.zeros((2, 2)), np.zeros(2), -0.25)

    def test_nonfinite_budget_is_rejected(self) -> None:
        for budget in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "finite"):
                    extend_with_budget(np.zeros((2, 2)), np.zeros(2), budget)

    def test_q_must_be_a_two_dimensional_square_matrix(self) -> None:
        for q in (np.zeros(2), np.zeros((2, 3)), np.zeros((2, 2, 1))):
            with self.subTest(shape=q.shape):
                with self.assertRaisesRegex(ValueError, "square"):
                    extend_with_budget(q, np.zeros(2), 1.0)

    def test_w_must_be_a_compatible_vector(self) -> None:
        for w in (np.zeros((2, 1)), np.zeros(3)):
            with self.subTest(shape=w.shape):
                with self.assertRaisesRegex(ValueError, "vector|dimension"):
                    extend_with_budget(np.zeros((2, 2)), w, 1.0)

    def test_nonfinite_q_entry_reports_the_measured_count(self) -> None:
        q = np.array([[0.0, float("nan")], [0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, r"q.*nonfinite.*1"):
            extend_with_budget(q, np.zeros(2), 1.0)

    def test_nonfinite_w_entry_reports_the_measured_count(self) -> None:
        w = np.array([0.0, float("inf")])
        with self.assertRaisesRegex(ValueError, r"w.*nonfinite.*1"):
            extend_with_budget(np.zeros((2, 2)), w, 1.0)

    def test_nonhermitian_q_reports_operator_norm_violation(self) -> None:
        q = np.array([[1.0, 0.25], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, r"Hermitian violation=.*0\.25"):
            extend_with_budget(q, np.zeros(2), 2.0)
        small_q = np.array([[1.0e-15, 1.0e-16], [0.0, 1.0e-15]])
        with self.assertRaisesRegex(
            ValueError, r"Hermitian violation=.*(?:e-17|e-16)"
        ):
            extend_with_budget(small_q, np.zeros(2), 3.0e-15)

    def test_indefinite_q_reports_negative_eigenvalue_violation(self) -> None:
        q = np.diag([1.0, -0.125])
        with self.assertRaisesRegex(ValueError, r"PSD violation=.*0\.125"):
            extend_with_budget(q, np.zeros(2), 2.0)
        small_q = np.diag([1.0e-15, -1.0e-16])
        with self.assertRaisesRegex(
            ValueError, r"PSD violation=.*(?:e-17|e-16)"
        ):
            extend_with_budget(small_q, np.zeros(2), 1.0e-15)

    def test_trace_above_budget_reports_excess(self) -> None:
        q = np.diag([0.75, 0.5])
        with self.assertRaisesRegex(ValueError, r"trace-budget violation=.*0\.25"):
            extend_with_budget(q, np.zeros(2), 1.0)
        small_q = np.diag([1.0e-15, 0.0])
        with self.assertRaisesRegex(ValueError, r"trace-budget violation=.*e-16"):
            extend_with_budget(small_q, np.array([0.0, 1.0]), 1.0e-16)

    def test_tail_norm_above_one_reports_excess(self) -> None:
        w = np.array([0.0, 1.25])
        with self.assertRaisesRegex(ValueError, r"tail-norm violation=.*0\.25"):
            extend_with_budget(np.zeros((2, 2)), w, 1.0)

    def test_zero_budget_rejects_nonzero_q_even_below_generic_tolerance(self) -> None:
        q = np.diag([np.finfo(float).eps, 0.0])
        with self.assertRaisesRegex(ValueError, r"zero-budget Q norm=.*e-16"):
            extend_with_budget(q, np.zeros(2), 0.0)


class ExactSuffixSplitTests(unittest.TestCase):
    def test_right_suffix_split_has_the_declared_exponent(self) -> None:
        n, a, c, start, stop = 6, 1, 0.5, 1, 3
        s = sqrt(0.75)
        expected_u = np.array([0.0, s, s / 2, s / 4, s / 8, s / 16])
        expected_x = np.array([0.0, s, s / 2, s / 4, 0.0, 0.0])
        expected_w = np.array([0.0, 0.0, 0.0, 0.0, s, s / 2])
        x, gamma, w = split_right_suffix(n, a, c, start, stop)

        np.testing.assert_allclose(right_suffix(n, a, c), expected_u, atol=1e-15)
        np.testing.assert_allclose(x, expected_x, atol=1e-15)
        self.assertAlmostEqual(gamma, 0.5**3, places=15)
        np.testing.assert_allclose(w, expected_w, atol=1e-15)
        np.testing.assert_allclose(x + gamma * w, expected_u, atol=1e-15)

    def test_left_split_is_exactly_the_reflected_right_split(self) -> None:
        n, b, c, start, stop = 7, 5, 0.4, 3, 5
        x_left, gamma_left, w_left = split_left_suffix(n, b, c, start, stop)
        reflected_start = n - 1 - stop
        reflected_stop = n - 1 - start
        x_right, gamma_right, w_right = split_right_suffix(
            n, n - 1 - b, c, reflected_start, reflected_stop
        )

        np.testing.assert_allclose(
            left_suffix(n, b, c), right_suffix(n, n - 1 - b, c)[::-1]
        )
        np.testing.assert_allclose(x_left, x_right[::-1])
        self.assertAlmostEqual(gamma_left, gamma_right, places=15)
        np.testing.assert_allclose(w_left, w_right[::-1])
        np.testing.assert_allclose(
            x_left + gamma_left * w_left, left_suffix(n, b, c)
        )

    def test_every_block_split_has_bounded_gamma_and_tail_norm(self) -> None:
        n, c = 9, 0.73
        for start, stop in ((0, 2), (3, 5), (6, 8)):
            for anchor in range(start, stop + 1):
                with self.subTest(side="right", block=(start, stop), anchor=anchor):
                    x, gamma, w = split_right_suffix(
                        n, anchor, c, start, stop
                    )
                    self.assertGreaterEqual(gamma, 0.0)
                    self.assertLessEqual(gamma, 1.0)
                    self.assertLessEqual(float(np.vdot(w, w).real), 1.0 + 1e-14)
                    np.testing.assert_allclose(
                        x + gamma * w, right_suffix(n, anchor, c)
                    )
                with self.subTest(side="left", block=(start, stop), anchor=anchor):
                    x, gamma, w = split_left_suffix(n, anchor, c, start, stop)
                    self.assertGreaterEqual(gamma, 0.0)
                    self.assertLessEqual(gamma, 1.0)
                    self.assertLessEqual(float(np.vdot(w, w).real), 1.0 + 1e-14)
                    np.testing.assert_allclose(
                        x + gamma * w, left_suffix(n, anchor, c)
                    )

    def test_c0_and_c1_suffixes_are_direct_physical_branches(self) -> None:
        np.testing.assert_array_equal(
            right_suffix(4, 1, 0.0), np.array([0.0, 1.0, 0.0, 0.0])
        )
        np.testing.assert_array_equal(
            left_suffix(4, 2, 0.0), np.array([0.0, 0.0, 1.0, 0.0])
        )
        np.testing.assert_array_equal(right_suffix(4, 1, 1.0), np.zeros(4))
        np.testing.assert_array_equal(left_suffix(4, 2, 1.0), np.zeros(4))


class WeightedLocalCertificateTests(unittest.TestCase):
    def test_DK0D_direction_entries_trace_and_m4_fixture(self) -> None:
        m, c = 4, 0.5
        k0 = _frame_sum_suffix_certificate(m)
        d = np.diag([1.0, 0.5, 0.25, 0.125])
        expected_congruence = d @ k0 @ d
        expected = 0.75 * 0.5 ** (-6) * expected_congruence
        actual = weighted_local_certificate(m, c, k0)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(float(np.trace(expected_congruence)), 1.75, places=14)
        self.assertLessEqual(float(np.trace(expected_congruence)), float(np.trace(k0)))
        self.assertAlmostEqual(float(np.trace(actual)), 84.0, places=12)

    def test_weighted_local_certificate_dominates_all_local_suffixes(self) -> None:
        m, c = 5, 0.71
        s = sqrt(1.0 - c * c)
        certificate = weighted_local_certificate(
            m, c, _frame_sum_suffix_certificate(m)
        )
        for anchor in range(m):
            vector = np.array(
                [s * c ** (j - anchor) if j >= anchor else 0.0 for j in range(m)]
            )
            projector = _outer(vector)
            _assert_psd_relative(
                self, certificate - projector, certificate, projector
            )

    def test_local_endpoint_branches_are_identity_and_zero(self) -> None:
        np.testing.assert_array_equal(
            weighted_local_certificate(3, 0.0), np.eye(3)
        )
        np.testing.assert_array_equal(
            weighted_local_certificate(3, 1.0), np.zeros((3, 3))
        )

    def test_invalid_overlap_or_unweighted_base_is_rejected(self) -> None:
        for c in (-0.1, 1.1):
            with self.subTest(c=c):
                with self.assertRaises(ValueError):
                    weighted_local_certificate(3, c)
        with self.assertRaises(ValueError):
            weighted_local_certificate(3, 0.6, np.eye(3))


class TensorAndGlobalCertificateTests(unittest.TestCase):
    def test_tensor_order_identity_and_strict_pair_order(self) -> None:
        x = np.array([1.0, 0.0, 0.5])
        y = np.array([0.0, 0.4, 0.0])
        a = _outer(x) + np.diag([0.0, 0.3, 0.0])
        b = _outer(y) + np.diag([0.2, 0.0, 0.1])
        tensor_slack = np.kron(a, b) - np.kron(_outer(x), _outer(y))
        identity_rhs = np.kron(a - _outer(x), b) + np.kron(
            _outer(x), b - _outer(y)
        )
        np.testing.assert_allclose(tensor_slack, identity_rhs, atol=1e-15)
        _assert_psd_relative(
            self,
            tensor_slack,
            np.kron(a, b),
            np.kron(_outer(x), _outer(y)),
        )

        strict_indices = [u * 3 + v for u, v in _strict_pairs(3)]
        expected = np.kron(a, b)[np.ix_(strict_indices, strict_indices)]
        np.testing.assert_allclose(tensor_cell_certificate(a, b), expected)

    def test_physical_rows_keep_singletons_and_projectors_have_s4(self) -> None:
        n, c = 2, 0.6
        rows = physical_hull_rows(n, c)
        self.assertEqual(rows.shape, (3, 1))
        np.testing.assert_array_equal(rows[[0, 2]], np.zeros((2, 1)))
        self.assertAlmostEqual(rows[1, 0], 1.0 - c * c, places=15)
        self.assertAlmostEqual(_outer(rows[1])[0, 0], (1.0 - c * c) ** 2, places=15)

    def test_global_n4_n5_certificate_assigns_every_interval_to_one_cell(self) -> None:
        for n, c, block_count in ((4, 0.55, 2), (5, 0.72, 2)):
            with self.subTest(n=n):
                certificate, cells = global_hull_certificate(
                    n, c, block_count, return_cells=True
                )
                rows = physical_hull_rows(n, c)
                self.assertEqual(rows.shape[0], n * (n + 1) // 2)
                self.assertEqual(rows.shape[1], n * (n - 1) // 2)
                self.assertEqual(set(cells), {(0, 0), (0, 1), (1, 1)})
                np.testing.assert_allclose(
                    certificate, sum(cells.values(), np.zeros_like(certificate))
                )

                minimum_size, remainder = divmod(n, block_count)
                sizes = tuple(
                    minimum_size + (1 if p < remainder else 0)
                    for p in range(block_count)
                )
                blocks: list[tuple[int, int]] = []
                cursor = 0
                for local_size in sizes:
                    blocks.append((cursor, cursor + local_size - 1))
                    cursor += local_size
                interval_to_row = {
                    interval: index for index, interval in enumerate(_intervals(n))
                }
                assigned_count = 0
                for (p, q), cell in cells.items():
                    self.assertGreater(float(np.trace(cell).real), 0.0)
                    left_start, left_stop = blocks[p]
                    right_start, right_stop = blocks[q]
                    assigned = [
                        (a, b)
                        for a in range(left_start, left_stop + 1)
                        for b in range(right_start, right_stop + 1)
                        if a <= b
                    ]
                    self.assertTrue(assigned)
                    assigned_count += len(assigned)
                    for interval in assigned:
                        row = rows[interval_to_row[interval]]
                        projector = _outer(row)
                        _assert_psd_relative(
                            self, cell - projector, cell, projector
                        )
                self.assertEqual(assigned_count, n * (n + 1) // 2)
                for row in rows:
                    projector = _outer(row)
                    _assert_psd_relative(
                        self, certificate - projector, certificate, projector
                    )

    def test_global_endpoint_certificates_use_direct_physical_branches(self) -> None:
        n, block_count = 5, 2
        identity, cells_zero = global_hull_certificate(
            n, 0.0, block_count, return_cells=True
        )
        np.testing.assert_array_equal(identity, np.eye(n * (n - 1) // 2))
        np.testing.assert_array_equal(
            identity, sum(cells_zero.values(), np.zeros_like(identity))
        )
        rows_zero = physical_hull_rows(n, 0.0)
        for row in rows_zero:
            _assert_psd_relative(self, identity - _outer(row), identity, _outer(row))

        zero, cells_one = global_hull_certificate(
            n, 1.0, block_count, return_cells=True
        )
        np.testing.assert_array_equal(zero, np.zeros_like(zero))
        np.testing.assert_array_equal(
            zero, sum(cells_one.values(), np.zeros_like(zero))
        )
        np.testing.assert_array_equal(physical_hull_rows(n, 1.0), np.zeros((15, 10)))

    def test_minimum_slack_reports_an_infeasible_rank_one_family(self) -> None:
        certificate = np.zeros((2, 2))
        rows = np.array([[1.0, 0.0], [0.0, 0.5]])
        self.assertAlmostEqual(
            minimum_certificate_slack(certificate, rows), -1.0, places=15
        )


def _zero_extend(
    q: np.ndarray, w: np.ndarray, third_parameter: float
) -> np.ndarray:
    local = np.asarray(q)
    tail = np.asarray(w)
    size = local.shape[0] if local.ndim == 2 else tail.size
    return np.zeros((size, size), dtype=np.result_type(local, tail, float))


def _zero_optimized(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _zero_extend(q, w, 1.0)


def run_behavioral_zero_stub_red() -> unittest.result.TestResult:
    """Replay the 32-test Task 4 behavior RED with shape-safe zero stubs."""
    replacements = {
        "right_suffix": _zero_right_suffix,
        "left_suffix": _zero_left_suffix,
        "split_right_suffix": _zero_split,
        "split_left_suffix": _zero_split,
        "extend_certificate": _zero_extend,
        "extend_with_budget": _zero_extend,
        "optimized_extension": _zero_optimized,
        "weighted_local_certificate": _zero_weighted_local,
        "tensor_cell_certificate": _zero_tensor,
        "physical_hull_rows": _zero_physical,
        "global_hull_certificate": _zero_global,
        "minimum_certificate_slack": _zero_slack,
    }
    original = {name: globals()[name] for name in replacements}
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in (
        WeightedBlockTailBudgetTests,
        ExactSuffixSplitTests,
        WeightedLocalCertificateTests,
        TensorAndGlobalCertificateTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    try:
        globals().update(replacements)
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals().update(original)


def run_validation_branch_mutants_red() -> dict[str, object]:
    """Delete one copied production guard while preserving every other line."""

    def copied_with_one_guard_deleted(omitted: str):
        def mutant(q: np.ndarray, w: np.ndarray, budget: float) -> np.ndarray:
            if isinstance(budget, bool) or not isinstance(budget, Real):
                raise ValueError("budget must be a finite real number")
            limit = float(budget)
            if omitted != "nonfinite_budget" and not isfinite(limit):
                raise ValueError(f"budget must be finite; measured budget={limit}")
            if omitted != "negative_budget" and limit < 0.0:
                raise ValueError(
                    f"budget must be nonnegative; measured budget={limit}"
                )

            local = np.asarray(q)
            bad_q_shape = (
                local.ndim != 2
                or local.shape[0] != local.shape[1]
                or not np.issubdtype(local.dtype, np.number)
            )
            if omitted != "q_shape" and bad_q_shape:
                raise ValueError(
                    "q must be a two-dimensional numeric square matrix; "
                    f"measured shape={local.shape}"
                )
            q_nonfinite = int(
                local.size - np.count_nonzero(np.isfinite(local))
            )
            if omitted != "q_finite" and q_nonfinite:
                raise ValueError(
                    f"q has nonfinite entries; measured count={q_nonfinite}"
                )

            dimension = local.shape[0]
            operator_scale = float(np.linalg.norm(local, 2)) if dimension else 0.0
            hermitian_violation = float(
                np.linalg.norm(local - local.conjugate().T, 2)
            )
            hermitian_tolerance = block_tail_probe._scaled_tolerance(
                operator_scale, dimension
            )
            if (
                omitted != "hermitian"
                and hermitian_violation > hermitian_tolerance
            ):
                raise ValueError(
                    "q Hermitian violation="
                    f"{hermitian_violation:.17g} exceeds tolerance="
                    f"{hermitian_tolerance:.17g}"
                )
            hermitian = (local + local.conjugate().T) / 2.0
            minimum = (
                float(np.linalg.eigvalsh(hermitian)[0]) if dimension else 0.0
            )
            psd_violation = max(0.0, -minimum)
            psd_tolerance = block_tail_probe._scaled_tolerance(
                operator_scale, dimension
            )
            if omitted != "psd" and psd_violation > psd_tolerance:
                raise ValueError(
                    f"q PSD violation={psd_violation:.17g} exceeds tolerance="
                    f"{psd_tolerance:.17g}"
                )
            if minimum < 0.0:
                eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
                clipped = np.maximum(eigenvalues, 0.0)
                hermitian = (
                    eigenvectors * clipped
                ) @ eigenvectors.conjugate().T
                hermitian = (hermitian + hermitian.conjugate().T) / 2.0
            local_trace = fsum(
                float(value.real) for value in np.diag(hermitian)
            )

            tail = np.asarray(w)
            bad_w_rank = tail.ndim != 1 or not np.issubdtype(
                tail.dtype, np.number
            )
            if omitted != "w_shape" and bad_w_rank:
                raise ValueError(
                    "w must be a one-dimensional numeric vector; "
                    f"measured shape={tail.shape}"
                )
            if tail.shape[0] != dimension:
                raise ValueError(
                    f"w dimension mismatch; measured size={tail.shape[0]}, "
                    f"expected size={dimension}"
                )
            w_nonfinite = int(
                tail.size - np.count_nonzero(np.isfinite(tail))
            )
            if omitted != "w_finite" and w_nonfinite:
                raise ValueError(
                    f"w has nonfinite entries; measured count={w_nonfinite}"
                )
            tail_norm = float(np.linalg.norm(tail, 2))
            tail_violation = max(0.0, tail_norm - 1.0)
            tail_tolerance = block_tail_probe._scaled_tolerance(
                max(1.0, tail_norm), dimension
            )
            if omitted != "tail_norm" and tail_violation > tail_tolerance:
                raise ValueError(
                    f"tail-norm violation={tail_violation:.17g}"
                )
            if limit == 0.0:
                measured_norm = float(np.linalg.norm(local, 2))
                if omitted != "zero_budget" and measured_norm != 0.0:
                    raise ValueError(f"zero-budget Q norm={measured_norm:.17g}")
                return np.outer(tail, tail.conjugate())
            trace_violation = max(0.0, local_trace - limit)
            if omitted != "trace_budget" and trace_violation > 0.0:
                raise ValueError(
                    f"trace-budget violation={trace_violation:.17g}"
                )
            alpha = limit ** -0.5
            return (
                (1.0 + alpha) * hermitian
                + (1.0 + 1.0 / alpha) * np.outer(tail, tail.conjugate())
            )

        return mutant

    cases = (
        ("negative_budget", np.zeros((2, 2)), np.zeros(2), -0.25,
         r"budget must be nonnegative; measured budget=-0\.25"),
        ("nonfinite_budget", np.zeros((2, 2)), np.zeros(2), float("nan"),
         r"budget must be finite"),
        ("q_shape", np.zeros(2), np.zeros(2), 1.0, r"square matrix"),
        ("w_shape", np.zeros((2, 2)), np.zeros((2, 1)), 1.0,
         r"one-dimensional|dimension"),
        ("q_finite", np.array([[0.0, np.nan], [0.0, 0.0]]), np.zeros(2),
         1.0, r"q.*nonfinite.*1"),
        ("w_finite", np.zeros((2, 2)), np.array([0.0, np.inf]), 1.0,
         r"w.*nonfinite.*1"),
        ("hermitian", np.array([[1.0, 0.25], [0.0, 1.0]]), np.zeros(2),
         2.0, r"Hermitian violation=.*0\.25"),
        ("psd", np.diag([1.0, -0.125]), np.zeros(2), 2.0,
         r"PSD violation=.*0\.125"),
        ("trace_budget", np.diag([0.75, 0.5]), np.zeros(2), 1.0,
         r"trace-budget violation=.*0\.25"),
        ("tail_norm", np.zeros((2, 2)), np.array([0.0, 1.25]), 1.0,
         r"tail-norm violation=.*0\.25"),
        ("zero_budget", np.diag([np.finfo(float).eps, 0.0]), np.zeros(2),
         0.0, r"zero-budget Q norm=.*e-16"),
    )

    total_tests = total_failures = total_errors = 0
    traceback_types: set[str] = set()
    mutant_names: list[str] = []
    original = globals()["extend_with_budget"]
    for name, q, w, budget, expected_regex in cases:
        mutant = copied_with_one_guard_deleted(name)
        globals()["extend_with_budget"] = mutant

        class FocusedGuardContract(unittest.TestCase):
            def runTest(self) -> None:
                try:
                    with self.assertRaisesRegex(ValueError, expected_regex):
                        extend_with_budget(q, w, budget)
                except AssertionError:
                    raise
                except Exception as error:
                    raise AssertionError(
                        f"deleted {name} guard reached unchanged downstream "
                        f"code and raised {type(error).__name__}: {error}"
                    ) from error

        try:
            suite = unittest.TestSuite([FocusedGuardContract()])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        finally:
            globals()["extend_with_budget"] = original
        failure_types = {
            "AssertionError" if "AssertionError" in traceback else "non-AssertionError"
            for _, traceback in result.failures
        }
        if (
            result.testsRun != 1
            or len(result.failures) < 1
            or result.errors
            or failure_types != {"AssertionError"}
        ):
            raise AssertionError(
                f"validation mutant survived or errored: {name}; "
                f"tests={result.testsRun}, failures={len(result.failures)}, "
                f"errors={len(result.errors)}, types={sorted(failure_types)}"
            )
        mutant_names.append(name)
        traceback_types.update(failure_types)
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)

    loader = unittest.TestLoader()
    green_suite = unittest.TestSuite()
    for test_case in (
        WeightedBlockTailBudgetTests,
        ExactSuffixSplitTests,
        WeightedLocalCertificateTests,
        TensorAndGlobalCertificateTests,
    ):
        green_suite.addTests(loader.loadTestsFromTestCase(test_case))
    green = unittest.TextTestRunner(stream=io.StringIO()).run(green_suite)
    if green.testsRun != 32 or green.failures or green.errors:
        raise AssertionError(
            "post-mutation reusable suite did not restore to 32/32 GREEN"
        )
    return {
        "mutants": len(mutant_names),
        "mutant_names": tuple(mutant_names),
        "tests": total_tests,
        "failures": total_failures,
        "errors": total_errors,
        "traceback_types": tuple(sorted(traceback_types)),
        "post_green_tests": green.testsRun,
        "post_green_failures": len(green.failures),
        "post_green_errors": len(green.errors),
    }


def run_tiny_trace_discard_mutant_red() -> unittest.result.TestResult:
    """Show that a positive trace may not be rounded into the zero branch."""
    original = globals()["optimized_extension"]

    def discard_tiny_positive(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        if 0.0 < float(np.trace(q).real) < 1.0e-12:
            return _outer(np.asarray(w))
        return original(q, w)

    suite = unittest.TestSuite(
        [
            WeightedBlockTailBudgetTests(
                "test_small_positive_cost_is_never_discarded"
            )
        ]
    )
    try:
        globals()["optimized_extension"] = discard_tiny_positive
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals()["optimized_extension"] = original


def run_shared_extension_coefficient_mutant_red() -> unittest.result.TestResult:
    """Corrupt the shared Q coefficient in both extension call paths."""
    original_raw = block_tail_probe._raw_extension_formula

    def wrong_shared_coefficient(
        q: np.ndarray, w: np.ndarray, alpha: float
    ) -> np.ndarray:
        local = np.asarray(q)
        tail = np.asarray(w)
        return (
            (2.0 + alpha) * local
            + (1.0 + 1.0 / alpha) * np.outer(tail, tail.conjugate())
        )

    suite = unittest.TestSuite(
        [
            WeightedBlockTailBudgetTests(
                "test_extension_coefficients_match_independent_literal"
            ),
            WeightedBlockTailBudgetTests(
                "test_small_positive_cost_is_never_discarded"
            ),
        ]
    )
    try:
        block_tail_probe._raw_extension_formula = wrong_shared_coefficient
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        block_tail_probe._raw_extension_formula = original_raw


def run_unconjugated_zero_outer_mutant_red() -> unittest.result.TestResult:
    """Replace the complex zero-branch projector by outer(w,w) without conj."""
    original_global = globals()["extend_with_budget"]
    original_module = block_tail_probe.extend_with_budget

    def wrong_unconjugated_outer(
        q: np.ndarray, w: np.ndarray, budget: float
    ) -> np.ndarray:
        validated = original_module(q, w, budget)
        if float(budget) == 0.0:
            tail = np.asarray(w)
            return np.outer(tail, tail)
        return validated

    suite = unittest.TestSuite(
        [
            WeightedBlockTailBudgetTests(
                "test_exact_zero_budget_uses_tail_projector"
            )
        ]
    )
    try:
        globals()["extend_with_budget"] = wrong_unconjugated_outer
        block_tail_probe.extend_with_budget = wrong_unconjugated_outer
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals()["extend_with_budget"] = original_global
        block_tail_probe.extend_with_budget = original_module


def run_missing_last_diagonal_cell_mutant_red() -> unittest.result.TestResult:
    """Zero the final diagonal cell while preserving its dictionary key."""
    original = globals()["global_hull_certificate"]

    def omit_last_diagonal(
        n: int, c: float, block_count: int, *, return_cells: bool = False
    ):
        certificate, cells = original(n, c, block_count, return_cells=True)
        key = (block_count - 1, block_count - 1)
        removed = cells[key]
        mutated_cells = dict(cells)
        mutated_cells[key] = np.zeros_like(removed)
        mutated_certificate = certificate - removed
        return (
            (mutated_certificate, mutated_cells)
            if return_cells
            else mutated_certificate
        )

    suite = unittest.TestSuite(
        [
            TensorAndGlobalCertificateTests(
                "test_global_n4_n5_certificate_assigns_every_interval_to_one_cell"
            )
        ]
    )
    try:
        globals()["global_hull_certificate"] = omit_last_diagonal
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals()["global_hull_certificate"] = original


def run_skipped_unweighted_dominance_mutant_red() -> unittest.result.TestResult:
    """Keep the weighted formula but omit only the K0 suffix-dominance audit."""
    original = globals()["weighted_local_certificate"]

    def skip_dominance(
        m: int, c: float, k0: np.ndarray | None = None
    ) -> np.ndarray:
        size = int(m)
        overlap = float(c)
        base = np.asarray(k0)
        if size < 1 or not 0.0 < overlap < 1.0:
            raise ValueError("invalid local-certificate parameters")
        if (
            base.shape != (size, size)
            or not np.issubdtype(base.dtype, np.number)
            or not np.isfinite(base).all()
            or not np.allclose(base, base.conjugate().T, atol=1.0e-13)
        ):
            raise ValueError("invalid Hermitian k0")
        if float(np.linalg.eigvalsh(base)[0]) < -1.0e-13:
            raise ValueError("k0 must be positive semidefinite")
        diagonal = np.power(overlap, np.arange(size))
        congruence = diagonal[:, None] * base * diagonal[None, :]
        factor = (1.0 - overlap * overlap) * overlap ** (-2 * (size - 1))
        return factor * congruence

    suite = unittest.TestSuite(
        [
            WeightedLocalCertificateTests(
                "test_invalid_overlap_or_unweighted_base_is_rejected"
            )
        ]
    )
    try:
        globals()["weighted_local_certificate"] = skip_dominance
        return unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        globals()["weighted_local_certificate"] = original


def run_deterministic_certificate_stress() -> dict[str, float | int]:
    """Replay all finite global and suffix stress cases deterministically."""
    global_cases = 0
    suffix_cases = 0
    minimum_raw_slack = float("inf")
    for n in range(2, 9):
        for c in (0.05, 0.2, 0.5, 0.8, 0.97):
            for block_count in range(1, min(n, 4) + 1):
                certificate = global_hull_certificate(n, c, block_count)
                rows = physical_hull_rows(n, c)
                slack = minimum_certificate_slack(certificate, rows)
                minimum_raw_slack = min(minimum_raw_slack, slack)
                scale = max(
                    1.0,
                    float(np.linalg.norm(certificate, 2)),
                    max(float(np.vdot(row, row).real) for row in rows),
                )
                if slack < -1.0e-12 * scale:
                    raise AssertionError((n, c, block_count, slack, scale))
                global_cases += 1
    for n in range(2, 15):
        for c in (0.0, 0.13, 0.71, 1.0):
            blocks = np.array_split(np.arange(n), min(n, 4))
            for block in blocks:
                start, stop = int(block[0]), int(block[-1])
                for anchor in range(start, stop + 1):
                    x, gamma, w = split_right_suffix(n, anchor, c, start, stop)
                    np.testing.assert_allclose(
                        x + gamma * w, right_suffix(n, anchor, c)
                    )
                    suffix_cases += 1
                    x, gamma, w = split_left_suffix(n, anchor, c, start, stop)
                    np.testing.assert_allclose(
                        x + gamma * w, left_suffix(n, anchor, c)
                    )
                    suffix_cases += 1
    return {
        "global_cases": global_cases,
        "suffix_cases": suffix_cases,
        "minimum_raw_slack": minimum_raw_slack,
    }


def run_deterministic_budget_stress() -> dict[str, float | int]:
    """Audit singular complex instances of the abstract budgeted theorem."""
    generator = np.random.default_rng(20260829)
    cases = 0
    minimum_relative_slack = float("inf")
    maximum_cost_excess = float("-inf")
    for dimension in range(1, 7):
        for _ in range(200):
            x = generator.normal(size=dimension) + 1j * generator.normal(
                size=dimension
            )
            x /= max(1.0, float(np.linalg.norm(x)))
            # Q=xx* is deliberately singular whenever dimension > 1.
            q = _outer(x)
            trace = float(np.trace(q).real)
            budget = trace * (1.0 + float(generator.random()))
            w = generator.normal(size=dimension) + 1j * generator.normal(
                size=dimension
            )
            w_norm = float(np.linalg.norm(w))
            if w_norm:
                w *= float(generator.random()) / w_norm
            gamma = float(generator.random())
            certificate = extend_with_budget(q, w, budget)
            vector = x + gamma * w
            projector = _outer(vector)
            scale = max(
                1.0,
                float(np.linalg.norm(certificate, 2)),
                float(np.linalg.norm(projector, 2)),
            )
            relative_slack = _minimum_eigenvalue(
                certificate - projector
            ) / scale
            minimum_relative_slack = min(minimum_relative_slack, relative_slack)
            cost_excess = float(np.trace(certificate).real) - (
                budget**0.5 + 1.0
            ) ** 2
            maximum_cost_excess = max(maximum_cost_excess, cost_excess)
            if relative_slack < -1.0e-12 or cost_excess > 1.0e-12:
                raise AssertionError(
                    (dimension, relative_slack, cost_excess, budget)
                )
            cases += 1
    return {
        "cases": cases,
        "minimum_relative_slack": minimum_relative_slack,
        "maximum_cost_excess": maximum_cost_excess,
    }


def run_deterministic_rank_deficient_scan() -> dict[str, float | int]:
    """Check real and complex AA* inputs without altering rank deficiency."""
    generator = np.random.default_rng(20260830)
    cases = 0
    false_rejections = 0
    minimum_relative_slack = float("inf")
    maximum_trace_excess = float("-inf")
    for complex_case in (False, True):
        for dimension in range(2, 6):
            for rank in range(1, dimension):
                for _ in range(2):
                    a = generator.normal(size=(dimension, rank))
                    if complex_case:
                        a = a + 1j * generator.normal(
                            size=(dimension, rank)
                        )
                    a /= max(1.0, float(np.linalg.norm(a, 2)))
                    q = a @ a.conjugate().T
                    z = generator.normal(size=rank)
                    if complex_case:
                        z = z + 1j * generator.normal(size=rank)
                    z /= max(1.0, float(np.linalg.norm(z)))
                    x = a @ z
                    w = generator.normal(size=dimension)
                    if complex_case:
                        w = w + 1j * generator.normal(size=dimension)
                    w /= max(1.0, float(np.linalg.norm(w)))
                    try:
                        certificate = optimized_extension(q, w)
                    except ValueError:
                        false_rejections += 1
                        raise
                    budget = float(np.trace(q).real)
                    trace_excess = float(np.trace(certificate).real) - (
                        sqrt(budget) + 1.0
                    ) ** 2
                    maximum_trace_excess = max(
                        maximum_trace_excess, trace_excess
                    )
                    for gamma in (0.0, 0.37, 1.0):
                        vector = x + gamma * w
                        projector = _outer(vector)
                        scale = max(
                            1.0,
                            float(np.linalg.norm(certificate, 2)),
                            float(np.linalg.norm(projector, 2)),
                        )
                        relative_slack = _minimum_eigenvalue(
                            certificate - projector
                        ) / scale
                        minimum_relative_slack = min(
                            minimum_relative_slack, relative_slack
                        )
                        if relative_slack < -5.0e-14:
                            raise AssertionError(
                                (complex_case, dimension, rank, relative_slack)
                            )
                    if trace_excess > 5.0e-13:
                        raise AssertionError(
                            (complex_case, dimension, rank, trace_excess)
                        )
                    cases += 1
    return {
        "cases": cases,
        "false_rejections": false_rejections,
        "minimum_relative_slack": minimum_relative_slack,
        "maximum_trace_excess": maximum_trace_excess,
    }


if __name__ == "__main__":
    unittest.main()
