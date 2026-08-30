"""Exact behavior tests for the weighted hull coordinate identity."""

from fractions import Fraction
from math import pi
import unittest

from proofs.weighted_hull_finite_audit import (
    complex_phase_congruence_check,
    direct_ge_two_gram_fraction,
    direct_hull_coordinate_matrix_fraction,
    hull_pairs,
    interval_labels,
    old_qdzw_hull_matrix_fraction,
    tensor_compressed_hull_matrix_fraction,
    weighted_hull_gram_fraction,
    weighted_hull_matrix_fraction,
)


class WeightedHullSentinelTests(unittest.TestCase):
    def test_n2_boundary_hull_has_hand_calculated_amplitude(self) -> None:
        matrix = weighted_hull_matrix_fraction(2, Fraction(3, 5))
        row = interval_labels(2).index((0, 1))
        column = hull_pairs(2).index((0, 1))

        self.assertEqual(matrix[row][column], Fraction(16, 25))

    def test_n4_interior_hull_has_hand_calculated_amplitude(self) -> None:
        matrix = weighted_hull_matrix_fraction(4, Fraction(1, 2))
        row = interval_labels(4).index((0, 3))
        column = hull_pairs(4).index((1, 2))

        self.assertEqual(matrix[row][column], Fraction(3, 16))


class WeightedHullCoordinateConstructionTests(unittest.TestCase):
    def test_four_constructions_match_hand_calculated_coordinate_tables(self) -> None:
        z = Fraction(0)
        cases = (
            (
                2,
                Fraction(3, 5),
                Fraction(4, 5),
                [[z], [Fraction(16, 25)], [z]],
            ),
            (
                3,
                Fraction(3, 5),
                Fraction(4, 5),
                [
                    [z, z, z],
                    [Fraction(16, 25), z, z],
                    [Fraction(48, 125), Fraction(16, 25), Fraction(48, 125)],
                    [z, z, z],
                    [z, z, Fraction(16, 25)],
                    [z, z, z],
                ],
            ),
            (
                4,
                Fraction(3, 5),
                Fraction(4, 5),
                [
                    [z, z, z, z, z, z],
                    [Fraction(16, 25), z, z, z, z, z],
                    [Fraction(48, 125), Fraction(16, 25), z, Fraction(48, 125), z, z],
                    [
                        Fraction(144, 625),
                        Fraction(48, 125),
                        Fraction(16, 25),
                        Fraction(144, 625),
                        Fraction(48, 125),
                        Fraction(144, 625),
                    ],
                    [z, z, z, z, z, z],
                    [z, z, z, Fraction(16, 25), z, z],
                    [z, z, z, Fraction(48, 125), Fraction(16, 25), Fraction(48, 125)],
                    [z, z, z, z, z, z],
                    [z, z, z, z, z, Fraction(16, 25)],
                    [z, z, z, z, z, z],
                ],
            ),
            (
                2,
                Fraction(5, 13),
                Fraction(12, 13),
                [[z], [Fraction(144, 169)], [z]],
            ),
            (
                3,
                Fraction(5, 13),
                Fraction(12, 13),
                [
                    [z, z, z],
                    [Fraction(144, 169), z, z],
                    [Fraction(720, 2197), Fraction(144, 169), Fraction(720, 2197)],
                    [z, z, z],
                    [z, z, Fraction(144, 169)],
                    [z, z, z],
                ],
            ),
            (
                4,
                Fraction(5, 13),
                Fraction(12, 13),
                [
                    [z, z, z, z, z, z],
                    [Fraction(144, 169), z, z, z, z, z],
                    [Fraction(720, 2197), Fraction(144, 169), z, Fraction(720, 2197), z, z],
                    [
                        Fraction(3600, 28561),
                        Fraction(720, 2197),
                        Fraction(144, 169),
                        Fraction(3600, 28561),
                        Fraction(720, 2197),
                        Fraction(3600, 28561),
                    ],
                    [z, z, z, z, z, z],
                    [z, z, z, Fraction(144, 169), z, z],
                    [z, z, z, Fraction(720, 2197), Fraction(144, 169), Fraction(720, 2197)],
                    [z, z, z, z, z, z],
                    [z, z, z, z, z, Fraction(144, 169)],
                    [z, z, z, z, z, z],
                ],
            ),
        )

        for n, c, s, expected in cases:
            constructions = (
                ("weighted", weighted_hull_matrix_fraction(n, c)),
                ("direct", direct_hull_coordinate_matrix_fraction(n, c, s)),
                ("tensor", tensor_compressed_hull_matrix_fraction(n, c)),
                ("old_qdzw", old_qdzw_hull_matrix_fraction(n, c, s)),
            )
            for name, actual in constructions:
                with self.subTest(n=n, c=c, construction=name):
                    self.assertEqual(actual, expected)


class WeightedHullIndexAndValidationTests(unittest.TestCase):
    def test_zero_based_labels_have_required_dimensions_and_strict_hulls(self) -> None:
        self.assertEqual(
            interval_labels(4),
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3),
            ],
        )
        self.assertEqual(
            hull_pairs(4),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )
        self.assertEqual(len(interval_labels(4)), 10)
        self.assertEqual(len(hull_pairs(4)), 6)
        self.assertTrue(all(u < v for u, v in hull_pairs(4)))

    def test_every_construction_keeps_singleton_rows_present_and_zero(self) -> None:
        c = Fraction(3, 5)
        s = Fraction(4, 5)
        expected_zero_row = [Fraction(0)] * 6
        constructions = (
            weighted_hull_matrix_fraction(4, c),
            direct_hull_coordinate_matrix_fraction(4, c, s),
            tensor_compressed_hull_matrix_fraction(4, c),
            old_qdzw_hull_matrix_fraction(4, c, s),
        )

        for matrix in constructions:
            with self.subTest(construction_rows=len(matrix)):
                self.assertEqual(len(matrix), 10)
                self.assertTrue(all(len(row) == 6 for row in matrix))
                for singleton_row in (0, 4, 7, 9):
                    self.assertEqual(matrix[singleton_row], expected_zero_row)

    def test_weighted_coordinate_rejects_overlap_outside_closed_unit_interval(self) -> None:
        for invalid_c in (Fraction(-1, 5), Fraction(6, 5)):
            with self.subTest(c=invalid_c):
                with self.assertRaises(ValueError):
                    weighted_hull_matrix_fraction(3, invalid_c)

    def test_nonpositive_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            interval_labels(0)
        with self.assertRaises(ValueError):
            hull_pairs(0)


class WeightedHullGramTests(unittest.TestCase):
    def test_both_gram_constructions_match_hand_calculated_full_matrices(self) -> None:
        z = Fraction(0)
        cases = (
            (
                2,
                Fraction(3, 5),
                [
                    [z, z, z],
                    [z, Fraction(256, 625), z],
                    [z, z, z],
                ],
            ),
            (
                3,
                Fraction(3, 5),
                [
                    [z, z, z, z, z, z],
                    [z, Fraction(256, 625), Fraction(768, 3125), z, z, z],
                    [z, Fraction(768, 3125), Fraction(11008, 15625), z, Fraction(768, 3125), z],
                    [z, z, z, z, z, z],
                    [z, z, Fraction(768, 3125), z, Fraction(256, 625), z],
                    [z, z, z, z, z, z],
                ],
            ),
            (
                2,
                Fraction(5, 13),
                [
                    [z, z, z],
                    [z, Fraction(20736, 28561), z],
                    [z, z, z],
                ],
            ),
            (
                3,
                Fraction(5, 13),
                [
                    [z, z, z, z, z, z],
                    [z, Fraction(20736, 28561), Fraction(103680, 371293), z, z, z],
                    [z, Fraction(103680, 371293), Fraction(4541184, 4826809), z, Fraction(103680, 371293), z],
                    [z, z, z, z, z, z],
                    [z, z, Fraction(103680, 371293), z, Fraction(20736, 28561), z],
                    [z, z, z, z, z, z],
                ],
            ),
        )

        for n, c, expected in cases:
            for name, actual in (
                ("coordinate_product", weighted_hull_gram_fraction(n, c)),
                ("direct_sector_sum", direct_ge_two_gram_fraction(n, c)),
            ):
                with self.subTest(n=n, c=c, construction=name):
                    self.assertEqual(actual, expected)

    def test_n4_gram_entries_match_independent_hand_calculations(self) -> None:
        for name, gram in (
            ("coordinate_product", weighted_hull_gram_fraction(4, Fraction(1, 2))),
            ("direct_sector_sum", direct_ge_two_gram_fraction(4, Fraction(1, 2))),
        ):
            with self.subTest(construction=name, entry="full_interval_norm"):
                self.assertEqual(gram[3][3], Fraction(243, 256))
            with self.subTest(construction=name, entry="nested_cross_term"):
                self.assertEqual(gram[3][5], Fraction(9, 64))
            with self.subTest(construction=name, entry="disjoint_cross_term"):
                self.assertEqual(gram[1][8], Fraction(0))


class WeightedHullComplexPhaseTests(unittest.TestCase):
    def test_explicit_n3_tensor_states_obey_diagonal_unitary_congruence(self) -> None:
        self.assertTrue(
            complex_phase_congruence_check(3, Fraction(3, 5), pi / 7)
        )


class WeightedHullEndpointTests(unittest.TestCase):
    def test_c1_hull_is_exactly_zero_without_defining_an_excitation_basis(self) -> None:
        expected = [[Fraction(0)] * 6 for _ in range(10)]

        self.assertEqual(
            weighted_hull_matrix_fraction(4, Fraction(1)), expected
        )
        self.assertEqual(
            direct_hull_coordinate_matrix_fraction(
                4, Fraction(1), Fraction(0)
            ),
            expected,
        )
        self.assertEqual(
            tensor_compressed_hull_matrix_fraction(4, Fraction(1)),
            expected,
        )

    def test_c0_direct_computational_basis_matches_hand_calculated_table(self) -> None:
        z = Fraction(0)
        o = Fraction(1)
        expected = [
            [z, z, z, z, z, z],
            [o, z, z, z, z, z],
            [z, o, z, z, z, z],
            [z, z, o, z, z, z],
            [z, z, z, z, z, z],
            [z, z, z, o, z, z],
            [z, z, z, z, o, z],
            [z, z, z, z, z, z],
            [z, z, z, z, z, o],
            [z, z, z, z, z, z],
        ]

        # The old q^2DZW and complex-phase helpers are intentionally absent:
        # both rely on divisions or gauge machinery excluded from this audit.
        self.assertEqual(
            weighted_hull_matrix_fraction(4, Fraction(0)), expected
        )
        self.assertEqual(
            direct_hull_coordinate_matrix_fraction(
                4, Fraction(0), Fraction(1)
            ),
            expected,
        )
        self.assertEqual(
            tensor_compressed_hull_matrix_fraction(4, Fraction(0)),
            expected,
        )

    def test_old_qdzw_representation_rejects_c0(self) -> None:
        with self.assertRaises(ValueError):
            old_qdzw_hull_matrix_fraction(
                3, Fraction(0), Fraction(1)
            )

    def test_endpoint_gram_matrices_follow_direct_computational_basis(self) -> None:
        active_rows_at_c0 = (1, 2, 3, 5, 6, 8)
        for name, gram in (
            ("coordinate_c0", weighted_hull_gram_fraction(4, Fraction(0))),
            ("direct_c0", direct_ge_two_gram_fraction(4, Fraction(0))),
        ):
            for i in range(10):
                for j in range(10):
                    expected = Fraction(
                        int(i == j and i in active_rows_at_c0)
                    )
                    with self.subTest(construction=name, i=i, j=j):
                        self.assertEqual(gram[i][j], expected)

        expected_c1 = [[Fraction(0)] * 10 for _ in range(10)]
        self.assertEqual(
            weighted_hull_gram_fraction(4, Fraction(1)), expected_c1
        )
        self.assertEqual(
            direct_ge_two_gram_fraction(4, Fraction(1)), expected_c1
        )


if __name__ == "__main__":
    unittest.main()
