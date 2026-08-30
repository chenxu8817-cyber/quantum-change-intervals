"""Compact-lambda selection and compatibility tests after Task 4 extraction.

Generic suffix, extension, local, tensor, cell, and finite-global behavior was
moved to ``tests.test_weighted_block_tail``.  This module now tests only the
compact-lambda choice of blocks, full-hypothesis accounting, and legacy public
re-exports needed by the already accepted Task 3 interface.
"""

from __future__ import annotations

import unittest

import numpy as np

import proofs.weighted_block_tail_probe as reusable_probe
import proofs.weighted_hull_asymptotic_probe as continuum_probe


class ContinuumCertificateCompatibilityTests(unittest.TestCase):
    def test_task3_public_certificate_api_reexports_reusable_definitions(self) -> None:
        names = (
            "extend_certificate",
            "extend_with_budget",
            "global_hull_certificate",
            "left_suffix",
            "minimum_certificate_slack",
            "optimized_extension",
            "physical_hull_rows",
            "right_suffix",
            "split_left_suffix",
            "split_right_suffix",
            "tensor_cell_certificate",
            "weighted_local_certificate",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(continuum_probe, name), getattr(reusable_probe, name)
                )

    def test_compact_lambda_block_choice_builds_a_full_feasible_family(self) -> None:
        parameters = continuum_probe.continuum_parameters(8, 1.0)
        n = parameters["n"]
        block_count = parameters["block_count"]
        certificate, cells = continuum_probe.global_hull_certificate(
            n, parameters["c"], block_count, return_cells=True
        )
        rows = continuum_probe.physical_hull_rows(n, parameters["c"])
        hypothesis_count = n * (n + 1) // 2
        hull_dimension = n * (n - 1) // 2

        self.assertEqual(rows.shape, (hypothesis_count, hull_dimension))
        self.assertEqual(len(cells), block_count * (block_count + 1) // 2)
        self.assertEqual(
            set(cells),
            {
                (p, q)
                for p in range(block_count)
                for q in range(p, block_count)
            },
        )
        np.testing.assert_allclose(
            certificate, sum(cells.values(), np.zeros_like(certificate))
        )
        slack = continuum_probe.minimum_certificate_slack(certificate, rows)
        scale = max(1.0, float(np.linalg.norm(certificate, 2)))
        self.assertGreaterEqual(slack, -1.0e-12 * scale)

    def test_legacy_and_reusable_calls_have_identical_finite_results(self) -> None:
        old_result = continuum_probe.global_hull_certificate(
            5, 0.72, 2, return_cells=True
        )
        new_result = reusable_probe.global_hull_certificate(
            5, 0.72, 2, return_cells=True
        )
        np.testing.assert_array_equal(old_result[0], new_result[0])
        self.assertEqual(set(old_result[1]), set(new_result[1]))
        for key in old_result[1]:
            np.testing.assert_array_equal(old_result[1][key], new_result[1][key])


if __name__ == "__main__":
    unittest.main()
