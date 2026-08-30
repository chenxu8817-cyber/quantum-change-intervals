from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper1_analytics import fixed_length_limit_quadrature
from paper1_make_figures import (
    fixed_length_limit_curve,
    make_all_figures,
    make_figure_3,
)


ROOT = Path(__file__).resolve().parents[1]


class Paper1FigureTests(unittest.TestCase):
    def test_vectorized_fixed_length_curve_matches_quadrature(self) -> None:
        for length in (1, 2, 4, 8):
            c_values = np.array([0.0, 0.25, 0.7, 0.95, 1.0])
            actual = fixed_length_limit_curve(c_values, length)
            expected = np.array(
                [fixed_length_limit_quadrature(length, c * c) for c in c_values]
            )
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)

    def test_all_figures_export_pdf_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = make_all_figures(Path(directory))
            self.assertEqual(len(outputs), 6)
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "figure1_model_geometry.pdf",
                    "figure1_model_geometry.png",
                    "figure2_analytic_limits.pdf",
                    "figure2_analytic_limits.png",
                    "figure3_finite_size.pdf",
                    "figure3_finite_size.png",
                },
            )
            for path in outputs:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def test_figure3_rejects_an_incomplete_sdp_grid(self) -> None:
        source = ROOT / "certified_sdp_results.csv"
        lines = source.read_text(encoding="utf-8-sig").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            incomplete = temporary / "incomplete_sdp.csv"
            incomplete.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete rectangular"):
                make_figure_3(
                    temporary / "output",
                    sdp_data_path=incomplete,
                )


if __name__ == "__main__":
    unittest.main()
