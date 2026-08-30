"""Integration tests for the Paper I critical-SRM diagnostic export."""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper1"
    / "quantum_revision_ultracritical"
    / "code"
    / "unknown_length_full_srm_probe.py"
)
FROZEN_CSV = (
    ROOT
    / "paper1"
    / "quantum_revision_ultracritical"
    / "data"
    / "critical_srm_diagnostics.csv"
)


class CriticalSrmCsvExportTests(unittest.TestCase):
    def test_documented_grid_writes_frozen_csv_and_keeps_stdout(self) -> None:
        """Catch a missing CSV export or a change to the documented grid."""
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "critical_srm_diagnostics.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--n",
                    "12",
                    "18",
                    "24",
                    "32",
                    "--tau",
                    "0.25",
                    "1",
                    "4",
                    "--retained",
                    "--output",
                    str(output),
                ],
                cwd=SCRIPT.parent.parent,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated_text = output.read_text(encoding="utf-8")
            frozen_text = FROZEN_CSV.read_text(encoding="utf-8")

        self.assertEqual(
            generated_text.splitlines()[0], frozen_text.splitlines()[0]
        )
        generated_rows = list(csv.DictReader(io.StringIO(generated_text)))
        frozen_rows = list(csv.DictReader(io.StringIO(frozen_text)))
        self.assertEqual(len(generated_rows), 12)
        self.assertEqual(len(generated_rows), len(frozen_rows))
        for generated, frozen in zip(
            generated_rows, frozen_rows, strict=True
        ):
            self.assertEqual(generated.keys(), frozen.keys())
            self.assertEqual(generated["n"], frozen["n"])
            self.assertEqual(generated["tau"], frozen["tau"])
            for field in generated.keys() - {"n", "tau"}:
                expected = float(frozen[field])
                observed = float(generated[field])
                tolerance = 5.0e-7 * max(1.0, abs(expected))
                self.assertTrue(
                    math.isclose(
                        observed,
                        expected,
                        rel_tol=5.0e-7,
                        abs_tol=tolerance,
                    ),
                    f"{field}: observed {observed}, expected {expected}",
                )

        self.assertEqual(completed.stdout.count("n="), 12)
        self.assertEqual(completed.stdout.count("retained "), 12)


if __name__ == "__main__":
    unittest.main()
