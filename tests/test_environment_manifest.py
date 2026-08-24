from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_manifest import collect_manifest  # noqa: E402


class EnvironmentManifestTests(unittest.TestCase):
    def test_manifest_records_runtime_solver_and_blas_information(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manifest = collect_manifest(
                command=["unit-test"], random_seed=1729
            )
        self.assertEqual(caught, [])
        self.assertEqual(manifest["random_seed"], 1729)
        self.assertIn("python", manifest)
        self.assertIn("packages", manifest)
        self.assertIn("numpy", manifest["packages"])
        self.assertIn("cvxpy", manifest["packages"])
        self.assertIn("installed_solvers", manifest)
        self.assertIn("thread_environment", manifest)
        self.assertIn("pyproject.toml", manifest["file_sha256"])
        self.assertIn(
            "certified_sdp_results.csv", manifest["file_sha256"]
        )
        json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()
