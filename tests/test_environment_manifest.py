from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_manifest import PAPER1_HASHED_FILES, collect_manifest  # noqa: E402


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
        self.assertEqual(
            manifest["python"]["executable"], Path(sys.executable).name
        )
        self.assertFalse(
            any(Path(token).is_absolute() for token in manifest["command"])
        )
        self.assertIn("pyproject.toml", manifest["file_sha256"])
        self.assertIn(
            "certified_sdp_results.csv", manifest["file_sha256"]
        )
        json.dumps(manifest)

    def test_paper1_profile_hashes_the_current_release_evidence(self) -> None:
        manifest = collect_manifest(
            command=["unit-test", "--profile", "paper1"],
            hashed_files=PAPER1_HASHED_FILES,
        )
        hashes = manifest["file_sha256"]
        required = {
            "paper1/quantum_revision_ultracritical/main.tex",
            "paper1/quantum_revision_ultracritical/content.tex",
            "paper1/quantum_revision_ultracritical/supplement.tex",
            "paper1/quantum_revision_ultracritical/supplement_content.tex",
            "paper1/quantum_revision_ultracritical/references.bib",
            "paper1/quantum_revision_ultracritical/code/unknown_length_full_srm_probe.py",
            "paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv",
            "paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.pdf",
            "paper1/quantum_revision_ultracritical/figures/figure2_analytic_limits.pdf",
            "paper1/quantum_revision_ultracritical/figures/figure3_finite_size.pdf",
            "proofs/weighted_hull_continuum_outer_probe.py",
            "proofs/weighted_hull_diagnostics.csv",
            "paper1/reproduced/paper1_fixed_and_growing_srm.csv",
            "paper1/reproduced/paper1_h0_certified_sdp.csv",
            "paper1/reproduced/certified_sdp_results.csv",
            "paper1/reproduced/srm_scaling_m1.csv",
            "paper1/build_release.py",
            "paper1_make_figures.py",
            "paper1_numerics.py",
            "environment_manifest.py",
            "tests/test_paper1_release_packaging.py",
        }
        self.assertTrue(required.issubset(hashes), required - set(hashes))
        self.assertFalse(
            any("quantum_submission" in relative for relative in hashes)
        )
        for relative, digest in hashes.items():
            self.assertTrue(ROOT.joinpath(relative).is_file(), relative)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
