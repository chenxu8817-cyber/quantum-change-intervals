from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment_manifest import (  # noqa: E402
    PAPER1_HASHED_FILES,
    _portable_numpy_configuration,
    _file_sha256,
    collect_manifest,
    verify_manifest_file,
)


class EnvironmentManifestTests(unittest.TestCase):
    def test_numpy_configuration_redacts_absolute_host_paths(self) -> None:
        raw = json.dumps(
            {
                "Build Dependencies": {
                    "blas": {
                        "include directory": "C:/Users/private/build/include",
                        "lib directory": "/home/private/build/lib",
                        "version": "0.3.27",
                    }
                }
            }
        )
        portable = _portable_numpy_configuration(raw)
        self.assertNotIn("C:/Users/private", portable)
        self.assertNotIn("/home/private", portable)
        self.assertIn("0.3.27", portable)
        self.assertEqual(portable.count("<absolute-path-redacted>"), 2)

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

    def test_verify_manifest_accepts_fresh_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracked = root / "tracked.txt"
            tracked.write_text("release candidate\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"file_sha256": {"tracked.txt": _file_sha256(tracked)}}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_manifest_file(
                    manifest_path,
                    root=root,
                    expected_files=["tracked.txt"],
                ),
                [],
            )

    def test_verify_manifest_detects_missing_and_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changed = root / "changed.txt"
            changed.write_text("before\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"file_sha256": {"changed.txt": _file_sha256(changed)}}
                ),
                encoding="utf-8",
            )
            changed.write_text("after\n", encoding="utf-8")
            issues = verify_manifest_file(
                manifest_path,
                root=root,
                expected_files=["changed.txt", "missing.txt"],
            )
            self.assertTrue(any("SHA-256 mismatch" in issue for issue in issues))
            self.assertIn("missing manifest hash entry: missing.txt", issues)

    def test_verify_manifest_rejects_nonportable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "file_sha256": {
                            "../outside.txt": "0" * 64,
                            "C:/private/file.txt": "0" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            issues = verify_manifest_file(manifest_path, root=root)
            self.assertTrue(any("escapes repository root" in issue for issue in issues))
            self.assertTrue(any("absolute path" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
