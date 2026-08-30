from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "paper1" / "build_release.py"


def _load_builder():
    if not BUILDER_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("paper1_build_release", BUILDER_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _fixture(root: Path) -> tuple[Path, Path]:
    manuscript = root / "paper1" / "quantum_revision_ultracritical"
    for name in (
        "main.tex",
        "content.tex",
        "supplement.tex",
        "supplement_content.tex",
        "references.bib",
        "quantumarticle.cls",
        "quantum.bst",
    ):
        _write(manuscript / name, f"fixture {name}\n")
    _write(manuscript / "build-main" / "main.bbl", "fixture bbl\n")
    _write(manuscript / "build-main" / "main.aux", "forbidden aux\n")
    _write(manuscript / "figures" / "figure1_model_geometry.pdf", b"%PDF-figure-1")
    _write(manuscript / "figures" / "figure2_analytic_limits.pdf", b"%PDF-figure-2")
    _write(manuscript / "figures" / "figure3_finite_size.pdf", b"%PDF-figure-3")
    _write(manuscript / "code" / "unknown_length_full_srm_probe.py", "print('probe')\n")
    _write(manuscript / "data" / "critical_srm_diagnostics.csv", "n,value\n3,0.5\n")

    for name in (
        ".python-version",
        "LICENSE",
        "pyproject.toml",
        "requirements-lock.txt",
        "paper1_analytics.py",
        "paper1_make_figures.py",
        "paper1_numerics.py",
        "quantum_interval_numerics.py",
        "sdp_certification.py",
        "srm_scaling.py",
        "verify_paper1_results.py",
        "environment_manifest.py",
        "interval_unknown_length_numerics.py",
    ):
        _write(root / name, f"fixture {name}\n")
    _write(root / "two_unknown_intervals_numerics.py", "forbidden Paper II\n")
    _write(root / "tests" / "test_paper1_numerics.py", "# fixture Paper I test\n")
    _write(root / "tests" / "test_paper1_release_packaging.py", "# fixture release test\n")
    _write(root / "tests" / "test_weighted_hull_exact.py", "# fixture theorem test\n")
    _write(root / "tests" / "test_m3_forest_factorization.py", "# forbidden Paper II test\n")
    for name in (
        "weighted_hull_finite_audit.py",
        "weighted_block_tail_probe.py",
        "weighted_hull_asymptotic_probe.py",
        "weighted_hull_continuum_outer_probe.py",
        "plot_task8_candidate_figure3.py",
    ):
        _write(root / "proofs" / name, "# fixture proof probe\n")
    _write(root / "proofs" / "weighted_hull_diagnostics.csv", "n,value\n3,0.5\n")
    _write(root / "proofs" / "unknown_length_hull_dominance_unified.md", "fixture proof\n")
    _write(root / "proofs" / "m3_forest_notes.md", "forbidden Paper II\n")
    _write(root / "paper1" / "REPRODUCING.md", "fixture reproduction guide\n")
    _write(root / "paper1" / "NUMERICAL_REPRODUCIBILITY.md", "fixture numerical guide\n")
    _write(root / "paper1" / "PAPER1_SCOPE_FREEZE.md", "fixture scope\n")
    _write(root / "paper1" / "PAPER1_RELEASE_MANIFEST.md", "fixture release manifest\n")
    _write(root / "paper1" / "build_release.py", "# fixture release builder\n")
    _write(root / "paper1" / "run_reproduction.ps1", "# fixture runner\n")
    _write(root / "paper1" / "paper1_fixed_and_growing_srm.csv", "n,value\n3,0.5\n")
    _write(root / "paper1" / "paper1_h0_certified_sdp.csv", "n,value\n3,0.5\n")
    for name in (
        "paper1_fixed_and_growing_srm.csv",
        "paper1_h0_certified_sdp.csv",
        "certified_sdp_results.csv",
        "srm_scaling_m1.csv",
    ):
        _write(root / "paper1" / "reproduced" / name, "n,value\n3,0.5\n")

    main_pdf = root / "compiled" / "main.pdf"
    supplement_pdf = root / "compiled" / "supplement.pdf"
    _write(main_pdf, b"%PDF-main")
    _write(supplement_pdf, b"%PDF-supplement")
    return main_pdf, supplement_pdf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Paper1ReleasePackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = _load_builder()
        self.assertIsNotNone(
            self.builder,
            "Paper I release packager has not been implemented",
        )

    def _build(self, root: Path, output: Path):
        main_pdf, supplement_pdf = _fixture(root)
        return self.builder.build_release(
            version="v1.3.0-paper1",
            main_pdf=main_pdf,
            supplement_pdf=supplement_pdf,
            output_dir=output,
            repository_root=root,
        )

    def test_arxiv_archive_has_exact_compile_ready_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outputs = self._build(base / "repo", base / "out")
            with zipfile.ZipFile(outputs["arxiv_source_zip"]) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "anc/supplement.pdf",
                        "content.tex",
                        "figures/figure1_model_geometry.pdf",
                        "figures/figure2_analytic_limits.pdf",
                        "figures/figure3_finite_size.pdf",
                        "main.bbl",
                        "main.tex",
                        "quantum.bst",
                        "quantumarticle.cls",
                        "references.bib",
                    ],
                )

    def test_explicit_main_bbl_overrides_a_stale_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            main_pdf, supplement_pdf = _fixture(repo)
            fresh_bbl = base / "clean-build" / "main.bbl"
            _write(fresh_bbl, b"fresh bibliography\n")
            outputs = self.builder.build_release(
                version="v1.3.0-paper1",
                main_pdf=main_pdf,
                supplement_pdf=supplement_pdf,
                main_bbl=fresh_bbl,
                output_dir=base / "out",
                repository_root=repo,
            )
            with zipfile.ZipFile(outputs["arxiv_source_zip"]) as archive:
                self.assertEqual(
                    archive.read("main.bbl"),
                    b"fresh bibliography\n",
                )

    def test_source_archives_include_paper1_assets_and_exclude_paper2_and_build_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outputs = self._build(base / "repo", base / "out")
            for key in ("source_zip", "github_release_zip"):
                with zipfile.ZipFile(outputs[key]) as archive:
                    members = archive.namelist()
                    self.assertIn(
                        "paper1/quantum_revision_ultracritical/supplement_content.tex",
                        members,
                    )
                    self.assertIn("paper1_make_figures.py", members)
                    self.assertIn("LICENSE", members)
                    self.assertIn("interval_unknown_length_numerics.py", members)
                    self.assertIn("paper1/build_release.py", members)
                    self.assertIn("paper1/PAPER1_RELEASE_MANIFEST.md", members)
                    self.assertIn("tests/test_paper1_release_packaging.py", members)
                    self.assertIn("tests/test_weighted_hull_exact.py", members)
                    self.assertIn(
                        "paper1/reproduced/certified_sdp_results.csv", members
                    )
                    self.assertIn(
                        "proofs/weighted_hull_finite_audit.py",
                        members,
                    )
                    self.assertIn(
                        "proofs/weighted_hull_continuum_outer_probe.py",
                        members,
                    )
                    self.assertTrue(all("\\" not in member for member in members))
                    forbidden_fragments = (
                        "build-main/",
                        "build-supplement/",
                        ".aux",
                        ".log",
                        "two_unknown_intervals",
                        "test_m3_forest",
                        "m3_forest",
                        "unknown_length_hull_dominance_unified.md",
                        "unknown_length_ultracritical_sector_report.md",
                        "weighted_hull_blind_audit.md",
                        "WEIGHTED_HULL_INTEGRATION_AUDIT.md",
                    )
                    self.assertFalse(
                        any(
                            fragment in member
                            for member in members
                            for fragment in forbidden_fragments
                        )
                    )

    def test_repeated_builds_have_identical_zip_hashes_and_normalized_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            main_pdf, supplement_pdf = _fixture(repo)
            first = self.builder.build_release(
                version="v1.3.0-paper1",
                main_pdf=main_pdf,
                supplement_pdf=supplement_pdf,
                output_dir=base / "out-a",
                repository_root=repo,
            )
            second = self.builder.build_release(
                version="v1.3.0-paper1",
                main_pdf=main_pdf,
                supplement_pdf=supplement_pdf,
                output_dir=base / "out-b",
                repository_root=repo,
            )
            for key in ("arxiv_source_zip", "source_zip", "github_release_zip"):
                self.assertEqual(_sha256(first[key]), _sha256(second[key]))
                with zipfile.ZipFile(first[key]) as archive:
                    self.assertTrue(
                        all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
                    )

    def test_inventory_and_sha256sums_cover_outputs_without_self_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outputs = self._build(base / "repo", base / "out")
            inventory = json.loads(outputs["inventory_json"].read_text(encoding="utf-8"))
            self.assertEqual(inventory["version"], "v1.3.0-paper1")
            self.assertNotIn(outputs["inventory_json"].name, inventory["artifacts"])
            self.assertNotIn("SHA256SUMS", inventory["artifacts"])

            checksum_lines = outputs["sha256sums"].read_text(encoding="utf-8").splitlines()
            recorded: dict[str, str] = {}
            for line in checksum_lines:
                digest, name = line.split("  ", 1)
                recorded[name] = digest
            self.assertNotIn("SHA256SUMS", recorded)
            self.assertIn(outputs["inventory_json"].name, recorded)
            for name, digest in recorded.items():
                self.assertEqual(digest, _sha256(outputs["sha256sums"].parent / name))

            copied_main = outputs["quantum_main_pdf"]
            copied_supplement = outputs["supplement_pdf"]
            self.assertEqual(copied_main.read_bytes(), b"%PDF-main")
            self.assertEqual(copied_supplement.read_bytes(), b"%PDF-supplement")

    def test_missing_required_source_fails_before_writing_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            main_pdf, supplement_pdf = _fixture(repo)
            (repo / "paper1" / "quantum_revision_ultracritical" / "main.tex").unlink()
            output = base / "out"
            with self.assertRaisesRegex(FileNotFoundError, "main.tex"):
                self.builder.build_release(
                    version="v1.3.0-paper1",
                    main_pdf=main_pdf,
                    supplement_pdf=supplement_pdf,
                    output_dir=output,
                    repository_root=repo,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
