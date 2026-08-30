"""Build deterministic, allowlisted Paper I release artifacts.

The packager deliberately ignores unlisted workspace files.  This keeps
Paper II material, build products, editor state, and transient logs out of the
Paper I archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Iterable, Mapping
import zipfile


MANUSCRIPT_REL = Path("paper1/quantum_revision_ultracritical")
REQUIRED_MANUSCRIPT_FILES = (
    "main.tex",
    "content.tex",
    "supplement.tex",
    "supplement_content.tex",
    "references.bib",
    "quantumarticle.cls",
    "quantum.bst",
    "figures/figure1_model_geometry.pdf",
    "figures/figure2_analytic_limits.pdf",
    "figures/figure3_finite_size.pdf",
)

ROOT_RELEASE_FILES = (
    ".python-version",
    "CITATION.cff",
    "LICENSE",
    "NUMERICAL_REPRODUCIBILITY.md",
    "README.md",
    "REPRODUCING.md",
    "environment_manifest.py",
    "interval_unknown_length_numerics.py",
    "paper1_analytics.py",
    "paper1_make_figures.py",
    "paper1_numerics.py",
    "pyproject.toml",
    "quantum_interval_numerics.py",
    "requirements-lock.txt",
    "reproduction_manifest.json",
    "sdp_certification.py",
    "srm_scaling.py",
    "verify_paper1_results.py",
    "certified_sdp_results.csv",
    "srm_scaling_m1.csv",
)

PAPER1_RELEASE_FILES = (
    "paper1/NUMERICAL_REPRODUCIBILITY.md",
    "paper1/PUBLIC_RELEASE_CHECKLIST.md",
    "paper1/README.md",
    "paper1/REPRODUCING.md",
    "paper1/ARXIV_V2_METADATA.md",
    "paper1/QUANTUM_COVER_LETTER.md",
    "paper1/PAPER1_RELEASE_MANIFEST.md",
    "paper1/build_release.py",
    "paper1/paper1_fixed_and_growing_srm.csv",
    "paper1/paper1_h0_certified_sdp.csv",
    "paper1/reproduction_manifest.json",
    "paper1/reproduced/paper1_fixed_and_growing_srm.csv",
    "paper1/reproduced/paper1_h0_certified_sdp.csv",
    "paper1/reproduced/certified_sdp_results.csv",
    "paper1/reproduced/srm_scaling_m1.csv",
    "paper1/run_reproduction.ps1",
    "paper1/verification_report.audit.json",
    "paper1/verification_report.json",
)

PAPER1_TEST_FILES = (
    "tests/test_environment_manifest.py",
    "tests/test_paper1_analytics.py",
    "tests/test_paper1_figures.py",
    "tests/test_paper1_numerics.py",
    "tests/test_paper1_release_packaging.py",
    "tests/test_quantum_interval_numerics.py",
    "tests/test_sdp_certification.py",
    "tests/test_srm_scaling.py",
    "tests/test_unknown_length_full_srm_probe.py",
    "tests/test_weighted_block_tail.py",
    "tests/test_weighted_hull_adaptive_blocks.py",
    "tests/test_weighted_hull_asymptotics.py",
    "tests/test_weighted_hull_continuum_certificate.py",
    "tests/test_weighted_hull_exact.py",
    "tests/test_weighted_hull_outer_ledger.py",
    "tests/test_weighted_hull_regimes.py",
    "tests/test_weighted_hull_sdp.py",
)

PAPER1_PROOF_FILES = (
    "proofs/weighted_hull_finite_audit.py",
    "proofs/weighted_block_tail_probe.py",
    "proofs/weighted_hull_asymptotic_probe.py",
    "proofs/weighted_hull_continuum_outer_probe.py",
    "proofs/weighted_hull_diagnostics.csv",
    "proofs/plot_task8_candidate_figure3.py",
)

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_version(version: str) -> str:
    if not re.fullmatch(r"v?[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("version must contain only letters, digits, '.', '_', and '-'")
    return version


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required release input is missing: {path}")
    return path


def _existing_files(root: Path, relative_paths: Iterable[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in relative_paths:
        path = root / Path(relative)
        if path.is_file():
            files[PurePosixPath(relative).as_posix()] = path
    return files


def _write_zip(path: Path, members: Mapping[str, Path]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(members):
            normalized = PurePosixPath(name).as_posix()
            if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
                raise ValueError(f"unsafe archive member: {name}")
            info = zipfile.ZipInfo(normalized, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name].read_bytes(), compresslevel=9)


def _collect_arxiv_members(
    manuscript_root: Path,
    supplement_pdf: Path,
    main_bbl: Path | None = None,
) -> dict[str, Path]:
    members = {
        name: _require_file(manuscript_root / Path(name))
        for name in REQUIRED_MANUSCRIPT_FILES
        if name not in {"supplement.tex", "supplement_content.tex"}
    }
    bibliography = (
        _require_file(main_bbl)
        if main_bbl is not None
        else manuscript_root / "build-main" / "main.bbl"
    )
    if bibliography.is_file():
        members["main.bbl"] = bibliography
    members["anc/supplement.pdf"] = supplement_pdf
    return members


def _collect_source_members(repository_root: Path) -> dict[str, Path]:
    manuscript_root = repository_root / MANUSCRIPT_REL
    members: dict[str, Path] = {}
    for relative in REQUIRED_MANUSCRIPT_FILES:
        source = _require_file(manuscript_root / Path(relative))
        archive_name = (MANUSCRIPT_REL / Path(relative)).as_posix()
        members[archive_name] = source
    optional_manuscript_files = (
        "code/unknown_length_full_srm_probe.py",
        "data/critical_srm_diagnostics.csv",
        "figures/figure1_model_geometry.png",
        "figures/figure1_model_geometry.svg",
        "figures/figure1_model_geometry.tiff",
        "figures/figure2_analytic_limits.png",
        "figures/figure3_finite_size.png",
        "FINAL_QUANTUM_REVIEW_20260831.md",
        "FINAL_PDF_QA_20260831.md",
    )
    for relative in optional_manuscript_files:
        source = manuscript_root / Path(relative)
        if source.is_file():
            members[(MANUSCRIPT_REL / Path(relative)).as_posix()] = source
    members.update(_existing_files(repository_root, ROOT_RELEASE_FILES))
    members.update(_existing_files(repository_root, PAPER1_RELEASE_FILES))
    members.update(_existing_files(repository_root, PAPER1_TEST_FILES))
    members.update(_existing_files(repository_root, PAPER1_PROOF_FILES))
    return members


def _artifact_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def build_release(
    *,
    version: str,
    main_pdf: str | Path,
    supplement_pdf: str | Path,
    main_bbl: str | Path | None = None,
    output_dir: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build all release artifacts and return their paths by logical name."""

    version = _validate_version(version)
    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    manuscript_root = root / MANUSCRIPT_REL
    main_pdf_path = _require_file(Path(main_pdf).resolve())
    supplement_pdf_path = _require_file(Path(supplement_pdf).resolve())

    # Validate and collect every required source before creating output_dir.
    main_bbl_path = None if main_bbl is None else Path(main_bbl).resolve()
    arxiv_members = _collect_arxiv_members(
        manuscript_root,
        supplement_pdf_path,
        main_bbl_path,
    )
    source_members = _collect_source_members(root)

    output = Path(output_dir).resolve()
    prefix = f"quantum-change-intervals-{version}"
    names = {
        "arxiv_source_zip": f"{prefix}-arxiv-source.zip",
        "source_zip": f"{prefix}-source.zip",
        "github_release_zip": f"{prefix}-github-release.zip",
        "quantum_main_pdf": f"{prefix}-quantum-main.pdf",
        "supplement_pdf": f"{prefix}-supplement.pdf",
        "inventory_json": f"{prefix}-inventory.json",
        "sha256sums": "SHA256SUMS",
    }
    paths = {key: output / name for key, name in names.items()}

    output.mkdir(parents=True, exist_ok=True)
    _write_zip(paths["arxiv_source_zip"], arxiv_members)
    _write_zip(paths["source_zip"], source_members)
    shutil.copyfile(paths["source_zip"], paths["github_release_zip"])
    shutil.copyfile(main_pdf_path, paths["quantum_main_pdf"])
    shutil.copyfile(supplement_pdf_path, paths["supplement_pdf"])

    primary_keys = (
        "arxiv_source_zip",
        "source_zip",
        "github_release_zip",
        "quantum_main_pdf",
        "supplement_pdf",
    )
    artifact_records = {
        paths[key].name: _artifact_record(paths[key]) for key in primary_keys
    }
    inventory = {
        "schema_version": 1,
        "version": version,
        "artifacts": artifact_records,
    }
    paths["inventory_json"].write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    checksum_keys = (*primary_keys, "inventory_json")
    checksum_lines = [
        f"{_sha256(paths[key])}  {paths[key].name}" for key in checksum_keys
    ]
    paths["sha256sums"].write_text(
        "\n".join(sorted(checksum_lines)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--main-pdf", required=True, type=Path)
    parser.add_argument("--supplement-pdf", required=True, type=Path)
    parser.add_argument(
        "--main-bbl",
        type=Path,
        help="Fresh main.bbl from the same clean build as --main-pdf.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = build_release(
        version=args.version,
        main_pdf=args.main_pdf,
        supplement_pdf=args.supplement_pdf,
        main_bbl=args.main_bbl,
        output_dir=args.output_dir,
        repository_root=args.repository_root,
    )
    for key in sorted(outputs):
        print(f"{key}: {outputs[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
