"""Generate a machine-readable reproduction environment manifest."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".local_pydeps"
if os.environ.get("QCI_USE_LOCAL_DEPS") == "1" and LOCAL_DEPS.exists():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(".local_pydeps is a CPython 3.12 cache")
    sys.path.insert(0, str(LOCAL_DEPS))

import cvxpy as cp
import numpy as np


TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "cvxpy",
    "clarabel",
    "scs",
    "osqp",
]
THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
HASHED_FILES = [
    ".python-version",
    "LICENSE",
    "pyproject.toml",
    "requirements-lock.txt",
    "REPRODUCING.md",
    "NUMERICAL_REPRODUCIBILITY.md",
    "quantum_change_interval_research.md",
    "quantum_interval_numerics.py",
    "sdp_certification.py",
    "fixed_m_sdp_grid.py",
    "interval_unknown_length_numerics.py",
    "srm_scaling.py",
    "tests/test_m3_forest_factorization.py",
    "proofs/EnergyPayment.lean",
    "certified_sdp_results.csv",
    "certified_sdp_m2.csv",
    "certified_sdp_m3.csv",
    "srm_scaling_m1.csv",
    "srm_scaling_m2.csv",
    "srm_scaling_m3.csv",
]
PAPER1_HASHED_FILES = [
    ".python-version",
    "LICENSE",
    "pyproject.toml",
    "requirements-lock.txt",
    "environment_manifest.py",
    "CITATION.cff",
    "README.md",
    "REPRODUCING.md",
    "NUMERICAL_REPRODUCIBILITY.md",
    "paper1/README.md",
    "paper1/REPRODUCING.md",
    "paper1/NUMERICAL_REPRODUCIBILITY.md",
    "paper1/PUBLIC_RELEASE_CHECKLIST.md",
    "paper1/PAPER1_RELEASE_MANIFEST.md",
    "paper1/ARXIV_V2_METADATA.md",
    "paper1/QUANTUM_COVER_LETTER.md",
    "paper1/build_release.py",
    "paper1/run_reproduction.ps1",
    "paper1/verification_report.json",
    "paper1_analytics.py",
    "paper1_numerics.py",
    "paper1_make_figures.py",
    "verify_paper1_results.py",
    "quantum_interval_numerics.py",
    "sdp_certification.py",
    "interval_unknown_length_numerics.py",
    "srm_scaling.py",
    "tests/test_environment_manifest.py",
    "tests/test_paper1_analytics.py",
    "tests/test_paper1_numerics.py",
    "tests/test_paper1_release_packaging.py",
    "tests/test_paper1_figures.py",
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
    "certified_sdp_results.csv",
    "srm_scaling_m1.csv",
    "paper1/paper1_fixed_and_growing_srm.csv",
    "paper1/paper1_h0_certified_sdp.csv",
    "paper1/reproduced/paper1_fixed_and_growing_srm.csv",
    "paper1/reproduced/paper1_h0_certified_sdp.csv",
    "paper1/reproduced/certified_sdp_results.csv",
    "paper1/reproduced/srm_scaling_m1.csv",
    "proofs/weighted_hull_finite_audit.py",
    "proofs/weighted_block_tail_probe.py",
    "proofs/weighted_hull_asymptotic_probe.py",
    "proofs/weighted_hull_continuum_outer_probe.py",
    "proofs/weighted_hull_diagnostics.csv",
    "proofs/plot_task8_candidate_figure3.py",
    "paper1/quantum_revision_ultracritical/main.tex",
    "paper1/quantum_revision_ultracritical/content.tex",
    "paper1/quantum_revision_ultracritical/supplement.tex",
    "paper1/quantum_revision_ultracritical/supplement_content.tex",
    "paper1/quantum_revision_ultracritical/references.bib",
    "paper1/quantum_revision_ultracritical/quantumarticle.cls",
    "paper1/quantum_revision_ultracritical/quantum.bst",
    "paper1/quantum_revision_ultracritical/code/unknown_length_full_srm_probe.py",
    "paper1/quantum_revision_ultracritical/data/critical_srm_diagnostics.csv",
    "paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.pdf",
    "paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.png",
    "paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.svg",
    "paper1/quantum_revision_ultracritical/figures/figure1_model_geometry.tiff",
    "paper1/quantum_revision_ultracritical/figures/figure2_analytic_limits.pdf",
    "paper1/quantum_revision_ultracritical/figures/figure2_analytic_limits.png",
    "paper1/quantum_revision_ultracritical/figures/figure3_finite_size.pdf",
    "paper1/quantum_revision_ultracritical/figures/figure3_finite_size.png",
    "paper1/quantum_revision_ultracritical/FINAL_PDF_QA_20260901.md",
    "paper1/quantum_revision_ultracritical/FINAL_QUANTUM_REVIEW_20260901.md",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_command(command: list[str]) -> list[str]:
    """Remove host-specific absolute paths from the public manifest."""

    portable: list[str] = []
    for token in command:
        candidate = Path(token)
        if not candidate.is_absolute():
            portable.append(token)
            continue
        try:
            portable.append(candidate.resolve().relative_to(ROOT).as_posix())
        except ValueError:
            portable.append(candidate.name)
    return portable


def _portable_numpy_configuration(raw: str) -> str:
    """Redact absolute build paths from NumPy's public configuration text."""

    try:
        configuration = json.loads(raw)
    except json.JSONDecodeError:
        # NumPy's current output is JSON when PyYAML is unavailable.  If a
        # future version changes that format, fail closed instead of exposing
        # an unreviewed host path in a public manifest.
        return "<unparsed NumPy configuration omitted>"

    def redact(value: object) -> object:
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str) and (
            PureWindowsPath(value).is_absolute()
            or PurePosixPath(value).is_absolute()
        ):
            return "<absolute-path-redacted>"
        return value

    return json.dumps(redact(configuration), indent=2, ensure_ascii=False)


def collect_manifest(
    command: list[str] | None = None,
    random_seed: int = 1729,
    hashed_files: list[str] | None = None,
) -> dict[str, object]:
    """Collect versions, solver availability, BLAS data, and file hashes."""
    numpy_config = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Install .* for better output",
            category=UserWarning,
        )
        with contextlib.redirect_stdout(numpy_config):
            np.show_config()
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    selected_files = HASHED_FILES if hashed_files is None else hashed_files
    hashes = {
        relative: _file_sha256(ROOT / relative)
        for relative in selected_files
        if (ROOT / relative).exists()
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": _portable_command(command if command is not None else sys.argv),
        "random_seed": random_seed,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "installed_solvers": sorted(cp.installed_solvers()),
        "thread_environment": {
            variable: os.environ.get(variable)
            for variable in THREAD_VARIABLES
        },
        "numpy_configuration": _portable_numpy_configuration(
            numpy_config.getvalue()
        ),
        "file_sha256": hashes,
    }


def verify_manifest_file(
    manifest_path: Path,
    *,
    root: Path = ROOT,
    expected_files: list[str] | None = None,
) -> list[str]:
    """Return portability, completeness, and SHA-256 errors for a manifest.

    The verifier deliberately checks the files on disk rather than trusting
    the manifest's recorded environment metadata.  An empty return value is a
    release-gate pass.
    """

    issues: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read manifest {manifest_path}: {exc}"]

    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        return ["manifest field 'file_sha256' is missing or is not an object"]

    if expected_files is not None:
        for relative in expected_files:
            if relative not in hashes:
                issues.append(f"missing manifest hash entry: {relative}")

    resolved_root = root.resolve()
    for relative, recorded_digest in hashes.items():
        if not isinstance(relative, str) or not isinstance(recorded_digest, str):
            issues.append("manifest hash entries must map strings to strings")
            continue
        if (
            PureWindowsPath(relative).is_absolute()
            or PurePosixPath(relative).is_absolute()
        ):
            issues.append(f"absolute path in manifest: {relative}")
            continue
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root):
            issues.append(f"path escapes repository root: {relative}")
            continue
        if not candidate.is_file():
            issues.append(f"hashed file is missing: {relative}")
            continue
        current_digest = _file_sha256(candidate)
        if current_digest != recorded_digest:
            issues.append(
                f"SHA-256 mismatch: {relative} "
                f"(recorded {recorded_digest}, current {current_digest})"
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproduction_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--profile",
        choices=["all", "paper1"],
        default="all",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the existing --output manifest instead of rewriting it",
    )
    args = parser.parse_args()
    selected_files = PAPER1_HASHED_FILES if args.profile == "paper1" else None
    if args.verify_existing:
        expected_files = PAPER1_HASHED_FILES if args.profile == "paper1" else HASHED_FILES
        issues = verify_manifest_file(
            args.output,
            expected_files=expected_files,
        )
        if issues:
            for issue in issues:
                print(f"ERROR: {issue}", file=sys.stderr)
            raise SystemExit(1)
        print(f"verified {args.output.resolve()}")
        return

    files_to_hash = PAPER1_HASHED_FILES if args.profile == "paper1" else HASHED_FILES
    missing_files = [relative for relative in files_to_hash if not (ROOT / relative).is_file()]
    if missing_files:
        for relative in missing_files:
            print(f"ERROR: required hashed file is missing: {relative}", file=sys.stderr)
        raise SystemExit(1)
    manifest = collect_manifest(
        random_seed=args.seed,
        hashed_files=selected_files,
    )
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
