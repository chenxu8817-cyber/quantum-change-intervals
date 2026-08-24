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
from pathlib import Path


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
    "pyproject.toml",
    "requirements-lock.txt",
    "paper1/README.md",
    "paper1/PROOF_AUDIT.md",
    "paper1/REPRODUCING.md",
    "paper1/NUMERICAL_REPRODUCIBILITY.md",
    "paper1/PUBLIC_RELEASE_CHECKLIST.md",
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
    "tests/test_paper1_figures.py",
    "tests/test_quantum_interval_numerics.py",
    "tests/test_sdp_certification.py",
    "tests/test_srm_scaling.py",
    "certified_sdp_results.csv",
    "srm_scaling_m1.csv",
    "paper1/paper1_fixed_and_growing_srm.csv",
    "paper1/paper1_h0_certified_sdp.csv",
    "paper1/reproduced/certified_sdp_results.csv",
    "paper1/reproduced/srm_scaling_m1.csv",
    "paper1/figures/figure1_model_geometry.pdf",
    "paper1/figures/figure1_model_geometry.png",
    "paper1/figures/figure2_analytic_limits.pdf",
    "paper1/figures/figure2_analytic_limits.png",
    "paper1/figures/figure3_finite_size.pdf",
    "paper1/figures/figure3_finite_size.png",
    "paper1/quantum_submission/main.tex",
    "paper1/quantum_submission/content.tex",
    "paper1/quantum_submission/references.bib",
    "paper1/quantum_submission/quantumarticle.cls",
    "paper1/quantum_submission/quantum.bst",
    "paper1/quantum_submission/main.bbl",
    "paper1/quantum_submission/main.pdf",
    "paper1/quantum_submission/README.md",
    "paper1/quantum_submission/figures/figure1_model_geometry.pdf",
    "paper1/quantum_submission/figures/figure1_model_geometry.png",
    "paper1/quantum_submission/figures/figure2_analytic_limits.pdf",
    "paper1/quantum_submission/figures/figure2_analytic_limits.png",
    "paper1/quantum_submission/figures/figure3_finite_size.pdf",
    "paper1/quantum_submission/figures/figure3_finite_size.png",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "command": command if command is not None else sys.argv,
        "random_seed": random_seed,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
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
        "numpy_configuration": numpy_config.getvalue(),
        "file_sha256": hashes,
    }


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
    args = parser.parse_args()
    selected_files = PAPER1_HASHED_FILES if args.profile == "paper1" else None
    manifest = collect_manifest(
        random_seed=args.seed,
        hashed_files=selected_files,
    )
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
