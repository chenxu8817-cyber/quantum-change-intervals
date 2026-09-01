"""Generate Paper I fixed-length, growing-length, and H0 checks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from paper1_analytics import (
    fixed_length_gram,
    fixed_length_limit_quadrature,
)
from quantum_interval_numerics import p1, srm_quantities
from sdp_certification import certify_minimum_error


ROOT = Path(__file__).resolve().parent


def fixed_and_growing_rows(overlaps: list[float]) -> list[dict[str, object]]:
    """Return exact SRM rows for fixed and growing known lengths."""
    rows: list[dict[str, object]] = []
    for overlap in overlaps:
        r = overlap**2
        for candidate_count in (10, 20, 40, 80):
            schedules = {
                "fixed_i2": 2,
                "sqrt_growth": max(1, math.ceil(math.sqrt(candidate_count))),
                "balanced_growth": candidate_count // 2,
            }
            for schedule, length in schedules.items():
                gram = fixed_length_gram(candidate_count, length, r)
                quantities = srm_quantities(gram)
                target = (
                    fixed_length_limit_quadrature(length, r)
                    if schedule == "fixed_i2"
                    else p1(r)
                )
                rows.append(
                    {
                        "schedule": schedule,
                        "c": overlap,
                        "r": r,
                        "N": candidate_count,
                        "i": length,
                        "n": candidate_count + length - 1,
                        "target": target,
                        "srm": quantities["srm"],
                        "trace_lower_bound": quantities["trace_lower_bound"],
                        "helstrom_upper": quantities["helstrom_upper"],
                        "srm_minus_target": float(quantities["srm"]) - target,
                    }
                )
    return rows


def h0_gram(candidate_count: int, length: int, overlap: float) -> np.ndarray:
    """Unweighted Gram for H0 plus a known-length interval family."""
    r = overlap**2
    anomaly = fixed_length_gram(candidate_count, length, r)
    gram = np.ones((candidate_count + 1, candidate_count + 1))
    gram[1:, 1:] = anomaly
    gram[0, 1:] = overlap**length
    gram[1:, 0] = overlap**length
    return gram


def h0_rows(
    overlaps: list[float], solver: str = "CLARABEL"
) -> list[dict[str, object]]:
    """Return safeguarded floating-point H0 detection-localization rows."""
    rows: list[dict[str, object]] = []
    for overlap in overlaps:
        for prior_h0 in (0.2, 0.5):
            for candidate_count in (3, 5):
                for length in (1, 2):
                    gram = h0_gram(candidate_count, length, overlap)
                    priors = np.array(
                        [prior_h0]
                        + [
                            (1.0 - prior_h0) / candidate_count
                            for _ in range(candidate_count)
                        ]
                    )
                    certificate = certify_minimum_error(
                        gram,
                        solver=solver,
                        priors=priors,
                    )
                    rows.append(
                        {
                            "c": overlap,
                            "pi0": prior_h0,
                            "N": candidate_count,
                            "i": length,
                            **certificate,
                        }
                    )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, nargs="+", default=[0.5, 0.8, 0.95])
    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument(
        "--fixed-output",
        type=Path,
        default=ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv",
    )
    parser.add_argument(
        "--h0-output",
        type=Path,
        default=ROOT / "paper1" / "paper1_h0_certified_sdp.csv",
    )
    args = parser.parse_args()
    if any(not 0.0 <= overlap <= 1.0 for overlap in args.c):
        raise ValueError("all overlaps must lie in [0,1]")
    fixed_rows = fixed_and_growing_rows(args.c)
    h0_certificate_rows = h0_rows(args.c, solver=args.solver)
    write_rows(args.fixed_output, fixed_rows)
    write_rows(args.h0_output, h0_certificate_rows)
    print(f"wrote {args.fixed_output} ({len(fixed_rows)} rows)")
    print(f"wrote {args.h0_output} ({len(h0_certificate_rows)} rows)")


if __name__ == "__main__":
    main()
