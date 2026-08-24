"""Small-size primal/dual SDP certificates for fixed-m interval models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from quantum_interval_numerics import (
    exact_interval_gram,
    p1,
    srm_quantities,
)
from sdp_certification import certify_minimum_error


ROOT = Path(__file__).resolve().parent


def certify_interval_case(
    n: int,
    interval_count: int,
    overlap: float,
    solver: str = "CLARABEL",
    max_candidates: int = 40,
) -> dict[str, float | int | str]:
    gram, points = exact_interval_gram(n, interval_count, overlap)
    if len(points) > max_candidates:
        raise MemoryError(
            "SDP candidate count exceeds configured certification limit: "
            f"{len(points)} > {max_candidates}"
        )
    quantities = srm_quantities(gram)
    certificate = certify_minimum_error(gram, solver=solver)
    return {
        "m": interval_count,
        "n": n,
        "candidate_count": len(points),
        "c": overlap,
        "target_p1_power_2m": p1(overlap) ** (2 * interval_count),
        **quantities,
        **certificate,
        "primal_minus_srm": (
            float(certificate["primal_objective"])
            - float(quantities["srm"])
        ),
        "dual_minus_srm": (
            float(certificate["dual_objective"])
            - float(quantities["srm"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument(
        "--c", type=float, nargs="+", default=[0.5, 0.9, 0.99]
    )
    parser.add_argument(
        "--solver", choices=["CLARABEL", "SCS"], default="CLARABEL"
    )
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "certified_fixed_m_sdp_results.csv",
    )
    args = parser.parse_args()
    rows = []
    for overlap in args.c:
        for n in args.n:
            row = certify_interval_case(
                n=n,
                interval_count=args.m,
                overlap=overlap,
                solver=args.solver,
                max_candidates=args.max_candidates,
            )
            rows.append(row)
            print(
                "m={m} n={n} M={candidate_count} c={c:.3f} "
                "SRM={srm:.9f} primal={primal_objective:.9f} "
                "dual={dual_objective:.9f} gap={absolute_gap:.2e}".format(
                    **row
                ),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
