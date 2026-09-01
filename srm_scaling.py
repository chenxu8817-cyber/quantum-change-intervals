"""Resource-aware exact dense SRM scaling experiments.

This script deliberately refuses infeasible dense Gram allocations.  It is for
exact medium-size calculations; matrix-free approximations must use a separate
method label and are not silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from quantum_interval_numerics import (
    exact_interval_gram,
    p1,
    srm_quantities,
)


ROOT = Path(__file__).resolve().parent


def estimate_dense_resources(
    n: int, interval_count: int
) -> dict[str, int]:
    candidate_count = math.comb(n + 1, 2 * interval_count)
    gram_bytes = 8 * candidate_count**2
    # Gram, eigenvectors, square root, and eigensolver workspace.
    estimated_peak_bytes = 4 * gram_bytes
    return {
        "candidate_count": candidate_count,
        "gram_bytes": gram_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def _correlation_length(overlap: float) -> float:
    if overlap <= 0.0:
        return 0.0
    if overlap >= 1.0:
        return math.inf
    return 1.0 / abs(math.log(overlap))


def run_exact_case(
    n: int,
    interval_count: int,
    overlap: float,
    max_gram_bytes: int,
) -> dict[str, float | int | str]:
    estimate = estimate_dense_resources(n, interval_count)
    if estimate["gram_bytes"] > max_gram_bytes:
        gib = estimate["gram_bytes"] / 2**30
        limit_gib = max_gram_bytes / 2**30
        raise MemoryError(
            "dense Gram estimate exceeds configured limit: "
            f"{gib:.2f} GiB > {limit_gib:.2f} GiB "
            f"(m={interval_count}, n={n}, "
            f"M={estimate['candidate_count']})"
        )
    started = time.perf_counter()
    gram, points = exact_interval_gram(n, interval_count, overlap)
    quantities = srm_quantities(gram)
    elapsed = time.perf_counter() - started
    correlation_length = _correlation_length(overlap)
    scaled_length = (
        math.inf
        if correlation_length == 0.0
        else (0.0 if math.isinf(correlation_length) else n / correlation_length)
    )
    return {
        "method": "exact_dense_eigh",
        "m": interval_count,
        "n": n,
        "candidate_count": len(points),
        "c": overlap,
        "target_p1_power_2m": p1(overlap) ** (2 * interval_count),
        "correlation_length": correlation_length,
        "n_over_correlation_length": scaled_length,
        **estimate,
        **quantities,
        "srm_minus_target": (
            float(quantities["srm"])
            - p1(overlap) ** (2 * interval_count)
        ),
        "runtime_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument(
        "--c",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.9, 0.95, 0.99],
    )
    parser.add_argument(
        "--max-gram-gib",
        type=float,
        default=2.0,
        help="Refuse cases whose Gram alone exceeds this size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "srm_scaling_results.csv",
    )
    args = parser.parse_args()
    if args.m < 1:
        raise ValueError("m must be positive")
    if any(n < 2 * args.m - 1 for n in args.n):
        raise ValueError("every n must satisfy n >= 2m-1")
    if any(not 0.0 <= overlap <= 1.0 for overlap in args.c):
        raise ValueError("every c must lie in [0,1]")

    max_gram_bytes = int(args.max_gram_gib * 2**30)
    rows = []
    for overlap in args.c:
        for n in args.n:
            row = run_exact_case(
                n=n,
                interval_count=args.m,
                overlap=overlap,
                max_gram_bytes=max_gram_bytes,
            )
            rows.append(row)
            print(
                "m={m} n={n:>3d} M={candidate_count:>6d} c={c:.3f} "
                "SRM={srm:.9f} target={target_p1_power_2m:.9f} "
                "n/xi={n_over_correlation_length:.3f} "
                "time={runtime_seconds:.2f}s".format(**row),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
