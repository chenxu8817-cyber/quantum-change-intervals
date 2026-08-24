"""Numerical checks for the two-unknown-interval theorem.

Candidates are four boundaries 0 <= x1 < x2 < x3 < x4 <= n and the
anomalous set is [x1,x2) union [x3,x4). The script:

1. exhaustively tests the endpoint-matching dichotomy used in Lemma E.1;
2. constructs the exact physical Gram matrix c**|S_x symmetric-difference S_y|;
3. computes the full-candidate SRM and normalized square-root-trace lower bound;
4. compares both quantities with the conjectured/proved limit p1(c)**4.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


def endpoint_candidates(n: int) -> list[tuple[int, int, int, int]]:
    return list(itertools.combinations(range(n + 1), 4))


def incidence_matrix(
    n: int, points: list[tuple[int, int, int, int]]
) -> np.ndarray:
    incidence = np.zeros((len(points), n), dtype=np.int16)
    for row, (x1, x2, x3, x4) in enumerate(points):
        incidence[row, x1:x2] = 1
        incidence[row, x3:x4] = 1
    return incidence


def exact_gram(n: int, overlap: float) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    points = endpoint_candidates(n)
    incidence = incidence_matrix(n, points)
    weights = incidence.sum(axis=1)
    symmetric_difference = (
        weights[:, None] + weights[None, :] - 2 * incidence @ incidence.T
    )
    return overlap**symmetric_difference, points


def srm_data(gram: np.ndarray) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-9:
        raise ValueError(f"Gram matrix is not PSD: {eigenvalues.min()}")
    root = (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    ) @ eigenvectors.T
    diagonal = np.diag(root)
    success = float(np.mean(diagonal * diagonal))
    trace_lower = float((np.trace(root) / len(gram)) ** 2)

    q = diagonal / np.trace(root)
    uniform = np.full(len(gram), 1.0 / len(gram))
    helstrom_upper = trace_lower + math.sqrt(float(eigenvalues.max())) * float(
        np.abs(q - uniform).sum()
    )
    return success, trace_lower, helstrom_upper


def p1(q: float, grid_size: int = 200_000) -> float:
    if q == 0.0:
        return 1.0
    if q == 1.0:
        return 0.0
    theta = 2.0 * math.pi * np.arange(grid_size) / grid_size
    symbol = (1.0 - q * q) / (
        1.0 - 2.0 * q * np.cos(theta) + q * q
    )
    alpha = float(np.mean(np.sqrt(symbol)))
    return alpha * alpha


def interval_mask(x: tuple[int, int, int, int]) -> int:
    x1, x2, x3, x4 = x
    left = ((1 << (x2 - x1)) - 1) << x1
    right = ((1 << (x4 - x3)) - 1) << x3
    return left | right


def validate_endpoint_dichotomy(max_n: int) -> None:
    """Exhaustively verify: D != E implies D >= L, for small sizes."""
    for n in range(4, max_n + 1):
        all_points = endpoint_candidates(n)
        for length in range(1, n // 3 + 1):
            points = [
                x
                for x in all_points
                if min(x[k + 1] - x[k] for k in range(3)) >= length
            ]
            masks = [interval_mask(x) for x in points]
            for row, x in enumerate(points):
                for col, y in enumerate(points):
                    symmetric_difference = (masks[row] ^ masks[col]).bit_count()
                    endpoint_distance = sum(
                        abs(x[k] - y[k]) for k in range(4)
                    )
                    if symmetric_difference > endpoint_distance:
                        raise AssertionError(
                            f"D <= E failed: n={n}, L={length}, x={x}, y={y}"
                        )
                    if (
                        symmetric_difference != endpoint_distance
                        and symmetric_difference < length
                    ):
                        raise AssertionError(
                            "endpoint dichotomy failed: "
                            f"n={n}, L={length}, x={x}, y={y}, "
                            f"D={symmetric_difference}, E={endpoint_distance}"
                        )


def run_case(n: int, overlap: float) -> dict[str, float | int]:
    gram, points = exact_gram(n, overlap)
    srm, trace_lower, helstrom_upper = srm_data(gram)
    target = p1(overlap) ** 4
    return {
        "c": overlap,
        "n": n,
        "M": len(points),
        "target_p1_c_pow4": target,
        "full_srm": srm,
        "trace_lower_bound": trace_lower,
        "helstrom_gram_upper": helstrom_upper,
        "full_srm_minus_target": srm - target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--c", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--validate-through", type=int, default=13)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("two_unknown_intervals_numerics.csv"),
    )
    args = parser.parse_args()

    validate_endpoint_dichotomy(args.validate_through)
    rows = [run_case(n, overlap) for overlap in args.c for n in args.n]

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "c={c:.2f} n={n:>2d} M={M:>4d} "
            "target={target_p1_c_pow4:.9f} SRM={full_srm:.9f} "
            "traceLB={trace_lower_bound:.9f} "
            "HelstromUB={helstrom_gram_upper:.9f}".format(**row)
        )
    print(f"wrote {args.csv.resolve()}")


if __name__ == "__main__":
    main()
