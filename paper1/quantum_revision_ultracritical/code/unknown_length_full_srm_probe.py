"""Diagnostics for local square-root recombination in the critical interval model.

This script is exploratory support for ``unknown_length_full_srm_attempt.md``.
It compares the diagonal of the square root of the full Gram matrix with the
sum of the square-root diagonals of the 0-, 1-, and >=2-hull sectors.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def interval_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    labels = [(a, b) for a in range(n) for b in range(a, n)]
    return (
        np.asarray([a for a, _ in labels], dtype=np.int64),
        np.asarray([b for _, b in labels], dtype=np.int64),
    )


def root_diagonal(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    values, vectors = eigh(matrix, check_finite=False, overwrite_a=True)
    roots = np.sqrt(np.maximum(values, 0.0))
    diagonal = np.square(vectors) @ roots
    return diagonal, float(np.sum(roots))


def sector_grams(n: int, c: float) -> tuple[np.ndarray, ...]:
    a, b = interval_arrays(n)
    lengths = b - a + 1
    left = np.maximum(a[:, None], a[None, :])
    right = np.minimum(b[:, None], b[None, :])
    overlap = np.maximum(0, right - left + 1)
    s2 = 1.0 - c * c

    common = np.power(
        c, lengths[:, None] + lengths[None, :], dtype=np.float64
    )
    x0 = common
    x1 = common * (s2 / (c * c)) * overlap

    # The exact >=2 hull-compressed sector is the full Gram remainder.
    symmetric_difference = (
        lengths[:, None] + lengths[None, :] - 2 * overlap
    )
    full = np.power(c, symmetric_difference, dtype=np.float64)
    xge2 = full - x0 - x1
    xge2 = (xge2 + xge2.T) / 2.0
    return x0, x1, xge2, full


def one_row(n: int, tau: float) -> dict[str, float]:
    m = n * (n + 1) // 2
    c = 1.0 - tau / (n * math.log(n) ** 2)
    x0, x1, x2, full = sector_grams(n, c)
    d0, t0 = root_diagonal(x0)
    d1, t1 = root_diagonal(x1)
    d2, t2 = root_diagonal(x2)
    d, trace = root_diagonal(full)
    sector = d0 + d1 + d2
    scale = math.sqrt(m)
    difference = scale * (d - sector)
    eta = scale * d
    sector_eta = scale * sector
    target = 1.0 + 2.0 * math.sqrt(tau) / math.pi + (
        math.sqrt(2.0) * tau / math.pi**2
    )
    return {
        "n": float(n),
        "tau": tau,
        "trace_amplitude": trace / scale,
        "sector_trace_amplitude": (t0 + t1 + t2) / scale,
        "target": target,
        "min_full_minus_sector": float(np.min(difference)),
        "max_full_minus_sector": float(np.max(difference)),
        "rms_full_minus_sector": float(np.sqrt(np.mean(difference**2))),
        "mean_full_minus_sector": float(np.mean(difference)),
        "rms_full_target": float(np.sqrt(np.mean((eta - target) ** 2))),
        "rms_sector_target": float(
            np.sqrt(np.mean((sector_eta - target) ** 2))
        ),
        "max_full_eta": float(np.max(eta)),
        "max_sector_eta": float(np.max(sector_eta)),
    }


def one_excitation_state_matrix(n: int, c: float) -> np.ndarray:
    """Return physical rows x interval labels for the k=1 subensemble."""
    a, b = interval_arrays(n)
    sites = np.arange(n)[:, None]
    incidence = (a[None, :] <= sites) & (sites <= b[None, :])
    lengths = b - a + 1
    return (
        math.sqrt(1.0 - c * c)
        * incidence.astype(np.float64)
        * np.power(c, lengths - 1, dtype=np.float64)[None, :]
    )


def one_excitation_dual(n: int, tau: float) -> dict[str, float]:
    """Exploratory SCS dual for the one-excitation sector."""
    import cvxpy as cp

    m = n * (n + 1) // 2
    c = 1.0 - tau / (n * math.log(n) ** 2)
    states = one_excitation_state_matrix(n, c)
    y = cp.Variable((n, n), symmetric=True)
    constraints = [y >> 0]
    for index in range(m):
        vector = states[:, index]
        constraints.append(y - np.outer(vector, vector) / m >> 0)
    problem = cp.Problem(cp.Minimize(cp.trace(y)), constraints)
    value = problem.solve(
        solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, verbose=False
    )
    gram = states.T @ states
    diagonal, trace = root_diagonal(gram)
    p_srm = float(np.mean(diagonal**2))
    p_trace = float((trace / m) ** 2)
    return {
        "n": float(n),
        "tau": tau,
        "M_times_k1_opt": float(m * value),
        "M_times_k1_srm": float(m * p_srm),
        "M_times_k1_trace": float(m * p_trace),
        "k1_target": float((2.0 * math.sqrt(tau) / math.pi) ** 2),
    }


def complement_projection(*vectors: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(vectors).astype(np.float64)
    basis, _ = np.linalg.qr(matrix)
    rank = np.linalg.matrix_rank(matrix)
    basis = basis[:, :rank]
    return np.eye(matrix.shape[0]) - basis @ basis.T


def retained_orthogonal_diagnostics(
    n: int, tau: float, block_count: int | None = None
) -> dict[str, float]:
    """Check the exact row-orthogonal retained comparison at finite size."""
    a, b = interval_arrays(n)
    m = len(a)
    c = 1.0 - tau / (n * math.log(n) ** 2)
    q = math.sqrt(1.0 - c * c) / c
    sites = np.arange(n)
    b1 = (
        (a[:, None] <= sites[None, :])
        & (sites[None, :] <= b[:, None])
    ).astype(np.float64)
    pairs = [(s, t) for s in range(n) for t in range(s + 1, n)]
    b2 = np.asarray(
        [
            [(left <= s and t <= right) for s, t in pairs]
            for left, right in zip(a, b, strict=True)
        ],
        dtype=np.float64,
    )

    q1 = complement_projection(b1.T @ np.ones(m))
    if block_count is None:
        block_count = max(2, int(math.sqrt(math.log(n + 1))))
    blocks = [np.asarray(x, dtype=np.int64) for x in np.array_split(sites, block_count)]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    q2 = np.zeros((len(pairs), len(pairs)), dtype=np.float64)
    for p, left_block in enumerate(blocks):
        q_left = complement_projection(
            np.ones(len(left_block)), left_block.astype(np.float64) + 1.0
        )
        for right_block in blocks[p + 1 :]:
            q_right = complement_projection(
                np.ones(len(right_block)),
                (n - right_block).astype(np.float64),
            )
            indices = [
                pair_index[(int(s), int(t))]
                for s in left_block
                for t in right_block
            ]
            local = np.kron(q_left, q_right)
            q2[np.ix_(indices, indices)] = local

    x0 = np.ones((m, m))
    x1 = q * q * b1 @ b1.T
    x2 = q**4 * b2 @ b2.T
    retained1 = q * q * b1 @ q1 @ b1.T
    retained2 = q**4 * b2 @ q2 @ b2.T
    full = x0 + x1 + x2
    retained = x0 + retained1 + retained2
    d_full, trace_full = root_diagonal(full.copy())
    d_retained, trace_retained = root_diagonal(retained.copy())
    additive_basis = np.column_stack(
        [
            np.ones(m),
            *[(a == value).astype(float) for value in range(n)],
            *[(b == value).astype(float) for value in range(n)],
        ]
    )
    return {
        "retained_min_gram_slack": float(
            np.min(np.linalg.eigvalsh((full - retained + (full - retained).T) / 2))
        ),
        "retained_min_root_diagonal_slack": float(np.min(d_full - d_retained)),
        "retained_trace_gap_over_sqrt_M": float(
            (trace_full - trace_retained) / math.sqrt(m)
        ),
        "b1_vacuum_orthogonality": float(
            np.linalg.norm(np.ones(m) @ b1 @ q1)
        ),
        "b2_additive_orthogonality": float(
            np.linalg.norm(additive_basis.T @ b2 @ q2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[12, 18, 24, 32])
    parser.add_argument("--tau", nargs="+", type=float, default=[0.25, 1.0, 4.0])
    parser.add_argument("--sdp", action="store_true")
    parser.add_argument("--retained", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records: list[dict[str, float]] = []
    for n in args.n:
        for tau in args.tau:
            row = one_row(n, tau)
            record = dict(row)
            print(" ".join(f"{key}={value:.9g}" for key, value in row.items()))
            if args.sdp:
                sector = one_excitation_dual(n, tau)
                record.update(sector)
                print(
                    "k1_sdp "
                    + " ".join(
                        f"{key}={value:.9g}" for key, value in sector.items()
                    )
                )
            if args.retained:
                retained = retained_orthogonal_diagnostics(n, tau)
                record.update(retained)
                print(
                    "retained "
                    + " ".join(
                        f"{key}={value:.9g}"
                        for key, value in retained.items()
                    )
                )
            records.append(record)

    if args.output is not None and records:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(records[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {key: f"{value:.9g}" for key, value in record.items()}
                )


if __name__ == "__main__":
    main()
