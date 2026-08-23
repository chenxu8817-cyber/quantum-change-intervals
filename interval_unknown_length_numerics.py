"""Finite-size checks for unknown-length quantum interval localization.

The hypotheses are all nonempty intervals I=[a,b] in {1,...,n}, with
uniform prior.  Their Gram matrix is G[I,J] = c**|I symmetric_difference J|.

The script reports
  * the SRM success probability,
  * the Gram spectral lower bound,
  * the Sentis-et-al. Helstrom upper bound,
  * the exact finite-dimensional minimum-error value from the SDP dual,
  * the conjectured/proved asymptotic limit p1(c)**2.

Dependencies are kept in the workspace-local .local_pydeps directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".local_pydeps"
if os.environ.get("QCI_USE_LOCAL_DEPS") == "1" and LOCAL_DEPS.exists():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(".local_pydeps is a CPython 3.12 cache")
    sys.path.insert(0, str(LOCAL_DEPS))

import cvxpy as cp
import numpy as np
from scipy.special import ellipk

from sdp_certification import certify_minimum_error


def intervals(n: int) -> list[tuple[int, int]]:
    """All nonempty inclusive intervals, ordered lexicographically."""
    return [(a, b) for a in range(n) for b in range(a, n)]


def interval_gram(n: int, c: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Exact Gram matrix G[I,J] = c**|I symmetric_difference J|."""
    ints = intervals(n)
    endpoints = np.asarray(ints, dtype=int)
    left = endpoints[:, 0]
    right = endpoints[:, 1]
    lengths = right - left + 1

    intersection = np.maximum(
        0,
        np.minimum(right[:, None], right[None, :])
        - np.maximum(left[:, None], left[None, :])
        + 1,
    )
    exponent = lengths[:, None] + lengths[None, :] - 2 * intersection
    gram = np.power(float(c), exponent, dtype=float)
    return ints, gram


def endpoint_product_gram(
    ints: list[tuple[int, int]], c: float
) -> np.ndarray:
    """The comparison 2D Toeplitz kernel c**(|a-a'|+|b-b'|)."""
    endpoints = np.asarray(ints, dtype=int)
    exponent = (
        np.abs(endpoints[:, None, 0] - endpoints[None, :, 0])
        + np.abs(endpoints[:, None, 1] - endpoints[None, :, 1])
    )
    return np.power(float(c), exponent, dtype=float)


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-9 * scale:
        raise ValueError(f"matrix is not PSD: lambda_min={np.min(eigenvalues):.3e}")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def gram_quantities(gram: np.ndarray) -> dict[str, float]:
    """SRM and the two Gram bounds for equal priors."""
    m = gram.shape[0]
    root = psd_sqrt(gram)
    diagonal = np.diag(root)
    root_trace = float(np.sum(diagonal))
    spectral_lower = (root_trace / m) ** 2
    srm = float(np.sum(diagonal**2) / m)
    q = diagonal / root_trace
    q_l1 = float(np.sum(np.abs(q - 1.0 / m)))
    lambda_max = float(np.linalg.eigvalsh(gram)[-1])
    helstrom_upper = spectral_lower + np.sqrt(lambda_max) * q_l1
    return {
        "spectral_lower": spectral_lower,
        "srm": srm,
        "q_l1": q_l1,
        "lambda_max": lambda_max,
        "helstrom_upper": min(1.0, helstrom_upper),
    }


def minimum_error_sdp(
    gram: np.ndarray, solver: str = "CLARABEL", verbose: bool = False
) -> tuple[float, str]:
    """Solve the Helstrom/Yuen-Kennedy-Lax dual in the Gram span.

    For equal priors, columns v_j of sqrt(G/m) are subnormalized canonical
    state vectors.  The dual is

        minimize Tr(Y),  subject to Y >= v_j v_j^T for every j.
    """
    m = gram.shape[0]
    root_weighted = psd_sqrt(gram) / np.sqrt(m)
    y = cp.Variable((m, m), symmetric=True)
    constraints = []
    for j in range(m):
        v = root_weighted[:, j]
        constraints.append(y - np.outer(v, v) >> 0)
    problem = cp.Problem(cp.Minimize(cp.trace(y)), constraints)

    if solver.upper() == "CLARABEL":
        try:
            value = problem.solve(
                solver="CLARABEL",
                tol_gap_abs=2e-8,
                tol_gap_rel=2e-8,
                tol_feas=2e-8,
                max_iter=500,
                verbose=verbose,
            )
        except cp.error.SolverError:
            value = problem.solve(
                solver="SCS",
                eps=2e-6,
                max_iters=100_000,
                verbose=verbose,
            )
    else:
        value = problem.solve(
            solver="SCS",
            eps=2e-6,
            max_iters=100_000,
            verbose=verbose,
        )

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP failed with status {problem.status}")
    return float(value), str(problem.status)


def p1(c: float) -> float:
    """One-change-point asymptotic success probability."""
    if c <= 0:
        return 1.0
    if c >= 1:
        return 0.0
    return float(4 * (1 - c * c) * ellipk(c * c) ** 2 / np.pi**2)


def run_case(n: int, c: float, solver: str, verbose: bool) -> dict[str, object]:
    ints, gram = interval_gram(n, c)
    product = endpoint_product_gram(ints, c)
    quantities = gram_quantities(gram)
    product_quantities = gram_quantities(product)
    certificate = certify_minimum_error(
        gram,
        solver=solver,
        verbose=verbose,
    )
    limit = p1(c) ** 2
    return {
        "n": n,
        "M": len(ints),
        "c": c,
        "limit_p1_squared": limit,
        **quantities,
        **certificate,
        # The postprocessed primal and dual objectives are genuine feasible
        # lower and upper bounds.  Keep ``sdp_opt`` only as a legacy alias for
        # the conservative upper endpoint; figures and prose use the interval.
        "sdp_lower": certificate["primal_feasible_objective"],
        "sdp_upper": certificate["dual_feasible_objective"],
        "sdp_opt": certificate["dual_feasible_objective"],
        "sdp_status": certificate["dual_status"],
        "sdp_lower_minus_srm": (
            float(certificate["primal_feasible_objective"]) - quantities["srm"]
        ),
        "sdp_upper_minus_srm": (
            float(certificate["dual_feasible_objective"]) - quantities["srm"]
        ),
        "sdp_minus_srm": (
            float(certificate["dual_feasible_objective"]) - quantities["srm"]
        ),
        "sdp_minus_limit": (
            float(certificate["dual_feasible_objective"]) - limit
        ),
        "product_srm": product_quantities["srm"],
        "product_spectral_lower": product_quantities["spectral_lower"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--c", type=float, nargs="+", default=[0.3, 0.6, 0.8])
    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "unknown_length_sdp_srm.csv"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.n_min < 1 or args.n_max < args.n_min:
        raise ValueError("require 1 <= n-min <= n-max")
    if any(c < 0 or c > 1 for c in args.c):
        raise ValueError("all overlaps c must lie in [0,1]")

    rows: list[dict[str, object]] = []
    for c in args.c:
        for n in range(args.n_min, args.n_max + 1):
            row = run_case(n, c, args.solver, args.verbose)
            rows.append(row)
            print(
                "n={n:2d} M={M:3d} c={c:.2f}  limit={limit_p1_squared:.8f} "
                "SRM={srm:.8f} primal={primal_objective:.8f} "
                "dual={dual_objective:.8f} pd_gap={absolute_gap:.2e} "
                "safe=[{sdp_lower:.8f},{sdp_upper:.8f}] "
                "rP={primal_equality_residual_fro:.1e} "
                "rD={dual_psd_violation:.1e}".format(**row),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
