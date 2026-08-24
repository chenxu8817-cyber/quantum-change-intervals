"""Floating-point primal/dual SDP verification for state discrimination.

Historical public API and output-field names retain 'certify' for backward
compatibility. The computations use conventional IEEE double precision, not
interval arithmetic or exact-rational certification.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".local_pydeps"
if os.environ.get("QCI_USE_LOCAL_DEPS") == "1" and LOCAL_DEPS.exists():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(".local_pydeps is a CPython 3.12 cache")
    sys.path.insert(0, str(LOCAL_DEPS))

import cvxpy as cp
import numpy as np


def canonical_weighted_states(
    gram: np.ndarray,
    rank_tolerance: float = 1e-12,
    priors: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Factor the prior-weighted Gram into canonical state columns."""
    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    hypothesis_count = gram.shape[0]
    if priors is None:
        prior_array = np.full(hypothesis_count, 1.0 / hypothesis_count)
    else:
        prior_array = np.asarray(priors, dtype=float)
        if prior_array.shape != (hypothesis_count,):
            raise ValueError("priors must have one entry per hypothesis")
        if np.any(prior_array < 0.0):
            raise ValueError("priors must be nonnegative")
        if not np.isclose(float(prior_array.sum()), 1.0, atol=1e-12):
            raise ValueError("priors must sum to one")
    raw_symmetrized = (gram + gram.T) / 2
    raw_eigenvalues = np.linalg.eigvalsh(raw_symmetrized)
    weights = np.sqrt(prior_array)
    weighted_gram = weights[:, None] * gram * weights[None, :]
    symmetrized = (weighted_gram + weighted_gram.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
    lambda_max = float(eigenvalues[-1])
    threshold = rank_tolerance * max(1.0, lambda_max)
    if float(eigenvalues[0]) < -100.0 * threshold:
        raise ValueError(
            f"gram is not PSD: lambda_min={float(eigenvalues[0]):.6e}"
        )
    keep = eigenvalues > threshold
    states = np.sqrt(eigenvalues[keep])[:, None] * eigenvectors[:, keep].T
    metadata: dict[str, float | int] = {
        "rank": int(np.count_nonzero(keep)),
        "hypothesis_count": hypothesis_count,
        "rank_threshold": threshold,
        # Preserve the historical meaning of these public fields: they
        # describe the unweighted physical Gram matrix.
        "gram_lambda_min": float(raw_eigenvalues[0]),
        "gram_lambda_max": float(raw_eigenvalues[-1]),
        "weighted_gram_lambda_min": float(eigenvalues[0]),
        "weighted_gram_lambda_max": lambda_max,
        "prior_min": float(prior_array.min()),
        "prior_max": float(prior_array.max()),
    }
    return states, metadata


def _solve_options(solver: str, verbose: bool) -> dict[str, object]:
    if solver.upper() == "CLARABEL":
        return {
            "solver": "CLARABEL",
            "tol_gap_abs": 2e-9,
            "tol_gap_rel": 2e-9,
            "tol_feas": 2e-9,
            "max_iter": 1000,
            "verbose": verbose,
        }
    if solver.upper() == "SCS":
        return {
            "solver": "SCS",
            "eps": 2e-7,
            "max_iters": 200_000,
            "verbose": verbose,
        }
    raise ValueError(f"unsupported solver: {solver}")


def _solve_primal(
    states: np.ndarray, solver: str, verbose: bool
) -> tuple[list[np.ndarray], str, int, float]:
    rank, hypothesis_count = states.shape
    measurements = [
        cp.Variable((rank, rank), symmetric=True)
        for _ in range(hypothesis_count)
    ]
    constraints = [measurement >> 0 for measurement in measurements]
    constraints.append(sum(measurements) == np.eye(rank))
    objective_terms = []
    for column, measurement in enumerate(measurements):
        rho = np.outer(states[:, column], states[:, column])
        objective_terms.append(cp.sum(cp.multiply(rho, measurement)))
    problem = cp.Problem(cp.Maximize(sum(objective_terms)), constraints)
    started = time.perf_counter()
    problem.solve(**_solve_options(solver, verbose))
    elapsed = time.perf_counter() - started
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"primal SDP failed with status {problem.status}")
    values = [
        np.asarray(measurement.value, dtype=float)
        for measurement in measurements
    ]
    iterations = int(problem.solver_stats.num_iters or 0)
    return values, str(problem.status), iterations, elapsed


def _solve_dual(
    states: np.ndarray, solver: str, verbose: bool
) -> tuple[np.ndarray, str, int, float]:
    rank, hypothesis_count = states.shape
    dual = cp.Variable((rank, rank), symmetric=True)
    constraints = []
    for column in range(hypothesis_count):
        rho = np.outer(states[:, column], states[:, column])
        constraints.append(dual - rho >> 0)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(np.eye(rank), dual))),
        constraints,
    )
    started = time.perf_counter()
    problem.solve(**_solve_options(solver, verbose))
    elapsed = time.perf_counter() - started
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"dual SDP failed with status {problem.status}")
    iterations = int(problem.solver_stats.num_iters or 0)
    return (
        np.asarray(dual.value, dtype=float),
        str(problem.status),
        iterations,
        elapsed,
    )


def _negative_eigenvalue_violation(matrix: np.ndarray) -> float:
    matrix = (matrix + matrix.T) / 2
    return max(0.0, -float(np.linalg.eigvalsh(matrix)[0]))


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    """Return the real symmetric part used by every spectral repair."""
    return (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2


def _roundoff_margin(
    matrices: list[np.ndarray] | tuple[np.ndarray, ...],
) -> float:
    """A scale-aware margin for rechecking finite-precision PSD constraints."""
    dimension = max(matrix.shape[0] for matrix in matrices)
    scale = max(
        1.0,
        *(float(np.linalg.norm(matrix, ord=2)) for matrix in matrices),
    )
    return 256.0 * np.finfo(float).eps * max(1, dimension) * scale


def _safe_dual_bound(
    dual: np.ndarray,
    rhos: list[np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    """Shift a numerical dual point toward strict dual feasibility.

    If ``delta`` is the largest negative-slack violation, then
    ``dual + (delta + margin) I`` dominates every weighted state.  The final
    floating-point recheck diagnoses the eigensolver roundoff used to estimate
    delta; it is not a directed-rounding proof of feasibility.
    """
    dual_symmetric = _symmetric(dual)
    slacks = [_symmetric(dual_symmetric - rho) for rho in rhos]
    margin = _roundoff_margin([dual_symmetric, *rhos, *slacks])
    shift = max(_negative_eigenvalue_violation(slack) for slack in slacks)
    shift += margin
    identity = np.eye(dual_symmetric.shape[0])
    dual_safe = dual_symmetric + shift * identity

    minimum_slack = min(
        float(np.linalg.eigvalsh(_symmetric(dual_safe - rho))[0])
        for rho in rhos
    )
    if minimum_slack < margin / 2.0:
        correction = margin - minimum_slack
        shift += correction
        dual_safe = dual_safe + correction * identity
        minimum_slack = min(
            float(np.linalg.eigvalsh(_symmetric(dual_safe - rho))[0])
            for rho in rhos
        )

    violation = max(
        _negative_eigenvalue_violation(dual_safe - rho) for rho in rhos
    )
    return dual_safe, shift, minimum_slack, violation


def _psd_factor(matrix: np.ndarray) -> np.ndarray:
    """Return B with B B^T equal to the PSD-cone projection of matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(matrix))
    positive = eigenvalues > 0.0
    if not np.any(positive):
        return np.zeros((matrix.shape[0], 0), dtype=float)
    return eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])[None, :]


def _safe_primal_bound(
    measurements: list[np.ndarray],
    rhos: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Repair numerical POVM matrices into a feasible finite-precision POVM.

    The raw matrices are projected onto the PSD cone.  With
    ``S = sum_j M_j^+``, congruence by ``S^{-1/2}`` gives a POVM in exact
    arithmetic.  We retain Gram factors throughout so that each normalized
    element is manifestly PSD.  A final common contraction plus a PSD remainder
    assigned to the first outcome makes the floating-point completeness check
    conservative as well.
    """
    rank = measurements[0].shape[0]
    factors = [_psd_factor(measurement) for measurement in measurements]
    projected = [factor @ factor.T for factor in factors]
    total = _symmetric(sum(projected, np.zeros((rank, rank), dtype=float)))
    total_eigenvalues, total_eigenvectors = np.linalg.eigh(total)
    margin = _roundoff_margin([total, *projected, *rhos])
    if float(total_eigenvalues[0]) <= margin:
        raise RuntimeError(
            "PSD-projected primal sum is not numerically positive definite; "
            "the solver output cannot be safely normalized"
        )
    inverse_square_root = (
        total_eigenvectors
        * (1.0 / np.sqrt(total_eigenvalues))[None, :]
    ) @ total_eigenvectors.T

    normalized_factors = [inverse_square_root @ factor for factor in factors]
    normalized = [factor @ factor.T for factor in normalized_factors]
    identity = np.eye(rank)
    # A positive floor converts the theoretical PSD Gram products into
    # matrices whose rechecked floating-point eigenvalues are strictly
    # positive as well.  The subsequent common normalization absorbs it.
    normalized = [_symmetric(matrix) + margin * identity for matrix in normalized]
    normalized_total = _symmetric(
        sum(normalized, np.zeros((rank, rank), dtype=float))
    )

    # gamma * normalized_total <= I with a strict roundoff margin.  Hence the
    # remainder is PSD and may be assigned to any outcome without lowering
    # feasibility.  The first outcome is used deterministically.
    spectral_radius = float(np.linalg.eigvalsh(normalized_total)[-1])
    denominator = max(1.0, spectral_radius) + margin
    contraction = 1.0 / denominator
    remainder = _symmetric(identity - contraction * normalized_total)
    repaired = [contraction * matrix for matrix in normalized]
    repaired[0] = _symmetric(repaired[0] + remainder)

    completeness = sum(repaired, np.zeros((rank, rank), dtype=float)) - np.eye(rank)
    equality_fro = float(np.linalg.norm(completeness, ord="fro"))
    equality_op = float(np.linalg.norm(completeness, ord=2))
    psd_violation = max(
        _negative_eigenvalue_violation(measurement) for measurement in repaired
    )
    minimum_eigenvalue = min(
        float(np.linalg.eigvalsh(_symmetric(measurement))[0])
        for measurement in repaired
    )
    objective = float(
        sum(
            np.trace(rho @ measurement)
            for rho, measurement in zip(rhos, repaired)
        )
    )
    return repaired, {
        "objective": objective,
        "contraction": contraction,
        "regularization_floor": margin,
        "equality_fro": equality_fro,
        "equality_op": equality_op,
        "psd_violation": psd_violation,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def certify_minimum_error(
    gram: np.ndarray,
    solver: str = "CLARABEL",
    rank_tolerance: float = 1e-12,
    verbose: bool = False,
    priors: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Solve both SDPs and independently recompute floating-point residuals."""
    states, metadata = canonical_weighted_states(
        gram, rank_tolerance=rank_tolerance, priors=priors
    )
    rank, hypothesis_count = states.shape
    measurements, primal_status, primal_iterations, primal_time = (
        _solve_primal(states, solver=solver, verbose=verbose)
    )
    dual, dual_status, dual_iterations, dual_time = _solve_dual(
        states, solver=solver, verbose=verbose
    )

    rhos = [
        np.outer(states[:, column], states[:, column])
        for column in range(hypothesis_count)
    ]
    primal_objective = float(
        sum(
            np.trace(rho @ measurement)
            for rho, measurement in zip(rhos, measurements)
        )
    )
    dual_objective = float(np.trace(dual))
    signed_gap = dual_objective - primal_objective
    absolute_gap = abs(signed_gap)
    relative_signed_gap = signed_gap / max(
        1e-15,
        0.5 * (abs(primal_objective) + abs(dual_objective)),
    )
    relative_gap = abs(relative_signed_gap)

    completeness = sum(measurements) - np.eye(rank)
    primal_equality_fro_raw = float(np.linalg.norm(completeness, ord="fro"))
    primal_equality_fro = float(
        primal_equality_fro_raw / (1.0 + math.sqrt(rank))
    )
    primal_equality_op = float(np.linalg.norm(completeness, ord=2))
    primal_psd_violation = max(
        _negative_eigenvalue_violation(measurement)
        for measurement in measurements
    )
    slacks = [dual - rho for rho in rhos]
    dual_psd_violation = max(
        _negative_eigenvalue_violation(slack) for slack in slacks
    )
    complementarity = float(
        sum(
            abs(float(np.trace(measurement @ slack)))
            for measurement, slack in zip(measurements, slacks)
        )
    )

    repaired_measurements, primal_safe = _safe_primal_bound(measurements, rhos)
    primal_feasible_objective = primal_safe["objective"]
    primal_safety_contraction = primal_safe["contraction"]
    primal_feasible_equality_fro = primal_safe["equality_fro"]
    primal_feasible_completeness = (
        sum(repaired_measurements, np.zeros((rank, rank), dtype=float))
        - np.eye(rank)
    )
    primal_feasible_equality_op = float(
        np.linalg.norm(primal_feasible_completeness, ord=2)
    )
    primal_feasible_psd_violation = max(
        _negative_eigenvalue_violation(measurement)
        for measurement in repaired_measurements
    )

    (
        dual_safe,
        dual_safety_shift,
        dual_feasible_min_slack,
        dual_feasible_psd_violation,
    ) = _safe_dual_bound(dual, rhos)
    dual_feasible_objective = float(np.trace(dual_safe))
    feasible_bound_gap = dual_feasible_objective - primal_feasible_objective
    relative_feasible_bound_gap = feasible_bound_gap / max(
        1e-15,
        0.5
        * (abs(primal_feasible_objective) + abs(dual_feasible_objective)),
    )

    return {
        **metadata,
        "solver": solver.upper(),
        "cvxpy_version": cp.__version__,
        "primal_status": primal_status,
        "dual_status": dual_status,
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "signed_gap": signed_gap,
        "absolute_gap": absolute_gap,
        "relative_signed_gap": relative_signed_gap,
        "relative_gap": relative_gap,
        "primal_equality_residual_fro_raw": primal_equality_fro_raw,
        "primal_equality_residual_fro": primal_equality_fro,
        "primal_equality_residual_op": primal_equality_op,
        "primal_psd_violation": primal_psd_violation,
        "dual_psd_violation": dual_psd_violation,
        "complementarity_residual": complementarity,
        "primal_feasible_objective": primal_feasible_objective,
        "dual_feasible_objective": dual_feasible_objective,
        "feasible_bound_gap": feasible_bound_gap,
        "relative_feasible_bound_gap": relative_feasible_bound_gap,
        "primal_safety_contraction": primal_safety_contraction,
        "primal_regularization_floor": primal_safe["regularization_floor"],
        "dual_safety_shift": dual_safety_shift,
        "dual_feasible_min_slack": dual_feasible_min_slack,
        "primal_feasible_equality_residual_fro": primal_feasible_equality_fro,
        "primal_feasible_equality_residual_op": primal_feasible_equality_op,
        "primal_feasible_psd_violation": primal_feasible_psd_violation,
        "primal_feasible_min_eigenvalue": primal_safe["minimum_eigenvalue"],
        "dual_feasible_psd_violation": dual_feasible_psd_violation,
        "primal_feasible_combined_violation": max(
            primal_safe["equality_op"], primal_feasible_psd_violation
        ),
        "primal_iterations": primal_iterations,
        "dual_iterations": dual_iterations,
        "primal_solve_time_seconds": primal_time,
        "dual_solve_time_seconds": dual_time,
    }
