"""Floating-point primal/dual SDP verification for state discrimination.

Historical public API and output-field names retain 'certify' for backward
compatibility. The computations use conventional IEEE double precision, not
interval arithmetic or exact-rational certification.
"""

from __future__ import annotations

import hashlib
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

    The raw matrices are projected onto the PSD cone.  If
    ``S = sum_j M_j^+`` and ``alpha = max(1, lambda_max(S))``, then every
    ``M_j^+/alpha`` is PSD and their sum is bounded by the identity.  The PSD
    remainder is assigned to the first outcome.  A final convex mixture with
    the uniform POVM makes every effect positive definite while preserving
    completeness.  This construction does not infer feasibility from a small
    completeness residual and never inverts ``S``.
    """
    rank = measurements[0].shape[0]
    factors = [_psd_factor(measurement) for measurement in measurements]
    projected = [_symmetric(factor @ factor.T) for factor in factors]
    total = _symmetric(sum(projected, np.zeros((rank, rank), dtype=float)))
    margin = _roundoff_margin([total, *projected, *rhos])
    identity = np.eye(rank)
    spectral_radius = float(np.linalg.eigvalsh(total)[-1])
    alpha = max(1.0, spectral_radius)
    # One outward representable step makes the spectral contraction
    # conservative in ordinary IEEE-double arithmetic.  If the subsequent
    # diagnostic recheck still sees a negative remainder, enlarge alpha by a
    # scale-aware correction and recompute.
    alpha = float(np.nextafter(alpha, math.inf))
    repaired = [_symmetric(matrix / alpha) for matrix in projected]
    repaired_total = _symmetric(
        sum(repaired, np.zeros((rank, rank), dtype=float))
    )
    remainder = _symmetric(identity - repaired_total)
    minimum_remainder = float(np.linalg.eigvalsh(remainder)[0])
    if minimum_remainder < 0.0:
        alpha = float(
            np.nextafter(
                alpha + (-minimum_remainder + margin) * max(1.0, alpha),
                math.inf,
            )
        )
        repaired = [_symmetric(matrix / alpha) for matrix in projected]
        repaired_total = _symmetric(
            sum(repaired, np.zeros((rank, rank), dtype=float))
        )
        remainder = _symmetric(identity - repaired_total)
    repaired[0] = _symmetric(repaired[0] + remainder)

    # Strictly regularize without assuming that the projected sum is
    # invertible.  If q is the number of outcomes and f is the identity floor,
    # then
    #
    #   M_j -> (1 - q f) M_j + f I
    #
    # is a convex mixture with the uniform POVM and still sums to I.  The
    # scale-aware correction dominates any negative eigenvalue seen in the
    # floating-point recheck of the nominally PSD repaired effects.
    hypothesis_count = len(repaired)
    base_minimum_eigenvalue = min(
        float(np.linalg.eigvalsh(_symmetric(measurement))[0])
        for measurement in repaired
    )
    regularization_floor = margin + max(0.0, -base_minimum_eigenvalue)
    mixing_weight = hypothesis_count * regularization_floor
    if not 0.0 < mixing_weight < 0.5:
        raise RuntimeError(
            "scale-aware primal regularization is unexpectedly large; "
            "the solver output cannot be safely repaired"
        )
    repaired = [
        _symmetric(
            (1.0 - mixing_weight) * measurement
            + regularization_floor * identity
        )
        for measurement in repaired
    ]

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
        "contraction": (1.0 - mixing_weight) / alpha,
        "regularization_floor": regularization_floor,
        "equality_fro": equality_fro,
        "equality_op": equality_op,
        "psd_violation": psd_violation,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def _validated_weighted_states(source_states: np.ndarray) -> np.ndarray:
    """Return the sole real float64 source factor consumed by both SDPs."""
    raw = np.asarray(source_states)
    if np.iscomplexobj(raw):
        raise ValueError("source_states must be real")
    states = np.array(raw, dtype=float, copy=True, order="C")
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("source_states must be a nonempty rank-by-count matrix")
    if states.shape[0] > states.shape[1]:
        raise ValueError("source_states cannot have more rows than hypotheses")
    if not np.all(np.isfinite(states)):
        raise ValueError("source_states must be finite")
    return states


def _array_sha256(array: np.ndarray) -> str:
    """Fingerprint a float64 artifact with its shape and byte order bound."""
    contiguous = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    shape = ",".join(str(value) for value in contiguous.shape)
    prefix = f"float64-le|{contiguous.ndim}|{shape}|".encode("ascii")
    return hashlib.sha256(prefix + contiguous.tobytes(order="C")).hexdigest()


def _certify_weighted_states_core(
    states: np.ndarray,
    *,
    solver: str,
    verbose: bool,
) -> dict[str, object]:
    """Solve and repair both SDPs on exactly the supplied source factor."""
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

    source_gram = _symmetric(states.T @ states)
    weighted_eigenvalues = np.linalg.eigvalsh(source_gram)
    source_states_artifact = np.array(states, copy=True, order="C")
    source_gram_artifact = np.array(source_gram, copy=True, order="C")
    repaired_artifacts = tuple(
        np.array(measurement, copy=True, order="C")
        for measurement in repaired_measurements
    )
    dual_artifact = np.array(dual_safe, copy=True, order="C")

    return {
        "rank": rank,
        "hypothesis_count": hypothesis_count,
        "rank_threshold": 0.0,
        "weighted_gram_lambda_min": float(weighted_eigenvalues[0]),
        "weighted_gram_lambda_max": float(weighted_eigenvalues[-1]),
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
        "certificate_artifact_schema": "weighted_state_certificate_v1",
        "source_weighted_states": source_states_artifact,
        "source_weighted_gram": source_gram_artifact,
        "source_weighted_states_sha256": _array_sha256(
            source_states_artifact
        ),
        "source_weighted_gram_sha256": _array_sha256(source_gram_artifact),
        "repaired_primal_povm": repaired_artifacts,
        "safe_dual_operator": dual_artifact,
    }


def certify_minimum_error_from_weighted_states(
    source_states: np.ndarray,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> dict[str, object]:
    """Solve directly on one retained weighted-state factor.

    This entry point performs no Gram reconstruction, canonicalization, rank
    truncation, or normalization.  The copied factor returned in the artifact
    payload is the same copied factor consumed by both solver problems.
    """
    states = _validated_weighted_states(source_states)
    return _certify_weighted_states_core(
        states,
        solver=solver,
        verbose=verbose,
    )


def certify_minimum_error(
    gram: np.ndarray,
    solver: str = "CLARABEL",
    rank_tolerance: float = 1e-12,
    verbose: bool = False,
    priors: np.ndarray | None = None,
) -> dict[str, object]:
    """Backward-compatible Gram entry point for the floating-point SDP."""
    states, metadata = canonical_weighted_states(
        gram, rank_tolerance=rank_tolerance, priors=priors
    )
    certificate = _certify_weighted_states_core(
        _validated_weighted_states(states),
        solver=solver,
        verbose=verbose,
    )
    certificate.update(metadata)
    for artifact_field in (
        "certificate_artifact_schema",
        "source_weighted_states",
        "source_weighted_gram",
        "source_weighted_states_sha256",
        "source_weighted_gram_sha256",
        "repaired_primal_povm",
        "safe_dual_operator",
    ):
        certificate.pop(artifact_field, None)
    return certificate
