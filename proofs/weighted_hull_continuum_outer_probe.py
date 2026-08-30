"""Finite-dimensional SRM and floating-SDP weighted-hull diagnostics.

These computations are numerical probes, not proofs of an asymptotic claim.
Every row uses the external hypothesis count ``M_n = n(n + 1) / 2``.  Dense
resource estimates are logged and checked before a physical Gram is allocated.
The reported peak is a conservative dense-linear-algebra array proxy, not a
strict bound on the memory of CVXPY, a solver, or the Python process.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum_interval_numerics import p1, srm_quantities
from sdp_certification import (
    canonical_weighted_states,
    certify_minimum_error,
    certify_minimum_error_from_weighted_states,
)


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "weighted_hull_diagnostics_v8_route_b"
DEFAULT_MAX_SRM_N = 48
DEFAULT_MAX_SDP_N = 5
MAX_EXPLICIT_SDP_N = 7
DEFAULT_MAX_PEAK_BYTES = 512 * 2**20
DEFAULT_MAX_MATRIX_DIMENSION = 1200
CONSERVATIVE_DENSE_ARRAY_COUNT = 16
ESTIMATED_PEAK_SCOPE = (
    "conservative_dense_linear_algebra_array_proxy_not_process_peak"
)
DIAGNOSTIC_GRID_SCOPE = "default_srm_cutoff_n48_frozen_grid_through_n32"
PROBABILITY_RELATIVE_TOLERANCE = 1.0e-8
COMPLETENESS_RESIDUAL_TOLERANCE = 1.0e-9
PRIMAL_PSD_RESIDUAL_TOLERANCE = 1.0e-9
DUAL_PSD_RELATIVE_TOLERANCE = 1.0e-9
SPECTRAL_RECONSTRUCTION_RELATIVE_TOLERANCE = 1.0e-10
ARTIFACT_SCALAR_RELATIVE_TOLERANCE = 4096.0 * np.finfo(float).eps

HULL_AUDIT_FIELDS = frozenset(
    {
        "hull_subtraction_max_abs_error",
        "hull_subtraction_max_elementwise_envelope",
        "hull_subtraction_fro_error",
        "hull_subtraction_fro_envelope",
        "hull_subtraction_audit_status",
        "hull_subtraction_audit_kind",
    }
)
SOURCE_METADATA_FIELDS = frozenset(
    {
        "rank",
        "hypothesis_count",
        "rank_threshold",
        "gram_lambda_min",
        "gram_lambda_max",
        "weighted_gram_lambda_min",
        "weighted_gram_lambda_max",
        "prior_min",
        "prior_max",
    }
)
SPECTRAL_AUDIT_FIELDS = frozenset(
    {
        "trusted_source_snapshot_sha256",
        "source_gram_lambda_min",
        "source_gram_lambda_max",
        "source_gram_factorization_residual_fro_relative",
        "source_gram_factorization_residual_op_relative",
        "source_gram_support_rank",
        "srm_sqrt_reconstruction_residual_fro_relative",
        "srm_sqrt_reconstruction_residual_op_relative",
        "srm_sqrt_support_rank",
        "source_reconstructed_sqrt_residual_fro_relative",
        "source_reconstructed_sqrt_residual_op_relative",
        "spectral_audit_status",
        "source_target_trace",
        "source_reconstructed_trace",
        "source_difference_trace_norm",
        "source_difference_fro_norm",
        "source_difference_op_norm",
        "source_difference_trace_norm_relative",
        "source_difference_fro_norm_relative",
        "source_difference_op_norm_relative",
        "source_lambda_min_target",
        "source_lambda_max_target",
        "source_lambda_min_reconstructed",
        "source_lambda_max_reconstructed",
        "source_sqrt_difference_computed_fro",
        "source_sqrt_difference_bound_trace",
        "source_sqrt_difference_bound_spectral",
        "source_sqrt_difference_bound",
        "source_transfer_delta",
        "source_transfer_budget",
        "source_transfer_status",
    }
)
SOURCE_IDENTITY_FIELDS = frozenset(
    {
        "trusted_source_snapshot_sha256",
        "trusted_source_snapshot_pre_sha256",
        "trusted_source_snapshot_post_sha256",
        "trusted_source_snapshot_validation_status",
        "trusted_source_snapshot_validation_reason",
        "trusted_source_snapshot_validation_phase",
    }
)
CERTIFICATE_INPUT_AUDIT_FIELDS = frozenset(
    {
        "certificate_input_factor_pre_sha256",
        "certificate_input_factor_post_sha256",
        "certificate_input_factor_max_abs_error",
        "certificate_input_factor_pre_writeable",
        "certificate_input_factor_post_writeable",
        "certificate_input_factor_validation_status",
        "certificate_input_factor_validation_reason",
    }
)
SAFE_VALIDATION_FIELDS = frozenset(
    {
        "certificate_rank",
        "certificate_expected_rank",
        "certificate_rank_tolerance",
        "certificate_probability_scale",
        "certificate_probability_tolerance",
        "certificate_residual_tolerance",
        "certificate_completeness_tolerance",
        "certificate_primal_psd_tolerance",
        "certificate_dual_psd_tolerance",
        "validated_lower",
        "validated_upper",
    }
)
ARTIFACT_METRIC_FIELDS = frozenset(
    {
        "certificate_artifact_schema",
        "certificate_artifact_source_states_sha256",
        "certificate_artifact_source_gram_sha256",
        "certificate_artifact_source_states_max_abs_error",
        "certificate_artifact_source_gram_max_abs_error",
        "certificate_artifact_primal_symmetry_error_fro",
        "certificate_artifact_primal_completeness_fro",
        "certificate_artifact_primal_completeness_op",
        "certificate_artifact_primal_psd_violation",
        "certificate_artifact_primal_min_eigenvalue",
        "certificate_artifact_primal_objective",
        "certificate_artifact_dual_symmetry_error_fro",
        "certificate_artifact_dual_min_slack",
        "certificate_artifact_dual_psd_violation",
        "certificate_artifact_dual_trace",
    }
)
CERTIFICATE_FIELDS = frozenset(
    {
        "absolute_gap",
        "certificate_artifact_schema",
        "complementarity_residual",
        "cvxpy_version",
        "dual_feasible_min_slack",
        "dual_feasible_objective",
        "dual_feasible_psd_violation",
        "dual_iterations",
        "dual_objective",
        "dual_psd_violation",
        "dual_safety_shift",
        "dual_solve_time_seconds",
        "dual_status",
        "feasible_bound_gap",
        "hypothesis_count",
        "primal_equality_residual_fro",
        "primal_equality_residual_fro_raw",
        "primal_equality_residual_op",
        "primal_feasible_combined_violation",
        "primal_feasible_equality_residual_fro",
        "primal_feasible_equality_residual_op",
        "primal_feasible_min_eigenvalue",
        "primal_feasible_objective",
        "primal_feasible_psd_violation",
        "primal_iterations",
        "primal_objective",
        "primal_psd_violation",
        "primal_regularization_floor",
        "primal_safety_contraction",
        "primal_solve_time_seconds",
        "primal_status",
        "rank",
        "rank_threshold",
        "relative_feasible_bound_gap",
        "relative_gap",
        "relative_signed_gap",
        "repaired_primal_povm",
        "safe_dual_operator",
        "signed_gap",
        "solver",
        "source_weighted_gram",
        "source_weighted_gram_sha256",
        "source_weighted_states",
        "source_weighted_states_sha256",
        "weighted_gram_lambda_max",
        "weighted_gram_lambda_min",
    }
)

COMPACT_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
COMPACT_N_VALUES = (5, 8, 16)
OUTER_N_VALUES = (8, 16, 32)

CSV_FIELDS = (
    "schema_version",
    "schedule",
    "lambda_target",
    "n",
    "external_M_n",
    "c",
    "lambda",
    "h",
    "p1_squared",
    "P_tr",
    "P_SRM",
    "repaired_primal_value",
    "shifted_dual_value",
    "strongest_measurement_value",
    "floating_primal_dual_span",
    "floating_primal_dual_span_relative",
    "repaired_primal_over_p1_squared",
    "shifted_dual_over_p1_squared",
    "strongest_measurement_over_p1_squared",
    "repaired_primal_over_P_SRM",
    "P_SRM_over_shifted_dual",
    "primal_completeness_residual_fro",
    "primal_completeness_residual_op",
    "primal_psd_violation",
    "primal_min_eigenvalue",
    "dual_min_slack_after_shift",
    "dual_psd_violation_after_shift",
    "artifact_recomputation_status",
    "artifact_recomputation_reason",
    "diagnostic_kind",
    "legacy_bound_fields_status",
    "P_SRM_transfer_budget",
    "P_SRM_safe_lower",
    "L_raw",
    "U_raw",
    "L_safe",
    "U_safe",
    "strongest_certified_lower",
    "P_tr_over_p1_squared",
    "P_SRM_over_p1_squared",
    "L_safe_over_p1_squared",
    "U_safe_over_p1_squared",
    "strongest_lower_over_p1_squared",
    "P_tr_over_P_SRM",
    "L_safe_over_P_SRM",
    "P_SRM_over_U_safe",
    "L_safe_over_U_safe",
    "raw_safe_gap",
    "raw_safe_gap_relative",
    "safe_gap",
    "safe_gap_relative",
    "primal_feasible_equality_residual_fro",
    "primal_feasible_equality_residual_op",
    "primal_feasible_psd_violation",
    "primal_feasible_min_eigenvalue",
    "dual_feasible_min_slack",
    "dual_feasible_psd_violation",
    "P_opt",
    "P_opt_status",
    "srm_status",
    "sdp_status",
    "primal_status",
    "dual_status",
    "status",
    "solver",
    "cvxpy_version",
    "gram_bytes",
    "estimated_peak_bytes",
    "estimated_dense_array_count",
    "estimated_peak_scope",
    "eigensolver_dimension",
    "eigensolver_cubic_work_units",
    "source_gram_lambda_min",
    "source_gram_lambda_max",
    "source_gram_factorization_residual_fro_relative",
    "source_gram_factorization_residual_op_relative",
    "source_gram_support_rank",
    "trusted_source_snapshot_sha256",
    "trusted_source_snapshot_pre_sha256",
    "trusted_source_snapshot_post_sha256",
    "trusted_source_snapshot_validation_status",
    "trusted_source_snapshot_validation_reason",
    "trusted_source_snapshot_validation_phase",
    "srm_sqrt_reconstruction_residual_fro_relative",
    "srm_sqrt_reconstruction_residual_op_relative",
    "srm_sqrt_support_rank",
    "source_reconstructed_sqrt_residual_fro_relative",
    "source_reconstructed_sqrt_residual_op_relative",
    "spectral_audit_status",
    "source_target_trace",
    "source_reconstructed_trace",
    "source_difference_trace_norm",
    "source_difference_fro_norm",
    "source_difference_op_norm",
    "source_difference_trace_norm_relative",
    "source_difference_fro_norm_relative",
    "source_difference_op_norm_relative",
    "source_lambda_min_target",
    "source_lambda_max_target",
    "source_lambda_min_reconstructed",
    "source_lambda_max_reconstructed",
    "source_sqrt_difference_computed_fro",
    "source_sqrt_difference_bound_trace",
    "source_sqrt_difference_bound_spectral",
    "source_sqrt_difference_bound",
    "source_transfer_delta",
    "source_transfer_budget",
    "source_transfer_status",
    "hull_subtraction_max_abs_error",
    "hull_subtraction_max_elementwise_envelope",
    "hull_subtraction_fro_error",
    "hull_subtraction_fro_envelope",
    "hull_subtraction_audit_status",
    "hull_subtraction_audit_kind",
    "certificate_rank",
    "certificate_expected_rank",
    "certificate_rank_tolerance",
    "certificate_probability_scale",
    "certificate_probability_tolerance",
    "certificate_residual_tolerance",
    "certificate_completeness_tolerance",
    "certificate_primal_psd_tolerance",
    "certificate_dual_psd_tolerance",
    "certificate_validation_status",
    "certificate_validation_reason",
    "certificate_input_factor_pre_sha256",
    "certificate_input_factor_post_sha256",
    "certificate_input_factor_max_abs_error",
    "certificate_input_factor_pre_writeable",
    "certificate_input_factor_post_writeable",
    "certificate_input_factor_validation_status",
    "certificate_input_factor_validation_reason",
    "certificate_source_binding_status",
    "certificate_source_binding_reason",
    "certificate_artifact_schema",
    "certificate_artifact_source_states_sha256",
    "certificate_artifact_source_gram_sha256",
    "certificate_artifact_source_states_max_abs_error",
    "certificate_artifact_source_gram_max_abs_error",
    "certificate_artifact_primal_symmetry_error_fro",
    "certificate_artifact_primal_completeness_fro",
    "certificate_artifact_primal_completeness_op",
    "certificate_artifact_primal_psd_violation",
    "certificate_artifact_primal_min_eigenvalue",
    "certificate_artifact_primal_objective",
    "certificate_artifact_dual_symmetry_error_fro",
    "certificate_artifact_dual_min_slack",
    "certificate_artifact_dual_psd_violation",
    "certificate_artifact_dual_trace",
    "certificate_artifact_validation_status",
    "certificate_artifact_validation_reason",
    "enclosure_kind",
    "numeric_guarantee",
    "diagnostic_grid_scope",
    "interpretation",
)


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return n


def _validate_overlap(c: float) -> float:
    overlap = float(c)
    if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError("c must be finite and lie in [0,1]")
    return overlap


def _interval_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    labels = [(left, right) for left in range(n) for right in range(left, n)]
    left = np.fromiter((label[0] for label in labels), dtype=np.int64)
    right = np.fromiter((label[1] for label in labels), dtype=np.int64)
    return left, right


def _explicit_product_state_gram(n: int, c: float) -> np.ndarray:
    """Build physical states directly; used only for the mandatory n<=4 audit."""
    zero = np.array([1.0, 0.0])
    anomaly = np.array([c, math.sqrt(max(0.0, 1.0 - c * c))])
    states: list[np.ndarray] = []
    for left in range(n):
        for right in range(left, n):
            state = np.array([1.0])
            for site in range(n):
                local = anomaly if left <= site <= right else zero
                state = np.kron(state, local)
            states.append(state)
    matrix = np.asarray(states, dtype=float)
    return matrix @ matrix.T


def physical_interval_gram(n: int, c: float) -> np.ndarray:
    """Return ``G[I,J] = c**|I symmetric-difference J|``.

    The interval order is the zero-based lexicographic order
    ``(0,0), (0,1), ..., (n-1,n-1)``.  For ``n <= 4`` the formula is actively
    checked against independently assembled tensor-product states.
    """
    site_count = _validate_n(n)
    overlap = _validate_overlap(c)
    left, right = _interval_arrays(site_count)
    lengths = right - left + 1
    intersection = np.maximum(
        0,
        np.minimum(right[:, None], right[None, :])
        - np.maximum(left[:, None], left[None, :])
        + 1,
    )
    symmetric_difference = (
        lengths[:, None] + lengths[None, :] - 2 * intersection
    )
    gram = np.power(overlap, symmetric_difference, dtype=float)
    if site_count <= 4:
        explicit = _explicit_product_state_gram(site_count, overlap)
        if not np.allclose(gram, explicit, rtol=2e-14, atol=2e-14):
            difference = float(np.max(np.abs(gram - explicit)))
            raise AssertionError(
                "physical Gram failed independent product-state check: "
                f"n={site_count}, c={overlap}, max_abs={difference:.6e}"
            )
    return gram


def _vacuum_gram(n: int, c: float) -> np.ndarray:
    left, right = _interval_arrays(n)
    lengths = right - left + 1
    amplitudes = np.power(c, lengths, dtype=float)
    return np.multiply.outer(amplitudes, amplitudes)


def _one_excitation_gram(n: int, c: float) -> np.ndarray:
    left, right = _interval_arrays(n)
    lengths = right - left + 1
    coordinates = np.zeros((len(left), n), dtype=float)
    local_excitation = math.sqrt(max(0.0, 1.0 - c * c))
    row_amplitudes = local_excitation * np.power(
        c, lengths - 1, dtype=float
    )
    for row, (interval_left, interval_right) in enumerate(zip(left, right)):
        coordinates[row, interval_left : interval_right + 1] = row_amplitudes[
            row
        ]
    return coordinates @ coordinates.T


def _stable_hull_gram(n: int, c: float) -> np.ndarray:
    """Evaluate the nonnegative binomial-tail formula for ``G_{>=2}``.

    For intervals ``I,J``, put ``k=|I intersect J|``, ``d=|I symmetric
    difference J|``, ``q=c**2``, and ``x=1-q``.  The returned entry is

    ``c**d * sum(comb(k,t) * x**t * q**(k-t), t=2..k)``.

    This positive-term expression avoids the cancellation in
    ``G_full-G_vacuum-G_one`` near ``c=1``.
    """
    site_count = _validate_n(n)
    overlap = _validate_overlap(c)
    left, right = _interval_arrays(site_count)
    lengths = right - left + 1
    intersection = np.maximum(
        0,
        np.minimum(right[:, None], right[None, :])
        - np.maximum(left[:, None], left[None, :])
        + 1,
    )
    symmetric_difference = (
        lengths[:, None] + lengths[None, :] - 2 * intersection
    )
    q = overlap * overlap
    x = 1.0 - q
    tails = np.zeros(site_count + 1, dtype=float)
    for k in range(2, site_count + 1):
        tails[k] = math.fsum(
            math.comb(k, t) * x**t * q ** (k - t)
            for t in range(2, k + 1)
        )
    stable = np.power(overlap, symmetric_difference, dtype=float) * tails[
        intersection
    ]
    return (stable + stable.T) / 2.0


def _hull_gram_with_audit(
    n: int, c: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Build both hull formulas and compare their IEEE-double evaluations."""
    site_count = _validate_n(n)
    overlap = _validate_overlap(c)
    full_gram = physical_interval_gram(site_count, overlap)
    vacuum_gram = _vacuum_gram(site_count, overlap)
    one_excitation_gram = _one_excitation_gram(site_count, overlap)
    raw = full_gram - vacuum_gram - one_excitation_gram
    raw = (raw + raw.T) / 2.0
    stable = _stable_hull_gram(site_count, overlap)

    error = np.abs(raw - stable)
    arithmetic_scale = (
        np.abs(full_gram)
        + np.abs(vacuum_gram)
        + np.abs(one_excitation_gram)
        + np.abs(stable)
    )
    # This deliberately conservative roundoff proxy covers two subtractions,
    # the length-n dot product in G_1, and the positive binomial sum.  It is an
    # empirical IEEE-double consistency envelope, not a backward-error theorem,
    # interval-arithmetic enclosure, or directed-rounding proof.
    roundoff_factor = (64.0 + 16.0 * site_count) * np.finfo(float).eps
    elementwise_bound = roundoff_factor * np.maximum(
        arithmetic_scale, np.finfo(float).tiny
    )
    max_error = float(np.max(error, initial=0.0))
    max_bound = float(np.max(elementwise_bound, initial=0.0))
    fro_error = float(np.linalg.norm(raw - stable, ord="fro"))
    fro_bound = float(np.linalg.norm(elementwise_bound, ord="fro"))
    finite = all(
        math.isfinite(value)
        for value in (max_error, max_bound, fro_error, fro_bound)
    )
    elementwise_passed = bool(np.all(error <= elementwise_bound))
    fro_passed = fro_error <= fro_bound
    status = "passed" if finite and elementwise_passed and fro_passed else "failed"
    audit: dict[str, float | str] = {
        "hull_subtraction_max_abs_error": max_error,
        "hull_subtraction_max_elementwise_envelope": max_bound,
        "hull_subtraction_fro_error": fro_error,
        "hull_subtraction_fro_envelope": fro_bound,
        "hull_subtraction_audit_status": status,
        "hull_subtraction_audit_kind": (
            "empirical_ieee_double_consistency_envelope_not_error_bound_or_proof"
        ),
    }
    return stable, full_gram, audit


def independent_hull_gram(n: int, c: float) -> np.ndarray:
    """Audit an independent raw subtraction and return the stable hull Gram.

    The full physical Gram, vacuum Gram, and one-excitation Gram are assembled
    by three separate constructions and actually subtracted.  That raw result
    is checked against a separately evaluated positive binomial-tail formula;
    after the audit passes, the stable formula is returned.  This function has
    no dependency on the Task 1 weighted-coordinate helper.
    """
    stable, _, audit = _hull_gram_with_audit(n, c)
    if audit["hull_subtraction_audit_status"] != "passed":
        raise AssertionError(
            "raw subtraction consistency envelope failed: "
            f"max_abs={audit['hull_subtraction_max_abs_error']:.6e}, "
            "max_elementwise_envelope="
            f"{audit['hull_subtraction_max_elementwise_envelope']:.6e}, "
            f"fro={audit['hull_subtraction_fro_error']:.6e}, "
            f"fro_envelope={audit['hull_subtraction_fro_envelope']:.6e}"
        )
    return stable


def estimate_dense_resources(n: int) -> dict[str, int | str]:
    """Estimate dense-array resources without allocating a matrix.

    ``estimated_peak_bytes`` is a conservative multi-array proxy for the dense
    linear algebra in this probe.  It is not a strict peak-memory upper bound
    for CVXPY, an SDP solver, Python, or the complete process.
    """
    site_count = _validate_n(n)
    external_count = site_count * (site_count + 1) // 2
    gram_bytes = 8 * external_count**2
    return {
        "n": site_count,
        "external_M_n": external_count,
        "candidate_count": external_count,
        "gram_bytes": gram_bytes,
        "estimated_peak_bytes": (
            CONSERVATIVE_DENSE_ARRAY_COUNT * gram_bytes
        ),
        "estimated_dense_array_count": CONSERVATIVE_DENSE_ARRAY_COUNT,
        "estimated_peak_scope": ESTIMATED_PEAK_SCOPE,
        "eigensolver_dimension": external_count,
        "eigensolver_cubic_work_units": external_count**3,
    }


def _strict_positive_definite_gram(gram: np.ndarray) -> bool:
    """Detect exact or numerical rank loss without a positive cutoff.

    Cholesky is intentionally used as a sign/rank condition, not as a
    user-chosen spectral truncation rule.  Thus a resolvable tiny positive
    direction is retained, while a singular all-ones Gram fails closed.
    """
    matrix = np.asarray(gram, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        return False
    symmetric = (matrix + matrix.T) / 2.0
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        return False
    return bool(
        np.all(np.isfinite(factor))
        and np.all(np.diag(factor) > 0.0)
    )


def _trusted_source_snapshot_audit(
    gram: np.ndarray,
    trusted_source_snapshot: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, float | int | str]]:
    """Recompute all source authority from ``(G, retained S)`` alone.

    The caller may obtain ``S`` from any factorization path.  This pure audit
    deliberately accepts no factorization metadata: it independently forms
    ``A = G/M`` and ``B = S.T @ S``, checks their supports and spectra, and
    derives every transfer bound and certificate-binding scalar from those two
    actual matrices.  The IEEE-double checks are residual diagnostics, not a
    directed-rounding proof.
    """
    raw_physical = np.asarray(gram)
    raw_source = np.asarray(trusted_source_snapshot)
    if np.iscomplexobj(raw_physical) or np.iscomplexobj(raw_source):
        raise ValueError("physical Gram and retained source must be real")
    physical = np.array(raw_physical, dtype=float, copy=True, order="C")
    source_states = np.array(raw_source, dtype=float, copy=True, order="C")
    if (
        physical.ndim != 2
        or physical.shape[0] != physical.shape[1]
        or physical.shape[0] < 1
    ):
        raise ValueError("physical Gram must be nonempty and square")
    hypothesis_count = int(physical.shape[0])
    if (
        source_states.ndim != 2
        or source_states.shape[1] != hypothesis_count
        or source_states.shape[0] < 1
        or source_states.shape[0] > hypothesis_count
    ):
        raise ValueError("retained source has incompatible dimensions")
    if not (
        np.all(np.isfinite(physical))
        and np.all(np.isfinite(source_states))
    ):
        raise ValueError("physical Gram and retained source must be finite")

    physical = (physical + physical.T) / 2.0
    target = physical / float(hypothesis_count)
    reconstructed = source_states.T @ source_states
    reconstructed = (reconstructed + reconstructed.T) / 2.0
    raw_gram_is_strictly_positive = _strict_positive_definite_gram(physical)

    raw_eigenvalues = np.linalg.eigvalsh(physical)
    target_eigenvalues, target_eigenvectors = np.linalg.eigh(target)
    (
        reconstructed_square_root_eigenvalues,
        reconstructed_eigenvectors,
    ) = np.linalg.eigh(reconstructed)
    # Use eigvalsh for the authoritative source spectrum because the direct
    # certificate API reports its binding metadata with the same routine.
    reconstructed_eigenvalues = np.linalg.eigvalsh(reconstructed)
    target_roots = np.sqrt(np.maximum(target_eigenvalues, 0.0))
    reconstructed_roots = np.sqrt(
        np.maximum(reconstructed_square_root_eigenvalues, 0.0)
    )
    target_square_root = (
        target_eigenvectors * target_roots[None, :]
    ) @ target_eigenvectors.T
    reconstructed_square_root = (
        reconstructed_eigenvectors * reconstructed_roots[None, :]
    ) @ reconstructed_eigenvectors.T
    target_square_reconstruction = target_square_root @ target_square_root
    reconstructed_square_reconstruction = (
        reconstructed_square_root @ reconstructed_square_root
    )

    difference = (target - reconstructed + (target - reconstructed).T) / 2.0
    difference_eigenvalues = np.linalg.eigvalsh(difference)
    difference_trace_norm = float(np.sum(np.abs(difference_eigenvalues)))
    difference_fro_norm = float(np.linalg.norm(difference, ord="fro"))
    difference_op_norm = float(np.linalg.norm(difference, ord=2))
    target_trace = float(np.trace(target))
    reconstructed_trace = float(np.trace(reconstructed))
    lambda_min_target = float(target_eigenvalues[0])
    lambda_max_target = float(target_eigenvalues[-1])
    lambda_min_reconstructed = float(reconstructed_eigenvalues[0])
    lambda_max_reconstructed = float(reconstructed_eigenvalues[-1])
    square_root_difference_computed = float(
        np.linalg.norm(
            target_square_root - reconstructed_square_root, ord="fro"
        )
    )

    # Powers--Stoermer and positive-definite Sylvester estimates are computed
    # independently; their minimum is the actual transfer authority.
    square_root_bound_trace = math.sqrt(max(0.0, difference_trace_norm))
    if lambda_min_target > 0.0 and lambda_min_reconstructed > 0.0:
        square_root_bound_spectral = difference_fro_norm / (
            math.sqrt(lambda_min_target)
            + math.sqrt(lambda_min_reconstructed)
        )
    else:
        square_root_bound_spectral = math.inf
    square_root_bound = min(
        square_root_bound_trace, square_root_bound_spectral
    )
    transfer_delta = math.sqrt(
        2.0 * (target_trace + reconstructed_trace)
    ) * square_root_bound
    transfer_budget = float(np.nextafter(transfer_delta, math.inf))

    tiny = np.finfo(float).tiny
    trace_scale = max(abs(target_trace), tiny)
    fro_scale = max(float(np.linalg.norm(target, ord="fro")), tiny)
    op_scale = max(float(np.linalg.norm(target, ord=2)), tiny)
    difference_trace_relative = difference_trace_norm / trace_scale
    difference_fro_relative = difference_fro_norm / fro_scale
    difference_op_relative = difference_op_norm / op_scale
    target_square_fro = float(
        np.linalg.norm(target_square_reconstruction - target, ord="fro")
        / fro_scale
    )
    target_square_op = float(
        np.linalg.norm(target_square_reconstruction - target, ord=2)
        / op_scale
    )
    reconstructed_fro_scale = max(
        float(np.linalg.norm(reconstructed, ord="fro")), tiny
    )
    reconstructed_op_scale = max(
        float(np.linalg.norm(reconstructed, ord=2)), tiny
    )
    reconstructed_square_fro = float(
        np.linalg.norm(
            reconstructed_square_reconstruction - reconstructed, ord="fro"
        )
        / reconstructed_fro_scale
    )
    reconstructed_square_op = float(
        np.linalg.norm(
            reconstructed_square_reconstruction - reconstructed, ord=2
        )
        / reconstructed_op_scale
    )
    source_support_rank = int(
        np.count_nonzero(reconstructed_eigenvalues > 0.0)
    )
    target_support_rank = int(np.count_nonzero(target_eigenvalues > 0.0))

    numeric_values = (
        target_trace,
        reconstructed_trace,
        difference_trace_norm,
        difference_fro_norm,
        difference_op_norm,
        difference_trace_relative,
        difference_fro_relative,
        difference_op_relative,
        lambda_min_target,
        lambda_max_target,
        lambda_min_reconstructed,
        lambda_max_reconstructed,
        square_root_difference_computed,
        square_root_bound_trace,
        square_root_bound_spectral,
        square_root_bound,
        transfer_delta,
        transfer_budget,
        target_square_fro,
        target_square_op,
        reconstructed_square_fro,
        reconstructed_square_op,
    )
    finite = bool(
        np.all(np.isfinite(target))
        and np.all(np.isfinite(reconstructed))
        and all(math.isfinite(value) for value in numeric_values)
    )
    residuals_pass = max(
        difference_trace_relative,
        difference_fro_relative,
        difference_op_relative,
        target_square_fro,
        target_square_op,
        reconstructed_square_fro,
        reconstructed_square_op,
    ) <= SPECTRAL_RECONSTRUCTION_RELATIVE_TOLERANCE
    support_pass = (
        int(source_states.shape[0]) == hypothesis_count
        and source_support_rank == hypothesis_count
        and target_support_rank == hypothesis_count
        and raw_gram_is_strictly_positive
        and lambda_min_target > 0.0
        and lambda_min_reconstructed > 0.0
    )
    square_root_scale = max(
        float(np.linalg.norm(target_square_root, ord="fro")),
        float(np.linalg.norm(reconstructed_square_root, ord="fro")),
        tiny,
    )
    bound_audit_pass = square_root_difference_computed <= (
        square_root_bound
        + SPECTRAL_RECONSTRUCTION_RELATIVE_TOLERANCE * square_root_scale
    )
    transfer_pass = (
        transfer_delta >= 0.0
        and transfer_budget >= transfer_delta
        and target_trace >= 0.0
        and reconstructed_trace >= 0.0
    )
    status = (
        "passed"
        if finite
        and residuals_pass
        and support_pass
        and bound_audit_pass
        and transfer_pass
        else "failed"
    )
    source_metadata: dict[str, float | int] = {
        "rank": int(source_states.shape[0]),
        "hypothesis_count": hypothesis_count,
        "rank_threshold": 0.0,
        "gram_lambda_min": float(raw_eigenvalues[0]),
        "gram_lambda_max": float(raw_eigenvalues[-1]),
        "weighted_gram_lambda_min": lambda_min_reconstructed,
        "weighted_gram_lambda_max": lambda_max_reconstructed,
        "prior_min": 1.0 / hypothesis_count,
        "prior_max": 1.0 / hypothesis_count,
    }
    audit: dict[str, float | int | str] = {
        "trusted_source_snapshot_sha256": _artifact_array_sha256(
            source_states
        ),
        "source_gram_lambda_min": float(raw_eigenvalues[0]),
        "source_gram_lambda_max": float(raw_eigenvalues[-1]),
        "source_gram_factorization_residual_fro_relative": (
            difference_fro_relative
        ),
        "source_gram_factorization_residual_op_relative": (
            difference_op_relative
        ),
        "source_gram_support_rank": source_support_rank,
        "srm_sqrt_reconstruction_residual_fro_relative": target_square_fro,
        "srm_sqrt_reconstruction_residual_op_relative": target_square_op,
        "srm_sqrt_support_rank": target_support_rank,
        "source_reconstructed_sqrt_residual_fro_relative": (
            reconstructed_square_fro
        ),
        "source_reconstructed_sqrt_residual_op_relative": (
            reconstructed_square_op
        ),
        "spectral_audit_status": status,
        "source_target_trace": target_trace,
        "source_reconstructed_trace": reconstructed_trace,
        "source_difference_trace_norm": difference_trace_norm,
        "source_difference_fro_norm": difference_fro_norm,
        "source_difference_op_norm": difference_op_norm,
        "source_difference_trace_norm_relative": difference_trace_relative,
        "source_difference_fro_norm_relative": difference_fro_relative,
        "source_difference_op_norm_relative": difference_op_relative,
        "source_lambda_min_target": lambda_min_target,
        "source_lambda_max_target": lambda_max_target,
        "source_lambda_min_reconstructed": lambda_min_reconstructed,
        "source_lambda_max_reconstructed": lambda_max_reconstructed,
        "source_sqrt_difference_computed_fro": (
            square_root_difference_computed
        ),
        "source_sqrt_difference_bound_trace": square_root_bound_trace,
        "source_sqrt_difference_bound_spectral": square_root_bound_spectral,
        "source_sqrt_difference_bound": square_root_bound,
        "source_transfer_delta": transfer_delta,
        "source_transfer_budget": transfer_budget,
        "source_transfer_status": status,
    }
    return source_metadata, audit


def _spectral_reconstruction_audit(
    gram: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int], dict[str, float | int | str]]:
    """Construct a zero-cutoff source and report a compatibility audit.

    The returned metadata is intentionally non-authoritative to callers that
    retain or transform the returned factor.  Such callers must first freeze a
    private snapshot and invoke ``_trusted_source_snapshot_audit`` again.
    """
    physical = np.asarray(gram, dtype=float)
    physical = (physical + physical.T) / 2.0
    source_states, _ = canonical_weighted_states(
        physical, rank_tolerance=0.0
    )
    source_metadata, audit = _trusted_source_snapshot_audit(
        physical, source_states
    )
    return source_states, source_metadata, audit


def _canonical_source_bundle(
    gram: np.ndarray,
    _pinned_source_auditor=_trusted_source_snapshot_audit,
) -> tuple[np.ndarray, dict[str, float | int], dict[str, float | int | str]]:
    """Build and audit one retained source through a definition-time core."""
    physical = np.asarray(gram, dtype=float)
    physical = (physical + physical.T) / 2.0
    source_states, _ = canonical_weighted_states(
        physical, rank_tolerance=0.0
    )
    source_metadata, audit = _pinned_source_auditor(
        physical, source_states
    )
    return source_states, source_metadata, audit


def _probability_scale(*values: float) -> float:
    """Return the true probability scale, never an unconditional unit scale."""
    finite_magnitudes = [abs(float(value)) for value in values if math.isfinite(float(value))]
    if not finite_magnitudes:
        raise ValueError("probability scale requires a finite value")
    return max(finite_magnitudes)


def _relative_enclosure_gap(lower: float, upper: float) -> float:
    """Recompute a gap relative to the enclosure's own probability scale."""
    lower_value = float(lower)
    upper_value = float(upper)
    scale = max(
        np.finfo(float).tiny,
        0.5 * (abs(lower_value) + abs(upper_value)),
    )
    return (upper_value - lower_value) / scale


def _metadata_values_match(expected: object, actual: object) -> bool:
    """Compare source metadata without hiding tiny spectra behind unit scale."""
    try:
        expected_value = float(expected)
        actual_value = float(actual)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(expected_value) or not math.isfinite(actual_value):
        return False
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        return actual_value.is_integer() and int(actual_value) == int(expected)
    scale = max(
        abs(expected_value), abs(actual_value), np.finfo(float).tiny
    )
    return abs(expected_value - actual_value) <= (
        128.0 * np.finfo(float).eps * scale
    )


def _certificate_binds_to_source(
    source_metadata: dict[str, float | int],
    certificate: dict[str, object],
) -> tuple[bool, str]:
    metadata_fields = (
        "rank",
        "hypothesis_count",
        "rank_threshold",
        "weighted_gram_lambda_min",
        "weighted_gram_lambda_max",
    )
    for field in metadata_fields:
        if field not in source_metadata or field not in certificate:
            return False, f"missing_source_metadata:{field}"
        if not _metadata_values_match(
            source_metadata[field], certificate[field]
        ):
            return False, f"source_metadata_mismatch:{field}"
    return True, "passed"


def _artifact_array_sha256(array: np.ndarray) -> str:
    """Independently fingerprint a float64 artifact and its exact shape."""
    contiguous = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    shape = ",".join(str(value) for value in contiguous.shape)
    prefix = f"float64-le|{contiguous.ndim}|{shape}|".encode("ascii")
    return hashlib.sha256(prefix + contiguous.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class _Float64ArrayAuthority:
    """Immutable exact-byte authority for one C-order float64 array."""

    shape: tuple[int, ...]
    dtype_string: str
    c_contiguous: bool
    writeable: bool
    payload: bytes
    sha256: str


def _capture_float64_array_authority(
    array: np.ndarray,
    *,
    require_readonly: bool,
) -> _Float64ArrayAuthority:
    """Capture identity before any untrusted numerical helper sees an array."""
    raw = np.asarray(array)
    if raw.dtype != np.dtype(np.float64):
        raise ValueError("authority array must have exact float64 dtype")
    if not raw.flags.c_contiguous:
        raise ValueError("authority array must be C contiguous")
    if require_readonly and raw.flags.writeable:
        raise ValueError("authority array must be readonly")
    if not np.all(np.isfinite(raw)):
        raise ValueError("authority array must be finite")
    return _Float64ArrayAuthority(
        shape=tuple(int(value) for value in raw.shape),
        dtype_string=raw.dtype.str,
        c_contiguous=bool(raw.flags.c_contiguous),
        writeable=bool(raw.flags.writeable),
        payload=raw.tobytes(order="C"),
        sha256=_artifact_array_sha256(raw),
    )


def _authority_readonly_view(
    authority: _Float64ArrayAuthority,
) -> np.ndarray:
    """Return a non-owning view backed by immutable authority bytes."""
    view = np.frombuffer(
        authority.payload,
        dtype=np.dtype(authority.dtype_string),
    ).reshape(authority.shape)
    if view.flags.writeable:
        raise AssertionError("immutable bytes unexpectedly produced a writable view")
    return view


def _authority_writable_copy(
    authority: _Float64ArrayAuthority,
) -> np.ndarray:
    """Return a disposable solver copy reconstructed only from authority bytes."""
    return np.array(
        _authority_readonly_view(authority),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def _exact_authority_audit(
    array: np.ndarray,
    authority: _Float64ArrayAuthority,
    *,
    label: str,
    helper_sha256: object | None = None,
    require_helper_sha256: bool = False,
) -> tuple[bool, str, str]:
    """Recheck every structural and exact-value identity component."""
    post_sha256 = ""
    try:
        raw = np.asarray(array)
        dtype_matches = (
            raw.dtype == np.dtype(np.float64)
            and raw.dtype.str == authority.dtype_string
        )
        shape_matches = tuple(raw.shape) == authority.shape
        layout_matches = bool(raw.flags.c_contiguous) == authority.c_contiguous
        writeable_matches = bool(raw.flags.writeable) == authority.writeable
        finite = bool(np.all(np.isfinite(raw)))
        if dtype_matches and shape_matches:
            expected = np.frombuffer(
                authority.payload,
                dtype=np.dtype(authority.dtype_string),
            ).reshape(authority.shape)
            values_match = bool(np.array_equal(raw, expected))
        else:
            values_match = False
        bytes_match = bool(
            dtype_matches
            and shape_matches
            and bool(raw.flags.c_contiguous)
            and raw.tobytes(order="C") == authority.payload
        )
        post_sha256 = _artifact_array_sha256(raw)
        hash_matches = post_sha256 == authority.sha256
        helper_hash_present = helper_sha256 is not None
        helper_hash_matches = bool(
            (not require_helper_sha256 and not helper_hash_present)
            or (
                helper_hash_present
                and str(helper_sha256) == authority.sha256
            )
        )
        if not dtype_matches:
            reason = f"{label}_dtype_mutated"
        elif not shape_matches:
            reason = f"{label}_shape_mutated"
        elif not layout_matches:
            reason = f"{label}_layout_mutated"
        elif not writeable_matches:
            reason = f"{label}_writeable_flag_mutated"
        elif not finite:
            reason = f"{label}_became_nonfinite"
        elif not values_match:
            reason = f"{label}_values_mutated"
        elif not bytes_match:
            reason = f"{label}_bytes_mutated"
        elif not hash_matches:
            reason = f"{label}_hash_mutated"
        elif require_helper_sha256 and not helper_hash_present:
            reason = f"{label}_helper_sha_missing"
        elif not helper_hash_matches:
            reason = f"{label}_helper_sha_mismatch"
        else:
            reason = "passed"
    except Exception as exc:
        reason = f"{label}_identity_audit_exception:{type(exc).__name__}"
    return reason == "passed", reason, post_sha256


def _trusted_source_snapshot_identity_audit(
    trusted_source_snapshot: np.ndarray,
    authority: _Float64ArrayAuthority,
    *,
    phase: str,
    helper_sha256: object | None = None,
    require_helper_sha256: bool = False,
) -> tuple[bool, str, dict[str, str]]:
    """Bind every source-consumer phase to the pre-helper exact authority."""
    passed, reason, post_sha256 = _exact_authority_audit(
        trusted_source_snapshot,
        authority,
        label="trusted_source_snapshot",
        helper_sha256=helper_sha256,
        require_helper_sha256=require_helper_sha256,
    )
    metrics = {
        "trusted_source_snapshot_sha256": authority.sha256,
        "trusted_source_snapshot_pre_sha256": authority.sha256,
        "trusted_source_snapshot_post_sha256": post_sha256,
        "trusted_source_snapshot_validation_status": (
            "passed" if passed else "failed"
        ),
        "trusted_source_snapshot_validation_reason": reason,
        "trusted_source_snapshot_validation_phase": str(phase),
    }
    return passed, reason, metrics


def _certificate_input_factor_audit(
    trusted_source_snapshot: np.ndarray,
    solver_input: np.ndarray,
    *,
    before_bytes: bytes,
    before_sha256: str,
    before_writeable: bool,
) -> tuple[bool, str, dict[str, float | str]]:
    """Verify that a certificate callee did not mutate its caller-owned copy."""
    metrics: dict[str, float | str] = {
        "certificate_input_factor_pre_sha256": before_sha256,
        "certificate_input_factor_post_sha256": "",
        "certificate_input_factor_max_abs_error": "",
        "certificate_input_factor_pre_writeable": str(
            bool(before_writeable)
        ).lower(),
        "certificate_input_factor_post_writeable": "",
    }
    try:
        snapshot = np.asarray(trusted_source_snapshot)
        post = np.asarray(solver_input)
        if (
            snapshot.dtype != np.dtype(np.float64)
            or not snapshot.flags.c_contiguous
            or snapshot.flags.writeable
        ):
            reason = "invalid_trusted_source_snapshot"
            metrics["certificate_input_factor_validation_status"] = "failed"
            metrics["certificate_input_factor_validation_reason"] = reason
            return False, reason, metrics
        post_hash = _artifact_array_sha256(post)
        metrics["certificate_input_factor_post_sha256"] = post_hash
        metrics["certificate_input_factor_post_writeable"] = str(
            bool(post.flags.writeable)
        ).lower()
        if post.shape == snapshot.shape:
            metrics["certificate_input_factor_max_abs_error"] = float(
                np.max(np.abs(np.asarray(post, dtype=float) - snapshot), initial=0.0)
            )
        shape_matches = post.shape == snapshot.shape
        dtype_matches = post.dtype == np.dtype(np.float64)
        layout_matches = bool(post.flags.c_contiguous)
        writeable_matches = bool(post.flags.writeable) == bool(
            before_writeable
        )
        finite = bool(np.all(np.isfinite(post)))
        values_match = bool(shape_matches and np.array_equal(post, snapshot))
        bytes_match = bool(
            shape_matches
            and dtype_matches
            and layout_matches
            and post.tobytes(order="C") == before_bytes
        )
        hash_matches = post_hash == before_sha256
        if not shape_matches:
            reason = "solver_input_shape_mutated"
        elif not dtype_matches:
            reason = "solver_input_dtype_mutated"
        elif not layout_matches:
            reason = "solver_input_layout_mutated"
        elif not writeable_matches:
            reason = "solver_input_writeable_flag_mutated"
        elif not finite:
            reason = "solver_input_became_nonfinite"
        elif not values_match:
            reason = "solver_input_values_mutated"
        elif not bytes_match:
            reason = "solver_input_bytes_mutated"
        elif not hash_matches:
            reason = "solver_input_hash_mutated"
        else:
            reason = "passed"
    except (TypeError, ValueError, OverflowError) as exc:
        reason = f"solver_input_audit_exception:{type(exc).__name__}"
    passed = reason == "passed"
    metrics["certificate_input_factor_validation_status"] = (
        "passed" if passed else "failed"
    )
    metrics["certificate_input_factor_validation_reason"] = reason
    return passed, reason, metrics


def _artifact_scalars_match(
    expected: float,
    reported: object,
    *,
    natural_scale: float,
) -> bool:
    """Compare a recomputed scalar on its probability/operator unit scale."""
    try:
        reported_value = float(reported)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(expected) or not math.isfinite(reported_value):
        return False
    scale = max(
        abs(expected),
        abs(reported_value),
        abs(float(natural_scale)),
        np.finfo(float).tiny,
    )
    return abs(expected - reported_value) <= (
        ARTIFACT_SCALAR_RELATIVE_TOLERANCE * scale
    )


class _NamespaceValidationError(ValueError):
    """Raised internally when an untrusted return namespace is malformed."""


def _is_sha256_text(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _normalize_plain_numeric_namespace(
    value: object,
    *,
    namespace: str,
    allowed_fields: frozenset[str],
    required_fields: frozenset[str],
    integer_fields: frozenset[str] = frozenset(),
    string_fields: frozenset[str] = frozenset(),
    sha256_fields: frozenset[str] = frozenset(),
    blankable_fields: frozenset[str] = frozenset(),
) -> dict[str, float | int | str]:
    """Copy an exact built-in dict into finite built-in primitive values."""
    if type(value) is not dict:
        raise _NamespaceValidationError(f"{namespace}:not_plain_dict")
    raw = value
    keys = set(dict.keys(raw))
    if any(type(key) is not str for key in keys):
        raise _NamespaceValidationError(f"{namespace}:non_string_key")
    unexpected = keys - allowed_fields
    missing = required_fields - keys
    if unexpected:
        raise _NamespaceValidationError(
            f"{namespace}:unexpected_field:{sorted(unexpected)[0]}"
        )
    if missing:
        raise _NamespaceValidationError(
            f"{namespace}:missing_field:{sorted(missing)[0]}"
        )

    normalized: dict[str, float | int | str] = {}
    for field in sorted(keys):
        item = dict.__getitem__(raw, field)
        if field in blankable_fields and item == "":
            normalized[field] = ""
        elif field in sha256_fields:
            if not _is_sha256_text(item):
                raise _NamespaceValidationError(
                    f"{namespace}:invalid_sha256:{field}"
                )
            normalized[field] = item
        elif field in string_fields:
            if type(item) is not str:
                raise _NamespaceValidationError(
                    f"{namespace}:non_plain_string:{field}"
                )
            normalized[field] = item
        elif field in integer_fields:
            if type(item) is bool or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise _NamespaceValidationError(
                    f"{namespace}:non_integer:{field}"
                )
            numeric = float(item)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise _NamespaceValidationError(
                    f"{namespace}:nonfinite_or_fractional_integer:{field}"
                )
            normalized[field] = int(numeric)
        else:
            if type(item) is bool or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise _NamespaceValidationError(
                    f"{namespace}:non_numeric:{field}"
                )
            numeric = float(item)
            if not math.isfinite(numeric):
                raise _NamespaceValidationError(
                    f"{namespace}:nonfinite:{field}"
                )
            normalized[field] = numeric
    return normalized


def _project_namespace(
    row: dict[str, float | int | str],
    namespace: dict[str, float | int | str],
    allowed_fields: frozenset[str],
) -> None:
    """Project only explicitly allowed primitive fields into a row."""
    for field in sorted(allowed_fields & set(namespace)):
        row[field] = namespace[field]


_CERTIFICATE_STRING_FIELDS = frozenset(
    {
        "certificate_artifact_schema",
        "cvxpy_version",
        "dual_status",
        "primal_status",
        "solver",
    }
)
_CERTIFICATE_SHA256_FIELDS = frozenset(
    {"source_weighted_gram_sha256", "source_weighted_states_sha256"}
)
_CERTIFICATE_INTEGER_FIELDS = frozenset(
    {"rank", "hypothesis_count", "primal_iterations", "dual_iterations"}
)
_CERTIFICATE_ARTIFACT_FIELDS = frozenset(
    {
        "repaired_primal_povm",
        "safe_dual_operator",
        "source_weighted_gram",
        "source_weighted_states",
    }
)


def _normalize_certificate_namespace(
    value: object,
) -> dict[str, object]:
    """Reject hostile mappings before any certificate lookup or iteration."""
    if type(value) is not dict:
        raise _NamespaceValidationError("certificate_namespace_invalid:not_plain_dict")
    raw = value
    keys = set(dict.keys(raw))
    if any(type(key) is not str for key in keys):
        raise _NamespaceValidationError(
            "certificate_namespace_invalid:non_string_key"
        )
    unexpected = keys - CERTIFICATE_FIELDS
    missing = CERTIFICATE_FIELDS - keys
    if unexpected:
        raise _NamespaceValidationError(
            "certificate_namespace_invalid:unexpected_field:"
            f"{sorted(unexpected)[0]}"
        )
    if missing:
        raise _NamespaceValidationError(
            "certificate_namespace_invalid:missing_field:"
            f"{sorted(missing)[0]}"
        )

    normalized: dict[str, object] = {}
    for field in sorted(keys):
        item = dict.__getitem__(raw, field)
        if field in _CERTIFICATE_ARTIFACT_FIELDS:
            normalized[field] = item
        elif field in _CERTIFICATE_SHA256_FIELDS:
            if not _is_sha256_text(item):
                raise _NamespaceValidationError(
                    f"certificate_namespace_invalid:invalid_sha256:{field}"
                )
            normalized[field] = item
        elif field in _CERTIFICATE_STRING_FIELDS:
            if type(item) is not str:
                raise _NamespaceValidationError(
                    f"certificate_namespace_invalid:non_plain_string:{field}"
                )
            normalized[field] = item
        elif field in _CERTIFICATE_INTEGER_FIELDS:
            if type(item) is bool or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise _NamespaceValidationError(
                    f"certificate_namespace_invalid:non_integer:{field}"
                )
            numeric = float(item)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise _NamespaceValidationError(
                    "certificate_namespace_invalid:"
                    f"nonfinite_or_fractional_integer:{field}"
                )
            normalized[field] = int(numeric)
        else:
            if type(item) is bool or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise _NamespaceValidationError(
                    f"certificate_namespace_invalid:non_numeric:{field}"
                )
            numeric = float(item)
            if not math.isfinite(numeric):
                raise _NamespaceValidationError(
                    f"certificate_namespace_invalid:nonfinite:{field}"
                )
            normalized[field] = numeric
    return normalized


def _independently_validate_certificate_artifacts(
    trusted_source_states: np.ndarray,
    solver_input: np.ndarray,
    certificate: dict[str, object],
) -> tuple[bool, str, dict[str, float | str]]:
    """Recompute every advertised feasible artifact against retained ``S``."""
    metrics: dict[str, float | str] = {}

    def reject(reason: str) -> tuple[bool, str, dict[str, float | str]]:
        return False, reason, metrics

    try:
        trusted = np.asarray(trusted_source_states)
        if np.iscomplexobj(trusted):
            return reject("trusted_source_is_complex")
        trusted = np.asarray(trusted, dtype=float)
        if trusted.ndim != 2 or not np.all(np.isfinite(trusted)):
            return reject("invalid_trusted_source")
        caller_solver_input = np.asarray(solver_input)
        rank, hypothesis_count = trusted.shape
        if np.iscomplexobj(caller_solver_input):
            return reject("solver_input_is_complex")
        if (
            caller_solver_input.shape != trusted.shape
            or caller_solver_input.dtype != np.dtype(np.float64)
            or not np.all(np.isfinite(caller_solver_input))
        ):
            return reject("solver_input_malformed")
        if not np.array_equal(caller_solver_input, trusted):
            return reject("solver_input_not_identical_to_retained_source")

        def aliases_caller_storage(array: np.ndarray) -> bool:
            return bool(
                np.shares_memory(array, trusted)
                or np.shares_memory(array, caller_solver_input)
            )

        raw_source_artifact = certificate["source_weighted_states"]
        artifact_raw = np.asarray(raw_source_artifact)
        if aliases_caller_storage(artifact_raw):
            return reject("source_states_artifact_aliases_caller_storage")
        if np.iscomplexobj(artifact_raw):
            return reject("source_states_are_complex")
        artifact_states = np.asarray(artifact_raw, dtype=float)
        if artifact_states.shape != trusted.shape:
            return reject("source_states_shape_mismatch")
        if not np.all(np.isfinite(artifact_states)):
            return reject("source_states_nonfinite")
        source_states_error = float(
            np.max(np.abs(artifact_states - trusted), initial=0.0)
        )
        metrics["certificate_artifact_source_states_max_abs_error"] = (
            source_states_error
        )
        if not np.array_equal(artifact_states, trusted):
            return reject("source_states_not_identical_to_retained_factor")

        source_states_sha256 = _artifact_array_sha256(artifact_states)
        metrics["certificate_artifact_source_states_sha256"] = (
            source_states_sha256
        )
        if (
            str(certificate.get("source_weighted_states_sha256", ""))
            != source_states_sha256
        ):
            return reject("source_states_fingerprint_mismatch")

        gram_raw = np.asarray(certificate["source_weighted_gram"])
        if aliases_caller_storage(gram_raw):
            return reject("source_gram_artifact_aliases_caller_storage")
        if np.iscomplexobj(gram_raw):
            return reject("source_gram_is_complex")
        artifact_gram = np.asarray(gram_raw, dtype=float)
        if artifact_gram.shape != (hypothesis_count, hypothesis_count):
            return reject("source_gram_shape_mismatch")
        if not np.all(np.isfinite(artifact_gram)):
            return reject("source_gram_nonfinite")
        expected_gram = trusted.T @ trusted
        expected_gram = (expected_gram + expected_gram.T) / 2.0
        source_gram_error = float(
            np.max(np.abs(artifact_gram - expected_gram), initial=0.0)
        )
        metrics["certificate_artifact_source_gram_max_abs_error"] = (
            source_gram_error
        )
        gram_scale = max(
            float(np.linalg.norm(expected_gram, ord=2)),
            np.finfo(float).tiny,
        )
        gram_tolerance = (
            ARTIFACT_SCALAR_RELATIVE_TOLERANCE
            * max(1, hypothesis_count)
            * gram_scale
        )
        if source_gram_error > gram_tolerance:
            return reject("source_gram_not_bound_to_retained_factor")
        gram_symmetry = float(
            np.linalg.norm(artifact_gram - artifact_gram.T, ord="fro")
        )
        if gram_symmetry > gram_tolerance:
            return reject("source_gram_not_symmetric")
        source_gram_sha256 = _artifact_array_sha256(artifact_gram)
        metrics["certificate_artifact_source_gram_sha256"] = (
            source_gram_sha256
        )
        if (
            str(certificate.get("source_weighted_gram_sha256", ""))
            != source_gram_sha256
        ):
            return reject("source_gram_fingerprint_mismatch")

        raw_effects = certificate["repaired_primal_povm"]
        if type(raw_effects) is not tuple:
            return reject("primal_povm_malformed:expected_tuple")
        if len(raw_effects) != hypothesis_count:
            return reject("primal_povm_outcome_count_mismatch")
        raw_effect_matrices: list[np.ndarray] = []
        primal_symmetry = 0.0
        for raw_effect in raw_effects:
            effect_raw = np.asarray(raw_effect)
            if aliases_caller_storage(effect_raw):
                return reject("primal_effect_artifact_aliases_caller_storage")
            if type(raw_effect) is not np.ndarray:
                return reject("primal_effect_malformed:expected_ndarray")
            if np.iscomplexobj(effect_raw):
                return reject("primal_effect_is_complex")
            effect = np.asarray(effect_raw, dtype=float)
            if effect.shape != (rank, rank):
                return reject("primal_effect_shape_mismatch")
            if not np.all(np.isfinite(effect)):
                return reject("primal_effect_nonfinite")
            primal_symmetry = max(
                primal_symmetry,
                float(np.linalg.norm(effect - effect.T, ord="fro")),
            )
            raw_effect_matrices.append(effect)
        metrics["certificate_artifact_primal_symmetry_error_fro"] = (
            primal_symmetry
        )
        if primal_symmetry > COMPLETENESS_RESIDUAL_TOLERANCE:
            return reject("primal_effect_symmetry_residual")

        identity = np.eye(rank)
        raw_completeness = (
            sum(raw_effect_matrices, np.zeros((rank, rank), dtype=float))
            - identity
        )
        raw_completeness_fro = float(
            np.linalg.norm(raw_completeness, ord="fro")
        )
        raw_completeness_op = float(np.linalg.norm(raw_completeness, ord=2))
        raw_primal_minimum = min(
            float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
            for effect in raw_effect_matrices
        )
        raw_primal_violation = max(0.0, -raw_primal_minimum)
        rhos = [
            np.outer(trusted[:, index], trusted[:, index])
            for index in range(hypothesis_count)
        ]
        raw_primal_objective = math.fsum(
            float(np.trace(rho @ effect))
            for rho, effect in zip(rhos, raw_effect_matrices)
        )

        # Re-feasibilize from the actual effect artifacts.  Small residuals
        # are diagnostics only: they are never treated as feasibility.  Each
        # symmetric effect is projected to the PSD cone, all effects are
        # contracted by alpha >= lambda_max(sum F_j), and the PSD remainder is
        # assigned deterministically to the first outcome.
        projected_effects: list[np.ndarray] = []
        for effect in raw_effect_matrices:
            symmetric_effect = (effect + effect.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_effect)
            projected = (
                (eigenvectors * np.maximum(eigenvalues, 0.0)[None, :])
                @ eigenvectors.T
            )
            projected = (projected + projected.T) / 2.0
            projected_floor = (
                256.0
                * np.finfo(float).eps
                * max(1, rank)
                * max(1.0, float(np.linalg.norm(projected, ord=2)))
            )
            projected_effects.append(projected + projected_floor * identity)
        projected_total = (
            sum(projected_effects, np.zeros((rank, rank), dtype=float))
        )
        projected_total = (projected_total + projected_total.T) / 2.0
        alpha = max(
            1.0, float(np.linalg.eigvalsh(projected_total)[-1])
        )
        alpha = float(np.nextafter(alpha, math.inf))
        effects = [
            (effect / alpha + (effect / alpha).T) / 2.0
            for effect in projected_effects
        ]
        normalized_total = sum(
            effects, np.zeros((rank, rank), dtype=float)
        )
        normalized_total = (normalized_total + normalized_total.T) / 2.0
        remainder = (identity - normalized_total)
        remainder = (remainder + remainder.T) / 2.0
        remainder_minimum = float(np.linalg.eigvalsh(remainder)[0])
        if remainder_minimum < 0.0:
            scale = max(
                1.0,
                float(np.linalg.norm(projected_total, ord=2)),
            )
            correction = (
                -remainder_minimum
                + 256.0 * np.finfo(float).eps * max(1, rank) * scale
            )
            alpha = float(
                np.nextafter(alpha + correction * max(1.0, alpha), math.inf)
            )
            effects = [
                (effect / alpha + (effect / alpha).T) / 2.0
                for effect in projected_effects
            ]
            normalized_total = sum(
                effects, np.zeros((rank, rank), dtype=float)
            )
            normalized_total = (normalized_total + normalized_total.T) / 2.0
            remainder = identity - normalized_total
            remainder = (remainder + remainder.T) / 2.0
        effects[0] = (effects[0] + remainder)
        effects[0] = (effects[0] + effects[0].T) / 2.0

        completeness = (
            sum(effects, np.zeros((rank, rank), dtype=float)) - identity
        )
        completeness_fro = float(np.linalg.norm(completeness, ord="fro"))
        completeness_op = float(np.linalg.norm(completeness, ord=2))
        primal_minimum = min(
            float(np.linalg.eigvalsh((effect + effect.T) / 2.0)[0])
            for effect in effects
        )
        primal_violation = max(0.0, -primal_minimum)
        primal_objective = float(
            np.nextafter(
                math.fsum(
                    float(np.trace(rho @ effect))
                    for rho, effect in zip(rhos, effects)
                ),
                -math.inf,
            )
        )
        metrics.update(
            {
                "certificate_artifact_primal_completeness_fro": (
                    completeness_fro
                ),
                "certificate_artifact_primal_completeness_op": (
                    completeness_op
                ),
                "certificate_artifact_primal_psd_violation": (
                    primal_violation
                ),
                "certificate_artifact_primal_min_eigenvalue": (
                    primal_minimum
                ),
                "certificate_artifact_primal_objective": primal_objective,
            }
        )
        if primal_violation > PRIMAL_PSD_RESIDUAL_TOLERANCE:
            return reject("primal_repair_failed:psd_residual")
        if (
            completeness_fro > COMPLETENESS_RESIDUAL_TOLERANCE
            or completeness_op > COMPLETENESS_RESIDUAL_TOLERANCE
        ):
            return reject("primal_repair_failed:completeness_residual")

        dual_raw = np.asarray(certificate["safe_dual_operator"])
        if aliases_caller_storage(dual_raw):
            return reject("dual_operator_artifact_aliases_caller_storage")
        if np.iscomplexobj(dual_raw):
            return reject("dual_operator_is_complex")
        dual = np.asarray(dual_raw, dtype=float)
        if dual.shape != (rank, rank):
            return reject("dual_operator_shape_mismatch")
        if not np.all(np.isfinite(dual)):
            return reject("dual_operator_nonfinite")
        dual_symmetry = float(np.linalg.norm(dual - dual.T, ord="fro"))
        dual_symmetric = (dual + dual.T) / 2.0
        raw_dual_trace = float(np.trace(dual_symmetric))
        raw_slack_minima = [
            float(
                np.linalg.eigvalsh(
                    (dual_symmetric - rho + (dual_symmetric - rho).T)
                    / 2.0
                )[0]
            )
            for rho in rhos
        ]
        raw_dual_minimum_slack = min(raw_slack_minima)
        raw_dual_violation = max(0.0, -raw_dual_minimum_slack)

        # Recompute the exact algebraic shift from the actual slack matrices.
        # The outward representable step and recheck are numerical safeguards,
        # not a directed-rounding or interval-arithmetic certificate.
        dual_shift = max(0.0, -raw_dual_minimum_slack)
        if dual_shift > 0.0:
            dual_shift = float(np.nextafter(dual_shift, math.inf))
        dual_safe = dual_symmetric + dual_shift * identity
        dual_minimum_slack = min(
            float(np.linalg.eigvalsh(dual_safe - rho)[0]) for rho in rhos
        )
        if dual_minimum_slack < 0.0:
            dual_shift = float(
                np.nextafter(
                    dual_shift - dual_minimum_slack
                    + 256.0
                    * np.finfo(float).eps
                    * max(1, rank)
                    * max(1.0, float(np.linalg.norm(dual_symmetric, ord=2))),
                    math.inf,
                )
            )
            dual_safe = dual_symmetric + dual_shift * identity
            dual_minimum_slack = min(
                float(np.linalg.eigvalsh(dual_safe - rho)[0])
                for rho in rhos
            )
        dual_violation = max(0.0, -dual_minimum_slack)
        dual_trace = float(
            np.nextafter(
                raw_dual_trace + rank * dual_shift,
                math.inf,
            )
        )
        metrics.update(
            {
                "certificate_artifact_dual_symmetry_error_fro": (
                    dual_symmetry
                ),
                "certificate_artifact_dual_min_slack": dual_minimum_slack,
                "certificate_artifact_dual_psd_violation": dual_violation,
                "certificate_artifact_dual_trace": dual_trace,
            }
        )
        if dual_symmetry > COMPLETENESS_RESIDUAL_TOLERANCE:
            return reject("dual_artifact_symmetry_residual")
        if dual_minimum_slack < 0.0 or dual_violation > 0.0:
            return reject("dual_repair_failed:negative_slack")

        probability_scale = _probability_scale(
            raw_primal_objective,
            raw_dual_trace,
            1.0 / hypothesis_count,
        )
        scalar_checks = (
            (
                "primal_feasible_objective",
                raw_primal_objective,
                probability_scale,
            ),
            (
                "dual_feasible_objective",
                raw_dual_trace,
                probability_scale,
            ),
            (
                "primal_feasible_equality_residual_fro",
                raw_completeness_fro,
                1.0,
            ),
            (
                "primal_feasible_equality_residual_op",
                raw_completeness_op,
                1.0,
            ),
            (
                "primal_feasible_psd_violation",
                raw_primal_violation,
                1.0,
            ),
            (
                "primal_feasible_min_eigenvalue",
                raw_primal_minimum,
                1.0,
            ),
            (
                "dual_feasible_min_slack",
                raw_dual_minimum_slack,
                1.0,
            ),
            (
                "dual_feasible_psd_violation",
                raw_dual_violation,
                1.0,
            ),
        )
        for field, recomputed, scale in scalar_checks:
            if not _artifact_scalars_match(
                recomputed,
                certificate.get(field),
                natural_scale=scale,
            ):
                return reject(f"artifact_scalar_mismatch:{field}")

        raw_gap = raw_dual_trace - raw_primal_objective
        if not _artifact_scalars_match(
            raw_gap,
            certificate.get("feasible_bound_gap"),
            natural_scale=probability_scale,
        ):
            return reject("artifact_scalar_mismatch:feasible_bound_gap")
    except Exception as exc:
        return reject(f"invalid_certificate_artifact:{type(exc).__name__}")

    metrics["certificate_artifact_schema"] = str(
        certificate.get("certificate_artifact_schema", "")
    )
    if metrics["certificate_artifact_schema"] != (
        "weighted_state_certificate_v1"
    ):
        return reject("artifact_schema_mismatch")
    return True, "passed", metrics


def _validate_certificate_artifacts(
    trusted_source_states: np.ndarray,
    solver_input: np.ndarray,
    certificate: dict[str, object],
) -> tuple[bool, str, dict[str, float | str]]:
    """Patchable validation facade; orchestration also recomputes independently."""
    return _independently_validate_certificate_artifacts(
        trusted_source_states,
        solver_input,
        certificate,
    )


def _artifact_metric_ledgers_match(
    reported: dict[str, float | int | str],
    independent: dict[str, float | int | str],
    *,
    hypothesis_count: int,
) -> tuple[bool, str]:
    """Bind a validator ledger to an independent pass over actual artifacts."""
    if set(reported) != ARTIFACT_METRIC_FIELDS:
        return False, "artifact_namespace_invalid:reported_key_set"
    if set(independent) != ARTIFACT_METRIC_FIELDS:
        return False, "artifact_namespace_invalid:independent_key_set"
    probability_scale = _probability_scale(
        float(independent["certificate_artifact_primal_objective"]),
        float(independent["certificate_artifact_dual_trace"]),
        1.0 / hypothesis_count,
    )
    probability_fields = {
        "certificate_artifact_primal_objective",
        "certificate_artifact_dual_trace",
    }
    for field in sorted(ARTIFACT_METRIC_FIELDS):
        expected = independent[field]
        actual = reported[field]
        if field in {
            "certificate_artifact_schema",
            "certificate_artifact_source_states_sha256",
            "certificate_artifact_source_gram_sha256",
        }:
            if type(actual) is not str or actual != expected:
                suffix = (
                    ":trusted_snapshot_sha"
                    if field
                    == "certificate_artifact_source_states_sha256"
                    else ""
                )
                return False, f"artifact_metrics_mismatch:{field}{suffix}"
            continue
        scale = probability_scale if field in probability_fields else 1.0
        if not _artifact_scalars_match(
            float(expected),
            actual,
            natural_scale=scale,
        ):
            return False, f"artifact_metrics_mismatch:{field}"
    return True, "passed"


def _validate_safe_certificate(
    certificate: dict[str, object],
    *,
    hypothesis_count: int,
    trace_lower: float | None,
    srm_probability: float | None,
) -> tuple[bool, str, dict[str, float | int]]:
    """Validate the floating-point enclosure before exposing safe fields."""
    for key, raw_value in certificate.items():
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            try:
                finite = math.isfinite(float(raw_value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                return False, f"nonfinite_certificate_field:{key}", {}
    required_numeric = (
        "rank",
        "hypothesis_count",
        "rank_threshold",
        "primal_feasible_objective",
        "dual_feasible_objective",
        "feasible_bound_gap",
        "relative_feasible_bound_gap",
        "primal_feasible_equality_residual_fro",
        "primal_feasible_equality_residual_op",
        "primal_feasible_psd_violation",
        "primal_feasible_min_eigenvalue",
        "dual_feasible_min_slack",
        "dual_feasible_psd_violation",
    )
    try:
        values = {key: float(certificate[key]) for key in required_numeric}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return False, f"missing_or_invalid_numeric_field:{type(exc).__name__}", {}
    if not all(math.isfinite(value) for value in values.values()):
        return False, "nonfinite_certificate_field", {}

    rank = values["rank"]
    reported_count = values["hypothesis_count"]
    if rank != float(hypothesis_count) or not rank.is_integer():
        return False, "rank_loss", {}
    if reported_count != float(hypothesis_count) or not reported_count.is_integer():
        return False, "hypothesis_count_mismatch", {}
    if values["rank_threshold"] != 0.0:
        return False, "nonzero_rank_threshold", {}

    accepted_statuses = {"optimal", "optimal_inaccurate"}
    primal_status = str(certificate.get("primal_status", "")).lower()
    dual_status = str(certificate.get("dual_status", "")).lower()
    if primal_status not in accepted_statuses or dual_status not in accepted_statuses:
        return False, "unacceptable_solver_status", {}

    lower = values["primal_feasible_objective"]
    upper = values["dual_feasible_objective"]
    scale_inputs = [lower, upper, 1.0 / hypothesis_count]
    if srm_probability is not None:
        scale_inputs.append(float(srm_probability))
    probability_scale = _probability_scale(*scale_inputs)
    probability_tolerance = PROBABILITY_RELATIVE_TOLERANCE * probability_scale
    dual_psd_tolerance = (
        DUAL_PSD_RELATIVE_TOLERANCE / float(hypothesis_count)
    )
    metrics: dict[str, float | int] = {
        "certificate_rank": int(rank),
        "certificate_expected_rank": hypothesis_count,
        "certificate_rank_tolerance": 0.0,
        "certificate_probability_scale": probability_scale,
        "certificate_probability_tolerance": probability_tolerance,
        "certificate_residual_tolerance": COMPLETENESS_RESIDUAL_TOLERANCE,
        "certificate_completeness_tolerance": COMPLETENESS_RESIDUAL_TOLERANCE,
        "certificate_primal_psd_tolerance": PRIMAL_PSD_RESIDUAL_TOLERANCE,
        "certificate_dual_psd_tolerance": dual_psd_tolerance,
    }

    if (
        values["primal_feasible_equality_residual_fro"]
        > COMPLETENESS_RESIDUAL_TOLERANCE
        or values["primal_feasible_equality_residual_op"]
        > COMPLETENESS_RESIDUAL_TOLERANCE
    ):
        return False, "primal_completeness_residual", metrics
    if (
        values["primal_feasible_psd_violation"]
        > PRIMAL_PSD_RESIDUAL_TOLERANCE
        or values["primal_feasible_psd_violation"] < 0.0
        or values["primal_feasible_min_eigenvalue"]
        < -PRIMAL_PSD_RESIDUAL_TOLERANCE
    ):
        return False, "primal_psd_residual", metrics
    if (
        values["dual_feasible_psd_violation"] > dual_psd_tolerance
        or values["dual_feasible_psd_violation"] < 0.0
        or values["dual_feasible_min_slack"] < -dual_psd_tolerance
    ):
        return False, "dual_slack_residual", metrics
    if abs(values["feasible_bound_gap"] - (upper - lower)) > probability_tolerance:
        return False, "inconsistent_reported_gap", metrics
    if lower < -probability_tolerance or upper > 1.0 + probability_tolerance:
        return False, "probability_range_violation", metrics
    if lower > upper + probability_tolerance:
        return False, "lower_above_upper", metrics
    if trace_lower is not None and srm_probability is not None:
        if float(trace_lower) > float(srm_probability) + probability_tolerance:
            return False, "trace_lower_above_srm", metrics
    if srm_probability is not None:
        if float(srm_probability) > upper + probability_tolerance:
            return False, "srm_above_upper", metrics
    metrics.update(
        {
            "validated_lower": lower,
            "validated_upper": upper,
        }
    )
    return True, "passed", metrics


def _blank_row(
    n: int,
    c: float,
    schedule: str,
    estimate: dict[str, int | str],
    solver: str,
) -> dict[str, float | int | str]:
    probability = p1(c)
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule": str(schedule),
        "lambda_target": "",
        "n": n,
        "external_M_n": estimate["external_M_n"],
        "c": c,
        "lambda": n * (1.0 - c),
        "h": n * probability,
        "p1_squared": probability**2,
        "P_tr": "",
        "P_SRM": "",
        "repaired_primal_value": "",
        "shifted_dual_value": "",
        "strongest_measurement_value": "",
        "floating_primal_dual_span": "",
        "floating_primal_dual_span_relative": "",
        "repaired_primal_over_p1_squared": "",
        "shifted_dual_over_p1_squared": "",
        "strongest_measurement_over_p1_squared": "",
        "repaired_primal_over_P_SRM": "",
        "P_SRM_over_shifted_dual": "",
        "primal_completeness_residual_fro": "",
        "primal_completeness_residual_op": "",
        "primal_psd_violation": "",
        "primal_min_eigenvalue": "",
        "dual_min_slack_after_shift": "",
        "dual_psd_violation_after_shift": "",
        "artifact_recomputation_status": "not_computed",
        "artifact_recomputation_reason": "not_computed",
        "diagnostic_kind": "not_computed",
        "legacy_bound_fields_status": "deprecated_blank",
        "P_SRM_transfer_budget": "",
        "P_SRM_safe_lower": "",
        "L_raw": "",
        "U_raw": "",
        "L_safe": "",
        "U_safe": "",
        "strongest_certified_lower": "",
        "P_tr_over_p1_squared": "",
        "P_SRM_over_p1_squared": "",
        "L_safe_over_p1_squared": "",
        "U_safe_over_p1_squared": "",
        "strongest_lower_over_p1_squared": "",
        "P_tr_over_P_SRM": "",
        "L_safe_over_P_SRM": "",
        "P_SRM_over_U_safe": "",
        "L_safe_over_U_safe": "",
        "raw_safe_gap": "",
        "raw_safe_gap_relative": "",
        "safe_gap": "",
        "safe_gap_relative": "",
        "primal_feasible_equality_residual_fro": "",
        "primal_feasible_equality_residual_op": "",
        "primal_feasible_psd_violation": "",
        "primal_feasible_min_eigenvalue": "",
        "dual_feasible_min_slack": "",
        "dual_feasible_psd_violation": "",
        "P_opt": "",
        "P_opt_status": "not_computed",
        "srm_status": "not_computed",
        "sdp_status": "not_computed",
        "primal_status": "not_computed",
        "dual_status": "not_computed",
        "status": "not_computed",
        "solver": solver.upper(),
        "cvxpy_version": "",
        "gram_bytes": estimate["gram_bytes"],
        "estimated_peak_bytes": estimate["estimated_peak_bytes"],
        "estimated_dense_array_count": estimate["estimated_dense_array_count"],
        "estimated_peak_scope": estimate["estimated_peak_scope"],
        "eigensolver_dimension": estimate["eigensolver_dimension"],
        "eigensolver_cubic_work_units": estimate[
            "eigensolver_cubic_work_units"
        ],
        "source_gram_lambda_min": "",
        "source_gram_lambda_max": "",
        "source_gram_factorization_residual_fro_relative": "",
        "source_gram_factorization_residual_op_relative": "",
        "source_gram_support_rank": "",
        "trusted_source_snapshot_sha256": "",
        "trusted_source_snapshot_pre_sha256": "",
        "trusted_source_snapshot_post_sha256": "",
        "trusted_source_snapshot_validation_status": "not_computed",
        "trusted_source_snapshot_validation_reason": "not_computed",
        "trusted_source_snapshot_validation_phase": "not_computed",
        "srm_sqrt_reconstruction_residual_fro_relative": "",
        "srm_sqrt_reconstruction_residual_op_relative": "",
        "srm_sqrt_support_rank": "",
        "source_reconstructed_sqrt_residual_fro_relative": "",
        "source_reconstructed_sqrt_residual_op_relative": "",
        "spectral_audit_status": "not_computed",
        "source_target_trace": "",
        "source_reconstructed_trace": "",
        "source_difference_trace_norm": "",
        "source_difference_fro_norm": "",
        "source_difference_op_norm": "",
        "source_difference_trace_norm_relative": "",
        "source_difference_fro_norm_relative": "",
        "source_difference_op_norm_relative": "",
        "source_lambda_min_target": "",
        "source_lambda_max_target": "",
        "source_lambda_min_reconstructed": "",
        "source_lambda_max_reconstructed": "",
        "source_sqrt_difference_computed_fro": "",
        "source_sqrt_difference_bound_trace": "",
        "source_sqrt_difference_bound_spectral": "",
        "source_sqrt_difference_bound": "",
        "source_transfer_delta": "",
        "source_transfer_budget": "",
        "source_transfer_status": "not_computed",
        "hull_subtraction_max_abs_error": "",
        "hull_subtraction_max_elementwise_envelope": "",
        "hull_subtraction_fro_error": "",
        "hull_subtraction_fro_envelope": "",
        "hull_subtraction_audit_status": "not_computed",
        "hull_subtraction_audit_kind": "not_computed",
        "certificate_rank": "",
        "certificate_expected_rank": "",
        "certificate_rank_tolerance": "",
        "certificate_probability_scale": "",
        "certificate_probability_tolerance": "",
        "certificate_residual_tolerance": "",
        "certificate_completeness_tolerance": "",
        "certificate_primal_psd_tolerance": "",
        "certificate_dual_psd_tolerance": "",
        "certificate_validation_status": "",
        "certificate_validation_reason": "",
        "certificate_input_factor_pre_sha256": "",
        "certificate_input_factor_post_sha256": "",
        "certificate_input_factor_max_abs_error": "",
        "certificate_input_factor_pre_writeable": "",
        "certificate_input_factor_post_writeable": "",
        "certificate_input_factor_validation_status": "",
        "certificate_input_factor_validation_reason": "",
        "certificate_source_binding_status": "",
        "certificate_source_binding_reason": "",
        "certificate_artifact_schema": "",
        "certificate_artifact_source_states_sha256": "",
        "certificate_artifact_source_gram_sha256": "",
        "certificate_artifact_source_states_max_abs_error": "",
        "certificate_artifact_source_gram_max_abs_error": "",
        "certificate_artifact_primal_symmetry_error_fro": "",
        "certificate_artifact_primal_completeness_fro": "",
        "certificate_artifact_primal_completeness_op": "",
        "certificate_artifact_primal_psd_violation": "",
        "certificate_artifact_primal_min_eigenvalue": "",
        "certificate_artifact_primal_objective": "",
        "certificate_artifact_dual_symmetry_error_fro": "",
        "certificate_artifact_dual_min_slack": "",
        "certificate_artifact_dual_psd_violation": "",
        "certificate_artifact_dual_trace": "",
        "certificate_artifact_validation_status": "",
        "certificate_artifact_validation_reason": "",
        "enclosure_kind": "",
        "numeric_guarantee": (
            "ieee_double_residual_checked_not_directed_rounding_proof"
        ),
        "diagnostic_grid_scope": DIAGNOSTIC_GRID_SCOPE,
        "interpretation": "finite_size_diagnostic_not_proof",
    }


def _ratio(numerator: float | int | str, denominator: float | int | str):
    if numerator == "" or denominator == "":
        return ""
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return ""
    return float(numerator) / denominator_float


def _populate_ratios(row: dict[str, float | int | str]) -> None:
    target = row["p1_squared"]
    for value_field, ratio_field in (
        ("P_tr", "P_tr_over_p1_squared"),
        ("P_SRM", "P_SRM_over_p1_squared"),
        ("L_safe", "L_safe_over_p1_squared"),
        ("U_safe", "U_safe_over_p1_squared"),
        (
            "strongest_certified_lower",
            "strongest_lower_over_p1_squared",
        ),
    ):
        row[ratio_field] = _ratio(row[value_field], target)
    row["P_tr_over_P_SRM"] = _ratio(row["P_tr"], row["P_SRM"])
    row["L_safe_over_P_SRM"] = _ratio(row["L_safe"], row["P_SRM"])
    row["P_SRM_over_U_safe"] = _ratio(row["P_SRM"], row["U_safe"])
    row["L_safe_over_U_safe"] = _ratio(row["L_safe"], row["U_safe"])


_SDP_CLEAR_FIELDS = frozenset(
    {
        "L_raw",
        "U_raw",
        "L_safe",
        "U_safe",
        "raw_safe_gap",
        "raw_safe_gap_relative",
        "safe_gap",
        "safe_gap_relative",
        "primal_feasible_equality_residual_fro",
        "primal_feasible_equality_residual_op",
        "primal_feasible_psd_violation",
        "primal_feasible_min_eigenvalue",
        "dual_feasible_min_slack",
        "dual_feasible_psd_violation",
    }
    | ARTIFACT_METRIC_FIELDS
)


def _fail_closed_namespace(
    row: dict[str, float | int | str],
    *,
    failure: str,
    reason: str,
    need_srm: bool,
    need_sdp: bool,
) -> dict[str, float | int | str]:
    """Return a primitive-only row after clearing every SDP-derived value."""
    for field in _SDP_CLEAR_FIELDS:
        row[field] = ""
    row["P_opt"] = ""
    row["P_opt_status"] = "not_computed"
    if need_srm and row["P_SRM"] == "":
        row["srm_status"] = failure
    if need_sdp:
        row["sdp_status"] = failure
        row["primal_status"] = failure
        row["dual_status"] = failure
    row["certificate_validation_status"] = "failed"
    row["certificate_validation_reason"] = reason
    row["certificate_source_binding_status"] = failure
    row["certificate_source_binding_reason"] = reason
    row["certificate_artifact_validation_status"] = "failed"
    row["certificate_artifact_validation_reason"] = reason
    row["enclosure_kind"] = failure
    row["status"] = (
        "srm_only_certificate_failed" if row["P_SRM"] != "" else failure
    )
    _populate_ratios(row)
    return row


def _fail_closed_trusted_source_snapshot(
    row: dict[str, float | int | str],
    *,
    need_srm: bool,
    need_sdp: bool,
    reason: str,
    identity_metrics: dict[str, str],
) -> dict[str, float | int | str]:
    """Erase enclosure outputs when the caller-owned source authority drifts."""
    failure = "not_computed_trusted_source_snapshot_mutated"
    _project_namespace(row, identity_metrics, SOURCE_IDENTITY_FIELDS)
    row["spectral_audit_status"] = "failed"
    row["source_transfer_status"] = "failed"
    return _fail_closed_namespace(
        row,
        failure=failure,
        reason=reason,
        need_srm=need_srm,
        need_sdp=need_sdp,
    )


def _normalize_hull_result(
    result: object,
    *,
    hypothesis_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if type(result) is not tuple or len(result) != 3:
        raise _NamespaceValidationError("hull_namespace_invalid:malformed_tuple")
    stable, physical, audit = result
    for label, matrix in (("stable", stable), ("physical", physical)):
        if type(matrix) is not np.ndarray:
            raise _NamespaceValidationError(
                f"hull_namespace_invalid:{label}_not_ndarray"
            )
        if (
            matrix.shape != (hypothesis_count, hypothesis_count)
            or matrix.dtype != np.dtype(np.float64)
            or not np.all(np.isfinite(matrix))
        ):
            raise _NamespaceValidationError(
                f"hull_namespace_invalid:{label}_matrix_invalid"
            )
    normalized_audit = _normalize_plain_numeric_namespace(
        audit,
        namespace="hull_namespace_invalid",
        allowed_fields=HULL_AUDIT_FIELDS,
        required_fields=HULL_AUDIT_FIELDS,
        string_fields=frozenset(
            {"hull_subtraction_audit_status", "hull_subtraction_audit_kind"}
        ),
    )
    return stable, physical, normalized_audit


def _normalize_source_authority_result(
    result: object,
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    if type(result) is not tuple or len(result) != 2:
        raise _NamespaceValidationError("source_namespace_invalid:malformed_tuple")
    metadata, audit = result
    normalized_metadata = _normalize_plain_numeric_namespace(
        metadata,
        namespace="source_namespace_invalid",
        allowed_fields=SOURCE_METADATA_FIELDS,
        required_fields=SOURCE_METADATA_FIELDS,
        integer_fields=frozenset({"rank", "hypothesis_count"}),
    )
    normalized_audit = _normalize_plain_numeric_namespace(
        audit,
        namespace="source_namespace_invalid",
        allowed_fields=SPECTRAL_AUDIT_FIELDS,
        required_fields=SPECTRAL_AUDIT_FIELDS,
        integer_fields=frozenset(
            {"source_gram_support_rank", "srm_sqrt_support_rank"}
        ),
        string_fields=frozenset(
            {"spectral_audit_status", "source_transfer_status"}
        ),
        sha256_fields=frozenset({"trusted_source_snapshot_sha256"}),
    )
    return normalized_metadata, normalized_audit


def _source_authority_ledgers_match(
    reported_metadata: dict[str, float | int | str],
    reported_audit: dict[str, float | int | str],
    canonical_metadata: dict[str, float | int | str],
    canonical_audit: dict[str, float | int | str],
) -> tuple[bool, str]:
    """Compare a patchable source report with a caller-pinned recomputation."""
    if set(reported_metadata) != SOURCE_METADATA_FIELDS:
        return False, "source_transfer_ledger_mismatch:reported_metadata_keys"
    if set(canonical_metadata) != SOURCE_METADATA_FIELDS:
        return False, "source_transfer_ledger_mismatch:canonical_metadata_keys"
    if set(reported_audit) != SPECTRAL_AUDIT_FIELDS:
        return False, "source_transfer_ledger_mismatch:reported_audit_keys"
    if set(canonical_audit) != SPECTRAL_AUDIT_FIELDS:
        return False, "source_transfer_ledger_mismatch:canonical_audit_keys"

    for field in sorted(SOURCE_METADATA_FIELDS):
        if not _metadata_values_match(
            canonical_metadata[field], reported_metadata[field]
        ):
            return False, f"source_transfer_ledger_mismatch:{field}"

    exact_text_fields = {
        "trusted_source_snapshot_sha256",
        "spectral_audit_status",
        "source_transfer_status",
    }
    integer_fields = {"source_gram_support_rank", "srm_sqrt_support_rank"}
    for field in sorted(SPECTRAL_AUDIT_FIELDS):
        expected = canonical_audit[field]
        actual = reported_audit[field]
        if field in exact_text_fields:
            if type(actual) is not str or actual != expected:
                return False, f"source_transfer_ledger_mismatch:{field}"
        elif field in integer_fields:
            if int(actual) != int(expected):
                return False, f"source_transfer_ledger_mismatch:{field}"
        elif not _metadata_values_match(expected, actual):
            return False, f"source_transfer_ledger_mismatch:{field}"

    delta = float(canonical_audit["source_transfer_delta"])
    budget = float(canonical_audit["source_transfer_budget"])
    target_trace = float(canonical_audit["source_target_trace"])
    reconstructed_trace = float(
        canonical_audit["source_reconstructed_trace"]
    )
    if not (
        budget >= delta >= 0.0
        and target_trace >= 0.0
        and reconstructed_trace >= 0.0
    ):
        return False, "source_transfer_ledger_mismatch:invalid_ordering"
    return True, "passed"


def _normalize_audit_triplet(
    result: object,
    *,
    namespace: str,
    allowed_fields: frozenset[str],
    integer_fields: frozenset[str] = frozenset(),
    string_fields: frozenset[str] = frozenset(),
    sha256_fields: frozenset[str] = frozenset(),
    blankable_fields: frozenset[str] = frozenset(),
    require_all_metrics: bool = True,
) -> tuple[bool, str, dict[str, float | int | str]]:
    if type(result) is not tuple or len(result) != 3:
        raise _NamespaceValidationError(f"{namespace}:malformed_tuple")
    valid, reason, metrics = result
    if type(valid) is not bool:
        raise _NamespaceValidationError(f"{namespace}:non_plain_bool")
    if type(reason) is not str:
        raise _NamespaceValidationError(f"{namespace}:non_plain_reason")
    normalized = _normalize_plain_numeric_namespace(
        metrics,
        namespace=namespace,
        allowed_fields=allowed_fields,
        required_fields=(
            allowed_fields if require_all_metrics or valid else frozenset()
        ),
        integer_fields=integer_fields,
        string_fields=string_fields,
        sha256_fields=sha256_fields,
        blankable_fields=blankable_fields,
    )
    if valid and reason != "passed":
        raise _NamespaceValidationError(f"{namespace}:true_with_nonpassed_reason")
    return valid, reason, normalized


def _normalize_identity_audit_result(
    result: object,
) -> tuple[bool, str, dict[str, float | int | str]]:
    return _normalize_audit_triplet(
        result,
        namespace="identity_namespace_invalid",
        allowed_fields=SOURCE_IDENTITY_FIELDS,
        string_fields=frozenset(
            {
                "trusted_source_snapshot_validation_status",
                "trusted_source_snapshot_validation_reason",
                "trusted_source_snapshot_validation_phase",
            }
        ),
        sha256_fields=frozenset(
            {
                "trusted_source_snapshot_sha256",
                "trusted_source_snapshot_pre_sha256",
                "trusted_source_snapshot_post_sha256",
            }
        ),
    )


def _normalize_input_audit_result(
    result: object,
) -> tuple[bool, str, dict[str, float | int | str]]:
    return _normalize_audit_triplet(
        result,
        namespace="input_namespace_invalid",
        allowed_fields=CERTIFICATE_INPUT_AUDIT_FIELDS,
        string_fields=frozenset(
            {
                "certificate_input_factor_pre_writeable",
                "certificate_input_factor_post_writeable",
                "certificate_input_factor_validation_status",
                "certificate_input_factor_validation_reason",
            }
        ),
        sha256_fields=frozenset(
            {
                "certificate_input_factor_pre_sha256",
                "certificate_input_factor_post_sha256",
            }
        ),
        blankable_fields=frozenset(
            {
                "certificate_input_factor_post_sha256",
                "certificate_input_factor_max_abs_error",
                "certificate_input_factor_post_writeable",
            }
        ),
    )


def _normalize_validation_result(
    result: object,
) -> tuple[bool, str, dict[str, float | int | str]]:
    return _normalize_audit_triplet(
        result,
        namespace="validation_namespace_invalid",
        allowed_fields=SAFE_VALIDATION_FIELDS,
        integer_fields=frozenset(
            {"certificate_rank", "certificate_expected_rank"}
        ),
        require_all_metrics=False,
    )


def _normalize_artifact_result(
    result: object,
    *,
    namespace: str = "artifact_namespace_invalid",
) -> tuple[bool, str, dict[str, float | int | str]]:
    return _normalize_audit_triplet(
        result,
        namespace=namespace,
        allowed_fields=ARTIFACT_METRIC_FIELDS,
        string_fields=frozenset({"certificate_artifact_schema"}),
        sha256_fields=frozenset(
            {
                "certificate_artifact_source_states_sha256",
                "certificate_artifact_source_gram_sha256",
            }
        ),
        require_all_metrics=False,
    )


def _populate_analytic_endpoint(
    row: dict[str, float | int | str], overlap: float
) -> None:
    hypothesis_count = int(row["external_M_n"])
    probability = 1.0 if overlap == 0.0 else 1.0 / hypothesis_count
    theoretical_rank = hypothesis_count if overlap == 0.0 else 1
    row.update(
        {
            "P_tr": probability,
            "P_SRM": probability,
            "P_SRM_transfer_budget": 0.0,
            "P_SRM_safe_lower": probability,
            "L_raw": probability,
            "U_raw": probability,
            "L_safe": probability,
            "U_safe": probability,
            "strongest_certified_lower": probability,
            "raw_safe_gap": 0.0,
            "raw_safe_gap_relative": 0.0,
            "safe_gap": 0.0,
            "safe_gap_relative": 0.0,
            "srm_status": "analytic_endpoint_exact",
            "sdp_status": "analytic_endpoint_exact",
            "primal_status": "analytic_endpoint_exact",
            "dual_status": "analytic_endpoint_exact",
            "status": "analytic_endpoint",
            "source_gram_support_rank": theoretical_rank,
            "source_gram_factorization_residual_fro_relative": 0.0,
            "source_gram_factorization_residual_op_relative": 0.0,
            "srm_sqrt_reconstruction_residual_fro_relative": 0.0,
            "srm_sqrt_reconstruction_residual_op_relative": 0.0,
            "srm_sqrt_support_rank": theoretical_rank,
            "source_reconstructed_sqrt_residual_fro_relative": 0.0,
            "source_reconstructed_sqrt_residual_op_relative": 0.0,
            "spectral_audit_status": "analytic_endpoint_exact_no_evd",
            "trusted_source_snapshot_validation_status": (
                "analytic_endpoint_exact_no_snapshot"
            ),
            "trusted_source_snapshot_validation_reason": (
                "analytic_endpoint_exact_no_snapshot"
            ),
            "trusted_source_snapshot_validation_phase": (
                "analytic_endpoint_exact_no_snapshot"
            ),
            "source_target_trace": 1.0,
            "source_reconstructed_trace": 1.0,
            "source_difference_trace_norm": 0.0,
            "source_difference_fro_norm": 0.0,
            "source_difference_op_norm": 0.0,
            "source_difference_trace_norm_relative": 0.0,
            "source_difference_fro_norm_relative": 0.0,
            "source_difference_op_norm_relative": 0.0,
            "source_lambda_min_target": (
                1.0 / hypothesis_count if overlap == 0.0 else 0.0
            ),
            "source_lambda_max_target": (
                1.0 / hypothesis_count if overlap == 0.0 else 1.0
            ),
            "source_lambda_min_reconstructed": (
                1.0 / hypothesis_count if overlap == 0.0 else 0.0
            ),
            "source_lambda_max_reconstructed": (
                1.0 / hypothesis_count if overlap == 0.0 else 1.0
            ),
            "source_sqrt_difference_computed_fro": 0.0,
            "source_sqrt_difference_bound_trace": 0.0,
            "source_sqrt_difference_bound_spectral": 0.0,
            "source_sqrt_difference_bound": 0.0,
            "source_transfer_delta": 0.0,
            "source_transfer_budget": 0.0,
            "source_transfer_status": "analytic_endpoint_exact_no_evd",
            "hull_subtraction_audit_status": "analytic_endpoint_exact_no_dense_allocation",
            "hull_subtraction_audit_kind": (
                "analytic_endpoint_exact_no_dense_allocation"
            ),
            "certificate_probability_scale": probability,
            "certificate_probability_tolerance": 0.0,
            "certificate_validation_status": "analytic_endpoint_exact",
            "certificate_validation_reason": "analytic_endpoint_exact",
            "certificate_input_factor_validation_status": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_input_factor_validation_reason": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_source_binding_status": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_source_binding_reason": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_artifact_schema": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_artifact_validation_status": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "certificate_artifact_validation_reason": (
                "analytic_endpoint_exact_no_certificate"
            ),
            "enclosure_kind": "analytic_endpoint_exact",
            "numeric_guarantee": "analytic_endpoint_exact_formula",
        }
    )
    _populate_ratios(row)


def _validate_diagnostic_limits(
    max_srm_n: int,
    max_sdp_n: int,
    max_peak_bytes: int,
    max_matrix_dimension: int,
) -> None:
    for name, value in (
        ("max_srm_n", max_srm_n),
        ("max_sdp_n", max_sdp_n),
        ("max_peak_bytes", max_peak_bytes),
        ("max_matrix_dimension", max_matrix_dimension),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_peak_bytes == 0 or max_matrix_dimension == 0:
        raise ValueError("resource budgets must be positive")


def _legacy_diagnostic_row(
    n: int,
    c: float,
    *,
    schedule: str,
    max_srm_n: int = DEFAULT_MAX_SRM_N,
    max_sdp_n: int = DEFAULT_MAX_SDP_N,
    max_peak_bytes: int = DEFAULT_MAX_PEAK_BYTES,
    max_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
    allow_srm_above_default: bool = False,
    allow_sdp_above_default: bool = False,
    solver: str = "CLARABEL",
    resource_log: Callable[[dict[str, int | str]], None] | None = None,
    _canonical_source_auditor=_trusted_source_snapshot_audit,
    _canonical_artifact_validator=_independently_validate_certificate_artifacts,
) -> dict[str, float | int | str]:
    """Return one estimate-first finite diagnostic row.

    ``L_raw`` and ``U_raw`` come from the direct weighted-state SDP API;
    ``L_safe`` and ``U_safe`` additionally include the audited source-factor
    transfer budget.  No value is written to ``P_opt`` because a floating-point
    finite solve is retained as an enclosure rather than advertised as an exact
    optimum.
    """
    site_count = _validate_n(n)
    overlap = _validate_overlap(c)
    _validate_diagnostic_limits(
        max_srm_n,
        max_sdp_n,
        max_peak_bytes,
        max_matrix_dimension,
    )
    estimate = estimate_dense_resources(site_count)
    if resource_log is not None:
        resource_log(dict(estimate))
    row = _blank_row(site_count, overlap, schedule, estimate, solver)

    if (
        estimate["estimated_peak_bytes"] > max_peak_bytes
        or estimate["eigensolver_dimension"] > max_matrix_dimension
    ):
        row["srm_status"] = "not_computed_resource_guard"
        row["sdp_status"] = "not_computed_resource_guard"
        row["primal_status"] = "not_computed_resource_guard"
        row["dual_status"] = "not_computed_resource_guard"
        row["status"] = "not_computed_resource_guard"
        return row

    # Both physical endpoints have exact ensemble probabilities.  Keep this
    # before every dense Gram, EVD, SRM, CVXPY, and solver path.
    if overlap == 0.0 or overlap == 1.0:
        _populate_analytic_endpoint(row, overlap)
        return row

    if site_count <= max_srm_n:
        if site_count > DEFAULT_MAX_SRM_N and not allow_srm_above_default:
            row["srm_status"] = "not_computed_opt_in_required"
        else:
            row["srm_status"] = "eligible"

    if site_count <= max_sdp_n:
        if site_count > MAX_EXPLICIT_SDP_N:
            row["sdp_status"] = "not_computed_unsupported_above_n7"
        elif site_count > DEFAULT_MAX_SDP_N and not allow_sdp_above_default:
            row["sdp_status"] = "not_computed_opt_in_required"
        else:
            row["sdp_status"] = "eligible"

    need_srm = row["srm_status"] == "eligible"
    need_sdp = row["sdp_status"] == "eligible"
    if not need_srm and not need_sdp:
        return row

    try:
        _, gram, hull_audit = _normalize_hull_result(
            _hull_gram_with_audit(site_count, overlap),
            hypothesis_count=int(row["external_M_n"]),
        )
    except Exception as exc:
        return _fail_closed_namespace(
            row,
            failure="not_computed_hull_audit_failed",
            reason=f"hull_namespace_invalid:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    _project_namespace(row, hull_audit, HULL_AUDIT_FIELDS)

    # Bind the helper-selected matrix directly to the requested physical
    # ensemble.  This local construction deliberately does not call any Gram
    # builder: the labels and symmetric-difference exponents come straight
    # from the caller's (n,c) parameters.
    labels = tuple(
        (left, right)
        for left in range(site_count)
        for right in range(left, site_count)
    )
    direct_physical = np.empty(
        (len(labels), len(labels)), dtype=np.float64
    )
    for row_index, (left_a, right_a) in enumerate(labels):
        for column_index, (left_b, right_b) in enumerate(labels):
            length_a = right_a - left_a + 1
            length_b = right_b - left_b + 1
            intersection = max(
                0,
                min(right_a, right_b) - max(left_a, left_b) + 1,
            )
            symmetric_difference = (
                length_a + length_b - 2 * intersection
            )
            direct_physical[row_index, column_index] = (
                overlap**symmetric_difference
            )
    physical_formula_tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(direct_physical))))
    )
    if not np.allclose(
        gram,
        direct_physical,
        rtol=32.0 * np.finfo(float).eps,
        atol=physical_formula_tolerance,
    ):
        return _fail_closed_namespace(
            row,
            failure="not_computed_physical_gram_formula_mismatch",
            reason="physical_gram_formula_mismatch:n_c_authority",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    if row["hull_subtraction_audit_status"] != "passed":
        if need_srm:
            row["srm_status"] = "not_computed_hull_audit_failed"
        if need_sdp:
            row["sdp_status"] = "not_computed_hull_audit_failed"
            row["primal_status"] = "not_computed_hull_audit_failed"
            row["dual_status"] = "not_computed_hull_audit_failed"
        row["status"] = "not_computed_hull_audit_failed"
        return row

    try:
        physical_gram_snapshot = np.array(
            gram,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        physical_gram_snapshot.setflags(write=False)
        physical_gram_authority = _capture_float64_array_authority(
            physical_gram_snapshot,
            require_readonly=True,
        )
        initial_source_result = _spectral_reconstruction_audit(
            _authority_readonly_view(physical_gram_authority)
        )
        if type(initial_source_result) is not tuple or len(initial_source_result) != 3:
            raise _NamespaceValidationError(
                "source_namespace_invalid:initial_malformed_tuple"
            )
        audited_source_states = initial_source_result[0]
        if type(audited_source_states) is not np.ndarray:
            raise _NamespaceValidationError(
                "source_namespace_invalid:initial_states_not_ndarray"
            )
        trusted_source_snapshot = np.array(
            audited_source_states,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        trusted_source_snapshot.setflags(write=False)
        trusted_source_authority = _capture_float64_array_authority(
            trusted_source_snapshot,
            require_readonly=True,
        )
    except Exception as exc:
        if need_srm:
            row["srm_status"] = "not_computed_source_transfer_exception"
        if need_sdp:
            row["sdp_status"] = "not_computed_source_transfer_exception"
            row["primal_status"] = "not_computed_source_transfer_exception"
            row["dual_status"] = "not_computed_source_transfer_exception"
        row["spectral_audit_status"] = "failed"
        row["source_transfer_status"] = "failed"
        reason = f"source_namespace_invalid:{type(exc).__name__}:{exc}"
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_exception",
            reason=reason,
            need_srm=need_srm,
            need_sdp=need_sdp,
        )

    # The exact identity ledger is captured before the authoritative helper.
    # Metadata and transfer values returned beside the earlier factorization
    # are deliberately discarded.
    source_metadata: dict[str, float | int | str] | None = None
    spectral_audit: dict[str, float | int | str] | None = None
    source_helper_exception: Exception | None = None
    source_namespace_reason = ""
    raw_source_result: object = None
    try:
        raw_source_result = _trusted_source_snapshot_audit(
            _authority_readonly_view(physical_gram_authority),
            trusted_source_snapshot,
        )
        source_metadata, spectral_audit = _normalize_source_authority_result(
            raw_source_result
        )
    except _NamespaceValidationError as exc:
        source_namespace_reason = str(exc)
    except Exception as exc:
        source_helper_exception = exc

    helper_sha256 = None
    helper_ledger_structurally_available = bool(
        type(raw_source_result) is tuple
        and len(raw_source_result) == 2
        and type(raw_source_result[1]) is dict
    )
    if helper_ledger_structurally_available:
        helper_sha256 = dict.get(
            raw_source_result[1], "trusted_source_snapshot_sha256"
        )
    try:
        (
            source_identity_passed,
            source_identity_reason,
            source_identity_metrics,
        ) = _normalize_identity_audit_result(
            _trusted_source_snapshot_identity_audit(
                trusted_source_snapshot,
                trusted_source_authority,
                phase="post_authority_helper",
                helper_sha256=helper_sha256,
                require_helper_sha256=(
                    source_helper_exception is None
                    and helper_ledger_structurally_available
                ),
            )
        )
    except Exception as exc:
        return _fail_closed_namespace(
            row,
            failure="not_computed_trusted_source_snapshot_mutated",
            reason=f"identity_namespace_invalid:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    _project_namespace(row, source_identity_metrics, SOURCE_IDENTITY_FIELDS)
    if not source_identity_passed:
        return _fail_closed_trusted_source_snapshot(
            row,
            need_srm=need_srm,
            need_sdp=need_sdp,
            reason=source_identity_reason,
            identity_metrics=source_identity_metrics,
        )

    if source_namespace_reason:
        row["spectral_audit_status"] = "failed"
        row["source_transfer_status"] = "failed"
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_failed",
            reason=source_namespace_reason,
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    if source_helper_exception is not None:
        row["spectral_audit_status"] = "failed"
        row["source_transfer_status"] = "failed"
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_exception",
            reason=(
                "source_transfer_exception:"
                f"{type(source_helper_exception).__name__}"
            ),
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    if source_metadata is None or spectral_audit is None:
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_failed",
            reason="source_namespace_invalid:missing_normalized_result",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )

    # The patchable helper is diagnostic only.  Recompute every source
    # spectrum, square-root relation, trace, rank, delta, and budget through a
    # definition-time pinned core, compare both ledgers, and expose only the
    # pinned values.
    try:
        canonical_source_metadata, canonical_spectral_audit = (
            _normalize_source_authority_result(
                _canonical_source_auditor(
                    _authority_readonly_view(physical_gram_authority),
                    _authority_readonly_view(trusted_source_authority),
                )
            )
        )
        source_ledgers_match, source_ledger_reason = (
            _source_authority_ledgers_match(
                source_metadata,
                spectral_audit,
                canonical_source_metadata,
                canonical_spectral_audit,
            )
        )
    except Exception as exc:
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_failed",
            reason=(
                "source_transfer_ledger_mismatch:canonical_exception:"
                f"{type(exc).__name__}:{exc}"
            ),
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    if not source_ledgers_match:
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_failed",
            reason=source_ledger_reason,
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    source_metadata = canonical_source_metadata
    spectral_audit = canonical_spectral_audit
    _project_namespace(row, spectral_audit, SPECTRAL_AUDIT_FIELDS)
    if row["trusted_source_snapshot_sha256"] != trusted_source_authority.sha256:
        return _fail_closed_namespace(
            row,
            failure="not_computed_source_transfer_failed",
            reason="source_namespace_invalid:authority_sha_mismatch",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    if (
        row["spectral_audit_status"] != "passed"
        or row["source_transfer_status"] != "passed"
    ):
        if need_srm:
            row["srm_status"] = "not_computed_source_transfer_failed"
        if need_sdp:
            row["sdp_status"] = "not_computed_source_transfer_failed"
            row["primal_status"] = "not_computed_source_transfer_failed"
            row["dual_status"] = "not_computed_source_transfer_failed"
        row["status"] = "not_computed_source_transfer_failed"
        return row

    srm_result: dict[str, float | int] | None = None
    if need_srm or need_sdp:
        try:
            srm_result = srm_quantities(
                _authority_readonly_view(physical_gram_authority)
            )
            trace_lower = float(srm_result["trace_lower_bound"])
            srm_probability = float(srm_result["srm"])
        except Exception as exc:
            row["srm_status"] = "not_computed_srm_exception"
            if need_sdp:
                row["sdp_status"] = "not_computed_srm_exception"
                row["primal_status"] = "not_computed_srm_exception"
                row["dual_status"] = "not_computed_srm_exception"
            row["status"] = "not_computed_srm_exception"
            row["certificate_validation_reason"] = type(exc).__name__
            return row
        if not (
            math.isfinite(trace_lower)
            and math.isfinite(srm_probability)
            and 0.0 <= trace_lower <= 1.0
            and 0.0 <= srm_probability <= 1.0
        ):
            row["srm_status"] = "not_computed_invalid_srm"
            if need_sdp:
                row["sdp_status"] = "not_computed_invalid_srm"
                row["primal_status"] = "not_computed_invalid_srm"
                row["dual_status"] = "not_computed_invalid_srm"
            row["status"] = "not_computed_invalid_srm"
            return row
        srm_scale = _probability_scale(
            trace_lower,
            srm_probability,
            1.0 / int(row["external_M_n"]),
        )
        if trace_lower > srm_probability + PROBABILITY_RELATIVE_TOLERANCE * srm_scale:
            row["srm_status"] = "not_computed_invalid_srm_order"
            if need_sdp:
                row["sdp_status"] = "not_computed_invalid_srm_order"
                row["primal_status"] = "not_computed_invalid_srm_order"
                row["dual_status"] = "not_computed_invalid_srm_order"
            row["status"] = "not_computed_invalid_srm_order"
            return row
        row["P_tr"] = trace_lower
        row["P_SRM"] = srm_probability
        # P_SRM is evaluated directly on the physical target Gram, so the
        # source-factor transfer used by the SDP is unrelated to this explicit
        # target measurement.  These compatibility fields therefore record a
        # zero SRM transfer budget and the target SRM value itself.
        row["P_SRM_transfer_budget"] = 0.0
        row["P_SRM_safe_lower"] = srm_probability
        row["strongest_certified_lower"] = srm_probability
        row["srm_status"] = (
            "computed" if need_srm else "computed_for_sdp_validation"
        )

    if need_sdp:
        try:
            (
                source_identity_passed,
                source_identity_reason,
                source_identity_metrics,
            ) = _normalize_identity_audit_result(
                _trusted_source_snapshot_identity_audit(
                    trusted_source_snapshot,
                    trusted_source_authority,
                    phase="before_solver",
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_trusted_source_snapshot_mutated",
                reason=f"identity_namespace_invalid:{type(exc).__name__}:{exc}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, source_identity_metrics, SOURCE_IDENTITY_FIELDS)
        if not source_identity_passed:
            return _fail_closed_trusted_source_snapshot(
                row,
                need_srm=need_srm,
                need_sdp=need_sdp,
                reason=source_identity_reason,
                identity_metrics=source_identity_metrics,
            )

        row["certificate_expected_rank"] = int(row["external_M_n"])
        row["certificate_rank_tolerance"] = 0.0
        solver_input = _authority_writable_copy(trusted_source_authority)
        solver_input_before_bytes = trusted_source_authority.payload
        solver_input_before_sha256 = trusted_source_authority.sha256
        solver_input_before_writeable = bool(solver_input.flags.writeable)
        certificate: dict[str, object] | None = None
        certificate_exception: Exception | None = None
        try:
            certificate = certify_minimum_error_from_weighted_states(
                solver_input,
                solver=solver,
            )
        except Exception as exc:
            certificate_exception = exc

        try:
            (
                source_identity_passed,
                source_identity_reason,
                source_identity_metrics,
            ) = _normalize_identity_audit_result(
                _trusted_source_snapshot_identity_audit(
                    trusted_source_snapshot,
                    trusted_source_authority,
                    phase="after_solver",
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_trusted_source_snapshot_mutated",
                reason=f"identity_namespace_invalid:{type(exc).__name__}:{exc}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, source_identity_metrics, SOURCE_IDENTITY_FIELDS)
        if not source_identity_passed:
            return _fail_closed_trusted_source_snapshot(
                row,
                need_srm=need_srm,
                need_sdp=need_sdp,
                reason=source_identity_reason,
                identity_metrics=source_identity_metrics,
            )

        try:
            input_unchanged, input_reason, input_metrics = (
                _normalize_input_audit_result(
                    _certificate_input_factor_audit(
                        _authority_readonly_view(trusted_source_authority),
                        solver_input,
                        before_bytes=solver_input_before_bytes,
                        before_sha256=solver_input_before_sha256,
                        before_writeable=solver_input_before_writeable,
                    )
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_input_factor_mutated",
                reason=f"input_namespace_invalid:{type(exc).__name__}:{exc}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, input_metrics, CERTIFICATE_INPUT_AUDIT_FIELDS)
        if not input_unchanged:
            input_failure = "not_computed_certificate_input_factor_mutated"
            row["sdp_status"] = input_failure
            row["primal_status"] = input_failure
            row["dual_status"] = input_failure
            row["certificate_validation_status"] = "failed"
            row["certificate_validation_reason"] = input_reason
            row["certificate_source_binding_status"] = input_failure
            row["certificate_source_binding_reason"] = input_failure
            row["certificate_artifact_validation_status"] = input_failure
            row["certificate_artifact_validation_reason"] = input_failure
            row["enclosure_kind"] = input_failure
            row["status"] = "srm_only_certificate_failed"
            _populate_ratios(row)
            return row

        if certificate_exception is not None:
            row["sdp_status"] = "not_computed_certificate_exception"
            row["primal_status"] = "not_computed_certificate_exception"
            row["dual_status"] = "not_computed_certificate_exception"
            row["certificate_validation_status"] = "failed"
            row["certificate_validation_reason"] = (
                f"certificate_exception:{type(certificate_exception).__name__}"
            )
            row["certificate_source_binding_status"] = (
                "not_computed_certificate_exception"
            )
            row["certificate_source_binding_reason"] = (
                "not_computed_certificate_exception"
            )
            row["certificate_artifact_validation_status"] = "failed"
            row["certificate_artifact_validation_reason"] = (
                "not_computed_certificate_exception"
            )
            row["enclosure_kind"] = "not_computed_certificate_exception"
            row["status"] = "srm_only_certificate_failed"
            _populate_ratios(row)
            return row

        try:
            certificate = _normalize_certificate_namespace(certificate)
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_validation_failed",
                reason=(
                    "certificate_namespace_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )

        row["certificate_rank"] = int(certificate["rank"])
        row["primal_status"] = certificate["primal_status"]
        row["dual_status"] = certificate["dual_status"]
        row["cvxpy_version"] = certificate["cvxpy_version"]

        try:
            valid, reason, validation = _normalize_validation_result(
                _validate_safe_certificate(
                    certificate,
                    hypothesis_count=int(row["external_M_n"]),
                    trace_lower=(
                        float(row["P_tr"]) if row["P_tr"] != "" else None
                    ),
                    srm_probability=(
                        float(row["P_SRM"])
                        if row["P_SRM"] != ""
                        else None
                    ),
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_validation_failed",
                reason=(
                    "validation_namespace_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(
            row,
            {
                field: value
                for field, value in validation.items()
                if field not in {"validated_lower", "validated_upper"}
            },
            SAFE_VALIDATION_FIELDS,
        )
        row["certificate_validation_status"] = "passed" if valid else "failed"
        row["certificate_validation_reason"] = reason
        if not valid:
            row["certificate_source_binding_status"] = (
                "not_computed_certificate_validation_failed"
            )
            row["certificate_source_binding_reason"] = (
                "not_computed_certificate_validation_failed"
            )
            row["sdp_status"] = "not_computed_certificate_validation_failed"
            row["enclosure_kind"] = (
                "not_computed_certificate_validation_failed"
            )
            row["status"] = "srm_only_certificate_failed"
            _populate_ratios(row)
            return row

        try:
            source_bound, binding_reason = _certificate_binds_to_source(
                source_metadata, certificate
            )
            if type(source_bound) is not bool or type(binding_reason) is not str:
                raise _NamespaceValidationError(
                    "source_binding_namespace_invalid:return_type"
                )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_source_binding_failed",
                reason=(
                    "source_binding_namespace_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        row["certificate_source_binding_status"] = (
            "passed" if source_bound else "failed"
        )
        row["certificate_source_binding_reason"] = binding_reason
        if not source_bound:
            row["sdp_status"] = (
                "not_computed_certificate_source_binding_failed"
            )
            row["enclosure_kind"] = (
                "not_computed_certificate_source_binding_failed"
            )
            row["status"] = "srm_only_certificate_failed"
            _populate_ratios(row)
            return row

        artifact_result: object = None
        artifact_exception: Exception | None = None
        try:
            artifact_result = _validate_certificate_artifacts(
                _authority_readonly_view(trusted_source_authority),
                solver_input,
                certificate,
            )
        except Exception as exc:
            artifact_exception = exc

        try:
            (
                source_identity_passed,
                source_identity_reason,
                source_identity_metrics,
            ) = _normalize_identity_audit_result(
                _trusted_source_snapshot_identity_audit(
                    trusted_source_snapshot,
                    trusted_source_authority,
                    phase="after_validator",
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_trusted_source_snapshot_mutated",
                reason=f"identity_namespace_invalid:{type(exc).__name__}:{exc}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, source_identity_metrics, SOURCE_IDENTITY_FIELDS)
        if not source_identity_passed:
            return _fail_closed_trusted_source_snapshot(
                row,
                need_srm=need_srm,
                need_sdp=need_sdp,
                reason=source_identity_reason,
                identity_metrics=source_identity_metrics,
            )

        try:
            input_unchanged, input_reason, input_metrics = (
                _normalize_input_audit_result(
                    _certificate_input_factor_audit(
                        _authority_readonly_view(trusted_source_authority),
                        solver_input,
                        before_bytes=trusted_source_authority.payload,
                        before_sha256=trusted_source_authority.sha256,
                        before_writeable=solver_input_before_writeable,
                    )
                )
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_input_factor_mutated",
                reason=f"input_namespace_invalid:{type(exc).__name__}:{exc}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, input_metrics, CERTIFICATE_INPUT_AUDIT_FIELDS)
        if not input_unchanged:
            input_failure = "not_computed_certificate_input_factor_mutated"
            row["sdp_status"] = input_failure
            row["primal_status"] = input_failure
            row["dual_status"] = input_failure
            row["certificate_validation_status"] = "failed"
            row["certificate_validation_reason"] = input_reason
            row["certificate_source_binding_status"] = input_failure
            row["certificate_source_binding_reason"] = input_failure
            row["certificate_artifact_validation_status"] = input_failure
            row["certificate_artifact_validation_reason"] = input_failure
            row["enclosure_kind"] = input_failure
            row["status"] = "srm_only_certificate_failed"
            _populate_ratios(row)
            return row

        artifact_failure = (
            "not_computed_certificate_artifact_validation_failed"
        )
        if artifact_exception is not None:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=(
                    "certificate_artifact_validation_exception:"
                    f"{type(artifact_exception).__name__}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        try:
            artifact_valid, artifact_reason, artifact_metrics = (
                _normalize_artifact_result(artifact_result)
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=(
                    "artifact_namespace_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        if not artifact_valid:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=artifact_reason,
                need_srm=need_srm,
                need_sdp=need_sdp,
            )

        try:
            (
                independent_valid,
                independent_reason,
                independent_metrics,
            ) = _normalize_artifact_result(
                _canonical_artifact_validator(
                    _authority_readonly_view(trusted_source_authority),
                    solver_input,
                    certificate,
                ),
                namespace="artifact_independent_namespace_invalid",
            )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=(
                    "artifact_independent_namespace_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        if not independent_valid:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=f"independent_artifact_invalid:{independent_reason}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        try:
            ledgers_match, ledger_reason = _artifact_metric_ledgers_match(
                artifact_metrics,
                independent_metrics,
                hypothesis_count=int(row["external_M_n"]),
            )
            if type(ledgers_match) is not bool or type(ledger_reason) is not str:
                raise _NamespaceValidationError(
                    "artifact_metrics_matcher_invalid:return_type"
                )
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=(
                    "artifact_metrics_matcher_invalid:"
                    f"{type(exc).__name__}:{exc}"
                ),
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        if not ledgers_match:
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason=ledger_reason,
                need_srm=need_srm,
                need_sdp=need_sdp,
            )

        artifact_source_sha256 = independent_metrics[
            "certificate_artifact_source_states_sha256"
        ]
        certificate_source_sha256 = certificate[
            "source_weighted_states_sha256"
        ]
        if not (
            artifact_source_sha256 == trusted_source_authority.sha256
            and certificate_source_sha256 == trusted_source_authority.sha256
            and row["certificate_input_factor_pre_sha256"]
            == trusted_source_authority.sha256
            and row["certificate_input_factor_post_sha256"]
            == trusted_source_authority.sha256
        ):
            return _fail_closed_namespace(
                row,
                failure=artifact_failure,
                reason="artifact_source_sha_mismatch:trusted_snapshot_sha",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        _project_namespace(row, independent_metrics, ARTIFACT_METRIC_FIELDS)
        row["certificate_artifact_validation_status"] = "passed"
        row["certificate_artifact_validation_reason"] = "passed"

        try:
            raw_lower = float(
                independent_metrics["certificate_artifact_primal_objective"]
            )
            raw_upper = float(
                independent_metrics["certificate_artifact_dual_trace"]
            )
            transfer_budget = float(row["source_transfer_budget"])
            lower = max(
                0.0,
                float(np.nextafter(raw_lower - transfer_budget, -math.inf)),
            )
            upper = min(
                1.0,
                float(np.nextafter(raw_upper + transfer_budget, math.inf)),
            )
            target_srm_lower = float(row["P_SRM"])
            strongest_lower = max(lower, target_srm_lower)
            final_probability_scale = _probability_scale(
                lower,
                upper,
                float(row["P_SRM"]),
                1.0 / int(row["external_M_n"]),
            )
            final_probability_tolerance = (
                PROBABILITY_RELATIVE_TOLERANCE * final_probability_scale
            )
            final_values = (
                raw_lower,
                raw_upper,
                transfer_budget,
                lower,
                upper,
                target_srm_lower,
                strongest_lower,
                final_probability_scale,
                final_probability_tolerance,
            )
            raw_gap = raw_upper - raw_lower
            padded_gap = upper - lower
            raw_gap_relative = _relative_enclosure_gap(raw_lower, raw_upper)
            padded_gap_relative = _relative_enclosure_gap(lower, upper)
        except Exception as exc:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_validation_failed",
                reason=f"final_scalar_extraction_exception:{type(exc).__name__}",
                need_srm=need_srm,
                need_sdp=need_sdp,
            )
        final_valid = all(math.isfinite(value) for value in final_values)
        final_reason = "passed"
        if not final_valid:
            final_reason = "nonfinite_source_transfer_enclosure"
        elif lower > upper + final_probability_tolerance:
            final_valid = False
            final_reason = "padded_lower_above_upper"
        elif (
            float(row["P_SRM"])
            > upper + final_probability_tolerance
        ):
            final_valid = False
            final_reason = "srm_above_padded_upper"
        elif strongest_lower > upper + final_probability_tolerance:
            final_valid = False
            final_reason = "strongest_lower_above_padded_upper"
        if not final_valid:
            return _fail_closed_namespace(
                row,
                failure="not_computed_certificate_validation_failed",
                reason=final_reason,
                need_srm=need_srm,
                need_sdp=need_sdp,
            )

        row.update(
            {
                "L_raw": raw_lower,
                "U_raw": raw_upper,
                "L_safe": lower,
                "U_safe": upper,
                "raw_safe_gap": raw_gap,
                "raw_safe_gap_relative": raw_gap_relative,
                "safe_gap": padded_gap,
                "safe_gap_relative": padded_gap_relative,
                "primal_feasible_equality_residual_fro": float(
                    independent_metrics[
                        "certificate_artifact_primal_completeness_fro"
                    ]
                ),
                "primal_feasible_equality_residual_op": float(
                    independent_metrics[
                        "certificate_artifact_primal_completeness_op"
                    ]
                ),
                "primal_feasible_psd_violation": float(
                    independent_metrics[
                        "certificate_artifact_primal_psd_violation"
                    ]
                ),
                "primal_feasible_min_eigenvalue": float(
                    independent_metrics[
                        "certificate_artifact_primal_min_eigenvalue"
                    ]
                ),
                "dual_feasible_min_slack": float(
                    independent_metrics[
                        "certificate_artifact_dual_min_slack"
                    ]
                ),
                "dual_feasible_psd_violation": float(
                    independent_metrics[
                        "certificate_artifact_dual_psd_violation"
                    ]
                ),
                "sdp_status": "computed_safe_enclosure",
                "enclosure_kind": (
                    "residual_checked_floating_point_enclosure"
                ),
                "certificate_probability_scale": final_probability_scale,
                "certificate_probability_tolerance": (
                    final_probability_tolerance
                ),
            }
        )
        row["strongest_certified_lower"] = strongest_lower

    row["status"] = "sdp_certified" if need_sdp else "srm_only"
    if not need_sdp:
        row["enclosure_kind"] = "srm_measurement_lower_bound_only"
    _populate_ratios(row)
    return row


_ROUTE_B_DIAGNOSTIC_FIELDS = frozenset(
    {
        "repaired_primal_value",
        "shifted_dual_value",
        "strongest_measurement_value",
        "floating_primal_dual_span",
        "floating_primal_dual_span_relative",
        "repaired_primal_over_p1_squared",
        "shifted_dual_over_p1_squared",
        "strongest_measurement_over_p1_squared",
        "repaired_primal_over_P_SRM",
        "P_SRM_over_shifted_dual",
        "primal_completeness_residual_fro",
        "primal_completeness_residual_op",
        "primal_psd_violation",
        "primal_min_eigenvalue",
        "dual_min_slack_after_shift",
        "dual_psd_violation_after_shift",
    }
)


def _populate_route_b_ratios(row: dict[str, float | int | str]) -> None:
    row["P_tr_over_p1_squared"] = _ratio(row["P_tr"], row["p1_squared"])
    row["P_SRM_over_p1_squared"] = _ratio(
        row["P_SRM"], row["p1_squared"]
    )
    row["P_tr_over_P_SRM"] = _ratio(row["P_tr"], row["P_SRM"])
    row["repaired_primal_over_p1_squared"] = _ratio(
        row["repaired_primal_value"], row["p1_squared"]
    )
    row["shifted_dual_over_p1_squared"] = _ratio(
        row["shifted_dual_value"], row["p1_squared"]
    )
    row["strongest_measurement_over_p1_squared"] = _ratio(
        row["strongest_measurement_value"], row["p1_squared"]
    )
    row["repaired_primal_over_P_SRM"] = _ratio(
        row["repaired_primal_value"], row["P_SRM"]
    )
    row["P_SRM_over_shifted_dual"] = _ratio(
        row["P_SRM"], row["shifted_dual_value"]
    )


def _route_b_fail(
    row: dict[str, float | int | str],
    *,
    failure: str,
    reason: str,
    need_srm: bool,
    need_sdp: bool,
) -> dict[str, float | int | str]:
    for field in _ROUTE_B_DIAGNOSTIC_FIELDS:
        row[field] = ""
    row["P_opt"] = ""
    row["P_opt_status"] = "not_computed"
    row["artifact_recomputation_status"] = "failed"
    row["artifact_recomputation_reason"] = reason
    row["diagnostic_kind"] = failure
    if need_srm and row["P_SRM"] == "":
        row["srm_status"] = failure
    if need_sdp:
        row["sdp_status"] = failure
        row["primal_status"] = failure
        row["dual_status"] = failure
    row["status"] = (
        "srm_only_artifact_diagnostic_failed"
        if row["P_SRM"] != ""
        else failure
    )
    _populate_route_b_ratios(row)
    return row


def _populate_route_b_endpoint(
    row: dict[str, float | int | str], overlap: float
) -> None:
    hypothesis_count = int(row["external_M_n"])
    probability = 1.0 if overlap == 0.0 else 1.0 / hypothesis_count
    row.update(
        {
            "P_tr": probability,
            "P_SRM": probability,
            "repaired_primal_value": probability,
            "shifted_dual_value": probability,
            "strongest_measurement_value": probability,
            "floating_primal_dual_span": 0.0,
            "floating_primal_dual_span_relative": 0.0,
            "primal_completeness_residual_fro": 0.0,
            "primal_completeness_residual_op": 0.0,
            "primal_psd_violation": 0.0,
            "primal_min_eigenvalue": 0.0,
            "dual_min_slack_after_shift": 0.0,
            "dual_psd_violation_after_shift": 0.0,
            "srm_status": "analytic_endpoint_exact",
            "sdp_status": "analytic_endpoint_exact",
            "primal_status": "analytic_endpoint_exact",
            "dual_status": "analytic_endpoint_exact",
            "artifact_recomputation_status": "analytic_endpoint_exact",
            "artifact_recomputation_reason": "analytic_endpoint_exact",
            "diagnostic_kind": "analytic_endpoint_exact",
            "status": "analytic_endpoint",
        }
    )
    _populate_route_b_ratios(row)


def diagnostic_row(
    n: int,
    c: float,
    *,
    schedule: str,
    max_srm_n: int = DEFAULT_MAX_SRM_N,
    max_sdp_n: int = DEFAULT_MAX_SDP_N,
    max_peak_bytes: int = DEFAULT_MAX_PEAK_BYTES,
    max_matrix_dimension: int = DEFAULT_MAX_MATRIX_DIMENSION,
    allow_srm_above_default: bool = False,
    allow_sdp_above_default: bool = False,
    solver: str = "CLARABEL",
    resource_log: Callable[[dict[str, int | str]], None] | None = None,
    _canonical_source_builder=_canonical_source_bundle,
    _canonical_artifact_validator=_independently_validate_certificate_artifacts,
) -> dict[str, float | int | str]:
    """Return one Route-B floating diagnostic with a two-pass truth path.

    The public path performs one definition-time pinned source/physical audit
    and one definition-time pinned artifact recomputation.  It reports a
    re-feasibilized primal value, a shifted-dual value, and their floating
    span.  None is published as a certificate, enclosure, or bound, and
    ``P_opt`` remains uncomputed.
    """
    site_count = _validate_n(n)
    overlap = _validate_overlap(c)
    _validate_diagnostic_limits(
        max_srm_n,
        max_sdp_n,
        max_peak_bytes,
        max_matrix_dimension,
    )
    estimate = estimate_dense_resources(site_count)
    if resource_log is not None:
        resource_log(dict(estimate))
    row = _blank_row(site_count, overlap, schedule, estimate, solver)

    if (
        estimate["estimated_peak_bytes"] > max_peak_bytes
        or estimate["eigensolver_dimension"] > max_matrix_dimension
    ):
        row["srm_status"] = "not_computed_resource_guard"
        row["sdp_status"] = "not_computed_resource_guard"
        row["primal_status"] = "not_computed_resource_guard"
        row["dual_status"] = "not_computed_resource_guard"
        row["artifact_recomputation_status"] = "not_computed_resource_guard"
        row["artifact_recomputation_reason"] = "not_computed_resource_guard"
        row["diagnostic_kind"] = "not_computed_resource_guard"
        row["status"] = "not_computed_resource_guard"
        return row

    if overlap == 0.0 or overlap == 1.0:
        _populate_route_b_endpoint(row, overlap)
        return row

    if site_count <= max_srm_n:
        row["srm_status"] = (
            "not_computed_opt_in_required"
            if site_count > DEFAULT_MAX_SRM_N
            and not allow_srm_above_default
            else "eligible"
        )
    if site_count <= max_sdp_n:
        if site_count > MAX_EXPLICIT_SDP_N:
            row["sdp_status"] = "not_computed_unsupported_above_n7"
        elif site_count > DEFAULT_MAX_SDP_N and not allow_sdp_above_default:
            row["sdp_status"] = "not_computed_opt_in_required"
        else:
            row["sdp_status"] = "eligible"

    need_srm = row["srm_status"] == "eligible"
    need_sdp = row["sdp_status"] == "eligible"
    if not need_srm and not need_sdp:
        return row

    try:
        _, physical, hull_audit = _normalize_hull_result(
            _hull_gram_with_audit(site_count, overlap),
            hypothesis_count=int(row["external_M_n"]),
        )
    except Exception as exc:
        return _route_b_fail(
            row,
            failure="not_computed_hull_audit_failed",
            reason=f"hull_audit_failed:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    _project_namespace(row, hull_audit, HULL_AUDIT_FIELDS)
    if row["hull_subtraction_audit_status"] != "passed":
        return _route_b_fail(
            row,
            failure="not_computed_hull_audit_failed",
            reason="hull_subtraction_audit_failed",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )

    # Directly bind the helper-selected physical matrix to (n,c).
    labels = tuple(
        (left, right)
        for left in range(site_count)
        for right in range(left, site_count)
    )
    direct_physical = np.empty_like(physical, dtype=np.float64)
    for row_index, (left_a, right_a) in enumerate(labels):
        for column_index, (left_b, right_b) in enumerate(labels):
            length_a = right_a - left_a + 1
            length_b = right_b - left_b + 1
            intersection = max(
                0,
                min(right_a, right_b) - max(left_a, left_b) + 1,
            )
            distance = length_a + length_b - 2 * intersection
            direct_physical[row_index, column_index] = overlap**distance
    physical_tolerance = (
        32.0
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(direct_physical))))
    )
    if not np.allclose(
        physical,
        direct_physical,
        rtol=32.0 * np.finfo(float).eps,
        atol=physical_tolerance,
    ):
        return _route_b_fail(
            row,
            failure="not_computed_physical_gram_formula_mismatch",
            reason="physical_gram_formula_mismatch:n_c_authority",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    physical_snapshot = np.array(
        physical, dtype=np.float64, copy=True, order="C"
    )
    physical_snapshot.setflags(write=False)

    # First and only source/physical truth pass.
    try:
        source_result = _canonical_source_builder(physical_snapshot)
        if type(source_result) is not tuple or len(source_result) != 3:
            raise _NamespaceValidationError("source_bundle_malformed_tuple")
        source_states_raw, source_metadata_raw, source_audit_raw = source_result
        if type(source_states_raw) is not np.ndarray:
            raise _NamespaceValidationError("source_bundle_states_not_ndarray")
        source_states = np.array(
            source_states_raw, dtype=np.float64, copy=True, order="C"
        )
        source_states.setflags(write=False)
        source_metadata, source_audit = _normalize_source_authority_result(
            (source_metadata_raw, source_audit_raw)
        )
        if (
            int(source_metadata["rank"]) != int(row["external_M_n"])
            or int(source_metadata["hypothesis_count"])
            != int(row["external_M_n"])
            or source_audit["spectral_audit_status"] != "passed"
            or source_audit["source_transfer_status"] != "passed"
        ):
            raise ValueError("source_bundle_failed_full_support_audit")
        source_authority = _capture_float64_array_authority(
            source_states, require_readonly=True
        )
    except Exception as exc:
        return _route_b_fail(
            row,
            failure="not_computed_source_audit_failed",
            reason=f"source_audit_failed:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )
    _project_namespace(row, source_audit, SPECTRAL_AUDIT_FIELDS)

    try:
        srm_result = srm_quantities(physical_snapshot)
        trace_value = float(srm_result["trace_lower_bound"])
        srm_value = float(srm_result["srm"])
        if not (math.isfinite(trace_value) and math.isfinite(srm_value)):
            raise ValueError("nonfinite_srm")
        row["P_tr"] = trace_value
        row["P_SRM"] = srm_value
        row["srm_status"] = "computed"
    except Exception as exc:
        return _route_b_fail(
            row,
            failure="not_computed_srm_failed",
            reason=f"srm_failed:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )

    if not need_sdp:
        row["status"] = "srm_only"
        row["diagnostic_kind"] = "full_physical_gram_srm_only"
        row["artifact_recomputation_status"] = "not_computed"
        row["artifact_recomputation_reason"] = "outside_small_sdp_cutoff"
        _populate_route_b_ratios(row)
        return row

    solver_input = _authority_writable_copy(source_authority)
    try:
        certificate = _normalize_certificate_namespace(
            certify_minimum_error_from_weighted_states(
                solver_input,
                solver=solver,
            )
        )
        row["primal_status"] = str(certificate["primal_status"])
        row["dual_status"] = str(certificate["dual_status"])
        row["cvxpy_version"] = str(certificate["cvxpy_version"])
        accepted = {"optimal", "optimal_inaccurate"}
        if (
            row["primal_status"].lower() not in accepted
            or row["dual_status"].lower() not in accepted
        ):
            raise ValueError("unacceptable_solver_status")

        # Second and only artifact truth pass.
        artifact_valid, artifact_reason, artifact_metrics = (
            _normalize_artifact_result(
                _canonical_artifact_validator(
                    _authority_readonly_view(source_authority),
                    solver_input,
                    certificate,
                ),
                namespace="artifact_recomputation_malformed",
            )
        )
        if not artifact_valid:
            raise ValueError(f"artifact_recomputation_failed:{artifact_reason}")

        primal_value = float(
            artifact_metrics["certificate_artifact_primal_objective"]
        )
        dual_value = float(
            artifact_metrics["certificate_artifact_dual_trace"]
        )
        strongest_value = max(primal_value, float(row["P_SRM"]))
        span = dual_value - primal_value
        scale = max(
            np.finfo(float).tiny,
            0.5 * (abs(primal_value) + abs(dual_value)),
        )
        values = (primal_value, dual_value, strongest_value, span, scale)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite_floating_diagnostic")
        if primal_value > dual_value or strongest_value > dual_value:
            raise ValueError("inconsistent_floating_primal_dual_order")

        row.update(
            {
                "repaired_primal_value": primal_value,
                "shifted_dual_value": dual_value,
                "strongest_measurement_value": strongest_value,
                "floating_primal_dual_span": span,
                "floating_primal_dual_span_relative": span / scale,
                "primal_completeness_residual_fro": float(
                    artifact_metrics[
                        "certificate_artifact_primal_completeness_fro"
                    ]
                ),
                "primal_completeness_residual_op": float(
                    artifact_metrics[
                        "certificate_artifact_primal_completeness_op"
                    ]
                ),
                "primal_psd_violation": float(
                    artifact_metrics[
                        "certificate_artifact_primal_psd_violation"
                    ]
                ),
                "primal_min_eigenvalue": float(
                    artifact_metrics[
                        "certificate_artifact_primal_min_eigenvalue"
                    ]
                ),
                "dual_min_slack_after_shift": float(
                    artifact_metrics["certificate_artifact_dual_min_slack"]
                ),
                "dual_psd_violation_after_shift": float(
                    artifact_metrics[
                        "certificate_artifact_dual_psd_violation"
                    ]
                ),
                "artifact_recomputation_status": "passed",
                "artifact_recomputation_reason": "passed",
                "sdp_status": "computed_residual_checked_diagnostic",
                "status": "sdp_floating_diagnostic",
                "diagnostic_kind": (
                    "residual_checked_floating_primal_dual_span"
                ),
            }
        )
    except Exception as exc:
        return _route_b_fail(
            row,
            failure="not_computed_artifact_diagnostic_failed",
            reason=f"artifact_diagnostic_failed:{type(exc).__name__}:{exc}",
            need_srm=need_srm,
            need_sdp=need_sdp,
        )

    _populate_route_b_ratios(row)
    return row


def _diagnostic_cases(
    max_srm_n: int,
    max_sdp_n: int = DEFAULT_MAX_SDP_N,
) -> list[dict[str, float | int | str]]:
    compact_ns = set(COMPACT_N_VALUES)
    compact_ns.update(range(DEFAULT_MAX_SDP_N + 1, max_sdp_n + 1))
    if max_srm_n > DEFAULT_MAX_SRM_N:
        compact_ns.add(max_srm_n)
    outer_ns = set(OUTER_N_VALUES)
    if max_srm_n > DEFAULT_MAX_SRM_N:
        outer_ns.add(max_srm_n)

    cases: list[dict[str, float | int | str]] = []
    for n in sorted(value for value in compact_ns if value <= max_srm_n):
        for lambda_value in COMPACT_LAMBDAS:
            if lambda_value < n:
                cases.append(
                    {
                        "schedule": "compact_lambda",
                        "n": n,
                        "lambda_target": lambda_value,
                        "c": 1.0 - lambda_value / n,
                    }
                )
    schedule_functions = (
        ("outer_log_log_n", lambda n: math.log(math.log(n))),
        ("outer_sqrt_log_n", lambda n: math.sqrt(math.log(n))),
        ("outer_n_one_third", lambda n: n ** (1.0 / 3.0)),
    )
    for schedule, function in schedule_functions:
        for n in sorted(value for value in outer_ns if value <= max_srm_n):
            lambda_value = float(function(n))
            cases.append(
                {
                    "schedule": schedule,
                    "n": n,
                    "lambda_target": lambda_value,
                    "c": 1.0 - lambda_value / n,
                }
            )
    return cases


def _write_diagnostics_csv(
    output: Path,
    rows: Iterable[dict[str, float | int | str]],
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate finite weighted-hull SRM/floating-SDP diagnostics; "
            "finite trends are not proof."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "weighted_hull_diagnostics.csv",
    )
    parser.add_argument("--max-srm-n", type=int, default=DEFAULT_MAX_SRM_N)
    parser.add_argument("--max-sdp-n", type=int, default=DEFAULT_MAX_SDP_N)
    parser.add_argument(
        "--max-peak-gib",
        type=float,
        default=DEFAULT_MAX_PEAK_BYTES / 2**30,
        help=(
            "Fail closed before allocation above this conservative dense-"
            "array proxy; it is not a strict CVXPY/process peak bound."
        ),
    )
    parser.add_argument(
        "--max-matrix-dimension",
        type=int,
        default=DEFAULT_MAX_MATRIX_DIMENSION,
        help="Fail closed before allocation above this eigensolver dimension.",
    )
    parser.add_argument(
        "--allow-srm-above-48",
        action="store_true",
        help="Explicitly opt into SRM above n=48 after the resource log.",
    )
    parser.add_argument(
        "--allow-sdp-above-5",
        action="store_true",
        help="Explicitly opt into SDP for 5<n<=7 after the resource log.",
    )
    parser.add_argument("--solver", default="CLARABEL")
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.max_srm_n < 1:
        raise ValueError("--max-srm-n must be positive")
    if not 0 <= args.max_sdp_n <= MAX_EXPLICIT_SDP_N:
        raise ValueError("--max-sdp-n must lie in [0,7]")
    if not math.isfinite(args.max_peak_gib) or args.max_peak_gib <= 0.0:
        raise ValueError("--max-peak-gib must be finite and positive")
    if args.max_matrix_dimension < 1:
        raise ValueError("--max-matrix-dimension must be positive")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _validate_cli_args(args)
    max_peak_bytes = int(args.max_peak_gib * 2**30)
    rows: list[dict[str, float | int | str]] = []

    def log_estimate(estimate: dict[str, int | str]) -> None:
        print(
            "RESOURCE_ESTIMATE "
            f"n={estimate['n']} M_n={estimate['external_M_n']} "
            f"gram_bytes={estimate['gram_bytes']} "
            f"estimated_peak_bytes={estimate['estimated_peak_bytes']} "
            f"estimated_peak_scope={estimate['estimated_peak_scope']} "
            f"eigensolver_dimension={estimate['eigensolver_dimension']} "
            "eigensolver_work="
            f"{estimate['eigensolver_cubic_work_units']}",
            flush=True,
        )

    for case in _diagnostic_cases(args.max_srm_n, args.max_sdp_n):
        row = diagnostic_row(
            int(case["n"]),
            float(case["c"]),
            schedule=str(case["schedule"]),
            max_srm_n=args.max_srm_n,
            max_sdp_n=args.max_sdp_n,
            max_peak_bytes=max_peak_bytes,
            max_matrix_dimension=args.max_matrix_dimension,
            allow_srm_above_default=args.allow_srm_above_48,
            allow_sdp_above_default=args.allow_sdp_above_5,
            solver=args.solver,
            resource_log=log_estimate,
        )
        row["lambda_target"] = float(case["lambda_target"])
        rows.append(row)
        print(
            f"ROW schedule={row['schedule']} n={row['n']} "
            f"lambda={float(row['lambda']):.9g} status={row['status']} "
            f"srm_status={row['srm_status']} sdp_status={row['sdp_status']}",
            flush=True,
        )
    _write_diagnostics_csv(args.output, rows)
    print(f"wrote {args.output.resolve()} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
