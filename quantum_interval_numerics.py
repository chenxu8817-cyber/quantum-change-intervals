"""Shared numerical primitives for fixed-m quantum interval localization."""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".local_pydeps"
if os.environ.get("QCI_USE_LOCAL_DEPS") == "1" and LOCAL_DEPS.exists():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(".local_pydeps is a CPython 3.12 cache")
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
from scipy.special import ellipk


Boundary = tuple[int, ...]


def interval_candidates(n: int, interval_count: int) -> list[Boundary]:
    """Return all strictly increasing 2m-boundary tuples."""
    if interval_count < 1:
        raise ValueError("interval_count must be positive")
    if n < 2 * interval_count - 1:
        return []
    return list(itertools.combinations(range(n + 1), 2 * interval_count))


def interval_mask(boundaries: Boundary) -> int:
    """Encode the anomalous set as a Python integer bit mask."""
    if len(boundaries) % 2:
        raise ValueError("a boundary tuple must have even length")
    mask = 0
    for left, right in zip(boundaries[::2], boundaries[1::2]):
        mask |= ((1 << (right - left)) - 1) << left
    return mask


def incidence_matrix(
    n: int, points: list[Boundary]
) -> np.ndarray:
    """Return the candidate-by-site binary incidence matrix."""
    incidence = np.zeros((len(points), n), dtype=np.int16)
    for row, boundaries in enumerate(points):
        for left, right in zip(boundaries[::2], boundaries[1::2]):
            incidence[row, left:right] = 1
    return incidence


def symmetric_difference_size(x: Boundary, y: Boundary) -> int:
    """Exact symmetric-difference cardinality of two interval unions."""
    return (interval_mask(x) ^ interval_mask(y)).bit_count()


def endpoint_distance(x: Boundary, y: Boundary) -> int:
    """The l1 distance between corresponding ordered boundaries."""
    if len(x) != len(y):
        raise ValueError("boundary tuples must have equal length")
    return sum(abs(left - right) for left, right in zip(x, y))


def exact_interval_gram(
    n: int, interval_count: int, overlap: float
) -> tuple[np.ndarray, list[Boundary]]:
    """Construct G[x,y] = c**|S_x symmetric-difference S_y|."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0,1]")
    points = interval_candidates(n, interval_count)
    incidence = incidence_matrix(n, points)
    weights = incidence.sum(axis=1)
    symmetric_difference = (
        weights[:, None] + weights[None, :] - 2 * incidence @ incidence.T
    )
    gram = np.power(float(overlap), symmetric_difference, dtype=float)
    return gram, points


def psd_sqrt(
    matrix: np.ndarray, negative_tolerance: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Return the principal PSD square root and raw eigenvalues."""
    symmetrized = (matrix + matrix.T.conj()) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -negative_tolerance * scale:
        raise ValueError(
            "matrix is not PSD: "
            f"lambda_min={float(eigenvalues[0]):.6e}, scale={scale:.6e}"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    root = (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T.conj()
    return root, eigenvalues


def srm_quantities(gram: np.ndarray) -> dict[str, float | int]:
    """Compute exact dense SRM and Gram-bound diagnostics."""
    hypothesis_count = gram.shape[0]
    if gram.shape != (hypothesis_count, hypothesis_count):
        raise ValueError("gram must be square")
    root, eigenvalues = psd_sqrt(gram)
    diagonal = np.real(np.diag(root))
    root_trace = float(diagonal.sum())
    trace_lower = (root_trace / hypothesis_count) ** 2
    srm = float(np.mean(diagonal**2))
    q = diagonal / root_trace
    q_l1 = float(np.abs(q - 1.0 / hypothesis_count).sum())
    lambda_max = float(eigenvalues[-1])
    helstrom_upper = min(
        1.0, trace_lower + math.sqrt(max(0.0, lambda_max)) * q_l1
    )
    rank_tolerance = 1e-12 * max(1.0, lambda_max)
    numerical_rank = int(np.count_nonzero(eigenvalues > rank_tolerance))
    positive = eigenvalues[eigenvalues > rank_tolerance]
    condition_number = (
        float(lambda_max / positive[0]) if positive.size else math.inf
    )
    return {
        "trace_lower_bound": trace_lower,
        "srm": srm,
        "q_l1": q_l1,
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": lambda_max,
        "numerical_rank": numerical_rank,
        "condition_number_on_support": condition_number,
        "helstrom_upper": helstrom_upper,
    }


def p1(overlap: float) -> float:
    """One-change-point asymptotic success probability."""
    if overlap <= 0.0:
        return 1.0
    if overlap >= 1.0:
        return 0.0
    return float(
        4.0
        * (1.0 - overlap * overlap)
        * ellipk(overlap * overlap) ** 2
        / math.pi**2
    )


def validate_endpoint_dichotomy(
    max_interval_count: int, extra_sites: int
) -> None:
    """Exhaustively verify D<=E and D!=E => D>=L on small instances."""
    for interval_count in range(1, max_interval_count + 1):
        endpoint_count = 2 * interval_count
        for n in range(endpoint_count, endpoint_count + extra_sites + 1):
            all_points = interval_candidates(n, interval_count)
            max_length = n // (endpoint_count - 1)
            for length in range(1, max_length + 1):
                points = [
                    x
                    for x in all_points
                    if min(
                        x[k + 1] - x[k]
                        for k in range(endpoint_count - 1)
                    )
                    >= length
                ]
                masks = [interval_mask(x) for x in points]
                for row, x in enumerate(points):
                    for col, y in enumerate(points):
                        difference = (masks[row] ^ masks[col]).bit_count()
                        distance = endpoint_distance(x, y)
                        if difference > distance:
                            raise AssertionError(
                                "D <= E failed: "
                                f"m={interval_count}, n={n}, L={length}, "
                                f"x={x}, y={y}, D={difference}, E={distance}"
                            )
                        if difference != distance and difference < length:
                            raise AssertionError(
                                "endpoint dichotomy failed: "
                                f"m={interval_count}, n={n}, L={length}, "
                                f"x={x}, y={y}, D={difference}, E={distance}"
                            )
