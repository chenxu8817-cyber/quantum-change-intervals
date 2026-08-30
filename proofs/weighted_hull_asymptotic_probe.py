"""Finite probes for compact-lambda and moving-outer weighted-hull proofs.

The scalar, balanced-block, Volterra, and rectangular-pinching utilities are
shared by the compact-lambda and adaptive moving-outer analyses.  Reusable
block--tail certificate utilities live in
:mod:`proofs.weighted_block_tail_probe` and are re-exported here to preserve
the Task 3 public API.
"""

from __future__ import annotations

from math import ceil, isfinite, log, pi, sqrt
from numbers import Integral, Real
from operator import index as integer_index
from typing import Literal, Sequence, TypedDict

import numpy as np
from scipy.special import ellipk

from proofs.weighted_block_tail_probe import (
    extend_certificate,
    extend_with_budget,
    global_hull_certificate,
    left_suffix,
    minimum_certificate_slack,
    optimized_extension,
    physical_hull_rows,
    right_suffix,
    split_left_suffix,
    split_right_suffix,
    tensor_cell_certificate,
    weighted_local_certificate,
)


class ContinuumParameters(TypedDict):
    """Balanced-block parameters for ``c = 1 - lambda_value / n``."""

    n: int
    lambda_value: float
    c: float
    ell: float
    p1: float
    block_count: int
    minimum_block_size: int
    sizes: tuple[int, ...]
    m_epsilon: float
    log_m_over_ell: float


class AdaptiveBlockParameters(TypedDict):
    """Exact finite ledger for the adaptive moving-overlap partition."""

    n: int
    c: float
    lambda_value: float
    h: float
    ell: float
    p1: float
    block_count: int
    minimum_block_size: int
    sizes: tuple[int, ...]
    lambda_over_B: float
    B_over_h: float
    m_p1: float
    log_m_over_ell: float


class OuterErrorScales(TypedDict):
    """Finite diagnostic scales used by the moving-outer proof ledger."""

    local_similarity: float
    volterra_repair: float
    log_match: float
    tail_overhead: float
    diagonal_cells: float
    one_excitation: float
    vacuum: float
    total_scale_proxy: float


class RegimeDiagnostics(TypedDict):
    """Finite scales and a caller-selected threshold label.

    ``threshold_regime`` is only a finite diagnostic band.  It is not an
    asymptotic classifier and does not certify that a sequence lies in any
    theorem regime.
    """

    n: int
    c: float
    lambda_value: float
    h: float
    ell: float
    p1: float
    tau: float
    inner_lambda_max: float
    outer_lambda_min: float
    threshold_regime: Literal["inner", "continuum", "outer"]


def ell(c: float) -> float:
    """Return ``pi J(c) = 2 K(c^2)`` for ``0 <= c < 1``."""
    overlap = _finite_real(c, "c")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("ell requires 0 <= c < 1")
    return float(2.0 * ellipk(overlap * overlap))


def p1(c: float) -> float:
    """Return the endpoint-safe one-boundary success law."""
    overlap = _finite_real(c, "c")
    if overlap == 1.0:
        return 0.0
    if not 0.0 <= overlap < 1.0:
        raise ValueError("p1 requires 0 <= c <= 1")
    length = ell(overlap)
    return float((1.0 - overlap * overlap) * length * length / (pi * pi))


def weighted_volterra(m: int, c: float) -> np.ndarray:
    """Return ``c**(u-a)`` on the upper triangle and zero below it."""
    size = _positive_int(m, "m")
    overlap = _finite_real(c, "c")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("c must lie in [0,1]")

    rows = np.arange(size)[:, None]
    columns = np.arange(size)[None, :]
    exponents = columns - rows
    mask = exponents >= 0
    out = np.zeros((size, size), dtype=float)
    out[mask] = np.power(overlap, exponents[mask])
    return out


def balanced_block_sizes(n: int, block_count: int) -> tuple[int, ...]:
    """Split ``n`` sites into nonempty sizes differing by at most one."""
    site_count = _positive_int(n, "n")
    count = _positive_int(block_count, "block_count")
    if count > site_count:
        raise ValueError("block_count cannot exceed n")
    minimum_size, remainder = divmod(site_count, count)
    return tuple(
        minimum_size + (1 if block < remainder else 0)
        for block in range(count)
    )


def continuum_parameters(n: int, lambda_value: float) -> ContinuumParameters:
    """Return the balanced compact-lambda ledger with ``B = ceil(ell)``."""
    site_count = _positive_int(n, "n")
    lambda_number = _finite_real(lambda_value, "lambda_value")
    if not 0.0 < lambda_number <= site_count:
        raise ValueError("lambda_value must satisfy 0 < lambda_value <= n")
    overlap = 1.0 - lambda_number / site_count
    length = ell(overlap)
    block_count = int(ceil(length))
    if block_count > site_count:
        raise ValueError("n is too small for ceil(ell(c)) nonempty blocks")
    sizes = balanced_block_sizes(site_count, block_count)
    minimum_size = min(sizes)
    return {
        "n": site_count,
        "lambda_value": lambda_number,
        "c": overlap,
        "ell": length,
        "p1": p1(overlap),
        "block_count": block_count,
        "minimum_block_size": minimum_size,
        "sizes": sizes,
        "m_epsilon": minimum_size * (1.0 - overlap),
        "log_m_over_ell": log(minimum_size) / length,
    }


def adaptive_block_parameters(n: int, c: float) -> AdaptiveBlockParameters:
    """Return the adaptive moving-overlap ledger when ``B <= n``.

    The finite API mirrors the asymptotic parameter lemma: ``n`` is an integer
    at least two, ``0 <= c < 1``, and the approved adaptive block count must
    admit a nonempty balanced partition.  The theorem proves that the final
    condition holds eventually in its moving-overlap regime.
    """
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be an integer with n >= 2")
    site_count = int(n)
    if site_count < 2:
        raise ValueError("n must be an integer with n >= 2")

    overlap = _finite_real(c, "c")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("c must satisfy 0 <= c < 1")

    try:
        site_count_float = float(site_count)
    except OverflowError as error:
        raise ValueError(
            "n is too large for a finite floating-point ledger"
        ) from error
    if not isfinite(site_count_float):
        raise ValueError("n is too large for a finite floating-point ledger")

    length = ell(overlap)
    probability = p1(overlap)
    lambda_value = site_count_float * (1.0 - overlap)
    hull_scale = site_count_float * probability
    if (
        not isfinite(length)
        or not isfinite(probability)
        or not isfinite(lambda_value)
        or not isfinite(hull_scale)
    ):
        raise ValueError(
            "adaptive ledger requires finite ell, p1, lambda, and h: "
            f"ell={length!r}, p1={probability!r}, "
            f"lambda={lambda_value!r}, h={hull_scale!r}"
        )
    if hull_scale <= 0.0:
        raise ValueError(
            "positive finite hull scale required before normalization: "
            f"h={hull_scale!r}"
        )
    if length <= 0.0 or probability <= 0.0 or lambda_value <= 0.0:
        raise ValueError(
            "adaptive ledger requires positive ell, p1, and lambda: "
            f"ell={length!r}, p1={probability!r}, lambda={lambda_value!r}"
        )

    real_block_scale = sqrt(hull_scale) + lambda_value * sqrt(length)
    if not isfinite(real_block_scale):
        raise ValueError(
            "adaptive block scale must be finite: "
            f"sqrt(h)+lambda*sqrt(ell)={real_block_scale!r}"
        )
    block_count = int(ceil(real_block_scale))
    if block_count > site_count:
        raise ValueError(
            "finite request is before the partition regime: "
            f"B={block_count} exceeds n={site_count} for c={overlap:.17g}; "
            f"lambda={lambda_value:.17g}, h={hull_scale:.17g}, "
            f"ell={length:.17g}"
        )

    sizes = balanced_block_sizes(site_count, block_count)
    minimum_size = min(sizes)
    return {
        "n": site_count,
        "c": overlap,
        "lambda_value": lambda_value,
        "h": hull_scale,
        "ell": length,
        "p1": probability,
        "block_count": block_count,
        "minimum_block_size": minimum_size,
        "sizes": sizes,
        "lambda_over_B": lambda_value / block_count,
        "B_over_h": block_count / hull_scale,
        "m_p1": minimum_size * probability,
        "log_m_over_ell": log(minimum_size) / length,
    }


def outer_error_scale_proxy(n: int, c: float) -> OuterErrorScales:
    """Return the sufficient moving-outer diagnostic scale ledger.

    Hidden constants in the proof's ``O(.)`` terms are absent.  Consequently,
    ``total_scale_proxy`` is neither an upper confidence bound nor a certified
    finite-``n`` error bound.
    """
    parameters = adaptive_block_parameters(n, c)
    lambda_value = parameters["lambda_value"]
    hull_scale = parameters["h"]
    length = parameters["ell"]
    block_count = parameters["block_count"]
    minimum_size = parameters["minimum_block_size"]
    if minimum_size <= 2 or lambda_value <= 0.0 or hull_scale <= 0.0:
        raise ValueError("outside diagnostic ledger domain")

    terms: OuterErrorScales = {
        "local_similarity": lambda_value / block_count,
        "volterra_repair": (
            log(log(minimum_size)) / log(minimum_size)
        ),
        "log_match": abs(log(minimum_size) / length - 1.0),
        "tail_overhead": sqrt(block_count / hull_scale),
        "diagonal_cells": 1.0 / block_count,
        "one_excitation": (
            log(parameters["n"])
            / (sqrt(lambda_value) * length * length)
        ),
        "vacuum": 1.0 / hull_scale,
        "total_scale_proxy": 0.0,
    }
    terms["total_scale_proxy"] = sum(terms.values())
    return terms


def regime_diagnostics(
    n: int,
    c: float,
    *,
    inner_lambda_max: float,
    outer_lambda_min: float,
) -> RegimeDiagnostics:
    """Return finite scales and a caller-selected threshold-band label.

    The thresholds are mandatory finite inputs satisfying
    ``0 < inner_lambda_max < outer_lambda_min``.  The returned label reports
    only which closed/open finite band contains ``lambda = n * (1 - c)``:
    it is not an asymptotic classifier and is not evidence that a sequence
    satisfies an inner, continuum, or moving-outer theorem.
    """
    site_count = _positive_int(n, "n")
    if site_count < 2:
        raise ValueError("n must be an integer with n >= 2")
    overlap = _finite_real(c, "c")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("c must satisfy 0 <= c < 1")
    inner_max = _finite_real(inner_lambda_max, "inner_lambda_max")
    outer_min = _finite_real(outer_lambda_min, "outer_lambda_min")
    if not 0.0 < inner_max < outer_min:
        raise ValueError(
            "thresholds must satisfy "
            "0 < inner_lambda_max < outer_lambda_min"
        )

    try:
        site_count_float = float(site_count)
    except OverflowError as error:
        raise ValueError(
            "n is too large for a finite floating-point diagnostic"
        ) from error
    if not isfinite(site_count_float):
        raise ValueError("n is too large for a finite floating-point diagnostic")

    length = ell(overlap)
    probability = p1(overlap)
    lambda_value = site_count_float * (1.0 - overlap)
    hull_scale = site_count_float * probability
    tau = lambda_value * log(site_count_float) ** 2
    if not all(
        isfinite(value)
        for value in (length, probability, lambda_value, hull_scale, tau)
    ):
        raise ValueError("finite regime scales are required")

    if lambda_value <= inner_max:
        threshold_regime: Literal["inner", "continuum", "outer"] = "inner"
    elif lambda_value >= outer_min:
        threshold_regime = "outer"
    else:
        threshold_regime = "continuum"

    return {
        "n": site_count,
        "c": overlap,
        "lambda_value": lambda_value,
        "h": hull_scale,
        "ell": length,
        "p1": probability,
        "tau": tau,
        "inner_lambda_max": inner_max,
        "outer_lambda_min": outer_min,
        "threshold_regime": threshold_regime,
    }


def plateau_overlap(n: int, anchors: Sequence[int]) -> float:
    """Return the exact plateau value on a validated finite anchor prefix.

    ``anchors`` must contain at least two integers, start at two or later, be
    strictly increasing, and obey ``N[k + 1] = N[k] ** 2`` exactly.  A finite
    list represents only the half-open range ``[anchors[0], anchors[-1])``;
    requests outside that range are rejected rather than extrapolated.
    """
    site_count = _positive_int(n, "n")
    try:
        raw_anchors = tuple(anchors)
    except TypeError as error:
        raise ValueError("anchors must be an iterable of integers") from error
    if len(raw_anchors) < 2:
        raise ValueError("anchors must contain at least two values")
    normalized = tuple(_positive_int(anchor, "anchor") for anchor in raw_anchors)
    if normalized[0] < 2:
        raise ValueError("the first anchor must be at least two")
    for left, right in zip(normalized, normalized[1:]):
        if right <= left:
            raise ValueError("anchors must be strictly increasing")
        if right != left * left:
            raise ValueError("anchors must satisfy N[k + 1] = N[k] ** 2")
    if not normalized[0] <= site_count < normalized[-1]:
        raise ValueError("n lies outside the finite anchor prefix")

    for left, right in zip(normalized, normalized[1:]):
        if left <= site_count < right:
            return 1.0 - 1.0 / float(left * left)
    raise AssertionError("validated anchor prefix did not contain n")


def hermitian_dilation(a: np.ndarray) -> np.ndarray:
    """Return ``[[0, A], [A*, 0]]`` for a finite numeric matrix ``A``."""
    array = _finite_numeric_matrix(a)
    row_count, column_count = array.shape
    return np.block(
        [
            [np.zeros((row_count, row_count), dtype=array.dtype), array],
            [array.conjugate().T, np.zeros((column_count, column_count), dtype=array.dtype)],
        ]
    )


def rectangular_block_pinching(
    a: np.ndarray,
    row_blocks: Sequence[Sequence[int]],
    col_blocks: Sequence[Sequence[int]],
) -> np.ndarray:
    """Keep only matched, pairwise-disjoint rectangular row/column blocks."""
    array = _finite_numeric_matrix(a)
    try:
        raw_row_blocks = tuple(row_blocks)
        raw_col_blocks = tuple(col_blocks)
    except TypeError as error:
        raise ValueError("row_blocks and col_blocks must be iterable") from error
    if len(raw_row_blocks) != len(raw_col_blocks):
        raise ValueError("row_blocks and col_blocks must have equal length")

    rows = _validated_blocks(raw_row_blocks, array.shape[0], "row")
    columns = _validated_blocks(raw_col_blocks, array.shape[1], "column")
    out = np.zeros_like(array)
    for row_block, column_block in zip(rows, columns):
        out[np.ix_(row_block, column_block)] = array[
            np.ix_(row_block, column_block)
        ]
    return out


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_numeric_matrix(a: np.ndarray) -> np.ndarray:
    array = np.asarray(a)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("a must be a two-dimensional numeric matrix")
    if not np.isfinite(array).all():
        raise ValueError("a must contain only finite entries")
    return array


def _validated_blocks(
    blocks: Sequence[Sequence[int]], limit: int, family: str
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for block in blocks:
        try:
            raw_indices = tuple(block)
        except TypeError as error:
            raise ValueError(f"each {family} block must be iterable") from error
        indices: list[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, bool):
                raise ValueError(f"{family} indices must be integers")
            try:
                location = integer_index(raw_index)
            except TypeError as error:
                raise ValueError(f"{family} indices must be integers") from error
            if not 0 <= location < limit:
                raise ValueError(f"{family} index out of range")
            if location in seen:
                raise ValueError(f"{family} blocks must be pairwise disjoint")
            seen.add(location)
            indices.append(location)
        normalized.append(tuple(indices))
    return tuple(normalized)


__all__ = [
    "AdaptiveBlockParameters",
    "ContinuumParameters",
    "OuterErrorScales",
    "RegimeDiagnostics",
    "adaptive_block_parameters",
    "balanced_block_sizes",
    "continuum_parameters",
    "ell",
    "extend_certificate",
    "extend_with_budget",
    "global_hull_certificate",
    "hermitian_dilation",
    "left_suffix",
    "minimum_certificate_slack",
    "optimized_extension",
    "outer_error_scale_proxy",
    "p1",
    "plateau_overlap",
    "physical_hull_rows",
    "rectangular_block_pinching",
    "regime_diagnostics",
    "right_suffix",
    "split_left_suffix",
    "split_right_suffix",
    "tensor_cell_certificate",
    "weighted_local_certificate",
    "weighted_volterra",
]
