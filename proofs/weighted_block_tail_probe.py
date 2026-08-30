"""Finite-dimensional utilities for the reusable weighted block--tail lemma.

All public validation is numerical but scale aware.  The mathematical lemma
itself is exact: no inverse, regularization, or continuity argument is used.
"""

from __future__ import annotations

from math import fsum, isfinite, sqrt
from numbers import Integral, Real
from operator import index as integer_index

import numpy as np


_MACHINE_EPSILON = float(np.finfo(float).eps)
_ROUNDING_FACTOR = 64.0


def _scaled_tolerance(scale: float, dimension: int) -> float:
    """Return a roundoff allowance scaled by norm, size, and machine epsilon."""
    return (
        _ROUNDING_FACTOR
        * _MACHINE_EPSILON
        * max(1, int(dimension))
        * abs(float(scale))
    )


def _finite_budget(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("budget must be a finite real number")
    budget = float(value)
    if not isfinite(budget):
        raise ValueError(f"budget must be finite; measured budget={budget}")
    if budget < 0.0:
        raise ValueError(f"budget must be nonnegative; measured budget={budget}")
    return budget


def _numeric_square(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(
            f"{name} must be a two-dimensional numeric square matrix; "
            f"measured shape={array.shape}"
        )
    nonfinite = int(array.size - np.count_nonzero(np.isfinite(array)))
    if nonfinite:
        raise ValueError(f"{name} has nonfinite entries; measured count={nonfinite}")
    return array


def _numeric_vector(
    value: np.ndarray, expected_size: int, name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"{name} must be a one-dimensional numeric vector; "
            f"measured shape={array.shape}"
        )
    if array.shape[0] != expected_size:
        raise ValueError(
            f"{name} dimension mismatch; measured size={array.shape[0]}, "
            f"expected size={expected_size}"
        )
    nonfinite = int(array.size - np.count_nonzero(np.isfinite(array)))
    if nonfinite:
        raise ValueError(f"{name} has nonfinite entries; measured count={nonfinite}")
    return array


def _validated_positive_operator(q: np.ndarray) -> tuple[np.ndarray, float, float]:
    local = _numeric_square(q, "q")
    dimension = local.shape[0]
    operator_scale = float(np.linalg.norm(local, 2)) if dimension else 0.0
    hermitian_violation = float(np.linalg.norm(local - local.conjugate().T, 2))
    hermitian_tolerance = _scaled_tolerance(operator_scale, dimension)
    if hermitian_violation > hermitian_tolerance:
        raise ValueError(
            "q Hermitian violation="
            f"{hermitian_violation:.17g} exceeds tolerance="
            f"{hermitian_tolerance:.17g}"
        )
    hermitian = (local + local.conjugate().T) / 2.0
    minimum_eigenvalue = (
        float(np.linalg.eigvalsh(hermitian)[0]) if dimension else 0.0
    )
    psd_violation = max(0.0, -minimum_eigenvalue)
    psd_tolerance = _scaled_tolerance(operator_scale, dimension)
    if psd_violation > psd_tolerance:
        raise ValueError(
            f"q PSD violation={psd_violation:.17g} exceeds tolerance="
            f"{psd_tolerance:.17g}"
        )
    if minimum_eigenvalue < 0.0:
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        clipped = np.maximum(eigenvalues, 0.0)
        hermitian = (eigenvectors * clipped) @ eigenvectors.conjugate().T
        hermitian = (hermitian + hermitian.conjugate().T) / 2.0
    trace = fsum(float(value.real) for value in np.diag(hermitian))
    return hermitian, operator_scale, trace


def _raw_extension_formula(
    q: np.ndarray, w: np.ndarray, alpha: float
) -> np.ndarray:
    """Evaluate only the shared extension formula, without validation."""
    return (
        (1.0 + alpha) * q
        + (1.0 + 1.0 / alpha) * np.outer(w, w.conjugate())
    )


def extend_certificate(q: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Return ``(1+alpha)Q + (1+alpha^-1)|w><w|`` for ``alpha>0``."""
    local = _numeric_square(q, "q")
    tail = _numeric_vector(w, local.shape[0], "w")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite positive real number")
    scale = float(alpha)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError(f"alpha must be finite and positive; measured alpha={scale}")
    return _raw_extension_formula(local, tail, scale)


def extend_with_budget(
    q: np.ndarray, w: np.ndarray, budget: float
) -> np.ndarray:
    """Return the checked singular-safe extension with an upper trace budget.

    Positive budgets always use the declared budget scale, however small.
    The exact zero branch accepts only the exact zero operator.
    """
    limit = _finite_budget(budget)
    supplied = np.asarray(q)
    local, operator_scale, local_trace = _validated_positive_operator(q)
    return _assemble_prevalidated_extension(
        local, supplied, w, limit, operator_scale, local_trace
    )


def _assemble_prevalidated_extension(
    local: np.ndarray,
    supplied: np.ndarray,
    w: np.ndarray,
    limit: float,
    operator_scale: float,
    local_trace: float,
) -> np.ndarray:
    """Assemble from one validated Q representative without revalidation."""
    tail = _numeric_vector(w, local.shape[0], "w")
    dimension = local.shape[0]

    tail_norm = float(np.linalg.norm(tail, 2))
    tail_violation = max(0.0, tail_norm - 1.0)
    # The constraint has the fixed threshold one, unlike the homogeneous
    # operator tests above, so its natural scale is max(1, ||w||).
    tail_tolerance = _scaled_tolerance(max(1.0, tail_norm), dimension)
    if tail_violation > tail_tolerance:
        raise ValueError(
            f"tail-norm violation={tail_violation:.17g} exceeds tolerance="
            f"{tail_tolerance:.17g}; measured norm={tail_norm:.17g}"
        )

    if limit == 0.0:
        # Exact zero-budget semantics use the norm of the supplied matrix,
        # not the PSD-clipped representative.
        measured_norm = float(np.linalg.norm(supplied, 2))
        if measured_norm != 0.0:
            raise ValueError(
                "zero-budget Q norm="
                f"{measured_norm:.17g}; exact Q=0 is required"
            )
        return np.outer(tail, tail.conjugate())

    trace_violation = max(0.0, local_trace - limit)
    trace_scale = max(operator_scale, abs(local_trace), abs(limit))
    trace_tolerance = _scaled_tolerance(trace_scale, dimension)
    # Trace feasibility is one-sided and exact for the validated PSD
    # representative.  There is no Loewner-safe operation that can shrink Q
    # to the budget while preserving an unknown Q >= xx*.  The homogeneous
    # tolerance is reported diagnostically, but every positive excess is
    # rejected.  Stable summation above avoids an avoidable accumulation loss.
    if trace_violation > 0.0:
        raise ValueError(
            f"trace-budget violation={trace_violation:.17g} exceeds tolerance="
            f"{trace_tolerance:.17g}; measured trace={local_trace:.17g}, "
            f"budget={limit:.17g}"
        )

    alpha = limit ** -0.5
    return _raw_extension_formula(local, tail, alpha)


def optimized_extension(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Use ``tr(Q)`` as the budget, with the exact zero branch preserved."""
    # Derive the optimized budget from the same Hermitian PSD representative
    # and stable diagonal sum used by ``extend_with_budget``.  This avoids a
    # one-ulp disagreement between ``np.trace`` and the checked trace without
    # shrinking Q or relaxing the one-sided trace constraint.
    supplied = np.asarray(q)
    local, operator_scale, budget = _validated_positive_operator(q)
    return _assemble_prevalidated_extension(
        local, supplied, w, budget, operator_scale, budget
    )


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


def _closed_overlap(c: float) -> float:
    overlap = _finite_real(c, "c")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("c must lie in [0,1]")
    return overlap


def _site_index(value: int, n: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer site index")
    try:
        location = integer_index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer site index") from error
    if not 0 <= location < n:
        raise ValueError(f"{name} is outside the site range")
    return int(location)


def _validated_interval(
    start: int, stop: int, n: int, name: str
) -> tuple[int, int]:
    left = _site_index(start, n, f"{name}_start")
    right = _site_index(stop, n, f"{name}_stop")
    if left > right:
        raise ValueError(f"{name} must have start <= stop")
    return left, right


def _balanced_block_sizes(n: int, block_count: int) -> tuple[int, ...]:
    site_count = _positive_int(n, "n")
    count = _positive_int(block_count, "block_count")
    if count > site_count:
        raise ValueError("block_count cannot exceed n")
    minimum_size, remainder = divmod(site_count, count)
    return tuple(
        minimum_size + (1 if block < remainder else 0)
        for block in range(count)
    )


def right_suffix(n: int, a: int, c: float) -> np.ndarray:
    """Return ``sqrt(1-c^2)c^(k-a) 1[k>=a]``, including endpoints."""
    size = _positive_int(n, "n")
    anchor = _site_index(a, size, "a")
    overlap = _closed_overlap(c)
    out = np.zeros(size, dtype=float)
    exponents = np.arange(size - anchor)
    out[anchor:] = sqrt(1.0 - overlap * overlap) * np.power(
        overlap, exponents
    )
    return out


def left_suffix(n: int, b: int, c: float) -> np.ndarray:
    """Return the reflected normalized suffix at the right endpoint."""
    size = _positive_int(n, "n")
    anchor = _site_index(b, size, "b")
    overlap = _closed_overlap(c)
    return right_suffix(size, size - 1 - anchor, overlap)[::-1].copy()


def split_right_suffix(
    n: int, a: int, c: float, block_start: int, block_stop: int
) -> tuple[np.ndarray, float, np.ndarray]:
    """Split a right suffix into its local part and one common tail."""
    size = _positive_int(n, "n")
    start, stop = _validated_interval(block_start, block_stop, size, "block")
    anchor = _site_index(a, size, "a")
    if not start <= anchor <= stop:
        raise ValueError("a must lie in the declared block")
    overlap = _closed_overlap(c)
    full = right_suffix(size, anchor, overlap)
    local = full.copy()
    local[stop + 1 :] = 0.0
    tail = np.zeros(size, dtype=float)
    if stop + 1 < size:
        tail[stop + 1 :] = right_suffix(size - stop - 1, 0, overlap)
    gamma = overlap ** (stop + 1 - anchor)
    return local, float(gamma), tail


def split_left_suffix(
    n: int, b: int, c: float, block_start: int, block_stop: int
) -> tuple[np.ndarray, float, np.ndarray]:
    """Obtain the left split only by coordinate reflection."""
    size = _positive_int(n, "n")
    start, stop = _validated_interval(block_start, block_stop, size, "block")
    anchor = _site_index(b, size, "b")
    if not start <= anchor <= stop:
        raise ValueError("b must lie in the declared block")
    local, gamma, tail = split_right_suffix(
        size,
        size - 1 - anchor,
        c,
        size - 1 - stop,
        size - 1 - start,
    )
    return local[::-1].copy(), gamma, tail[::-1].copy()


def weighted_local_certificate(
    m: int, c: float, k0: np.ndarray | None = None
) -> np.ndarray:
    """Return the weighted local certificate, with direct endpoint branches."""
    size = _positive_int(m, "m")
    overlap = _closed_overlap(c)
    if overlap == 0.0:
        return np.eye(size)
    if overlap == 1.0:
        return np.zeros((size, size), dtype=float)

    if k0 is None:
        indices = np.arange(size)
        base = np.minimum(indices[:, None], indices[None, :]) + 1.0
    else:
        base, _, _ = _validated_positive_operator(k0)
        if base.shape != (size, size):
            raise ValueError(
                f"k0 dimension mismatch; measured shape={base.shape}, "
                f"expected shape={(size, size)}"
            )

    base_scale = max(1.0, float(np.linalg.norm(base, 2)))
    dominance_tolerance = _scaled_tolerance(base_scale, size) * 8.0
    for anchor in range(size):
        suffix = np.zeros(size, dtype=base.dtype)
        suffix[anchor:] = 1
        projector = np.outer(suffix, suffix.conjugate())
        slack = (base - projector + (base - projector).conjugate().T) / 2.0
        violation = max(0.0, -float(np.linalg.eigvalsh(slack)[0]))
        if violation > dominance_tolerance:
            raise ValueError(
                "k0 suffix-dominance violation="
                f"{violation:.17g} exceeds tolerance={dominance_tolerance:.17g}; "
                f"anchor={anchor}"
            )

    diagonal = np.power(overlap, np.arange(size))
    congruence = diagonal[:, None] * base * diagonal[None, :]
    factor = (1.0 - overlap * overlap) * overlap ** (-2 * (size - 1))
    return factor * congruence


def tensor_cell_certificate(
    right_certificate: np.ndarray, left_certificate: np.ndarray
) -> np.ndarray:
    """Tensor positive endpoint certificates and compress to strict pairs."""
    right = _numeric_square(right_certificate, "right_certificate")
    left = _numeric_square(left_certificate, "left_certificate")
    if right.shape != left.shape:
        raise ValueError("endpoint certificates must have equal dimensions")
    size = right.shape[0]
    indices = [u * size + v for u in range(size) for v in range(u + 1, size)]
    product = np.kron(right, left)
    return product[np.ix_(indices, indices)]


def physical_hull_rows(n: int, c: float) -> np.ndarray:
    """Return all physical hull rows, retaining zero singleton hypotheses."""
    size = _positive_int(n, "n")
    overlap = _closed_overlap(c)
    intervals = [(a, b) for a in range(size) for b in range(a, size)]
    pairs = [(u, v) for u in range(size) for v in range(u + 1, size)]
    physical = 1.0 - overlap * overlap
    return np.asarray(
        [
            [
                physical * overlap ** ((u - a) + (b - v))
                if a <= u < v <= b
                else 0.0
                for u, v in pairs
            ]
            for a, b in intervals
        ],
        dtype=float,
    )


def _blocks_from_sizes(sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    position = 0
    for local_size in sizes:
        blocks.append((position, position + local_size - 1))
        position += local_size
    return blocks


def _endpoint_cell_projectors(
    n: int, blocks: list[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    hull_size = len(pairs)
    cells: dict[tuple[int, int], np.ndarray] = {}
    for p in range(len(blocks)):
        for q in range(p, len(blocks)):
            operator = np.zeros((hull_size, hull_size), dtype=float)
            p_start, p_stop = blocks[p]
            q_start, q_stop = blocks[q]
            for coordinate, (u, v) in enumerate(pairs):
                if p_start <= u <= p_stop and q_start <= v <= q_stop:
                    operator[coordinate, coordinate] = 1.0
            cells[(p, q)] = operator
    return cells


def global_hull_certificate(
    n: int, c: float, block_count: int, *, return_cells: bool = False
):
    """Sum strict-triangle endpoint-cell certificates for the full hull."""
    size = _positive_int(n, "n")
    overlap = _closed_overlap(c)
    count = _positive_int(block_count, "block_count")
    sizes = _balanced_block_sizes(size, count)
    blocks = _blocks_from_sizes(sizes)
    hull_size = size * (size - 1) // 2

    if overlap == 0.0:
        cells = _endpoint_cell_projectors(size, blocks)
        certificate = sum(cells.values(), np.zeros((hull_size, hull_size)))
        return (certificate, cells) if return_cells else certificate
    if overlap == 1.0:
        cells = {
            (p, q): np.zeros((hull_size, hull_size), dtype=float)
            for p in range(count)
            for q in range(p, count)
        }
        certificate = np.zeros((hull_size, hull_size), dtype=float)
        return (certificate, cells) if return_cells else certificate

    right_certificates: list[np.ndarray] = []
    left_certificates: list[np.ndarray] = []
    for start, stop in blocks:
        local_size = stop - start + 1
        local = weighted_local_certificate(local_size, overlap)
        embedded_right = np.zeros((size, size), dtype=local.dtype)
        embedded_right[start : stop + 1, start : stop + 1] = local
        right_tail = split_right_suffix(size, start, overlap, start, stop)[2]
        right_certificates.append(
            optimized_extension(embedded_right, right_tail)
        )

        reflected_local = local[::-1, ::-1]
        embedded_left = np.zeros((size, size), dtype=local.dtype)
        embedded_left[start : stop + 1, start : stop + 1] = reflected_local
        left_tail = split_left_suffix(size, stop, overlap, start, stop)[2]
        left_certificates.append(optimized_extension(embedded_left, left_tail))

    certificate = np.zeros((hull_size, hull_size), dtype=float)
    cells: dict[tuple[int, int], np.ndarray] = {}
    for p in range(count):
        for q in range(p, count):
            cell = tensor_cell_certificate(
                right_certificates[p], left_certificates[q]
            )
            cells[(p, q)] = cell
            certificate = certificate + cell
    return (certificate, cells) if return_cells else certificate


def minimum_certificate_slack(
    certificate: np.ndarray, state_rows: np.ndarray
) -> float:
    """Return the least eigenvalue over all certificate/projector slacks."""
    operator = _numeric_square(certificate, "certificate")
    rows = np.asarray(state_rows)
    if (
        rows.ndim != 2
        or not np.issubdtype(rows.dtype, np.number)
        or not np.isfinite(rows).all()
        or rows.shape[1] != operator.shape[0]
        or rows.shape[0] == 0
    ):
        raise ValueError("state_rows must be a nonempty finite compatible matrix")
    minimum = float("inf")
    for row in rows:
        slack = operator - np.outer(row, row.conjugate())
        slack = (slack + slack.conjugate().T) / 2.0
        minimum = min(minimum, float(np.linalg.eigvalsh(slack)[0]))
    return minimum


__all__ = [
    "extend_certificate",
    "extend_with_budget",
    "global_hull_certificate",
    "left_suffix",
    "minimum_certificate_slack",
    "optimized_extension",
    "physical_hull_rows",
    "right_suffix",
    "split_left_suffix",
    "split_right_suffix",
    "tensor_cell_certificate",
    "weighted_local_certificate",
]
