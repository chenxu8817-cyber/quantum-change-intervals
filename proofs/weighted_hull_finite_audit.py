"""Exact finite-dimensional audits for the weighted hull identity.

Python indices are zero-based closed pairs.  Manuscript indices are their
one-based counterparts.  Hull columns are always strict pairs ``u < v``.
"""

from cmath import exp as complex_exp
from fractions import Fraction
from math import comb, sqrt


def interval_labels(n: int) -> list[tuple[int, int]]:
    """Return all zero-based closed interval labels in lexicographic order."""
    if n < 1:
        raise ValueError("n must be positive")
    return [(a, b) for a in range(n) for b in range(a, n)]


def hull_pairs(n: int) -> list[tuple[int, int]]:
    """Return all zero-based strict hull pairs in lexicographic order."""
    if n < 1:
        raise ValueError("n must be positive")
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def weighted_hull_matrix_fraction(
    n: int, c: Fraction
) -> list[list[Fraction]]:
    """Return the exact physical hull coordinate matrix."""
    if not Fraction(0) <= c <= Fraction(1):
        raise ValueError("c must lie in [0,1]")
    s2 = Fraction(1) - c * c
    return [
        [
            s2 * c ** ((u - a) + (b - v))
            if a <= u < v <= b
            else Fraction(0)
            for (u, v) in hull_pairs(n)
        ]
        for (a, b) in interval_labels(n)
    ]


def direct_hull_coordinate_matrix_fraction(
    n: int, c: Fraction, s: Fraction
) -> list[list[Fraction]]:
    """Sum excitation subsets grouped by their first and last sites."""
    return [
        [
            sum(
                Fraction(comb(v - u - 1, k))
                * s ** (k + 2)
                * c ** (b - a + 1 - k - 2)
                * s**k
                * c ** (v - u - 1 - k)
                for k in range(v - u)
            )
            if a <= u < v <= b
            else Fraction(0)
            for (u, v) in hull_pairs(n)
        ]
        for (a, b) in interval_labels(n)
    ]


def tensor_compressed_hull_matrix_fraction(
    n: int, c: Fraction
) -> list[list[Fraction]]:
    """Build the Volterra factors, then compress their tensor product."""
    volterra = [
        [
            c ** (u - a) if a <= u else Fraction(0)
            for u in range(n)
        ]
        for a in range(n)
    ]
    return [
        [
            (Fraction(1) - c * c) * volterra[a][u] * volterra[v][b]
            if a <= u < v <= b
            else Fraction(0)
            for (u, v) in hull_pairs(n)
        ]
        for (a, b) in interval_labels(n)
    ]


def old_qdzw_hull_matrix_fraction(
    n: int, c: Fraction, s: Fraction
) -> list[list[Fraction]]:
    """Evaluate the earlier q-squared DZW representation entry by entry."""
    if not Fraction(0) < c < Fraction(1):
        raise ValueError("old q^2DZW form requires 0 < c < 1")
    q_squared = (s / c) ** 2
    return [
        [
            q_squared * c ** (b - a + 1) * c ** (-(v - u - 1))
            if a <= u < v <= b
            else Fraction(0)
            for (u, v) in hull_pairs(n)
        ]
        for (a, b) in interval_labels(n)
    ]


def weighted_hull_gram_fraction(
    n: int, c: Fraction
) -> list[list[Fraction]]:
    """Form the exact hull Gram matrix as the coordinate product A A^T."""
    coordinates = weighted_hull_matrix_fraction(n, c)
    return [
        [
            sum(
                (left * right for left, right in zip(row_i, row_j)),
                Fraction(0),
            )
            for row_j in coordinates
        ]
        for row_i in coordinates
    ]


def direct_ge_two_gram_fraction(
    n: int, c: Fraction
) -> list[list[Fraction]]:
    """Sum all common excitation subsets of cardinality at least two."""
    labels = interval_labels(n)
    s_squared = Fraction(1) - c * c
    gram: list[list[Fraction]] = []
    for a, b in labels:
        row: list[Fraction] = []
        length_i = b - a + 1
        for d, e in labels:
            length_j = e - d + 1
            overlap = max(0, min(b, e) - max(a, d) + 1)
            row.append(
                sum(
                    (
                        Fraction(comb(overlap, k))
                        * s_squared**k
                        * c ** (length_i + length_j - 2 * k)
                        for k in range(2, overlap + 1)
                    ),
                    Fraction(0),
                )
            )
        gram.append(row)
    return gram


def complex_phase_congruence_check(n: int, c: Fraction, phi: float) -> bool:
    """Check G' = D* G D using explicit tensor-product interval states."""
    if not Fraction(0) <= c <= Fraction(1):
        raise ValueError("c must lie in [0,1]")
    labels = interval_labels(n)
    c_float = float(c)
    s_float = sqrt(1.0 - c_float * c_float)
    phase = complex_exp(1j * phi)
    inverse_phase = complex_exp(-1j * phi)
    zero_state = (1.0 + 0.0j, 0.0 + 0.0j)
    anomaly_state = (c_float * phase, s_float + 0.0j)
    gauged_anomaly_state = tuple(
        inverse_phase * amplitude for amplitude in anomaly_state
    )

    def interval_state(
        interval: tuple[int, int], anomaly: tuple[complex, complex]
    ) -> list[complex]:
        a, b = interval
        state = [1.0 + 0.0j]
        for site in range(n):
            local_state = anomaly if a <= site <= b else zero_state
            state = [
                prefix * local
                for prefix in state
                for local in local_state
            ]
        return state

    def gram(states: list[list[complex]]) -> list[list[complex]]:
        return [
            [
                sum(
                    left_amplitude.conjugate() * right_amplitude
                    for left_amplitude, right_amplitude in zip(left, right)
                )
                for right in states
            ]
            for left in states
        ]

    original_states = [
        interval_state(interval, anomaly_state) for interval in labels
    ]
    gauged_states = [
        interval_state(interval, gauged_anomaly_state) for interval in labels
    ]
    original_gram = gram(original_states)
    gauged_gram = gram(gauged_states)
    diagonal = [
        complex_exp(-1j * (b - a + 1) * phi) for a, b in labels
    ]

    return all(
        abs(
            gauged_gram[i][j]
            - diagonal[i].conjugate()
            * original_gram[i][j]
            * diagonal[j]
        )
        <= 1e-12
        for i in range(len(labels))
        for j in range(len(labels))
    )
