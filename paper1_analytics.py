"""Analytic and finite-size checks used by Paper I.

The functions in this module cover only the one-interval models retained in
Paper I.  They are deliberately independent of the fixed-m forest code used
by Paper II.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipe


def fixed_length_gram(candidate_count: int, length: int, r: float) -> np.ndarray:
    """Return G[a,b] = r**min(|a-b|, length)."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must lie in [0,1]")
    indices = np.arange(candidate_count)
    distances = np.abs(indices[:, None] - indices[None, :])
    return np.power(float(r), np.minimum(distances, length), dtype=float)


def single_change_gram(candidate_count: int, r: float) -> np.ndarray:
    """Return the Toeplitz Gram Q[a,b] = r**|a-b|."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must lie in [0,1]")
    indices = np.arange(candidate_count)
    return np.power(float(r), np.abs(indices[:, None] - indices[None, :]))


def fixed_length_symbol(theta: float | np.ndarray, length: int, r: float):
    """Evaluate f_{i,r}(theta) from Theorem 3.1."""
    if length < 1:
        raise ValueError("length must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must lie in [0,1]")
    theta_array = np.asarray(theta)
    value = np.full_like(theta_array, 1.0 - r**length, dtype=float)
    for distance in range(1, length):
        value += 2.0 * (r**distance - r**length) * np.cos(
            distance * theta_array
        )
    return float(value) if value.ndim == 0 else value


def fixed_length_limit_quadrature(length: int, r: float) -> float:
    """Numerically evaluate the fixed-length Toeplitz integral."""
    if r <= 0.0:
        return 1.0
    if r >= 1.0:
        return 0.0
    integral, _ = quad(
        lambda theta: math.sqrt(max(0.0, fixed_length_symbol(theta, length, r))),
        0.0,
        2.0 * math.pi,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )
    return float((integral / (2.0 * math.pi)) ** 2)


def i1_finite_optimum(candidate_count: int, r: float) -> float:
    """Finite-size optimum for a known one-site anomalous interval."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must lie in [0,1]")
    numerator = math.sqrt(1.0 + (candidate_count - 1) * r)
    numerator += (candidate_count - 1) * math.sqrt(1.0 - r)
    return float((numerator / candidate_count) ** 2)


def i2_elliptic_limit(r: float) -> float:
    """Closed form for the two-site fixed-length limit."""
    if not 0.0 <= r <= 1.0:
        raise ValueError("r must lie in [0,1]")
    if r >= 1.0:
        return 0.0
    parameter = 4.0 * r / (1.0 + 3.0 * r)
    return float(
        4.0
        * (1.0 - r)
        * (1.0 + 3.0 * r)
        * ellipe(parameter) ** 2
        / math.pi**2
    )


def fixed_length_circulant_gram(
    candidate_count: int, length: int, r: float
) -> np.ndarray:
    """Return the legal circulant comparison Gram from Theorem 3.1.

    This constructor is intended for candidate_count >= 2*length, the regime
    in which the local diagonals do not collide around the cycle.
    """
    if candidate_count < 2 * length:
        raise ValueError("require candidate_count >= 2*length")
    indices = np.arange(candidate_count)
    ordinary = np.abs(indices[:, None] - indices[None, :])
    cyclic = np.minimum(ordinary, candidate_count - ordinary)
    residual = np.zeros_like(cyclic, dtype=float)
    for distance in range(length):
        residual[cyclic == distance] = r**distance - r**length
    return r**length * np.ones_like(residual) + residual

