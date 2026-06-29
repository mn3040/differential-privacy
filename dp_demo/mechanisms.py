"""Laplace and Gaussian mechanisms implemented from scratch.

The formulas mirror the standard textbook mechanisms used by libraries such as
Google Differential Privacy and OpenDP, but intentionally avoid depending on
either library. This file is small on purpose: it is meant to be read.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyBudget:
    """Privacy parameters for an approximate-DP query."""

    epsilon: float
    delta: float = 0.0

    def validate_for_laplace(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta != 0:
            raise ValueError("Laplace mechanism is pure DP, so delta must be 0")

    def validate_for_gaussian(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("Gaussian mechanism requires 0 < delta < 1")


def laplace_noise(scale: float, rng: random.Random | None = None) -> float:
    """Sample Laplace(0, scale) noise using inverse transform sampling."""

    if scale < 0:
        raise ValueError("scale must be non-negative")
    if scale == 0:
        return 0.0

    rng = rng or random
    u = rng.random() - 0.5
    return -scale * math.copysign(math.log1p(-2.0 * abs(u)), u)


def gaussian_noise(sigma: float, rng: random.Random | None = None) -> float:
    """Sample Gaussian(0, sigma) noise."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return 0.0

    rng = rng or random
    return rng.gauss(0.0, sigma)


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Return b = sensitivity / epsilon for the Laplace mechanism."""

    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return sensitivity / epsilon


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Return the classic Gaussian sigma for (epsilon, delta)-DP.

    This is the common textbook calibration:
        sigma >= sqrt(2 ln(1.25 / delta)) * sensitivity / epsilon

    It is simple and conservative for epsilon in the usual small-epsilon regime.
    """

    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon


def release_laplace(
    true_value: float,
    sensitivity: float,
    budget: PrivacyBudget,
    rng: random.Random | None = None,
) -> float:
    """Release a numeric query with Laplace noise."""

    budget.validate_for_laplace()
    return true_value + laplace_noise(laplace_scale(sensitivity, budget.epsilon), rng)


def release_gaussian(
    true_value: float,
    sensitivity: float,
    budget: PrivacyBudget,
    rng: random.Random | None = None,
) -> float:
    """Release a numeric query with Gaussian noise."""

    budget.validate_for_gaussian()
    sigma = gaussian_sigma(sensitivity, budget.epsilon, budget.delta)
    return true_value + gaussian_noise(sigma, rng)
