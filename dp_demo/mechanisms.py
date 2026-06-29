"""Laplace, Gaussian, and Exponential mechanisms implemented from scratch.

The formulas mirror the standard textbook mechanisms used by libraries such as
Google Differential Privacy and OpenDP, but intentionally avoid depending on
either library. This file is small on purpose: it is meant to be read.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


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

    def validate_for_exponential(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta != 0:
            raise ValueError("Exponential mechanism is pure DP, so delta must be 0")


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


def exponential_weights(
    utilities: Sequence[float],
    sensitivity: float,
    epsilon: float,
) -> list[float]:
    """Return selection probabilities for the exponential mechanism.

        Pr[output = r] proportional to exp(epsilon * utility(r) / (2 * sensitivity))

    The `/ 2` in the exponent (rather than `/ 1` as in Laplace) accounts for the
    fact that a single changed row can move the utility of *one* candidate up
    while moving another candidate's utility down, so the score can swing by
    twice the sensitivity between neighboring datasets.
    """

    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not utilities:
        raise ValueError("utilities must be non-empty")

    scores = [epsilon * u / (2.0 * sensitivity) for u in utilities]
    # Subtract the max score before exponentiating (log-sum-exp trick) so the
    # exponential mechanism stays numerically stable for large epsilon/utility.
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def release_exponential(
    candidates: Sequence[T],
    utilities: Sequence[float],
    sensitivity: float,
    budget: PrivacyBudget,
    rng: random.Random | None = None,
) -> tuple[T, list[float]]:
    """Privately select a candidate using the exponential mechanism.

    Unlike Laplace/Gaussian, this perturbs the *selection probability* rather
    than adding noise to a number, so it works for categorical or otherwise
    non-numeric outputs (e.g. "which species is most common?").
    """

    budget.validate_for_exponential()
    if len(candidates) != len(utilities):
        raise ValueError("candidates and utilities must be the same length")

    probabilities = exponential_weights(utilities, sensitivity, budget.epsilon)
    rng = rng or random
    selected = rng.choices(candidates, weights=probabilities, k=1)[0]
    return selected, probabilities
