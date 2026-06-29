"""Privacy budget composition and sensitivity-comparison helpers.

Running several DP queries against the same dataset spends privacy budget
each time. These functions answer "what's my total epsilon (and delta) after
k queries?" two different ways, plus expose the local-vs-global sensitivity
distinction used to justify why composition bounds are conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CompositionResult:
    epsilon: float
    delta: float
    method: str
    explanation: str


def sequential_composition(epsilons: Sequence[float], deltas: Sequence[float] | None = None) -> CompositionResult:
    """Basic composition: privacy costs simply add up.

        epsilon_total = sum(epsilon_i)
        delta_total   = sum(delta_i)

    This is exact and assumption-free, but pessimistic: it does not account
    for the fact that independent noise draws partially cancel out.
    """

    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    deltas = deltas or [0.0] * len(epsilons)
    if len(deltas) != len(epsilons):
        raise ValueError("epsilons and deltas must be the same length")

    total_epsilon = sum(epsilons)
    total_delta = sum(deltas)
    return CompositionResult(
        epsilon=total_epsilon,
        delta=total_delta,
        method="sequential",
        explanation=(
            f"{len(epsilons)} mechanisms composed sequentially: "
            f"epsilon_total = sum(epsilon_i) = {total_epsilon:.6g}, "
            f"delta_total = sum(delta_i) = {total_delta:.6g}."
        ),
    )


def advanced_composition(k: int, epsilon: float, delta: float, delta_prime: float) -> CompositionResult:
    """Tighter bound for k repetitions of the *same* (epsilon, delta)-DP mechanism.

        epsilon_total = sqrt(2 k ln(1/delta')) * epsilon + k * epsilon^2

    This is the standard (Dwork-Rothblum-Vadhan style) advanced composition
    bound: it grows roughly with sqrt(k) instead of k, at the cost of an
    extra failure probability delta' and only applying when every query
    shares the same (epsilon, delta).
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")
    if not 0 < delta_prime < 1:
        raise ValueError("delta_prime must satisfy 0 < delta_prime < 1")

    first_term = math.sqrt(2 * k * math.log(1.0 / delta_prime)) * epsilon
    second_term = k * epsilon**2
    total_epsilon = first_term + second_term
    total_delta = k * delta + delta_prime

    return CompositionResult(
        epsilon=total_epsilon,
        delta=total_delta,
        method="advanced",
        explanation=(
            f"{k} repeats of ({epsilon:.4g}, {delta:.4g})-DP composed with advanced "
            f"composition (failure slack delta'={delta_prime:.4g}): "
            f"epsilon_total = sqrt(2*{k}*ln(1/{delta_prime:.4g}))*{epsilon:.4g} "
            f"+ {k}*{epsilon:.4g}^2 = {first_term:.6g} + {second_term:.6g} "
            f"= {total_epsilon:.6g}, delta_total = k*delta + delta' = {total_delta:.6g}."
        ),
    )


def global_sensitivity_mean(lower: float, upper: float, n: int) -> float:
    """Worst-case replacement-neighbor sensitivity of a clipped mean over any dataset of size n."""

    if n <= 0:
        raise ValueError("n must be positive")
    return (upper - lower) / n


def local_sensitivity_mean(values: Sequence[float], lower: float, upper: float) -> float:
    """Replacement-neighbor sensitivity of the mean for *this specific* dataset.

    Global sensitivity assumes the swapped-in value could be anywhere in
    [lower, upper]. Local sensitivity instead uses the actual observed range
    of `values`, which is never larger and is often much smaller -- the
    catch is that publishing it (without extra care) can itself leak
    information about the dataset, which is why most DP libraries default
    to global sensitivity for the noise calibration.
    """

    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    observed_lower, observed_upper = min(values), max(values)
    worst_case_neighbor_value = max(upper - observed_lower, observed_upper - lower)
    return worst_case_neighbor_value / n
