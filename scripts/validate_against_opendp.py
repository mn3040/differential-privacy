"""Compare this demo's mechanisms against OpenDP measurements.

This script is intentionally optional: the project itself has no runtime
dependency on OpenDP. Install OpenDP in your environment, then run:

    python scripts/validate_against_opendp.py

The comparison uses the same true query value and the same calibrated noise
scale for both implementations, then checks empirical output moments against
the theoretical distribution moments.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass

from dp_demo.dataset import load_dataset
from dp_demo.mechanisms import (
    PrivacyBudget,
    gaussian_sigma,
    laplace_scale,
    release_gaussian,
    release_laplace,
)
from dp_demo.queries import run_query


@dataclass(frozen=True)
class MechanismValidation:
    mechanism: str
    samples: int
    true_value: float
    sensitivity: float
    scale: float
    expected_mean: float
    expected_std: float
    demo_mean: float
    demo_std: float
    opendp_mean: float
    opendp_std: float
    mean_tolerance: float
    std_tolerance: float
    passed: bool


def _require_opendp():
    try:
        import opendp.prelude as dp
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenDP is not installed. Run `python -m pip install opendp` in this environment "
            "to enable trusted-reference validation."
        ) from exc

    dp.enable_features("contrib")
    return dp


def _stats(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def validate_laplace(samples: int, seed: int) -> MechanismValidation:
    dp = _require_opendp()
    dataset = load_dataset("iris")
    query = run_query(
        dataset,
        query="sum",
        mechanism="laplace",
        epsilon=2.15,
        delta=0.0,
        column="petal_width",
        category="versicolor",
        seed=seed,
    )
    scale = laplace_scale(query.sensitivity, query.epsilon)

    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    opendp_laplace = dp.m.make_laplace(*space, scale=scale)

    rng = random.Random(seed)
    demo_values = [
        release_laplace(query.true_value, query.sensitivity, PrivacyBudget(query.epsilon), rng)
        for _ in range(samples)
    ]
    opendp_values = [float(opendp_laplace(query.true_value)) for _ in range(samples)]

    expected_std = math.sqrt(2.0) * scale
    return _build_result("laplace", samples, query.true_value, query.sensitivity, scale, expected_std, demo_values, opendp_values)


def validate_gaussian(samples: int, seed: int) -> MechanismValidation:
    dp = _require_opendp()
    dataset = load_dataset("iris")
    query = run_query(
        dataset,
        query="sum",
        mechanism="gaussian",
        epsilon=4.0,
        delta=0.5,
        column="petal_width",
        category="versicolor",
        seed=seed,
    )
    sigma = gaussian_sigma(query.sensitivity, query.epsilon, query.delta)

    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    opendp_gaussian = dp.m.make_gaussian(*space, scale=sigma)

    rng = random.Random(seed)
    budget = PrivacyBudget(query.epsilon, query.delta)
    demo_values = [
        release_gaussian(query.true_value, query.sensitivity, budget, rng)
        for _ in range(samples)
    ]
    opendp_values = [float(opendp_gaussian(query.true_value)) for _ in range(samples)]

    return _build_result("gaussian", samples, query.true_value, query.sensitivity, sigma, sigma, demo_values, opendp_values)


def _build_result(
    mechanism: str,
    samples: int,
    true_value: float,
    sensitivity: float,
    scale: float,
    expected_std: float,
    demo_values: list[float],
    opendp_values: list[float],
) -> MechanismValidation:
    demo_mean, demo_std = _stats(demo_values)
    opendp_mean, opendp_std = _stats(opendp_values)
    # Loose enough to avoid false negatives from sampling noise, tight enough
    # to catch an obvious scale/calibration mismatch.
    mean_tolerance = expected_std * 0.12
    std_tolerance = expected_std * 0.18
    passed = (
        abs(demo_mean - true_value) <= mean_tolerance
        and abs(opendp_mean - true_value) <= mean_tolerance
        and abs(demo_std - expected_std) <= std_tolerance
        and abs(opendp_std - expected_std) <= std_tolerance
    )
    return MechanismValidation(
        mechanism=mechanism,
        samples=samples,
        true_value=true_value,
        sensitivity=sensitivity,
        scale=scale,
        expected_mean=true_value,
        expected_std=expected_std,
        demo_mean=demo_mean,
        demo_std=demo_std,
        opendp_mean=opendp_mean,
        opendp_std=opendp_std,
        mean_tolerance=mean_tolerance,
        std_tolerance=std_tolerance,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate demo mechanisms against OpenDP measurements.")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.samples < 1000:
        raise SystemExit("--samples should be at least 1000 for stable empirical moments")

    results = [validate_laplace(args.samples, args.seed), validate_gaussian(args.samples, args.seed)]
    print(json.dumps([asdict(result) for result in results], indent=2))
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
