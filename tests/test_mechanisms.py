import math
import random
import unittest

from dp_demo.mechanisms import (
    PrivacyBudget,
    exponential_weights,
    gaussian_sigma,
    laplace_scale,
    release_exponential,
    release_gaussian,
    release_laplace,
)


class MechanismTests(unittest.TestCase):
    def test_laplace_scale_is_sensitivity_over_epsilon(self) -> None:
        self.assertEqual(laplace_scale(2.0, 0.5), 4.0)

    def test_gaussian_sigma_uses_classic_calibration(self) -> None:
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) * 1.0 / 1.0
        self.assertAlmostEqual(gaussian_sigma(1.0, 1.0, 1e-5), expected)

    def test_seeded_releases_are_repeatable(self) -> None:
        budget = PrivacyBudget(epsilon=1.0)
        a = release_laplace(10.0, 1.0, budget, random.Random(7))
        b = release_laplace(10.0, 1.0, budget, random.Random(7))
        self.assertEqual(a, b)

    def test_gaussian_requires_delta(self) -> None:
        with self.assertRaises(ValueError):
            release_gaussian(1.0, 1.0, PrivacyBudget(epsilon=1.0, delta=0.0))

    def test_exponential_weights_favor_higher_utility(self) -> None:
        weights = exponential_weights([10.0, 0.0], sensitivity=1.0, epsilon=1.0)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[0], weights[1])

    def test_exponential_weights_uniform_for_equal_utility(self) -> None:
        weights = exponential_weights([5.0, 5.0, 5.0], sensitivity=1.0, epsilon=1.0)
        for w in weights:
            self.assertAlmostEqual(w, 1.0 / 3.0)

    def test_exponential_rejects_nonzero_delta(self) -> None:
        with self.assertRaises(ValueError):
            release_exponential(["a", "b"], [1.0, 2.0], 1.0, PrivacyBudget(epsilon=1.0, delta=1e-6))

    def test_exponential_is_seeded_repeatable(self) -> None:
        budget = PrivacyBudget(epsilon=1.0)
        a, _ = release_exponential(["a", "b", "c"], [1.0, 5.0, 2.0], 1.0, budget, random.Random(7))
        b, _ = release_exponential(["a", "b", "c"], [1.0, 5.0, 2.0], 1.0, budget, random.Random(7))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
