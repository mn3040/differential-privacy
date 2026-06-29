import math
import random
import unittest

from dp_demo.mechanisms import (
    PrivacyBudget,
    gaussian_sigma,
    laplace_scale,
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


if __name__ == "__main__":
    unittest.main()
