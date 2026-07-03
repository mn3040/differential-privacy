import unittest

from dp_demo.composition import (
    advanced_composition,
    advanced_savings_sweep,
    global_sensitivity_mean,
    local_sensitivity_mean,
    sequential_composition,
)


class CompositionTests(unittest.TestCase):
    def test_sequential_composition_sums_budgets(self) -> None:
        result = sequential_composition([0.5, 0.3, 0.2])
        self.assertAlmostEqual(result.epsilon, 1.0)
        self.assertAlmostEqual(result.delta, 0.0)

    def test_advanced_composition_matches_worked_example(self) -> None:
        # 100 queries at (epsilon=0.1, delta=1e-5), delta'=1e-5 -> epsilon_total ~= 5.8
        result = advanced_composition(k=100, epsilon=0.1, delta=1e-5, delta_prime=1e-5)
        self.assertAlmostEqual(result.epsilon, 5.8, places=2)

    def test_advanced_composition_is_tighter_than_sequential_for_large_k(self) -> None:
        k, epsilon = 100, 0.1
        sequential = sequential_composition([epsilon] * k)
        advanced = advanced_composition(k=k, epsilon=epsilon, delta=1e-5, delta_prime=1e-5)
        self.assertLess(advanced.epsilon, sequential.epsilon)

    def test_advanced_savings_sweep_shows_crossover(self) -> None:
        rows = advanced_savings_sweep([10, 100], [0.1], delta=1e-5, delta_prime=1e-5)
        small_k, large_k = rows
        self.assertLess(small_k.savings_percent, 0.0)
        self.assertAlmostEqual(large_k.savings_percent, 42.0, places=1)
        self.assertGreater(large_k.savings_percent, small_k.savings_percent)

    def test_local_sensitivity_never_exceeds_global(self) -> None:
        lower, upper, n = 1.0, 7.0, 10
        values = [2.0, 2.1, 1.9, 2.2, 2.0, 2.1, 1.8, 2.3, 2.0, 2.1]
        self.assertEqual(len(values), n)
        self.assertLessEqual(
            local_sensitivity_mean(values, lower, upper),
            global_sensitivity_mean(lower, upper, n),
        )


if __name__ == "__main__":
    unittest.main()
