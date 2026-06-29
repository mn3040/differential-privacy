import unittest

from dp_demo.dataset import load_dataset
from dp_demo.queries import run_query


class QueryTests(unittest.TestCase):
    def test_count_query_uses_unit_sensitivity(self) -> None:
        result = run_query(load_dataset("iris"), "count", "laplace", epsilon=1.0, seed=3)
        self.assertEqual(result.true_value, 30.0)
        self.assertEqual(result.sensitivity, 1.0)

    def test_category_mean_uses_filtered_n(self) -> None:
        result = run_query(
            load_dataset("iris"),
            "mean",
            "laplace",
            epsilon=1.0,
            column="petal_length",
            category="setosa",
            seed=3,
        )
        self.assertEqual(result.rows_used, 10)
        self.assertAlmostEqual(result.true_value, 1.45)
        self.assertAlmostEqual(result.sensitivity, 0.6)

    def test_laplace_rejects_nonzero_delta(self) -> None:
        with self.assertRaises(ValueError):
            run_query(load_dataset("iris"), "count", "laplace", epsilon=1.0, delta=1e-6)

    def test_mode_category_uses_exponential_mechanism(self) -> None:
        result = run_query(load_dataset("iris"), "mode_category", "exponential", epsilon=2.0, seed=5)
        self.assertEqual(result.true_label, "setosa")
        self.assertEqual(result.sensitivity, 1.0)
        self.assertIn(result.released_label, ("setosa", "versicolor", "virginica"))
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_mode_category_requires_exponential_mechanism(self) -> None:
        with self.assertRaises(ValueError):
            run_query(load_dataset("iris"), "mode_category", "laplace", epsilon=1.0)

    def test_exponential_mechanism_only_applies_to_mode_category(self) -> None:
        with self.assertRaises(ValueError):
            run_query(load_dataset("iris"), "count", "exponential", epsilon=1.0)

    def test_pums_dataset_mean_income(self) -> None:
        result = run_query(load_dataset("pums"), "mean", "laplace", epsilon=1.0, column="income", seed=1)
        self.assertEqual(result.rows_used, 1000)
        self.assertGreater(result.true_value, 0)

    def test_pums_mode_category_is_married_or_unmarried(self) -> None:
        result = run_query(load_dataset("pums"), "mode_category", "exponential", epsilon=2.0, seed=1)
        self.assertIn(result.true_label, ("married", "unmarried"))


if __name__ == "__main__":
    unittest.main()
