import unittest

from dp_demo.dataset import load_iris
from dp_demo.queries import run_query


class QueryTests(unittest.TestCase):
    def test_count_query_uses_unit_sensitivity(self) -> None:
        result = run_query(load_iris(), "count", "laplace", epsilon=1.0, seed=3)
        self.assertEqual(result.true_value, 30.0)
        self.assertEqual(result.sensitivity, 1.0)

    def test_species_mean_uses_filtered_n(self) -> None:
        result = run_query(
            load_iris(),
            "mean",
            "laplace",
            epsilon=1.0,
            column="petal_length",
            species="setosa",
            seed=3,
        )
        self.assertEqual(result.rows_used, 10)
        self.assertAlmostEqual(result.true_value, 1.45)
        self.assertAlmostEqual(result.sensitivity, 0.6)

    def test_laplace_rejects_nonzero_delta(self) -> None:
        with self.assertRaises(ValueError):
            run_query(load_iris(), "count", "laplace", epsilon=1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
