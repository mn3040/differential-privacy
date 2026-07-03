import json
import shutil
import subprocess
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path

from dp_demo.dataset import load_dataset
from dp_demo.queries import run_query


ROOT = Path(__file__).resolve().parent.parent


def _run_js_query(dataset_key: str, params: dict[str, object]) -> dict[str, object]:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("Node.js is required for Python/JS parity tests")

    script = textwrap.dedent(
        """
        const fs = require("fs");
        const path = require("path");
        const vm = require("vm");

        const sandbox = { module: { exports: {} }, console };
        const appSource = fs.readFileSync(path.join("docs", "app.js"), "utf8");
        vm.runInNewContext(appSource, sandbox, { filename: "docs/app.js" });
        const { buildDatasetFromCsv, runQuery } = sandbox.module.exports;

        const datasetKey = process.argv[1];
        const params = JSON.parse(process.argv[2]);
        const file = datasetKey === "pums" ? "pums_california_1000.csv" : "iris_sample.csv";
        const csv = fs.readFileSync(path.join("docs", "data", file), "utf8");
        const dataset = buildDatasetFromCsv(datasetKey, csv);
        process.stdout.write(JSON.stringify(runQuery(dataset, params)));
        """
    )
    completed = subprocess.run(
        [node, "-e", script, dataset_key, json.dumps(params)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class JavaScriptParityTests(unittest.TestCase):
    def assert_numeric_query_matches_js(self, dataset_key: str, params: dict[str, object]) -> None:
        py = asdict(run_query(load_dataset(dataset_key), **params))
        js = _run_js_query(dataset_key, params)

        self.assertEqual(js["query"], py["query"])
        self.assertEqual(js["mechanism"], py["mechanism"])
        self.assertEqual(js["column"], py["column"])
        self.assertEqual(js["category"], py["category"])
        self.assertEqual(js["rowsUsed"], py["rows_used"])
        self.assertAlmostEqual(js["trueValue"], py["true_value"], places=9)
        self.assertAlmostEqual(js["noisyValue"], py["noisy_value"], places=9)
        self.assertAlmostEqual(js["sensitivity"], py["sensitivity"], places=9)

    def test_laplace_mean_matches_js(self) -> None:
        self.assert_numeric_query_matches_js(
            "iris",
            {
                "query": "mean",
                "mechanism": "laplace",
                "epsilon": 1.25,
                "delta": 0.0,
                "column": "petal_length",
                "category": "setosa",
                "seed": 42,
            },
        )

    def test_gaussian_sum_matches_js(self) -> None:
        self.assert_numeric_query_matches_js(
            "iris",
            {
                "query": "sum",
                "mechanism": "gaussian",
                "epsilon": 4.0,
                "delta": 0.5,
                "column": "petal_width",
                "category": "versicolor",
                "seed": 123,
            },
        )

    def test_exponential_mode_matches_js(self) -> None:
        params = {
            "query": "mode_category",
            "mechanism": "exponential",
            "epsilon": 2.0,
            "delta": 0.0,
            "column": "income",
            "category": None,
            "seed": 7,
        }
        py = asdict(run_query(load_dataset("pums"), **params))
        js = _run_js_query("pums", params)

        self.assertEqual(js["trueLabel"], py["true_label"])
        self.assertEqual(js["releasedLabel"], py["released_label"])
        self.assertEqual(js["rowsUsed"], py["rows_used"])
        self.assertEqual(set(js["probabilities"]), set(py["probabilities"]))
        for label, probability in py["probabilities"].items():
            self.assertAlmostEqual(js["probabilities"][label], probability, places=9)


if __name__ == "__main__":
    unittest.main()
