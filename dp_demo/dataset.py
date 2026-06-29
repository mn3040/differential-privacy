"""Dataset loading, clipping domains, and the multi-dataset registry."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Dataset:
    """A loaded CSV plus the public metadata needed to run private queries on it.

    `numeric_columns` and `categories` are *public* facts about the column
    domain (e.g. "age is between 18 and 95"), not derived from the loaded
    rows. Using a public bound instead of the data's actual min/max is what
    makes the resulting sensitivity a global sensitivity (see README).
    """

    key: str
    label: str
    source: str
    rows: tuple[dict[str, float | str], ...]
    numeric_columns: dict[str, tuple[float, float]]
    category_column: str
    categories: tuple[str, ...]

    def value(self, row: dict[str, float | str], column: str) -> float:
        if column not in self.numeric_columns:
            raise ValueError(f"unsupported numeric column: {column}")
        return float(row[column])

    def clip(self, value: float, column: str) -> float:
        lower, upper = self.numeric_columns[column]
        return min(max(value, lower), upper)

    def category(self, row: dict[str, float | str]) -> str:
        return str(row[self.category_column])


def _load_csv(
    path: Path,
    numeric_fields: tuple[str, ...],
    category_field: str,
    category_map: dict[str, str] | None = None,
) -> tuple[dict[str, float | str], ...]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float | str] = {field: float(raw[field]) for field in numeric_fields}
            label = raw[category_field]
            row[category_field] = category_map.get(label, label) if category_map else label
            rows.append(row)
    return tuple(rows)


def _load_iris() -> Dataset:
    numeric_columns = {
        "sepal_length": (4.0, 8.0),
        "sepal_width": (2.0, 4.5),
        "petal_length": (1.0, 7.0),
        "petal_width": (0.0, 2.5),
    }
    rows = _load_csv(DATA_DIR / "iris_sample.csv", tuple(numeric_columns), "species")
    return Dataset(
        key="iris",
        label="Iris flowers (Fisher, 1936)",
        source="Bundled public-domain excerpt of Fisher's Iris dataset.",
        rows=rows,
        numeric_columns=numeric_columns,
        category_column="species",
        categories=("setosa", "versicolor", "virginica"),
    )


def _load_pums() -> Dataset:
    numeric_columns = {
        "age": (18.0, 95.0),
        "educ": (0.0, 16.0),
        "income": (0.0, 420500.0),
    }
    rows = _load_csv(
        DATA_DIR / "pums_california_1000.csv",
        tuple(numeric_columns),
        "married",
        category_map={"0": "unmarried", "1": "married"},
    )
    return Dataset(
        key="pums",
        label="California PUMS demographics (sample of 1000)",
        source="OpenDP test datasets: opendp/dp-test-datasets, data/PUMS_california_demographics_1000.",
        rows=rows,
        numeric_columns=numeric_columns,
        category_column="married",
        categories=("married", "unmarried"),
    )


_LOADERS = {
    "iris": _load_iris,
    "pums": _load_pums,
}

DATASET_KEYS = tuple(_LOADERS)

_CACHE: dict[str, Dataset] = {}


def load_dataset(key: str = "iris") -> Dataset:
    """Load (and cache) one of the bundled datasets by key."""

    if key not in _LOADERS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASET_KEYS)}")
    if key not in _CACHE:
        _CACHE[key] = _LOADERS[key]()
    return _CACHE[key]
