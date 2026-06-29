"""Dataset loading and clipping domains for the demo."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris_sample.csv"

NUMERIC_COLUMNS = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.0, 2.5),
}

SPECIES = ("setosa", "versicolor", "virginica")


@dataclass(frozen=True)
class IrisRow:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

    def value(self, column: str) -> float:
        if column not in NUMERIC_COLUMNS:
            raise ValueError(f"unsupported numeric column: {column}")
        return float(getattr(self, column))


def load_iris(path: Path = DATA_PATH) -> list[IrisRow]:
    """Load a compact public-domain excerpt of Fisher's Iris dataset."""

    rows: list[IrisRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(
                IrisRow(
                    sepal_length=float(raw["sepal_length"]),
                    sepal_width=float(raw["sepal_width"]),
                    petal_length=float(raw["petal_length"]),
                    petal_width=float(raw["petal_width"]),
                    species=raw["species"],
                )
            )
    return rows


def clip(value: float, column: str) -> float:
    lower, upper = NUMERIC_COLUMNS[column]
    return min(max(value, lower), upper)
