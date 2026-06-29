"""Command-line query interface for the DP demo."""

from __future__ import annotations

import argparse

from .dataset import NUMERIC_COLUMNS, SPECIES, load_iris
from .queries import MECHANISMS, QUERY_TYPES, run_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private queries over the Iris dataset.")
    parser.add_argument("--query", choices=QUERY_TYPES, default="mean")
    parser.add_argument("--mechanism", choices=MECHANISMS, default="laplace")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--column", choices=tuple(NUMERIC_COLUMNS), default="petal_length")
    parser.add_argument("--species", choices=SPECIES, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for repeatable demos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_query(load_iris(), **vars(args))

    print(f"Query:       {result.query}")
    print(f"Mechanism:   {result.mechanism}")
    print(f"Rows used:   {result.rows_used}")
    if result.column:
        print(f"Column:      {result.column}")
    if result.species:
        print(f"Species:     {result.species}")
    print(f"Epsilon:     {result.epsilon}")
    print(f"Delta:       {result.delta}")
    print(f"Sensitivity: {result.sensitivity:.6g}")
    print(f"True value:  {result.true_value:.6g}")
    print(f"Noisy value: {result.noisy_value:.6g}")
    print(f"Why:         {result.explanation}")


if __name__ == "__main__":
    main()
