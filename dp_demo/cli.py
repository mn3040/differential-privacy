"""Command-line query interface for the DP demo."""

from __future__ import annotations

import argparse
import sys

from .composition import advanced_composition, sequential_composition
from .dataset import DATASET_KEYS, load_dataset
from .queries import MECHANISMS, QUERY_TYPES, run_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private queries over a bundled dataset.")
    subparsers = parser.add_subparsers(dest="command")

    query_parser = subparsers.add_parser("query", help="Run a single private query (default).")
    query_parser.add_argument("--dataset", choices=DATASET_KEYS, default="iris")
    query_parser.add_argument("--query", choices=QUERY_TYPES, default="mean")
    query_parser.add_argument("--mechanism", choices=MECHANISMS, default="laplace")
    query_parser.add_argument("--epsilon", type=float, default=1.0)
    query_parser.add_argument("--delta", type=float, default=0.0)
    query_parser.add_argument("--column", default="petal_length", help="Numeric column (varies by dataset).")
    query_parser.add_argument("--category", default=None, help="Optional category filter (varies by dataset).")
    query_parser.add_argument("--seed", type=int, default=None, help="Optional seed for repeatable demos.")

    compose_parser = subparsers.add_parser(
        "compose", help="Compute the total privacy budget spent across repeated queries."
    )
    compose_parser.add_argument("--method", choices=("sequential", "advanced"), default="sequential")
    compose_parser.add_argument("--k", type=int, default=1, help="Number of repeated queries.")
    compose_parser.add_argument("--epsilon", type=float, default=0.1, help="Per-query epsilon.")
    compose_parser.add_argument("--delta", type=float, default=0.0, help="Per-query delta.")
    compose_parser.add_argument(
        "--delta-prime", type=float, default=1e-5, help="Extra failure slack used only by --method advanced."
    )

    return parser


def _print_query(args: argparse.Namespace) -> None:
    kwargs = vars(args)
    kwargs.pop("command", None)
    dataset = load_dataset(kwargs.pop("dataset"))
    result = run_query(dataset, **kwargs)

    print(f"Dataset:     {dataset.label}")
    print(f"Query:       {result.query}")
    print(f"Mechanism:   {result.mechanism}")
    print(f"Rows used:   {result.rows_used}")
    if result.column:
        print(f"Column:      {result.column}")
    if result.category:
        print(f"Category:    {result.category}")
    print(f"Epsilon:     {result.epsilon}")
    print(f"Delta:       {result.delta}")
    print(f"Sensitivity: {result.sensitivity:.6g}")
    if result.released_label is not None:
        print(f"True mode:       {result.true_label}")
        print(f"Released mode:   {result.released_label}")
        print(f"Probabilities:   {result.probabilities}")
    else:
        print(f"True value:  {result.true_value:.6g}")
        print(f"Noisy value: {result.noisy_value:.6g}")
    print(f"Why:         {result.explanation}")


def _print_compose(args: argparse.Namespace) -> None:
    if args.method == "sequential":
        result = sequential_composition([args.epsilon] * args.k, [args.delta] * args.k)
    else:
        result = advanced_composition(args.k, args.epsilon, args.delta or 1e-6, args.delta_prime)

    print(f"Method:          {result.method}")
    print(f"k:               {args.k}")
    print(f"Epsilon total:   {result.epsilon:.6g}")
    print(f"Delta total:     {result.delta:.6g}")
    print(f"Why:             {result.explanation}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("query", "compose", "-h", "--help"):
        argv = ["query", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compose":
        _print_compose(args)
    else:
        _print_query(args)


if __name__ == "__main__":
    main()
