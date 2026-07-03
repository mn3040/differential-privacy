"""Render the advanced composition savings chart with matplotlib.

The checked-in chart is derived from docs/data/composition_savings_sweep.csv.
Matplotlib is intentionally a developer-only dependency:

    python -m pip install matplotlib
    python scripts/render_composition_chart.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "docs" / "data" / "composition_savings_sweep.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "composition_savings.svg"


def read_series(path: Path, epsilon: float) -> tuple[list[int], list[float]]:
    ks: list[int] = []
    savings: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["epsilon_per_query"]) == epsilon:
                ks.append(int(row["k"]))
                savings.append(float(row["savings_percent"]))
    if not ks:
        raise ValueError(f"no rows found for epsilon={epsilon}")
    return ks, savings


def render_chart(input_path: Path, output_path: Path, epsilon: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ks, savings = read_series(input_path, epsilon)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=160)
    fig.patch.set_facecolor("#0b0d14")
    ax.set_facecolor("#0b0d14")

    ax.plot(ks, savings, color="#5fd0c0", marker="o", linewidth=2.5, markersize=6)
    ax.axhline(0, color="#343b55", linewidth=1.2, linestyle="--")
    ax.grid(True, color="#252a3d", linewidth=0.8, alpha=0.85)

    ax.set_title("Advanced composition savings trend", loc="left", color="#d8dce8", pad=14)
    ax.set_xlabel("repeated queries (k)", color="#7d849d")
    ax.set_ylabel("epsilon savings vs sequential", color="#7d849d")
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    for k, saving in zip(ks, savings):
        ax.annotate(
            f"{saving:.1f}%",
            (k, saving),
            textcoords="offset points",
            xytext=(0, 9 if saving >= 0 else -16),
            ha="center",
            color="#d8dce8",
            fontsize=8,
        )

    for spine in ax.spines.values():
        spine.set_color("#343b55")
    ax.tick_params(colors="#7d849d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "svg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the composition savings chart from CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    render_chart(args.input, args.output, args.epsilon)


if __name__ == "__main__":
    main()
