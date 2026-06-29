"""Private query planning over a loaded Dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dataset import Dataset
from .mechanisms import PrivacyBudget, release_exponential, release_gaussian, release_laplace


MECHANISMS = ("laplace", "gaussian", "exponential")
QUERY_TYPES = ("count", "sum", "mean", "mode_category")


@dataclass(frozen=True)
class QueryResult:
    query: str
    mechanism: str
    column: str | None
    category: str | None
    epsilon: float
    delta: float
    true_value: float
    noisy_value: float
    sensitivity: float
    rows_used: int
    explanation: str
    true_label: str | None = None
    released_label: str | None = None
    probabilities: dict[str, float] | None = None


def run_query(
    dataset: Dataset,
    query: str,
    mechanism: str,
    epsilon: float,
    delta: float = 0.0,
    column: str | None = None,
    category: str | None = None,
    seed: int | None = None,
) -> QueryResult:
    """Run a differentially private query over a dataset's rows."""

    if query not in QUERY_TYPES:
        raise ValueError(f"query must be one of: {', '.join(QUERY_TYPES)}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of: {', '.join(MECHANISMS)}")
    if category is not None and category not in dataset.categories:
        raise ValueError(f"category must be one of: {', '.join(dataset.categories)}")
    if query in {"sum", "mean"} and column not in dataset.numeric_columns:
        raise ValueError(f"{query} requires a numeric column")
    if query == "mode_category" and mechanism != "exponential":
        raise ValueError("mode_category requires the exponential mechanism")
    if query != "mode_category" and mechanism == "exponential":
        raise ValueError("the exponential mechanism only applies to the mode_category query")

    filtered = [row for row in dataset.rows if category is None or dataset.category(row) == category]
    rng = random.Random(seed)
    budget = PrivacyBudget(epsilon=epsilon, delta=delta)

    result_column = column

    if query == "mode_category":
        result_column = None
        counts = {c: sum(1 for row in filtered if dataset.category(row) == c) for c in dataset.categories}
        true_label = max(counts, key=counts.get)
        sensitivity = 1.0  # adding/removing one row changes any one category's count by at most 1
        utilities = [float(counts[c]) for c in dataset.categories]
        released_label, probabilities = release_exponential(
            list(dataset.categories), utilities, sensitivity, budget, rng
        )
        explanation = (
            f"Exponential mechanism selects a value of '{dataset.category_column}' with probability "
            "proportional to exp(epsilon * count / (2 * sensitivity)) instead of adding noise to a "
            "number, so the released answer is always a real category."
        )
        return QueryResult(
            query=query,
            mechanism=mechanism,
            column=result_column,
            category=category,
            epsilon=epsilon,
            delta=delta,
            true_value=float(counts[true_label]),
            noisy_value=float(counts[released_label]),
            sensitivity=sensitivity,
            rows_used=len(filtered),
            explanation=explanation,
            true_label=true_label,
            released_label=released_label,
            probabilities=dict(zip(dataset.categories, probabilities)),
        )

    if query == "count":
        result_column = None
        true_value = float(len(filtered))
        sensitivity = 1.0
        explanation = "Count sensitivity is 1 because one changed row can alter a count by at most one."
    else:
        assert column is not None
        lower, upper = dataset.numeric_columns[column]
        values = [dataset.clip(dataset.value(row, column), column) for row in filtered]
        if query == "sum":
            true_value = float(sum(values))
            sensitivity = max(abs(lower), abs(upper))
            explanation = (
                f"Sum clips {column} to [{lower}, {upper}], so add/remove sensitivity is "
                f"max(abs(lower), abs(upper)) = {sensitivity}."
            )
        else:
            true_value = float(sum(values) / len(values)) if values else 0.0
            sensitivity = (upper - lower) / max(len(values), 1)
            explanation = (
                f"Mean clips {column} to [{lower}, {upper}] and uses replacement-neighbor "
                f"sensitivity (upper - lower) / n = {sensitivity:.6g}."
            )

    if mechanism == "laplace":
        noisy_value = release_laplace(true_value, sensitivity, budget, rng)
    else:
        noisy_value = release_gaussian(true_value, sensitivity, budget, rng)

    return QueryResult(
        query=query,
        mechanism=mechanism,
        column=result_column,
        category=category,
        epsilon=epsilon,
        delta=delta,
        true_value=true_value,
        noisy_value=noisy_value,
        sensitivity=sensitivity,
        rows_used=len(filtered),
        explanation=explanation,
    )
