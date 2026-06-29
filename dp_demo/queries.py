"""Private query planning over the bundled Iris dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dataset import NUMERIC_COLUMNS, SPECIES, IrisRow, clip
from .mechanisms import PrivacyBudget, release_gaussian, release_laplace


MECHANISMS = ("laplace", "gaussian")
QUERY_TYPES = ("count", "sum", "mean")


@dataclass(frozen=True)
class QueryResult:
    query: str
    mechanism: str
    column: str | None
    species: str | None
    epsilon: float
    delta: float
    true_value: float
    noisy_value: float
    sensitivity: float
    rows_used: int
    explanation: str


def run_query(
    rows: list[IrisRow],
    query: str,
    mechanism: str,
    epsilon: float,
    delta: float = 0.0,
    column: str | None = None,
    species: str | None = None,
    seed: int | None = None,
) -> QueryResult:
    """Run a differentially private query over the Iris rows."""

    if query not in QUERY_TYPES:
        raise ValueError(f"query must be one of: {', '.join(QUERY_TYPES)}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of: {', '.join(MECHANISMS)}")
    if species is not None and species not in SPECIES:
        raise ValueError(f"species must be one of: {', '.join(SPECIES)}")
    if query in {"sum", "mean"} and column not in NUMERIC_COLUMNS:
        raise ValueError(f"{query} requires a numeric column")

    filtered = [row for row in rows if species is None or row.species == species]
    rng = random.Random(seed)
    budget = PrivacyBudget(epsilon=epsilon, delta=delta)

    result_column = column

    if query == "count":
        result_column = None
        true_value = float(len(filtered))
        sensitivity = 1.0
        explanation = "Count sensitivity is 1 because one changed row can alter a count by at most one."
    else:
        assert column is not None
        lower, upper = NUMERIC_COLUMNS[column]
        values = [clip(row.value(column), column) for row in filtered]
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
        species=species,
        epsilon=epsilon,
        delta=delta,
        true_value=true_value,
        noisy_value=noisy_value,
        sensitivity=sensitivity,
        rows_used=len(filtered),
        explanation=explanation,
    )
