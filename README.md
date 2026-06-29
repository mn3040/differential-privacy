# Differential Privacy Library Demo

This is a small educational differential privacy project. It implements the
Laplace and Gaussian mechanisms from scratch in Python, then exposes a query
interface over a compact excerpt of Fisher's public Iris dataset.

The goal is to make the privacy math visible:

- **epsilon** controls the privacy/noise tradeoff. Smaller epsilon means more noise.
- **delta** is only used by the Gaussian mechanism. Laplace is pure DP with `delta = 0`.
- **sensitivity** is the maximum amount one person's row can change the query.
- **clipping bounds** make sensitivity finite for numeric sums and means.

## Run the CLI

```powershell
python -m dp_demo.cli --query mean --column petal_length --species setosa --mechanism laplace --epsilon 1 --seed 42
```

Gaussian requires a nonzero delta:

```powershell
python -m dp_demo.cli --query count --mechanism gaussian --epsilon 1 --delta 0.000001 --seed 42
```

## Run the Web Demo

```powershell
python -m dp_demo.app --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

## What Is Implemented

### Laplace mechanism

For a numeric query `f(D)` with L1 sensitivity `S`, the release is:

```text
f(D) + Laplace(0, S / epsilon)
```

This gives pure epsilon-DP when the sensitivity is correct.

### Gaussian mechanism

For a numeric query `f(D)` with L2 sensitivity `S`, the release is:

```text
f(D) + Normal(0, sigma)
sigma = sqrt(2 ln(1.25 / delta)) * S / epsilon
```

This is the classic textbook calibration for approximate `(epsilon, delta)`-DP.

## Query Sensitivities

The demo supports `count`, `sum`, and `mean`.

- `count`: sensitivity is `1`.
- `sum`: values are clipped to a public range, then sensitivity is the largest
  possible clipped magnitude.
- `mean`: values are clipped to a public range, then replacement-neighbor
  sensitivity is `(upper - lower) / n`.

For production systems, use a mature library such as OpenDP or Google
Differential Privacy. This project is deliberately small so the mechanism math
is easy to inspect.

## Tests

```powershell
python -m unittest discover
```

## Learn By Experimenting

See [docs/learning_path.md](docs/learning_path.md) for a guided sequence of
queries that explains sensitivity, epsilon, delta, and mechanism calibration.
