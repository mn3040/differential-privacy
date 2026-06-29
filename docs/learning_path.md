# Learning Path

Use this project as a lab notebook. The best way to learn differential privacy
is to change one parameter at a time and predict what should happen before you
run the query.

## 1. Start With Count

```powershell
python -m dp_demo.cli --query count --mechanism laplace --epsilon 1 --seed 1
```

Count is the cleanest query because its sensitivity is always `1`. If one row
changes, the count can move by at most one.

Try smaller epsilon:

```powershell
python -m dp_demo.cli --query count --mechanism laplace --epsilon 0.1 --seed 1
```

The true value is the same, but the noisy value should jump farther away.

## 2. Compare Laplace And Gaussian

Laplace is pure DP:

```text
delta = 0
```

Gaussian is approximate DP:

```text
0 < delta < 1
```

Try:

```powershell
python -m dp_demo.cli --query count --mechanism gaussian --epsilon 1 --delta 0.000001 --seed 1
```

Gaussian noise is calibrated with:

```text
sigma = sqrt(2 ln(1.25 / delta)) * sensitivity / epsilon
```

Smaller delta makes sigma larger.

## 3. Inspect Sensitivity For Means

```powershell
python -m dp_demo.cli --query mean --column petal_length --species setosa --epsilon 1 --seed 1
```

The demo clips petal length to `[1.0, 7.0]`. For a replacement-neighbor mean,
one changed row can move the mean by:

```text
(upper - lower) / n
```

For `setosa`, the sample has `n = 10`, so sensitivity is:

```text
(7.0 - 1.0) / 10 = 0.6
```

## 4. Read The Code In This Order

1. `dp_demo/mechanisms.py`: noise distributions and calibration formulas.
2. `dp_demo/dataset.py`: public clipping bounds.
3. `dp_demo/queries.py`: sensitivity choices and private releases.
4. `dp_demo/cli.py`: command-line interface.
5. `dp_demo/app.py`: browser interface.

The key habit is to ask: "What is the maximum effect one row can have?" Once
you can answer that, the mechanism calibration becomes much less mysterious.
