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
python -m dp_demo.cli --query mean --column petal_length --category setosa --epsilon 1 --seed 1
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

## 4. Pick A Category Without Adding Noise To A Number

Laplace and Gaussian both add noise to a number. But "which species is most
common?" doesn't have a number to add noise to — the output has to be a real
category value, or the answer is meaningless. That's what the exponential
mechanism is for: instead of perturbing the answer, it perturbs the
*probability of selecting* each candidate.

```powershell
python -m dp_demo.cli --query mode_category --mechanism exponential --epsilon 2 --seed 1
```

Each category gets a utility score (its row count), and the selection
probability is:

```text
Pr[output = category] proportional to exp(epsilon * count / (2 * sensitivity))
```

Sensitivity is `1` here too: adding or removing one row changes any single
category's count by at most one. Try a very small epsilon and watch the
probabilities flatten toward uniform — at `epsilon -> 0` you're picking a
category uniformly at random, which leaks nothing but also tells you nothing.
Switch `--dataset pums` to try this on `married`/`unmarried` instead of
species.

## 5. Spend Your Budget Across Multiple Queries

Every query above costs privacy budget. Run the same kind of query twice and
the costs add up. `dp_demo/composition.py` implements two ways to total that
cost:

```powershell
python -m dp_demo.cli compose --method sequential --k 3 --epsilon 0.3
python -m dp_demo.cli compose --method advanced --k 100 --epsilon 0.1 --delta 1e-5 --delta-prime 1e-5
```

Sequential composition just adds epsilons: `k` queries at `epsilon` each cost
`k * epsilon`, exactly, no assumptions. Advanced composition is a tighter,
`sqrt(k)`-shaped bound for many repeats of the *same* `(epsilon, delta)`
mechanism, at the cost of an extra small failure probability `delta'`. Run
both commands above and compare: sequential composition of 100 queries at
`epsilon = 0.1` costs `10.0`; advanced composition costs about `5.8`. The gap
is the whole reason advanced composition exists — pure addition is correct
but wasteful once you're running dozens of queries.

## 6. Global Vs. Local Sensitivity

Every sensitivity figure used so far is a *global* sensitivity: the worst
case over every possible neighboring dataset, which is why the mean's
sensitivity formula uses the public clipping bounds `[lower, upper]` rather
than the data actually observed. `dp_demo/composition.py` also has
`local_sensitivity_mean`, which uses the *observed* min/max of the filtered
rows instead:

```powershell
python -c "from dp_demo.composition import global_sensitivity_mean, local_sensitivity_mean; print(global_sensitivity_mean(1.0, 7.0, 10))"
```

Local sensitivity is never larger than global sensitivity (less noise,
better accuracy) but it depends on the dataset itself — and publishing a
noise scale that depends on the data can leak information through the back
door, which is why this demo's mechanisms calibrate to global sensitivity by
default. Treat `local_sensitivity_mean` as a comparison tool, not a
drop-in replacement.

## 7. Read The Code In This Order

1. `dp_demo/mechanisms.py`: noise distributions, calibration formulas, and
   the exponential mechanism's selection weights.
2. `dp_demo/dataset.py`: public clipping bounds.
3. `dp_demo/queries.py`: sensitivity choices and private releases.
4. `dp_demo/composition.py`: budget composition and sensitivity comparisons.
5. `dp_demo/cli.py`: command-line interface (`query` and `compose`).
6. `dp_demo/app.py`: browser interface.

The key habit is to ask: "What is the maximum effect one row can have?" Once
you can answer that, the mechanism calibration becomes much less mysterious.
