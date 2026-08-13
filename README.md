# driftkit

Correct, dependency-light stability metrics for tabular models — PSI, CSI, WOE/IV and calibration drift.

Depends on `numpy` and `pandas`. Nothing else.

```bash
pip install driftkit
```

## Why another drift library

Most PSI implementations you'll find in a notebook are subtly wrong, and the popular monitoring frameworks that get it right bring a large dependency tree with them.

The single most common bug is re-binning the comparison sample:

```python
# WRONG — each sample gets its own quantiles
expected_pct = pd.qcut(reference, 10).value_counts(normalize=True)
actual_pct = pd.qcut(current, 10).value_counts(normalize=True)
```

Quantile bins are equal-frequency *by construction*, so both histograms come out uniform and PSI collapses toward zero no matter how far the population has actually moved. `driftkit` makes this mistake hard: bins are learned once on the reference sample, frozen in a `BinSpec`, and applied unchanged thereafter.

There is a test for exactly this. On a 100-point mean shift, the correct calculation reads **3.28**; naive re-binning reads **0.00** — not "a bit low", but exactly zero, because re-fitting quantiles on each sample forces both histograms to be identical by construction.

Also handled, because these are the things that bite in production:

- **Out-of-range values.** Outer bin edges are `-inf`/`+inf`, so a value below anything in the reference set lands in bin 0 instead of silently becoming NaN.
- **Missing data.** NaN gets its own bin rather than being dropped. A feed that starts returning nulls *is* drift, and dropping nulls hides it precisely when you need to see it.
- **Empty bins.** `ln(actual/expected)` is undefined when either side is empty. Instead of an arbitrary epsilon, `driftkit` uses additive smoothing (Jeffreys prior, `alpha=0.5`) — the well-defined version of the same idea.
- **Tied features.** Duplicate quantile edges collapse instead of producing empty bins that inflate PSI with noise.

## Usage

### Population drift on one feature

```python
import numpy as np
from driftkit import psi

rng = np.random.default_rng(0)
reference = rng.normal(700, 50, 10_000)  # training population
current = rng.normal(660, 60, 10_000)  # this month's scoring window

result = psi(reference, current)

result.value  # 0.5244
result.interpretation  # 'significant shift'
```

Every headline number can be traced back to the bins that caused it:

```python
result.top_bins.head(3)
```

| bin | expected_count | actual_count | expected_pct | actual_pct | contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[-inf, 637.049)` | 1000 | 3524 | 0.100 | 0.352 | 0.3177 |
| `[764.127, inf)` | 1000 | 405 | 0.100 | 0.041 | 0.0537 |
| `[742.792, 764.127)` | 1000 | 415 | 0.100 | 0.042 | 0.0514 |

The bottom decile of the reference population now holds 35% of the current one — that single bin is 61% of the total PSI.

### Monitoring a whole frame

```python
from driftkit import monitor

report = monitor(reference_df, current_df)
print(report.to_markdown(limit=5))
report.unstable  # features with PSI >= 0.25
report.results["fico"]  # full per-bin detail for one feature
```

### Comparing many periods against one baseline

Fit the bins once so the series is comparable over time:

```python
from driftkit import fit_bins, psi

spec = fit_bins(reference["fico"], n_bins=10)
trend = [psi(reference["fico"], period["fico"], bins=spec).value for period in months]
```

### Which input moved the score

PSI on the model output tells you the population moved. CSI on an input tells you *which feature* moved it — and with scorecard points, how many points that's worth:

```python
from driftkit import csi

result = csi(reference["fico"], current["fico"], points=fico_points, bins=spec)
result.table["score_impact"].sum()  # expected point movement from this feature
```

### WOE encoding and Information Value

```python
from driftkit import WOEEncoder

encoder = WOEEncoder(n_bins=10).fit(X_train, y_train)
X_woe = encoder.transform(X_train)

encoder.information_values_  # {'fico': 0.809, 'dti': 0.078, 'noise': 0.001}
encoder.summary()  # per-bin WOE table, sorted by IV
```

Follows the scikit-learn transformer protocol, so it drops into a `Pipeline` — but doesn't import scikit-learn, because the arithmetic doesn't need it.

Sign convention is stated explicitly: `WOE = ln(P(bin | y=0) / P(bin | y=1))`. Positive WOE means the bin is over-represented among non-events. The opposite convention flips every coefficient's sign, so it's worth being unambiguous.

### Calibration drift

Discrimination and calibration fail independently. A model whose AUC hasn't moved can still have drifted badly if its probabilities no longer mean what they used to:

```python
from driftkit import calibration_report

report = calibration_report(y_true, y_prob)

report.ece  # 0.0039  — expected calibration error
report.bias  # -0.0032 — mean prediction minus observed rate
report.reliability  # 0.000022 — calibration term (lower is better)
report.resolution  # 0.0824   — discrimination term (higher is better)
report.table  # per-bin reliability diagram data
```

`reliability - resolution + uncertainty ≈ brier` (Murphy's decomposition). The `uncertainty` term is base-rate variance — a property of the sample, not the model, and the reason Brier scores move when the base rate moves even if nothing about the model changed.

## On the interpretation bands

`interpret_psi` uses the conventional retail-credit bands: `<0.10` stable, `0.10–0.25` moderate, `>0.25` significant.

These are industry convention, not a hypothesis test. They carry no sample-size correction, so on a few hundred rows a "significant" reading is often noise, and on several million rows a "stable" reading can still hide a shift that matters. Treat them as a triage prompt, not a verdict.

## API

| Function | Purpose |
| --- | --- |
| `psi(expected, actual)` | Population Stability Index |
| `csi(expected, actual, points=...)` | Characteristic Stability Index, with score impact |
| `psi_from_counts(expected, actual)` | PSI from pre-binned count vectors |
| `monitor(reference_df, current_df)` | PSI across every shared column |
| `fit_bins(x, n_bins)` | Learn a frozen `BinSpec` |
| `WOEEncoder` | Weight of Evidence transformer |
| `information_value(x, y)` | IV of one feature |
| `calibration_report(y_true, y_prob)` | Brier, ECE, Murphy decomposition |

## Development

```bash
pip install -e ".[dev]"
pytest --cov=driftkit
ruff check . && mypy
```

## License

MIT
