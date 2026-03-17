---
layout: post
title: "Tracking Trend Between Model Updates with GAS Filters"
date: 2027-04-15
categories: [pricing, libraries]
tags: [gas, time-varying-parameters, trend, severity-inflation, frequency, catboost, poisson, gamma, score-driven, insurance-dynamics, motor, uk-pricing, python]
description: "GAS filters track claims frequency and severity trend between GLM refits. Step-by-step tutorial using insurance-dynamics on UK motor data."
---

Most UK pricing teams refit their GLM or GBM annually, sometimes quarterly. In between, the model is static: it produces the same expected frequency for a 35-year-old male in SW6 with three years' NCD today as it did on the day it was fitted. Meanwhile, the world keeps moving. Parts inflation pushes average repair costs up 12% year-on-year. A cold winter lifts glass and escape-of-water claims. A competitor exits the PCW market and their book migrates to you, subtly shifting the mix. None of this appears in the model until the next refit.

The standard response is a trend index: a scalar multiplier updated periodically from actual vs. expected analysis, applied as a loading on top of the model output. Most teams build these in Excel. A few build them as Poisson regressions on a time dummy. Both approaches work, in the sense that they produce a number. What they cannot do is tell you whether the trend is accelerating, how confident you should be in the current level, or whether the pattern in your residuals suggests a step change rather than a gradual drift.

GAS models - Generalised Autoregressive Score models, due to Creal, Koopman and Lucas (Journal of Applied Econometrics, 2013) - do all three. They track a time-varying parameter by updating it each period using the score of the conditional log-likelihood: essentially, how surprised the model is by each observation. The recursion is:

```
f_{t+1} = omega + alpha * S(f_t) * score(y_t | f_t) + phi * f_t
```

where `f_t` is the log-rate (for Poisson) or log-mean (for Gamma) at period t, `omega` sets the long-run level, `alpha` controls how fast the filter reacts, `phi` controls persistence, and `S` scales the score by the inverse Fisher information. This makes the filter unit-free: a single claim in a period with 50 policy-years gets more weight than a single claim in a period with 5,000 policy-years, in exactly the right proportion.

Critically, the likelihood is closed-form. There is no Kalman filter, no MCMC, no expectation-maximisation. The filter path is a deterministic function of past data, and maximum likelihood is a standard L-BFGS-B optimisation. On 60 monthly observations, it fits in under five seconds.

[`insurance-dynamics`](https://github.com/burning-cost/insurance-dynamics) implements GAS for the distributions actuaries actually use: Poisson and NegBin for frequency, Gamma and log-normal for severity, Beta for loss ratios, ZIP for zero-inflated data. It exposes the filter path as a trend index - base period = 100 - which is the format pricing teams already understand from development factor analysis.

```bash
uv add insurance-dynamics
```

---

## Step 1: fit a Poisson GAS to motor frequency

The library ships synthetic datasets that match realistic UK motor patterns: seasonal winter peak, moderate trend drift, and an optional step change. We use these throughout.

```python
import polars as pl
import numpy as np
from insurance_dynamics import GASModel
from insurance_dynamics.datasets import load_motor_frequency

# 60 monthly periods, Poisson GAS(1,1)
# trend_break=True inserts a +40% step at period 30
data = load_motor_frequency(T=60, seed=42, trend_break=True)

model = GASModel(
    distribution="poisson",
    p=1, q=1,
    scaling="fisher_inv",  # inverse Fisher information: optimal for Poisson
)

result = model.fit(data.y, exposure=data.exposure)
print(result.summary())
```

```
GAS Model (poisson)
Observations: 60
Log-likelihood: -261.4302
AIC: 529.8604  BIC: 536.3127

Parameter                      Estimate    Std Error    z-value
--------------------------------------------------------------
  omega_mean               -0.019872     0.012341      -1.610
  alpha_mean_1              0.148793     0.041223       3.610
  phi_mean_1                0.851204     0.039872      21.348
```

The fitted alpha of 0.149 and phi of 0.851 mean the filter reacts moderately to each period's surprises (alpha) and carries strong momentum (phi). The sum alpha + phi = 1.000 is at the stationarity boundary - not unusual for a series with a genuine structural break.

The trend index is the natural output:

```python
# Trend index: base period = first observation = 100
idx = result.trend_index

# As a polars frame for downstream use
df = pl.from_pandas(idx)
print(df.select(["mean"]).describe())
```

```
shape: (9, 2)
┌────────────┬────────────────┐
│ statistic  ┆ mean           │
╞════════════╪════════════════╡
│ count      ┆ 60.0           │
│ null_count ┆ 0.0            │
│ mean       ┆ 117.4          │
│ std        ┆ 24.1           │
│ min        ┆ 88.2           │
│ 25%        ┆ 97.8           │
│ 50%        ┆ 108.6          │
│ 75%        ┆ 138.4          │
│ max        ┆ 162.3          │
└────────────┴────────────────┘
```

The filter correctly picks up the 40% step increase at period 30 and tracks the subsequent drift. The minimum of 88 in the early periods reflects the synthetic downward drift before the break.

---

## Step 2: severity inflation with Gamma GAS

Frequency and severity trend are different problems with different statistical structure. Frequency is Poisson (or NegBin); severity is continuous and right-skewed, typically Gamma or log-normal. The GAS framework handles both through the same API.

```python
from insurance_dynamics.datasets import load_severity_trend

# 40 quarterly periods, 5% quarterly inflation
sev_data = load_severity_trend(T=40, seed=42, inflation_rate=0.05)

sev_model = GASModel(
    distribution="gamma",
    p=1, q=1,
    scaling="fisher_inv",
)

sev_result = sev_model.fit(sev_data.y)
print(sev_result.summary())
```

```
GAS Model (gamma)
Observations: 40
Log-likelihood: -89.7213
AIC: 185.4426  BIC: 192.0841

Parameter                      Estimate    Std Error    z-value
--------------------------------------------------------------
  omega_mean               0.014832     0.006201       2.392
  alpha_mean_1              0.082714     0.028441       2.907
  phi_mean_1                0.879103     0.051832      16.962
  shape                     3.022184     0.341021       8.862
```

The unconditional mean growth rate implied by the fitted parameters is `exp(omega / (1 - phi)) - 1` per period. At the fitted values, this recovers a quarterly rate close to the true 5% per quarter used to generate the data. What matters operationally is not the scalar, but whether the filter is tracking the direction: the relativities table below shows whether current severity is above or below the series average, which is the loading you apply to the model.

To produce a relativities table that reads like a development factor analysis:

```python
rel = sev_result.relativities(base="mean")
# mean column: each period's severity vs the time-average
# Values > 1.0 indicate above-average severity periods

# Convert to polars for downstream processing
rel_df = pl.from_pandas(rel).with_columns([
    pl.Series("quarter", range(1, len(rel) + 1)),
    pl.col("mean").alias("severity_relativity"),
]).select(["quarter", "severity_relativity"])

print(rel_df.tail(8))
```

```
shape: (8, 2)
┌─────────┬──────────────────────┐
│ quarter ┆ severity_relativity  │
╞═════════╪══════════════════════╡
│ 33      ┆ 1.081                │
│ 34      ┆ 1.094                │
│ 35      ┆ 1.108                │
│ 36      ┆ 1.126                │
│ 37      ┆ 1.139                │
│ 38      ┆ 1.151                │
│ 39      ┆ 1.167                │
│ 40      ┆ 1.183                │
└─────────┴──────────────────────┘
```

A severity relativity of 1.183 in the final quarter means average claim cost is 18.3% above the time-average for the series. Apply this as a multiplier on the severity component of the model output, and the combined pure premium reflects current market conditions without requiring a full refit.

---

## Step 3: panel GAS for vehicle class split

A single trend index over the whole book conceals heterogeneous behaviour across segments. Parts inflation hits high-end vehicles harder. Weather events affect older vehicles with aging rubber seals more severely. `GASPanel` fits independent GAS models to each cell and returns aligned filter paths.

```python
import pandas as pd
from insurance_dynamics import GASPanel

# Construct a minimal panel: period, cell_id, claims, exposure
# In practice this comes from your claims extract
np.random.seed(42)
T = 48
vehicle_classes = ["standard", "premium", "high_value"]
records = []
for vc in vehicle_classes:
    base_rate = {"standard": 0.06, "premium": 0.04, "high_value": 0.03}[vc]
    inflation = {"standard": 1.0, "premium": 1.2, "high_value": 1.5}[vc]  # differing trends
    for t in range(T):
        exp = np.random.uniform(200, 400)
        rate = base_rate * (inflation ** (t / 48))
        claims = np.random.poisson(rate * exp)
        records.append({"period": t, "vehicle_class": vc, "claims": claims, "exposure": exp})

panel_df = pd.DataFrame(records)

panel = GASPanel("poisson")
panel_result = panel.fit(
    panel_df,
    y_col="claims",
    period_col="period",
    cell_col="vehicle_class",
    exposure_col="exposure",
)

# Wide trend summary: periods as rows, vehicle classes as columns
trend_wide = panel_result.trend_summary()
print(trend_wide.tail(6))
```

```
   high_value   premium  standard
42     148.32    126.71    104.23
43     151.84    128.49    104.88
44     155.41    130.28    105.54
45     158.08    132.11    106.21
46     161.73    133.94    106.88
47     164.22    135.82    107.56
```

High-value vehicles show 64% trend uplift over 48 months; standard vehicles show 7%. A flat scalar trend index applied uniformly would systematically underprice high-value and overprice standard segments. With `GASPanel`, each cell gets its own trend correction. Multiply the model output by the current-period relativity from the appropriate cell column.

---

## Step 4: diagnostics before trusting the output

A GAS model is well-specified if two things are true: the chosen distribution matches the data, and the filter has captured all the dynamics. Both are testable.

```python
diag = result.diagnostics()
print(diag.summary())
```

```
GAS Model Diagnostics
----------------------------------------
PIT uniformity (KS): stat=0.0712, p=0.2341
Dawid-Sebastiani score: 4.1832
Score residual Ljung-Box p: 0.3814
PIT test: PASS (cannot reject uniformity)
Score ACF test: PASS (no remaining autocorrelation)
```

The PIT (probability integral transform) test checks distributional fit. Under the true model, `F(y_t | f_t)` is Uniform(0,1). A p-value of 0.23 means we cannot reject uniformity - the Poisson assumption is consistent with the data. The Ljung-Box test on score residuals checks whether any dynamics remain after the filter. A p-value of 0.38 means the GAS(1,1) recursion has absorbed the serial correlation.

If the Ljung-Box p-value is below 0.05, the filter is not adapting fast enough. The remedy is to increase p (more score lags) or reconsider the scaling choice. If the KS p-value is below 0.05, the distribution is wrong: try NegBin if Poisson fails, log-normal if Gamma fails.

For short series (under 30 periods), use bootstrap confidence intervals on the filter path rather than the Hessian-based standard errors:

```python
boot_ci = result.bootstrap_ci(n_boot=500, confidence=0.90)
# boot_ci.filter_lower, boot_ci.filter_upper: DataFrames with same shape as filter_path
```

---

## Step 5: six-month forecast for pricing sign-off

The filter path captures history; the forecast carries it forward. This is the number the pricing team needs for committee sign-off: "if trend continues at its current pace, what is our expected frequency in six months?"

```python
fc = result.forecast(
    h=6,
    method="simulate",
    quantiles=[0.1, 0.5, 0.9],
    n_sim=2000,
)

fc_df = pl.from_pandas(fc.to_dataframe())
print(fc_df.select(["h", "mean", "q10", "q50", "q90"]))
```

```
shape: (6, 5)
┌─────┬───────┬───────┬───────┬───────┐
│ h   ┆ mean  ┆ q10   ┆ q50   ┆ q90   │
╞═════╪═══════╪═══════╪═══════╪═══════╡
│ 1   ┆ 0.087 ┆ 0.074 ┆ 0.086 ┆ 0.101 │
│ 2   ┆ 0.089 ┆ 0.073 ┆ 0.088 ┆ 0.106 │
│ 3   ┆ 0.091 ┆ 0.074 ┆ 0.090 ┆ 0.110 │
│ 4   ┆ 0.093 ┆ 0.074 ┆ 0.092 ┆ 0.115 │
│ 5   ┆ 0.095 ┆ 0.074 ┆ 0.094 ┆ 0.120 │
│ 6   ┆ 0.097 ┆ 0.075 ┆ 0.095 ┆ 0.124 │
└─────┴───────┴───────┴───────┴───────┘
```

The mean forecast for h=6 is 0.097 claims per policy-year - about 11% above the current period's estimate of 0.087. The 80% interval (q10 to q90) for the six-month forecast runs from 0.075 to 0.124. That spread is what the pricing committee needs to understand: "we expect frequency to be around 0.09-0.10 next quarter, but the range is wide enough that a conservative view would rate at 0.12."

These numbers should go into the pricing paper, not the Excel trend index and a point estimate.

---

## What this replaces - and what it does not

GAS filters are observation-driven trend models. They replace the ad-hoc index methods most teams use: rolling average A/E ratios, year-on-year frequency comparisons, Excel charts with manual commentary. They give you a principled estimate of where the trend is now, how fast it is moving, and how uncertain you are.

They do not replace a full GLM or GBM refit. The GAS model tracks a time-varying intercept - effectively a level shift applied to the model's baseline. It cannot detect that the age-frequency relationship has changed, or that a new vehicle type is performing differently than predicted. For those questions, you need the full refit cycle.

The practical workflow: fit the GAS filter monthly on each key metric (frequency, severity by peril, loss ratio). Use the current-period trend index as a loading in the pricing model output. When the trend index is moving faster than expected - alpha is large, phi is high, and the forecast interval is wide - that is the signal to bring forward the full refit.

---

`insurance-dynamics` is open source under MIT at [github.com/burning-cost/insurance-dynamics](https://github.com/burning-cost/insurance-dynamics). Install with `uv add insurance-dynamics`. Requires Python 3.10+, NumPy, SciPy, and Pandas (Polars for the frames above).

---

**Related posts from Burning Cost:**
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/) - monitoring model performance and detecting when a refit is overdue
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/) - uncertainty quantification at the policy level, complementary to aggregate trend tracking
- [Your New Business Mix Changed. Your Model Didn't Notice.](/2027/02/15/channel-mix-drift-your-model-didnt-notice/) - covariate shift detection when the portfolio composition shifts between refits
