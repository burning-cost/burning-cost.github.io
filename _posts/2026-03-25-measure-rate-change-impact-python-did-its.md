---
layout: post
title: "Measuring Rate Change Impact with DiD and ITS in Python"
date: 2026-03-25
categories: [techniques]
tags: [difference-in-differences, interrupted-time-series, rate-change-evaluation, causal-inference, python, insurance-causal, tutorial]
description: "A hands-on tutorial for the RateChangeEvaluator in insurance-causal v0.6.0. DiD when you have a control group. ITS when you don't. Real code, real API."
---

[Previously we argued](/2026/03/13/your-rate-change-didnt-prove-anything/) that before/after loss ratio comparisons are not evidence that a rate change worked. This post is the follow-up: here is the code that produces actual evidence.

`insurance-causal` v0.6.0 ships a `rate_change` sub-package built around a single entry point, `RateChangeEvaluator`. It handles the two situations a pricing team actually faces:

- **DiD** (Difference-in-Differences): you changed rates on some segments and left others alone. You have a control group.
- **ITS** (Interrupted Time Series): you changed rates across the entire book. No control group exists.

The evaluator chooses the method automatically based on whether your data contains both treated and untreated segments. The API is `fit()`, `summary()`, and a set of plot methods. 84 tests passing. Source at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal).

---

## The data format

The library expects a pandas DataFrame with one row per segment-period. The required columns depend on which method will be used:

| Column | Required for | Purpose |
|--------|-------------|---------|
| `outcome` (or any name) | Both | The thing you are measuring (loss ratio, claim frequency, conversion rate) |
| `period` | Both | Time index — integer, "YYYY-QN" string, or pandas Period |
| `treated` (or any name) | DiD | 0 for control segments, 1 for treated segments |
| `earned_exposure` (optional but strongly recommended) | Both | Exposure weights (policy-years, earned premium) |
| `segment` (optional) | DiD | Unit identifier for cluster-robust standard errors |

The library ships a synthetic data generator so you can validate the methodology before touching live data:

```python
from insurance_causal.rate_change import make_rate_change_data

# DiD panel: 40 segments, 16 quarterly periods, rate change at period 9
# True ATT = -0.05 (five percentage point reduction in claim frequency)
df = make_rate_change_data(
    n_segments=40,
    n_periods=16,
    treatment_period=9,
    true_att=-0.05,
    seed=0,
)

print(df.columns.tolist())
# ['segment', 'period', 'quarter', 'treated', 'outcome', 'earned_exposure', 'rate_change']

print(df.shape)
# (640, 7)

print(df['treated'].value_counts())
# 1    320
# 0    320
```

The `earned_exposure` column here is log-normally distributed around 500 policy-years per segment-period. Pass `exposure_cv=0.5` to control heterogeneity — this matters because the estimator uses WLS with exposure weights, and thin segments should contribute less to the estimate than thick ones.

---

## DiD: when you have a control group

This is the standard case: a motor segment received a 12% rate increase in Q1 2023, commercial lines did not. Or young drivers were targeted for a rate correction while the over-40 book held flat. Anywhere you have treated and untreated segments observed over the same periods, DiD applies.

```python
from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data

df = make_rate_change_data(n_segments=40, true_att=-0.05, seed=0)

evaluator = RateChangeEvaluator(
    outcome_col='outcome',
    treatment_period=9,
    unit_col='segment',
    weight_col='earned_exposure',
)

evaluator.fit(df)
print(evaluator.summary())
```

Output:

```
============================================================
Rate Change Evaluation Summary
============================================================
Method         : DiD
Outcome        : outcome
Treatment at   : period 9

Difference-in-Differences (TWFE)
----------------------------------------
ATT             : -0.0498***
Std error       : 0.0012
95% CI          : [-0.0521, -0.0475]
p-value         : 0.0000
SE type         : cluster
Observations    : 640
Units           : 40 (20 treated)
Periods         : 16
Parallel trends : F=0.63, p=0.644 (pass)

Significance: *** p<0.01  ** p<0.05  * p<0.1
============================================================
```

The true ATT was -0.05. The estimator recovers -0.0498. The parallel trends test — a joint F-test on the pre-treatment period-by-treatment interaction coefficients — passes with p=0.644. The assumption that treated and control segments were on the same trajectory before period 9 is consistent with the data.

The ATT here is in outcome units. If your outcome column is `loss_ratio` and the ATT is -0.048, that is a 4.8 percentage point causal reduction in the loss ratio attributable to the rate change — not a simple before/after comparison, but an estimate of what would not have happened without the intervention.

### Parallel trends: the event study plot

The formal test is useful, but the event study plot is what you show in the board pack. Each coefficient is the treatment-control difference in a single pre-treatment period relative to the period immediately before treatment (period T-1 = 0 by construction). If the pre-treatment coefficients bounce around zero without trend, the parallel trends assumption is plausible.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
evaluator.plot_event_study(ax=ax)
ax.set_title("Event study: claim frequency, young drivers Q1 2023 rate change")
plt.tight_layout()
plt.savefig("event_study.png", dpi=150)
```

`plot_event_study()` is only available when the method is DiD. If the evaluator used ITS (because no control group was found), it raises `RuntimeError`. This is intentional — there is no parallel trends assumption to test under ITS.

### Per-segment results

The base `RateChangeEvaluator` gives you the pooled ATT across all treated segments. If you want to know whether the rate change hit young drivers differently from urban drivers, run the evaluator separately on each sub-population. The `fit()` call is cheap — under a second on a 40-segment panel — so iterating over segments is practical:

```python
# Identify control segments (treated == 0) from the full panel
control_segments = df[df['treated'] == 0]['segment'].unique().tolist()
segments_of_interest = df[df['treated'] == 1]['segment'].unique()[:3].tolist()

for seg in segments_of_interest:
    df_sub = df[df['segment'].isin([seg] + control_segments)]
    ev = RateChangeEvaluator(
        outcome_col='outcome',
        treatment_period=9,
        unit_col='segment',
        weight_col='earned_exposure',
    )
    ev.fit(df_sub)
    result = ev._result.did
    print(f"{seg}: ATT={result.att:+.4f} (p={result.p_value:.3f})")
```

Note that running a separate estimator per segment reduces cluster count — the evaluator falls back from cluster-robust SE to HC3 when fewer than 20 clusters are available, and raises a warning accordingly.

---

## ITS: when the whole book moved

Sometimes there is no control group. A market-wide claims inflation response, a book-wide re-rating after an Ogden change, a simultaneous repricing across all channels because the old rates were simply wrong. When `treated` is all ones — or when you do not pass `treated_col` at all — `RateChangeEvaluator` selects ITS automatically and raises a warning:

```python
from insurance_causal.rate_change import make_rate_change_data, RateChangeEvaluator

df_its = make_rate_change_data(
    n_periods=20,
    treatment_period=9,
    true_level_shift=-0.02,
    true_slope_change=-0.003,
    seed=0,
    mode='its',
)

# No unit_col needed for ITS — this is aggregate time series data
evaluator_its = RateChangeEvaluator(
    outcome_col='outcome',
    treatment_period=9,
    weight_col='earned_exposure',
)

evaluator_its.fit(df_its)
print(evaluator_its.summary())
```

Output:

```
============================================================
Rate Change Evaluation Summary
============================================================
Method         : ITS
Outcome        : outcome
Treatment at   : period 9

Interrupted Time Series (Segmented Regression)
----------------------------------------
Level shift     : -0.0208***
  Std error     :  0.0021
  95% CI        : [-0.0250, -0.0167]
  p-value       :  0.0000
Slope change    : -0.002847***
  Std error     :  0.000412
  95% CI        : [-0.003690, -0.002004]
  p-value       :  0.0000
Pre-trend slope : +0.001012
Pre periods     : 8
Post periods    : 12
R-squared       : 0.967
HAC lags        : 4
Seasonality     : yes
============================================================
```

ITS decomposes the rate change effect into two components:

**Level shift**: the immediate step change at the intervention point. In this case, -0.0208 — the model says the outcome dropped about 2 percentage points at period 9, relative to the counterfactual (the pre-trend extrapolated forward).

**Slope change**: the change in the quarterly trend after the intervention. Here -0.002847 per period. After four post-treatment quarters, the cumulative effect is: `-0.0208 + (-0.002847 × 4) = -0.032` — a 3.2 percentage point reduction.

You can access `effect_at_k()` directly on the `ITSResult` object:

```python
its_result = evaluator_its._result.its

for k in [0, 1, 2, 4, 8]:
    print(f"k={k}: {its_result.effect_at_k(k):+.4f}")

# k=0: -0.0208
# k=1: -0.0237
# k=2: -0.0265
# k=4: -0.0322
# k=8: -0.0436
```

The `plot_its()` method shows the observed series against the counterfactual (pre-trend extrapolated) and shades the gap between them:

```python
fig, ax = plt.subplots(figsize=(9, 4))
evaluator_its.plot_its(df_its, ax=ax)
ax.set_title("ITS: book-wide re-rating, Q1 2023")
plt.tight_layout()
plt.savefig("its_counterfactual.png", dpi=150)
```

### ITS is weaker — the evaluator tells you this

ITS identification rests on a strong assumption: nothing else changed at period 9 other than the rate action. The evaluator says this plainly. When you fit ITS, it warns:

```
UserWarning: No control group found. Using ITS (Interrupted Time Series).
ITS identification is weaker than DiD — the causal estimate is valid only
if no concurrent events affected the outcome.
```

This is not defensive boilerplate. The Ogden rate, GIPP repricing, COVID, the 2022-Q3 claims inflation peak — all of these would contaminate an ITS estimate if the rate change happened to coincide with them. The library ships a `UK_INSURANCE_SHOCKS` dictionary and checks proximity automatically:

```python
from insurance_causal.rate_change._shocks import check_shock_proximity

msgs = check_shock_proximity("2022-Q1", proximity_quarters=2)
for m in msgs:
    print(m)

# Treatment period '2022-Q1' is within 2 quarters of UK market shock at 2022-Q1:
# GIPP (General Insurance Pricing Practices) — FCA PS21/5 effective January 2022.
# Causal estimates may be confounded by this concurrent event.
```

The current shock dictionary covers the Ogden changes (2017-Q1, 2019-Q3), COVID onset (2020-Q1, 2020-Q2), whiplash reform (2021-Q2), GIPP (2022-Q1), claims inflation peak (2022-Q3, 2023-Q1), and the FCA motor market study (2024-Q1). If your treatment period is within two quarters of any of these, you will see a warning in the summary. You cannot suppress the warning by pretending the shock did not happen.

---

## Minimum data requirements

ITS has hard floors: the estimator refuses to run with fewer than four pre-treatment periods and warns below eight. This is not arbitrary. With four pre-treatment quarters, you cannot reliably estimate a pre-trend, a seasonal pattern, and a slope change simultaneously. The segmented regression model has:

```
Y_t = β₀ + β₁·t + β₂·D_t + β₃·(t − T)·D_t + Σ γ_q·Q_q + ε_t
```

Four unknown parameters before seasonality is even added. Eight pre-treatment periods is the practical minimum for a credible estimate.

Standard errors use Newey-West HAC (heteroskedasticity and autocorrelation consistent), with lag length set to `floor(sqrt(n_periods))`. For a 20-period series that is 4 lags. Monthly data over two years gives you about 5 lags. The point is: quarterly insurance time series have autocorrelated residuals, and the standard errors reflect this.

For DiD, the practical minimum is 20 segments for cluster-robust standard errors. Below that threshold, the estimator switches to HC3 and warns you. HC3 is still valid but is less reliable when the number of clusters is small.

---

## What the evaluator does not do

We should say this plainly rather than bury it.

**No staggered adoption correction.** If different segments received the rate change at different times, the TWFE estimator the library uses can produce biased estimates — a known problem documented by Callaway and Sant'Anna (2021) and Goodman-Bacon (2021). The evaluator detects staggered adoption and raises a warning. If you have a staggered book, use `insurance-causal-policy` instead, which implements Callaway-Sant'Anna properly.

**No synthetic control.** The DiD here assumes you have a naturally occurring control group. If your control segments are not credibly comparable to the treated segments, the parallel trends test will fail and the ATT estimate will be biased. The `insurance-causal-policy` library's `SDIDEstimator` constructs a weighted synthetic control; that is the right tool when your control group is heterogeneous.

**No inference on ITS without adequate pre-periods.** Eight is not magic — it is a practical floor below which the pre-trend estimate is too noisy to be useful. If your book only has four quarters of history before the rate change, ITS will run but the output is not credible. The warning is there; take it seriously.

---

## Installation

```bash
pip install insurance-causal
```

Python 3.10 or later. Dependencies: pandas, numpy, scipy, statsmodels. The `rate_change` sub-package uses statsmodels WLS/OLS for estimation and Newey-West HAC for ITS standard errors — no optional extras required.

Source and issue tracker: [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal)

---

The point of this library is not that it is sophisticated. DiD and ITS are forty-year-old methods. The point is that they give you something a before/after chart cannot: a counterfactual. What would have happened to that segment's loss ratio if you had not changed the rates? That question is now answerable in about fifteen lines of Python, and the answer comes with a confidence interval and a pre-trend test.

---

## See also

- [Your Rate Change Didn't Prove Anything](/2026/03/13/your-rate-change-didnt-prove-anything/) — the conceptual argument for why before/after comparisons fail under FCA TR24/2, and how `insurance-causal-policy` handles the harder SDID case with staggered adoption
- [Causal Fixed Effects for Rate Change Evaluation](/2026/03/12/insurance-causal-panel/) — panel fixed effects as a complementary identification strategy
- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/15/causal-price-elasticity-tutorial/) — why the identification problems in rate change evaluation and price elasticity estimation share the same root
