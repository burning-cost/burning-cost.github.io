---
layout: post
title: "Separating Structural Inflation from the Underwriting Cycle with insurance-trend v0.1.5"
date: 2026-03-31
categories: [techniques, libraries]
tags: [insurance-trend, inflation, trend, Harvey, structural-time-series, kalman, statsmodels, UnobservedComponents, motor, pricing, reserving, python]
description: "The new InflationDecomposer in insurance-trend v0.1.5 uses the Harvey structural time series model to separate persistent cost inflation from cyclical effects. Using the raw trend to set a pricing assumption is a classification error, not just imprecision."
---

Every pricing review includes a conversation about claims inflation. Someone produces a chart of year-on-year severity changes — maybe with a CPI overlay — and the team picks a number. That number goes into the technical price.

The problem is that the series on that chart is not claims inflation. It is three or four different things stacked on top of each other: a structural cost trend, a position in the underwriting cycle, a seasonal pattern, and observation noise. Setting a single trend assumption from the raw series means embedding all of those components permanently into your price. The structural component should be embedded. The cyclical component should not. Getting this wrong in either direction costs money.

The Harvey structural time series model has been the principled solution to this problem since 1989. `insurance-trend` v0.1.5 adds `InflationDecomposer`, which wraps `statsmodels.tsa.statespace.UnobservedComponents` in an API that takes quarterly severity data in and hands back the decomposition. Sixty-two tests pass.

---

## What the raw series conflates

Take motor own damage severity over ten years. Vehicle repair costs have been running at roughly 4–6% per annum for structural reasons: ADAS recalibration, parts complexity, flat-rate labour shortages. That is genuine persistent inflation and it should flow through into a rate projection.

But over the same period there was a used car price spike in 2021–2022 (semiconductor shortage, post-lockdown demand) that pushed average write-off settlements up by 15–20% before reverting over 2023–2024. There was a post-COVID repair backlog that artificially depressed frequency and inflated severity per claim. There is a bodily injury market cycle running at roughly four to seven years, driven by credit hire volumes, litigation appetite, and insurer behaviour around reserving.

If you fit a straight line through the last eight quarters of that series, you are probably measuring the cyclical peak plus the structural trend. You set a 9% trend assumption when the right number for a forward-looking technical price is something closer to 5%. Then the cycle turns and your book is expensive relative to the market.

Conversely, if you measure through a cyclical trough you understate the structural component and underprice on the way up.

---

## The Harvey model

The Harvey (1989) local linear trend plus stochastic cycle model decomposes the observed series as:

```
y_t = mu_t + psi_t + seasonal_t + eps_t
```

where:
- `mu_t` is the structural level, evolving as a random walk with a time-varying drift `beta_t` (the slope)
- `psi_t` is the stochastic cycle — a mean-reverting oscillation with estimated period and damping factor
- `seasonal_t` is an optional calendar component
- `eps_t` is irregular noise

The slope `beta_t` is itself a random walk: `beta_t = beta_{t-1} + zeta_t`. This makes the structural trend stochastic rather than fixed — appropriate for insurance data where the underlying cost escalator genuinely shifts over time.

All variance parameters (`sigma^2_level`, `sigma^2_slope`, `sigma^2_cycle`, `sigma^2_irregular`) and the cycle frequency `lambda` are estimated by maximum likelihood via the Kalman filter. The Kalman smoother then extracts each component using the full-sample signal.

We work in log space (`log_transform=True` is the default). Insurance severity indices are log-normally distributed and the multiplicative structure in original space becomes additive in log space, which is what the state-space model requires.

---

## Using InflationDecomposer

The call for ten years of quarterly severity data. The simulated series here has a known 5% structural rate and a 5-year cycle:

```python
import numpy as np
from insurance_trend import InflationDecomposer

# Quarterly motor OD severity index, rebased to 100 at 2016Q1
# True structural: 5% pa. True cycle: 5-year period, amplitude 7%.
rng = np.random.default_rng(42)
n = 40  # ten years
t = np.arange(n, dtype=float)
structural = np.exp(0.05 / 4 * t)
cycle_comp = 0.07 * np.sin(2 * np.pi * t / 20 + np.pi / 3)  # phase offset
noise = rng.normal(0, 0.012, n)
severity_index = 100.0 * structural * np.exp(cycle_comp + noise)

periods = [f"{y}Q{q}" for y in range(2016, 2027) for q in range(1, 5)][:n]

decomposer = InflationDecomposer(
    series=severity_index,
    periods=periods,
    cycle=True,
    cycle_period_bounds=(3, 10),    # years; covers typical underwriting cycle
    periods_per_year=4,
    fit_kwargs={"method": "powell"}, # Powell is more reliable on short series
)
result = decomposer.fit()
print(result.summary())
```

```
=== Claims Inflation Decomposition (Harvey Structural Model) ===
Observations          : 40
Periods per year      : 4
Log-transformed input : True

--- Trend ---
Structural trend (pa) : 5.16%
Total trend (pa)      : 4.72%

--- Cycle ---
Estimated period      : 5.1 years
Current position      : +4.15% (above structural trend)

--- Model fit ---
AIC                   : -189.5
BIC                   : -180.0
Converged             : True
```

The structural rate (5.16%) is the number to put in your technical price. The current position (+4.15% above trend) is a reserving and cycle management signal — severity is running above the structural level right now, and if the model is right it will revert.

The total OLS trend is 4.72%: lower than the structural rate because the cycle happened to be in a depressed phase earlier in the sample, pulling down the full-period average. A pricing actuary using the simple OLS number would understate the persistent cost trend.

---

## Reading the output

**`structural_rate`** is the persistent annual inflation to embed in forward projections. It is derived from the mean of the Kalman-smoothed slope state `beta_t` over the full sample: `exp(mean(beta_t) * periods_per_year) - 1`. The mean rather than the final-period slope is deliberate — the end-of-sample Kalman state has higher uncertainty, and the mean is more representative of the typical underlying drift.

**`cyclical_position`** tells you where you are in the cycle right now. For log-transformed data: `exp(cycle[-1]) - 1`. A value of +0.04 means the latest observation is 4% above the structural trend due to cyclical effects. This is positive when the cycle is running hot (post-COVID repair backlog, credit hire peak) and negative in a soft market.

**`cycle_period`** is the estimated cycle length in years, derived from the fitted frequency parameter `lambda` (radians per period). For UK motor bodily injury you typically see 4–6 years. For motor OD the recent data suggests shorter cycles tied to used car price dynamics.

**`total_trend_rate`** is the simple OLS trend on the observed series — equivalent to what you would get from a log-linear fit. It combines structural and cyclical effects and is included for comparison. When it diverges significantly from `structural_rate`, the cycle is doing most of the work.

**The decomposition table** gives you the full time series for each component:

```python
df = result.decomposition_table()
# Columns: period, observed, trend, cycle, seasonal, irregular
# In log space: observed ≈ trend + cycle + seasonal + irregular
```

The reconstruction is exact up to floating-point precision. The irregular component is genuinely the residual, not a placeholder.

---

## When to use `cycle=True` vs `cycle=False`

Use `cycle=True` (the default) when you have at least 16 quarterly observations and believe there is a meaningful underwriting cycle in your series. For most UK personal lines — motor OD, motor BI, household — this is almost always true. The default bounds of `(2, 12)` years cover the plausible range; narrow them if you have a prior on the cycle length.

Use `cycle=False` for short series (fewer than 16 observations), or when you are fitting to a derived index that has already had cyclical effects removed (e.g., a frequency-normalised severity index). With `cycle=False`, the model reduces to a local linear trend — stochastic level plus slope — which still gives a better structural estimate than OLS but does not attempt to separate the cycle.

The hard minimums: 16 observations with `cycle=True`, 8 without. In both cases, 24+ is recommended and 40+ is where the MLE is genuinely reliable. Below 24 observations, treat the structural rate as indicative rather than precise.

```python
# Short series or already-adjusted index
decomposer_simple = InflationDecomposer(
    series=adjusted_index,
    periods=periods,
    cycle=False,       # local linear trend only
    periods_per_year=4,
)
```

---

## Adding seasonality

The `seasonal` parameter accepts the number of periods in a seasonal cycle. For quarterly data with annual seasonality:

```python
decomposer = InflationDecomposer(
    series=severity_index,
    periods=periods,
    cycle=True,
    seasonal=4,        # annual seasonality in quarterly data
    periods_per_year=4,
)
```

For most UK motor severity data, seasonal effects are real but small relative to the structural and cyclical components. Household is more seasonal (Q4 escape of water, Q1 subsidence). If you include `seasonal=4` and the estimated seasonal component is near zero, statsmodels will collapse its variance to a boundary solution — not an error, just the MLE confirming it is not needed for that series.

---

## How it fits with the rest of insurance-trend

`InflationDecomposer` addresses a different question from the other fitters in the library. `SeverityTrendFitter` and `LossCostTrendFitter` estimate the overall trend with bootstrap CIs and structural break detection — they answer "what was the trend over this window?". `InflationDecomposer` answers "how much of that trend is structural, how much is cyclical, and where are we in the cycle now?".

A natural workflow for a motor pricing review:

1. Run `LossCostTrendFitter` on the full severity history to get a single-number trend with CI and any detected structural breaks (COVID, Ogden rate changes).
2. Run `InflationDecomposer` on the post-break segment (or the full series if no break is detected) to decompose structural vs cyclical.
3. The `structural_rate` from step 2 goes into the technical price. The `cyclical_position` informs your cycle management view and large loss reserve loading.

You can also chain `InflationDecomposer` after `SeverityTrendFitter` with an external index. Deflate severity by the ONS HPTH parts cost index first (using `SeverityTrendFitter` with `external_index`), then run `InflationDecomposer` on the residual superimposed inflation series to understand whether the residual is structural or cyclical. That three-layer decomposition — economic inflation, structural superimposed inflation, cyclical — is closer to what the underlying loss cost is actually doing.

---

## A note on convergence

The Harvey model with a stochastic cycle has 4–5 variance parameters to estimate by MLE. L-BFGS-B (the default in statsmodels) can fail to converge on short or noisy series. If you see `Converged: False`, pass `fit_kwargs={"method": "powell"}` — Powell tends to be more robust on this class of problem. The parameter estimates from a non-converged fit are not necessarily wrong, but you should not trust the AIC/BIC comparisons.

Variance parameters collapsing to zero (a boundary solution where MLE decides one component has no variance) is common and not a problem. It means the data does not support that component. statsmodels still reports `Converged: True` in many boundary cases because the optimiser reached the requested tolerance.

---

## Install

```
uv add insurance-trend
```

`InflationDecomposer` requires `statsmodels>=0.14.5`, which has been a dependency of the library since v0.1.2.

The full source is at [github.com/actuarial-foss/insurance-trend](https://github.com/actuarial-foss/insurance-trend). The decomposer lives in `src/insurance_trend/inflation.py`.
