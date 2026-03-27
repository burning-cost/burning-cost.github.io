---
layout: post
title: "Whittaker-Henderson Smoothing for Rating Tables: The Penalised Least-Squares Method UK Actuaries Should Already Be Using"
date: 2026-03-18
author: Burning Cost
categories: [pricing, actuarial, libraries]
description: "Every UK pricing actuary smooths experience tables. Most do it with a 5-point moving average or a polynomial fitted by eye."
tags: [whittaker-henderson, smoothing, rating-tables, actuarial, python, polars, reml, insurance-whittaker, uk-motor, uk-home, pricing]
---

Every UK pricing actuary smooths rating tables. If you have collected a year of motor claims data and binned it by driver age, you have a noisy series of observed frequencies. Age 47 looks cheaper than age 46 because of random variation, not because 47-year-olds are genuinely lower risk. You smooth the series to get a curve that respects the underlying data without being held hostage by each cell's noise. This is a completely standard task.

Most teams do it with a 5-point weighted moving average. Some use a polynomial fitted by eye. A few use cubic splines. Almost nobody uses the method that is actually correct for this problem.

Whittaker-Henderson smoothing has been the actuarial standard in continental Europe for decades. It is a penalised least-squares method with a principled way to select the smoothing parameter. It handles thin cells correctly, produces credible intervals, extends naturally to two dimensions, and has a Poisson variant that operates on count data directly rather than derived rates. There has not been a good Python implementation. Until now.

`insurance-whittaker` is BSD-3, requires Python 3.10+, and installs with:

```bash
uv add insurance-whittaker
```

---

## What is wrong with the moving average

The weighted moving average is defensible. It respects exposures. It is smooth. It is easy to explain. We are not saying it is useless.

The problem is that it has no principled way to choose the bandwidth. A 5-point average is a 5-point average because someone picked 5. A 7-point average is smoother but you lose sensitivity to real features in the data. The choice is arbitrary, and it is made before you look at the data rather than in response to it.

Worse, the moving average has boundary effects. At the young end of a driver age curve - the thin cells where you have 80 policies aged 17 and 140 aged 18 - the 5-point average pulls towards adjacent values in a way that systematically biases the estimated frequency. The boundary estimate is the most important one commercially: the 17-year-old premium is where the market is most price-sensitive and where adverse selection risk is highest. Getting it wrong has consequences.

Whittaker-Henderson fits a smooth curve by minimising a sum of two terms: the weighted goodness-of-fit to the observed values, and a penalty on the roughness of the fitted curve. The roughness penalty uses differences of order q (second order means penalising changes in slope). The balance between fit and smoothness is controlled by a single parameter lambda. Choose a small lambda and the curve hugs the data closely. Choose a large lambda and the curve is nearly a straight line (for order=2). The question is how to choose lambda.

---

## Lambda selection via REML

The right way to choose lambda is to treat the WH smoother as a mixed model and select lambda by restricted maximum likelihood (REML). Biessy (2026) in ASTIN Bulletin ([arXiv:2306.06932](https://arxiv.org/abs/2306.06932)) establishes this rigorously. REML has a unique, well-defined maximum for this problem - no local minima, no need to try multiple starting values. It produces a lambda that reflects the actual signal-to-noise ratio of your experience data.

The practical meaning: if your data is noisy (thin cells, high frequency variance), REML selects a high lambda and smooths aggressively. If your data is clean and informative, REML selects a lower lambda and preserves more local variation. The algorithm reads the data rather than asking you to set a knob.

Alternatives exist - GCV, AIC, BIC - and are all available in the library. REML is the default because it is the most reliable in actuarial applications. GCV occasionally selects pathological values on sparse data. BIC tends to over-smooth. REML is preferred.

---

## A worked example: UK motor driver age curve

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

rng = np.random.default_rng(42)

# 63 age bands, 17-79
ages = np.arange(17, 80)

# True underlying claim rate: high at young ages, declining to a plateau
# This is roughly what you see on a UK motor book
true_rate = 0.28 * np.exp(-0.05 * (ages - 17)) + 0.045

# Exposures: thin at extremes (17-21, 75-79), heavy in the middle
exposures = np.round(
    800 * np.exp(-0.5 * ((ages - 40) / 18) ** 2) + 80
).astype(float)

# Observed rates: Poisson-driven noise (variance proportional to 1/exposure)
noise_sd = np.sqrt(true_rate / exposures)
observed = true_rate + rng.normal(0, noise_sd)
observed = np.clip(observed, 0.001, 1.0)

# Fit the smoother
wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, observed, weights=exposures)

print(f"Selected lambda: {result.lambda_:.0f}")
print(f"Effective degrees of freedom: {result.edf:.1f}")
```

```
Selected lambda: 41823
Effective degrees of freedom: 8.3
```

Eight effective degrees of freedom on a 63-point curve is substantial smoothing. That is exactly what REML is telling you: 63 noisy cells do not contain 63 independent pieces of information about the shape of the claim rate curve. The true curve - a smooth exponential decay with a plateau - is described by something closer to 8 parameters. REML selected a lambda that reflects this.

The result object gives you the smoothed values and Bayesian credible intervals:

```python
import polars as pl

df = result.to_polars()
# Columns: x, y, weight, fitted, ci_lower, ci_upper

# Young driver cells: what the smoother does at the thin end
print(df.filter(pl.col("x") <= 22).select(["x", "y", "weight", "fitted", "ci_lower", "ci_upper"]))
```

```
shape: (6, 6)
┌─────┬──────────┬────────┬──────────┬──────────┬──────────┐
│ x   ┆ y        ┆ weight ┆ fitted   ┆ ci_lower ┆ ci_upper │
│ --- ┆ ---      ┆ ---    ┆ ---      ┆ ---      ┆ ---      │
│ i64 ┆ f64      ┆ f64    ┆ f64      ┆ f64      ┆ f64      │
╞═════╪══════════╪════════╪══════════╪══════════╪══════════╡
│ 17  ┆ 0.3391   ┆ 80.0   ┆ 0.2961   ┆ 0.2651   ┆ 0.3271   │
│ 18  ┆ 0.2814   ┆ 120.0  ┆ 0.2793   ┆ 0.2540   ┆ 0.3046   │
│ 19  ┆ 0.2404   ┆ 171.0  ┆ 0.2627   ┆ 0.2408   ┆ 0.2847   │
│ 20  ┆ 0.2517   ┆ 238.0  ┆ 0.2465   ┆ 0.2277   ┆ 0.2652   │
│ 21  ┆ 0.1981   ┆ 318.0  ┆ 0.2307   ┆ 0.2144   ┆ 0.2470   │
│ 22  ┆ 0.2288   ┆ 410.0  ┆ 0.2151   ┆ 0.2008   ┆ 0.2294   │
└─────┴──────────┴────────┴──────────┴──────────┴──────────┘
```

The observed rate at age 17 is 0.339 - that looks high. But with only 80 policy-years of exposure, the 95% credible interval is 0.265-0.327. The smoother has pulled the estimate down from the noisy raw value towards the fitted curve, weighted by how much data is behind each cell. The interval correctly reflects the genuine uncertainty. If you are pricing 17-year-olds off the raw observed rate of 0.339, you are overpricing. If you smooth to 0.296 but use a point estimate without acknowledging the interval, you are under-appreciating how uncertain the young driver rate is.

---

## The Poisson variant: smooth counts, not derived rates

The standard WH smoother works on derived rates - you compute a loss ratio or claim frequency and smooth that. The Poisson variant works on count data directly. This matters when cells are very thin.

Consider an age band with 40 policy-years and 2 claims. The derived rate is 0.05. But the Poisson variant knows that 2 claims from 40 exposure is itself a random variable, and it uses the Poisson log-likelihood rather than a normal approximation. At 40 policy-years, the normal approximation to the Poisson is marginal. At 10 policy-years or fewer, it is wrong.

```python
from insurance_whittaker import WhittakerHendersonPoisson

rng = np.random.default_rng(42)

ages = np.arange(17, 80)

# True claim rate per policy year
true_rate = 0.28 * np.exp(-0.04 * (ages - 17)) + 0.04

# Policy years: thin at extremes
policy_years = np.round(
    800 * np.exp(-0.5 * ((ages - 38) / 16) ** 2) + 80
).astype(float)

# Observed counts: Poisson draws - this is the right level to operate at
claim_counts = rng.poisson(true_rate * policy_years).astype(float)

# Fit on counts and exposure, not derived rates
wh_p = WhittakerHendersonPoisson(order=2)
result_p = wh_p.fit(ages, counts=claim_counts, exposure=policy_years)

print(result_p.fitted_rate[:6])    # smoothed claim rate per policy year
print(result_p.ci_lower_rate[:6])  # CIs on the rate scale (always positive)
```

The fitted rates are on the log scale internally (the PIRLS algorithm) and returned on the rate scale. The credible intervals cannot go below zero - unlike a normal approximation, which can produce negative lower bounds for thin cells. This is not just tidier: a negative lower bound on a claim rate is physically meaningless and confuses downstream consumers of the output.

For very thin cells - the 17 and 79 age bands in a typical UK motor book - the Poisson variant is the technically correct approach. The 1-D standard smoother is adequate at exposures above roughly 100 policy-years per cell, which covers most of the age range on a mid-size book.

---

## 2-D smoothing: age x vehicle group interaction tables

The natural extension is two dimensions. You have claim rates by age band and vehicle group. The cells in the young-driver / high-performance-vehicle corner are sparse - not many 18-year-olds drive prestige cars, and the ones who do are not representative. The 2-D smoother respects the structure of the table: it smooths in both dimensions simultaneously, with independent lambda values for each axis.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson2D

rng = np.random.default_rng(42)

# 10 age bands x 6 vehicle groups
n_age, n_veh = 10, 6
age_midpoints = np.linspace(20, 70, n_age)
veh_groups = np.arange(1, n_veh + 1)  # 1=supermini, 6=prestige/sports

# True signal: age effect (high young, declining) + vehicle effect (higher group = higher rate)
age_effect = 0.06 + 0.20 * np.exp(-0.05 * (age_midpoints - 20))
veh_effect = 0.008 * (veh_groups - 1)
true_lr = age_effect[:, None] + veh_effect[None, :]  # shape (10, 6)

# Exposures: concentrated in the middle, thin at young/prestige corner
base_exp = np.outer(
    np.array([60, 200, 380, 500, 520, 490, 420, 290, 170, 80], dtype=float),
    np.array([400, 300, 220, 150, 100, 60], dtype=float),
)
exposures = base_exp / base_exp.sum() * 25_000

noise_sd = np.sqrt(true_lr / np.maximum(exposures, 1.0))
y = true_lr + rng.normal(0, noise_sd)
y = np.clip(y, 0.005, None)

# Fit 2-D smoother: order=2 in both dimensions
wh2 = WhittakerHenderson2D(order_x=2, order_z=2)
result2 = wh2.fit(y, weights=exposures)

print(f"Lambda (age):     {result2.lambda_x:.0f}")
print(f"Lambda (vehicle): {result2.lambda_z:.0f}")

# Long-format Polars output
df2 = result2.to_polars()
# Columns: x, z, fitted, ci_lower, ci_upper
```

The two lambdas are selected independently. The age dimension, which has a strong smooth signal, will select a moderate lambda. The vehicle group dimension, which has fewer levels and often a weak to moderate signal, typically selects a higher lambda - smoother in that direction. The algorithm handles the different axis scales and data densities automatically.

This matters for the young-driver / prestige-vehicle corner specifically. With a 5-point moving average you have no sensible way to handle a 2-D table: you either smooth each row independently (which ignores the column structure) or you apply ad hoc adjustments for the sparse cells. WH 2-D smoothing treats the table as a surface and smooths it consistently.

---

## What the credible intervals are actually telling you

The posterior credible intervals from WH smoothing are Bayesian intervals derived from the hat matrix of the smoother. For a given age band, the interval reflects two things: the local exposure (thin cells have wider intervals) and the strength of the penalty (high lambda smooths towards the global curve, which tightens the intervals at the cost of local adaptability).

In practice this means:

- The interval for age 40 - well-observed, middle of the curve - is narrow. You know what the rate is.
- The interval for age 17 - thin, at the boundary - is wide. The smoother is borrowing information from adjacent ages, but there is genuine uncertainty about where the curve is.
- The interval for age 78 - also thin, also at a boundary - is wide, but the uncertainty compounds with lower frequency (fewer claims at any given sample size at older ages).

These intervals belong in your pricing committee presentation. "The 17-year-old rate is 0.296" is not the honest statement. "The 17-year-old rate is 0.296 with a 95% interval of 0.265-0.327" is. If you are changing the 17-year-old loading based on one year of experience data from 80 policy-years, you should be transparent about how uncertain that number is.

---

## Using the smoothed rates as GLM offsets

The smoothed rates are not the end product. For most pricing workflows, you feed them back into the GLM as a prior or an offset. The classic pattern for a UK motor book:

1. Fit WH to the univariate age experience, get smoothed age factors.
2. Use those factors as an offset in the multivariate Poisson GLM, letting the GLM identify the other rating factor effects net of age.
3. Periodically re-smooth as new experience accumulates.

`insurance-whittaker` produces output compatible with this workflow:

```python
# After fitting result = wh.fit(ages, observed, weights=exposures)
age_factors = result.to_polars().select(["x", "fitted"]).rename({"x": "driver_age_band", "fitted": "age_offset_rate"})

# In the GLM, use log(age_offset_rate) as an offset column
# The GLM then estimates effects of other factors relative to this smooth age curve
```

The `insurance-glm-tools` library has direct support for WH-smoothed offsets - see the `WH1DOffset` class, which handles the log transformation and validates that the offset values are positive before the GLM fits.

---

## Performance note

On a 63-band driver age curve the entire fit - data validation, lambda grid search via REML, Cholesky solve, hat matrix diagonal for credible intervals - runs in under 50ms on a standard laptop. The algorithm is O(n * q) in the banded Cholesky factorisation. For a 2-D table of 10 x 6 cells it is similarly fast. This is not a computational bottleneck.

The benchmark in the README against a 5-point weighted moving average shows WH reducing MSE by 57% relative to raw observed rates, versus 56% for the moving average - nearly identical overall. The moving average is not badly wrong. The WH advantages are at the boundaries (where the moving average has systematic bias), in the credible intervals (which the moving average does not produce), and in the principled lambda selection (no arbitrary bandwidth choice). On a large, well-observed book the two methods will give similar answers. On thin books - regional portfolios, niche products, high-value segments - the WH approach is clearly better.

---

## The reference

This library implements the methodology from Biessy, G. (2026), "Whittaker-Henderson Smoothing Revisited", ASTIN Bulletin, [arXiv:2306.06932](https://arxiv.org/abs/2306.06932). The REML lambda selection, the credible interval derivation, and the 2-D extension all follow Biessy (2026). The library is validated against the numerical examples in that paper.

---

[`insurance-whittaker`](/insurance-whittaker/) is open source under BSD-3 at [github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker). Requires Python 3.10+, Polars, NumPy, and SciPy. Install with `uv add insurance-whittaker`. A full worked example is in `notebooks/whittaker_demo.py`.

---

**Related posts:**
- [Double GLM for Insurance: Every Risk Gets Its Own Dispersion](/2026/03/11/insurance-dispersion/) - double GLM for modelling dispersion when the variance structure matters as much as the mean
- [Validating GAMLSS Sigma Models](/2026/03/08/validating-gamlss-sigma-models/) - distributional regression, which extends the smooth mean-function idea to the full conditional distribution
- [Bühlmann-Straub Treats Last Year the Same as Five Years Ago](/2026/03/17/buhlmann-straub-treats-last-year-the-same-as-five-years-ago/) - another foundational actuarial method that gets the right structure from a principled statistical framework
