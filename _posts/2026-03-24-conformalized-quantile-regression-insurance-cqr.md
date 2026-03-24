---
layout: post
title: "Conformalized Quantile Regression for Insurance: Adaptive Intervals That Know Which Risks Are Hard"
date: 2026-03-24
author: Burning Cost
categories: [conformal-prediction, techniques, libraries]
tags: [conformal-prediction, insurance-conformal, CQR, quantile-regression, prediction-intervals, heteroscedastic, tweedie, zero-inflation, catboost, lightgbm, python]
description: "Standard conformal prediction applies one correction to every risk. CQR applies different corrections to different risks — wider for young drivers and commercial fleets, narrower for standard motor. We shipped ConformalisedQuantileRegression in insurance-conformal v0.6.2."
---

Standard split conformal prediction has one correction term. You compute nonconformity scores on a calibration set, take the (1–α)(1 + 1/n) quantile of those scores, and add it symmetrically to every prediction. On a homogeneous book that is fine. On a UK motor portfolio where a young driver in a modified car and a retired teacher in a standard hatchback both sit in the same calibration set, it is wasteful at best and misleading at worst.

[insurance-conformal v0.6.2](https://github.com/burning-cost/insurance-conformal) ships `ConformalisedQuantileRegression` — CQR for short, from Romano, Patterson & Candès (NeurIPS 2019) — which replaces the single correction term with per-risk adaptive intervals. This post explains why that matters for insurance, how the algorithm works, and what the code looks like.

---

## The problem with a single correction term

Standard conformal prediction wraps a point forecast — say, a CatBoost Tweedie model. Its correction quantile is calibrated on residuals from that model across all risk types simultaneously. Because residuals from high-risk policies dominate (Var(Y) ∝ μ^p, so a commercial fleet policy with a £12,000 expected loss has roughly 50× more variance than a £400 standard motor risk at p=1.5), the calibration quantile is inflated.

The result: low-risk policies get wider intervals than they need. You are covering the entire heteroscedastic distribution with one number, and that number is set by the hardest tail.

Pearson-weighted conformal — which we use in `InsuranceConformalPredictor` — partly addresses this by normalising residuals by the Tweedie variance function before computing the calibration quantile. That corrects for heteroscedasticity in the nonconformity score. But the correction still comes from the point forecast: you are trusting the Tweedie variance function to accurately characterise how residuals scale with the mean.

CQR abandons the point forecast entirely. You train two quantile models directly — one for the lower tail, one for the upper — and conformal calibration corrects for any systematic miscoverage in those raw quantile outputs. The interval width is then a direct function of the spread between the two quantile model predictions at each individual risk.

---

## How CQR works

The algorithm is split conformal, applied to quantile model residuals rather than point forecast residuals.

**Training.** Fit two quantile models on your training data — one targeting the α/2 quantile (say, the 5th percentile) and one targeting the 1–α/2 quantile (the 95th percentile). Any sklearn-compatible quantile objective works: CatBoost `Quantile:alpha=`, LightGBM `objective="quantile"`, sklearn `GradientBoostingRegressor(loss="quantile")`.

**Calibration.** On a held-out calibration set, compute the CQR nonconformity score for each observation:

```
s_i = max(q_lo(x_i) − y_i,  y_i − q_hi(x_i))
```

This score is negative when y_i falls inside the raw interval — a well-covered point — and positive when it falls outside. Compute the (1–α)(1+1/n) quantile of these scores across the calibration set; call it q̂.

**Prediction.** For a new observation, construct:

```
[q_lo(x) − q̂,  q_hi(x) + q̂]
```

The conformal correction q̂ is a single scalar applied symmetrically — but because the raw quantile predictions q_lo(x) and q_hi(x) vary by risk, the final interval width varies too. Risks where the two quantile models disagree strongly get wide intervals. Risks where they agree closely get narrow ones.

**Why this gives better adaptation than pearson_weighted conformal.** Pearson-weighted conformal uses the Tweedie variance function to normalise residuals — it assumes variance scales as μ^p. CQR makes no such assumption. The quantile models learn the actual conditional distribution, including any non-Tweedie behaviour in the tail: heavy-tailed commercial risks, spatial clustering, convictions interaction effects. If the tails behave differently from what Tweedie predicts, the quantile models capture it.

---

## Why zero-inflation is handled naturally

A Tweedie model with p=1.5 fits both the probability mass at zero and the continuous severity simultaneously. The compound Poisson-Gamma interpretation forces a specific relationship between the zero mass and the severity distribution — one that may not hold in your data.

A quantile model has no such constraint. For a low-risk policy where the true 5th percentile of the loss distribution is zero, the lower quantile model learns to predict zero. The lower interval bound clips at zero anyway (`clip_lower=0.0` by default), and the upper interval reflects the severity tail. The calibration step then corrects for any systematic miscoverage.

You do not need to specify a Tweedie power, handle excess zeros separately, or add a two-part model component. The quantile models adapt.

---

## The code

```python
from catboost import CatBoostRegressor
from insurance_conformal import ConformalisedQuantileRegression

# Fit lower and upper quantile models on training data
# (X_train, y_train must be completely separate from calibration)
lo = CatBoostRegressor(
    loss_function="Quantile:alpha=0.05",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)
hi = CatBoostRegressor(
    loss_function="Quantile:alpha=0.95",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)

lo.fit(X_train, y_train)
hi.fit(X_train, y_train)

# Conformal calibration on held-out calibration set
cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
cqr.calibrate(X_cal, y_cal)

# Prediction — returns a Polars DataFrame
intervals = cqr.predict_interval(X_test, alpha=0.10)
# Columns: lower, q_lo, q_hi, upper
# lower/upper carry the coverage guarantee
# q_lo/q_hi are the raw quantile outputs (diagnostic)
```

The `q_lo` and `q_hi` columns are the raw quantile model predictions before the conformal correction. The difference `upper - lower` is the interval width; compare this across risk deciles to verify that adaptation is working.

The correction term itself is available as `cqr.correction_quantile(0.10)`. A positive value means the raw quantile models were undercovering — the conformal step widened them. If the correction is consistently large and positive, your quantile models may need more iterations or a different architecture.

---

## Checking coverage by risk decile

The coverage guarantee from CQR is marginal — correct on average, not per-segment. In practice, the adaptive nature of CQR means per-decile coverage is usually more uniform than with standard conformal, but you should verify rather than assume.

```python
diag = cqr.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
# columns: decile, mean_midpoint, mean_width, n_obs, coverage, target_coverage
```

If coverage in the top two deciles (highest mean midpoint, i.e. highest-risk policies) is below 90% while the bottom half runs at 95%+, you have the standard conformal failure mode: the calibration quantile is being set by the low-risk majority and is insufficient for the high-risk tail. CQR typically narrows this gap because the quantile models learn the high-risk tail directly, rather than relying on a single global correction.

---

## When to use CQR versus standard conformal

Use `InsuranceConformalPredictor` with `pearson_weighted` scoring when:
- You already have a Tweedie or Poisson point forecast
- Your portfolio is reasonably homogeneous (personal lines motor, standard risk mix)
- You need SCR upper bounds or premium sufficiency control from the same object

Use `ConformalisedQuantileRegression` when:
- Your book has genuinely heterogeneous tail behaviour — commercial lines mixed with personal, or a subportfolio where the severity tail departs from Tweedie
- Zero-inflation is a real problem and you want the lower bound to adapt to the zero mass directly
- You are already running quantile GBMs for pricing percentiles (common in casualty and Lloyd's market work) and want to add a coverage guarantee at minimal additional cost

The practical trade-off: CQR requires training two quantile models instead of one point forecast, which roughly doubles your GBM training cost. On 50,000 policies with CatBoost at 500 iterations, that is two minutes versus one. Not a constraint for quarterly rate cycles.

---

## Installation

```bash
uv add insurance-conformal

# With CatBoost:
uv add "insurance-conformal[catboost]"

# With LightGBM:
uv add "insurance-conformal[lightgbm]"
```

`ConformalisedQuantileRegression` is in scope from the package root:

```python
from insurance_conformal import ConformalisedQuantileRegression
```

---

## What CQR does not fix

CQR corrects interval width. It does not correct interval location. If your quantile models are systematically biased — trained on data that no longer represents your current portfolio — the conformal calibration will give you correctly-wide intervals centred on the wrong place.

Before using CQR outputs for capital allocation or reinsurance structuring, run `coverage_by_decile` on a recent holdout period and check that the interval midpoints make actuarial sense. The conformal guarantee tells you the width is right. It does not tell you the centre is right.

---

**References:**
- Romano, Patterson & Candès (2019). "Conformalized Quantile Regression." *NeurIPS 32.* [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
- [insurance-conformal on GitHub](https://github.com/burning-cost/insurance-conformal)
- [insurance-conformal v0.6.2 changelog](https://github.com/burning-cost/insurance-conformal/blob/main/CHANGELOG.md)

**Related:**
- [MAPIE vs insurance-conformal: Why Generic Conformal Prediction Breaks on Insurance Data](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) — the structural differences between standard conformal and insurance-specific scoring
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the foundational post on split conformal for insurance
- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/) — quantile regression for tail pricing in insurance
