---
layout: post
title: "Conformalised Quantile Regression: Prediction Intervals That Actually Adapt to Risk"
date: 2026-03-24
categories: [techniques]
tags: [conformal-prediction, quantile-regression, prediction-intervals, insurance-conformal, catboost, lightgbm, motor, reserving, reinsurance, uk-motor]
description: "ConformalisedQuantileRegression in insurance-conformal v0.6.2 gives you statistically guaranteed prediction intervals that are wide for high-risk segments and narrow for low-risk ones. Here is how to use it."
author: Burning Cost
---

Standard conformal prediction gives you a coverage guarantee. It does not give you intervals that are useful at the policy level. A 19-year-old in London gets the same ±£X as a 45-year-old in rural Norfolk because the conformal correction is a single scalar applied uniformly to the whole book. The intervals are valid in aggregate. For any individual risk they are uninformative.

Conformalised Quantile Regression fixes this. `ConformalisedQuantileRegression` in [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) v0.6.2 produces heteroscedastic intervals — wider where the data is genuinely more variable, narrower where it is stable — while retaining the marginal coverage guarantee.

```bash
uv add "insurance-conformal>=0.6.2"
```

---

## Why standard conformal is not enough

The basic split conformal approach wraps a point forecast. You compute nonconformity scores on a calibration set — typically Pearson residuals for Tweedie models — find the 90th percentile of those scores, and add it symmetrically to every prediction. The interval width is determined by a single number: that calibration quantile.

This produces correct marginal coverage. At the portfolio level, roughly 90% of observed losses fall inside the 90% interval. But at segment level the picture is different. High-variance segments — young drivers, commercial fleets, cat-exposed properties — get intervals that are too narrow. Low-variance segments get intervals that are too wide. The single correction term is borrowed from whichever segments dominate the calibration set.

The practical consequence: if you are using prediction intervals for reserve ranges, reinsurance attachment decisions, or large loss loading estimates, standard conformal intervals are anchored at the wrong width for the risks you actually care about.

---

## How CQR works

Romano, Patterson & Candès (NeurIPS 2019, [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)) observed that if you start from a quantile regression model rather than a point forecast, you get adaptivity for free. The idea is straightforward.

Train two models directly on quantile objectives: a lower quantile model `model_lo` at level α/2 and an upper quantile model `model_hi` at level 1 − α/2. For 90% intervals, those are the 5th and 95th percentile models.

On the calibration set, compute a nonconformity score per observation:

```
s_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
```

The score is negative when the observed loss sits inside the raw interval (well-covered) and positive when it falls outside (not covered). Take the (1 − α)(1 + 1/n) quantile of these scores across the calibration set to get a correction term q̂.

Final interval: `[q_lo(x) - q̂,  q_hi(x) + q̂]`

The correction q̂ adjusts for systematic miscoverage in the quantile models. If the models were perfectly calibrated, q̂ would be near zero and the raw quantile outputs would already achieve the target. In practice there is always some misspecification, and q̂ absorbs it. Crucially, the width `q_hi(x) - q_lo(x)` varies by risk because it comes from the quantile models — the correction q̂ shifts both bounds equally.

---

## Zero-inflation and motor claims

UK personal lines motor has a claim frequency somewhere between 5% and 15% depending on the book. Most policies have exactly zero annual claims. A Tweedie model with power p ≈ 1.5 handles this through the compound structure — Poisson frequency times Gamma severity — but the resulting intervals are always positive at both ends.

Quantile regression does not have this problem. For a low-risk policy, the 5th percentile of annual loss is £0. The lower quantile model learns to predict zero for the bottom of the distribution. CQR inherits this: `lower` in `predict_interval()` is clipped at 0.0 by default (`clip_lower=0.0`), so the interval `[0, upper]` naturally represents the distribution for a low-risk policy, while a young driver in London gets an interval `[£120, £1,800]`. Standard conformal would give both risks intervals of equal width centred on their respective point forecasts.

---

## The API

`ConformalisedQuantileRegression` takes two pre-fitted quantile models — one for the lower tail, one for the upper — and a calibration set. It works with any quantile objective: CatBoost `Quantile:alpha=`, LightGBM `objective="quantile"`, sklearn `GradientBoostingRegressor(loss="quantile")`.

```python
from catboost import CatBoostRegressor
from insurance_conformal import ConformalisedQuantileRegression

# Train quantile models on the training set (NOT the calibration set)
lo = CatBoostRegressor(
    loss_function="Quantile:alpha=0.05",
    iterations=500,
    depth=6,
    verbose=0,
)
hi = CatBoostRegressor(
    loss_function="Quantile:alpha=0.95",
    iterations=500,
    depth=6,
    verbose=0,
)
lo.fit(X_train, y_train)
hi.fit(X_train, y_train)

# Calibrate on a held-out set
cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
cqr.calibrate(X_cal, y_cal)

# Predict intervals on new data
intervals = cqr.predict_interval(X_test, alpha=0.10)
```

`predict_interval()` returns a Polars DataFrame with four columns: `lower`, `q_lo`, `q_hi`, `upper`. The `lower` and `upper` columns carry the coverage guarantee. The `q_lo` and `q_hi` columns are the raw quantile model outputs before the conformal correction — useful for diagnostics, which we will come to.

The data split matters. Train quantile models on years 1–4, calibrate on year 5, predict on year 6. Training and calibrating on the same data invalidates the coverage guarantee. For a temporal motor portfolio this is the natural split.

---

## The `correction_quantile` diagnostic

The first thing to check after calibration is whether your quantile models are doing useful work.

```python
q_hat = cqr.correction_quantile(alpha=0.10)
print(f"Conformal correction: £{q_hat:.0f}")
```

`correction_quantile()` returns the single scalar q̂ that gets added to and subtracted from every interval. It represents how much the raw quantile models were under- or over-covering.

A large positive q̂ — say, £300 on a motor book with a mean claim around £400 — means the quantile models are systematically missing. Either they were too aggressive in their quantile estimates during training, or there is distribution shift between train and calibration periods. A negative q̂ means the models over-covered: the raw intervals were already wider than needed.

More importantly: a large correction relative to the mean interval width tells you CQR is rescuing a poorly specified quantile model. The adaptivity is still real — the width `q_hi(x) - q_lo(x)` still varies by risk — but a large fixed correction term added uniformly across the book narrows the gap between CQR and standard conformal. If q̂ is larger than the mean of `(q_hi - q_lo)`, your quantile models need work before CQR can add much over basic conformal.

The correction is cached after first use (`cal_quantile_` attribute), so calling `correction_quantile()` at multiple alpha levels is cheap.

---

## Coverage by risk band

CQR gives marginal coverage — correct on average across the whole calibration set. It does not guarantee conditional coverage for any subgroup. To check whether the intervals are actually adaptive rather than secretly collapsing to fixed width:

```python
coverage_table = cqr.coverage_by_decile(X_test, y_test, alpha=0.10)
print(coverage_table)
```

`coverage_by_decile()` bins the test set into deciles of the midpoint prediction `(q_lo + q_hi) / 2` and reports coverage and mean interval width per decile. The output has columns `decile`, `mean_midpoint`, `mean_width`, `n_obs`, `coverage`, and `target_coverage`.

What you want to see: `mean_width` increasing monotonically from decile 1 to decile 10. If the interval width is roughly flat across deciles, your quantile models are not learning a useful spread signal — they are predicting approximately the same quantile gap everywhere, and the conformal correction is doing all the work. That makes CQR equivalent to standard conformal and you should ask whether the quantile models need more features or deeper trees.

What you do not want to see: `coverage` in decile 10 dropping well below 0.90. Systematic under-coverage in the highest-risk decile is the failure mode that matters most for reserve ranges and large loss loading. If that number is below 0.80 at a 90% target, there is a model specification problem in the upper quantile.

---

## Practical uses

**Reserve ranges.** A 90% prediction interval per policy at the individual risk level is directly usable as a reserve range in a development context. CQR intervals are wider for the risks where the range matters — attritional large losses, commercial lines where severity is volatile — and narrow for low-severity personal lines. Using standard conformal or a symmetric Tweedie interval gives you one width for the whole book.

**Reinsurance pricing.** The upper quantile `q_hi` and the corrected `upper` bound are natural inputs to excess-of-loss pricing. For a risk excess of £50,000, what matters is the probability mass above that threshold. CQR's upper quantile model is trained directly on the upper tail; the conformal correction ensures the threshold is calibrated to the actual data. Feed `intervals["upper"]` into your per-risk XL pricing and you have a statistically grounded attachment distribution rather than a parametric assumption.

**Large loss loading.** A standard safety loading formula loads a multiple of the standard deviation. With CQR you have something cleaner: the 95th percentile of the predictive interval is a direct estimate of the large loss threshold for that risk. For a commercial fleet policy with `upper = £28,000`, load the difference between the 95th percentile and expected loss. For a private car with `upper = £900`, the large loss loading is near zero. No distribution assumption required.

---

## LightGBM alternative

If you are using LightGBM, the setup is identical in structure:

```python
import lightgbm as lgb
from insurance_conformal import ConformalisedQuantileRegression

params_lo = {
    "objective": "quantile",
    "alpha": 0.05,
    "n_estimators": 500,
    "num_leaves": 63,
    "verbose": -1,
}
params_hi = {**params_lo, "alpha": 0.95}

lo = lgb.LGBMRegressor(**params_lo)
hi = lgb.LGBMRegressor(**params_hi)
lo.fit(X_train, y_train)
hi.fit(X_train, y_train)

cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
cqr.calibrate(X_cal, y_cal)
intervals = cqr.predict_interval(X_test, alpha=0.10)
```

The class accepts anything with a `predict(X)` method. Polars and pandas DataFrames and numpy arrays all work as inputs.

---

## What to watch for

Two failure modes are specific to insurance data.

**Quantile crossing.** If `q_lo > q_hi` for some predictions — the lower quantile model predicts more than the upper quantile model — your interval has negative width before the correction. This happens when the two models are trained independently and one overfits. The conformal correction will still produce a valid interval (`lower` will be capped at `clip_lower=0.0`), but the raw `q_lo` and `q_hi` columns will show the crossing. Check `(intervals["q_hi"] - intervals["q_lo"]).min()` after fitting. Any negative value means your quantile models need regularisation.

**Coverage at zero.** For a motor book where 90% of policies have no claims, the lower interval bound is £0 for almost everyone. Coverage statistics at low alpha levels will be dominated by this mass of zeros. Always run `coverage_by_decile()` restricted to non-zero claims if you want to understand interval quality for the severity component:

```python
mask = y_test > 0
coverage_nonzero = cqr.coverage_by_decile(X_test[mask], y_test[mask], alpha=0.10)
```

---

## The broader picture

`ConformalisedQuantileRegression` sits alongside the existing `InsuranceConformalPredictor` and `LocallyWeightedConformal` in the library. The choice depends on your base model.

If you already have a Tweedie or Poisson point forecast and do not want to train additional quantile models, `InsuranceConformalPredictor` with `nonconformity="pearson_weighted"` is lower overhead. If you have a distributional GBM producing a per-risk phi estimate, `LocallyWeightedConformal` is the right wrapper.

CQR is the right choice when you want interval width to be directly driven by the data distribution in each region of feature space, not by a variance function assumption. The quantile models learn the spread from the data; the conformal calibration makes the coverage guarantee rigorous. For high-stakes interval estimates — reserve ranges, attachment pricing, large loss loading on commercial lines — that combination is worth the cost of training two additional models.

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

- [Conformal Prediction Intervals for Insurance: Getting Coverage Guarantees from Tweedie Models](/techniques/2025/12/insurance-conformal-split-conformal-tweedie/) -- the original library post
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/03/04/per-risk-volatility-scoring-with-distributional-gbms/) -- related approach to heteroscedastic modelling
- [Large Loss Loading for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) -- practical context for the reinsurance and large loss use cases above
