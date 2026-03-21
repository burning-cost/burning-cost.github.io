---
layout: post
title: "Your Year-End Large Loss Loading Is a Finger in the Air"
date: 2026-03-14
categories: [pricing, libraries, tutorials]
tags: [large-loss, quantile-regression, TVaR, ILF, catboost, insurance-quantile, motor, severity, tail-risk, python, polars]
description: "December is the season for year-end rate reviews where someone adds a flat 8% large loss loading to every segment regardless of tail weight."
---

December is the season for year-end rate reviews. Boards want sign-off, reinsurance treaties are being re-priced, and pricing actuaries everywhere are reaching for the same tool: a flat large loss loading applied to the burning cost mean. Pick a number, 6%, maybe 8%, argue about it for half an hour, document it as "consistent with historical average severity inflation" and move on.

The problem is not that this approach is dishonest. The problem is that it is the same number for your lowest-risk segment and your highest-risk one. A 25-year-old driver in an ABI group 40 vehicle and a 55-year-old driver in an ABI group 10 vehicle do not have the same tail distribution. The flat loading is not conservative for the first and not generous for the second. It is just wrong for both of them in opposite directions.

What you actually need is a per-segment large loss loading: the expected excess of the tail over the mean, computed from a model that has learned the conditional tail distribution from the data. That is what `insurance-quantile` gives you.

```bash
uv add insurance-quantile
# or
uv add insurance-quantile
```

---

## The mechanics of a large loss loading

Define the loading precisely:

```
loading_i = TVaR_alpha(i) - E[Y_i]
```

`TVaR_alpha(i)` is the Tail Value at Risk for risk `i` at confidence level `alpha`: the expected loss given it exceeds its alpha-th percentile. `E[Y_i]` is the mean loss prediction from a standard Tweedie or Gamma severity model. The loading is additive. It represents how much extra the rate should be above the pure burning cost to cover large claims with adequate confidence.

For personal lines motor, alpha=0.95 is standard. For commercial lines or any book where aggregate stop-loss treaties attach at high quantiles, alpha=0.99 is more appropriate.

The traditional approach estimates `TVaR_alpha` implicitly by applying a flat percentage to the mean. This assumes every risk has the same ratio of TVaR to mean, which is the same as assuming every risk has the same tail shape. That is the assumption we are relaxing.

---

## Why a Tweedie model cannot give you this

A Tweedie GBM is correctly specified for `E[Y | X]`. It tells you the expected loss per risk given its features. What it cannot tell you is how much of that expected loss sits in the tail versus the body of the distribution.

Two risks with identical Tweedie predictions can have completely different tail behaviour if their feature values imply different severity dispersion. A high-powered vehicle in an area with higher proportion of commercial traffic generates BI claims differently than a typical private car. The tail weight is genuinely different. The Tweedie model's variance-power is a single global hyperparameter. It cannot represent heteroskedastic tail weights.

Quantile regression directly estimates `Q(alpha | X)` for each risk separately. CatBoost's `MultiQuantile` loss trains all quantile levels in a single model with shared feature representations, which is efficient and prevents the kind of crossing that occurs when you fit separate models per quantile.

---

## Fitting the model

The workflow requires two models: a standard Tweedie model for the mean, and a `QuantileGBM` for the quantiles. Both are fitted on non-zero claims (severity model, not full frequency-severity), which is the right approach for large loss loading.

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_quantile import QuantileGBM, large_loss_loading, per_risk_tvar

# Fit a Tweedie model for the mean prediction
# X_train and y_train are pl.DataFrame and pl.Series of non-zero claim amounts
tweedie = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    verbose=0,
)
tweedie.fit(X_train.to_numpy(), y_train.to_numpy())

# Fit the quantile GBM
# Include 0.95 and 0.99 at minimum for TVaR at alpha=0.95 to be well-estimated
model = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    fix_crossing=True,
    iterations=500,
)
model.fit(X_train, y_train)
```

`fix_crossing=True` applies per-row isotonic regression at prediction time. CatBoost's `MultiQuantile` loss can produce crossing predictions for individual risks despite enforcing correct orderings in the loss function. The fix is essentially free computationally.

---

## Computing the loading

```python
# Per-risk TVaR and loading on the validation set
tvar_result = per_risk_tvar(model, X_val, alpha=0.95)

loading = large_loss_loading(
    model_mean=tweedie,
    model_quantile=model,
    X=X_val,
    alpha=0.95,
)

# loading is a Polars Series of additive per-risk loadings
# Attach to your rating output
output = X_val.with_columns([
    tvar_result.values.alias("tvar_95"),
    tvar_result.var_values.alias("var_95"),
    loading.alias("large_loss_loading"),
])
```

You can now look at the loading distribution by segment. The key question is whether high-risk segments are genuinely getting higher loadings than low-risk ones. If your flat-percentage approach was applying 7% to everyone, but the quantile model gives you 4% for low-risk and 14% for high-risk, you have found something worth acting on.

---

## Increased Limits Factors

ILFs arise in commercial lines and high-limit personal lines where the same risk may be written at different policy limits. The ILF from limit L1 to limit L2 is:

```
ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)]
```

This is the factor you multiply the L1 rate by to price coverage at L2. The library estimates it by numerically integrating the survival function derived from the quantile predictions:

```python
from insurance_quantile import ilf

# For a model fitted with quantiles including 0.99 or higher,
# compute the per-risk ILF from £100k to £500k
ilf_values = ilf(
    model=model,
    X=X_val,
    basic_limit=100_000,
    higher_limit=500_000,
    n_integration_points=200,
)
# ilf_values: per-risk Polars Series
# A value of 1.4 means the £500k limit costs 40% more than the £100k limit
```

The key thing here is that `ilf()` returns a per-risk factor, not a single table lookup. A risk with a heavier predicted tail gets a larger ILF for the same limit pair than a risk with a light tail. This is what ILF tables from industry aggregate data cannot give you: the ILF conditioned on this specific risk's features.

For best results at high limits, include 0.99 and 0.995 in your quantile levels when fitting. The ILF is integrating the survival function over the interval [L1, L2], and that integral is sensitive to how well the far tail is estimated.

---

## What the output looks like

Run this on a real motor book and you will typically see:

- Mean loading across the portfolio close to the flat percentage you were using before (otherwise the total premium volume changes, which is a separate decision)
- Significant spread around that mean: high-risk segments at 1.5-2x the portfolio mean loading, low-risk segments below it
- A strong correlation between loading magnitude and features associated with bodily injury severity: age of third party, vehicle power, area

That spread is the information you were throwing away with the flat rate. In a competitive market, correctly loading the high-risk tail and releasing rate on the low-risk tail is a retention and selection improvement, not just an accuracy exercise.

---

## When not to use this

The README is explicit about this and we agree with it: if you have only a few hundred large claims in the training window, the tail quantiles are estimated from a very small number of data points regardless of method. A parametric Gamma's regularisation via a global shape parameter may genuinely help in that regime.

Also: if your actuarial deliverable requires a smooth, monotone ILF curve by limit, quantile regression is not constrained to be monotone in the limit dimension without additional work. For that use case, a parametric severity distribution (Pareto, Generalised Pareto, or a Pareto-Gamma composite) is more appropriate. The library's `insurance-distributional` sibling handles that.

For a book with several thousand large claims across a few years of development, and genuine heteroskedasticity in the tail by risk segment, `insurance-quantile` is the right tool.

---

## Year-end checklist

If you are running a December rate review, the questions to ask are:

1. Is your large loss loading the same number for all segments, or does it vary by the drivers of tail severity?
2. Have you looked at whether your highest-risk segments are generating disproportionately large claims, or just more of them? Frequency and severity tail behaviour are not the same thing.
3. If your book has changed mix year-on-year, has the appropriate large loss loading changed with it? A flat percentage applied to a Tweedie mean is stable by construction, which is not the same as being right.

The code to replace the flat loading with per-segment TVaR-derived loadings is about 30 lines. The modelling infrastructure takes a day. The actuarial argument for why the result is defensible is straightforward: it is the mean of the conditional tail distribution, estimated from a model that has been calibrated on your data.

That is a better answer than "we used 7%, consistent with last year."

---

`insurance-quantile` is open source under MIT at [github.com/burning-cost/insurance-quantile](https://github.com/burning-cost/insurance-quantile). Requires Python 3.10+, CatBoost 1.2+, and Polars 1.0+.

- [Your Model Validation Is a Checklist, Not a Test](/2026/03/11/model-validation-pra-ss123/) — building a defensible sign-off process alongside the technical output
- [Your Severity Model Assumes the Wrong Variance](/2026/03/10/your-severity-model-assumes-the-wrong-variance/) — why the Tweedie variance function matters and when to use a Gamma or inverse Gaussian instead
- [Spliced Severity Distributions: When One Distribution Is Not Enough](/2026/03/06/spliced-severity-distributions-when-one-distribution-isnt-enough/) — parametric alternative for closed-form ILFs
