---
layout: post
title: "Composite Severity Regression: Getting the Tail Right Without Throwing Away the Body"
date: 2026-03-23
categories: [libraries, pricing, severity]
tags: [composite-models, spliced-distributions, severity, heavy-tail, GPD, Burr, mode-matching, ILF, motor-BI, EL, reinsurance, python, insurance-severity, lognormal, gamma, MLE]
description: "A Gamma GLM systematically underestimates large motor BI claims because no single parametric family fits both the £5k bulk of claims and the £500k+ tail. insurance-severity implements spliced severity models — lognormal or gamma body below a threshold, GPD or Burr XII above — with the threshold varying by policyholder covariates via mode-matching. 2,818 lines, 106 tests."
---

Every UK motor BI pricing model has the same structural problem. The Gamma GLM fits the body of the claim distribution well — the £3k–50k range where 90% of claims live. Then a £400k claim appears, and the Gamma's thin tail gives it a probability mass that is off by an order of magnitude. You know this, so you apply an aggregate large loss loading, maybe 15% or 18%, derived from some tail factor that your predecessor calibrated three years ago on a different book of business. The loading does not condition on vehicle group, driver age, or region. It is the same for a young driver in a modified VW as for a fleet manager in a mid-range saloon.

This is how large loss misallocation compounds. The young driver is underpriced at high limits. The fleet account is overpriced. The aggregate is approximately right, but the cross-subsidy is not visible until you lose the fleet account to a competitor who priced it better.

[`insurance-severity`](https://github.com/burning-cost/insurance-severity) takes a different approach: fit a composite (spliced) severity model that treats the body and tail as genuinely different distributions, connected at a threshold. The body is lognormal or gamma. The tail is GPD or Burr XII. The threshold can vary by policyholder. 2,818 lines of source, 106 tests passing. MIT-licensed, on PyPI.

```bash
uv add insurance-severity
```

---

## The structural problem with single-family severity models

UK motor BI claim severity has a tail index (ξ in GPD notation) somewhere in the range 0.1 to 0.5. A Gamma distribution has a tail index of effectively zero — it is exponentially bounded. Fitting a Gamma to the full claim distribution is a model error, not a parameter error. No matter how many thousands of claims you use to fit it, the Gamma will never reproduce the frequency of £200k+ BI claims because its tail decays too fast. The same applies to the lognormal, which is heavier-tailed than the Gamma but still materially thinner than the empirical distribution at high quantiles.

The actuarial response to this has traditionally been to accept the GLM for the body, then apply an Increased Limit Factor (ILF) schedule derived from a separate heavy-tailed fit to the large loss experience. ILF schedules are useful, but they have a problem: they are aggregate. The ILF at the £250k limit is the same for every risk, or at best varies by rating group with a coarse classification. It does not respond to the continuous covariate structure of your pricing model.

A composite model solves this by making the body-to-tail transition a formal part of the statistical model, estimated jointly with the regression structure.

---

## How composite models work

The density of a composite (spliced) severity model is:

```
f(y) = π * f_body*(y)    for 0 < y ≤ t
f(y) = (1-π) * f_tail*(y)  for y > t
```

where f_body* is the body density truncated above at threshold t, f_tail* is the tail density truncated below at t, and π is the probability mass assigned to the body. The truncated densities integrate to 1 over their respective domains, so the composite integrates to 1.

Three things need to be estimated: the body parameters, the tail parameters, and the threshold t (and π follows from the others).

The naive approach is to pick t from a mean excess plot, then fit body and tail independently. This works, but it gives you a fixed t for the whole portfolio. A young driver in a high-NCD band does not have the same severity profile as a commercial vehicle operator; the transition from "routine claim" to "large claim" should plausibly happen at different absolute amounts for different risks.

The mode-matching approach makes t covariate-dependent. You set the threshold equal to the mode of the tail distribution. Since the tail scale parameter varies with covariates via a log-link regression, the mode — and therefore the threshold — also varies with covariates. Each policyholder gets their own splice point.

---

## The GPD incompatibility

There is a constraint that is not obvious from reading the literature and is not documented in any of the R packages that implement composite models.

GPD with shape parameter ξ ≥ 0 has its mode at zero. Not at some small positive value — at exactly zero. The GPD density for ξ > 0 is a strictly decreasing function from the threshold. For ξ = 0 (exponential case) it is flat at the threshold and then decreases. There is no interior mode.

Mode-matching requires setting the threshold equal to the tail mode. If the tail mode is zero, the threshold is zero, which means everything is in the tail and the body is empty. The model degenerates.

For insurance large losses, ξ is almost always positive. UK motor BI claims fitted above a £100k threshold typically give ξ estimates around 0.15–0.40. GPD is the right distribution for the tail in the EVT sense, but it cannot be used with mode-matching.

The Burr XII distribution does have a positive interior mode, provided its δ parameter exceeds 1:

```
mode_Burr(α, δ, β) = β * [(δ-1)/(α*δ+1)]^{1/δ}    for δ > 1
```

This is positive and finite for any β, any α > 0, and δ > 1. The library enforces δ > 1 via reparameterisation (fitting log(δ-1) rather than log(δ)) so the constraint is never violated during optimisation.

The practical rule: use `LognormalBurrComposite` with `threshold_method="mode_matching"` for covariate-dependent thresholds. Use `LognormalGPDComposite` or `GammaGPDComposite` with a fixed or profile-likelihood threshold when GPD is the right EVT choice.

---

## Basic usage

### Lognormal-Burr with mode-matching

The main model for covariate-dependent threshold pricing:

```python
import numpy as np
from insurance_severity.composite import LognormalBurrComposite

# y: 1-D array of positive claim amounts (e.g. motor BI in £)
model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=5)
model.fit(y)

print(model.summary(y))
# LognormalBurrComposite
#   Threshold method : mode_matching
#   Threshold        : 87,423.1
#   Body weight (pi) : 0.8812
#   n_body           : 44,059
#   n_tail           : 5,941
#   Body params      : [mu=9.21, sigma=1.43]
#   Tail params      : [alpha=2.34, delta=1.81, beta=95200]
#   Log-likelihood   : -538,241.33
#   AIC              : 1,076,492.66

# Value at Risk and TVaR
print(f"VaR 99.5%  : £{model.var(0.995):,.0f}")
print(f"TVaR 99.5% : £{model.tvar(0.995):,.0f}")

# Randomized quantile residuals — should look N(0,1)
resid = model.quantile_residuals(y)
```

The threshold of £87k emerged from the data via mode-matching — we did not specify it. The body weight of 0.881 means about 88% of claims by count are in the body. Both are outputs, not inputs.

### Lognormal-GPD with a fixed threshold

When you have a natural threshold from the treaty structure — the XL attachment at £250k, or the deductible at £100k — use fixed threshold with GPD:

```python
from insurance_severity.composite import LognormalGPDComposite

model = LognormalGPDComposite(
    threshold=250_000,
    threshold_method="fixed",
)
model.fit(y)

# GPD shape and scale
xi, sigma = model.tail_params_
print(f"GPD ξ={xi:.3f}, σ={sigma:,.0f}")

# ILF at £500k and £1m basic limit £250k
print(f"ILF(£500k)  : {model.ilf(500_000, basic_limit=250_000):.4f}")
print(f"ILF(£1m)    : {model.ilf(1_000_000, basic_limit=250_000):.4f}")
print(f"ILF(£5m)    : {model.ilf(5_000_000, basic_limit=250_000):.4f}")
```

Or let the data choose the threshold:

```python
model = LognormalGPDComposite(
    threshold_method="profile_likelihood",
    n_threshold_grid=60,
    threshold_quantile_range=(0.75, 0.97),
)
model.fit(y)
print(f"Profile-selected threshold: £{model.threshold_:,.0f}")
```

The profile likelihood grid evaluates the log-likelihood at each candidate threshold with body and tail fitted independently, then picks the maximum. It is more expensive than fixed (one pass versus 60) but gives you a data-driven threshold without assuming mode-matching.

---

## Regression with covariate-dependent thresholds

`CompositeSeverityRegressor` wraps any composite model and adds covariates to the tail scale parameter via a log-link. For mode-matching models, this automatically makes the threshold covariate-dependent:

```
log(β_i) = X_i @ w          (Burr scale, per policyholder)
t_i = β_i * h(α, δ)          (threshold, per policyholder)
μ_i = σ² + log(t_i)          (lognormal mu, per policyholder)
```

where h(α, δ) is a shape-parameter-only function. The threshold is proportional to exp(linear predictor). This is the Liu, Li & Shi (Insurance: Mathematics and Economics, 2024) framework implemented in Python.

```python
import pandas as pd
from insurance_severity.composite import CompositeSeverityRegressor, LognormalBurrComposite

# X_train: DataFrame with covariates
# Columns here: vehicle_group (0/1 encoded), driver_age_std, region_north
reg = CompositeSeverityRegressor(
    composite=LognormalBurrComposite(threshold_method="mode_matching"),
    feature_cols=["vehicle_group", "driver_age_std", "region_north"],
    n_starts=3,
)
reg.fit(X_train, y_train)

print(reg.summary())
# CompositeSeverityRegressor
#   Base composite    : LognormalBurrComposite
#   Mode-matching     : True
#   Intercept (log)   : 10.8742
#   Coefficients      : [ 0.421  -0.183   0.097]
#   Shape params      : [alpha=2.31, delta=1.79]
#   Body params       : [nan, sigma=1.41]
#   Mean pi           : 0.8794
#   Log-likelihood    : -537,188.22

# Per-policyholder thresholds
thresholds = reg.predict_thresholds(X_test)
print(f"Threshold range: £{thresholds.min():,.0f} – £{thresholds.max():,.0f}")

# Expected severity per policyholder
predicted_means = reg.predict(X_test)
```

The regression coefficient of 0.421 on vehicle_group means the tail scale parameter (and therefore the threshold) is exp(0.421) = 1.52× higher for that vehicle group — the point at which claims transition from "routine" to "large" is 52% higher in absolute terms. This is the covariate-dependent threshold in action.

For GPD regression with a fixed global threshold:

```python
from insurance_severity.composite import CompositeSeverityRegressor, LognormalGPDComposite

reg_gpd = CompositeSeverityRegressor(
    composite=LognormalGPDComposite(
        threshold=250_000,
        threshold_method="fixed",
    ),
    feature_cols=["vehicle_group", "driver_age_std"],
)
reg_gpd.fit(X_train, y_train)
```

Here the threshold is shared; only the GPD scale σ_i varies with covariates. The GPD shape ξ is estimated globally.

---

## ILF schedules by policyholder

The regression model can compute an ILF schedule per observation — each policyholder gets their own set of increased limit factors derived from their individual body and tail fit:

```python
limits = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
ilf_matrix = reg.compute_ilf(X_test, limits=limits, basic_limit=1_000_000)

# ilf_matrix shape: (n_policies, 6)
# Row i: ILF schedule for policyholder i
import pandas as pd
ilf_df = pd.DataFrame(
    ilf_matrix,
    columns=[f"ILF_{int(l/1000)}k" for l in limits],
)
print(ilf_df.head())
#    ILF_100k  ILF_250k  ILF_500k  ILF_1000k  ILF_2000k  ILF_5000k
# 0    0.8234    0.9471    0.9821     1.0000     1.0241     1.0587
# 1    0.7912    0.9318    0.9779     1.0000     1.0289     1.0661
```

The ILF at the basic limit is by definition 1.000. Below the basic limit the ILF is less than 1 (limited coverage). Above the basic limit the ILF grows, with the rate of growth determined by the tail distribution. Policyholders with different tail-scale coefficients will have materially different ILF curves at high limits — exactly what you want when pricing excess layers.

---

## Diagnostics

### Mean excess plot

Before fitting, check whether a composite structure is warranted. A mean excess plot that curves upward at some threshold and then flattens (or continues linearly) is evidence of a heavy tail:

```python
from insurance_severity.composite import LognormalBurrComposite

model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(y)

# Plot: fitted density vs empirical histogram, log y-axis
model.plot_fit(y, log_scale=True)
```

### Quantile residuals

After fitting, check model adequacy with randomised quantile residuals (Dunn & Smyth, 1996). For a correctly specified model, these should look like draws from N(0,1):

```python
resid = model.quantile_residuals(y)

# Check: should be approximately standard normal
import numpy as np
print(f"Mean: {resid.mean():.3f} (expect ~0)")
print(f"Std : {resid.std():.3f} (expect ~1)")
print(f"Skew: {((resid - resid.mean())**3).mean() / resid.std()**3:.3f}")

# If you have matplotlib:
import matplotlib.pyplot as plt
from scipy.stats import probplot
probplot(resid, plot=plt)
plt.title("Quantile residuals — composite model")
plt.show()
```

Heavy tails in the residuals above the threshold indicate the tail distribution is not capturing enough mass. If you see this with Burr, try comparing AIC against `LognormalGPDComposite` with profile likelihood threshold — sometimes the GPD is genuinely a better fit even without mode-matching.

### AIC/BIC comparison

```python
from insurance_severity.composite import (
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
)

models = {
    "LN-Burr (mode-match)": LognormalBurrComposite(threshold_method="mode_matching"),
    "LN-GPD (profile)": LognormalGPDComposite(threshold_method="profile_likelihood"),
    "Gamma-GPD (profile)": GammaGPDComposite(threshold_method="profile_likelihood"),
}

for name, m in models.items():
    m.fit(y)
    print(f"{name:25s}  AIC={m.aic(y):,.0f}  BIC={m.bic(y):,.0f}  t={m.threshold_:,.0f}")
```

---

## Which model to use

**`LognormalBurrComposite` with `threshold_method="mode_matching"`** is the default choice for UK motor BI pricing with covariates. Use it when you want per-policyholder thresholds, when you have no strong prior on where the body-tail transition should occur, and when you want ILF schedules conditioned on risk characteristics. The Burr XII tail is not GPD, but it is flexible enough for ξ in the range typical of UK motor BI, and it supports mode-matching where GPD cannot.

**`LognormalGPDComposite` with `threshold_method="fixed"`** is correct when the threshold is determined by the contract structure — reinsurance attachment points, deductibles, policy limits. PI/D&O pricing where the deductible is the natural splice point is the canonical use case. GPD is the EVT-motivated choice above a threshold; use it where the threshold is not in question.

**`LognormalGPDComposite` with `threshold_method="profile_likelihood"`** is the data-driven version of the above, when the threshold is not specified by contract and you want to use GPD. The profile likelihood grid search is slower (roughly proportional to `n_threshold_grid` fits) but avoids the Burr approximation. Use it for EL or PL books where the threshold location is uncertain and you prefer the GPD's EVT interpretation.

**`GammaGPDComposite`** is the natural choice if you want continuity with a Gamma GLM for the body. The Gamma GLM frequency × severity decomposition produces Gamma-distributed severities. Combining a Gamma body with a GPD tail gives you a model where the body is exactly what your GLM assumes, with a principled heavy-tailed extension above the threshold.

**`CompositeSeverityRegressor` wrapping `LognormalBurrComposite`** is the full covariate-dependent threshold model. It requires more data than the standalone models — you need enough tail observations across covariate cells to identify the tail scale regression. As a rough guide, you want at least 500–1,000 observations above the modal threshold; below that, the body/tail regression coefficients are poorly identified. Bootstrap confidence intervals (`reg.bootstrap_ci(X, y, n_bootstrap=200)`) tell you how much to trust the covariate effects.

---

## Implementation notes

A few things worth knowing before deploying this in production.

**Individual-varying thresholds during optimisation.** For the regression model with mode-matching, each gradient evaluation reclassifies observations between body and tail based on the current threshold estimates. With 100,000 policies this is a vectorised NumPy operation and is fast. The optimizer (L-BFGS-B) typically converges in 80–200 iterations for well-scaled data.

**Multi-start initialisation.** The composite log-likelihood can be multimodal, especially when the threshold is poorly identified by the data. The library defaults to 5 starts for standalone models and 3 for regression models. If you get a poor fit — threshold at the edge of the grid, near-zero tail weight — increase `n_starts`. The warm start (fitting the standalone model first, then using its parameters to initialise the regression) materially reduces the starting-point sensitivity.

**Normalization terms.** The body density is a truncated lognormal or truncated gamma, integrating to 1 over [0, t]. The tail density is the corresponding truncated heavy-tailed distribution, integrating to 1 over (t, ∞). Both normalisation constants appear in the log-likelihood. A common error in composite model implementations is to omit the normalisation — this produces a biased likelihood and biased parameter estimates. The library includes them throughout.

**Tail sample size.** GPD parameter estimates are unreliable below about 30 exceedances. The library emits a `UserWarning` if you have fewer than 30 observations above the threshold. For small books, use a low threshold (high quantile, not high absolute value) to get enough tail data, or consider pooling across years before fitting.

---

## Academic context

The composite framework implemented here draws primarily from two papers:

Liu, Li & Shi (Insurance: Mathematics and Economics, 2024) established the varying-threshold GBII composite regression framework. The covariate-dependent threshold via mode-matching, the reparameterisation to handle shape constraints, and the warm-start initialisation strategy from their paper are all present in the library's regression implementation.

Fung, Li et al. (arXiv:2208.01262, 2022) developed the Lognormal-T composite regression models with varying threshold. The lognormal body with Burr XII tail variant — the most practically useful for actuarial work — is the model in `LognormalBurrComposite`.

Neither group has published a Python implementation. The R packages (evmix, ReIns, CompLognormal) cover univariate fitting only — no regression, no covariate-dependent threshold. The Python gap is total.

---

**[insurance-severity on GitHub](https://github.com/burning-cost/insurance-severity)** — 106 tests, MIT-licensed, PyPI. Library #47.
