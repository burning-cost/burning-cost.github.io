---
layout: post
title: "scikit-learn TweedieRegressor vs insurance-distributional: Why Fixed Tweedie Isn't Enough for Insurance Pricing"
date: 2026-03-22
author: Burning Cost
categories: [distributional, glm, libraries, comparisons]
description: "sklearn's TweedieRegressor is a well-engineered GLM. It fits a fixed-power Tweedie model correctly. The problem is that insurance pricing needs per-risk variance, not a single portfolio-level dispersion parameter - and that is not what it was built to do."
tags: [tweedie-regression-python, sklearn-tweedieregressor, tweedie-insurance-pricing, distributional-regression, GAMLSS, insurance-distributional, per-risk-volatility, heteroscedasticity-insurance]
---

scikit-learn's `TweedieRegressor` is a legitimate tool for insurance pricing. It implements a GLM with Tweedie log-likelihood, handles the log-link correctly, exposes regularisation, and integrates into the full scikit-learn pipeline ecosystem. If you are building your first pure premium model or running a benchmark, reaching for `TweedieRegressor` is not a mistake.

The limitation is structural: it fixes the power parameter `p` and estimates a single dispersion scalar across the entire portfolio. For a textbook GLM that is correct and expected. For insurance pricing, where the whole point is per-risk differentiation, a single dispersion parameter is throwing away half the actuarial information.

This post is a direct comparison. The case against sklearn here is about scope, not quality.

```bash
uv add insurance-distributional
uv add insurance-distributional-glm
```

---

## What TweedieRegressor does, and does it well

`TweedieRegressor` fits a generalised linear model in the exponential dispersion family with Tweedie variance function `Var[Y] = phi * mu^p`. You supply `p`; the model estimates the mean structure. With `p=1` you get Poisson (frequency), `p=2` is Gamma (severity), and for `1 < p < 2` you get the compound Poisson-Gamma that actuaries use for pure premiums - non-negative observations with a point mass at zero.

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

model = Pipeline([
    ("scaler", StandardScaler()),
    ("tweedie", TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000)),
])

model.fit(X_train, y_train, tweedie__sample_weight=exposure_train)
y_pred = model.predict(X_test)
```

The scikit-learn implementation is correct. The deviance loss function, the IRLS solver, the treatment of the log-link, the `sample_weight` mechanism for exposure - all of it is properly implemented. The documentation is clear, the test coverage is strong, and the interface is stable across releases.

For a first-pass pricing model, a regularisation benchmark, or an input into an ensemble, `TweedieRegressor` is perfectly adequate.

The problem emerges when you ask: *which policies in my portfolio are genuinely more volatile, and by how much?* That question cannot be answered by a model that produces only E[Y|X].

---

## The fixed-dispersion problem

A Tweedie GLM with fixed `p` estimates one `phi` - a scalar dispersion parameter that applies to every policy in the portfolio. `Var[Y_i|X_i] = phi * mu_i^p`. The mean `mu_i` varies by risk; the dispersion `phi` does not.

That is the standard GLM assumption. It is also directly at odds with what we observe in insurance portfolios.

Two motor policies priced at a pure premium of £350 are not interchangeable. Policy A is a standard family hatchback, moderate driver age, five years' NCD - a low-frequency, moderate-severity risk where the variance around £350 is modest. Policy B is a young driver on a sports car with one year's NCD - a low-frequency, high-severity risk where the variance around £350 is substantial. Their means are similar. Their dispersion is not.

A constant-`phi` model assigns them identical variance, identical uncertainty, identical safety loading. It has no mechanism to do otherwise.

This matters downstream in three ways that pricing committees should care about:

**Safety loading.** The actuarially correct technical price is `mu * (1 + k * CoV)` where `k` reflects your risk appetite and `CoV = sqrt(phi) * mu^(p/2 - 1)`. If `phi` is constant, the loading is the same fraction for every risk at a given mean. High-CV risks are underloaded; low-CV risks are overloaded. A portfolio-level `phi` hides this.

**Reinsurance.** The policies that drive your XL exposure are not necessarily the ones with the highest mean loss. They are the ones with the highest conditional variance - the risks where the distribution has a heavier right tail. You cannot identify these without per-risk dispersion estimates.

**Underwriter referrals.** If you flag risks for underwriter review, flagging on mean-based metrics alone misses the high-severity, low-frequency policies that look cheap on average but have a wide distribution of outcomes.

```python
# What sklearn gives you
y_pred = model.predict(X_test)
# shape: (n_test,) — one mean prediction per policy
# Variance: phi * y_pred^p, where phi is a single fitted scalar
# No per-risk variance. No CoV per policy.
```

This is not a bug. It is exactly what a GLM was designed to return. The question is whether it is sufficient.

---

## What [`insurance-distributional`](/insurance-distributional/) does instead

[insurance-distributional](https://github.com/burning-cost/insurance-distributional) implements distributional gradient boosting for insurance: you get both `mu_i` and `phi_i` per policy, making variance a function of covariates rather than a portfolio constant.

The approach follows So & Valdez (ASTIN Best Paper 2024, arXiv:2406.16206), implemented with CatBoost as the boosting engine. The dispersion model uses the Smyth-Jørgensen double GLM approach: after fitting the mean model, squared Pearson residuals `d_i = (y_i - mu_hat_i)^2 / V(mu_hat_i)` are used as the target for a second GBM. From v0.1.3 this uses K=3 cross-fitting to prevent the phi model from training on in-sample residuals - a fix that brought empirical coverage at the 80% level from 42% to 80.4% on the benchmark dataset.

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

pred = model.predict(X_test, exposure=exposure_test)

pred.mean              # E[Y|X] — pure premium per policy
pred.variance          # Var[Y|X] — conditional variance per policy
pred.cov               # CoV = SD/mean — per policy, not per portfolio
pred.volatility_score()  # same as cov — for safety loading
```

The `pred` object is what changes the conversation. Every policy in `X_test` gets its own `phi` estimate, its own `variance`, its own `cov`. A young driver on a sports car gets a higher `phi` than a middle-aged driver on a family hatchback, even if their means are similar. The model has actually learned this from the data rather than assuming it away.

### The four distribution classes

| Class | Distribution | Use case |
|-------|-------------|----------|
| `TweedieGBM` | Compound Poisson-Gamma (mu, phi) | Motor/home pure premium |
| `GammaGBM` | Gamma (mu, phi) | Severity-only models |
| `ZIPGBM` | Zero-Inflated Poisson (lambda, pi) | Pet/travel/breakdown frequency |
| `NegBinomialGBM` | Negative Binomial (mu, r) | Overdispersed fleet frequency |

`ZIPGBM` is worth noting separately. For pet or travel insurance, there is a structural distinction between policyholders who will never claim (structural zeros) and those who might. A standard Poisson model cannot separate these. `ZIPGBM` models both the zero-inflation probability `pi` and the underlying Poisson rate `lambda`:

```python
from insurance_distributional import ZIPGBM

model = ZIPGBM()
model.fit(X_zip_train, y_zip_train)
pred = model.predict(X_zip_test)

pred.mu    # observable mean = (1 - pi) * lambda
pred.pi    # structural zero probability per risk
```

scikit-learn has no equivalent. `TweedieRegressor` with `p` close to 1 approximates Poisson but cannot separate the zero-inflation process from the claim rate.

### Honest benchmark numbers

From the `insurance-distributional` benchmark (6,000 synthetic UK motor severity observations, heteroskedastic DGP with `phi` varying 0.42–1.18 across vehicle age and vehicle group, Databricks serverless, v0.1.3):

| Metric | Constant-phi Gamma GLM | GammaGBM (per-risk phi) |
|--------|----------------------|------------------------|
| Gamma deviance | 1.201 | 0.959 |
| Coverage at 80% | 74.8% | 80.4% |
| Coverage at 95% | 90.1% | 94.9% |
| Phi correlation with true DGP | 0.000 (flat) | +0.702 |
| Safety loading spread | 0 | 0.086 |

The constant-phi model systematically under-covers (74.8% actual at the 80% nominal level) because it applies the same uncertainty envelope to low-phi and high-phi risks alike. The per-risk model is near-nominal. The phi correlation of +0.70 means the GBM correctly ranks which risks are more dispersed - useful for underwriter referrals even before you trust the absolute phi values.

One honest caveat from the README: the absolute scale of `pred.phi` carries a systematic upward bias of 5–12% (predicted CoV 0.81–1.12 vs true 0.77–1.01). The ordering is reliable; the absolute value needs cross-validation before use in a literal pricing formula. That is a known limitation, documented clearly.

---

## For GLM teams: insurance-distributional-glm

[insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) takes the same idea into a GLM framework: GAMLSS (Generalised Additive Models for Location, Scale and Shape), which R has had since Rigby and Stasinopoulos (2005) but which had no production-ready Python implementation until now.

The core idea is the same as the GBM variant, but with explicit linear predictors for each distribution parameter:

```
log(mu_i)    = x_i^T beta_mu        # mean depends on risk factors
log(sigma_i) = z_i^T beta_sigma      # CV depends on (possibly different) risk factors
```

This is the natural extension of a standard GLM for pricing teams. The mean submodel is exactly what you already have. The sigma submodel adds a second linear predictor - potentially using a different set of rating factors - to explain why some risks are more volatile than others.

```python
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_value", "ncd_years"],
        "sigma": ["age_band", "vehicle_group"],     # different factors for variance
    },
)
model.fit(df, y)
model.summary()   # relativities for both mu and sigma submodels
```

The fitting uses the Rigby-Stasinopoulos (RS) algorithm: cycle through each distribution parameter, update via IRLS whilst holding the others fixed. On 25,000 observations, wall-clock time is comparable to a single Gamma GLM - the per-iteration IRLS steps are cheap.

The benchmark result is striking. On a synthetic UK motor severity dataset where CV genuinely depends on vehicle class and distribution channel (vehicle class D via broker has roughly 3× the CV of vehicle class A direct), a standard Gamma GLM assigns `sigma = 0.467` to every policy. The GAMLSS model recovers sigma with a correlation of 0.998 against the true DGP. The mean deviance is barely different (0.2385 vs 0.2385) - when the mean model is well-identified, GAMLSS does not improve mean fit. What it improves is calibration at the segment level: 95% PI coverage is 94.25% vs 93.87%.

The `relativities()` output is designed for actuarial review:

```python
# Multiplicative factors per rating level, just like a standard GLM one-way
rel = model.relativities(parameter="mu")
rel_sigma = model.relativities(parameter="sigma")
# exp(coefficient) = multiplicative effect on predicted mean / CV
```

For a pricing team that wants to understand why some risks attract higher sigma, this output is auditable in the same way a standard GLM one-way analysis is auditable.

The library also includes `quantile_residuals` (Dunn & Smyth 1996) and a `worm_plot` for model diagnostics - the standard GAMLSS toolkit that R users will recognise.

---

## The decision framework

**Use sklearn's `TweedieRegressor` when:**
- You want a quick, well-tested baseline or a regularisation benchmark
- You are fitting a model that feeds into an ensemble where per-risk variance is not the output
- You are in a pipeline context and scikit-learn compatibility matters more than distributional flexibility
- You genuinely believe `phi` is constant across your portfolio (this is rarely true for a heterogeneous book, but for a tightly defined risk segment it may hold)

**Use `insurance-distributional` (TweedieGBM/GammaGBM) when:**
- You need per-risk variance estimates - safety loading, underwriter referral thresholds, reinsurance attachment
- Your book has material heteroscedasticity (it almost certainly does)
- You are working with pet or travel insurance and need to separate structural zeros from claim rate (ZIPGBM)
- You have fleet or commercial frequency data with genuine overdispersion beyond what Poisson implies (NegBinomialGBM)
- You want calibrated prediction intervals with near-nominal coverage at 80%/90%/95%

**Use `insurance-distributional-glm` (GAMLSS) when:**
- Your team works with GLMs and you want interpretable sigma relativities alongside mean relativities
- You need an audit trail showing which factors drive volatility - for FCA governance or internal model review
- You want model selection across distributional families (Gamma vs LogNormal vs InverseGaussian) with GAIC
- You need the output to look like a standard GLM one-way analysis, not a GBM prediction

The two libraries are not competing. `insurance-distributional` has the stronger predictive performance (CatBoost GBM, phi correlation +0.70 on the heteroskedastic benchmark). `insurance-distributional-glm` has the more interpretable output and sits closer to the GLM workflow most UK pricing teams already use. For a mature pricing stack, the natural order is: GAMLSS first for structured understanding of the variance drivers, distributional GBM for the final model where predictive performance takes priority.

---

## On the fixed-p assumption

One more difference worth flagging. sklearn's `TweedieRegressor` requires you to specify `p` in advance. The standard actuarial approach is profile likelihood: fit the model for each `p` on a grid (say 1.2, 1.4, 1.5, 1.6, 1.8) and pick the best by deviance. This is external to sklearn - you iterate manually.

`insurance-distributional-glm` handles this through `choose_distribution()` and grid-search over `Tweedie(power)`, with GAIC for model selection. It treats finding the right `p` as part of the modelling problem, not a prerequisite to it.

`TweedieGBM` takes `power` as a parameter and does not profile it automatically. The README recommends profile likelihood as a preprocessing step, which is the right approach - but it does put the manual iteration back on you. For motor pure premium pricing in the UK, `p=1.5` is a reasonable starting point; the actual optimal `p` for a mixed personal lines book is typically between 1.4 and 1.7.

---

## Side-by-side summary

| | sklearn TweedieRegressor | insurance-distributional (TweedieGBM) | insurance-distributional-glm (GAMLSS) |
|---|---|---|---|
| Output | E[Y\|X] only | E[Y\|X], Var[Y\|X], CoV per risk | E[Y\|X], sigma per risk, relativities |
| Dispersion | Single scalar phi | Per-risk phi via GBM | Per-risk sigma via linear predictor |
| Model type | GLM | Gradient boosted | GLM/IRLS |
| Interpretability | Standard GLM relativities | SHAP for variable importance | Explicit sigma relativities |
| Calibrated intervals | No | Yes (post v0.1.3 cross-fitting fix) | Near-nominal (25k benchmark) |
| Zero-inflation | No | ZIPGBM | ZIP family |
| Overdispersion (count) | No | NegBinomialGBM | NBI family |
| sklearn compatibility | Full | No (CatBoost native) | No (Polars/numpy native) |
| Fitting speed | Fast | Slower (two GBMs) | Comparable to single GLM |

sklearn `TweedieRegressor` is in the top row because it does less - not because it does it worse. Everything it promises, it delivers. The question is whether E[Y|X] alone is enough for the pricing decisions you need to make.

For most UK insurance pricing work, it is not.

---

`insurance-distributional` is at [github.com/burning-cost/insurance-distributional](https://github.com/burning-cost/insurance-distributional). CatBoost-based, Python 3.10+, MIT.

`insurance-distributional-glm` is at [github.com/burning-cost/insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm). numpy/scipy, Polars-native, Python 3.10+, MIT.

---

**Related posts:**
- [Your Technical Price Ignores Variance](/2026/03/05/insurance-distributional/) - the full argument for per-risk volatility estimation, So & Valdez (2024) in detail
- [GAMLSS in Python, Finally](/2026/03/10/insurance-distributional-glm/) - the insurance-distributional-glm launch post with full benchmark numbers

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/19/equipy-vs-insurance-fairness/) - optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/19/econml-vs-insurance-causal-inference-pricing/) - causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - model monitoring
- [NannyML vs insurance-monitoring](/2026/03/21/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) - statistical drift tests
