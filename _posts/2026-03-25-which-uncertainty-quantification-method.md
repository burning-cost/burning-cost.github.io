---
layout: post
title: "Which Uncertainty Quantification Method Should You Use?"
date: 2026-03-25
author: Burning Cost
categories: [techniques, uncertainty-quantification]
tags: [uncertainty-quantification, conformal, gamlss, distributional-regression, ngboost, catboost, bayesian, bootstrap, insurance-pricing, solvency-ii, python]
description: "A decision flowchart for pricing actuaries choosing between conformal prediction, GAMLSS, probabilistic GBMs, Bayesian Last Layer, and bootstrap. Opinionated recommendations with code."
---

The question of which uncertainty quantification (UQ) method to use comes up in almost every pricing modernisation project, and it is never answered cleanly. The literature lists five or six viable approaches, each with genuine advantages, and most practitioners default to bootstrap intervals because that is what the last project used. This post replaces that default with a structured decision.

The flowchart below has five branch points. Work through them in order. The first question that applies determines your method.

---

## The decision tree

```
START
│
├─ Do you need a finite-sample coverage guarantee
│  that holds regardless of the model or DGP?
│
│   YES ─────────────────────────────────────────► CONFORMAL PREDICTION
│                                                   (insurance-conformal)
│   NO
│   │
│   ├─ Do you need the full conditional distribution
│   │  per risk — not just an interval, but a
│   │  fitted distribution you can integrate?
│   │
│   │   YES ──────────────────────────────────────► GAMLSS / DISTRIBUTIONAL REGRESSION
│   │                                               (insurance-distributional-glm
│   │                                               or insurance-distributional)
│   │   NO
│   │   │
│   │   ├─ Is your mean model a GBM (CatBoost,
│   │   │  XGBoost, LightGBM) and you want
│   │   │  uncertainty without changing the
│   │   │  training pipeline?
│   │   │
│   │   │   YES ──────────────────────────────────► PROBABILISTIC GBM
│   │   │                                           (NGBoost or CatBoost quantile)
│   │   │   NO
│   │   │   │
│   │   │   ├─ Is computational cost a hard
│   │   │   │  constraint? (batch scoring at scale,
│   │   │   │  or inference latency matters)
│   │   │   │
│   │   │   │   YES ─────────────────────────────► BAYESIAN LAST LAYER
│   │   │   │                                      (BLL / linearised Laplace)
│   │   │   │   NO
│   │   │   │   │
│   │   │   │   └───────────────────────────────► BOOTSTRAP / PARAMETRIC CIs
```

The methods are not mutually exclusive and two of them (conformal + distributional) compose cleanly. But start here.

---

## Node 1: Conformal prediction

**When to choose it:** You need a coverage guarantee. This is non-negotiable in two contexts: (a) reserving or capital inputs where you are making a formal statement about interval validity, and (b) Solvency II Article 101 contexts where the 99.5th percentile needs to be defensible to a PRA internal model team.

Conformal prediction is the only method here that guarantees finite-sample marginal coverage — formally, at least ⌈(1−α)(n+1)⌉/n fraction of test risks will have their actual loss fall inside the interval. That guarantee is distribution-free: it requires no assumption about the claim DGP, only exchangeability between calibration and test data.

For insurance claims the non-conformity score matters. Do not use the default absolute residual. Use the Pearson-weighted score, which normalises by `ŷ^(p/2)` for a Tweedie model with power p. Without it you reproduce the same tail undercoverage you were trying to fix.

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.utils import temporal_split
from insurance_conformal import solvency_capital_range

# Calibrate on held-out data (temporal split is critical)
X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y, calibration_frac=0.20, date_col="accident_year"
)

cp = InsuranceConformalPredictor(
    model=model,                 # any fitted sklearn-compatible model
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

# Per-risk 90% prediction intervals
intervals = cp.predict_interval(X_test, alpha=0.10)

# Solvency II SCR bounds (99.5th percentile, distribution-free)
scr = solvency_capital_range(cp, X_test, alpha=0.005)
print(scr.total_scr)
```

The `coverage_by_decile()` diagnostic is the sanity check you should always run. A well-calibrated run with `pearson_weighted` should be flat across deciles within Wilson score confidence bands.

For regulatory audit contexts, `InsuranceConformalPredictor` is also model-agnostic — the calibration step wraps any fitted model. This matters if the PRA review team wants to run the same methodology against an alternative model.

---

## Node 2: GAMLSS / distributional regression

**When to choose it:** You need more than an interval. You need a fitted conditional distribution per risk — something you can integrate to price XL layers, compute VaR at multiple levels, or feed into a frequency-severity copula.

GAMLSS models each distributional parameter (location, scale, shape) as a function of covariates via separate linear predictors. A Gamma-GAMLSS on severity, for example, models both `mu` and `sigma` as functions of risk characteristics, rather than assuming a constant dispersion across the portfolio.

For GLM-style models with interpretable rating factors, use `insurance-distributional-glm`. For GBM-driven distributional models — where you want CatBoost fitting both the mean and dispersion simultaneously — use `insurance-distributional`.

```python
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu": ["driver_age", "vehicle_age", "ncd_years", "region"],
        "sigma": ["driver_age", "vehicle_age"],  # dispersion varies by risk
    },
)
model.fit(X_train, y_train, exposure=exposure_train)

# Full conditional distribution objects (scipy frozen dists)
dists = model.predict_distribution(X_test)
quantiles_99 = [d.ppf(0.995) for d in dists]   # per-risk 99.5th percentile

# Or the GBM version for richer non-linear dispersion
from insurance_distributional import TweedieGBM

gbm = TweedieGBM(power=1.5)
gbm.fit(X_train, y_train, exposure=exposure_train)
pred = gbm.predict(X_test, exposure=exposure_test)
pred.variance        # per-risk Var[Y|X] — not a portfolio scalar
pred.volatility_score()  # CoV per risk: sqrt(phi(x)) * mu(x)^(p/2 - 1)
```

The key output is `predict_distribution()` on the GLM side and `pred.variance` / `pred.volatility_score()` on the GBM side. These are the inputs to XL pricing, safety loading schedules, and IFRS 17 risk adjustment — use cases where a mere interval is insufficient.

Note that distributional regression does not provide a coverage guarantee. If you need both the full distribution and a formal coverage bound, compose the two: fit a GAMLSS, then conformalise the output. That is valid and we have done it.

---

## Node 3: Probabilistic GBM

**When to choose it:** Your mean model is already a gradient boosting machine, you are confident in the model specification, and you want uncertainty estimates without rebuilding the training pipeline or introducing a separate calibration set.

Two approaches:

**NGBoost** treats uncertainty quantification as a primary objective. It fits a parametric distribution (Gaussian, LogNormal, or custom) at each step using Natural Gradient boosting (Duan et al., 2020). The output is a fully specified conditional distribution, similar to GAMLSS but with a boosting engine. The downside is that NGBoost is slower than standard GBMs and the parametric family is a harder assumption than conformal calibration.

**CatBoost quantile regression** trains multiple quantile models (e.g. 5th, 50th, 95th) simultaneously via the pinball loss. It is fast, straightforward, and integrates with your existing CatBoost infrastructure. The trade-off: quantile crossing is possible without post-processing, and there is no parametric distribution object to integrate analytically.

```python
from ngboost import NGBRegressor
from ngboost.distns import LogNormal

ngb = NGBRegressor(Dist=LogNormal, n_estimators=500, verbose=False)
ngb.fit(X_train, y_train)
pred_dist = ngb.pred_dist(X_test)
lower = pred_dist.dist.ppf(0.05)
upper = pred_dist.dist.ppf(0.95)

# Or: CatBoost quantile — simpler, faster, no parametric assumption
from catboost import CatBoostRegressor
cb_q05 = CatBoostRegressor(loss_function="Quantile:alpha=0.05", verbose=0)
cb_q95 = CatBoostRegressor(loss_function="Quantile:alpha=0.95", verbose=0)
cb_q05.fit(X_train, y_train)
cb_q95.fit(X_train, y_train)
```

Neither approach provides a coverage guarantee. If coverage at specific levels is required, wrap either with a conformal calibration step (conformalized quantile regression, or CQR, is directly supported in `insurance-conformal`).

We prefer CatBoost quantile for operational work — it is fast, auditable, and easy to explain to a review panel. NGBoost is better when you genuinely need the parametric distribution object and want boosted fitting.

---

## Node 4: Bayesian Last Layer

**When to choose it:** Computational cost is a hard constraint and you are working with a neural network mean model. BLL (Bayesian Last Layer, or linearised Laplace approximation) adds uncertainty estimates to a pre-trained network at near-zero marginal cost during inference.

The approach (Kristiadi et al., arXiv:2302.10975) replaces the final linear layer with a Bayesian linear model fitted on the frozen penultimate layer features. The posterior over the last layer weights is Gaussian and closed-form, so prediction is fast — one forward pass plus a linear algebra computation against a fixed covariance matrix.

For insurance pricing the main use case is parameter uncertainty in neural rate models: you have a pre-trained network (e.g. an embedding model for telematics or geospatial factors) and you want to quantify how confident the model is in regions of the feature space with sparse training data.

BLL does not provide a distribution-free coverage guarantee. The uncertainty estimates are Bayesian credible intervals, conditional on the model being correctly specified. For regulatory purposes this is a weaker claim than conformal prediction. Use BLL for internal risk management and model quality diagnostics, not for capital inputs submitted to the PRA.

```python
from laplace import Laplace

# Your pre-trained neural network (any PyTorch nn.Module)
la = Laplace(model, "regression", subset_of_weights="last_layer", hessian_structure="full")
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

# Posterior predictive — mean and variance per risk
mean, var = la(X_test_tensor, pred_type="glm", link_approx="probit")
```

The `laplace` package (Daxberger et al., 2021) is the standard implementation. This is lightweight and fast — fitting the Laplace approximation on the last layer typically takes seconds even on a large network. Full-network Laplace (not last-layer) is computationally heavier and rarely justified for insurance applications.

---

## Node 5: Bootstrap / parametric confidence intervals

**When to choose it:** You have ruled out the other four, or you are operating in a context where interpretability of the uncertainty source matters more than statistical rigour. Bootstrap CIs are easy to explain — "we refitted the model 500 times on resampled data and measured the spread" — and that explanation carries weight in non-technical governance forums.

The use case we find most defensible: quantifying parameter uncertainty in a GLM rate model for a regulatory sign-off where the reviewer wants to understand sensitivity to training data, not a probabilistic bound on individual predictions. A bootstrap of GLM coefficients gives you a credible interval on each rating factor, which is something a conformal interval on individual claims does not directly provide.

The main failure mode is that bootstrap intervals for individual predictions (not parameters) are notoriously optimistic in the tail. They resample the training data, not future data, so they conflate estimation uncertainty with predictive uncertainty. Do not use bootstrap prediction intervals as capital inputs.

```python
import numpy as np
from sklearn.linear_model import TweedieRegressor

n_boot = 500
predictions = np.zeros((n_boot, len(X_test)))

for i in range(n_boot):
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    m = TweedieRegressor(power=1.5, alpha=0.0)
    m.fit(X_train[idx], y_train[idx])
    predictions[i] = m.predict(X_test)

lower = np.percentile(predictions, 5, axis=0)
upper = np.percentile(predictions, 95, axis=0)
```

For parametric CIs (delta method or asymptotic Wald intervals), use the standard GLM output directly. For Tweedie GLMs in statsmodels, `result.conf_int()` returns Wald intervals on coefficients. These are parameter intervals, not prediction intervals — do not conflate them.

---

## Comparison table

| Method | Coverage guarantee | Full conditional distribution | Computational cost | Regulatory acceptance | Burning Cost library |
|---|---|---|---|---|---|
| Conformal prediction | Yes — finite-sample, distribution-free | No (interval only, unless composed) | Low (one calibration pass) | Strong — model-agnostic, PRA-friendly | `insurance-conformal` |
| GAMLSS / distributional GLM | No | Yes — scipy frozen dist per risk | Medium (iterative IRLS) | Moderate — parametric assumption required | `insurance-distributional-glm` |
| Distributional GBM (TweedieGBM) | No | Partial — mean and variance | Medium (CatBoost training × 2) | Moderate — harder to audit than GLM | `insurance-distributional` |
| Probabilistic GBM (NGBoost) | No | Yes — parametric family | High (slower than standard GBM) | Low — less familiar to reviewers | `ngboost` (external) |
| CatBoost quantile | No (crossing possible) | No (quantiles only) | Low (standard CatBoost) | Moderate — simple to explain | `catboost` (external) |
| Bayesian Last Layer | No (credible, not frequentist) | Gaussian approximation | Very low (last layer only) | Low for capital; fine for monitoring | `laplace` (external) |
| Bootstrap / parametric CIs | No (training-data resampling) | No | Medium (n_boot refits) | High for parameter CIs; low for prediction CIs | — |

---

## What we actually use

In practice, we reach for conformal prediction first in any context where the uncertainty estimate feeds into a number someone will rely on — reserving, treaty pricing, capital. The formal coverage guarantee is worth the minor overhead of the calibration set. The `pearson_weighted` score in `insurance-conformal` handles heteroskedastic tail variance, which is the failure mode that makes parametric intervals dangerous on real motor books.

We reach for `insurance-distributional-glm` when the output is the full conditional distribution and interpretable rating factor coefficients both matter — for example, pricing a reinsurance layer on a commercial book where the underwriter needs to see the mu and sigma models separately.

We use `TweedieGBM` from `insurance-distributional` when we are already in a GBM workflow and want per-risk dispersion without a separate modelling exercise. The CRPS and log-score diagnostics are more informative than MSE for evaluating whether the uncertainty estimates are actually calibrated.

Bootstrap stays for governance documents where the audience is a non-technical board and the question is "what happens if we re-estimate the model on next year's data?" — a question about parameter stability, not predictive uncertainty.

---

## Installation

```bash
uv add insurance-conformal           # conformal prediction
uv add insurance-distributional-glm  # GAMLSS / distributional GLM
uv add insurance-distributional      # distributional GBM (TweedieGBM)
```

All three are on PyPI and tested against Python 3.11+.

---

**Related posts:**
- [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/)
- [Per-Risk Volatility Scoring: How to Replace Your Constant Phi with a Distributional GBM](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/)
- [Reserve Range with a Conformal Guarantee](/2025/12/10/reserve-range-conformal-guarantee/)
