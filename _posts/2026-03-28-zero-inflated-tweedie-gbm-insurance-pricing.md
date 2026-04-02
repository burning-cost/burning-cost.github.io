---
layout: post
title: "Zero-Inflated Tweedie GBM for Insurance Pricing"
description: "insurance-distributional v0.3.0 ships ZeroInflatedTweedieGBM — the first open-source implementation of So & Valdez (2024) Scenario 2. When standard Tweedie gets structural zeros wrong, this is what to use instead."
date: 2026-03-28
categories: [pricing, machine-learning]
tags: [Tweedie, zero-inflated, ZI-Tweedie, insurance-distributional, CatBoost, So-Valdez-2024, ASTIN-Best-Paper, frequency-severity, pure-premium, structural-zeros, distributional-regression, Python]
---

Standard Tweedie handles zeros through the compound Poisson structure: a zero claim outcome is just a Poisson draw of zero claim counts, and the mass at zero comes out of the mean and power parameter. For most motor and home books, this is fine. Tweedie(p=1.5) with a well-specified GBM fits the zero inflation implicitly.

But some lines have more zeros than Tweedie can accommodate. Not because the data is poorly specified — because the zeros have a genuinely different generative mechanism. Think no-claims bonus systems that cause low-risk policyholders to under-report small claims. Commercial lines with policy features that produce structural non-claims. Telematics products where driver behaviour creates a bimodal distribution: a mass of zero-claim low-risk drivers, and a positive tail of active claimers.

In those situations, a Tweedie model is mis-specified. You can push p toward 1 (more mass at zero), but then you distort the severity distribution for the positive tail. You cannot simultaneously fit both with a single parameter.

This is what zero-inflated Tweedie is for. And as of [insurance-distributional v0.3.0](https://github.com/burning-cost/insurance-distributional), it is now pip-installable.

---

## The model

Zero-inflated Tweedie mixes a point mass at zero with a Tweedie distribution for positive claims:

```
P(Y = 0 | x) = q(x) + (1 - q(x)) * P_Tweedie(Y = 0 | x)
f(Y = y | x) = (1 - q(x)) * f_Tweedie(y | x)    for y > 0
```

The combined expected value is:

```
E[Y | x] = (1 - q(x)) * mu(x)
```

where `q(x)` is the structural zero probability and `mu(x)` is the Tweedie mean for the positive component.

The model has two moving parts: `q(x)` and `mu(x)`. Standard Tweedie has one. The gain is that you can independently model why a policyholder has zero claims (structural behaviour, captured by `q`) versus how much they claim when they do claim (severity distribution, captured by `mu` and `phi`). These can have entirely different covariate structures.

So & Valdez (2024, arXiv:2406.16206, Applied Soft Computing) developed two scenarios for fitting both parameters jointly with CatBoost. Scenario 1 uses separate, unrelated tree ensembles for `logit(q)` and `log(mu)`, updated by coordinate descent. Scenario 2 — their novel contribution and the one that won the ASTIN Best Paper award — imposes a functional link: `q = 1/(1 + mu^gamma)`, where `gamma` is a scalar that controls how tightly the zero probability is linked to the mean. As `gamma -> 0`, Scenario 2 reduces to standard Tweedie.

`ZeroInflatedTweedieGBM` implements the Scenario 2 two-stage estimation approach: a CatBoost binary classifier for `q(x)` and a CatBoost Tweedie regressor for `mu(x)`, estimated on appropriate data subsets and combined at prediction time. This is the first open-source pip-installable implementation of this model class.

---

## Why a standalone class, not a DistributionalGBM subclass

Every other model in the library — `TweedieGBM`, `GammaGBM`, `ZIPGBM`, `NegBinomialGBM` — is a subclass of `DistributionalGBM`. The base class assumes a single parametric family estimated on all n observations via a single CatBoost objective.

`ZeroInflatedTweedieGBM` breaks that assumption in a structural way. Stage 1 (the zero classifier) trains on all n observations. Stage 2 (the Tweedie regressor for positive amounts) trains only on the `n_pos` observations where `y > 0`. There is no single CatBoost objective that can capture both simultaneously without custom gradient surgery.

We considered implementing the Scenario 2 functional link directly as a custom CatBoost objective — the paper provides the analytic gradients and Hessians — but the maintenance overhead of custom CatBoost objectives is substantial and the practical difference in predictive performance is small once you have a well-specified two-stage model. The two-stage approach is also easier to diagnose: you can inspect the classifier and the severity model separately.

So `ZeroInflatedTweedieGBM` is a standalone class. It does not return a `DistributionalPrediction` object. `predict()` returns an ndarray of combined means; `predict_components()` returns a dict.

---

## API

```python
from insurance_distributional import ZeroInflatedTweedieGBM
import numpy as np

model = ZeroInflatedTweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

# Combined mean: E[Y|x] = (1-q) * mu
y_pred = model.predict(X_test, exposure=exposure_test)

# Decomposed predictions
components = model.predict_components(X_test, exposure=exposure_test)
# components is a dict with keys:
#   "zero_prob"      — q(x): P(structural zero | x)
#   "severity_mean"  — mu(x): E[Y | Y > 0, x]
#   "combined_mean"  — (1-q) * mu, same as predict()

# sklearn-compatible zero probabilities
proba = model.predict_proba(X_test)   # shape (n, 2)

# Model evaluation
tweedie_deviance = model.score(X_test, y_test, exposure=exposure_test)
nll = model.log_score(X_test, y_test, exposure=exposure_test)
```

The `power` parameter controls the Tweedie variance function for the positive component: `p=1.5` is the standard choice for UK motor pure premium, `p=1.9` suits heavy-tailed severity. The same defaults apply as `TweedieGBM`.

You can pass CatBoost parameters to each stage separately:

```python
model = ZeroInflatedTweedieGBM(
    power=1.5,
    catboost_params_classifier={"iterations": 400, "depth": 5},
    catboost_params_tweedie={"iterations": 500, "depth": 6},
    cat_features=["vehicle_class", "area_code"],
)
```

The model pickles cleanly — useful for Databricks MLflow logging and for serving via ONNX or a simple joblib registry.

---

## Installation

```bash
uv add insurance-distributional
# or
uv add insurance-distributional
```

v0.3.0 is on PyPI now. Requires Python 3.10+, CatBoost, NumPy, SciPy.

---

## When to use it versus standard Tweedie

Standard `TweedieGBM` is the right default for most motor and home books. The compound Poisson structure already accommodates a mass at zero, and the power parameter implicitly controls the degree of zero-inflation. If your zero rate is consistent with what Tweedie(p=1.5) predicts given your mean, there is no benefit to adding a second stage.

Use `ZeroInflatedTweedieGBM` when you have evidence of excess zeros beyond what Tweedie predicts. Practically, this means:

**Check the residual zero rate.** Fit `TweedieGBM`, predict `mu_hat` and `phi_hat`, compute the implied `P(Y=0 | x)` from the Tweedie CDF, and compare to the observed zero rate by segment. If the model systematically under-predicts zeros for certain covariate combinations — low-value risks, long-tenure policyholders, certain vehicle classes — that is structural zero-inflation.

**Consider the data generating process.** No-claims bonus systems that create rational under-reporting of small claims are a textbook case. The zero is not a zero-count Poisson outcome — it is a policyholder choosing not to claim. A ZI model captures this by letting `q(x)` learn from the features that predict non-claiming behaviour (tenure, NCD level, policy excess) independently of the severity features.

**Check the zero rate by subgroup.** So & Valdez used a synthetic telematics dataset with 97.3% zero claims. That is an extreme case — most UK personal lines books sit at 65–85%. But even at 75% zeros, if the Tweedie-predicted zero probability varies by only 5 percentage points across your risk range while the observed rate varies by 15 points, you have an identifiable structural zero-inflation problem.

If you have a specific high-excess commercial product where zeros are common across all risk types, a simpler test is: does adding a zero-inflation stage reduce out-of-sample Tweedie deviance? Run a cross-validated comparison. If the deviance improvement is less than 0.5%, standard Tweedie is fine and the additional complexity is not justified.

---

## The paper

So & Valdez (2024) — full title: *Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics*, arXiv:2406.16206, published in Applied Soft Computing 2025 — won the ASTIN Best Paper award for 2024. The award matters not because of prestige but because ASTIN judges papers on practical relevance, not just theoretical novelty. The committee considered this work applicable enough to real insurance pricing to merit the top prize.

The paper demonstrates that ZI-Tweedie variants substantially outperform standard Tweedie on their synthetic telematics dataset across unit deviance, mean absolute deviation, and Gini coefficients. Scenario 2 (the functional link approach) and Scenario 1 (independent `q` and `mu`) perform comparably in their experiments, with Scenario 1 having a slight edge on some metrics at the cost of 2× the trees.

There is no other open-source implementation of either scenario. XGBoostLSS and LightGBMLSS do not support Tweedie at all. Commercial tools (Emblem, Radar) implement their own ZI variants, but the implementations are opaque and not pip-installable for integration into custom pipelines.

`ZeroInflatedTweedieGBM` in insurance-distributional v0.3.0 fills that gap.

The library is [on GitHub](https://github.com/burning-cost/insurance-distributional). If you fit this to real data and want to share results or flag an edge case, open an issue or start a discussion.
