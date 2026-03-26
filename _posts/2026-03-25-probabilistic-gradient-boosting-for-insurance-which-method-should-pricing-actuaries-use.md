---
layout: post
title: "Probabilistic Gradient Boosting for Insurance: Which Method Should Pricing Actuaries Use?"
date: 2026-03-25
categories: [pricing, techniques, research]
tags: [distributional-regression, gradient-boosting, xgboostlss, ngboost, catboost, lightgbm, tweedie, insurance-distributional, benchmarking, proper-scoring-rules, crps, uk-motor, actuarial, research, python]
description: "Chevalier & Côté (EAJ 2025) benchmark nine GBM variants on five insurance datasets. We read it so you don't have to, then show where insurance-distributional fits in."
---

The question comes up whenever a pricing team considers moving beyond a Tweedie GLM: which gradient boosting method should we use, and does going probabilistic actually help?

For the last few years the honest answer has been "it depends, and we don't have good insurance-specific evidence." That changed in December 2024. Dominik Chevalier and Marie-Pier Côté at Université Laval published a systematic comparison — arXiv:2412.14916, subsequently accepted in the European Actuarial Journal (August 2025, doi: 10.1007/s13385-025-00428-5) — that is the most rigorous insurance benchmark of probabilistic GBM methods to date.

Here is what they found, and what it means for a UK pricing team deciding which tools to reach for.

---

## What the paper actually benchmarks

Nine algorithms, five publicly available insurance datasets, two modelling tasks (claim frequency with Poisson, claim severity with Gamma).

The nine algorithms split into two groups:

**Point prediction:** GBM (scikit-learn), XGBoost, LightGBM, CatBoost, EGBM

**Probabilistic:** XGBoostLSS, NGBoost, PGBM, cyc-GBM

The five datasets include `freMTPL2freq` and `freMTPL2sev` from the French motor third-party liability book (the standard academic insurance benchmarks), plus three others. Exposure is handled via offset or sample weight throughout.

Three evaluation dimensions:

1. **Computational efficiency** — training time in seconds
2. **Predictive accuracy** — Poisson and Gamma deviance on held-out test sets
3. **Model adequacy** — log-score and CRPS as proper scoring rules, plus coverage calibration of prediction intervals

This is the right evaluation structure. Deviance measures mean prediction quality. Proper scoring rules measure distributional quality. Coverage tells you whether the stated 90% interval actually contains 90% of observations. All three dimensions matter for pricing.

---

## The findings

### LightGBM dominates on efficiency

LightGBM is the fastest algorithm tested, often by a factor of two or more relative to XGBoost and CatBoost. This is not surprising — LightGBM's histogram-based algorithm is specifically engineered for speed. What the paper confirms is that this speed advantage comes with essentially no accuracy cost on these datasets. If you need to iterate quickly across a large feature set, LightGBM is the right default.

CatBoost is slower but shows a consistent advantage on datasets with high-cardinality categorical variables. That is its intended use case: ordered target statistics for variables like vehicle make/model, occupation code, and postcode, without manual encoding. For UK personal lines, where a 2,000-level vehicle grouping is typical, this matters.

### XGBoostLSS wins among probabilistic methods

Of the four probabilistic variants, XGBoostLSS is the clear winner. It is the most computationally efficient probabilistic method and achieves better coverage calibration than the alternatives. XGBoostLSS (Alexander März, StatMixedML) fits separate tree ensembles for each distributional parameter simultaneously, deriving gradients and Hessians from the full negative log-likelihood via PyTorch autograd.

The runners-up tell a cautionary story:

**NGBoost** (Duan et al., Stanford, ICML 2020) uses natural gradient updates with a scoring rule objective. It works, but it is slow — the natural gradient computation is expensive. On these datasets it loses in three of four average rank comparisons. We would not recommend NGBoost for production pricing work unless you have a specific reason to prefer it.

**PGBM** (Sprangers et al., KDD 2021) takes a different approach: it estimates the variance of leaf predictions across trees and fits a distribution post-hoc from the estimated mean and variance. It is fast and clever, but the distribution is determined after the fact rather than jointly estimated. This limits the distributional families available and produces less reliable calibration than XGBoostLSS.

**cyc-GBM** (Zakrisson, 2023) implements cyclic coordinate descent over distributional parameters, mimicking the GAMLSS EM algorithm with trees. It loses in three of four tables and is effectively abandoned — the GitHub repository has had no commits since July 2023. We would not use this.

### EGBM: interpretable and competitive

The most strategically interesting finding is the performance of EGBM — Explainable Gradient Boosting Machine, from the interpret-ml project (Microsoft). EGBM is fully interpretable: it produces shape functions for each feature that are directly readable, not post-hoc approximations. No SHAP required.

The paper finds EGBM is competitive with the black-box GBM methods on accuracy. Not identical, but not materially worse. This has direct implications for regulatory and governance pressure in UK pricing: if EGBM can match XGBoost on Poisson deviance while producing natively interpretable outputs, the usual accuracy–interpretability trade-off dissolves.

We think this will attract actuarial attention. The IFoA working parties on machine learning in underwriting have been reluctant to endorse black-box methods. If EGBM starts showing up in their literature, expect it to become a governance-friendly alternative to GBM+SHAP relativities.

### Do probabilistic methods beat point estimators?

On mean prediction — Poisson or Gamma deviance — probabilistic GBMs are not better than their point-prediction equivalents. XGBoostLSS does not fit the mean better than XGBoost. The probabilistic constraint costs you nothing on mean accuracy, but it also gains you nothing. You fit probabilistically for the distributional output, not for a better mean.

On distributional quality — log-score, CRPS, coverage — the probabilistic methods do better, as expected. A model designed to output a full distribution will produce better-calibrated intervals than a point model. This is not a surprise, but it is useful to have confirmed on insurance data.

The practical question is whether better-calibrated intervals are worth the additional complexity and fit time. If your downstream use is mean prediction only, they are not. If you need calibrated uncertainty — for IFRS 17 risk adjustments, safety loading differentiation, or reinsurance optimisation — they are.

---

## What the paper does not cover

The benchmark tests Poisson frequency and Gamma severity separately. It does not test:

- **Tweedie (compound Poisson-Gamma)** — the most common single-model formulation for pure premium in UK personal lines pricing
- **Zero-inflated distributions** — relevant for lines with structural excess of zeros (home contents frequency, commercial property)
- **Exposure weighting beyond offset** — policy-year exposure requires a proper multiplicative offset in the mean model; the paper handles this via sample weights which is not identical

None of these are gaps in the authors' intent — they chose public benchmark datasets and standard actuarial frequency/severity models. But they do mean you cannot read these results directly as a recommendation for a pure premium Tweedie model on a UK motor book.

---

## Where `insurance-distributional` fits

Our [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) library is CatBoost-based distributional regression with the distributions and diagnostics that insurance actuaries actually need.

```bash
uv add insurance-distributional
```

The library implements: `TweedieGBM`, `GammaGBM`, `ZIPGBM`, `NegBinomialGBM`, and `FlexCodeDensity`. CatBoost is the backend for all of them — not because CatBoost wins the Chevalier & Côté benchmark (it does not, taken overall), but because it is the right backend for UK personal lines data with high-cardinality categoricals.

We are not claiming `insurance-distributional` beats XGBoostLSS. The EAJ 2025 paper did not test a CatBoost-native distributional model, so there is no direct comparison. What we can say is: if you need Tweedie, or zero-inflated count models, or exposure-weighted distributional regression on a UK motor book, XGBoostLSS does not have those. `insurance-distributional` does.

### Fitting a Tweedie distributional model

The basic fit takes the same form as a standard scikit-learn model:

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

pred = model.predict(X_test, exposure=exposure_test)

print(pred.mean)              # E[Y | X] — pure premium, exposure-adjusted
print(pred.variance)          # Var[Y | X] — conditional variance
print(pred.volatility_score())  # CoV per risk: sqrt(Var) / E
```

The `exposure` parameter enters as a multiplicative offset in the log-linear mean model — `log(E[Y]) = log(exposure) + Xb` — not as a sample weight. That is the correct actuarial formulation, and it differs from how the EAJ 2025 benchmark handles exposure.

### Scoring with proper scoring rules

The paper uses log-score and CRPS as its distributional quality metrics. Both are available directly from `insurance-distributional`:

```python
crps_val = model.crps(X_test, y_test, exposure=exp_test)
log_s    = model.log_score(X_test, y_test, exposure=exp_test)

print(f"CRPS:      {crps_val:.4f}")
print(f"Log-score: {log_s:.4f}")
```

CRPS is in the same units as the claims outcome. Log-score is the negative log-likelihood of the data under the fitted distribution. Use both: CRPS is more robust to outliers, log-score is more sensitive to tail calibration.

### PIT diagnostics

The benchmark in the paper checks coverage calibration of prediction intervals. The equivalent in `insurance-distributional` is the probability integral transform:

```python
from insurance_distributional import pit_values
import numpy as np

pit = pit_values(y_test, pred)

# Under a well-calibrated distribution, PIT values are uniform on [0, 1]
# Reference standard deviation for uniform: 0.289
print(f"PIT std dev: {pit.std():.3f}  (reference: 0.289)")
```

A value materially below 0.289 indicates the model is under-dispersed — intervals that are too narrow, the same failure mode the paper identifies in point estimators when used to construct intervals via simulation.

### Zero-inflated frequency models

For lines with structural zeros — commercial property, pet insurance, home contents — the Poisson assumption is wrong at the mass point. `ZIPGBM` handles this directly:

```python
from insurance_distributional import ZIPGBM

model_zip = ZIPGBM()
model_zip.fit(X_train, y_train, exposure=exposure_train)

pred_zip = model_zip.predict(X_test, exposure=exposure_test)
print(pred_zip.mean)      # E[Y | X] = (1 - pi) * lambda * exposure
print(pred_zip.pi)        # structural zero probability per risk
```

XGBoostLSS does support zero-inflated Poisson (ZIP), but not Tweedie. If your line has both zero-inflation and continuous severity (which is the typical case — claims are rare events with continuous amounts), you need either two separate models or a Tweedie approximation. `insurance-distributional` gives you the Tweedie path natively.

---

## Our recommendation

For pricing actuaries choosing between approaches in 2026:

**If you need a fast, accurate mean model:** LightGBM. The EAJ 2025 paper confirms it — it is the most efficient option and does not materially sacrifice accuracy. You already knew this, but now you have a peer-reviewed citation.

**If you need full distributional output with XGBoost or LightGBM as your backend:** XGBoostLSS. It is the best-benchmarked probabilistic GBM available, it is actively maintained (v0.6.1, December 2025), and it supports 26 distributions. Its gaps are Tweedie (not implemented) and CatBoost (CatBoostLSS is explicitly abandoned).

**If you need Tweedie, ZIP, or NegBinomial with exposure offsets on UK personal lines data:** `insurance-distributional`. Not because it has been independently benchmarked against XGBoostLSS — it has not — but because it is the only option that implements those distributions with the actuarial vocabulary pricing teams need. The CatBoost backend is the right choice for high-cardinality categoricals.

**If you are under governance pressure to use interpretable methods:** EGBM deserves a serious look. The EAJ 2025 paper finds it competitive with black-box GBMs on Poisson and Gamma deviance. The IFoA's model transparency guidance will be relevant here, and EGBM produces native shape functions without post-hoc explanation.

**If someone proposes NGBoost, PGBM, or cyc-GBM:** ask why. NGBoost is slow. PGBM is indirect. cyc-GBM is abandoned. None of them have a compelling case over XGBoostLSS for insurance use.

One honest caveat: distributional GBMs are not always better than a well-calibrated Tweedie GLM. The benchmarks in the paper use deviance and proper scoring rules as metrics. Those metrics favour models that fit the data's distributional shape. If your downstream use is mean prediction only — and most pricing models are ultimately used to set expected loss — a GLM with a GBM mean and Pearson residual diagnostics will often be sufficient. The distributional machinery earns its place specifically when you need calibrated intervals: IFRS 17 risk adjustments, safety loading differentiation, underwriter referral rules, reinsurance attachment analysis.

---

## Related

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — practical walkthrough of `TweedieGBM` on a synthetic UK motor portfolio, including CoV calculations for safety loading
- [`insurance-gam`](https://github.com/burning-cost/insurance-gam) — interpretable additive models for insurance pricing; the alternative to EGBM if you prefer the GAM family
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — distribution-free prediction intervals via conformal prediction; no distributional assumption required, valid coverage guarantees by construction

Source: Chevalier, D. & Côté, M.-P. (2025). From point to probabilistic gradient boosting for claim frequency and severity prediction. *European Actuarial Journal*. doi:10.1007/s13385-025-00428-5. arXiv:2412.14916.
