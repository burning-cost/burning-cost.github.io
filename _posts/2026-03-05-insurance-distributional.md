---
layout: post
title: "insurance-distributional: Distributional GBMs for UK Insurance Pricing"
date: 2026-03-05
categories: [libraries]
tags: [distributional-regression, tweedie, catboost, dispersion, volatility, motor, pet, zip, zero-inflation, python, astin, insurance-distributional]
description: "Library release: insurance-distributional. First open-source implementation of the ASTIN 2024 Best Paper approach. TweedieGBM, ZIPGBM, GammaGBM, NegBinomialGBM."
---

[`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) is now available.

```bash
uv add insurance-distributional
```

It is the first open-source Python implementation of the distributional GBM approach from So & Valdez (*Applied Soft Computing*, doi:10.1016/j.asoc.2025.113226; arXiv 2406.16206), which won the ASTIN Best Paper at the IAA Joint Colloquium in Brussels, September 2024.

---

## What it does

A standard Tweedie GBM produces one number per risk: E[Y|X]. Two risks at the same expected loss cost look identical to your pricing model, regardless of how different their conditional variance profiles are. `insurance-distributional` adds a second model for the dispersion parameter φ, estimated via the Smyth-Jørgensen double GLM method (ASTIN Bulletin, 2002), giving you Var[Y|X] per risk alongside the mean.

Four model classes:

| Class | Distribution | Primary use |
|---|---|---|
| `TweedieGBM` | Compound Poisson-Gamma | UK motor/home pure premium |
| `GammaGBM` | Gamma | Severity-only models |
| `ZIPGBM` | Zero-Inflated Poisson | Pet/travel frequency |
| `NegBinomialGBM` | Negative Binomial | Overdispersed frequency |

No PyTorch. No Pyro. CatBoost, NumPy, SciPy, Polars.

---

## Minimum viable motor workflow

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)
pred = model.predict(X_val, exposure=exposure_val)

pred.mean              # E[Y|X] — unchanged from your existing model
pred.volatility_score()  # CoV per risk: sqrt(Var[Y|X]) / E[Y|X]
```

The `volatility_score()` — the coefficient of variation per risk — is the new output. It is dimensionless and directly comparable across risks of different sizes. Use it for safety loadings (`P = μ · (1 + k · CoV)`), underwriter referral thresholds, reinsurance attachment optimisation, IFRS 17 risk adjustment inputs, and per-risk capital allocation under Solvency II.

No commercial pricing tool (Emblem, Radar, Guidewire) currently outputs CoV per risk. This is the gap the library fills.

---

## Note on version

Use v0.1.3 or later. Earlier versions had a φ estimation bug in the coverage diagnostic functions — near-zero φ due to in-sample μ overfitting — that caused `coverage()` results to be unreliable. The mean prediction (`pred.mean`) was unaffected in all versions. v0.1.3 fixes this with K=3 cross-fitting and Gamma deviance loss.

---

For the full technical discussion — why XGBoostLSS and NGBoost cannot do this, why CatBoost is the right base for UK personal lines, how the Smyth-Jørgensen dispersion model works, and five specific applications of the volatility score — see [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/04/per-risk-volatility-scoring-with-distributional-gbms/).

Source, tests, and API documentation on [GitHub](https://github.com/burning-cost/insurance-distributional). MIT licence, Python 3.10+.
