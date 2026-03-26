---
layout: page
title: "insurance-conformal"
description: "Distribution-free prediction intervals for insurance GBM and GLM pricing models. Finite-sample coverage guarantees. Insurance-specific non-conformity scores for Tweedie and Poisson claims."
permalink: /insurance-conformal/
---

Distribution-free prediction intervals for insurance pricing models. The only requirement is exchangeable calibration and test data — no distributional assumptions, no model specification requirements. Current version: v0.6.0.

**[View on GitHub](https://github.com/burning-cost/insurance-conformal)** &middot; **[PyPI](https://pypi.org/project/insurance-conformal/)**

---

## The problem

Parametric Tweedie prediction intervals assume a single dispersion parameter across all risks. On a heterogeneous UK motor book, this over-covers low-risk policies and under-covers high-risk policies — exactly where getting it wrong is most expensive.

Conformal prediction fixes this without distributional assumptions. The key improvement over generic conformal implementations is the non-conformity score: for Tweedie/Poisson data, the correct score is the locally-weighted Pearson residual `|y - ŷ| / ŷ^(p/2)`, which accounts for the inherent heteroscedasticity of insurance claims. The raw absolute residual — which most conformal implementations use — treats a 1-unit error on a £100 risk identically to a 1-unit error on a £10,000 risk.

Benchmark result: `pearson_weighted` produces 13.4% narrower intervals than parametric Tweedie at identical 90% coverage (50,000 synthetic UK motor policies, CatBoost Tweedie(p=1.5), temporal 60/20/20 split).

---

## Installation

```bash
pip install insurance-conformal
pip install "insurance-conformal[catboost]"   # with CatBoost
pip install "insurance-conformal[lightgbm]"   # with LightGBM
pip install "insurance-conformal[all]"         # everything
```

---

## Quick start

```python
import numpy as np
from insurance_conformal import InsuranceConformalPredictor

# Assuming a fitted CatBoost Tweedie model
cp = InsuranceConformalPredictor(
    model=fitted_catboost_tweedie,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)
# Returns Polars DataFrame: lower | upper | prediction | coverage_target
```

Locally-adaptive intervals (24% narrower than standard pearson_weighted):

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=fitted_catboost, tweedie_power=1.5)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)
intervals = lw.predict_interval(X_test, alpha=0.10)
```

Online conformal with retrospective adjustment — handles distribution shifts (UK motor inflation, Ogden rate changes):

```python
from insurance_conformal import RetroAdj

model = RetroAdj(bandwidth=1.0, lambda_reg=0.1, window_size=250)
model.fit(y_train, X_train)
lower, upper = model.predict_interval(y_test, X_test, alpha=0.10)
```

---

## Key API reference

### `InsuranceConformalPredictor`

Standard split conformal predictor with insurance-specific non-conformity scores.

```python
InsuranceConformalPredictor(
    model,                              # fitted regression model
    nonconformity="pearson_weighted",   # score: pearson_weighted | pearson | deviance | anscombe | raw
    distribution="tweedie",             # tweedie | poisson | gamma
    tweedie_power=1.5,                  # Tweedie variance power (p)
)
```

- `.calibrate(X_cal, y_cal, exposure=None)` — calibrates on held-out data
- `.predict_interval(X, alpha=0.10)` — returns Polars DataFrame with lower/upper bounds
- `.coverage(X_test, y_test)` — empirical coverage on test data

### Non-conformity scores

```python
from insurance_conformal import (
    pearson_weighted_score,   # |y - yhat| / yhat^(p/2) — recommended for Tweedie/Poisson
    pearson_score,            # |y - yhat| / yhat^(1/2)
    deviance_score,           # deviance residual
    anscombe_score,           # Anscombe residual
    raw_score,                # |y - yhat| — use only for Gaussian
)
```

### `LocallyWeightedConformal`

Two-stage conformal with secondary spread model for adaptive-width intervals.

- `.fit(X_train, y_train)` — fits base model and spread model
- `.calibrate(X_cal, y_cal)` — calibrates conformity scores
- `.predict_interval(X, alpha)` — adaptive-width intervals

### `SCRReport`

Solvency II Solvency Capital Requirement upper bounds with conformal coverage guarantees.

```python
from insurance_conformal import SCRReport

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
print(scr.to_markdown())
```

### `RetroAdj` (v0.6.0)

Online conformal inference with retrospective adjustment. Recovers from abrupt distribution shifts within 1–3 steps versus ~200 steps for standard ACI.

```python
RetroAdj(
    bandwidth=1.0,      # KRR bandwidth
    lambda_reg=0.1,     # KRR regularisation
    window_size=250,    # sliding window
)
```

### Diagnostic tools

```python
from insurance_conformal import (
    CoverageDiagnostics,            # coverage by quantile, feature group
    ert_coverage_gap,               # conditional coverage diagnostic
    subgroup_coverage,              # coverage by arbitrary grouping variable
    width_efficiency_comparison,    # compare interval widths across predictors
)
```

### `insurance_conformal.risk` subpackage — Conformal Risk Control

Controls expected loss directly (E[L(C_lambda(X), Y)] ≤ alpha) rather than just coverage probability.

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)
result = psc.predict(premium_new)
# result["upper_bound"] = risk-controlled loading factor per policy
```

### `insurance_conformal.multivariate` subpackage

Joint multi-output conformal prediction for simultaneous frequency/severity intervals.

```python
from insurance_conformal.multivariate import JointConformalPredictor

predictor = JointConformalPredictor(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    alpha=0.05,
    method='lwc',
)
predictor.calibrate(X_cal, Y_cal)
joint_set = predictor.predict(X_test)
```

---

## Benchmark

| Score | Interval width | Coverage (90% target) |
|-------|---------------|----------------------|
| parametric Tweedie | 1.000 (baseline) | 90.1% |
| `raw` | 1.18 | 90.3% |
| `pearson_weighted` | **0.866** | 90.0% |
| `LocallyWeightedConformal` | **0.883** | 90.0% |

13.4% narrower intervals than parametric at identical coverage. Tested on 50,000 synthetic UK motor policies (heteroskedastic Gamma DGP, temporal 60/20/20 split, seed=42).

---

## Related blog posts

- [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [Conformal Risk Control for Insurance Reserves](https://burning-cost.github.io/2026/03/13/insurance-conformal-risk/)
- [Multivariate Conformal Prediction for Joint Frequency-Severity](https://burning-cost.github.io/2026/03/13/insurance-multivariate-conformal/) — joint frequency/severity intervals now implemented in `insurance_conformal.multivariate`
- [Reserve Range with Conformal Coverage Guarantee](https://burning-cost.github.io/2026/03/16/reserve-range-conformal-guarantee/)
- [Conformal Prediction for Non-Exchangeable Claims Time Series](https://burning-cost.github.io/2026/03/15/conformal-prediction-for-non-exchangeable-claims-time-series/)
