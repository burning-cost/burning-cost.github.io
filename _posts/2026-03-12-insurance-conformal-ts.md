---
layout: post
title: "Your Claim Frequency Forecast Has No Honest Error Bars"
date: 2026-03-12
categories: [libraries, pricing, uncertainty]
tags: [conformal-prediction, time-series, ACI, EnbPI, SPCI, claims-forecasting, Poisson, non-exchangeable, insurance-conformal-ts, python]
description: "Standard conformal prediction assumes exchangeability — an assumption violated by every time series. insurance-conformal-ts implements ACI, EnbPI, SPCI, and Conformal PID for insurance claims time series with Poisson/NB nonconformity scores and exposure weighting."
post_number: 76
---

Every claims forecasting workflow produces a point estimate. Most produce confidence intervals. Almost none of those intervals are valid in the statistical sense — they do not contain the true value at the stated frequency.

The problem is exchangeability. Standard conformal prediction produces finite-sample valid prediction intervals under one assumption: the calibration data and test data are exchangeable. In insurance time series — monthly claim counts, quarterly frequency development — that assumption is always violated. Yesterday's data is not exchangeable with today's. The distribution shifts seasonally, with macroeconomic conditions, with underwriting mix changes.

[`insurance-conformal-ts`](https://github.com/burning-cost/insurance-conformal-ts) implements the conformal prediction methods designed for non-exchangeable data.

```bash
uv add insurance-conformal-ts
```

## Why exchangeability fails in insurance time series

Consider a monthly claims frequency series for a motor portfolio. You want a prediction interval for next month's frequency. You calibrate conformal prediction on the last 24 months. But:

- Claims frequency in 2021-2023 had a different level than 2024-2025 due to post-COVID frequency normalisation
- Seasonal patterns (higher frequency in Q4) mean calibration residuals from summer months are not exchangeable with winter months
- Changes in underwriting appetite shift the risk mix over time

Standard split conformal prediction ignores all of this and produces intervals that undercover in distributional shift periods.

## ACI: adaptive conformal inference

Adaptive Conformal Inference (Zaffran et al. 2022 ICML) maintains coverage by adjusting the quantile level adaptively based on recent coverage misses. If your intervals have been missing the last few observations, ACI widens them. If coverage is above target, it narrows them.

```python
from insurance_conformal_ts import ACI

aci = ACI(
    base_model=fitted_frequency_model,
    target_coverage=0.90,
    gamma=0.005,  # adaptation rate
    score="poisson"  # Poisson nonconformity score
)
aci.calibrate(X_cal, y_cal, exposure=exposure_cal)

intervals = aci.predict_interval(X_test, exposure=exposure_test)
```

The adaptation rate `gamma` controls the speed of adaptation. Higher gamma responds faster to distribution shift but produces more variable interval widths.

## EnbPI: ensemble prediction intervals for online learning

EnbPI (Xu & Xie 2021 ICML) maintains an ensemble of base models and uses leave-one-out residuals to construct online prediction intervals. It updates as new observations arrive without needing to refetch historical data.

```python
from insurance_conformal_ts import EnbPI

enbpi = EnbPI(
    base_learners=[model1, model2, model3],
    n_bootstrap=50,
    score="negative_binomial",
    exposure_col="earned_exposure"
)
enbpi.fit(X_train, y_train, exposure=exposure_train)

# Online update as new months arrive
for month_X, month_y, month_exposure in new_data:
    interval = enbpi.predict_interval(month_X, month_exposure)
    enbpi.update(month_X, month_y, month_exposure)
```

## SPCI for covariate-conditional coverage

SPCI (Xu & Xie 2022) produces prediction intervals with conditional coverage guarantees — coverage is valid not just marginally but conditional on the covariates. For insurance, this means intervals that are valid for each risk segment, not just on average.

```python
from insurance_conformal_ts import SPCI

spci = SPCI(
    base_model=fitted_model,
    conditional_model="random_forest",
    target_coverage=0.90,
    score="poisson"
)
spci.calibrate(X_cal, y_cal, exposure=exposure_cal)
intervals = spci.predict_interval(X_test, exposure=exposure_test)
```

## Conformal PID controller

The Conformal PID controller (Angelopoulos et al. 2024) treats prediction interval width as a control variable and applies PID feedback to maintain the target coverage. It handles distribution shift more smoothly than ACI — no discrete adaptation events, just continuous feedback.

```python
from insurance_conformal_ts import ConformalPID

pid = ConformalPID(
    base_model=fitted_model,
    target_coverage=0.90,
    Kp=0.1, Ki=0.01, Kd=0.001,
    score="poisson"
)
pid.calibrate(X_cal, y_cal)
intervals = pid.predict_interval(X_test)
```

## Multi-step-ahead intervals for quarterly reviews

Insurance pricing reviews operate quarterly — you need valid intervals over a 3-month horizon, not just one step ahead.

```python
from insurance_conformal_ts import MSCP

mscp = MSCP(
    base_model=fitted_model,
    max_horizon=3,
    score="poisson",
    joint=True  # simultaneous coverage over all horizons
)
mscp.calibrate(X_cal, y_cal, exposure=exposure_cal)

intervals = mscp.predict_interval(X_test, horizon=3)
# Returns: [(lower_h1, upper_h1), (lower_h2, upper_h2), (lower_h3, upper_h3)]
```

With `joint=True`, the intervals are constructed to cover all three horizons simultaneously at the stated level — the correct interpretation for a quarterly forecast review.

## Poisson and Negative Binomial scores

The choice of nonconformity score matters. For claim count data:

```python
from insurance_conformal_ts import PoissonScore, NegativeBinomialScore

# Poisson score: (y - mu) / sqrt(mu) — appropriate for low counts
score = PoissonScore(exposure_weighted=True)

# NB score: accounts for overdispersion — better for volatile claim series
score = NegativeBinomialScore(theta=2.0, exposure_weighted=True)
```

Exposure weighting adjusts the score by earned exposure — a month with more policies contributes proportionally more to calibration.

## Relationship to insurance-conformal

[`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) handles cross-sectional Tweedie/Poisson models where the exchangeability assumption holds. That's the right tool for cross-sectional validation of a fitted pricing model.

`insurance-conformal-ts` handles the temporal case. Use it for:
- Monthly frequency forecasts for experience rate monitoring
- Quarterly loss ratio projections for pricing committee
- Trend uncertainty quantification beyond bootstrap resampling

---

`insurance-conformal-ts` is available on [PyPI](https://pypi.org/project/insurance-conformal-ts/) and [GitHub](https://github.com/burning-cost/insurance-conformal-ts). 150 tests. Databricks notebook included.

Related: [insurance-conformal](https://burning-cost.github.io/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) for cross-sectional conformal prediction, [insurance-trend](https://burning-cost.github.io/2026/03/13/insurance-trend/) for trend analysis, [insurance-monitoring](https://burning-cost.github.io/2026/03/03/your-pricing-model-is-drifting/) for deployed model drift detection.
