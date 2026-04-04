---
layout: post
title: "Conformal Prediction for Insurance Python: A Frequency-Severity Tutorial"
date: 2026-04-04
categories: [conformal-prediction, insurance-pricing, libraries]
tags: [conformal-prediction, frequency-severity, FrequencySeverityConformal, prediction-intervals, insurance-conformal, python, tutorial, Graziadei-2023, exchangeability, coverage-guarantee, motor, Poisson, Gamma, uk-motor]
author: Burning Cost
description: "A step-by-step tutorial on conformal prediction for insurance Python models, specifically the frequency-severity decomposition. Covers the calibration subtlety that breaks naive approaches, using the insurance-conformal library with a synthetic UK motor dataset."
---

Every Poisson-Gamma pricing model we have ever seen has the same problem. It produces a point estimate of expected loss and, if pushed, a parametric prediction interval derived from the Gamma dispersion. That interval assumes the variance function is correctly specified globally — one dispersion parameter, applied uniformly across every cell from the no-NCD 19-year-old in East London to the 15-year NCD retired driver in rural Shropshire.

The problem is not that the Gamma assumption is particularly bad. It is that the interval cannot adapt to segments where the true dispersion differs from the book average. And the segments where it differs most are always the ones your reinsurer asks most questions about.

Conformal prediction for insurance Python models is the practical fix. [We have written about split conformal for Tweedie pure premium models before.](/2026/03/31/conformal-prediction-intervals-insurance-pricing-python/) This post covers the specific challenge of **frequency-severity models** — and a calibration mistake that is easy to make and expensive to miss.

---

## The frequency-severity setup

Most UK motor and property pricing teams use a two-stage model:

1. **Frequency model** (Poisson GLM or GBM): predicts expected claim count per policy year.
2. **Severity model** (Gamma GLM or GBM): predicts average cost per claim, conditional on a claim occurring.

Expected loss is the product. Intervals are less straightforward.

The naive approach is to generate intervals from each model independently and multiply the bounds. This is wrong for two reasons: the stages are correlated, and the resulting interval has no coverage guarantee.

`FrequencySeverityConformal` from [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) implements the correct approach — a single conformity score over the composed prediction, with a finite-sample coverage guarantee on the aggregate loss. The algorithm is from Graziadei, Janett, Embrechts and Bucher (arXiv:2307.13124).

```bash
pip install insurance-conformal
```

---

## The calibration subtlety that breaks naive implementations

Before the code: the key insight in Graziadei et al. (2023) is about what you feed the severity model at calibration time.

At training time, the severity model sees `(X, d_observed, y)` for policies with at least one claim — the observed count `d_observed` is available. At prediction time for new policies, `d_observed` does not exist; you use `mu_hat(X)`, the predicted frequency, as the input to the severity model.

If you use `d_observed` at calibration too, the score distribution at calibration differs from the score distribution at test time. Exchangeability — the only real assumption underpinning conformal coverage — breaks. The coverage guarantee fails, typically by under-covering high-frequency risks where predicted frequency diverges most from one.

`FrequencySeverityConformal` handles this automatically: at calibration it feeds `mu_hat(X_cal)` (predicted frequency) into the severity model, not the observed claim count. This is the implementation detail that matters.

---

## A complete worked example

We will build a synthetic UK motor portfolio with a known data-generating process, demonstrate the failure of the naive approach, and show correct conformal prediction intervals.

### Step 1: Generate a synthetic motor portfolio

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor

RNG = np.random.default_rng(2026)
N = 30_000

# Rating factors
driver_age    = RNG.integers(18, 80, N).astype(float)
ncd_years     = RNG.integers(0, 9, N).astype(float)
vehicle_group = RNG.integers(1, 21, N).astype(float)
area          = RNG.integers(0, 6, N).astype(float)
exposure      = RNG.uniform(0.25, 1.0, N)

# True frequency (Poisson mean)
log_freq = (
    -2.8
    + 0.025 * np.maximum(0, 25 - driver_age)
    - 0.08  * ncd_years
    + 0.04  * vehicle_group
    + 0.06  * area
)
true_freq = np.exp(log_freq) * exposure

d = RNG.poisson(true_freq)  # observed claim counts

# True severity (Gamma mean, conditional on claim)
claim_mask = d > 0
log_sev = (
    6.5
    + 0.03 * vehicle_group[claim_mask]
    + 0.04 * area[claim_mask]
    - 0.01 * ncd_years[claim_mask]
)
shape_gamma = 3.0
true_sev_mean = np.exp(log_sev)
y_claims = RNG.gamma(
    shape=shape_gamma,
    scale=true_sev_mean / shape_gamma,
)

# Aggregate loss (zero for no-claim policies)
y = np.zeros(N)
y[claim_mask] = y_claims

X = np.column_stack([driver_age, ncd_years, vehicle_group, area, exposure])

# Temporal split: 60% train / 20% calibration / 20% test
n_train = int(0.60 * N)
n_cal   = int(0.20 * N)

X_train, d_train, y_train = X[:n_train],         d[:n_train],         y[:n_train]
X_cal,   d_cal,   y_cal   = X[n_train:n_train+n_cal], d[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
X_test,  d_test,  y_test  = X[n_train+n_cal:],   d[n_train+n_cal:],   y[n_train+n_cal:]

zero_frac = (y_test == 0).mean()
print(f"Test set: {len(y_test):,} policies, {zero_frac:.1%} zero claims")
# Test set: 6,000 policies, ~63% zero claims
```

### Step 2: Fit and calibrate `FrequencySeverityConformal`

```python
from insurance_conformal.claims import FrequencySeverityConformal

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(max_iter=300),
    sev_model=GammaRegressor(max_iter=300),
)

# fit() trains both models on the training split
# d_train = observed claim counts; y_train = aggregate loss
fs.fit(X_train, d_train, y_train)

# calibrate() computes conformity scores using predicted frequency
# — NOT observed claim counts — on the calibration set
fs.calibrate(X_cal, d_cal, y_cal)

print(f"Calibration observations: {fs.n_calibration_:,}")
print(f"Calibrated: {fs.is_calibrated_}")
```

### Step 3: Generate 90% prediction intervals

```python
intervals = fs.predict_interval(X_test, alpha=0.10)
# Returns a polars DataFrame with columns: lower, point, upper

lower = intervals["lower"].to_numpy()
upper = intervals["upper"].to_numpy()
point = intervals["point"].to_numpy()

covered = ((y_test >= lower) & (y_test <= upper)).mean()
mean_width = (upper - lower).mean()

print(f"Empirical coverage: {covered:.3f}  (target: 0.900)")
print(f"Mean interval width: £{mean_width:,.0f}")
```

On a realistic synthetic book, this produces coverage near the 90% target. The finite-sample guarantee (`P(y in interval) >= 1 - alpha`) holds regardless of whether the GLMs are correctly specified — the guarantee is on the composed prediction, not on the individual model assumptions.

### Step 4: Check per-segment coverage

Marginal coverage in aggregate is necessary but not sufficient. The diagnostics that matter are per-segment.

```python
from insurance_conformal import subgroup_coverage

# Coverage by vehicle group band
veh_band = np.where(
    vehicle_group[n_train+n_cal:] <= 7,  "vg01-07",
    np.where(
        vehicle_group[n_train+n_cal:] <= 14, "vg08-14",
        "vg15-20"
    )
)

sg = subgroup_coverage(
    predictor=fs,
    X_test=X_test,
    y_test=y_test,
    alpha=0.10,
    groups=veh_band,
    group_name="vehicle_group",
)
print(sg.select(["vehicle_group", "n_obs", "empirical_coverage", "mean_width"]))
```

---

## What the naive approach gets wrong

To make the calibration problem concrete, here is what happens if you misuse observed claim counts at calibration time. We can simulate this by constructing the conformity scores incorrectly:

```python
# WRONG: feeding observed counts into the severity model at calibration
# This is what a naive implementation would do

freq_pred_cal  = fs.freq_model_.predict(X_cal)     # predicted frequency
sev_input_wrong = np.column_stack([X_cal[d_cal > 0], d_cal[d_cal > 0]])   # observed d
sev_input_right = np.column_stack([X_cal[d_cal > 0], freq_pred_cal[d_cal > 0]])  # predicted mu

sev_pred_wrong = fs.sev_model_.predict(sev_input_wrong)
sev_pred_right = fs.sev_model_.predict(sev_input_right)

# Correlation between predicted frequency and severity prediction error
# is larger with observed counts — score distribution is not exchangeable
corr_wrong = np.corrcoef(
    d_cal[d_cal > 0],
    np.abs(y_cal[d_cal > 0] - sev_pred_wrong)
)[0, 1]
corr_right = np.corrcoef(
    d_cal[d_cal > 0],
    np.abs(y_cal[d_cal > 0] - sev_pred_right)
)[0, 1]

print(f"Score-frequency correlation (wrong): {corr_wrong:.3f}")
print(f"Score-frequency correlation (right): {corr_right:.3f}")
```

The scores using observed counts are systematically correlated with claim frequency — multi-claim policies produce bigger errors because the severity model was trained expecting `d=1` but receives `d=3`. The correct approach zeroes that correlation out. Exchangeability holds; the guarantee holds.

---

## Limitations to be honest about

**IBNR on recent accident years.** If you calibrate on development-year 0 or 1 data, `y_cal` underestimates ultimate claims. The conformity scores are too small, so the quantile `q*` is too low, and the resulting intervals are too narrow. Calibrate only on accident years with at least three years of development, or apply chain-ladder IBNR factors to `y_cal` before calling `calibrate()`.

**Zero-claim policies.** The coverage guarantee applies to the aggregate loss including zero-claim policies. If you want conditional intervals — interval for total loss given a claim — you need a different conformity scoring approach. `FrequencySeverityConformal` produces unconditional intervals.

**Calibration sample size.** The library recommends `n_cal >= 2,000`. Below that, interval widths are materially wider and more variable. On a small book, pool accident years to reach the minimum.

---

## What this is good for

Frequency-severity conformal prediction intervals are most useful when:

- You need a defensible uncertainty range on aggregate loss per policy — for pricing review documentation or reinsurance submissions.
- Your book is heterogeneous enough that a single Gamma dispersion parameter fails to adapt across segments.
- You want a coverage statement that does not rely on correct model specification — "P(actual loss in interval) >= 90%, finite-sample valid" — rather than "this is the 90th percentile of a Gamma distribution with dispersion estimated from book experience".

The interval is wider than a well-specified parametric interval would be on a dataset where the parametric model is exactly correct. On real books it is usually narrower in the high-risk tail, where parametric intervals over-estimate dispersion, and correctly wider in segments with genuine excess variance.

For pure premium (Tweedie) models rather than two-stage models, use [`InsuranceConformalPredictor`](https://github.com/burning-cost/insurance-conformal) directly with `nonconformity="pearson_weighted"`.

---

## Code and library

Full library: [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — ~1,165 downloads/month on PyPI.

```bash
pip install insurance-conformal
```

The `FrequencySeverityConformal` class is in `insurance_conformal.claims`. The worked example above runs without CatBoost or LightGBM — it uses sklearn's `PoissonRegressor` and `GammaRegressor` throughout.

---

*Related:*
- [Conformal Prediction Intervals for Insurance Pricing in Python](/blog/2026/03/31/conformal-prediction-intervals-insurance-pricing-python/) — the Tweedie pure premium version using the Pearson nonconformity score
- [Does Conformal Prediction Actually Work for Insurance Claims?](/blog/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/) — benchmark results on freMTPL2: coverage, width, and where the intervals fail
- [When Does Your Conformal Model Break? Monitoring Coverage Without False Alarms](/blog/2026/04/04/anytime-valid-conformal-monitoring-coverage-sequential-testing/) — the sequential testing problem that invalidates monthly coverage checks
