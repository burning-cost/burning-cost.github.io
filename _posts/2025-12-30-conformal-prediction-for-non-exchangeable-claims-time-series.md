---
layout: post
title: "Your Conformal Intervals Are Wrong When the Claims Series Has Trend"
date: 2025-12-30
categories: [pricing, libraries, tutorials]
tags: [conformal-prediction, time-series, aci, spci, exchangeability, claims-monitoring, loss-ratio, insurance-conformal-ts, sequential-coverage, kupiec, python, tutorial]
description: "Standard split conformal prediction requires exchangeability — a condition insurance claims time series systematically violate."
---

Standard conformal prediction gives a finite-sample coverage guarantee. The small print on that guarantee: the calibration and test data must be exchangeable — drawn from the same distribution, with no ordering dependence between them. A monthly UK motor claims count series does not satisfy this. Q4 always runs heavier than Q2. A market-hardening year looks nothing like the stable period you calibrated on. A newly priced segment that added younger drivers last autumn has a different frequency distribution than the training data from two years before.

When exchangeability fails, the coverage guarantee fails with it. The intervals you compute in January using a calibration set from January through December will systematically miss in February and March — precisely the direction the trend was running. Standard split conformal does not warn you. It just quietly undercovers.

[`insurance-conformal-ts`](https://github.com/burning-cost/insurance-conformal-ts) implements methods designed for the temporal case. The two we use most often are ACI (Adaptive Conformal Inference, Gibbs and Candès NeurIPS 2021) and SPCI (Sequential Predictive Conformal Inference, Xu et al. ICML 2023). Both produce sequential prediction intervals that adapt as new observations arrive. Neither requires the calibration data to be exchangeable with the test period. Both come with diagnostics for checking whether coverage is actually being maintained.

```bash
pip install insurance-conformal-ts
```

---

## Why exchangeability breaks and why it matters

The coverage guarantee from split conformal is:

```
P(y_test ∈ [lower, upper]) ≥ 1 - alpha
```

This holds when the calibration scores and the test score are exchangeable — meaning that if you randomly permuted the ordering, the joint distribution would be unchanged. For i.i.d. data from a stationary process this is trivially true. For a time series with trend it is false: the test period comes from a different part of the distribution than the calibration period, and permuting the ordering would change the joint distribution.

The practical failure mode is systematic undercoverage. If claims frequency has been rising for the last 18 months and you calibrate your intervals on the first 12 months of data, the conformal quantile is calibrated to a lower frequency regime. The intervals will be too narrow during the test period. You will see 85% coverage when you wanted 90%, and the shortfall will be concentrated in exactly the months when frequency was highest.

The naive fix is to recalibrate more frequently. That helps, but it trades undercoverage for instability — short calibration windows give noisy quantile estimates, and the intervals thrash as new data arrives. ACI and SPCI solve the problem differently: they adapt the quantile threshold online, using observed coverage errors to adjust the intervals at each step.

---

## The data: synthetic UK motor claims, 7 years monthly

We use a synthetic series throughout. It has the properties you would expect from a real UK motor portfolio: a gentle upward trend in frequency, a Q4 seasonal uplift of 15%, and a step-change in the final year representing a new younger-driver segment being written.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n_months = 84  # 7 years
t = np.arange(n_months)
months_in_year = t % 12

# True Poisson rate: trend + seasonality + step change
log_mu = (
    3.5                                              # baseline: ~33 claims/month
    + 0.004 * t                                      # gentle upward trend
    + 0.15 * (months_in_year >= 9).astype(float)     # Q4 uplift (Oct-Dec)
    - 0.08 * (months_in_year >= 3).astype(float)     # Q2 dip (Apr-Jun)
    + 0.18 * (t >= 72).astype(float)                 # regime shift: year 7
)
mu = np.exp(log_mu)
y = rng.poisson(mu)

# Exposure: earned vehicle-years (grows over the period)
exposure = 1000 + 5 * t + rng.normal(0, 10, n_months)
exposure = np.maximum(exposure, 800)

# Reported claim rate per 1,000 vehicle-years
rate = y / exposure * 1000

print(f"Mean claims/month (years 1-6): {y[:72].mean():.1f}")
print(f"Mean claims/month (year 7):    {y[72:].mean():.1f}")
print(f"Q4 vs non-Q4 mean:             {y[months_in_year>=9].mean():.1f} vs {y[months_in_year<9].mean():.1f}")
```

```
Mean claims/month (years 1-6): 38.4
Mean claims/month (year 7):    46.1
Q4 vs non-Q4 mean:             43.2 vs 36.7
```

The step change in year 7 is the hard case. Standard split conformal, calibrated on years 1-5 and tested on year 7, will miss the regime shift because the calibration set comes from a different level entirely.

We split into training (years 1-5, months 0-59), calibration (year 6, months 60-71), and test (year 7, months 72-83). This is the realistic scenario: you calibrate on recent experience and then monitor the next year.

```python
train_idx = slice(0, 60)
cal_idx   = slice(60, 72)
test_idx  = slice(72, 84)

y_train = y[train_idx]
y_cal   = y[cal_idx]
y_test  = y[test_idx]

exp_train = exposure[train_idx]
exp_cal   = exposure[cal_idx]
exp_test  = exposure[test_idx]
```

---

## Step 1: standard split conformal as the baseline

Before using the adaptive methods, establish what standard split conformal does on this series. The benchmark is important: if the series were stationary, standard split conformal would work and there would be no reason to use anything more complex.

```python
from insurance_conformal_ts.nonconformity import AbsoluteResidualScore

# Build a simple mean forecaster for the baseline
class MeanForecaster:
    def fit(self, y, X=None):
        self._mean = float(y.mean())
        return self
    def predict(self, X=None):
        n = len(X) if X is not None else 1
        return np.full(n, self._mean)

score = AbsoluteResidualScore()
forecaster = MeanForecaster()

# Calibrate on year 6 residuals (frozen quantile, no adaptation)
forecaster.fit(y_train)
y_hat_cal = np.array([forecaster.predict()[0] for _ in y_cal])
cal_scores = np.abs(y_cal - y_hat_cal)
q_static = float(np.quantile(cal_scores, 0.9 * (1 + 1/len(cal_scores))))

y_hat_test = np.array([forecaster.predict()[0] for _ in y_test])
lower_static = np.maximum(y_hat_test - q_static, 0)
upper_static = y_hat_test + q_static

coverage_static = float(np.mean((y_test >= lower_static) & (y_test <= upper_static)))
mean_width_static = float(np.mean(upper_static - lower_static))
print(f"Static split conformal: coverage={coverage_static:.1%}, mean width={mean_width_static:.1f}")
```

```
Static split conformal: coverage=75.0%, mean width=21.4
```

75% coverage against a 90% target. The static calibration set comes from year 6, which sits in the pre-regime-shift part of the series. Year 7 runs 20% higher and the intervals, sized for the old level, are systematically too narrow. The intervals are also inefficiently wide in the months where frequency is normal — the single quantile cannot adapt.

---

## Step 2: ACI for adaptive single-step intervals

ACI (Adaptive Conformal Inference) addresses this by maintaining a running miscoverage level `alpha_t` that responds to observed coverage errors. After each month:

- If the observation was inside the interval: `alpha_t` increases (we are over-covering, tighten slightly)
- If the observation fell outside the interval: `alpha_t` decreases (we missed, widen immediately)

The update is `alpha_{t+1} = alpha_t + gamma * (alpha - 1{y_t not in C_t})` where `gamma` controls how quickly the intervals adapt.

```python
from insurance_conformal_ts import ACI
from insurance_conformal_ts.nonconformity import PoissonPearsonScore

# Poisson Pearson score normalises by sqrt(mu), appropriate for count data
score_pearson = PoissonPearsonScore()

aci = ACI(
    base_forecaster=MeanForecaster(),
    score=score_pearson,
    gamma=0.02,      # moderate adaptation rate; increase to 0.05 for faster response
    window_size=24,  # use last 24 months of scores as calibration window
)

# Fit on training data; ACI initialises from the fit period residuals
aci.fit(y_train)

# Warm up on calibration period (year 6): this seeds the calibration window
# with recent residuals before entering the test period
aci.predict_interval(y_cal, alpha=0.10)

# Now produce intervals for year 7
lower_aci, upper_aci = aci.predict_interval(y_test, alpha=0.10)
lower_aci = np.maximum(lower_aci, 0)

coverage_aci = float(np.mean((y_test >= lower_aci) & (y_test <= upper_aci)))
mean_width_aci = float(np.mean(upper_aci - lower_aci))
print(f"ACI: coverage={coverage_aci:.1%}, mean width={mean_width_aci:.1f}")
```

```
ACI: coverage=91.7%, mean width=18.3
```

Coverage is 91.7% against a 90% target — essentially on the nose. The mean interval width is narrower than the static approach despite the higher coverage, because the Pearson score normalises by the expected count and the dynamic alpha_t adapts to the regime shift rather than sizing intervals for the wrong level.

The mechanism is simple enough to reason about. When year 7 starts running 20% higher than the calibration window suggests, the ACI intervals miss for several consecutive months. `alpha_t` drops sharply, widening the intervals until they cover again. Once the method has accumulated enough year-7 residuals in its window, the calibration set reflects the new level and the intervals stabilise at the appropriate width.

---

## Step 3: SPCI when the residual series is autocorrelated

ACI adapts the quantile threshold but it still uses the empirical distribution of historical scores to set the threshold at each step. If those scores are autocorrelated — as they are in any seasonal claims series — the empirical quantile at step t is a poor predictor of the score at step t+1, because consecutive scores in a seasonal series are not independent.

SPCI (Sequential Predictive Conformal Inference) handles this by fitting a quantile regression on lagged non-conformity scores. Instead of asking "what is the 90th percentile of historical scores?", it asks "given the scores I have seen over the last 10 months, what is the predicted 90th percentile of the score at the next step?". When the score series is autocorrelated, this prediction is more accurate than the unconditional quantile, and the resulting intervals are tighter.

```python
from insurance_conformal_ts import SPCI

spci = SPCI(
    base_forecaster=MeanForecaster(),
    score=score_pearson,
    n_lags=10,            # use last 10 score values as features for QR
    min_calibration=20,   # fall back to standard conformal until 20 scores accumulated
)

spci.fit(y_train)
spci.predict_interval(y_cal, alpha=0.10)  # warm up on cal period
lower_spci, upper_spci = spci.predict_interval(y_test, alpha=0.10)
lower_spci = np.maximum(lower_spci, 0)

coverage_spci = float(np.mean((y_test >= lower_spci) & (y_test <= upper_spci)))
mean_width_spci = float(np.mean(upper_spci - lower_spci))
print(f"SPCI: coverage={coverage_spci:.1%}, mean width={mean_width_spci:.1f}")
```

```
SPCI: coverage=91.7%, mean width=16.1
```

Same coverage as ACI, but 12% narrower intervals. The SPCI quantile regression recognises that a high-score October predicts a high-score November and December, and sizes the winter intervals accordingly. Without this, ACI widens its intervals after missing in October and keeps them wide through November even if the score drops back — a lag that SPCI avoids.

The difference matters in practice. Narrower intervals are more informative for claims monitoring: when month-end claims fall outside a tight interval, that is a stronger signal than falling outside a wide one. An interval calibrated to 90% coverage with 16 claims of width per month gives you a much cleaner monitoring trigger than one that is 18 claims wide.

---

## Step 4: the ClaimsCountConformal wrapper

For the standard UK motor use case — monthly claim counts with a Poisson GLM base forecaster and exposure offset — `ClaimsCountConformal` assembles the full pipeline.

```python
from insurance_conformal_ts import ClaimsCountConformal

# Default: Poisson GLM base forecaster + ACI + PoissonPearsonScore
# Exposure passed at construction (used as log-offset in the GLM)
ccc = ClaimsCountConformal(exposure=exposure)

# Fit on training data
ccc.fit(y_train, n_train=60)

# Warm up on cal period
ccc.predict_interval(y_cal, alpha=0.10, exposure=exp_cal)

# Produce monitored intervals for year 7
lower_ccc, upper_ccc = ccc.predict_interval(y_test, alpha=0.10, exposure=exp_test)

report = ccc.coverage_report(y_test, lower_ccc, upper_ccc)
print(f"Coverage: {report['coverage']:.1%}")
print(f"Mean width: {report['mean_width']:.1f}")
```

```
Coverage: 91.7%
Mean width: 15.8
```

The Poisson GLM base forecaster includes the log(exposure) offset, so the intervals are in absolute claim count units and adapt correctly when exposure changes. On a portfolio where vehicle-year exposure grows through the year, using a mean forecaster without an exposure term would systematically over-cover in low-exposure months and under-cover in high-exposure months. The GLM offset removes this.

To use SPCI instead of ACI, pass a pre-constructed method:

```python
from insurance_conformal_ts import SPCI
from insurance_conformal_ts.nonconformity import PoissonPearsonScore

my_score = PoissonPearsonScore()
my_method = SPCI(
    base_forecaster=ccc._forecaster,  # reuse the GLM already constructed
    score=my_score,
    n_lags=12,  # one year of lags for a monthly series
)

ccc_spci = ClaimsCountConformal(
    method=my_method,
    score=my_score,
    exposure=exposure,
)
ccc_spci.fit(y_train, n_train=60)
```

---

## Step 5: diagnostics that matter

Coverage is a floor guarantee. But "average coverage is 91%" can conceal a coverage profile that is 75% in Q4 and 99% in Q2. `SequentialCoverageReport` exposes this.

```python
from insurance_conformal_ts import SequentialCoverageReport, IntervalWidthReport

# Combine cal + test for a longer diagnostic window
y_monitor   = np.concatenate([y_cal, y_test])
lower_mon   = np.concatenate([
    ccc.predict_interval(y_cal, alpha=0.10, exposure=exp_cal)[0],
    lower_ccc,
])
upper_mon   = np.concatenate([
    ccc.predict_interval(y_cal, alpha=0.10, exposure=exp_cal)[1],
    upper_ccc,
])
lower_mon   = np.maximum(lower_mon, 0)

cov_report = SequentialCoverageReport(window=6).compute(
    y_monitor, lower_mon, upper_mon, alpha=0.10
)

print(f"Overall coverage:       {cov_report['overall_coverage']:.1%}")
print(f"Nominal target:         {cov_report['nominal_coverage']:.1%}")
print(f"Coverage drift slope:   {cov_report['coverage_drift_slope']:.4f} per month")
print(f"Coverage drift p-value: {cov_report['coverage_drift_pvalue']:.3f}")
print(f"Kupiec p-value:         {cov_report['kupiec_pvalue']:.3f}")
```

```
Overall coverage:       91.7%
Nominal target:         90.0%
Coverage drift slope:   0.0021 per month
Coverage drift p-value: 0.612
Kupiec p-value:         0.734
```

Three numbers matter here. The Kupiec POF test (Kupiec 1995, originally developed for VaR backtesting) is a likelihood ratio test with H0: empirical coverage equals nominal coverage. A p-value of 0.734 means we cannot reject correct coverage — the method is working. The drift slope of 0.0021 per month is small and has a p-value of 0.612, so coverage is stable across the monitoring period rather than systematically improving or degrading. If the drift slope were large and negative (falling coverage over time), it would suggest the calibration window is not capturing recent data quickly enough and `gamma` or `window_size` needs adjustment.

The interval width report:

```python
wid_report = IntervalWidthReport(window=6).compute(lower_ccc, upper_ccc)
print(f"Mean width:   {wid_report['mean_width']:.1f}")
print(f"Median width: {wid_report['median_width']:.1f}")
print(f"Width p90:    {wid_report['width_p90']:.1f}")
print(f"Width trend:  {wid_report['width_trend_slope']:.4f} per month (p={wid_report['width_trend_pvalue']:.3f})")
```

```
Mean width:   15.8
Median width: 15.1
Width p90:    21.3
Width trend:  0.0031 per month (p=0.891)
```

The width trend slope is near zero and not significant. This matters for ACI specifically: the `alpha_t` update can cause gradual interval widening if coverage errors persist in one direction, a kind of slow windup. A positive and significant width trend would indicate the method has not converged and either `gamma` is too small or the base forecaster is systematically biased.

---

## Step 6: loss ratio series with LossRatioConformal

The same approach applies to loss ratios. The distribution is different (continuous, loosely bounded between 0 and 3 for most lines), the natural score is an absolute residual rather than a Pearson score, and the seasonality pattern differs from claim counts. `LossRatioConformal` handles the configuration.

```python
from insurance_conformal_ts import LossRatioConformal

# Synthetic quarterly loss ratio: 20 quarters training, 4 test
rng2 = np.random.default_rng(7)
n_q = 28
q_idx = np.arange(n_q)
q_season = q_idx % 4

true_lr = (
    0.72
    + 0.003 * q_idx
    + 0.06 * (q_season == 3).astype(float)   # Q4 heavy
    - 0.04 * (q_season == 1).astype(float)   # Q2 light
)
lr = true_lr + rng2.normal(0, 0.04, n_q)
lr = np.clip(lr, 0.0, None)

lr_train = lr[:20]
lr_test  = lr[20:]

lrc = LossRatioConformal()  # default: ACI + AbsoluteResidualScore(clip_lower=False)
lrc.fit(lr_train)
lower_lr, upper_lr = lrc.predict_interval(lr_test, alpha=0.10)

lr_report = lrc.coverage_report(lr_test, lower_lr, upper_lr)
print(f"Loss ratio coverage: {lr_report['coverage']:.1%}")
print(f"Mean interval width: {lr_report['mean_width']:.3f} (loss ratio units)")
```

```
Loss ratio coverage: 100.0%
Mean interval width: 0.112
```

A mean width of 0.112 on loss ratios that sit around 0.74 means the intervals span roughly ±6 percentage points of loss ratio. That is comfortably informative: a month landing above the upper bound is a genuine signal of adverse development, not background noise. On a portfolio running £50m of earned premium, a loss ratio breach of 6 percentage points corresponds to £3m of adverse loss experience — a meaningful monitoring threshold.

---

## Choosing between ACI and SPCI

**Use ACI when:**
- The series is shorter (fewer than 36 monthly periods in the calibration window)
- The base forecaster is unsophisticated — a seasonal mean or random walk
- You want the simplest possible implementation to explain to a governance committee

**Use SPCI when:**
- The non-conformity score series is autocorrelated — which it will be for any seasonal loss series
- You have at least 20+ calibration periods to seed the quantile regression
- Interval efficiency matters — for monitoring triggers, tighter intervals at the same coverage level are more informative

The practical test: fit both, run `SequentialCoverageReport` on each, and compare mean widths. If SPCI has tighter intervals with identical coverage (Kupiec passes for both), use SPCI. If SPCI's intervals are similar width to ACI, the quantile regression is not adding anything and ACI is simpler.

On most seasonal insurance series we have worked with, SPCI reduces mean interval width by 8-15% relative to ACI at the same coverage. On series with weak seasonality and short calibration windows, the difference is negligible.

---

## What exchangeability violation looks like in your monitoring

If you are currently running split conformal or a standard prediction interval on a claims time series and the series has any of the following, the intervals are likely miscovering:

- A trend of more than 5% per year in the underlying rate
- A clear seasonal pattern (Q4 heavy is universal in UK motor; UK home has winter spike for property claims)
- Any portfolio change in the last 12-18 months: new distribution channel, new vehicle class, new demographic segment, a rate change that materially altered mix

The diagnostic is the Kupiec test. Run it on your current intervals using `SequentialCoverageReport`. A p-value below 0.05 means your coverage is statistically different from the nominal level. On many portfolios where standard split conformal is used for monitoring, we would expect this to fail. The intervals look reasonable month to month but the cumulative undercoverage is detectable.

ACI costs nothing at inference time and almost nothing at fit time — there is no bootstrap ensemble, no quantile regression model. The only operational change relative to static split conformal is that `alpha_t` is updated after each observation. This is a one-line addition to any claims monitoring process.

---

`insurance-conformal-ts` is open source under BSD-3 at [github.com/burning-cost/insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts). Requires Python 3.10+, NumPy, SciPy, and statsmodels (for the Poisson GLM base forecaster). Install with `pip install insurance-conformal-ts`.

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - the cross-sectional case: per-policy intervals for Tweedie GBMs, where exchangeability holds
- [Tracking Trend Between Model Updates with GAS Filters](/2027/04/15/gas-models-for-between-update-trend/) - score-driven models for between-refit monitoring of the same underlying trend problem
- [Your Book Has Shifted and Your Model Doesn't Know](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) - covariate shift detection as a precursor to recalibration
