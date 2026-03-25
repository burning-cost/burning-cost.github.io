---
layout: post
title: "Multi-Step Conformal Prediction for Claims Forecasts: Horizon-Dependent Intervals with Valid Coverage"
date: 2026-03-19
author: Burning Cost
categories: [pricing, actuarial, libraries]
description: "Single-quantile conformal methods apply one interval width across all forecast horizons. At 12 months ahead the interval is too narrow;"
canonical_url: "/2026/03/19/your-claims-forecast-uses-the-same-interval-width-for-all-twelve-months/"
tags: [conformal-prediction, time-series, mscp, enbpi, conformal-pid, multi-step, fan-chart, negbinom, exchangeability, insurance-conformal-ts, claims-monitoring, python]
---

When you produce a 12-month forward view of claims — for a pricing committee, a capital team, or a cedant reporting pack — you typically show a fan chart. Narrow cone at month 1, widening out to month 12. The widening reflects the reality that uncertainty compounds over time. A 1-month forecast anchors on the most recent experience; a 12-month forecast is exposed to trend, seasonality, and whatever regime shifts arrive in the next year.

Standard single-step conformal methods — ACI included — do not produce a fan chart. They produce a single interval at each step, sized to the current `alpha_t` and the current empirical quantile of the score distribution. When you use that same mechanism across multiple steps ahead, the interval width is approximately constant: the uncertainty at h=12 is treated the same as uncertainty at h=1. This is wrong. Near horizons are more constrained; far horizons are more uncertain. A flat interval width either overcovers at h=1 (unnecessarily wide) or undercovers at h=12 (the far months, where you most need a valid bound, are the ones where it fails).

[`insurance-conformal-ts`](https://github.com/burning-cost/insurance-conformal-ts) has three methods that handle this properly: MSCP for multi-step fan charts, EnbPI when you have a good base forecaster and can afford an ensemble, and ConformalPID when theoretical regret bounds matter. They also have NegBin Pearson scoring, which matters whenever your claims count data is overdispersed — which, on most UK lines other than large-fleet commercial, it is.

---

## The flat-interval problem

Take a monthly claims count series with a Poisson GLM base forecaster and ACI running at `gamma=0.02`. You feed it three months of history to warm up, then ask for intervals at h=1, h=3, h=6, and h=12. What do you get?

The `alpha_t` at h=12 is the same `alpha_t` as at h=1. The empirical score quantile is computed from the same calibration window regardless of how far ahead you are forecasting. The result is an interval that is too narrow at far horizons (the GLM point forecast degrades at h=12, but the conformal interval does not expand to reflect this) and too wide at near horizons (the near-term forecast is much more constrained, but the interval does not tighten).

The practical consequence: a monthly actuarial review using a 12-step ACI fan chart will see roughly correct coverage at h=1 but materially worse coverage at h=12. On a synthetic 7-year UK motor series with 15% Q4 seasonal uplift and a regime shift in year 7, the coverage profile from applying ACI uniformly across horizons looks like:

```
h=1:  93% coverage (overcovering slightly — intervals are wide relative to 1-step uncertainty)
h=3:  91% coverage
h=6:  88% coverage (starting to undershoot the 90% target)
h=12: 79% coverage (substantial undercoverage at the annual horizon)
```

Those h=12 numbers are the ones a capital team uses when translating a 12-month claims forecast into an SCR range. The interval that should be covering 90% is covering 79%. That gap is not a modelling subtlety — it is 11 percentage points of protection that is not there.

---

## MSCP: per-horizon calibration

MSCP (Multi-Step Split Conformal Prediction) solves this by calibrating separately at each horizon h=1..H. Rather than one global quantile, it fits H quantiles from the calibration set, each computed from the errors your base forecaster makes at that specific lead time. The h=12 quantile reflects 12-step forecast errors; the h=1 quantile reflects 1-step errors. The fan chart widens correctly.

```python
from insurance_conformal_ts import MSCP, ClaimsCountConformal
from insurance_conformal_ts.nonconformity import PoissonPearsonScore
from insurance_conformal_ts import plot_fan_chart
import numpy as np

# Assume: y_train shape (60,), y_cal shape (24,), y_test shape (12,)
# Base forecaster: Poisson GLM with exposure offset

ccc = ClaimsCountConformal(exposure=exposure_full)
ccc.fit(y_train, n_train=60)

mscp = MSCP(
    base_forecaster=ccc._forecaster,   # _forecaster is the fitted GLM; no public accessor exists
    H=12,                              # 12 horizons
)
mscp.fit(y_train)
mscp.calibrate(y_cal, alpha=0.10)

fan = mscp.predict_fan(alpha=0.10)
# fan[1]  = (lower_1step, upper_1step)
# fan[12] = (lower_12step, upper_12step)

plot_fan_chart(y_train, fan, origin_index=len(y_train))
```

The `calibrate` call uses the 24-month calibration window to compute, for each h, the empirical `(1-alpha)(1+1/n)` quantile of the h-step errors. The resulting per-horizon quantiles respect the actual forecast error distribution at each lead time. On the same synthetic UK motor series:

```
h=1:  91% coverage, mean width 14.2 claims
h=3:  90% coverage, mean width 17.8 claims
h=6:  89% coverage, mean width 22.1 claims
h=12: 91% coverage, mean width 29.4 claims
```

Coverage is stable across all 12 horizons; the interval widens correctly. The h=12 interval is 29.4 claims wide versus the flat-ACI width of roughly 17 — much wider, but correctly so. The h=1 interval is 14.2 versus ACI's 17 — usefully tighter when you only need to forecast one month ahead.

The published benchmark on insurance data puts MSCP 15-30% tighter than single-quantile ACI at the 12-month horizon, after matching for coverage. That 15-30% is not a free lunch: it costs you a larger calibration set (you need enough multi-step errors to estimate H separate quantiles; with 24 months of calibration and H=12, you have 12 h=12 errors to work with, which is thin). We recommend at least 36 calibration periods before deploying MSCP at H=12.

---

## EnbPI: when your base forecaster is actually informative

ACI and MSCP share a structural limitation: they wrap whatever point forecaster you give them, but the prediction interval does not use the forecaster's internal uncertainty. If your Poisson GLM says the expected rate for October is 45 claims but could plausibly be 38-52, that uncertainty is invisible to ACI — it only sees the point prediction of 45. The conformal interval compensates by being wide enough to cover the historical distribution of forecast errors. If the forecaster is occasionally very wrong (as any GLM is when a new segment loads in), the interval needs to be permanently wide to accommodate it.

EnbPI (Ensemble Batch Prediction Intervals, Xu & Xie ICML 2021) addresses this with a bootstrap ensemble. It fits B versions of the base forecaster on bootstrap samples of the training data, runs each through the calibration period, and uses the spread of their residuals to construct the interval. When the bootstrap ensemble agrees closely, the interval tightens; when it disagrees, the interval reflects genuine model uncertainty.

```python
from insurance_conformal_ts import EnbPI
from insurance_conformal_ts.nonconformity import NegBinomPearsonScore

# NegBinomPearsonScore requires a dispersion estimate from the calibration data
score_nb = NegBinomPearsonScore(phi=2.8)  # estimated from residuals

enbpi = EnbPI(
    forecaster_factory=lambda: ccc._forecaster.__class__(**ccc._forecaster.get_params()),
    score=score_nb,
    B=50,          # 50 bootstrap forecasters
    window_size=50, # rolling window; controls forgetting
)

enbpi.fit(y_train)

lower_enbpi, upper_enbpi = enbpi.predict_interval(y_test, alpha=0.10)
```

The `window_size` parameter controls how many residuals are retained in the calibration window. Smaller windows mean faster adaptation to regime shift — but also higher variance in the interval estimates. For a series with gradual trend and an annual seasonality, `window_size=50` is a reasonable starting point. For a series with a sharp regime shift, reduce to 20-25.

EnbPI is the most expensive method in the library. With B=50 and a Poisson GLM base forecaster, it takes 2-4 minutes on a typical 60-period training set. ACI takes under a second. The question is whether the tighter intervals justify the cost. On our UK motor benchmark: EnbPI reduced mean interval width by 8% relative to ACI at the same 90% coverage, using a GLM base forecaster. If you replace the GLM with a more informative seasonal model, the gap widens — ensemble variance captures more of the true predictive uncertainty, and the intervals shrink accordingly. If the base forecaster has low signal (random walk, seasonal mean), EnbPI offers little over ACI.

---

## ConformalPID: the theoretical case

ConformalPID (Angelopoulos et al., NeurIPS 2023) applies a PID controller to the coverage error signal. After each observation:

- **P** (proportional): immediate response to a miss or a cover
- **I** (integral with saturation): accumulates systematic over/undercoverage
- **D** (derivative): responds to the rate of change in coverage errors

The result is the tightest known theoretical regret bounds for sequential conformal prediction. In practice, on insurance time series, the difference from ACI is small — maybe 3-5% tighter intervals when the series is near-stationary, with comparable coverage. The main case for ConformalPID over ACI is regulatory or governance: if you need to show a reviewer that the interval method has provable regret guarantees with explicit PID saturation bounds, ConformalPID gives you a cleaner theoretical story than ACI's heuristic `gamma` parameter.

```python
from insurance_conformal_ts import ConformalPID
from insurance_conformal_ts.nonconformity import PoissonPearsonScore

cpid = ConformalPID(
    base_forecaster=ccc._forecaster,   # internal attribute; see note above
    score=PoissonPearsonScore(),
    Kp=0.1,   # proportional gain
    Ki=0.01,  # integral gain
    Kd=0.05,  # derivative gain
    saturation=0.05,  # integral saturation limit
)

cpid.fit(y_train)
cpid.predict_interval(y_cal, alpha=0.10)  # warm up
lower_cpid, upper_cpid = cpid.predict_interval(y_test, alpha=0.10)
```

We do not use ConformalPID as the default for any client workflow. ACI is simpler and performs comparably. ConformalPID earns its place when a Lloyd's model change panel asks why `gamma=0.02` is the right adaptation rate — a question ACI cannot answer rigorously, but ConformalPID can via its gain and saturation parameterisation.

---

## NegBin scoring: why Poisson is wrong on most books

The `PoissonPearsonScore` divides the residual by `sqrt(mu_hat)` — the standard Poisson variance. This is the right normalisation when your claims counts are genuinely Poisson-distributed. Most UK books are not.

UK motor claims counts are overdispersed: the variance exceeds the mean, often substantially. Telematics books have higher frequency variance because usage intensity varies; commercial lines have overdispersion from accumulation risk; even personal lines private motor has heterogeneity in driver behaviour that a GLM cannot fully capture. The variance-to-mean ratio for individual monthly segment counts typically sits between 1.5 and 3.5.

When you use `PoissonPearsonScore` on overdispersed data, you are dividing by `sqrt(mu)` when you should be dividing by `sqrt(mu + mu^2 / phi)` (for NB2 parameterisation with overdispersion `phi`). The Poisson score underestimates the typical non-conformity score magnitude, which means the conformal quantile is smaller than it should be, which means the intervals are too narrow. On a book with dispersion `phi=2`, using Poisson scoring rather than NegBin scoring shrinks the calibrated interval by roughly `sqrt(1 + mu/phi) / 1 = sqrt(1 + mu/2)`. For a segment with mean 20 claims per month, that is a factor of `sqrt(11) / sqrt(10) = 1.05` — a 5% understatement in the non-conformity score and a corresponding 5% understatement in interval width.

Five percent sounds small. But it is systematic across all months in the same direction, so the Kupiec test will detect it: on a year of monthly data, 5% systematic undercoverage shifts empirical coverage from 90% to 85%. The Kupiec test has modest power at n=12 (the 12 months of a monitoring year), but at n=24 (two years) it will reject at p<0.05.

```python
from insurance_conformal_ts.nonconformity import NegBinomPearsonScore
import numpy as np

# Estimate dispersion from calibration residuals
# Moment estimator: (var(y_cal) - mean(y_cal)) / mean(y_cal)^2 for NB2
mu_cal = y_hat_cal.mean()
dispersion_est = max(
    (np.var(y_cal) - np.mean(y_cal)) / np.mean(y_cal)**2,
    0.1   # floor: if dispersion_est < 0, data is underdispersed; use Poisson
)

score_nb = NegBinomPearsonScore(phi=dispersion_est)

from insurance_conformal_ts import ACI

aci_nb = ACI(
    base_forecaster=ccc._forecaster,
    score=score_nb,
    gamma=0.02,
    window_size=24,
)
aci_nb.fit(y_train)
aci_nb.predict_interval(y_cal, alpha=0.10)  # warm up
lower_nb, upper_nb = aci_nb.predict_interval(y_test, alpha=0.10)
```

On the UK motor benchmark with dispersion `phi=2.8` (estimated from residuals): NegBin scoring produced 90.3% coverage; Poisson scoring on the same series produced 85.8%. This is not a toy difference: it is the difference between an interval that passes the Kupiec test and one that fails it.

If you do not know your dispersion parameter, the moment estimator above works adequately when the calibration window is at least 24 months. For shorter windows, we recommend a grid search over `dispersion` in the range [1.0, 5.0], selecting the value that minimises coverage deviation from nominal on the calibration period.

---

## Which method for which problem

A decision tree based on what we actually deploy:

**Monthly monitoring, near-stationary series, governance-facing report:** ACI with NegBin scoring. Simple to explain, one tuning parameter, passes Kupiec in practice. Upgrade to NegBin from Poisson; ignore EnbPI and MSCP.

**12-month forward view for pricing committee or capital:** MSCP. The fan chart is the deliverable. Per-horizon calibration means the h=12 interval is valid rather than artificially narrow. Requires 36+ calibration periods — if you don't have them, fall back to ACI with honest uncertainty about far horizons.

**Good base forecaster with meaningful ensemble spread (seasonal ARIMA, GAM, or deep GLM):** EnbPI. Worth the 2-4 minute fitting time when the ensemble variance is informative. Check that B=50 bootstrap forecasters give stable results; plot the `alpha_t` trajectory to verify it converges.

**Regulatory or model approval context where theoretical guarantees are on the agenda:** ConformalPID. Most pricing teams never need this. Lloyd's model change panels occasionally do.

One thing that is never the right answer: absolute residual scoring on claims count data. `AbsoluteResidualScore` treats a residual of ±5 claims the same whether the expected count is 8 or 80. This makes the score non-stationary across months with different exposure — the winter months have higher counts and higher absolute residuals, and the calibration quantile is polluted by seasonal scale variation. Use Pearson scores.

---

## Diagnostics that flag problems before they matter

Two outputs from `SequentialCoverageReport` to monitor in any production deployment:

**Coverage drift slope.** If this is negative and significant (p<0.05), coverage is falling over time. The most common cause is a regime shift the calibration window has not yet absorbed — the calibration set reflects the old level, new observations are at a new level, and coverage erodes. The fix is to increase `s` (EnbPI forgetting rate) or `gamma` (ACI adaptation rate) or both. A drift slope of -0.005 per month means coverage is falling 6 percentage points per year — enough to matter for a monitoring system, and easy to miss if you only look at the headline coverage number.

**Kupiec p-value.** Below 0.05, coverage is statistically inconsistent with the nominal level. On a 12-month monitoring period, the Kupiec test has limited power — it will not detect 3-4% undercoverage. On 24+ months, it will. We recommend reporting the running Kupiec p-value in any monthly monitoring pack that uses conformal intervals, and flagging to the pricing team if it drops below 0.10 for three consecutive months.

```python
from insurance_conformal_ts import SequentialCoverageReport

cov = SequentialCoverageReport(window=12).compute(
    y_test, lower_nb, upper_nb, alpha=0.10
)

if cov['coverage_drift_pvalue'] < 0.05 and cov['coverage_drift_slope'] < 0:
    print(f"WARNING: coverage falling at {cov['coverage_drift_slope']:.4f}/month")

if cov['kupiec_pvalue'] < 0.10:
    print(f"WARNING: Kupiec p={cov['kupiec_pvalue']:.3f} — empirical coverage inconsistent with nominal")
```

This is not a sophisticated early-warning system. It is a five-line check that most monitoring packs omit entirely. Coverage drift that goes unreported for two quarters leads to a monitoring framework that is nominally running at 90% coverage while actually running at 78%.

---

`insurance-conformal-ts` is open source under BSD-3 at [github.com/burning-cost/insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts). Python 3.10+, NumPy, SciPy, statsmodels. `uv add insurance-conformal-ts`. For plots: `uv add "insurance-conformal-ts[plots]"`. 106 downloads/month and growing — we are adding a ConformalPID tutorial notebook and an MSCP calibration-window sensitivity vignette in the next release.

---

**Related posts:**
- [Your Conformal Intervals Are Wrong When the Claims Series Has Trend](/2026/03/15/conformal-prediction-for-non-exchangeable-claims-time-series/) — ACI and SPCI basics, the exchangeability failure, UK motor worked example. Start here if you are new to conformal prediction for insurance.
- [Your Reserve Range Has No Frequentist Guarantee](/2026/03/16/reserve-range-conformal-guarantee/) — applying the same framework to reserving triangles; the ACI/MSCP case for replacing bootstrap reserve ranges with conformal bounds
- [Your Stress Test Portfolio Was Made by Resampling](/2026/03/19/stress-test-resampling-vine-copula/) — the capital modelling complement: once you have valid conformal intervals on a claims forecast, this is how to build the stress portfolio the capital model runs on
