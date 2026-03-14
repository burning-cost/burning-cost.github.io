---
layout: post
title: "Your Claims Forecast Interval Is Wrong. Here Is Why."
date: 2026-03-12
categories: [libraries, pricing, conformal-prediction]
tags: [conformal-prediction, time-series, claims-forecasting, ARIMA, MSCP, ACI, SPCI, PID, EnbPI, Poisson, pricing, insurance-conformal-ts]
description: "ARIMA intervals assume Gaussian residuals and stationarity. Insurance claims data violates both by construction. insurance-conformal-ts implements five distribution-free methods for sequential prediction intervals, including MSCP — the only method that clearly met the 90% coverage target in the 2026 benchmark."
---

When you present those intervals at a pricing committee without mentioning either assumption, you are not being precise. You are being precisely wrong.

Go to a quarterly pricing review at any UK insurer and ask where the uncertainty band around the claims forecast comes from. You will get one of three answers. First: "It's an ARIMA interval." Second: "We widened the ARIMA interval by judgement." Third, less common but honest: "We don't really have one."

The ARIMA interval is the interesting case. It is not worthless — it does reflect something about forecast uncertainty — but it rests on two assumptions that insurance data violates by construction. First, it assumes Gaussian residuals. Motor claims severity is not Gaussian; it is right-skewed, fat-tailed, and was running at 20% year-on-year inflation in late 2022. Second, it assumes stationarity. A series with claims inflation trend, exposure changes, NCD mix drift, and the occasional Ogden rate ruling is not stationary.

We built [`insurance-conformal-ts`](https://github.com/burning-cost/insurance-conformal-ts) to provide a rigorous alternative.

## The exchangeability problem

Our earlier library `insurance-conformal` gives distribution-free prediction intervals for cross-sectional pricing models: fit a Poisson GBM, calibrate non-conformity scores, get valid coverage without assuming anything about the residual distribution. It works because split conformal prediction's coverage guarantee holds whenever the calibration set and the test point are *exchangeable* — drawn from the same distribution, in any order.

A time series is exchangeable in exactly no meaningful sense. The residual from quarter t correlates with the residual from quarter t-1. The variance in 2022 was higher than the variance in 2020. The mean shifted when the Ogden rate changed. Swapping observations in time changes the problem.

Apply standard conformal prediction to a time series anyway, and the coverage guarantee disappears. You still get an interval, but you have no mathematical assurance it contains the true value at any particular frequency.

This is not a theoretical concern. A 2026 benchmark (Sabashvili, arXiv:2601.18509) tested several methods on a large monthly sales dataset. Standard split conformal intervals failed to meet the 90% coverage target. So did EnbPI. The methods that actually work have different theoretical foundations.

## Five methods, one insurance workflow

`insurance-conformal-ts` implements five methods for sequential prediction intervals, each solving a different failure mode of exchangeability.

**ACI (Gibbs and Candès, 2021)** is the simplest. It treats the coverage level as a dynamic variable, updated by online gradient descent. If the last interval missed the true value, alpha tightens slightly (wider intervals next time). If it covered, alpha relaxes. After a claims inflation shock — the post-COVID used car repricing that hit motor severity in 2021–22, for instance — ACI's intervals widen automatically over the following months as coverage failures accumulate. It is not a perfect alarm, but it is a real one.

**EnbPI (Xu and Xie, 2021 ICML)** trains a bootstrap ensemble of forecasters and maintains a sliding window of recent residuals for calibration. No model re-fitting. The coverage guarantee relaxes the iid requirement to *strongly mixing* errors, which is reasonable for most stationary series. Its weakness: it failed the 90% coverage benchmark in 2026. We include it as the baseline and because it is in MAPIE; for regulatory use, it is not the first choice.

**SPCI (Xu and Xie, 2023 ICML)** fits a quantile regression on lagged residuals to predict the next residual's quantile conditional on recent history. The result is *conditional* coverage — intervals calibrated to current market conditions, not just averaged over all time. For a Poisson GLM without ARIMA residual whitening, which is the standard pricing setup, SPCI exploits the autocorrelation left in the residuals. This is where it earns its keep.

**Conformal PID (Angelopoulos, Candès and Tibshirani, NeurIPS 2023)** reframes interval calibration as a control theory problem. ACI is proportional control: corrects coverage error proportionally each step. PID adds integral control. If a Poisson GLM fitted in 2022 is systematically underpredicting because claims inflation has run above the model's trend assumption, the coverage shortfall accumulates over time. The integral term detects the systematic bias and applies a permanent offset correction — directly analogous to applying a claims trend adjustment. Best theoretical guarantees of the five methods.

**MSCP (arXiv:2601.18509, January 2026)** is the benchmark winner and the reason we built this library. Multi-Step Split Conformal Prediction handles the multi-horizon forecasting case that every quarterly pricing review actually needs. For each forecast horizon h = 1, 2, ..., H, it calibrates a separate quantile from residuals at that specific horizon. 3-step-ahead forecast errors are larger than 1-step-ahead errors — always — and pooling them into a single calibration set conflates this. MSCP gives calibrated, horizon-specific intervals, and it clearly met the 90% coverage target in the 2026 benchmark.

None of EnbPI, SPCI, or MSCP existed in any pip-installable library as of March 2026. MAPIE v1.3.0 has EnbPI and ACI; GitHub issues requesting SPCI (#370) have been in the backlog since 2023 with no delivery date. PID's reference code is conda-only. MSCP is not packaged anywhere.

## The insurance workflow that is actually missing

The algorithmic gap is real, but it is not the whole gap. None of the academic implementations know what a Poisson GLM is.

Standard conformal uses absolute residuals: |y_t - ŷ_t|. For count data, this is exposure-dependent. A portfolio with 10,000 policies has higher absolute claim count variance than one with 1,000 at the same underlying frequency. Use absolute residuals as your non-conformity score, and high-exposure periods dominate the calibration quantile — making intervals too wide for small portfolios and too narrow for large ones.

The correct score is the Pearson residual:

```
s_t = (y_t - mu_t) / sqrt(mu_t)                     [Poisson]
s_t = (y_t - mu_t) / sqrt(mu_t * (1 + mu_t/phi))    [Negative Binomial]
```

where mu_t = lambda_t * E_t (rate times exposure). These are asymptotically N(0,1) under a correctly specified model, so the quantiles are well-calibrated and exposure-adjusted automatically. The library also implements exposure-adjusted scores (y/E - mu/E) for direct rate comparison.

None of the time series conformal papers use these score types. They are defined in the insurance-conformal literature (Manna et al., ASMBI 2025) for cross-sectional models. `insurance-conformal-ts` ports them into the sequential calibration framework.

## Code

The primary use case is multi-step quarterly forecasting:

```python
from insurance_conformal_ts import PoissonTimeSeriesConformal

# y: claim counts per quarter, e: earned exposure per quarter
model = PoissonTimeSeriesConformal(y_train, exposure_train, method="mscp")
model.fit()

# Forecast 4 quarters ahead
forecast = model.predict(X_future, exposure_future, horizon=4)

# Fan chart for the underwriting committee
forecast.plot_fan()

# Horizon-specific coverage summary
print(forecast.coverage_summary())
```

`ClaimsCountConformal` is a higher-level wrapper that handles the full pipeline: Poisson GLM with offset via statsmodels, Pearson residual calibration, and a `SequentialCoverageReport` including a Kupiec test. For loss ratio work, `LossRatioConformal` accepts earned premium as the exposure denominator.

The `SequentialCoverageReport` is the deliverable for model documentation. It shows rolling coverage by window, Kupiec test result, interval width series, and a flag if coverage has drifted more than 5 percentage points from nominal. The point is not just to produce intervals — it is to give you something auditable.

## What to use when

For a quarterly pricing review with 40–80 quarters of data:

- **Multi-step fan chart (h=1..4):** MSCP. It met the benchmark. It is the simplest of the five methods for multi-horizon use.
- **Loss ratio trend with automatic inflation adaptation:** ACI or PID. ACI is simpler; PID is better if the model has a known systematic trend error.
- **Conditional intervals that respect autocorrelation:** SPCI. Better interval width than MSCP, but requires T > 50 observations and performs best when residuals are autocorrelated (Poisson GLM without ARIMA whitening).
- **Solvency II reserve uncertainty:** MSCP at 99.5% level. The discrete Poisson distribution makes conformal intervals slightly conservative, which is acceptable for regulatory use.

Do not use SPCI or EnbPI for regulatory deliverables if your series is shorter than 50 observations. At T = 30, the quantile regressor in SPCI does not have enough data to stabilise. MSCP is the most robust to short series.

## Honest limitations

The 2026 benchmark tested AutoARIMA as the base forecaster. SPCI failed there because AutoARIMA whitens residuals, leaving nothing for SPCI's quantile regressor to exploit. For insurance: a Poisson GLM without explicit AR terms will typically have autocorrelated residuals, which is better for SPCI. But we cannot guarantee this from first principles — run the `SequentialCoverageReport` and check.

For Poisson data with small expected counts (mu < 10), prediction interval lower bounds may be negative after back-transformation. The library clips at zero and documents the conservative bias. For very large portfolios (mu > 500), the Poisson approximation is good; for small commercial books, consider whether a discrete distribution wrapper is appropriate.

MSCP's coverage guarantee is per-horizon. It does not guarantee that all four horizons are simultaneously covered. If you need joint coverage across all forecast steps — the full fan chart inside the envelope — use the `JointMSCP` variant, which applies a Bonferroni correction and is correspondingly wider.

None of this replaces knowing your data. Conformal methods are distribution-free but not assumption-free. They assume the future distribution is not too different from the calibration window. If you are launching a new product line or entering a class affected by a structural regulatory change, recalibrate before presenting intervals.

## How it fits the stack

`insurance-conformal-ts` sits alongside, not replacing, `insurance-conformal`. That library handles cross-sectional intervals: given 50,000 policies in a pricing model, what is the prediction interval for each individual policy's claims? This library handles sequential intervals: given 80 quarters of aggregate claims, what is the interval around each of the next four quarters' totals?

The two libraries share non-conformity score design. They do not share methods.

---

The library is at [`github.com/burning-cost/insurance-conformal-ts`](https://github.com/burning-cost/insurance-conformal-ts). The notebook `notebooks/01_mscp_motor_quarterly.py` runs the full MSCP workflow on synthetic UK motor data with claims inflation regime shifts. Five modules, 150 tests.

The committee slide with the ARIMA interval will exist for a while yet. But now the interval next to it can be labelled "conformal, 90%, Kupiec: pass" — and if someone asks where the error bars come from, you have an answer.

---

*insurance-conformal-ts is part of the Burning Cost open-source stack. Related libraries: [insurance-conformal](https://github.com/burning-cost/insurance-conformal) (cross-sectional prediction intervals), [insurance-garch](https://github.com/burning-cost/insurance-garch) (claims inflation volatility modelling), [insurance-trend](https://github.com/burning-cost/insurance-trend) (severity and frequency trend analysis).*

---

**Related articles from Burning Cost:**
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [GARCH Modelling for Insurance Inflation Uncertainty](/2026/03/25/insurance-gas/)
- [Trend Analysis for Insurance Pricing](/2026/03/13/insurance-trend/)
