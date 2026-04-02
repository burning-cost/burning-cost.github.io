---
layout: post
title: "MAPIE vs insurance-conformal: Why Generic Conformal Prediction Breaks on Insurance Data"
date: 2026-03-20
author: Burning Cost
categories: [conformal-prediction, libraries, comparisons]
description: "MAPIE is the standard Python library for conformal prediction, but it wasn't designed for insurance. Here is what goes wrong with exposure-weighted portfolios and Tweedie models, and what insurance-conformal does differently."
tags: [MAPIE, conformal-prediction, insurance-conformal, Tweedie, prediction-intervals, coverage, exposure, Python]
---

If you search "conformal prediction Python", MAPIE is the first result. It deserves to be. It is well-engineered, well-documented, scikit-learn compatible, and used in production across financial services, healthcare, and manufacturing. For classification and general regression problems it is the right tool.

For insurance pricing models it breaks in ways that are not immediately obvious.

This post explains specifically what goes wrong, shows the code comparison, and describes what [insurance-conformal](/insurance-conformal/) does differently. If you are not working on insurance pricing, stop here and use MAPIE.

---

## Background: what conformal prediction does

Split conformal prediction is simple in principle. You hold out a calibration set that your model never trained on. You compute a non-conformity score - some measure of how wrong your model's prediction is - for every calibration observation. You sort those scores and take the `ceil((n+1)(1-alpha)/n)` quantile. Call it `q_hat`. Then your prediction interval for a new observation is:

```
[yhat - q_hat, yhat + q_hat]
```

The coverage guarantee - `P(y in interval) >= 1 - alpha` - holds without any distributional assumption, as long as calibration and test data are exchangeable.

MAPIE implements this correctly. The guarantee is genuine. The problem is what you put in as the non-conformity score.

---

## Problem 1: the wrong non-conformity score for insurance

MAPIE's default non-conformity score is the absolute residual `|y - yhat|`. This is correct for problems where errors scale roughly uniformly with the response.

Insurance claims do not behave like that. If your model predicts £500 for a low-risk policy and the actual claim is £700, the error is £200. If your model predicts £5,000 for a high-risk policy and the actual claim is £5,200, the error is also £200. MAPIE treats these identically. They are not identical: the second policy performed broadly as expected, the first is a significant overshoot.

For Tweedie and Poisson models, variance scales as `mu^p` where `p` is the Tweedie power. The correct non-conformity score accounts for this:

```
score(y, yhat) = |y - yhat| / yhat^(p/2)
```

This is the Pearson residual, scaled by the expected standard deviation under the Tweedie family. Using it rather than the raw residual means your calibration quantile is computed on a scale where residuals are actually comparable across risk levels.

**MAPIE with default score:**

```python
from mapie.regression import MapieRegressor

mapie = MapieRegressor(estimator=model, method="plus", cv="prefit")
mapie.fit(X_cal, y_cal)
y_pred, y_pi = mapie.predict(X_test, alpha=0.10)
# y_pi shape: (n, 2, 1) → lower and upper bounds
```

This works. The coverage guarantee holds. But the calibration quantile is derived from absolute residuals on insurance data where claims range from £0 to £50,000. On a heteroskedastic book, the calibration quantile will be inflated by large residuals from high-risk policies - which means intervals are unnecessarily wide for low-risk policies and structured uniformly when they should be narrower for low-risk and proportionally wider for high-risk.

**insurance-conformal with pearson_weighted score:**

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)
# Polars DataFrame: columns lower, point, upper
```

The score hierarchy matters. For Tweedie/Poisson models, narrowest intervals come from `pearson_weighted`, then `deviance`, then `anscombe`, then raw `pearson`, then `raw`. Coverage is guaranteed in all cases - but `raw` produces intervals that are 13–14% wider than necessary on a UK motor book where high-mean risks are genuinely more dispersed.

---

## Problem 2: no exposure handling

Every insurance model is a rate model. Expected claims cost is rate × exposure. A policy that ran for six months contributes 0.5 years of exposure; a policy cancelled after two weeks contributes 0.038 years. If you treat these the same as annual policies in your calibration set, your non-conformity scores are not exchangeable - short-exposure policies systematically have smaller residuals not because your model is better at predicting them, but because they had less time to generate claims.

MAPIE has no mechanism for this. It treats calibration observations as exchangeable, which they are not when exposure varies.

insurance-conformal accepts an `exposure` array at calibration time and adjusts the non-conformity scores accordingly:

```python
# exposure_cal: numpy array of policy durations (0.083 to 1.0)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
intervals = cp.predict_interval(X_test, alpha=0.10, exposure=exposure_test)
```

Without exposure weighting, the calibration quantile is contaminated by the easier-to-predict short-duration policies. The resulting intervals will be too narrow for annual policies - the opposite of what you want.

---

## Problem 3: no coverage diagnostics by risk segment

The conformal coverage guarantee is marginal: `P(y in interval) >= 1 - alpha` averaged over all observations. A model can achieve 90% aggregate coverage while covering only 65% of high-risk policies. MAPIE gives you aggregate coverage. It does not give you a tool to check whether that coverage is uniform across risk deciles.

```python
# MAPIE — you have to write this yourself
coverage = np.mean((y_test >= y_pi[:, 0, 0]) & (y_test <= y_pi[:, 1, 0]))
print(f"Aggregate coverage: {coverage:.3f}")
# No per-decile breakdown, no segment analysis
```

```python
# insurance-conformal — built in
diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
#    decile  mean_predicted  n_obs  coverage  target_coverage
# 0       1        1,035        1000     0.929             0.90
# 1       2        1,184        1000     0.924             0.90
# ...
# 9      10        2,344        1000     0.906             0.90

cp.summary(X_test, y_test, alpha=0.10)
```

This matters for pricing. A reinsurance attachment point is set based on the high-risk tail of the book. If your intervals under-cover the top decile, you are making capacity decisions on unreliable uncertainty estimates. Aggregate coverage statistics will not reveal this.

---

## Problem 4: no frequency-severity support

Non-life pricing separates claims into frequency (how often) and severity (how much). The expected aggregate loss is `E[frequency] × E[severity | claim]`. Conformal prediction for this decomposition has a specific subtlety: you cannot use observed claim counts at calibration time, because that creates a distributional mismatch between calibration scores and test scores that breaks the coverage guarantee.

The correct approach, from Graziadei et al. (arXiv:2307.13124), is to feed the *predicted* frequency from the frequency model into the severity model at both calibration and test time. MAPIE cannot do this - it is a single-model wrapper.

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from insurance_conformal.claims import FrequencySeverityConformal

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(),
    sev_model=GammaRegressor(),
)

fs.fit(X_train, d_train, y_train)   # d_train = observed claim counts
fs.calibrate(X_cal, d_cal, y_cal)  # d_cal passed for validation only; scores use mu_hat(x)
intervals = fs.predict_interval(X_test, alpha=0.10)
```

The coverage guarantee holds because calibration scores use `mu_hat(x)`, not observed counts. This is not a workaround - it is the theoretically correct protocol for two-stage models.

---

## Benchmark: what the score choice costs

The numbers below are from the insurance-conformal GBM benchmark: 50,000 synthetic UK motor policies, CatBoost Tweedie(p=1.5) as the point forecast, heteroskedastic Gamma DGP where high-mean risks are genuinely more dispersed than Tweedie(1.5) predicts. Temporal 60/20/20 split (30k train, 10k calibration, 10k test). 90% target coverage. Databricks serverless, seed=42.

We have not benchmarked MAPIE directly on this dataset - the comparison to MAPIE is structural (absolute vs Pearson-weighted score), not a head-to-head run. The headline finding is what the correct score choice achieves compared to the parametric baseline:

_Note: these benchmarks compare insurance-conformal's scoring modes against a parametric Tweedie baseline, not a direct MAPIE head-to-head. The structural argument for why MAPIE's absolute-residual approach produces wider intervals is discussed below._

| Method | Aggregate coverage | Worst-decile coverage | Mean interval width (£) |
|--------|-------------------|----------------------|------------------------|
| Parametric Tweedie (global sigma) | 0.931 | 0.904 | 4,393 |
| insurance-conformal (`pearson_weighted`) | 0.902 | 0.879 | 3,806 |
| insurance-conformal (locally-weighted) | 0.903 | **0.906** | 3,881 |

The parametric approach estimates a single sigma on the calibration set. Because the DGP has genuinely higher dispersion at higher means, the single sigma overestimates uncertainty for low-risk policies (aggregate coverage 93.1% signals over-width) while barely holding the top-decile target. Conformal `pearson_weighted` corrects this: 90.2% aggregate, 13.4% narrower intervals. The locally-weighted variant adds a secondary model to predict residual spread and achieves 90.6% in the top decile - useful when per-decile coverage matters for reinsurance attachment decisions.

Using MAPIE with default absolute residuals on this data would produce valid aggregate coverage but wider intervals than `pearson_weighted` - by the same mechanism as the raw score, which the benchmark shows costs 13–14% in width. That is the structural argument, not a measured number.

Full benchmark: `benchmarks/benchmark_gbm.py` in the insurance-conformal repository.

---

## When to use MAPIE

Use MAPIE for any conformal prediction problem that is not insurance pricing:

- Classification (it has dedicated classifiers via `MapieClassifier`)
- Regression problems where errors scale roughly uniformly with the response
- Any scikit-learn compatible model where you want confidence sets without parametric assumptions
- Multi-label problems, anomaly detection wrappers

MAPIE's `method="plus"` (jackknife+) is theoretically superior to split conformal for small calibration sets. MAPIE handles cross-conformal and CV+ variants that insurance-conformal does not implement. If you are a data scientist working outside insurance, MAPIE is the right default.

---

## When to use insurance-conformal

- Tweedie or Poisson pricing models with heteroskedastic residuals
- Any portfolio with genuine exposure variation (monthly direct debits, mid-term adjustments)
- Frequency-severity decomposition models
- When you need per-decile coverage diagnostics rather than aggregate statistics
- When you need premium sufficiency control (`PremiumSufficiencyController`) rather than raw interval coverage
- When you want SCR stress-test bounds at the 99.5th percentile for internal model validation

---

## Installation

```bash
uv add insurance-conformal

# with CatBoost:
uv add "insurance-conformal[catboost]"

# with everything:
uv add "insurance-conformal[all]"
```

The library has no MAPIE dependency. The split conformal algorithm is 20 lines of code for the core quantile function plus the score implementations. We chose to own this rather than wrapping MAPIE, because MAPIE does not expose the score functions we need as a stable API.

**Version note:** The structural comparisons in this post were written with MAPIE 0.8.x in mind. MAPIE's API has changed across versions (particularly the `MapieRegressor` interface and `method` arguments). If you are using a different version, check the MAPIE changelog before assuming the code snippets above translate directly.

---

## Summary

MAPIE is an excellent library. If you are doing conformal prediction outside insurance, use it. The problem is not that MAPIE is wrong - it is that insurance data violates the implicit assumptions built into a general-purpose implementation.

Specifically: insurance residuals are not homoskedastic, exposure creates non-exchangeability if ignored, the coverage guarantee needs to be checked by risk segment rather than in aggregate, and frequency-severity models require a specific calibration protocol that single-model wrappers cannot replicate.

[insurance-conformal](https://github.com/burning-cost/insurance-conformal) handles all of this. It is distribution-free, has no parametric assumptions, and produces 90% prediction intervals that hold in aggregate - and, with the locally-weighted variant, hold across risk deciles on a heterogeneous UK motor book.

The benchmark numbers are in `benchmarks/benchmark_gbm.py`. Run them.

---

**Related:**
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - the foundational post
- [insurance-conformal on GitHub](https://github.com/burning-cost/insurance-conformal)
- Manna et al. (2025), *Applied Stochastic Models in Business and Industry* - theoretical basis for `pearson_weighted` and locally-weighted conformal in insurance
- Graziadei et al. (arXiv:2307.13124) - frequency-severity conformal prediction protocol

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/19/equipy-vs-insurance-fairness/) - optimal transport fairness
- [EconML vs insurance-causal](/2026/03/19/econml-vs-insurance-causal-inference-pricing/) - causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - model monitoring
- [NannyML vs insurance-monitoring](/2026/03/21/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) - statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) - distributional regression
