---
layout: post
title: "Distribution-Free Solvency Capital from Conformal Prediction"
date: 2026-03-11
categories: [libraries, pricing, risk, solvency]
tags: [conformal-prediction, solvency-ii, scr, catboost, locally-weighted, hong, ert, coverage, insurance-conformal, python, capital-modelling]
description: "insurance-conformal v0.2 adds SCRReport for distribution-free 99.5% upper bounds, LocallyWeightedConformal for intervals ~24% narrower than standard split conformal, model-free prediction via Hong (2025), and extended diagnostics for checking conditional coverage by rating segment."
---

The standard Solvency II SCR calculation requires you to estimate the 99.5th percentile of your loss distribution — the 1-in-200-year event. Every major internal model uses parametric assumptions to get there: fit residuals to a distribution, compute the quantile analytically. The distribution assumption is not tested. It is chosen to be tractable.

Conformal prediction offers a different path. Given a calibration set, it produces an upper bound at any specified quantile — including 99.5% — with a finite-sample, distribution-free coverage guarantee. No assumed distribution. The only requirement is that calibration and test data are exchangeable.

[`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) v0.2 builds on the split conformal machinery from v0.1 to add SCR reporting, locally-weighted intervals that are roughly 24% narrower, model-free prediction for situations where you have no existing regression model, and extended diagnostics for checking coverage by rating segment.

```bash
uv add "insurance-conformal[catboost]"
```

---

## The SCR problem

A Solvency II Standard Formula insurer computes the non-life underwriting risk SCR from volatility factors applied to net written premium. An internal model insurer computes the 99.5th percentile of the aggregate loss distribution directly. Both approaches produce a number. The question is whether that number is honest.

The Standard Formula approach is calibrated from industry data and systematically wrong for any insurer whose book differs from the industry average. Internal models use parametric assumptions — log-normal, gamma, Pareto tails — that are tractable but not validated at the extremes where the regulation cares most.

Conformal prediction does not replace your actuarial model. It wraps it. You fit your Tweedie GBM or GLM as normal, calibrate the conformal predictor on a held-out period, and then extract 99.5% upper bounds with a theoretical guarantee: at least 99.5% of future losses will fall below this bound, across the population of calibration-and-test pairs drawn from the same distribution.

The guarantee is marginal — it is a population-level statement, not a conditional one — but it is genuine, finite-sample, and requires no distributional assumptions. That is a different quality of claim to what parametric models offer.

---

## SCRReport: conformal bounds for regulatory capital

```python
from insurance_conformal import TweedieConformal
from insurance_conformal.scr import SCRReport

# Fit and calibrate your conformal predictor
predictor = TweedieConformal(model=gbm, tweedie_power=1.5)
predictor.calibrate(X_cal, y_cal)

# Wrap it for SCR reporting
report = SCRReport(predictor)

# 99.5% upper bounds per risk — the SCR quantile
scr_bounds = report.solvency_capital_requirement(X_new, alpha=0.005)
# DataFrame: lower, point, upper (upper = 99.5th percentile bound)

# Aggregate view
agg = report.aggregate_scr(X_new)
print(agg)
# {'total_expected_loss': 4_230_412,
#  'total_scr': 9_847_331,
#  'scr_ratio': 2.33,
#  'n_risks': 12_450}
```

The `scr_ratio` is the aggregate SCR divided by aggregate expected loss — the loading above best estimate that the capital bound represents.

The coverage validation table is the piece that makes this useful for regulatory documentation:

```python
table = report.coverage_validation_table(X_test, y_test)
print(table)
#   alpha   target_coverage  empirical_coverage  n_obs  pass
#   0.005          0.9950              0.9953   2200   True
#   0.010          0.9900              0.9905   2200   True
#   0.050          0.9500              0.9512   2200   True
#   0.100          0.9000              0.9023   2200   True
#   0.200          0.8000              0.8031   2200   True

print(report.to_markdown())
# Renders a formatted regulatory submission table
```

The table checks coverage at multiple alpha levels simultaneously. If your intervals are well-calibrated, all rows pass. If a row fails — empirical coverage materially below target — you have a score mismatch or non-exchangeability problem, and you need to resolve that before using the bounds in any regulatory context.

One point to be explicit about: conformal gives marginal coverage. Whether that is sufficient for Solvency II purposes depends on your internal model validation framework. The table is evidence, not a pass/fail gate.

---

## Locally-weighted intervals: ~24% narrower

v0.1 split conformal with Pearson-weighted scores already produces substantially narrower intervals than naive conformal. v0.2's `LocallyWeightedConformal` goes further.

The idea, from Manna et al. (ASMBI 2025), is to fit a secondary model that predicts the expected absolute residual for each observation. Where the secondary model predicts a large residual, the observation is hard to predict and gets a wide interval. Where it predicts a small residual, the interval tightens. The coverage guarantee is maintained — the secondary model just allocates the width budget more efficiently across the risk distribution.

```python
from insurance_conformal import LocallyWeightedConformal

# Two-stage fit: base model + spread model
lwc = LocallyWeightedConformal(
    model=catboost_gbm,        # your fitted base predictor
    tweedie_power=1.5,
)

# fit() trains the CatBoost spread model on training data
# (spread model must not see calibration data)
lwc.fit(X_train, y_train)
lwc.calibrate(X_cal, y_cal)

# Intervals include a 'spread' diagnostic column
intervals = lwc.predict_interval(X_test, alpha=0.10)
print(intervals.head())
#        lower   point    upper   spread
# 0     0.0283  0.1134   0.3891   0.847
# 1     0.0091  0.0365   0.1251   0.762
# 2     0.3140  1.2560   3.8420   1.423
```

The `spread` column is the secondary model's prediction for each observation — a relative measure of how hard the point is to predict. A spread of 1.42 means this observation is predicted to have 42% higher residual magnitude than the average calibration point. The interval for that risk is correspondingly wider.

The 24% width reduction relative to standard `TweedieConformal` comes from the benchmark in the Manna et al. paper on held-out motor claims data (n=15,000, Tweedie p=1.5, CatBoost base model). On your data, the reduction will vary. The diagnostic to check is `width_efficiency_comparison()`:

```python
from insurance_conformal.diagnostics_ext import width_efficiency_comparison

comparison = width_efficiency_comparison(
    predictors={"split": standard_conformal, "locally_weighted": lwc},
    X=X_test,
    alpha=0.10,
)
# Returns mean interval width and empirical coverage for each predictor
```

Run this before committing to the two-stage approach. If your base model is already well-calibrated and your calibration set is small, the spread model may not have enough signal to beat the simpler approach.

---

## Model-free prediction: no regression model required

Hong (2025, arXiv:2503.03659) showed that conformal prediction does not require a regression model at all. You can construct prediction intervals directly from historical data using an h-transformation framework.

`HongModelFree` implements this:

```python
from insurance_conformal.hong import HongModelFree

# No model required — learns directly from data
mf = HongModelFree()
mf.fit(X_train, y_train)   # stores training data
mf.calibrate(X_cal, y_cal)

intervals = mf.predict_interval(X_new, alpha=0.10)
```

The intervals are wider than model-based conformal — sometimes substantially so. That is expected: without a regression model, you have less information to work with.

Where this is useful:

**New book pricing.** You have acquired a new scheme or portfolio with 18 months of loss data and no fitted model. You need SCR-style bounds quickly, before you have run full model development. `HongModelFree` gives you defensible bounds with what you have.

**Benchmark validation.** If your model-based intervals are narrower than `HongModelFree` intervals, that is reassuring — the model is adding genuine predictive information. If they are wider, something is wrong with your score function.

**Auditing third-party models.** `HongTransformConformal` accepts any callable as the h-transformation, not just sklearn-compatible models:

```python
from insurance_conformal.hong import HongTransformConformal

# h can be any callable — a GBM, a GLM, a pricing engine, a lambda
htc = HongTransformConformal(h=external_pricing_model.predict)
htc.calibrate(X_cal, y_cal)
intervals = htc.predict_interval(X_new, alpha=0.10)
```

This is useful when your pricing model does not have a clean sklearn API — a Radar export, a vendor model, an in-house pricing engine. You pass the predict function directly and the conformal machinery wraps it.

One caveat on `HongModelFree`: the construction assumes an additive relationship between covariates and outcomes. Insurance pricing models are multiplicative (log-linear). The model-free intervals are conservative for log-linear data and should be used as benchmarks rather than production intervals. The warning in the library is explicit about this.

---

## Extended diagnostics: where your intervals are lying

Marginal coverage — overall 90.x% on a holdout — tells you little. What you need is conditional coverage: do your intervals cover correctly for motor liability vs. comprehensive, for new drivers vs. experienced, for London vs. rural?

v0.2 adds two diagnostics for this.

### ERT coverage gap

The Expected Ranking Test (ERT) coverage gap bins observations by predicted value and checks coverage in each bin against the target:

```python
from insurance_conformal.diagnostics_ext import ert_coverage_gap

gaps = ert_coverage_gap(predictor, X_test, y_test, alpha=0.10, n_bins=10)
print(gaps)
#    bin  mean_pred  n_obs  coverage  target  gap    flagged
#      1     0.021    220    0.923   0.900   0.023   False
#      2     0.048    220    0.915   0.900   0.015   False
#      ...
#      9     1.540    220    0.878   0.900  -0.022   False
#     10     3.200    220    0.832   0.900  -0.068   True   *
#     -1       NaN   2200    0.901   0.900   0.001   False  (overall)
```

Bins flagged `True` have coverage gaps more than 2 standard errors from zero. The summary row (bin = -1) gives overall mean absolute error of the coverage gap across bins. An MAE above 0.01 is worth investigating. Above 0.03 and you have a systematic conditional coverage problem that the marginal figure is hiding.

### Subgroup coverage

For segment-level checks — by vehicle class, by region, by distribution channel:

```python
from insurance_conformal.diagnostics_ext import subgroup_coverage

# vehicle_class is a categorical array aligned with X_test
results = subgroup_coverage(
    predictor, X_test, y_test,
    groups=vehicle_class_array,
    alpha=0.10,
)
print(results)
#   group        n_obs  coverage  target  gap     flagged
#   motorbike      340    0.861   0.900  -0.039   True  *
#   van            890    0.898   0.900  -0.002   False
#   private_car   8100    0.903   0.900   0.003   False
#   commercial     420    0.874   0.900  -0.026   True  *
```

Motorbikes and commercial vehicles are flagged here. Those are exactly the segments where parametric approaches tend to fail — small volumes, heavy tails, distribution shapes that don't match the majority class. Conformal should handle them better than parametric if your score function is correctly specified. If the subgroup diagnostic flags them, your score is not transferring well across segments.

The fix: recalibrate with a segment-specific calibration set, or switch to `LocallyWeightedConformal` where the spread model can learn segment-specific difficulty.

---

## What changed from v0.1

v0.1 shipped with SplitConformal, CVPlusConformal, CQR, TweedieConformal, PoissonConformal, and ConformalReport. Those are unchanged. v0.2 adds:

| Module | What it adds |
|---|---|
| `LocallyWeightedConformal` | CatBoost spread model, ~24% narrower intervals (Manna et al. ASMBI 2025) |
| `HongModelFree` | No point predictor needed — direct from data (Hong arXiv:2503.03659) |
| `HongTransformConformal` | h-transform framework, accepts any callable predictor (Hong arXiv:2601.21153) |
| `SCRReport` | 99.5% upper bounds + multi-alpha coverage validation table |
| `ert_coverage_gap` | Conditional coverage by predicted-value bin, 2 SE flagging |
| `subgroup_coverage` | Coverage by arbitrary categorical grouping variable |
| `width_efficiency_comparison` | Side-by-side interval width comparison across predictors |

136 new tests, 186 total. All v0.1 tests pass unchanged.

---

## Install and links

```bash
uv add insurance-conformal            # base
uv add "insurance-conformal[catboost]"  # + LocallyWeightedConformal
uv add "insurance-conformal[all]"       # + matplotlib diagnostics
```

[GitHub](https://github.com/burning-cost/insurance-conformal) · [PyPI](https://pypi.org/project/insurance-conformal/)

The v0.1 post covers the score function theory and why the Pearson-weighted score outperforms the naive absolute residual for Tweedie data: [Why Your Prediction Intervals Are Lying to You](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/).

---

**Related reading:**
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the foundational library: score functions, calibration sets, and split conformal for Tweedie models
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — when the constraint is on expected monetary shortfall rather than coverage probability
- [Frequency and Severity Are Two Outputs. You Have One Prediction Interval.](/2026/03/13/insurance-multivariate-conformal/) — joint conformal prediction sets for multi-output models via [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) (multivariate module), with joint coverage rather than marginal coverage
