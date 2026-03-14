---
layout: post
title: "Distribution-Free Solvency Capital from Conformal Prediction"
date: 2026-03-11
categories: [libraries, pricing, risk, solvency]
tags: [conformal-prediction, solvency-ii, solvency-uk, scr, catboost, locally-weighted, hong, ert, coverage, insurance-conformal, python, capital-modelling, pra-ps924]
description: "Hong (2025) showed a parametric GLM achieves 57.8% actual coverage at 99.5% nominal — the SCR quantile. Conformal prediction held at 99.6% with no distributional assumption. insurance-conformal v0.2 builds on this: SCRReport, LocallyWeightedConformal (~24% narrower intervals), HongModelFree, and extended conditional coverage diagnostics."
---

Your internal model sets the SCR at 99.5% VaR. Hong (2025, arXiv:2503.03659) tested what a parametric GLM actually achieves at that quantile on personal injury claims data: 22,036 settled UK claims, realistic distributional misspecification. The empirical coverage was 57.8%.

You set capital for a 1-in-200 event. Your model is protecting against roughly a 1-in-1.7 event.

Conformal prediction, under the same misspecification, held at 99.6%. And it does this with no distributional assumption whatsoever — only the exchangeability of calibration and future data.

[`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) v0.2 builds the full SCR workflow around this: distribution-free 99.5% upper bounds with regulatory-format output, locally-weighted intervals roughly 24% narrower than standard split conformal, model-free prediction when you have no fitted model, and extended diagnostics for conditional coverage by rating segment.

```bash
uv add "insurance-conformal[catboost]"
```

---

## Why parametric intervals fail at the tail

A GLM prediction interval is derived from the assumed error distribution. At the 95th or 99th percentile, moderate distributional misspecification produces moderate coverage loss. At 99.5% — where the SCR lives — the same misspecification is catastrophic. The tail is precisely where distributional assumptions are least tested by data and most consequential for solvency.

Hong's Table 2 shows the mechanism: the GLM interval at 99.5% coverage has width ratio 0.07 relative to the conformal interval. The parametric interval is seven times narrower than the distribution-free interval — and achieves 57.8% actual coverage. The GLM is not being precise; it is overconfident in the wrong direction. A random forest does better at 98.8% coverage, but still falls short of nominal.

The Standard Formula approach has a related problem: it is calibrated from industry data and systematically wrong for any insurer whose book differs from the industry average. Internal models use parametric assumptions — log-normal, gamma, Pareto tails — that are tractable but not validated at the extremes where the regulation cares most.

Conformal prediction does not replace your actuarial model. It wraps it. You fit your Tweedie GBM or GLM as normal, calibrate the conformal predictor on a held-out period, and extract 99.5% upper bounds with a theoretical guarantee: at least 99.5% of future losses will fall below this bound, across the population of calibration-and-test pairs drawn from the same distribution.

The guarantee is marginal — it is a population-level statement, not a conditional one — but it is genuine, finite-sample, and requires no distributional assumptions. That is a different quality of claim to what parametric models offer.

---

## The Solvency UK angle: PRA PS9/24

Under Solvency UK (PRA PS9/24, which replaces Solvency II post-Brexit for UK insurers), the SCR is computed at 99.5% VaR over a one-year horizon. The standard approach derives this from a fitted model's distributional assumption. If that assumption is wrong — which Hong demonstrates it can be, badly — the SCR is understated.

The conformal approach is not a replacement for actuarial models. It is a validation layer. Run your GLM, compute your parametric SCR. Then run `HongModelFree` on the same data, extract the conformal SCR, and compare. If they diverge materially, your distributional assumption is doing serious work. You should know that before your internal model validation team asks.

Three limitations are worth stating plainly.

**Marginal, not conditional, coverage.** The 99.5% guarantee is averaged over the covariate distribution. For portfolio-level SCR, this is appropriate. For per-policy capital allocation, you need mondrian conformal or conformalized quantile regression.

**Exchangeability.** The guarantee requires that training and test observations are exchangeable — drawn from the same distribution. UK personal injury claims from 1989 to 1999 are not exchangeable with 2025 claims. Restrict calibration data to recent years. We recommend no more than five years for motor and three years for lines with faster mix shift.

**Scaling.** `HongConformal.predict_interval()` computes a matrix of shape (n_new, n_train). For n_train = 50,000 and n_new = 10,000, this is 500 million floats. The library uses chunked batching with a configurable `batch_size` parameter (default: 1,000 new observations per batch).

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

## Model-free SCR: Hong's order-statistic shortcut

Hong (2025) identified a property of a specific nonconformity measure that eliminates the grid search of standard full conformal entirely. The measure is linear in the new observation's value — so the prediction region is always an interval with an upper bound at a single order statistic. O(n log n): a sort and an index lookup.

`HongModelFree` implements this. No regression model needed:

```python
from insurance_conformal.hong import HongModelFree
from insurance_conformal.scr import SCRReport

hong = HongModelFree()
hong.fit(X_train, y_train)

scr = SCRReport(hong)
capital = scr.solvency_capital_requirement(X_train, alpha=0.005)
print(f"SCR upper bound: £{capital:,.0f}")

# Full coverage table across alpha values
print(scr.coverage_table(X_test, y_test))
#    alpha  nominal_coverage  actual_coverage  mean_width  n_test
# 0  0.005             0.995            0.996  544093.0    4408
# 1  0.010             0.990            0.991  498721.0    4408
# 2  0.050             0.950            0.951  312440.0    4408
```

Hong's paper computed an SCR upper bound of £544,093 for the personal injury dataset (Table 7). That number required no distributional assumption about injury claims — only that the 22,036 training claims and the new claim were drawn from the same generating process.

Where this is useful beyond direct SCR estimation:

**New book pricing.** You have acquired a portfolio with 18 months of loss data and no fitted model. `HongModelFree` gives you defensible bounds before full model development.

**Benchmark validation.** If your model-based intervals are narrower than `HongModelFree` intervals, the model is adding genuine predictive information. If they are wider, something is wrong with your score function.

**Auditing third-party models.** `HongTransformConformal` accepts any callable as the h-transformation:

```python
from insurance_conformal.hong import HongTransformConformal

# h can be any callable — a GBM, a GLM, a pricing engine, a lambda
htc = HongTransformConformal(h=external_pricing_model.predict)
htc.calibrate(X_cal, y_cal)
intervals = htc.predict_interval(X_new, alpha=0.10)
```

This is useful when your pricing model does not have a clean sklearn API — a Radar export, a vendor model, an in-house pricing engine.

One caveat: `HongModelFree` assumes an additive relationship between covariates and outcomes. Insurance pricing models are multiplicative (log-linear). The model-free intervals are conservative for log-linear data and should be used as benchmarks rather than production intervals.

---

## Locally-weighted intervals: ~24% narrower

v0.2's `LocallyWeightedConformal` fits a secondary CatBoost model that predicts the expected absolute residual for each observation. Where the secondary model predicts a large residual, the interval widens. Where it predicts small, the interval tightens. The coverage guarantee is maintained — the secondary model allocates the width budget more efficiently across the risk distribution.

```python
from insurance_conformal import LocallyWeightedConformal

lwc = LocallyWeightedConformal(
    model=catboost_gbm,        # your fitted base predictor
    tweedie_power=1.5,
)

# fit() trains the CatBoost spread model on training data
# (spread model must not see calibration data)
lwc.fit(X_train, y_train)
lwc.calibrate(X_cal, y_cal)

intervals = lwc.predict_interval(X_test, alpha=0.10)
print(intervals.head())
#        lower   point    upper   spread
# 0     0.0283  0.1134   0.3891   0.847
# 1     0.0091  0.0365   0.1251   0.762
# 2     0.3140  1.2560   3.8420   1.423
```

The `spread` column is the secondary model's prediction for each observation. A spread of 1.42 means this observation is predicted to have 42% higher residual magnitude than the average calibration point.

The 24% width reduction relative to standard `TweedieConformal` comes from the Manna et al. (ASMBI 2025) benchmark on motor claims data (n=15,000, Tweedie p=1.5, CatBoost base model). Check whether it holds on your data with `width_efficiency_comparison()`:

```python
from insurance_conformal.diagnostics_ext import width_efficiency_comparison

comparison = width_efficiency_comparison(
    predictors={"split": standard_conformal, "locally_weighted": lwc},
    X=X_test,
    alpha=0.10,
)
```

Run this before committing to the two-stage approach. If your calibration set is small, the spread model may not have enough signal to beat the simpler approach.

---

## Extended diagnostics: conditional coverage by segment

Marginal coverage — overall 90.x% on a holdout — tells you little. What you need is conditional coverage: do your intervals cover correctly for motor liability vs. comprehensive, for new drivers vs. experienced, for London vs. rural?

### ERT coverage gap

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

Bins flagged `True` have coverage gaps more than 2 standard errors from zero. MAE above 0.03 and you have a systematic conditional coverage problem that the marginal figure is hiding.

### Subgroup coverage

```python
from insurance_conformal.diagnostics_ext import subgroup_coverage

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

Motorbikes and commercial vehicles — exactly the segments where parametric approaches tend to fail. The fix: recalibrate with a segment-specific calibration set, or switch to `LocallyWeightedConformal`.

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

There was no Python implementation of any of this before v0.2. Hong's papers have no accompanying code. The Manna et al. R code at `alokesh17/conformal_LightGBM_tweedie` is research scripts — not installable, no tests, does not implement Hong's method. MAPIE and crepes provide generic split conformal but with no Tweedie scores and no SCR reporting.

Read Hong (2025) alongside this library. The paper is twelve pages and Table 7 is worth the whole thing.

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
