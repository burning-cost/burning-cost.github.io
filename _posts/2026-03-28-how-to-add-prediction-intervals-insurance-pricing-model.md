---
layout: post
title: "How to Add Prediction Intervals to an Insurance Pricing Model"
date: 2026-03-28
categories: [tutorials]
tags: [conformal-prediction, prediction-intervals, tweedie, insurance-conformal, catboost, uncertainty, python]
description: "Point estimates from pricing models are incomplete. This tutorial shows how to add distribution-free prediction intervals to a CatBoost Tweedie model using insurance-conformal — with per-decile coverage validation and a note on what 'coverage' actually means for pricing."
---

Every GBM pricing model we have seen in production produces a single number per risk: the predicted pure premium. The point estimate. That number goes into the tariff, gets loaded for expenses and profit, and becomes the quoted price.

What is missing is any statement about uncertainty. "We predict £482 for this risk" is a different claim from "we predict £482, and the 90% interval is £180–£1,140." Both contain useful information. The second contains the information a reinsurance underwriter needs, an SCR analyst needs, and that a pricing review should ask about when the model is performing differently across segments.

This tutorial shows how to attach distribution-free prediction intervals to a Tweedie pricing model using [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal). The approach — split conformal prediction — requires no distributional assumptions and gives a finite-sample coverage guarantee: at least 90% of future policies will have their true loss within the stated interval, regardless of whether the Tweedie variance assumption holds in the tail.

---

## Installation

```bash
pip install "insurance-conformal[catboost]"
```

Or with uv:

```bash
uv add "insurance-conformal[catboost]"
```

---

## Why not just use the parametric Tweedie interval?

A Tweedie(p) model assumes Var(Y) ∝ μ^p. On a heterogeneous book, that assumption holds reasonably well in the body of the distribution but breaks in the tail — large, unusual risks are genuinely more volatile than their predicted mean would suggest under a fixed dispersion parameter.

The practical consequence: on a 50,000-policy UK motor book with a heteroskedastic Gamma DGP (measured on Databricks serverless, 2026-03-21, seed=42):

| Approach | Aggregate coverage @ 90% target | Top-decile coverage | Mean interval width |
|---|---|---|---|
| Parametric Tweedie | 93.1% | 90.4% | £4,393 |
| Conformal (`pearson_weighted`) | 90.2% | 87.9% | £3,806 (−13.4%) |
| Locally-weighted conformal | 90.3% | 90.6% | £3,881 (−11.7%) |

The parametric baseline over-covers in aggregate (93.1% when you asked for 90%) because it is too conservative on low-risk policies — and that excess width does not buy you coverage in the high-risk decile where you actually need it. The conformal approach hits the target in aggregate and, with local weighting, hits it in the top decile too.

The 13% narrower intervals are not free — they come from the conformal procedure correctly allocating uncertainty rather than applying a flat Tweedie dispersion parameter to all risks equally.

---

## The data split

Split conformal prediction requires three non-overlapping datasets:

- **Training set**: fit the model
- **Calibration set**: compute the nonconformity scores (residuals) used to set the interval width
- **Test set**: evaluate coverage

The calibration set must not overlap with training. The test set must not overlap with either. A temporal split is ideal for insurance: train on years 1–3, calibrate on year 4, test on year 5. This respects the exchangeability assumption while being honest about temporal dependency.

```python
import numpy as np
import polars as pl
import catboost
from insurance_conformal import InsuranceConformalPredictor

# Assume df is a Polars DataFrame with columns:
# vehicle_age, driver_age, mileage, ncd_years, area_risk, loss_cost, exposure
# Sorted by policy_year

n = len(df)
train_end = int(0.60 * n)
cal_end   = int(0.80 * n)

features = ["vehicle_age", "driver_age", "mileage", "ncd_years", "area_risk"]

X_train = df[:train_end].select(features)
y_train = df[:train_end]["loss_cost"].to_numpy()

X_cal   = df[train_end:cal_end].select(features)
y_cal   = df[train_end:cal_end]["loss_cost"].to_numpy()

X_test  = df[cal_end:].select(features)
y_test  = df[cal_end:]["loss_cost"].to_numpy()

# Train a Tweedie CatBoost model
pool_train = catboost.Pool(
    data=X_train.to_pandas(),
    label=y_train,
    weight=df[:train_end]["exposure"].to_numpy(),
)
model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
model.fit(pool_train)
```

---

## Wrapping the model for conformal intervals

```python
# Wrap the fitted model
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # correct for Tweedie
    tweedie_power=1.5,
)

# Calibrate on held-out year-4 data
cp.calibrate(X_cal, y_cal)

# Generate 90% prediction intervals on test set
# Returns a Polars DataFrame: lower, point, upper
intervals = cp.predict_interval(X_test, alpha=0.10)

print(intervals.head(5))
```

Output:
```
shape: (5, 3)
┌──────────┬───────────┬──────────┐
│ lower    ┆ point     ┆ upper    │
│ ---      ┆ ---       ┆ ---      │
│ f64      ┆ f64       ┆ f64      │
╞══════════╪═══════════╪══════════╡
│  68.40   ┆  241.30   ┆  854.10  │
│ 182.50   ┆  523.80   ┆ 1,462.20 │
│  31.20   ┆   98.70   ┆  319.40  │
│  44.80   ┆  134.20   ┆  427.90  │
│ 312.40   ┆  891.50   ┆ 2,487.60 │
└──────────┴───────────┴──────────┘
```

The `point` column is the CatBoost prediction unchanged. The `lower` and `upper` are the conformal bounds.

---

## The critical check: per-decile coverage

Global coverage of 90% is not enough. A model that covers 100% of low-risk policies and 70% of high-risk policies will show 90% aggregate coverage while systematically failing where it matters.

```python
# Per-decile coverage validation
coverage_table = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(coverage_table)
```

Output (abbreviated):
```
shape: (10, 4)
┌────────┬──────────────────┬─────────────────────┬───────────────┐
│ decile ┆ mean_point_pred  ┆ empirical_coverage  ┆ mean_width    │
│ ---    ┆ ---              ┆ ---                 ┆ ---           │
│ i64    ┆ f64              ┆ f64                 ┆ f64           │
╞════════╪══════════════════╪═════════════════════╪═══════════════╡
│ 1      ┆   87.20          ┆  0.928              ┆   268.30      │
│ 2      ┆  163.40          ┆  0.914              ┆   432.70      │
│ ...    ┆  ...             ┆  ...                ┆   ...         │
│ 9      ┆  934.80          ┆  0.891              ┆ 2,814.10      │
│ 10     ┆ 2,341.60         ┆  0.878              ┆ 6,923.40      │
```

The top decile coverage of 87.8% is below the 90% target — an expected outcome with the `pearson_weighted` nonconformity score. If top-decile coverage is a hard requirement (for reinsurance pricing or SCR stress-testing), use the locally-weighted variant instead.

---

## Locally-weighted intervals for better tail coverage

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=model, tweedie_power=1.5)
lw.fit(X_train.to_pandas(), y_train)
lw.calibrate(X_cal.to_pandas(), y_cal)
intervals_lw = lw.predict_interval(X_test.to_pandas(), alpha=0.10)

# Per-decile check
coverage_lw = lw.coverage_by_decile(X_test.to_pandas(), y_test, alpha=0.10)
```

The locally-weighted variant fits a variance model alongside the mean model. High-variance risks get wider intervals; low-variance risks get narrower ones. Top-decile coverage comes in at roughly 90.6% against a 90% target on the benchmark portfolio.

The cost is a slightly wider mean interval width (£3,881 vs £3,806 for the standard variant) — because width is being reallocated from low-risk to high-risk policies rather than reduced overall.

---

## What 'coverage' actually means for pricing

A 90% prediction interval in conformal prediction means: if you were to price 1,000 new policies with these intervals, at least 900 of them would have their actual loss within the stated bounds. This is a finite-sample statement about future policies, not an asymptotic one.

It does not mean: 90% of past policies in the training set fell within their intervals. The calibration set is used to set the interval width so that the guarantee holds prospectively.

Three uses for these intervals in an insurance context:

**1. Reinsurance attachment.** The upper bound at 99.5% confidence is the claim amount above which your reinsurance treaty responds. Computing this per policy, rather than from a portfolio aggregate, lets you identify which individual risks are driving your reinsurance cost.

```python
upper_995 = cp.predict_interval(X_test, alpha=0.005)["upper"]
```

**2. SCR stress-testing.** Solvency II internal model validation requirements ask internal model teams to validate the predictive distribution of the model, not just point estimates (aligned to SS1/23 best practice principles, which apply formally to banks but set the benchmark for insurer model governance too). A coverage table by risk segment — showing the model meets its stated confidence level across all deciles — is the direct evidence the validation team needs.

**3. Pricing review alerts.** In a monitoring dashboard, track the empirical coverage rate on new business month-by-month. When coverage in any decile drops below the target, the model is producing intervals that are too narrow for that segment — a signal of distribution shift worth investigating before it shows up in claims.

---

## When not to use conformal prediction

**When your calibration set is too small.** Conformal prediction's finite-sample guarantee requires at least ~200–500 calibration policies to be reliable. Below that, the quantile estimate from the calibration nonconformity scores is noisy and the intervals will be wider than necessary to compensate.

**When your data is not exchangeable.** The coverage guarantee assumes calibration and test data come from the same distribution. If your calibration period was 2022 and you are pricing in 2026, the guarantee may not hold. Use a temporal split and monitor coverage on recent data.

**When you just need a simpler story.** For a head of pricing who wants to know "why is this risk expensive," a prediction interval does not help. It answers a different question — "how uncertain are we about the actual claim amount for this risk." Know which question you are being asked.

---

## Verdict

Adding conformal prediction intervals to a Tweedie pricing model takes under 20 lines of code after the model is fitted. The output is 13% narrower than parametric Tweedie intervals in aggregate, with a finite-sample coverage guarantee that does not depend on the variance assumption holding.

The per-decile coverage table is the check to run every time you recalibrate. If top-decile coverage is below target, switch to `LocallyWeightedConformal`. If it is above target, your intervals are too conservative — the parametric baseline is wasting width on low-risk policies.

```bash
pip install "insurance-conformal[catboost]"
```

Source, benchmarks, and full Databricks notebooks at [GitHub](https://github.com/burning-cost/insurance-conformal). The benchmark comparing parametric, conformal, and locally-weighted approaches is at `benchmarks/benchmark.py`.

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [Does Conformal Prediction Actually Work for Insurance Pricing?](/2026/03/23/does-conformal-prediction-work-insurance-pricing/)
- [Conformal Prediction for Solvency II SCR Validation](/2026/03/26/conformal-prediction-for-solvency-ii-scr-validation/)
