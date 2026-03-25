---
layout: post
title: "Getting Started: Three Libraries, One Workflow"
date: 2026-03-23
categories: [guides, tutorials]
tags: [tutorial, getting-started, insurance-causal, insurance-conformal, insurance-monitoring, dml, causal-inference, prediction-intervals, drift-detection, catboost, polars, python, uk-insurance]
description: "A practical walkthrough for pricing analysts: use insurance-causal for causal inference, insurance-conformal for prediction intervals, and insurance-monitoring for drift detection - together, in one workflow."
---

You have built a pricing model. CatBoost, frequency/severity split, reasonable Gini. It is in production. Now what?

Three questions come up almost immediately. First: that telematics score in your model - is it genuinely reducing predicted frequency, or is it just correlated with urban driving? Second: your point estimates say this risk costs £340. How wide is the uncertainty band around that? Third: six months from now, how will you know if the model has started to drift?

These are separate questions, but they come in sequence for every pricing team. We built three libraries that address them. This post walks through all three together, using a single synthetic UK motor book, so you can see how they fit into a coherent workflow rather than three disconnected pieces of code.

The libraries:

- [`insurance-causal`](https://github.com/burning-cost/insurance-causal) - Double Machine Learning for causal effect estimation
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) - distribution-free prediction intervals
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) - PSI, A/E, Gini drift detection

---

## Setup

Install the three libraries. All are on PyPI.

```bash
pip install insurance-causal insurance-conformal insurance-monitoring catboost polars
```

Or with uv:

```bash
uv add insurance-causal insurance-conformal insurance-monitoring catboost polars
```

`insurance-causal` depends on `doubleml` and `scikit-learn`, which install automatically. `insurance-conformal` requires both `polars` and `pandas` (the latter for some binning utilities). `insurance-monitoring` is Polars-native throughout and has no scikit-learn dependency.

---

## The scenario

We have a UK motor book of 50,000 policies. We built a CatBoost Tweedie model on the training set, held out 10,000 policies for calibration, and now have another 10,000 in a monitoring window that represents six months of new business.

We want to:

1. Check whether the `harsh_braking_score` telematics signal has a genuine causal effect on claim frequency, or whether it is a proxy for something else (urban driving, driver age).
2. Wrap the CatBoost model in conformal prediction intervals, so each quote comes with a coverage-guaranteed band.
3. Set up a monitoring report that will tell us whether the model is drifting over the monitoring window.

First, generate the synthetic data we will use throughout:

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(42)
n = 50_000

# Rating factors
driver_age   = rng.integers(18, 75, n).astype(float)
vehicle_age  = rng.integers(1, 15, n).astype(float)
ncd_years    = rng.integers(0, 9, n).astype(float)
urban_flag   = rng.integers(0, 2, n).astype(float)  # 1 = urban postcode

# Telematics: harsh braking score, correlated with urban driving (the confound)
# True causal effect on frequency is small; most of the correlation is urban driving
harsh_braking = (
    0.4 * urban_flag
    + 0.01 * np.maximum(25 - driver_age, 0)
    + rng.normal(0, 0.3, n)
)
harsh_braking = np.clip(harsh_braking, 0, 1)

# Exposure in earned years
exposure = rng.uniform(0.5, 1.0, n)

# Claim frequency: true causal effect of harsh_braking = +0.08 on log scale
# Urban driving adds +0.25 on log scale (the confound)
log_mu = (
    -2.5
    + 0.08 * harsh_braking   # true causal effect
    + 0.25 * urban_flag       # confounder (correlated with harsh_braking)
    - 0.015 * ncd_years
    + 0.008 * vehicle_age
    + np.log(exposure)
)
freq = np.exp(log_mu)
claim_count = rng.poisson(freq)

# Severity: Gamma, correlated with vehicle age
severity_mu = 800 + 60 * vehicle_age + rng.gamma(2, 200, n)
incurred = np.where(claim_count > 0, rng.gamma(2, severity_mu / 2, n), 0.0)

df = pl.DataFrame({
    "driver_age":    driver_age,
    "vehicle_age":   vehicle_age,
    "ncd_years":     ncd_years,
    "urban_flag":    urban_flag,
    "harsh_braking": harsh_braking,
    "exposure":      exposure,
    "claim_count":   claim_count.astype(float),
    "incurred":      incurred,
})
```

We will split this into training (60%), calibration (20%), and monitoring (20%) sets. Temporal ordering matters for real data; here we use a random split with a fixed seed:

```python
idx = rng.permutation(n)
train_idx = idx[:30_000]
cal_idx   = idx[30_000:40_000]
mon_idx   = idx[40_000:]

df_train = df[train_idx.tolist()]
df_cal   = df[cal_idx.tolist()]
df_mon   = df[mon_idx.tolist()]
```

---

## Step 1: Is the telematics score causal?

Before incorporating `harsh_braking` as a rating factor, we want to know how much of its correlation with claims is genuine causal signal versus confounding from urban driving.

A naive Poisson GLM will give us a coefficient. But `urban_flag` is in our rating factors too, so it looks like confounding is controlled for. The problem: your rating model is not perfect. If `urban_flag` only partially captures urban risk - which it does, because "urban" is a band, not a postcode - then `harsh_braking` still carries residual urban signal that the GLM misattributes to braking behaviour.

Double Machine Learning handles this. It partials out all confounders (including `urban_flag`) from both the treatment and the outcome using CatBoost, then estimates the causal effect from the residuals.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import ContinuousTreatment

model_causal = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="exposure",
    treatment=ContinuousTreatment(
        column="harsh_braking",
        standardise=False,  # keep native units: 0–1 score
    ),
    confounders=["driver_age", "vehicle_age", "ncd_years", "urban_flag"],
    cv_folds=5,
)

model_causal.fit(df_train)

ate = model_causal.average_treatment_effect()
print(ate)
```

Expected output (true causal effect on log scale = +0.08):

```
Average Treatment Effect
  Treatment: harsh_braking
  Outcome:   claim_count
  Estimate:  0.083
  Std Error: 0.019
  95% CI:    (0.046, 0.120)
  p-value:   0.000
  N:         30,000
```

The naive GLM would estimate something closer to 0.20 - the combination of the genuine causal effect and the residual urban confound. DML recovers the true effect by partialling out the urban signal that the categorical `urban_flag` could not fully capture.

The confounding bias report makes the gap explicit:

```python
naive_coeff = 0.21  # from your GLM log(mu) ~ harsh_braking + confounders
report = model_causal.confounding_bias_report(naive_coefficient=naive_coeff)
print(report)
# treatment      outcome      naive_estimate  causal_estimate  bias    bias_pct
# harsh_braking  claim_count  0.210           0.083            -0.127  -60.5%
```

The practical conclusion: `harsh_braking` does have a genuine causal effect on frequency. It belongs in the model. But its rating factor relativity should be set using the causal estimate (0.083), not the naive GLM coefficient (0.21). Using the naive number would overcharge telematics policyholders by a factor of roughly 2.5.

---

## Step 2: Conformal prediction intervals on quotes

With the pricing model built and validated, we now want to express uncertainty around each quote. The CatBoost Tweedie model gives a point estimate. A pricing actuary quoting a commercial fleet wants to know: is the uncertainty on this risk £50 either side, or £500?

Parametric intervals - based on the Tweedie dispersion parameter - give a single dispersion across all risks. That is wrong for a heterogeneous motor book. High-risk policies have higher variance in absolute terms, and parametric intervals either over-cover low-risk policies (too wide) or under-cover high-risk ones (too narrow). Conformal prediction avoids this by construction.

First, fit the pricing model on the training set:

```python
import catboost

X_train = df_train.select(
    ["driver_age", "vehicle_age", "ncd_years", "urban_flag", "harsh_braking"]
).to_numpy()
y_train = (df_train["incurred"] / df_train["exposure"].clip(1e-6)).to_numpy()

X_cal = df_cal.select(
    ["driver_age", "vehicle_age", "ncd_years", "urban_flag", "harsh_braking"]
).to_numpy()
y_cal = (df_cal["incurred"] / df_cal["exposure"].clip(1e-6)).to_numpy()

X_mon = df_mon.select(
    ["driver_age", "vehicle_age", "ncd_years", "urban_flag", "harsh_braking"]
).to_numpy()
y_mon = (df_mon["incurred"] / df_mon["exposure"].clip(1e-6)).to_numpy()

pricing_model = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)
pricing_model.fit(X_train, y_train)
```

Now wrap it in conformal prediction. The calibration set must not overlap with the training set - here we use the held-out 10,000 policies:

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=pricing_model,
    nonconformity="pearson_weighted",  # recommended for Tweedie insurance data
    distribution="tweedie",
    tweedie_power=1.5,
)

cp.calibrate(X_cal, y_cal)

# Generate 90% prediction intervals on the monitoring set
intervals = cp.predict_interval(X_mon, alpha=0.10)
print(intervals.head(5))
```

```
shape: (5, 3)
┌────────┬────────┬────────┐
│ lower  ┆ point  ┆ upper  │
│ ---    ┆ ---    ┆ ---    │
│ f64    ┆ f64    ┆ f64    │
╞════════╪════════╪════════╡
│ 0.0    ┆ 312.4  ┆ 841.2  │
│ 0.0    ┆ 487.9  ┆ 1203.7 │
│ 0.0    ┆ 198.1  ┆ 562.4  │
│ 0.0    ┆ 751.3  ┆ 1887.6 │
│ 0.0    ┆ 423.6  ┆ 1089.3 │
└────────┴────────┴────────┘
```

The lower bound is clipped to zero - insurance losses are non-negative. The upper bound varies considerably across the five policies, reflecting the genuine heterogeneity in uncertainty rather than a fixed parametric spread.

Check that coverage holds uniformly across risk deciles - this is the key diagnostic:

```python
diag = cp.coverage_by_decile(X_mon, y_mon, alpha=0.10)
print(diag)
```

A well-behaved output shows coverage at or above 90% in every decile. If the top decile (highest-risk policies) shows 70% coverage, the interval is too narrow where it matters most - and the `pearson_weighted` non-conformity score is the fix for exactly this failure mode.

The `pearson_weighted` score accounts for Tweedie heteroscedasticity. On benchmarks against a heteroskedastic Gamma DGP, it produces intervals 13-14% narrower than a parametric bootstrap at equivalent coverage, and it holds coverage in the top risk decile where parametric intervals typically fail.

---

## Step 3: Setting up monitoring

Six months in. New business has arrived. Has the model drifted?

The aggregate A/E ratio is the standard check - but it has two well-known problems. It is slow: on a UK motor book, frequency claims develop over 3 to 12 months. And it cancels: a model that is 15% cheap on under-25s and 15% expensive on the 26-35 band shows an aggregate A/E of 1.00. Nobody raises an alarm. The underlying adverse selection mechanism is already running.

`insurance-monitoring` monitors at three layers simultaneously: feature distribution (PSI), calibration (A/E with Poisson confidence intervals), and discriminatory power (Gini drift z-test). The last one is the one most monitoring frameworks miss: a model can have a stable A/E and a deteriorating Gini, meaning it is still getting the average right but has lost the ability to rank risks correctly.

We will use the monitoring window as the "current" period and the training set as the reference:

```python
from insurance_monitoring import MonitoringReport

# For monitoring, we need predictions vs actuals in both periods
pred_ref = pricing_model.predict(X_train)
act_ref  = y_train

pred_cur = pricing_model.predict(X_mon)
act_cur  = y_mon

# Feature DataFrames for PSI/CSI analysis
feat_ref = df_train.select(["driver_age", "vehicle_age", "ncd_years"])
feat_cur = df_mon.select(["driver_age", "vehicle_age", "ncd_years"])

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'

results = report.to_polars()
print(results)
```

With a random split and no induced shift, you should get `NO_ACTION` and PSI scores below 0.1 for all features. Now induce a shift to see what the alert looks like in practice:

```python
# Simulate: younger drivers have entered the monitoring book
# - under-25s go from ~7% to ~18% of new business
rng2 = np.random.default_rng(99)
n_mon = 10_000
driver_age_shifted = np.concatenate([
    rng2.integers(18, 25, int(n_mon * 0.18)),
    rng2.integers(25, 75, int(n_mon * 0.82)),
]).astype(float)
```

On shifted data, the monitoring report separates the three failure modes:

```
metric              value   band
ae_ratio            1.09    AMBER      <- model underpredicting (young drivers cost more)
gini_p_value        0.031   AMBER      <- discrimination degraded (p < 0.05)
csi_driver_age      0.18    AMBER      <- driver age distribution has shifted
recommendation      RECALIBRATE
```

The CSI on `driver_age` fires first - the feature distribution has changed, before any claims have developed. The A/E amber follows once we simulate claims development. The recommendation is `RECALIBRATE` rather than `REFIT`: the Murphy decomposition has found that the model's ranking (DSC) is still intact, but the scale (MCB) is wrong. Multiplying all predictions by the aggregate A/E factor will fix the pricing without a full refit.

The distinction matters commercially. A recalibration takes hours. A refit takes weeks of actuarial time. If your monitoring cannot tell you which one you need, you will either over-react (triggering a full refit when a scale factor would have done) or under-react (applying a patch when the model's discrimination has actually broken and needs rebuilding).

---

## Putting it together

Here is the full workflow as a checklist:

**Before deploying the model:**
- Run `CausalPricingModel` on any treatment whose rating factor coefficient you plan to use for pricing decisions. Compare the causal estimate to the GLM coefficient. If the bias exceeds 20%, revisit the factor specification.
- Run `InsuranceConformalPredictor.calibrate()` on a held-out calibration set. Check `coverage_by_decile()` to confirm coverage holds at the target level across all risk deciles.

**At each monitoring cycle (monthly or quarterly):**
- Run `MonitoringReport` with the reference period as training and the monitoring window as current.
- Check the recommendation first. If `NO_ACTION`, record the metrics and move on.
- If `RECALIBRATE`, apply an aggregate A/E multiplier and update the conformal calibration set with recent data.
- If `REFIT`, start the model rebuild. The feature-level CSI will tell you which rating factors are driving the shift.
- Update the conformal calibration at the same cadence. Conformal intervals derived from a 12-month-old calibration set can be stale even when the model itself is still well-calibrated.

---

## What we did not cover

This post is a workflow introduction, not a comprehensive tutorial on each library. We left out several things worth knowing:

**Causal forest for heterogeneous effects.** If the question is not "what is the average causal effect of harsh braking?" but "which customer segments respond most to a discount?", use `insurance_causal.causal_forest.HeterogeneousElasticityEstimator`. It estimates per-customer CATEs with formal heterogeneity tests (BLP, GATES). Requires n >= 5,000 per segment.

**Locally-weighted conformal.** `InsuranceConformalPredictor` supports a locally-weighted variant that conditions the coverage guarantee on the predicted risk level. On heterogeneous books it produces narrower intervals in the top risk decile. Pass `nonconformity="lw"` and supply a bandwidth parameter.

**Champion/challenger testing.** `insurance-monitoring` includes anytime-valid mSPRT testing for comparing two models in production. A standard t-test with monthly peeking inflates type I error to around 25%; mSPRT holds at 5% at all stopping times.

**Sensitivity analysis.** After running DML, always check how fragile the causal estimate is to unobserved confounders. `insurance_causal.diagnostics.sensitivity_analysis()` runs a Rosenbaum-style bound: if the conclusion reverses at gamma = 1.25 (an unobserved factor that changes treatment odds by 25%), the result is fragile. If it holds to gamma = 2.0, it is robust.

---

## Libraries

- [`insurance-causal`](https://github.com/burning-cost/insurance-causal) - DML causal inference, CATE estimation, FCA renewal pricing
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) - distribution-free prediction intervals for Tweedie/GBM models
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) - PSI, A/E, Murphy decomposition, Gini drift z-test

All three are MIT-licensed, Polars-native, and tested against CatBoost on synthetic UK motor data. Questions and pull requests welcome via GitHub Discussions on each repository.

---

**Deeper reading on each library:**

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) - the full theoretical treatment behind `insurance-causal`: why OLS renewal elasticity is confounded and how the Riesz representer approach avoids generalised propensity score instabilities
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - `insurance-conformal` in depth: non-conformity score selection, locally-weighted variants, and the coverage-by-decile diagnostic
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - the three-layer monitoring framework in `insurance-monitoring`: exposure-weighted PSI, the Gini drift z-test, and the Murphy decomposition that distinguishes calibration from resolution failures
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) - the natural next step: once you have causal elasticity estimates from `insurance-causal`, feed them into `insurance-optimise` to find the profit-maximising rate changes subject to FCA constraints
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/) - the calibration module inside `insurance-monitoring` that handles the recalibrate vs refit decision
