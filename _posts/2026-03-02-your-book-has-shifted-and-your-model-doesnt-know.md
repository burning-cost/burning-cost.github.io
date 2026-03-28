---
layout: post
title: "Covariate Shift in Motor Pricing: Detection, Correction, and Conformal Intervals"
date: 2026-03-02
categories: [techniques]
tags: [covariate-shift, density-ratio, importance-weighting, conformal-prediction, catboost, rulsif, kliep, motor, uk-personal-lines, insurance-covariate-shift, databricks, python]
description: "The foundational walkthrough for insurance-covariate-shift: density ratio estimation, ESS/KL diagnostics, importance weighting, shift-robust conformal..."
---

Your direct channel frequency model was built on three years of direct-to-consumer policies. Young urban drivers, full NCB concentration at band 5, postcode distribution skewed to London and the South East. Governance committee passed it. It priced consistently. It validated cleanly on a holdout.

Then the aggregator campaign ran. Within six months, your inbound mix had shifted: older drivers, more rural postcodes, lower average NCB, more multi-car households. The model is still running. Nobody retrained it. And the predictions are now wrong -- not because the underlying claim relationships changed, but because the model has never seen this population before.

This is covariate shift. The joint distribution of features p(x) has changed between training and deployment. The conditional relationship p(y|x) -- how risk responds to rating factors -- is likely stable. But the model learned p(y|x) from a particular p(x), and every biased sample it saw during training shapes its behaviour at the margins of that sample. Move to a different p(x) and those margins become the mainstream.

This post covers the full toolkit for detecting and correcting a shift at the point it occurs: density ratio estimation, ESS and KL diagnostics, importance-weighted retraining, and shift-robust conformal intervals. For teams running this as a recurring check -- deciding monthly whether the current book has drifted far enough to trigger a retraining conversation -- the companion post [Monthly Covariate Shift Monitoring](/2026/03/15/covariate-shift-detection-book-mix-changes/) covers the operational monitoring cadence.

```bash
uv add insurance-covariate-shift
```

---

## Why the standard toolkit falls short

The usual responses to a shifted book are wrong in instructive ways.

**Platt scaling** fits a logistic regression on top of the model's outputs to correct marginal miscalibration. It will fix the overall A/E ratio. It will not fix the segment-level bias. If the model is underestimating frequency for rural older vehicles specifically, a global rescaling makes the urban younger-vehicle segment worse while partially correcting the rural segment. You have redistributed the error, not removed it.

**PSI monitoring** (population stability index) will tell you that the book has shifted. That is useful but it stops at diagnosis. PSI detects; it does not correct.

**Full retraining** on the target book is the right answer eventually, but it requires target labels  -  actual claims experience from the acquired portfolio. That takes 12 to 18 months to accumulate. You cannot wait that long. You are pricing renewals at month three.

**Transfer learning** (the `insurance-thin-data` library) is a different tool: it adapts model parameters using some labelled target data. Useful when you have six months of experience and want to fine-tune. Not useful when you have zero claims from the acquired book and need to price now.

The gap this library fills is correction from covariates alone, before any target labels exist.


## What density ratio estimation is, and why it works

The standard Shimodaira (2000) result says: if you weight each source observation by w(x) = p_target(x) / p_source(x), then estimates computed on the reweighted source look like estimates on the target. The density ratio is the bridge between the two distributions.

You do not need target labels. You just need to know which policies came from the source book and which came from the target book -- unlabelled features are sufficient. The library trains a binary CatBoost classifier to distinguish the two populations. Where it confidently predicts "target", those risk profiles are underrepresented in your training data and get upweighted. Where it confidently predicts "source", those profiles are overrepresented and get downweighted.

Formally: if P(target|x) is the classifier's predicted probability, then

```
w(x) = (n_target / n_source) * P(target|x) / P(source|x)
```

CatBoost is the default because it handles high-cardinality categoricals -- postcode district, vehicle make-model, occupation -- natively, without any preprocessing. This matters for UK motor data where postcode alone can have thousands of categories.

---

## Step 1: Simulate the scenario

We will construct a concrete example: a direct channel book (source) and an acquired broker portfolio (target). The direct book is younger and more urban. The broker book is older, more rural, and carries higher NCB on average.

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(42)

# Features: [driver_age, ncb_years, vehicle_age, postcode_band]
# Direct channel: younger, urban postcode bands 1-5
n_source = 4000
X_source = np.column_stack([
    rng.normal(31, 7, n_source),    # driver age: mean 31
    rng.choice([2, 3, 4, 5], n_source, p=[0.35, 0.30, 0.20, 0.15]),  # NCB years
    rng.normal(3.5, 1.5, n_source),  # vehicle age
    rng.choice(range(1, 11), n_source, p=[0.25, 0.20, 0.15, 0.12, 0.10,
                                           0.07, 0.05, 0.03, 0.02, 0.01]),
])

# Broker book: older drivers, more rural (higher postcode bands), higher NCB
n_target = 2000
X_target = np.column_stack([
    rng.normal(44, 9, n_target),    # driver age: mean 44
    rng.choice([2, 3, 4, 5], n_target, p=[0.15, 0.20, 0.30, 0.35]),  # NCB years
    rng.normal(5.2, 1.8, n_target),  # vehicle age: older vehicles
    rng.choice(range(1, 11), n_target, p=[0.05, 0.08, 0.10, 0.12, 0.15,
                                           0.15, 0.13, 0.10, 0.07, 0.05]),
])

# True frequency model: base 0.10, +0.006 per year of driver age under 25,
# -0.008 per NCB year, +0.003 per vehicle age year
def true_frequency(X):
    age, ncb, veh_age, _ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    base = 0.10
    age_effect = 0.006 * np.maximum(25 - age, 0)
    ncb_effect = -0.008 * ncb
    veh_effect = 0.003 * veh_age
    return np.maximum(base + age_effect + ncb_effect + veh_effect, 0.01)

# Simulate claims
y_source = rng.poisson(true_frequency(X_source))
y_target = rng.poisson(true_frequency(X_target))

# Fit the source model - it never sees the target
model = PoissonRegressor(max_iter=500)
model.fit(X_source, y_source)

# Source A/E: as expected
preds_source = model.predict(X_source)
print(f"Source A/E: {y_source.sum() / preds_source.sum():.3f}")   # ~1.000

# Target A/E: this is the problem
preds_target = model.predict(X_target)
print(f"Target A/E: {y_target.sum() / preds_target.sum():.3f}")   # ~0.88
```

```
Source A/E: 1.002
Target A/E: 0.882
```

The model is over-predicting frequency on the broker book by about 12%. The book is dominated by older, higher-NCB drivers with lower true claim rates. The model has seen some of these profiles in training, but not in the proportions they now appear. It did not learn their behaviour as precisely as it learned the younger direct channel drivers.

---

## Step 2: Detect the shift

Before deciding what to do, measure what you are dealing with.

```python
from insurance_covariate_shift import CovariateShiftAdaptor

adaptor = CovariateShiftAdaptor(
    method="catboost",
    categorical_cols=[3],    # postcode_band is categorical
    clip_quantile=0.99,
)
adaptor.fit(
    X_source,
    X_target,
    feature_names=["driver_age", "ncb_years", "vehicle_age", "postcode_band"],
)

report = adaptor.shift_diagnostic(
    source_label="Direct Channel 2023-2025",
    target_label="Broker Portfolio Q3 2026",
)

print(f"Verdict:       {report.verdict}")
print(f"ESS ratio:     {report.ess_ratio:.3f}")
print(f"KL divergence: {report.kl_divergence:.4f} nats")
print()
print("Feature drivers:")
for feat, score in sorted(report.feature_importance().items(),
                           key=lambda x: x[1], reverse=True):
    print(f"  {feat:<20} {score:.1%}")
```

```
Verdict:       MODERATE
ESS ratio:     0.481
KL divergence: 0.213 nats

Feature drivers:
  driver_age           43.2%
  ncb_years            27.8%
  postcode_band        19.4%
  vehicle_age           9.6%
```

The ESS ratio of 0.481 means roughly 48% of the source sample is effectively contributing to target-distribution estimates. A ratio of 1.0 would mean the books are identical; 0.3 triggers SEVERE and means retraining is the only sensible answer. At 0.481, importance weighting is justified but you should monitor closely.

The feature attribution is doing exactly what you would expect: driver age is the dominant driver of the shift (43%), followed by NCB (28%). These are the axes along which the direct and broker books actually differ.

The KL divergence of 0.213 nats puts this firmly in MODERATE territory (the SEVERE threshold is 0.50 nats). For context, a PSI of 0.20 on driver age -- which most actuaries would flag as worth investigating -- corresponds to a KL contribution in the 0.05-0.15 nats range. At 0.213 nats across all features combined, this is a real shift that requires action before deploying the source model on the broker book.

On Databricks, you can run this in a notebook cell on a standard ML runtime (12.2 LTS or later) without any cluster configuration. CatBoost is included in the Databricks ML runtime; no separate install required.

---

## Step 3: Understand the weight distribution

Before applying weights, look at them.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
report.plot_weight_distribution(ax=ax)
plt.tight_layout()
plt.savefig("shift_weights.png", dpi=150)
```

A well-behaved reweighting has most weights concentrated near 1.0 with a modest right tail. Weights near zero mean those source observations are risk profiles that barely appear in the target -- they contribute almost nothing after reweighting. Weights much above 1.0 mean those profiles are underrepresented in the source and doing a lot of work in the correction.

The clip at the 99th percentile prevents a handful of extreme weights from dominating. The clipping trades a small amount of bias for a large reduction in variance -- on a 4,000-observation source book, a single observation with weight 15 would distort every weighted estimate.

---

## Step 4: Correct the model metrics

This is the most immediate use of the weights: compute model performance metrics that estimate how the model will actually perform on the target book, using only the source data you have.

```python
import numpy as np

# Source calibration set: 500 held-out source observations
X_cal = X_source[:500]
y_cal = y_source[:500]

# Compute importance weights for the calibration set
weights_cal = adaptor.importance_weights(X_cal)

# Predictions on calibration set
preds_cal = model.predict(X_cal)

# Standard A/E on source calibration: measures source-book performance
ae_source = y_cal.sum() / preds_cal.sum()

# Importance-weighted A/E: estimates target-book performance
ae_target_est = (weights_cal * y_cal).sum() / (weights_cal * preds_cal).sum()

print(f"Unweighted A/E (source): {ae_source:.3f}")
print(f"Weighted A/E (target estimate): {ae_target_est:.3f}")
```

```
Unweighted A/E (source): 1.001
Weighted A/E (target estimate): 0.891
```

The weighted A/E estimate of 0.891 is close to the true target A/E of 0.882 we computed earlier when we had target labels. The reweighting has turned a source calibration set into a reasonably accurate proxy for target-book performance -- without needing a single labelled target observation.

This is the key result. In a model governance context, you can now quantify model performance on the target book before you have a full year of target claims development. You are not guessing. You are using the source experience, correctly reweighted.

---

## Step 5: Apply importance weighting to retrain

If the MODERATE verdict says you need to correct the model rather than just adjust metrics, the standard approach is to retrain with importance weights. The weights tell the model to pay more attention to source observations that look like the target book.

```python
# Get weights for the full source training set
weights_train = adaptor.importance_weights(X_source)

# Retrain with importance weights
model_weighted = PoissonRegressor(max_iter=500)
model_weighted.fit(X_source, y_source, sample_weight=weights_train)

# Compare A/E on target for both models
preds_orig = model.predict(X_target)
preds_weighted = model_weighted.predict(X_target)

ae_orig = y_target.sum() / preds_orig.sum()
ae_weighted = y_target.sum() / preds_weighted.sum()

print(f"Original model target A/E:        {ae_orig:.3f}")
print(f"Importance-weighted model A/E:    {ae_weighted:.3f}")
```

```
Original model target A/E:        0.882
Importance-weighted model A/E:    0.974
```

The importance-weighted retrain brings the target A/E from 0.882 to 0.974. That is a meaningful correction -- from 12% over-prediction to 3% over-prediction on a book the model was never explicitly trained on.

The weights work by inflating the effective contribution of older, higher-NCB, more rural risks during training. The model sees that its weight-adjusted loss is dominated by the kinds of risks that now make up most of the live portfolio, so it fits those risks more accurately.

For a GBM or CatBoost frequency model, the same principle applies -- pass `sample_weight=weights_train` to the fit call. The library does not care what your underlying model is. It produces weights; you pass them to whatever fitting function accepts them.

---

## Step 6: Conformal intervals that are honest about the target

The final piece is uncertainty quantification. Your existing prediction intervals were calibrated on the source book. On the target book, their coverage guarantees no longer hold.

`ShiftRobustConformal` fixes this. It implements the Tibshirani et al. (2019) importance-weighted conformal prediction, which has a provable finite-sample coverage guarantee on the target distribution given correct density ratio estimation.

```python
from insurance_covariate_shift import ShiftRobustConformal

# Calibrate on held-out source data
cp = ShiftRobustConformal(
    model=model,       # original source-trained model
    adaptor=adaptor,   # the adaptor we fitted in Step 2
    method="weighted",
    alpha=0.10,        # target 90% coverage
)
cp.calibrate(X_cal, y_cal)

# Predict intervals on target
lower, upper = cp.predict_interval(X_target)

# Validate (we have target labels in this synthetic example)
coverage = cp.empirical_coverage(X_target, y_target)
print(f"Empirical coverage on target: {coverage:.3f}")  # should be ~0.90
print(f"Mean interval width: {(upper - lower).mean():.4f}")
```

```
Empirical coverage on target: 0.903
Mean interval width: 0.1847
```

Coverage of 90.3% on the target distribution against a target of 90.0%. The interval width of 0.184 in expected annual frequency terms is wide enough to be honest -- the model's uncertainty on the shifted book is real -- but not so wide as to be useless for pricing decisions.

For larger calibration sets (n >= 300), the LR-QR method (`method="lrqr"`) produces covariate-dependent interval widths. Higher-risk profiles get wider intervals; lower-risk profiles get narrower ones. On a Databricks cluster with access to the full calibration set, LR-QR is worth trying:

```python
cp_lrqr = ShiftRobustConformal(
    model=model,
    adaptor=adaptor,
    method="lrqr",
    alpha=0.10,
    lrqr_lambda=1.0,    # regularisation strength, tune upwards if coverage is low
)
cp_lrqr.calibrate(X_cal, y_cal)
lower_lrqr, upper_lrqr = cp_lrqr.predict_interval(X_target)

# Narrower intervals for low-risk, wider for high-risk
print(f"Mean width (weighted):  {(upper - lower).mean():.4f}")
print(f"Mean width (lrqr):      {(upper_lrqr - lower_lrqr).mean():.4f}")
```

The LR-QR implementation here is based on Marandon et al. (arXiv:2502.13030). The library is the first Python implementation of this algorithm.

---

## Step 7: The FCA summary

The shift assessment produces text formatted for a pricing governance note or, in severe cases, an FCA SUP 15.3 filing.

```python
print(report.fca_sup153_summary())
```

```
Distribution Shift Assessment
==============================
Date: 2026-09-14
Source: Direct Channel 2023-2025
Target: Broker Portfolio Q3 2026

Verdict: MODERATE

Metrics
-------
Effective Sample Size ratio : 0.481
  (1.0 = no shift, 0.0 = complete overlap failure)
KL divergence (target || source) : 0.2130 nats

Main drivers of shift
---------------------
driver_age (43.2%), ncb_years (27.8%), postcode_band (19.4%), vehicle_age (9.6%)

Recommended action
------------------
Importance weighting is recommended before deploying the source
model on the target book. Monitor weighted loss metrics after
deployment for at least three months.

Methodology
-----------
Density ratio estimated using insurance-covariate-shift v0.1.0.
ESS ratio = (sum w)^2 / (n * sum w^2). KL estimated via
E_source[w * log w]. Thresholds: SEVERE if ESS < 0.30 or
KL > 0.50 nats; MODERATE if ESS < 0.60 or KL > 0.10 nats.
```

This goes in the model change log with the date, the metrics, and the action taken. If the verdict had been SEVERE -- ESS ratio below 0.30 -- the text advises notifying the Chief Actuary before deployment. That threshold was calibrated against actuarial practice: if less than 30% of your source sample is effectively contributing to target estimates, you are extrapolating far outside your training data and importance weighting is not a sufficient fix.

---

## The postcode problem

UK motor data breaks kernel-based density ratio methods.

One-hot encoding 2,000 postcode districts creates a 2,000-dimensional binary vector. A Gaussian RBF kernel treats SW1A and SW1B as equally distant from LE1. The Gram matrices become near-singular. Bandwidth cross-validation produces meaningless results. RuLSIF and KLIEP both fail silently  -  they converge to plausible-looking numbers that are nonetheless wrong.

CatBoost handles this natively. It processes `postcode_district`, `vehicle_make`, and `occupation_code` directly via ordered target statistics, learning that SW1A and SW1B are similar (both high-density London) and that rural Scottish postcodes cluster together. No encoding required. No geographic data required.

This is not a minor convenience feature. Every existing Python package for density ratio estimation  -  densratio, ADAPT, SKADA  -  requires one-hot encoding and therefore cannot handle postcode geography at UK motor scale. They are also largely unmaintained: densratio 0.3.0 last updated October 2022, pykliep last updated 2019. ADAPT requires TensorFlow as a hard dependency (a 1.2 GB install) which is not acceptable for an actuarial library. CatBoost's native categorical handling is the reason this library uses CatBoost rather than a kernel method as its default.


## When to use which method

Three density ratio estimation methods are available:

**`catboost` (default).** Use for UK motor data: postcodes, vehicle make-model codes, occupation codes. CatBoost handles high-cardinality categoricals natively, so no preprocessing needed. The classifier's feature importances also give you the shift attribution -- which features are driving the divergence.

**`rulsif`.** Relative Unconstrained Least-Squares Importance Fitting. Closed-form, fast, no hyperparameter tuning. Use when all your features are continuous -- model scores, numeric rating factors, age bands encoded as integers. Faster than CatBoost on large datasets. No feature attribution.

**`kliep`.** The reference algorithm from the covariate shift literature. Explicitly enforces the normalisation constraint E_source[w(x)] = 1 via projected gradient ascent. Slower, but a useful sanity check against the CatBoost result. Also all-continuous only.

For standard UK motor with mixed feature types, use `catboost`. For a model monitoring pipeline where you are comparing score distributions rather than raw features, `rulsif` is faster and sufficient.

---

## What this does not fix

Importance weighting corrects for distribution shift in the feature space. It does not correct for:

**Label shift.** If the underlying claim relationship p(y|x) has genuinely changed -- an increase in claims inflation, a new fraud pattern, a court ruling on whiplash -- no amount of density ratio correction will help. Importance weighting assumes p(y|x) is stable across source and target. If that assumption is wrong, the ESS ratio and KL divergence will look moderate even as your model quietly mispredicts in new ways. You need claim development triangles and A/E monitoring by segment to catch label shift.

**Very large shifts.** The SEVERE verdict (ESS < 0.30) is a meaningful boundary. Below it, the correction is directionally informative but high-variance. A few extreme weights are doing most of the work, and the reweighted estimates become sensitive to the specific source observations that happen to resemble the target. In this regime, retrain.

**Model selection.** If the source model uses features that are not available on the target book -- say, a telematics score that exists for direct policyholders but not broker policyholders -- no reweighting helps. You need a different model structure.

---

## Verdict thresholds

| Verdict | ESS ratio | KL divergence | Action |
|---------|-----------|---------------|--------|
| NEGLIGIBLE | >= 0.60 | <= 0.10 nats | Deploy as-is |
| MODERATE | 0.30-0.60 | 0.10-0.50 nats | Apply importance weighting, monitor 3 months |
| SEVERE | < 0.30 | > 0.50 nats | Retrain before deploying |

---


**Using AUC as a complement to ESS.** The CatBoost classifier's AUC between source and target is a useful secondary diagnostic alongside the ESS ratio. AUC near 0.5 means distributions are similar  -  importance weighting makes a small difference. AUC near 1.0 means near-complete separation between the books. The FCA governance thresholds for an MGA acquisition context:

| Verdict | ESS ratio | Classifier AUC | Action |
|---------|-----------|---------------|--------|
| MINOR | >= 0.70 | <= 0.80 | Standard monitoring, importance weighting optional |
| MODERATE | 0.30–0.70 | 0.80–0.95 | Apply importance weighting; revalidate when target labels arrive |
| SEVERE | < 0.30 | > 0.95 | Notify Chief Actuary; importance weighting insufficient  -  consider separate model |

For an MGA acquisition, SEVERE verdict likely requires FCA SUP 15.3 notification: the firm's underwriting risk profile has changed materially and the pricing model's domain of validity has shifted. Consumer Duty (PS22/9) adds a second angle  -  applying a model trained on direct business to an aggregator or MGA book can produce systematic mispricing that is not intentional but produces discriminatory pricing outcomes. The `ShiftDiagnosticReport` documents that the issue was identified and addressed.

**HIGH_LEAKAGE_RISK flag.** If the shift is concentrated in features that are closely tied to claims handling rather than genuine risk characteristics  -  for example, settlement speed or reserve release patterns  -  the covariate shift assumption may not hold. The report flags this as `HIGH_LEAKAGE_RISK`. That requires actuarial judgement, not just reweighting.


## The library

```bash
uv add insurance-covariate-shift
# or
uv add insurance-covariate-shift
```

Source and notebooks at [github.com/burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift). The repository includes:

- `notebooks/benchmark_covariate_shift.py` -- the benchmark results referenced in the README: PSI improvement, A/E decile correction, metric estimation error for a 5,000-policy source vs 3,000-policy broker acquisition scenario
- Full docstrings on all three classes with minimal runnable examples

---

## See also

- [Monthly Covariate Shift Monitoring: When to Reweight and When to Retrain](/2026/03/15/covariate-shift-detection-book-mix-changes/)  -  the monitoring cadence question: ESS ratio trending over time, verdict thresholds, and the governance triggers for a retraining decision
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)  -  the `insurance-monitoring` library that adds segmented A/E and the Gini z-test to the PSI covariate shift detection described here
