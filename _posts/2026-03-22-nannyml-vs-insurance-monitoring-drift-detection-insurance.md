---
layout: post
title: "NannyML vs insurance-monitoring: Drift Detection Built for Insurance Portfolios"
date: 2026-03-22
author: Burning Cost
categories: [monitoring, model-risk, libraries, comparisons]
description: "NannyML is the best general-purpose ML monitoring library for teams without ground truth labels. For insurance pricing, it doesn't do exposure-weighted PSI, segmented A/E ratios, Gini discrimination drift, or PRA SS3/17 framing. Here is the direct comparison."
tags: [nannyml-insurance, nannyml-model-monitoring-insurance, NannyML, insurance-monitoring, CBPE, PSI, ae-ratio, gini-drift, mSPRT, sequential-testing, model-drift-insurance-pricing, PRA-SS3-17, python, uk-insurance, polars]
---

If you search "ML model monitoring Python", NannyML appears at or near the top. It earns that position. The confidence-based performance estimation (CBPE) capability - estimating model performance metrics when ground truth labels are delayed or unavailable - is genuinely clever and has no close equivalent in the open-source ecosystem. The visualisations are the best in class across monitoring tools. For a data science team monitoring a classification model in a delayed-label setting, NannyML is probably the right default.

For insurance pricing model monitoring, it misses the point in ways that compound.

This post is the direct comparison: what NannyML does, where `insurance-monitoring` diverges, and the exact decision tree for which to use when. If you are not monitoring insurance pricing models, use NannyML - this post is not for you.

---

## What NannyML does well

The critique below is honest, so the credit should be too.

NannyML's CBPE is its headline feature, and it deserves the attention. In a typical insurance model deployment the ground truth - actual claims experience - arrives 9 to 18 months after the policy inception. CBPE estimates model performance (AUC, accuracy, F1, or similar) using only the model's confidence scores on unlabelled analysis data, without waiting for ground truth. The API is minimal:

```python
import nannyml as nml

estimator = nml.CBPE(
    problem_type='classification_binary',
    y_pred_proba='predicted_probability',
    y_pred='prediction',
    y_true='employed',
    metrics=['roc_auc'],
    chunk_size=5000,
)
estimator.fit(reference_df)
estimated_performance = estimator.estimate(analysis_df)
figure = estimated_performance.plot()
```

The plots NannyML produces - chunked time series with alert shading, confidence bands, reference versus analysis overlays - are the kind of thing you could show to a CRO without explanation. Clean, well-designed, and correct.

The univariate drift detection is broad: KS, Jensen-Shannon, Wasserstein, Hellinger, chi-squared, and PSI are all available. The `UnivariateDriftCalculator` runs all features in one call and produces a ranked drift table:

```python
calculator = nml.UnivariateDriftCalculator(
    column_names=feature_cols,
    chunk_size=5000,
)
calculator.fit(reference_df)
drift = calculator.calculate(analysis_df)
```

`AlertCountRanker` identifies which features have drifted most across chunks - useful for triage when you have 30 rating factors and need to know where to look first.

NannyML also has `DLE` (Direct Loss Estimation), which estimates regression model performance without ground truth - the regression analogue of CBPE. For a claims severity model where paid loss takes time to mature, this is a genuine capability.

None of this changes what follows. NannyML is a well-engineered library with a clear design philosophy. The problem is that the philosophy does not transfer to insurance pricing without modification.

---

## What insurance pricing actually needs from a monitoring tool

Three requirements distinguish insurance pricing from generic ML monitoring, and NannyML addresses none of them by design.

### 1. Exposure-weighted PSI and CSI

PSI is the operational standard for insurance model drift monitoring. Every major UK insurer, Lloyd's syndicate, and regulatory submission uses some version of it. NannyML implements PSI - but unweighted.

For insurance, rows in a monitoring dataset are policies, and policies vary enormously in how much claims exposure they represent. An annual fleet policy covering 80 vehicles that ran for twelve months contributes one row to NannyML's PSI calculation. A 30-day moped policy cancelled after a month also contributes one row. In claims exposure terms, these differ by a factor of roughly 100.

The problem is not academic. UK comparison site business is disproportionately short-duration - monthly pay, annual intent, frequent lapse. Annual renewals are a different population. If young drivers are entering via monthly-pay comparison site policies while your reference data is annual renewals, unweighted PSI underestimates the driver age shift because the young-driver policies each count once despite contributing 0.25 car-years.

`insurance-monitoring` exposes an `exposure_weights` parameter directly on the PSI and CSI functions:

```python
from insurance_monitoring.drift import psi, csi
import polars as pl

# NannyML / unweighted approach: every policy counts once
# Reads amber at 0.18 in the benchmark

# Insurance-correct: weight by earned car-years
psi_weighted = psi(
    reference=driver_ages_ref,
    current=driver_ages_cur,
    n_bins=10,
    exposure_weights=earned_exposure_cur,
    reference_exposure=earned_exposure_ref,
)
# Reads red at 0.27 in the benchmark — a different governance conclusion

# Per-feature CSI heatmap across all rating factors
feature_ref = pl.DataFrame({
    "driver_age":  [...],
    "vehicle_age": [...],
    "ncd_years":   [...],
})
feature_cur = pl.DataFrame({
    "driver_age":  [...],
    "vehicle_age": [...],
    "ncd_years":   [...],
})
csi_table = csi(feature_ref, feature_cur,
                features=["driver_age", "vehicle_age", "ncd_years"])
# Returns: feature | csi | band
```

In the `insurance-monitoring` benchmark - 50,000 training policies, 2019–2021 reference, 2023 current cohort with 2x young driver oversampling - unweighted PSI reads 0.18 (amber), exposure-weighted reads 0.27 (red). These produce different governance outputs. The weighted version is correct.

### 2. Actuarial A/E ratios with statistical calibration

NannyML does not have a concept of A/E ratio. This is not a gap in implementation - it is out of scope for a general ML library.

A/E (actual claims divided by expected claims) is the primary metric for insurance model calibration monitoring. An aggregate A/E of 1.00 is reassuring. An aggregate A/E of 1.00 that is composed of 1.55 in the under-25 cohort and 0.85 in the over-50 cohort means your model is generating adverse selection: good risks are leaving (overpriced) and bad risks are staying (underpriced). NannyML's prediction drift detection will tell you the distribution of your model outputs has shifted. It cannot tell you whether actual claims experience is tracking your predictions at segment level.

`insurance-monitoring` implements segmented A/E with exact Garwood (Poisson) confidence intervals:

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with Poisson CI
result = ae_ratio_ci(actual, predicted, exposure=exposure)
print(f"A/E: {result['ae']:.3f}  [{result['lower']:.3f}, {result['upper']:.3f}]")
# A/E: 1.082  [1.031, 1.137]

# Segment-level decomposition — where is the model misfiring?
seg = ae_ratio(actual, predicted, exposure=exposure, segments=age_bands)
# segment | actual | expected | ae_ratio | n_policies
# 17-24   |    48  |      31  |    1.55  |       312
# 25-34   |    91  |      87  |    1.05  |       847
# 50+     |    61  |      72  |    0.85  |       901
```

The Garwood interval is not cosmetic. At 30 observed claims - a realistic monitoring window for a specialist product, a new entrant cohort, or a thinly traded perimeter - a normal approximation understates the confidence interval by 15–20%. Teams using normal approximations generate spurious significant deviations on thin segments, train their actuaries to discount alarms, and eventually miss real problems.

### 3. Gini discrimination drift testing

This is the failure mode that destroys books silently, and it is invisible to every general-purpose monitoring tool including NannyML.

A pricing model can maintain a stable aggregate A/E while losing its ability to rank risks. The mechanism is covariate shift: as portfolio composition changes, the model's relativities become stale. It predicts the right mean but can no longer separate a good risk from a bad risk within any cohort. The consequence is adverse selection. NannyML monitors whether the distribution of model outputs has shifted - which is useful but orthogonal to whether the model's *ranking* is still working.

`insurance-monitoring` implements the Gini drift z-test from arXiv 2510.04556 - a formal hypothesis test with bootstrap variance estimation:

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=50_000,
    n_current=15_000,
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur,   current_predicted=pred_cur,
)
# GiniDriftResult(z_statistic=-1.93, p_value=0.054, gini_change=-0.03, significant=False)
```

The default alpha is 0.32 (the one-sigma rule), following the arXiv 2510.04556 recommendation. The rationale: the cost of missing a Gini decline is measurable in loss ratio; the cost of a false-positive investigation is two actuarial days. The test produces a governance deliverable - bootstrap histogram, CI shading, training Gini line, monitor Gini line, z and p annotation - formatted to the standard expected in an IFoA model validation pack.

---

## Side-by-side: the same monitoring problem, two tools

Here is the core comparison: a UK motor portfolio, 18 months post-deployment, monitoring for the standard failure modes.

**NannyML approach:**

```python
import nannyml as nml
import pandas as pd

# Step 1: univariate drift across all features
calculator = nml.UnivariateDriftCalculator(
    column_names=["driver_age", "vehicle_age", "ncd_years", "area"],
    chunk_size=2000,
)
calculator.fit(reference_df)
drift = calculator.calculate(analysis_df)
drift.filter(column_names=["driver_age"]).plot()

# Step 2: estimate whether model performance has degraded
# (classification models only — CBPE requires confidence scores)
estimator = nml.CBPE(
    problem_type='classification_binary',
    y_pred_proba='risk_score',
    y_pred='prediction',
    y_true='had_claim',
    metrics=['roc_auc'],
    chunk_size=2000,
)
estimator.fit(reference_df)
estimated = estimator.estimate(analysis_df)
estimated.plot()
```

This tells you: (a) have any features shifted distributionally, and (b) has estimated AUC changed. Useful. But it does not tell you A/E by age band, whether the Gini has degraded significantly, whether to recalibrate or refit, or what to write in your model risk documentation.

**insurance-monitoring approach:**

```python
import polars as pl
import numpy as np
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",   # GLM family for Murphy decomposition
)

print(report.recommendation)
# 'RECALIBRATE' | 'REFIT' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

df = report.to_polars()
# metric                  | value  | band
# ae_ratio                | 1.08   | amber
# gini_current            | 0.39   | amber
# gini_p_value            | 0.054  | amber
# csi_driver_age          | 0.14   | amber
# murphy_discrimination   | 0.041  | amber
# murphy_miscalibration   | 0.003  | green
# recommendation          | nan    | RECALIBRATE
```

The `RECALIBRATE` vs `REFIT` verdict comes from the Murphy decomposition (Lindholm & Wüthrich, Scandinavian Actuarial Journal 2025): if global miscalibration (GMCB) dominates, a multiplier fix is all you need - hours of work; if local miscalibration (LMCB) dominates, the model's ranking is structurally broken - weeks of work to refit. Getting this wrong is expensive in either direction.

---

## The CBPE question

The one thing NannyML does that `insurance-monitoring` does not: CBPE.

If you are monitoring a binary classification model - a churn predictor, a fraud classifier, an underwriting accept/decline - and ground truth labels are delayed, CBPE is useful. It estimates AUC and F1 from the model's confidence distribution without waiting for observed outcomes.

But for Poisson frequency and Gamma severity models - the standard UK motor pricing GLM setup - CBPE does not apply. CBPE requires calibrated probability outputs. Poisson frequency models output expected claim rates (events per unit exposure), not calibrated binary probabilities. NannyML's documentation explicitly scopes CBPE to classification.

For insurance frequency models, the analogue of "waiting for ground truth" is the IBNR problem: monitoring recent accident months requires chain-ladder adjustment before the A/E ratio is reliable. This is an actuarial problem that NannyML does not address, and `insurance-monitoring`'s documentation explicitly flags: "the A/E ratio and balance test are only reliable on mature accident periods. For motor, at least 12 months of claims development. For liability, 24+ months."

---

## NannyML's other genuine strengths

Beyond CBPE, a few NannyML capabilities are worth naming explicitly.

**Visualisation.** NannyML's plots are genuinely better than anything `insurance-monitoring` ships. The chunked time-series monitoring plot with alert highlighting and reference shading is the right format for an executive dashboard. `insurance-monitoring` outputs structured Polars DataFrames, which are right for actuary-built pipelines but require a dashboard layer on top.

**Multivariate drift.** NannyML has `DataReconstructionDriftCalculator` - a PCA-based reconstruction error method that catches multivariate distributional shifts that column-by-column tests miss. `insurance-monitoring` has `InterpretableDriftDetector` (TRIPODD, Panda et al. 2025) for feature-attribution-aware drift, but does not implement PCA reconstruction.

**Model-agnostic breadth.** NannyML works with any model that produces predictions and optionally confidence scores - sklearn, XGBoost, LightGBM, custom pipelines. `insurance-monitoring` assumes Poisson or Tweedie distributed targets (the standard non-life actuarial setup) and has less utility outside that context.

**DLE for regression.** Direct Loss Estimation allows performance estimation for regression models without ground truth. For a claims severity model this is relevant, though the 9–18 month lag before severity claims mature is handled differently by insurers (chain-ladder reserving) than by DLE's estimation approach.

---

## When to use each

**Use NannyML if:**
- You are monitoring any ML model that is not an insurance pricing GLM or GBM
- You need ground-truth-free performance estimation (CBPE, DLE) for delayed-label settings
- You need clean monitoring visualisations for engineering or product dashboards
- You are monitoring a fraud, churn, or underwriting classification model alongside your pricing models
- You want multivariate drift detection via reconstruction error

**Use insurance-monitoring if:**
- You are monitoring a UK motor, home, commercial lines, or other non-life pricing model
- You need exposure-weighted PSI/CSI where a fleet policy contributes proportionally to car-years, not policy count
- You need segmented A/E ratios with exact Garwood confidence intervals to catch cohort-level mispricing
- You need the Gini drift z-test to detect ranking degradation before adverse selection compounds
- You need Murphy decomposition to distinguish RECALIBRATE (multiplier fix) from REFIT (model rebuild)
- You need `PITMonitor` for calibration change detection that does not false-alarm under repeated monthly testing
- You need `SequentialTest` for champion/challenger experiments that are checked monthly without FPR inflation
- You need `MonitoringReport` producing a structured output for PRA SS3/17 model risk documentation

The two tools are not alternatives for the same problem. NannyML is excellent for general ML monitoring with strong performance estimation and visualisation. `insurance-monitoring` is purpose-built for the actuarial monitoring workflow: exposure-weighted distributions, A/E by segment, Gini drift significance testing, and the RECALIBRATE-or-REFIT decision. A pricing team in a larger organisation might run both - NannyML for engineering-facing dashboards, `insurance-monitoring` for the actuarial outputs that feed model governance.

---

## Installation

```bash
pip install insurance-monitoring
# or:
uv add insurance-monitoring
```

Polars-native throughout. No scikit-learn dependency. Python 3.10+.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).

---

**Related:**
- [Alibi Detect vs insurance-monitoring: Drift Detection for Insurance Pricing Models](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) - the same comparison against Alibi Detect
- [Why Evidently Isn't Enough for Insurance Pricing Model Monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - comparison against Evidently
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - detailed walkthrough of each monitoring layer
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - why aggregate A/E is a lagging indicator for covariate shift

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/22/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/22/equipy-vs-insurance-fairness/) - optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) - causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/22/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - model monitoring
- [Alibi Detect vs insurance-monitoring](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) - statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) - distributional regression
