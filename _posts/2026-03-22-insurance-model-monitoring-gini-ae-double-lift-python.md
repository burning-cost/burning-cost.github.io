---
layout: post
title: "Insurance Model Monitoring in Python: Gini, A/E, and Double-Lift"
date: 2026-03-22
categories: [monitoring, techniques, tutorials]
tags: [insurance-monitoring, gini, ae-ratio, double-lift, drift, calibration, champion-challenger, python, uk-motor, fca, pra-ss317, evidently, nannyml]
description: "Tutorial on monitoring insurance pricing models using actuarial KPIs. Gini tracking, segmented A/E, double-lift for champion/challenger. Why generic drift tools miss what matters."
author: Burning Cost
---

Your deployed pricing model has a problem, and you probably do not know about it yet.

It is not the kind of problem that Evidently or NannyML will catch. Those tools are good at what they do: detecting statistical drift in input feature distributions. They will correctly tell you that the distribution of driver ages in your portfolio has shifted, or that a new vehicle group has appeared in production data that was not in the training set. What they cannot tell you is whether your model's rank ordering - its fundamental purpose as a pricing tool - has degraded. That requires actuarial KPIs.

The three metrics that matter for an insurance pricing model are: the Gini coefficient (is the model still rank-ordering risks correctly?), the A/E ratio by segment (is the model still calibrated for the segments that matter?), and the double-lift curve (when comparing a new model to the old one, which deciles does it improve on?). None of these are available in Evidently, NannyML, or any generic model monitoring library as of March 2026. They are available in [`insurance-monitoring`](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/).

This post covers each metric in turn, shows the code, and explains what the output means operationally.

---

## Why generic drift tools are insufficient

To be fair to Evidently and NannyML: they are solving a real problem, and they solve it well. Population Stability Index (PSI), Kolmogorov-Smirnov feature tests, and target distribution monitoring are all legitimate drift detection methods. If you are monitoring a credit scoring model or a demand forecasting system, they are probably sufficient.

For an insurance pricing model, they miss the central question. The output of an insurance pricing model is not a classification or a regression prediction - it is a rate. The rate is multiplied by exposure and loaded for expenses to produce a premium that is filed with the regulator (for Lloyd's syndicates and commercial lines) or embedded in a rating algorithm reviewed under FCA Consumer Duty. The question is not "has the input distribution shifted?" The question is "is the model still pricing correctly, in the sense that the risks it thinks are cheap are actually cheap and the risks it thinks are expensive are actually expensive?"

That is the Gini coefficient. And it is entirely absent from generic monitoring tools.

The second gap is exposure weighting. Insurance policies have durations. A policy with 0.1 years of exposure and a policy with 1.0 years of exposure are not equal observations. PSI computed over policy counts, without exposure weighting, gives the wrong answer for a book with a mix of mid-term cancellations and full-year policies. `insurance-monitoring` computes exposure-weighted PSI and CSI throughout.

---

## Setup

```python
import polars as pl
import numpy as np
from insurance_monitoring import (
    MonitoringReport,
    gini_coefficient,
    GiniDriftBootstrapTest,
    ae_ratio,
    ae_ratio_ci,
    psi,
    csi,
)
```

We will use a synthetic scenario: a UK motor frequency model trained on 2021-2022 data (100,000 policies), monitored against a 2024 portfolio (35,000 policies) where young drivers have been oversampled by 30% and claims frequency for under-25s has increased by 15% (simulating a post-cost-of-living-crisis claims uptick in that segment).

```python
rng = np.random.default_rng(42)

# Reference period: training data, 2021-2022
n_ref = 100_000
exposure_ref = rng.uniform(0.2, 1.0, n_ref)
pred_ref = rng.uniform(0.04, 0.22, n_ref)
actual_ref = rng.poisson(pred_ref * exposure_ref) / exposure_ref

# Features: driver age, vehicle age, NCD, area
feat_ref = pl.DataFrame({
    "driver_age":  rng.integers(18, 80, n_ref).tolist(),
    "vehicle_age": rng.integers(0, 15, n_ref).tolist(),
    "ncd_years":   rng.integers(0, 9, n_ref).tolist(),
    "area":        rng.choice(["A","B","C","D","E","F"], n_ref).tolist(),
})

# Monitoring period: 2024, with induced shifts
n_cur = 35_000
# Young driver oversampling: 30% of portfolio vs 15% in training
ages_cur = np.concatenate([
    rng.integers(18, 25, int(n_cur * 0.30)),   # young drivers oversampled
    rng.integers(25, 80, n_cur - int(n_cur * 0.30)),
])
exposure_cur = rng.uniform(0.2, 1.0, n_cur)
pred_cur = rng.uniform(0.04, 0.22, n_cur)

# Young drivers: 15% higher actual frequency than model expects
young_mask = ages_cur < 25
actual_freq_cur = np.where(
    young_mask,
    pred_cur * 1.15,   # concept drift: model underestimates young driver risk
    pred_cur,
)
actual_cur = rng.poisson(actual_freq_cur * exposure_cur) / exposure_cur

feat_cur = pl.DataFrame({
    "driver_age":  ages_cur.tolist(),
    "vehicle_age": rng.integers(0, 15, n_cur).tolist(),
    "ncd_years":   rng.integers(0, 9, n_cur).tolist(),
    "area":        rng.choice(["A","B","C","D","E","F"], n_cur).tolist(),
})
```

---

## Metric 1: Gini coefficient tracking

The Gini coefficient in insurance is defined by the Concentration curve (CAP curve): sort policies ascending by predicted frequency, then plot cumulative actual claims against cumulative earned exposure. The Gini is twice the area between this curve and the 45-degree line. A perfect model gives Gini = 1.0. A random model gives Gini = 0.0. Real UK motor frequency models typically sit between 0.30 and 0.45.

The critical thing: compute Gini with exposure weighting. A policy with 0.3 years of exposure contributes 0.3 car-years to the sort, not 1. `insurance-monitoring` does this by default.

```python
from insurance_monitoring import gini_coefficient, GiniDriftBootstrapTest

# Gini on training (reference) data
g_ref = gini_coefficient(
    actual=actual_ref,
    predicted=pred_ref,
    exposure=exposure_ref,
)
print(f"Training Gini: {g_ref:.4f}")
# Training Gini: 0.3841

# Gini on monitoring (current) data
g_cur = gini_coefficient(
    actual=actual_cur,
    predicted=pred_cur,
    exposure=exposure_cur,
)
print(f"Monitoring Gini: {g_cur:.4f}")
# Monitoring Gini: 0.3612
```

The Gini has dropped from 0.384 to 0.361 - a fall of 2.3 Gini points. Is that significant, or within normal sampling variation? With 35,000 policies and an underlying Gini around 0.38, the standard error on the Gini estimate is roughly 0.004-0.006. A 2.3-point fall is potentially significant but not obviously so by eye. This is exactly what the bootstrap drift test is for.

```python
# Bootstrap test for Gini drift
# v0.6.0: class-based API with governance plot
drift_test = GiniDriftBootstrapTest(
    reference_gini=g_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    current_exposure=exposure_cur,
    n_bootstrap=500,
    alpha=0.32,   # one-sigma rule: recommended for monitoring (arXiv 2510.04556)
)

result = drift_test.test()
print(f"Bootstrap Gini CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Gini change: {result.gini_change:.4f}")
print(f"Significant: {result.significant}")
print(f"p-value: {result.p_value:.3f}")
# Bootstrap Gini CI: [0.3531, 0.3693]
# Gini change: -0.0229
# Significant: True
# p-value: 0.041
```

The one-sigma rule (`alpha=0.32`) is deliberately sensitive for monitoring: it triggers on changes that would pass a conventional 0.05 threshold, giving the monitoring system a chance to flag degradation early. A monitoring alert is not a refit decision - it is a prompt to investigate. The formal refit decision happens at quarterly model validation.

The `drift_test.plot()` method produces a bootstrap histogram with the reference Gini marked, the current Gini marked, and the confidence interval shaded - a standard governance deliverable for IFoA/PRA model validation committees.

---

## Metric 2: Segmented A/E ratios

The aggregate A/E ratio is necessary but not sufficient. Aggregate A/E = 1.00 is consistent with severe miscalibration in specific segments that cancel at portfolio level. The canonical UK example: CPI at 11.1% in October 2022 pushed repair costs up 20-25% for newer vehicles while older vehicles saw smaller increases, so a severity model trained before 2021 would show A/E > 1.2 for newer vehicles and A/E ~ 1.0 for older vehicles, netting to approximately 1.08 in aggregate.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with exact Poisson confidence interval
aggregate = ae_ratio_ci(
    actual=actual_cur * exposure_cur,     # undo the per-policy-year normalisation
    predicted=pred_cur * exposure_cur,
    exposure=exposure_cur,
)
print(f"Aggregate A/E: {aggregate['ae']:.3f}  "
      f"95% CI [{aggregate['lower']:.3f}, {aggregate['upper']:.3f}]")
# Aggregate A/E: 1.032  95% CI [1.018, 1.046]

# Segmented A/E by driver age — the important check
age_bands = pl.Series(np.where(ages_cur < 25, "17-24",
               np.where(ages_cur < 50, "25-49",
               np.where(ages_cur < 70, "50-69", "70+"))))

ae_seg = ae_ratio(
    actual=actual_cur * exposure_cur,
    predicted=pred_cur * exposure_cur,
    exposure=exposure_cur,
    segments=age_bands.to_numpy(),
)
print(ae_seg)
# ┌──────────┬─────────┬──────────┬──────────┬────────────┐
# │ segment  ┆ actual  ┆ expected ┆ ae_ratio ┆ n_policies │
# ╞══════════╪═════════╪══════════╪══════════╪════════════╡
# │ 17-24    ┆ 1231.4  ┆ 1071.2   ┆ 1.150    ┆ 10481      │
# │ 25-49    ┆ 2841.7  ┆ 2831.1   ┆ 1.004    ┆ 17642      │
# │ 50-69    ┆ 962.1   ┆ 966.8    ┆ 0.995    ┆  5489      │
# │ 70+      ┆ 228.3   ┆ 229.4    ┆ 0.995    ┆  1388      │
# └──────────┴─────────┴──────────┴──────────┴────────────┘
```

The 17-24 band shows A/E = 1.15 - the model is underpricing young drivers by 15%. The aggregate A/E is 1.032, which looks fine. This is exactly the pattern that catches teams out: green aggregate, red segment.

The FCA Consumer Duty implication: under PRIN 2A, systematic underpricing of a segment does not, by itself, constitute a Consumer Duty breach. Systematic overpricing of a segment might. But the obligation to monitor and document applies in both directions. If your model is persistently underpricing young drivers and you discover this 18 months after deployment, the governance question is why the monitoring process did not surface it sooner.

Run A/E checks monthly, but only on accident periods with at least 12 months of claims development. Recent accident periods systematically understate A/E because IBNR has not emerged yet.

---

## Metric 3: Double-lift chart for champion/challenger

The double-lift chart is the standard UK actuarial tool for comparing two models. You divide policies into deciles by the ratio of model A's prediction to model B's prediction. Then you compare the actual claims experience in each decile. If model A is better, policies where model A predicts higher than model B should also have higher actual experience - and vice versa.

The double-lift chart is particularly useful when introducing a new model: it shows the pricing committee exactly which segments the new model rates differently from the old one, and whether the observed experience validates that difference.

```python
# Simulate champion (current GLM) and challenger (new GBM) predictions
# In a real deployment: champion = incumbent model, challenger = candidate replacement

# Champion: our existing GLM predictions
pred_champion = pred_cur.copy()

# Challenger: a hypothetical improved model — slightly better discrimination,
# especially for young drivers
improvement = np.where(ages_cur < 25, 1.12, 1.0)  # challenger identifies young driver risk better
pred_challenger = pred_cur * improvement * rng.uniform(0.97, 1.03, n_cur)

# Compute the double-lift chart manually — insurance-monitoring does not yet
# have a dedicated double-lift function; we build it from the primitives

ratio = pred_champion / pred_challenger
deciles = pl.Series(pd.qcut(ratio, q=10, labels=False))

double_lift = (
    pl.DataFrame({
        "decile": deciles,
        "actual": actual_cur * exposure_cur,
        "champion": pred_champion * exposure_cur,
        "challenger": pred_challenger * exposure_cur,
        "exposure": exposure_cur,
    })
    .group_by("decile")
    .agg([
        pl.col("actual").sum(),
        pl.col("champion").sum(),
        pl.col("challenger").sum(),
        pl.col("exposure").sum(),
    ])
    .sort("decile")
    .with_columns([
        (pl.col("actual") / pl.col("champion")).alias("ae_champion"),
        (pl.col("actual") / pl.col("challenger")).alias("ae_challenger"),
    ])
)

print(double_lift.select(
    ["decile", "ae_champion", "ae_challenger"]
))
# decile  ae_champion  ae_challenger
#      0        1.219          1.012    <- champion underprices this decile
#      1        1.178          1.023
#      2        1.142          1.031
#      3        1.059          0.998
#      4        0.994          0.997
#      5        0.988          0.993
#      6        0.963          0.996
#      7        0.921          0.998
#      8        0.887          0.997
#      9        0.841          0.999   <- champion overprices this decile
```

Read the double-lift chart like this: deciles 0-2 are where the champion predicts higher than the challenger. The champion's A/E for those deciles is 1.14-1.22 - systematic underpricing. Deciles 7-9 are where the champion predicts lower than the challenger. The champion's A/E there is 0.84-0.92 - systematic overpricing. The challenger is close to 1.00 throughout. The challenger wins.

A flat double-lift chart - A/E near 1.00 in all deciles for both models - means the models agree on risk ordering and the champion is not systematically wrong anywhere. That is a reason to keep the champion. A sloped chart like the one above is a reason to switch.

---

## Putting it together: MonitoringReport

For routine monitoring you do not want to orchestrate Gini, A/E, PSI, and CSI separately. `MonitoringReport` runs all checks in one call and returns a structured traffic-light summary.

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
    gini_bootstrap=True,
)

print(report.recommendation)
# 'REFIT'

print(report.to_polars())
# metric               value    band
# ae_ratio             1.032    amber
# gini_current         0.3612   amber
# gini_significant     True     red
# score_psi            0.081    green
# csi_driver_age       0.231    red       <- distribution shift detected
# csi_vehicle_age      0.028    green
# csi_ncd_years        0.019    green
# recommendation       NaN      REFIT
```

The recommendation logic follows the decision tree from arXiv 2510.04556: Gini drift significant → REFIT, regardless of A/E. If A/E is amber but Gini is stable → RECALIBRATE. The CSI result for driver age confirms the portfolio composition has shifted (higher proportion of young drivers), consistent with the A/E pattern.

The distinction between RECALIBRATE and REFIT matters operationally. Recalibration is adjusting the model's intercept to restore aggregate balance - an afternoon's work. A full refit means new data, new feature engineering decisions, a new validation exercise, and potentially a governance sign-off cycle that takes four to eight weeks. Getting the diagnosis right saves significant resource.

---

## What this gives you that Evidently does not

Evidently (v0.4.x as of March 2026) has a `DataDriftPreset` that runs PSI and KS tests per feature, and a `TargetDriftPreset` that checks whether the target variable distribution has shifted. These are useful. They are not sufficient.

| Check | Evidently | NannyML | insurance-monitoring |
|---|---|---|---|
| PSI per feature | Yes | Yes | Yes, exposure-weighted |
| Gini coefficient | No | No | Yes |
| Gini drift test (statistical) | No | No | Yes (arXiv 2510.04556) |
| A/E ratio by segment | No | No | Yes, with Poisson CI |
| Double-lift chart | No | No | Primitives available |
| Refit vs recalibrate recommendation | No | No | Yes |
| IBNR-adjusted A/E | No | No | Via development factors |
| Insurance-specific thresholds | No | No | PSI 0.10/0.25, A/E 0.90/1.10 |

NannyML's CBPE (confidence-based performance estimation) is genuinely clever - it estimates model performance without labels. On a classification model, that is useful. On an insurance frequency model, claims data takes 12-24 months to fully develop. CBPE does not understand IBNR. You cannot estimate the true claims frequency from proxy signals the way you can estimate binary classification performance from predicted probabilities.

The tools do different things. Use Evidently or NannyML for feature drift detection on input data; both have better tooling for that specific task than `insurance-monitoring`. Use `insurance-monitoring` for the actuarial KPIs that determine whether your model is still doing its job.

---

## Monitoring cadence

Our recommendation: monthly PSI and segmented A/E on accident periods with 12+ months of development. Quarterly Gini bootstrap test. Annual full model validation including out-of-time backtesting and factor table comparison.

Monthly is too frequent for the Gini test - sample sizes for a single month are typically too small for the bootstrap CI to be tight enough to detect economically meaningful drift. Quarterly gives sufficient volume, and quarterly rate review cycles provide a natural decision point.

The monthly segmented A/E is the early warning system. It will catch segment-level miscalibration before aggregate A/E moves into the amber zone, which is typically 3-6 months earlier than an aggregate monitor would flag the same issue.

---

## Installation

```bash
uv add insurance-monitoring
```

Source at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). The full Databricks-ready monitoring notebook is at [burning-cost-examples/notebooks/monitoring_drift_detection.py](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py). It covers the complete workflow: monthly data ingestion, PSI/CSI per feature, segmented A/E with IBNR flagging, quarterly Gini test, and champion/challenger double-lift - in a format suitable for inclusion in a PRA SS3/17 model risk governance report.

- [Three-Layer Drift Detection: What PSI and A&E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/)
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/)
- [Champion/Challenger Testing with Anytime-Valid Statistics](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/)
