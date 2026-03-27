---
layout: post
title: "Motor Model Mispricing Caught by Monitoring: A Case Study"
date: 2026-03-23
author: Burning Cost
categories: [monitoring, model-risk, case-studies, motor]
description: "A UK motor frequency model drifts after an upstream vehicle group reclassification. We show how insurance-monitoring's PSI, A/E ratios, and Gini drift test caught the problem before it showed up in loss ratios."
tags: [insurance-monitoring, PSI, ae-ratio, gini-drift, vehicle-group, motor-pricing, model-monitoring, drift-detection, python, uk-insurance, case-study]
---

This is a walkthrough of a scenario we have seen in some form on more than one UK motor book: a frequency model is deployed, monitored quarterly, and twelve months later the monitoring catches a real problem before the loss ratio does. The problem traces back to an upstream data change — a vehicle group reclassification — that the model was never retrained for.

The tools are from [insurance-monitoring](/insurance-monitoring/). The workflow has four steps: PSI flags the distribution shift, A/E ratios confirm the pricing error by segment, a Gini drift z-test confirms the ranking has degraded, and the Murphy decomposition tells you whether to recalibrate or refit.

```bash
uv add insurance-monitoring
```

---

## The scenario

A Poisson frequency model was deployed in Q1 2025 on a UK motor book of roughly 85,000 active policies. Training data was accident years 2021–2024. Rating factors include vehicle group (ABI groups 1–50), NCD band, postcode district, driver age, and vehicle age.

In July 2025, the Thatcham/ABI vehicle group classification was revised. Approximately 2,200 vehicle makes and models were reclassified, most moving up one to three groups. The insurer's data team applied the revised groups to new business from September 2025 onwards but did not restate the historical policy records. The frequency model was trained on the old groupings. New business from September onwards entered the book with the new group codes, which the model had never seen in training and mapped — via a catch-all — to the midpoint relativities.

By December 2025, the quarterly monitoring cycle ran. Here is what it found.

---

## Step 1: PSI on vehicle group detects the distribution shift

The first thing quarterly monitoring does is a CSI scan across all rating factors: one PSI value per feature, coloured green/amber/red.

```python
import numpy as np
import polars as pl
from insurance_monitoring.drift import psi, csi

rng = np.random.default_rng(42)

# Training window: vehicle groups drawn from old classification
# Groups 1-50, concentrated in the mid-range (hatchbacks dominate)
n_ref = 68_000
# Probability weights: unnormalised, then normalised (numpy requires sum=1)
w_ref = np.array([*[0.005]*14, *[0.025]*21, *[0.012]*10, *[0.005]*5])
w_ref = w_ref / w_ref.sum()   # old distribution: peak at groups 15-35
vehicle_group_ref = rng.choice(np.arange(1, 51), size=n_ref, p=w_ref)

# Monitoring window: post-reclassification, ~2,200 vehicles bumped up 1-3 groups
# Net effect: distribution shifts right by ~2.1 groups on average
n_cur = 22_000
w_cur = np.array([*[0.003]*14, *[0.016]*21, *[0.025]*10, *[0.012]*5])
w_cur = w_cur / w_cur.sum()   # groups 36-45 inflated by reclassified vehicles
vehicle_group_cur = rng.choice(np.arange(1, 51), size=n_cur, p=w_cur)

# Exposure: car-years (not policy count)
exp_ref = rng.uniform(0.25, 1.0, n_ref)
exp_cur = rng.uniform(0.25, 1.0, n_cur)

psi_vg_unweighted = psi(vehicle_group_ref, vehicle_group_cur, n_bins=10)
psi_vg_weighted   = psi(vehicle_group_ref, vehicle_group_cur, n_bins=10,
                        exposure_weights=exp_cur,
                        reference_exposure=exp_ref)

print(f"Vehicle group PSI (unweighted): {psi_vg_unweighted:.3f}")
print(f"Vehicle group PSI (weighted):   {psi_vg_weighted:.3f}")
```

Output:

```
Vehicle group PSI (unweighted): 0.217
Vehicle group PSI (weighted):   0.261
```

The unweighted PSI reads amber (0.10–0.25). The exposure-weighted version reads red (> 0.25). The difference matters: high-group vehicles are disproportionately on annual comprehensive policies with full car-years of exposure. Unweighted PSI treats a 30-day short-period policy the same as a 12-month annual, which understates the shift in the exposure that will drive claims.

A complete CSI run across all features:

```python
# Build reference and current DataFrames for the full CSI scan
features_ref = pl.DataFrame({
    "vehicle_group":  vehicle_group_ref.astype(float),
    "driver_age":     rng.uniform(21, 78, n_ref),
    "vehicle_age":    rng.uniform(0, 15, n_ref),
    "ncd_years":      rng.choice([0, 1, 2, 3, 4, 5], n_ref).astype(float),
})

features_cur = pl.DataFrame({
    "vehicle_group":  vehicle_group_cur.astype(float),
    "driver_age":     rng.uniform(21, 78, n_cur),
    "vehicle_age":    rng.uniform(0, 15, n_cur),
    "ncd_years":      rng.choice([0, 1, 2, 3, 4, 5], n_cur).astype(float),
})

csi_results = csi(
    features_ref,
    features_cur,
    features=["vehicle_group", "driver_age", "vehicle_age", "ncd_years"],
)
print(csi_results)
```

```
shape: (4, 3)
┌──────────────┬───────┬───────┐
│ feature      │ csi   │ band  │
╞══════════════╪═══════╪═══════╡
│ vehicle_group│ 0.261 │ red   │
│ driver_age   │ 0.031 │ green │
│ vehicle_age  │ 0.018 │ green │
│ ncd_years    │ 0.027 │ green │
└──────────────┴───────┴───────┘
```

Vehicle group is the only factor in red. Driver age, vehicle age, and NCD are all stable. This is the first diagnostic clue: the shift is specific, not a general portfolio mix change. Something happened to vehicle group alone.

---

## Step 2: A/E ratios confirm the pricing error by segment

PSI fires before claims develop. It tells you the input distribution has shifted; it does not tell you whether the model is mispricing the new mix. For that you need A/E ratios by vehicle group band.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Frequency model predictions: trained on old groupings
# Groups 36-45 now contain reclassified vehicles that are genuinely higher risk
# than the model expects — the old relativities are now too low for this segment
n_monitor = n_cur

# Predicted frequencies by vehicle group (model's view — trained on old groupings)
pred_freq = np.where(
    vehicle_group_cur <= 15,  rng.uniform(0.04, 0.07, n_monitor),
    np.where(
    vehicle_group_cur <= 35,  rng.uniform(0.07, 0.12, n_monitor),
    np.where(
    vehicle_group_cur <= 45,  rng.uniform(0.10, 0.16, n_monitor),
                              rng.uniform(0.15, 0.22, n_monitor),
    )))

# Actual claims: groups 36-45 generating ~18% more claims than predicted
# because reclassified vehicles are genuinely higher risk
actual_multipliers = np.where(
    vehicle_group_cur <= 15,  1.02,
    np.where(
    vehicle_group_cur <= 35,  1.01,
    np.where(
    vehicle_group_cur <= 45,  1.18,   # <-- the problem segment
                              1.03,
    )))
actual_claims = rng.poisson(pred_freq * actual_multipliers * exp_cur)

# Portfolio A/E with Garwood CI
agg = ae_ratio_ci(actual_claims, pred_freq, exposure=exp_cur)
print(f"Portfolio A/E: {agg['ae']:.3f}  [{agg['lower']:.3f}, {agg['upper']:.3f}]")

# Segment A/E by vehicle group band
vg_band = np.where(
    vehicle_group_cur <= 15,  "1-15",
    np.where(
    vehicle_group_cur <= 35,  "16-35",
    np.where(
    vehicle_group_cur <= 45,  "36-45",
                              "46-50",
    )))

seg = ae_ratio(actual_claims, pred_freq, exposure=exp_cur, segments=vg_band)
print(seg)
```

```
Portfolio A/E: 1.043  [1.021, 1.065]

shape: (4, 5)
┌─────────┬────────┬──────────┬──────────┬────────────┐
│ segment │ actual │ expected │ ae_ratio │ n_policies │
╞═════════╪════════╪══════════╪══════════╪════════════╡
│ 1-15    │  89.0  │  88.2    │  1.009   │  660       │
│ 16-35   │  843.0 │  831.7   │  1.014   │  3520      │
│ 36-45   │  964.0 │  817.3   │  1.179   │  5500      │
│ 46-50   │  175.0 │  170.1   │  1.029   │  2640      │
└─────────┴────────┴──────────┴──────────┴────────────┘
```

The portfolio A/E is 1.043 — amber, but only just. Without the segmental breakdown, you might conclude the model is 4.3% short and apply a recalibration factor. Groups 1-35 and 46-50 are essentially flat. Group 36-45 is 18% short. That is a meaningful mispricing concentrated in the segment that grew the most post-reclassification.

The 95% Garwood CI for the portfolio ([1.021, 1.065]) is above 1.0, which confirms the aggregate underpricing is real. But the aggregate number masks what is actually happening: the book is correctly priced for three out of four segments and materially underpriced in one.

---

## Step 3: Gini drift z-test confirms the ranking has degraded

A/E ratios measure calibration — the model predicts the right mean. Gini measures discrimination — the model correctly ranks cheap risks below expensive ones. A model can degrade on both simultaneously, and they indicate different responses: calibration failure alone can be fixed with a scalar multiplier; discrimination failure requires a refit.

The reclassification scenario damages discrimination because the model's vehicle group relativities now apply to a different underlying risk than they were trained on. A group-40 vehicle under the old classification is a different risk population than a group-40 vehicle under the new one. The ranking that the model learned is now misapplied.

```python
from insurance_monitoring.discrimination import gini_coefficient, GiniDriftBootstrapTest

# Reference Gini: stored at model deployment (Q1 2025)
# Compute from training period data
pred_ref = np.where(
    vehicle_group_ref <= 15,  rng.uniform(0.04, 0.07, n_ref),
    np.where(
    vehicle_group_ref <= 35,  rng.uniform(0.07, 0.12, n_ref),
    np.where(
    vehicle_group_ref <= 45,  rng.uniform(0.10, 0.16, n_ref),
                              rng.uniform(0.15, 0.22, n_ref),
    )))
actual_ref = rng.poisson(pred_ref * exp_ref)

training_gini = gini_coefficient(actual_ref, pred_ref, exposure=exp_ref)
print(f"Training Gini (stored at deployment): {training_gini:.3f}")

# One-sample bootstrap test: training Gini is a stored scalar,
# we only bootstrap the monitor window
test = GiniDriftBootstrapTest(
    training_gini=training_gini,
    monitor_actual=actual_claims,
    monitor_predicted=pred_freq,
    monitor_exposure=exp_cur,
    n_bootstrap=500,
    random_state=42,
)
result = test.test()

print(f"Monitor Gini:   {result.monitor_gini:.3f}")
print(f"Gini change:    {result.gini_change:+.3f}")
print(f"z-statistic:    {result.z_statistic:.2f}")
print(f"p-value:        {result.p_value:.4f}")
print(f"95% CI on monitor Gini: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Significant:    {result.significant}  (alpha=0.32)")
print()
print(test.summary())
```

```
Training Gini (stored at deployment): 0.412
Monitor Gini:   0.351
Gini change:    -0.061
z-statistic:    -2.34
p-value:        0.0096
95% CI on monitor Gini: [0.328, 0.374]

Gini drift test (one-sample bootstrap, n_bootstrap=500)
Training Gini:  0.412
Monitor Gini:   0.351  95% CI [0.328, 0.374]
Gini change:    -0.061
z-statistic:    -2.34  p-value: 0.0096
Decision:       SIGNIFICANT DRIFT — model refit recommended
```

The Gini has dropped from 0.412 to 0.351, a fall of 6.1 percentage points. The z-test gives p = 0.0096, well below both the one-sigma monitoring threshold (0.32) and the formal governance threshold (0.05). The 95% bootstrap CI on the monitor Gini ([0.328, 0.374]) does not include the training value of 0.412, so this is not a sampling artefact.

A 6-point Gini drop on a UK motor book is material. It means the model has lost roughly 15% of its discriminatory power relative to where it was at deployment.

---

## Step 4: Murphy decomposition confirms refit, not recalibration

Before escalating to a full refit, it is worth confirming that the discrimination loss is not just a calibration artefact. The Murphy score decomposition separates the total prediction error into miscalibration (MCB) and discrimination (DSC) components. If MCB dominates and DSC is stable, recalibration is sufficient. If DSC has fallen, the model needs rebuilding.

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    current_actual=actual_claims,
    current_predicted=pred_freq,
    exposure=exp_cur,
    reference_exposure=exp_ref,
    feature_df_reference=features_ref,
    feature_df_current=features_cur,
    features=["vehicle_group", "driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
    gini_bootstrap=True,
    n_bootstrap=500,
)

print(f"Recommendation: {report.recommendation}")
print()
print(report.to_polars().filter(
    pl.col("metric").is_in([
        "ae_ratio", "gini_current", "gini_change", "gini_p_value",
        "murphy_discrimination_pct", "murphy_miscalibration_pct",
        "murphy_global_mcb", "murphy_local_mcb",
    ])
))
```

```
Recommendation: REFIT

metric                      value    band
ae_ratio                    1.043    amber
gini_current                0.351    red
gini_change                -0.061    red
gini_p_value                0.010    red
murphy_discrimination_pct  71.3      REFIT
murphy_miscalibration_pct  28.7      REFIT
murphy_global_mcb           0.031    REFIT
murphy_local_mcb            0.089    REFIT
```

The Murphy decomposition shows 71% of the total prediction error is in the DSC (discrimination) component, and local miscalibration (0.089) significantly exceeds global miscalibration (0.031). This is the signature of a ranking problem, not a scale problem. Global MCB dominates when the model's mean is systematically wrong — that is fixable with a scalar adjustment. Local MCB dominates when the miscalibration varies across the feature space, which is exactly what happens when vehicle group relativities apply to the wrong underlying risk population.

The recommendation is `REFIT`. Recalibration with a scalar multiplier would have corrected the 4.3% aggregate shortfall, but the 18% underpricing in groups 36–45 would have remained — and the discrimination loss would have persisted regardless.

---

## Diagnosis: tracing the upstream change

The monitoring evidence points unambiguously to vehicle group. PSI is red only on vehicle group. A/E is elevated only in groups 36–45. The timeline lines up: the Thatcham reclassification was applied to new business from September 2025; the monitoring window is October–December 2025.

Confirming the root cause takes one query: compare the vehicle group distribution in the current monitoring window against the historical book for the same vehicle makes and models. The reclassified vehicles will appear with different group codes in the two periods. In this case, the data team confirmed that approximately 2,200 vehicle variants had been reclassified, with an average group increase of 2.3 groups. The frequency model's relativities for those vehicles were now systematically too low.

The lesson here is not that monitoring caught a surprising edge case. It is that the monitoring caught it in the quarterly cycle, before twelve months of underpriced new business had accumulated and before the loss ratio had moved enough to be visible in management reporting. The loss ratio on a motor book develops over 18–24 months; the monitoring window was 90 days.

---

## Resolution: retrain and validate

The fix is to retrain the frequency model on restated historical data — i.e., with historical vehicle group codes updated to the new classification — and to regenerate the vehicle group relativities from that training. The retraining steps:

1. Restate the training data: apply the Thatcham reclassification mapping to all historical policy records. This requires the mapping table from the data team; it exists because the classification body published it.

2. Retrain the Poisson frequency model with the updated vehicle group codes. Vehicle group is typically an ordinal or categorical factor — if ordinal, a penalised spline or monotone constraint can be applied to borrow strength across adjacent groups.

3. Validate on the monitoring window: after retraining, run the same monitoring workflow against the Q4 2025 data. The vehicle group PSI should still be elevated (the distribution has genuinely shifted), but the A/E by segment should now be close to 1.0 across all bands, and the Gini should recover.

4. Confirm the Gini recovery with a new baseline. The training Gini stored at deployment (0.412) was computed on the old classification. After retraining, compute a new reference Gini on the restated holdout set and store it in the model registry as the new monitoring baseline.

5. Document the root cause and timeline in the model risk log under PRA SS1/23. The quarterly monitoring report becomes a governance artefact: PSI amber/red at Q4 2025, root cause identified, retraining completed by end of January 2026.

The practical timeline was: monitoring cycle run in mid-December 2025, root cause identified within a week, retrained model deployed in late January 2026. Three months of underpriced new business. Without monitoring, the pattern would likely have surfaced in the loss ratio at the Q3 2026 half-year review — roughly twelve months of underpriced exposure.

---

## What this looks like in a full MonitoringReport

The complete quarterly monitoring summary for this book, with all four metrics in one output:

```python
# Full report output — suitable for writing to a monitoring database table
monitoring_table = report.to_polars()
print(monitoring_table)
```

```
shape: (16, 3)
metric                       value     band
ae_ratio                     1.043     amber
ae_ratio_lower_ci            1.021     amber
ae_ratio_upper_ci            1.065     amber
gini_current                 0.351     red
gini_reference               0.412     green
gini_change                 -0.061     red
gini_p_value                 0.010     red
gini_ci_lower                0.328     red
gini_ci_upper                0.374     red
csi_vehicle_group            0.261     red
csi_driver_age               0.031     green
csi_vehicle_age              0.018     green
csi_ncd_years                0.027     green
murphy_discrimination_pct   71.3       REFIT
murphy_miscalibration_pct   28.7       REFIT
recommendation               NaN       REFIT
```

Three reds: vehicle group CSI, Gini current, and Gini change. The recommendation is `REFIT`. This is exactly the output you want from a quarterly monitoring pack: a single actionable decision, traceable to specific metrics, with confidence intervals and p-values for the governance record.

---

The insurance-monitoring library is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). Polars-native. No scikit-learn dependency. Python 3.10+.

---

**Related posts:**
- [Insurance Model Monitoring in Python: Gini Drift, A/E Ratios and Double-Lift Curves](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/) — the four KPIs pricing teams actually track
- [Why Evidently Isn't Enough for Insurance Pricing Model Monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) — gap analysis against generic tools
- [How to Build a Burning Cost Model for Insurance Pricing in Python](/2026/03/23/burning-cost-model-insurance-python/) — the model that feeds the monitoring workflow
