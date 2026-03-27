---
layout: post
title: "Model Drift Detection on 677k Policies: PSI, A/E, and Gini Tests on freMTPL2"
date: 2026-03-27
categories: [monitoring, pricing, techniques]
tags: [model-drift, psi, ae-ratio, gini-drift, monitoring, frequency-model, poisson-glm, fremtpl2, insurance-monitoring, motor, model-validation]
description: "We fitted a Poisson GLM on the first third of freMTPL2 (677k French motor policies) and monitored it across two later temporal segments without refitting. PSI, A/E ratios with Wilson CIs, segmented age-band analysis, and GiniDriftTest — all from insurance-monitoring."
---

The monitoring problem in insurance pricing is not detecting drift. Any actuary with a spreadsheet and quarterly data can eyeball an A/E ratio. The problem is detecting drift *systematically*, across multiple dimensions, with appropriate statistical thresholds, before the model has been 7% cheap for eighteen months and someone asks why you did not notice sooner.

We benchmarked [`insurance-monitoring`](/insurance-monitoring/) against freMTPL2 — 677,991 French motor third-party liability policies from OpenML (dataset ID 41214) — to see what a structured monitoring workflow looks like at real portfolio scale.

---

## The scenario

We split freMTPL2 into three equal thirds by row order (row order correlates loosely with policy vintage via the IDpol identifier, following the approach of Noll, Salzmann & Wüthrich, 2020). Period 1 is the training window; periods 2 and 3 are monitoring windows. We fit a Poisson GLM on period 1 and then hold it fixed — no refit, no recalibration — while applying it to periods 2 and 3.

This mirrors production. Your model was fitted on last year's data. You are now applying it to this year's policies. The question is: what has changed, and has the model noticed?

```python
from insurance_monitoring import (
    psi,
    csi,
    ae_ratio,
    ae_ratio_ci,
    GiniDriftTest,
)
from insurance_monitoring.drift import ks_test, wasserstein_distance
from insurance_monitoring.thresholds import PSIThresholds
```

Note that `psi`, `csi`, `ae_ratio`, `ae_ratio_ci`, and `GiniDriftTest` are all available from the top-level package. `ks_test` and `wasserstein_distance` live in `insurance_monitoring.drift` — import them from there.

The dataset after cleaning: roughly 226,000 policies per segment, with features including driver age, vehicle age, vehicle power, BonusMalus score, and population density of the policyholder's commune.

---

## PSI: covariate drift before you look at claims

PSI (Population Stability Index) answers the question before any outcome data is needed: has the portfolio population changed? This is the first thing to check in any monitoring review, because covariate shift can cause miscalibration even if the underlying risk relationship has not changed.

```python
CONTINUOUS_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
thresholds = PSIThresholds()

for feat in CONTINUOUS_FEATURES:
    score = psi(
        df_p1[feat].to_numpy(),
        df_p2[feat].to_numpy(),
        n_bins=10,
        exposure_weights=df_p2["Exposure"].to_numpy(),
        reference_exposure=df_p1["Exposure"].to_numpy(),
    )
    band = thresholds.classify(score)
    print(f"{feat:<15} {score:.4f}  [{band}]")
```

The `exposure_weights` and `reference_exposure` parameters matter here. The standard PSI formula weights bins by policy count. Insurance policies have varying exposure — a one-month policy and an annual policy are not the same observation. Passing exposure arrays to both sides gives you a symmetrically exposure-weighted PSI, which is the correct version for an insurance book.

`PSIThresholds.classify()` returns `'green'` (PSI < 0.10), `'amber'` (0.10–0.25), or `'red'` (> 0.25).

The `csi()` function wraps `psi()` across all features at once and returns a Polars DataFrame — one row per feature with `feature`, `csi`, and `band` columns. That is what goes in the monitoring pack as the CSI heat map:

```python
csi_p2 = csi(df_p1.select(CONTINUOUS_FEATURES), df_p2.select(CONTINUOUS_FEATURES),
              features=CONTINUOUS_FEATURES)
```

On freMTPL2, BonusMalus is the most volatile feature across temporal thirds — exactly what you would expect in a French motor book, where bonus-malus scores shift as drivers accumulate or lose claim-free years over time. Density also moves: the geographic composition of the book changes as the insurer grows or shrinks in different regions.

### Why PSI is not enough on its own

PSI is heuristic. There is no p-value, and the standard thresholds (0.10, 0.25) come from 1990s credit scoring practice, not from any formal statistical derivation. It is good for operational dashboards because it is fast and consistent, but at the scale of freMTPL2 — 226k policies per segment — even economically trivial distribution shifts will produce measurable PSI.

For formal quarterly testing we supplement PSI with KS tests:

```python
for feat in CONTINUOUS_FEATURES:
    ks = ks_test(df_p1[feat].to_numpy(), df_p3[feat].to_numpy())
    wd = wasserstein_distance(df_p1[feat].to_numpy(), df_p3[feat].to_numpy())
    sig = "YES" if ks["significant"] else "no"
    print(f"{feat:<15}  KS={ks['statistic']:.4f}  p={ks['p_value']:.4f}  sig={sig}  "
          f"Wasserstein={wd:.2f}")
```

The `wasserstein_distance()` function returns a distance in the original feature units. "BonusMalus shifted by 3.2 points" is considerably more useful in a pricing committee meeting than "PSI = 0.14". At n~226k, the KS test will flag almost anything as significant; the Wasserstein distance tells you whether the shift is materially large.

---

## Fitting the Poisson GLM

The monitoring workflow requires a model. We use scikit-learn's `PoissonRegressor` — a log-linear Poisson GLM, the standard for insurance frequency — fitted only on period 1:

```python
from sklearn.linear_model import PoissonRegressor

model = PoissonRegressor(alpha=1e-4, max_iter=300)
model.fit(X_p1_norm, y_p1 / exp_p1, sample_weight=exp_p1)
```

The response variable is claim frequency (claims / exposure), with exposure as `sample_weight`. Features are z-score normalised using period 1 statistics; periods 2 and 3 are normalised with the same period 1 mean and standard deviation — the model knows nothing about later periods.

In-sample A/E on period 1 should be close to 1.0 by construction; it is worth checking because a large in-sample A/E would mean the GLM did not converge properly.

---

## A/E ratio monitoring

The A/E ratio is the universal actuarial calibration check. `ae_ratio()` computes the exposure-weighted ratio of actual to expected. `ae_ratio_ci()` adds Wilson-style confidence intervals:

```python
ae_p2 = ae_ratio(y_p2, rate_pred_p2, exposure=exp_p2)
ae_p3 = ae_ratio(y_p3, rate_pred_p3, exposure=exp_p3)

ci_p2 = ae_ratio_ci(y_p2, rate_pred_p2, exposure=exp_p2)
ci_p3 = ae_ratio_ci(y_p3, rate_pred_p3, exposure=exp_p3)
```

The confidence intervals matter because they tell you whether a departure from 1.0 is statistically distinguishable from noise at the exposure you have. A sustained A/E of 1.07 over 50,000 car-years is actionable; an A/E of 1.07 over 800 car-years is probably just sampling variation.

freMTPL2 is cross-sectional rather than a true panel, so the temporal split produces thirds that are fairly similar — you should not expect dramatic A/E drift from a simple row-order split on this dataset. The value of the exercise is in the infrastructure and the workflow: the same code running on a genuine multi-year UK motor book will find real drift.

### Aggregate A/E hides segment problems

The most important monitoring principle, and the one most often violated: never conclude your model is well-calibrated from the aggregate A/E alone. A book-level A/E of 1.00 is entirely consistent with the model being 15% cheap on drivers under 25 and 13% expensive on drivers over 60. The errors cancel in aggregate.

The `ae_ratio()` function accepts a `segments` array for exactly this reason:

```python
def age_band(age_arr):
    return np.where(age_arr < 25, "18-24",
           np.where(age_arr < 35, "25-34",
           np.where(age_arr < 45, "35-44",
           np.where(age_arr < 55, "45-54",
           np.where(age_arr < 65, "55-64", "65+")))))

bands = age_band(df_p3["DrivAge"].to_numpy())
segmented = ae_ratio(y_p3, mu_pred_p3 / exp_p3, exposure=exp_p3, segments=bands)
print(segmented.sort("segment"))
```

Run this for your top three or four rating factors. If any segment has an A/E outside the 0.90–1.10 range and a sample size where the confidence interval does not straddle 1.0, that is an actionable finding. One segment being 12% cheap while another is 11% expensive is not a rounding error — it is a model that has gone stale in a specific part of the rating structure.

---

## GiniDriftTest: has ranking power changed?

Aggregate calibration can be fine while ranking power has quietly deteriorated. A model can predict the right average frequency while losing its ability to distinguish high-risk from low-risk policyholders within any given segment. When that happens, your pricing is fair in aggregate but wrong at the individual level — you are subsidising the bad risks from the good risks.

`GiniDriftTest` implements the asymptotic z-test from Wüthrich, Merz & Noll (2025, arXiv:2510.04556). It tests H0: Gini(reference period) = Gini(monitoring period):

```python
gini_test_p3 = GiniDriftTest(
    reference_actual=y_p1,
    reference_predicted=mu_pred_p1,
    monitor_actual=y_p3,
    monitor_predicted=mu_pred_p3,
    reference_exposure=exp_p1,
    monitor_exposure=exp_p3,
    n_bootstrap=200,
    alpha=0.05,
    random_state=42,
)
result_p3 = gini_test_p3.test()
print(gini_test_p3.summary())
```

`GiniDriftTestResult` gives you `gini_reference`, `gini_monitor`, `delta` (monitor minus reference), `z_statistic`, `p_value`, and `significant`. The `summary()` method returns a governance-ready paragraph you can drop directly into a model validation report.

A few things worth knowing about this test:

**Alpha default is 0.32, not 0.05.** The arXiv paper recommends the one-sigma rule for routine monitoring — earlier detection at the cost of more false positives. For governance escalation or regulatory reporting, pass `alpha=0.05`. The benchmark notebook uses `alpha=0.05` explicitly.

**Power at scale.** At n~226k per segment, the bootstrap z-test has enough power to detect Gini changes of ±0.01 or less. For a smaller UK motor book — say 8,000 policies in a monitoring quarter — use `GiniDriftBootstrapTest` from `insurance_monitoring.discrimination` instead, which gives percentile confidence intervals rather than a z-test and is more appropriate when the normal approximation may not hold.

**The test requires raw data for both periods.** If you only have the stored training Gini scalar (which is the common case for models that have been in production for two years without archiving the training predictions), use `gini_drift_test_onesample()` from `insurance_monitoring.discrimination`.

---

## PSI and A/E catch different problems

This is the operational insight from the benchmark, and it is worth stating directly because it is regularly missed.

PSI detects covariate shift — the input distribution has changed — without any reference to claim outcomes. If 30% of your new business is now young drivers when it was 18% at training time, PSI catches that immediately. A/E does not: if the model's relativities for young drivers happen to still be roughly right, A/E will look fine even though your model is now extrapolating more heavily into a high-risk segment.

A/E detects output miscalibration — the model's predictions are systematically wrong relative to actual claims. But A/E can look fine even when PSI is red: if the portfolio has shifted towards older, safer drivers and the model underprices them slightly, the drift could be self-cancelling for a while.

You need both. A practical monitoring protocol runs PSI monthly as a leading indicator (no claim data needed, available immediately) and A/E quarterly as a lagging indicator (requires claim development). For the full three-layer framework — PSI, segmented A/E, and Gini drift testing — see [three-layer drift detection for deployed pricing models](/2026/03/03/your-pricing-model-is-drifting/). Gini monitoring runs quarterly with A/E. Any amber or red PSI finding triggers an investigation that includes segmented A/E, not a wait until Q-end.

---

## Putting it together

The full monitoring workflow in code is roughly forty lines across the three checks. The `MonitoringReport` class wraps all of this into a single call if you want traffic-light output without assembling the pieces yourself:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=y_p1,
    reference_predicted=rate_pred_p1,
    current_actual=y_p3,
    current_predicted=rate_pred_p3,
    exposure=exp_p3,
    reference_exposure=exp_p1,
)
print(report.recommendation)
print(report.to_polars())
```

For the full benchmark with segmented A/E and explicit PSI per feature, assemble the pieces as we have done above — `MonitoringReport` gives you the aggregate view; the individual functions give you the diagnostic depth.

The notebook is at [github.com/pricing-frontier/insurance-monitoring](https://github.com/pricing-frontier/insurance-monitoring) (`notebooks/benchmark_fremtpl2_drift.py`), Databricks-compatible. The dataset downloads automatically from OpenML if the cluster has network access; if not, there is a synthetic fallback that matches the freMTPL2 schema exactly and runs the same code.

```
pip install insurance-monitoring scikit-learn
```

No further dependencies. The library uses polars, numpy, and scipy internally.
