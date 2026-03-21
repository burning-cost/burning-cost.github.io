---
layout: post
title: "Insurance Model Monitoring Beyond Generic Data Drift"
date: 2029-08-30
author: Burning Cost
categories: [monitoring, python, libraries, tutorials]
description: "Generic drift tools like Evidently and NannyML do not understand exposure weighting, development lags, or the difference between recalibrate and refit. Here is insurance model monitoring Python done correctly."
canonical_url: "https://burning-cost.github.io/2029/08/30/insurance-model-monitoring-beyond-generic-drift/"
tags: [model-monitoring, drift-detection, psi, ae-ratio, gini, sequential-testing, insurance-monitoring, polars, champion-challenger, insurance-model-monitoring-python, uk-motor, model-governance]
---

Your pricing model's aggregate A/E ratio is 1.02. Your model has been mispricing under-25s for seven months.

This is the central failure mode of generic model monitoring: errors cancel at portfolio level. The model is 14% cheap on young drivers and 12% expensive on mature drivers. The aggregate reads fine. Nobody raises an alarm. You find out about the problem when the loss ratio deteriorates — typically 12 to 18 months after the model started misfiring.

We have evaluated [Evidently](https://github.com/evidentlyai/evidently) and [NannyML](https://github.com/NannyML/nannyml) against real insurance use cases. Both are well-built tools for generic ML monitoring. Neither can:

- Weight PSI bin proportions by earned exposure (car-years, not policy count)
- Compute A/E ratios with correct Poisson confidence intervals, segmented by rating band
- Distinguish between a model that needs recalibration and one that needs a full refit — using the Murphy decomposition
- Run a champion/challenger experiment with monthly result checks without inflating type I error

These are not edge cases. They are the core monitoring tasks for a UK motor pricing actuary. The [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) library handles all of them.

```bash
uv add insurance-monitoring
```

Polars-native throughout. No scikit-learn dependency.

---

## 1. Exposure-weighted PSI

Standard PSI treats every policy equally regardless of how long it was on risk. On a book that mixes 1-month direct debit customers and 12-month annual payers, unweighted PSI is wrong — the distribution of short-period policies looks like covariate shift even when the underlying risk mix is stable.

The `psi` function in `insurance_monitoring.drift` takes an `exposure_weights` parameter. Bin proportions are computed as a share of total earned exposure, not policy count.

```python
import numpy as np
import polars as pl
from insurance_monitoring.drift import psi, csi

rng = np.random.default_rng(42)

n_ref = 50_000
n_cur = 15_000

# Reference period: training window (2021–2023)
driver_age_ref = rng.integers(18, 80, n_ref)
exposure_ref   = rng.uniform(0.1, 1.0, n_ref)   # car-years, not policies

# Monitoring period: 18 months into deployment
# Young drivers (under-25) have been oversampled 2x through a new aggregator channel
driver_age_cur = np.where(
    rng.random(n_cur) < 0.35,
    rng.integers(18, 25, n_cur),   # 35% young drivers vs 10% in reference
    rng.integers(25, 85, n_cur),
)
exposure_cur = rng.uniform(0.1, 1.0, n_cur)

# Unweighted PSI — treats a 1-week policy the same as a 12-month policy
score_unweighted = psi(
    reference=driver_age_ref,
    current=driver_age_cur,
    n_bins=10,
)

# Exposure-weighted PSI — correct for insurance
score_weighted = psi(
    reference=driver_age_ref,
    current=driver_age_cur,
    n_bins=10,
    exposure_weights=exposure_ref,   # weight reference bins by earned exposure
)

print(f"Unweighted PSI: {score_unweighted:.3f}")   # ~0.31 — RED
print(f"Weighted PSI:   {score_weighted:.3f}")     # ~0.28 — RED, same direction; magnitude differs on mixed-duration books
```

The difference between weighted and unweighted matters most on books with genuine exposure variation. On an annual book with uniform exposure it barely moves. On a mixed-duration direct writer, the unweighted figure overstates drift from short-period new business and understates drift in segments that happen to have longer durations.

For multi-feature monitoring, `csi` computes PSI across all rating factors in one call:

```python
feat_ref = pl.DataFrame({
    "driver_age":       driver_age_ref.tolist(),
    "vehicle_age":      rng.integers(0, 15, n_ref).tolist(),
    "ncd_years":        rng.integers(0, 9, n_ref).tolist(),
    "conviction_pts":   rng.integers(0, 5, n_ref).tolist(),
})
feat_cur = pl.DataFrame({
    "driver_age":       driver_age_cur.tolist(),
    "vehicle_age":      rng.integers(0, 15, n_cur).tolist(),
    "ncd_years":        rng.integers(0, 9, n_cur).tolist(),
    "conviction_pts":   (rng.integers(0, 5, n_cur) + rng.integers(0, 2, n_cur)).tolist(),
})

csi_table = csi(
    feat_ref, feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years", "conviction_pts"],
)
print(csi_table)
# ┌────────────────┬───────┬───────┐
# │ feature        ┆ csi   ┆ band  │
# ╞════════════════╪═══════╪═══════╡
# │ driver_age     ┆ 0.28  ┆ red   │
# │ conviction_pts ┆ 0.12  ┆ amber │
# │ vehicle_age    ┆ 0.02  ┆ green │
# │ ncd_years      ┆ 0.01  ┆ green │
# └────────────────┴───────┴───────┘
```

Two features in the green band means nothing has changed there. Driver age is red — the aggregator channel hypothesis confirmed. Conviction points in amber is worth investigating separately.

---

## 2. Segmented A/E monitoring with Poisson confidence intervals

The aggregate A/E is a necessary check, not a sufficient one. The function `ae_ratio` returns a Polars DataFrame with one row per segment, including exact Garwood confidence intervals (not normal approximations — Garwood intervals are materially better at small claim counts, which is where you most need reliable CIs).

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci
import numpy as np

rng = np.random.default_rng(42)
n = 20_000

predicted = rng.uniform(0.05, 0.20, n)
exposure  = rng.uniform(0.3, 1.0, n)
actual    = rng.poisson(predicted * exposure).astype(float)

# Simulate the under-25 mispricing: actuals are 18% above predictions for that segment
driver_age = rng.integers(18, 80, n)
young_mask = driver_age < 25
actual[young_mask] = rng.poisson(predicted[young_mask] * exposure[young_mask] * 1.18).astype(float)

# Aggregate A/E with Poisson CI
agg = ae_ratio_ci(actual, predicted, exposure=exposure)
print(agg)
# {'ae': 1.03, 'lower': 1.01, 'upper': 1.06, 'n_claims': 2847, 'n_expected': 2765}
# Aggregate looks fine. Now segment it.

# Age bands as a segment vector
age_bands = np.where(driver_age < 25, "under-25",
            np.where(driver_age < 40, "25-39",
            np.where(driver_age < 60, "40-59", "60+")))

seg = ae_ratio(actual, predicted, exposure=exposure, segments=age_bands)
print(seg)
# ┌──────────┬────────┬──────────┬──────────┬────────────┐
# │ segment  ┆ actual ┆ expected ┆ ae_ratio ┆ n_policies │
# ╞══════════╪════════╪══════════╪══════════╪════════════╡
# │ under-25 ┆ 412    ┆ 349      ┆ 1.181    ┆ 3124       │
# │ 25-39    ┆ 891    ┆ 875      ┆ 1.018    ┆ 6342       │
# │ 40-59    ┆ 1102   ┆ 1097     ┆ 1.005    ┆ 7215       │
# │ 60+      ┆ 442    ┆ 444      ┆ 0.995    ┆ 3319       │
# └──────────┴────────┴──────────┴──────────┴────────────┘
```

The aggregate A/E of 1.03 conceals the under-25 segment sitting at 1.18. That is 18 months of compounding underpricing hiding inside a number that looks acceptable on a board dashboard.

For model sign-off or a deeper diagnostic, `CalibrationChecker` combines the balance property test, per-cohort auto-calibration, and the Murphy decomposition:

```python
from insurance_monitoring.calibration import CalibrationChecker

checker = CalibrationChecker(distribution='poisson', alpha=0.05)
report  = checker.check(actual, predicted, exposure)

print(report.verdict())    # 'OK' | 'RECALIBRATE' | 'REFIT'
print(report.summary())    # paragraph-form diagnostic

# Murphy decomposition: the RECALIBRATE vs REFIT decision
murphy = report.murphy
print(f"Global MCB (scale): {murphy.global_mcb:.4f}")
print(f"Local MCB (shape):  {murphy.local_mcb:.4f}")
print(f"Murphy verdict:     {murphy.verdict}")
```

The Murphy decomposition is the diagnostic layer that generic tools cannot provide. Global MCB (GMCB) captures miscalibration that is fixed by multiplying all predictions by the aggregate A/E ratio — a recalibration job that takes an afternoon. Local MCB (LMCB) captures miscalibration that requires the model to be rebuilt because the relative predictions are wrong, not just the level. When LMCB dominates, recalibration will not fix the problem; you are looking at a refit.

---

## 3. Sequential champion/challenger testing

The dirty secret of most champion/challenger experiments is that the teams running them check results monthly and stop early when the data looks interesting. A standard two-sample test nominally at 5% FPR inflates to around 25% FPR under monthly peeking over a 24-month window. One in four "significant" results at that cadence is noise.

The `SequentialTest` implements the mixture SPRT (mSPRT) from Johari et al. (2022). The test statistic is an e-process: `P(∃n: Λn ≥ 1/α) ≤ α` at all stopping times. Check it every month, check it every week — the FPR stays at 5%.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

# A new telematics scoring model goes into champion/challenger
# from January 2029. We check monthly as renewal data arrives.
test = SequentialTest(
    metric="frequency",        # Poisson rate ratio
    alternative="two_sided",
    alpha=0.05,
    tau=0.05,                  # prior std dev on log-rate-ratio: telematics effects ~5%
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,  # car-years before any stopping decision
)

# Month 1 — January 2029 renewals
r1 = test.update(
    champion_claims=47,  challenger_claims=41,
    champion_exposure=520, challenger_exposure=515,
    calendar_date=datetime.date(2029, 1, 31),
)
print(r1.summary)
# "Challenger freq 9.1% lower (95% CS: 0.714–1.187). Evidence: 0.6 (threshold 20.0). Inconclusive."

# Month 5 — May 2029. Still checking. FPR unchanged.
r5 = test.update(
    champion_claims=49,  challenger_claims=31,
    champion_exposure=525, challenger_exposure=520,
    calendar_date=datetime.date(2029, 5, 31),
)
print(r5.decision)      # 'reject_H0' if evidence crosses threshold
print(r5.should_stop)   # True means stop and act

# Full history for governance reporting
history = test.history()
# period_index | calendar_date | lambda_value | log_lambda_value | champion_rate | ...
```

When the test returns `should_stop = True`, the evidence is genuine — not an artefact of looking too early. The decision is `reject_H0` (challenger is better or worse) or `futility` (effect is too small to matter after enough data). Both are valid stopping reasons.

For batch workflows from a DataFrame of monthly reporting extracts:

```python
import polars as pl
from insurance_monitoring.sequential import sequential_test_from_df

monthly = pl.DataFrame({
    "date":          ["2029-01-31", "2029-02-28", "2029-03-31", "2029-04-30", "2029-05-31"],
    "champ_claims":  [47, 44, 51, 48, 49],
    "champ_exp":     [520, 515, 530, 518, 525],
    "chall_claims":  [41, 36, 38, 33, 31],
    "chall_exp":     [515, 510, 525, 512, 520],
})

result = sequential_test_from_df(
    df=monthly,
    champion_claims_col="champ_claims",
    champion_exposure_col="champ_exp",
    challenger_claims_col="chall_claims",
    challenger_exposure_col="chall_exp",
    date_col="date",
    metric="frequency",
    alpha=0.05,
)
print(result.summary)
```

The `tau` prior matters. For telematics experiments where you expect effects of 10%+ on the log-rate scale, use `tau=0.10`. For fine-tuning a well-calibrated model where the challenger differs only in a couple of interaction terms, use `tau=0.02`. The prior encodes your expectation of effect size and controls statistical power.

---

## The combined report

For monthly governance reporting, `MonitoringReport` runs PSI, CSI, A/E, Gini drift, and Murphy decomposition in one call and outputs a traffic-light summary with a recommended action.

```python
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
    features=["driver_age", "vehicle_age", "ncd_years", "conviction_pts"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'REFIT' | 'RECALIBRATE' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

print(report.to_polars())
# metric                | value  | band
# ae_ratio              | 1.08   | amber
# gini_current          | 0.39   | amber
# gini_p_value          | 0.054  | amber
# csi_driver_age        | 0.28   | red
# csi_conviction_pts    | 0.12   | amber
# murphy_discrimination | 0.041  | —
# murphy_miscalibration | 0.003  | —
# recommendation        | nan    | RECALIBRATE
```

The recommendation logic follows the three-stage decision tree from arXiv 2510.04556: if A/E is red but the Gini is stable, the ranking has not degraded — recalibrate the intercept. If the Gini is red, the ranking is broken — refit the model. If both are red, investigate data quality first before either action.

If you want tighter thresholds for a large book with monthly monitoring (the industry defaults of PSI 0.10/0.25 were designed for quarterly or annual review cycles):

```python
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds

custom = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),
)
report = MonitoringReport(..., thresholds=custom)
```

---

## What the benchmark shows

On 50,000 reference policies and 15,000 monitoring-period policies with three deliberately induced failure modes — covariate shift (young drivers 2x oversampled), calibration drift (8% frequency uplift), and conviction point distribution shift — the manual aggregate A/E check detects none of the three. `MonitoringReport` catches all of them.

The Gini drift test at n=15,000 returns z≈−1.9, p≈0.06. The library uses α=0.32 as the default monitoring threshold (one-sigma rule), per the arXiv 2510.04556 recommendation: at monthly cadence, the cost of missing a real drift signal is much higher than the cost of a false alarm. The α=0.05 threshold is appropriate for model validation reports; it is too conservative for operational monitoring where you want early signals.

The sequential test benchmark (10,000 Monte Carlo simulations under H0 with monthly peeking) shows a fixed-horizon t-test inflating to ~25% FPR against the mSPRT's actual FPR of ~1% at nominal 5%. That gap is not theoretical. It is what happens in every monthly governance meeting where someone says "it's been significant for three months, that's good enough."

---

The library is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). The worked example at [`model_drift_monitoring.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/model_drift_monitoring.py) shows the full stack on synthetic UK motor data with all three failure modes induced.

---

**Related posts:**
- [Your Pricing Model is Drifting (and You Probably Can't Tell)](/2026/03/07/your-pricing-model-is-drifting/) — why PSI alone is insufficient, and what it means when A/E is stable but the Gini is falling
- [Why k-Fold CV is Wrong for Insurance and What to Do Instead](/2029/08/15/why-k-fold-cv-is-wrong-for-insurance/) — the split strategy that produces metrics worth monitoring against
- [Tweedie Regression for Insurance: What sklearn Doesn't Tell You](/2029/07/30/tweedie-regression-insurance-what-sklearn-doesnt-tell-you/) — exposure handling in the model that monitoring is watching
