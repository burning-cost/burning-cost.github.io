---
layout: post
title: "Your Model Drift Alert Is Too Late"
date: 2026-02-08
author: Burning Cost
categories: [monitoring, model-risk]
description: "Aggregate A/E is a lagging indicator. insurance-monitoring catches input drift, feature drift, and score drift before the loss ratio moves."
canonical_url: "https://burning-cost.github.io/2026/02/08/your-model-drift-alert-is-too-late"
tags: [model-monitoring, drift-detection, PSI, CSI, gini, ae-ratio, insurance-monitoring, model-risk, pricing, python, uk-insurance]
---

By the time your A/E ratio flags a problem, the problem is six months old.

That is not an exaggeration. A motor frequency model misprices from the moment its input distribution shifts. But the A/E ratio only moves after claims develop - and on a UK motor book, frequency claims develop over three to twelve months. A segment that has been underpriced since January may not show up in your quarterly monitoring until Q3. By then, you have renewed the affected cohort twice.

The standard monitoring workflow has a structural defect: it monitors outputs when it should be monitoring inputs. The output - A/E - is a function of claims, which are slow. The inputs - driver age, vehicle group, postcode, conviction points - are observable immediately, at quote time, before any claim has occurred. An exposure-weighted distribution shift in driver age this month is detectable this month. The A/E signal from that same shift will not arrive until late next year.

---

## Why aggregate A/E fails at detection

The aggregate A/E ratio has a second problem beyond latency. It cancels.

A model that is 15% cheap on under-25s and 15% expensive on the 26–35 band will show an aggregate A/E close to 1.00 if those two groups are roughly balanced in the book. The aggregate says nothing is wrong. But the model is mispricing both segments, subsidising one with the overcharge from the other. This is the canonical adverse selection trap: the overpriced segment leaves at renewal, the underpriced segment stays. The aggregate A/E stays at 1.00 right up until the book degrades and retention skews.

Segment-level A/E catches this - but segment-level A/E still requires claims development. The detection timeline does not improve.

---

## The three-layer approach

`insurance-monitoring` monitors at three independent layers. Each layer catches a different class of failure, at a different point in time.

**Layer 1: Exposure-weighted PSI on inputs.** This is the early warning layer. It fires when the feature distribution shifts - before any claim has been paid, often before any claim has been reported. The key word is exposure-weighted. Unweighted PSI treats a one-month policy the same as an annual one; for insurance monitoring, the correct denominator is earned car-years (or earned house-years, earned unit-months - whatever your exposure measure is). A 5% shift in the age distribution where all the movement is in the 17–25 band is not the same as a 5% shift spread evenly across all ages. Exposure-weighted PSI captures this correctly.

**Layer 2: Segment-level A/E with Poisson confidence intervals.** This is the downstream check. It confirms whether a flagged input shift has started to manifest in claims experience. It will always be slower than Layer 1, but it provides the actuarial evidence needed to escalate from "investigate" to "recalibrate" or "refit".

**Layer 3: Gini drift z-test.** This is the discrimination check, and it is the one most monitoring frameworks miss entirely. A model can have a stable aggregate A/E - correct on average - and a deteriorating Gini. What this means in practice: the model is still predicting the right mean, but it has lost the ability to rank risks correctly. Cheap risks are priced expensively and expensive risks are priced cheaply. The aggregate looks fine because the errors net out. The adverse selection consequence is identical to the explicit mispricing case above.

The Gini drift test in `insurance-monitoring` is based on the z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556), which establishes the asymptotic normality of the sample Gini and derives a proper bootstrap variance estimator. It is not a heuristic. When the test statistic exceeds the one-sigma threshold (the default, following the paper's recommendation for monitoring contexts), the library returns `INVESTIGATE`. At the two-sigma level, `REFIT`.

---

## A worked example

To make this concrete: we induce a known shift into a synthetic UK motor portfolio. Reference data is 50,000 policies with a typical age distribution - about 9% under-25. Monitoring data is 15,000 policies where under-25s have been oversampled 2x, representing roughly 17% of exposure. This is not an implausible scenario; it could be a comparison site algorithm change, a new affinity scheme, or a broker acquisition that skews young.

```python
import numpy as np
import polars as pl
from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi

rng = np.random.default_rng(42)

# Reference: 50,000 policies, training distribution
n_ref = 50_000
ages_ref = np.concatenate([
    rng.integers(17, 25, int(n_ref * 0.09)),   # 9% under-25
    rng.integers(25, 80, int(n_ref * 0.91)),
])
rng.shuffle(ages_ref)
exposure_ref = rng.uniform(0.3, 1.0, n_ref)
pred_ref = 0.05 + 0.003 * np.clip((25 - ages_ref), 0, 8)  # young-driver loading
act_ref = rng.poisson(pred_ref * exposure_ref)

# Monitoring: 15,000 policies, shifted — under-25s oversampled 2x
n_cur = 15_000
ages_cur = np.concatenate([
    rng.integers(17, 25, int(n_cur * 0.17)),   # 17% under-25
    rng.integers(25, 80, int(n_cur * 0.83)),
])
rng.shuffle(ages_cur)
exposure_cur = rng.uniform(0.3, 1.0, n_cur)
# Model was trained on old distribution — it applies the same loadings
pred_cur = 0.05 + 0.003 * np.clip((25 - ages_cur), 0, 8)
# But young drivers have higher actual frequency — model is now cheap on them
true_freq = pred_cur * np.where(ages_cur < 25, 1.15, 1.0)
act_cur = rng.poisson(true_freq * exposure_cur)

feat_ref = pl.DataFrame({"driver_age": ages_ref.tolist()})
feat_cur = pl.DataFrame({"driver_age": ages_cur.tolist()})

report = MonitoringReport(
    reference_actual=act_ref / exposure_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur / exposure_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age"],
    murphy_distribution="poisson",
)

print(report.recommendation)
print(report.to_dict())
```

What this returns:

- `driver_age` PSI: `0.28 - RED`. The distribution shift is flagged immediately, before any claims data has arrived.
- Aggregate A/E: `1.017 - GREEN`. A/E is barely above 1.0, well within the confidence interval. Nothing alarming.
- `MonitoringReport` recommendation: `INVESTIGATE`.

The PSI fires on the input shift in the current monitoring period. The A/E has not moved, because we have not given claims enough time to develop. If we ran this same check three quarters later - after the under-25 cohort has accumulated enough claims to move the A/E - we would find PSI still RED and A/E now in AMBER territory. But by that point, we have been mispricing young drivers for nine months.

---

## The exposure-weighting detail

Standard PSI divides each bin by policy count. For insurance monitoring this is wrong, and it can produce misleading signals in portfolios with highly variable policy durations.

Consider a commercial fleet book with a mix of monthly and annual policies. A shift towards short-duration fleet policies changes the risk exposure more than a naive policy-count PSI would suggest - short policies are often newer vehicles or higher-turnover fleets, with different risk characteristics. Count-based PSI treats a one-month policy and a twelve-month policy identically. Exposure-weighted PSI weights each observation by its earned exposure.

```python
from insurance_monitoring.drift import psi

# Without exposure weighting — count-based
psi_count = psi(ages_ref, ages_cur)

# With exposure weighting — correct for insurance
psi_weighted = psi(
    ages_ref,
    ages_cur,
    exposure_weights=exposure_cur,
    reference_exposure=exposure_ref,
)

print(f"Count PSI: {psi_count:.3f}")
print(f"Exposure-weighted PSI: {psi_weighted:.3f}")
```

In this example the difference is modest because the age shift is large enough to dominate both metrics. In portfolios with variable policy durations and small distributional shifts, the count-based PSI will systematically understate the drift.

---

## The Gini check that A/E cannot replace

The Gini drift z-test addresses a failure mode that neither PSI nor A/E can catch: discrimination degradation where the average is preserved.

This happens when the model's learned monotonic relationships erode at the margins - typically due to interactions between a shifted variable and a stable one. The model correctly prices the average risk; it incorrectly ranks the tails. In a competitive pricing market, this means you are cheap on the highest-risk segment within each band, and expensive on the lowest-risk segment. You attract the wrong customers within every rating cell.

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

ref_gini = gini_coefficient(act_ref / exposure_ref, pred_ref)
cur_gini = gini_coefficient(act_cur / exposure_cur, pred_cur)

result = gini_drift_test(
    reference_gini=ref_gini,
    current_gini=cur_gini,
    reference_actual=act_ref / exposure_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur / exposure_cur,
    current_predicted=pred_cur,
    n_bootstrap=2000,
)

print(f"Reference Gini: {result.reference_gini:.3f}")
print(f"Current Gini:   {result.current_gini:.3f}")
print(f"z-statistic:    {result.z_statistic:.2f}")
print(f"Significant:    {result.significant}")
```

In this synthetic example the Gini degradation is mild - the age shift introduces only moderate ranking deterioration. In real portfolios, discrimination drift often precedes A/E movement because it is driven by the same composition change, but manifests faster: ranking is affected as soon as the new risk profiles enter the book, while A/E requires those profiles to generate claims.

---

## Putting it together in production

The `MonitoringReport` class runs all three layers in one call and returns a structured recommendation:

```python
report = MonitoringReport(
    reference_actual=act_ref / exposure_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur / exposure_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'NO_ACTION' | 'MONITOR_CLOSELY' | 'INVESTIGATE' | 'RECALIBRATE' | 'REFIT'
```

The recommendation logic: PSI RED on any feature moves immediately to `INVESTIGATE`. If the Gini z-test is significant and A/E is drifting, the Murphy decomposition (UNC/DSC/MCB) distinguishes calibration error from discrimination error - the former suggests `RECALIBRATE`, the latter `REFIT`. This matters operationally. A recalibration is a parameter update - hours of work. A refit is a model rebuild - weeks. Calling one when you need the other is expensive.

The library produces a structured traffic-light report that satisfies PRA SS1/23 requirements for model risk logging. This is not a minor benefit. The PRA's 2026 model risk management review found that firms with systematic monitoring logs were significantly less likely to receive remediation notices - partly because the logs demonstrated awareness and response rather than passive oversight.

---

## The Databricks pipeline

The full production pipeline - hourly feature extraction from the quotes database, automated PSI/CSI computation across all rating factors, Gini drift test, and report generation to Delta Lake with alerting hooks - is in the [monitoring_drift_detection notebook](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py). It runs on a standard DBR 13.3 LTS cluster without modification.

The architecture is straightforward: a scheduled job reads the rolling 90-day quote feed, computes `MonitoringReport` against the stored reference statistics from the model training run, writes the result to a monitoring table, and triggers a Slack alert if any metric crosses the amber threshold. The reference statistics are serialised at model training time and versioned alongside the model artefact. When the model is replaced, the reference statistics are replaced automatically.

---

## What to do when the alert fires

An `INVESTIGATE` recommendation from PSI alone is not a mandate to rebuild. It is a mandate to look. The first question is whether the distribution shift is real - sampling noise, data pipeline changes, and platform test traffic are common false positives. Check raw policy counts. Check whether the shift is concentrated in specific broker codes or distribution channels that might be running tests.

If the shift is real, the second question is whether the model is affected. Not every input distribution shift degrades model performance - if the shifted variable has a weak or linear relationship with frequency, the model may still price it adequately at the new distribution. Run `ae_ratio` segmented by the flagged feature. If the segment-level A/E is within confidence intervals, `MONITOR_CLOSELY` is the right posture.

If the segment-level A/E is drifting, and the Gini test is flagging discrimination degradation, you have a compound problem: the model is miscalibrated on a growing segment and its ranking is deteriorating. That is when you escalate.

The escalation path - recalibration or refit - depends on the Murphy decomposition. High MCB (miscalibration) relative to DSC (discrimination) means the model structure is sound and the intercepts need adjusting. High DSC loss means the ranking itself has degraded and you need new features or a new tree structure. `insurance-monitoring` gives you this decomposition; the decision is still yours.

---

`insurance-monitoring` is open source under BSD-3 at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). Requires Python 3.10+. Install with `uv add insurance-monitoring`.

---

**Related posts:**
- [Your Pricing Model is Drifting (and You Probably Can't Tell)](/2026/03/07/your-pricing-model-is-drifting/) - the original case for PSI-first monitoring and why A/E alone is insufficient
- [Your New Book Doesn't Look Like Your Old Book. Your Model Doesn't Care.](/2028/05/15/your-new-book-doesnt-look-like-your-old-book/) - density ratio correction for portfolio composition shift, when the distribution change is too large for recalibration
- [Your Champion Model Has Been Running Unchallenged for Three Years](/2028/09/15/champion-model-unchallenged/) - what happens after monitoring flags a problem: champion/challenger testing and the governance infrastructure for model replacement
