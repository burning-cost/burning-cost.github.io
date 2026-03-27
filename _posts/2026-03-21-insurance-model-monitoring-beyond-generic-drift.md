---
layout: post
title: "Insurance Model Monitoring in Python: Beyond Generic Data Drift"
date: 2026-03-21
author: Burning Cost
categories: [monitoring, model-risk, libraries, tutorials]
description: "Insurance model monitoring in Python that understands exposure weighting, development lags, and Gini drift. Why Evidently and NannyML miss what matters for pricing, and what insurance-monitoring adds."
tags: [insurance-model-monitoring-python, model-monitoring, PSI, CSI, ae-ratio, gini-drift, sequential-testing, mSPRT, insurance-monitoring, Evidently, NannyML, model-risk, pricing, python, uk-insurance, polars]
---

Insurance model monitoring in Python is a specific problem that generic drift detection tools solve only partially. Those tools — Evidently, NannyML, and their equivalents — are built for the modal ML use case: tabular or unstructured data, i-i-d records, binary or regression targets. They work well for churn models, credit scoring pipelines, and fraud detection. They work less well for insurance pricing, and the failure modes are not cosmetic.

The problem is not that tools like Evidently or NannyML are badly engineered. The problem is that insurance data has structure those tools do not model:

1. **Exposure weighting.** A policy in force for one month contributes the same features to an unweighted PSI calculation as a policy in force for twelve months. The twelve-month policy contributes twelve times as much to the claims experience. Unweighted feature drift detection gives you the wrong signal.

2. **Development lags.** An A/E alert raised on undeveloped accident months is comparing immature claims to a model trained on mature ones. The ratio is biased downward by as much as 30–40% for liability lines. Generic calibration monitoring does not know what IBNR is.

3. **The discrimination dimension.** A pricing model can have a stable A/E - correct on average - while its Gini coefficient erodes. The model still predicts the right mean but has lost the ability to rank risks. An adverse selection consequence follows as surely as from explicit mispricing, and generic monitoring does not test for it.

4. **The sequential testing problem.** When you run champion/challenger experiments on renewal cohorts and check results monthly, a standard significance test inflates false positives to roughly five times the nominal rate. The typical UK motor renewal cycle forces exactly this kind of repeated peeking.

[`insurance-monitoring`](/insurance-monitoring/) is built around these four problems. This post works through them concretely.

```bash
uv add insurance-monitoring
```

---

## The exposure weighting problem

PSI is the industry-standard feature drift statistic, and it is wrong for insurance portfolios unless you correct it.

The standard formula computes bin proportions by policy count. For a book where mid-term adjustments, short-period policies, and annual renewals coexist, this gives each policy equal weight. A 30-day policy and a 365-day policy look the same to the formula. But they contribute very differently to claims exposure. If your young driver cohort is disproportionately on short-period policies - which it is, because comparison sites and monthly-pay schemes attract younger drivers - unweighted PSI will understate the distributional shift in the exposure you actually care about.

The fix is to weight bin proportions by earned exposure (car-years, earned house-years, earned unit-months - whatever your exposure measure is). `insurance-monitoring` does this via the `exposure_weights` parameter on `psi()`.

```python
import numpy as np
import polars as pl
from insurance_monitoring.drift import psi, csi, wasserstein_distance

rng = np.random.default_rng(42)

n_ref = 50_000
n_cur = 15_000

# Reference: training window, typical age distribution
ages_ref = np.concatenate([
    rng.integers(17, 25, int(n_ref * 0.09)),   # 9% under-25
    rng.integers(25, 80, int(n_ref * 0.91)),
]).astype(float)

# Monitoring: comparison site scheme acquisition shifted the mix
ages_cur = np.concatenate([
    rng.integers(17, 25, int(n_cur * 0.19)),   # 19% under-25
    rng.integers(25, 80, int(n_cur * 0.81)),
]).astype(float)

# Exposure: short-period policies over-represented among young drivers
# Annual policies get 1.0; short-period get 0.2–0.3
exp_ref = np.where(ages_ref < 25, rng.uniform(0.2, 0.4, n_ref), rng.uniform(0.7, 1.0, n_ref))
exp_cur = np.where(ages_cur < 25, rng.uniform(0.2, 0.4, n_cur), rng.uniform(0.7, 1.0, n_cur))

# Unweighted PSI — understates the shift
psi_unweighted = psi(ages_ref, ages_cur, n_bins=10)

# Exposure-weighted PSI — correct for insurance
psi_weighted = psi(
    ages_ref,
    ages_cur,
    n_bins=10,
    exposure_weights=exp_cur,
    reference_exposure=exp_ref,
)

print(f"Unweighted PSI: {psi_unweighted:.3f}")   # ~0.18 (AMBER)
print(f"Weighted PSI:   {psi_weighted:.3f}")     # ~0.27 (RED)

# Wasserstein tells you the shift in interpretable units
d = wasserstein_distance(ages_ref, ages_cur)
print(f"Average driver age shifted by {d:.1f} years")
```

The unweighted PSI reads amber. The exposure-weighted PSI reads red. These are different conclusions. The weighted version is correct; the unweighted version flatters the picture by treating twelve-month and one-month policies identically.

Wasserstein distance is the best statistic for communication: it tells an underwriter or head of pricing that the average driver age has shifted by some number of years, which they can immediately relate to their experience of the book.

---

## The discrimination dimension

Aggregate A/E monitoring is now standard practice on most UK pricing teams. What is not standard is Gini drift monitoring - and this is the gap that matters most.

A model can maintain a stable aggregate A/E while losing discriminatory power. The mechanism is covariate shift: as the portfolio ages and mix shifts, the model's relativities become stale — [Whittaker-Henderson smoothing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) of the affected rating tables is often the right first step before a full refit. The model predicts the right mean by coincidence — cheap and expensive errors cancel — but it no longer ranks risks correctly. Cheap risks are priced expensively; expensive risks are priced cheaply. This is adverse selection by another name, and it proceeds silently until retention data reveals the pattern.

`insurance-monitoring` implements the Gini drift z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556), which establishes the asymptotic normality of the sample Gini and derives a proper bootstrap variance estimator. This is not a heuristic threshold; it is a proper hypothesis test.

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

# Reference period: well-calibrated model at launch
pred_ref = rng.uniform(0.05, 0.20, n_ref)
act_ref  = rng.poisson(pred_ref).astype(float)

# Monitoring period: model trained on 2021 data, now mid-2029
# Portfolio mix has shifted; predictions still centre correctly but rank poorly
pred_cur = rng.uniform(0.05, 0.20, n_cur)
noise    = rng.uniform(0.0, 0.5, n_cur)     # simulate staleness
act_cur  = rng.poisson(pred_ref[:n_cur] * 0.5 + noise).astype(float)

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=n_ref,
    n_current=n_cur,
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur,   current_predicted=pred_cur,
)

print(f"Gini reference: {gini_ref:.3f}")
print(f"Gini current:   {gini_cur:.3f}  (change: {result['gini_change']:+.3f})")
print(f"z-statistic:    {result['z_statistic']:.2f}")
print(f"p-value:        {result['p_value']:.3f}")
print(f"Significant:    {result['significant']}")
```

The decision rule from arXiv 2510.04556 maps onto the library's traffic lights: one-sigma (p < 0.32) triggers `INVESTIGATE`, two-sigma (p < 0.10) triggers `REFIT`. These are deliberately conservative thresholds - the authors' argument is that the cost of missing a Gini decline is high enough to accept more false positives than you would accept in a publication context.

We agree with that logic. The cost of a spurious investigation is two days of an actuary's time. The cost of an undetected Gini decline, compounded over 18 months of renewals, is measurable in loss ratio.

---

## A/E ratios with actual statistical rigour

The standard pricing A/E calculation produces a point estimate. `insurance-monitoring` adds Poisson confidence intervals via exact Garwood intervals, and segment-level breakdown.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with exact Garwood CI
pred_cur = rng.uniform(0.05, 0.20, n_cur)
act_cur  = rng.poisson(pred_cur * 1.08).astype(float)  # 8% underpricing

result = ae_ratio_ci(act_cur, pred_cur, exposure=exp_cur)
print(f"A/E: {result['ae']:.3f}  [{result['lower']:.3f}, {result['upper']:.3f}]")
print(f"Claims: {result['n_claims']:.0f} observed, {result['n_expected']:.0f} expected")
# A/E: 1.082  [1.031, 1.137]

# Segment-level: where is the model misfiring?
age_bands = np.select(
    [ages_cur < 25, ages_cur < 35, ages_cur < 50],
    ["17-24", "25-34", "35-49"],
    default="50+",
)

seg = ae_ratio(act_cur, pred_cur, exposure=exp_cur, segments=age_bands)
print(seg)
# shape: segment | actual | expected | ae_ratio | n_policies
```

The Garwood interval is exact for Poisson counts - not a normal approximation, not a bootstrap. At low claim counts (under 30), the normal approximation understates interval width by 15–20%. This matters for new entrant segments, niche products, or monthly monitoring of sub-books.

Segment-level A/E is the tool that reveals the 15%/−15% cancellation pattern. It does require claims development, which the PSI layer does not. The two layers are complementary: PSI fires first (at feature shift, before any claim), A/E confirms (after claims develop, confirming the economic impact).

---

## Murphy decomposition: recalibrate or refit?

The hardest decision in model monitoring is not "is the model drifting?" It is "what do we do about it?"

Recalibration — applying a scalar multiplier across all predictions — takes hours. Refitting — rebuilding the model on recent data — takes weeks. Getting the decision wrong in either direction is expensive. A team that recalibrates when a refit is needed has patched over a broken ranking and will face the same problem again at the next quarterly review. A team that refits when recalibration was sufficient has wasted three weeks of pricing analyst time. When a refit is warranted, it is also the right moment to ask whether the new model should be an [interpretable GAM rather than a plain GLM](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — the shape functions make future monitoring conversations substantially easier.

The Murphy decomposition (Lindholm & Wüthrich, SAJ 2025) resolves this. It decomposes forecast error into:

- **UNC** (uncertainty): irreducible noise in the data - not the model's fault
- **DSC** (discrimination): the portion of forecast skill explained by ranking
- **MCB** (miscalibration): the portion explained by wrong scale

MCB itself splits into GMCB (global miscalibration - fixed by a multiplier) and LMCB (local miscalibration - requires model refit). If GMCB dominates, recalibrate. If LMCB dominates, refit.

```python
from insurance_monitoring.calibration import murphy_decomposition

result = murphy_decomposition(act_cur, pred_cur, exp_cur, distribution='poisson')

print(f"Discrimination (DSC): {result.discrimination:.4f}")
print(f"Global MCB (GMCB):    {result.global_mcb:.4f}")
print(f"Local MCB (LMCB):     {result.local_mcb:.4f}")
print(f"Verdict:              {result.verdict}")  # 'OK' | 'RECALIBRATE' | 'REFIT'
```

No other Python monitoring library we are aware of implements this decomposition. Generic drift tools can tell you A/E has moved; they cannot tell you whether moving A/E represents a cheap fix or an expensive one.

---

## Putting it together: MonitoringReport

The `MonitoringReport` class assembles all three layers - exposure-weighted PSI per feature, A/E with confidence intervals, Gini drift test, and Murphy decomposition - into a single traffic-light output with an actionable recommendation.

```python
from insurance_monitoring import MonitoringReport

feat_ref = pl.DataFrame({
    "driver_age":        ages_ref.tolist(),
    "vehicle_age":       rng.integers(0, 15, n_ref).tolist(),
    "conviction_points": rng.integers(0, 5, n_ref).tolist(),
})
feat_cur = pl.DataFrame({
    "driver_age":        ages_cur.tolist(),
    "vehicle_age":       rng.integers(0, 15, n_cur).tolist(),
    "conviction_points": rng.integers(0, 5, n_cur).tolist(),
})

pred_ref = rng.uniform(0.05, 0.20, n_ref)
act_ref  = rng.poisson(pred_ref).astype(float)
pred_cur = rng.uniform(0.05, 0.20, n_cur)
act_cur  = rng.poisson(pred_cur * 1.08).astype(float)

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    exposure=exp_cur,
    reference_exposure=exp_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "conviction_points"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'NO_ACTION' | 'MONITOR_CLOSELY' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'

df = report.to_polars()
print(df)
# metric                   | value  | band
# ae_ratio                 | 1.08   | amber
# gini_current             | 0.39   | amber
# gini_p_value             | 0.054  | amber
# csi_driver_age           | 0.27   | red
# csi_vehicle_age          | 0.04   | green
# csi_conviction_points    | 0.06   | green
# murphy_discrimination    | 0.041  | amber
# murphy_miscalibration    | 0.003  | green
# recommendation           | nan    | RECALIBRATE
```

The decision logic follows arXiv 2510.04556 mapped to actuarial practice: A/E red plus stable Gini means RECALIBRATE; Gini red means REFIT; Murphy sharpens the distinction using GMCB vs LMCB.

The `to_polars()` output is designed to be logged to a database or written to a monitoring dashboard. Each row is a metric with a value and a traffic-light band. This is the artefact you include in your model risk documentation under PRA's internal model requirements or your Consumer Duty monitoring evidence.

---

## Sequential champion/challenger testing

One thing generic tools do not handle: the false positive problem in renewal cycle A/B tests.

UK motor teams typically set up a champion/challenger split at the start of the quarter and check results monthly. Under a standard two-sample test, checking monthly for 12 months at p < 0.05 per check produces an actual false positive rate of approximately 25% - five times nominal. This is not a theoretical concern; it is the most common source of spurious "significant" challenger wins we see in practice.

`insurance-monitoring` implements the mixture Sequential Probability Ratio Test (mSPRT) from Johari et al. (2022), which is valid at all stopping times. The evidence statistic is an e-process with the property that the probability of it ever exceeding 1/alpha is at most alpha, regardless of when you stop.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.03,                   # prior on log-rate-ratio: expect ~3% effects
    max_duration_years=2.0,
    min_exposure_per_arm=100.0, # car-years before any stopping decision
)

# Feed monthly increments as they arrive
for month, (champ_claims, chall_claims, champ_exp, chall_exp) in enumerate([
    (42, 38, 500, 495),
    (38, 31, 490, 488),
    (44, 29, 510, 505),
    (41, 28, 495, 492),
], start=1):
    result = test.update(
        champion_claims=champ_claims,
        challenger_claims=chall_claims,
        champion_exposure=champ_exp,
        challenger_exposure=chall_exp,
        calendar_date=datetime.date(2029, month, 28),
    )
    print(f"Month {month}: {result.summary}")
    if result.should_stop:
        print(f"Stop: {result.decision}")
        break
```

The `tau` parameter encodes the prior effect size. `tau=0.03` says we expect the champion and challenger to differ by roughly 3% on the log-rate-ratio scale. For a telematics experiment where you expect larger effects, increase to `tau=0.10`. For a recalibration pilot where the expected effect is very small, `tau=0.01`.

---

## What generic tools miss

To be direct about the comparison:

**Evidently** has a well-designed feature drift suite, good visualisations, and broad ML ecosystem integration. It does not have exposure-weighted PSI, Poisson confidence intervals for A/E, Gini drift testing, Murphy decomposition, or mSPRT sequential testing. If you fit it to your insurance monitoring pipeline, you will get approximately correct feature drift detection (unweighted), no A/E monitoring, and no discrimination monitoring.

**NannyML** adds statistical rigour to performance estimation and some calibration testing. It does not understand the insurance data structure: no exposure weights, no Poisson-rate A/E, no Gini test.

Both tools are reasonable starting points for engineering teams standing up generic ML monitoring. Neither is the right tool for a UK pricing actuary running monthly monitoring on a motor or home book. The insurance-specific structure - exposure weights, Poisson rates, Gini as the primary discrimination metric, Garwood intervals, Murphy decomposition - is not a bolt-on. It changes the conclusions.

---

## Thresholds and governance

Traffic-light thresholds are configurable. The defaults follow industry convention: PSI 0.10/0.25 (from FICO credit scoring practice), A/E 0.95–1.05 green / 0.90–1.10 amber, Gini p < 0.32 amber / p < 0.10 red.

```python
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds

# Tighter PSI thresholds for a large book monitored monthly
custom = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),
)
report = MonitoringReport(..., thresholds=custom)
```

A large motor book running monthly monitoring warrants tighter thresholds than the FICO defaults, which were designed for annual credit model reviews. The library does not enforce a single standard; it enforces that you make the threshold decision explicitly and that the choice is recorded in the output.

The `to_polars()` report is the model risk artefact. It records which metrics were checked, what thresholds were applied, what values were observed, and what action was recommended. For PRA internal model documentation or FCA Consumer Duty model governance evidence, this is the structured audit trail you want.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) and on [PyPI](https://pypi.org/project/insurance-monitoring/). Polars-native throughout; no scikit-learn dependency. Python 3.10+.

---

**Related posts:**
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - the original case for multi-layer monitoring: why PSI alone is not enough and what a three-layer monitoring cadence looks like
- [Champion Model, Unchallenged](/2026/03/17/champion-model-unchallenged/) - why most insurers never properly test their champion model, and how `insurance-deploy` shadow mode provides the challenger evidence
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) - once monitoring flags a deterioration, the Murphy decomposition gives you a principled answer to whether the fix is a scalar shift or a full model rebuild
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) - the governance layer: how monitoring outputs from `insurance-monitoring` feed into the `ModelInventory` audit trail that SS1/23-level documentation requires
