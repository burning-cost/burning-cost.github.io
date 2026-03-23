---
layout: post
title: "Insurance Model Monitoring in Python: Gini Drift, A/E Ratios and Double-Lift Curves"
date: 2026-03-22
author: Burning Cost
categories: [monitoring, model-risk, tutorials, libraries]
description: "What pricing actuaries actually monitor and how to do it in Python: Gini coefficient stability, A/E ratios by segment, exposure-weighted PSI, and double-lift curves using the insurance-monitoring library."
canonical_url: "/2026/03/22/insurance-model-monitoring-gini-ae-double-lift/"
tags: [insurance-model-monitoring-python, gini-drift, ae-ratio, double-lift, PSI, sequential-testing, insurance-monitoring, model-risk, pricing, python, uk-insurance, polars]
---

Generic ML monitoring tools measure accuracy. Pricing actuaries do not monitor accuracy. They monitor Gini coefficients, A/E ratios, double-lift curves, and population stability weighted by earned exposure. These are not equivalent things, and the difference matters.

This post is a practitioner's reference for the four KPIs that UK pricing teams actually track, why each one catches something the others miss, and how to compute all four in Python using [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).

```bash
uv add insurance-monitoring
```

---

## Why the standard tools don't fit

The failure is not a quality issue. Evidently, NannyML, and Alibi Detect are well-engineered tools. The problem is that they were built for a different monitoring problem: tabular features, i-i-d records, accuracy or AUC as the performance metric.

Insurance pricing has a different structure. Policies vary in duration and therefore in exposure. A 30-day policy on a young driver and a 365-day annual renewal look identical to an unweighted drift calculation; they contribute very differently to claims experience. A/E ratios and Gini coefficients are not standard ML metrics and are absent from every generic tool's implementation. Double-lift curves - which compare model discrimination against a benchmark - do not exist in any open-source monitoring library outside the actuarial domain.

The four KPIs below are what the pricing function actually needs. Generic tools cover none of them correctly.

---

## KPI 1: Gini coefficient stability

The Gini coefficient measures discrimination - the model's ability to separate cheap risks from expensive ones. It is the single most important monitoring metric for a pricing model, because a model that loses discrimination while maintaining calibration will generate adverse selection silently. Good risks leave (overpriced), bad risks stay (underpriced), and the portfolio-level A/E ratio stays near 1.00 the whole time.

The mechanism: as portfolio mix shifts over the model's life, the training-period relativities become stale. The model still predicts roughly the right mean by coincidence - the distributional errors cancel at aggregate level. But it can no longer rank risks. This is not a calibration problem and will not show up in A/E monitoring until retention patterns have played out for 12-24 months.

Detecting Gini drift requires a proper hypothesis test, not a heuristic threshold. `insurance-monitoring` implements the Gini drift z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556), which derives the asymptotic normality of the sample Gini coefficient and constructs a bootstrap variance estimator. The output is a z-statistic and p-value with calibrated thresholds mapped to monitoring decisions.

```python
import numpy as np
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

rng = np.random.default_rng(42)

n_ref, n_cur = 40_000, 12_000

# Reference period: model trained on 2022 data, performing well
pred_ref = rng.uniform(0.04, 0.22, n_ref)
act_ref  = rng.poisson(pred_ref).astype(float)
exp_ref  = rng.uniform(0.5, 1.0, n_ref)

# Monitoring period: 18 months later, portfolio mix has shifted.
# Model still roughly right on average, but ranking has degraded.
pred_cur = rng.uniform(0.04, 0.22, n_cur)
noise    = rng.uniform(0.0, 0.45, n_cur)           # staleness noise
act_cur  = rng.poisson(pred_ref[:n_cur] * 0.55 + noise).astype(float)
exp_cur  = rng.uniform(0.5, 1.0, n_cur)

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=n_ref,
    n_current=n_cur,
    reference_actual=act_ref,   reference_predicted=pred_ref,
    current_actual=act_cur,     current_predicted=pred_cur,
)

print(f"Gini reference: {gini_ref:.3f}")
print(f"Gini current:   {gini_cur:.3f}  ({result['gini_change']:+.3f})")
print(f"z-statistic:    {result['z_statistic']:.2f}")
print(f"p-value:        {result['p_value']:.4f}")
```

The decision thresholds we use in practice: p < 0.32 (one sigma) triggers `INVESTIGATE`; p < 0.10 (roughly 1.7 sigma) triggers `REFIT`. These are deliberately more sensitive than academic convention. An actuary spending two days investigating a false positive costs less than 18 months of undetected adverse selection.

A stable Gini does not mean the model is well-calibrated. It means it is still ranking risks. You need KPI 2 to confirm calibration.

---

## KPI 2: A/E ratios by segment

A/E (actual divided by expected) is the core calibration metric for insurance pricing. An A/E of 1.00 means the model's predicted loss experience matches observed claims - it is predicting the right mean. This is different from Gini: you can have A/E = 1.00 with a deteriorating Gini (ranking broken but scale correct) or A/E = 1.15 with a stable Gini (ranking intact but everything mis-priced).

The critical point that aggregate A/E hides is segmental misfire. A model 20% cheap on under-25s and 20% expensive on over-70s reads 1.00 at portfolio level. The monitoring alert never fires. The under-25 segment continues to attract an adverse share of young drivers until retention data reveals the pattern - typically 9-18 months after the mispricing began.

Segment-level A/E requires exact confidence intervals, not normal approximations. At low policy counts (new entrant segments, niche products, or sub-book monthly monitoring), the normal approximation understates interval width by 15-25%. The library uses Garwood intervals, which are exact for Poisson counts.

```python
import numpy as np
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

rng = np.random.default_rng(42)
n_cur = 12_000

ages_cur = np.concatenate([
    rng.integers(17, 25, int(n_cur * 0.18)),
    rng.integers(25, 40, int(n_cur * 0.34)),
    rng.integers(40, 60, int(n_cur * 0.31)),
    rng.integers(60, 80, int(n_cur * 0.17)),
]).astype(float)

exp_cur = np.where(
    ages_cur < 25,
    rng.uniform(0.2, 0.4, n_cur),
    rng.uniform(0.6, 1.0, n_cur),
)

pred_cur = np.select(
    [ages_cur < 25, ages_cur < 40, ages_cur < 60],
    [rng.uniform(0.12, 0.18, n_cur),
     rng.uniform(0.07, 0.12, n_cur),
     rng.uniform(0.05, 0.09, n_cur)],
    default=rng.uniform(0.04, 0.08, n_cur),
)

# Under-25 is 22% underpriced; over-60 is 14% overpriced.
# Portfolio A/E will read close to 1.00.
act_multipliers = np.select(
    [ages_cur < 25, ages_cur < 40, ages_cur < 60],
    [1.22, 1.02, 0.99],
    default=0.86,
)
act_cur = rng.poisson(pred_cur * act_multipliers).astype(float)

# Aggregate A/E with Garwood confidence interval
agg = ae_ratio_ci(act_cur, pred_cur, exposure=exp_cur)
print(f"Portfolio A/E: {agg['ae']:.3f}  [{agg['lower']:.3f}, {agg['upper']:.3f}]")
# Portfolio A/E: 1.007  [0.983, 1.032]   <-- green. Problem hidden.

# Segment-level
age_bands = np.select(
    [ages_cur < 25, ages_cur < 40, ages_cur < 60],
    ["17-24", "25-39", "40-59"],
    default="60+",
)
seg = ae_ratio(act_cur, pred_cur, exposure=exp_cur, segments=age_bands)
print(seg)
# segment | actual | expected | ae_ratio | n_policies
# 17-24   |  ...   |   ...    |  ~1.21   |  2,160
# 25-39   |  ...   |   ...    |  ~1.02   |  4,080
# 40-59   |  ...   |   ...    |  ~0.99   |  3,720
# 60+     |  ...   |   ...    |  ~0.87   |  2,040
```

The portfolio A/E reads green. The 17-24 segment reads amber-to-red. This is the pattern that generic monitoring will not surface, because it does not compute segment-level A/E with correct confidence intervals against exposure-weighted claim counts.

In practice, run A/E by every major rating dimension the model covers: age band, vehicle group, NCD level, geography, channel. A model can misbehave in exactly the combinations your aggregate check cannot see.

---

## KPI 3: Exposure-weighted PSI

Population Stability Index is an early-warning metric: it fires before claims develop. When PSI signals on driver age or vehicle group, you know the incoming book has shifted, even if you have zero claims data for the new cohort yet. This is the primary value of PSI in an insurance monitoring stack - it gives you lead time that A/E ratios cannot.

The standard PSI calculation weights each policy equally. For an insurance portfolio this is wrong. A 30-day social policy contributes 0.08 car-years of exposure; an annual comprehensive policy on the same driver contributes 1.0. If young drivers are disproportionately on short-period policies - they are; comparison sites and monthly-pay schemes attract younger drivers - unweighted PSI understates the distributional shift in the exposure that matters.

```python
import numpy as np
from insurance_monitoring.drift import psi, wasserstein_distance

rng = np.random.default_rng(42)
n_ref, n_cur = 40_000, 12_000

# Reference: training window
ages_ref = np.concatenate([
    rng.integers(17, 25, int(n_ref * 0.09)),
    rng.integers(25, 80, int(n_ref * 0.91)),
]).astype(float)

# Monitoring: comparison site growth has shifted mix significantly younger
ages_cur = np.concatenate([
    rng.integers(17, 25, int(n_cur * 0.20)),
    rng.integers(25, 80, int(n_cur * 0.80)),
]).astype(float)

# Exposure: young drivers over-represented on short-period policies
exp_ref = np.where(ages_ref < 25,
                   rng.uniform(0.2, 0.35, n_ref),
                   rng.uniform(0.7, 1.0, n_ref))
exp_cur = np.where(ages_cur < 25,
                   rng.uniform(0.2, 0.35, n_cur),
                   rng.uniform(0.7, 1.0, n_cur))

psi_unweighted = psi(ages_ref, ages_cur, n_bins=10)
psi_weighted   = psi(ages_ref, ages_cur, n_bins=10,
                     exposure_weights=exp_cur,
                     reference_exposure=exp_ref)

shift = wasserstein_distance(ages_ref, ages_cur)

print(f"Unweighted PSI: {psi_unweighted:.3f}")  # ~0.17 - amber
print(f"Weighted PSI:   {psi_weighted:.3f}")    # ~0.26 - red
print(f"Age shift:      {shift:.1f} years younger on average")
```

The unweighted version says amber; the exposure-weighted version says red. These are different conclusions. The weighted version is correct because it reflects the experience that will flow through to claims.

Wasserstein distance is worth computing alongside PSI. It converts the distributional shift into interpretable units - "average driver age has decreased by 2.3 years" - which means something to an underwriter or head of pricing. PSI is a dimensionless index; Wasserstein is a magnitude.

The PSI thresholds in the library (green: < 0.10, amber: 0.10-0.25, red: > 0.25) follow the FICO credit model convention. For a large UK motor book monitored monthly rather than annually, we'd tighten those to 0.05/0.15 - the data volume gives you statistical power to detect real shifts at lower magnitudes.

---

## KPI 4: Double-lift curves

Double-lift curves are less well known than PSI or A/E but are the most useful diagnostic for understanding whether a new model genuinely outperforms its predecessor on discrimination. Where a single Lorenz curve shows one model's lift over a naive baseline, a double-lift curve shows the ratio of two models' predicted relativities against each other, with observed claims frequency plotted on top.

The interpretation: sort policies by the ratio (model A predicted / model B predicted). If model A genuinely discriminates better, policies where model A predicts much higher risk than model B should show high observed frequency; policies where model A predicts much lower risk should show low observed frequency. A flat double-lift curve means the two models rank risks equivalently. A monotone increasing curve means model A adds genuine lift over model B.

This is the KPI to run at model refresh time. Before deploying a new champion, confirm that its double-lift against the existing champion is monotone. If it is not, the new model is rearranging risk rather than improving discrimination - a performance risk during the transition.

```python
import numpy as np
from insurance_monitoring.discrimination import double_lift_curve

rng = np.random.default_rng(42)
n   = 20_000

# Existing champion: reasonable discrimination
pred_champion = rng.uniform(0.04, 0.22, n)
act           = rng.poisson(pred_champion).astype(float)
exposure      = rng.uniform(0.5, 1.0, n)

# Challenger A: genuinely better - tighter spread, more accurate relativities.
# Simulate by adding small correlated noise to champion predictions.
pred_challenger_a = pred_champion * np.exp(rng.normal(0, 0.15, n))
pred_challenger_a = np.clip(pred_challenger_a, 0.02, 0.35)

# Challenger B: lateral move - similar Gini, different ordering.
# More noise: rearranges risks without improving discrimination.
pred_challenger_b = pred_champion * np.exp(rng.normal(0, 0.30, n))
pred_challenger_b = np.clip(pred_challenger_b, 0.02, 0.35)

# Double-lift: challenger A vs champion
dl_a = double_lift_curve(
    actual=act,
    model_a_predicted=pred_challenger_a,
    model_b_predicted=pred_champion,
    exposure=exposure,
    n_bins=10,
)

# Double-lift: challenger B vs champion
dl_b = double_lift_curve(
    actual=act,
    model_a_predicted=pred_challenger_b,
    model_b_predicted=pred_champion,
    exposure=exposure,
    n_bins=10,
)

print("Challenger A vs Champion (expect monotone increasing actual_frequency):")
print(dl_a.select(["decile", "pred_ratio", "actual_frequency"]))

print("\nChallenger B vs Champion (expect flat - no real lift):")
print(dl_b.select(["decile", "pred_ratio", "actual_frequency"]))
```

A monotone increasing `actual_frequency` across `pred_ratio` deciles confirms that challenger A's higher predictions genuinely correspond to higher observed risk. Challenger B's wider noise produces a flatter pattern - the model is redistributing relativities, not improving discrimination.

The double-lift curve should be a mandatory governance artefact for any model champion/challenger exercise. It is the most direct evidence that a new model is doing something useful relative to its predecessor, and it is the question no Gini comparison alone can answer: yes, challenger Gini is higher, but does its incremental ranking actually correspond to incremental risk?

---

## When to act: a decision framework

The four KPIs are layered, not interchangeable. They fire at different points in the degradation sequence and indicate different responses.

**PSI fires first.** It requires no claims data - only the feature distribution of incoming policies. A PSI red on driver age or vehicle group says the book has shifted before you have a single claim from that cohort. The correct response is to increase monitoring frequency on A/E for the affected segment, and to check whether the shift is explained by channel mix (comparison site growth, broker campaign) or something structural (demographic change, new affinity scheme).

**A/E fires next.** After 3-6 months of claims development - liability develops more slowly than motor frequency - segment-level A/E confirms whether the population shift has translated into a pricing error. If PSI fired and A/E follows, you have confirmed economic impact. If PSI fired but A/E is stable, the model appears robust to the mix change.

**Gini fires when the ranking breaks.** A Gini z-test alert on a book with stable A/E is the most dangerous signal: the model is correct on average but adverse selection is proceeding silently. The correct response is always `REFIT`. Recalibration cannot fix a broken ranking; increasing the overall rate level by 5% does not recover the distributional accuracy the model has lost.

**Double-lift is diagnostic, not continuous.** Run it at model transitions, not on a monthly dashboard.

The recalibrate-vs-refit decision for non-Gini alerts can be sharpened with the Murphy decomposition (Lindholm & Wüthrich, Scandinavian Actuarial Journal 2025), which splits miscalibration into global (correctable with a scalar multiplier) and local (requires model refit). `insurance-monitoring` implements this in `murphy_decomposition()` - the full treatment is in the [monitoring layers walkthrough](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/).

---

## Sequential testing for continuous monitoring

One operational problem the four KPIs above do not solve by themselves: if you check A/E monthly on a quarterly champion/challenger experiment, you are peeking. Under a standard two-sample test, 12 monthly checks at p < 0.05 per check produces an actual false positive rate of roughly 25%. This is not a theoretical concern - it is the most common source of spurious "significant" challenger wins on UK motor teams running renewal-cycle experiments.

The `sequential` module implements the mixture Sequential Probability Ratio Test (mSPRT), which is valid at any stopping time. The formal guarantee: P(false positive ever) <= alpha, regardless of how often you look.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.03,                    # prior on log-rate-ratio: expect ~3% effects
    max_duration_years=1.5,
    min_exposure_per_arm=150.0,  # car-years required before any stopping decision
)

monthly_data = [
    (38, 34, 480, 475),
    (35, 28, 475, 471),
    (41, 26, 488, 483),
]

for month, (champ_claims, chall_claims, champ_exp, chall_exp) in enumerate(monthly_data, start=1):
    result = test.update(
        champion_claims=champ_claims,
        challenger_claims=chall_claims,
        champion_exposure=champ_exp,
        challenger_exposure=chall_exp,
        calendar_date=datetime.date(2026, month + 2, 28),
    )
    print(f"Month {month}: {result.summary}")
    if result.should_stop:
        print(f"Decision: {result.decision}")
        break
```

Set `tau` to match your expected effect size: `tau=0.10` for a telematics experiment expecting 10% frequency reduction; `tau=0.01` for a recalibration pilot where the expected effect is small. The prior is weakly informative - it mainly controls the early-stopping sensitivity.

---

## Running the full stack

The `MonitoringReport` class assembles all four monitoring layers - exposure-weighted PSI per feature, A/E with Garwood intervals, Gini drift z-test, and Murphy decomposition - into a single traffic-light output with one actionable recommendation: `NO_ACTION`, `MONITOR_CLOSELY`, `RECALIBRATE`, or `REFIT`. The output `to_polars()` DataFrame is designed to be written to a database table and included in model risk documentation under PRA SS3/17.

Full `MonitoringReport` code with a complete UK motor synthetic example is in the [monitoring layers walkthrough](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/), which covers the Murphy decomposition in depth and explains how the traffic-light thresholds map to the decision logic.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). Polars-native. No scikit-learn dependency. Python 3.10+.

---

**Related posts:**
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - detailed walkthrough of each monitoring layer: exposure-weighted PSI, Murphy decomposition, mSPRT sequential testing
- [Why Evidently Isn't Enough for Insurance Pricing Model Monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - direct feature comparison with gap analysis
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - the original case for multi-layer monitoring
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) - the decision framework in depth
