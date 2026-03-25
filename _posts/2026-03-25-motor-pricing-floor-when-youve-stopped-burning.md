---
layout: post
title: "The Motor Pricing Floor - How to Know When You've Stopped Burning"
date: 2026-03-25
categories: [monitoring, pricing, strategy]
tags: [underwriting-cycle, loss-ratio, PSI, sequential-testing, mSPRT, A/E, Gini, portfolio-optimisation, insurance-monitoring, insurance-optimise, motor, python, tutorial]
description: "How to detect when a motor book has hit the floor of its underwriting cycle - using PSI on new business mix, segment-level A/E, Gini stability, and mSPRT to know when the next move is deterioration, not improvement."
---

The soft market is ending. You just do not know exactly when.

That ambiguity is the real problem. The combined ratio has been improving for six or seven quarters. The CFO wants to know when to start building reserves rather than releasing them. The commercial director wants to know whether the next rate cut will win volume or just burn margin. And the pricing team - if it is doing its job - should have a view on all three, ideally before the loss ratio has already started moving in the wrong direction.

This post is about the leading indicators that tell you the floor is near or already reached: the mix shifts in new business, the early signs in reserve development, the segment-level A/E patterns that arrive before the aggregate does. And then the practical question of when the data actually says stop cutting - using the mSPRT sequential test to detect when rate reductions have ceased improving loss ratios.

All code uses real APIs from [insurance-monitoring](https://burning-cost.github.io/insurance-monitoring) and [insurance-optimise](https://burning-cost.github.io/insurance-optimise). Nothing is invented.

---

## What "the floor" actually means

In a UK motor soft market, net rate reductions compound. From mid-2019 through 2021 (pre-whiplash reform), UK private motor combined operating ratios ran above 100% at most carriers - rates had fallen faster than inflation, injury claims were still inflating, and reinsurance costs were rising. The ABI's market statistics showed private motor GWP falling in nominal terms whilst claim costs rose. The "floor" is the point at which further rate cuts produce no corresponding benefit: loss ratios stop improving, or - the version nobody admits to at the time - the improvement was illusory because it reflected underwriting year immaturity, not genuine cost reduction.

There are two distinct senses of the floor:

**The technical floor**: the rate at which, in expectation, you are pricing below cost. The optimiser in `insurance-optimise` enforces this explicitly via `ConstraintConfig(technical_floor=True)` - it prevents multipliers from falling below 1.0 (price below technical price). But technical floors are only as good as the underlying technical price, and in a prolonged soft market the technical price itself may be stale.

**The cycle floor**: the macroeconomic turning point at which adverse development pressure, competitor exhaustion, and mix deterioration combine to start pushing loss ratios back up. This is what this post is about - and it happens 3 to 6 months before any aggregate loss ratio confirms it.

The gap between the two is why monitoring matters. You may still be pricing above your technical price whilst the cycle has already turned.

---

## Leading indicator 1: New business mix shift

The first sign of floor proximity is not a claims signal - it is a volume signal. When you are in a competitive soft market, the cheapest carriers are writing a disproportionate share of new business. As those carriers approach or breach their technical floors, they start to hesitate. The mix of new business across the market begins to shift toward risks that had been declined or declined-and-shopped.

The operational signal is PSI on your new business incoming risk profile versus the stable risk profile from 18 to 24 months ago. A PSI below 0.10 is normal background noise. PSI between 0.10 and 0.25 warrants investigation. PSI above 0.25 means your new business book looks materially different from the cohort your pricing model was calibrated on - and the direction of the shift matters as much as the magnitude.

```python
import polars as pl
from insurance_monitoring.drift import psi

# nb_current: new business written in last 3 months
# nb_reference: new business written 18-24 months ago
# Both DataFrames have columns: driver_age, vehicle_age, ncd_years,
#   annual_mileage, area_code, vehicle_group

nb_reference = pl.read_parquet("nb_reference_2024q2.parquet")
nb_current = pl.read_parquet("nb_current_2025q4.parquet")

# Exposure-weighted PSI on driver age
# Use earned_car_years as the weight — a one-month policy should not
# dilute the signal from an annual policy
psi_age = psi(
    reference=nb_reference["driver_age"],
    current=nb_current["driver_age"],
    n_bins=10,
    exposure_weights=nb_current["earned_car_years"],
    reference_exposure=nb_reference["earned_car_years"],
)

print(f"PSI driver_age (NB vs 18mo reference): {psi_age:.3f}")
# PSI 0.08 → green, normal background noise
# PSI 0.19 → amber, investigate: are we writing younger drivers than before?
# PSI 0.31 → red, significant shift: model calibration at risk
```

The segment that typically moves first is young drivers (under 25) and recently-passed (NCD 0-1). When competitors at the floor start pulling back, the young driver book concentrates at carriers still offering competitive rates. If your PSI on the under-25 sub-book starts moving amber whilst the whole-book PSI is still green, that is the early warning.

The right complement to PSI here is `wasserstein_distance` on the annual mileage distribution - it gives you a number in the original units ("average declared mileage has shifted by 1,200 miles per annum") which communicates more directly to underwriters than a dimensionless index.

---

## Leading indicator 2: Segment A/E patterns before they hit aggregate

The aggregate A/E ratio is a lagging indicator. Claims from an accident year do not fully develop for 12 to 18 months in UK motor (longer for injury-heavy cohorts), so the aggregate A/E you are monitoring today is describing what happened 12 months ago, not what is happening now.

Segment-level A/E tells you more, faster - because high-frequency, low-severity segments (minor property damage, windscreen) develop quickly and show A/E deterioration within 6 to 9 months of underwriting. If windscreen A/E is rising whilst the overall book A/E is flat, that is not necessarily a pricing problem - but if third-party property damage A/E is rising in the young driver segment, that is both a pricing signal and a frequency signal.

```python
from insurance_monitoring.calibration import ae_ratio
import numpy as np

# claims: actual claim counts, 2024 accident year at 12mo development
# freq_pred: model predicted frequency (claims per car-year)
# exposure: earned car-years
# age_band: ["<25", "25-39", "40-54", "55-69", "70+"]

claims_12mo = df["claims_12mo_developed"].to_numpy()
freq_pred = df["model_frequency"].to_numpy()
exposure = df["earned_car_years"].to_numpy()
age_band = df["age_band"]

# Aggregate A/E — the number everyone already knows
agg_ae = ae_ratio(
    actual=claims_12mo,
    predicted=freq_pred,
    exposure=exposure,
)
print(f"Aggregate A/E: {agg_ae:.3f}")   # e.g. 0.96 — looks fine

# Segment A/E by age band — the number that actually tells you something
seg_ae = ae_ratio(
    actual=claims_12mo,
    predicted=freq_pred,
    exposure=exposure,
    segments=age_band,
)

print(seg_ae)
# segment  actual  expected  ae_ratio  n_policies
# <25       412     318      1.295      8,412
# 25-39     891     876      1.017     31,540
# 40-54     632     641      0.986     28,810
# 55-69     285     298      0.957     14,200
# 70+       110     119      0.924      4,180
```

In this output, the aggregate A/E of 0.96 would not trigger any monitoring alert. The under-25 A/E of 1.295 almost certainly would, if the monitoring team were looking at it - but it only surfaces in the segment cut. A well-run pricing function is running this split monthly across at least: age band, NCD band, vehicle group, channel (direct vs broker vs PCW), and region.

The practical question is what A/E threshold triggers action. We use a simple rule: if the 95% confidence interval for a segment A/E excludes 1.0 in the adverse direction, it is flagged for review. `ae_ratio_ci` computes the Wilson interval on the claim count:

```python
from insurance_monitoring.calibration import ae_ratio_ci

# For the under-25 segment: 412 actual claims, 318 expected
ci = ae_ratio_ci(actual_claims=412, expected_claims=318, confidence=0.95)
print(f"A/E 95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
# A/E 95% CI: (1.177, 1.419) — entirely above 1.0
# The deterioration is real, not noise
```

---

## Leading indicator 3: Gini stability under adverse selection

As mix deteriorates, the Gini coefficient tells you something different from A/E: it tells you whether the model is still correctly ranking risks or whether the incoming new business contains a profile the model has not seen in quantity before.

In a normal operating environment, Gini should be stable between consecutive 12-month windows. A Gini that is falling whilst A/E is rising is the worst combination: you are under-predicting on average (A/E > 1) and the model is also losing its ability to distinguish cheap from expensive risks (Gini falling). In that state, a recalibration will not fix it - you need a refit.

The `GiniDriftBootstrapTest` runs the one-sample bootstrap test from arXiv 2510.04556, Algorithm 3. It treats the training-time Gini as fixed and tests whether the monitoring period Gini is significantly lower:

```python
from insurance_monitoring.discrimination import GiniDriftBootstrapTest

# predicted_rate: model predicted frequency per car-year
# actual_claims: binary 0/1 claim indicator per policy
# exposure: earned car-years per policy

gini_test = GiniDriftBootstrapTest(
    reference_gini=0.41,          # Gini at model training time (stored scalar)
    predicted=predicted_rate,
    actual=actual_claims,
    exposure=exposure,
    n_bootstrap=2000,
    alpha=0.10,                   # one-sigma rule for monitoring (earlier signal)
)
result = gini_test.test()

print(result.monitor_gini)        # e.g. 0.37
print(result.gini_change)         # e.g. -0.04
print(result.significant)         # True if p < alpha
print(gini_test.summary())
# Monitor Gini: 0.370 (95% CI: 0.351, 0.389)
# Training Gini: 0.410
# Change: -0.040 (p = 0.023, significant at alpha=0.10)
# Recommendation: REFIT — calibration AND discrimination have degraded
```

A Gini drop of 4 percentage points is meaningful in a UK private motor book. It means the model is less able to separate the high-frequency young drivers from the low-frequency experienced drivers - and if this coincides with the mix shift we saw in the PSI signal, it is because the incoming new business contains a distribution of risks the model has limited experience with.

---

## When the sequential test says stop: mSPRT on rate cut experiments

Everything above is observational - monitoring the existing book for signs of deterioration. The most important application is prospective: when you are running a rate cut experiment (champion = current rates, challenger = proposed reduction), the mSPRT tells you in real time whether the cut is actually improving loss ratios or has ceased to provide any benefit.

The intuition is simple. When a rate cut improves loss ratios, it does so because the volume increase brings in risks that were, on average, better than the risks you were writing at the higher rate. Once you are at the floor, further cuts bring in the same marginal risk profile - or worse - and loss ratios stop improving or start deteriorating. The sequential test detects the point at which the evidence for improvement is no longer accumulating.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

# Testing whether a 3% rate reduction improves claim frequency
# Champion: current book at existing rates
# Challenger: cells selected for 3% reduction (split by postcode sector)

test = SequentialTest(
    metric="loss_ratio",       # compound frequency × severity e-value
    alternative="less",        # we expect challenger LR < champion LR if cut is working
    alpha=0.05,
    tau=0.03,                  # prior: ~3% effects (calibrated to typical motor rate moves)
    max_duration_years=1.5,
    futility_threshold=0.10,   # stop early if evidence of improvement < 0.10
    min_exposure_per_arm=500.0, # car-years per arm before any stopping decision
)

# Monthly update loop — January through June 2026
monthly_data = [
    # (date, champ_claims, champ_exposure, chall_claims, chall_exposure,
    #   champ_sev_sum, champ_sev_ss, chall_sev_sum, chall_sev_ss)
    (datetime.date(2026, 1, 31),  142, 890,  61, 380,  1240.3, 10820.1,  528.7, 4610.4),
    (datetime.date(2026, 2, 28),  138, 875,  64, 372,  1198.5, 10441.2,  551.3, 4802.7),
    (datetime.date(2026, 3, 31),  151, 902,  67, 385,  1312.8, 11450.3,  582.4, 5080.2),
    (datetime.date(2026, 4, 30),  144, 888,  62, 379,  1255.1, 10940.6,  539.8, 4706.5),
    (datetime.date(2026, 5, 31),  149, 895,  69, 383,  1291.4, 11240.7,  599.2, 5222.3),
    (datetime.date(2026, 6, 30),  153, 908,  71, 391,  1330.6, 11598.4,  616.8, 5374.1),
]

for (date, cc, ce, lc, le, css, css2, lss, lss2) in monthly_data:
    result = test.update(
        champion_claims=cc,
        champion_exposure=ce,
        challenger_claims=lc,
        challenger_exposure=le,
        calendar_date=date,
        champion_severity_sum=css,
        champion_severity_ss=css2,
        challenger_severity_sum=lss,
        challenger_severity_ss=lss2,
    )
    print(f"{date}: {result.summary}")

# After 6 months:
# Decision: futility
# Challenger LR 1.8% lower (95% CS: 0.982–1.054). Evidence: 0.07 (threshold 20.0). Futility.
```

The futility decision is the key output here. The challenger has a slightly better loss ratio - 1.8% lower - but the evidence is only 0.07 against a rejection threshold of 20.0. The anytime-valid confidence sequence (0.982 to 1.054) straddles 1.0. There is no evidence the rate cut is working, and the mSPRT is terminating the experiment early.

This is the correct action at the floor: stop cutting, because the data has already told you the cuts are not improving your position. The fact that you cannot prove deterioration either is beside the point - you need positive evidence of improvement to justify the premium revenue sacrifice.

---

## Tying it together: constraint-based rate setting when you suspect the floor

Once the monitoring signals are pointing at the floor - PSI moving amber on NB mix, segment A/E climbing in the young driver cohort, Gini drift test approaching significance, sequential experiments reaching futility - the question for the optimiser changes. In a normal soft market the constraint is `lr_max` (do not price below the technical floor on aggregate). At the floor, you want to add a floor constraint as well.

`ConstraintConfig` in `insurance-optimise` supports `lr_min` for exactly this:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig
import numpy as np

# At the suspected floor: hold aggregate LR between 68% and 74%
# Do not allow the optimiser to chase volume by cutting below 68% LR
config = ConstraintConfig(
    lr_min=0.68,            # floor: stop pricing aggressively below this
    lr_max=0.74,            # ceiling: stay technically sound
    retention_min=0.80,     # still protect the renewal book
    max_rate_change=0.10,   # tighter cap: ±10% as the market turns
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=df["technical_premium"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["base_demand"].to_numpy(),
    elasticity=df["elasticity"].to_numpy(),
    renewal_flag=df["is_renewal"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    constraints=config,
)

result = opt.optimise()
print(result)
# OptimisationResult:
#   Status: optimal
#   Expected profit: £3.1M
#   Aggregate LR: 0.712          (constrained between lr_min and lr_max)
#   Retention rate: 0.821
#   Policies at lr_min floor: 3,240 (12.9%)  ← the rate-cutting cells, now blocked
```

The "policies at lr_min floor" figure tells you which cells the optimiser would have cut further if you had let it. At the floor of the cycle, these are the cells most likely to experience adverse development in the next 12 to 18 months. Tracking this number over consecutive monthly optimisation runs gives you a forward indicator of where your next pain will come from - the cells where the market is likely to harden first.

---

## A practical monitoring workflow

Put this all together as a monthly cadence:

1. **PSI on NB mix** - run on all major rating factor distributions. Flag any PSI > 0.15 for investigation. If PSI > 0.25 on young driver or NCD-0 sub-books, escalate.

2. **Segment A/E** - run for all accident years at current development. Focus on 6-month and 9-month development for frequency, since these develop fast enough to be actionable. Flag any segment A/E CI excluding 1.0.

3. **Gini drift** - run the one-sample bootstrap test quarterly, not monthly. Gini requires more data to stabilise. A significant Gini decline combined with A/E > 1.0 is a refit signal, not just a recalibration signal.

4. **mSPRT on active experiments** - any champion/challenger rate cut experiment should be monitored monthly with the futility threshold set. An experiment reaching futility is evidence you are at or past the floor in that cell.

5. **lr_min constraint review** - quarterly, review whether the lr_min constraint in the optimiser is still appropriate. At the floor, tighten it. When the market is hardening, the lr_min becomes a discipline mechanism, not just a technical guardrail.

The leading indicators arrive in this order: PSI on NB mix, then segment A/E in fast-developing cover types, then mSPRT futility on rate experiments, then aggregate A/E, then reserve development. By the time the last two are signalling, the floor is already history - you are now on the way back up.

---

## Further reading

- [Sequential A/B Testing for Insurance Champion/Challenger](/2026/03/24/sequential-ab-testing-insurance-champion-challenger/) - full walkthrough of mSPRT with UK motor synthetic data
- [Rate Change Evaluation with DiD and ITS](/2026/03/25/measure-rate-change-impact-python-did-its/) - for evaluating book-wide rate moves after the fact
- [insurance-monitoring documentation](https://burning-cost.github.io/insurance-monitoring) - PSI, A/E, Gini drift, SequentialTest
- [insurance-optimise documentation](https://burning-cost.github.io/insurance-optimise) - PortfolioOptimiser, ConstraintConfig, lr_min/lr_max
