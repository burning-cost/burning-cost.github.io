---
layout: post
title: "The Motor Pricing Floor — How to Know When You've Stopped Burning"
date: 2026-03-25
categories: [pricing, monitoring, motor]
tags: [underwriting-cycle, motor, loss-ratio, psi, ae-ratio, gini, sequential-testing, msprt, insurance-monitoring, insurance-optimise, uk-motor]
description: "Leading indicators for the floor of the motor underwriting cycle: PSI on new business mix, segmented A/E ratios, Gini stability tests, and mSPRT for detecting when rate cuts stop working."
---

The underwriting cycle for UK motor personal lines has a well-documented shape. Rates fall, loss ratios improve, volume grows, capital chases the margin. Then something tips. Loss ratios bottom out, reserve adequacy quietly deteriorates, and the claims that looked clean at 12 months develop badly at 36. By the time the combined ratio starts rising again, you have already been repricing yesterday's risk at tomorrow's cost.

The question pricing teams rarely formalise is: how do you know you are at the floor? Not in retrospect, but in real time, with the tools available to a working pricing actuary.

This post is about that detection problem. We work through four complementary leading indicators — new business mix drift, segmented A/E ratios, Gini stability, and sequential testing on rate cut experiments — and show how to implement them using [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) and [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise).

---

## What "the floor" actually means

The floor is not a single event. It is a regime transition — the point at which the market's collective rate adequacy has been exhausted and the next marginal rate cut no longer improves underwriting profit. In a soft market, cutting rates grows volume and, for a period, still clears the technical hurdle because the underlying risk cost is falling too (or you think it is). At the floor, the two effects have decoupled: cutting further means writing unprofitable business.

The textbook signal is the combined ratio. UK motor's combined ratio troughed around 96% in 2014-15, with the market only hardening materially after the Ogden rate shock in March 2017. A decade later, the same pattern: combined ratios were recovering into 2021 before used car valuations corrected sharply, repair costs accelerated with parts shortages and labour inflation, and the 2022-23 hardening came. Both cycles show the same lag. Reserving actuaries were revising IBNR upward four to six quarters before the pricing market visibly moved.

We think of the floor in terms of three structural observations:

1. **The marginal risk coming through the door is getting worse.** New business mix shifts as better risks stop shopping (they have found their rate) and adverse risks keep searching. The rate your model quotes to a new enquiry is evaluated against a changing pool, not the pool it was trained on.

2. **Segment-level A/E ratios stop converging.** In a rate-adequate environment, A/E tends toward 1.0 across segments as rate reductions are absorbed by genuine improvement. At the floor, specific segments begin to show A/E > 1.1 — the first places where your rate is below cost.

3. **Reserve adequacy starts eroding in the most recent accident years.** This is the lagging signal, but it is the most reliable. Track 12-month-to-18-month and 18-month-to-24-month development factors by accident quarter. When the development starts going adverse on recent quarters, the floor is already behind you.

---

## Leading indicator 1: PSI on new business mix

Population Stability Index is the standard tool for detecting mix shift, and it is particularly useful here because the signal we care about is a _relative_ shift between new business and renewals. In a soft market, new business mix shifts toward the adverse end as competitors undercut you on better risks. Your NB book gets worse on average, but since rates are also falling, gross loss ratios can stay flat or improve — until they do not.

```bash
uv add insurance-monitoring
```

```python
import numpy as np
from insurance_monitoring.drift import psi

# Reference period: NB driver age distribution at model training
# Current period: NB driver age distribution this quarter
nb_driver_age_reference = ...  # your training window NB
nb_driver_age_current   = ...  # current quarter NB

nb_psi = psi(
    reference=nb_driver_age_reference,
    current=nb_driver_age_current,
    n_bins=10,
    exposure_weights=nb_exposure_current,       # earned car-years, current
    reference_exposure=nb_exposure_reference,   # earned car-years, reference
)

print(f"NB driver age PSI: {nb_psi:.3f}")
# < 0.10 : stable
# 0.10-0.25: investigate — who is coming through the door?
# > 0.25 : significant shift — your NB book has changed materially
```

The interpretation we care about at the floor is not just "has the mix changed?" but "which direction?" A PSI of 0.20 on driver age that is driven by a shift _toward younger drivers_ is a different problem to one driven by ageing. Run this monthly, and track the direction of the Wasserstein distance — `wasserstein_distance()` from the same module returns the shift in original units ("NB driver age has moved 2.1 years younger since Q3 2024"). That is a number you can show to an underwriting director.

The comparison that matters is **NB PSI versus renewal PSI on the same feature**. If renewal driver age is stable but NB is drifting younger, that is adverse selection operating in real time: younger drivers are disproportionately shopping, and you are winning some of that business at rates set for the old mix. By the time the A/E ratio reflects it, you will have 12 months of poorly priced exposure on risk.

---

## Leading indicator 2: Segmented A/E ratios

The aggregate loss ratio is the last place you want to detect the floor — it is already behind you when it moves. What matters is which segments are breaking first.

```python
import numpy as np
from insurance_monitoring.calibration import ae_ratio

# Segment by vehicle age band — older vehicles tend to break first
# as parts costs accelerate
segments = np.where(
    vehicle_age <= 3,  "0-3yr",
    np.where(vehicle_age <= 7, "4-7yr", "8yr+")
)

ae_by_vehicle_age = ae_ratio(
    actual=claim_counts,
    predicted=model_frequency,
    exposure=car_years,
    segments=segments,
)

print(ae_by_vehicle_age)
# Returns a Polars DataFrame:
# segment | actual | expected | ae_ratio | n_policies
# 0-3yr   | 840    | 830      | 1.012    | 11200
# 4-7yr   | 1120   | 1010     | 1.109    | 9800
# 8yr+    | 890    | 760      | 1.171    | 7100
```

An A/E of 1.17 on older vehicles is not noise. At a frequency of 8% and 7,100 policies, that is roughly 100 excess claims per annum versus expectation. In a 12-month accident year at £3,500 average cost, you are looking at £350k of unmodelled loss already accumulated — before IBNR development.

The segments that tend to break first in a UK motor soft market are:

- **Older vehicles (8+ years):** parts costs inflate faster than your severity model updates, and IBNR development is longer because of disputes and third-party claims on older vehicles.
- **Young drivers on short-term or monthly policies:** adverse selection concentrates here as insurers undercut each other on conversion rates, and IBNR is thin at short durations.
- **High vehicle group codes (6-7):** prestige and performance vehicles where repair costs have a longer tail.

Track A/E at the segment level every quarter with at minimum 12 months of development on the accident year before concluding it is adequate. The floor tends to show up as a consistent pattern of A/E > 1.05 on two or three of these segments simultaneously, not as a single outlier.

---

## Leading indicator 3: Gini stability

A stable or rising Gini coefficient at the portfolio level is consistent with rate adequacy: your model is correctly ranking risks, the market is pricing on similar relativities, and the risks you are winning are the ones your model expects. Gini deterioration at the floor has a specific signature: the discriminatory power of your model falls because the pool of risks presenting for quotes has changed shape.

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test_onesample

# Gini at model training (stored as a governance artefact)
training_gini = 0.312  # from model sign-off, Q1 2024

# Current monitoring period
current_gini = gini_coefficient(
    actual=claim_counts_current,
    predicted=model_frequency_current,
    exposure=car_years_current,
)

# One-sample bootstrap test: is the current Gini significantly lower
# than the training Gini? (Algorithm 3, arXiv:2510.04556)
result = gini_drift_test_onesample(
    actual=claim_counts_current,
    predicted=model_frequency_current,
    exposure=car_years_current,
    reference_gini=training_gini,
    n_bootstrap=1000,
    alpha=0.05,
)

print(f"Training Gini: {training_gini:.3f}")
print(f"Current Gini:  {current_gini:.3f}  (change: {result.gini_change:+.3f})")
print(f"Significant:   {result.significant}")
```

The Gini drift result we are looking for at the floor is not just a statistically significant decline — it is a _directional_ decline on the new business book specifically, with the renewal Gini stable or improving. That pattern says: the renewal book is still being priced competently, but the market is writing NB at inadequate rates and your relativities are being swamped by portfolio mix effects.

Run `gini_drift_test_onesample` separately on NB and renewal sub-portfolios. If renewal Gini is at 0.31 and NB Gini has dropped to 0.24, with `significant=True`, you are watching adverse selection in the Gini data before you see it in the loss ratio.

---

## Leading indicator 4: Sequential testing on rate cut experiments

The most direct test for whether you are at the floor is an experiment: does cutting rates on a specific segment still improve loss ratios? In a soft market with genuine cost reduction, rate cuts increase volume at acceptable or improving loss ratios — demand elasticity is working in your favour. At the floor, cutting rates increases volume but the new volume comes in with a worse risk profile, and the loss ratio moves against you.

`insurance-monitoring` implements the mixture Sequential Probability Ratio Test (mSPRT, Johari et al. 2022) for exactly this kind of champion/challenger test. The key property for pricing is that mSPRT is anytime-valid: you can look at results monthly as experience develops without inflating your type I error. A standard t-test at each quarter-end gives you a 5% error rate per look; with four looks per year, your actual error rate is around 19%.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

# Champion: current rate level on 40-49yr vehicle group 5 segment
# Challenger: -8% rate cut, written from Q3 2024
test = SequentialTest(
    metric="loss_ratio",
    alternative="two_sided",
    alpha=0.05,
    tau=0.05,                  # prior std dev on log-LR-ratio: expect ±5% effects
    max_duration_years=2.0,
    min_exposure_per_arm=250.0,  # car-years before any decision
)

# Feed quarterly increments as experience develops
quarters = [
    # (date, champ_claims, champ_exposure, chall_claims, chall_exposure,
    #  champ_sev_sum, champ_sev_ss, chall_sev_sum, chall_sev_ss)
    (datetime.date(2024, 9, 30),  42, 380,  38, 395,  ...),
    (datetime.date(2024, 12, 31), 48, 392,  51, 408,  ...),
    (datetime.date(2025, 3, 31),  44, 385,  57, 401,  ...),
    (datetime.date(2025, 6, 30),  46, 378,  62, 397,  ...),
]

for (date, cc, ce, chc, che, css, csss, chss, chsss) in quarters:
    result = test.update(
        champion_claims=cc, champion_exposure=ce,
        challenger_claims=chc, challenger_exposure=che,
        calendar_date=date,
        champion_severity_sum=css, champion_severity_ss=csss,
        challenger_severity_sum=chss, challenger_severity_ss=chsss,
    )
    print(result.summary)
```

What the floor looks like in the mSPRT output: by Q4 2024 the challenger loss ratio is moving in the wrong direction. The rate cut brought volume — the challenger arm has more exposure — but the challenger claims rate is rising. By Q2 2025, if the evidence is accumulating toward `reject_H0` with `rate_ratio > 1.0`, you have your answer: cutting rates on this segment is making things worse, not better. The e-process value `lambda_value` rising past `1/alpha = 20` while the rate ratio is above 1.0 is the signal that the cut has failed.

The `futility_threshold` parameter is useful here too. Set it to 0.1: if the evidence for any effect has collapsed below that threshold after 250 car-years in each arm, the experiment is futile — the effect is too small to detect in a reasonable time horizon, which itself is useful information about where you are in the cycle.

---

## Tying it together: the floor checklist

When we see three or more of the following simultaneously, we treat it as a floor signal:

- **NB PSI > 0.15** on driver age or vehicle age, with NB skewing toward the adverse end relative to renewals
- **Segmented A/E > 1.10** on two or more of: 8yr+ vehicles, young driver NB, vehicle groups 6-7
- **NB Gini decline > 0.03** with `significant=True` from `gini_drift_test_onesample`
- **Rate cut experiments returning `rate_ratio > 1.05`** with evidence accumulating toward rejection (lambda > 10)
- **12-to-18 month development factors on the most recent accident quarter going adverse** versus the prior year same quarter

None of these alone is conclusive. The PSI can spike because of a product change or a panel switch, not because of cycle dynamics. A/E can move for IBNR reasons. Gini can decline because you updated the model and the new version has lower discrimination on the monitoring set — a well-known trap.

But three or more signals together, sustained over two consecutive quarters, is the pattern we would stake a reserve recommendation on. By that point, the constraint to impose on the optimiser is not a lower loss ratio target — it is a floor on the loss ratio:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

# At the floor: stop cutting. Impose lr_min to prevent further
# deterioration, and enforce technical_floor so no policy is
# priced below cost.
config = ConstraintConfig(
    lr_min=0.68,          # do not let aggregate LR fall below this
    lr_max=0.75,          # still protecting the upside
    technical_floor=True, # m_i >= 1.0 everywhere
    max_rate_change=0.10, # contain the rate movement either direction
)
```

The `lr_min` constraint in `ConstraintConfig` is the explicit mechanism for saying "we believe the book is at the floor and further rate reductions are not in the portfolio's interest." It prevents the optimiser from chasing volume through inadequate pricing when the demand model still shows elasticity — because at the floor, that elasticity is a trap, not an opportunity.

---

## Timing and IBNR

One honest caveat. Everything in this post is subject to the development problem. Claim counts at 6 months look better than they will at 36 months, especially for bodily injury, third-party property damage, and complex liability claims. Our 12-month minimum development lag recommendation for the A/E calculation is conservative for large claims but reasonable for frequency.

The mSPRT sequential test partially addresses this: by running it for two years with monthly updates, you are accumulating development incrementally rather than fixing a point-in-time snapshot. But you are still working with incurred claims that are not fully developed. If your reserving team is strengthening recent accident year IBNR — adding to the estimates, not releasing — treat that as a leading indicator of its own, independent of anything the monitoring tools show.

The floor of the underwriting cycle is ultimately a reserving problem wearing a pricing hat.

---

```bash
uv add insurance-monitoring insurance-optimise
```

Source: [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) · [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

- [Your Book Has Shifted and Your Model Doesn't Know](/2026/09/14/your-book-has-shifted-and-your-model-doesnt-know/) — PSI and CSI in depth for pricing model monitoring
- [How to Detect Covariate Interactions Your GLM Missed](/2026/11/14/how-to-detect-covariate-interactions-your-glm-missed/) — segment-level diagnostics for rate inadequacy
