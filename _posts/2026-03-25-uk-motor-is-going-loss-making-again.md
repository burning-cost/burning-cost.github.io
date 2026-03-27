---
layout: post
title: "UK Motor Is Going Loss-Making Again — Here's the Data"
date: 2026-03-25
categories: [market-intelligence, pricing]
tags: [uk-motor, combined-ratio, claims-inflation, rate-adequacy, loss-ratio, insurance-monitoring, insurance-causal, insurance-optimise, market-intelligence, python]
description: "EY forecasts a 111% net combined ratio for UK motor in 2026. WTW documents a 13% annual premium fall. Here is what the data shows and what pricing teams should do about it."
---

UK motor is heading back into loss. Not as a possibility, not as a tail scenario — as the central forecast from the sector's most-watched analysts. The numbers are clear enough that we think pricing teams should be treating 2026 as a rate correction year, not a volume protection year.

Here is what the data shows, why it is happening, and what to do about it.

---

## The numbers

EY's December 2025 Motor Insurance Results Analysis reported a net combined ratio (NCR) of 97% for 2024 — the first underwriting profit since 2021. They forecast 101% for 2025 and 111% for 2026. To be concrete: for every £1 of premium earned in 2026, the sector is expected to pay out £1.11 in claims and expenses. That is a structural loss, not a rounding error.

The premium side of the equation is what has driven this reversal. EY estimates consumer premiums fell 10% in 2025, an average saving of £54 per policy. WTW's January 2026 Motor Insurance Premium Tracker, which analyses nearly 28 million policies, put the annual premium fall at 13% — a £111 reduction taking the average policy to £726, down from a peak of £995 in December 2023. Eight consecutive quarters of falling premiums. The WTW figure for age-17 drivers is striking: a 25% fall, bringing average premiums for that cohort from £2,568 to £1,932.

The ABI's Q3 2025 Motor Insurance Premium Tracker, covering actual customer prices rather than quoted rates, reported an average premium of £551 for Q3 2025 — £56 lower year-on-year. Total claims payouts for Q3 2025 reached £3 billion, with repair costs alone accounting for £1.9 billion of that, or 64% of total claims.

EY's 2026 forecast assumes no reserve releases. Insurers with conservative reserve positions from the 2022–2024 hard market may see partial offset, but the underwriting trajectory is the same regardless.

---

## Why it is happening

Three forces are running simultaneously, and they compound each other.

**Claims inflation has not stopped.** The ABI documented a record £11.7 billion in motor claims paid across 2024 — 17% higher than 2023. Repair costs are the primary driver. Modern vehicles with integrated sensors, cameras, and ADAS systems require specialist repair facilities and longer cycle times. Even a minor low-speed impact now frequently triggers ADAS recalibration. Paint costs are up 16%, spare parts up 11%, and replacement vehicle costs up 47% versus 2019 levels (ABI). The shortage of skilled ADAS-qualified technicians has not resolved; if anything, the UK's training pipeline has not kept pace with the vehicle parc turnover towards EVs and hybrid systems.

Bodily injury tells a more nuanced story. WTW's Tim Rourke noted in January 2026 that bodily injury claims have fallen from 16% of insurers' spend in 2021 to 9% in 2025, largely attributable to the whiplash reforms enacted via the Civil Liability Act 2018 and the Official Injury Claim portal. That structural improvement is real. But serious injury claims — those involving long-term care costs — continue to inflate above CPI, driven by care worker shortages and the 2025 Ogden rate review uncertainty. The OGD rate review remains outstanding at time of writing and represents a meaningful reserve risk for bodily injury-heavy books.

**The price comparison market amplified the premium cycle.** Aggregators accelerated both the premium rise of 2022–2023 and the subsequent fall. When premiums peaked in late 2023, consumer media coverage of the affordability crisis was intense — ABI and FCA both ran public commentary on it. Insurers under competitive pressure from aggregators and the FCA's Consumer Duty framework (PS22/9) responded by cutting rates through 2024 and 2025. The 2024 premium increase of 14% (EY) represented the market correcting for prior underpricing. That correction was immediately reversed by competition in 2025. The sector's collective inability to hold discipline during a soft cycle is not new, but the aggregator-mediated speed of price transmission has made cycles shorter and more violent.

**Model error compounds the problem.** Pricing models were calibrated on data from 2019–2023, a period of unusually severe claims inflation. The severity uplift applied to 2024 and 2025 pricing reflected the inflation that had already occurred — not the inflation that is continuing. Many teams' frequency models have not been validated against Q3 and Q4 2025 claims emergence. If your model's expected loss cost reflects 2023 repair prices, your technical premium is understated by the cumulative effect of 12–18 months of additional repair cost inflation.

---

## Leading indicators your pricing team should watch

Before you can respond to a deteriorating market, you need to know your book is deteriorating — ideally before the quarterly management accounts tell you. Three metrics matter most.

**A/E ratio by accident quarter.** The actual-to-expected ratio on paid claims, tracked by accident quarter with a development lag correction, is the earliest reliable signal. A/E ratios drifting above 1.05 on recent accident quarters, before claims reach their expected development point, suggest your model's expected cost is understated. [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) has this built in:

```python
from insurance_monitoring import ae_ratio, ae_ratio_ci, MonitoringReport

# ae_ratio returns the A/E scalar; ae_ratio_ci adds Poisson confidence intervals
ratio = ae_ratio(actual=paid_claims, predicted=model_expected)
result = ae_ratio_ci(actual=paid_claims, predicted=model_expected, alpha=0.05)
lower, upper = result["lower"], result["upper"]

print(f"A/E: {ratio:.3f}  95% CI: [{lower:.3f}, {upper:.3f}]")
```

An A/E above 1.0 is not automatically a crisis — reserve margins and development assumptions explain much of it. But an A/E that has moved from 0.98 to 1.06 over three accident quarters without a corresponding change in development assumptions is a signal worth investigating.

**PSI on repair cost distributions.** If your pricing model uses vehicle group as a proxy for repair cost, monitor PSI on vehicle group and age distributions monthly. The mix shift towards newer vehicles with higher ADAS content will elevate your repair cost without changing the rating factor, because the rating factor is calibrated to historical costs for that group. The exposure-weighted PSI from `insurance-monitoring` is the right tool here — unweighted PSI treats a 0.1 car-year policy identically to a 1.0 car-year policy:

```python
from insurance_monitoring import psi

psi_value = psi(
    reference=reference_vehicle_age,
    current=current_vehicle_age,
    n_bins=10,
    exposure_weights=current_exposure,
    reference_exposure=reference_exposure,
)

# PSI > 0.25: significant distribution shift, model revalidation warranted
print(f"Vehicle age PSI: {psi_value:.4f}")
```

**MonitoringReport for a complete picture.** For teams running monthly model governance, the combined report gives you A/E, Gini, and PSI in one call with traffic-light outputs:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=reference_claims,
    reference_predicted=reference_expected,
    current_actual=current_claims,
    current_predicted=model_expected,
    feature_df_reference=reference_features,
    feature_df_current=current_features,
    exposure=current_exposure,
)
print(report.recommendation)
print(report.to_dict())
```

The output distinguishes between calibration failure (A/E outside thresholds) and discrimination failure (Gini drift), which matters for deciding whether the response is a scalar adjustment or a model refit.

---

## How to respond

Monitoring tells you the book is burning. It does not tell you what to do about it. The response has three components.

**Separate market effect from portfolio effect.** Before you can set rates, you need to know how much of the A/E deterioration is the market (everyone's repair costs are up) versus your portfolio (your mix has shifted towards higher-risk segments). These look the same in aggregate A/E but require completely different responses. [insurance-causal](/2026/03/10/rate-change-lapse-evaluation-causal-inference/)'s `RateChangeEvaluator` uses a difference-in-differences framework to isolate the portfolio-specific effect:

```python
from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data

evaluator = RateChangeEvaluator(
    outcome_col="loss_ratio",
    treatment_period=9,          # month index when market hardened
    unit_col="segment",
    weight_col="earned_exposure",
)
result = evaluator.fit(df).summary()
print(result)
```

If your deterioration tracks the market (EY's 111% NCR trajectory), you have a pricing adequacy problem. If your deterioration exceeds the market, you have a portfolio selection problem as well.

**Rate adequacy by segment.** A portfolio-level rate increase decision made without segment-level profitability analysis will produce an uneven outcome: adequately priced segments will be pushed past competitive rates and lose volume, while underpriced segments get inadequate correction. The standard approach is to compute technical loss ratios at segment level — by vehicle group, age band, and channel — then apply rate change factors calibrated to the technical premium gap.

**Constrained rate optimisation.** The FCA's Consumer Duty framework (PS22/9) constrains how you can move rates, particularly on renewal books where existing customers may face significant increases. [insurance-optimise](/2026/03/07/insurance-optimise/) handles this directly — you specify loss ratio targets, maximum rate change limits per policy, volume retention floors, and the ENBP constraint from PS21/5, and it solves for the profit-maximising premium set within those constraints:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.75,            # target loss ratio: pull back from current trajectory
    retention_min=0.82,     # minimum volume retention
    max_rate_change=0.25,   # cap rate increases at 25% per policy
    enbp_buffer=0.01,       # PS21/5 margin
)

opt = PortfolioOptimiser(
    technical_price=technical_prices,
    expected_loss_cost=expected_loss_costs,
    p_demand=conversion_probs,
    elasticity=price_elasticities,
    renewal_flag=is_renewal,
    enbp=enbp_benchmarks,
    constraints=config,
)

result = opt.optimise()
print(f"Expected profit: £{result.profit:,.0f}")
print(f"Volume retention: {result.expected_retention:.1%}")
```

The key constraint tension in the current market is `lr_max` versus `max_rate_change`. A 111% NCR implies you need material rate increases to return to adequacy, but Consumer Duty limits how much you can move in a single renewal cycle. The optimiser surfaces this tension explicitly: if the constraints are infeasible, it tells you, which is more useful than a number that looks right but reflects an impossible set of requirements.

---

## What we are watching

The pace of premium recovery will be the critical variable. EY forecasts a 3% premium increase for 2026 — about £15 per policy on average. Given the 10% fall in 2025, that is not close to adequate for a sector heading towards 111% NCR. The market needs something closer to 8–12% to stabilise underwriting results by 2027, but whether competitive dynamics allow it is a different question.

The Ogden discount rate review is the largest single reserve uncertainty. The current rate of -0.25% (set in 2019) is under review. A further reduction — which the actuarial consensus considers likely given long-term investment return assumptions — would increase bodily injury reserves and push 2026 NCRs higher than EY's central forecast.

On the claims side, the ADAS repair cost question is structural rather than cyclical. Every year, more of the vehicle parc is newer, more complex, and more expensive to repair. That trajectory does not reverse. Teams pricing older-vehicle segments may see some relative improvement, but the book-wide repair cost trajectory is upward.

We will update our monitoring tracking on this as ABI Q4 2025 data releases and as EY's mid-year 2026 analysis becomes available — typically June.

---

*Sources: [EY UK Motor Insurance Results Analysis, December 2025](https://www.ey.com/en_uk/newsroom/2025/12/ey-latest-motor-insurance-results-analysis); [WTW UK Car Insurance Premium Tracker, January 2026](https://www.wtwco.com/en-gb/news/2026/01/uk-car-insurance-premiums-continue-to-slide-with-13-percent-annual-fall); [ABI Motor Insurance Premium Tracker, November 2025](https://www.abi.org.uk/news/news-articles/2025/11/three-straight-quarters-of-falling-motor-premiums/); [ABI Motor Claims Data, July 2025](https://www.abi.org.uk/news/news-articles/2025/7/motor-premiums-fall---but-repair-and-theft-costs-keep-revving-up-claims/).*
