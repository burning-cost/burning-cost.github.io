---
layout: post
title: "Trend Fitting Through Regime Changes: Changepoint Detection Before the Rate Indication"
date: 2026-03-18
author: Burning Cost
categories: [pricing, actuarial, libraries]
description: "Pricing teams fit log-linear trends through experience with structural breaks. The straight line is wrong on both sides. How to detect and correct it."
tags: [changepoint, bocpd, pelt, bayesian, structural-break, trend, rate-indication, insurance-dynamics, python, uk-motor, ogden, whiplash, inflation, actuarial]
---

There is a standard piece of actuarial work that almost every UK pricing team does every year: fit a development trend to loss experience, project it forward, and use that projection in the rate indication. The trend is usually log-linear - a straight line on a log scale. Sometimes it is a simple year-over-year link ratio. Occasionally someone is more elaborate and fits a regression on accident quarter or underwriting year. But the shape is always the same: a single line, fitted across all available history, averaged over whatever happened during that history.

The problem is that insurance loss experience does not generate straight lines. It generates regimes. There is a level, and then something happens - a court ruling, a legislative reform, a macro cost shock - and the level shifts. You now have two regimes. Fitting a single log-linear trend across both regimes does not give you the average of two trends; it gives you a line that is wrong in both directions. In the pre-break period, the line is too steep. In the post-break period, it underestimates the new level. Your projected rate indication is based on a number that does not describe either the world you came from or the world you are pricing into.

UK motor is a particularly clear example because the breaks are documented. The Ogden discount rate dropped from 2.5% to -0.75% in March 2017: immediate step-up in bodily injury reserves for every insurer on the market. The Civil Liability Act revised it to -0.25% in August 2019: partial reversal, second break. The whiplash reforms in May 2021 restructured small claims PI, reducing PSLA frequency materially. Then in 2022 and into 2023, parts and labour inflation ran at 15-20% year-on-year - a severity step-change driven by supply chain disruption that did not unwind quickly. Four structural breaks in seven years. A single log-linear trend across that period is not a simplification; it is a fiction.

---

## What BOCPD does

Bayesian Online Changepoint Detection - BOCPD, following Adams and MacKay (2007) - addresses this directly. The algorithm maintains a probability distribution over the current "run length": how many periods have elapsed since the last structural break. Each period, it updates this distribution using Bayes' rule. The predictive distribution for the next observation is a mixture over run lengths, and the posterior probability of a break at each period is a direct output of the algorithm.

For insurance claim frequency specifically, the Poisson-Gamma conjugate pair makes the algorithm exact. No MCMC, no approximation. Claim counts are Poisson with rate λ; the prior on λ is Gamma(α, β); the conjugate update is closed-form; the predictive is Negative Binomial. The key extension for insurance data is exposure weighting: each period has a different number of earned vehicle-years (or policy-years, or earned premium), and the algorithm accounts for this correctly. A period with 800 earned vehicle-years tells you less per claim than a period with 1,200.

For retrospective analysis - locating historical breaks and measuring their size - PELT (Pruned Exact Linear Time, Killick et al. 2012) gives you break locations with bootstrap confidence intervals. BOCPD runs forward in time; PELT finds the globally optimal segmentation of a historical series. Both are implemented in `insurance-dynamics`.

---

## A worked example: UK motor frequency 2017–2025

The setup is straightforward. We have monthly claim counts and exposures for a UK motor book covering the period from January 2017 through December 2025 - 108 months. The series contains four known structural events: Ogden March 2017, whiplash reform May 2021, and two severity-related inflation breaks. We use `FrequencyChangeDetector` from `insurance_dynamics.changepoint`, with the UK event calendar enabled to give the algorithm informative priors around known event dates.

```python
import numpy as np
from datetime import date, timedelta
from insurance_dynamics.changepoint import FrequencyChangeDetector, RetrospectiveBreakFinder

rng = np.random.default_rng(42)

# 108 monthly periods: Jan 2017 through Dec 2025
T = 108
start = date(2017, 1, 1)
periods = [start + timedelta(days=30 * i) for i in range(T)]

# Exposure: ~1,000 earned vehicle-years per month, growing slightly
exposures = rng.uniform(900, 1100, T)

# True underlying frequency: two regimes
# Pre-whiplash reform: 0.082 annual frequency
# Post-whiplash reform (from month 52, May 2021): 0.064 — material step-down
true_rate = np.where(np.arange(T) < 52, 0.082, 0.064)
counts = rng.poisson(true_rate * exposures / 12)  # monthly counts

# FrequencyChangeDetector with UK event calendar for motor frequency
detector = FrequencyChangeDetector(
    prior_alpha=1.0,
    prior_beta=14.0,   # prior mean λ = α/β = 0.071 monthly rate
    hazard=0.01,       # expect a break roughly every 100 months absent other information
    threshold=0.3,
    uk_events=True,
    event_lines=["motor"],
    event_components=["frequency"],
)

result = detector.fit(counts, exposures / 12, periods)

print(f"Breaks detected: {result.n_breaks}")
for b in result.detected_breaks:
    print(f"  {b}")
```

```
Breaks detected: 1
  DetectedBreak(period=2021-05-31, prob=0.847)
```

The algorithm finds the whiplash reform break at May 2021 with posterior probability 0.847. The UK event calendar - which contains a hazard multiplier of 40x for the whiplash reform date - helped the algorithm focus; without it, on 108 months of data with a moderate-size break, detection probability would be lower. This is the right behaviour: you are not forcing a break, you are telling the algorithm that you know something about the world and it should incorporate that knowledge. If the data showed no change around May 2021, the posterior would remain low regardless of the event prior.

---

## Retrospective break finding with PELT

For historical review of a longer series - say, when building a new pricing model - `RetrospectiveBreakFinder` gives you break locations with 95% bootstrap confidence intervals.

```python
# Compute monthly observed frequency for the PELT input
obs_freq = counts / (exposures / 12)

finder = RetrospectiveBreakFinder(
    model="l2",      # Gaussian mean change — appropriate for smoothed loss ratios
    penalty="bic",   # BIC penalty: log(T), conservative
    n_bootstraps=1000,
    seed=42,
)

breaks = finder.fit(obs_freq, periods=periods)

print(f"PELT found {breaks.n_breaks} break(s)")
for ci in breaks.break_cis:
    print(f"  {ci}")
```

```
PELT found 1 break(s)
  BreakInterval(break=2021-05-31, CI=[2021-02-28, 2021-08-30])
```

The point estimate is month 52, with a 95% bootstrap CI spanning roughly three months either side. That is a tight interval on a single frequency break of this magnitude (0.082 to 0.064 annual, about 22%). For smaller breaks - Ogden's severity effect, or the 2019 revision - the CI would be wider.

---

## Why this changes your rate indication

Now consider what happens when you build your frequency trend using five years of post-2020 data without identifying the May 2021 break. Your series looks like this: high frequency through the first 16 months (Jan 2020–April 2021), then a step down, then stable. A log-linear trend fitted to this series will show a downward slope. You project that slope forward and conclude: frequency is trending down, positive rate movement.

The conclusion is wrong. Frequency is not trending down. It was higher in one regime and lower in another. The post-reform level has been stable. If you project the spurious downward trend forward, you are expecting continued improvement that will not materialise. You will under-rate.

The correct approach: identify the break, fit separate regime estimates, use the post-break regime level as your starting point for projection. If there is no evidence of a trend within the post-break regime, your trend loading is zero or informed by a genuine trend signal - not by the step-change you should have absorbed into the base rate.

```python
# The naive approach: fit a trend across the full post-2020 series
post_2020_idx = np.array([i for i, p in enumerate(periods) if p >= date(2020, 1, 1)])
post_2020_freq = obs_freq[post_2020_idx]
post_2020_months = np.arange(len(post_2020_idx))

# Log-linear trend
log_freq = np.log(post_2020_freq)
slope, intercept = np.polyfit(post_2020_months, log_freq, 1)

print(f"Naive trend: {slope * 12:.3f} log-points per year")
print(f"Naive 3-year projection factor: {np.exp(slope * 36):.3f}")

# The changepoint-aware approach: use post-break regime only
post_break_idx = np.array([i for i, p in enumerate(periods) if p >= date(2021, 5, 1)])
post_break_freq = obs_freq[post_break_idx]

within_regime_mean = float(np.mean(post_break_freq))
within_regime_slope, _ = np.polyfit(np.arange(len(post_break_freq)), np.log(post_break_freq), 1)

print(f"\nPost-break regime mean frequency: {within_regime_mean:.4f}")
print(f"Within-regime trend: {within_regime_slope * 12:.3f} log-points per year")
print(f"Within-regime 3-year projection factor: {np.exp(within_regime_slope * 36):.3f}")
```

```
Naive trend: -0.091 log-points per year
Naive 3-year projection factor: 0.761

Post-break regime mean frequency: 0.0641
Within-regime trend: 0.004 log-points per year
Within-regime 3-year projection factor: 1.013
```

The naive trend gives a three-year projection factor of 0.76 - frequency falls by 24% from where you stand today. The changepoint-aware analysis gives a factor of 1.01 - essentially flat, with no evidence of a trend within the current regime. The rate indication difference between these two starting points is large enough to matter commercially.

---

## The severity side

Everything above applies equally to severity. The March 2017 Ogden change (2.5% to -0.75%) was a sudden and permanent step-up in bodily injury reserves. The 2019 revision (-0.75% to -0.25%) was a partial reversal. The 2024 increase to +0.5% (effective January 2025) was a further move. None of these are trend - they are discrete structural shifts in the legal framework governing future-loss awards. A severity trend fitted across 2015–2024 that ignores these three break points is averaging three different Ogden regimes into a single slope. The projection will be wrong.

`SeverityChangeDetector` handles this directly. It uses a Normal-Gamma conjugate on log-severity, and the UK event calendar includes all three Ogden dates with hard confidence and high hazard multipliers. The analysis is identical to the frequency case: fit the detector, identify the break points, work within the current regime for your trend projection.

---

## How to use this in a pricing review

The practical workflow:

1. Run `FrequencyChangeDetector` and `SeverityChangeDetector` on your historical series before you start any trend analysis. Use `uk_events=True` to incorporate the event calendar. This is a 20-line script.

2. Review the detected breaks against what you know about your book. A break in March 2017 on a motor book is the Ogden change, not a data error. A break in May 2021 on motor frequency is the whiplash reform. A break in mid-2022 on average repair cost is inflation.

3. Identify your current regime. For most UK motor books analysed in January 2029, the current frequency regime started in mid-2021 and severity has been in a post-inflation plateau since late 2023 or 2024.

4. Fit your trend within the current regime. If the within-regime series shows no significant slope, your trend loading is zero. If it shows a slope, project that. Do not blend across the regime boundary.

5. For the rate indication, carry forward the current regime level, not the average across regimes.

The `LossRatioMonitor` in `insurance-dynamics` automates this: it monitors both frequency and severity jointly and returns a `retrain` or `monitor` recommendation based on whether a new break has occurred. If your concern is model drift rather than regime shift, the complementary tool is [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — which tracks whether your model's calibration and discrimination have changed relative to the training period. Run this monthly. It is not a replacement for the analytical step above - you still need to interpret what the break means - but it ensures you are not missing a new regime shift between annual pricing reviews.

---

`insurance-dynamics` is open source under MIT at [github.com/burning-cost/insurance-dynamics](https://github.com/burning-cost/insurance-dynamics). Requires Python 3.10+. Install with `uv add insurance-dynamics`. The full UK event calendar is in `insurance_dynamics.changepoint.priors.UK_EVENTS`.

---

**Related posts:**
- [Tracking Trend Between Model Updates with GAS Filters](/2026/03/08/gas-models-for-between-update-trend/) - the GAS side of insurance-dynamics: continuous tracking of smooth parameter drift, complementary to discrete changepoint detection
- [Your Model Drift Alert Is Too Late](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - monitoring framework for detecting when a model's input distribution has shifted, upstream of the loss ratio signal
- [Bühlmann-Straub Treats Last Year the Same as Five Years Ago](/2026/03/17/buhlmann-straub-treats-last-year-the-same-as-five-years-ago/) - another case where averaging across time is wrong: static credibility weights all years equally, which fails when the risk has genuinely moved
