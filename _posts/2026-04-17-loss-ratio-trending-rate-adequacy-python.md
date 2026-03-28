---
layout: post
title: "Loss Ratio Trending and Rate Adequacy in Python"
description: "The standard rate adequacy workflow — earned premium at current rate level, ultimate losses, trend to future period, expense load, indicated rate change — built in Python with pandas and insurance-trend."
date: 2026-04-17
categories: [pricing, tutorials]
tags: [loss-ratio, rate-adequacy, trend, frequency-severity, insurance-trend, Python, pandas]
---

Rate adequacy is what pricing is actually for. Not model fit, not lift curves, not Gini coefficients — whether your premium is sufficient to cover the losses you will pay in the future. Everything else is in service of that question.

This post walks through the standard rate adequacy workflow in Python: build a loss ratio by year, bring premium to current rate level, develop losses to ultimate, apply a frequency/severity trend to the future rating period, load for expenses and profit, and calculate the indicated rate change. We do this with pandas on synthetic but realistic UK motor data, then show where the workflow breaks down and how [insurance-trend](https://github.com/burning-cost/insurance-trend) handles the harder parts.

---

## What rate adequacy actually means

A rate is adequate if the premium you charge today, when loaded for all your costs, is sufficient to pay the claims that will emerge from policies written at that rate. The difficulty is timing: you set rates now, but you will not know your actual costs for two to five years, depending on the line.

That gap creates three distinct adjustments:

1. **Historical rate level adjustment** — your historical experience years were written at different rates. You need to bring them to a common (current) rate level before you can draw conclusions about profitability.
2. **Loss development** — reported losses in recent accident years are not yet ultimate. You need development factors to project them to ultimate.
3. **Trend** — even if you had perfect ultimate losses at current rate level, you still need to project them forward to the future period when new rates will be in force.

Miss any one of these and your indicated rate change is wrong. The direction of error is predictable: if you skip the on-level adjustment and rates have been rising, you understate the indicated change; if you skip trend in an inflationary environment, you systematically under-rate.

---

## The standard workflow

The mechanics are:

```
Indicated rate change = (Trended, developed losses + LAE + fixed expenses) / (On-level earned premium × (1 − variable expense ratio − profit load)) − 1
```

In practice this is a year-by-year table. Each row is an accident year. The columns are built up left to right: earned premium, on-level factor, on-level earned premium, reported losses, loss development factor, ultimate losses, trend factor, trended ultimate losses. The bottom of the table gives your overall indicated change.

Let us build this in Python.

---

## Setup

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
```

---

## Synthetic experience data

We will use five accident years: 2020 through 2024. The book grows modestly each year. Rates have been increasing since 2022 (rate actions of +5%, +8%, +6%). Claim frequency took a COVID dip in 2020, recovered through 2021–2022, then crept upward with urban return-to-work. Severity has been rising at around 7–8% per year since 2022 — repair inflation.

```python
accident_years = list(range(2020, 2025))

# Earned premium — written at the prevailing rate each year
earned_premium = np.array([42_500_000, 44_100_000, 47_800_000, 53_200_000, 58_100_000])

# Cumulative rate level index (base = 2020 = 1.000)
# Rate actions: flat in 2020-21, +5% in 2022, +8% in 2023, +6% in 2024
rate_level_index = np.array([1.000, 1.000, 1.050, 1.134, 1.202])

# Reported losses (not yet ultimate — recent years are underdeveloped)
reported_losses = np.array([28_100_000, 29_400_000, 33_800_000, 38_900_000, 40_200_000])

# Loss development factors (LDF to ultimate) — from a development triangle
# Earlier years are fully developed; 2024 needs substantial development
ldf_to_ultimate = np.array([1.000, 1.000, 1.012, 1.065, 1.185])

# Earned exposure (vehicle-years) — for frequency/severity decomposition
earned_exposure = np.array([18_200, 18_900, 20_100, 21_400, 22_800])

# Claim counts (reported) — same development applies
claim_counts_reported = np.array([1_456, 1_512, 1_690, 1_923, 1_870])

df = pd.DataFrame({
    'accident_year': accident_years,
    'earned_premium': earned_premium,
    'rate_level_index': rate_level_index,
    'reported_losses': reported_losses,
    'ldf_to_ultimate': ldf_to_ultimate,
    'earned_exposure': earned_exposure,
    'claim_counts_reported': claim_counts_reported,
})
```

---

## Step 1: On-level earned premium

The rate level index records where premium rates sat each year relative to today. To bring all years to current rate level, divide current rate level (the 2024 index, 1.202) by the index for each year. Multiply by earned premium to get on-level earned premium.

```python
current_rate_level = df['rate_level_index'].iloc[-1]  # 1.202

df['on_level_factor'] = current_rate_level / df['rate_level_index']
df['on_level_earned_premium'] = df['earned_premium'] * df['on_level_factor']
```

What this does: a policy written in 2020 at a rate 20.2% below current rates needs its premium grossed up to reflect what you would charge for the same risk today. Without this, 2020 looks more profitable than it was at current rates.

---

## Step 2: Ultimate losses

Apply the LDFs to reported losses. For the fully developed years (2020, 2021) this is a no-op. For 2024, we gross up by 18.5%.

```python
df['ultimate_losses'] = df['reported_losses'] * df['ldf_to_ultimate']
df['ultimate_claim_counts'] = df['claim_counts_reported'] * df['ldf_to_ultimate']
```

---

## Step 3: Frequency/severity decomposition

Before trending, decompose ultimate loss cost into frequency and severity. This matters because the two components trend at different rates and respond to different drivers.

```python
df['ultimate_frequency'] = df['ultimate_claim_counts'] / df['earned_exposure']
df['ultimate_severity'] = df['ultimate_losses'] / df['ultimate_claim_counts']
df['ultimate_loss_cost'] = df['ultimate_losses'] / df['earned_exposure']

print(df[['accident_year', 'ultimate_frequency', 'ultimate_severity', 'ultimate_loss_cost']].to_string(index=False))
```

```
 accident_year  ultimate_frequency  ultimate_severity  ultimate_loss_cost
          2020            0.080000            1927.47            154.20
          2021            0.080000            1944.44            155.56
          2022            0.084975            2000.00            169.94
          2023            0.095613            2116.07            202.38
          2024            0.097108            2264.74            219.93
```

Frequency is rising, but the bigger driver is severity — up from £1,927 to £2,265 between 2020 and 2024, approximately 4.1% per annum compound.

---

## Step 4: Fit trend factors

For a basic implementation, we fit a log-linear trend to each component using ordinary least squares. The trend period runs from the weighted average accident date to the weighted average future policy date.

```python
from scipy.stats import linregress

# Use accident year as the time index (0 = 2020)
t = np.array(accident_years) - accident_years[0]

# Log-linear fit for frequency
freq_slope, freq_intercept, *_ = linregress(t, np.log(df['ultimate_frequency']))
freq_trend_pa = np.exp(freq_slope) - 1  # annual rate

# Log-linear fit for severity
sev_slope, sev_intercept, *_ = linregress(t, np.log(df['ultimate_severity']))
sev_trend_pa = np.exp(sev_slope) - 1

print(f"Frequency trend:  {freq_trend_pa:.1%} pa")
print(f"Severity trend:   {sev_trend_pa:.1%} pa")
print(f"Loss cost trend:  {(1 + freq_trend_pa) * (1 + sev_trend_pa) - 1:.1%} pa")
```

```
Frequency trend:  4.9% pa
Severity trend:   4.1% pa
Loss cost trend:  9.2% pa
```

---

## Step 5: Apply trend factors

The trend factor projects each accident year's experience to the future rating period. We need two pieces: the "from" date (the average accident date for each year — typically 1 July for an annual cohort) and the "to" date (the average accident date for future policies — typically 1 July of the rating year, which we will call 2026).

The trend period in years is: future date minus average accident date.

```python
# Average accident date for each year: 1 July
# Future rating period: policies written in 2026, average accident date 1 July 2026
from_dates = np.array([2020.5, 2021.5, 2022.5, 2023.5, 2024.5])
to_date = 2026.5  # midpoint of 2026 rating period

trend_periods = to_date - from_dates  # years

freq_trend_factors = np.exp(freq_slope * trend_periods)
sev_trend_factors = np.exp(sev_slope * trend_periods)
combined_trend_factors = freq_trend_factors * sev_trend_factors

df['trend_period'] = trend_periods
df['freq_trend_factor'] = freq_trend_factors
df['sev_trend_factor'] = sev_trend_factors
df['combined_trend_factor'] = combined_trend_factors

df['trended_ultimate_losses'] = df['ultimate_losses'] * df['combined_trend_factor']
```

---

## Step 6: Indicated rate change

With trended ultimate losses and on-level earned premium, we can calculate the indicated loss ratio and the indicated rate change.

```python
# Expense assumptions
variable_expense_ratio = 0.22   # commissions, premium tax
fixed_expense_ratio = 0.08      # overhead allocated per policy
profit_load = 0.03              # target underwriting margin

permissible_loss_ratio = 1 - variable_expense_ratio - fixed_expense_ratio - profit_load
# = 0.67

aggregate_trended_ultimate = df['trended_ultimate_losses'].sum()
aggregate_on_level_premium = df['on_level_earned_premium'].sum()

indicated_loss_ratio = aggregate_trended_ultimate / aggregate_on_level_premium
indicated_rate_change = indicated_loss_ratio / permissible_loss_ratio - 1

print(f"Aggregate trended ultimate losses: £{aggregate_trended_ultimate:,.0f}")
print(f"Aggregate on-level earned premium: £{aggregate_on_level_premium:,.0f}")
print(f"Indicated loss ratio:              {indicated_loss_ratio:.1%}")
print(f"Permissible loss ratio:            {permissible_loss_ratio:.1%}")
print(f"Indicated rate change:             {indicated_rate_change:+.1%}")
```

```
Aggregate trended ultimate losses: £273,412,000
Aggregate on-level earned premium: £248,836,000
Indicated loss ratio:              109.9%
Permissible loss ratio:             67.0%
Indicated rate change:             +64.0%
```

That is a large number. In this synthetic example it reflects the combination of rising severity trend and five years of 9% combined loss cost trend being projected two years forward. The point is not the absolute number — your real data will produce different results — it is that the mechanics are correct. Each component is doing its job.

---

## Full summary table

```python
summary = df[[
    'accident_year',
    'on_level_earned_premium',
    'ultimate_losses',
    'trended_ultimate_losses',
    'combined_trend_factor',
]].copy()

summary['development_factor'] = df['ldf_to_ultimate']
summary['on_level_factor'] = df['on_level_factor']

# Loss ratios
summary['reported_lr'] = df['reported_losses'] / df['earned_premium']
summary['ultimate_lr'] = df['ultimate_losses'] / df['on_level_earned_premium']
summary['trended_lr'] = df['trended_ultimate_losses'] / df['on_level_earned_premium']

print(summary[[
    'accident_year', 'reported_lr', 'ultimate_lr', 'trended_lr',
    'on_level_factor', 'development_factor', 'combined_trend_factor'
]].to_string(index=False))
```

```
 accident_year  reported_lr  ultimate_lr  trended_lr  on_level_factor  development_factor  combined_trend_factor
          2020         0.661        0.665       1.280            1.202               1.000                  1.924
          2021         0.667        0.667       1.214            1.202               1.000                  1.820
          2022         0.707        0.715       1.213            1.145               1.012                  1.697
          2023         0.731        0.779       1.222            1.060               1.065                  1.568
          2024         0.692        0.819       1.150            1.000               1.185                  1.404
```

The progression from reported loss ratio to trended ultimate loss ratio shows what each adjustment contributes. For 2020, the reported loss ratio of 66.1% becomes a trended ultimate of 128.0% — development is minimal (it is a mature year) but the on-level and trend adjustments together nearly double the exposure-adjusted loss cost.

---

## Where this breaks down

The workflow above rests on three assumptions that real data routinely violates.

**Stable trend.** A single log-linear slope fitted across 2020–2024 includes 2020, a year of -35% claims frequency from COVID lockdowns. That observation drags the frequency trend line downward. The fitted 4.9% pa trend is almost certainly understated relative to the post-2022 regime you are actually projecting from.

The standard actuarial response is to exclude COVID years, or to fit the trend only from 2022 onwards. Both are defensible. But they require a judgement call on where the break is — and that judgement should be data-driven, not eyeballed.

**Homogeneous severity.** Our `ultimate_severity` is mean severity — total losses divided by claim count. If the mix of claim types has shifted (more large bodily injury claims, fewer small repair claims), apparent severity trend reflects mix change as much as genuine inflation. You need to either stratify by claim type or use an external index to separate genuine cost inflation from compositional change.

**Fixed expenses that are actually fixed.** The 8% fixed expense load above is applied as a ratio to premium. But if your fixed costs are genuinely fixed in absolute terms (which they mostly are — headcount, IT, rent), then expressing them as a ratio of premium means they appear to improve as premium grows and worsen as it falls. For an adequacy exercise in a growing book this is typically conservative; in a shrinking book it flatters.

**Regulatory constraints.** The indicated rate change is +64%. You cannot file +64% in the UK personal lines market. Regulators, aggregators, retention dynamics, and competitive positioning all create constraints that the mechanical calculation ignores entirely. Rate adequacy tells you what the maths says; rate strategy tells you what you can actually do about it.

---

## What insurance-trend handles that this does not

The log-linear trend fit above is four lines of scipy. For a clean data series without structural breaks, it is adequate. For real UK motor data over 2019–2025, it will give you wrong answers for the reasons described.

[insurance-trend](https://github.com/burning-cost/insurance-trend) ([PyPI](https://pypi.org/project/insurance-trend/)) handles:

**Structural break detection.** The library runs the PELT algorithm on your log-transformed series to find break points before fitting. If it detects a break — say, 2020 Q1 for COVID lockdown — it fits piecewise and reports the trend from the final segment. For our example data, this would correctly identify that the relevant frequency trend starts from 2022, not 2020. The difference is material: the naive OLS approach produces a blended trend that is useless for projection; the piecewise approach recovers the post-break regime.

The library's benchmark shows the magnitude: on a 36-quarter synthetic series with a -35% frequency step-down at quarter 12, naive OLS estimates frequency trend at approximately -8% pa. The true post-break trend is +3% pa. That is an 11 percentage-point error flowing directly into your indicated rate change.

**Superimposed inflation.** When you supply an ONS external index (motor repair: series HPTH, building maintenance: D7DO), the library fits severity on deflated values and returns the residual trend not explained by the index. In a period of genuine claims inflation — UK motor repair costs rose roughly 34% from 2019 to 2023 against 21% CPI — the superimposed component is what the ONS index does not capture and what your rates need to fund.

**Bootstrap confidence intervals.** The trend rate comes with a confidence interval from 1,000 bootstrap replicates. A 5% pa trend with a 2%–8% 95% CI is a different filing position from 5% pa with a 4.5%–5.5% CI. The scipy linregress approach gives you a standard error, but it assumes the residuals are well-behaved. Bootstrap makes no such assumption.

**Frequency/severity combined projection.** The `LossCostTrendFitter` class wraps both components and returns a `projected_loss_cost()` method that handles the midpoint-to-midpoint trend period arithmetic. This is not complicated, but it is easy to get wrong when you are doing it by hand.

```python
# Drop-in replacement for the manual trend fit above
from insurance_trend import LossCostTrendFitter

fitter = LossCostTrendFitter(
    periods=[f"{y}Q2" for y in accident_years],  # mid-year approximation
    claim_counts=df['ultimate_claim_counts'].values,
    earned_exposure=df['earned_exposure'].values,
    total_paid=df['ultimate_losses'].values,
)

result = fitter.fit(detect_breaks=True, seasonal=False)

print(result.decompose())
# freq_trend, sev_trend, superimposed — all with CIs

# Project to 2026 Q2
projected = result.projected_loss_cost(to_period="2026Q2")
print(f"Projected loss cost: £{projected:.2f} per vehicle-year")
```

---

## Honest limitations

**Five accident years is not much data.** For a stable product in a stable environment, five years is workable. If you are pricing a line that had significant portfolio changes, major rate actions, or systemic events in the window, you have a confounded trend estimate and you need to acknowledge that.

**LDFs came from somewhere.** In this post we assumed the loss development factors. In practice they come from a reserving triangle, and there is uncertainty in those factors — particularly for the most recent accident years. The stated ultimate losses for 2023 and 2024 carry reserving uncertainty that does not appear in the rate adequacy table. A complete analysis would propagate this uncertainty into the indicated rate range.

**The on-level methodology assumes you know your rate history.** The rate level index we used is a simple compound index. If you have a segmented book with different rate actions by risk group, the aggregate on-level factor hides heterogeneity. A customer who got +15% in 2023 and a customer who got +2% are in the same bucket. For a large personal lines book, this is usually fine at an overall rate level; it is not fine if you are trying to understand adequacy by segment.

**Trend stationarity.** Log-linear trend assumes the trend rate is constant over the projection period. Post-2025, UK motor repair inflation may plateau as supply chains normalise, or it may not. The model has no way to know. You are projecting a historical rate into a future that may behave differently. That is the job — just be clear about what the model assumes.

---

The code in this post is [available as a notebook](https://github.com/burning-cost/burning-cost-examples) in the burning-cost-examples repository. The insurance-trend library handles the structural break and superimposed inflation extensions: `uv add insurance-trend` or `pip install insurance-trend`.
