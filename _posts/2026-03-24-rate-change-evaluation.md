---
layout: post
title: "Did Your Rate Change Actually Work? Causal Evaluation for Insurance Pricing"
date: 2026-03-24
author: Burning Cost
categories: [pricing, causal-inference, tutorials]
tags: [difference-in-differences, interrupted-time-series, rate-change-evaluation, did, its, twfe, parallel-trends, exposure-weighting, causal-inference, insurance-causal, post-hoc-analysis, loss-ratio, conversion-rate, fca-consumer-duty, uk-insurance, statsmodels, callaway-santanna-2021, goodman-bacon-2021, kontopantelis-2015]
description: "After every rate review, pricing teams ask whether it worked. Most compare before/after loss ratios without controlling for anything — a comparison that is biased by seasonal trends, regression to the mean, and concurrent market shifts. Difference-in-Differences and Interrupted Time Series fix this. Here is how to apply them to UK insurance data using the RateChangeEvaluator in insurance-causal."
---

After every rate review, the same question lands in someone's inbox: did it work?

The standard answer is a comparison. You pull the loss ratio for young drivers before the 8% rate increase in January, compare it to the loss ratio after, note it has moved in the right direction, and declare success. The deck goes to the Chief Actuary. Everyone is satisfied.

The problem is that this comparison is almost certainly wrong — not slightly wrong, but structurally wrong in ways that will mislead you consistently and in the same direction. This post explains why, and shows how to do the evaluation properly using Difference-in-Differences and Interrupted Time Series.

---

## Why before/after comparison is biased

A simple before/after comparison of loss ratios answers the question: *what changed?* It does not answer: *what did the rate change cause?* The difference matters enormously in practice.

Four mechanisms will bias a naive before/after comparison in insurance:

**Secular trends.** Claims inflation in UK motor has run at 8-12% per annum since 2022. If you raise rates 6% in April and your loss ratio improves by 3 points by September, some of that is your rate action. Some of it is claims inflation slowing slightly. Some of it is competitors who raised rates more aggressively pulling adverse risks away from your book. You cannot disentangle them with a before/after.

**Seasonality.** UK motor frequency is materially higher in Q4 (darkness, ice, school runs) than Q3. A rate change implemented in July will look more effective than one implemented in October simply because Q4 follows October. If you are comparing the six months after to the six months before, you have introduced a systematic bias.

**Regression to the mean.** Rate changes are not random. They are triggered by adverse loss ratio experience. A cohort with a run of bad luck in 2024 gets a rate increase in January 2025. Their 2025 experience improves. Is that the rate increase? Or is it partly reversion to their true underlying frequency after an unusually bad period? Without a control group, you cannot separate the two.

**Concurrent market changes.** The March 2021 GIPP (General Insurance Pricing Practices) FCA rules, the April 2023 whiplash reforms, the November 2024 Ogden rate change — any of these landing in your evaluation window will confound a before/after comparison. UK insurance markets are not stable in ways that make simple time comparisons credible.

None of these biases are subtle. In a market with 10% annual claims inflation, a segment with genuinely no price elasticity will look like a 10% rate increase "worked" if evaluated naively.

---

## Difference-in-Differences

The fundamental insight of DiD is simple: use a control group to soak up everything that would have changed anyway.

Suppose in January you increase rates 8% on drivers aged 17-25, and leave rates unchanged for drivers aged 26-35. Both groups were previously priced similarly. If claims inflation, seasonality, and regression to the mean affect both groups equally, then the control group (26-35) tells you what would have happened to the treated group (17-25) in the absence of the rate change.

The DiD estimator is:

```
ATT = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
```

In a 2×2 table:

|                  | Before (Oct–Dec 2024) | After (Jan–Mar 2025) | Change |
|------------------|-----------------------|----------------------|--------|
| Young (treated)  | 0.74                  | 0.69                 | -0.05  |
| Older (control)  | 0.68                  | 0.66                 | -0.02  |
| **DiD**          |                       |                      | **-0.03** |

The young drivers' loss ratio fell by 5 points. But the control group fell by 2 points in the same period — picking up the underlying trend, seasonal effects, and any concurrent market changes. Our best estimate of the causal effect of the rate increase is -3 points. The naive before/after would have given us -5 points: 67% too large.

### The parallel trends assumption

DiD rests on one identifying assumption: absent the treatment, the treated group would have followed the same trend as the control group. This is called parallel trends.

In plain English: if we had not raised rates on young drivers, their loss ratio would have moved in the same direction and by the same amount as the older drivers' loss ratio.

This cannot be verified directly — we never observe the counterfactual. But we can test it in the pre-treatment periods. If young and older drivers' loss ratios moved in parallel for the six quarters before January 2025, that is evidence (not proof) the assumption holds.

We test this with an event study: estimate the DiD coefficient separately for each period relative to treatment, using only pre-treatment periods. The pre-treatment coefficients should be statistically indistinguishable from zero. If they are not — if young drivers were already diverging from older drivers before the rate change — then parallel trends fails and the DiD estimate is unreliable.

The choice of control group matters considerably. Young and older drivers on the same product line in the same regions are a reasonable control. Young drivers on motor versus older drivers on home are not — the loss ratio trends for those products will diverge for completely unrelated reasons.

### Exposure weighting is not optional

Insurance outcomes are measured at different levels of statistical reliability depending on how many earned policy years sit behind them. A segment with 50 policies and a 70% loss ratio is almost pure noise. A segment with 50,000 policies and a 70% loss ratio is a reliable estimate. An unweighted DiD treats them identically.

The correct specification is weighted least squares with earned exposure years (or earned premium, for loss ratio outcomes) as weights. For a 2×2 DiD with segment-level data:

```python
import statsmodels.formula.api as smf

model = smf.wls(
    'loss_ratio ~ treated + post + treated_post',
    data=df,
    weights=df['earned_exposure']
).fit(cov_type='HC3')
```

For panel data with multiple segments and periods, you add unit and time fixed effects (two-way fixed effects, TWFE):

```python
formula = 'loss_ratio ~ treated_post + C(segment) + C(period)'
model = smf.wls(formula, data=df, weights=df['earned_exposure']).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['segment']}
)
```

The coefficient on `treated_post` is the ATT — your causal estimate of the rate change effect.

Cluster standard errors at the segment level, not the policy level. The variation that identifies the DiD is at the segment × period level. With fewer than 20 segments, cluster SEs can be unreliable; use HC3 instead and note the limitation.

### Staggered adoption

One common complexity: rate changes are rarely deployed on a single date to a single group. More typically, you increase rates on young drivers in January, on high-value vehicles in March, on London risks in May. Different segments are treated at different times.

Standard TWFE with staggered adoption is biased. Goodman-Bacon (2021, *Journal of Econometrics* 225:254–277) showed that the TWFE coefficient is a weighted average of all 2×2 DiD comparisons in your data — including comparisons that use already-treated units as controls, which is wrong. In samples with many segments and varied treatment timing, the bias can be severe.

If your rate change was staggered, use Callaway & Sant'Anna (2021, *Journal of Econometrics* 225:200–230) instead of standard TWFE. The `insurance-causal` library flags staggered adoption automatically and redirects you to `StaggeredEstimator`.

---

## Interrupted Time Series

DiD requires a control group. Sometimes there is no reasonable control group — the rate change was applied to the entire book, or the entire product line. In that case, use Interrupted Time Series.

ITS asks a different question: does the time series of your outcome change in level or slope at the moment of the rate change, relative to what the pre-treatment trend predicted?

The model is segmented regression:

```
Y_t = β₀ + β₁·t + β₂·D_t + β₃·(t − T)·D_t + ε_t
```

where:
- `t` is a time counter (1, 2, 3, ...)
- `T` is the period of the rate change
- `D_t` = 1 if `t ≥ T` (post-intervention indicator)
- `(t − T)·D_t` is time *since* the intervention, active only in the post-period

The four parameters tell a complete story:
- `β₀`: the baseline level at the start of the series
- `β₁`: the pre-intervention trend — how was the loss ratio moving before you acted?
- `β₂`: the **immediate level shift** at the moment of the rate change — did the loss ratio jump up or down?
- `β₃`: the **change in slope** after the rate change — did the trend accelerate or decelerate?

The counterfactual at period `T + k` (k quarters after the change) is `β₀ + β₁·(T+k)` — the pre-trend extrapolated forward. The observed outcome is that plus `β₂ + β₃·k`. The causal effect grows over time if `β₃ ≠ 0`.

### A common parameterisation error

There is a well-documented mistake in the applied ITS literature (Ewusie et al., 2021, *International Journal of Epidemiology* 50:1011). Analysts sometimes code the post-intervention time variable as `t_post = t − T` for *all* periods after T, including applying it to the pre-period. This is wrong. The slope-change term must be zero in the pre-period — it is an interaction, `(t − T) × D_t`. Getting this wrong produces a model where the pre-period trend and post-period trend are estimated from the same coefficient, which is not what you want.

### Insurance-specific ITS considerations

**Seasonality is a serious problem.** UK motor loss ratios in Q4 are systematically higher than Q3 by around 8-12 percentage points in most portfolios. A rate change implemented in October will appear to fail if you do not control for the Q4 seasonal pattern. Include quarter fixed effects in the ITS regression. Without them, ITS and seasonal patterns are confounded.

**Minimum pre-periods.** ITS estimates the pre-intervention trend, then extrapolates it. With fewer than 8 pre-treatment periods, the trend estimate is unreliable. With fewer than 4, do not run ITS at all — there is not enough data to identify a trend, and you will fit noise. `RateChangeEvaluator` will refuse to run ITS and tell you why.

**Autocorrelation.** Monthly or quarterly loss ratio series are autocorrelated. Standard OLS standard errors will be too small. Use Newey-West (HAC) standard errors with bandwidth `sqrt(T)`.

**Concurrent shocks.** ITS has no control group, so it cannot account for anything else that happened at the same time as your rate change. The library maintains a list of known UK insurance market shocks and warns you if your intervention period falls within two quarters of one: Ogden rate changes, GIPP, whiplash reform, and COVID.

---

## When to use which

The choice is straightforward:

**Use DiD when:**
- Part of your book received the rate change and part did not — any segment-specific or regional rate action
- You can identify a credible control group (similar product, similar distribution channel, similar claim characteristics) where parallel trends is defensible
- You have at least 4 pre-treatment periods for both groups

**Use ITS when:**
- The entire book or product line received the rate change — no control group is possible
- You are doing early exploratory analysis before a full DiD setup
- You need to produce evidence for FCA under Consumer Duty that a whole-book pricing action had the intended effect
- You accept the stronger assumptions and are willing to be explicit about them

ITS is not a second-class method. For whole-book rate changes, it is the right tool. But the assumptions are stronger, and the results should carry more explicit caveats. Present ITS with the pre-trend, the level shift, the slope change, and a list of concurrent events that could confound the interpretation.

---

## Practical considerations

### IBNR and claim development

Loss ratio is the wrong outcome for a rate change evaluation if you are looking at recent periods. Incurred losses on recent cohorts are incomplete — you are comparing early-development incurred to earned premium, and the development pattern will distort your estimate systematically. Young driver claims (high frequency, moderate severity, short tail) develop faster than bodily injury claims. If your rate change is on a short-tail segment, you can work with loss ratios at 12-18 months of development. If you are touching long-tail liability, use claim frequency as the primary outcome and evaluate loss ratio separately at 24+ months.

### Regression to the mean

Rate changes are triggered by adverse experience. The cohort that prompted the review had a run of bad luck. Some of that bad luck was genuine bad risk; some was noise. The noise component will revert regardless of what you do. DiD controls for this if the control group was also adversely experienced (i.e., if the adverse period affected the whole book). If only the treated segment had the bad run, DiD will not fully separate the rate change effect from mean reversion. Be explicit about this limitation when presenting results.

### FCA Consumer Duty evidence

Since July 2023, UK insurers have a duty to demonstrate that pricing produces fair value. The FCA expects firms to monitor the outcomes of pricing changes and produce evidence that they are achieving the intended effect without creating consumer harm. A before/after comparison does not satisfy this — it is not causal evidence. A properly specified DiD or ITS with documented assumptions, tested parallel trends, and explicit uncertainty quantification is the right approach. Keep the methodology documentation alongside the results.

---

## Using RateChangeEvaluator

The `insurance-causal` library implements both methods with insurance-appropriate defaults: exposure weighting, clustered standard errors, seasonal controls for ITS, staggered adoption detection, and the known UK shock list.

```python
import pandas as pd
from insurance_causal import RateChangeEvaluator

# DiD: segment-specific rate change
# df has columns: period, segment, treated (0/1), loss_ratio, earned_exposure
evaluator = RateChangeEvaluator(
    method='auto',          # selects DiD if treated column has 0 and 1
    outcome_col='loss_ratio',
    exposure_col='earned_exposure',
    cluster_col='segment'
)

did_result = evaluator.fit(df, treatment_period='2025-Q1')

print(did_result.att)           # average treatment effect on treated
print(did_result.att_ci)        # 95% confidence interval
did_result.plot_event_study()   # pre-trend test

# ITS: whole-book rate change
# ts_df has columns: period (datetime), loss_ratio, earned_exposure
its_result = evaluator.fit(ts_df, treatment_period='2025-Q1', method='its')

print(its_result.level_shift)   # β₂: immediate effect
print(its_result.slope_change)  # β₃: change in trend
its_result.plot_counterfactual() # observed vs extrapolated pre-trend
```

The `method='auto'` selector checks whether the data has a treated/control split. If it does, it runs DiD. If it does not, it runs ITS and prints a warning about the stronger identification assumptions.

---

## Common mistakes

**Not exposure-weighting.** An unweighted mean loss ratio for a segment with 40 policies gets the same weight as one with 40,000. The estimate will be dominated by small, noisy segments. Always use WLS with earned exposure.

**Standard TWFE with staggered treatment timing.** If your rate changes rolled out across segments over several months, vanilla TWFE is biased. The bias can go in either direction and can be large. Use Callaway-Sant'Anna or flag the issue and report it alongside the TWFE estimate.

**Wrong ITS parameterisation.** The slope-change term is `(t − T) × D_t` — an interaction that is zero in the pre-period. If you code it as `(t − T)` for all t > 1, you are not estimating what you think you are estimating.

**Too few pre-treatment periods.** Both DiD and ITS need pre-treatment data to test or estimate the pre-intervention trend. Six to eight quarters is the minimum for reliable results. With two or three quarters of pre-data, you cannot distinguish "parallel trends holds" from "we did not have enough data to detect a violation."

**Ignoring development in the outcome.** Evaluating loss ratio at 3 months of development is not the same as evaluating it at 12 months. Define your development point before the analysis and stick to it.

**Presenting point estimates without uncertainty.** The ATT from DiD is an estimate with a confidence interval. A point estimate of -3 loss ratio points with a 95% CI of [-8, +2] is a very different result from one with a CI of [-4, -2]. Always report the interval. A result that is statistically consistent with zero is not evidence the rate change worked.

---

The pricing actuary who uses DiD or ITS for post-hoc evaluation will occasionally find that a rate action they believed in had a smaller causal effect than the naive before/after suggested. That is uncomfortable, but it is also the point. The goal of post-hoc evaluation is not to confirm decisions already made — it is to calibrate future models, identify which segments respond to price, and demonstrate to the FCA that pricing outcomes are being properly monitored.

A before/after comparison tells you what happened. DiD and ITS tell you what you caused.

---

*The `RateChangeEvaluator` is part of [insurance-causal](https://github.com/burningcost/insurance-causal). Full API reference and worked examples are in the library documentation.*
