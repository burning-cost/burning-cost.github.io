---
layout: post
title: "Formal Statistical Tests for Insurance Pricing Model Drift"
date: 2026-04-01
categories: [techniques, monitoring]
tags: [model-monitoring, drift, calibration, cusum, murphy-decomposition, gini, deviance, bootstrap, poisson, insurance-monitoring, arXiv-2510-04556, arXiv-2510-25573, Wüthrich, Brauer, Menzel, Franck, python]
description: "insurance-monitoring v0.10.0 adds PricingDriftMonitor (Brauer/Menzel/Wüthrich arXiv:2510.04556) and CalibrationCUSUM (Franck et al. arXiv:2510.25573): formal statistical tests for annual pricing model review and continuous sequential calibration monitoring, replacing ad-hoc dashboard watching with REFIT/RECALIBRATE/OK verdicts and quantified ARL properties."
math: true
author: burning-cost
---

Most pricing monitoring setups we encounter are variations on the same theme: a dashboard, a few loss ratio charts by quarter, a Gini coefficient tracked over time. Someone watches the charts and makes a judgement call. The model gets the green light — until a materially bad year forces a retrospective conversation about whether the signals were there earlier.

The problem is not effort. Teams spend considerable time on monitoring. The problem is the absence of a decision rule. When the Gini drops from 0.42 to 0.39, is that drift, noise, or a new cohort composition? When the pure premium runs 6% over technical price for three consecutive months, is that a calibration failure or sampling variance on a thin book? A chart cannot answer those questions. A statistical test can.

`insurance-monitoring` v0.10.0 ships two new classes that replace watching with testing.

---

## What we mean by formal monitoring

A formal monitoring test is one that:

1. Specifies a null hypothesis: the model is working as intended
2. Produces a p-value or a statistic with known distributional properties
3. Leads to a predetermined decision: refit, recalibrate, or do nothing

The decision rule matters as much as the test statistic. Without it, you are back to a judgement call, now decorated with a number.

For annual model governance — the SR 11/7 (Fed) and PRA SS1/23 (banks) world, or for insurers the Solvency II model governance equivalent, where you are justifying model changes to a validation committee — you need the former. For continuous production monitoring where you want to catch a calibration shift inside the model year, you need the latter. The two new classes cover both.

---

## Annual review: PricingDriftMonitor

`PricingDriftMonitor` implements the two-step monitoring framework from Brauer, Menzel and Wüthrich (arXiv:2510.04556, October 2025). The framework asks two questions in sequence: has the model's ranking ability degraded, and has its calibration shifted?

### The Murphy decomposition

The framework is grounded in the Murphy decomposition of the expected deviance score. For a model with predictions $\hat\mu$ against outcomes $y$ with exposures $v$:

$$S(y, \hat\mu, v) = \text{UNC}(y, v) - \text{DSC}(y, \hat\mu, v) + \text{MCB}(y, \hat\mu, v)$$

where:

- **UNC** is the unconditional uncertainty — a property of the data, not the model
- **DSC** is discrimination: how much the model reduces uncertainty by ranking risks correctly
- **MCB** is miscalibration: the penalty for predicting the wrong level

DSC captures ranking ability. MCB captures whether predictions are on the right level. These are genuinely different things. A Gini coefficient that falls might mean the model is ranking risks worse — or it might mean a new rating factor has emerged that the model never saw. The MCB decomposition separates these.

MCB decomposes further:

$$\text{MCB} = \text{GMCB} + \text{LMCB}$$

**GMCB** (global miscalibration) is the penalty from a systematic level shift: the model is predicting the right relative risks but has the overall level wrong. This is fixable by a balance correction — a multiplicative scaler — without touching the model structure.

**LMCB** (local miscalibration) is the residual: after the best possible global correction, the model still mispredicts specific cohorts. This requires a refit.

The distinction has immediate operational meaning. A GMCB signal says: run an A/E analysis, apply a correction factor, done. An LMCB signal says: go back to the rating algorithm.

### Step 1: Gini ranking drift test

The first test (Algorithm 3 of the paper) checks whether the monitor period Gini is consistent with the reference period Gini's bootstrap distribution. In the reference period, we bootstrap 500 times to estimate $\hat\sigma[G_\text{ref}]$. At monitoring time:

$$z = \frac{G_\text{mon} - \hat\mathbb{E}[G_\text{ref}]}{\hat\sigma[G_\text{ref}]}$$

This is a critical detail. The denominator is the bootstrap SE of the reference distribution, not a one-sample SE of $G_\text{mon}$. We are testing whether the monitor Gini is a plausible draw from the distribution the reference Gini came from.

The default significance level is $\alpha = 0.32$ — the one-sigma rule, as Remark 3 of the paper calls it. At $\alpha = 0.32$ we catch roughly one-sigma deviations. This is intentional: the test is an early warning system, not a strict hypothesis test. A Gini drift at $p = 0.25$ is worth flagging. Use $\alpha = 0.05$ if you want a more conservative governance threshold.

### Step 2: Calibration tests

Both GMCB and LMCB are tested against their null distributions via Algorithm 1 bootstrap. Under $H_0$ (the model is correctly specified), we simulate $B = 500$ datasets from the assumed distribution with means $\hat\mu_i$, compute the Murphy component in each simulation, and use the fraction of simulated values exceeding the observed value as the p-value.

For Poisson: $Y_i^{(b)} \sim \text{Poisson}(\hat\mu_i \cdot v_i)$, then $y^{(b)} = Y^{(b)} / v$. This is the pure parametric bootstrap under the null model.

### Decision logic

The three tests produce a verdict:

| Gini rejected | LMCB rejected | GMCB rejected | Verdict |
|---|---|---|---|
| No | No | No | OK |
| No | No | Yes | RECALIBRATE |
| Yes | No | Any | REFIT |
| Any | Yes | Any | REFIT |

REFIT if rank structure or local cohort calibration has degraded. RECALIBRATE if only the global level has shifted. OK otherwise.

### API

```python
import numpy as np
from insurance_monitoring.pricing_drift import PricingDriftMonitor

# Reference period: last 12 months of in-force exposure
monitor = PricingDriftMonitor(
    distribution='poisson',
    n_bootstrap=500,
    alpha_gini=0.32,   # one-sigma early warning
    random_state=42,
)
monitor.fit(y_ref, mu_hat_ref, exposure=exposure_ref)

# Monitor period: current year
result = monitor.test(y_mon, mu_hat_mon, exposure=exposure_mon)
print(result.verdict)   # 'OK', 'RECALIBRATE', or 'REFIT'
print(result.summary()) # governance-ready paragraph
```

`result.summary()` produces a paragraph suitable for inclusion in a model validation report: Gini test statistic and p-value, Murphy component breakdown with percentage contributions, calibration test results, and the verdict with interpretation. No further formatting needed for a committee pack.

The full result structure exposes everything:

```python
result.gini.z_stat         # z-statistic for the Gini test
result.gini.p_value        # two-tailed p-value
result.murphy.dsc_pct      # discrimination as % of total deviance
result.murphy.mcb_pct      # miscalibration as % of total deviance
result.global_calib.p_value  # GMCB bootstrap p-value
result.local_calib.p_value   # LMCB bootstrap p-value
result.to_dict()           # flat dict for Delta table / JSON logging
```

For a Tweedie severity model, swap `distribution='gamma'` or `distribution='tweedie'` with appropriate `tweedie_power`. The deviance scoring function and bootstrap simulation adapt accordingly.

### What the decomposition looks like in practice

Suppose a UK motor frequency model trained on 2023 data is monitored against 2024 H1. A typical healthy result might show DSC at 61% of total deviance, MCB at 3%. A result worth acting on might show MCB at 14%, of which GMCB is 11% (global frequency inflation, fixable by multiplicative correction) and LMCB is 3% (not significant). That is a RECALIBRATE verdict: apply a correction factor, document the reason (perhaps a short-term claims frequency spike), no refit required.

A REFIT verdict would look like: Gini drops from 0.44 to 0.37 ($z = -2.1$, $p = 0.04$ at $\alpha = 0.32$), LMCB significant ($p = 0.02$). The model is ranking risks differently than it was fitted to. Something structural has changed in the risk pool — new usage patterns, changed population, a macroeconomic shift — and the model needs new training data.

---

## Continuous monitoring: CalibrationCUSUM

Annual reviews are necessary but not sufficient. A model that is miscalibrating from month three of a twelve-month review period carries that error for nine months before it surfaces. For books where calibration errors compound — commercial lines, specialty, anything with long development tails — that is a real problem.

`CalibrationCUSUM` implements the sequential monitoring approach from Franck, Driscoll, Szajnfarber and Woodall (arXiv:2510.25573, 2025). The idea is a CUSUM chart with dynamic control limits that maintain a constant conditional false alarm rate (CFAR) at each time step.

### The statistic

The CUSUM accumulates log-likelihood ratios comparing the calibrated null against a specified alternative. At time step $t$ with $n_t$ new observations:

$$W_t = \log \frac{f_\text{uncalibrated}(y_t; \delta_a, \gamma_a)}{f_\text{calibrated}(y_t)}$$

$$S_t = \max(0, S_{t-1} + W_t), \quad S_0 = 0$$

For binary claim indicators (Bernoulli mode), the alternative uses the linear log-odds (LLO) recalibration function:

$$g(x; \delta, \gamma) = \frac{\delta x^\gamma}{\delta x^\gamma + (1-x)^\gamma}$$

At $(\delta=1, \gamma=1)$ this is the identity — perfectly calibrated. $\delta > 1$ is an upward shift; $\gamma < 1$ is a scale compression (predictions too extreme relative to what is realised).

For claim count data (Poisson mode, our insurance adaptation), the LLR simplifies:

$$W_t = \sum_i \left[ y_i \log \delta_a - (\delta_a - 1) \hat\mu_i v_i \right]$$

where $\hat\mu_i$ is the model's rate prediction and $v_i$ is exposure. The alternative is $\delta_a \cdot \hat\mu_i$: a multiplicative rate shift.

### Dynamic probability control limits

Standard CUSUM charts use a fixed control limit $h$, chosen to give a target average run length (ARL) in the in-control state. That works when every time step has the same number of observations. In insurance, it does not: monthly new business volumes vary, policies on risk differ, renewal cycles create seasonality.

The DPCLs from Franck et al. fix this. At each time step, the control limit $h_t$ is the $(1 - \text{CFAR})$ quantile of the CUSUM statistic under the calibrated null, simulated from the actual predictions arriving at that step. The Monte Carlo pool of 5,000 paths is maintained across time steps, with alarmed paths replaced by resampled non-alarmed paths — preserving the conditional structure.

This gives a constant false alarm rate per time step by construction, regardless of volume variation. At `cfar=0.005`, in-control ARL0 = 200. Out-of-control ARL1 at $\delta_a = 2$ (event rate doubled) is approximately 37, from Table 1 of the paper.

### API

```python
from insurance_monitoring.cusum import CalibrationCUSUM

# Monthly frequency monitoring, Poisson mode
monitor = CalibrationCUSUM(
    delta_a=1.3,          # detect 30% rate inflation
    distribution='poisson',
    cfar=0.005,           # ARL0 = 200
    n_mc=5000,
    random_state=42,
)

# Called once per month as data arrives
alarm = monitor.update(mu_hat_month, y_claims_month, exposure=car_years_month)

if alarm:
    print(f"Alarm at t={alarm.time}, S_t={alarm.statistic:.3f}")
    print(f"Control limit: {alarm.control_limit:.3f}")
    print(f"W_t (LLR): {alarm.log_likelihood_ratio:.3f}")
```

For binary claim indicator data (cyber, critical illness, guarantee triggers):

```python
# Bernoulli mode — LLO alternative
monitor = CalibrationCUSUM(
    delta_a=2.0,    # shift-up alternative: event rate doubled
    gamma_a=1.0,    # pure shift, no scale change
    distribution='bernoulli',
    cfar=0.005,
)

alarm = monitor.update(p_model, y_binary)
```

After a genuine miscalibration event — say, a rate correction is applied mid-year — call `monitor.reset()` to restart the statistic from zero. The alarm history (`monitor.summary()`) is preserved, so you retain a record of all events across the monitoring lifecycle.

The `.plot()` method produces a chart of $S_t$ against the dynamic control limits $h_t$, with alarm times marked. This is the natural artefact for a model monitoring report: a visual record of the CUSUM trajectory and the points at which formal signals occurred.

### Choosing delta_a

The choice of $\delta_a$ is a design decision. It specifies what you are monitoring for. If you want early detection of a 30% rate increase, set `delta_a=1.3`. If you are more worried about large-scale miscalibration and can afford later detection, set `delta_a=2.0`.

For UK motor frequency, a 20–30% rate movement inside a year would be material for most books. We would suggest starting with `delta_a=1.25` and `cfar=0.005`, which gives ARL0=200 (roughly a false alarm every 17 years of monthly monitoring) and detects a sustained 25% rate shift within a few months.

---

## Annual and sequential monitoring together

These two classes address different questions across different time horizons. `PricingDriftMonitor` answers: at the annual review, does the model need a refit, a recalibration, or neither? `CalibrationCUSUM` answers: in the months since the last annual review, has a calibration shift already appeared?

A sensible combined workflow:

1. At year-end, run `PricingDriftMonitor.test()` on the full year's data. Use `result.summary()` in the governance pack.
2. Throughout the year, run `CalibrationCUSUM.update()` monthly. Any alarm triggers an intra-year review without waiting for the annual cycle.
3. If the CUSUM alarms, apply `monitor.reset()` after the intervention and document the event. The annual `PricingDriftMonitor` run at year-end will still tell you whether a more fundamental refit is warranted.

The two approaches are formally independent. The CUSUM monitors calibration drift sequentially. The annual monitor tests discrimination and calibration jointly, with a structured verdict for governance. Running both gives you intra-year detection with documented ARL properties, plus an annual formal verdict tied to a decision rule.

---

## Getting the library

```
uv add insurance-monitoring
```

Version 0.10.0. Both classes are available from the package top level:

```python
from insurance_monitoring.pricing_drift import PricingDriftMonitor
from insurance_monitoring.cusum import CalibrationCUSUM
```

755 tests pass across the full library. The `PricingDriftMonitor` tests cover the Gini z-test denominator (reference bootstrap SE, not monitor SE), the GMCB/LMCB bootstrap p-values under both calibrated and shifted simulations, and the three-branch decision logic. The `CalibrationCUSUM` tests cover CFAR targeting, alarm reset behaviour, pool resampling under near-exhaustion conditions, and Poisson/Bernoulli mode parity.

---

## The papers

Brauer, Jan, Benjamin Menzel, and Mario V. Wüthrich. "Monitoring Insurance Pricing Models." arXiv:2510.04556, October 2025. The source for the Murphy decomposition framework, the GMCB/LMCB split, and Algorithms 1–4.

Franck, Carter T., Shawn J. Driscoll, Zoe Szajnfarber, and William H. Woodall. "A Calibration Monitoring Approach for Probability Predictions." arXiv:2510.25573, 2025. The source for the DPCL CUSUM, LLO alternative hypothesis, and the ARL simulation results in Table 1.

---

## Related posts

- [Model Validation Under PRA SS1/23](/2025/07/13/model-validation-pra-ss123/) — the governance context for annual model review requirements
- [Your GLM Confidence Intervals Are Wrong After Lasso](/2026/04/01/penalized-glm-inference-bias-corrected-confidence-intervals-lasso/) — formal inference for model fitting, complementing formal inference for model monitoring
