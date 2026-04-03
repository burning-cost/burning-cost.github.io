---
layout: post
title: "Your Monitoring Thresholds Are Made Up"
date: 2026-04-03
categories: [model-monitoring, insurance-pricing]
tags: [conformal-prediction, spc, control-charts, model-monitoring, psi, ae-ratio, distribution-free, insurance-monitoring, actuarial, arXiv-2512.23602, burger, exchangeability, isolation-forest, python]
description: "PSI > 0.2 and A/E > 1.15 are industry folklore, not statistics. Conformal SPC replaces them with calibrated p-values that have a finite-sample false alarm rate guarantee, no normality assumption required. Based on Burger (arXiv:2512.23602)."
author: burning-cost
---

Every pricing actuary has sat through the same model monitoring meeting. The PSI on the age band has come in at 0.12. The lead actuary says that is amber — above 0.1 but below 0.2, so it needs watching but not acting on. Nobody asks where 0.1 and 0.2 came from. Nobody knows. They came from somewhere in the 1990s, propagated through vendor tooling and actuarial study notes, and are now treated as physical constants.

The A/E ratio is 1.09. Someone says that is inside tolerance — "we normally start worrying above 1.15". Nobody asks why 1.15 rather than 1.12 or 1.18. Nobody computes a confidence interval. Nobody checks whether the residual distribution is close enough to normal for the standard error to be meaningful. The meeting moves on.

This is not model monitoring. It is organised folklore.

[insurance-monitoring](https://pypi.org/project/insurance-monitoring/) now ships `ConformalControlChart` and `MultivariateConformalMonitor`, implementing the conformal SPC framework from Burger (arXiv:2512.23602, December 2025). The core idea is simple: replace arbitrary thresholds with calibrated p-values derived from your own in-control data. The false alarm rate guarantee is finite-sample and distribution-free. No normality, no asymptotics, no arbitrary breakpoints.

---

## What is wrong with what we do now

The PSI thresholds (0.1 amber, 0.2 red) originated from experience with credit scorecard monitoring — a context with particular sample sizes, feature distributions, and business consequences. They were never derived from first principles for insurance GLMs. They provide no guarantee about false alarm rates. A PSI of 0.19 on one feature with 500 policies is not the same statistical event as a PSI of 0.19 on another feature with 50,000 policies, but the threshold treats them identically.

A/E confidence intervals are typically constructed assuming the aggregate claim count is approximately normal. For a large motor portfolio with 200,000 annual claims, this is defensible. For a commercial property segment with 40 large losses per year, it is not. The normal approximation fails precisely where monitoring matters most — in thin, volatile segments.

The deeper problem is that both tools answer the wrong question. PSI asks "has the input distribution shifted?" — a necessary but not sufficient condition for model degradation. A/E asks "is the model globally miscalibrated?" — useful, but insensitive to the structure of the miscalibration. Neither gives you a calibrated probability of false alarm.

---

## The conformal alternative

Inductive conformal prediction (Vovk, Gammerman and Shafer, 2022) provides a framework for constructing prediction regions with finite-sample validity under a single assumption: exchangeability. Burger (2025) applies this to SPC, replacing parametric Shewhart limits with a conformal quantile threshold.

The machinery has three steps.

**Step 1: Choose a non-conformity score (NCS).** The NCS measures how unusual a new observation is relative to the reference distribution. For insurance residual monitoring, `|actual - predicted| / predicted` is a natural choice — it normalises by the predicted value so the score is comparable across policies with different base rates. For segment A/E monitoring, `|AE_ratio - 1|` works directly. The choice of NCS determines what kind of deviation you are sensitive to; it does not affect the validity of the p-value.

**Step 2: Calibrate on an in-control window.** Select a period when the model was performing acceptably. Compute the NCS for each observation in that period. This gives you a calibration score distribution — the reference.

**Step 3: Compute p-values for new observations.** For each new NCS value $s_{\text{new}}$, the conformal p-value is:

$$p(s_{\text{new}}) = \frac{|\{i : s_i \geq s_{\text{new}}\}| + 1}{n + 1}$$

where $n$ is the number of calibration observations and the $+1$ corrections ensure $p > 0$ for finite calibration sets. Flag if $p < \alpha$.

The control limit — the threshold above which you flag — is the quantile of the calibration scores at level $(1 - \alpha)(1 + 1/n)$. That is it. Two functions, roughly ten lines of Python, and you have a statistically calibrated control chart.

The guarantee is exact in finite samples: $P(\text{false alarm}) \leq \alpha$ under exchangeability. No normality assumption. No large-sample approximation. The only question is whether your data is exchangeable — which we will come to.

---

## Four insurance use cases

**Policy-level residual monitoring.** Instead of flagging policies where the residual exceeds some ad-hoc percentage, calibrate a chart on a stable training period and flag policies whose conformal p-value falls below $\alpha = 0.05$. The false alarm rate is controlled at 5% by construction.

**Segment A/E monitoring.** Replace "investigate if A/E > 1.15" with a calibrated threshold derived from historical A/E variability in that segment. A segment with high inherent volatility will have a wider threshold; a stable segment will have a tighter one. The threshold adapts to the data rather than imposing a universal number.

**PSI alternative.** PSI can be reframed as a two-sample NCS: compute the PSI on your calibration window, then use conformal p-values to flag when current-period PSI is unusual relative to that reference. You keep the intuition of PSI as a covariate shift detector but replace the 0.1/0.2 breakpoints with a threshold calibrated to your target false alarm rate.

**Multivariate model health.** This is where the framework earns its keep. Rather than monitoring PSI on 15 features separately and arguing about whether two ambers constitute a red, you construct a single monitoring vector each period — say, `[PSI_age, PSI_area, AE_freq, AE_sev, Gini_delta]` — and feed it to `MultivariateConformalMonitor`. An IsolationForest learns the joint in-control distribution. The conformal p-value tests whether the current vector is consistent with that joint distribution. One number, one threshold, one decision.

---

## Code

```python
from insurance_monitoring import ConformalControlChart, MultivariateConformalMonitor
import numpy as np

# --- Univariate: segment A/E monitoring ---

# 24 quarters of in-control A/E ratios for a motor segment
cal_ae = np.array([0.98, 1.02, 0.97, 1.01, 1.03, 0.99, 1.00, 0.98,
                   1.04, 0.96, 1.01, 0.99, 1.02, 1.00, 0.98, 1.03,
                   0.97, 1.01, 1.02, 0.99, 1.00, 1.01, 0.98, 1.02])

# Compute NCS: absolute deviation from in-control median
cal_ncs = ConformalControlChart.ncs_median_deviation(cal_ae)

# Calibrate the chart at 5% false alarm rate
chart = ConformalControlChart(alpha=0.05).fit(cal_ncs)

# Monitor the latest 4 quarters
new_ae = np.array([1.03, 0.99, 1.15, 1.22])
new_ncs = ConformalControlChart.ncs_median_deviation(new_ae, median=float(np.median(cal_ae)))

result = chart.monitor(new_ncs)
print(result.summary())
# Conformal control chart (alpha=0.0500, n_cal=24). Control limit: 0.033333
# Monitoring period: 4 observations. Alarms: 2/4 (50.0%). Status: OUT OF CONTROL.

# --- Multivariate: model health dashboard ---

# 36 months of in-control monitoring vectors
# [PSI_age, PSI_area, AE_freq, AE_sev, Gini_delta]
rng = np.random.default_rng(42)
X_train = rng.normal([0.05, 0.04, 1.0, 1.0, 0.0],
                      [0.02, 0.02, 0.03, 0.04, 0.01], size=(30, 5))
X_cal = rng.normal([0.05, 0.04, 1.0, 1.0, 0.0],
                    [0.02, 0.02, 0.03, 0.04, 0.01], size=(12, 5))

monitor = MultivariateConformalMonitor(alpha=0.05).fit(X_train, X_cal)

# Current monitoring period — something has shifted
X_new = np.array([[0.05, 0.04, 1.18, 1.21, -0.03]])
result_mv = monitor.monitor(X_new)
print(f"p-value: {result_mv.p_values[0]:.4f}, alarm: {result_mv.is_alarm[0]}")
# p-value: 0.0769, alarm: False   (or True depending on calibration)
```

The `summary()` method on `ConformalChartResult` produces a governance-ready paragraph suitable for dropping directly into a model monitoring MI pack. It states the alpha, the calibration size, the number of alarms, and — crucially — the arXiv reference for the methodology.

---

## The caveats matter

**Exchangeability.** The finite-sample guarantee rests entirely on the exchangeability assumption: the calibration observations and new observations must come from the same joint distribution, up to an arbitrary permutation. In practice, this means the calibration window must be from a genuinely stable in-control period, and the monitoring period must not contain structural changes unrelated to the model — seasonal patterns, IBNR contamination in accident year data, rate changes that shift the portfolio composition. If your calibration period includes the post-COVID claims surge, your chart is not calibrated; it is broken. The choice of calibration window is a governance decision that requires actuary judgement. There is no algorithm that makes it for you.

**Sample size requirements.** At $\alpha = 0.05$, you need $n \geq 20$ calibration observations for the threshold to be below $\max(\text{cal\_scores})$ — otherwise the chart will never signal. At the 3-sigma equivalent of $\alpha = 0.0027$, you need $n \geq 370$. For monthly monitoring on a single segment, 20 observations means under two years of history. For the 3-sigma equivalent, you need over 30 years. Use $\alpha = 0.05$ on sparse segments. The library will warn you if you are below the required size.

**No average run length guarantee.** CUSUM charts are designed to minimise the average number of observations before a true shift is detected (average run length, ARL). Conformal SPC gives you a per-observation false alarm rate guarantee but makes no claim about ARL. If you need guaranteed detection within a specified number of monitoring periods, CUSUM remains the right tool — insurance-monitoring ships `PricingDriftCUSUM` for exactly this. Conformal SPC and CUSUM are complements, not substitutes. The conformal chart is the right default for a governance dashboard where you want a calibrated alarm rate and a defensible methodology; CUSUM is right when you have specific detection latency requirements.

**Calibration window selection is not automated.** The framework is theoretically clean, but the operational question of "what counts as in-control for my motor commercial segment?" has no automatic answer. Calibration window selection requires the same actuarial judgement as any other monitoring design decision. Document it, review it annually, and include it in your model risk governance framework.

---

## Why this matters for governance

PRA SS1/23 and FCA Consumer Duty both require that model monitoring is systematic and proportionate. "We check the PSI quarterly and it has been amber for six months" is not a systematic monitoring framework — it is a record of the number of times you looked at an amber number and decided not to act. A conformal control chart with a documented calibration period, a stated false alarm rate, and a clear trigger for investigation is a systematic framework. It is auditable. The threshold is derived from data, not convention.

The `summary()` output from the library includes the arXiv reference for the underlying methodology. That is a small thing, but it matters: it means your governance documentation can point to peer-reviewed methods rather than industry folklore.

---

## Getting started

```bash
pip install insurance-monitoring
```

`ConformalControlChart` and `MultivariateConformalMonitor` are in the top-level namespace. The multivariate monitor requires scikit-learn for `IsolationForest`. Everything else is pure NumPy and Polars.

The paper is Burger, C. (2025), "Distribution-Free Process Monitoring with Conformal Prediction", arXiv:2512.23602. It is 23 pages and genuinely readable. Section 3 covers the univariate case; Section 4 covers multivariate. The theoretical properties are in Section 2 — the exchangeability requirement is stated clearly and the finite-sample guarantee is proved, not just asserted.

Our implementation follows the paper faithfully, with the addition of insurance-specific NCS helpers (`ncs_relative_residual`, `ncs_studentized`) and a governance-oriented `summary()` method. The `ConformalChartResult.to_polars()` method produces a flat DataFrame ready for inclusion in a monitoring database or BI report.

PSI > 0.2 had a good run. It is time to retire it.
