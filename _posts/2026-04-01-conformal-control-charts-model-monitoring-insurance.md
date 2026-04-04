---
layout: post
title: "Conformal Control Charts: Replacing PSI Thresholds With a Proper False Alarm Rate"
date: 2026-04-01
categories: [monitoring, model-risk, libraries]
tags: [conformal-prediction, SPC, control-charts, PSI, model-monitoring, false-alarm-rate, distribution-free, IsolationForest, OC-SVM, insurance-monitoring, arXiv-2512-23602, Burger, ConformalControlChart, MultivariateConformalMonitor, AE-monitoring, GLM-monitoring, model-risk, PRA-SS3-17, python]
description: "PSI thresholds of 0.1 and 0.2 are industry convention, not statistical calibration. Burger (arXiv:2512.23602, Dec 2025) replaces them with conformal p-values, giving a distribution-free false alarm rate guarantee that classical SPC charts cannot provide. We have shipped this in insurance-monitoring v0.11.0."
math: true
author: burning-cost
---

Most model monitoring pipelines in UK insurance pricing carry the same PSI thresholds they have always carried: 0.1 is yellow, 0.2 is red. Ask where those numbers came from and the answer is usually "they are industry convention" or "they are what the previous team used." They are not derived from your data. They do not correspond to a specific false alarm rate. They were not calibrated on your calibration period. They are, in the technical sense, arbitrary.

This matters more than it might appear. If your PSI threshold is too tight, you generate false alarms every quarter, the monitoring team habituates to them, and a genuine shift goes unnoticed in the noise. If it is too loose, real distributional change passes without alert. You do not know which of these you are in unless you can characterise the false alarm rate — which you cannot do with conventional PSI because the threshold has no distributional justification.

Christopher Burger ([arXiv:2512.23602](https://arxiv.org/abs/2512.23602), University of Mississippi, December 2025) provides the fix. His paper applies conformal prediction to statistical process control, replacing parametric Shewhart limits and arbitrary threshold conventions with a control limit that has a proven distribution-free false alarm rate guarantee. We have implemented both his univariate and multivariate constructions in `insurance-monitoring` v0.11.0.

```bash
uv add "insurance-monitoring>=0.11.0"
```

---

## What is wrong with PSI thresholds

PSI (Population Stability Index) is the right diagnostic concept. Comparing the distribution of a feature or model score between a reference period and a monitoring period is exactly what you should do. The problem is the thresholds.

The PSI statistic is:

$$\text{PSI} = \sum_{k=1}^K \left(p_k^{\text{ref}} - p_k^{\text{mon}}\right) \log\frac{p_k^{\text{ref}}}{p_k^{\text{mon}}}$$

This is an asymmetric KL-type divergence over binned distributions. It has no known distribution under the null hypothesis (reference and monitoring periods drawn from the same distribution) except under strong parametric assumptions that are never satisfied for insurance features. The thresholds of 0.1 and 0.2 were set empirically in the 1950s credit-scoring literature and transferred to insurance without rederivation.

Burger's approach replaces the question "is PSI above 0.2?" with "what is the probability that a stable model would produce this anomaly score?" The latter question has a clean answer: the conformal p-value. And importantly, the false alarm rate is controlled exactly — not asymptotically, not under normality, but in finite samples under the single assumption that your calibration and monitoring data are exchangeable (drawn from the same stable process).

---

## The conformal p-value

The construction is simpler than the name suggests. You have a calibration set of $n$ observations from an in-control period. For each observation you compute a non-conformity score (NCS) — a measure of how anomalous that observation is. For model monitoring, the natural NCS is the absolute residual: $s_i = |y_i - \hat{y}(x_i)|$.

At monitoring time, you compute the NCS for the new observation $s_{\text{new}}$ and ask: how often did the calibration set produce a score at least as large?

$$p(s_{\text{new}}) = \frac{\#\{i : s_i \geq s_{\text{new}}\} + 1}{n + 1}$$

The $+1$ in numerator and denominator is the finite-sample correction from inductive conformal prediction theory. It ensures $p > 0$ always — you cannot reject the null with certainty from a finite calibration set.

If you choose a significance level $\alpha$, you flag the observation as out-of-control when $p < \alpha$. The false alarm rate — the probability of flagging an in-control observation — is exactly $\alpha$ under exchangeability. Not approximately $\alpha$. Exactly $\alpha$, in finite samples. This is the guarantee that PSI thresholds cannot provide.

For a Shewhart-style chart (a threshold on the NCS rather than a p-value), the corresponding conformal threshold is:

$$q = \text{quantile}\left(S_{\text{cal}},\; \min\left[(1-\alpha)\left(1 + \frac{1}{n}\right),\; 1\right]\right)$$

This is the $(1-\alpha)(1+1/n)$-th quantile of the calibration scores, which is equivalent to the $\lceil(1-\alpha)(n+1)\rceil$-th order statistic. At $\alpha = 0.05$ (the most common insurance monitoring significance level), you need $n \geq 20$ calibration observations for the threshold to be well-defined. At $\alpha = 0.0027$ (the 3-sigma equivalent), you need $n \geq 370$.

---

## Univariate monitoring with ConformalControlChart

The `ConformalControlChart` class handles univariate monitoring: individual residuals, segment A/E ratios, feature median deviations, or any scalar anomaly measure you can define.

```python
from insurance_monitoring import ConformalControlChart, ConformalChartResult
import numpy as np

# --- Calibration phase ---
# 12 months of in-control residuals from a stable period
actual_cal = np.array([...])   # realised claim amounts
predicted_cal = np.array([...]) # model predictions

cal_scores = ConformalControlChart.ncs_absolute_residual(
    actual_cal, predicted_cal
)

chart = ConformalControlChart(alpha=0.05)
chart.fit(cal_scores)

print(f"Control limit: {chart.threshold_:.2f}")
print(f"Calibration n: {chart.n_cal_}")
```

At monitoring time:

```python
# New quarter's residuals
mon_scores = ConformalControlChart.ncs_absolute_residual(
    actual_monitoring, predicted_monitoring
)

result = chart.monitor(mon_scores)

print(f"Alarm rate this quarter: {result.alarm_rate:.3f}")
print(f"Expected under null:     {chart.alpha}")
print(f"N alarms:                {result.n_alarms}")

# 2-panel plot: NCS + control limit; p-values + alpha line
result.plot(title="Motor Frequency Model — Q1 2026 Monitoring")
```

For heteroskedastic insurance losses, the studentized NCS is more powerful — it adapts the control limit to local volatility:

```python
# Requires a local volatility estimate (e.g. from your GLM dispersion)
cal_vol = np.array([...])  # per-observation volatility estimate, calibration
mon_vol = np.array([...])  # per-observation volatility estimate, monitoring

cal_scores_stud = ConformalControlChart.ncs_studentized(
    actual_cal, predicted_cal, local_vol=cal_vol
)
mon_scores_stud = ConformalControlChart.ncs_studentized(
    actual_monitoring, predicted_monitoring, local_vol=mon_vol
)

chart_stud = ConformalControlChart(alpha=0.05)
chart_stud.fit(cal_scores_stud)
result_stud = chart_stud.monitor(mon_scores_stud)
```

The studentized NCS also produces adaptive interval widths: when local volatility spikes — for example, in a quarter with unusual large loss activity — the control limit widens. This is genuinely useful because a genuine large-loss quarter is distinct from a calibration failure; the adaptive width separates the two signals.

---

## The PSI replacement

Rather than computing PSI on model scores and comparing to 0.2, compute the conformal p-value on the score distribution directly:

```python
import numpy as np

# Reference score distribution: use median deviations per decile
ref_scores = model.predict(X_reference)
mon_scores_raw = model.predict(X_monitoring)

# NCS: deviation of each score from the calibration decile medians
cal_ncs = ConformalControlChart.ncs_median_deviation(ref_scores)

chart = ConformalControlChart(alpha=0.05)
chart.fit(cal_ncs)

mon_ncs = ConformalControlChart.ncs_median_deviation(
    mon_scores_raw,
    median=np.median(ref_scores)  # pin to calibration median
)
result = chart.monitor(mon_ncs)
```

The output is a p-value chart rather than a PSI time series. The control line at $\alpha = 0.05$ means: "we expect 5% of in-control observations to fall below this line." When the alarm rate consistently exceeds 5%, the score distribution has shifted. The statement is quantitative and defensible. "PSI exceeded 0.2 this quarter" is neither.

---

## Multivariate monitoring with MultivariateConformalMonitor

Univariate monitoring catches score distribution shift. It does not catch multivariate relationships degrading — for example, a model that still has the right marginal score distribution but has lost discrimination between renewal and lapse segments, or between low and high A/E segments simultaneously.

`MultivariateConformalMonitor` handles this by treating each monitoring cycle's state vector as a point in feature space and applying conformal anomaly detection:

```python
from insurance_monitoring import MultivariateConformalMonitor

# State vector per monitoring cycle:
# (PSI_age, PSI_vehicle_value, PSI_area, AE_frequency, AE_severity, Gini_trend)
# Shape: (n_cycles, n_features)

X_train = state_vectors_year1       # 12 monthly cycles from stable year 1
X_cal = state_vectors_year2         # 12 monthly cycles for calibration

monitor = MultivariateConformalMonitor(alpha=0.05)
monitor.fit(X_train, X_cal)

# Monitor new cycles
X_new = state_vectors_q1_2026       # shape (3, n_features) for 3 months
result = monitor.monitor(X_new)

print(result.p_values)    # p-value per monthly cycle
print(result.is_alarm)    # True where p < 0.05
print(result.summary())   # Governance paragraph for PRA SS3/17 report
```

The default anomaly detector is `IsolationForest(contamination=0.01)` from scikit-learn, which is an optional dependency. If you prefer a different detector — OC-SVM, an autoencoder, or a domain-specific anomaly model — pass it in:

```python
from sklearn.svm import OneClassSVM

svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
monitor = MultivariateConformalMonitor(alpha=0.05, model=svm)
monitor.fit(X_train, X_cal)
```

The interface requires only `.fit(X)` and `.decision_function(X)` or `.score_samples(X)`. The monitor negates the anomaly detector output (sklearn convention: higher score = more normal; the NCS convention is higher score = more anomalous) and applies the conformal p-value formula.

The output is a single p-value per monitoring cycle. This is the "overall model health" metric the PRA SS3/17 framework asks for: a single number per reporting period with a stated significance level and a documented calibration procedure.

---

## Governance output

`ConformalChartResult.summary()` generates a governance paragraph ready for a model monitoring report:

```python
print(result.summary())
```

Output (example):

> Conformal control chart calibrated on 1,200 in-control observations (January 2024 – December 2024). Significance level alpha = 0.05. Control limit: 0.4231 (absolute residual scale). Q1 2026 monitoring: 847 observations. Observed alarm rate: 0.067 (57 alarms). Expected under stable process: 0.05 (42 alarms). Excess alarms: 15 (36% above expected). Conclusion: borderline evidence of distributional shift; recommend investigation of segment-level A/E ratios before confirming.

This paragraph contains everything a model risk committee needs: calibration period, sample size, significance level, observed alarm rate, expected alarm rate under null, and a quantitative excess. None of these elements are present in a PSI report that says "PSI = 0.23, above the 0.20 red threshold."

---

## What to monitor and when

The NCS choice determines what shift you detect. Our recommended configuration for a standard UK motor pricing model:

| Monitoring objective | NCS | Calibration period |
|---|---|---|
| Model residual distribution shift | `ncs_absolute_residual` or `ncs_relative_residual` | 12 months in-control claims |
| Score/feature distribution shift | `ncs_median_deviation` on predicted score | Reference quarter scores |
| A/E monitoring by segment | Deviation of segment A/E ratio from 1.0 | 2–3 years of stable A/E history |
| Overall model state (multivariate) | `MultivariateConformalMonitor` on state vector | Full stable year of monthly vectors |

The relative residual (`|actual - predicted| / predicted`) is better than the absolute for heteroskedastic severity models, where large claims have naturally larger absolute residuals. The studentized version is best when you have a local volatility estimate — typically from the fitted GLM dispersion or a dedicated volatility model.

---

## The exchangeability constraint

The distribution-free guarantee requires exchangeability: calibration and monitoring observations drawn from the same distribution. For insurance monitoring this is a real constraint, not a technical nicety. Insurance data has seasonality (glass claims in winter, flood claims in storm seasons), trend (claims inflation, exposure mix drift), and structural breaks (COVID, Ogden rate changes). Calibration on data that includes extraordinary periods will inflate calibration NCS distributions and widen thresholds — giving you a false sense of security that the model is stable when in fact the thresholds are just too wide.

Our recommendation: calibrate on a 12-month window that excludes known extraordinary events. If your calibration period includes a hard frost event, an Ogden discount rate change quarter, or a significant MOJ reform announcement, exclude those months from the calibration set and document the exclusion. The audit trail should state: "calibration excludes Q4 2022 (Ogden rate change, +23% BI settlement uplift), N = 847 observations."

The `ConformalControlChart` emits a warning if the calibration set is too small: `n < 20` at `alpha=0.05` (threshold saturates at max calibration score) or `n < 370` at `alpha=0.0027` (3-sigma equivalent). Both warnings are worth heeding — a saturated threshold means every out-of-control observation is flagged, which defeats the purpose of calibration.

---

## Comparison with existing tools

`insurance-monitoring` already has `PITMonitor` (e-process based calibration testing) and `CalibrationCUSUM` (sequential mean shift detection). Conformal charts complement both:

- **vs PITMonitor**: PITMonitor tracks whether the model's probability integral transform is uniform — it detects calibration failure as a global property. The conformal chart detects individual out-of-control periods. PITMonitor is the ongoing PRA/FCA compliance metric; the conformal chart is the operational alert per monitoring cycle.

- **vs CalibrationCUSUM**: CUSUM accumulates evidence of a slow drift and has better power for gradual mean shift. The conformal p-value chart is better for sudden step changes and gives an interpretable p-value per period rather than a cumulative statistic. For most quarterly monitoring cycles, the conformal chart is more intuitive.

- **vs raw PSI**: PSI has no false alarm rate calibration. The conformal chart has an exact guarantee. If your model risk framework requires a stated false alarm rate — which PRA SS3/17 strongly implies it should — the conformal chart is the only distribution-free tool in the library that delivers one.

---

## The paper

Burger, Christopher. "Distribution-Free Process Monitoring with Conformal Prediction." arXiv:2512.23602. 29 December 2025.

The paper is nine pages. Section 3 (univariate charts) is the main result; Section 4 (multivariate p-value chart) is the practical extension. The reference code at [github.com/christopherburger/ConformalSPC](https://github.com/christopherburger/ConformalSPC) is a demonstration rather than a production implementation — two functions, no tests, no API. Our implementation adds the insurance-specific NCS helpers, the governance output, Polars integration, and the multivariate monitor class.

The key technical contribution is the finite-sample coverage guarantee. Conformal p-values under exchangeability satisfy $\Pr(p(X_{\text{new}}) < \alpha) \leq \alpha$ exactly in finite samples. Not asymptotically. This is a strictly stronger guarantee than classical Shewhart, where the 3-sigma false alarm rate of 0.0027 is only exact under normality.

---

## Related posts

- [Insurance Model Monitoring in Python: Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — `insurance-monitoring` library overview
- [Alibi Detect vs insurance-monitoring: What Each Is Actually For](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) — comparison with the generic drift detection ecosystem
- [PITMonitor: Continuous Calibration Monitoring With E-Processes](/2026/03/14/pitmonitor-calibration-monitoring-e-processes/) — the complementary calibration monitoring tool
- [Conformalized Quantile Regression for Insurance Prediction Intervals](/2026/03/24/conformalised-quantile-regression-insurance-prediction-intervals/) — conformal coverage guarantees applied to prediction intervals rather than monitoring
