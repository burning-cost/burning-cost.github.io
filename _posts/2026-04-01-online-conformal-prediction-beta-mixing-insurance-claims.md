---
layout: post
title: "Online Conformal Validity on Claims Data: Beta-Mixing Conditions and the Seasonal Caveat"
date: 2026-04-01
categories: [techniques, pricing]
tags: [conformal-prediction, online-learning, time-series, beta-mixing, non-exchangeable, ACI, EnbPI, WeightedConformalPredictor, WCP, ConformalPID, insurance-conformal-ts, seasonal-adjustment, distribution-shift, motor, claims-inflation, arXiv-2511-13608, python]
description: "Standard conformal prediction gives valid coverage only when calibration and test data are exchangeable. For insurance models deployed for 12+ months — through claims inflation cycles, seasonal patterns, and portfolio drift — exchangeability is simply not true. The beta-mixing framework from arXiv:2511.13608 gives the conditions under which coverage guarantees survive temporal dependence. The key result for UK practitioners: GLM residuals are fine; raw seasonal claim counts are not."
math: true
author: burning-cost
---

Insurance pricing models are not trained and retired in a controlled laboratory. They are deployed, forgotten, and quietly mis-calibrated for months or years. A motor frequency model fitted on 2022 data is still running in a 2024 portfolio during a claims inflation cycle that has added 15–20% to bodily injury costs. The conformal prediction interval calibrated on the training window will be systematically too narrow. That is not a subtle statistical concern — it affects your reserve adequacy and your Solvency II SCR.

The online conformal prediction methods in `insurance-conformal-ts` — ACI, EnbPI, SPCI, ConformalPID, and the new `WeightedConformalPredictor` in v0.2.0 — all claim to maintain coverage under distribution shift. Most practitioners take this on faith. This post explains what conditions are actually required for those guarantees to hold, what breaks them for insurance data, and which method to use for which problem.

---

## Why exchangeability fails in insurance

Standard split conformal prediction gives a coverage guarantee under one assumption: the calibration scores and test scores are exchangeable — drawn i.i.d. from the same distribution. The guarantee is sharp and distribution-free, which is why it has become popular.

Insurance time series violate exchangeability in at least three ways:

**Serial correlation.** Monthly claim counts in adjacent months are correlated. A model calibrated on January–June will have correlated residuals. The standard conformal proof assumes i.i.d. residuals; serial correlation inflates the effective variance and can cause the empirical quantile to be estimated too precisely, producing intervals that are nominally 90% but actually achieve lower coverage.

**Distribution shift.** UK motor claims inflation ran at roughly 10% per annum in 2022–2023. A model fitted before the inflation cycle produces systematically underestimated predictions, so the nonconformity scores on test data are higher than on calibration data. The coverage guarantee fails because the test distribution has shifted away from the calibration distribution.

**Seasonality.** Home insurance claims peak in Q4 (storms, burst pipes). Motor claims peak in Q1 (ice, fog). A model calibrated on data from March–August will have calibration scores drawn from a different part of the seasonal cycle than test scores from November–January.

All three are distinct problems. Serial correlation is about dependence within the calibration set. Distribution shift is about the gap between calibration and test. Seasonality is a specific form of distribution shift that is periodic and deterministic.

---

## The beta-mixing framework

A recent survey paper — arXiv:2511.13608, "A Gentle Introduction to Conformal Time Series Forecasting" — provides the theoretical conditions under which conformal prediction retains approximate coverage guarantees for dependent data. The central concept is beta-mixing.

**Definition.** A stochastic process is beta-mixing if its temporal dependence decays as observations get further apart in time. Formally, the beta-mixing coefficient $\beta(a)$ measures (in total variation) how much the joint distribution of observations separated by lag $a$ differs from the product of their marginals:

$$\beta(a) = \left\| P_{-\infty:0,\, a:\infty} - P_{-\infty:0} \otimes P_{a:\infty} \right\|_{\mathrm{TV}}$$

The process is beta-mixing if $\beta(a) \to 0$ as $a \to \infty$. Intuitively: observations that are far enough apart in time become approximately independent.

**Why this matters for calibration concentration.** The key question for conformal prediction is whether the empirical distribution of calibration scores is a good proxy for the true marginal distribution. Under i.i.d. sampling, concentration follows from the standard CLT. Under beta-mixing, the effective sample size is reduced by serial correlation — the relevant quantity is:

$$\tilde{\sigma}(a) = \sqrt{v + \frac{2}{a} \sum_{k < a} (a - k)\, \beta(k)}$$

where $v = \mathrm{Var}[Z_1]$. This is the effective standard deviation of the sample mean after accounting for correlation. When $\beta(k)$ decays geometrically (fast mixing), $\tilde{\sigma}(a)$ stays bounded and calibration concentration proceeds at the standard $O(1/\sqrt{n_{\mathrm{cal}}})$ rate. When $\beta(k)$ decays slowly, the effective variance grows and you need a much larger calibration set to get the same precision.

The practical upshot (Theorem A.3.1 from the paper) is that marginal coverage holds approximately:

$$P\!\left(Y_i \in C_{1-\alpha}(X_i)\right) \geq 1 - \alpha - \varepsilon_{\mathrm{cal}} - \delta_{\mathrm{cal}} - \varepsilon_{\mathrm{train}}$$

The slack $\varepsilon_{\mathrm{train}}$ is what captures distribution shift between calibration and test — this is the term that spikes during claims inflation. The slack $\varepsilon_{\mathrm{cal}}$ is what captures calibration concentration under dependence — this is what beta-mixing controls.

---

## Which insurance series are beta-mixing?

This is the practically important question, and the answer is not uniform.

**GLM residuals on monthly motor data.** If your model captures trend and seasonality correctly, the residuals should behave like a low-order moving average process. MA($q$) processes are beta-mixing with geometric decay — $\beta(a) \sim \rho^a$ for some $\rho < 1$. The calibration concentration result applies with small slack. **The coverage guarantee holds.** This is the case where online conformal methods are theoretically justified.

**Rolling 12-month loss ratios.** Rolling aggregation induces MA(11) autocorrelation by construction. Beta-mixing still holds, but $\tilde{\sigma}(a)$ is non-trivial at lags 1–11. In practice: use a calibration window of at least 36 months to ensure the effective sample size is large enough.

**Seasonal raw claim counts with no seasonal adjustment.** This is the failure case. A deterministic periodic signal — claims that peak every December regardless of your model — means the beta-mixing coefficients at the annual lag do not decay to zero. They cycle. A purely periodic process is not beta-mixing. If you run conformal prediction on raw motor or home claim counts without deseasonalising, the coverage guarantee is vacuous. **Always apply conformal prediction to model residuals, not raw counts.**

**Claims inflation drift.** This is not a mixing failure — it is a shift in the test distribution ($\varepsilon_{\mathrm{train}}$ grows). Beta-mixing holds on the residuals. ACI, EnbPI, and `WeightedConformalPredictor` all handle this through their adaptive mechanisms. The beta-mixing framework tells you the calibration concentration is fine; it is the test decoupling assumption that breaks down, and the online methods are specifically designed to repair that.

**Long UK insurance market cycles.** The soft/hard cycle in UK liability and commercial lines runs roughly 5–10 years. If your calibration window spans more than one cycle, you may be in polynomial mixing territory ($\beta(a) \sim a^{-r}$). The practical fix is to cap calibration windows at 12–18 months and rely on the online update mechanisms rather than a long static calibration window.

---

## The seasonal caveat in full

The seasonal warning deserves more than a footnote because it is the most common mistake we see in practice.

Consider a home insurance model being monitored with monthly claim counts. The raw series has a pronounced December–January spike. The model captures some of this through seasonal features, but not perfectly. The residuals still have a seasonal component.

If you calibrate conformal prediction on those residuals directly, the January calibration scores will be higher (large December residuals) than the July scores (low summer residuals). When you apply the fitted predictor to next December's data, the calibration quantile is wrong because the seasonal composition of your calibration window does not match December's distribution.

**The fix is simple.** Deseasonalise the residuals before computing nonconformity scores. The standard approach is to fit a seasonal dummy or harmonic regression on the residuals and use the deseasonalised residuals as inputs to the conformal predictor. Alternatively, use a base forecaster that explicitly handles seasonality (SARIMA, Prophet, or a GLM with month dummies), so that the conformal layer only sees white-noise-ish residuals.

If deseasonalisation is impractical — for example, in a real-time scoring pipeline — use Mondrian conformal prediction partitioned by month. This is more conservative (12 separate calibration windows) but avoids the mixing violation entirely.

---

## Four strategies for non-exchangeable conformal prediction

arXiv:2511.13608 classifies all online conformal methods into four strategies. This taxonomy is useful because it tells you what each method is actually doing when it adapts.

| Strategy | Methods | Mechanism |
|---|---|---|
| Reweighting | WCP | Upweight recent calibration scores, discount stale ones |
| Refreshing | EnbPI | Replace stale residuals with fresh ones (hard cutoff) |
| Adapting target | ACI, ConformalPID | Adjust the miscoverage level $\alpha_t$ online |
| Blocking | BCP | Permute temporal blocks for p-values |

Block conformal (BCP) is not in `insurance-conformal-ts` and we do not plan to add it. The empirical result from the survey is unambiguous: under mean shift — which is what UK claims inflation looks like — BCP achieves only 0.81–0.84 coverage at a 0.90 nominal level, worse than standard split conformal. BCP works only for stationary short-memory processes, which are the least interesting case for insurance.

The three strategies that actually work are reweighting, refreshing, and adapting. They address the same problem through different mechanisms, and the choice between them matters.

---

## WeightedConformalPredictor: the reweighting strategy

`WeightedConformalPredictor` (WCP) is new in `insurance-conformal-ts` v0.2.0. It is the one strategy in the taxonomy that was missing from the library.

The idea is direct. Instead of treating all calibration scores equally, assign higher weight to recent observations:

$$w_i = \rho^{a_i}, \quad \tilde{w}_i = \frac{w_i}{\sum_j w_j}$$

where $a_i$ is the age of observation $i$ in steps and $\rho \in (0,1)$ is the decay parameter. The prediction interval is formed at the weighted $(1-\alpha)$-quantile of the calibration scores under $\tilde{w}$.

The weighted quantile is computed by sorting calibration scores ascending and finding the first score where the cumulative weight exceeds $1 - \alpha$. This is pure numpy — no new dependencies.

The decay parameter controls how quickly old observations are forgotten:

- $\rho = 0.99$: half-weight at lag 69 steps. For monthly data, about 5.8 years. Suitable for commercial lines with slow-moving cycles.
- $\rho = 0.95$: half-weight at lag 14 steps. For monthly data, about 1.2 years. Suitable for motor/home lines with 2–3 year inflation cycles.
- $\rho = 0.90$: half-weight at lag 7 steps. For monthly data, about 7 months. Aggressive — appropriate if you suspect a rapid regime change.

**Why WCP rather than ACI?** ACI adapts $\alpha_t$ reactively — it widens intervals after missing. WCP adapts preemptively — by continuously upweighting recent observations, the quantile estimate drifts towards the current distribution before a coverage error is observed. For gradual drift (the UK motor claims inflation pattern from 2021–2023), WCP produces narrower intervals than ACI because it does not wait for misses to start responding. For abrupt shifts (a sudden change in the underlying regime), ACI's feedback loop tends to converge faster.

---

## API walkthrough

```python
import numpy as np
from insurance_conformal_ts import WeightedConformalPredictor, PoissonPearsonScore

# Assume: glm fitted on training data, monthly motor claim counts
# y_train: np.ndarray of observed counts, shape (n_train,)
# X_train: np.ndarray of features, shape (n_train, p)
# glm.predict(X): returns expected counts (exposure-adjusted)

wcp = WeightedConformalPredictor(
    base_forecaster=glm,
    score=PoissonPearsonScore(),
    rho=0.95,          # half-weight at ~14 months for motor
    window_size=200,   # keep last 200 calibration scores
    burn_in=5,         # return infinite intervals until 5 scores seen
)

wcp.fit(y_train, X_train)

# predict_interval runs the sequential loop internally.
# At each step t it predicts the interval, then updates the
# calibration window with the observed y[t] before moving to t+1.
# This matches the monthly monitoring cadence: one interval per period,
# exponential weights updated as actuals arrive.
lower, upper = wcp.predict_interval(y_test, X_test, alpha=0.1)

# wcp now holds an updated calibration window covering the full test period.
# Feed subsequent batches directly:
#   lower_next, upper_next = wcp.predict_interval(y_next, X_next, alpha=0.1)
```

For the seasonal caveat — if your GLM does not fully capture seasonality, deseasonalise residuals before fitting:

```python
from insurance_conformal_ts import WeightedConformalPredictor, AbsoluteResidualScore
from sklearn.linear_model import LinearRegression

# Deseasonalise residuals with month dummies before conformal calibration
residuals_train = y_train - glm.predict(X_train)
month_dummies = np.eye(12)[month_indices_train]  # (n_train, 12)
seasonal_model = LinearRegression(fit_intercept=False).fit(month_dummies, residuals_train)
deseasonalised = residuals_train - seasonal_model.predict(month_dummies)

# Wrap your GLM to output deseasonalised residuals as scores
# Use a custom NonConformityScore that applies the seasonal correction
# ... then proceed as above with the corrected residuals
```

In practice, the cleanest approach is to add monthly dummy variables directly to the GLM so the base forecaster handles seasonality, and the conformal layer sees approximately mixing residuals.

---

## Which method to use and when

This is the decision most posts leave vague. We will be direct.

**Use ACI** when you want simplicity and the drift is primarily abrupt — a sudden shift in the claims environment that you need to recover from quickly. ACI with `gamma=0.005` is a reasonable default for UK motor. The single hyperparameter is interpretable: smaller `gamma` gives slower response, larger `gamma` gives wider intervals in stable periods.

**Use EnbPI** when you have a heterogeneous book with clear data breaks — a new reinsurance structure, an acquired portfolio, a change in product terms. EnbPI's hard replacement of stale residuals means old data from a different product structure does not pollute the calibration window. The `M` bootstrap parameter should be at least 20 for stable quantile estimates.

**Use `WeightedConformalPredictor`** when drift is gradual and you care about interval efficiency — you want intervals that are tight in stable periods and expand smoothly under inflation, without the jagged behaviour of ACI's discrete coverage-error feedback. UK motor frequency monitoring over a 2–3 year cycle is exactly this use case.

**Use ConformalPID** when you have long deployment horizons and care about cumulative regret bounds. The PID controller has the best theoretical guarantees for long-run cumulative miscoverage. For FCA-reportable monitoring where you track coverage over the year, ConformalPID gives you the strongest formal argument.

**For seasonal data, all of the above**: deseasonalise first. The method choice is secondary to getting the residual structure right.

---

## What to check before trusting any of these

The coverage guarantees are not unconditional. Before relying on online conformal prediction in production:

**Check calibration window length.** For rolling 12-month loss ratios (MA(11) correlation), you need at least 36 months of calibration data for the beta-mixing concentration to give you meaningful precision. With 12 months of monthly data, you have perhaps 6–8 effective independent observations after accounting for autocorrelation. That is not enough to estimate a 90th percentile.

**Plot sequential coverage.** After deployment, track the rolling empirical coverage over 12-month windows. A well-calibrated online predictor should stay within $\pm 2$–3 percentage points of the nominal level. Persistent undercoverage means your decay parameter is too small (you are discounting old data too aggressively or not aggressively enough). The `SequentialCoverageReport` diagnostic in `insurance-conformal-ts` produces this plot directly.

**Do not apply conformal prediction to raw seasonal series without checking.** If the autocorrelation function of your nonconformity scores has a spike at lag 12 (annual for monthly data), the beta-mixing condition is violated and the coverage guarantee does not apply. Fix the base model first.

**Be aware of long-range dependence.** If you have 10+ years of loss ratio data exhibiting a soft/hard market cycle, the series may have Hurst exponent $H > 0.5$ and polynomial beta-mixing decay. In this case, limit calibration windows to 12–18 months. This is a conservative choice but the coverage guarantee becomes vacuous with longer windows when mixing is polynomial.

---

## Getting the library

```python
uv add insurance-conformal-ts
```

`WeightedConformalPredictor` is available from v0.2.0. The existing ACI, EnbPI, SPCI, and ConformalPID are in all versions from v0.1.0.

```python
from insurance_conformal_ts import (
    ACI,
    EnbPI,
    ConformalPID,
    WeightedConformalPredictor,  # v0.2.0+
    PoissonPearsonScore,
    NegBinomPearsonScore,
    ExposureAdjustedScore,
    SequentialCoverageReport,
)
```

Source: `src/insurance_conformal_ts/methods.py`. Tests: `tests/test_methods.py`.

---

## The paper

"A Gentle Introduction to Conformal Time Series Forecasting." arXiv:2511.13608 [stat.ML]. November 2024.

---

## Related posts

- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/2026/04/01/shape-adaptive-conformal-prediction/) — ShapeAdaptiveCP (MOPI) for conditional coverage across rating cells
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — conformal risk control and the `insurance-conformal` library overview
- [Conformal Prediction for Insurance Pricing: Intervals, Risk Control, and the Practical Toolkit](/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — when conformal prediction works, when it does not, and what calibration data you actually need
- [Conditional Coverage in Conformal Prediction: Model Selection with CVI](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — checking whether your conformal guarantees hold conditionally, not just marginally
