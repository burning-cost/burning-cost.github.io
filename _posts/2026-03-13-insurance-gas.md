---
layout: post
title: "GAS Models for Claims Inflation: Trend Estimation With a Likelihood"
date: 2026-03-13
categories: [libraries, pricing, trends]
tags: [GAS, generalised-autoregressive-score, trend, frequency, severity, loss-ratio, Poisson, Gamma, Beta, time-series, observation-driven, MLE, PRA, claims-inflation, insurance-dynamics, python]
description: "The PRA's June 2023 Dear Chief Actuary letter found personal lines claims inflation allowances ranging from 0.6% to 9% — a 15× spread that can't be explained by genuine portfolio differences. insurance-dynamics brings GAS models to Python for the first time, giving pricing actuaries a statistically grounded alternative to moving averages."
---

The PRA's June 2023 Dear Chief Actuary letter on motor and home claims inflation did not mince its words. Across the personal lines firms surveyed, claims inflation allowances ranged from 0.6% to 9%. That is not analytical disagreement. That is fifteen different firms applying fifteen different informal judgements to the same underlying problem, and the spread is so wide it tells you something structural: the profession does not have an agreed analytical method.

The typical process is: take a moving average over some window (12 months? 24?), eyeball whether the recent trend looks like an outlier, apply a judgement adjustment, document it as "management overlay", get it signed off. No likelihood. No standard error. No formal way to distinguish signal from noise.

[`insurance-dynamics`](https://github.com/burning-cost/insurance-dynamics) is our response to this. It brings Generalised Autoregressive Score (GAS) models to Python for the first time — the first maintained Python implementation anywhere. The equivalent R library, `gasmodel`, has supported 36 distributions since 2022. Python has had nothing but an abandoned PyFlux (last commit 2018) and a statsmodels GitHub issue open since 2016.

```bash
uv add insurance-dynamics
```

---

## What GAS models actually do

GAS models, introduced by Creal, Koopman and Lucas (2013) and independently by Harvey (2013), solve a specific problem: how do you update a time-varying parameter as new data arrives, without knowing the true state of the world?

The answer is to use the score — the gradient of the log-likelihood with respect to the parameter — as the forcing variable. The recursion is:

```
f_{t+1} = omega + alpha * S(f_t) * nabla(y_t, f_t) + phi * f_t
```

where `f_t` is the time-varying parameter (log frequency, log severity, logit loss ratio), `nabla` is the score of the log-likelihood, `S` is a scaling matrix (inverse Fisher information by default), `alpha` controls how fast the filter reacts to new observations, and `phi` controls how persistent the current level is.

The key insight is that the score is proportional to the surprise in the observation. If claims frequency comes in 30% higher than the current filtered level, the score is large and positive and the filter reacts strongly. If claims come in exactly as expected, the score is near zero and the filter barely moves. The filter is self-calibrating — it adapts faster when the model is wrong by more.

The other crucial property is that GAS models are observation-driven. The filter path is a deterministic function of past data. That means the full data likelihood is just a product of closed-form densities, and fitting is standard L-BFGS-B optimisation. No MCMC, no Kalman smoother, no numerical integration. MLE is fast and the standard errors come from the Hessian.

---

## Why this matters for frequency and severity tracking

The distributions that matter for pricing actuaries are not Gaussian. Claim counts follow Poisson or negative binomial. Severities follow Gamma or log-normal. Loss ratios live on (0, 1) and want a Beta. Each of these has a natural score function that respects the support and variance structure of the distribution.

A moving average on Poisson claim counts treats a week with 1,000 earned policies the same as a week with 10,000. The score-based update automatically up-weights the more informative observation because Fisher information scales with exposure. The GAS filter with Poisson distribution and log-exposure offset is doing exposure-weighted inference — something a moving average cannot do without bespoke engineering.

The same logic applies to severity. Large claims contain more information about the true severity level than small claims, but a simple average treats them symmetrically. With a Gamma or log-normal GAS model, the score accounts for the variance structure of the distribution, so the filter is naturally influenced more by observations that are informative under the assumed model.

---

## The Python gap

We are not claiming GAS models are new. The Creal, Koopman, Lucas (2013) paper has been cited over 2,000 times. The R `gasmodel` package (Holy and Zouhar, 2024) is maintained and supports 36 distributions. There are implementations in Julia and Matlab.

Python has nothing. PyFlux implemented GAS in 2017 and has not had a meaningful commit since 2018. It is incompatible with modern NumPy and fails to install on Python 3.10+. The statsmodels issue requesting GAS support has been open since 2016 with no resolution.

Every Python pricing actuary who wanted GAS models has had three options: use R, port the equations themselves, or settle for moving averages. `insurance-dynamics` closes that gap.

---

## Using it

The quickest path for a pricing team is Poisson frequency monitoring with exposure:

```python
from insurance_dynamics.gas import GASModel
from insurance_dynamics.gas.datasets import load_motor_frequency

# 60 monthly periods with a trend break at period 40
data = load_motor_frequency(T=60, trend_break=True)

model = GASModel(
    distribution="poisson",   # claim frequency with log link
    p=1, q=1,                 # GAS(1,1): one lag of score, one lag of filter
    scaling="fisher_inv",     # optimal scaling for exponential family
)
result = model.fit(data.y, exposure=data.exposure)

print(result.summary())
# Params: omega=-3.21 (0.08), alpha=0.14 (0.03), phi=0.87 (0.04)
# Log-lik: -812.3 | AIC: 1630.6 | BIC: 1643.8
# Score residuals: mean=-0.02, std=1.01 [expected ~iid(0,1)]

# The time-varying rate, indexed to 100 at t=0
result.trend_index.plot(title="Frequency Trend Index")

# Relativities vs the time-average
result.relativities(base="mean")
```

The `trend_index` output is deliberately in the format pricing actuaries already use for development factor analyses — base period = 100, subsequent periods as relatives. You can put it in a monitoring pack without reformatting.

For severity, swap the distribution:

```python
model = GASModel(distribution="gamma", p=1, q=1)
result = model.fit(data.severity, exposure=data.claim_count)
```

For loss ratio monitoring:

```python
model = GASModel(distribution="beta", p=1, q=1)
result = model.fit(data.loss_ratio)  # must be in (0,1)
```

Panel data — tracking multiple rating cells simultaneously — uses `GASPanel`:

```python
from insurance_dynamics.gas import GASPanel

panel = GASPanel("poisson")
panel_result = panel.fit(
    df,
    y_col="claims",
    period_col="month",
    cell_col="vehicle_class",
    exposure_col="exposure",
)

panel_result.trend_summary()   # wide DataFrame: periods × vehicle classes
```

The diagnostics are formal rather than visual. The score residuals should be approximately iid(0,1) for a well-specified model. The library tests this with a probability integral transform histogram and a Ljung-Box test on the residual autocorrelation:

```python
diag = result.diagnostics()
print(diag.summary())
# PIT: uniform (p=0.41)
# Ljung-Box lag-12: statistic=11.2, p=0.51 — no autocorrelation
# Dawid-Sebastiani score: -1.84
```

If the Ljung-Box is significant, the filter is not fully capturing the autocorrelation structure — the usual response is to increase p or q.

---

## Where it sits in the toolkit

We think of trend analysis as having three distinct problems:

1. **Has a structural break occurred?** That is [`insurance-dynamics`](https://github.com/burning-cost/insurance-dynamics) — PELT and window-based detection with actuarial output formats.

2. **What is the current rate of change?** That is `insurance-dynamics` — continuous adaptive estimation of the time-varying parameter, updated at each period.

3. **What is the projected trend to apply to future periods?** That is [`insurance-trend`](https://github.com/burning-cost/insurance-trend) — GLM-based trend fitting with bootstrap confidence intervals for use in pricing sign-off.

These are not competitors. The rational workflow is: use `insurance-dynamics` to identify regime boundaries, fit `insurance-dynamics` within each regime to get the adaptive trend path, then use `insurance-trend` to project the current regime forward with formal uncertainty bounds.

---

## The question the PRA was asking

The Dear Chief Actuary letter was asking, in effect: how did you arrive at your inflation allowance? What data? What method? What uncertainty?

A GAS-fitted Poisson frequency model gives you answers to all three questions. The data is every monthly observation with its earned exposure. The method is MLE on the GAS(1,1) likelihood. The uncertainty is a bootstrap confidence interval on the filter path or, equivalently, a formal standard error on omega, alpha, and phi.

The 0.6%–9% dispersion in the PRA's survey is partly genuine portfolio difference — motor mix, geographic concentration, repair network. But a 15× range is not explainable by portfolio heterogeneity alone. Some of it is different informal judgements applied to the same underlying signal. A formal statistical model does not eliminate all disagreement, but it makes the remaining disagreement legible: two firms can look at each other's alpha and phi estimates and have a grounded conversation about why their filters are responding differently.

That is what a likelihood buys you. A moving average does not have one.

---

**[insurance-dynamics on GitHub](https://github.com/burning-cost/insurance-dynamics)** — MIT-licensed, PyPI. The first maintained Python GAS library.

---

**Related reading:**
- [Trend Selection Is Not Actuarial Judgment: A Python Approach](/2026/03/13/insurance-trend/) — GLM-based trend fitting with bootstrap confidence intervals for use in pricing sign-off; the complement to GAS for the structural trend component
- [When Did Your Loss Ratio Actually Change?](/2026/03/13/insurance-changepoint/) — Bayesian change-point detection for identifying when the regime shifted before fitting any trend model
- [MinTrace Reconciliation for Insurance Pricing Hierarchies](/2026/03/11/your-peril-forecasts-dont-add-up/) — hierarchical reconciliation of frequency and severity trends across perils; the forecasting layer that GAS estimates feed into
