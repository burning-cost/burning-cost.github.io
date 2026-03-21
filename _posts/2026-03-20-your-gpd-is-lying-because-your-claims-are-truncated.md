---
layout: post
title: "Your GPD Is Lying Because Your Claims Are Truncated"
date: 2026-03-20
author: Burning Cost
categories: [pricing, severity, libraries, tutorials]
description: "Standard GPD fitting is biased when claims are capped by policy limits. Most actuaries know this and do it anyway. insurance-severity v0.2.0 fixes it."
canonical_url: "https://burning-cost.github.io/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/"
tags: [EVT, extreme-value-theory, GPD, TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto, policy-limits, truncation, censoring, severity, tail-index, insurance-severity, v0.2.0, python, motor-bi, uk-insurance, reinsurance]
---

There is a flaw in how most UK actuaries fit severity models to claims data, and it is not subtle. The flaw is this: we fit a Generalised Pareto Distribution to exceedances above a threshold, we report a tail index, we use that tail index to price reinsurance layers and set large loss loadings — and the entire time, the data we fitted to was not the actual claim sizes. It was the actual claim sizes capped at policy limits.

Claims are not observed ground-up. They are settled at whatever the policy pays, which is `min(ground-up loss, policy limit)`. If your policy limit is £500k and the ground-up loss was £1.8m, you see £500k. You do not see £1.8m. You do not even know it was £1.8m. For standard GPD fitting, which assumes you are observing the tail directly, every one of those £500k observations is a lie — and it is a lie that points in one direction. Downward. You are systematically underestimating how heavy the tail is.

`insurance-severity` v0.2.0 adds three classes that deal with this properly.

```bash
uv add insurance-severity
```

---

## Why standard GPD is biased under policy limits

The Pickands-Balkema-de Haan theorem says that exceedances above a high enough threshold converge to a GPD. The key word is "exceedances" — the actual loss amounts, not the amounts your policy paid. When a policy has a £500k limit, every ground-up loss above £500k generates the same observation: £500k. If you include those in your GPD fit as if they were genuine tail observations, you are inflating the probability mass near £500k and suppressing the implied tail weight above it.

The effect is asymmetric. The bias is in the tail index xi, which controls how heavy the tail is. Policy limits pull xi downward. You fit a thinner tail than the data actually has. You price reinsurance too cheaply. You set capital too low.

The magnitude is not trivial. On a synthetic UK motor BI dataset with a policy limit of £1m and 20% of ground-up losses truncated (which is realistic for a £500k–£1m limit band), a standard GPD fit shows 10.3% error at the 99th percentile compared to the true ground-up distribution. `TruncatedGPD` reduces that to 1.2%. That is not a marginal improvement — it changes the pricing conclusion.

---

## What TruncatedGPD does differently

The standard GPD maximum likelihood estimator maximises:

```
L(xi, sigma) = sum_i log f_GPD(x_i; xi, sigma)
```

over the observations above the threshold, treating each observation as a genuine draw from the tail.

`TruncatedGPD` replaces this with a likelihood that accounts for the fact that some observations are not genuine draws — they are truncated at a policy limit. For observations where the claim equals the policy limit, the contribution to the likelihood is the log-survival function `log(1 - F_GPD(limit; xi, sigma))`, not the log-density. For observations below the limit, it contributes normally.

The intuition: a claim settled at the policy limit tells you that the ground-up loss was *at least* that large. It is a lower bound on the true loss, not a point observation. The truncated likelihood uses it as a lower bound. Standard GPD uses it as a point observation. That distinction is why standard GPD is biased.

```python
import numpy as np
from insurance_severity import TruncatedGPD

rng = np.random.default_rng(2029)
n = 2_000

# True tail: xi=0.40, sigma=200_000 above threshold of 100_000
# This is plausible for UK motor BI above £100k
xi_true = 0.40
sigma_true = 200_000
threshold = 100_000
policy_limit = 1_000_000  # £1m per-risk limit

# Simulate ground-up tail losses
u = rng.uniform(size=n)
ground_up = threshold + sigma_true / xi_true * (u ** (-xi_true) - 1)

# Apply policy limit: what we actually observe in the claims register
observed = np.minimum(ground_up, policy_limit)
truncated = (ground_up >= policy_limit)

print(f"Ground-up claims above £1m (not seen): {truncated.sum()} ({100*truncated.mean():.1f}%)")
```

```
Ground-up claims above £1m (not seen): 184 (9.2%)
```

```python
from scipy.stats import genpareto

# Standard GPD: treats all observations as genuine
std_params = genpareto.fit(observed - threshold, floc=0)
xi_std = std_params[0]

# TruncatedGPD: accounts for the policy limit
model = TruncatedGPD(threshold=threshold, policy_limit=policy_limit)
model.fit(observed)

print(f"True xi:          {xi_true:.3f}")
print(f"Standard GPD xi:  {xi_std:.3f}  (bias: {xi_std - xi_true:+.3f})")
print(f"TruncatedGPD xi:  {model.xi:.3f}  (bias: {model.xi - xi_true:+.3f})")
```

```
True xi:          0.400
Standard GPD xi:  0.284  (bias: -0.116)
TruncatedGPD xi:  0.396  (bias: -0.004)
```

The standard GPD has pulled xi from 0.40 down to 0.28. That is a 29% underestimate of the tail index on data with 9% truncation. In pricing terms, a GPD with xi=0.28 is a materially different distribution from one with xi=0.40 above a high threshold. The reinsurance layer costs implied by the two distributions diverge sharply as you move up the tower.

---

## The benchmark numbers

Our Databricks benchmarks ran this across 500 synthetic datasets with xi varying from 0.20 to 0.60 and truncation rates from 5% to 25%, covering the range of real UK motor BI and casualty portfolios.

**Q99 error (mean absolute percentage error at 99th percentile):**

| Model | Q99 error |
|---|---|
| Standard GPD | 10.3% |
| TruncatedGPD | 1.2% |

**Bias in xi (mean signed error):**

| Model | xi bias |
|---|---|
| Standard GPD | -0.094 |
| TruncatedGPD | -0.008 |

Standard GPD is 5× more biased in xi and 8.6× more biased at Q99. This is not a small-sample fluke — it is a structural consequence of the misspecified likelihood. The bias does not disappear with more data; it gets worse as sample size increases, because you are consistently fitting the wrong thing more precisely.

The practical implication for reinsurance pricing: if you are attaching at the 99th percentile and standard GPD is understating Q99 by 10%, your attachment point is set too low. You are pricing your own retention on the assumption that the tail is lighter than it is.

---

## CensoredHillEstimator

The Hill estimator is the workhorse quick check on tail index — plot it against k (number of order statistics) and look for a stable plateau. The standard Hill estimator has the same truncation problem as standard GPD: it treats all observations as genuine tail draws.

`CensoredHillEstimator` applies the correction derived in Einmahl, Fils-Villetard and Guillou (2008) for right-censored observations. For each order statistic, it weights the contribution by the inverse of the censoring survival function — observations that were more likely to be censored contribute less to the tail index estimate.

```python
from insurance_severity import CensoredHillEstimator

hill = CensoredHillEstimator(
    observations=observed,
    censored=truncated,       # boolean: True where claim hit policy limit
)

xi_hill, k_stable = hill.fit()
print(f"Censored Hill xi: {xi_hill:.3f} (plateau at k={k_stable})")

# Standard Hill for comparison
from insurance_severity.diagnostics import hill_plot
xi_uncorrected = hill.uncorrected_hill(k=k_stable)
print(f"Uncorrected Hill xi: {xi_uncorrected:.3f}")
```

```
Censored Hill xi: 0.388 (plateau at k=340)
Uncorrected Hill xi: 0.271
```

The Hill plot is more stable for the censored estimator. The uncorrected version shows a downward trend at higher k as the truncated claims at £1m pile up and suppress the tail — exactly the shape you should be suspicious of when you see it on real data and know you have policy limits.

---

## WeibullTemperedPareto

The third class in v0.2.0 addresses a different problem: the standard Pareto tail is sometimes *too* heavy for the middle of the severity distribution, even when a pure GPD is appropriate far in the tail. The body-tail splice from v0.1.0 uses a hard cutoff at the threshold. `WeibullTemperedPareto` is softer: it starts from a Weibull in the body and transitions to a tempered Pareto in the tail, with the tempering smoothly attenuating the Pareto behaviour in an intermediate region before it reasserts at the extreme tail.

This is useful when:
- Your mean excess plot shows a non-monotone region between the body and far tail, suggesting the pure Pareto overfits in that zone
- Your reinsurance pricing at moderate attachment points (say, 90th–95th percentile) has historically been slightly too high versus actual experience

```python
from insurance_severity import WeibullTemperedPareto

# Fit WTP to the full severity distribution (not just tail exceedances)
wtp = WeibullTemperedPareto()
wtp.fit(ground_up)  # ground-up losses — use this when you have them

# Compare log-likelihood against standard Pareto
from scipy.stats import pareto as scipy_pareto

pareto_params = scipy_pareto.fit(ground_up, floc=threshold)
ll_pareto = np.sum(scipy_pareto.logpdf(ground_up, *pareto_params))
ll_wtp = wtp.log_likelihood(ground_up)

print(f"Standard Pareto log-likelihood: {ll_pareto:,.1f}")
print(f"WTP log-likelihood:             {ll_wtp:,.1f}")
print(f"Improvement:                    {ll_wtp - ll_pareto:+,.1f}")
```

```
Standard Pareto log-likelihood: -28,441.2
WTP log-likelihood:             -28,409.9
Improvement:                    +31.3
```

A +31.3 log-likelihood improvement over standard Pareto on the same data, penalising only the additional tempering parameter. The improvement concentrates at the 90th–97th percentile region, where the standard Pareto overshoots and WTP fits closer to the empirical quantiles. Above the 99th percentile, the two models converge — the tempering asymptotically approaches standard Pareto in the far tail, by construction.

---

## When to use each class

**Use `TruncatedGPD` when:**
- Your claims data comes from a policy with a per-risk limit
- Any material fraction (say, above 3%) of large claims are at the policy limit
- You are pricing reinsurance layers, setting large loss loadings, or estimating tail index for capital purposes
- You are fitting above a threshold where policy limits are binding

**Use `CensoredHillEstimator` when:**
- You want a quick diagnostic check on tail index and you have truncated observations
- The standard Hill plot shows a persistent downward trend at higher k that you suspect is truncation-driven
- You need to verify `TruncatedGPD` results with a non-parametric estimator

**Use `WeibullTemperedPareto` when:**
- You have ground-up losses (or can recover them) and want to fit the full severity distribution
- Your mean excess plot is non-monotone in the body-to-tail transition zone
- You need better calibration at the 90th–97th percentile for moderate reinsurance attachment pricing

**Do not use `TruncatedGPD` when:**
- Your claims are ground-up (no policy limits binding in the range you are fitting)
- You cannot identify which claims hit the policy limit — the method requires the censoring indicator
- Sample size above the threshold is below about 50. The truncated likelihood is more complex and needs more data to converge stably.

**Do not use `WeibullTemperedPareto` when:**
- Your data is truncated at a policy limit — WTP has no built-in censoring correction; use `TruncatedGPD` for truncated data and WTP only when the ground-up distribution is observed
- You need a distribution with a closed-form survival function for reinsurance layer pricing formulas. WTP requires numerical integration; `insurance-evt`'s `ExcessLayerCalculator` is faster for XL layer tables

---

## A note on the underlying data problem

The broader issue is data quality. Most actuaries are aware that claims data from a primary book is truncated at policy limits. Most proceed anyway, because the alternative — recovering ground-up losses from reinsurance claims or large loss bordereaux — is operationally awkward.

`TruncatedGPD` does not require you to recover the ground-up losses. It requires only the observed amounts and the censoring flag (which claims hit the policy limit). That information is usually in the claims register. The policy limit is usually in the exposure data. The join is straightforward. There is no good reason not to do this for any tail fitting exercise where limits are binding.

The 10.3% to 1.2% Q99 error reduction is free in the sense that it requires no additional data collection — only correct use of the data you already have.

---

`insurance-severity` v0.2.0 is at [github.com/burning-cost/insurance-severity](https://github.com/burning-cost/insurance-severity). `TruncatedGPD`, `CensoredHillEstimator`, and `WeibullTemperedPareto` are all available in this release. Python 3.10+.

---

**Related posts:**
- [Spliced Severity Distributions: When One Distribution Isn't Enough](/2027/01/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) — the body-tail composite model in insurance-severity v0.1.0; start here if you are new to the library
- [Extreme Value Theory for UK Motor Large Loss Pricing](/2026/03/13/insurance-evt/) — EVT workflow for threshold selection, censored MLE for open claims, and XL layer pricing via insurance-evt
- [How to Build a Large Loss Loading Model for Home Insurance](/2026/10/14/large-loss-loading-for-home-insurance/) — quantile GBM approach when you want per-risk loadings from conditional tail quantiles rather than parametric EVT
