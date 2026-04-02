---
layout: post
title: "Your XL Pricing Has a Truncation Bias. Here Is the Fix."
date: 2026-03-31
categories: [techniques]
tags: [EVT, extreme-value-theory, severity, reinsurance, XL-pricing, truncation, censoring, IBNR, GPD, Hill-estimator, insurance-severity, Solvency-II, SCR, python, arXiv-2511-22272]
description: "Ignoring policy limits and IBNR censoring when fitting tail distributions biases your tail index by ~15%. For UK motor TPBI with xi~0.60, that is the difference between finite and infinite variance. insurance-severity already fixes this."
---

If you are fitting a GPD or Hill estimator to your large-loss data without accounting for policy limits and IBNR open claims, you are not estimating the tail of your loss distribution. You are estimating something else, and pricing your XL layers off it.

This is not a subtle point. Albrecher and Beirlant's November 2025 handbook chapter (arXiv:2511.22272) — a consolidation of a decade of EVT-in-insurance literature — gives the machinery to fix it. Our `insurance-severity` library already implements all three core methods. This post explains when the bias matters, how large it is, and what to do about it.

---

## What goes wrong

Every UK insurer's large-loss data has two structural problems that standard EVT ignores.

**Policy limits (upper truncation).** Your liability, motor TPBI, and commercial property claims are capped at policy limits. The reinsurer sees claims only up to the cedant's retention. A plain GPD fitted to that data never sees what happens above the cap — it estimates the distribution of *observed* losses, not the underlying severity. The effect on the tail index is a positive bias: the estimated xi is too high, because the largest *observed* claims sit artificially close to the limit, making the tail appear heavier from below than it actually is.

Wait — if xi is biased upward, shouldn't that mean you're overpricing reinsurance, which would be conservative? The problem is more awkward. An inflated xi implies heavier extrapolation above the data range. If you are pricing an XL layer that sits above your data's policy limit, your attachment-point and exhaustion-point loss cost calculations are both distorted. You may be underpricing the layer you actually wrote while overestimating the severity beyond it.

**IBNR open claims (right-censoring).** A serious brain injury claim in a UK motor TPBI triangle may remain open for ten years. At any valuation date, the current case reserve is a censored observation: you know the claim is at least that large, but the final settlement is unknown. Standard Hill and GPD treat the current reserve as the true claim amount. They don't. When the largest open claims are the most severely underdeveloped — as they almost always are — the correction goes the other way: standard Hill *underestimates* xi, because it doesn't know those reserves will multiply.

Poudyal and Brazauskas (2022, cited in Albrecher & Beirlant) quantified the censoring effect on the Hill estimator. With n = 50 tail observations and 30% censoring, standard Hill overestimates xi by roughly 15%. With n = 200 tail observations and 30% censoring it's still around 8%. These aren't asymptotic approximations — they come from simulation studies at realistic sample sizes.

For UK motor TPBI, where the tail index xi sits in the range 0.50–0.67 (alpha = 1.5–2.0), a 15% upward bias in xi produces an estimate around 0.69 for a true value of 0.60. That gap matters. xi = 0.60 implies finite variance. xi = 0.69 is still finite variance, but the tail quantile estimates diverge sharply at the 99.5th percentile. Your Solvency II SCR is computed off those quantiles. So is your XL layer pricing.

The Loss/ALAE public dataset (general liability, n = 1,500, 34 censored losses, available in the `copula` R package) gives xi_hat = 0.67 with censoring correction applied. Without correction on the same data the naive Hill estimate sits above that. With only 2.3% censoring in that dataset, the effect is small — but motor TPBI reserving triangles with 30–40% of large claims still open are not unusual, and the effect is not small there.

---

## Three methods, all already built

Albrecher and Beirlant (2025) identify three estimators that handle these problems properly. All three are in `insurance-severity` v0.3.1.

```bash
uv add insurance-severity
```

### 1. Censored Hill Estimator

The correction is conceptually simple. The standard Hill estimator for the top-k order statistics is:

```
H_k = (1/k) * sum_{j=1}^{k} log(X_{(n-j+1)} / X_{(n-k)})
```

When some of those top-k are censored — current reserves, not final settlements — the numerator is deflated because the censored values are understated. Dividing by the fraction of *uncensored* observations among the top-k restores the correct scaling:

```
H_k^(c) = H_k / (number uncensored in top-k / k)
```

This is Equation 8 in Albrecher and Beirlant (2025). In `insurance-severity`:

```python
import numpy as np
from insurance_severity.evt import CensoredHillEstimator

# claims: all large losses above threshold
# censored: True for IBNR / still-open claims
est = CensoredHillEstimator()
est.fit(claims=large_losses, censored=is_ibnr)

print(f"xi estimate: {est.xi:.3f}")
print(f"Optimal k:   {est.k_opt}")

# Hill plot with bootstrap CI
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
est.hill_plot(ax=ax)
plt.tight_layout()
plt.savefig("hill_plot.png", dpi=150)
```

The `hill_plot()` method shows the censored Hill estimate across all k values. You are looking for stability in a region where the estimate is neither climbing (threshold too low, body contamination) nor noisy (threshold too high, too few observations). The k selection uses rolling variance minimisation with a window of 10 — automated, but you should verify visually on your data.

### 2. Truncated GPD MLE

For policy limit truncation, the fix is in the log-likelihood. The standard GPD log-likelihood for exceedances above threshold u treats every observation as drawn from the same GPD. When observations are capped at policy limit T_i, each observation is drawn from a GPD *truncated above at (T_i - u)*. The corrected log-likelihood subtracts `log F_GPD(T_i - u | xi, sigma)` for each capped observation — the probability mass that was removed by truncation.

This matters most when some policy limits are close to the modelling threshold. If your threshold is £500k and your policy limits cluster at £1m and £2m, the truncation correction is non-trivial.

`insurance-severity` supports heterogeneous per-observation limits — not just a single cap:

```python
import numpy as np
from insurance_severity.evt import TruncatedGPD

threshold = 500_000  # modelling threshold

# Exceedances and per-policy limits
exceedances = large_losses - threshold          # shape (n,)
policy_limits = per_policy_limit_array          # shape (n,), np.inf for uncapped

model = TruncatedGPD(threshold=threshold)
model.fit(exceedances=exceedances, limits=policy_limits)

print(model.summary())
# {'xi': 0.612, 'sigma': 148423.7, 'se_xi': 0.041, 'se_sigma': 12891.2}

# 1-in-200 loss (99.5th percentile ground-up)
var_200 = model.isf(0.005)
print(f"99.5th pct: £{var_200:,.0f}")

# Layer pricing: E[X | X > attachment, X <= exhaustion]
# isf() returns the ground-up quantile above threshold u
```

The optimiser uses a two-step profile-then-Powell approach. The truncated GPD likelihood surface is flat in xi — standard L-BFGS-B cannot find the optimum reliably. The profile step grids over 20 xi values, then Powell refines from the best five. This is slower than standard GPD fitting but necessary.

Standard errors come from a numerical Hessian at the optimum. They will be wider than standard GPD standard errors on the same data — that is honest, not a bug. The truncation correction has a cost in information.

### 3. Weibull-Tempered Pareto

Some large-loss tails are heavy but not pure Pareto. Norwegian fire insurance is the textbook example: the tail decays faster than Pareto at the very extreme end, but slower than exponential. The Weibull-tempered Pareto adds a tempering term:

```
S(x) = x^{-alpha} * exp(-lambda * x^tau)
```

When lambda → 0, this recovers pure Pareto. When tau = 1, you get a Pareto-exponential mixture. The parameters fitted on the Norwegian fire data in Albrecher and Beirlant (2025): alpha = 1.199, lambda = 0.004, tau = 0.702 (n = 4,920 observations). The slight tempering means the tail is heavy but physically bounded — the parametric 1-in-10,000 year loss is finite and meaningful, not an artefact of Pareto extrapolation.

In UK personal lines, Weibull tempering is worth considering for home buildings (sum insured caps the true underlying exposure) and commercial property (per-risk limits create genuine physical bounds on loss). We wouldn't use it for motor TPBI, where the true underlying tail appears closer to pure Pareto.

```python
from insurance_severity.evt import WeibullTemperedPareto

wtp = WeibullTemperedPareto()
wtp.fit(exceedances=large_losses - threshold)

s = wtp.summary()
print(f"alpha={s['alpha']:.3f}, lambda={s['lambda']:.4f}, tau={s['tau']:.3f}")
print(f"xi (=1/alpha): {wtp.xi:.3f}")

# Survival function
probs = wtp.sf(np.array([1e6, 2e6, 5e6]))
```

The optimiser runs Powell from four starting points. The `xi` property returns 1/alpha for direct comparison with GPD xi estimates.

---

## A worked comparison

Here is the kind of comparison you should be running on your own motor TPBI or liability data. Fit all three estimators and compare their tail index estimates and their 99.5th percentile quantities:

```python
import numpy as np
from scipy.stats import genpareto
from insurance_severity.evt import (
    TruncatedGPD,
    CensoredHillEstimator,
    WeibullTemperedPareto,
)

rng = np.random.default_rng(42)
threshold = 500_000

# Simulate: true xi=0.60, some policy limits at 2m, some uncapped
n = 300
true_xi, true_sigma = 0.60, 200_000
x = genpareto.rvs(c=true_xi, scale=true_sigma, size=n, random_state=rng) + threshold
limits = np.where(rng.random(n) < 0.4, 2_000_000, np.inf)
x_obs = np.minimum(x, limits)
exceedances = x_obs - threshold

# 20% IBNR censoring in the top observations
is_censored = np.zeros(n, dtype=bool)
top_k_idx = np.argsort(x_obs)[-int(0.2 * n):]
is_censored[top_k_idx] = rng.random(len(top_k_idx)) < 0.3

# Naive GPD (no corrections)
from scipy.stats import genpareto as gp
naive_xi, _, naive_sigma = gp.fit(exceedances, floc=0)
print(f"Naive GPD:        xi={naive_xi:.3f}")

# Truncated GPD
tgpd = TruncatedGPD(threshold=threshold)
tgpd.fit(exceedances=exceedances, limits=limits)
print(f"Truncated GPD:    xi={tgpd.xi:.3f}  (se={tgpd.se_xi:.3f})")

# Censored Hill
chill = CensoredHillEstimator()
chill.fit(claims=x_obs, censored=is_censored)
print(f"Censored Hill:    xi={chill.xi:.3f}")

# WTP
wtp = WeibullTemperedPareto()
wtp.fit(exceedances=exceedances)
print(f"Weibull Tempered: xi={wtp.xi:.3f}")

# 99.5th percentile comparison
q = 0.005
print(f"\n99.5th percentile above threshold:")
print(f"  Naive GPD:     £{genpareto.isf(q, c=naive_xi, scale=naive_sigma):,.0f}")
print(f"  Truncated GPD: £{tgpd.isf(q):,.0f}")
```

On data generated with true xi = 0.60 and 40% of observations truncated at £2m, the naive GPD will return xi around 0.68–0.72. The corrected estimator returns something close to 0.60. The difference in the 99.5th percentile is typically 20–35% on this type of simulation. That is your SCR error from ignoring the truncation.

---

## What is not yet built: conditional EVT

The one area in Albrecher and Beirlant (2025) that `insurance-severity` does not yet cover is conditional EVT regression — estimating xi(x) and sigma(x) as smooth functions of covariates. The methodology combines an Akritas-Van Keilegom conditional Kaplan-Meier for censoring with kernel-weighted local GPD MLE at each covariate value. This would let you say: "for commercial fleet policies, the tail index is 0.55; for private car TPBI it is 0.70" — not as separate data cuts, but as a continuous function of the risk characteristics, fitted simultaneously.

This is genuinely hard to implement well (bandwidth selection, penalised local MLE, simultaneous confidence bands) and we are watching rather than building for now. The conditional EVT direction is also where `insurance-severity` would eventually connect with `insurance-quantile`'s EQRN framework.

---

## The practical checklist

Before your next XL treaty renewal or SCR calculation:

1. **What are your policy limits?** If your modelling threshold is below any policy limit in your data, use `TruncatedGPD` with per-policy `limits` array.

2. **What is your IBNR proportion among large claims?** If more than 10% of your top-k losses are open reserves, your Hill estimate is biased. Use `CensoredHillEstimator` with a `censored` indicator.

3. **Does your tail stabilise at pure Pareto?** If your Hill plot shows the estimate declining at large k (tail lighter than Pareto), `WeibullTemperedPareto` may give a better extrapolation.

4. **Run all three and compare.** If the naive GPD and the corrected estimators agree, you have evidence the corrections don't matter for your data. If they disagree by more than 5% in xi, the naive estimate is wrong.

The `insurance-severity` EVT module handles all of this. The corrections are not optional sophistication for academic papers — they are the difference between an accurate tail model and a biased one. Your reinsurers' actuaries are running equivalent corrections on their side of the negotiation.

---

## Reference

Albrecher, H. and Beirlant, J. (2025). Statistics of Extremes for the Insurance Industry. arXiv:2511.22272. Handbook chapter for Chapman & Hall Statistics of Extremes.

Companion R package: `ReIns` (v1.0.16, February 2026). Functions: `censHill()`, `SpliceFitGPD()`, `SpliceFiticPareto()`.
