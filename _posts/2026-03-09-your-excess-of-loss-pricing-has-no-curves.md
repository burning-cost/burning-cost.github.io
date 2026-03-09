---
layout: post
title: "Your Excess-of-Loss Pricing Has No Curves"
date: 2026-03-09
categories: [reinsurance, libraries, pricing]
tags: [mbbefd, exposure-rating, ilf, xl-pricing, swiss-re, bernegger, london-market, reinsurance, python, insurance-ilf]
description: "Without exposure curves, per-risk XL pricing is guesswork dressed up as arithmetic. insurance-ilf brings MBBEFD distributions, Swiss Re curve families, ILF tables, truncated/censored MLE fitting, and the Lee diagram to Python for the first time."
---

Per-risk excess-of-loss pricing has a dirty secret: for many of the teams doing it, the exposure rating is not really exposure rating. It is experience rating applied to aggregate loss triangles, with attachment factors drawn from a table that someone produced six years ago and nobody has refitted since.

The correct approach — fitting an MBBEFD exposure curve to destruction-rate data, deriving Increased Limits Factors analytically, and pricing the XL layer as the integral of the severity distribution across the layer — has been standard actuarial practice since Stefan Bernegger's 1997 ASTIN paper. The Swiss Re curve family, which parameterises the whole of MBBEFD space with a single scalar c, has been in the Lloyd's and London market toolbox for decades. None of this is new.

What is new is doing it in Python. Until now, the only open-source implementation of the MBBEFD distribution family was the `mbbefd` package in R. Python had nothing.

[`insurance-ilf`](https://github.com/burning-cost/insurance-ilf) is our 24th open-source library, and our first in commercial lines and London market reinsurance. Five modules, 129 tests, v0.1.0 on PyPI.

```bash
pip install insurance-ilf
```

---

## Why exposure rating matters here

Experience rating relies on historical loss frequency to estimate future losses. For per-risk property XL, where the layer only responds to large losses and large losses are rare by construction, it falls apart. If your five-year burning cost on a £1m xs £2m layer is zero because no single risk penetrated the layer in that period, you have learned almost nothing. The layer is not risk-free — it is just infrequently hit.

Exposure rating replaces the burning cost calculation with a parametric model of the severity distribution. Instead of "what did losses look like in the past," you ask "what fraction of the underlying severity distribution falls inside the XL layer?" The answer is the exposure curve.

---

## The MBBEFD distribution

The MBBEFD family was introduced by Bernegger (1997) as a framework for modelling the destruction rate — the fraction of maximum possible loss (MPL) that a given loss represents. If a factory worth £10m burns for £3m, the destruction rate is 0.3.

The distribution is mixed: continuous density on [0, 1) and a point mass at x = 1 for total losses. The exposure curve for MBBEFD(g, b) is:

```
G(x) = ln[(g-1)b/(1-b) + (1-gb)/(1-b) * b^x] / ln(gb)
```

G(x) is the proportion of expected loss below fraction x of MPL — concave, running from G(0) = 0 to G(1) = 1. The concavity encodes right-skewed severity: most losses are small relative to MPL. The total loss probability is 1/g.

```python
from insurance_ilf import MBBEFDDistribution

# Swiss Re Y2 curve (c=2.0) — medium commercial property
dist = MBBEFDDistribution(g=7.69, b=9.02)

print(dist.total_loss_prob())    # 0.130 — 13% of losses are total
print(dist.exposure_curve(0.5)) # G(0.5) — EL fraction below 50% of MPL
```

---

## Swiss Re curves: the London market baseline

Swiss Re parameterised a one-dimensional subfamily of MBBEFD using a single scalar c:

```
b = exp(3.1 - 0.15c(1 + c))
g = exp(c(0.78 + 0.12c))
```

The five standard curves span from Y1 (c=1.5, light manufacturing, sprinklered) through Y2 (c=2.0, warehousing), Y3 (c=3.0, heavy industrial), Y4 (c=4.0, petrochemical) to Lloyd's (c=5.0, industrial complexes). Higher c concentrates more mass toward total loss. When you have no claims data to fit, you pick the curve matching your risk class and use it directly. This is the standard London market approach.

```python
from insurance_ilf import swiss_re_curve, all_swiss_re_curves

y2 = swiss_re_curve(2.0)
y4 = swiss_re_curve(4.0)

# G(0.5) for Y2 vs Y4: how much EL sits below 50% of MPL
print(y2.exposure_curve(0.5))  # higher — mass concentrated at small fractions
print(y4.exposure_curve(0.5))  # lower — mass concentrated near total loss

curves = all_swiss_re_curves()  # dict: 'Y1', 'Y2', 'Y3', 'Y4', 'Lloyds'
```

---

## Fitting to claims data

When you have sufficient history — Bernegger suggests 50+ large losses — you fit an MBBEFD directly via MLE. The log-likelihood surface is non-convex in (g, b), so we use multi-start L-BFGS-B with six starting points drawn from the Swiss Re c-parameter grid plus moment-matching and random perturbations. If L-BFGS-B fails, we fall back to differential evolution as a global search.

```python
from insurance_ilf import fit_mbbefd, swiss_re_curve
import numpy as np

true_dist = swiss_re_curve(3.0)
data = true_dist.rvs(300, rng=np.random.default_rng(42))

result = fit_mbbefd(data, method='mle')
print(result)
# FittingResult(g=18.3, b=3.1, loglik=-87.4, aic=178.8, converged=True)
```

### The truncation and censoring problem

Real claims data is subject to deductibles (left truncation) and policy limits (right censoring). A deductible means losses below it are absent from your data entirely, not just unobserved. Fitting on raw data without accounting for this biases the severity distribution away from the tails — which is precisely where XL pricing cares.

The correct likelihood follows Aigner, Hirz and Leitinger (2024), *European Actuarial Journal*. For deductible d and policy limit c as fractions of MPL:

```
log L(θ | Y) = log f_θ(Y) × 1{Y < c}
             + log(1 - F_θ(c⁻)) × 1{Y = c}
             - log(1 - F_θ(d))
```

The final term normalises for truncation — you condition on the event {X > d} because smaller losses are not in your data.

```python
result = fit_mbbefd(
    data,
    truncation=0.10,  # deductible = 10% of MPL
    censoring=0.80,   # policy limit = 80% of MPL
)
```

We implemented this from scratch. On a book with a £50k deductible and £10m limit fitted against a Swiss Re Y3 curve, ignoring the truncation correction can shift the implied attachment loading by several percentage points. For a large treaty, that is real money.

---

## ILF tables

An Increased Limits Factor is the ratio of limited expected values: ILF(L, B) = G(L/MPL) / G(B/MPL), with the basic limit B as denominator. The ILF at L = B is 1.0 by construction.

```python
from insurance_ilf import ilf_table

dist = swiss_re_curve(2.0)
table = ilf_table(
    dist=dist,
    limits=[500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000],
    basic_limit=1_000_000,
    mpl=5_000_000,
)
print(table)
```

```
      limit       lev    ilf  marginal_ilf
0   500000   0.05893  0.734         0.734
1  1000000   0.08028  1.000         0.266
2  2000000   0.11026  1.374         0.374
3  3000000   0.12879  1.605         0.231
4  5000000   0.15271  1.903         0.298
```

The marginal ILF declines with each successive layer: 37.4% for the £1m–£2m slice, 23.1% for £2m–£3m. This is the concavity of the exposure curve — each pound of coverage at a higher limit carries less expected loss than the one before it.

---

## Pricing the XL layer

The per-risk XL pricing formula from Clark (2014, CAS Study Note) for a layer [AP, AP+Limit] on a risk with policy limit PL and MPL:

```
EL = SP × [G(min(PL, AP+Limit)/MPL) - G(min(PL, AP)/MPL)]
```

The min(PL, ...) terms are load-bearing: the reinsurer cannot recover beyond the policy limit regardless of where the layer ceiling sits.

In practice, you work from a risk profile table — risks grouped into sum-insured bands with counts and subject premiums.

```python
import pandas as pd
from insurance_ilf import per_risk_xl_rate, swiss_re_curve

profile = pd.DataFrame({
    'sum_insured': [500_000, 1_000_000, 2_000_000],
    'premium':     [ 50_000,    80_000,    60_000],
    'count':       [    100,        80,        30],
})

result = per_risk_xl_rate(
    risk_profile=profile,
    dist=swiss_re_curve(2.0),
    attachment=1_000_000,
    limit=1_000_000,
)

print(f"Technical rate: {result['technical_rate']:.4f}")  # 0.0412
print(f"Expected loss:  £{result['total_expected_loss']:,.0f}")  # £7,740
```

Risks with sum insured below the attachment contribute nothing to the layer — their policy limits sit below the retention, and the exposure curve difference is zero. The library handles this correctly across any risk profile.

---

## The Lee diagram

The Lee diagram ranks individual losses by destruction rate and plots cumulative loss proportion against cumulative risk proportion. A straight diagonal would mean all risks contribute equally — which does not happen in property insurance. The curve bends above the diagonal in proportion to how concentrated large losses are.

```python
from insurance_ilf import lee_diagram
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
lee_diagram(losses=claim_amounts, mpl=mpls, dist=fitted_dist, ax=ax)
```

If the fitted MBBEFD G(x) tracks the empirical curve well, your parametric model captures the concentration correctly. If the data sits consistently above the fitted curve at large destruction rates, move to a heavier Swiss Re curve or fit with a higher c. The Lee diagram is also the standard tool for explaining to cedants why their largest risks dominate XL loss experience.

The `GoodnessOfFit` class handles the formal diagnostics — KS test and Anderson-Darling test on the continuous part, PP and QQ plot data — and the `compare_curves` function overlays multiple MBBEFD curves on a single plot for visual comparison.

---

## What this is for

`insurance-ilf` makes the London market per-risk XL pipeline scriptable: read a risk profile, fit MBBEFD with truncated/censored likelihood, compare against the Swiss Re family, build the ILF table, price the layer, plot the Lee diagram. All in versioned, testable Python.

The Aigner et al. (2024) truncated/censored likelihood is the part we are most satisfied with. It is mathematically straightforward but consistently omitted in practice — including in the original R `mbbefd` package. Omitting it produces biased parameters. For XL pricing, biased severity parameters mean systematically mispriced attachment loadings. This is the gap the library closes.

---

`insurance-ilf` is open source under the MIT licence at [github.com/burning-cost/insurance-ilf](https://github.com/burning-cost/insurance-ilf). Install with `pip install insurance-ilf`. 129 tests passing. Requires Python 3.10+, NumPy, SciPy, and Pandas.
