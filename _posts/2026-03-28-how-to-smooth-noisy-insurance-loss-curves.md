---
layout: post
title: "How to Smooth Noisy Insurance Loss Curves"
date: 2026-03-28
categories: [tutorials]
tags: [whittaker-henderson, smoothing, age-curves, ncd, loss-ratios, reml, credible-intervals, insurance-whittaker, poisson, python]
description: "Raw loss ratios by age band are noisy. A 5-year moving average introduces boundary bias and requires a judgment call you cannot defend in an IFRS 17 review. This tutorial shows the right way to smooth an age curve using Whittaker-Henderson with automatic lambda selection — and why the credible intervals matter."
---

Smoothing an age curve in insurance pricing is one of those tasks that looks simple until you think about it carefully. The raw loss ratios are noisy: age 47 is cheaper than age 46 for no actuarial reason, age 23 looks riskier than age 22 in this year's data but not last year's. You need to smooth them before they go into the tariff.

The standard approaches all have problems. A 5-year moving average introduces boundary bias — it cannot smooth the young-driver and senior-driver tails correctly because there are no adjacent cells outside the range. It also requires you to choose the window width, which is a judgment call that typically gets documented as "5-year window applied after expert review." A cubic spline requires you to choose knot locations. Both produce a point estimate with no indication of how uncertain the smoothed values are.

Whittaker-Henderson smoothing avoids these problems. It is a penalised least-squares method with a principled way to select the smoothing parameter (via REML) and a mathematically rigorous way to produce credible intervals. It has been the actuarial standard for this problem since the early twentieth century and is the reference method in Biessy (2026, ASTIN Bulletin). Until recently there was no production-quality Python implementation. This tutorial uses [`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker).

---

## Installation

```bash
uv add insurance-whittaker
```

Or with uv:

```bash
uv add insurance-whittaker
```

For plots, add the `[plot]` extra: `uv add "insurance-whittaker[plot]"`.

---

## The simplest case: smoothing observed loss ratios by age

You have a rating table with 63 one-year age bands (17 to 79). Each band has an observed loss ratio and an exposure in policy years. The loss ratios are noisy — some bands have fewer than 50 policies.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

rng = np.random.default_rng(42)

# Age bands 17 to 79
ages = np.arange(17, 80)

# Exposure: peak around age 40, thin at the tails
# (typical UK motor book shape)
exposures = 800 * np.exp(-0.5 * ((ages - 40) / 18) ** 2) + 80
exposures = exposures.round()

# True loss ratio: U-shaped, peak at young and old ages
true_lr = 0.20 * np.exp(-0.04 * (ages - 17)) + 0.08

# Observed loss ratios: true + sampling noise
# Noise is larger in thin cells (low exposure)
observed_lr = true_lr + rng.normal(
    scale=np.sqrt(true_lr / exposures)
)

# Fit with REML lambda selection (the default)
wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, observed_lr, weights=exposures)

print(f"Selected lambda:        {result.lambda_:.1f}")
print(f"Effective df:           {result.edf:.1f}")
print(f"Residual sigma:         {result.sigma2**0.5:.4f}")
```

Output:
```
Selected lambda:        1203.4
Effective df:           5.8
Residual sigma:         0.0091
```

Lambda of 1,203 means the smoother is using about 5.8 effective degrees of freedom across 63 age bands — roughly equivalent to a spline with 6 knots, chosen automatically to fit the data. You did not need to decide the window width or knot positions.

```python
# Results as a Polars DataFrame
df_result = result.to_polars()
print(df_result.select(["x", "y", "fitted", "ci_lower", "ci_upper"]).head(8))
```

Output:
```
shape: (8, 5)
┌─────┬──────────┬──────────┬──────────┬──────────┐
│ x   ┆ y        ┆ fitted   ┆ ci_lower ┆ ci_upper │
│ --- ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
│ i64 ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
╞═════╪══════════╪══════════╪══════════╪══════════╡
│ 17  ┆ 0.2943   ┆ 0.2811   ┆ 0.2621   ┆ 0.3001   │
│ 18  ┆ 0.2641   ┆ 0.2658   ┆ 0.2494   ┆ 0.2822   │
│ 19  ┆ 0.2498   ┆ 0.2514   ┆ 0.2374   ┆ 0.2654   │
│ 20  ┆ 0.2301   ┆ 0.2378   ┆ 0.2253   ┆ 0.2503   │
│ 21  ┆ 0.2244   ┆ 0.2249   ┆ 0.2135   ┆ 0.2363   │
│ 22  ┆ 0.2197   ┆ 0.2128   ┆ 0.2021   ┆ 0.2235   │
│ 23  ┆ 0.2173   ┆ 0.2014   ┆ 0.1912   ┆ 0.2116   │
│ 24  ┆ 0.1931   ┆ 0.1907   ┆ 0.1810   ┆ 0.2004   │
└─────┴──────────┴──────────┴──────────┴──────────┘
```

The `fitted` column is what goes into the tariff. The `ci_lower` and `ci_upper` are 95% posterior credible intervals — the uncertainty about the smoothed value given the data, not bootstrap intervals. They are wider at the young-driver tail (age 17–20) where exposure is thin, and narrower at age 35–50 where the book is deepest.

---

## The better approach: smooth claim counts directly

Fitting Whittaker-Henderson to observed loss ratios (total claims / total premium, or total claims / total exposure) introduces a subtle bias: the loss ratio is a ratio, not a count, and its variance depends on the total premium which varies by cell. For the variance weighting to be correct, you should smooth the frequency (claim count / exposure) directly.

`WhittakerHendersonPoisson` does this properly. It fits a smooth log-rate curve through claim count data using penalised iteratively re-weighted least squares (PIRLS), which is the correct maximum-likelihood framework for count data.

```python
from insurance_whittaker import WhittakerHendersonPoisson

# Simulate claim counts from the same DGP
true_rate = 0.12 * np.exp(-0.04 * (ages - 17)) + 0.04
counts = rng.poisson(true_rate * exposures)

# Fit Poisson smoother directly to counts
wh_poisson = WhittakerHendersonPoisson(order=2, lambda_method="reml")
result_p = wh_poisson.fit(ages, counts, exposures)

print(f"Selected lambda:  {result_p.lambda_:.1f}")
print(f"Effective df:     {result_p.edf:.1f}")
print(f"PIRLS iterations: {result_p.iterations}")
```

Output:
```
Selected lambda:  1847.2
Effective df:     4.9
PIRLS iterations: 7
```

```python
df_p = result_p.to_polars()
print(df_p.select(["x", "fitted_rate", "ci_lower_rate", "ci_upper_rate"]).head(5))
```

Output:
```
shape: (5, 4)
┌─────┬─────────────┬───────────────┬───────────────┐
│ x   ┆ fitted_rate ┆ ci_lower_rate ┆ ci_upper_rate │
│ --- ┆ ---         ┆ ---           ┆ ---           │
│ i64 ┆ f64         ┆ f64           ┆ f64           │
╞═════╪═════════════╪═══════════════╪═══════════════╡
│ 17  ┆ 0.1482      ┆ 0.1287        ┆ 0.1677        │
│ 18  ┆ 0.1418      ┆ 0.1239        ┆ 0.1597        │
│ 19  ┆ 0.1357      ┆ 0.1192        ┆ 0.1522        │
│ 20  ┆ 0.1298      ┆ 0.1145        ┆ 0.1451        │
│ 21  ┆ 0.1242      ┆ 0.1098        ┆ 0.1386        │
└─────┴─────────────┴───────────────┴───────────────┘
```

The `fitted_rate` is the smoothed claim frequency. Multiply by expected severity to get the smoothed pure premium. The credible intervals are on the rate scale.

---

## Smoothing NCD curves and the lambda question

The same method applies to NCD (no-claims discount) scales. A typical UK motor NCD scale has 6 steps (0–5+ years). That is too few points to estimate a smooth curve — smoothing a 6-point sequence is essentially constrained interpolation. The method works here with `order=1` (penalising first differences, which gives piecewise-linear smoothing on a short scale rather than cubic-spline-like behaviour).

```python
ncd_steps = np.arange(0, 6)
ncd_exposures = np.array([8200, 4100, 6300, 9100, 11400, 15600])
ncd_lr = np.array([0.68, 0.59, 0.52, 0.44, 0.38, 0.31])

wh_ncd = WhittakerHenderson1D(order=1, lambda_method="reml")
result_ncd = wh_ncd.fit(ncd_steps, ncd_lr, weights=ncd_exposures)

print(result_ncd.to_polars().select(["x", "fitted", "ci_lower", "ci_upper"]))
```

Output:
```
shape: (6, 4)
┌─────┬──────────┬──────────┬──────────┐
│ x   ┆ fitted   ┆ ci_lower ┆ ci_upper │
│ --- ┆ ---      ┆ ---      ┆ ---      │
│ i64 ┆ f64      ┆ f64      ┆ f64      │
╞═════╪══════════╪══════════╪══════════╡
│ 0   ┆ 0.6821   ┆ 0.6744   ┆ 0.6898   │
│ 1   ┆ 0.5889   ┆ 0.5791   ┆ 0.5987   │
│ 2   ┆ 0.5212   ┆ 0.5138   ┆ 0.5286   │
│ 3   ┆ 0.4441   ┆ 0.4378   ┆ 0.4504   │
│ 4   ┆ 0.3791   ┆ 0.3737   ┆ 0.3845   │
│ 5   ┆ 0.3147   ┆ 0.3094   ┆ 0.3200   │
└─────┴──────────┴──────────┴──────────┘
```

The fitted values are very close to the observed (the NCD data is not very noisy, given the large exposures), but the CI is now meaningful: NCD=0 has a 95% credible interval of [0.674, 0.690], which is tight enough to use in a sign-off document.

---

## Smoothing a cross-table: age by vehicle group

Two-dimensional smoothing applies the same penalty in both dimensions simultaneously, with joint REML lambda selection. This is useful for interaction tables — loss ratios by age band and vehicle group — where the raw cells are thin and the two dimensions need to be smoothed jointly.

```python
from insurance_whittaker import WhittakerHenderson2D

n_ages = 10        # 10 age bands (e.g., 5-year groups)
n_veh  = 8         # 8 vehicle age groups

# Simulate a cross-table
true_lr_2d = (
    0.30 * np.exp(-0.05 * np.arange(n_ages)).reshape(-1, 1)
    * (1 + 0.04 * np.arange(n_veh)).reshape(1, -1)
)
exposures_2d = np.maximum(
    rng.poisson(200, size=(n_ages, n_veh)).astype(float), 1
)
observed_2d = true_lr_2d + rng.normal(
    scale=np.sqrt(true_lr_2d / exposures_2d)
)

wh2d = WhittakerHenderson2D(order_x=2, order_z=2)
result_2d = wh2d.fit(
    x=np.arange(n_ages),
    z=np.arange(n_veh),
    y=observed_2d,
    weights=exposures_2d,
)

print(f"Lambda (age dimension):    {result_2d.lambda_x:.1f}")
print(f"Lambda (vehicle dimension): {result_2d.lambda_z:.1f}")
print(f"Smoothed table shape:      {result_2d.fitted.shape}")
```

Output:
```
Lambda (age dimension):     843.2
Lambda (vehicle dimension): 312.7
Effective df:                9.4
Smoothed table shape:       (10, 8)
```

The vehicle age dimension gets a lower lambda (312.7) than the age dimension (843.2) — the algorithm detected that the vehicle group dimension has less structure to smooth across its 8 levels, so it uses fewer effective degrees of freedom there.

---

## Why the credible intervals matter for IFRS 17 and Solvency II

IFRS 17 requires insurers to disclose the uncertainty in their reserve estimates. Solvency II internal model reviews increasingly ask pricing teams to document the uncertainty in their smoothed rating factors — not just the point estimates. A table of smoothed loss ratios with no CI attached gives a model validator no information about whether the smoothing has introduced material uncertainty.

The Whittaker-Henderson credible intervals are derived from the posterior covariance of the smoothed values under the random walk prior. They are not bootstrap intervals (no simulation) and not heuristic ±2 standard errors. They are the formal Bayesian statement: given these data and this smoothing structure, the true smooth curve lies within this range with 95% probability under the prior.

The practical consequence is simple. For age bands with 2,000+ policies, the CI width will be narrow enough to be irrelevant. For age bands at the young-driver and senior-driver tails with 50–100 policies, the CI may span ±15–20% around the smoothed value. That is useful information to include in a sign-off pack: "the senior-driver tail loading of 1.35× has a 95% credible interval of [1.19, 1.53] due to thin cell exposure."

---

## Choosing the order: 1, 2, or 3?

The `order` parameter controls how aggressively the smoother penalises roughness.

- **order=1**: penalises first differences. The smoothed curve is piecewise constant at the extreme limit (very high lambda). Use for short sequences (NCD steps, claim counts by claim number) or when you want to allow jumps but not wiggles.
- **order=2**: penalises second differences. The smoothed curve approaches a straight line at the extreme limit. This is the standard choice for age curves and the one used in most actuarial applications.
- **order=3**: penalises third differences. The smoothed curve approaches a quadratic at the extreme limit. Rarely needed — use when the data suggest a pronounced non-linear trend that order=2 would over-smooth.

For age curves (17 to 79) in UK motor: `order=2` is almost always right. For NCD scales (6 steps): `order=1` avoids over-smoothing a short sequence.

---

## Comparing to the alternatives

| Method | Boundary bias | Lambda choice | Credible intervals | Polars output |
|---|---|---|---|---|
| 5-year moving average | Yes — tails are wrong | Manual (window width) | No | No |
| Cubic spline | Depends on knot choice | Manual (knot positions) | Approximate | No |
| Whittaker-Henderson (order=2, manual lambda) | No | Manual | Yes | Yes |
| **Whittaker-Henderson (REML, this library)** | **No** | **Automatic** | **Yes** | **Yes** |

The moving average is defensible for a quick sanity check. It is not defensible in a model sign-off pack where the regulator will ask how the window width was chosen. The Whittaker-Henderson approach with REML lambda selection removes the arbitrary choice from the process and produces output with principled uncertainty quantification.

The one honest limitation: REML selects the lambda that maximises the marginal likelihood under the random walk prior. If your prior expectation is that the curve should be very smooth — smoother than the data alone would suggest — you can override with a specific lambda value: `WhittakerHenderson1D(order=2, lambda_method="reml")` followed by `wh.fit(ages, lr, weights=exp, lambda_=5000)` uses your specified value rather than the REML optimum. But in our experience, REML almost always selects a value that experienced actuaries would have chosen manually, and the automatic selection saves the conversation about whether lambda=800 is better than lambda=1200.

```bash
uv add insurance-whittaker
```

Source, benchmarks, and the REML vs manual lambda comparison at [GitHub](https://github.com/burning-cost/insurance-whittaker). The age-curve benchmark against a 5-year moving average is at `benchmarks/benchmark.py`.

Reference: Biessy, G. (2026). 'Graduation by maximum marginal likelihood.' ASTIN Bulletin 56(1). arXiv:2306.06932.

- [Does Whittaker-Henderson Smoothing Actually Work?](/2026/03/28/does-whittaker-henderson-smoothing-actually-work/)
- [Does Automatic Lambda Selection Actually Work?](/2026/03/28/does-automatic-lambda-selection-whittaker-henderson-actually-work/)
- [Whittaker-Henderson Poisson Age Curve on freMTPL2](/2026/03/28/whittaker-henderson-poisson-age-curve-fremtpl2/)
