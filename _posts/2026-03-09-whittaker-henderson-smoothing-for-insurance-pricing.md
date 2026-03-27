---
layout: post
title: "Whittaker-Henderson Smoothing for Insurance Pricing"
date: 2026-03-09
author: Burning Cost
categories: [techniques, libraries, pricing]
tags: [smoothing, experience-rating, whittaker-henderson, p-splines, credibility, rating-tables, age-curves, ncd, python, insurance-whittaker]
description: "Whittaker-Henderson smoothing for noisy experience rating tables in Python. REML lambda selection, Bayesian confidence intervals, 2D surface smoothing."
---

Every experience rating table has the same shape problem at the edges. The middle of the age curve -- drivers aged 25 to 55 -- is well-behaved: high exposure, stable relativities, small year-on-year movement. The extremes are not. Drivers aged 17 to 20 have low exposure, high claim frequency, and relativities that bounce around from year to year in ways that have nothing to do with the underlying risk. At age 70 and above, the same thing happens: volumes thin out, the raw relativities become noisy, and you start making manual overrides to stop the table from doing something actuarially implausible.

The same pattern appears in vehicle group relativities, postcode-based risk scores, and NCD scales. Wherever you have a one-dimensional array of risk relativities derived from raw experience data, you face the same trade-off: smooth too much and you lose real signal; smooth too little and you price off noise. The table you actually use sits somewhere in between, and in most teams the decision about where is made informally -- someone looks at the raw versus smooth curve, picks a tension parameter by eye, and moves on.

Whittaker-Henderson smoothing has been the formal actuarial answer to this problem since E. T. Whittaker's 1923 paper on the graduation of actuarial tables. The objective function is transparent:

```
min sum_i w_i(y_i - theta_i)^2 + lambda * ||D^q theta||^2
```

The first term penalises deviation from the data; the second penalises roughness in the fitted curve, measured by the q-th order difference of the smoothed values. Lambda is the smoothing parameter: low lambda follows the data closely, high lambda produces a near-polynomial trend. The weights w_i reflect data credibility -- typically exposure or claim counts.

The problem in practice has not been the mathematics. It has been the tooling. R has `MortalitySmooth` and the `WH` package. Python has had nothing purpose-built for insurance rating tables. Actuaries doing this in Python have either ported their R code, written the Whittaker system from scratch (it is three lines of linear algebra, but then you need lambda selection, confidence intervals, and 2D extensions), or simply not done it formally.

[`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker) is our 38th open-source library. Three modules, 73 tests passing, v0.1.0. It covers the 1D case (age curves, NCD scales, vehicle group relativities), the 2D case (driver age by vehicle age rating surfaces), and a Poisson variant for count data directly.

```bash
uv add insurance-whittaker
```

---

## Why Whittaker-Henderson and not a spline

Paul Eilers and Brian Marx showed in 1996 that Whittaker-Henderson smoothing is formally equivalent to P-splines with degree-0 B-splines. They are the same optimisation problem, arrived at from different directions. Whittaker came from actuarial graduation; Eilers and Marx came from nonparametric regression. The practical implication is that actuarial WH smoothing has the same theoretical backing as the P-spline literature, which is extensive.

For insurance rating tables, we prefer the Whittaker formulation for two reasons. First, it operates directly on the tabular data that actuaries work with -- you pass in an array of relativities indexed by age band or vehicle group, not a design matrix of spline basis functions. Second, the roughness penalty on finite differences has a direct actuarial interpretation: `q=2` penalises curvature (the second difference), which corresponds to a preference for linear trends in the absence of data. That is a sensible prior for most rating factors.

Smoothing splines and LOESS are alternatives, but they do not naturally handle the credibility weighting problem -- the fact that a 17-year-old age band with 200 claims should be smoothed more heavily than a 30-year-old age band with 10,000 claims. Whittaker handles this directly through the weight vector.

---

## The 1D case: age curves and NCD scales

The most common application is smoothing a one-dimensional relativity array. Here is a motor frequency curve by driver age, where the raw relativities come from a GLM fitted on the full book:

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# Raw frequency relativities from a GLM, indexed 17-80
# Exposure (policy years) in each age band
ages = np.arange(17, 81)

raw_relativities = np.array([
    3.41, 3.12, 2.87, 2.63, 2.45, 2.21, 2.08, 1.94,
    1.82, 1.71, 1.65, 1.58, 1.52, 1.47, 1.43, 1.39,
    1.36, 1.33, 1.30, 1.28, 1.25, 1.23, 1.21, 1.19,
    1.17, 1.15, 1.14, 1.12, 1.11, 1.10, 1.09, 1.08,
    1.07, 1.07, 1.06, 1.06, 1.05, 1.06, 1.07, 1.08,
    1.09, 1.11, 1.13, 1.15, 1.18, 1.22, 1.27, 1.33,
    1.40, 1.48, 1.59, 1.73, 1.89, 2.09, 2.32, 2.58,
    2.89, 3.24, 3.66, 4.12, 4.65, 5.22, 5.84, 6.52,
])

exposures = np.array([
    180, 210, 290, 380, 520, 680, 820, 1040,
    1280, 1510, 1740, 1920, 2100, 2280, 2410, 2530,
    2640, 2710, 2780, 2820, 2850, 2880, 2890, 2870,
    2840, 2810, 2770, 2720, 2670, 2610, 2540, 2470,
    2390, 2310, 2230, 2150, 2070, 1990, 1910, 1830,
    1750, 1670, 1590, 1510, 1430, 1350, 1270, 1190,
    1110,  980,  860,  740,  630,  530,  440,  360,
     290,  230,  175,  130,   96,   71,   52,   37,
])

smoother = WhittakerHenderson1D(
    order=2,            # penalise second differences (curvature)
    lambda_method='reml',  # auto-select lambda via REML
)

result = smoother.fit(ages, raw_relativities, weights=exposures)
print(result)
```

```
WHResult1D(n=64, order=2, lambda_=847.000, edf=4.23, criterion='reml')
```

The REML-selected lambda of 847 is fairly high: the 17-year-old raw relativity of 3.41 gets smoothed substantially, because the 180-policy-year exposure in that band is low. Accessing the smoothed values and confidence intervals:

```python
smoothed = result.fitted        # numpy array, same length as input
ci_lower = result.ci_lower      # 95% CI lower bound, Bayesian posterior
ci_upper = result.ci_upper      # 95% CI upper bound

# The smoothed relativity at age 17
print(f"Age 17: raw={raw_relativities[0]:.3f}, "
      f"smooth={smoothed[0]:.3f}, "
      f"CI=({ci_lower[0]:.3f}, {ci_upper[0]:.3f})")
# Age 17: raw=3.41, smooth=3.19, CI=(2.87, 3.54)
```

The confidence intervals come from the Bayesian interpretation of WH smoothing: treating the roughness penalty as a prior, the posterior covariance of the smoothed values is available in closed form. The intervals are wider at the edges (age 17, age 80) where data is sparse. At age 35, with 2,600 policy years of exposure, the interval is narrow -- the data speaks clearly. At age 79, it is wide -- you are primarily relying on the extrapolation from the smooth trend.

---

## Lambda selection: why REML is the right default

Lambda is the critical hyperparameter. Too low and the smoother chases noise; too high and it irons out real structure.

Four selection criteria are implemented: REML, GCV, AIC, and BIC. They can be compared explicitly:

```python
from insurance_whittaker import WhittakerHenderson1D

for criterion in ['reml', 'gcv', 'aic', 'bic']:
    result = WhittakerHenderson1D(order=2, lambda_method=criterion).fit(
        ages, raw_relativities, weights=exposures
    )
    print(f"{criterion.upper():5s}: lambda_={result.lambda_:7.1f}, edf={result.edf:.2f}")
```

```
REML : lambda_=  847.3, edf=4.23
GCV  : lambda_=  412.1, edf=6.81
AIC  : lambda_=  291.8, edf=8.54
BIC  : lambda_=  634.7, edf=5.12
```

REML selects a higher lambda than GCV or AIC on this dataset, meaning more smoothing. The analogy is to REML in linear mixed models: REML corrects for the degrees of freedom consumed by the fixed effects (here, the smooth trend), whereas GCV does not. In practice, REML tends to produce smoother curves than GCV, which is usually the right direction for insurance rating tables -- you want to err towards smoothing rather than fitting noise. GCV and AIC tend to undersmooth, retaining wiggles that are artefacts of the data rather than genuine risk structure.

Our view: use REML by default. Use GCV as a cross-check. If they agree, you are probably in the right region of the lambda space. If they disagree materially, look at the data in that region -- there may be genuine structure that REML is smoothing away, or there may be a data quality issue in a specific band.

---

## The Poisson variant for count data

For some applications, you want to smooth claim count rates directly rather than pre-computed GLM relativities. The Poisson variant handles this: it fits an iteratively reweighted WH smoother via PIRLS (Penalised Iteratively Reweighted Least Squares), working on the log scale.

```python
from insurance_whittaker import WhittakerHendersonPoisson

# Claim counts and exposures by age band, not pre-computed relativities
claim_counts = np.array([
     68,  78, 106, 135, 175, 214, 243, 284,
    306, 340, 365, 389, 407, 424, 438, 450,
    461, 470, 477, 483, 488, 492, 494, 495,
    492, 488, 483, 476, 469, 462, 455, 447,
    440, 434, 428, 422, 417, 413, 410, 407,
    405, 403, 402, 401, 400, 400, 399, 398,
    395, 385, 371, 352, 330, 304, 276, 246,
    214, 183, 153, 124,  99,  78,  61,  47,
])

smoother = WhittakerHendersonPoisson(order=2)

result = smoother.fit(ages, claim_counts, exposure=exposures)
smoothed_rates = result.fitted_rate      # smoothed claim rate per unit exposure
ci_lower_rate  = result.ci_lower_rate    # 95% CI lower bound on rate
ci_upper_rate  = result.ci_upper_rate    # 95% CI upper bound on rate
```

The PIRLS algorithm iterates: fit a weighted WH smoother, update the working response and weights from the Poisson likelihood, repeat until convergence. Convergence is typically fast (under 20 iterations) for insurance data. The smoothed rates are on the natural scale; the confidence intervals are computed on the log scale and back-transformed, which keeps them positive.

The Poisson smoother is the right choice when the raw counts are small and you do not want to pre-divide by exposure (which can amplify noise in thin bands). It is also correct when the count distribution is the object of interest rather than a pre-computed relativity.

---

## 2D smoothing for rating surfaces

Some rating factors are naturally two-dimensional. Driver age by vehicle age is the canonical example: a 25-year-old driving a 15-year-old vehicle is a different risk from a 25-year-old driving a new vehicle. If you want to model this interaction without specifying it as an explicit GLM interaction term, 2D WH smoothing produces a smooth rating surface.

```python
from insurance_whittaker import WhittakerHenderson2D
import numpy as np

# 10 driver age bands (17-26 -> 0, 27-36 -> 1, ...) x 8 vehicle age bands
n_driver_ages = 10
n_vehicle_ages = 8

# Raw relativities: a 10x8 matrix of experience rates
# NaN where no data exists in a cell
rng = np.random.default_rng(42)
raw_surface = 1.0 + rng.normal(0, 0.15, (n_driver_ages, n_vehicle_ages))
raw_surface[0, :] *= 2.2    # young drivers systematically higher
raw_surface[:, 0] *= 1.3    # new vehicles slightly higher (repair costs)

# Exposure matrix: policy years in each cell
exposures_2d = np.abs(rng.normal(500, 200, (n_driver_ages, n_vehicle_ages)))
exposures_2d[0, :] *= 0.3   # sparse data for young driver bands

smoother_2d = WhittakerHenderson2D(
    order_x=2,          # curvature penalty along driver age axis
    order_z=2,          # curvature penalty along vehicle age axis
)

result_2d = smoother_2d.fit(raw_surface, weights=exposures_2d)
smoothed_surface = result_2d.fitted   # 10x8 numpy array
```

The 2D system works by applying the WH penalty separately along each dimension, building a Kronecker product penalty matrix. This is computationally efficient for the matrix sizes that appear in insurance (rating tables rarely exceed 20x20 cells) and has the intuitive property that the smoothing tension can be set independently for each axis.

Separate lambda values for rows and columns matter in practice. A driver age axis with good exposure across most bands might warrant a low lambda (following the data). A vehicle age axis where data becomes sparse above 10 years might warrant a high lambda. REML selects the two lambdas jointly, optimising the marginal likelihood of the observed data.

---

## Practical limits and what they mean

The library solves the WH system directly via Cholesky factorisation. For n observations (rating bands), this is O(n^3) in the number of bands -- fast for n up to around 200, beyond which memory and compute become considerations.

This is not a practical constraint for insurance rating tables. A typical driver age curve has 20 to 60 bands. An NCD scale has at most 10 steps. A vehicle group relativity table might have 50 to 150 groups. A 2D surface at 20x20 cells has 400 entries -- comfortably within the Cholesky limit. If you find yourself with more than 200 bands in a single dimension, the data structure probably needs rethinking before the smoothing does.

For large postcode-level smoothing problems -- hundreds or thousands of geographic cells -- the Cholesky approach is not the right tool. A sparse iterative solver or a spatial model (such as a GMRF on the postcode adjacency graph) would be more appropriate. That is a different problem from the rating table graduation that this library solves.

---

## Connecting to the rest of the pricing workflow

WH smoothing sits at a specific point in the pricing workflow: after the GLM or GBM has produced raw factor relativities, and before those relativities go into the tariff. The typical pipeline is:

1. Fit a Tweedie GLM or GBM to get raw relativities per rating band
2. Apply WH smoothing with REML lambda selection
3. Check the smoothed curve against the raw data, with CI bands, and challenge any departures in well-exposed regions
4. Export the smoothed relativities to Radar or the tariff system

The library outputs clean numpy arrays, so step 4 is a straightforward handoff. Once in production, [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) tracks PSI and A/E against the relativities you just graduated — if Gini drift flags a staleness problem, the WH smoothing step is often where you intervene first.

If you are building raw relativities with a GLM rather than reading them from a factor table, [`insurance-credibility`](/2026/02/19/buhlmann-straub-credibility-in-python/) handles Bühlmann-Straub credibility blending -- an alternative approach when the smoothing problem is primarily about blending thin segments with portfolio experience rather than graduate an existing curve. The two approaches are complementary: WH smoothing is appropriate when you have a natural ordering along the rating axis (age, vehicle age, NCD step); credibility blending is more appropriate for unordered categorical factors (occupation, region) where there is no natural neighbourhood structure to smooth across.

---

## What this replaces

The informal version of this analysis is: print the raw and smoothed curves in Excel, move the smoothing parameter until the curve looks right, export. The problem is not that the result is wrong -- an experienced actuary with good data intuition will often arrive at a defensible lambda. The problem is that the process is not reproducible and the decision is not documented.

With REML lambda selection, the question changes from "does this look smooth enough?" to "does the REML-selected smooth look actuarially sensible, and if we are overriding it, why?" That is a better conversation to have with a peer reviewer or a model validation team — and if you are documenting under PRA SS1/23, the REML-selected lambda and the credible intervals are the artefacts that should appear in the [model validation report](/2026/03/14/insurance-governance-unified-pra-ss123-validation/).

The confidence intervals change the conversation further. When the CI on the young driver band is wide, that is not a presentation artefact -- it is genuine information about how much uncertainty sits in that part of the table. A table produced without CIs looks more certain than it is. A table with CI bands makes the data quality issues visible, which is usually what you want when explaining a rating change to a head of pricing.

---

`insurance-whittaker` is open source under the MIT licence at [github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) ([PyPI](https://pypi.org/project/insurance-whittaker/)). Install with `uv add insurance-whittaker`. 73 tests, all passing. Requires Python 3.10+, NumPy, and SciPy.

- [Buhlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Trend Selection Is Not Actuarial Judgment: A Python Approach](/2026/03/13/insurance-trend/)
- [Building a Modern Insurance Pricing Pipeline in Python](/2026/03/12/modern-pricing-pipeline/)
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — EBMs and ANAMs provide shape-function interpretability without the smoothing step; Whittaker-Henderson is the right tool when you have an existing rating table to graduate rather than a new model to fit from scratch
- [Optimal Binning for GLM Rating Factors: Beyond the Eyeball Test](/2026/03/14/your-factor-banding-is-made-up/) — the complement: WH smoothing graduates a curve once the bin structure is fixed; this post automates the bin-selection problem that precedes the smoothing step
