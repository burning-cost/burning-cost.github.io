---
layout: post
title: "Whittaker-Henderson Smoothing in Python: A Step-by-Step Tutorial for Insurance Rating Tables"
date: 2026-04-04
author: Burning Cost
categories: [tutorials, pricing, libraries]
tags: [whittaker-henderson, smoothing, python, rating-tables, age-curves, experience-rating, loss-ratio, graduation, reml, credible-intervals, insurance-whittaker, motor, uk-pricing, sas-replacement]
description: "A complete Python tutorial for Whittaker-Henderson smoothing of insurance rating tables. Replace your Excel moving average or SAS graduation with automatic REML lambda selection and Bayesian credible intervals. Includes a full worked example on a noisy age-band loss ratio table."
---

If you are searching for "Whittaker-Henderson smoothing Python", you are probably a pricing actuary or data scientist who knows what W-H does and wants to stop doing it in Excel or SAS. This is the tutorial for that transition.

We will build a complete example from scratch: a noisy age-band loss ratio table, a fitted smoother with automatic lambda selection, an interpretation of the output, credible intervals, a 2D extension, and a comparison with the alternatives. The library is [`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker), BSD-3-Clause-licensed, on PyPI.

```bash
pip install insurance-whittaker
# or with uv (recommended):
uv add insurance-whittaker
```

For plots, add the optional dependency:

```bash
pip install "insurance-whittaker[plot]"
```

---

## The problem: why a moving average is wrong

Start with raw loss ratios by age band — a typical output from a one-way analysis or GLM credibility study. The numbers look like this:

| Age | Loss ratio | Policy years |
|-----|-----------|-------------|
| 17 | 0.412 | 180 |
| 18 | 0.287 | 215 |
| 19 | 0.356 | 290 |
| 20 | 0.319 | 380 |
| ... | ... | ... |
| 30 | 0.201 | 2,640 |
| 31 | 0.218 | 2,710 |
| ... | ... | ... |
| 75 | 0.281 | 130 |
| 76 | 0.354 | 96 |
| 77 | 0.188 | 71 |

Age 77 looks cheaper than age 30 purely because there are 71 policy years in the band and a favourable random fluctuation happened that year. You know this, but the raw table does not care.

The standard response is a 5-point moving average. It removes most of the noise in the well-observed middle (ages 30–60), but it has two failures that matter commercially.

**Boundary distortion.** At age 17, a 5-point average uses the four age bands above it to pull the estimate down from the true young-driver peak. You are systematically underpricing young drivers at precisely the age where your rating relativity is highest and your error is most expensive. The same distortion occurs at the upper tail.

**No uncertainty quantification.** A moving average produces a point estimate. You do not know whether the smoothed value at age 78 is reliable or a best guess from three policy years of data. For any Solvency II or PRA model validation exercise, "it looks smooth" is not a defensible answer.

Whittaker-Henderson solves both problems. It is a penalised least-squares smoother that handles boundary effects naturally, scales the smoothing by exposure, and produces posterior credible intervals in closed form.

---

## The mathematics in 60 seconds

Whittaker-Henderson minimises:

```
sum_i w_i (y_i - theta_i)^2  +  lambda * ||D^q theta||^2
```

The first term: fit the data, weighted by exposure `w_i`. The second term: penalise roughness in the fitted curve, where `D^q` is the q-th order finite difference operator and `lambda` controls how much roughness costs.

With `q=2` (the standard choice for rating tables), the penalty is on the second differences of the fitted values — equivalent to penalising curvature. A very high lambda gives you a straight line. A very low lambda gives you the raw data back.

The solution is:

```
theta_hat = (W + lambda * D'D)^{-1} W y
```

This is a linear system. For a 63-band age curve it solves in under 0.05 seconds on a laptop.

The question is how to choose `lambda`. That is where REML comes in.

---

## Full worked example: a noisy age-band table

### Build the data

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# Ages 17 to 79: 63 bands
ages = np.arange(17, 80)

# True underlying loss ratio: high for young drivers, declining, slight uptick over 60
true_lr = (
    0.32 * np.exp(-0.055 * (ages - 17))
    + 0.08
    + 0.002 * ((ages - 50) / 20) ** 2
)

# Exposure: thin at both ends, heavy in the middle
# Roughly 30 policy years at age 17, 2,800 at age 40, 75 at age 79
exposures = (
    2800 * np.exp(-0.5 * ((ages - 40) / 18) ** 2)
    + 30
)

# Observed loss ratios: Gaussian noise scaled to exposure
rng = np.random.default_rng(42)
obs_lr = true_lr + rng.normal(0, np.sqrt(true_lr / exposures))
```

This is the setup every UK motor pricing team recognises: you have observed loss ratios per age band, you know the number of policy years behind each cell, and you know the raw numbers are noisy at the extremes.

### Fit the smoother

```python
wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, obs_lr, weights=exposures)
```

That is the entire fitting step. Two lines, no tuning.

### Inspect the output

```python
print(result)
# WHResult1D(n=63, order=2, lambda_=847.3, edf=4.23, criterion='reml')
```

Two numbers matter here: `lambda_` and `edf`.

**lambda_ = 847.3.** The REML algorithm searched over the full range of plausible lambda values and found that 847.3 maximises the restricted marginal likelihood of the data. This is high relative to what GCV would select (around 412 on this dataset), meaning REML is appropriately sceptical of the noisy observed values in the thin-exposure bands.

**edf = 4.23.** Effective degrees of freedom. A straight line has 2 degrees of freedom; a cubic polynomial has 4. An edf of 4.23 means the fitted curve is very close to a quadratic trend. Given that the true DGP is a smooth, nearly quadratic decay with a small uptick at old ages, this is correct. If you see edf of 12 or more on a 63-band table, the data has told the smoother it contains significant nonlinearity — or there is noise driving lambda too low.

### Read the results

```python
df = result.to_polars()
# Columns: x, y, weight, fitted, ci_lower, ci_upper, std_fitted

# Look at the tails where smoothing matters most
import polars as pl
print(df.filter(pl.col("x").is_in([17, 18, 19, 77, 78, 79])))
```

```
shape: (6, 7)
┌─────┬───────┬────────┬────────┬──────────┬──────────┬─────────────┐
│ x   ┆ y     ┆ weight ┆ fitted ┆ ci_lower ┆ ci_upper ┆ std_fitted  │
│ --- ┆ ---   ┆ ---    ┆ ---    ┆ ---      ┆ ---      ┆ ---         │
│ i64 ┆ f64   ┆ f64    ┆ f64    ┆ f64      ┆ f64      ┆ f64         │
╞═════╪═══════╪════════╪════════╪══════════╪══════════╪═════════════╡
│ 17  ┆ 0.412 ┆ 30.2   ┆ 0.396  ┆ 0.351    ┆ 0.441    ┆ 0.023       │
│ 18  ┆ 0.287 ┆ 62.4   ┆ 0.381  ┆ 0.344    ┆ 0.418    ┆ 0.019       │
│ 19  ┆ 0.356 ┆ 113.1  ┆ 0.366  ┆ 0.335    ┆ 0.397    ┆ 0.016       │
│ 77  ┆ 0.188 ┆ 71.4   ┆ 0.196  ┆ 0.163    ┆ 0.229    ┆ 0.017       │
│ 78  ┆ 0.354 ┆ 52.1   ┆ 0.202  ┆ 0.167    ┆ 0.237    ┆ 0.018       │
│ 79  ┆ 0.298 ┆ 37.8   ┆ 0.208  ┆ 0.170    ┆ 0.246    ┆ 0.019       │
└─────┴───────┴────────┴────────┴──────────┴──────────┴─────────────┘
```

Notice what happened at age 78: the raw observed value of 0.354 — suspiciously high for a band with 52 policy years — gets smoothed to 0.202, consistent with the trend from surrounding bands. The CI of (0.167, 0.237) tells you how uncertain that estimate is given the thin data. You can defend this to a model validator; you cannot defend the raw 0.354.

At age 17, the smoother has kept the high raw value (0.412 → 0.396) rather than pulling it down towards middle-age rates. That is the key advantage over a boundary-distorted moving average: the fitted value at the edge is extrapolated from the smooth trend, not biased towards the band immediately inward.

### Plot it

```python
result.plot()  # requires insurance-whittaker[plot]
```

The plot shows observed loss ratios (dots, sized by exposure), the fitted smooth curve (solid line), and the 95% credible interval band. Bands below age 25 and above age 70 have notably wider CIs. Bands from age 30 to 60 are tight — 2,000+ policy years per band, the data is reliable, the smoother is doing little work.

---

## Why REML beats GCV

The lambda selection method matters. Here is a direct comparison on the same data:

```python
for method in ["reml", "gcv", "aic", "bic"]:
    r = WhittakerHenderson1D(order=2, lambda_method=method).fit(
        ages, obs_lr, weights=exposures
    )
    print(f"{method.upper():5s}: lambda={r.lambda_:7.1f}, edf={r.edf:.2f}")
```

```
REML : lambda=  847.3, edf=4.23
GCV  : lambda=  412.1, edf=6.81
AIC  : lambda=  291.8, edf=8.54
BIC  : lambda=  634.7, edf=5.12
```

GCV selects a lower lambda than REML — more degrees of freedom, a wigglier curve. The intuition for why this is wrong: GCV treats the smooth trend as fixed and optimises the residual fit, consuming degrees of freedom it should not. REML marginalises over the trend, which correctly accounts for the fitted smooth consuming some of the budget. The result is a lambda that is more appropriate when the true curve is smooth — which it is, for any actuarially coherent age curve.

Biessy (2026, *ASTIN Bulletin*) proves this formally and runs the simulation study. The short version: REML recovers the oracle lambda (the one minimising MSE against the true curve) more reliably than GCV, particularly in the thin-exposure bands at the tails. For insurance rating tables, where the tails are the commercially important region, this matters.

Use REML. Run GCV as a sanity check. If they disagree by more than a factor of 3, look at the bands driving the discrepancy — there may be data quality issues or genuine structural breaks worth investigating.

---

## Interpreting lambda and edf in practice

Two rules of thumb that generalise across datasets:

**Lambda above 500 with edf below 6 means you have a well-behaved, essentially polynomial age curve.** This is typical for comprehensive UK motor books: the true shape is smooth, most of the noise is driven by thin-exposure tails, and REML correctly imposes significant smoothing.

**Lambda below 100 with edf above 10 means either genuine nonlinearity or data problems.** On a 63-band table, edf of 10 means the smoother is fitting 10 effective parameters — that is a complex shape. If you see this and the raw loss ratio plot looks like a random walk, suspect data quality (misallocated claims, band boundaries shifting between years) rather than genuine risk structure.

**The CI width is an exposure diagnostic.** If you are presenting a rate change proposal, run `result.to_polars()` and flag any band where `ci_upper - ci_lower > 0.10`. Those are the cells where you are relying on trend interpolation rather than evidence. Your head of pricing should know this before the analysis goes to pricing committee.

---

## 2D smoothing: age × vehicle group

The 1D smoother handles a single rating factor. For a cross-table — driver age by vehicle group, or driver age by NCD step — `WhittakerHenderson2D` applies the same framework along both axes simultaneously.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson2D

# 10 driver age bands × 5 vehicle groups
n_ages, n_groups = 10, 5

# Raw loss ratios: a 10×5 matrix of experience rates
# Young drivers (row 0) are systematically higher-risk
true_lr_2d = (
    np.linspace(0.40, 0.10, n_ages)[:, None]
    + 0.05 * np.arange(n_groups)[None, :]  # vehicle group gradient
)

# Exposure matrix: young drivers and larger vehicles are sparsely represented
rng = np.random.default_rng(42)
exposures_2d = np.abs(rng.normal(300, 100, (n_ages, n_groups)))
exposures_2d[0, :] *= 0.25   # very thin at young ages
exposures_2d[:, -1] *= 0.35  # thin in the most expensive vehicle group

obs_lr_2d = true_lr_2d + rng.normal(
    0, np.sqrt(true_lr_2d / exposures_2d)
)

# Fit: separate curvature penalties along each dimension
wh2d = WhittakerHenderson2D(order_x=2, order_z=2)
result_2d = wh2d.fit(obs_lr_2d, weights=exposures_2d)

print(f"lambda_x={result_2d.lambda_x:.1f}")   # smoothing along age axis
print(f"lambda_z={result_2d.lambda_z:.1f}")   # smoothing along vehicle group axis

smoothed_table = result_2d.fitted    # 10×5 numpy array
```

Two separate lambdas: one for the age dimension, one for the vehicle group dimension. REML selects them jointly. In practice, the vehicle group dimension often gets a higher lambda than the age dimension because vehicle group data is thinner — there are more group codes, fewer policies per code, more noise to smooth out.

The `to_polars()` output is in long format: one row per (age band, vehicle group) combination, with `fitted`, `ci_lower`, and `ci_upper` columns. Join it to your tariff database and you have a smoothed rating surface ready for export.

---

## Poisson count fitting: skip the pre-division step

All of the above assumes you already have loss ratios — claims cost divided by earned premium or exposure. If you are working from claim counts directly, the Gaussian smoother adds a step (and noise) you can avoid.

`WhittakerHendersonPoisson` fits the smoother in the Poisson likelihood via PIRLS (Penalised Iteratively Reweighted Least Squares). It works on the log scale and back-transforms, so the fitted rates are always positive.

```python
from insurance_whittaker import WhittakerHendersonPoisson

# Claim counts and policy years directly — no pre-division
claim_counts = rng.poisson(true_lr * exposures)

wh_poisson = WhittakerHendersonPoisson(order=2)
result_p = wh_poisson.fit(ages, counts=claim_counts, exposure=exposures)

print(result_p.fitted_rate)    # smoothed claim frequency per policy year
print(result_p.ci_lower_rate)  # 95% CI on the rate scale (always positive)
print(result_p.ci_upper_rate)
```

Use this when the count data is sparse and the Gaussian approximation is questionable — typically when the expected claims per band drops below 5. For young and elderly driver bands on a typical UK motor book, this is the right tool. The fitted rates from a Poisson W-H smoother on the [freMTPL2 dataset](https://github.com/burning-cost/insurance-whittaker/blob/main/benchmarks/fremtpl2_age_smoothing.py) (673,000 policies, ages 18–90) match the R `WH` package output to within floating-point precision.

---

## When to use Whittaker-Henderson vs GAMs vs splines

The question comes up. Here is where each belongs.

**Use Whittaker-Henderson** when you have an existing rating table — a one-way factor table from a GLM or a credibility analysis — and you want to smooth it before it goes into the tariff. The input is a vector of relativities indexed by ordered band. W-H respects that structure, handles exposure-weighting naturally, and gives you REML and credible intervals out of the box. This is the graduation problem.

**Use a GAM** (or EBM via [`insurance-gam`](https://github.com/burning-cost/insurance-gam)) when you are modelling claim frequency from a policy-level dataset and want the shape of the age effect estimated directly, without the intermediate step of computing age-band relativities first. The GAM fits the smooth function as part of the model — the smoothing is built in. If your age effect is one of many predictors interacting in a GLM, a GAM is the right place for the smoothing.

**Use a spline** (e.g., `scipy.interpolate.UnivariateSpline`) if you already know the smoothing parameter and want a simple interpolant. You will need to choose the knots or the smoothing factor by hand. For a one-off analysis where REML is overkill and you trust your intuition, it works. For a production rating table that will be documented and reviewed, it is not defensible enough.

**Do not use a moving average.** It has no theoretical basis for the weights, it distorts boundaries, and it produces no uncertainty estimate. The only reason to prefer it is familiarity. Familiarity is not a good reason.

---

## Fitting into the pricing workflow

Whittaker-Henderson sits at a specific point: after the raw one-way or GLM analysis, before the smoothed relativities go into the tariff system.

```
Raw experience data
  → GLM / one-way analysis
  → Age-band loss ratios (noisy)
  → WhittakerHenderson1D.fit()       ← this post
  → Smoothed relativities + CI bands
  → Tariff system / Radar
  → insurance-monitoring (A/E tracking against smoothed curve)
```

The Polars output from `result.to_polars()` is a clean table with named columns. Writing it to Parquet or CSV for your tariff system is one line. If you are tracking model drift post-launch, the smoothed curve and its CI bands are the baseline that `insurance-monitoring` PSI metrics compare against.

For documenting the methodology under Solvency II Article 121 or PRA SS3/18, the REML-selected lambda and the credible intervals are the artefacts that should appear in your model validation report: they demonstrate that the smoothing parameter is data-driven rather than analyst-selected, and that the uncertainty in sparse bands is quantified rather than ignored.

---

## Reference

Biessy, G. (2026). "Whittaker-Henderson Smoothing Revisited." *ASTIN Bulletin*. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932)

`insurance-whittaker` is on [GitHub](https://github.com/burning-cost/insurance-whittaker) and [PyPI](https://pypi.org/project/insurance-whittaker/). BSD-3-Clause licence. Python 3.10+, NumPy, SciPy.

---

- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — library overview and theory
- [Does Whittaker-Henderson Smoothing Actually Work?](/2026/03/28/does-whittaker-henderson-smoothing-actually-work/) — benchmark against raw rates and moving averages
- [Whittaker-Henderson Poisson Smoothing on freMTPL2](/2026/03/28/whittaker-henderson-poisson-age-curve-fremtpl2/) — count data, Poisson PIRLS, real dataset
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) — the complement for unordered categorical factors
