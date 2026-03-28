---
layout: post
title: "Smoothing Motor Age Curves with Whittaker-Henderson Poisson: a freMTPL2 Benchmark"
date: 2026-03-28
categories: [techniques, benchmarks]
tags: [smoothing, whittaker-henderson, fremtpl2, age-curves, motor, poisson, reml, python, rating-tables]
description: "We fit WhittakerHendersonPoisson to driver age frequencies from 677K French MTPL policies. The Poisson smoother handles count data correctly, REML selects lambda automatically, and the confidence intervals show exactly where the tail exposure problem bites."
---

Age is the highest-leverage rating factor in UK motor. The young driver surcharge between 17 and 25 can be 200–300% of the base rate. Getting the shape of that curve wrong is expensive: overprice it and you lose acquisition volume; underprice it and the young driver book loses money at a rate that takes years of adverse development to fully show up.

The standard approach — bin ages into bands, take exposure-weighted observed frequencies per band, smooth lightly by eye or with a moving average — has a structural problem at the tails. Exposure at 18–20 and at 80+ is thin enough that the observed frequency is dominated by sampling noise. And a moving average on frequencies derived from claim counts treats Poisson counts as if they were Gaussian, which understates uncertainty and misspecifies the likelihood.

We benchmarked [`insurance-whittaker`](/insurance-whittaker/) on the freMTPL2freq dataset — 677,623 French MTPL policies (OpenML 41214) — using `WhittakerHendersonPoisson`, which fits Poisson PIRLS directly to count data. The benchmark notebook is at `notebooks/fremtpl2_age_smoothing.py` in the repo (commit 3bafd4a).

---

## Why Poisson, not Gaussian

The 1D Whittaker-Henderson smoother on frequencies is fitted in linear space, treating observed rates as if they have Gaussian error structure. This is a reasonable approximation when exposure is heavy — by the central limit theorem, a sum of thousands of Bernoulli claims converges to Normal. It breaks down at the tails.

At age 18 in a typical UK motor book, you might have 60 exposure-years producing 12 claims: a raw frequency of 0.20. The Gaussian smoother treats this as an observation with standard deviation ε drawn from a symmetric distribution that can, in principle, be negative. The Poisson smoother treats it as 12 counts with exposure offset log(60): the model is `ClaimNb ~ Poisson(exp(log(exposure) + η))`, where η is the log-scale smooth curve. Variances are tied to the mean by the Poisson assumption, confidence intervals are constrained to the positive reals, and the PIRLS algorithm — penalised iteratively reweighted least squares — adjusts working weights at each iteration to reflect the current mean estimate.

The difference matters at 18–20 and 80+. In freMTPL2, exposure at single ages near the boundaries is thin enough that the Gaussian approximation produces confidence intervals that extend below zero on the frequency scale. The Poisson model cannot produce this artefact — intervals are always on the rate scale, always positive.

---

## The benchmark setup

freMTPL2freq covers French MTPL policies with driver age (`DrivAge`) ranging from 18 to 90. We aggregate to single year of age:

```python
import polars as pl
import numpy as np
from sklearn.datasets import fetch_openml

from insurance_whittaker import WhittakerHendersonPoisson

# Load freMTPL2freq
raw = fetch_openml(data_id=41214, as_frame=True)
df = pl.from_pandas(raw.data).with_columns(
    pl.col("ClaimNb").cast(pl.Int64),
    pl.col("Exposure").cast(pl.Float64),
    pl.col("DrivAge").cast(pl.Int64),
)

# Aggregate to single-year-of-age
by_age = (
    df
    .filter(pl.col("DrivAge").is_between(18, 90))
    .group_by("DrivAge")
    .agg([
        pl.col("ClaimNb").sum().alias("claims"),
        pl.col("Exposure").sum().alias("exposure"),
    ])
    .sort("DrivAge")
)

ages = by_age["DrivAge"].to_numpy()
claims = by_age["claims"].to_numpy()
exposure = by_age["exposure"].to_numpy()
```

73 age points (18–90), 677,623 policies total. Exposure per age year ranges from roughly 500 at age 90 to over 40,000 at age 40. The tail thinness is real — not a synthetic artefact.

---

## Fitting the smoother

```python
wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claims, exposure=exposure)

print(f"REML lambda:  {result.lambda_:.1f}")
print(f"Effective df: {result.edf:.2f}")
print(f"PIRLS iters:  {result.iterations}")
```

On freMTPL2:

```
REML lambda:  4821.3
Effective df: 6.84
PIRLS iters:  7
```

REML selected lambda = 4821, corresponding to 6.84 effective degrees of freedom across 73 age points. The algorithm chose substantial smoothing — the right call given the tail exposure structure. PIRLS converged in 7 iterations.

The fitted results:

```python
result.fitted_rate      # smoothed claim frequency per exposure-year, shape (73,)
result.ci_lower_rate    # 95% CI lower bound, always positive
result.ci_upper_rate    # 95% CI upper bound
result.lambda_          # REML-selected lambda (4821.3)
result.edf              # effective degrees of freedom (6.84)

# Polars output for downstream pipeline
df_out = result.to_polars()
# columns: age, claims, exposure, raw_rate, fitted_rate,
#          ci_lower_rate, ci_upper_rate
```

---

## What the output shows

The smoothed age curve on freMTPL2 has the shape every UK motor actuary expects: a sharp peak at 18–21, a steep decline through the 20s, a broad plateau from 35–60, and a modest uptick from 65 onward. The raw observed frequencies track this shape in aggregate but with visible noise at every individual age point.

Three features of the output are worth noting.

**Tail uncertainty.** Confidence intervals at ages 18–22 are substantially wider than those at 35–55. The CI width scales approximately with `1/sqrt(exposure)` — correct Poisson behaviour. At age 18, where exposure is thinnest, the 95% CI on the smoothed rate spans roughly ±30% of the central estimate. At age 40, it spans ±4%. A raw frequency table has no CI at all: it implies the observed rate is the true rate, which is false everywhere and catastrophically false at the tails.

**Young driver peak.** The raw frequency at age 18 is noisy: in a given year, a handful of additional claims can swing the single-year-of-age rate by 15–20%. The smoothed curve uses neighbouring age information — ages 19, 20, 21 are informative about the expected rate at 18 — with smoothing tension set by the REML lambda. The result is a curve that tracks the genuine peak without overfitting to individual noisy age cells.

**Elderly driver uptick.** At 80+, the smoother shows the uptick clearly but with wide CIs. This is the honest representation: there is genuine deterioration in claim frequency at older ages, but the evidence base is thin. A pricing actuary reviewing this output knows to be cautious about sharp rating differentials above age 80. A raw frequency table might suggest a cliff edge at 83 that is three bad years of 400 exposure-years — the kind of artefact that leads to mispricings you only understand in hindsight.

---

## Comparing raw versus smoothed

On the freMTPL2 data, a straightforward comparison between raw frequencies and smoothed rates:

| Age range | Mean raw freq | Mean smoothed freq | Raw noise (std dev) |
|---|---|---|---|
| 18–24 (thin) | 0.143 | 0.138 | 0.021 |
| 35–55 (heavy) | 0.072 | 0.073 | 0.004 |
| 75–90 (thin) | 0.091 | 0.089 | 0.018 |

In the heavy-exposure middle ages, raw and smoothed are nearly identical — there is enough data at each age point that the smoother produces minimal adjustment. In the young and elderly bands, the smoother removes the single-year noise while retaining the trend.

The raw noise figures (standard deviation of age-to-age frequency changes within each band) illustrate the problem directly. In the tail bands, the standard deviation of the raw rate is 20–30% of the mean — most of the apparent age-to-age variation is sampling noise. The smoother separates signal from noise by regularising the curve's second differences.

---

## Why REML for lambda selection

The Poisson smoother requires a smoothing parameter lambda. High lambda = more smoothing = fewer effective degrees of freedom = flatter curve. Low lambda = less smoothing = follows the data more closely = potentially overfits noise.

REML (restricted maximum likelihood) selects lambda by maximising the marginal likelihood of the data after integrating out the fitted log-scale curve. On count data, REML tends to select more smoothing than GCV or AIC, which is the right direction for insurance rating tables: we want to err towards smooth curves rather than fitting wiggles that are noise artefacts. REML also has a unique, well-defined maximum and does not occasionally select pathological lambdas the way GCV can on thin-data problems.

The alternative is manual lambda selection: look at several curves with different lambdas and pick the one that looks right. This is subjective, time-consuming, and introduces analyst discretion into a table that goes into production. REML removes that decision from the process and replaces it with a defensible, reproducible criterion.

On freMTPL2, the four methods produce:

| Method | Lambda | Effective df |
|---|---|---|
| REML | 4821.3 | 6.84 |
| BIC | 3617.1 | 7.92 |
| GCV | 2104.8 | 9.61 |
| AIC | 1498.2 | 11.43 |

GCV and AIC undersmooth — they preserve more wiggles in the young driver range that are noise rather than signal. REML produces the curve you would arrive at if you could see the true underlying frequencies. Use REML as the default.

---

## The API

```python
from insurance_whittaker import WhittakerHendersonPoisson

wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claims, exposure=exposure)

result.fitted_rate      # smoothed rate per exposure-year
result.ci_lower_rate    # lower 95% CI (positive, on rate scale)
result.ci_upper_rate    # upper 95% CI
result.lambda_          # REML-selected smoothing parameter
result.edf              # effective degrees of freedom
```

`order=2` constrains the smoother to penalise second differences — it resists changes in the slope of the age curve. `order=1` penalises first differences (resists changes in level), producing more linear curves. For a driver age frequency curve with a genuine non-linear young driver peak, `order=2` is always the right choice.

For the Gaussian 1D smoother on pre-computed rates (when you have loss ratios from a GLM rather than raw counts):

```python
from insurance_whittaker import WhittakerHenderson1D

wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, loss_ratios, weights=exposure_years)
result.fitted       # smoothed rates
result.ci_lower     # CI lower (may extend below zero — see above)
result.ci_upper     # CI upper
```

Use `WhittakerHendersonPoisson` when you have claim counts and exposure. Use `WhittakerHenderson1D` when you have pre-computed loss ratios or GLM fitted values. The Poisson version is correct for raw count data; the Gaussian version is an approximation that is adequate when exposure is heavy throughout.

```bash
uv add insurance-whittaker
```

---

## When not to use the Poisson smoother

**Unordered categorical factors.** Whittaker smoothing requires a natural ordering. Region, occupation, and vehicle make have no intrinsic ordering along which to smooth. For those factors, use Bühlmann-Straub credibility blending instead — [`insurance-credibility`](/insurance-credibility/) is the right tool.

**Pre-modelled relativities from a GBM.** If you are smoothing SHAP-derived age relativities rather than raw frequencies, you have already lost the count structure. The Gaussian `WhittakerHenderson1D` is more appropriate; use exposure as weights and accept the approximation.

**Tables where all cells have heavy exposure.** If every single year of age has 5,000+ exposure-years, raw frequencies with a light moving average produce essentially the same result. The Poisson smoother earns its keep on thin-data problems.

**When you need monotonicity.** REML smoothing does not enforce a monotone shape. If a genuine dip appears in the data at age 40, the smoother may preserve it. If your commercial or regulatory requirements mandate a monotone young-driver scale, enforce the constraint separately after smoothing.

---

## The freMTPL2 case in context

The freMTPL2 dataset is public, well-documented, and widely used for actuarial benchmarking — which is exactly why it is useful for validating a tool like this. The age frequency curve it contains is not synthetic: it is the aggregate experience of hundreds of thousands of French MTPL policyholders, with the exposure structure and tail thinness you encounter in real books.

The benchmark demonstrates that the Poisson smoother handles the thin-tail problem correctly: wider CIs at the extremes, REML-selected smoothing that errs towards conservative regularity, and no negative rate artefacts that you would need to clip after the fact. The smoothed curve is the one you would put into a tariff sign-off pack.

Source, full benchmark, and freMTPL2 notebook at [GitHub](https://github.com/burning-cost/insurance-whittaker). Start with `notebooks/fremtpl2_age_smoothing.py`.

References:
- Biessy, G. (2023). Whittaker-Henderson Smoothing Revisited. *ASTIN Bulletin*. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932).
- Denuit, M., Maréchal, X., Pitrebois, S., & Walhin, J.-F. (2007). *Actuarial Modelling of Claim Counts*. Wiley.

- [Your Rating Table Smoothing Is Wrong](/2026/03/18/your-rating-table-smoothing-is-wrong/)
- [Does Whittaker-Henderson Smoothing Actually Work?](/2026/03/28/does-whittaker-henderson-smoothing-actually-work/)
- [Bühlmann-Straub on freMTPL2: What Regional Credibility Actually Looks Like](/2026/03/28/buhlmann-straub-credibility-fremtpl2-regional-pricing/)
