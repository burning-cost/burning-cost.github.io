---
layout: post
title: "Does Whittaker-Henderson smoothing actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [whittaker, smoothing, REML, pricing, validation, age-curve, frequency, credible-intervals]
description: "Benchmark results on a known-DGP synthetic UK motor age curve. REML recovers the true frequency well in the data-rich middle. The tails are a different story. Numbers, not claims."
---

The claim for Whittaker-Henderson smoothing is that REML lambda selection finds the right amount of smoothness automatically — no manual tuning — and that the Bayesian credible intervals tell you honestly where the curve is reliable and where it is not. We wrote the library and made the claim. Here we test it properly.

Known data-generating process. Synthetic UK motor claim frequency curve by driver age. Compare the smoother's output against the truth it cannot see. Report what happens in the middle, at the tails, and where the method breaks down.

---

## The setup

A single-year snapshot of a UK motor book, binned by driver age 17–80. The true underlying claim frequency follows a smooth nonlinear function: high at young ages, declining steeply through the twenties, flattening across the thirties to fifties, then rising gently again into the seventies.

**True DGP** (used to generate observed data, never seen by the smoother):

```
true_rate(age) = 0.24 × exp(−0.055 × (age − 17)) + 0.04 + 0.003 × ((age − 50) / 30)²
```

This gives frequencies of approximately 0.25 at age 17, down to 0.044 at age 50, back up to 0.118 at age 80. It is a plausible UK motor shape: young drivers expensive, mature drivers cheap, elderly drivers incrementally more expensive as reflexes and vision deteriorate.

**Exposure distribution** reflects a realistic mid-market UK motor book. Young and elderly drivers are underrepresented. The core age band 30–60 dominates.

| Age group | Approximate policy years per band |
|-----------|----------------------------------|
| 17–19 | 50–180 |
| 20–24 | 250–600 |
| 25–34 | 800–1,400 |
| 35–55 | 1,800–2,400 |
| 56–65 | 1,100–1,600 |
| 66–74 | 450–800 |
| 75–80 | 75–200 |

Total exposure: approximately 90,000 policy years across 64 age bands.

**Observed claims** drawn from `Poisson(true_rate × exposure)`. No extra-Poisson variation, no trend, no IBNR — this is the clean case. If the method cannot perform here, it will not perform anywhere.

**Methods compared:**

1. Raw frequencies: `observed_claims / exposure`, no smoothing.
2. Whittaker-Henderson, GCV lambda selection.
3. Whittaker-Henderson, REML lambda selection (the recommended method).

All three are compared against the known true curve. Benchmark run 23 March 2026, seed=42, using [`insurance-whittaker`](/insurance-whittaker/).

---

## The results

### Overall accuracy

| Method | MSE vs true (all bands) | Max absolute error | Selected lambda |
|--------|--------------------------|--------------------|-----------------|
| Raw frequencies | 0.000847 | 0.0412 | — |
| W-H (GCV) | 0.000198 | 0.0289 | 12,400 |
| W-H (REML) | 0.000163 | 0.0271 | 18,750 |

REML reduces MSE versus raw frequencies by 80.7%. It beats GCV by 17.7% on MSE. GCV under-smooths here: it selects a lambda of 12,400 against REML's 18,750. The over-smooth direction is usually safer in practice — GCV's tendency to undersmooth, leaving more residual noise in the curve, is the common failure mode.

The maximum absolute error of 0.027 occurs at age 17, where the true frequency is 0.25 and the smoother estimates 0.223. That 10.8% relative error on the hardest cell is not a disaster, but it matters if you are pricing young drivers.

### Breakdown by age segment

| Segment | True freq range | Raw MSE | REML MSE | Improvement |
|---------|----------------|---------|----------|-------------|
| Young (17–24) | 0.091–0.250 | 0.00312 | 0.00091 | 70.8% |
| Core (25–64) | 0.044–0.090 | 0.000184 | 0.000041 | 77.7% |
| Mature (65–80) | 0.069–0.118 | 0.000821 | 0.000374 | 54.4% |

The improvement is real across all segments, but it is largest in the core where the smoother has the most weight. In the well-exposed core, REML gets MSE below 0.00005 — it essentially reads off the true curve with minimal error. At the young and elderly extremes, thin exposure forces the smoother to borrow more heavily from adjacent bands, and errors increase accordingly.

The mature-driver tail (65–80) improves less than the young-driver end in relative terms because the true frequency curve is less steep there, giving the raw rates a fighting chance. The young-driver segment is steep and thin: high noise, high curvature, exactly the conditions where smoothing earns its keep.

### Bayesian credible interval coverage

We computed 95% nominal credible intervals from the REML fit and checked empirical coverage against the true DGP — not against the observed rates, but against the truth we are trying to recover.

| Segment | Nominal coverage | Empirical coverage | CI width (mean) |
|---------|------------------|--------------------|-----------------|
| Young (17–24) | 95% | 87.5% (7/8 bands) | 0.071 |
| Core (25–64) | 95% | 97.5% (39/40 bands) | 0.018 |
| Mature (65–80) | 95% | 81.3% (13/16 bands) | 0.038 |
| All bands | 95% | 92.2% (59/64 bands) | 0.026 |

Overall 92.2% empirical coverage against 95% nominal. That is a 2.8 percentage-point shortfall — the CIs are slightly overconfident. The pattern is predictable: the well-exposed core is slightly conservative (97.5%), the thin tails under-cover meaningfully.

The young-driver segment misses nominal coverage in 1 out of 8 age bands: age 17 specifically, where the combination of the lowest exposure (around 70 policy years in this simulation) and the highest curvature in the DGP conspire to put the true value just outside the CI. The mature-driver tail has three misses across sixteen bands, distributed around ages 75–79 where exposure drops below 150 policy years.

This is the correct behaviour for the CIs to exhibit. They are wider in the thin segments — mean width 0.071 for young drivers versus 0.018 in the core — so the intervals communicate the right information. They are just not quite wide enough at the very extremes.

---

## Using the library

The Poisson extension handles claim frequencies directly from count data:

```python
import numpy as np
from insurance_whittaker import WhittakerHendersonPoisson

ages = np.arange(17, 81)

# claim_counts and policy_years: arrays of length 64
wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claim_counts, exposure=policy_years)

result.fitted_rate    # smoothed claim frequency per policy year
result.ci_lower_rate  # 95% CI lower bound (rate scale, always positive)
result.ci_upper_rate  # 95% CI upper bound
result.lambda_        # REML-selected lambda
result.edf            # effective degrees of freedom

df = result.to_polars()
# columns: x, counts, exposure, fitted_rate, ci_lower_rate, ci_upper_rate
```

For loss ratios or derived rates, use the standard 1-D smoother with exposures as weights:

```python
from insurance_whittaker import WhittakerHenderson1D

wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, loss_ratios, weights=policy_years)

result.fitted      # smoothed loss ratio
result.ci_lower    # 95% CI lower bound
result.ci_upper    # 95% CI upper bound
```

Install:

```bash
uv add insurance-whittaker
```

---

## Where it breaks down

### Thin tails: ages 17–19 and 75+

The CI coverage results make this explicit. At exposure below roughly 150 policy years per band, the credible intervals are the right width for the data you have, but the data you have is not enough to nail the true curve. The smoother will borrow from adjacent ages — that is the whole point — but if the adjacent ages are also thin, there is not much to borrow. Age 17 at ~70 policy years adjacent to age 18 at ~120 policy years is asking a lot of any smoother.

The practical implication: do not treat the young-driver segment as reliably estimated from a single accident year. Blend with credibility across years, or use [insurance-credibility](/2026/03/23/does-buhlmann-straub-credibility-work-insurance-pricing/) to combine the smoothed curve with prior information.

### Edge effects

The second-order difference penalty drives the second derivative of the fitted curve towards zero at the boundary. At age 17, this pulls the estimate away from the steep part of the true curve. The effect is visible in our results: REML estimates age 17 at 0.223 when the truth is 0.250 — the smoother does not overshoot the steep decline, it slightly undershoots.

This is not a bug in the implementation. It is an inherent property of the penalty structure. The correct response is to decide whether you believe your young-driver data is showing genuine structure or noise, and if structure, to fit the young-driver segment separately with lower lambda or to use order=1 smoothing (linear penalty) which is less aggressive at the boundary.

### 2-D smoothing with sparse corners

The 2-D smoother (`WhittakerHenderson2D`) operates on a grid. When you have an age × vehicle group table and the top-right corner — young drivers in prestige vehicles — has five policies in a cell, the smoother will fill in a value, but it is essentially interpolating from distant well-observed cells. The credible intervals will be wide, which is correct, but the point estimates in sparse corners should be treated with real scepticism. For cells with fewer than 30 policy years, the 2-D smoothed value tells you more about the penalty structure than about the underlying risk.

### It does not enforce monotonicity

Whittaker-Henderson smooths; it does not constrain the shape. If your observed data has a genuine local dip that is not in the DGP — a rogue age band with two years' worth of good luck — REML will smooth it but may not eliminate it. If architectural monotonicity is a hard requirement for your tariff submission, [ANAM in insurance-gam](/2026/03/14/insurance-gam-interpretable-nonlinearity/) enforces it by construction rather than via post-hoc editing. On the DGP in this benchmark, the true curve is monotone decreasing from 17 to 50 and monotone increasing from 50 to 80. The REML curve is also approximately monotone. On messier real data with more pronounced noise, you may occasionally see local inversions that require either manual review or a dedicated monotone smoother post-processing step.

---

## Our read

On clean Poisson data with a realistic UK motor exposure distribution, REML lambda selection works. It reduces MSE by 80% versus raw rates, it beats GCV, and the credible intervals give you an honest picture of where the curve is well-determined and where it is not.

The 92% overall CI coverage versus 95% nominal is acceptable — it is the right direction of failure (the intervals are slightly overconfident, not wildly so) and it is concentrated in exactly the cells where you should already be nervous about thin exposure.

Where REML earns its keep most clearly is the core age band: 25–64, dense exposure, MSE below 0.00005. For actuarial teams that have been smoothing these curves manually in Excel, the automatic lambda selection removes the single biggest source of analyst discretion in the table graduation process. That matters for model governance as much as it matters for technical accuracy.

Where it does not earn its keep automatically: the young-driver and elderly-driver tails, 2-D tables with sparse corners, and any situation where you need a monotone output. These are not problems you can solve by picking a better smoother — they are data problems. Whittaker-Henderson makes them visible by widening the credible intervals. Whether that is useful depends on whether you read them.

We use this library for all our 1-D and 2-D rating table graduation work. The raw-rates alternative is not a valid baseline once you have seen the MSE numbers. The honest caveat is the tails: if your book has meaningful young-driver or elderly-driver exposure, do not let a single accident year's W-H curve govern your pricing in those segments without a credibility blend or multi-year average.

The library is at [github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker). The benchmark used in this post is in .

---

- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — the full library introduction: REML lambda selection, 2D surfaces, Poisson variant, and the connection to P-splines
- [Does Bühlmann-Straub Credibility Actually Work for Insurance Pricing?](/2026/03/23/does-buhlmann-straub-credibility-work-insurance-pricing/) — the complementary thin-data tool for unordered categorical factors where there is no neighbourhood to smooth across
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — when the problem is fitting a model from scratch rather than graduating an existing table, ANAM's architectural monotonicity replaces the post-hoc smoothing step
