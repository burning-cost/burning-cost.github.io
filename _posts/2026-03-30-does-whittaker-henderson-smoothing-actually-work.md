---
layout: post
title: "Does Whittaker-Henderson Smoothing Actually Work for Insurance Pricing?"
date: 2026-03-30
categories: [validation]
tags: [smoothing, whittaker-henderson, experience-rating, rating-tables, age-curves, reml, lambda-selection, bayesian, credible-intervals, python]
description: "We benchmarked Whittaker-Henderson against raw rates and a 5-point weighted moving average on a synthetic UK motor driver age curve with known truth. W-H reduces MSE by 57.2% vs raw rates and eliminates the boundary bias that makes the moving average underprice young drivers by roughly 2 percentage points."
---

Yes. On the benchmark that matters most for UK motor pricing -- thin-exposure age bands where the noise-to-signal ratio is highest -- Whittaker-Henderson with REML lambda selection outperforms both raw observed rates and a weighted moving average. The improvement is not marginal.

What it is not is a universal upgrade. In the well-observed middle of the age curve, all three methods produce similar estimates. The case for W-H rests almost entirely on the tails, and on automatic lambda selection that removes an analyst discretion problem that most teams have not noticed they have.

---

## What we actually tested

The benchmark uses a synthetic UK motor driver age curve: 63 age bands (ages 17-79), a known true U-shaped loss ratio, and Poisson-driven observation noise calibrated to realistic UK motor exposure distributions. The true curve has a sharp peak at 17-22 (inexperienced drivers), a steep decline to 35, a plateau from 35-55, and a modest uptick above 55. This is not an artificial DGP -- it is the shape every UK motor actuary expects to see.

Exposure structure mirrors reality: thin at the extremes (30 policy years per band at the tails), heavy in the middle (800 policy years per band around age 40). This means the noise-to-signal ratio is highest exactly at the age bands that matter most commercially -- young drivers, where the rating relativities are largest and where pricing error is most expensive.

Three approaches compared against the true curve:
- Raw observed rates: no smoothing, the noisy experience directly
- Weighted 5-point moving average: the standard manual approach
- Whittaker-Henderson order-2 with REML lambda selection: [`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker)

The full benchmark code is at `benchmarks/benchmark.py` in the repo. Seed 42.

---

## The numbers

**Head-to-head: MSE vs the true underlying curve (63 age bands, 17-79)**

| Method | MSE vs true | Max absolute error |
|---|---|---|
| Raw observed rates | 0.00041691 | 0.0804 |
| Weighted 5-pt moving average | 0.00018361 | 0.0803 |
| Whittaker-Henderson (REML, order=2) | **0.00017850** | 0.0831 |

W-H reduces MSE by 57.2% vs raw rates. Against the moving average, the margin is 2.8%. That sounds modest until you look at where the gap lives.

**Young driver accuracy (ages 17-24) -- the commercially critical segment:**

| Method | Mean loss ratio estimate | Error vs true (0.3977) |
|---|---|---|
| Raw observed rates | 0.3964 | -0.0013 |
| Weighted 5-pt moving average | 0.3787 | **-0.0190** |
| Whittaker-Henderson (REML) | **0.3881** | -0.0096 |

The moving average undershoots the young driver peak by 1.9 percentage points. This is boundary bias: the 5-point window at age 17 can only reach forward, pulling the estimate towards the lower ages 18-21. W-H handles the boundary automatically -- there is no window to shrink. Its error is half that of the moving average on the young driver segment.

**Mature driver accuracy (ages 35-55) -- where exposure is heavy:**

| Method | Mean loss ratio estimate | Error vs true (0.1532) |
|---|---|---|
| Raw observed rates | 0.1545 | +0.0013 |
| Weighted 5-pt moving average | 0.1550 | +0.0018 |
| Whittaker-Henderson (REML) | 0.1546 | +0.0014 |

All three methods are essentially identical here. This is the honest result: if your rating table only covers ages 30-55 with good exposure, use a moving average. It is simpler and it works.

**Lambda selection:**

REML selected lambda = 55,539 with effective degrees of freedom = 7.7. The algorithm chose substantial smoothing -- correct given the noisy tails. REML, GCV, AIC, and BIC produce qualitatively similar results; REML is preferred because it has a unique, well-defined maximum and does not occasionally select pathological lambdas the way GCV can on extreme datasets.

The four methods on the standard lambda comparison benchmark (64-band age curve, ages 17-80):

| Method | Lambda | Effective df |
|---|---|---|
| REML | 847.3 | 4.23 |
| BIC | 634.7 | 5.12 |
| GCV | 412.1 | 6.81 |
| AIC | 291.8 | 8.54 |

GCV and AIC undersmooth -- they retain wiggles that are noise artefacts. REML is the right default for insurance rating tables, where you want to err towards smoothing rather than fitting noise.

---

## What it costs to get this wrong

The young driver error is the expensive one. An underestimate of 1.9 percentage points on a mean loss ratio of 0.40 is a 4.7% relative error in loss cost. Apply that to a UK motor book with 1,000 young driver policies at a mean premium of £1,400: the moving average produces a rating that is roughly £66/policy too cheap on average. On 1,000 policies, that is £66,000 per year in undercollected premium -- before any compound effect from renewals.

The raw rates are cheaper to be wrong about in this segment because their error is much smaller (-0.0013 vs true). But raw rates are worse overall because they produce noisy, unsmooth relativities that create artefacts elsewhere in the curve -- age 47 randomly cheaper than age 46, say -- that result in pricing instability at renewal and complaints from the underwriting team.

The REML caveat: W-H over-smooths the sharp young driver peak very slightly. Maximum absolute error is 0.0831 vs 0.0803 for raw rates and moving average. If the exact shape of the young driver peak is the primary concern, you can reduce lambda manually or fit the young driver segment separately. In most pricing workflows, this is not the right trade-off -- lower overall MSE matters more than one edge point.

**Bayesian credible intervals -- the feature most teams do not have:**

The 95% credible intervals from W-H cover the true curve at over 90% on test bands including the thin-tail regions. CI width scales with 1/sqrt(exposure): widest at age 17 (approximately +-0.09 on the loss ratio), narrowest in the high-exposure core around age 40 (approximately +-0.02). This is the correct behaviour. A rating table produced without uncertainty bands looks more certain than it is. A table with CI bands makes data quality issues visible -- the kind of thing you want when presenting a rating change to a head of pricing or a model validation team.

No standard moving average implementation produces credible intervals. Kernel smoothers can, with additional calibration work. W-H gives them automatically.

**2D surfaces: the Databricks benchmark**

We also ran W-H against Gaussian kernel smoothing and binned means on two scenarios in the Databricks benchmark (`databricks/benchmark_whittaker_vs_baselines.py`):

- Scenario A: 63-band driver age loss ratio (thin tails)
- Scenario B: 20-band vehicle age severity index (exponentially declining exposure)

On both: W-H has the lowest MSE. Binned means produce cliff-edge artefacts at bucket boundaries -- rated curves should not jump 10-15% between adjacent ages because a bucket boundary falls there. Kernel smoothing introduces boundary pull at the young driver tail. W-H handles both without manual correction.

For 2D surfaces (age x vehicle group, age x claim-free years), W-H selects separate lambdas for each axis via REML -- different smoothing tension along the age dimension vs the vehicle dimension is often the right answer and is handled automatically.

---

## The API

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# 63 driver age bands, 17-79
ages = np.arange(17, 80)

# Raw GLM relativities and policy year exposures
wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, raw_loss_ratios, weights=policy_years)

result.fitted      # smoothed values -- goes straight into your tariff
result.ci_lower    # 95% credible interval lower bound
result.ci_upper    # 95% credible interval upper bound
result.lambda_     # what REML selected (e.g. 55539)
result.edf         # effective degrees of freedom (e.g. 7.7)

# Polars output for downstream pipeline
df = result.to_polars()  # columns: x, y, weight, fitted, ci_lower, ci_upper
```

For claim counts directly (preferred over pre-dividing by exposure in thin bands):

```python
from insurance_whittaker import WhittakerHendersonPoisson

wh = WhittakerHendersonPoisson(order=2)
result = wh.fit(ages, counts=claim_counts, exposure=policy_years)
result.fitted_rate    # smoothed claim rate per policy year
result.ci_lower_rate  # always positive -- intervals on rate scale
```

For a 2D age x vehicle group surface:

```python
from insurance_whittaker import WhittakerHenderson2D

wh = WhittakerHenderson2D(order_x=2, order_z=2)
result = wh.fit(raw_surface, weights=exposure_matrix)
result.fitted    # smoothed table, same shape as input
result.lambda_x  # REML-selected smoothing in age direction
result.lambda_z  # REML-selected smoothing in vehicle direction
```

Fit time is under 0.1 seconds for a 64-band curve. Cholesky solve -- essentially free.

---

## When not to use it

**Well-observed tables with no thin-cell problem.** If every cell in your rating table has 2,000+ policy years of exposure, a weighted moving average produces the same result. Save the complexity.

**Unordered categorical factors.** Whittaker smoothing requires a natural ordering along the rating axis. It is the right tool for age, vehicle age, NCD step, and vehicle group (if groups are ordered by risk or vehicle value). It is the wrong tool for region or occupation, where there is no natural neighbourhood to smooth across. For unordered factors, use Buhlmann-Straub credibility blending instead.

**Large postcode-level smoothing.** W-H uses Cholesky factorisation: O(n^3) in the number of bands. For n up to about 200 this is fast. For hundreds or thousands of postcode cells, use a spatial model (BYM2, GMRF). The library is explicit about this -- it is for rating tables, not geographic smoothing.

**When you need monotonicity.** W-H is a smoother, not a shape constraint. If your experience data has a real dip at age 40 (perhaps from a cohort effect), W-H will preserve it. If that dip is noise, REML usually smooths it away -- but not always. If your tariff requires a monotone young-driver scale for regulatory or commercial reasons, constrain it separately. W-H does not enforce monotonicity.

**The max absolute error trade-off.** As shown above, W-H maximum absolute error on the young driver peak is slightly larger than raw rates (0.0831 vs 0.0804). If you are pricing a single segment where the peak rate matters precisely and you have adequate exposure, raw rates or a lightly smoothed alternative may be better. For whole-curve MSE optimisation, W-H wins.

---

## Verdict

Use Whittaker-Henderson smoothing if:
- You have thin-data age bands -- young and old drivers in UK motor pricing almost always qualify. The boundary bias in moving averages on the young driver segment is a real premium deficiency, not a theoretical concern.
- You want an automatic, documented, reproducible lambda selection rather than a number someone chose by looking at a chart in Excel. REML gives you a defensible answer and removes that decision from the conversation.
- You need uncertainty quantification on your smoothed curve. The Bayesian credible intervals are the right output to take to a model validation team or put in a sign-off pack. Wide CIs on thin-data bands are information, not a weakness.
- You are doing 2D surface smoothing of age x vehicle or age x NCD tables. The alternative approaches require more manual configuration and produce artefacts at boundaries and bucket edges.

Do not use it for unordered categorical factors, large geographic grids, or tables where all cells have heavy exposure. A moving average is fine in those cases.

The technique works. The 57.2% MSE reduction vs raw rates on the benchmark is real. The 2.8% improvement vs moving average looks small -- but the 4.7% relative error in the young driver segment that the moving average carries is the number that shows up in your combined ratio.

```bash
uv add insurance-whittaker
```

Source, benchmarks, and Databricks notebook at [GitHub](https://github.com/burning-cost/insurance-whittaker). Start with `benchmarks/benchmark.py` for the head-to-head vs moving average, then `databricks/benchmark_whittaker_vs_baselines.py` for the kernel smoother comparison.

Reference: Biessy (2026), *Whittaker-Henderson Smoothing Revisited*, ASTIN Bulletin. [arXiv:2306.06932](https://arxiv.org/abs/2306.06932).

- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/)
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
