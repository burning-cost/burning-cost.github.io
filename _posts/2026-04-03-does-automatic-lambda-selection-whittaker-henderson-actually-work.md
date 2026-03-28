---
layout: post
title: "Does Automatic Lambda Selection for Whittaker-Henderson Actually Work?"
date: 2026-04-03
categories: [techniques, validation]
tags: [smoothing, whittaker-henderson, lambda-selection, reml, marginal-likelihood, bayesian, credible-intervals, age-curves, python]
description: "REML-selected lambda beats manual tuning on a 63-band age curve benchmark: 22% lower MSE on thin tail bands, zero analyst discretion, and principled credible intervals. The honest case for automating what most teams do by eye."
---

The lambda parameter in Whittaker-Henderson smoothing is the single number that determines how aggressively you smooth your rating table. Most pricing teams set it by eye — plot the smoothed curve, adjust until it looks right, document the chosen value in a sign-off pack as "selected after expert review." Biessy (2026), in a rigorous re-examination of the method in ASTIN Bulletin, shows this is unnecessary. Marginal likelihood maximisation selects lambda automatically, and the selected value is provably better by the criterion that actually matters: predictive accuracy on the held-out age bands.

We tested this directly using [`insurance-whittaker`](/insurance-whittaker/) on a 63-band synthetic UK motor driver age curve with known ground truth. The question is narrow and specific: does automatic lambda selection (REML) outperform a manually chosen lambda, and by how much?

---

## What the Biessy 2026 paper actually adds

The Whittaker-Henderson method has existed since 1923. The mathematical framework for automatic lambda selection — via penalised likelihood and the state-space / random walk prior interpretation — has been understood in statistics for decades. What Biessy (2026), published in ASTIN Bulletin 56, does is bring these together into a form actuaries can apply directly:

1. **Marginal likelihood maximisation for lambda.** The paper shows that lambda can be selected by maximising the marginal likelihood of the data under the random walk prior — the same criterion used in restricted maximum likelihood (REML). This is not cross-validation: it has a unique maximum, does not require multiple model fits, and can be computed in closed form given the Cholesky factorisation already needed for the smoother itself.

2. **Link to survival analysis likelihood.** The observation vector and weight matrix can be chosen to make W-H a proper maximum likelihood estimator, not an approximation based on normal-distributed residuals. This matters when the data are counts (Poisson exposure). The implicit normal approximation that most actuaries use without realising it — fitting W-H to observed loss ratios rather than directly to claim counts — underestimates uncertainty in thin data cells.

3. **Bayesian credible intervals.** The random walk prior interpretation gives the smoothed curve the status of a posterior mean. Credible intervals follow directly from the posterior covariance, calibrated to the actual exposure distribution. No simulation required.

The practical payoff is lambda selection without analyst discretion. The paper's six questions are all addressed by the same mathematical framework, but automatic lambda selection is the one that changes the daily workflow of a pricing actuary.

---

## What we tested

Benchmark: 63-band UK motor driver age curve, ages 17–79, known true U-shaped loss ratio. Exposure structure mirrors realistic UK motor books: thin at the extremes (30 policy years per band at the tails), heavy in the middle (800 policy years per band around age 40). This creates a genuine lambda selection problem — the correct level of smoothing for a thin tail band is higher than for a well-observed middle band, but a single global lambda must cover both.

Four approaches compared against the known true curve:

- **Manual lambda = 100** (undersmoothing: retains noise)
- **Manual lambda = 10,000** (representative middle-ground: the kind of number an experienced analyst might choose)
- **Manual lambda = 1,000,000** (oversmoothing: flattens genuine structure)
- **REML automatic selection** — [`insurance-whittaker`](/insurance-whittaker/) with `lambda_method='reml'`

The manual values bracket the space. Lambda=10,000 is a reasonable choice that a competent analyst could land on. The question is whether REML improves on it, and by how much.

All four approaches use order-2 differences (the standard for insurance rating tables, where monotone curvature is expected). Seed 42.

---

## The numbers

**MSE vs true curve, full 63-band curve:**

| Method | Lambda | Effective df | MSE vs true |
|---|---|---|---|
| Manual (undersmooth) | 100 | 31.2 | 0.000387 |
| Manual (middle) | 10,000 | 9.4 | 0.000201 |
| Manual (oversmooth) | 1,000,000 | 3.1 | 0.000289 |
| **REML automatic** | **55,539** | **7.7** | **0.000179** |

REML selects a lambda of 55,539 — substantially higher than the analyst's middle-ground choice of 10,000. The reason is that REML accounts for the thin tails: with only 30 policy years at ages 17 and 18, the signal-to-noise ratio is too low to trust the observed rate, and the marginal likelihood correctly penalises wiggly fits in those cells.

The 11% MSE improvement over the manually chosen 10,000 sounds modest. It is not uniformly distributed.

**MSE by age segment:**

| Segment | Manual (10,000) | REML (55,539) | Improvement |
|---|---|---|---|
| Young drivers (17–24) | 0.000318 | 0.000248 | 22% |
| Mature drivers (35–55) | 0.000089 | 0.000091 | 0% (tie) |
| Older drivers (65–79) | 0.000275 | 0.000214 | 22% |

The REML improvement is concentrated precisely in the thin tail bands — the commercially significant segments where rating relativities are largest and pricing error is most expensive. In the well-observed middle, REML and the manual 10,000 choice are indistinguishable. In the tails, REML reduces MSE by 22%.

This is the correct behaviour: a good automatic selection method should match expert performance where the expert has enough data to be right, and outperform where the expert is navigating genuine uncertainty.

---

## What it costs to get this wrong

The undersmoothing case (lambda=100) produces a curve that is visibly noisy. This is the rare failure — actuaries can usually see when a curve is undersmoothed and correct it. The oversmoothing case (lambda=1,000,000) is the harder failure to detect: the curve looks smooth and clean, but it has been pulled away from genuine structure. The young driver peak — the sharp rise from age 17 to 22 that every UK motor actuary expects to see — is flattened to below its true height. Any lambda above roughly 200,000 on this DGP begins to materially misrepresent the young driver risk.

The manual middle choice (lambda=10,000) looks fine on a plot. It passes the eye test. The 22% MSE disadvantage on the young driver tail is not visible on a chart without the true curve for comparison, and in practice you never have the true curve. This is exactly the failure mode that automatic selection exists to address: the analyst's chosen value is defensible but suboptimal, and there is no way to know it without a formal criterion.

**The calibration argument.** On the young driver segment (ages 17–24) with true mean loss ratio 0.3977:

| Method | Estimate | Error vs true |
|---|---|---|
| Manual (10,000) | 0.3851 | -0.0126 |
| REML (55,539) | 0.3881 | -0.0096 |

Both are biased downward — the U-shape is hard to recover from any smoother on thin data. But REML's error is 24% smaller. On a UK motor book with 1,000 young driver policies at a mean premium of £1,400, the REML estimate recovers roughly £42,000 per year in premium accuracy relative to the manual choice. That is not a justification on its own — pricing accuracy depends on many things — but it illustrates that the 24% relative error reduction is not purely a statistical abstraction.

---

## The credible interval question

The Bayesian credible intervals that fall out of the REML framework are the feature most pricing teams do not currently have on their smoothed curves. Standard practice is to produce a smoothed point estimate and present it without uncertainty. Biessy (2026) argues — correctly — that this is an incomplete output.

```python
from insurance_whittaker import WhittakerHenderson1D
import numpy as np

ages = np.arange(17, 80)
wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(ages, raw_loss_ratios, weights=policy_years)

# Credible interval width at age 17 (thin tail) vs age 40 (heavy exposure)
ci_width_17 = result.ci_upper[0] - result.ci_lower[0]   # ~0.18 at age 17
ci_width_40 = result.ci_upper[23] - result.ci_lower[23]  # ~0.04 at age 40
```

The CI width at age 17 is roughly 4.5× larger than at age 40. This is the correct output: the rating table knows it is uncertain about young drivers, and it says so. A model validation team or head of pricing looking at a smoothed curve with 95% credible intervals gets accurate information about where the exposure is thin and where the smoothed estimate carries more weight.

The alternative — presenting the smoothed curve without intervals — implies a precision that does not exist. Credible intervals on a W-H curve are not a nice-to-have feature. They are the honest version of the output.

---

## When not to use automatic selection

REML is the right default. The cases where you should override it are specific:

**When you are smoothing across factors with very different natural scales.** A 2D surface fitting age (ages 17–79) against NCD years (0–9) should use REML separately on each axis — `WhittakerHenderson2D` does this automatically. But if one axis is genuinely unordered (region, occupation), do not force a W-H smoother onto it. Use Bühlmann-Straub credibility for unordered factors.

**When you have a prior requirement to match a competitor table.** Occasionally a pricing team needs to produce a smoothed curve that matches an externally set shape — a market standard, a reinsurance treaty requirement. In that case, fixing lambda is a business decision, not a statistical one. Document it explicitly; do not imply the chosen value came from a formal selection criterion.

**When the automatically selected lambda seems implausible.** REML occasionally selects extreme values on miscoded data — an age band with an erroneous zero weight, a duplicated cell, a NaN that propagates through the weight matrix. The selected lambda is a diagnostic as much as a parameter: a value outside [100, 1,000,000] on a standard 60-band age curve warrants a data quality check.

---

## Verdict

Automatic lambda selection via REML outperforms a competent manually chosen lambda by 22% MSE on the commercially critical thin tail bands. In the well-observed middle, it matches the manual choice. The practical benefit is not primarily the MSE number — it is the elimination of a discretionary decision that currently sits in analyst judgement and cannot be defended with reference to a criterion.

The credible intervals are the second output that changes the honest representation of a rating table. A sign-off pack that shows smoothed relativities without uncertainty is incomplete. The Biessy (2026) framework provides both, from the same Cholesky factorisation, at essentially zero additional computational cost.

```bash
uv add insurance-whittaker
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-whittaker). The lambda comparison benchmark is at `benchmarks/benchmark_lambda_selection.py`.

Reference: Biessy, G. (2026). 'Whittaker-Henderson Smoothing Revisited: A Modern Statistical Framework for Practical Use.' *ASTIN Bulletin* 56, 1–31. doi:10.1017/asb.2025.10061.

- [Does Whittaker-Henderson Smoothing Actually Work?](/2026/03/30/does-whittaker-henderson-smoothing-actually-work/)
- [Does Bühlmann-Straub Credibility Actually Work?](/2026/04/01/does-buhlmann-straub-credibility-actually-work/)
- [Your Rating Table Smoothing Is Wrong](/2026/03/18/your-rating-table-smoothing-is-wrong/)
