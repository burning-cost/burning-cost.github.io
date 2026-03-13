---
layout: post
title: "Distributional Refinement Networks for Insurance Pricing: insurance-severity"
date: 2026-03-10
categories: [libraries, pricing, capital]
tags: [distributional-regression, neural-networks, GLM, DRN, tail-risk, solvency-ii, ifrs-17, reinsurance, crps, python, insurance-severity]
description: "Your GLM predicts a mean. DRN refines it into a full predictive distribution — per risk, per policy, with parametric tails for the extremes that matter for capital and reinsurance. insurance-severity implements Avanzi et al. (2024) with production training, vectorised inference, and actuarial validation tooling."
---

Your GLM predicts that a young driver with a sports car has an expected annual severity of £4,200. It predicts that a 45-year-old driving a family saloon has an expected severity of £2,100. The mean is 2:1. Fine. That is what GLMs are built to estimate.

What the GLM does not tell you is whether the young driver's 99th percentile loss is 4x higher, 8x higher, or 15x higher than the family saloon driver's, even after adjusting for the mean difference. For premium pricing, the mean is enough. For capital allocation, reinsurance treaty pricing, and IFRS 17 risk adjustments, the tail is everything.

[`insurance-severity`](https://github.com/burning-cost/insurance-severity) is a Python library implementing Distributional Refinement Networks (Avanzi, Dong, Laub & Wong, arXiv:2406.00998, 2024). It takes any fitted baseline model (a statsmodels Gamma GLM, a CatBoost model) and refines it into a full predictive distribution. The result: per-risk quantiles, CRPS scoring, and adjustment factors that show exactly where the distribution deviates from the baseline and why.

```bash
uv add insurance-severity
```

137 tests. No Lightning dependency. Vectorised inference throughout.

---

## The problem with mean-only models

GLMs have been the pricing workhorse for decades, and rightly so. A Gamma GLM with log link gives you interpretable rating factors, multiplicative structure that survives regulatory scrutiny, and a mean estimate that is well-calibrated when the model is well-specified. Most UK motor pricing systems are still built on this foundation.

The limitation is structural. A Gamma GLM models E[Y|X]. The shape parameter (the dispersion) is assumed constant across all risks. Every risk in the portfolio is described by the same distributional family with the same shape; only the mean shifts. This is rarely true in practice.

A no-claims-discount group 0 driver with three prior faults has a different severity *distribution* from a NCB group 9 driver with a clean record, not just a different mean. The variance, skewness, and tail behaviour all differ. When you price using only the mean, you are implicitly assuming those distributional differences do not exist. They do.

The consequences show up in three places where actuaries need distributional accuracy most.

**Capital modelling.** Solvency II SCR requires the 99.5th percentile of aggregate losses. If you roll up mean estimates with a fixed-shape assumption, your SCR is wrong: not directionally, but in the risk differentials across segments. The segments with genuinely fatter tails are undercharging capital; the segments with thinner tails are over-reserving.

**Reinsurance pricing.** An excess-of-loss treaty depends on E[max(Y − R, 0)] where R is the retention. That expectation is entirely driven by the tail beyond R. A Gamma tail calibrated to the mean is often too thin for large property losses and motor third-party bodily injury claims, where the severity distribution has a multi-modal character (attritional layer, large loss layer) that a single Gamma cannot capture.

**IFRS 17 risk adjustment.** The risk adjustment compensates for uncertainty in the amount and timing of claims cash flows. It is not a mean estimate; it is a function of distributional uncertainty. Industry shortcuts such as CV-based or ATAR methods are approximations. A full predictive distribution is the principled approach.

---

## How DRN works

DRN does not replace the baseline model. It refines it.

The approach, from Avanzi et al. (2024), starts from the baseline CDF. Take a fitted Gamma GLM. For each risk, the GLM specifies the probability mass in each of K histogram bins across the loss range. These baseline probabilities encode what the GLM believes about the distribution of losses for that risk.

A neural network then learns multiplicative adjustments to those bin probabilities. For each risk, the network outputs K log-adjustment values, one per bin. The adjustment softmax is applied on top of the log-baseline probabilities:

```
DRN_logit_k = log(b_k) + delta_k(x)
p_k^DRN = softmax(DRN_logit_k)
```

When `delta_k = 0` for all bins, `p_k = b_k` exactly: the DRN reduces to the GLM. The network trains to learn the *deviation* from the baseline. This has a critical practical consequence: the DRN starts as the GLM and refines from there. Bad training does not corrupt the baseline; it just fails to improve on it.

The output is an `ExtendedHistogram`: a spliced distribution combining parametric GLM tails with a piecewise uniform middle region. Below the lowest cutpoint and above the highest, the tail is governed entirely by the baseline distribution. In the middle region, where most claims fall, the DRN's histogram takes over. The tails have proper parametric behaviour, not an arbitrary histogram boundary. For SCR applications, we fix the upper cutpoint at the 99.7th percentile of training claims, ensuring the DRN histogram controls the 99.5th percentile used in Solvency II calculations.

The adjustment factors `a_k = p_k^DRN / b_k` are directly interpretable: `a_k > 1` means the DRN assigns more mass to bin k than the GLM baseline would. These are the audit trail for FCA product governance, showing exactly how the distributional model deviates from the rating model at each loss level.

Training uses Joint Binary Cross-Entropy (JBCE) loss rather than standard NLL. For each cutpoint, JBCE treats the problem as a binary question: does the loss fall at or below this threshold? The CDF at cutpoints is computed from the PMF via cumulative sum, stable and differentiable. NLL for histogram distributions becomes numerically unstable when bins have varying widths or sparse coverage; JBCE does not.

---

## The API

The workflow is three steps: fit a baseline, wrap it, fit the DRN.

```python
from insurance_severity.drn import DRN, GLMBaseline
import statsmodels.api as sm

# Fit a baseline GLM
glm = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log())).fit()
baseline = GLMBaseline(glm, family='gamma')

# Refine with DRN
drn = DRN(baseline, hidden_size=75, max_epochs=500, patience=30)
drn.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Full distributional predictions
dist = drn.predict_distribution(X_test)
mean = dist.mean()
q95 = dist.quantile(0.95)
crps = drn.score(X_test, y_test, metric='crps')
```

The `dist` object is an `ExtendedHistogramBatch`: a vectorised distributional output with the full method surface: `mean()`, `variance()`, `quantile(p)`, `cdf(y)`, `pdf(y)`, `sample(n)`. All vectorised; no Python loops over observations.

For SCR applications, use `dist.quantile(0.995)` per policy to get per-risk 1-in-200 estimates. For reinsurance treaty pricing:

```python
import numpy as np

def excess_mean(dist, retention, y_max):
    y_grid = np.linspace(retention, y_max, 1000)
    sf = 1 - dist.cdf(y_grid)  # (n, 1000) survival function
    return np.trapz(sf, y_grid)  # (n,) per-policy expected excess loss
```

The adjustment factors are available directly:

```python
a_k = drn.adjustment_factors(X_test)  # (n, K) — interpretable
```

Plot these by rating factor to see where the distributional model diverges from the GLM mean estimate, and at which loss levels. This is the output that supports a regulatory conversation about whether your distributional model is adding genuine signal or fitting noise.

---

## Why DRN, not the alternatives?

The distributional insurance modelling space has several existing approaches. We have a clear view of where each sits.

**Mixture Density Networks (MDN)** fit a mixture of Gaussians directly to the data, with no baseline and no actuarial prior. In practice, MDNs are prone to mode collapse, the mixture components are not interpretable, and there is no guarantee the mean aligns with your rating model. For a portfolio where the GLM is already approved by actuarial management and regulators, replacing it entirely with an MDN is neither practical nor defensible.

**CANN (Schelldorfer & Wüthrich, 2019)** refines the GLM mean via a neural network skip connection. This is the right philosophical approach: augment, do not replace. But CANN only refines the mean. You get a better point estimate; you still have no distributional output.

**[`insurance-distributional`](/2026/03/05/insurance-distributional/)** (our CatBoost-based distributional GBM) models the distribution parameters directly, mean and dispersion simultaneously. It is fast (no neural network training), works well for operational pricing, and produces quantile estimates. It can produce quantile crossing at the extremes, and beyond the training data range the tails are undefined. We use `insurance-distributional` for fast quantile estimates in pricing; we use `insurance-severity` when distributional correctness matters: capital, reinsurance, reserve uncertainty.

The honest comparison:

| | `insurance-distributional` | `insurance-severity` |
|---|---|---|
| Training time | Minutes | 30 min (100k obs, CPU) |
| Quantile crossing | Possible | Impossible (softmax guarantee) |
| Tail beyond training data | Undefined | Parametric baseline |
| GLM baseline integration | No | Yes |
| Audit trail | None | Adjustment factors `a_k` |
| Best for | Operational pricing | Capital, reinsurance, reserve |

---

## What the paper shows

Avanzi et al. (2024) benchmark DRN against GLM, CANN, DDR (Deep Distribution Regression), and MDN on insurance datasets. DRN outperforms all four on NLL, CRPS, and quantile loss at the 25th, 50th, 75th, 95th, and 99th percentiles.

The improvement is largest at the 95th and 99th percentiles — precisely the region that matters for tail risk pricing and Solvency II SCR. At the mean (RMSE), DRN matches the GLM baseline. This is by design: the mean regularisation term and GLM initialisation force the DRN mean to stay close to the baseline, so you do not lose rating model consistency when adding distributional accuracy.

The paper was presented at the Insurance Data Science Conference at Bayes Business School, London, in June 2025, by Tian Dong (UNSW). The research code exists as `pip install drn` (v0.0.1, github.com/EricTianDong/drn).

Our `insurance-severity` takes that research implementation and adds what production use requires: external baseline integration (any fitted model, not just the library's own GLM class), a raw PyTorch training loop without Lightning (dropping a ~100MB dependency), vectorised histogram CDF inference using `torch.searchsorted` (O(n log K) versus the original's O(n*K) Python loop), fixed cutpoint count at 100-200 regardless of dataset size (the original scales with n, producing 10,000 cutpoints at n=100k), and full model serialisation that saves GLM coefficients alongside network weights.

---

## UK regulatory applications

**Solvency II SCR.** Per-policy distributional forecasting with `drn.predict_quantile(0.995)`, followed by Monte Carlo simulation across the portfolio, gives a SCR roll-up based on individually calibrated severity distributions rather than a single assumed Gamma shape. The parametric tail at the upper boundary means the 99.5th percentile is not an extrapolation from a histogram edge.

**IFRS 17 risk adjustment.** The full distributional output from DRN allows the RA to be set at the 75th percentile confidence level of the aggregate distribution, statistically grounded, auditable, and reproducible, rather than using CV-based or ATAR approximations.

**FCA PROD 4 (product governance).** FCA PROD 4.2.9 requires demonstration that products offer fair value for the target market. DRN enables showing the full loss distribution by customer segment. If mean-only pricing leads to systematic underpricing of the 95th percentile for one segment relative to another (which a GLM mean model cannot detect), the adjustment factors `a_k` expose this. That is the evidence trail.

**Reinsurance treaty pricing.** UK excess-of-loss treaties require expected excess loss above the retention. With `ExtendedHistogramBatch.cdf()`, this calculation uses the full per-risk distributional output rather than a Gamma tail assumption. The improvement is largest for property damage (partial vs total loss bimodality) and motor bodily injury (attritional vs catastrophic injury distributions).

---

## Getting started

```bash
uv add insurance-severity
```

Dependencies: `torch>=2.0`, `numpy`, `scipy`, `statsmodels`. No Lightning. No SHAP in v1. Gamma and LogNormal baseline families are supported; Inverse Gaussian and Tweedie are in v2.

The library has 137 tests covering baseline integration, training stability, distributional output, and serialisation. The training loop includes gradient clipping, early stopping with configurable patience, and `baseline_start=True` by default. The DRN initialises as the GLM and refines from there, so training instability cannot regress below the baseline.

Full API documentation and worked examples at [github.com/burning-cost/insurance-severity](https://github.com/burning-cost/insurance-severity).

---

## See also

- [**insurance-distributional**: Distributional GBM for operational pricing](/2026/03/05/insurance-distributional/) — CatBoost-based joint modelling of mean and dispersion. Faster than DRN for pricing; use DRN when tail correctness is non-negotiable.
- [**insurance-gam**: Additive Neural Attention Models for pricing](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/) — interpretable neural network that can serve as a DRN baseline when you need more covariate flexibility than a GLM but more interpretability than a GBM.
- [**insurance-monitoring**: Calibration diagnostics for distributional models](/2026/03/07/insurance-calibration/) — reliability diagrams, CRPS decomposition, and coverage testing. Run these on DRN output before presenting distributional results to capital or reinsurance teams.
