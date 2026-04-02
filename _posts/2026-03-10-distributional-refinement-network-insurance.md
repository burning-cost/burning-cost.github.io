---
layout: post
title: "GLMs Predict Means. DRN Predicts Everything Else."
date: 2026-03-10
categories: [techniques, libraries]
tags: [drn, distributional, neural-network, glm, quantile, solvency-ii, scr, crps, pytorch, insurance-severity]
description: "Distributional Refinement Networks wrap any GLM to produce a full predictive distribution. insurance-severity - neural severity modelling for UK motor pricing."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-drn` has been consolidated into [`insurance-severity`](/insurance-severity/). Install with `uv add insurance-severity` — all functionality described here is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-severity)
</div>


Here is the problem with GLMs: they are correct on average. That is not a backhanded compliment. It is a precise statement of what a GLM does. Given a set of covariates, it predicts the conditional mean of the response. If your Gamma GLM is well-specified, the mean prediction for a 19-year-old male driving a modified hatchback will be accurate in expectation. The mean is fine.

The tail is not fine.

A young driver shifts the 99th percentile of claim severity far more than the median. The same covariates that push average severity up by 30% may push the 99.5th percentile up by 80%. A GLM with a Gamma distribution will stretch the whole distribution proportionally - it has no mechanism to distort the shape by covariate. If the distributional form is wrong in the tail, the mean can still be right while the Value-at-Risk is significantly wrong. And for Solvency II, for per-policy SCR, for tail risk pricing, the tail is what you are selling.

Distributional GBMs go further. CatBoost's quantile and expectile modes let you target different parts of the distribution, and our [`insurance-distributional`](/2026/03/05/insurance-distributional/) library fits full parametric distributional families (Tweedie, Gamma) via gradient boosting. But neither approach builds on a GLM baseline - they start from scratch, which means discarding the actuarial calibration that took years to embed.

[`insurance-severity`](https://github.com/burning-cost/insurance-severity) takes a different path. It wraps any baseline model - GLM, GBM, whatever you have - with a neural network that refines the entire predictive distribution without discarding the baseline. The baseline's calibration is preserved exactly when the neural network has nothing useful to add. When it does have something to add, it adjusts the distribution bin by bin, differently for each covariate profile. 2,835 lines, 137 tests, v0.1.0.

```bash
uv add insurance-severity
```

---

## The architecture

DRN - Distributional Refinement Network - comes from Avanzi, Dong, Laub, and Wong at the University of Melbourne and UNSW, published June 2024 (arXiv:2406.00998) and presented at the Insurance Data Science Conference at Bayes Business School in June 2025.

The key insight is this: rather than predicting a full distribution from scratch, DRN refines the baseline distribution you already have. The mechanism is straightforward once you see it.

First, choose K+1 cutpoints spanning the range of your target variable. These divide the outcome space into K bins. The default scheme picks cutpoints at empirical quantiles of the training data, with a minimum-observations constraint to avoid degenerate bins.

Second, for each policy, use the frozen baseline (GLM or GBM) to compute baseline bin probabilities. If the baseline predicts a Gamma distribution with mean mu and dispersion phi, the probability mass in bin k is just the CDF difference: F(c_{k+1} | x) - F(c_k | x). You now have a K-vector of baseline probabilities per policy.

Third, pass the same covariate vector through a feedforward neural network. The network outputs K log-adjustment values - one per bin.

Fourth, the refined distribution is:

```
p_k(x) = softmax( log(b_k(x)) + delta_k(x) )
```

where `b_k(x)` is the baseline bin probability and `delta_k(x)` is the neural network output for bin k. When all deltas are zero, `softmax(log(b_k))` recovers `b_k` exactly - the DRN reduces to the GLM baseline. The neural network can only improve on the baseline, not corrupt it, and initialising with `baseline_start=True` starts training exactly at the baseline.

The tails - outcomes below the lowest cutpoint or above the highest - use the baseline distribution unchanged. This is the sensible design choice: extreme observations are rare, neural networks have little signal there, and the baseline's parametric form (Gamma, LogNormal, InverseGaussian) handles tails better than a histogram with sparse bins.

---

## Fitting

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from insurance_severity.drn import DRN, GLMBaseline

# GLMBaseline wraps a fitted statsmodels GLM — fit the GLM first
sm_glm = smf.glm(
    "claim_amount ~ age_band + vehicle_group + ncd + region",
    data=df_train,
    family=sm.families.Gamma(sm.families.links.Log()),
).fit()
baseline = GLMBaseline(sm_glm)

# Wrap with DRN
drn = DRN(
    baseline=baseline,
    hidden_size=75,
    num_hidden_layers=2,
    dropout_rate=0.2,
    baseline_start=True,   # initialise at baseline — recommended
    lr=1e-3,
    max_epochs=200,
    patience=20,           # early stopping
)
drn.fit(X_train, y_train)
```

The `baseline_start=True` flag forces the network weights to initialise such that all delta_k are zero at the start of training. This gives training a clean starting point - the model begins as a GLM and departs only where the data justifies it. Without it, random initialisation can destabilise early training.

The loss function is joint binary cross-entropy (JBCE) over bin indicators, not the standard NLL. Each bin becomes a binary classification problem: does the observation fall below this cutpoint? JBCE over K cutpoints is equivalent to optimising the CDF shape at each cutpoint simultaneously. The result is a better-calibrated distribution than NLL minimisation on a small number of quantiles.

---

## Extracting distributions

The output of `drn.predict_distribution(X)` is an `ExtendedHistogramBatch` - a vectorised object representing one full predictive distribution per policy. All operations are vectorised across the batch.

```python
# Predict full distributions for 10,000 policies
dist = drn.predict_distribution(X_holdout)   # ExtendedHistogramBatch of shape (10_000,)

# Conditional mean — should match GLM closely if baseline is well-calibrated
means = dist.mean()             # shape (10_000,)

# Quantiles — CDF-inverted, vectorised
median    = dist.quantile(0.50)
p75       = dist.quantile(0.75)
p99       = dist.quantile(0.99)
scr_level = dist.quantile(0.995)  # Solvency II SCR per policy

# CRPS — proper scoring rule for distributional accuracy
crps = dist.crps(y_holdout)     # shape (10_000,) — lower is better

# Expected Shortfall above 99th percentile
es99 = dist.expected_shortfall(0.99)

print(f"Mean CRPS: {crps.mean():.4f}")
print(f"Median policy SCR percentile: £{np.median(scr_level):,.0f}")
```

The CDF and quantile methods use numpy searchsorted for O(n log K) performance - no Python loops over observations. For the tails (below c_0 or above c_K) the implementation falls back to the baseline parametric CDF, which preserves correct behaviour even for extreme claims.

---

## Solvency II SCR per policy

This is the use case that justifies building DRN rather than using a GLM with a parametric tail.

Solvency II requires insurers to hold capital sufficient to survive a 1-in-200 year event. At the portfolio level, this is handled through internal models or standard formula. But per-policy SCR - the marginal contribution of a single policy to the portfolio's 99.5th percentile - requires a full predictive distribution for that policy.

A GLM with a Gamma distribution will give you a 99.5th percentile, but it assumes the Gamma shape parameter is constant across policies. That assumption fails exactly where it matters most: for the policies whose covariate profiles push the tail hardest. DRN lets the distribution shape vary freely by covariate, constrained only by what the data supports.

```python
# Per-policy SCR (99.5th percentile of individual claim severity)
scr = dist.quantile(0.995)

# Ratio of SCR to mean — measures tail heaviness by policy
tail_ratio = scr / dist.mean()

# Segment by age band (for a motor portfolio)
age_band = X_holdout['age_band']
for band in sorted(age_band.unique()):
    mask = age_band == band
    print(f"Age {band}: mean tail ratio = {tail_ratio[mask].mean():.2f}")
```

On a typical UK motor portfolio, DRN reveals that the tail-to-mean ratio varies materially by age band - something a constant-shape Gamma cannot capture. That variation is directly priceable: policies with heavy tails warrant a loading above the mean-calibrated premium.

---

## Interpretability: adjustment factors

DRN is a neural network, which raises the usual model risk questions. The `adjustment_factors()` method addresses this directly.

```python
# Adjustment factors: DRN probability / baseline probability, per bin
factors = drn.adjustment_factors(X_holdout)
# shape: (n_policies, K) — values > 1 mean DRN assigns more mass than GLM

# For a specific high-risk policy profile (factors is a Polars DataFrame)
young_driver_idx = int((X_holdout['age'] < 22).arg_true()[0])
af = factors.row(young_driver_idx)

cutpoints = drn.cutpoints  # K+1 values
bin_centres = 0.5 * (cutpoints[:-1] + cutpoints[1:])

import matplotlib.pyplot as plt
plt.bar(bin_centres, af, width=np.diff(cutpoints) * 0.8)
plt.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Claim amount (£)')
plt.ylabel('DRN / GLM probability ratio')
plt.title('How DRN adjusts the GLM distribution: age < 22 driver')
plt.show()
```

A factor above 1.0 in the upper bins means DRN assigns more probability to large claims than the GLM would. A factor below 1.0 in the middle bins means DRN reallocates some mass away from the body. This plot is directly auditable - a model risk reviewer can look at it and ask whether the shift is directionally sensible for the covariate profile.

The CANN architecture (Schelldorfer and Wuthrich, 2019) gave actuaries an interpretability handle on mean adjustments via skip connections. DRN extends that to the whole distribution: you can see not just whether the neural network is pushing the mean up, but whether it is specifically fattening the tail.

---

## Diagnostics

The `DRNDiagnostics` class provides the three checks we run on every model before sign-off.

```python
from insurance_severity.drn import DRNDiagnostics

diag = DRNDiagnostics(drn)

# PIT histogram — should be uniform if calibration is right
diag.pit_histogram(X_holdout, y_holdout)

# Quantile calibration — empirical vs predicted coverage
diag.quantile_calibration(X_holdout, y_holdout)

# CRPS by segment — where is the model weakest?
diag.crps_by_segment(
    X_holdout, y_holdout,
    segment_col='age_band'
)
```

The PIT (Probability Integral Transform) histogram is the distributional equivalent of a residual plot. If `P(Y <= y | x)` is uniform over [0,1] when evaluated at the true observation y, the distribution is calibrated. A U-shaped PIT means the model is under-dispersed - it is concentrating too much mass near the mean. An inverted-U means over-dispersion. Spikes at 0 or 1 mean the model is too frequently predicting values outside the observed range.

We require a PIT histogram that passes visual inspection and a Kolmogorov-Smirnov p-value above 0.05 before we count a DRN model as production-ready.

---

## When to use DRN vs distributional GBM

These two libraries answer different questions.

Use distributional GBM when: you do not have a GLM baseline, you want tree-based feature interactions, and a parametric distributional family (Tweedie, Gamma) is adequate for your use case.

Use DRN when: you have an existing GLM that has been validated and is in production, and you want distributional refinement without discarding that validation work. Or when you have reason to believe the Gamma shape is heterogeneous across policies - which is almost always true for UK motor injury severity. Or when you need Solvency II alignment and want to defend the model to an internal model team.

The two approaches are complementary. We use the GLM as baseline for DRN on lines where GLM pricing is well-established (motor, property). We use distributional GBM on lines where GLM is less embedded and tree models dominate.

---

## Model risk considerations

DRN introduces neural network parameters on top of a well-understood GLM baseline. There are three things to watch.

First, overfitting in the upper bins. With few extreme observations, the neural network can overfit bin probabilities in the tail. The KL regularisation term in the loss (`drn_regularisation`) penalises large departures from the baseline; increase `kl_alpha` if you see validation CRPS diverging from training CRPS. **Note:** the `kl_alpha` regularisation parameter and the associated KL penalty are additions in this library implementation and are not present in the original DRN paper (arXiv:2406.00998). The core architecture — bin probabilities, JBCE loss, and baseline refinement — follows the paper; the KL term is a practical enhancement for production use where overfitting in sparse tail bins is a real concern.

Second, the baseline is frozen. The GLM is not updated during DRN training. If the GLM mean prediction is wrong for a particular segment, DRN cannot correct the mean - it can only adjust the shape. Run `diag.quantile_calibration` and verify that DRN's 50th percentile tracks the GLM mean before trusting the tail.

Third, save and load requires re-providing the baseline. `torch.save` serialises the DRN network weights, not the fitted GLM. When loading a saved model, pass the baseline explicitly:

```python
drn_loaded = DRN.load('drn_v1.pt', baseline=baseline)
```

This is a deliberate design choice - it forces explicit baseline versioning rather than silently loading a stale GLM.

---

## Summary

GLMs predict means. Even well-calibrated, parametric-tailed GLMs assume the distributional shape is fixed across policies. DRN relaxes that assumption using a neural network that adjusts bin probabilities directly, initialises at the baseline, and degrades gracefully to the GLM when it has nothing to add.

The output - `ExtendedHistogramBatch` - gives you a full predictive distribution per policy: CDF, quantiles, mean, CRPS, Expected Shortfall. The adjustment factors are directly auditable. The diagnostics are the same ones you would use to validate any distributional model.

For Solvency II capital modelling, tail risk loading, or anywhere you need the 99th percentile rather than the mean, DRN is the tool we reach for.

---

## See also

- [Distributional GBMs for insurance pricing](/2026/03/05/insurance-distributional/)
- [Actuarial Neural Additive Models](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/)
- [Calibration testing: beyond the residual plot](/2026/03/09/insurance-calibration/)
- [Conformal prediction intervals for insurance](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
