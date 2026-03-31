---
layout: post
title: "Distribution-Free Prediction Intervals for Insurance Pricing — Conformal Methods That Actually Work"
seo_title: "Conformal Prediction Intervals for Insurance Pricing: GLM Coverage Failures and the Distribution-Free Alternative"
date: 2026-03-31
categories: [statistics, techniques]
tags: [conformal-prediction, tweedie, GLM, GBM, LightGBM, insurance-pricing, prediction-intervals, capital-modelling, SCR, Consumer-Duty, rate-adequacy, python, PRA, Solvency-II, uncertainty-quantification]
description: "GLM prediction intervals targeting 95% coverage achieved 57.8% actual coverage on real personal injury data. Conformal prediction fixes this without distributional assumptions. Here is the theory, the empirical evidence, and the Python."
author: burning-cost
---

Actuaries routinely produce prediction intervals. They are used in capital modelling, rate adequacy testing, and increasingly in Consumer Duty frameworks. The standard machinery — Wald intervals from a Tweedie GLM, parametric bootstrap, Bayesian credible intervals — all share a common failure mode: they assume the model is correctly specified. When it is not, the intervals can be badly wrong.

How badly wrong? Liang Hong (arXiv:2503.03659, 2025) fits a Tweedie GLM to 22,036 personal injury claims from the Jong & Heller (2008) dataset. He targets 95% prediction intervals. The actual empirical coverage: **57.8%**. Not a rounding error. Nearly four in ten observations fall outside the stated 95% bounds.

Conformal prediction addresses this without distributional assumptions. We do not think it replaces GLMs. We think it replaces the prediction interval step that comes after fitting one.

---

## Why standard intervals fail

The standard approach has three structural problems, which Hong identifies directly.

**Model misspecification.** Tweedie GLMs are convenient approximations. The true claim distribution is not Tweedie compound Poisson-Gamma with constant dispersion. Nobody thinks it is. But the parametric interval treats it as if it were.

**Selection effect.** Once you have used AIC or BIC to select your model from a candidate set, the model is no longer an independent object — it was chosen to fit this data. Prediction intervals that do not account for this selection are optimistic. Hong & Martin (2018) formalised this problem; it is not commonly appreciated in actuarial practice.

**Lack of finite-sample validity.** GLM interval theory is asymptotic. It tells you that as n → ∞, coverage approaches 1 − α. With finite data — even 22,000 observations — coverage can be substantially below nominal, especially under misspecification.

Conformal prediction sidesteps all three. The coverage guarantee is:

$$P^{n+1}\{Y_{n+1} \in C_\alpha(X_{n+1}; Z^n)\} \geq 1 - \alpha$$

for all n and all exchangeable distributions P. No asymptotic. No distributional assumption. The only assumption is exchangeability — that calibration and test data come from the same distribution. For UK pricing this mostly holds within a product year; we discuss the limits below.

---

## Three approaches, one framework

Three papers published between July 2025 and January 2026 collectively define the state of the art. The `insurance-conformal` library implements all of them.

### 1. Model-free full conformal (Hong, arXiv:2503.03659)

The simplest approach requires no model at all. Define adjusted training responses:

$$W_i = Y_i + \frac{1}{n} \sum_{j=1}^p (X_{n+1,j} - X_{ij})$$

The 95% prediction interval for a new risk is simply (0, W₍ₖ₎), where k = min(n, ⌊(n+1)(0.95) + 1⌋) — the k-th order statistic of the adjusted W values. This is O(n log n). No model, no grid search over candidate response values.

The W correction adjusts each training claim by the sum of covariate differences between the new risk and each training observation, divided by n. It is a crude linear adjustment — it does not use any regression model. This is precisely why the interval is guaranteed regardless of model specification: there is no model.

The cost is wider intervals when a model would have been informative.

```python
from insurance_conformal.claims.hong import HongConformal
import numpy as np

hc = HongConformal()
hc.fit(X_train, y_train)
intervals = hc.predict_interval(X_test, alpha=0.05)
# intervals.shape == (n_test, 2); lower bound is always 0
```

For SCR estimation, set `alpha=0.005` to get a finite-sample valid 99.5% upper bound directly citable in PRA model reviews under SS1/23.

### 2. The h-transformation (Hong, arXiv:2601.21153)

The model-free approach is conservative because it ignores regression structure. Hong's 2026 paper introduces a bridge: decompose Y = h(X) + W. Since training observations are exchangeable, residuals W_i = Y_i − h(X_i) are iid regardless of how h was chosen. A valid iid interval for W gives a valid regression interval for Y:

$$\hat{I}_\alpha = [0,\ W_{(r)} + h(X_{n+1}))$$

where r = min{n, ⌊(n+1)(1 − α) + 1⌋} and W₍ᵣ₎ is the r-th order statistic of calibration residuals.

Here h can be **any** fitted model — your existing pricing GBM, your GLM, anything. A well-fitted h produces small residuals W_i, so W₍ᵣ₎ is small, so the interval is narrow. Table 2 from the paper quantifies this on simulated data (oracle 90% interval length = 3.26):

| Choice of h | Mean interval length | Overhead vs oracle |
|---|---|---|
| h = 0 (no model) | 5.23 | 60% |
| h = linear fit | 3.47 | 6% |
| h = log(1+t) | 3.84 | 18% |

A linear h gets you within 6% of the oracle. This is the practical argument for using your existing pricing model: not to trust its distributional assumptions, but to reduce residual variance so the conformal interval is tight.

```python
from insurance_conformal.claims.hong import HongTransformConformal
from lightgbm import LGBMRegressor

h_model = LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5)
htc = HongTransformConformal(h_model=h_model)
htc.fit(X_train, y_train, X_cal, y_cal)
intervals = htc.predict_interval(X_test, alpha=0.05)
```

### 3. Tweedie non-conformity scores (Manna et al., arXiv:2507.06921)

Manna, Sett, Dey, Gu, Schifano and He — a University of Connecticut / Travelers collaboration — test three Tweedie-specific non-conformity scores for split conformal prediction on real insurance data (AutoClaim, n = 10,296, 100 simulation repetitions, α = 0.05):

**Pearson residual:**
$$R_{i,\text{Pear}} = \frac{|Y_i - \hat{\mu}_i|}{\hat{\mu}_i^{p/2}}$$

This standardises by the Tweedie standard deviation up to the dispersion parameter φ. Crucially, φ cancels in the calibration quantile computation — the conformal Pearson score is φ-free.

**Deviance residual:** Do not use with LightGBM. The deviance score requires root-finding to invert intervals per observation. On heavy-tailed claims it is numerically unstable: mean interval width of 26.32 vs 13.96 for locally weighted Pearson in Table 3. The root-finding instability is not a fringe case; it is the systematic behaviour.

**Locally weighted Pearson (the winner):**
$$R^*_{i,\text{Pear}} = \frac{R_{i,\text{Pear}}}{\hat{\rho}(X_i)}$$

where ρ̂ is a second LightGBM model fitted on training Pearson residuals. This second model captures heteroscedasticity beyond what V(μ) = μᵖ models — it relaxes the constant-φ assumption non-parametrically. Table 3 results:

| Method | Model | Coverage (mean ± SD) | Width (mean ± SD) |
|---|---|---|---|
| Pearson | GLMNET | 0.9495 ± 0.0061 | 14.76 ± 0.62 |
| Pearson | LightGBM | 0.9496 ± 0.0057 | 14.32 ± 0.55 |
| **LW Pearson** | **LightGBM** | **0.9502 ± 0.0054** | **13.96 ± 0.51** |
| Anscombe | LightGBM | 0.9495 ± 0.0059 | 17.79 ± 0.50 |
| Deviance | LightGBM | 0.9496 ± 0.0060 | 26.32 ± 1.34 |

All methods achieve nominal coverage — that is expected, it is what conformal guarantees. The differentiator is interval width. Locally weighted Pearson with LightGBM is the empirically superior method.

```python
from insurance_conformal.claims.tweedie import TwoStageLWConformal
from lightgbm import LGBMRegressor

mean_model = LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5)

cp = TwoStageLWConformal(mean_model=mean_model, p=1.5)
cp.fit(X_train, y_train)       # trains mean model + spread model
cp.calibrate(X_cal, y_cal)     # computes calibration scores
intervals = cp.predict_interval(X_test, alpha=0.05)
# shape (n_test, 2); lower bounds clipped at 0
```

---

## What the guarantee actually is

Conformal gives **marginal (joint) coverage**: across the population of future risks with similar features, at least (1 − α) of prediction intervals will contain the true outcome. It does not give **conditional coverage**: coverage ≥ 1 − α for every individual covariate value x.

This is not a limitation of the implementation. Vovk (2012) and Lei & Wasserman (2014) proved that conditional coverage with bounded intervals is unachievable with finite samples. Marginal validity is the best possible finite-sample guarantee.

For capital modelling and rate adequacy, marginal coverage is the right concept anyway. These are portfolio-level tests. For individual policyholder transparency under Consumer Duty, the population-level framing is also defensible — it matches how the FCA defines outcomes in PS22/9.

---

## Practical applications for UK pricing

### Capital modelling — SCR

Setting α = 0.005 gives a finite-sample valid 99.5% upper bound on individual claims. This is stronger than the parametric Tweedie SCR in common use. The conformal bound holds regardless of distributional assumption, and is directly auditable: "we computed this on a held-out calibration set and the coverage guarantee is finite-sample." That is citable in PRA model reviews under SS1/23.

**Important caveat:** the sum of individual conformal bounds is not itself a conformal interval. Aggregation violates exchangeability. For aggregate portfolio SCR you need a portfolio-level conformal framework, not the sum of per-policy bounds.

### Rate adequacy testing

Run `TwoStageLWConformal` on your held-out validation set. For each pricing segment, flag any segment where more than 5% of risks have their premium below the 5th percentile conformal lower bound on expected claims. This is distribution-free evidence of potential under-pricing — more robust than testing premium < GLM point estimate, because the conformal lower bound accounts for model uncertainty.

### Consumer Duty

FCA PS22/9 requires firms to demonstrate fair outcomes for identifiable groups of customers. Prediction intervals provide transparent uncertainty quantification. A pricing team can demonstrate that its intervals contain the true outcome at the stated confidence level, on held-out data, without relying on parametric distributional assumptions. This is harder to challenge than "our GLM says so."

---

## The exchangeability assumption in UK practice

Conformal requires calibration and test data to be exchangeable — drawn from the same distribution. In UK motor and property, this can fail over time:

- Claims inflation (especially bodily injury in the post-Ogden rate change environment)
- EV and PHEV penetration shifting the risk profile of the vehicle fleet
- Model drift as the book composition changes with distribution mix shifts

We recommend a rolling 12–24 month calibration window rather than using full history. For sequential pricing environments where you want online coverage guarantees, the library's `RetroAdj` class (implementing Jun & Ohn 2025) handles distribution shift explicitly.

---

## Comparison with what actuaries currently do

| Method | Requires correct model? | Coverage guarantee | Computational cost |
|---|---|---|---|
| GLM parametric interval | Yes — distribution and φ | Asymptotic only; 57.8% actual on real data | Negligible |
| Parametric bootstrap | Yes — must resample from correct distribution | Asymptotic | ~1,000 model refits |
| Wild bootstrap | Heteroscedasticity-robust, but model-based | Asymptotic | ~1,000 model refits |
| Bayesian credible interval | Yes — prior + correct likelihood | Posterior validity | MCMC or conjugate approx |
| Conformal (LW Pearson, split) | No | Finite-sample marginal | Single calibration pass |

The implementation cost of conformal is a single pass through a held-out calibration set. You keep your existing GLM or GBM. You add the calibration step. The return is a coverage guarantee that holds regardless of whether your model is correctly specified — which it is not, and you know it is not.

---

## What is not yet in the library

The three papers are well-covered in `insurance-conformal`. Five gaps remain, documented in our Phase 36 plan:

**Exposure weighting.** In UK aggregate pricing, Y_i is aggregate claims with exposure e_i. The correct exposure-weighted Pearson score is |Y_i − e_i·μ̂_i| / (e_i·μ̂_i)^{p/2}. Without this correction, high-exposure policies dominate the calibration score distribution and the resulting quantile is biased. None of the three papers address this.

**Score selection API.** There is no built-in method to select the best non-conformity score by minimum width at nominal coverage on a validation set. We want `select_score(['pearson', 'lw_pearson', 'anscombe'])` returning the winner with a summary table.

**Leaf-level conditional coverage diagnostics.** For GBM models you can check conditional coverage per terminal leaf — which leaves of the tree have sub-nominal coverage. This identifies where the exchangeability assumption is locally violated.

**Anscombe exponent discrepancy.** Manna et al. use exponent 1 − p/3 in the Anscombe score. The library uses the theoretically correct variance-stabilising exponent (2 − p)/2. Both are valid; they produce different interval widths. This needs documentation rather than a code change.

**Asymmetric interval suppression.** Manna et al. find that splitting α to produce separate lower and upper tail quantiles consistently gives wider intervals than symmetric ones, because each tail is estimated from n₂/2 effective calibration points. The library should document and enforce the symmetric interval default for Tweedie data.

---

## The core claim

Actuaries have been producing prediction intervals with an unacknowledged assumption: that the model is correctly specified. The intervals look credible because they have confidence levels attached to them. But a GLM confidence level is an asymptotic property of the interval procedure under the assumed model, not an empirically validated coverage claim.

Conformal prediction inverts this. The coverage guarantee comes first, from exchangeability. The model is used only to make the intervals narrower, not to justify them. The 57.8% coverage on personal injury data is not a fringe result — it is what happens when you use parametric intervals on real insurance data and your model is wrong.

The switch is a single calibration pass on held-out data. The output is an auditable coverage claim that holds for all (n, P). We think this is the correct approach for capital adequacy, rate adequacy, and Consumer Duty transparency. The only obstacle is inertia.

---

## References

1. Liang Hong. "Conformal Prediction of Future Insurance Claims in the Regression Problem." arXiv:2503.03659v3, September 2025.
2. Liang Hong. "A New Strategy for Finite-Sample Valid Prediction of Future Insurance Claims in the Regression Setting." arXiv:2601.21153v1, January 2026.
3. Manna, Sett, Dey, Gu, Schifano & He (The Travelers). "Distribution-Free Inference for LightGBM and GLM with Tweedie Loss." arXiv:2507.06921v1, July 2025.
4. Romano, Patterson & Candès. "Conformalized Quantile Regression." NeurIPS 2019. arXiv:1905.03222.
5. Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
6. Lei & Wasserman. "Distribution-free prediction bands for non-parametric regression." *JRSS-B*, 2014.
7. Hong & Martin. "Valid model-free prediction of future insurance claims." *Insurance: Mathematics and Economics*, 2018.
