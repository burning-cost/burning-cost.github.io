---
layout: post
title: "We Changed Our Minds: Boulevard EBM CIs Are Going Into insurance-gam"
date: 2026-04-04
categories: [model-governance, interpretability]
tags: [ebm, confidence-intervals, boulevard, insurance-gam, model-governance, kernel-ridge-regression, arXiv-2601.18857, pricing]
description: "In March we said we weren't building Boulevard EBM confidence intervals. We've reversed that decision. MSE/Gaussian models only — but that covers enough governance use cases to be worth shipping. Here is what BoulevardEBM does and where it breaks."
math: true
author: Burning Cost
---

In [our March review of Fang et al. (AISTATS 2026)]({{ '/2026/03/26/boulevard-ebm-confidence-intervals-without-bootstrapping/' | relative_url }}), we concluded that Boulevard EBM confidence intervals were not going into `insurance-gam`. The asymptotic normality requires Gaussian noise. Insurance responses are not Gaussian. We scored the paper 11/20 for insurance applicability and moved on.

We have reconsidered. We are building it.

The reason is not that the Gaussian limitation disappeared. It has not. The reason is that we kept encountering the same governance question in EBM review meetings: *is that rating factor actually significant, or is it fitting noise?* Boulevard is the only way to answer that question analytically, without 100 bootstrap refits. For MSE models — loss development, continuous LR targets, log-residual analysis — it is directly applicable. For Poisson and Gamma, we will offer a clearly-labelled heuristic with an explicit warning. That is the honest position.

---

## What the governance problem actually is

Your regulator — whether under PRA CP6/24 or Solvency II Article 121 — wants to know that the variables in your pricing model are material. An EBM shape function is a continuous curve. If the driver age shape function rises at age 19, drops unexpectedly at age 22, and rises again at age 25, you need to be able to say whether that non-monotonicity is signal or noise. "We looked at it and it seemed plausible" is not a satisfying answer to a model validation team.

The standard response is bootstrapping. InterpretML's `outer_bags` parameter (set to 100 for anything submission-quality) trains 100 separate EBMs and reports spread across bags as an uncertainty estimate. On a UK motor book of 500,000 policies with 20 features, that is 100 full training runs — roughly 8 hours on a single machine. For annual pricing cycles this is survivable. For investigative work, feature selection sweeps, or any workflow where you want to test 30 candidate variables and check which ones have non-zero marginal effects, it is a genuine bottleneck.

Boulevard, when its assumptions hold, eliminates that cost.

---

## The mechanism

The standard EBM boosting update accumulates trees additively:

$$\hat{f}_b = \hat{f}_{b-1} + \eta \cdot \tilde{t}_b$$

Boulevard replaces this with a moving average:

$$\hat{f}_b \leftarrow \frac{b-1}{b} \cdot \hat{f}_{b-1} + \frac{\lambda}{b} \cdot \tilde{t}_b$$

Each new tree contributes $1/b$ of its value. As $b$ grows, each additional tree matters less — harmonic decay. This is a Robbins-Monro stochastic approximation update, and it has a known fixed point: kernel ridge regression. The feature-$k$ estimate converges to:

$$\tilde{y}_k^* = J_n K^{(k)} \left[\lambda I + J_n K\right]^{-1} y$$

where $K^{(k)}$ is the kernel induced by trees fitted on feature $k$, and $J_n$ is the centering matrix. The prediction at any point $x$ is a linear function of the training labels $y$.

Linear functions of Gaussian variables are Gaussian. That is where the confidence intervals come from. Theorem 4.12 of Fang et al. gives:

$$\hat{f}^{(k)}(x) \pm z_{1-\alpha/2} \cdot c_E^{-1} \cdot \hat{\sigma} \cdot \left\| r_E^{(k)}(x) \right\|$$

where $r_E^{(k)}(x)$ is the influence vector at $x$ — how much each training observation pulls the prediction — and $c_E$ is a correction for the shrinkage structure. This interval achieves asymptotic nominal coverage as $n \to \infty$. No bootstrap. No refit. Compute it once, at prediction time.

The computational trick: EBMs discretise inputs into $m \leq 512$ bins. The tree kernel factorises as $S_n^{(k)} = b_n^{(k)} B^{(k)} (b_n^{(k)})^T$ where $B^{(k)}$ is $m \times m$, not $n \times n$. The matrix inversion is $O(m^3)$ — roughly 134 million floating-point operations for $m = 512$, done once at training time. Per-query inference is $O(m^2)$. For a 500,000-row dataset, this is not a bottleneck.

---

## What we are shipping

`BoulevardEBM` replaces the standard InterpretML boosting loop with the moving-average update. `BoulevardInference` computes the bin-space kernel, solves the $m \times m$ system, and produces confidence intervals.

```python
import numpy as np
from insurance_gam.ebm import BoulevardEBM, BoulevardInference

rng = np.random.default_rng(42)
n = 5_000
X = rng.standard_normal((n, 4))
# Additive Gaussian target — this is the regime where Boulevard CIs are valid
y = 2.0 * X[:, 0] + np.sin(3 * X[:, 1]) - 0.5 * X[:, 2] + rng.normal(0, 0.5, n)

model = BoulevardEBM(
    lambda_=1.0,          # regularisation strength, analogous to shrinkage in standard EBM
    max_rounds=500,       # boosting rounds
    max_bins=256,         # bins per feature
    loss="mse",           # mse only — see below for the Poisson warning
)
model.fit(X, y)

inference = BoulevardInference(model)
inference.fit(X, y)  # computes the bin-space kernel and sigma_hat

# Confidence bands on a grid for feature 0 (driver age, log-vehicle-value, etc.)
x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
shape_est, ci_lower, ci_upper = inference.shape_ci(feature=0, x_values=x_grid, alpha=0.05)

# Feature significance: does the 95% CI exclude zero at all grid points?
significant = not np.any((ci_lower <= 0) & (ci_upper >= 0))
print(f"Feature 0 excludes zero across range: {significant}")
```

The `shape_ci` method returns the shape function estimate and pointwise 95% confidence bands. Plotting these for each rating factor gives your governance committee something concrete: not "the EBM found a non-linear effect" but "the non-linear effect at ages 19–22 has a 95% confidence interval that excludes zero, based on asymptotic normality of the Boulevard ensemble".

---

## The Poisson problem, clearly stated

We said it in March and we are saying it again: the CLT requires homoscedastic Gaussian noise. For claim frequency (Poisson), claim severity (Gamma), or pure premium (Tweedie), this does not hold. The variance structure is wrong. Boulevard CIs applied directly to a Poisson EBM would produce intervals with no valid coverage guarantee.

`BoulevardEBM` will raise an error if you pass `loss="poisson"` and `compute_ci=True` without an explicit override flag. The override flag exists because there is a heuristic that is sometimes useful: fit on log-residuals from a calibrated Poisson GLM, apply Boulevard MSE theory to those residuals, and treat the CIs as approximate. Log-residuals from a well-calibrated multiplicative model are closer to homoscedastic than the raw response — not perfectly so, but close enough to be informative in exploratory work.

We will not pretend this has the same guarantee as the Gaussian case. The output will be labelled `approx_ci` rather than `ci`, and the documentation will be explicit. If you want theoretically valid uncertainty quantification for Poisson EBMs, the answer is still `outer_bags=100` and a long coffee break. We are watching the Hooker group's publication trajectory — an exponential family extension is the logical next step from this paper, and if it appears, we will implement it.

---

## Using this in governance submissions

For Solvency II Article 121(d) (use test documentation) and PRA CP6/24 (model risk identification for insurance models), the materiality of each variable needs to be defensible. Boulevard CIs give you two things regulators can read:

**Shape function plots with confidence bands.** Each rating factor gets a curve with 95% CI shading. A factor where the CI spans zero across most of the range is a candidate for removal. A factor where the CI is narrow and well above zero is material. This is a sharper statement than "we looked at the SHAP values and they seemed reasonable."

**Feature significance tests.** For each feature, you can report the minimum absolute value of the shape function across the rating range divided by the CI half-width — a signal-to-noise ratio. Features below a threshold (say, SNR < 1.5) are borderline; features above it are unambiguously material. This is defensible methodology, not eyeballing.

The honest caveat for a model governance pack: "Confidence intervals are derived from Boulevard asymptotic normality (Fang et al., AISTATS 2026, Theorem 4.12), which assumes Gaussian residuals. This model uses MSE loss on [target variable], which satisfies the assumption approximately. Intervals should be interpreted as approximate for non-Gaussian targets."

That is a better position than no uncertainty quantification at all.

---

## What changed since March

The March post said we needed either a public implementation from the Hooker group, or an exponential family extension, before reconsidering. Neither has appeared.

What changed is that we kept being asked to answer the feature significance question, and kept giving unsatisfying answers. "Run 100 bags" is correct but slow. "Look at the shape function" is not evidence. Boulevard, with its limitations clearly stated, is the best available answer for MSE cases, and the heuristic extension to log-residuals is useful enough for Poisson exploratory work to be worth shipping with a clear warning.

We changed our minds. That is the position.

---

The paper is at [arXiv:2601.18857](https://arxiv.org/abs/2601.18857). `insurance-gam` is at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam). The March explainer of the Boulevard mechanism is [here]({{ '/2026/03/26/boulevard-ebm-confidence-intervals-without-bootstrapping/' | relative_url }}).
