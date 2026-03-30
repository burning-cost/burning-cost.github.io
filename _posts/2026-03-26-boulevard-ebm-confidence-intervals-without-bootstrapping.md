---
layout: post
title: "Boulevard: EBM Confidence Intervals Without Bootstrapping"
date: 2026-03-26
categories: [techniques]
tags: [ebm, gam, explainable-boosting, confidence-intervals, statistical-inference, boulevard, kernel-ridge-regression, gaussian, poisson, tweedie, interpretML, insurance-gam, pricing]
description: "Fang, Tan, Pipping, and Hooker (AISTATS 2026) show that replacing additive boosting with a moving-average update makes EBMs converge to kernel ridge regression — and that means asymptotic normality, analytic confidence intervals, and no bootstrapping. Elegant. But the theory is Gaussian-only, which is the entire problem for insurance."
---

InterpretML's bagged EBM already gives you uncertainty bands on rating factor shapes. Run 14 bags (or 100 if you want anything reliable), fit a full EBM on each subsample, and take the spread across bags as your confidence interval. It works. It is also slow — on a UK motor book of 500,000 policies, 100 bags means 100 full training runs, which is O(B × n × p × T). For exploration it is fine. For production retraining pipelines or large-scale hyperparameter searches, it becomes a bottleneck you notice.

Fang, Tan, Pipping, and Hooker — "Statistical Inference for Explainable Boosting Machines", arXiv:2601.18857, accepted AISTATS 2026 — propose a different path. Replace the standard additive boosting update with a moving average. Do it carefully, and the ensemble converges to a kernel ridge regression fixed point. Kernel ridge regression has a known limiting distribution. That means you get asymptotically exact confidence intervals from a single training run, at a fraction of the cost.

The catch for insurance is substantial and we will not bury it: the asymptotic normality requires Gaussian noise. Insurance responses are not Gaussian. We score this paper 11/20 for insurance applicability, and we are not implementing Boulevard CIs in `insurance-gam`. But the idea is genuinely novel and the mechanism is worth understanding.

---

## The moving-average update

Standard gradient boosting accumulates trees additively. After _b_ rounds, the ensemble is:

```
f̂_b = f̂_{b-1} + η · t̃_b
```

where _η_ is the fixed shrinkage parameter and _t̃_b_ is the new tree. The scaling of each tree's contribution is constant regardless of how many trees have already been fitted.

Boulevard replaces this with:

```
f̂_b ← ((b-1)/b) · f̂_{b-1} + (λ/b) · t̃_b
```

The new tree contributes _1/b_ of its value. As _b_ grows, each additional tree has less influence than the last — a harmonic decay. This is a stochastic approximation (Robbins-Monro type) update, and it has a known fixed point.

Zhou and Hooker (JMLR 23:183, 2022) showed that this update converges to kernel ridge regression for vanilla GBMs. The current paper extends that result to EBMs, where boosting cycles over features _k = 1 ... p_ rather than fitting a single joint tree. After enough cycles, each feature's learned function _f̂^(k)_ converges to:

```
ỹ_k* = J_n K^(k) [λI + J_n K]^{-1} y
```

where _K^(k) = E[S^(k)]_ is the kernel induced by the trees fitted on feature _k_, and _J_n_ is the centering matrix. Each prediction is a linear function of the training labels _y_ via this kernel. Linear functions of Gaussian random variables are Gaussian. That is where the confidence intervals come from.

---

## What the confidence intervals actually look like

For feature _k_ at a query point _x_, the interval is:

```
f̂_E^(k)(x) ± z_{1-α/2} · c_E^{-1} · σ̂ · ‖r_E^(k)(x)‖
```

where _r_E^(k)(x)_ is the per-feature influence vector (how much each training observation pulls the prediction at _x_), _c_E_ is an algorithmic correction for the shrinkage structure, and _σ̂_ is the residual standard deviation. The theorem (Theorem 4.12) is that this interval achieves asymptotic nominal coverage as _n_ → ∞.

The key computational trick is that the influence vectors are computed in bin space. EBMs use histogram binning — with _m_ ≤ 512 bins per feature, the tree kernel _S_n^(k)_ factorises as:

```
S_n^(k) = b_n^(k) B^(k) (b_n^(k))^T
```

where _B^(k)_ is _m × m_, not _n × n_. The _m × m_ matrix inversion is O(m³). For _m_ = 512, that is about 134 million floating-point operations — once, at training time, independent of _n_. Per-query inference is O(m²). For prediction intervals on a 500,000-row dataset with 100 bootstrap bags, InterpretML needs 100 full training runs; Boulevard needs one training run plus a 512 × 512 matrix solve.

The paper also reports a minimax rate result (Corollary 4.13): Boulevard achieves _O(n^{-2/3})_ MSE for nonparametric Lipschitz GAM estimation, matching the lower bound from Yuan and Zhou (2015). This confirms Boulevard is rate-optimal — not just computationally convenient, but statistically tight.

---

## Comparing to InterpretML's current approach

InterpretML's bagging is not theoretically grounded as asymptotic inference. The `outer_bags` parameter (default 14, recommended 100 for stability) trains _B_ separate EBMs on 85% subsamples and reports mean ± k × standard deviation across bags. This is an empirical dispersion proxy, not a confidence interval derived from a CLT. In finite samples, with correlated features and moderate n, coverage can drift from nominal with no warning.

Boulevard, if the assumptions hold, gives coverage that converges to nominal as _n_ grows. That is a harder guarantee.

On predictive accuracy, the paper shows Boulevard achieving comparable MSE to standard InterpretML EBM on UCI benchmarks (Wine Quality, Air Quality, Obesity), while being substantially more resistant to overfitting than vanilla GBMs. The moving-average structure provides regularisation automatically — no early stopping required. Whether or not you want the CIs, an EBM that degrades more gracefully without tuning stopping rounds is useful.

---

## Why this does not work for insurance

The asymptotic normality result requires:

```
y = β + Σ_k f^(k)(x^(k)) + ε,    ε ~ N(0, σ²)
```

Homoscedastic Gaussian noise. Insurance responses are not like this:

- Claim frequency: Poisson or negative binomial
- Claim severity: Gamma or lognormal
- Pure premium: Tweedie (compound Poisson-Gamma)

The confidence interval formula plugs in _σ̂_, the residual standard deviation. For a Poisson response, this is not the right dispersion parameter — the variance scales with the mean, and using a single pooled _σ̂_ misrepresents the uncertainty structure entirely. Applying Boulevard CIs to a Tweedie EBM would produce intervals that look sensible but have no valid coverage guarantee. The authors are explicit that exponential family responses are out of scope.

This is not a criticism of the paper. It is a Gaussian regression paper. But it is the constraint that matters for our use case.

There is a partial workaround sometimes floated: apply Boulevard theory to the residuals on the log scale, arguing that log-scale residuals for a multiplicative model are approximately Gaussian. This is informal justification, not theory — log-scale residuals for Poisson are not homoscedastic, and the approximation quality depends on the signal-to-noise ratio in the data. We would not rely on this in a production pricing model without a proper exponential family extension.

---

## What would need to happen for this to transfer

The machinery is:

1. Establish that the Boulevard moving-average update, applied with canonical link and working weights, converges to an iteratively reweighted kernel ridge regression (IRKRR) fixed point for exponential family responses.
2. Derive the asymptotic distribution of the IRKRR estimator with the feature-wise (EBM) training structure.
3. Use the GLM delta method to propagate uncertainty from the linear predictor scale to the response scale.

This is new theoretical work. The Gaussian case uses the fact that the score function (gradient of log-likelihood) is linear in _ε_, which makes the stochastic approximation analysis clean. For Poisson with log link, the score function involves _y/μ - 1_ where _μ_ depends on the current estimate — the analysis is more involved. For Tweedie, the variance function _V(μ) = μ^p_ adds another layer of complication.

The Hooker group (Giles Hooker, Cornell, currently the main theoretical force behind Boulevard) has published precursor work on statistical inference for GBM regression (Fang, Tan, Hooker, NeurIPS 2025, arXiv:2509.23127). The progression is logical: GBM regression → EBM regression → EBM exponential family. Whether that third step appears in 2026 or 2027 we do not know.

---

## Where this sits for insurance-gam

[insurance-gam](/insurance-gam/) wraps InterpretML's `ExplainableBoostingRegressor` directly. Boulevard would require a different training loop — the moving-average update is not a post-processing step on a standard trained EBM, it is a different boosting algorithm. You cannot implement Boulevard CIs as a wrapper. You would need to rewrite the boosting engine.

As of March 2026, there is no public implementation of Boulevard EBMs from the authors. No GitHub repo, no PyPI package. The paper's algorithm descriptions are clear enough to implement from scratch — the bin-space factorisation is the key piece — but that is a non-trivial build, and it would produce Gaussian-only CIs that are not directly applicable to our models.

We are not building this yet. The decision: watch for either a public implementation from the Hooker group, or an exponential family extension, and reassess. If either appears, Boulevard becomes an immediate BUILD candidate for `insurance-gam`.

The existing `outer_bags` approach in InterpretML remains the default. It is slow at B=100, but it is honest about what it is: an empirical uncertainty estimate that does not claim theoretical coverage guarantees. For regulatory submissions consistent with PRA SS1/23 model risk management requirements, an empirically validated uncertainty band with acknowledged limitations is defensible. A misapplied Gaussian CI is not.

---

The paper is at [arXiv:2601.18857](https://arxiv.org/abs/2601.18857). The prior work on Boulevard for GBMs is Zhou and Hooker (2022) at [arXiv:1806.09762](https://arxiv.org/abs/1806.09762).
