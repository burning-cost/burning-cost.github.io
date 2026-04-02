---
layout: post
title: "RobustMMDGLM: Severity Modelling When Gamma GLMs Break"
date: 2026-04-02
categories: [insurance-pricing, machine-learning]
tags: [glm, severity, robust-estimation, mmd, admm, l1-regularisation, insurance-glm-tools, gamma-glm, lasso, outliers, arXiv-2602.21132, actuarial]
description: "insurance-glm-tools v0.2.0 ships RobustMMDGLM — a Gamma GLM that automatically downweights large losses and selects features via L1, based on Kang & Kang (2026). Replaces ad-hoc loss capping with a principled robust estimator."
author: burning-cost
---

Every severity modeller has encountered the problem, even if they have not named it. You fit a Gamma GLM on bodily injury claims, review the relativities table, and one factor has shifted by 15% since last year. You trace it back to three large losses in a single postcode band. You cap the losses at the 99th percentile, refit, and the factor settles down. You log the intervention in your model documentation and move on.

This works. It is also artisanal, unrepeatable, and indefensible to a Solvency II auditor who asks what principled criterion justified the 99th percentile rather than the 97th or the 95th.

[insurance-glm-tools v0.2.0](https://pypi.org/project/insurance-glm-tools/) ships `RobustMMDGLM`, which replaces this workflow with a robust estimator grounded in the theory of Maximum Mean Discrepancy. You fit once, the algorithm automatically downweights outliers, and you get multiplicative relativities that are stable under contamination. The implementation follows Kang & Kang (arXiv:2602.21132, 2026).

---

## Why standard Gamma GLMs break under large losses

The standard Gamma GLM minimises the Gamma deviance, which is a log-likelihood derived estimator. Under correctly specified data, MLE is optimal. The problem is the objective function's behaviour under misspecification.

The Gamma log-likelihood treats each observation's contribution as proportional to `y/mu - log(y/mu)`. For a claim that is ten times the expected value — a large bodily injury loss in a book where the average is £8,000 — this contribution is large. The IRLS algorithm adjusts the coefficient vector to reduce it, pulling factors toward the outliers. With enough large claims concentrated in a rating cell, the MLE estimator shifts the relativity for that cell materially.

MMD takes a different approach. Instead of measuring the discrepancy between each observed y_i and its fitted mu_i individually, it measures the discrepancy between the *empirical distribution* of Y and the *fitted model distribution* via a kernel function:

```
MMD²(P, Q) = E[K(Y, Y')] - 2·E[K(Y, Ŷ)] + E[K(Ŷ, Ŷ')]
```

The kernel K is Gaussian in Y-space: `K(y1, y2) = exp(-(y1-y2)²/2h²)`. The bandwidth h is set by the median heuristic on y. A large claim at £80,000 in a £8,000-mean book is a long way from the typical sample. The Gaussian kernel evaluates to near zero for that pair — the observation contributes almost nothing to the gradient. The estimator is effectively capping large deviations geometrically, without any user-specified threshold.

This is the key theoretical result in Kang & Kang: MMD is a proper scoring rule, so it identifies the true distribution at zero contamination; and its kernel structure makes it bounded under contamination, so large deviations cannot move the estimate arbitrarily far. You get consistency under clean data and robustness under contamination.

---

## L1 regularisation and feature selection

`RobustMMDGLM` adds an L1 penalty on the coefficient vector:

```
min_theta  MMD²(empirical Y, model(theta)) + lambda * ||theta||_1
```

The L1 term drives small coefficients to zero, giving automatic feature selection. For severity modelling this is practically useful: enrichment data suppliers offer dozens of postcode-level features (deprivation indices, drive time to hospital, property density), and many of them are noise at the individual claim level. A standard Gamma GLM with that many covariates requires stepwise selection or regularisation on top; `RobustMMDGLM` does both robustness and selection simultaneously.

Setting `lambda_="cv"` triggers five-fold cross-validation over a log-spaced lambda grid, selecting the value that minimises held-out MMD loss:

```python
from insurance_glm_tools.robust import RobustMMDGLM

model = RobustMMDGLM(family="gamma", lambda_="cv", cv_folds=5, random_state=42)
model.fit(X_train, y_train, exposure=vehicle_years_train,
          feature_names=feature_cols)
```

Or specify lambda directly if you know what you want:

```python
model = RobustMMDGLM(family="gamma", lambda_=0.05, random_state=42)
model.fit(X_train, y_train, exposure=vehicle_years_train,
          feature_names=feature_cols)
```

---

## The API

After fitting, three methods cover most use cases:

```python
# Relativities table — multiplicative factors for Gamma/Poisson
rels = model.relativities()
#   feature         coefficient  relativity  nonzero
#   vehicle_age     -0.142       0.868       True
#   ncd_years       -0.089       0.915       True
#   deprivation_idx  0.000       1.000       False   <- zeroed by L1
#   ...

# Selected features only (non-zero coefficients)
active = model.selected_features()
# ['vehicle_age', 'ncd_years', 'postcode_band', ...]

# Predictions at new data
mu_hat = model.predict(X_test, exposure=vehicle_years_test)
```

The CV path is available if you want to inspect how MMD loss and sparsity vary across the lambda grid:

```python
cv_path = model.cv_path()
# DataFrame with: lambda, mean_mmd_loss, std_mmd_loss, n_nonzero
```

---

## Confidence intervals via bootstrap

Because the optimisation is ADMM-based rather than likelihood-based, there is no Hessian, and Fisher-information standard errors do not apply. Confidence intervals require bootstrap. The API is explicit about this:

```python
ci_df = model.bootstrap_relativities_from_data(
    X_train, y_train,
    exposure=vehicle_years_train,
    n_boot=200,
    ci=0.95,
)
#   feature      coefficient  relativity  lower   upper
#   vehicle_age  -0.142       0.868       0.831   0.906
#   ...
```

Each bootstrap replicate refits ADMM with the selected lambda. Two hundred replicates is slow on a large dataset — allow several minutes for a 100k-row severity model. If this is a bottleneck, reduce to 100 replicates for development and reserve 200 for production model documentation.

---

## How it compares with a standard Gamma GLM

We are not publishing a formal benchmark here because the answer depends entirely on your contamination level — clean data with a well-specified Gamma family will see MLE and RobustMMDGLM converge to similar estimates. The point of the robust estimator is what happens when data is not clean.

Conceptually, the comparison is this:

| | Standard Gamma GLM | RobustMMDGLM |
|---|---|---|
| Objective | Gamma deviance (MLE) | MMD loss (min over theta) |
| Influence of large losses | Unbounded — pulls estimates | Bounded — kernel saturates |
| Feature selection | Manual (stepwise, BIC, or Ridge/Lasso on top) | Built-in L1 |
| Standard errors | Fisher information (Hessian-based) | Bootstrap only |
| Optimisation | IRLS (fast) | ADMM + AdaGrad (slower) |
| Output | Multiplicative relativities | Multiplicative relativities |
| NestedGLM compatible | Yes | Yes |

The output is deliberately identical in format: `relativities()` returns the same DataFrame structure as the nested GLM pipeline. This means `RobustMMDGLM` can slot into an existing severity stage without restructuring downstream code.

---

## The ADMM optimiser

For those who want to understand what is happening under the hood: the L1-penalised MMD objective is not differentiable everywhere, so gradient descent is not directly applicable. The implementation uses ADMM (Alternating Direction Method of Multipliers), which splits the problem into a theta-step (minimise the smooth MMD term plus a quadratic proximity term) and a z-step (proximal operator for L1, which is soft-thresholding).

The theta-step runs AdaGrad with a per-coordinate adaptive learning rate — this matters because the MMD gradient's magnitude varies significantly across features depending on scale and sparsity. The default `adagrad_lr=0.1` and `rho=1.0` work reliably for standardised design matrices. If you are passing raw floats with wildly different scales, standardise first or increase `rho`.

For the Gamma family, the expected kernel integral `E[K_y(y, Ŷ)]` involves integrating a Gaussian kernel against a Gamma density. This is computed via 32-point Gauss-Laguerre quadrature (`n_quadrature=32`), which is accurate for practical bandwidth values. Kang & Kang derive the analytic form; the implementation follows their Appendix B.

---

## Practical guidance

A few things we learnt fitting this on held-out datasets:

**Standardise X.** The ADMM convergence is sensitive to feature scale. Run `sklearn.preprocessing.StandardScaler` before passing X. The relativities are in the standardised coefficient space, so interpret them as effects-per-standard-deviation unless you transform back.

**CV with `lambda_="cv"` is the right default.** The auto-constructed lambda grid spans five orders of magnitude centred on a data-dependent scale estimate. For most insurance severity datasets (50k–500k rows, 20–80 features) this takes three to eight minutes on a single core.

**For Poisson frequency models**, `RobustMMDGLM(family="poisson")` works the same way. The frequency application is less dramatic because claim counts are already bounded, but if your data has coding errors producing impossible count spikes (a row coded with 99 claims rather than 0), the MMD loss will downweight them rather than distorting your NCD or vehicle age factors.

---

## Installation

```
pip install insurance-glm-tools>=0.2.0
```

The `robust` subpackage has no additional mandatory dependencies beyond numpy, scipy, and pandas. sklearn is used only for warm-start initialisation and can be excluded by setting `init="zeros"`.

**Paper:** [arXiv:2602.21132](https://arxiv.org/abs/2602.21132) — Kang & Kang (2026) | **Library:** [insurance-glm-tools on PyPI](https://pypi.org/project/insurance-glm-tools/) | **GitHub:** [insurance-glm-tools](https://github.com/insurance-glm-tools/insurance-glm-tools)
