---
layout: post
title: "When Your Distributional Regression Has Too Many Parameters"
date: 2026-04-04
categories: [techniques, insurance-distributional]
tags: [distributional-regression, gamlss, normalizing-flows, regularization, identifiability, severity, elastic-net, lasso, arXiv-2603.25919, pricing]
description: "Regression by composition — the framework that generalises GAMLSS and transformation models — suffers from a subtle non-identifiability when you stack multiple flows. Kadhem (2026) derives flow-specific L1 and elastic-net penalties that restore uniqueness, with oracle properties and asymptotic guarantees. Here is what that means for heavy-tailed severity modelling."
math: true
author: Burning Cost
---

There is a failure mode in distributional regression that does not announce itself loudly. Your model converges, your log-likelihood is high, your diagnostics look fine. But your coefficients are large and opposite in sign across distribution parameters. You add a variable and an existing coefficient flips sign. You tighten convergence tolerances and the estimates drift rather than settle. The predictions are stable. The parameters are fiction.

This is non-identifiability in multi-flow distributional models, and it is more common than the literature acknowledges. A paper from Safaa K. Kadhem (arXiv:2603.25919, March 2026) diagnoses the problem precisely and provides a principled fix: flow-specific penalisation using L1 and elastic-net terms, with theoretical guarantees that the fix actually works.

---

## Regression by Composition

The regression-by-composition framework is worth understanding because it unifies a large part of what pricing actuaries already do.

The idea is to build a conditional distribution $F(y \mid x)$ by applying $K$ successive transformations — normalising flows — to a reference distribution (typically standard normal):

$$Y \mid x \sim (T_K \circ T_{K-1} \circ \cdots \circ T_1)(Z), \quad Z \sim N(0,1)$$

where each flow $T_k$ has parameters that depend on covariates through $\beta_k^\top x$.

When $K = 1$, this is a transformation model — a generalised linear model for a transformation of $Y$. When $K = 2$ with location and scale flows, you recover GAMLSS: the two flows separately model the location and dispersion parameters. When $K \geq 2$ with more flexible flow architectures, you get a fully nonparametric conditional distribution estimator. The same framework.

The appeal for severity modelling is direct. A single flow constrains you to a fixed distributional family (lognormal, gamma, Pareto). Two flows let you separately model how covariates affect the body of the distribution and how they affect the tail. Three flows could add a shape parameter that controls the transition between body and tail — useful for claims that are lognormal in the middle and Pareto in the extreme. Each incremental flow adds flexibility at the cost of additional parameters.

The cost, it turns out, is heavier than it first appears.

---

## The Non-Identifiability Problem

When $K \geq 2$, the composition $T_K \circ \cdots \circ T_1$ can often be reproduced by multiple distinct parameter configurations $(\beta_1, \ldots, \beta_K)$. Different flows commute, or partially commute, in ways that create a flat ridge in the likelihood surface. You can slide parameters along that ridge — increasing $\beta_1$ and decreasing $\beta_2$ by compensating amounts — without changing the implied distribution at all.

This is what GAMLSS practitioners have observed empirically for years without a clean theoretical description: coefficient instability when fitting $\mu$, $\sigma$, and $\nu$ simultaneously on the same covariates. You might recognise it as the optimiser seeming to "wander" or as parameter estimates that vary implausibly across bootstrapped resamples. The predictions are stable because the flat ridge preserves the predicted distribution; the parameters are not because you can move freely along it.

The standard practical responses — tighter priors, manual coefficient constraints, fitting $\mu$ first then holding it fixed while fitting $\sigma$ — are workarounds, not solutions. They impose identifiability by external constraint rather than by model design, and they require judgement calls that are hard to defend or reproduce.

Kadhem's Theorem 3.1 states the condition precisely: the unregularised objective function of a $K$-flow composition model does not, in general, have a unique minimiser when $K \geq 2$ and the flows share covariate structure. The flat ridge is not a pathological edge case. It is a structural feature of the model class.

---

## The Fix: Flow-Specific Penalisation

The proposed solution is to add flow-specific L1 (Lasso) or elastic-net penalties, with a separate $\lambda_k$ per flow:

$$\hat{\beta} = \underset{\beta}{\arg\min} \left\{ -\ell(\beta; \text{data}) + \sum_{k=1}^{K} \lambda_k \|\beta_k\|_1 \right\}$$

The key word is *flow-specific*. A single global penalty $\lambda \|\beta\|_1$ across all flows does not break the symmetry — it just scales all parameters down uniformly and the ridge persists. Different $\lambda_k$ per flow creates an asymmetry that forces the optimiser to commit: if two parameter configurations produce the same likelihood, the one with smaller per-flow penalties is preferred, and the penalties weight each flow differently. The ridge disappears.

Theorem 3.1 in the paper confirms this: with distinct, non-zero $\lambda_k$ per flow, the penalised objective has a unique minimiser. Identifiability is restored.

This is satisfying because it is not an ad hoc fix. The penalties are doing two things simultaneously: regularising the model (shrinking irrelevant coefficients towards zero, which is useful in its own right for high-dimensional pricing models) and breaking the symmetry that caused the identifiability failure.

---

## Theoretical Guarantees

The paper proves three results that matter for practical use.

**Consistency.** The penalised estimator $\hat{\beta}$ converges to the true $\beta^*$ at the $n^{-1/2}$ rate for components where the true coefficient is non-zero. You lose nothing asymptotically relative to maximum likelihood when you penalise — the rate is the same.

**Oracle property (adaptive Lasso).** The adaptive Lasso variant, which uses weights $w_{k,j} = |\hat{\beta}_{k,j}^{(0)}|^{-\gamma}$ derived from an initial estimate, selects the correct covariate support per flow with probability converging to one. In practice: the model learns which covariates genuinely affect each distributional parameter and which do not, per flow, with asymptotic correctness. This is the property you want when fitting, say, a two-flow severity model and asking "does driving age affect the tail shape or only the body location?"

**Algorithm.** Proximal gradient descent: gradient steps on the smooth log-likelihood component, followed by the proximal operator (soft-thresholding) on the L1 terms. The gradient computation chains through the composition using Jacobians: $\nabla_{\beta_k} \ell$ involves flows $k$ through $K$, not all flows. Cost per iteration is $O(K)$, linear in the number of flows. This matters because the naive approach of numerically differentiating through the full composition would be $O(K^2)$.

---

## The NHANES Demonstration

The paper's empirical section uses the NHANES dataset to model the distribution of blood lead concentration as a function of covariates including asthma diagnosis — a health outcome study, not insurance. But the demonstration of the identifiability problem is clear enough to be directly instructive.

Unregularised two-flow model: large coefficients with opposite signs on the same covariates across flows. The standard errors are inflated. The Labbe plots from the unregularised model are difficult to interpret because the coefficients are physically implausible. Adding or removing a single covariate causes substantial changes to existing estimates — the hallmark of a poorly identified model.

Regularised model (flow-specific elastic-net): coefficients are an order of magnitude smaller, stable across specifications, and directionally interpretable. The Labbe plots show a clear protective effect of reducing lead exposure — a result that was obscured rather than revealed by the unregularised model's noise. Automatic variable selection per flow, with some covariates active in flow 1 (location) but not flow 2 (scale), and vice versa.

The takeaway is not that penalisation gives you "better" results in some abstract sense. It is that the unregularised results were unreliable, and the regularised results are recoverable from the data. The paper is not trading bias for variance; it is trading the appearance of a fit for an actual fit.

---

## What This Means for Insurance Severity Modelling

Our [insurance-distributional](https://github.com/burning-cost/insurance-distributional) library implements GAMLSS-style distributional gradient boosting for pricing — modelling the full conditional distribution of losses rather than just $E[Y \mid x]$. The library currently uses single-parameter-per-distribution architectures (Tweedie, Gamma, ZIP, Negative Binomial), which sidesteps the multi-flow identifiability problem by construction.

The problem becomes live, however, when you want to go further: fitting models where separate covariate structures govern different aspects of the distribution. For heavy commercial lines severity — EL/PL bodily injury, large commercial property — this is the natural model architecture. The body of the distribution responds to risk characteristics like occupation, construction type, and sum insured. The tail responds to different things: accumulation potential, contractual caps, reinsurance structure. Fitting a single distributional family with the same covariates driving all parameters is a simplification that often works in personal lines but strains credibility in commercial.

A two-flow architecture directly addresses this:

- **Flow 1** (location/body): severity drivers that are well-understood and high-signal — vehicle type, NCD, young driver indicator.
- **Flow 2** (tail shape): drivers that specifically affect large claim potential — annual mileage, telematics percentile scores, claims history pattern.

Without flow-specific regularisation, fitting this model produces the non-identifiability problem Kadhem describes. With it, you get per-flow variable selection and stable estimates.

The specific gap the library has is that there is currently no exposure offset handling or excess-zero structure in the paper's formulation — both of which matter for earned-premium-weighted severity models. Those are engineering additions, not theoretical barriers. The core framework and the penalisation approach are directly portable.

For pricing actuaries already using GAMLSS (directly or via packages like `gamlss` in R), the diagnostic implication is immediate: if your coefficient estimates are unstable across specifications, or if you see large opposite-sign coefficients on the same variable across $\mu$, $\sigma$, and $\nu$ equations, that is the non-identifiability problem. Adding L1 penalties per distributional parameter equation — not a global penalty across all of them — is the theoretically grounded fix. There is also a model governance argument: Solvency II Article 121 requires that internal model parameters are explainable and justifiable. A non-identifiable multi-flow model cannot provide that justification. Penalised estimation is not just statistically cleaner; it is the version you can actually defend to a validation function.

---

## Caveats

This is a single-author preprint submitted March 2026 and not yet peer reviewed. The empirical results come from one health outcomes application; there is no insurance dataset in the paper.

**Computational cost unbenchmarked at scale.** Whether the proximal gradient algorithm converges in acceptable time on 200,000+ claims with $K = 3$ flows is not established. The $O(K)$ per-iteration cost is encouraging; the number of outer iterations required on high-dimensional insurance data is unknown.

**Hyperparameter selection.** Flow-specific $\lambda_k$ require tuning, presumably by cross-validation. With $K$ flows the grid expands multiplicatively — three flows and five candidate values each gives 125 combinations before you have added an elastic-net mixing parameter. The paper does not address this; it is a practical barrier to deployment that needs a worked solution.

**Oracle property conditions.** The oracle result requires the true data-generating process to be within the composition class. For insurance severity, we are always approximating. The oracle result is guidance about asymptotic behaviour rather than a finite-sample guarantee.

---

## Reference

Kadhem, S.K. (2026). Regularized Regression by Composition for Distributional Models. arXiv:2603.25919. Submitted March 26, 2026.

---

Related on Burning Cost:

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — insurance-distributional, distributional GBM for insurance pricing
- [Conformal Prediction for Insurance Python: A Frequency-Severity Tutorial](/2026/04/04/conformal-prediction-insurance-python/) — distribution-free alternative to parametric severity intervals, for when the distributional assumptions are uncertain
- [Validating GAMLSS Sigma Models: What the Diagnostics Miss](/2026/03/08/validating-gamlss-sigma-models/) — coefficient instability in GAMLSS and practical diagnostics
