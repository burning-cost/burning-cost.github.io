---
layout: post
title: "When Distributional Regression Gets Too Flexible: Identifiability Problems in Multi-Flow Models and How Penalisation Fixes Them"
date: 2026-04-04
categories: [modelling, insurance-distributional]
tags: [distributional-regression, normalizing-flows, GAMLSS, identifiability, regularisation, severity-modelling, arXiv-2603.25919]
description: "Kadhem (arXiv:2603.25919) identifies a fundamental non-identifiability problem in regression by composition and fixes it with flow-specific L1 and elastic-net penalties. For pricing actuaries building multi-stage transformation models for heavy-tailed severity, this is the theoretical underpinning you need before composing flows in production."
author: Burning Cost
math: true
---

GAMLSS is already pushing the limits of what most pricing teams can govern. Location, scale, and shape each modelled as functions of covariates — powerful, interpretable, just about governable. You know what each component does.

Regression by composition goes further. Instead of parameterising a fixed distributional family, it constructs the conditional distribution by applying a sequence of transformations — normalizing flows — to a reference distribution such as the standard normal. Each flow acts on the previous output. The result can approximate any distribution, not just the families GAMLSS supports. For heavy-tailed severity where features influence the body and the tail differently, this flexibility is genuinely attractive.

The problem Kadhem (arXiv:2603.25919, March 2026) identifies is that composing multiple flows creates non-identifiability: different parameter configurations produce identical distributions when the flows commute. The likelihood surface is flat in those directions. Estimation is unstable. Coefficients are uninterpretable. You have built a model that fits the data but cannot be recovered uniquely from it.

The fix is structured penalisation. Flow-specific L1 and elastic-net penalties break the symmetry between flows and restore a unique minimiser. Kadhem proves this formally and provides a proximal gradient algorithm to solve the penalised problem efficiently.

---

## The non-identifiability problem in concrete terms

Regression by composition applies $K$ successive transformations $g_1, \ldots, g_K$, each parameterised by a separate linear predictor with its own coefficient vector $\beta_k$. The response distribution is:

$$Y \mid x \sim (g_K(\cdot; \beta_K^\top x) \circ \cdots \circ g_1(\cdot; \beta_1^\top x)) \# \mathbb{P}_0$$

where $\mathbb{P}_0$ is the reference distribution and $\#$ denotes the pushforward. With $K = 1$ this is a transformation model. With $K = 2$ and location-scale flows, you recover GAMLSS. With $K \geq 2$ and flexible flows, you can represent essentially anything.

The identifiability failure occurs when flows commute: if $g_1(\cdot; a) \circ g_2(\cdot; b) = g_2(\cdot; b') \circ g_1(\cdot; a')$ for some $(a', b') \neq (a, b)$, then both parameterisations produce identical distributions. Under location-scale flows this happens whenever one flow's shift can be absorbed into the other. The likelihood is indifferent between them.

In practice this manifests as coefficient estimates that are large and opposite in sign across flows, cancelling through the composition. The model's predictions look fine. Its parameters are fiction. Any attempt to ask which covariates drive the tail shape versus the mean will get a meaningless answer.

---

## What penalisation does

Kadhem proposes flow-specific penalties:

$$\hat{\beta} = \arg\min_\beta \left\{ -\ell(\beta; \text{data}) + \sum_{k=1}^K \lambda_k \| \beta_k \|_1 \right\}$$

with separate regularisation strengths $\lambda_k$ per flow. Different flows play different roles; they should face different penalties. The three formal results are:

**Identifiability under penalisation.** The penalised objective has a unique minimiser even when the unpenalised likelihood has a flat ridge. The asymmetry between flow penalties breaks the symmetry between flows.

**Oracle property for adaptive Lasso.** If the true model is sparse, the adaptive Lasso variant selects the correct covariate support within each flow with probability approaching 1. Standard result, extended to the composition setting.

**Asymptotic consistency** at the usual $n^{-1/2}$ rate for non-zero components.

The proximal gradient algorithm alternates gradient steps on the smooth log-likelihood with proximal operators on the L1 terms. Jacobians propagate sequentially through the composition; the gradient for flow $k$ only requires the chain through flows $k$ to $K$, so cost per iteration scales linearly in $K$.

The NHANES application on asthma and blood lead concentration makes the practical point cleanly: the unregularised model produces implausible coefficients (optimiser wandering along the flat likelihood ridge); the penalised model produces stable estimates with automatic variable selection.

---

## Why this matters for insurance severity

Our [insurance-distributional](https://github.com/burning-cost/insurance-distributional) library implements distributional gradient boosting with GAMLSS-style parameterisation — Tweedie, Gamma, ZIP, and Negative Binomial families, each fit jointly across mean and dispersion. This covers the vast majority of UK personal lines severity work.

The Kadhem framework is the logical next step for two situations where parametric families are genuinely insufficient:

**Large commercial and liability severity.** Property damage liability and employers' liability claims are heavy-tailed, skewed, and multi-modal in ways no single parametric family captures. A two-flow model with separate covariate sets for body and tail dynamics is the natural architecture. Without regularisation, that model is non-identifiable.

**Telematics severity.** Trip-level features (hard braking, cornering speed) that predict a large claim may have entirely different effects at the small-claim end of the distribution. A flow for each segment, with per-flow feature selection, is the right structure.

In both cases, Kadhem's result means you can write down a principled regularisation strategy with formal backing. The alternative — applying ad hoc shrinkage and hoping the estimates stabilise — will not survive model governance scrutiny: Solvency II Article 121 requires coefficient-level justification for significant internal model components.

---

## Limitations

Single-author preprint, not yet peer reviewed. The more material gaps are:

**No insurance application.** The NHANES example works. Claim severity data has exposure offsets, excess zeros, and catastrophic right tails that will require further adaptation of the composition framework.

**Computational cost unbenchmarked at scale.** Whether the proximal gradient algorithm converges in acceptable time on 200,000 claims, 50 covariates, and $K = 3$ flows is not established. GAMLSS already tests standard optimisers at this scale. Multi-flow is harder.

**Hyperparameter selection.** Flow-specific $\lambda_k$ require tuning, presumably by cross-validation. With $K$ flows the grid expands multiplicatively. The paper does not address this.

---

## Our assessment

The identifiability problem is real. Any practitioner who has tried to compose multiple transformation stages and found the coefficients erratic will recognise it immediately. The penalisation fix is the right response, and the oracle property result means you get sparsity — automatic selection of which covariates influence each flow — as a by-product.

We will track this for a potential extension to `insurance-distributional`. The prerequisite is establishing computational feasibility on insurance-scale datasets before designing an API around it.

For now: if your severity model uses more than one transformation stage, read section 3 of this paper before your next governance committee meeting.

---

## Reference

Kadhem, S.K. (2026). Regularized Regression by Composition for Distributional Models. arXiv:2603.25919. Submitted March 26, 2026.

Related on Burning Cost:

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — insurance-distributional, distributional GBM for insurance pricing
