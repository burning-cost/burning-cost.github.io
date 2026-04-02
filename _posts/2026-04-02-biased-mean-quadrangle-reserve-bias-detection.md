---
layout: post
title: "Biased Mean Regression: A Principled Test for Systematic Reserve Errors"
date: 2026-04-02
categories: [research]
tags: [reserving, ibnr, quantile-regression, biased-mean, superexpectation, linear-programming, risk-quadrangle, actuarial, bias-detection, arXiv-2603-26901, insurance-quantile]
description: "Malandii & Uryasev's Biased Mean Quadrangle (arXiv:2603.26901) provides a linear-programming-based method for estimating E[Y]+x — a biased mean offset. For reserving actuaries, it offers a principled way to detect and quantify systematic over- or under-reserving without relying on ad-hoc diagnostic charts."
math: true
author: burning-cost
permalink: /2026/04/02/biased-mean-quadrangle-reserve-bias-detection/
---

Every reserving actuary has looked at an ultimates vs. actuals chart and thought: is this systematic? Claims on this segment are consistently settling 15% above our central estimate. Is that model bias or noise? Is it stable enough to adjust for?

The standard response is either a residual plot (qualitative) or an over/under triangle (directional but not rigorous). Malandii and Uryasev's *Biased Mean Quadrangle and Applications* (arXiv:2603.26901, March 2026) provides something more formal: an estimator that explicitly targets $\mathbb{E}[Y] + x$ — the mean plus a bias offset $x$ — and solves for both simultaneously via linear programming. For reserving teams, this is a surprisingly direct fit to the diagnostic question.

---

## The problem biased mean regression is solving

Standard regression minimises squared error, which targets $\mathbb{E}[Y \mid X]$. Quantile regression targets a specific percentile $\tau$ of the conditional distribution, minimising the asymmetric pinball loss. Biased mean regression sits between them: it estimates $\mathbb{E}[Y] + x$ where $x$ is a parameter the model determines endogenously. The "bias" in the name is not a bug — it is the quantity you are trying to measure.

The formal setup: the biased mean statistic with offset $x$ is defined as the value that minimises superexpectation error over the sample. Superexpectation is a risk measure in Uryasev's Risk Quadrangle framework, which organises four linked stochastic functionals — error, regret, risk, deviation — into a consistent algebraic structure. What matters practically is that minimising superexpectation error produces an estimator for $\mathbb{E}[Y] + x$, not just $\mathbb{E}[Y]$.

The estimator reduces to a linear programme. Given a training sample $\{y_i\}$, the optimal offset $x^*$ and predictions are recovered by solving:

$$\min_{x, \xi_i \geq 0} \; x + \frac{1}{n} \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad \xi_i \geq y_i - \hat{y}_i - x \; \forall i$$

This is the same structure as quantile regression, but the interpretation differs. In quantile regression at level $\tau$, the asymmetric weights $\tau$ and $1-\tau$ on positive and negative residuals direct the estimator towards a specific percentile. In biased mean regression, the constraint formulation recovers the mean plus whatever systematic offset $x^*$ best fits the data. The paper shows this is equivalent to a particular parameterisation of quantile regression, but the estimation target is different.

---

## Why this is a natural tool for reserve bias detection

Reserving bias testing usually works backwards: fit a model, form residuals, look for patterns. The biased mean approach inverts this. You run the estimator on your historical ultimates vs. your model estimates, and it returns $x^*$ directly — the systematic component of your error.

Consider the setup: you have $n$ accident-year/development-period cells where you made a reserve estimate $\hat{R}_i$ and observed the actual settlement $Y_i$. You want to know if the shortfalls $Y_i - \hat{R}_i$ cluster on one side of zero. A simple mean of residuals does this, but:

1. It gives equal weight to cells regardless of their uncertainty. Large development periods have noisier residuals.
2. It conflates the mean bias with the noise in any one estimate.
3. It does not produce a clean confidence interval without distributional assumptions.

The biased mean LP is more robust because the superexpectation error function down-weights large individual deviations. More importantly, the LP structure lets you add feature covariates to test whether bias is uniform or segment-specific. If motor BI and motor OD share the same $x^*$, the bias is likely a systematic loading error. If they diverge, the problem is in your segment-level frequency or severity assumptions.

The practical workflow: fit the LP with your historical training data (reserve estimate, settlement outcome), recover $x^*$, then test stability across cohorts. An $x^*$ that is positive and stable over five or more accident years is evidence of systematic under-reserving. An $x^*$ that oscillates around zero is noise.

---

## Connection to quantile regression and the Risk Quadrangle

The paper proves that biased mean regression with offset $x$ is equivalent to quantile regression at the level $\tau^* = F(\bar{y} - x)$, where $F$ is the empirical distribution and $\bar{y}$ is the sample mean. This connection matters because it means you can use quantile regression infrastructure — which is well-established and has good LP solvers — to perform biased mean estimation. You are not adopting a new toolchain; you are reinterpreting results from existing machinery.

The Risk Quadrangle framing is more than theoretical tidiness. The quadrangle structure guarantees that deviation, risk, regret, and error are internally consistent — changes to one imply specific changes to the others. If you are using CVaR (Expected Shortfall) for capital purposes and quantile regression for reserve diagnostics, they are speaking the same mathematical language. The biased mean sits in the same family.

The LP reduction also matters operationally. Reserve bias tests on large books can involve millions of claim records. A well-implemented LP scales linearly in sample size once the constraint matrix is sparse. This is materially faster than methods requiring iterative reweighting or bootstrap resampling.

---

## What this means for pricing teams

Pricing actuaries who set IBNR loadings have a direct stake in this. The standard approach is to take the reserving team's central estimate and apply a loading — typically 5–15% for volatile lines — chosen judgementally or based on historical volatility. But volatility-based loadings do not correct for systematic bias; they compensate for it with extra capital. If your central reserve is biased downward by $x^*$ percentage points on average, the principled response is to reduce the IBNR loading by $x^*$ and fix the model, not to pad the loading further.

The biased mean LP gives you a number to argue with in a review: *our reserve estimates on motor BI have a measured bias of +8% against settled claims over the last six accident years; we are proposing to adjust the central estimate rather than the loading*. That is a different conversation from a residual chart.

For pricing teams building in-house severity models, reserve bias propagates directly into pricing. If the IBNR on your exposure data is systematically understated, your ultimate loss ratios used for credibility weighting are wrong, and your pricing will be optimistic. The feedback loop from reserving bias to pricing inadequacy is well-known but rarely measured formally. Biased mean regression is one way to put a number on it.

**Using insurance-quantile.** Our [insurance-quantile](https://github.com/burning-cost/insurance-quantile) library implements CatBoost-based quantile and expectile regression. Since the biased mean LP reduces to quantile regression at a specific $\tau^*$, you can estimate $x^*$ directly: fit quantile GBMs at a range of $\tau$ values on your (reserve estimate, settlement) pairs and scan for the $\tau^*$ where $\hat{Q}_{\tau^*}$ equals your sample mean residual — that is the biased mean offset $x^*$ implied by the data. It is not the LP formulation, but it produces the same estimator on tabular claims data and has the advantage of handling non-linear feature interactions that a plain LP cannot.

---

## Limitations

This is a March 2026 preprint. The paper focuses on the theoretical framework and synthetic numerical experiments — there is no large-scale actuarial validation on real reserving data. The LP formulation is elegant, but implementing it properly on heterogeneous claim data requires decisions about how to handle censoring (open claims), aggregation level, and time-varying reporting patterns that the paper does not address. These are not fatal objections, but they mean the work from paper to production is non-trivial.

The connection to existing quantile regression infrastructure is genuinely useful, but it also means the technique is not doing anything fundamentally new on structured tabular data. Where it adds value is the explicit framing of bias as an estimable quantity — which changes how you use the diagnostic, even if the underlying computation is familiar.

We think this is worth tracking and building into a proper reserving diagnostic tool. The biased mean framing is cleaner than ad-hoc residual analysis, the LP is fast, and the Risk Quadrangle connection means it integrates naturally with the ES-based capital frameworks most UK firms are already running.

---

*arXiv:2603.26901 — Malandii & Uryasev, "Biased Mean Quadrangle and Applications", March 2026.*
