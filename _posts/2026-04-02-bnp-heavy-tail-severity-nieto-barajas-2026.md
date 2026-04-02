---
layout: post
title: "Your Threshold Is a Choice, Not a Fact: BNP Heavy-Tail Severity Modelling"
date: 2026-04-02
categories: [techniques, pricing, research]
tags: [severity, EVT, GPD, threshold-selection, BNP, bayesian-nonparametric, SGG, scaled-generalised-gaussian, heavy-tail, MCMC, spliced-composites, EQRN, insurance-severity, arXiv-2602.07228, Nieto-Barajas, tail-index, python, UK-motor]
description: "Nieto-Barajas (arXiv:2602.07228, 2026) proposes a Bayesian nonparametric mixture of Scaled Generalised Gaussian distributions that eliminates threshold selection entirely. The mathematics is clean. The production case is weak. Here is what it gets right, where it falls short, and what we use instead."
math: true
author: burning-cost
---

Pick a threshold. Any threshold. That is the instruction embedded in every EVT severity workflow, and it is the instruction that actuaries quietly make up an answer for, every time.

The mean excess plot slopes upward after £150k. The stability plot shows the shape parameter settling above £200k. You pick £175k because it splits the difference and because you have a capping exercise to finish before the pricing meeting. The GPD fit looks reasonable. You write "threshold selected at £175k based on MRL analysis" in the methodology note and move on.

But the number was a judgement call dressed up as analysis. And if you had picked £130k or £220k, your 99.5th percentile estimate would have shifted by fifteen to twenty percent. That is not a rounding error. In a commercial property XL layer or a motor BI severity load, it is a pricing decision worth arguing about.

Nieto-Barajas ([arXiv:2602.07228](https://arxiv.org/abs/2602.07228), January 2026) asks whether you need to make that call at all.

---

## The threshold problem is structural, not cosmetic

The Generalised Pareto Distribution is the canonical tool for tail modelling because of a theorem: exceedances above a sufficiently high threshold converge to a GPD, regardless of the underlying distribution. The theorem is solid. The word "sufficiently" is where all the trouble lives.

A threshold that is too low includes body-of-distribution claims in a fit designed for tail behaviour. You get a biased shape parameter and a tail that is thinner than the data supports. A threshold that is too high leaves you with fifty or eighty observations above the threshold, a shape parameter with a standard error of 0.3, and extrapolations to the 99.9th percentile that are driven more by sampling noise than by genuine tail structure.

Spliced composite models — LognormalGPD, GammaGPD — replace the threshold with a splice point, which is an equivalent choice dressed in different notation. You still have to decide where the bulk model hands off to the tail model. If you fit the splice point by maximum likelihood you get an objective answer, but it is an answer that depends heavily on the assumed bulk distribution. Change the bulk from Lognormal to Gamma and the estimated splice point shifts. The tail fit shifts with it.

The diagnostic tools — mean residual life plots, Hill estimator stability, parameter stability plots — reduce the range of plausible thresholds but do not eliminate it. Expert judgement fills the remaining gap, and expert judgement can move a reinsurance price by more than most people are comfortable admitting.

---

## What Nieto-Barajas proposes

The paper builds a Bayesian nonparametric (BNP) mixture model using the Scaled Generalised Gaussian (SGG) distribution as the mixture kernel.

The SGG is a flexible family parameterised by location $\mu$, scale $\sigma$, and a tail parameter $\alpha$. When $\alpha > 1$ the distribution is light-tailed. When $\alpha = 1$ it is Laplace. When $\alpha < 1$ the tails are heavier than exponential. When $\alpha \to 0$ the distribution becomes increasingly heavy-tailed. The innovation is using an infinite mixture of these: a Dirichlet Process prior over an infinite collection of SGG components, fitted by MCMC.

Because the model is nonparametric, it adapts to whatever tail structure the data actually contains. No threshold is required. No splice point is required. The posterior distribution over the tail parameter $\alpha$ does the work that the threshold diagnostic plots were supposed to do. Specifically, $P(\alpha < 1 | \text{data})$ gives the posterior mass assigned to heavy-tail behaviour — a single number that summarises how strongly the data supports a heavy tail, with uncertainty attached.

The paper demonstrates this on an auto insurance severity dataset of 1,383 losses. The fitted model recovers expected tail behaviour, and the posterior decomposition clearly separates the bulk regime (large $\alpha$, sub-exponential tail) from the tail regime (small $\alpha$, heavier than exponential). No threshold was selected. The posterior selected it, implicitly and probabilistically.

This is a genuine conceptual advance. The threshold you used to pick manually is now a derived quantity, with a full posterior distribution, rather than a fixed input. If the data do not support a single clear threshold, the posterior reflects that uncertainty. If the tail is actually a mixture of two regimes — a moderate heavy-tail and an extreme heavy-tail, as sometimes occurs in commercial liability — the model can represent both simultaneously.

---

## What the paper gets right

The core insight — that threshold selection is a modelling assumption that should carry uncertainty rather than a fixed parameter — is correct and underappreciated.

Most actuaries who run sensitivity analyses on their EVT fits treat the threshold as one of several inputs to vary. They fit at £150k, £175k, and £200k, report the range of shape parameter estimates, and document the spread as an uncertainty measure. This is the right instinct. What the BNP approach does is make that uncertainty explicit in the model rather than external to it.

The posterior $P(\alpha < 1 | \text{data})$ is a particularly useful construct. Rather than reporting "the tail is heavy," you can report "the data assign 73% posterior probability to heavy-tail behaviour." That is a quantified statement about evidence strength, not a binary assertion. For pricing governance and model documentation, that distinction matters.

The model also handles multimodal tail structure naturally. Spliced composites assume a single transition point between bulk and tail. The BNP mixture can place mass on several different tail regimes simultaneously. In datasets where the claim distribution reflects different sub-perils — minor incidents, moderate structural losses, and rare catastrophic losses, all mixed together — a single-regime tail model is always an approximation. The BNP mixture is a less restrictive approximation.

---

## Why we are not building it

Three problems prevent this from entering a production pricing workflow.

**No covariate conditioning.** The model is marginal. It fits the unconditional severity distribution of all claims pooled together. Insurance pricing does not want that. We want $P(X > x | \text{age, vehicle, territory, occupation})$. A two-stage approach — fit covariate-conditional expected severity with a GLM, then fit the BNP model to residuals — is theoretically possible but loses the tail decomposition benefits. The whole point of the model is that the posterior identifies tail regimes from the data. Once you have stratified by covariates or worked on residuals, you have far fewer observations per stratum and the MCMC becomes even less tractable.

**MCMC speed.** The paper reports fitting time of approximately 40 minutes on 1,383 observations, using a Fortran implementation. Python, which is the natural habitat of a production pricing tool, would be five to ten times slower without substantial engineering effort. A typical severity dataset for commercial motor or property has tens of thousands of claims. At that scale, the fitting time becomes a material constraint on model iteration cycles. For research purposes, 40 minutes is fine. For a weekly repricing cycle with model governance requirements, it is not.

**No Python implementation exists.** The paper's code is in Fortran. There is no published Python package. Building a reliable Python implementation of Dirichlet Process mixtures with SGG kernels — including proper MCMC diagnostics, convergence checks, and numerical stability for heavy-tailed data — is several weeks of work for a competent probabilistic programmer, and then needs to be maintained. We built `insurance-severity` because the maintenance burden of cobbling together ad hoc implementations was the problem we were trying to solve. Creating another unmaintained implementation is the wrong direction.

---

## What we use instead

For marginal severity tail estimation — the use case the paper directly addresses — `insurance-severity` provides spliced composite models: `LognormalGPD`, `GammaGPD`, and `WeibullGPD`. The splice point is fitted by maximum likelihood, which makes it objective given the bulk distribution assumption. It is not threshold-free, but it is transparent and fast, and the documentation makes the assumption explicit.

For the conditional tail quantile problem — where threshold-free flexibility matters most, because the effective threshold varies by risk — we have `EQRNSeverity` (Extreme Quantile Regression Network, from `insurance-severity` v0.2.2 onwards). EQRN handles covariate conditioning natively, fitting conditional tail quantiles without requiring a global threshold. The shape parameter $\xi$ and scale $\beta$ of the GPD become functions of the covariates. This addresses the core weakness of fixed-threshold EVT in a per-risk pricing context, and it does so at the cost of a neural network fit rather than MCMC — which is hours of computation not weeks of development.

If you are working at the portfolio level — aggregate severity for a reinsurance pricing exercise, say, where you genuinely want the marginal distribution rather than per-risk conditional tails — the spliced composites with threshold sensitivity analysis remain our recommendation. The SPQRx method (Majumder and Richards, [arXiv:2504.19994](https://arxiv.org/abs/2504.19994)) is a third option in this space that blends bulk and GPD continuously using covariate-conditional blending quantiles; we covered it [separately](/2026-04-01-spqrx-semi-parametric-severity-tail-without-threshold-selection/).

The BNP paper occupies the same niche as SPQRx — threshold-free marginal tail modelling — but without covariates and with a much heavier computational and implementation cost.

---

## The takeaway: sensitivity analysis is non-negotiable

Whatever method you use, threshold sensitivity is not optional.

If you are using a GPD with a fixed threshold, fit it at three or four threshold levels and document the spread in your shape parameter estimate. If the shape parameter is stable, you have evidence that your threshold choice is not dominating the result. If it is unstable, you have evidence that it is. Either outcome is useful information.

If you are using a spliced composite, vary the bulk distribution assumption. Fit LognormalGPD and GammaGPD to the same data. If the implied 99.5th percentile differs by more than ten percent, the splice architecture is influencing the tail estimate and that needs to be in your model documentation.

If you are using EQRN, vary the network architecture and the intermediate quantile levels. The flexibility that makes EQRN attractive also makes it possible to produce plausible-looking results from an architecture that is quietly interpolating rather than extrapolating.

What the Nieto-Barajas paper demonstrates — and this is its real contribution to practitioners, regardless of whether you use it — is that threshold sensitivity is not a calibration choice. It is an epistemic choice about where the tail begins. If your methodology requires you to make that choice as a fixed input, the uncertainty you are not capturing in the model is being silently transferred to the premium. That is worth knowing, even if the solution is not yet production-ready.

---

`insurance-severity` is available on PyPI:

```bash
uv add insurance-severity
```

The spliced composite classes (`LognormalGPD`, `GammaGPD`, `WeibullGPD`) and `EQRNSeverity` are documented in the [library reference](/insurance-distributional/).
