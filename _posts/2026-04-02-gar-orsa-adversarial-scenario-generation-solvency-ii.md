---
layout: post
title: "GAR for ORSA: Why Adversarial Scenario Generation Is Better Than Replaying History"
date: 2026-04-02
categories: [research]
tags: [orsa, solvency-ii, scenario-generation, adversarial, minimax, gar, var, expected-shortfall, elicitability, stress-testing, insurance-distributional, arXiv-2603-08553]
description: "Asadi & Li's Generative Adversarial Regression (arXiv:2603.08553) frames scenario generation as a minimax problem between a generator and an adversarial policy. For Solvency II ORSA teams, this is a substantive improvement over historical replay and parametric bootstrapping — and it integrates directly with insurance-distributional."
math: true
author: burning-cost
permalink: /2026/04/02/gar-orsa-adversarial-scenario-generation-solvency-ii/
---

The standard approach to ORSA stress testing in UK personal lines is one of two things: replay historical tail events (the 2010 floods, the 2017 storms, the 2020 lockdown), or run a parametric bootstrap where you assume some copula structure, fit it to your historical data, and draw scenarios from it. Both approaches have the same weakness — they are bounded by what has happened. The scenarios you produce are either observations from your dataset or samples from a distribution calibrated to your dataset. If the real tail risk is a combination of factors that has not yet co-occurred, you will not generate it.

Asadi and Li's *Generative Adversarial Regression* (arXiv:2603.08553, March 2026) takes a different approach. The question is not "what scenarios look like our historical data?" but "what scenarios would be worst for our risk positions, conditioned on the current covariate state?" That is the right question for ORSA.

---

## The minimax formulation

GAR sets up a two-player game. The generator $G$ produces risk scenarios conditioned on covariates $X$: claim counts, weather indices, macroeconomic factors, whatever is relevant to your line of business. The adversarial policy $\pi$ selects the loss function — the metric by which scenarios are evaluated. The generator wants to minimise the worst-case risk evaluation; the adversary wants to find the evaluation that reveals the largest gap between what the generator produces and the true conditional risk distribution.

Formally:

$$\min_G \max_{\pi \in \Pi} \; \mathbb{E}_\pi\left[\ell\left(G(X), Y\right)\right]$$

where $\Pi$ is a class of admissible policies (loss functions) and $\ell$ is the associated scoring function. The generator learns to produce scenarios that are robust to any evaluation in $\Pi$. The adversary learns to find the hardest evaluation for the current generator. At convergence, the generator cannot be gamed by any evaluation in the policy class.

This is not just an abstract game-theoretic formulation. The key insight is that $\Pi$ is defined using *elicitable* risk functionals — risk measures that can be characterised as the minimiser of some expected score. Quantiles (VaR) are elicitable; expected shortfall (ES/CVaR) alone is not, but it is *jointly elicitable* with VaR. This means the adversarial policy class can include both VaR and ES as simultaneous objectives, which is exactly what Solvency II requires for internal models at the 1-in-200 level.

---

## Why elicitability is load-bearing here

The reason elicitability matters is that it guarantees the adversarial training process has a well-defined objective. If you are trying to train a generator to produce scenarios that stress-test a particular risk measure, you need that risk measure to be expressible as the solution to an optimisation problem — otherwise the adversary cannot compute meaningful gradients.

VaR at level $\alpha$ is the $\alpha$-quantile of the loss distribution, which minimises the asymmetric absolute loss $\rho_\alpha(u) = u(\alpha - \mathbf{1}_{u<0})$. ES is the conditional mean above VaR, and while it is not individually elicitable, Fissler and Ziegel (2016) proved it is jointly elicitable with VaR through a family of Bregman-type scores. The paper uses this result directly: the adversarial policy class over (VaR, ES) pairs is well-defined, the gradient is computable, and training converges.

What you end up with is a generator that, when conditioned on current covariates, produces a distribution of scenarios that correctly represents the joint (VaR, ES) tail behaviour — not by matching historical marginals, but by satisfying the adversarial objective across the full policy class. It will generate scenarios that push your portfolio towards its tail risk, conditioned on where you are now.

---

## Why this beats historical replay and parametric bootstrapping

**Historical replay** is transparent and easy to explain to a board or regulator. Its deficiency is that it assumes the future tail is bounded by the past tail. For a UK home insurer in 2026, replaying 2007 floods for the climate risk component of your ORSA is not stress-testing; it is using a known event as a calibration point. More importantly, it cannot generate scenarios that combine risks in new ways — a cold winter with elevated repair inflation and a claims surge is not in the historical record.

**Parametric bootstrapping** addresses the first problem — you can sample from the tails of a fitted distribution, not just observed points — but it introduces dependence on distributional assumptions that are often wrong in the tails. If your copula is Gaussian and the true tail dependence is upper-tail concentrated (motor claims during freeze events, where frequency and severity are correlated at the extremes), your bootstrap scenarios will systematically understate the tail.

**GAR** addresses both. The generator is a neural network with no fixed distributional form. The adversarial training process pushes it to produce scenarios that are hard for any policy in the class, which means it will learn tail dependencies implicitly. The empirical results in the paper, using S&P 500 data as a test case, show GAR scenarios outperform unconditional sampling, econometric models, and direct predictive baselines on downstream risk evaluation — meaning the scenarios preserve the tail risk properties that matter for a portfolio manager (or, by analogy, an insurer).

---

## What this means for pricing teams

ORSA is nominally a reserving and capital function, but pricing teams in UK personal lines have direct exposure to ORSA outputs:

- **Reinsurance attachment calibration.** If your ORSA scenarios understate correlated tail risk, your XL attachment points are too low. You are buying less reinsurance than you should, which is a pricing error that shows up in the P&L when events correlate.
- **Large loss loadings.** Pricing-level large loss loadings for property catastrophe are often derived from scenario analysis. If the scenario set is historically bounded, the loading is too low.
- **Premium adequacy in adverse scenarios.** ORSA scenarios should include underwriting year scenarios — what does a competitive pricing cycle followed by an attritional loss spike look like? These are not in the historical record for most books.

The GAR formulation is particularly useful for the third of these: you can condition on a "current state" that includes market pricing conditions (combined ratio across the market, rate change indices) and generate scenarios for claims experience over the next 12–24 months. The adversarial training ensures the generator will find correlated stress scenarios that your current portfolio is exposed to, not generic historical replays.

**Integration with insurance-distributional.** We are building `GARScenarioGenerator` into [insurance-distributional](https://github.com/burning-cost/insurance-distributional) as part of Phase 52. The library currently implements distributional GLMs for Tweedie, gamma, and negative binomial responses. GAR extends this to conditional scenario generation: given a covariate vector describing the current risk state, generate a distribution of future loss scenarios that is adversarially trained to represent tail risk. The implementation will expose the same Polars-in, Polars-out interface as the existing distributional models, and will support (VaR, ES) and quantile-based policy classes as the adversarial objective.

---

## Honest assessment

This is a machine learning paper from a finance background, and the empirical validation is on S&P 500 daily returns. It is not an actuarial paper. The gap from financial returns to insurance claim distributions is real: insurance losses are heavy-tailed, zero-inflated, and have much longer observation delays than daily equity returns. The joint (VaR, ES) elicitability result carries over, but whether the adversarial training converges well on insurance data — where you have far fewer tail observations — is an open question.

The minimax formulation also requires enough data to train a generator network. If you are running ORSA on a niche commercial line with 500 claims per year, GAR is not appropriate. It is designed for lines with enough tail observations to train a conditional distribution: home, motor, or aggregated commercial property.

We think the most immediate use case is not pure ORSA compliance — which regulators still expect to be grounded in something interpretable — but as a challenge to existing scenario sets. Run GAR on your historical portfolio data, generate 10,000 adversarial scenarios, and compare them to the scenario set you are currently submitting to your board. If GAR finds correlated stresses that your current scenarios miss, that is a finding worth acting on. It does not replace the narrative scenarios in your ORSA report; it improves the technical basis that underpins them.

The paper is technically rigorous and the minimax framing is the right one for ORSA. We are publishing the implementation alongside this post so you can evaluate it on your own data.

---

*arXiv:2603.08553 — Asadi & Li, "Generative Adversarial Regression: Learning Conditional Risk Scenarios", March 2026.*
