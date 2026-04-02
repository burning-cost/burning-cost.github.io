---
layout: post
title: "Your Reserve Already Has a Risk Margin — What RL Actually Adds (and Doesn't)"
date: 2026-04-02
author: burning-cost
categories: [research]
tags: [reserving, rl, reinforcement-learning, ppo, cvar, risk-margin, solvency-uk, chain-ladder, odp-bootstrap, capital-modelling, actuarial, arXiv-2504-09396]
description: "A new paper (arXiv:2504.09396) uses PPO reinforcement learning with a CVaR constraint to manage a reserve buffer. The framing is interesting — but this is not reserving in the actuarial sense, and a PPO black-box will not get sign-off under Solvency UK."
permalink: /2026/04/02/cvar-rl-reserving-risk-margin-what-rl-adds/
math: true
---

A paper appeared on arXiv in April 2026 (arXiv:2504.09396) proposing a reinforcement learning approach to claims reserving, using Proximal Policy Optimisation with a CVaR constraint to manage a reserve buffer through changing macroeconomic regimes. The authors pitch this as a solution to a real problem: classical reserving methods produce point estimates or symmetric uncertainty intervals, neither of which is optimised for tail risk.

We find the CVaR framing genuinely interesting. We find the rest of it hard to recommend to anyone working in a UK general insurance reserving or capital team right now. The paper is worth reading carefully — which means reading it precisely enough to see what it is actually doing, and what it is not.

---

## What the paper actually does

The Markov decision process here takes incurred losses as inputs and produces reserve levels as actions. The agent — trained with PPO using Stable-Baselines3 — learns a policy that minimises a reward function penalising both reserve shortfall and CVaR of that shortfall, subject to macroeconomic regime shocks drawn from four Gaussian multipliers.

Note what is conspicuously absent: the paper does not fit development factors. It does not complete a triangle. It does not estimate IBNR from claims development patterns. It assumes you already have an IBNR best estimate from somewhere — chain-ladder, Bornhuetter-Ferguson, whatever — and then asks: given that estimate and some macro state, how much buffer should you hold above it?

This is not reserving. It is dynamic capital buffer management. The distinction is not pedantic. The hard actuarial problem in reserving is projecting an incomplete development triangle to ultimate — deciding how far an accident year's losses will eventually develop from where they stand today. The paper skips that problem entirely and works with the answer already in hand.

The CAS database (US, 1998–2007 accident years) provides the test data. That is a reasonable research dataset. It is not UK motor, or UK employers' liability, or any line shaped by the Ogden rate, periodical payment orders, or the post-Woolf liability environment. Whether the trained policies transfer to UK triangles is genuinely unknown.

---

## Chain-ladder plus ODP bootstrap already does this

The paper's claimed gap versus classical methods is that they produce no explicit CVaR-optimised reserve level. This is technically correct and practically overstated.

England and Verrall (2002) established the overdispersed Poisson bootstrap framework that has been standard in UK reserving since. It produces a full predictive distribution of ultimate claims, not a point estimate. CVaR at any percentile — including the 99.5th percentile relevant to the SCR — can be read directly from that distribution. The chain-ladder generates the central estimate; the ODP bootstrap wraps it in a credible uncertainty range. Risk margins are applied on top, under Solvency UK rules, to cover the cost of capital.

The existing toolkit, in other words, already handles tail risk in reserves. It does so transparently, with a documented basis, on a method that any actuarial reviewer can interrogate.

The RL approach adds one thing the bootstrap does not explicitly provide: a sequential policy — a rule for adjusting the buffer dynamically as new information arrives. That is an interesting research contribution. We are sceptical it is a useful practical contribution at this moment, for reasons that have nothing to do with the maths.

---

## The regulatory problem is not going away

Solvency UK (PRA PS2/24, PS15/24) requires actuarial sign-off on best estimates. The actuarial function must be able to document the basis, explain the methodology, and certify that the estimate is best-estimate in the regulatory sense.

A PPO neural network policy, trained on a reward function the actuary had no hand in designing, does not satisfy that requirement. This is not a criticism specific to RL — it applies to any black-box model used in the actuarial function output. Solvency II Article 83 (which Solvency UK adopts) requires the best estimate to be calculated using appropriate actuarial and statistical methods. "Appropriate" in the PRA's interpretation means documented, validated, and explainable to a reviewer who was not in the room when the model was built.

SHAP attributions exist for neural networks. RL policy attribution is an active research area. In five years there may be a credible path to an interpretable RL reserving policy that a Chief Actuary could sign. Today there is not. We think the paper should have engaged with this constraint directly rather than leaving it as an implicit reader exercise.

The CVaR framing, ironically, is where RL does have genuine regulatory relevance — not in reserving but in capital modelling. The SCR is a 99.5th percentile one-year value-at-risk measure. Internal model teams doing dynamic solvency simulation, ORSA stress testing, or ICAAP scenario analysis have a natural interest in policies that explicitly target tail-risk metrics. That application — capital buffer management under stress — is closer to what the paper actually demonstrates, and it does not run into the actuarial sign-off barrier in the same way.

---

## The macroeconomic regime modelling is thin

The paper's treatment of macroeconomic regimes is four Gaussian shock multipliers with curriculum learning to sequence their appearance during training. This is identified as a feature distinguishing the approach from static reserving methods.

It is worth comparing this to the Bayesian HMM reserving paper (arXiv:2406.19903, JRSS-A 2025), which models body and tail regime-switching in development factors using a proper hidden Markov model with principled inference. The contrast is instructive. The RL paper's regimes are simple multipliers with no structural justification; the HMM paper's regime transitions are estimated from the data, with uncertainty quantified. For UK lines where macro environment changes materially affect reserve adequacy — motor personal injury with Ogden rate changes, employers' liability with inflation, financial lines with credit cycles — the richer HMM approach is more credible.

Real macro-regime modelling for UK reserving would need to incorporate at least: the Ogden discount rate (currently -0.25%), the PPO election rate for large bodily injury claims, claims inflation measured independently of general CPI, legal environment shifts (post-LASPO, the Civil Liability Act). None of this is reflected in four Gaussian multipliers on a US casualty dataset.

---

## What the CVaR framing is genuinely worth

We should not end on pure scepticism. The paper's core insight — that reserve management can be framed as a sequential decision problem with an explicit tail-risk objective — is the right observation. Actuaries already make sequential buffer decisions implicitly. The RL framing makes those decisions into a learnable policy, which means they can in principle be optimised against historical data rather than set by judgment alone.

The application we find most credible is internal capital modelling: a team doing Solvency II ORSA simulations could use a CVaR-constrained RL policy to stress-test whether their capital buffer management strategy degrades under extreme scenarios. This does not require actuarial sign-off in the same way, it is stress-testing rather than best-estimate production, and the explicit CVaR objective maps cleanly onto the 99.5th percentile SCR target.

For that narrower application, the paper's framework is worth understanding. For the broader claim that this constitutes a practical advance in reserving, we remain unconvinced until three things change: the approach handles triangle completion rather than presupposing it, interpretability reaches a level acceptable to actuarial reviewers, and the methodology is validated on UK triangles with UK macro covariates.

---

*arXiv:2504.09396. CAS triangles, PPO + CVaR, Stable-Baselines3 v1.8.0 / Gymnasium v0.29. No public code repository at time of writing.*
