---
layout: post
title: "When Reserves Learn from Experience: What RL-Based Reserving Means for Your Pricing Loss Ratios"
date: 2026-04-02
categories: [techniques, reserving, pricing, actuarial]
tags: [reserving, rl, reinforcement-learning, sac, soft-actor-critic, mdp, pricing, loss-ratios, individual-claims, chain-ladder, wüthrich, avanzi, richman, ocl, reserve-uncertainty, solvency-ii, arXiv-2601-07637]
description: "Avanzi, Richman, Wüthrich et al. (arXiv:2601.07637) treat individual claim development as a Markov decision process, using Soft Actor-Critic to revise outstanding claim liability estimates over time. We look past the reserving mechanics to what this means for pricing actuaries relying on historical run-off to calibrate loss ratio margins."
math: true
author: burning-cost
---

Reinforcement learning has arrived in reserving. A paper from Avanzi, Richman, Wüthrich and colleagues (arXiv:2601.07637, January 2026) frames individual claims development as a Markov decision process and uses Soft Actor-Critic (SAC) to produce revised outstanding claim liability (OCL) estimates at each development period. This is a long way from chain-ladder.

We are not going to pretend this is heading into a UK reserving team's year-end workflow next quarter — the audit trail problem alone would give a Chief Actuary pause, and the paper's code is not public. But the right way to read this work is not "how might reserving actuaries use it?" It is: **what does a fundamentally different approach to reserve estimation tell us about the run-off properties we have been assuming in pricing?**

---

## The MDP framing, briefly

Chain-ladder is a cross-sectional estimator. At each valuation date it fits a model to the triangle as it currently stands. It has no memory of how it arrived at last quarter's estimate and does not optimise for the quality of the path through development, only the endpoint.

The RL framing is sequential by construction. Each open claim is an environment. At each development period the SAC agent observes the claim's current state — paid to date, case reserve, claim age, whatever features are available — and proposes a revised OCL. The reward function penalises two things simultaneously: prediction error (how far was the OCL from the eventual settlement?) and reserve instability (how much did the estimate jump between periods?). SAC's entropy regularisation reinforces this: the policy is pushed toward smooth revisions unless the evidence compels a large move.

The paper makes three non-trivial technical contributions beyond the headline RL application. First, a claim initialisation approach for open claims at the valuation date — this matters because you cannot simply start the MDP from scratch for a claim that is already two years into development. Second, a rolling-settlement training scheme that mimics genuine quarter-over-quarter development rather than fitting on the full run-off history at once. Third, importance weighting for large claims, which addresses the familiar problem that a handful of major losses dominate aggregate reserve error.

---

## The accuracy versus stability dilemma

Here is why the reward function is interesting to a pricing actuary rather than just a reserving actuary.

The tension between accuracy and stability is not an academic abstraction. It maps directly onto something we deal with every time we calibrate pricing margins on a line of business with long development. When reserves are volatile — large upward or downward development surprises quarter to quarter — the historical combined ratios we use to anchor pricing become unreliable. We load for uncertainty we cannot quantify cleanly. We argue about whether the prior accident year's adverse development was signal or noise.

Chain-ladder does not have an explicit view on this trade-off. It has no mechanism to say "we could be more accurate if we allowed larger quarterly revisions, but we choose not to because instability has downstream costs." The RL approach explicitly encodes that preference in the reward. The resulting run-off pattern — the sequence of OCL estimates from first development period to settlement — has a different statistical character than chain-ladder run-off. Revisions are smoother by design.

That matters when you are computing historical accident-year ultimate loss ratios for pricing. If the reserving approach has been chain-ladder and you switch to an RL-derived method, the run-off profile of your historical triangles looks different. Not just the level of the ultimates, but the pattern of when and how much development you observed in prior periods. Calibrating a Bornhuetter-Ferguson a priori ratio, or setting a reserve uncertainty loading, from a historical run-off that was chain-ladder derived will give you different answers than if it was RL derived. We think this is underappreciated.

---

## What you are inheriting when you calibrate pricing loss ratios

Most UK commercial pricing teams work from combined ratios derived from the statutory reserving basis — which, for the near term, means something chain-ladder adjacent. The key properties of CL run-off that affect pricing margin calibration are:

**Development factor instability.** CL development factors estimated on thin diagonals are noisy. Large prior-year development surprises flow directly into the historical loss ratios you are regressing against. If your pricing model uses five years of accident-year ultimates as the dependent variable, and three of those years have had material adverse development in the last two diagonals, your model is fitting partly to reserving noise.

**Smoothing versus accuracy.** CL makes no explicit choice to trade accuracy for smoothness. It just fits the triangle. In years where the book changes composition — higher severity mix, or a new distribution channel — the factors can be badly biased, and the bias is not predictable in advance.

**Correlation across accident years.** CL shares development factors across accident years. This creates correlation in the run-off pattern that is often not reflected in reserve uncertainty models. When pricing teams assess reserve risk for their combined ratio projections, this correlation tends to be ignored.

An RL-derived reserving approach, because it optimises explicitly for the accuracy-stability trade-off at the claim level, produces ultimates with different correlation and bias properties. Whether "different" means "better for pricing" depends on the book. But it is not neutral.

---

## The Wüthrich lineage and where this is heading

It is worth noting the research lineage here because it affects how seriously to take the longer-term trajectory. Mario Wüthrich is one of the few academics whose work has moved from paper to industry tool within a reasonable timeframe. The individual claims framework from Richman and Wüthrich (arXiv:2602.15385, arXiv:2603.11660) — which we covered recently in the context of [one-shot RBNS reserving](/2026/04/02/individual-claims-reserving-linear-regression-beats-deep-learning.html) — has already produced results clean enough to implement. The RL paper sits in the same intellectual tradition: individual claim states, micro-level dynamics, optimising over sequences.

Our `insurance-severity` library contains a `ProjectionToUltimate` implementation that draws on this same research group's work. The RBNS projection methodology there is a one-shot approach — it estimates ultimate from current claim state without modelling the full sequence. The RL approach in arXiv:2601.07637 is the sequential extension of the same underlying intuition. These are not isolated experiments; they are steps in a coherent programme.

Our honest assessment of the timeline for production adoption is three to five years, and only for firms that have already built individual claim-level data infrastructure. The obstacles are real:

- **Audit trail.** Solvency II Article 83 requires the best estimate to be explainable to the PRA. "A neural network policy learned from historical development" does not currently satisfy that bar. This is a solvable problem — SHAP-based attribution for RL policies is an active research area — but it is not solved yet.
- **Compute.** SAC training on a large casualty book is not something you run on a laptop overnight. This is decreasing as a constraint but it is a constraint now.
- **No public code.** Unlike some of the recent Wüthrich-lineage work, there is no implementation to clone and run against your own data. This limits the ability to pressure-test the claims before committing to a project.

---

## The pricing takeaway

We think the right immediate action for a pricing actuary reading this paper is not to commission a reserving RL project. It is to ask a sharper question of whatever reserve run-off history you are currently using to calibrate pricing margins: what are the implicit assumptions about accuracy and stability baked into that run-off, and are those assumptions helping or hiding the underlying claims economics?

Chain-ladder is not wrong. But it makes choices — about smoothing, about shared development factors, about what information to discard — that propagate into your loss ratio history in ways that are easy to forget about. The RL paper makes those same choices explicit and tradeable. That explicitness is the thing worth importing into pricing practice, well before the RL models themselves are audit-ready.

---

*arXiv:2601.07637 — "Reinforcement Learning for Micro-Level Reserving" — Avanzi, Majewski, Richman, Taylor, Wüthrich. January 2026.*
