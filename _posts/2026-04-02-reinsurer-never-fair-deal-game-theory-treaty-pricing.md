---
layout: post
title: "Why Your Reinsurer Is Never Offering You a Fair Deal"
date: 2026-04-02
author: burning-cost
categories: [research]
tags: [reinsurance, game-theory, treaty-pricing, stackelberg, pareto, bowley, p2p-insurance, lloyds, arXiv-2602.14223, actuarial]
description: "Boonen & Ghossoub (arXiv:2602.14223) prove that when a reinsurer sets terms first — Bowley/Stackelberg — the treaty is mathematically guaranteed to be worse for total welfare than the cooperative solution. The inefficiency is structural, not a matter of greed."
permalink: /2026/04/02/reinsurer-never-fair-deal-game-theory-treaty-pricing/
---

Every reinsurance renewal starts with the same ritual. The reinsurer produces a quote. Your team models the cession. You go back with a counter. The reinsurer moves a little. You agree something, or you don't.

This looks like negotiation. Formally, it is not. It is a Stackelberg game in which the reinsurer is the leader. They set the loading; you optimise your cession against it. The outcome of that game has a name — the Bowley solution — and it has a well-established mathematical property: it is always suboptimal for total welfare compared to the cooperatively negotiated solution.

That property is not obvious, and it is not widely appreciated in the market. A recent paper by Boonen and Ghossoub (arXiv:2602.14223, February 2026) establishes it rigorously in the context of peer-to-peer reinsurance pools, extending results from classical bilateral treaties into multi-party settings. We are not building this — the implementation territory is already covered — but the game-theoretic result is worth understanding if you price or buy reinsurance.

---

## The formal setup

Boonen and Ghossoub consider a pool of *n* participants (insurers or cedants), each with their own risk exposure. A reinsurer offers a treaty that shares losses across the pool in exchange for a loading above the actuarially fair premium. The cedants respond by choosing how much risk to cede.

Two solution concepts are in play.

**Pareto-optimal.** A treaty is Pareto-optimal if you cannot make any party better off without making at least one party worse off. This is the cooperatively negotiated outcome: all parties sit down together, the total surplus in the arrangement is maximised, and it is divided according to some agreed allocation. The reinsurer earns a fair return on capital; the cedants receive risk transfer at the lowest sustainable price. The total welfare — summed across all participants — is maximised.

**Bowley-optimal.** A treaty is Bowley-optimal (or Stackelberg-optimal) if the reinsurer maximises their own objective by choosing the treaty loading, knowing that the cedants will respond optimally to whatever loading is set. The reinsurer acts first; the cedants optimise against the reinsurer's choice. This is not joint maximisation — it is sequential maximisation, and the two are not equivalent.

The paper's core result (Proposition 3.3 and the welfare comparison in Section 4) is that the Bowley equilibrium loading is strictly higher than the Pareto-optimal loading. The reinsurer extracts surplus from the pool that, under the cooperative solution, would have been shared between reinsurer and cedants. Total welfare — summed across all n+1 parties — is lower under Bowley than under Pareto.

This holds not just for a single bilateral treaty, but for the full P2P pool structure with multiple cedants.

---

## What "extracts surplus" actually means

The intuition is straightforward. Under Pareto optimality, the reinsurer's loading is set to maximise the total size of the pie. The reinsurer gets a slice; the cedants get the rest; the total is as large as possible.

Under Bowley optimality, the reinsurer knows that the cedants will reduce their cession as loading increases. The reinsurer solves: given that the cedants will behave rationally in response, what loading maximises *my* expected profit? The answer is a higher loading — the reinsurer sacrifices some of the cedants' demand (they buy less reinsurance) in exchange for a higher margin on what they do buy.

The result is a smaller total pie. The reinsurer has a larger slice of a smaller amount. The cedants have a smaller slice than they would have received under the cooperative solution. And — crucially — the cedants are also *under-reinsured* relative to the cooperative optimum: they are carrying more risk than is socially efficient because the loading has priced them away from the optimal cession.

This is not a hypothetical pathology. It is the structural result of the reinsurer having first-mover advantage.

---

## The P2P context, and why it is niche

The paper is framed around peer-to-peer pools, because that is the context in which multi-party treaties are most formally modelled. In a P2P pool, each member is both cedant and, implicitly, reinsurer to the others. Boonen and Ghossoub extend the bilateral Pareto/Bowley framework from their earlier work (Boonen & Ghossoub, EJOR 2023) into this multi-party setting.

The honest position on UK relevance: P2P insurance as a product category is small and shrinking. Laka, which ran the most credible UK P2P cycling insurance model, was acquired by Allianz in 2023. Dinghy and Guardhog are niche. Bought By Many moved away from pooling mechanics. The P2P pool structure described in the paper does not map cleanly onto how Lloyd's syndicates or personal lines carriers operate.

For this reason we are not building the `P2TPReinsuranceGame` class that the paper implies. The P2P pooling mechanics already exist in `insurance-optimise` via `LinearRiskSharingPool`, and the cession optimisation problem is covered by `RobustReinsuranceOptimiser`. A third reinsurance-specific class, with limited UK use cases, would fragment the library without a clear audience.

---

## The part that does apply to London Market

The bilateral version of the Pareto/Bowley result is where the London Market connection sits.

When a Lloyd's syndicate buys proportional quota-share reinsurance, or places an XL treaty in the market, the reinsurer sets the terms: the rate on line, the cession structure, the reinstatement provisions. The syndicate responds. This is Stackelberg. It has always been Stackelberg. The question the paper makes precise is: how far is this from the cooperative solution, and who bears the cost?

The answer is that the cedant bears it. The Bowley loading is higher than the Pareto loading. The cedant is underinsured relative to the efficient outcome, and the welfare loss is borne by the cedant, not shared equally.

This has two practical implications for treaty negotiations.

**The cooperative alternative exists.** The paper proves that a Pareto-optimal treaty always exists — it is not a theoretical construct that requires perfect information or altruism from the reinsurer. A pricing team could, in principle, compute the Pareto-optimal loading given their own loss distribution and the reinsurer's cost of capital, and present it in the negotiation. "Here is the loading at which total surplus is maximised; here is our proposed allocation of that surplus." This reframes the conversation from "here is the reinsurer's quote, take it or leave it" to "here is the cooperative solution, what are you taking beyond it?"

**The surplus extraction is quantifiable.** For a given treaty structure, you can estimate the Pareto-optimal loading from your own model of the reinsurer's economics (their cost of capital, their diversification benefit, their target return on risk-adjusted capital). The difference between that loading and the market loading is, in the language of the paper, the reinsurer's Stackelberg rent. Knowing this number does not give you leverage you do not otherwise have — the reinsurer still has first-mover advantage — but it tells you whether the treaty is "expensive by 10%" or "expensive by 40%", which is useful.

---

## Why the result is counterintuitive

The usual framing in treaty negotiations is that the reinsurer's loading is "fair" if it reflects their cost of capital, and unfair if it includes an excess margin. The Boonen/Ghossoub result says this framing is incomplete. Even a loading that exactly covers the reinsurer's cost of capital can be Bowley-suboptimal if it is set unilaterally rather than cooperatively, because the Pareto-optimal solution would have the reinsurer earning the same return at a lower loading by taking a larger share of a larger total surplus.

The inefficiency is not greed — it is sequential structure. The reinsurer who sets terms first and then watches the cedant optimise cannot achieve the cooperative optimum, even in principle, unless they explicitly commit to the cooperative loading before the cedant responds. And there is no mechanism in the standard treaty market that enforces that commitment.

This is a genuine result. It says the treaty market, as structured, cannot achieve Pareto optimality in bilateral or multi-party negotiations unless pricing is made cooperative. That is not an observation you will find in a Lloyd's market report.

---

## What we are not doing with this

We looked at implementing `P2TPReinsuranceGame` alongside the existing reinsurance optimisation tools. The mathematics is not difficult — closed-form Propositions 3.3, 4.4, and 4.5 are matrix algebra throughout, roughly 350 lines including tests. But the audience is too narrow. UK pricing teams working on personal lines or Lloyd's treaties would get the bilateral insight from the theory; they do not need P2P pool mechanics.

The existing `RobustReinsuranceOptimiser` in `insurance-optimise` already handles the cedant-side question — what cession fraction maximises your risk-adjusted return given a reinsurer's loading? The Bowley insight complements this: whatever loading your reinsurer quotes, you are almost certainly being held above the Pareto optimum. The gap is the Stackelberg rent, and `RobustReinsuranceOptimiser` can help you minimise its impact on your portfolio even if you cannot eliminate it.

---

## The bottom line

The paper is not essential reading for a pricing actuary who buys cat XL. It is essential reading if you want the theoretical backing for something practitioners have always intuited: the reinsurer sets terms that are good for the reinsurer.

What the paper adds is precision. It is not just that the reinsurer's terms are profitable for them — it is that the Stackelberg structure guarantees a welfare loss relative to the cooperative solution, and that welfare loss always falls on the cedant side. The Pareto-optimal treaty exists. It is better for everyone in aggregate. And the standard treaty market mechanism cannot reach it.

That is a result worth knowing.

*Paper: Boonen & Ghossoub, "Pareto and Bowley Optima in Reinsurance Markets with Peer-to-Peer Risk Sharing", arXiv:2602.14223, February 2026.*
