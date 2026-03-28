---
layout: post
title: "Dynamic Pricing for UK Motor and Home Insurance — Build or Buy?"
date: 2026-04-22
categories: [pricing, strategy]
tags: [dynamic-pricing, FCA, GIPP, ENBP, price-elasticity, thompson-sampling, reinforcement-learning, OptiGrad, PCW, Earnix, Akur8, Radar-Live, insurance-elasticity, insurance-optimise, causal-inference, DML, UK-insurance, motor, home]
description: "A practitioner's guide to dynamic pricing in UK insurance: what GIPP actually permits, why your elasticity model likely has a 3-5x bias, and an honest assessment of whether Earnix, Akur8 or open-source tools can fill the gap."
seo_title: "Dynamic Pricing UK Insurance: FCA GIPP Rules, Elasticity Models, Build vs Buy (2026)"
---

The phrase "dynamic pricing" is doing a lot of work in UK insurance conversations right now. At one end you have vendors selling real-time AI optimisation engines. At the other you have headlines about "surge pricing" as if insurers were secretly multiplying premiums during thunderstorms. Neither picture is accurate. This post is for pricing actuaries trying to work out what is actually possible, what the FCA will and will not tolerate, and whether the tooling exists to do it properly.

The answer to the build-or-buy question is not satisfying: commercial platforms have the operational infrastructure but a documented gap in statistical rigour; open-source has the statistical rigour but material capability gaps for the UK-specific regulatory problem. We will be precise about both.

---

## What dynamic pricing actually means for UK insurers

In most industries, dynamic pricing means varying prices over time based on demand, availability or competitive conditions — airlines repricing seats every few minutes, hotels adjusting rack rates by the hour. Insurance dynamic pricing is fundamentally different, for three reasons.

First, the price quoted to a new customer is a commitment — it forms part of a binding contract. The insurer cannot reprice mid-policy the way an airline reprices an unsold seat. Dynamic pricing in insurance means varying prices *at quote time* across customers, across time, and across channels, not repricing existing contracts.

Second, the price must be defensible against actuarial adequacy requirements. A price optimised purely on conversion probability — ignore the risk, just win the quote — creates adverse selection. Any dynamic pricing system must operate as a *multiplier* on top of a risk premium, not as a replacement for it. The research formalises this: OptiGrad (Grari & Detyniecki, arXiv:2404.10275, April 2024) decomposes commercial price as `p_i = c(x_i) × h(x_i)` where `h(x_i)` is the frozen actuarial pure premium and `c(x_i)` is a learnable coefficient bounded in `[a,b]`. The risk model does not move; the pricing layer does.

Third, the PCW (price comparison website) is the dominant distribution channel for UK motor and home. The demand model is not simply "will this customer buy at this price?" but "what is the probability of conversion given own price, competitor prices, and rank position?" Elasticity at rank 1 versus rank 2 on a PCW is a step function, not a smooth curve. Most elasticity models ignore this, which is one reason they underestimate price sensitivity — more on this below.

So: dynamic pricing for UK insurers means a system that sets the pricing coefficient `c(x_i)` per quote, taking into account (a) the conversion probability at that price, (b) the target portfolio composition, (c) competitive position on the PCW, and (d) FCA regulatory constraints.

---

## What GIPP permits and what it prohibits

The FCA's General Insurance Pricing Practices rules (PS21/5, May 2021; PS21/11, November 2021; in force 1 January 2022) are precise about what is restricted. They are frequently misread.

**What GIPP prohibits:**
- Charging a renewing customer more than the Equivalent New Business Price (ENBP) — the price the insurer would charge that same risk if they walked in as a new customer through the same channel
- Price walking: the systematic practice of incrementing renewal prices year-on-year beyond claims inflation
- Sludge: practices that make switching or cancellation unnecessarily difficult

**What GIPP explicitly permits:**
- New business price optimisation — varying prices to acquire customers competitively
- Using ML and algorithmic models in the pricing chain
- Setting new business prices below technical premium to build a book
- Reducing renewal prices below ENBP
- Demand elasticity models applied to new business pricing

This distinction matters. The entire machine learning, optimisation, and demand modelling apparatus is available for new business pricing. GIPP does not restrict how sophisticated your new business pricing model is. What it restricts is the pricing of *existing customers at renewal*.

**The live tension:** Real-time new business price optimisation creates a compliance problem that almost nobody has solved cleanly. If your new business pricing system adjusts prices intraday — responding to PCW competitor movements, time-of-day conversion rates, or marketing spend levels — then the ENBP calculation becomes a moving target. A customer who received their renewal notice at 9am with an ENBP of £480 can, by 2pm, find the same risk quoted at £450 through the new business channel. Technically compliant on the renewal notice date. Practically a loyalty penalty.

The FCA's three-year evaluation of GIPP (EP25/2, July 2025) confirmed that price walking is largely eliminated — the headline metrics moved in the right direction, with average inception price rising from £248.52 to £260.92, driven by claims inflation rather than any reversal of GIPP compliance. But EP25/2 also documented that 28 of the 66 firms reviewed could not demonstrate ENBP compliance with sufficient granularity, and 27 firms had inadequate documentary evidence that their controls were working as intended. The root cause in most cases: complex algorithmic pricing made the ENBP calculation difficult to trace.

The implication for any dynamic pricing build is that ENBP compliance cannot be a bolt-on. It has to be an architectural constraint — either a hard bound in the optimiser, or a real-time monitoring system that flags when dynamic new business pricing has created a gap relative to outstanding renewal notices. No commercial platform we have assessed has this implemented as a documented feature. It is also the most significant gap in open-source tooling.

---

## The elasticity problem: your model is probably wrong by a factor of 3

Before discussing optimisation methods, we need to address the foundation they sit on. Elasticity models at most UK insurers have a serious bias problem that few pricing teams have corrected for.

Woodard & Yi (2020, Journal of Risk & Insurance 87(2):477–500) used instrumental variable estimation on US Federal Crop Insurance data and found that OLS-estimated elasticities are **3 to 5 times too inelastic** relative to IV estimates. The mechanism is endogeneity: premium rates are set actuarially based on risk profile, and risk profile is correlated with other demand drivers. If higher-risk customers also tend to have lower price sensitivity for other reasons — they have fewer alternatives, they value coverage more highly, they spend less time shopping — then naive regression on observed prices and conversion rates will confound the price effect with the selection effect. The OLS estimate of price sensitivity is biased toward zero.

Applied to UK motor PCW pricing: the practitioner consensus on semi-elasticity for private motor is roughly -1.5 to -3.0. That range has never been derived with explicit endogeneity correction on post-GIPP UK data. After correction, the true range is plausibly -3.0 to -6.0 or beyond. If that is right, pricing teams have been systematically leaving margin on the table — they believe customers are less price-sensitive than they actually are, which leads to over-retaining customers at prices they would have accepted as lower, and under-acquiring price-sensitive prospects who might have been won with modest discounts.

GIPP compounds this. Before January 2022, loyal customers were overcharged through price walking. Their revealed price sensitivity was artificially suppressed — they stayed despite rising prices because switching friction was high. Any elasticity model trained on pre-2022 data has the wrong signal for the retention book. Post-GIPP, renewal prices have converged toward new business prices. The price-sensitivity channel for retention is now driven by absolute price level against PCW alternatives, not by a loyalty buffer. Elasticity models need rebuilding on 2022+ data.

**Fixing the bias with DML.** Double Machine Learning (Chernozhukov et al., 2018) handles confounding from observed variables by partialling out the effect of covariates on both the treatment (price) and the outcome (conversion) before estimating the treatment effect. The `insurance-elasticity` library (Burning Cost, v0.1.1 on PyPI) ships `CausalForestDML` and a DR-Learner for heterogeneous treatment effect estimation — CATE estimation across customer segments rather than a single population average.

DML is not a complete solution. It handles confounders that you can measure and include in the model. If unobserved demand shocks drive both price and conversion — a competitor exiting a rating tier simultaneously changes both the prices you quote and the conversion rates you observe — then DML alone is insufficient. A proper instrument is needed: something that shifts prices but is unrelated to latent demand (index-linked rating changes, regulatory capital events that affect a competitor's capacity). That instrument is hard to find in insurance. The Woodard & Yi approach of using cost-shifters as instruments is cleaner in agricultural insurance than in UK motor, where pricing is more discretionary.

**The Bayesian posterior gap.** DML gives point estimates of the CATE. What optimisation algorithms actually need is a posterior distribution over the demand parameter — not just "the elasticity for 25-year-old males in London is -3.8" but "the elasticity for that segment has mean -3.8 and standard deviation 0.7." Without the posterior, any pricing optimiser is operating with false certainty. Kumar et al. (arXiv:2205.01875, 2022) proposed a two-stage approach for airline pricing: first, ML models to extract demand residuals; second, a Bayesian Dynamic GLM on those residuals to build a full posterior over the price-sensitivity parameter. Their simulation showed estimation error falling from 25% to 4% versus naive regression. No open-source insurance implementation of this two-stage approach currently exists.

---

## Technical approaches: what the literature actually shows

### Gradient-based optimisation: OptiGrad

OptiGrad (Grari & Detyniecki, arXiv:2404.10275, April 2024, AXA Research) is methodologically the most sophisticated pricing optimiser in the recent literature. It treats the entire chain — coefficient model, pure premium model, conversion model — as a differentiable computation graph and optimises end-to-end via gradient descent. The fairness extension (Fair-OptiGrad) adds an adversarial training loop to enforce demographic parity using the Hirschfeld-Gebelein-Rényi (HGR) coefficient, which measures nonlinear dependence between price and a sensitive attribute via two learned transformation networks.

This is genuinely new. Previous approaches either enforced fairness post-hoc (tweak the output, check the metric) or treated the premium model and conversion model as separate systems that could not be jointly optimised. OptiGrad does both jointly.

The UK applicability problem is substantial. The framework requires differentiable models throughout — neural networks as the pure premium model, not the GLMs and GBMs that most UK insurers deploy. It was tested on a synthetically enriched Kaggle dataset of 46,129 instances, not on real insurer data. There is no GIPP/ENBP constraint in the formulation. There is no PCW rank dynamics. The demographic parity fairness criterion is not the right criterion under UK law — the Equality Act 2010 and actuarial principles require that price differences be justified by risk differences, not that prices be equal across demographic groups. Direct application to a UK pricing problem would require substantial re-engineering, and the AXA Research team has not published code.

OptiGrad is worth understanding because the architecture is right — decomposing commercial price into a risk component and an optimisable coefficient, then jointly training the coefficient model with conversion and fairness objectives — even if the implementation needs to be rebuilt for the UK context.

### Thompson sampling for demand learning

Thompson sampling (Ganti et al., arXiv:1802.03050, 2018; extended to high-dimensional features in Mathematics 12(8):1123, 2024) is the correct Bayesian framing for the core insurance pricing problem: how do we learn demand while simultaneously setting prices, under uncertainty?

The mechanism: maintain a posterior distribution over the demand parameters (α, β) in a logit or power-law demand model. At each pricing period, sample a parameter draw θ̃ from the current posterior, set the revenue-maximising price given that draw, observe the accept/reject outcome, and update the posterior via Bayes' rule. The exploration-exploitation tradeoff is handled implicitly — when the posterior is wide (high uncertainty), sampled θ̃ values vary widely, producing natural price experiments. When the posterior has tightened, prices converge toward the estimated optimum.

Applied to insurance, the 2024 contextual extension is the relevant formulation: the demand parameters become functions of customer features, β(x) = x'γ, giving heterogeneous elasticity across the book. This is the right architecture.

What Thompson sampling does not address out of the box: the PCW rank effect (conversion depends on competitive position, not just own price), delayed feedback (renewal outcomes arrive months after quote), the ENBP hard constraint on the action space, and adverse selection risk (acquiring customers at a profitable price but the wrong risk composition). Each of these requires an extension. The constraints and feedback delay are tractable engineering problems; the PCW rank effect requires a structural model of competitor behaviour.

### RL for PCW pricing

The most practically motivated academic work on dynamic insurance pricing comes from the Alan Turing Institute. Two papers in 2023–2024 are worth reading carefully.

Treetanthiploet et al. (arXiv:2308.06935, August 2023) proposed a hybrid model-based/model-free RL approach for PCW pricing. The model-based phase offline pre-trains a market simulator — avoiding the cold-start problem of deploying a naive RL agent on a live book. The model-free phase uses contextual bandit updates on the live pricing policy. The paper demonstrates better sample efficiency than pure model-free approaches, which is the decisive practical advantage: you cannot run pricing experiments on your entire renewals book while your RL agent is exploring. Authors have real UK insurer connections. Code was not released.

Young et al. (arXiv:2408.00713, August 2024) addresses the problem that UK insurers actually face: not pure profit maximisation, but portfolio pursuit — achieving a target mix of risks by age, geography, vehicle type, and other dimensions. The paper introduces a **k-value dimensionality reduction** that makes the problem tractable. The key insight: define the k-value as

```
k_π(s,ρ,t) = 1 − [V_π(ρ∪{s},t) − V_π(ρ,t)] / C(s)
```

where V_π is the value function and C(s) is the expected premium for customer s. The optimal pricing policy reduces to:

```
π(s,ρ,t) = argmax_a P(accept|s,a) × (a − k_π(s,ρ,t+1))
```

This separates the portfolio-level value calculation from the per-customer pricing decision. The k-value depends only on portfolio state (ρ,t), not on individual customer features — enabling backward induction rather than high-dimensional experience replay. Results: +7% profit versus baseline (p<0.0001, Cohen's d=0.51), with portfolio composition targets maintained. Code is available at [github.com/EdwardJamesYoung/RL-portfolio-pursuit](https://github.com/EdwardJamesYoung/RL-portfolio-pursuit).

The caveats are material. The market is synthetic; T=1,000 customers per epoch; no non-stationarity is modelled; there is no GIPP/ENBP constraint anywhere in the formulation; and the licence is CC BY-NC 4.0, which prohibits commercial use. The GIPP gap is fixable in principle — adding a binary new/renewal flag to the state space and capping the action `a_t` at the ENBP for renewal customers — but it requires re-running the RL training with the modified MDP.

---

## Commercial platforms: what they can and cannot do

### Akur8

Akur8 is the most technically advanced commercial platform for demand and elasticity modelling in insurance as of March 2026. Its demand module claims separate static (conversion propensity) and dynamic (price sensitivity) models, ingestion of external competitor quote data, and proprietary algorithms including a "Derivative Lasso" for regularised elasticity estimation.

What Akur8 does well: it is purpose-built for insurance pricing, it integrates with the rate-building workflow, and it has real UK insurer deployments. The 2024 acquisitions of Arius (Milliman reserving) and Matrisk (filings intelligence) suggest a push toward a full actuarial platform.

The gaps we can document: no public evidence of endogeneity correction in the elasticity estimates. The "Derivative Lasso" algorithm is not publicly described — we cannot assess whether it addresses the Woodard & Yi bias. No Bayesian uncertainty quantification on elasticity. No documented GIPP/ENBP constraint in the rate optimisation module. As a proprietary black box, it is not auditable in the way that Consumer Duty and the emerging FCA/PRA AI principles require.

### Earnix

Earnix's Price-It platform is the older incumbent, now positioned as an AI decisioning engine with real-time GLM/ML deployment, Verisk integration, and a combined pricing/credit risk module (from the October 2024 update). Azure-backed infrastructure for real-time scoring is a genuine differentiator.

Our honest assessment: Earnix looks like rules-based optimisation over a rate relativities grid with modern deployment infrastructure. There is no public documentation of a Bayesian demand module, endogeneity handling, or GIPP-aware constraint. If Earnix has solved the endogeneity problem internally, that information is not available in any published source we have found.

### Radar Live (WTW)

Radar Live is the deployment infrastructure that most large UK insurers already use for actuarial models. The September 2024 update adding real-time Python execution is genuinely significant — it opens the door to Python ML models running at quote time, not just at batch. This closes the gap between Radar's strong UK market penetration and the modern ML stack.

But Radar Live is a deployment and rating engine, not an optimiser. There is no demand elasticity module, no bandit learning, no portfolio pursuit. An insurer running Radar Live for deployment still needs to build the optimisation logic, which means a dynamic pricing capability on top of Radar is a build, not a buy.

---

## Where the open-source tools are and where the gaps are

The honest landscape in March 2026:

| Capability | Status |
|---|---|
| DML elasticity estimation (CATE/ATE) | Available — `insurance-elasticity` v0.1.1 |
| ENBP-constrained static optimisation | Available — `insurance-optimise` |
| Bayesian posterior over elasticity | Not implemented anywhere (open or commercial) |
| PCW rank-position demand model | Not implemented anywhere |
| Contextual bandit with ENBP constraint | Not implemented anywhere |
| Off-policy evaluation for pricing | Not implemented anywhere (open source) |
| Dynamic ENBP compliance tracker | Not implemented anywhere |
| RL portfolio pursuit (research quality) | arXiv:2408.00713 + GitHub, CC BY-NC |

The `insurance-elasticity` library is the right starting point for endogeneity-corrected elasticity estimation. It ships `CausalForestDML` and a DR-Learner for heterogeneous treatment effects — CATE estimates segmented by customer features. What it does not provide is a full posterior distribution over the demand parameters, which is what you need to run Thompson sampling or uncertainty-aware optimisation. The DML output is a point estimate plus a confidence interval at each customer segment; that is not the same as a proper Bayesian posterior.

`insurance-optimise` handles the static portfolio optimisation problem — maximise expected margin subject to conversion rate floor, ENBP ceiling, and volume constraints — using SLSQP with shadow prices. It does not do dynamic or bandit-based repricing.

The biggest genuine gap, and the one with the highest regulatory consequence, is the ENBP constraint embedded in a live optimiser. Every paper we reviewed — OptiGrad, both Turing Institute RL papers, the Thompson sampling literature — optimises without a ENBP hard constraint. Every commercial platform either lacks it or does not document it. Getting this right is not primarily a machine learning problem; it is a data architecture problem. You need the ENBP for each renewal risk, computed at the time the renewal notice was sent, stored and queryable in real time when the new business price is set. Most insurers do not have that data architecture in place.

---

## What a build actually looks like

If you are evaluating whether to build dynamic pricing capability in-house, here is our honest assessment of the components and their difficulty.

**Component 1: Elasticity model with endogeneity correction.** Use DML via `insurance-elasticity`. Budget 3–6 weeks for a proper implementation on your book, including instrument selection for any residual endogeneity not handled by the DML. This is the single highest-leverage investment — correcting a 3x bias in your elasticity estimates will improve every downstream decision, regardless of which optimiser you use.

**Component 2: Demand model for PCW quoting.** Build a conversion model that conditions on PCW rank position, not just own price. This requires competitor price data — either from a data provider or reconstructed from your PCW rank history. Without it, your demand model is misspecified. Difficulty: medium-high; data availability is the binding constraint.

**Component 3: Static optimisation with ENBP constraint.** `insurance-optimise` covers this. It is a batch system, not real-time, which is appropriate for portfolio-level decisions. For intraday quote-level decisions, you would need a lighter online version.

**Component 4: ENBP compliance tracker.** Build this before anything else that involves live pricing optimisation. The architecture needs: ENBP stored per risk at renewal notice date; a comparison service queried at new business quote time; an alert when the gap exceeds a threshold. This is 4–8 weeks of data engineering, not ML.

**Component 5: Bandit or RL optimiser.** Do not start here. The Thompson sampling or RL layer is only useful once your elasticity model is correctly specified, your demand model conditions on competitive position, and your ENBP constraint is architecturally embedded. If you add a bandit optimiser to a mis-specified demand model, you will explore efficiently in the wrong direction. The Treetanthiploet et al. model-based pre-training approach is the right way to bootstrap — train a market simulator offline before deploying any live exploration.

---

## The build-or-buy verdict

Buy if you need operational infrastructure fast: Akur8 or Earnix for deployment, Radar Live for Python execution. None of them give you a statistically rigorous dynamic pricing system, but they give you the plumbing.

Build if statistical rigour matters: endogeneity-corrected elasticity, Bayesian uncertainty, ENBP as a hard constraint, off-policy evaluation for A/B pricing tests. The open-source libraries cover parts of this. The Bayesian posterior and ENBP-constrained bandit are genuine white space — no open-source implementation exists, and no commercial platform has documented them.

The uncomfortable truth is that neither option gives you a complete, regulatory-compliant, statistically-sound dynamic pricing system today. The academic literature is 12–18 months ahead of what is available in production-grade tooling, open or commercial. The k-value portfolio pursuit framework from Young et al. is the most practically useful recent result, and its code is on GitHub behind a non-commercial licence. The two-stage Bayesian elasticity posterior approach from Kumar et al. has no implementation at all.

If you are starting this work now, our recommended sequence: fix the elasticity bias first, build the ENBP compliance tracker second, then evaluate whether the conversion from static to dynamic optimisation is worth the additional complexity. Most insurers we are aware of would materially improve their pricing outcomes from step one alone, without touching the dynamic machinery at all.

---

*FCA GIPP rules: PS21/5 (May 2021) and PS21/11 (November 2021). GIPP three-year evaluation: EP25/2 (July 2025). OptiGrad: Grari & Detyniecki, arXiv:2404.10275, April 2024. RL portfolio pursuit: Young et al., arXiv:2408.00713, August 2024; code at github.com/EdwardJamesYoung/RL-portfolio-pursuit (CC BY-NC 4.0). PCW pricing RL: Treetanthiploet et al., arXiv:2308.06935, August 2023. Thompson sampling: Ganti et al., arXiv:1802.03050, 2018; contextual extension Mathematics 12(8):1123, 2024. Insurance demand endogeneity: Woodard & Yi, Journal of Risk & Insurance 87(2):477–500, 2020. Bayesian demand estimation: Kumar et al., arXiv:2205.01875, 2022. insurance-elasticity v0.1.1 and insurance-optimise: PyPI, Burning Cost.*
