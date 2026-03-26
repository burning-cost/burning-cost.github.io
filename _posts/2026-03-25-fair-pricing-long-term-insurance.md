---
layout: post
title: "Fair Pricing in Long-Term Insurance: What arXiv:2602.04791 Means for UK Actuaries"
date: 2026-03-25
categories: [fairness, regulation, life-insurance]
tags: [fairness, life-insurance, income-protection, pmi, long-term, multi-state, poisson, fca, consumer-duty, pure-protection, insurance-fair-longterm, insurance-fairness, arXiv, uk-personal-lines]
description: "A February 2026 paper from Lim, Xu, and Zhou cracks open the problem of fair pricing in multi-state insurance products — the ones that matter most for Consumer Duty obligations on life, IP, and PMI."
---

There is a structural problem with fairness in long-term insurance that nobody in general insurance ever had to think about, and that the actuarial profession has mostly ignored. Lim, Xu, and Zhou's February 2026 paper ([arXiv:2602.04791](https://arxiv.org/abs/2602.04791)) is the first serious attempt to fix it. For UK pricing teams working on income protection, private medical insurance, or any protection product with multi-year pricing dynamics, this paper is directly relevant now — not in five years when it filters into guidance.

---

## Why long-term insurance is a different fairness problem

When a pricing actuary at a motor insurer runs a proxy discrimination audit, the question is straightforward: does my model treat protected groups fairly at the point of sale? The product is annual. The premium is set once. The fairness obligation applies to that single pricing decision.

Long-term protection products do not work like this, and the difference is not cosmetic.

An income protection policy priced at age 35 might pay out at 52 if the insured becomes chronically ill. The premium is either locked in at outset or changes according to contractual terms agreed years earlier. The insurer's liability is a function not of a single risk score but of a chain of transition probabilities — the likelihood of moving from healthy to claiming, from claiming to recovered, from recovered to dead — compounded across two decades. A biased inception rate model does not produce a biased quote this year. It produces a distorted liability profile across the entire book, with cross-subsidies running in directions that are not visible at point of sale.

Three features make this genuinely harder than GI fairness:

**Adverse selection operates over time, not at inception.** In annual GI, adverse selection is an underwriting problem at renewal. In long-term IP, a biased model might systematically underprice inception risk for certain occupational groups, leading to claims experience that deteriorates over the benefit period in ways the model never flagged. The discrimination is baked in at inception but only visible in aggregate after years of claims.

**Premium lock-in removes the market correction mechanism.** Annual GI products are self-correcting to some extent: if you misprice a segment, they renew elsewhere. A 10-year IP policy with locked premiums offers no such correction. The consequences of a biased transition rate model are permanent for the duration of the contract.

**Cross-subsidisation is structural and intentional — but it can mask discrimination.** Pooling across age, sex (to the extent permitted), and health status is how protection insurance works. The question is whether the cross-subsidisation reflects legitimate risk pooling or protected-characteristic discrimination embedded in the transition model. These are genuinely hard to disentangle, and existing fairness tools offer no help because they assume a single-period regression problem.

---

## What Lim, Xu, and Zhou actually did

The paper's contribution is an elegant reframing rather than an entirely new algorithm. The key insight is that multi-state transition models — the actuarial workhorse for IP, long-term care, and similar products — can be reformulated as a collection of Poisson regression problems.

In a multi-state model, you estimate separate rates for each possible transition: healthy-to-sick, sick-to-recovered, sick-to-dead, and so on. Each rate has its own covariates and its own model. The standard actuarial approach is to fit each transition model independently using survival analysis, then chain the rates together using Kolmogorov forward equations to obtain the expected present value of future benefits — which is your premium.

The Poisson reformulation works because, over short intervals, the number of transitions from state A to state B follows a Poisson distribution with a rate that depends on time in state A and whatever covariates you include. If you restructure your data to have one row per person-period at risk of each transition, with an exposure offset for the time at risk, you get a standard Poisson GLM. And Poisson GLMs are regression models — exactly the class of models for which every existing fairness method was designed.

This means every tool in the cross-sectional fairness toolkit now applies, by construction, to long-term transition models:

- **Post-processing (discrimination-free pricing):** Marginalise transition rates over the distribution of protected characteristics, as Lindholm et al. (2022) showed for single-period GLMs. Apply to each transition rate separately, then chain through Kolmogorov equations. The marginalisation formula is identical in structure: the fair price is the expectation of the risk premium across the reference distribution of the protected characteristic, conditional on all legitimate rating factors.
- **Pre-processing (covariate transformation):** Apply optimal transport reweighting to the training data before fitting any transition model. This removes protected-characteristic information from the input distribution.
- **In-processing (adversarial or constrained training):** Impose fairness constraints directly during the Poisson regression fitting, using adversarial debiasing or fairness-penalised loss functions.

The authors demonstrate the framework on data from the University of Michigan Health and Retirement Study — long-term care transitions in an ageing US population — using the post-processing route. The demonstration is illustrative rather than a production-scale benchmark, but it is enough to establish that the mathematics works and that the approach is tractable.

---

## The Consumer Duty angle

The FCA's Consumer Duty [Principle 12 and PRIN 2A] requires firms to deliver good outcomes to retail customers, including fair value. For pure protection products, the FCA's interim report on MS24/1 (published 29 January 2026) flagged specific concerns about income protection claims ratios running at approximately 40% — well below what fair value assessments should tolerate — and about premium trajectories that leave customers worse off over the life of the contract than a rational expectations analysis would suggest.

[We wrote about the MS24/1 implications for pricing teams here.](/2026/03/25/fca-pure-protection-market-study-what-pricing-teams-should-prepare-for/)

The connection to arXiv:2602.04791 is direct. If your IP or PMI pricing uses multi-state transition models — which it should, because it is the actuarially correct approach — then the fairness of those models is the fairness of your long-term pricing trajectory. Consumer Duty fair value assessments that document the technical model without examining its fairness properties across protected groups are, in our view, inadequate. The FCA has not said this explicitly yet, but the trajectory of Consumer Duty enforcement and the direction of the pure protection market study make it predictable.

The paper gives UK actuaries a principled method to make that examination. The Poisson reformulation is not a research curiosity — it is an audit framework. For each transition in your state machine, you can now ask: is this rate model proxy-discriminatory with respect to ethnicity, socioeconomic status, or disability? And you can correct for that discrimination using the same marginalisation approach that applies to GI pricing.

---

## What this looks like for UK IP pricing teams specifically

The CMI framework for income protection uses a four-state model: Healthy, Sick/Claiming, Recovered, Dead. There are six transitions:

1. Healthy → Sick (inception rate — the most commercially significant)
2. Sick → Recovered (recovery/termination rate)
3. Sick → Dead (mortality while claiming)
4. Healthy → Dead (mortality while healthy)
5. Recovered → Sick (relapse — critical for long-duration policies)
6. Recovered → Dead

Occupation class is the main risk factor in UK IP pricing, and it is a well-known proxy for socioeconomic status and, to some degree, ethnicity. Using it as a rating factor is not inherently discriminatory — there is a legitimate risk basis — but if the model is using occupation class to pick up socioeconomic or ethnic variation beyond the direct occupational hazard, that is an indirect discrimination risk under Section 19 of the Equality Act 2010.

The Lim et al. framework lets you address this systematically. For each of the six transitions, you fit a Poisson GLM with occupation class, age, benefit period, and deferred period as covariates. You run a proxy discrimination diagnostic on each model separately — is occupation class proxying for ethnicity in the inception rate? In the recovery rate? The pattern may differ across transitions, which matters both for the correction method and for the regulatory narrative you build.

For PMI pricing, the state machine is simpler (Active, Claiming, Lapsed) but the fairness concern is analogous: pre-existing condition loadings and area-based rating factors correlate with protected characteristics, and a multi-year PMI product compounds those biases across renewal cycles.

---

## Where existing tools connect

Our [insurance-fairness](https://github.com/burningcost/insurance-fairness) library already implements the discrimination-free pricing approach for single-period GLMs — specifically the Lindholm marginalisation method. The `DiscriminationFreeRates` class computes marginalised premiums by integrating over a reference distribution of protected characteristics.

The Lim et al. paper establishes that the same mathematical operation applies at the transition rate level in a multi-state model. The Poisson reformulation means you can use `DiscriminationFreeRates` on each transition model independently, then feed the adjusted rates into a standard actuarial Kolmogorov solver to recover the fair long-term premium. This is not a stretch of the existing functionality — it is the existing functionality applied at the right level of abstraction.

We are building a dedicated library, `insurance-fair-longterm`, that wires this pipeline together end-to-end: `MultiStateTransitionFitter` (Poisson GLM per transition with exposure offset), `FairnessAdjuster` (pre/in/post-processing adapters that call through to `insurance-fairness` logic), `LongTermPremiumCalculator` (Kolmogorov forward equations), and `FairnessAuditReport` (parity metrics per transition and per product). We will publish when it is ready.

In the meantime, if your team wants to run the proxy discrimination diagnostics on existing IP transition models, `IndirectDiscriminationAudit` from `insurance-fairness` works on any regression output — including Poisson GLMs for transition rates — without modification.

---

## The honest limitations

Three things the paper does not resolve, and where UK practitioners should be careful.

**The Equality Act test is not the same as the actuarial test.** Discrimination-free pricing in the Lindholm sense removes protected-characteristic information from the premium. But Section 19 of the Equality Act asks whether a provision, criterion, or practice puts a protected group at a particular disadvantage, and whether that disadvantage is proportionate to a legitimate aim. The actuarial correction and the legal test are related but not identical, and firms that run the Lim et al. adjustment and declare themselves compliant have skipped several steps.

**The framework handles observed heterogeneity, not unobserved correlation.** If your transition rate models do not include occupation class or area because you dropped them on fairness grounds, the Poisson reformulation cannot audit for their residual influence through other covariates. You need the proxy discrimination diagnostic — `D_proxy` in `insurance-fairness-diag` — before and after the adjustment to check that you have not simply redistributed the problem.

**Long-term care data is not UK IP data.** The Health and Retirement Study demonstration in the paper is US-based and covers an older population than typical IP buyers. The mathematics transfers; the parameter values do not. UK practitioners need to apply the framework to CMI graduated rates and UK-specific morbidity data, which is less publicly available and more commercially sensitive.

---

## What to do with this now

The paper has been available on arXiv since 4 February 2026. It will not be in the IFoA working party report until late 2027 at the earliest, if ever. If you are pricing IP or PMI and using a multi-state model, the gap between the paper being available and your team knowing about it is your commercial problem to fix, not the regulator's.

Concretely: run the proxy discrimination audit on your inception rate model this month using `IndirectDiscriminationAudit`. Document the result. If you find material indirect discrimination in the Healthy → Sick transition, that is a finding your model documentation needs to address before the FCA's final MS24/1 report arrives in Q3 2026 — not after.

The Lim et al. framework gives actuaries the mathematical tools. The obligation to use them is already here.

---

*arXiv:2602.04791 — 'Fair Pricing in Long-Term Insurance: A Unified Framework', Lim, Xu, and Zhou (February 2026). [Abstract on arXiv.](https://arxiv.org/abs/2602.04791)*
