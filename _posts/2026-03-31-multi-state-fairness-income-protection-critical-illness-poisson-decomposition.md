---
layout: post
title: "Multi-State Fairness in Income Protection and Critical Illness: Why Lindholm on the Aggregate Premium Is Wrong"
date: 2026-03-31
categories: [fairness, life-insurance]
tags: [fairness, income-protection, critical-illness, long-term-care, multi-state, poisson-glm, Lindholm, Lim-Xu-Zhou, arXiv-2602-04791, Kolmogorov, optimal-transport, adversarial-debiasing, age-conditional, FCA-Consumer-Duty, Equality-Act-2010, insurance-fairness, UK-personal-lines, discrimination-free]
description: "Lim, Xu & Zhou (arXiv:2602.04791) show that multi-state models decompose into independent Poisson GLMs — one per transition — making every single-period fairness method directly applicable. But the right place to apply it is at the transition rate, not the aggregate premium. We explain why, and what it means for UK income protection and critical illness pricing."
math: true
author: burning-cost
---

We [wrote about arXiv:2602.04791 in March](https://burning-cost.github.io/fairness/regulation/life-insurance/2026/03/25/fair-pricing-long-term-insurance/) when the paper by Lim, Xu, and Zhou first surfaced. That post made the high-level argument: multi-state products have a fairness problem that annual GI tools cannot address, and this paper provides a framework. We want to go deeper now, because the implementation detail matters — and the naive approach (apply Lindholm to the final premium) is wrong in a specific, fixable way.

---

## The Poisson equivalence and what it enables

The starting point is a result that every UK actuarial team working on income protection or critical illness already uses implicitly, but may not have thought about in these terms.

In a multi-state model — the standard actuarial workhorse for income protection, long-term care, and critical illness — you estimate a separate transition intensity for each possible state change. For a four-state income protection model (Healthy, Sick, Recovered, Dead), you have six possible transitions: H→S, H→D, S→R, S→D, R→H, R→D. Each has its own rate $\lambda_m$ that varies with covariates and time.

The Poisson equivalence says: over a short interval during which the rate $\lambda_m$ is approximately constant, the count of transition events $T_{k,m}$ for individual $k$ follows:

$$T_{k,m} \sim \text{Poisson}(\tau_{k,m} \cdot \lambda_m(z_k, x_k, s_k))$$

where $\tau_{k,m}$ is the exposure (time at risk of transition $m$), $z_k$ are legitimate rating factors, $x_k$ is issue age, and $s_k$ is the sensitive attribute. In continuous time, this is exact. In the discretised panel data setting that any practical implementation requires, it is an approximation that becomes exact as the interval width shrinks.

What this means: each transition becomes a Poisson GLM with a log offset for exposure. Transition H→S is one regression problem. Transition S→R is another. They are independent. You fit them separately. And because they are Poisson GLMs, every tool built for single-period fairness now applies to each one directly — without modification.

This is the contribution of Lim, Xu, and Zhou (CUHK/UNSW Sydney/Waterloo, February 2026). It is elegant precisely because it does not require new algorithms. The multi-state fairness problem reduces to $M$ instances of a problem we already know how to solve.

---

## Where the naive approach breaks down

Here is the obvious but wrong implementation: fit your multi-state model as normal, compute the final premium $P$ using Kolmogorov forward equations, then apply Lindholm marginalisation to $P$ directly.

Lindholm marginalisation on the final premium would give:

$$P^*(z, x) = \int P(z, x, s) \, dP_n(s)$$

integrating the premium over the empirical distribution of the sensitive attribute $S$ conditional on the legitimate factors. This is the approach that works for annual GI products.

For multi-state products it is the wrong object to integrate. The premium $P$ is a nonlinear function of the transition rates $\lambda_1, \ldots, \lambda_M$ via the matrix exponential in the Kolmogorov equations. Applying marginalisation at the premium level does not guarantee that the underlying transition rates are discrimination-free — and there is no principled guarantee that the resulting premium satisfies the discrimination-free criterion at each transition, or that the correction is consistent across transitions with different exposure to the sensitive attribute.

The correct procedure is:

1. Compute the discrimination-free transition rate for each transition $m$ separately:

$$\lambda^*_m(z, x) = \int \hat{\lambda}_m(z, x, s) \, dP_n(s)$$

2. Use the corrected rates $\lambda^*_1, \ldots, \lambda^*_M$ to recompute the premium via Kolmogorov forward equations.

Step 2 gives you a premium that is guaranteed to be discrimination-free because it is computed from discrimination-free primitives. The marginalisation-at-premium-level shortcut gives you no such guarantee — the nonlinearity of the matrix exponential means the commutation does not hold.

This distinction matters in practice. Occupation class is the dominant rating factor for UK income protection and it is a well-documented socioeconomic and ethnicity proxy. The inception rate (H→S) and the recovery rate (S→R) have very different occupation class effects — and the sensitive attribute's influence on those two rates may partially cancel in the final premium while remaining fully present at the transition level. Correcting only the premium would leave proxy discrimination embedded in the transition structure.

---

## Three methods, three very different novelty levels

Lim, Xu, and Zhou cover three approaches for applying fairness corrections to multi-state models. They are not equally novel.

**Post-processing (per-transition Lindholm marginalisation)** is formally identical to applying Lindholm (2022) to each transition GLM separately. There is nothing novel in the algorithm itself — the contribution is establishing that the Poisson equivalence makes this valid and showing that the resulting premium inherits the discrimination-free property. For teams already using `LindholmCorrector` from `insurance-fairness`, the extension to multi-state products is operationally trivial: fit each transition model, run `LindholmCorrector` on each one's output, recompute the premium.

**In-processing (adversarial debiasing with shared encoder)** has moderate novelty. A shared feature encoder $W = f(Z)$ is trained to be independent of $S$ (via adversarial debiasing), and separate regression heads $g_m(W, x)$ are fitted per transition. The shared encoder constrains the representations to be free of sensitive information before any transition model sees them. This is conceptually clean — one fairness constraint, $M$ regression tasks — but requires a neural architecture that not all production environments support.

**Pre-processing (age-conditional optimal transport)** is where the genuine methodological novelty sits. Age is special in multi-state insurance models: it enters both the risk estimation (older policyholders have higher transition rates) and the premium calculation (the premium for a 45-year-old is not the same as for a 35-year-old, by design). If you apply standard OT covariate transformation to remove sensitive information from $Z$ unconditionally, you will partially scramble the age signal along with it.

The solution is age-conditional OT: compute a separate transport map $T_{x^*}: Z \to Z^{\perp_S}$ for each issue age $x^*$, enforcing independence from $S$ conditional on $X = x^*$. The resulting transformed features $Z^{\perp_S}$ are free of sensitive information within each age band, while age itself remains as a legitimate rating variable in both the transition models and the premium calculation. Each transport map is fitted using the age-conditional empirical distributions.

This is the only piece of the paper that does not reduce to a trivial application of an existing method. If you want pre-processing fairness for UK income protection, you need age-conditional OT — standard OT is the wrong tool.

---

## What this looks like in practice for a UK IP book

A standard UK income protection model uses the CMI's four-state framework: Healthy, Sick/Claiming, Recovered, Dead. Six transitions. The dominant covariates are age, occupation class, deferred period, and benefit amount.

Occupation class (typically Class 1–4 in UK IP, or the more granular SOC-based classification) correlates materially with socioeconomic status, and via that route with ethnicity. This is the proxy discrimination exposure.

Under the Lim/Xu/Zhou framework, the audit and correction process for each transition is:

1. Structure the panel data with one row per person-period at risk of the relevant transition, with the exposure as a time offset.
2. Fit the Poisson GLM: `glm(events ~ age + occ_class + deferred_period + offset(log(exposure)), family=poisson)` (or your GLM of choice).
3. Run `LindholmCorrector` on the fitted rates, marginalising over the empirical distribution of the sensitive proxy variable (e.g., a socioeconomic deprivation quintile derived from MOSAIC/Acorn, since direct ethnicity data is rarely held by UK insurers).
4. Collect the six corrected rate functions.
5. Recompute expected present values via Kolmogorov equations using corrected rates.

Step 3 is where the existing `insurance-fairness` library applies without modification. Steps 1, 2, 4, and 5 are standard actuarial work — the multi-state modelling infrastructure your pricing team already uses.

The audit output is six sets of Lindholm plots (predicted rate vs corrected rate by sensitive proxy group), one per transition. This is the evidence base for a Consumer Duty fair value assessment that addresses proxy discrimination in transition modelling.

---

## The income protection and critical illness angle specifically

The FCA's interim report on the pure protection market study (MS24/1, January 2026) raised specific concerns about IP claims ratios running around 40% of premiums — below what a fair value outcome would support — and about whether pricing trajectories are systematically worse for certain customer groups. The review is ongoing; the final report is expected Q3 2026.

The direction of travel is clear. Demonstrating discrimination-free pricing on multi-state products is going to become part of the Consumer Duty fair value attestation process. UK insurers who have only applied Lindholm-style corrections to their annual GI portfolios have a gap for their IP and critical illness products.

Critical illness pricing adds a further complication: the product typically pays on first diagnosis of a defined condition (cancer, heart attack, stroke). This can be modelled as a two-state model (Healthy, Claimed/Dead) with a single inception rate, which makes it closer to a single-period problem and therefore simpler to address. But more sophisticated CI models track recovery and recurrence, in which case the multi-state framework is directly applicable.

The paper uses HRS (Health and Retirement Study) data from the US — an elderly US population, which limits direct comparability with UK working-age IP. The methodology transfers; the calibrated rates do not. UK practitioners wanting to apply this need CMI (Continuous Mortality Investigation) data, ELSA (English Longitudinal Study of Ageing) as a proxy, or their own panel claims data with sufficient duration.

---

## What we have not implemented

There is no `insurance-fair-longterm` library yet. The framework would require roughly 1,500 lines of multi-state infrastructure — panel data handling, Kolmogorov equation solver, per-transition wrapper for `LindholmCorrector`, age-conditional OT transport maps — that we have not built because no existing library was the right base.

The post-processing path (apply `LindholmCorrector` per transition, recompute premium) is implementable today using existing tools. The pre-processing age-conditional OT path is not — it would need either a new module in `insurance-fairness` or a standalone implementation. We are tracking this as a gap; it is medium-priority given that the post-processing path covers the main regulatory requirement.

If you are working on IP or critical illness fairness and want to implement this now, the post-processing route is the pragmatic choice. The theoretical justification from Lim/Xu/Zhou is solid; the operational steps are straightforward if your team already has multi-state models fitted.

---

## Related posts

- [Fair Pricing in Long-Term Insurance: What arXiv:2602.04791 Means for UK Actuaries](https://burning-cost.github.io/fairness/regulation/life-insurance/2026/03/25/fair-pricing-long-term-insurance/) — the broader argument for why long-term fairness is structurally different
- [Claims Lifecycle Modelling: The Python Gap and How to Bridge It with Poisson GLMs](https://burning-cost.github.io/techniques/reserving/pricing/2026/03/26/multi-state-claims-lifecycle-poisson-glm-substitution/) — the Poisson GLM substitution for multi-state models in Python
- [Sequential Optimal Transport for Multi-Attribute Fairness in Insurance Pricing](https://burning-cost.github.io/fairness/machine-learning/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — the Hu-Ratz-Charpentier algorithm for multi-attribute OT correction, which applies at each transition after Lindholm
