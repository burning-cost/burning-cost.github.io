---
layout: post
title: "Fairness in the Tail: Why Mean Demographic Parity Is the Wrong Test"
date: 2026-04-04
categories: [fairness, insurance-fairness]
tags: [demographic-parity, tails, optimal-transport, fca-consumer-duty, equality-act, proxy-discrimination, arXiv-2604.02017, insurance-fairness, fair-value, FCA-EP25-2, wasserstein]
description: "Le, Denis and Hebiri (arXiv:2604.02017, April 2026) show that enforcing demographic parity over the full prediction distribution is both accuracy-costly and unnecessary. The right target is the tail. For UK insurers, this is where Consumer Duty exposure actually lives."
author: burning-cost
math: true
---

Your demographic parity test passes. Mean predicted premium for group A: £618. Mean predicted premium for group B: £621. Ratio: 1.005. Amber threshold is 0.80. You are well inside it. The Consumer Duty evidence pack gets its fairness metric.

Now look at the upper tail — the top 5% of the premium distribution, above £1,400. Group A has 3.8% of policies there. Group B has 9.1%. The disparity is not in the average; it is concentrated exactly where it causes the most harm.

This is not a contrived criticism. It is the central observation of Le, Denis and Hebiri ([arXiv:2604.02017](https://arxiv.org/abs/2604.02017), April 2026): mean demographic parity can be satisfied simultaneously with severe tail discrimination, because the averages can cancel even when the distributional shapes are very different. Their proposed fix — enforce parity only in the tails, not across the full distribution — turns out to be both more regulatorily relevant and less accuracy-costly than the current standard.

---

## Why full-distribution parity is an overcorrection

The standard demographic parity criterion in regression requires that the full conditional distributions of predictions are identical across groups:

$$F_{\hat{Y}|S=s}(t) = F_{\hat{Y}|S=s'}(t) \quad \text{for all } t \in \mathbb{R} \text{ and all groups } s, s'$$

This is a strong constraint. Satisfying it globally forces the entire prediction distribution — including the bulk of the book, where discriminatory harm is not contested — to be identical across groups. A motor book where young drivers in group A genuinely have higher accident rates than young drivers in group B will have different claim severity distributions. Constraining the full premium distribution to be identical sacrifices accuracy everywhere in order to achieve fairness everywhere, including in the region where differential pricing is actuarially justified.

The accuracy cost of global distributional DP is not negligible. The prior work by Chzhen et al. (2020) — whose optimal transport approach this paper extends — shows that the MSE penalty from full-distribution DP scales with the Wasserstein distance between the group-conditional prediction distributions. For books with genuinely different risk profiles across demographic groups, that penalty is not small.

The Le et al. argument is that most of this cost is unnecessary. Fairness concerns in regulated insurance pricing are concentrated at the extremes: high-premium policyholders facing renewal unaffordability, customers in the upper tail who pay materially more than a discrimination-free model would charge, and — at the other end — the question of whether low-risk customers in a protected group are being priced more expensively than they should be. The middle of the distribution, where competing risk effects are small and pricing differentials are modest, is not where discrimination claims originate.

---

## The DP-tails definition

Le et al. formalise this as follows. Fix a threshold $\alpha \in \mathbb{R}$ and an unfairness proportion $p \in [0,1]$. A regressor $g$ is $(\alpha, p)$-DP-tails fair if, for all groups $s, s'$:

$$F_{g|s}(t) = F_{g|s'}(t) \quad \text{for all } t \geq \alpha$$

and additionally $F_{g|s}(\alpha) = p$ for all $s$.

Reading this: above the threshold $\alpha$, the conditional CDFs of predictions must be identical across groups. Below $\alpha$, predictions are unconstrained. The second condition fixes the probability mass below the threshold to $p$ for every group — this is what makes the optimisation tractable and the OT connection explicit.

The full-distribution DP case is recovered as a special case by taking $\alpha \to -\infty$ and $p = 0$. Choosing $\alpha$ at the 90th percentile of predictions — so the top 10% of the premium distribution — gives a constraint that enforces distributional parity exactly where the regulator's attention is focused, while leaving the bulk of the book undisturbed.

---

## How the optimal transport correction works

The paper's central result is that finding the optimal $(\alpha, p)$-DP-tails fair predictor is equivalent to a Wasserstein barycenter problem, but restricted to the tail region.

Concretely: the solution $g^*_{\alpha,p}$ applies the quantile function of the Wasserstein barycenter distribution above $\alpha$, and leaves predictions unchanged below $\alpha$. In the tail, predictions for each group are remapped via a monotone quantile transport to a common target distribution. Below the threshold, the original predictions are preserved.

The data-driven estimator $\hat{g}^{\xi}_{\alpha,p}$ is constructed using:
1. A base predictor $\hat{f}$ trained on labelled data in the usual way.
2. An unlabelled calibration set used to estimate the conditional CDFs and quantile functions per group.
3. The tail correction applied as a post-processing step.

The key practical point: this is post-processing, not retraining. Any existing model — GBM, GLM, neural network — can have the tail correction applied on top, using only unlabelled calibration data. The base model does not need to change.

The fairness guarantee (Theorem 3.1 in the paper) shows that tail unfairness decays at rate $O(1/\sqrt{N})$ where $N$ is the calibration sample size — distribution-free, independent of the quality of the base predictor. The accuracy penalty relative to the unconstrained base model decreases as $\alpha$ increases (i.e., as the constrained region narrows).

---

## What the experiments actually show

The paper validates against three datasets: Law School GPA prediction (race as sensitive attribute), Communities and Crime (race-related features), and California Housing (geography as sensitive attribute). In all three cases:

- The tail-corrected predictor achieves lower MSE than the full-distribution OT corrector, at the cost of higher global Kolmogorov–Smirnov distance (expected — it is only correcting the tail).
- The reduction in tail unfairness $U_\alpha(g)$ matches the theoretical rate.
- The optimised variant $\hat{g}^\xi_\alpha$ — which selects $p$ to minimise the accuracy cost — consistently outperforms fixed-$p$ approaches.

Specifically: as the threshold $\alpha$ moves from left to right (constraining an increasingly narrow tail region), MSE falls toward the unconstrained level and KS distance rises toward the unconstrained level, with the tail unfairness measure $U_\alpha$ falling toward zero throughout. This is the empirical confirmation that DP-tails is a genuine Pareto improvement: you can trade global unfairness for accuracy without sacrificing local fairness where it matters.

---

## Why this is the right abstraction for UK insurance

The FCA's EP25/2 guidance on proxy discrimination (published July 2025) makes the regulatory focus explicit: outcomes in the upper tail of the premium distribution — high-premium segments, renewal cliff edges — are the scenarios most likely to constitute actual consumer harm. A customer paying £1,900 for motor insurance who faces a premium that is materially inflated by demographic correlation is a qualitatively different situation from a customer paying £700 facing the same percentage differential.

The Equality Act 2010, section 19 indirect discrimination test asks whether a provision, criterion or practice puts people sharing a protected characteristic at a *particular disadvantage*. The word particular matters. The legal test is about concrete disadvantage, not statistical mean shifts. High premiums with no actuarial justification are concrete. Small mean-level discrepancies are much harder to argue as disadvantage.

Consumer Duty (PRIN 2A.3) requires firms to demonstrate good outcomes across groups sharing protected characteristics. The outcomes that will attract regulatory attention — and that could support an indirect discrimination claim — are concentrated at the top of the premium distribution.

A firm that demonstrates mean parity has satisfied the most basic diagnostic. It has not demonstrated that the outcomes in its upper premium tail are fair. These are different things, and Le et al. give the statistical framework to distinguish them.

---

## The three-level fairness hierarchy this creates

We now have three levels of demographic parity evidence a UK pricing team should be able to produce:

**Level 1 — Mean parity.** $\mathbb{E}[\hat{Y} | S=0] \approx \mathbb{E}[\hat{Y} | S=1]$. The standard portfolio-level check. Necessary but not sufficient. An adversarial model can satisfy this while concentrating discriminatory premiums at the top end.

**Level 2 — Tier-boundary parity** (Charpentier et al., arXiv:2603.25224): enforce equal probability mass at pre-specified price tier boundaries — the governance bands you already report against. This is the right tool when you have a defined rate-band structure and want to audit parity at those bands.

**Level 3 — Tail parity** (Le et al., this paper): enforce distributional parity above a quantile threshold, where discrimination risk is highest. Does not require pre-specifying bands; targets the continuous upper tail above $\alpha$.

These are complements. Level 1 is the claim that the averages are broadly similar. Level 2 is the claim that no single governance tier has a disproportionate demographic composition. Level 3 is the claim that the high-premium end of the distribution is demographically balanced.

A Consumer Duty evidence pack that can speak to all three levels is substantially stronger than one that only addresses Level 1.

---

## What `insurance-fairness` does today and what is missing

The library's `demographic_parity_ratio` in `metrics.py` is a Level 1 check — exposure-weighted group means with bootstrap confidence intervals. `LocalizedParityCorrector` (v0.8.0) implements Level 2, following the Charpentier et al. framework.

Level 3 is not yet in the library. The Le et al. correction would live in a `TailParityAudit` class: accepts predictions, a sensitive attribute vector, and a quantile threshold $\alpha$; returns the tail unfairness statistic $U_\alpha(g)$ and the OT correction maps for each group. Implementation would follow the same post-processing architecture as `LocalizedParityCorrector`.

Several gaps need addressing before this is production-ready for UK personal lines:

**Exposure weighting.** The paper's OT construction assumes equal-weight observations. Insurance pricing models are not equal-weight — earned exposure must enter the calibration. A tail correction estimated on unweighted policy counts will be miscalibrated for any frequency model that uses earned car years as an offset.

**Continuous sensitive attributes.** UK proxy discrimination analysis typically uses continuous proxies — postcode-level deprivation indices, ONS ethnicity proportions at LSOA level — rather than binary group indicators. The paper addresses categorical $S$. Extension is not treated and is non-trivial.

**Sample size in the minority group tail.** The $O(1/\sqrt{N})$ fairness guarantee depends on $N$ being the size of the calibration set within each group. For a UK motor book with 60,000 policies and a protected group at 15% prevalence, the tail (top 10%) of the minority group has around 900 policies. Feasible, but small standard errors require careful handling.

**Monotonicity.** The within-group OT map is monotone (it is a quantile transform), so within-group risk ordering is preserved. Cross-group ordering is not guaranteed. In a Solvency II internal model context, you need a worked example confirming that actuarial risk ordering is not disturbed by the correction before deploying this in a validated model.

---

## Our view

The conceptual case is correct and important. Mean demographic parity is not an adequate fairness audit for UK insurance, and Le et al. provide the formal framework for saying precisely what a better audit looks like.

The theoretical machinery is sound: the Wasserstein barycenter connection is clean, the convergence results are in the expected form, and the post-processing architecture is well-suited to insurance deployment (you do not need to retrain your GBM). The empirical results across three datasets confirm the MSE-fairness tradeoff behaves as the theory predicts.

What this is not: a drop-in solution for a UK personal lines book this quarter. The missing exposure weighting is fundamental, not peripheral. Until there is a worked example on an insurance dataset — freMTPL2freq or AusPrivauto being the natural candidates — the gap between the regression setting in the paper and a production frequency-severity GLM stack remains for the practitioner to bridge.

We have `TailParityAudit` on the `insurance-fairness` roadmap, with exposure weighting and an insurance-specific worked example. We will cover it here when it ships.

For now: read the paper. Understand that your current fairness audit is a Level 1 test. Add tail parity monitoring to your Consumer Duty checklist. The FCA's attention is on the upper end of the premium distribution — that is where EP25/2 focuses, that is where section 19 claims originate, and that is where the Le et al. framework is designed to give you rigorous evidence.

---

## Reference

Le, N.S., Denis, C. and Hebiri, M. (2026). Demographic Parity Tails for Regression. arXiv:2604.02017. Submitted 2 April 2026.

---

Related on Burning Cost:

- [Your GLM Passes Mean Fairness. Does It Pass the Tier Test?](/2026/04/02/localized-demographic-parity-pricing-tiers-insurance-fairness-v080/)
- [Detection vs Mitigation: insurance-fairness and EquiPy Are Not Competing](/blog/2026/04/04/insurance-fairness-equipy-comparison/) — the EquiPy tail correction that implements exactly the kind of post-processing this paper calls for — Charpentier et al. tier-boundary parity, `insurance-fairness` v0.8.0
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — the LRTW framework and `FairnessAudit`
- [Per-Policyholder Proxy Discrimination Scores](/2026/04/03/sensitivity-proxy-discrimination/) — instance-level PD scores for Consumer Duty evidence
