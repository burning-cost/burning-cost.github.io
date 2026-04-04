---
layout: post
title: "Your Fairness Metric Passes. Your High-Risk Customers Are Still Being Discriminated Against."
date: 2026-04-04
categories: [research, fairness, insurance-fairness]
tags: [demographic-parity, tail-parity, optimal-transport, wasserstein, fairness, fca-consumer-duty, equality-act, arXiv-2604.02017, le-denis-hebiri, insurance-fairness, pricing]
description: "Le, Denis and Hebiri (arXiv:2604.02017, April 2026) prove that mean demographic parity is not tail parity. A model can satisfy full-distribution DP whilst concentrating discriminatory outcomes precisely at the high-premium end — exactly where Equality Act indirect discrimination risk is most acute. Their optimal transport approach offers a theoretically sounder remedy than correcting the entire distribution."
author: Burning Cost
math: true
---

We have been writing about the gap between mean demographic parity and where regulatory risk actually sits. [Mean parity](/2026/04/02/localized-demographic-parity-pricing-tiers-insurance-fairness-v080/) is Level 1 — necessary, not sufficient, and what most fairness reporting currently measures. Tier-boundary parity is Level 2 — where a portfolio-average ratio can mask systematically different tier placement across groups. This post is about Level 3: the tail.

Le, Denis and Hebiri (arXiv:2604.02017, April 2026) make a deceptively simple observation: enforcing demographic parity over the full predicted distribution is both accuracy-costly and, in many cases, unnecessary. Fairness complaints — legal, regulatory, and practical — cluster at the high end of the premium distribution. A protected group customer paying a mid-market premium is unlikely to bring an indirect discrimination claim. A protected group customer in the top decile paying materially more than an equivalent non-protected-group peer has the basis for one.

The paper formalises this intuition with optimal transport theory, proves risk bounds and fairness guarantees, and produces an interpretable algorithm. There are no insurance experiments in the paper. But the theoretical contribution is one of the clearest arguments we have seen for a targeted — rather than blanket — approach to DP enforcement in pricing.

---

## What full-distribution demographic parity actually does

Standard demographic parity for regression enforces:

$$P(\hat{Y} \leq t \mid A = 0) = P(\hat{Y} \leq t \mid A = 1) \quad \forall\, t$$

In words: the entire predicted distribution must be identical across groups defined by the sensitive attribute $A$. Every quantile of predicted premiums for group 0 must equal the corresponding quantile for group 1.

The standard approach to achieving this uses optimal transport: find a transport map that minimises the Wasserstein distance between group distributions while enforcing distributional equality. The EquiPy library (UQAM/SCOR, 2025) implements exactly this. Our `WassersteinCorrector` in insurance-fairness does the same.

The problem is that full-distribution DP correction applies a remapping everywhere — including at premium levels where discriminatory harm is not plausibly occurring and where accuracy is being sacrificed for no regulatory benefit.

Le et al. formalise the cost. Their setting: a regression model $\hat{\mu}(x)$ that estimates expected loss. The sensitive attribute $A \in \{0, 1\}$ may be a proxy (postcodes correlated with ethnicity, for instance). Full-distribution DP requires the entire predicted distribution to match across groups. The accuracy cost of this constraint — measured by mean squared error increase — applies uniformly across all risk levels, including the low-premium majority where the distributions may already overlap substantially.

The paper's argument: this is not the right constraint. It is too strong where fairness is not at issue and too blunt an instrument to address where it is.

---

## Tail demographic parity

The authors propose enforcing DP only over a tail region: predicted values above a threshold $\tau_\alpha$, where $\alpha$ is a tail probability parameter.

Define:

$$T_\alpha(A) = \{x : \hat{\mu}(x) > Q_\alpha(\hat{\mu} \mid A)\}$$

where $Q_\alpha(\hat{\mu} \mid A)$ is the $\alpha$-quantile of predicted premiums within group $A$. Tail parity then requires:

$$P(\hat{Y} \leq t \mid A = 0,\, \hat{Y} > \tau_\alpha) = P(\hat{Y} \leq t \mid A = 1,\, \hat{Y} > \tau_\alpha) \quad \forall\, t > \tau_\alpha$$

The distributions in the tail must match. Below $\tau_\alpha$, the correction is not applied. The model's original predictions are preserved for low- and mid-premium policyholders.

The correction itself is constructed via optimal transport on the tail sub-distribution. The geometric structure of OT means the resulting algorithm is interpretable: premiums in the tail are remapped monotonically, and the correction can be expressed as a Wasserstein-minimal adjustment to the tail of each group's predicted distribution.

The paper proves:
1. The tail-corrected predictor satisfies the tail DP constraint exactly (under the empirical distributions used for calibration).
2. Risk bounds: the excess prediction error introduced by the correction is bounded as a function of $\alpha$ and the sample sizes used to estimate the tail distributions. The bound degrades as $\alpha \to 1$ (very thin tails, few calibration points), which is an honest and important limitation.
3. The Wasserstein distance between the corrected distributions is minimised subject to the fairness constraint — so no accuracy is sacrificed beyond the minimum required to achieve parity.

The algorithm: estimate group-specific tail distributions from held-out data; compute the OT map between them; apply the map to in-scope predictions. The OT computation on the tail is cheaper than full-distribution OT because it operates on a subset of the data, but it is still non-trivial at production scale with large protected-group sub-samples.

---

## Why this maps directly onto the UK regulatory concern

The FCA and Equality Act 2010 indirect discrimination test ask whether a provision, criterion or practice puts persons sharing a protected characteristic at a particular disadvantage. The operative word is *disadvantage*. For insurance pricing, disadvantage has a specific shape: it is concentrated in the upper tail.

A female driver who pays a premium in line with her actuarial risk estimate, within a competitive range of what a male driver with the same profile pays — she is not disadvantaged in any meaningful legal or regulatory sense. A female driver who pays materially more than an identical male driver because a GBM has learned to exploit a proxy correlated with gender, at the high-risk end of the book where premiums are largest and cross-subsidies most impactful — that is a concrete, actionable disadvantage.

The FCA's Consumer Duty Outcome 4 (fair value for all groups) and the EP25/2 proxy discrimination guidance (July 2025) are both most legally exposed at exactly this point. A regulatory investigation triggered by a complaint about discriminatory pricing will not, in practice, centre on whether mid-market premiums are equally distributed across gender groups. It will examine whether the most expensive policies disproportionately fall on one group.

Full-distribution DP addresses this concern, but it does so by correcting parts of the distribution that are not under regulatory scrutiny. It imposes an accuracy cost on low-premium policyholders in order to achieve a fairness property that nobody is actually contesting at that premium level. The accuracy cost is real: GBM predictions that are optimally calibrated for the bulk of the book will be remapped even where they are already fair.

Tail DP is a Pareto improvement in principle. It concentrates the correction where the regulatory risk sits and leaves the rest of the distribution alone. The accuracy sacrifice is smaller. The fairness protection where it matters is the same.

---

## The three-level DP framework

With this paper, we now have a complete theoretical foundation for three distinct levels of demographic parity enforcement in insurance pricing:

**Level 1: Mean parity.** $E[\hat{Y} \mid A = 0] = E[\hat{Y} \mid A = 1]$. The portfolio average is the same across groups. This is what most fairness reporting currently measures. It is necessary but not sufficient: a model can satisfy mean parity while concentrating disparity in tiers or in the tail.

**Level 2: Tier-boundary parity.** The distribution of predicted premiums across pricing tiers is similar across groups. `LocalizedParityCorrector` (insurance-fairness v0.8.0, from arXiv:2603.25224) addresses this layer. It catches cases where mean parity masks systematically different tier placement across groups.

**Level 3: Tail parity.** The premium distributions in the upper tail match across groups. Le et al. (arXiv:2604.02017) provide the theoretical basis for this layer. No implementation exists in insurance-fairness yet — we flag this as an explicit gap.

A governance evidence pack that demonstrates all three layers is materially stronger than one demonstrating mean parity alone. The Equality Act s.19 test and Consumer Duty Outcome 4 are both more naturally expressed at Levels 2 and 3 than Level 1.

---

## What is not in the paper

The theoretical results are clean. The practical limits are worth being explicit about.

**No insurance experiments.** Le et al. mention insurance and credit in their motivation. Their empirical work does not include an insurance dataset. The numerical results — on the risk bound tightness as $\alpha$ varies, on the OT computation — are generic. Whether the risk bounds are tight enough to be usable for UK motor or home portfolios with typical protected-group sub-sample sizes is unknown. Our expectation, based on the calibration sample requirements we see with `WassersteinCorrector`, is that thin tail estimates on minority sub-groups will be the binding constraint. Confidence intervals on the tail OT map will need to be part of any production implementation.

**No exposure weighting.** The paper treats all observations equally. Insurance predictions have varying exposure periods; a policy with 0.25 years of exposure should not carry the same weight as an annual policy in any parity calculation. Any implementation will need to extend the tail distribution estimation to handle exposure-weighted CDFs.

**No monotonicity constraint.** The OT remapping in the tail does not guarantee that the corrected predictions preserve monotonicity in underlying risk factors. A high-mileage driver whose predicted premium is remapped downwards to achieve parity could end up with a premium below a lower-mileage driver in the same risk segment. Monotonicity constraints on the OT map are not discussed. This is relevant for insurance where rating factor monotonicity is both actuarially necessary and sometimes a regulatory expectation.

**OT at scale is non-trivial.** The computational cost of OT on the tail sub-distribution scales with sample size in a way that Wasserstein-based corrections on quantile-discretised distributions avoid. For a UK motor portfolio with 500,000 policies, tail OT on raw premium predictions is materially more expensive than the parametric `SequentialOTCorrector` approach currently in insurance-fairness. The paper's algorithm is correct; whether it is fast enough for production refresh cycles is a separate question.

---

## What this means for insurance-fairness

Current state in insurance-fairness:

- `WassersteinCorrector`: full-distribution OT correction. Implements mean and distributional DP across one or more sensitive attributes. Accurate, expensive, over-corrects in the bulk.
- `SequentialOTCorrector`: sequential Wasserstein correction for multiple attributes. More efficient than joint OT for multi-attribute settings.
- `LindholmCorrector`: DP correction via regression residualisation (from Lindholm, Richman, Tsanakas, Wüthrich, 2024).
- `DiscriminationFreePrice`: orthogonal projection correction, fast, less theoretically tight than OT.
- `LocalizedParityCorrector`: tier-boundary parity (new in v0.8.0).

What is missing: a `TailParityCorrector` implementing the Le et al. framework — tail-only OT correction with exposure weighting, monotonicity enforcement, and confidence intervals on the tail map. We intend to build this. The practical design questions are: how to set $\alpha$ (the tail probability cutoff) for a given regulatory context; how to handle instability in minority-group tail estimates; and whether to implement monotonicity as a post-correction step or as a constraint on the OT problem itself.

The tail probability $\alpha$ is not purely a statistical choice. Setting $\alpha = 0.90$ means correcting the top decile of premiums. Setting $\alpha = 0.75$ means correcting the top quartile. The right value depends on where the portfolio's regulatory exposure actually sits — which requires looking at the complaint and claims experience, not just the premium distribution.

---

## Our assessment

Le et al. (arXiv:2604.02017) make one genuinely useful theoretical contribution: they prove that full-distribution demographic parity is not necessary for regulatory compliance, that the relevant constraint is a tail constraint, and that this can be formalised and enforced with the same OT machinery that full-distribution DP uses.

We think they are right on the main argument. Blanket full-distribution DP correction is the wrong target. It wastes accuracy in the bulk and does not specifically protect where protection is needed. The UK regulatory framework — Equality Act s.19, Consumer Duty Outcome 4, FCA EP25/2 proxy guidance — is most naturally read as requiring tail-level protection, not distributional identity across the whole predicted range.

The gap between the paper and a production UK insurance implementation is real. No insurance experiments, no exposure weighting, no monotonicity, OT scaling questions unresolved. These are solvable problems, not theoretical objections. We will report back when we have built it.

For now: if your fairness governance evidence pack only demonstrates mean DP, you have addressed the easiest layer. The Equality Act and Consumer Duty obligations sit higher up the distribution. Check the tail.

---

## Reference

Le, N.S., Denis, C. and Hebiri, M. (2026). Demographic Parity Tails for Regression. arXiv:2604.02017. Submitted April 2, 2026.

Related on Burning Cost:

- [Your GLM Passes Mean Fairness. Does It Pass the Tier Test?](/2026/04/02/localized-demographic-parity-pricing-tiers-insurance-fairness-v080/) — LocalizedParityCorrector and the tier-boundary layer
- [The Fairness Impossibility You Cannot Optimise Away](/2026/04/04/nsga2-multi-objective-fairness-pareto-boonen-2512-24747/) — multi-objective trade-offs and why "passing fairness checks" is not enough
- [FCA EP25/2: Proxy Discrimination Guidance for Insurance Pricing](/2026/03/29/fca-ep25-2-proxy-discrimination-insurance-pricing/) — regulatory context for tail-level fairness
