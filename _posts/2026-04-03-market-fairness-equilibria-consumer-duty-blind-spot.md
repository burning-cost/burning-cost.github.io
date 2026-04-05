---
layout: post
title: "Consumer Duty Audits One Firm. This Paper Asks What Happens to the Market."
date: 2026-04-03
categories: [fairness, regulation]
tags: [Consumer-Duty, FCA, EP25-2, market-equilibrium, collusion-pathology, fairness-metrics, RL-regulation, MarketSim, income-exclusion, Citizens-Advice, motor-insurance, access-to-insurance, Nash-equilibrium, arXiv-2506-00140, proxy-discrimination]
description: "Thibodeau et al. build a multi-firm market simulator and train an RL social planner to design fairness tax schedules. The collusion result stops the paper cold: a cartel that excludes every income group equally passes standard demand-fairness metrics with a perfect score."
author: burning-cost
---

Here is a scenario worth working through carefully. Three firms collude. They set prices high enough that low-income customers cannot afford cover. Every income group opts out at roughly the same rate. Your fairness metric — the one that checks whether uptake rates differ across income groups — passes them all with flying colours.

Equal exclusion. Perfect score.

This is the central result from Thibodeau, Nekoei, Taïk, Rajendran and Farnadi (arXiv:2506.00140, June 2025). They build MarketSim, a multi-firm market simulator, and demonstrate formally that the pathology above is not a degenerate edge case. It is a predictable property of any fairness metric defined over relative demographic gaps rather than absolute access levels.

---

## The unit-of-analysis problem

Consumer Duty asks each firm to demonstrate fair value outcomes for its customers. For a pricing team, that means showing your model does not produce systematic disadvantage for identifiable groups — income cohorts, customers with protected characteristics, vulnerable consumers. The FCA's supervisory toolkit is correspondingly firm-level: skilled persons reviews, section 166 reviews, Consumer Duty fair value assessments. You show that *your* model is not the problem.

The paper asks a different question: what happens to the market when every firm individually satisfies those requirements?

The answer is that market-level exclusion can persist — and in some configurations, worsen — even as individual firm fairness scores improve. The reason is structural. In a competitive market calibrated to the US Census (and there is no UK-calibrated equivalent yet, which matters), low-income consumers are 3.3 times more price-sensitive than high-income consumers. The multinomial logit demand model captures this through income-specific beta coefficients: beta_Low = 0.825, beta_High = 0.25. Profit-maximising firms, best-responding to each other in Nash equilibrium via Powell's method, find the optimal response to that price sensitivity is not to cut prices — it is to price towards the less elastic segment. Low-income opt-out is the equilibrium outcome of rational competitive behaviour, not regulatory failure in any individual firm's book.

The FCA's EP25/2 paper (July 2025) — the evaluation of GIPP price-walking remedies — found that the PS21/5 loyalty penalty rules worked: renewal price gaps closed. It did not evaluate access gaps by income or demographic group. That was not EP25/2's remit. But it is the gap this paper is pointing at.

---

## The collusion pathology

The paper's MarketSim has three income groups (High, Middle, Low) and two firms in the insurance scenario, five in consumer credit. In the baseline unregulated competitive equilibrium, the low-income opt-out rate is materially higher than the high-income rate. This is the access problem — the same structural dynamic that Citizens Advice documented in 2022, when they found that households in predominantly ethnic-minority postcodes pay approximately £280 more per year for motor insurance than comparable-risk households in predominantly white postcodes. The mechanism is competitive pricing optimising against price sensitivity, which correlates with income, which correlates with ethnicity.

In the collusion scenario, firms coordinate on high prices. Absolute opt-out rates rise for all groups. The demographic *gap* in opt-out rates shrinks — all groups are being excluded at similar high rates, because the prices are uniformly prohibitive. The demand-fairness metric scores this near-perfectly.

The authors acknowledge this result. They do not resolve it in their metric design, and we think they are right not to obscure it. The pathology reveals something important: any fairness metric defined solely over relative gaps between groups will, by construction, award a passing grade to a cartel that prices everyone out equally. Equal exclusion is not fair access. If your fairness assessment cannot distinguish "every group has equal access" from "every group is equally excluded", it is measuring the wrong thing.

Several published fairness metrics in use for UK insurance are defined as demographic parity ratios — the ratio of uptake or claim rates across groups. The collusion result applies to all of them. It is not an academic objection to an exotic edge case; it is a fundamental property of relative-gap metrics.

---

## The RL social planner

The paper's proposed solution is an outcome-based tax bracket mechanism. A Soft Actor-Critic reinforcement learning agent plays the role of social planner. It observes each firm's demographic selection gap — the degree to which its uptake rates differ across income groups — and assigns the firm to one of twenty fairness brackets, each with a tax rate between 0 and 1 applied to effective margin:

```
Effective margin = (price − marginal cost) × (1 − tax rate)
```

The RL agent learns the bracket tax schedule to maximise a welfare function W = mean firm profit × global fairness score. The conceptual link to Consumer Duty is genuine: the mechanism taxes firms on their outcomes, not their methodology. The FCA does not prescribe which rating factors to use; it asks whether your outcomes are fair. A differential charge on firms whose market presence produces unfair access outcomes, regardless of the specific pricing structure causing it, is a recognisable formalisation of where outcome-based regulation could go.

The health insurance results are the paper's best numbers. Against the unregulated competitive baseline:

| Market state | Welfare score | Opt-out rate |
|---|---|---|
| Unregulated competitive | 0.572 | 0.137 |
| Linear tax | 0.575 | — |
| RL social planner | 0.633 | 0.121 |

Welfare up 11%, opt-out down from 13.7% to 12.1%, fairness score at 0.895 against an empirical upper bound of around 0.90 for this market structure. The RL planner meaningfully outperforms both the unregulated baseline and a simple linear tax.

---

## The credit market contradiction

The consumer credit results are less comfortable, and the paper deserves credit for presenting them without softening.

| Market state | Welfare | Fairness | Opt-out rate |
|---|---|---|---|
| Unregulated | 0.416 | 0.660 | 0.173 |
| RL planner | 0.477 | 0.767 | 0.218 |

Fairness improves 16%. Welfare improves 15%. Opt-out worsens: from 17.3% to 21.8%. Under the regulated market, 4.5 percentage points more consumers are excluded from credit than under the unregulated equilibrium.

The mechanism is what you would expect once you see it. The RL agent learns that taxing firms with unequal demographic uptake rates causes firms to raise prices uniformly — which reduces the demographic gap in opt-out rates, because the higher prices price everyone out more equally. This is the collusion pathology replicated at the regulatory level. The agent achieves "fairness" by finding the policy that most efficiently reproduces the cartel outcome.

The paper names this contradiction. It does not resolve it. The welfare function W = mean profit × fairness has no principled basis for the trade-off it embeds. The regularisation parameter lambda (100 for insurance, 10 for credit, chosen by hyperparameter search) determines how strongly fairness is weighted against profit in the optimisation. A different lambda produces a different "optimal" tax schedule, with different opt-out rates. The framing of the RL planner result as a principled regulatory mechanism understates this sensitivity.

We think the honest reading is: the credit result shows that improving relative demographic fairness and improving absolute access are not the same objective, and can actively conflict. A regulator that only measures relative gaps will optimise towards the cartel equilibrium. If the actual goal is access — getting more low-income consumers into the market, not just equalising their exclusion — the welfare function needs to incorporate absolute opt-out levels directly.

---

## Where this sits against UK regulation today

Citizens Advice's 2022 finding — £280/year ethnicity penalty in motor insurance — is the most concrete UK evidence of the access gap this paper models. The FCA's July 2025 EP25/2 work evaluated the loyalty pricing remedies from PS21/5; the access question is not its focus. The FCA's Consumer Duty (PS22/9) creates obligation at firm level. There is currently no FCA mechanism to implement an outcome-based fairness tax on firms based on their market-level demographic selection gaps.

That is not a complaint about the FCA's current priorities. It is a description of where the paper sits relative to current regulatory practice: well ahead of it, in territory that requires legislative and institutional infrastructure that does not exist.

The paper's US calibration is a further constraint. Income proportions from Pew Research Centre 2024, insurance coverage rates from US Census — these are not UK numbers. UK market concentration (top five motor insurers hold roughly 60% of GWP), UK income-risk correlations, and UK demand elasticities all differ materially from the model's parameters. The direction of the effects — competitive markets exclude low-income consumers, outcome-based mechanisms can partially correct this — is plausible for UK markets. The magnitudes are not transferable.

A pricing actuary at a UK personal lines insurer does not need this paper to change anything they do next quarter. No pricing model refits, no Consumer Duty assessment methodology changes, no regulatory submissions. The practical implications, such as they are, sit at the governance and monitoring level: are you tracking market-level opt-out rates by income proxy alongside your own book? Do you have any view on whether your market — not just your book — is leaving low-income consumers without access? If the Consumer Duty five-year review (plausible from 2028) extends its scope to market-wide access outcomes, the firms that have been asking that question will be better positioned than those that have not.

---

## What the paper actually contributes

The specific limitations are worth naming clearly before the conclusion.

The welfare function W = mean profit × fairness is arbitrary. There is no theorem establishing that a 5% profit gain exactly offsets a 5% fairness deterioration. The multiplicative form ensures both matter but does not justify the relative weight. Results are sensitive to lambda.

The RL planner has perfect information — it observes every firm's demographic selection gap in real time. Real regulators have none of this. Implementing anything resembling the tax bracket mechanism would require solving information, legal authority, and measurement problems the paper does not address.

The credit market contradiction is unresolved. More consumers excluded under the regulated market than the unregulated one is a result that should cause pause before treating the mechanism as a policy recommendation.

US calibration is not UK calibration.

What works is the framing. The paper demonstrates, in a concrete and replicable simulation, that market-level exclusion can persist even when every individual firm satisfies individual fairness requirements. The collusion pathology demonstrates that relative-gap fairness metrics are structurally inadequate as welfare measures. The credit market result demonstrates that improving relative demographic fairness and improving absolute access can conflict directly.

These are genuine contributions to how a regulator should think about market-level fairness, not just firm-level compliance. Consumer Duty is a firm-level framework. The paper shows — formally, with a calibrated model — why a firm-level framework is not sufficient if the actual goal is access. That is worth understanding now, before whatever comes after Consumer Duty.

---

## Reference

Thibodeau, C., Nekoei, A., Taïk, A., Rajendran, G. and Farnadi, G. (2025). Profit, Fairness, and Market Equilibria: A Reinforcement Learning Approach. arXiv:2506.00140. [arxiv.org](https://arxiv.org/abs/2506.00140)

Citizens Advice (2022). *Paying Over the Odds: Ethnicity and Car Insurance Pricing.* [citizensadvice.org.uk](https://www.citizensadvice.org.uk)

FCA (2025). EP25/2: Evaluation of the General Insurance Pricing Practices Remedies. July 2025. [fca.org.uk](https://www.fca.org.uk)
