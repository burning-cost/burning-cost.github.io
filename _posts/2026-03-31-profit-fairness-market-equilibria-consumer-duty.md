---
layout: post
title: "Profit-Fairness Market Equilibria: Why Consumer Duty Audits the Wrong Unit"
date: 2026-03-31
categories: [fairness, regulation]
tags: [Consumer-Duty, FCA, EP25-2, market-equilibrium, collusion-pathology, fairness-metrics, RL-regulation, MarketSim, income-exclusion, Citizens-Advice, motor-insurance, proxy-discrimination, Nash-equilibrium, access-to-insurance]
description: "Thibodeau et al. build a multi-firm market simulator and demonstrate the collusion pathology: a cartel that excludes every income group equally passes standard demand-fairness metrics. Consumer Duty audits one firm. The real problem is market-level."
author: burning-cost
---

Here is a fairness audit scenario worth sitting with. Three insurers collude. They set prices high enough that low-income customers cannot afford cover. Every income group opts out at the same rate. Your fairness metric — the one that checks whether uptake rates differ across income groups — passes them all with flying colours. The exclusion is equal, therefore the market is fair.

This is not a hypothetical constructed to embarrass fairness researchers. It is a formal result from Thibodeau, Nekoei, Taïk, Rajendran and Farnadi (arXiv:2506.00140, June 2025), demonstrated in a calibrated multi-firm market simulation. And it identifies a genuine blind spot in how UK regulators currently think about pricing fairness.

---

## What Consumer Duty actually audits

The FCA's Consumer Duty framework asks individual firms to demonstrate fair value outcomes. For pricing, that means showing your pricing model does not produce systematic disadvantage for identifiable customer groups — income cohorts, demographic groups, customers with protected characteristics. The toolkit is firm-level: proxy detection, demographic A/E analysis, fair value assessments against benefit benchmarks. You demonstrate that *your* model is not the problem.

The FCA's EP25/2 paper (July 2025) evaluated whether the GIPP remedies introduced under PS21/5 reduced the loyalty penalty for renewing customers. It found they did. It explicitly did not assess market-wide access gaps by income or demographic group. This is not a criticism of EP25/2 — that was not its remit. But the gap it leaves open is exactly the one this paper addresses.

The paper's contribution is specific: showing that a set of firms can each satisfy individual fairness requirements while the market equilibrium systematically excludes low-income consumers. Not because any single firm is behaving badly. Because the competitive equilibrium itself produces exclusion.

---

## The model

MarketSim, the simulator the authors build, is sparse in the way economics models usually are — and deliberately so. Three income groups (High, Middle, Low — proportioned from US Census and Pew Research Center 2024 data). Two firms in the insurance case, five in the credit case. Consumer utility follows a multinomial logit: utility equals a baseline attraction minus a price-sensitivity coefficient times the premium, where price sensitivity increases sharply with lower income. Low-income consumers in the insurance scenario are 3.3 times more price-sensitive than high-income consumers (beta_Low = 0.825, beta_High = 0.25).

Firms maximise profit via Powell's derivative-free method against a Nash equilibrium concept — they best-respond simultaneously. There is no coordination and no regulatory intervention in the baseline scenario.

The regulator in the paper is a Soft Actor-Critic reinforcement learning agent that learns a tax schedule. It partitions firms into twenty fairness brackets based on their demographic selection gap (how much their uptake rates differ across income groups). Each bracket carries a tax rate between 0 and 1 that reduces effective margin:

```
Effective margin = (price - marginal_cost) × (1 - tax_rate)
```

The RL agent learns the tax schedule to maximise a product of mean firm profit and a global fairness score. The welfare function is W = mean_profit × global_fairness — a multiplicative combination with no principled justification for why a 5% profit gain exactly offsets a 5% fairness loss. We return to this problem.

---

## The collusion pathology

In the unregulated competitive scenario, firms maximise profit by pricing low-income consumers out of the market. The opt-out rate for the low-income group is substantially higher than for the high-income group. This is the access-to-insurance problem — structurally identical to what Citizens Advice documented in 2022, showing that households in predominantly ethnic-minority postcodes pay approximately £280 more per year for motor insurance than households in comparable-risk predominantly white postcodes.

In the collusion scenario, firms coordinate on high prices. All income groups opt out at high rates. The demographic *gap* in opt-out rates is small — all groups are excluded near-equally. The demand-fairness metric, which checks whether uptake rates differ across groups, scores this as approximately fair.

The authors acknowledge this result but do not resolve it in their metric design. We think the failure to resolve it is actually the most clarifying part of the paper: it shows that any fairness metric defined over uptake *ratios* is fundamentally inadequate as a welfare measure. Equal exclusion is not fair access. A cartel that prices everyone out equally is not passing a meaningful fairness test. The metric needs to incorporate absolute opt-out levels, not just relative gaps between groups.

This is not an academic objection. Several published fairness metrics used in UK insurance — including some framed around the FCA's fair value guidance — are defined as demographic parity ratios: the ratio of uptake or claim rates across groups. The collusion result shows that these metrics will, by construction, score perfect cartel behaviour as perfect fairness. If your fairness assessment only measures relative gaps, it cannot detect absolute exclusion.

---

## The RL planner results

The RL social planner produces genuine improvements in the health insurance simulation:

| Market state | Welfare score | Opt-out rate |
|---|---|---|
| Unregulated competitive | 0.572 | 0.137 |
| Linear tax policy | 0.575 | — |
| RL social planner | 0.633 | 0.121 |
| Collusion | 1.000 (profit-normalised) | high, all groups |

The RL planner achieves an 11% welfare gain over unregulated competition, with the opt-out rate falling from 13.7% to 12.1%. The fairness score under the RL planner reaches 0.895, described as the empirical upper bound for this market structure. These are the best numbers in the paper.

In consumer credit (five lenders), the picture is less comfortable:

| Market state | Welfare | Fairness | Opt-out rate |
|---|---|---|---|
| Unregulated | 0.416 | 0.660 | 0.173 |
| RL planner | 0.477 | 0.767 | 0.218 |

Fairness improves 16%. The opt-out rate worsens from 17.3% to 21.8%. More consumers are excluded from credit under the regulated market than under the unregulated one.

The paper notes this contradiction without resolving it. An RL agent that improves relative fairness by driving prices high enough that an extra 4.5 percentage points of consumers exit the market is not achieving what any regulator would actually want. This is partly a metric design problem — the welfare function W = mean profit × fairness score has no principled basis — and partly a consequence of the RL agent optimising what it is given, which rewards closing the demographic gap without penalising an increase in absolute exclusion.

The choice of regularisation parameter lambda also matters more than the paper acknowledges. The authors use lambda = 100 for insurance and lambda = 10 for credit, selected by hyperparameter search. A different lambda produces a different "optimal" tax schedule. The framing of the result as a principled regulatory mechanism understates how sensitive the output is to this unjustified choice.

We want to be direct about what these numbers are and are not: simulation results from a model with three income groups, 2–5 firms, and US Census calibration. A UK motor insurer should not take the 11% welfare improvement or the 16% fairness improvement as predictions for what would happen under similar intervention in the UK market. The direction of effect is plausible; the magnitudes are not transferable.

---

## Where this connects to UK regulation

Citizens Advice documented in 2022 that households in predominantly ethnic-minority postcodes pay approximately £280 more per year for motor insurance than comparable-risk households in predominantly white postcodes. This is the real-world version of the market-level exclusion the paper models. The mechanism is not individual firm malice — it is that legitimate rating factors (postcode, vehicle type, claims history) encode income and demographic information, and competitive pricing dynamics amplify rather than dampen those differentials.

Consumer Duty asks each firm to demonstrate that its pricing does not produce systematic disadvantage. This is the right question at the firm level. But if every firm individually complies while competitive dynamics continue to price low-income consumers out, the market outcome is still exclusion. The paper formalises why this can happen, and why firm-level compliance metrics cannot detect it.

UK motor and home insurance are concentrated markets. The top five motor insurers hold approximately 60% of gross written premium. The collusion result is directly relevant to how the CMA and FCA think about market analysis: a market study focused only on price levels, without examining access disparities by income group, may miss the most material welfare effect.

The tax bracket mechanism the paper proposes — taxing firms on their demographic selection *outcomes*, not on their methodology — is structurally similar to Consumer Duty's outcome framing. The FCA does not tell you which rating factors to use; it asks whether your outcomes are fair. A differential charge on firms whose market presence produces bad access outcomes, regardless of the specific pricing methodology that causes it, is a plausible formalisation of where outcome-based regulation could evolve.

The caveat is significant: the paper assumes a regulator with real-time, accurate fairness scores for every firm in the market, and the authority to impose differential tax rates on that basis. No such regulator exists anywhere. The FCA's toolkit consists of supervisory reviews, skilled persons reviews, and principle-based rules. The gap between the paper's assumed regulatory capability and the FCA's actual capability is not small.

---

## The honest assessment

This paper is theoretical infrastructure for a direction of regulatory thinking, not a description of where regulation currently is or where it will be in two years. The appropriate time horizon for the ideas it contains is a Consumer Duty five-year review — plausible from 2028 onwards — if the FCA decides to extend its scope from individual firm compliance to market-level access outcomes.

The specific limitations are real and should be named:

**US calibration.** Income proportions from Pew Research, insurance coverage rates from US Census. These are not UK numbers. UK income-risk correlations, demand elasticities, and market structures differ materially. None of the numerical results can be applied to UK motor, UK home, or any real UK market without full recalibration.

**Welfare metric without a foundation.** W = mean_profit × global_fairness is a product with arbitrary weighting. Lambda = 100 for insurance, 10 for credit, set by hyperparameter search. A different lambda produces a different "optimal" policy. Presenting this as a principled regulatory mechanism overstates what has been demonstrated.

**The opt-out contradiction.** In consumer credit, the regulated market produces more exclusion than the unregulated one. The paper does not resolve this. A regulatory mechanism that improves demographic parity by raising prices uniformly — pushing everyone out equally — is not a solution to access problems. It is a restatement of the collusion pathology at the market level.

**No real regulator.** The RL planner has perfect information and unlimited authority. Real regulation has neither. Translating the tax bracket concept into anything the FCA could actually implement would require solving information, authority, and legal framework problems the paper does not address.

What works is the framing. The paper demonstrates, in a concrete and replicable model, that market-level exclusion can persist even when every individual firm satisfies individual fairness requirements. That is a genuine and underappreciated point.

---

## What this means for boards

For a UK pricing actuary or head of pricing reading this paper: the relevant insight is not the tax schedule. It is the structural argument — that a market can produce systematic exclusion even when every individual firm satisfies individual fairness requirements.

This is a board-level argument for why industry coordination on access standards, not just individual compliance, is the correct long-run response to Consumer Duty. If your firm achieves fair value scores and your competitors price low-income consumers out, the market outcome is still exclusion. Your Consumer Duty compliance has not solved the problem. It has given you a clean audit trail while the problem persists elsewhere.

The practical implications are not about implementing tax brackets. They are about how you frame market-level risk at board level. If the FCA's Consumer Duty review in 2028 extends its scope to market-wide access outcomes — a plausible step — firms that have been thinking only about their own pricing models will not have the analysis ready. The firms that will be prepared are those that have started tracking market-level opt-out rates by income segment and building the evidentiary base now, before the regulatory question arrives.

That is a monitoring and governance question, not a modelling question. It requires combining your own book data with market-level intelligence — renewals vs new business rates by income proxy, FNOL withdrawal rates by segment, PCW conversion rates. None of this requires building a MarketSim. It requires asking the question: is our market leaving low-income consumers without access, and are we part of that dynamic or not?

Consumer Duty's five-year review will ask. The firms that have been asking the question themselves will have better answers.

---

## Further reading

- [Sequential Optimal Transport for Multi-Attribute Fairness in Insurance Pricing](/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — firm-level fairness correction in the insurance-fairness library
- [Your Policyholders Are Playing a Game with Your NCD Ladder](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/) — the hunger-for-bonus framework and game-theoretic NCD pricing
